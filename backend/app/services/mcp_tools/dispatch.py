# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tools for job dispatch: dispatch_eval, dispatch_optimization,
dispatch_harness_build, request_user_input, and resume_job.

Fire-and-forget pattern: the coordinator calls ``dispatch_*``, gets a
``job_id`` back immediately, and the completion-turn service fires when
results are ready.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

from app.services.bedrock_mantle_mappers import MantleAPIRejected
from app.services.bedrock_runtime_model import (
    ConverseNoTextBlock,
    extract_converse_text,
)
from app.services.dataset_service import DatasetService
from app.services.harness_config import load_harness_config
from app.services.jobs import _publish_project_event, get_job_tracker
from app.services.mcp_response import mcp_error, mcp_json, mcp_success
from app.services.model_descriptors import MantleDescriptor, read_descriptors
from app.services.model_descriptors_async import persist_descriptors_locked
from app.services.optimization_state import OptimizationState
from app.services.project_paths import resolve_project_dir, safe_path
from app.services.task_error_sink import attach_error_sink

logger = logging.getLogger(__name__)

# Dispatch guard caps. Defined in ``services.job_admission`` and re-exported here
# because these names are part of this module's established surface (tests and
# messages reference them). One definition, so the REST routes and this module
# cannot drift apart — they did, and the caps existed only here.
#
# Per-user cap: a single user can have at most 3 non-terminal jobs (chosen so
# a user can run a standalone eval + an optimization loop, which itself holds
# the project eval lock).
# Account backstop: global ceiling across all users. At 1-10 users a full
# non-terminal scan is cheap (no new infra). A single user cannot
# consume the whole account budget because their own cap trips first (3 < 10).
from app.services.job_admission import (  # noqa: E402
    ACCOUNT_MAX_JOBS,
    PER_USER_MAX_JOBS,
    check_launch_admission,
    optimization_lock_holder_id,
)


async def _preflight_models_invocable(
    model_ids: list[str],
    *,
    descriptors: dict[str, MantleDescriptor] | None = None,
    project_id: str | None = None,
    project_dir: Path | None = None,
) -> list[str]:
    """Probe every model with a 1-token "Say OK" call on a short-timeout client.

    Catches models that entered a project's lineup AFTER
    ``validate_harness`` last ran — for example a model added to
    ``config.py`` directly, bypassing validation, which would then hang
    the dispatched eval for 20 minutes. A dead-fast probe here converts
    that into an instant clear error.

    Deliberately does NOT route runtime ids through the shared
    :class:`BedrockRuntimeModel`: its 4× retry × 300s production timeout would
    make a hung model take ~20 minutes here too. The runtime path uses
    a dedicated boto3 client with tight timeouts and 1 attempt; the
    mantle path uses ``BedrockMantleModel.preflight`` under
    a 20s wall-clock ceiling enforced via ``asyncio.wait_for`` —
    worst-case ~20s × hung models, run concurrently via ``asyncio.gather``.

    Invalidate side: when ``descriptors`` + ``project_id`` +
    ``project_dir`` are provided AND a mantle probe surfaces a stale
    descriptor (404 ``ModelRemovedFromMantle`` or 400
    ``MantleAPIRejected``), the failing id is removed from the on-disk
    descriptor file via ``persist_descriptors_locked``. Re-resolution
    fires on the next ``list_bedrock_models``. Transient errors
    (timeout, signature mismatch, connect error) DO NOT invalidate.

    Args:
        model_ids: Full list of ids to probe (runtime + mantle merged).
        descriptors: Map of mantle ids → descriptor for routing decisions.
            Empty for a project whose lineup is runtime-only (no descriptor
            file on disk), in which case every id takes the runtime probe
            path. Every production caller passes the project's map.
        project_id: Required for descriptor invalidation; without it,
            mantle probe failures still surface as bad ids but the
            descriptor stays untouched.
        project_dir: Required for descriptor invalidation; the on-disk
            descriptor file is the source of truth for the WRITE
            (read-fresh + delete + persist) to avoid TOCTOU.

    Returns:
        List of failure strings ``"{model_id} ({reason})"`` (empty list
        = all OK). Mantle invalidations are augmented with ``,
        descriptor invalidated — re-run list_bedrock_models to
        re-resolve``.
    """
    if not model_ids:
        return []

    descriptors = descriptors or {}
    runtime_ids = [m for m in model_ids if m not in descriptors]
    mantle_ids = [m for m in model_ids if m in descriptors]

    # ----- runtime probe path (unchanged contract) ----- #
    runtime_failures: list[str] = []
    if runtime_ids:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError, ReadTimeoutError

        config = Config(
            read_timeout=15,
            connect_timeout=5,
            retries={"max_attempts": 1},
        )
        # Build the client once; boto3 clients are thread-safe for the
        # converse() call we make below. Explicit region is required:
        # botocore's ambient region resolution reads only AWS_DEFAULT_REGION,
        # this backend's convention is AWS_REGION, so an ambient client fails
        # to construct wherever no AWS config file exists (containers, Fargate).
        # That failure lands before any model is probed.
        region = os.environ.get("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region, config=config)

        def _probe_runtime(model_id: str) -> str | None:
            try:
                resp = client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": "Say OK."}]}],
                    inferenceConfig={"maxTokens": 16},
                )
            except ReadTimeoutError:
                return "ReadTimeoutError"
            except ClientError as e:
                # The AWS error code lives in the response metadata; fall
                # back to the exception class name so the user always gets
                # something specific.
                code = e.response.get("Error", {}).get("Code") if hasattr(e, "response") else None
                return code or type(e).__name__
            except Exception as e:  # noqa: BLE001 — last-resort: surface, don't crash
                return type(e).__name__
            # Shape-based extraction, not content[0]: a reasoning model puts
            # ``reasoningContent`` first, and at the 16-token cap below it can
            # emit reasoning ONLY. Indexing position reported both shapes as
            # "empty output", i.e. marked an invokable model unavailable.
            try:
                text = extract_converse_text(resp, model_id=model_id)
            except ConverseNoTextBlock:
                # Invokable — it answered, just with no room left for text.
                # Reporting a failure here would cull a working model.
                return None
            except (KeyError, IndexError, TypeError):
                return "malformed response"
            return None if text else "empty output"

        runtime_results = await asyncio.gather(*[
            asyncio.to_thread(_probe_runtime, mid) for mid in runtime_ids
        ])
        runtime_failures = [
            f"{mid} ({reason})"
            for mid, reason in zip(runtime_ids, runtime_results)
            if reason is not None
        ]

    # ----- mantle probe path ----- #
    async def _probe_mantle(
        mid: str, descriptor: MantleDescriptor
    ) -> tuple[str | None, bool]:
        """Run preflight on one mantle id.

        Returns ``(reason, should_invalidate)``. ``reason=None`` means
        OK. ``should_invalidate=True`` means the descriptor is stale
        and the dispatch loop must remove it from disk; False means
        the failure is transient/environmental and the descriptor
        stays.
        """
        # Lazy import: keeps the runtime-only path dep-free for tests
        # that monkeypatch boto3 without setting up the mantle adapter.
        from app.services.bedrock_mantle_model import (
            BedrockMantleModel,
            MantleNotInvocable,
            ModelRemovedFromMantle,
        )

        try:
            model = BedrockMantleModel(descriptor, mid)
            await asyncio.wait_for(
                asyncio.to_thread(model.preflight),
                timeout=20.0,
            )
            return None, False
        except asyncio.TimeoutError:
            # Hit the 20s wall-clock ceiling. Transient by
            # construction (the descriptor may still be correct;
            # the network or upstream is just slow).
            return "TimeoutError", False
        except MantleAPIRejected as e:
            # 400 validation_error — the api/region pairing is wrong
            # for this id. Invalidate.
            return f"MantleAPIRejected: {e}", True
        except ModelRemovedFromMantle:
            # 404 — model gone from mantle. Invalidate.
            return "ModelRemovedFromMantle", True
        except MantleNotInvocable as e:
            # Transient/auth (connect error, sig mismatch, timeout
            # inside the adapter, schema drift, truncation, etc.) —
            # do NOT invalidate.
            return f"{e}", False
        except Exception as e:  # noqa: BLE001 — last-resort surface
            return type(e).__name__, False

    mantle_failures: list[str] = []
    to_invalidate: list[str] = []
    if mantle_ids:
        mantle_results = await asyncio.gather(*[
            _probe_mantle(mid, descriptors[mid]) for mid in mantle_ids
        ])
        for mid, (reason, should_invalidate) in zip(mantle_ids, mantle_results):
            if reason is None:
                continue
            if should_invalidate:
                to_invalidate.append(mid)
                mantle_failures.append(
                    f"{mid} ({reason}, descriptor invalidated — "
                    "re-run list_bedrock_models to re-resolve)"
                )
            else:
                mantle_failures.append(f"{mid} ({reason})")

    # ----- invalidate side: persist the deletions ----- #
    if to_invalidate and project_id and project_dir is not None:
        # Read fresh from disk to avoid TOCTOU — the descriptors dict
        # passed in was for routing decisions; the on-disk file is the
        # source of truth for the WRITE.
        disk_descriptors = read_descriptors(project_dir)
        changed = False
        for mid in to_invalidate:
            if mid in disk_descriptors:
                del disk_descriptors[mid]
                changed = True
                logger.info(
                    "[preflight] invalidated descriptor for %s; "
                    "re-resolution will fire on next list_bedrock_models",
                    mid,
                )
        if changed:
            await persist_descriptors_locked(
                project_dir, disk_descriptors, project_id
            )

    # Order: runtime failures first (preserve existing test
    # expectations), then mantle failures.
    return runtime_failures + mantle_failures


@tool(
    "dispatch_eval",
    "Dispatch a background evaluation run. Returns a job_id immediately — "
    "results arrive via a completion turn when the eval finishes. "
    "Do NOT poll check_eval_status in a loop after calling this.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "data_file": {"type": "string", "description": "Filename in data/raw/"},
            "target_model_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Bedrock model IDs to evaluate",
            },
            "prompt_version": {"type": "string", "description": "Prompt file name without .md extension"},
            "max_test_cases": {"type": "integer", "description": "Optional limit on test cases"},
            "judge_model_id": {"type": "string", "description": "Bedrock model ID for LLM-as-judge"},
            "data_source": {
                "type": "string",
                "enum": ["raw", "synthetic", "merged"],
                "description": "Which dataset to evaluate against. Defaults to 'raw'.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-supplied key so a repeated/double-sent Run-eval "
                               "collapses to one job. Same key within ~5 min = same job.",
            },
        },
        "required": ["project_id", "data_file", "prompt_version"],
    },
)
async def dispatch_eval(args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch an eval job and return immediately with the job_id.

    Validates inputs, creates a Job via JobTracker, and spawns the
    eval executor as a background task. The completion-turn service
    fires a coordinator turn when results are ready.
    """
    # Initialize job to None so the outer exception handler can tell
    # "raised before the job existed" from "raised after". The job
    # binding lives inside the try.
    job = None
    try:
        project_dir = resolve_project_dir(args["project_id"])

        # Resolve user_id and arch_split early so guards can use them
        from app.platform.types import current_request
        try:
            user_id = current_request().user_id
        except RuntimeError:
            # No request context (background-spawned dispatch — rare).
            from app.platform.auth import get_local_dev_user_id
            user_id = get_local_dev_user_id()

        # Resolve the arch_split flag ONCE at dispatch start.
        from app.platform.arch_split import resolve_arch_split
        arch_split = await resolve_arch_split(args["project_id"])

        # Resolve the project owner ONCE for lock/caps/idempotency. They key
        # on the OWNER, not on the request context (which background callers
        # lack), so every entry point and the reconciler agree on one partition.
        from app.services.agent_turn import _resolve_owner_id_for_project
        owner_id = _resolve_owner_id_for_project(args["project_id"])

        # Fence: concurrent eval already running
        # NEW PATH (arch_split=True): durable lock + caps + kill-switch
        # OLD PATH (arch_split=False): in-memory fence (unchanged)
        tracker = get_job_tracker()
        if arch_split:
            # Kill switch + per-user cap + account backstop, shared with the REST
            # launch routes so the two cannot diverge.
            refusal = await check_launch_admission(
                owner_id=owner_id, project_id=args["project_id"], kind="eval"
            )
            if refusal is not None:
                return mcp_error(refusal.message)

            # Durable eval lock will be claimed AFTER job creation (below)
        else:
            # OLD PATH: in-memory fence (byte-unchanged)
            existing_eval = tracker.get_active_job_by_type(args["project_id"], "eval")
            if existing_eval:
                return mcp_error(
                    f"An eval job is already running for this project "
                    f"(job_id: {existing_eval.job_id[:8]}). Wait for it to finish "
                    "or cancel it before dispatching another."
                )

        # Fence: optimization loop running
        # On the durable path (arch_split=True), read from the metadata store.
        # The EFS optimization directory may not exist in this process's view at
        # all (microVM, or post-restart before any iteration has written), so an
        # EFS read would report a loop that has already ended. On the old path
        # (arch_split=False), keep the EFS read verbatim.
        from app.services import optimization_state_repository as opt_repo

        loop_handle = await opt_repo.read(
            args["project_id"], project_dir, arch_split=arch_split
        )

        if loop_handle.is_live():
            return mcp_error(
                "Optimization loop is running — it dispatches evals internally. "
                "Wait for it to finish or cancel it first."
            )

        # Require harness
        if not (project_dir / "run_eval.py").exists():
            return mcp_error(
                "No eval harness found (run_eval.py missing). "
                "Complete Phase 3 (Eval Harness Creation) first."
            )

        # Resolve model IDs
        model_ids = args.get("target_model_ids") or []
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if not model_ids:
            return mcp_error("No model IDs provided. Use target_model_ids array.")

        # Validate data file
        data_dir = project_dir / "data" / "raw"
        data_path = safe_path(data_dir, args["data_file"])
        if data_path is None or not data_path.exists():
            return mcp_error(f"Data file not found: {args['data_file']}")

        # Validate prompt
        prompts_dir = project_dir / "prompts"
        prompt_path = safe_path(prompts_dir, f"{args['prompt_version']}.md")
        if prompt_path is None or not prompt_path.exists():
            return mcp_error(f"Prompt not found: {args['prompt_version']}")

        # Resolve data source
        data_source = args.get("data_source") or "raw"
        data_file_override: str | None = None
        if data_source != "raw":
            harness = load_harness_config(project_dir)
            field_mappings = (harness.field_mappings if harness else None) or {}
            if not field_mappings:
                return mcp_error("Cannot resolve synthetic data source without harness field_mappings.")
            try:
                data_file_override = DatasetService().resolve_data_file(
                    project_dir=project_dir,
                    data_source=data_source,
                    base_data_file=args["data_file"],
                    field_mappings=field_mappings,
                )
            except ValueError as exc:
                return mcp_error(f"Data source error: {exc}")

        # Create the job BEFORE the preflight probe. The preflight can take
        # up to ~20s (mantle models) and used to run first, so ``/activity``
        # reported ``eval: null`` for that whole window and the open page
        # only caught the eval on its next poll — a visible startup
        # "blackout". Creating + announcing the job up front makes the bar
        # appear the instant dispatch begins. The concurrent-eval fence
        # above already ran, so this job is the project's only active eval
        # and the fence correctly covers the preflight window too.
        job = tracker.create_job(
            project_id=args["project_id"],
            job_type="eval",
            config={
                "user_id": user_id,
                "arch_split": arch_split,
                "project_dir": str(project_dir),
                "model_ids": model_ids,
                "prompt_version": args["prompt_version"],
                "max_test_cases": args.get("max_test_cases"),
                "judge_model_id": args.get("judge_model_id"),
                "data_file_override": data_file_override,
                "source": "manual",
            },
        )
        # Announce the eval immediately so OptimizationStateProvider's
        # ``eval_job/started`` subscription flips the bar up without waiting
        # on the 3s poll.
        _publish_project_event(args["project_id"], {
            "resource": "eval_job",
            "type": "started",
            "job_id": job.job_id,
            "job_type": "eval",
        })

        # Idempotency dedup (NEW PATH only, separate from lock)
        # If the client supplied an idempotency_key, attempt to claim it.
        # A second dispatch with the same key within TTL collapses to the
        # first job (returns the original job_id with deduplicated=True).
        # This is SEPARATE from the eval lock: the lock rejects a DIFFERENT
        # concurrent eval; the idempotency key collapses the SAME click.
        #
        # Track whether THIS dispatch claimed the key (distinct variable, not
        # reusing `claimed` which gets reassigned by claim_eval_lock below).
        idempotency_claimed_by_us = False
        if arch_split and args.get("idempotency_key"):
            from app.platform import get_platform
            store = get_platform().metadata_store
            claimed, existing_job_id = await store.claim_idempotency_key(
                owner_id, args["idempotency_key"], job.job_id, ttl_seconds=300
            )
            if not claimed:
                # Key was already claimed by a prior dispatch — cancel this
                # just-created job and return the ORIGINAL job_id so the agent
                # can still track/poll/wait/cancel the collapsed-to job.
                # Do NOT set idempotency_claimed_by_us — this dispatch did not
                # claim the key, so it must not release it.
                tracker.update_state(job.job_id, "cancelled")
                return mcp_json({"job_id": existing_job_id, "deduplicated": True})
            # Key freshly claimed by THIS dispatch — mark it for release on failure
            idempotency_claimed_by_us = True

        # Claim the durable eval lock (NEW PATH only)
        # Ordering: create_job first (needed for the optimistic bar), then
        # claim the lock with the real job.job_id so terminal-release can
        # match it. If the claim fails, we fail the just-created job and
        # return an error — this prevents orphaning a phantom "running" eval
        # in the tracker.
        if arch_split:
            from app.platform import get_platform
            store = get_platform().metadata_store
            claimed = await store.claim_eval_lock(owner_id, args["project_id"], job.job_id)
            if not claimed:
                # Lock already held — release the idempotency key (if we claimed it)
                # before failing, so a retry can launch.
                if idempotency_claimed_by_us:
                    try:
                        await store.release_idempotency_key(
                            owner_id, args["idempotency_key"], job.job_id
                        )
                    except Exception as release_err:
                        logger.warning(
                            "[dispatch_eval] idempotency-key release failed (lock-claim-failure path): %s",
                            release_err,
                        )
                # Fail the just-created job and reject dispatch
                from app.services.resource_error import Codes, ResourceError
                tracker.update_state(
                    job.job_id,
                    "failed",
                    error=ResourceError(
                        code=Codes.HARNESS_SUBPROCESS_FAILED,
                        message="An eval is already running for this project.",
                    ),
                )
                return mcp_error(
                    "An eval is already running for this project (eval_already_running). "
                    "Wait for it to finish or cancel it before dispatching another."
                )

            # Wrap the post-claim body to ensure ANY exception fails the job
            # and releases the lock (via terminal transition). Without this, a raise
            # from read_descriptors or _preflight_models_invocable would leave the
            # lock held for 24h with no reconciler path to clear it.
            try:
                # Invocability preflight — catches models added to config.py
                # after ``validate_harness`` last ran. Without this, a hung
                # model silently consumes a full eval cycle (20+ min); with
                # it we fail in ~15s with a clear error. Includes the judge
                # so a broken judge doesn't get past the gate either. Runs
                # AFTER the cheap path/file checks so a misconfigured data
                # file fails fast without spending probe budget on models.
                judge_id = args.get("judge_model_id")
                check_ids = list(dict.fromkeys(
                    list(model_ids) + ([judge_id] if judge_id else [])
                ))
                descriptors = read_descriptors(project_dir)
                bad = await _preflight_models_invocable(
                    check_ids,
                    descriptors=descriptors,
                    project_id=args["project_id"],
                    project_dir=project_dir,
                )
                if bad:
                    # Release the idempotency key (if we claimed it) before failing,
                    # so a retry can launch after fixing the model lineup.
                    if idempotency_claimed_by_us:
                        try:
                            await store.release_idempotency_key(
                                owner_id, args["idempotency_key"], job.job_id
                            )
                        except Exception as release_err:
                            logger.warning(
                                "[dispatch_eval] idempotency-key release failed (bad-models path): %s",
                                release_err,
                            )
                    # Fail the job rather than orphaning it active — now that the
                    # job is created before the probe, an early ``return`` would
                    # otherwise leave a phantom "running" eval in the tracker that
                    # ``/activity`` would faithfully report forever.
                    from app.services.resource_error import Codes, ResourceError
                    tracker.update_state(
                        job.job_id,
                        "failed",
                        error=ResourceError(
                            code=Codes.HARNESS_SUBPROCESS_FAILED,
                            message="Models not invocable: " + "; ".join(bad),
                        ),
                    )
                    return mcp_error(
                        "Models not invocable: " + "; ".join(bad)
                        + ". Drop them from the lineup or fix their availability "
                        "and try again."
                    )

                # Preflight passed — launch via JobCompute (Fargate worker on cloud, in-process on dev)
                from app.platform.job_compute import get_job_compute
                from app.services import launch_gate

                # Settle the launch/cancel race immediately before starting the
                # worker, and no earlier: the claim is only worth anything if
                # nothing cancellable can happen between it and RunTask.
                #
                # The job has been visible — and cancellable — since before the
                # preflight above, which is a 15-20s window the user is more likely
                # to cancel during than any other. A cancel arriving in it finds no
                # compute_handle (it is written after RunTask returns) and no tagged
                # task (there is none yet), and used to conclude from that pair of
                # true observations that there was nothing to stop. It then marked
                # the job cancelled and released the project's eval lock, and this
                # code launched a worker for it anyway.
                decision = await launch_gate.claim_for_launch(
                    owner_id, args["project_id"], job.job_id
                )
                if decision is launch_gate.LaunchDecision.UNAVAILABLE:
                    # Nothing is known about cancellation, so launching is unsafe —
                    # and unlike the cancelled case nobody else owns this job, so it
                    # cannot be left pending for a reconciler to notice. Raise into
                    # the handler below, which frees the idempotency key and makes
                    # the terminal transition that releases the eval lock.
                    raise RuntimeError(
                        "could not confirm whether this evaluation was still "
                        "wanted before starting its worker"
                    )
                if decision is launch_gate.LaunchDecision.CANCELLED:
                    if idempotency_claimed_by_us:
                        try:
                            await store.release_idempotency_key(
                                owner_id, args["idempotency_key"], job.job_id
                            )
                        except Exception as release_err:
                            logger.warning(
                                "[dispatch_eval] idempotency-key release failed "
                                "(cancelled-before-launch path): %s",
                                release_err,
                            )
                    # The cancel that won the gate owns the job's terminal state,
                    # so do not write one here — it already published cancelled and
                    # released the lock. Just stop.
                    return mcp_error(
                        "The evaluation was cancelled before its worker started, "
                        "so nothing was launched."
                    )

                compute = get_job_compute()
                try:
                    handle = await compute.launch(
                        job_id=job.job_id,
                        owner_id=owner_id,
                        project_id=args["project_id"],
                    )
                except BaseException:
                    # No worker exists, so hand the gate back. Leaving it claimed
                    # would make every later cancel wait out its window and then
                    # report failure for a worker that never started.
                    await launch_gate.abandon_launch(
                        owner_id, args["project_id"], job.job_id
                    )
                    raise

                # Persist the compute_handle on the durable job record for cancel/status
                await store.update_job_state(
                    owner_id,
                    args["project_id"],
                    job.job_id,
                    compute_handle=str(handle),
                )

                # Start the durable progress bridge: poll the DDB event log
                # and republish progress onto the owner's /ws job:<id> stream so a detached
                # worker's per-case progress reaches the browser.
                # thread the pre-resolved arch_split boolean (resolve-once)
                from app.services.job_progress_republisher import maybe_start_poller
                await maybe_start_poller(args["project_id"], job.job_id, owner_id, arch_split)

                return mcp_json({
                    "job_id": job.job_id,
                    "status": "dispatched",
                    "models": model_ids,
                    "prompt_version": args["prompt_version"],
                })

            except Exception as e:
                # Release the idempotency key (if we claimed it) before failing.
                # This catches ANY raise from the post-claim try body: read_descriptors,
                # _preflight_models_invocable, get_job_compute, compute.launch,
                # update_job_state, maybe_start_poller — so failures on any of those
                # paths free the key for retry.
                if idempotency_claimed_by_us:
                    try:
                        await store.release_idempotency_key(
                            owner_id, args["idempotency_key"], job.job_id
                        )
                    except Exception as release_err:
                        logger.warning(
                            "[dispatch_eval] idempotency-key release failed (post-claim exception): %s",
                            release_err,
                        )
                # Post-claim exception — fail the job (terminal transition releases lock)
                from app.services.resource_error import Codes, ResourceError
                tracker.update_state(
                    job.job_id,
                    "failed",
                    error=ResourceError(
                        code=Codes.HARNESS_SUBPROCESS_FAILED,
                        message=f"Dispatch failed after lock claim: {e}",
                    ),
                )
                logger.error("[dispatch_eval] post-claim failure: %s", e, exc_info=True)
                return mcp_error(f"Dispatch error: {e}")
        else:
            # SECOND-EXECUTION-HOME GUARD: the old path runs the harness as an
            # in-process subprocess of whatever process executes this tool. In the
            # AgentCore microVM that means the eval runs inside the microVM —
            # invisible to the coordinator's JobTracker, reconciler and caps, and
            # killed by the microVM's own lifetime/idle limits. Fail loud instead:
            # arriving here from the microVM means the flags disagree (relay ON with
            # arch_split OFF), which is a misconfiguration, not a fallback.
            from app.platform.process_role import is_agentcore_runtime
            if is_agentcore_runtime():
                from app.services.resource_error import Codes, ResourceError
                logger.error(
                    "[dispatch_eval] REFUSING in-process eval in the AgentCore "
                    "runtime for project=%s — relay is ON but arch_split resolved "
                    "OFF, so the harness would run inside the microVM",
                    args["project_id"][:12],
                )
                tracker.update_state(
                    job.job_id,
                    "failed",
                    error=ResourceError(
                        code=Codes.HARNESS_SUBPROCESS_FAILED,
                        message=(
                            "Cannot run an in-process eval inside the AgentCore "
                            "runtime: the harness would execute in the microVM, "
                            "invisible to job tracking and bounded by the "
                            "microVM's lifetime."
                        ),
                        hint=(
                            "Both the relay and the durable-state path resolve "
                            "from EVAL_STUDIO_MODE, so they can only disagree if "
                            "the runtime image is missing EVAL_STUDIO_MODE=cloud. "
                            "Fix the runtime environment; there is no per-project "
                            "switch to flip."
                        ),
                    ),
                )
                return mcp_error(
                    "Cannot run an in-process eval inside the AgentCore runtime — "
                    "the eval must dispatch to a worker. This project has the relay "
                    "enabled but the durable-state path disabled; that combination "
                    "is not supported."
                )

            # Old path: spawn in-process via asyncio.create_task (preserved byte-for-byte)
            # Invocability preflight first
            judge_id = args.get("judge_model_id")
            check_ids = list(dict.fromkeys(
                list(model_ids) + ([judge_id] if judge_id else [])
            ))
            descriptors = read_descriptors(project_dir)
            bad = await _preflight_models_invocable(
                check_ids,
                descriptors=descriptors,
                project_id=args["project_id"],
                project_dir=project_dir,
            )
            if bad:
                from app.services.resource_error import Codes, ResourceError
                tracker.update_state(
                    job.job_id,
                    "failed",
                    error=ResourceError(
                        code=Codes.HARNESS_SUBPROCESS_FAILED,
                        message="Models not invocable: " + "; ".join(bad),
                    ),
                )
                return mcp_error(
                    "Models not invocable: " + "; ".join(bad)
                    + ". Drop them from the lineup or fix their availability "
                    "and try again."
                )

            task = asyncio.create_task(
                _run_eval_with_completion(job),
                name=f"job-eval-{job.job_id[:8]}",
            )
            job._task = task
            attach_error_sink(task, project_id=args["project_id"], resource="job")

            return mcp_json({
                "job_id": job.job_id,
                "status": "dispatched",
                "models": model_ids,
                "prompt_version": args["prompt_version"],
            })

    except Exception as e:
        # If a job was created but left non-terminal, fail it now. A job
        # created here and left active stays is_active, the concurrent-eval
        # fence reads is_active, so every subsequent dispatch for this project
        # is rejected until the process restarts. The terminal transition also
        # publishes the error to the UI and releases the durable eval lock, so
        # this single call covers both paths.
        #
        # The not-already-terminal guard: the arch_split post-claim handler
        # owns its own failure and returns, so reaching here with a terminal
        # job would mean a new path was added, and its error must not be
        # clobbered.
        if job is not None and job.state not in ("completed", "failed", "cancelled"):
            try:
                from app.services.resource_error import Codes, ResourceError
                tracker.update_state(
                    job.job_id,
                    "failed",
                    error=ResourceError(
                        code=Codes.HARNESS_SUBPROCESS_FAILED,
                        message=f"Dispatch failed after job creation: {e}",
                    ),
                )
            except Exception as bookkeeping_err:
                # A tracker failure (unknown id raises KeyError) must never
                # replace the real dispatch error with a bookkeeping error.
                logger.warning(
                    "[dispatch_eval] job-failing bookkeeping failed: %s",
                    bookkeeping_err,
                )
        logger.error("[dispatch_eval] FAILED: %s", e, exc_info=True)
        return mcp_error(f"Dispatch error: {e}")


async def _run_eval_with_completion(job) -> None:
    """Run eval executor and fire completion turn on finish."""
    from app.services.jobs.eval_executor import run_eval_job
    from app.services.completion_turn import request_completion_turn

    result = await run_eval_job(job)
    # Fire completion turn so coordinator reviews results
    await request_completion_turn(job.project_id, job)


@tool(
    "dispatch_optimization",
    "Start a background optimization loop. Reads parameters from harness config. "
    "Returns immediately with a job_id — results arrive via completion turns as "
    "each iteration finishes. Do NOT call dispatch_eval while this is running.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "Project UUID"},
            "target_metric_scores": {
                "type": "object",
                "description": "Override target pass rates (reads from config if omitted)",
            },
            "max_iterations": {"type": "integer", "description": "Override max iterations"},
            "plateau_window": {"type": "integer", "description": "Override plateau window"},
            "max_test_cases": {"type": "integer", "description": "Optional limit on test cases"},
        },
        "required": ["project_id"],
    },
)
async def dispatch_optimization(args: dict[str, Any]) -> dict[str, Any]:
    """Start a background optimization loop dispatched as per-iteration jobs.

    Reads parameters from harness config, creates OptimizationState, and
    spawns the optimization loop. Each iteration creates its own Job and
    fires a completion turn for the coordinator to review.
    """
    # Eval-lock ownership handoff bookkeeping. The lock is claimed BEFORE
    # the invocability preflight so the ~15-20s preflight window cannot admit a
    # concurrent manual eval. Everything between the claim and a successfully
    # spawned loop must release it again — there are a dozen validation returns and
    # an exception handler in between, so track the claim and release it in a
    # finally unless ownership was handed to the loop task.
    opt_lock_claimed = False
    opt_lock_owner: str | None = None
    opt_lock_holder_id: str | None = None
    lock_handed_to_loop = False
    try:
        from app.services.optimization_orchestrator import detect_latest_prompt

        project_dir = resolve_project_dir(args["project_id"])

        # Resolve user_id and arch_split BEFORE the existing-check (the
        # resolve-once discipline documented on ``arch_split.resolve_arch_split``).
        # This single resolved boolean is threaded into every iteration job below.
        from app.platform.types import current_request
        try:
            user_id = current_request().user_id
        except RuntimeError:
            from app.platform.auth import get_local_dev_user_id
            user_id = get_local_dev_user_id()

        from app.platform.arch_split import resolve_arch_split
        arch_split = await resolve_arch_split(args["project_id"])

        # Guard against duplicate starts. Rejects only a LIVE run — a None or
        # TERMINAL record may be overwritten below, and the handle read here carries
        # the version that overwrite needs.
        from app.services import optimization_state_repository as opt_repo

        start_guard = await opt_repo.read(
            args["project_id"], project_dir, arch_split=arch_split
        )
        if start_guard.is_live():
            return mcp_error("Optimization is already running.")

        # Resolve the project owner ONCE for lock/caps (same as dispatch_eval)
        from app.services.agent_turn import _resolve_owner_id_for_project
        owner_id = _resolve_owner_id_for_project(args["project_id"])

        # Guard against concurrent eval — starting optimization while
        # an eval is running would race for the harness subprocess
        # NEW PATH: durable lock + caps + kill-switch (same as dispatch_eval)
        # OLD PATH: in-memory fence (unchanged)
        tracker_check = get_job_tracker()
        if arch_split:
            from app.platform import get_platform
            store = get_platform().metadata_store

            # Kill switch + per-user cap + account backstop, shared with the REST
            # launch routes so the two cannot diverge.
            refusal = await check_launch_admission(
                owner_id=owner_id, project_id=args["project_id"], kind="optimization"
            )
            if refusal is not None:
                return mcp_error(refusal.message)

            # CLAIM the eval lock for the whole optimization run — do not merely
            # check it. The loop's eval dispatch
            # (optimization_orchestrator.dispatch_eval_job) takes no claim of its
            # own, so without one here the lock would stay free through the
            # ~15-20s invocability preflight below, and a manual eval dispatched
            # in that window would pass BOTH fences (its own lock check sees no
            # lock; this check has already returned) — two detached evals on one
            # project, violating one-eval-per-project.
            #
            # Claiming here closes the window: the run owns the lock from before
            # preflight until the loop reaches a terminal exit, where the tail of
            # the loop orchestrator releases it. The claim is keyed on the
            # OWNER partition and holds the LOOP's identity (a synthetic
            # optimization-scoped holder id), so per-iteration eval jobs — which
            # never claim — cannot release it out from under the run, while the
            # reconciler can still clear it if the loop dies.
            opt_lock_holder_id = optimization_lock_holder_id(args["project_id"])
            claimed_opt_lock = await store.claim_eval_lock(
                owner_id, args["project_id"], opt_lock_holder_id
            )
            if not claimed_opt_lock:
                return mcp_error(
                    "An eval is already running (eval_already_running). Wait for it to finish or cancel it "
                    "before starting optimization."
                )
            opt_lock_claimed = True
            opt_lock_owner = owner_id
        else:
            # OLD PATH: in-memory fence (byte-unchanged)
            existing_eval = tracker_check.get_active_job_by_type(args["project_id"], "eval")
            if existing_eval:
                return mcp_error(
                    f"An eval job is already running (job_id: {existing_eval.job_id[:8]}). "
                    "Cancel it or wait for it to finish before starting optimization."
                )

        # Read harness config
        harness = load_harness_config(project_dir)
        if harness is None:
            return mcp_error("No harness config found at src/config.py.")
        if not harness.models:
            return mcp_error("No models configured in harness config.")
        fm = harness.field_mappings or {}
        if "input" not in fm or "expected_output" not in fm:
            return mcp_error("field_mappings must include 'input' and 'expected_output'.")

        # No metrics guard here: metrics live as code in
        # src/metrics/metrics.py, and the harness's build_metrics fails loudly
        # if they are missing.

        # Resolve parameters: explicit args override the harness config; a
        # value missing from BOTH is an error naming the field — never an
        # invented default (this used to fabricate ExactMatch@0.85 / 10
        # iterations / plateau 3, so a half-configured project optimized
        # against targets nobody chose).
        target_scores = args.get("target_metric_scores") or harness.target_metric_scores
        max_iter = args.get("max_iterations") or harness.max_iterations
        plateau_win = args.get("plateau_window") or harness.plateau_window
        missing = [
            name
            for name, value in (
                ("target_metric_scores", target_scores),
                ("max_iterations", max_iter),
                ("plateau_window", plateau_win),
            )
            if not value
        ]
        if missing:
            return mcp_error(
                "Missing required optimization parameters (not in the call "
                f"args nor in src/config.py): {', '.join(missing)}. Set them "
                "in the harness config or pass them explicitly."
            )

        # Validate target VALUES are 0–1 pass rates, not raw units. Every
        # metric in the pipeline is normalized to a 0–1 pass rate (e.g. a
        # latency metric records 1.0 = "100% of cases under the ms
        # budget"). The ms budget belongs in the metric's threshold, NOT
        # here. A raw-unit target like ``{"Latency": 3000.0}`` makes
        # convergence (``score >= target`` with score ≤ 1.0) impossible —
        # the run can never hit target_reached and burns iterations to
        # plateau. Reject it loudly at dispatch so the misconfig surfaces
        # before a wasted run instead of silently never converging.
        out_of_range = {
            name: val for name, val in target_scores.items()
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0)
        }
        if out_of_range:
            return mcp_error(
                f"target_metric_scores must be 0–1 pass rates, got out-of-range "
                f"values: {out_of_range}. A pass rate is the fraction of cases "
                f"that meet the metric (e.g. 1.0 = every case passes). For a "
                f"threshold metric like latency, the target is the pass-rate "
                f"(e.g. 1.0 = all cases under budget) — the raw budget (3000ms) "
                f"belongs in the metric's threshold, not the target."
            )

        # Invocability preflight — once, before state creation. A
        # broken model in the lineup would otherwise stall every
        # iteration's eval for 20+ min before the optimizer gives up,
        # leaving a half-converged state file behind. Includes the
        # judge so a dead judge doesn't tank every iteration's scoring.
        check_ids = list(dict.fromkeys(
            list(harness.models)
            + ([harness.judge_model_id] if harness.judge_model_id else [])
        ))
        descriptors = read_descriptors(project_dir)
        bad = await _preflight_models_invocable(
            check_ids,
            descriptors=descriptors,
            project_id=args["project_id"],
            project_dir=project_dir,
        )
        if bad:
            return mcp_error(
                "Models not invocable: " + "; ".join(bad)
                + ". Drop them from the lineup or fix their availability "
                "and try again."
            )

        # Capture workload fingerprint. This path always runs the raw data
        # file (``data_file_override`` is None below), so there is no
        # override to fold in.
        workload = OptimizationState.workload_fingerprint_from_harness(
            harness, None
        )

        # Create optimization state. The version rides on the start-guard handle
        # read above, so a surviving TERMINAL record is overwritten correctly
        # (version None = genuine first create → attribute_not_exists guard;
        # int = overwrite an existing terminal record on the 2nd start).
        # ConcurrentUpdateError surfaces via mcp_error rather than crashing the
        # agent loop.
        from app.platform.metadata_store import ConcurrentUpdateError
        plateau_thresh = harness.plateau_threshold
        try:
            state = opt_repo.create(
                start_guard,
                target_metric_scores=target_scores,
                max_iterations=max_iter,
                plateau_window=plateau_win,
                plateau_threshold=plateau_thresh,
                workload=workload,
            ).state
        except ConcurrentUpdateError as e:
            return mcp_error(
                f"OptimizationState version conflict: {e}. Another writer updated "
                "the state concurrently. The agent should retry with fresh state."
            )

        # Detect starting prompt
        from app.services.optimization_orchestrator import detect_latest_prompt
        prompt_version = detect_latest_prompt(project_dir)

        # Optimization runs on whatever dataset the harness is configured
        # with — no auto-split. (The old train.csv auto-detection was
        # removed: it required a split_dataset feature that was never wired
        # end-to-end and crashed the harness. If a user wants to optimize
        # on a subset, they upload that subset as the dataset; validation
        # against held-out data is a separate user-triggered eval — see
        # coordinator Phase 7.)

        # user_id + arch_split were resolved ONCE above (before the existing-check)
        # and are threaded into EVERY iteration job config below so every iteration
        # in a run shares one flag value — a run is provably all-old or all-new,
        # never split mid-run (resolve-once discipline, see
        # ``arch_split.resolve_arch_split``).

        # Build iteration config template
        iteration_config = {
            "project_id": args["project_id"],
            "user_id": user_id,
            "arch_split": arch_split,
            "project_dir": str(project_dir),
            "model_ids": harness.models,
            "data_file": harness.data_file or "",
            "field_mappings": harness.field_mappings or {},
            "judge_model_id": harness.judge_model_id,
            "target_metric_scores": target_scores,
            "max_iterations": max_iter,
            "plateau_window": plateau_win,
            "max_test_cases": args.get("max_test_cases"),
            "data_file_override": None,
            "prompt_version": prompt_version,
            # the run holds the eval lock from before preflight. Thread the
            # owner partition + holder id so the loop's terminal tail can release
            # exactly the lock this dispatch claimed.
            "eval_lock_owner_id": opt_lock_owner,
            "eval_lock_holder_id": opt_lock_holder_id,
        }

        # Spawn the job-based optimization loop — but only in a process that can
        # own it. In the AgentCore microVM the running record written above IS
        # the dispatch request (dispatch-to-owner seam): the coordinator's
        # adoption sweep (startup + reconciler cadence, ~60s) observes a live
        # record with no loop task and spawns the loop there. Spawning here
        # instead would strand the loop when the microVM dies with the turn —
        # the execution-home guard in run_optimization_loop exists precisely to
        # refuse that, and a refusal there would land AFTER the state/lock side
        # effects above, wedging the project.
        from app.platform.process_role import is_agentcore_runtime
        from app.services.optimization_loop_orchestrator import (
            _loop_task_name,
            run_optimization_loop,
        )

        if is_agentcore_runtime():
            # Ownership of the eval lock travels with the record: the adopted
            # loop's terminal tail releases it, and the adoption sweep's
            # unadoptable path releases it if no loop can ever run.
            lock_handed_to_loop = True
            return mcp_json({
                "status": "dispatched",
                "execution": "queued_for_coordinator_adoption",
                "note": (
                    "Dispatched from the agent runtime. The coordinator adopts "
                    "and starts the loop within about a minute; iterations "
                    "report as they complete."
                ),
                "targets": state.target_metric_scores,
                "max_iterations": state.max_iterations,
                "models": harness.models,
                "starting_prompt": prompt_version,
            })

        attach_error_sink(
            asyncio.create_task(
                run_optimization_loop(iteration_config),
                name=_loop_task_name(args["project_id"]),
            ),
            project_id=args["project_id"],
            resource="optimization",
        )
        # Ownership of the eval lock now belongs to the loop task, which releases
        # it on every terminal exit. Stop the finally below from releasing it.
        lock_handed_to_loop = True

        return mcp_json({
            "status": "dispatched",
            "targets": state.target_metric_scores,
            "max_iterations": state.max_iterations,
            "models": harness.models,
            "starting_prompt": prompt_version,
        })

    except Exception as e:
        logger.error("[dispatch_optimization] FAILED: %s", e, exc_info=True)
        return mcp_error(f"Error: {e}")
    finally:
        # Release the eval lock on every exit that did NOT hand it to the loop:
        # a validation rejection (bad harness/metrics/targets), a failed preflight,
        # a state-version conflict, or an unexpected exception. Without this the
        # lock claimed above would survive to its 24h TTL and block all dispatches
        # on the project. Best-effort and holder-conditional, so it can never free
        # a lock this dispatch does not own.
        if opt_lock_claimed and not lock_handed_to_loop and opt_lock_owner:
            try:
                from app.platform import get_platform
                await get_platform().metadata_store.release_eval_lock(
                    opt_lock_owner, args["project_id"], opt_lock_holder_id
                )
                logger.info(
                    "[dispatch_optimization] Released eval lock for project=%s "
                    "(optimization never started)",
                    args["project_id"][:12],
                )
            except Exception:  # noqa: BLE001 — must not mask the original result
                logger.exception(
                    "[dispatch_optimization] Failed to release eval lock for "
                    "project=%s — the reconciler's stale-lock sweep is the backstop",
                    args["project_id"][:12],
                )




@tool(
    "dispatch_harness_build",
    "Dispatch a background harness build (scaffold + validate). Returns a job_id "
    "immediately — results arrive via a completion turn.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "Project UUID"},
            "scaffold": {"type": "boolean", "description": "Run scaffold step (default true)"},
            "validate": {"type": "boolean", "description": "Run validate step (default true)"},
        },
        "required": ["project_id"],
    },
)
async def dispatch_harness_build(args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a harness build job and return immediately."""
    try:
        project_dir = resolve_project_dir(args["project_id"])

        tracker = get_job_tracker()
        job = tracker.create_job(
            project_id=args["project_id"],
            job_type="harness_build",
            config={
                "project_id": args["project_id"],
                "project_dir": str(project_dir),
                "scaffold": args.get("scaffold", True),
                "validate": args.get("validate", True),
            },
        )

        from app.services.jobs.harness_executor import run_harness_build_job
        task = asyncio.create_task(
            _run_harness_with_completion(job),
            name=f"job-harness-{job.job_id[:8]}",
        )
        job._task = task
        attach_error_sink(task, project_id=args["project_id"], resource="job")

        return mcp_json({
            "job_id": job.job_id,
            "status": "dispatched",
        })

    except Exception as e:
        logger.error("[dispatch_harness_build] FAILED: %s", e, exc_info=True)
        return mcp_error(f"Error: {e}")


async def _run_harness_with_completion(job) -> None:
    """Run harness executor and fire completion turn on finish."""
    from app.services.jobs.harness_executor import run_harness_build_job
    from app.services.completion_turn import request_completion_turn

    await run_harness_build_job(job)
    await request_completion_turn(job.project_id, job)


@tool(
    "request_user_input",
    "Pause a running job and surface a question to the user in chat. "
    "The job transitions to 'awaiting_user' state. The question is emitted "
    "directly into the chat stream to guarantee delivery. When the user "
    "replies, call resume_job with their answer.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "job_id": {"type": "string", "description": "Job to pause"},
            "question": {"type": "string", "description": "Question to ask the user"},
        },
        "required": ["project_id", "job_id", "question"],
    },
)
async def request_user_input(args: dict[str, Any]) -> dict[str, Any]:
    """Pause a job and surface a question to the user.

    Transitions the job to "awaiting_user" state. Also publishes a
    project-bus event so the frontend shows a "Needs your input" badge
    and the question is guaranteed visible even if the coordinator's
    reply text doesn't include it verbatim.

    On the durable path (arch_split=True), rebuilds jobs from the store before
    reading the tracker so the agent can pause a durable job from the microVM,
    and the durable store wins on terminal state (refresh_terminal=True).
    On the old path (arch_split=False), rebuild is a no-op.
    """
    from app.platform.arch_split import resolve_arch_split

    tracker = get_job_tracker()

    # Resolve arch_split once and rebuild jobs from durable store before reading
    try:
        arch_split = await resolve_arch_split(args["project_id"])
        await tracker.rebuild_jobs_from_store(args["project_id"], arch_split, refresh_terminal=True)
    except Exception as e:
        # Store errors degrade to old-path behavior rather than raising into agent loop
        logger.warning(
            "[request_user_input] rebuild failed for project=%s: %s",
            args["project_id"][:12], e
        )

    job = tracker.get_job(args["job_id"])

    if not job:
        return mcp_error(f"Job not found: {args['job_id']}")
    if job.project_id != args["project_id"]:
        return mcp_error("Job does not belong to this project.")
    if not job.is_active:
        return mcp_error(f"Job is not active (state: {job.state}).")

    tracker.update_state(
        args["job_id"],
        "awaiting_user",
        pause_reason=args["question"],
    )

    # Publish on project bus so frontend shows the question regardless
    # of what the coordinator says in its reply text
    _publish_project_event(args["project_id"], {
        "resource": "job",
        "type": "awaiting_user",
        "job_id": args["job_id"],
        "job_type": job.type,
        "question": args["question"],
    })

    logger.info(
        "[request_user_input] job=%s paused: %s",
        args["job_id"][:8], args["question"][:100],
    )

    return mcp_success(
        f"Job paused. The following question has been surfaced to the user:\n\n"
        f"{args['question']}\n\n"
        "Include this question in your reply. When the user answers, "
        "call resume_job with their response."
    )


@tool(
    "resume_job",
    "Resume a paused job (state='awaiting_user') with the user's answer. "
    "A new coordinator turn fires with the user's answer so the subagent "
    "can be re-invoked with the additional context. If the user's answer "
    "changes intent (e.g. 'stop'), cancel the job instead of calling this.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "job_id": {"type": "string", "description": "Job to resume"},
            "user_input": {"type": "string", "description": "The user's answer to the paused question"},
        },
        "required": ["project_id", "job_id", "user_input"],
    },
)
async def resume_job(args: dict[str, Any]) -> dict[str, Any]:
    """Resume a paused job with the user's answer.

    Transitions the job from "awaiting_user" back to "running" and
    submits a completion turn with the user's answer as context so the
    coordinator can re-delegate to the subagent.

    On the durable path (arch_split=True), rebuilds jobs from the store before
    reading the tracker so the agent can resume a durable job from the microVM,
    and the durable store wins on terminal state (refresh_terminal=True).
    On the old path (arch_split=False), rebuild is a no-op.
    """
    from app.platform.arch_split import resolve_arch_split

    tracker = get_job_tracker()

    # Resolve arch_split once and rebuild jobs from durable store before reading
    try:
        arch_split = await resolve_arch_split(args["project_id"])
        await tracker.rebuild_jobs_from_store(args["project_id"], arch_split, refresh_terminal=True)
    except Exception as e:
        # Store errors degrade to old-path behavior rather than raising into agent loop
        logger.warning(
            "[resume_job] rebuild failed for project=%s: %s",
            args["project_id"][:12], e
        )

    job = tracker.get_job(args["job_id"])

    if not job:
        return mcp_error(f"Job not found: {args['job_id']}")
    if job.project_id != args["project_id"]:
        return mcp_error("Job does not belong to this project.")
    if job.state != "awaiting_user":
        return mcp_error(f"Job is not awaiting user input (state: {job.state}).")

    original_question = job.pause_reason or "Unknown question"

    # Attach user input and resume
    job.config["user_input"] = args["user_input"]
    job.pause_reason = None
    tracker.update_state(args["job_id"], "running")

    logger.info(
        "[resume_job] job=%s resumed with input: %s",
        args["job_id"][:8], args["user_input"][:100],
    )

    # Submit a coordinator turn with a resume-specific synthetic message
    # that gives the coordinator the user's answer + job context.
    from app.services.agent_turn import get_agent_turn_tracker

    synthetic_message = (
        f"User answered the paused question for {job.type} job.\n\n"
        f"Original question: {original_question}\n"
        f"User's answer: {args['user_input']}\n\n"
        f"Resume work. Re-delegate to the appropriate subagent with this answer included."
    )

    # Submit directly through the turn queue with the resume message
    # Authoritative optimization state resolved in async scope; passed into the
    # sync build below (which must stay sync — it runs inside the tracker's
    # _build closure). A read inside it could only be the EFS read.
    from app.services import optimization_state_repository as opt_repo

    resume_opt_state = await opt_repo.read_for_display(
        args["project_id"], resolve_project_dir(args["project_id"])
    )

    def _build(reserved: str):
        from app.services.coordinator_prompt import build_coordinator_prompt
        from app.services.agent_runner import start_agent_turn
        from app.services.project_paths import resolve_project_dir
        from app.main import get_project_manager

        pm = get_project_manager()
        project = pm.get_project(args["project_id"])
        if not project:
            raise RuntimeError(f"Project {args['project_id']} not found")

        # Pass the resolved project dir (cloud-mode-safe) — the resolver
        # gives us the owner-prefixed dir directly.
        project_dir = resolve_project_dir(args["project_id"])
        coordinator_prompt = build_coordinator_prompt(
            project, project_dir, resume_opt_state
        )
        return start_agent_turn(
            project_id=args["project_id"],
            prompt=synthetic_message,
            coordinator_prompt=coordinator_prompt,
            session_id=project.session_id,
            model=project.agent_model,
            origin="orchestrator",
            turn_id=reserved,
        )

    turn_tracker = get_agent_turn_tracker()
    turn_tracker.submit(
        project_id=args["project_id"],
        origin="orchestrator",
        build_turn=_build,
    )

    return mcp_success(f"Job {args['job_id'][:8]} resumed. Re-delegating with the user's answer.")
