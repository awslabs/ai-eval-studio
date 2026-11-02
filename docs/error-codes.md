# Error Codes

Catalog of the `code` values the backend attaches to the structured errors the
UI receives: `ResourceError.code`, and the `code` field of the project `error`
events that feed the toast layer. The frontend renders `message` + `hint` as-is
and does not branch on any code today; the code is the stable handle to branch
on if a specific remediation is ever needed.

**Rules:**
- Codes are lowercase, dotted, namespace-first: `<domain>.<slug>`.
- Never rename an existing code. The frontend has no generated copy of the
  catalog, so anything matching on a code would silently stop matching.
- Define a new code as a constant on the `Codes` class in
  `backend/app/services/resource_error.py`, at the bottom of the relevant
  section, and use the constant rather than a bare string at the call site.
  The `relay.*` codes and two of the `optimization.*` codes below are raised as
  bare strings rather than `Codes` constants.
- Only codes that some code path actually raises are listed here. `Codes` also
  defines several constants nothing raises.

The last column is guidance for whoever is looking at the error. Where the code
sets a `hint`, the guidance follows it.

---

## `harness.*`

Failures from the per-project standalone harness and the jobs that run it.

| Code | When it's emitted | What to check |
|---|---|---|
| `harness.subprocess_failed` | The catch-all failure for eval and harness jobs. The harness exited non-zero (details: `exit_code`, `stderr_tail`), exited 0 without a readable results file, or raised; an eval job ended in an unexpected state; a detached worker produced no results file or could not upload it; an eval dispatch was refused (an eval already running, models not invocable); or a harness validation or build job failed. The message names the case. | For a non-zero exit, the per-job log file has the full traceback. Otherwise follow the message. |
| `harness.uv_sync_failed` | `uv` is not on PATH, or `uv sync` exited non-zero (details: `exit_code`, `stderr`) or timed out. | Check the harness project's `pyproject.toml` and dependencies; install uv if it is missing. |

## `eval.*`

Failures detected in the results of an eval run.

| Code | When it's emitted | What to check |
|---|---|---|
| `eval.metric_errors_dominant` | ≥80% of metric scores across the run were errors (DeepEval `ignore_errors=True` swallowed the exceptions). Strong signal of infra breakage, not a prompt regression. Details include `errored_scores`, `total_scores`, `error_ratio`, `unique_errors`, `top_error_count`, `top_error_preview`. | The hint is chosen from the most common error (credentials, model access, throttling). |
| `eval.generation_errors_dominant` | ≥80% of cases produced no output (generation error or empty response). Metrics may still have "scored" the empty outputs, so this fires before metric-level detection. Details include `generation_failed_cases`, `total_cases`, `error_ratio`, `unique_errors`, `top_error_count`, `top_error_preview`. | As above; when the failures carry no error text, inspect the harness log for the full traceback before resuming. |
| `eval.generation_unreliable` | Target-model generation failed for more than the tolerated fraction of cases (default 2%, overridable via `EVAL_GENERATION_FAILURE_TOLERANCE_RATIO`) after the harness's own retries. This is the GENERATION domain (the model under test, not the LLM-as-a-judge) and the resumable band below `eval.generation_errors_dominant`. `details.models` lists each model over tolerance with `model_id`, `generation_failed_count`, `total_cases`, `generation_failed_ratio`, `endpoint` and `region`, plus `status` and `label` when an upstream HTTP status could be read from the errors. | With a 401/403 observed: check credentials and Bedrock model access. Otherwise retry the run, or reduce generation concurrency in the harness config. |
| `eval.judge_unreliable` | At least one model run reports `data_quality.reliability == "degraded"`: more than 5% of metric evaluations stayed null after the sequential retry, or a case dropped by `evaluate()` could not be recovered. Covers the LLM-as-a-judge only. `details.degraded_models` carries the per-model null and drop counts. | Throttling (429/503, or no readable status): raise the judge model's per-region throughput quota or pick a less-throttled judge, then resume. Any other status: check the judge model configuration and retry. |
| `eval.died_without_results` | A detached eval reached a terminal state in its durable record without emitting a results event — the worker died before completing. Details include `eval_job_id`. | Check the worker logs for credential expiry, a task kill, or a reconciler reap. Resolve the underlying cause and resume the optimization. |
| `eval.results_unavailable` | A detached eval completed, but its results file never materialized from S3 on the coordinator within the retry ceiling. The iteration is failed rather than scored against empty results. Details include `eval_job_id`. | Check the worker's S3 upload and the coordinator's materialization log; resume the optimization once resolved. |

## `optimization.*`

Failures of the optimization control loop.

| Code | When it's emitted | What to check |
|---|---|---|
| `optimization.project_not_found` | An optimization iteration found no optimization state for the project — typically because the project was deleted mid-run. | — (the run cannot continue) |
| `optimization.dispatch_failed` | An iteration could not run or finish its eval: the eval job vanished from the tracker mid-run (details: `eval_job_id`), or the eval failed without recording an error of its own. Also recorded on the optimization state when a live optimization record cannot be adopted because the harness config is missing or has no models (see `optimization.unadoptable`). | Check the eval job's log. For the adoption case, restore the harness config and start the optimization again. |
| `optimization.eval_wait_exceeded` | The loop's eval did not reach a terminal state within the wait ceiling (`OPTIMIZATION_WAIT_FOR_EVAL_MAX_S`, default 4 h), so a wedged eval or dead detached worker cannot pin the loop indefinitely. The eval is cancelled and the iteration failed. Details include `eval_job_id`, `elapsed_s`, `max_wait_s`. | Check the worker task and its logs for a hang or silent death; resume the optimization. |
| `optimization.wrong_execution_home` | The optimization loop was started inside the AgentCore runtime, which cannot own it — the loop must run on the coordinator — so it refused to start. Published as a project error event. | Retry the request. |
| `optimization.unadoptable` | The coordinator found a live optimization record with no loop running (for example after a restart) and could not adopt it, because the harness config is missing or has no models. The record is marked failed and its eval lock released. Published as a project error event; the state itself records `optimization.dispatch_failed`. | Restore the harness config, then start the optimization again. |

## `agent.*`

Failures of a chat turn in the Claude Agent SDK.

| Code | When it's emitted | What to check |
|---|---|---|
| `agent.cli_error` | The bundled `claude` CLI reported an error for a chat turn. The message carries the CLI's own text; details include `error_type`, `turn_id` and, when known, `api_error_status`. Also raised when a delegated subagent ran in the background and never reported back (details: `orphaned_subagents`). | The hint follows the failure: re-authenticate the AWS credentials Bedrock uses, wait out rate limiting (429), or retry an upstream error (5xx). For an orphaned subagent, send the request again. |
| `agent.sdk_stream_error` | An unexpected exception propagated out of the SDK message loop for a chat turn. Details include `exception` and `origin`. | Check the backend log for the traceback, and the Bedrock credentials if the error mentions tokens. |

## `relay.*`

Failures of a chat turn relayed to the AgentCore runtime. Each is published as
a terminal error on the turn's stream and as a project error event, with
`details.origin` set to `"relay"` and no hint. `relay.http_error`,
`relay.frame_idle_timeout`, `relay.turn_cap_exceeded`, `relay.no_result_frame`
and `relay.unexpected_error` also stop the runtime session before the turn's
claim is released, so a retry cannot race the still-running turn.

| Code | When it's emitted | What to check |
|---|---|---|
| `relay.runtime_arn_missing` | The coordinator has no `AGENTCORE_RUNTIME_ARN`, so there is no runtime to invoke. | Re-run `infra/scripts/dev-postdeploy.sh`, which reads the runtime ARN and redeploys the compute stack with it, or redeploy the compute stack with `-c agentcore_runtime_arn=...` (see `docs/runbooks/deploy.md`). |
| `relay.token_missing` | No access token was available to invoke the JWT-authorized runtime. | Sign in again and retry. |
| `relay.token_expired` | The runtime rejected the invoke with HTTP 401 or 403. | Sign in again and retry. |
| `relay.invocation_failed` | The runtime answered the invoke with any other non-2xx status. | Retry; the coordinator log has the response body. |
| `relay.http_error` | A transport error or read timeout while talking to the runtime. | Retry. |
| `relay.frame_idle_timeout` | No frame arrived for longer than `RELAY_FRAME_IDLE_LIMIT_S` (default 900 s) while the connection stayed open. | Retry; the project state is intact. |
| `relay.turn_cap_exceeded` | The turn ran longer than `RELAY_TURN_MAX_S` (default 7200 s). | Retry to continue; progress made so far is saved. |
| `relay.no_result_frame` | The runtime's stream ended without a result frame. | Retry. |
| `relay.unexpected_error` | Any other exception while relaying the turn. | Retry; check the coordinator log. |

## `system.*`

Backstop for unguarded crashes.

| Code | When it's emitted | What to check |
|---|---|---|
| `system.task_crashed` | An unhandled exception ended background work: an `asyncio.Task` guarded by `task_error_sink.attach_error_sink` (details: `task_name`, `exception`), the optimization loop (details: `exception`), or a single optimization iteration (details: `exception`, `iteration`). | See the backend log for the traceback. Retry if it is safe to do so. |

---

## Adding a new code

1. Add the string constant on `Codes` in `backend/app/services/resource_error.py`.
2. Add a row to this doc under the correct namespace.
3. Use the constant (not a bare string) at the call site so renames stay consistent.
4. If the frontend needs to render specific UI for this code, add a matching constant in `frontend/src/lib/types.ts` or a switch in `<ResourceErrorBanner>`'s caller.
