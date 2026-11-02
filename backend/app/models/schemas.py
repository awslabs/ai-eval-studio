# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for request validation and response serialization.

This module defines all the data models used throughout the API for:
- Project creation and configuration
- Chat message requests
- File upload responses
"""
from typing import Literal

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    """Request model for creating a new evaluation project.

    Attributes:
        name: Human-readable project name (1-200 characters).
        description: Optional project description for user reference.
    """
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""


class ProjectConfig(BaseModel):
    """Complete project configuration including metadata.

    This model represents the full state of a project, including immutable
    metadata (id, created_at) and mutable settings (agent_model, session_id).
    Target and judge model selection is handled in the harness config, not here.

    Attributes:
        id: Unique project identifier (UUID).
        name: Human-readable project name.
        description: Project description.
        created_at: ISO 8601 timestamp of project creation.
        agent_model: Claude tier alias (``"sonnet"`` or ``"opus"``) or an
            explicit Bedrock inference-profile ID. Resolved by the Claude
            Code CLI at turn time; aliases auto-upgrade when new Claude
            versions ship.
        session_id: Claude Agent SDK session ID for conversation continuity.
        owner_id: Cognito ``sub`` of the user who owns this project, or
            (in dev mode) the dev user id. Stamped at creation time and
            required for authz; records that lack it are unreachable.
        deleting: Durable deletion intent. Set ``True`` as the FIRST step
            of project deletion, before any store is touched. Once set, the
            project is committed to deletion: read paths (list/get) treat it
            as already gone, and a startup reconcile sweep finishes any
            delete interrupted by a crash. This is what makes delete
            reliable — an interrupted delete always self-completes rather
            than leaving an orphan.
    """
    id: str
    name: str
    description: str
    created_at: str
    agent_model: str = "opus"
    session_id: str | None = None
    owner_id: str | None = None
    deleting: bool = False


class ProjectUpdate(BaseModel):
    """Partial-update payload for :class:`ProjectConfig`.

    Every field is optional; an unset field is left untouched, so a caller can
    patch one field without restating the others.

    ``name`` and ``description`` are safe to change at any time: no path, S3
    key, or durable partition is derived from either — those all key on the
    project id.

    Attributes:
        agent_model: ``"sonnet"`` or ``"opus"``. Restricted to the two tier
            aliases because that is what the ModelSelector dropdown offers;
            other values can still be set by editing ``config.json`` directly.
        name: Replacement project name. Bounds match :class:`ProjectCreate`, so
            a name that could not be created cannot be renamed into either.
        description: Replacement description. An empty string is a meaningful
            value here — it clears the description — which is why absent and
            empty have to stay distinguishable.
    """
    agent_model: Literal["sonnet", "opus"] | None = None
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None


class ProjectSummary(BaseModel):
    """Lightweight project metadata for list views.

    Contains only the essential fields needed for displaying projects in lists,
    without the full configuration details. Enriched fields (last_activity,
    model_count, etc.) are computed at list time from the project filesystem.

    Attributes:
        id: Unique project identifier (UUID).
        name: Human-readable project name.
        description: Project description.
        created_at: ISO 8601 timestamp of project creation.
        last_activity: Most recent file modification time (ISO 8601) in the project.
        model_count: Number of unique model IDs across eval runs.
        latest_pass_rate: Pass rate from the most recent eval run (0.0-1.0).
        is_running: Whether an eval is currently running for this project, as
            far as THIS process knows. Read from the in-memory job tracker, which
            is empty after a coordinator restart and never sees a job dispatched
            by another process — so a detached worker can still be evaluating
            while this reports False. The project list is a sync function and
            cannot await the durable rehydrate that the activity endpoint does,
            which is where to look for an authoritative answer.
    """
    id: str
    name: str
    description: str
    created_at: str
    last_activity: str | None = None
    model_count: int = 0
    latest_pass_rate: float | None = None
    is_running: bool = False


class ChatRequest(BaseModel):
    """Request model for sending a chat message to the AI agent.

    Attributes:
        message: User message content (must be non-empty).
        project_id: ID of the project context for the conversation.
        session_id: Optional APP session id from a previous turn, used as a
            fallback if the backend failed to persist the session to the
            project config. It is never an AgentCore ``runtimeSessionId``:
            treating it as one let a caller name the microVM session its turn
            joined, since AgentCore does not enforce session-to-user mapping.
            On the relay path it reaches only the app-session slot, after a
            stored session has had its chance to win.
    """
    message: str = Field(..., min_length=1)
    project_id: str
    session_id: str | None = None


class UploadResponse(BaseModel):
    """Response model for a multi-file upload request.

    The frontend refetches the project file tree after a successful
    upload rather than reading these fields directly, so only minimal
    confirmation data is returned.

    Attributes:
        files: Project-relative paths of every file written to disk
            (includes the preserved archive plus each extracted entry).
        total_bytes: Total bytes written (archive + extracted contents).
    """
    files: list[str]
    total_bytes: int


class PresignedUploadInitRequest(BaseModel):
    """Request to initiate a large-upload presigned PUT.

    Attributes:
        filename: Original filename — used for the EFS landing path
            after the backend copies from S3 staging.
        content_type: Optional MIME type. The presigned URL embeds
            this so the browser PUT must match (S3 enforces).
        size_bytes: Declared size in bytes. **Required** so the
            backend can sign ``Content-Length`` into the presigned
            URL — that pins the byte count S3 will accept and prevents
            an authed attacker from filling the staging bucket up to
            the 24h lifecycle window.
    """
    filename: str
    content_type: str | None = None
    size_bytes: int


class PresignedUploadInitResponse(BaseModel):
    """Response with the presigned PUT URL.

    Attributes:
        upload_id: UUID identifying this upload — passed back to
            ``/uploads/complete`` so the backend knows which S3 staging
            object to copy.
        presigned_url: Pre-signed HTTPS PUT URL the browser uploads to.
            Valid for ~15 min. The frontend should retry on 4xx by
            requesting a fresh init.
        target_key: S3 object key under the staging bucket. Echoed for
            debugging; the URL is the only thing the browser needs.
        expires_at: ISO-8601 expiration timestamp.
        max_size_bytes: Per-upload cap enforced by the backend on
            completion.
    """
    upload_id: str
    presigned_url: str
    target_key: str
    expires_at: str
    max_size_bytes: int


class PresignedUploadCompleteRequest(BaseModel):
    """Confirmation that an out-of-band PUT to S3 staging finished.

    Attributes:
        upload_id: UUID returned by ``/uploads/init``.
        filename: Echoed for filename validation; must match the init
            request to prevent a different file from being claimed.
    """
    upload_id: str
    filename: str


class MetricConfig(BaseModel):
    """Configuration for a single evaluation metric.

    Attributes:
        name: Human-readable metric name (e.g., "Accuracy", "Relevance").
        type: Metric type identifier (default: "geval").
        criteria: Evaluation criteria description for the metric.
        evaluation_params: List of parameters to pass to the metric evaluator.
        threshold: Minimum score (0.0-1.0) for a test case to pass this metric.
        judge_model_id: Optional model ID to use for judging this metric.
    """
    name: str
    type: str = "geval"
    criteria: str = ""
    evaluation_params: list[str] = ["actual_output", "expected_output"]
    threshold: float = 0.5
    judge_model_id: str | None = None


class EvalRunConfig(BaseModel):
    """Configuration for launching an evaluation run.

    Attributes:
        target_model_id: Model ID to evaluate.
        prompt_version: Prompt template version identifier (default: "baseline").
        metrics: List of metrics to evaluate against.
        data_file: Path to the evaluation dataset file.
        max_test_cases: Optional limit on number of test cases to run.
        category_filter: Optional list of categories to include in the run.
    """
    target_model_id: str
    prompt_version: str = "baseline"
    metrics: list[MetricConfig] = []
    data_file: str = ""
    max_test_cases: int | None = None
    category_filter: list[str] | None = None


class EvalRunSummary(BaseModel):
    """Summary of a completed evaluation run.

    Attributes:
        run_id: Unique identifier for this evaluation run.
        timestamp: ISO 8601 timestamp when the run completed.
        target_model_id: Model ID that was evaluated.
        prompt_version: Prompt template version used in the run.
        total_cases: Total number of test cases executed.
        passed: Number of test cases that passed all metrics.
        failed: Number of test cases that failed at least one metric.
        errored: Number of test cases that encountered execution errors.
        metrics_summary: Aggregate scores for each metric (metric_name -> avg_score).
        cost_usd: Optional estimated cost in USD for the evaluation run.
    """
    run_id: str
    timestamp: str
    target_model_id: str
    prompt_version: str
    total_cases: int
    passed: int
    failed: int
    errored: int
    metrics_summary: dict[str, float] = {}
    cost_usd: float | None = None


class EvalCaseResult(BaseModel):
    """Result for a single test case within an eval run.

    Attributes:
        case_id: Unique identifier for this test case.
        input: Input/prompt text sent to the model.
        expected_output: Expected output from the ground truth dataset.
        actual_output: Actual output generated by the model.
        category: Optional category/tag for this test case.
        passed: Whether this test case passed all configured metrics.
        metrics: Per-metric results (metric_name -> {score, passed, reason}).
        latency_ms: Optional latency in milliseconds for model inference.
        token_usage: Optional token usage statistics (prompt_tokens, completion_tokens).
    """
    case_id: str
    input: str
    expected_output: str
    actual_output: str
    category: str | None = None
    passed: bool
    metrics: dict[str, dict] = {}
    latency_ms: float | None = None
    token_usage: dict[str, int] | None = None


class FileNode(BaseModel):
    """A node in the project file tree.

    Represents either a file or directory in the recursive tree structure.

    Attributes:
        name: File or directory name (e.g., "config.py").
        path: Relative path from the project root (e.g., "src/config.py").
        type: Node type, either "file" or "dir".
        size: File size in bytes (None for directories).
        children: List of child nodes (None for files, list for directories).
    """
    name: str
    path: str
    type: str  # "file" or "dir"
    size: int | None = None
    children: list["FileNode"] | None = None


class FileContentResponse(BaseModel):
    """Response for file content retrieval.

    Attributes:
        path: Relative path within the project directory.
        content: File content as UTF-8 text.
        size: File size in bytes.
        language: Detected language identifier for syntax highlighting.
        rows: Parsed CSV rows as list of dicts (only for CSV files).
        columns: CSV column names (only for CSV files).
        total_rows: Total row count in the underlying CSV, before any
            truncation. ``len(rows) < total_rows`` means the client only
            received a prefix and should surface that to the user.
    """
    path: str
    content: str
    size: int
    language: str
    rows: list[dict] | None = None
    columns: list[str] | None = None
    total_rows: int | None = None


class FileSaveRequest(BaseModel):
    """Request to save file content.

    Attributes:
        content: The text content to write to the file.
    """
    content: str


class FileSaveResponse(BaseModel):
    """Response after saving file content.

    Attributes:
        path: Relative path of the saved file.
        size: Resulting file size in bytes after writing.
    """
    path: str
    size: int


class SyntheticBatchSummary(BaseModel):
    """Summary of a synthetic data batch for list views.

    Mirrors the manifest fields the UI needs to render the left-pane
    batch list, with per-status counts aggregated from the manifest's
    ``cases`` array.

    Attributes:
        batch_id: Timestamped batch identifier (matches manifest filename).
        strategy: Generation strategy used (paraphrase, from_docs, from_scratch, failure_targeted).
        timestamp: ISO 8601-ish timestamp the batch was written at.
        seed_source: Optional free-text label describing the seed source.
        seed_run_id: Optional eval run id the batch was seeded from.
        total_cases: Total number of cases in the batch.
        pending: Count of cases currently awaiting review.
        approved: Count of cases currently approved.
        rejected: Count of cases currently rejected.
        csv_file: Filename of the companion CSV under data/synthetic/.
    """
    batch_id: str
    strategy: str
    timestamp: str
    seed_source: str | None = None
    seed_run_id: str | None = None
    total_cases: int
    pending: int
    approved: int
    rejected: int
    csv_file: str


class SyntheticCase(BaseModel):
    """A single synthetic case enriched with its CSV row values.

    Attributes:
        index: 0-based row index in the batch CSV (matches manifest ``cases[].index``).
        status: Current review state (``pending`` | ``approved`` | ``rejected``).
        values: CSV column name -> string value for this row.
    """
    index: int
    status: str
    values: dict[str, str]


class SyntheticBatchDetail(BaseModel):
    """Full synthetic batch manifest plus the enriched per-case rows.

    The ``rows`` array is assembled at read-time by zipping the manifest
    ``cases`` entries with the batch CSV. Never persisted to disk.

    Attributes:
        batch_id: Timestamped batch identifier.
        timestamp: ISO 8601-ish timestamp the batch was written at.
        strategy: Generation strategy used.
        seed_source: Optional free-text label describing the seed source.
        seed_run_id: Optional eval run id the batch was seeded from.
        total_cases: Total number of cases in the batch.
        field_mappings: Harness field mapping snapshot at generation time.
        csv_file: Companion CSV filename.
        rows: Enriched per-case rows (index + status + CSV values).
    """
    batch_id: str
    timestamp: str
    strategy: str
    seed_source: str | None = None
    seed_run_id: str | None = None
    total_cases: int
    field_mappings: dict[str, str]
    csv_file: str
    rows: list[SyntheticCase]


class SyntheticApproveRequest(BaseModel):
    """Request body for approving cases within a synthetic batch.

    Exactly one of ``indices`` (non-empty) or ``approve_all=True`` must be
    supplied; both absent/empty is a 400 at the router layer.

    Attributes:
        indices: Explicit 0-based row indices to approve. ``None`` or empty
            means "no individual selection provided".
        approve_all: If ``True``, every currently-pending case in the batch
            is approved (no-op for already-approved/rejected cases).
    """
    indices: list[int] | None = None
    approve_all: bool = False


class SyntheticRejectRequest(BaseModel):
    """Request body for rejecting cases within a synthetic batch.

    Attributes:
        indices: 0-based row indices to reject. Must be non-empty
            (enforced at the router layer).
    """
    indices: list[int]


class SyntheticBatchStatusResponse(BaseModel):
    """Response from approve/reject endpoints with updated counts + cases.

    Attributes:
        batch_id: Echoed batch identifier.
        pending: Updated pending count.
        approved: Updated approved count.
        rejected: Updated rejected count.
        cases: Updated raw cases array (``{"index", "status"}`` dicts, as
            stored in the manifest). Not the enriched ``SyntheticCase``
            shape — the UI only needs status flips for optimistic reconcile.
    """
    batch_id: str
    pending: int
    approved: int
    rejected: int
    cases: list[dict]


class OptimizationStartRequest(BaseModel):
    """Request body for ``POST /optimization/start`` and ``/resume``.

    Lets the UI pick which data source the optimization loop should use
    when dispatching evals. ``"raw"`` keeps the existing behaviour (uses
    the harness-configured data file / train split); ``"synthetic"`` and
    ``"merged"`` both require at least one approved synthetic case.

    Attributes:
        data_source: Selector driving ``DatasetService.resolve_data_file``.
    """
    data_source: Literal["raw", "synthetic", "merged"] = "raw"


class HarnessConfig(BaseModel):
    """Parsed harness configuration from src/config.py.

    Contains structured fields extracted via AST parsing from the project's
    harness configuration file. Includes eval settings, field mappings,
    and optimization parameters.

    Attributes:
        models: List of target model IDs to evaluate.
        data_file: Path to the evaluation dataset file.
        judge_model_id: Model ID used for LLM-as-a-judge metrics.
        prompt_version: Active prompt template version identifier.
        field_mappings: Unified field mapping dict. Tier 1 keys (input,
            expected_output, context, retrieval_context, tools_called,
            expected_tools) map to LLMTestCase params. Tier 2 keys map
            dataset columns to additional_metadata for display/filtering.
        target_metric_scores: Target pass rates per metric for optimization.
        max_iterations: Maximum optimization iterations allowed.
        plateau_window: Number of iterations to check for plateau detection.
        plateau_threshold: Pass-rate spread (0.0–1.0) below which the
            plateau window counts as flat. Defaults to 0.02 (2%).
    """
    models: list[str] = []
    data_file: str | None = None
    judge_model_id: str | None = None
    prompt_version: str | None = None
    field_mappings: dict[str, str] | None = None
    target_metric_scores: dict[str, float] | None = None
    max_iterations: int | None = None
    plateau_window: int | None = None
    plateau_threshold: float | None = None
    # ``ready``: the harness is RUNNABLE, not merely scaffolded. Computed by the
    # endpoint from what dispatch actually needs: models set in config.py, a
    # dataset selected, and the referenced prompt file present on disk. The
    # SPA's Start Optimization button gates on this; gating on config-parse
    # success alone would light it mid-build and offer an optimization that
    # fails immediately.
    ready: bool = False
