# syntax=docker/dockerfile:1.7
# AI Eval Studio backend — cloud image.
#
# Built by CDK (`ecs.ContainerImage.from_asset` in infra/cdk/compute_stack.py)
# during `cdk deploy`, for linux/arm64, and pushed into the CDK-managed asset
# ECR repository that `cdk bootstrap` creates. There is no separate application
# ECR repository. The coordinator and the worker tasks both run this image.
#
# Two stages:
#   builder — uv installs deps from the lockfile into /app/.venv, then
#             copies the source.
#   runtime — same base, just the venv + source. Non-root `app` user.
#
# Image-size baseline: ~666 MB on python:3.11-slim, measured on a
# linux/amd64 build (the deployed image is built for linux/arm64). The
# largest pieces are the Claude Code CLI bundle inside claude-agent-sdk
# (~221 MB), pandas (75 MB), and numpy (45 MB+28 MB libs). An early
# <500 MB budget was a guess; the real floor with this dependency list is
# ~650 MB, accepted. Future shrink levers if it ever matters:
# strip claude-agent-sdk's bundled CLI to the one platform we run
# (linux/arm64), or move pandas-heavy code paths to the worker
# container and keep the coordinator slim.
#
# The base image is a TAG, not a digest, and that is deliberate — image scanners
# flag it, so here is the reasoning rather than a silent choice. A digest pins
# the Debian package set at the moment it was written, and nothing in this repo
# bumps it; a build a year from now would then ship a year of unpatched base
# CVEs. Floating on ``3.11-slim`` means each build picks up the current patch
# level, and the Python MINOR is what is actually pinned, which is the part that
# would change behaviour. Application dependencies get the opposite treatment
# for the opposite reason: ``uv.lock`` is exact, and uv itself is pinned below.
# Pin the digest here if a release ever has to be bit-reproducible — and then
# own bumping it.

FROM python:3.11-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# uv via the official ghcr image, pinned. Bump in lockstep with the
# uv.lock generator on dev machines (`uv --version`).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Layer cache: install deps from the lockfile FIRST so the deps layer stays
# warm across source-only changes. `--no-install-project` skips installing
# the local package itself; the project source comes in below.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now copy source and finalize the venv (installs the project package).
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# -----------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    # DeepEval's telemetry module tries to ``mkdir .deepeval`` on import
    # in the current working directory. With a non-root user that fails
    # the import — which fails the whole app boot. Opt out cleanly.
    DEEPEVAL_TELEMETRY_OPT_OUT=YES

WORKDIR /app

# uv + uvx in the runtime stage. Both are infrastructure for the
# coordinator's runtime, not just dependency installation:
#
# 1. ``services/harness_runner.py`` shells out to ``uv sync`` and
#    ``uv run python -u run_eval.py`` against each per-project
#    ``pyproject.toml``. Eval jobs hard-fail without ``uv``.
# 2. Scaffolded eval projects under ``projects/{guid}/`` are
#    designed to be runnable standalone — they
#    each have their own ``pyproject.toml`` and depend on ``uv``.
# 3. ``uvx`` is the standard launcher for ``{"command": "uvx", ...}``
#    MCP servers if any are wired into ``ClaudeAgentOptions`` later.
#    The bundled ``claude`` CLI spawns those during its ``initialize()``
#    control request; without ``uvx`` on PATH the spawn fails and
#    initialize hangs for 60s (``Control request timeout: initialize``).
#    No such MCP is wired today, but keeping ``uvx`` here is cheap.
#
# Pinned to the same version as the builder stage so the lockfile
# resolver and the runtime tooling never drift.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

# Non-root user — UID/GID 1000 to match the EFS access point in
# ``infra/cdk/storage_stack.py`` (``ProjectsAp`` chowns its root
# directory and squashes all NFS ops to UID/GID 1000). Drift between
# the two values would leave the container unable to write to
# ``/mnt/projects``.
#
# Non-root is also load-bearing for the Claude Agent SDK: the bundled
# ``claude`` CLI refuses to start with ``--dangerously-skip-permissions``
# when ``getuid() === 0`` (verified by reading the bundled binary).
#
# Home is ``/home/app``, separate from the workdir at ``/app``. The
# bundled ``claude`` CLI writes config to ``$HOME/.claude/`` on every
# startup; if HOME isn't writable by the running user the CLI exits
# silently with rc=0 and zero stderr — confirmed empirically by
# running the CLI inside the live task with HOME=/app (silent exit)
# vs HOME=/tmp (full initialize). ``/app`` is owned by root because
# WORKDIR creates it before USER switches and the COPY --chown only
# affects the contents we copy in, not the directory itself.
RUN groupadd -r -g 1000 app \
    && useradd -r -u 1000 -g app -d /home/app -s /sbin/nologin app \
    && mkdir -p /home/app \
    && chown app:app /home/app
ENV HOME=/home/app

COPY --from=builder --chown=app:app /app /app

# Clear the write bits on the runtime source tree. This directory is executed
# by `credential_process` on every refresh (via the mint helper in
# app/scripts/mint_worker_credentials.py), so write access to it is equivalent
# to control of the minted session tags. Agent-generated harness code runs as
# UID 1000 (the same `app` user), and the harness subprocess drops no
# privileges.
#
# What this does: an ordinary write to the mint helper or anything it imports
# (app/platform/aws/sts_token_vending.py, app/platform/token_vending.py,
# app/platform/types.py) fails with a permission error.
#
# What it does NOT do: stop code that sets out to substitute tags. The tree is
# owned by `app` (the COPY --chown above), and an owner can always chmod its own
# files writable again; the task definitions do not make the root filesystem
# read-only. The credential_process config the worker writes under
# /tmp/workspace/<job> is writable by the same user too. So this is a guard
# against accidental edits, not a tenant boundary — the repository README's
# "Security scope" section describes the isolation limits.
#
# Legitimate writers: none — PYTHONDONTWRITEBYTECODE=1 prevents __pycache__
# writes, HOME=/home/app is separate and writable, and all runtime writes
# (project files, eval output, temp files) go to /tmp or /mnt/projects. The
# app boots and runs evals with this restriction in place.
RUN chmod -R a-w /app/app

USER app

EXPOSE 8100

# Default command: coordinator (FastAPI). The coordinator task definition in
# compute_stack.py sets the same command explicitly, and worker tasks replace it
# at launch with `python -m app.scripts.worker_main` (a RunTask container
# override in FargateTaskCompute), so this CMD is only the default.
#
# No HEALTHCHECK directive: the ALB target group already exercises
# `/api/health` (network_stack.py).
#
# `--ws-max-size` bounds what an unauthenticated WebSocket can make the server
# buffer; the reasoning is at the coordinator command in compute_stack.py, which
# carries the same value.
#
# `app.scripts.serve_tls` runs uvicorn over TLS with a certificate it generates
# at start-up, so the hop from the ALB to this container is encrypted.
CMD ["python", "-m", "app.scripts.serve_tls", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--ws-max-size", "65536"]
