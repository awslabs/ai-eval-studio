#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the CDK infra tests (infra/tests).
#
# They need a runner of their own: `infra/.venv` carries `aws_cdk` but no
# `pytest`, and `backend/.venv` carries `pytest` but no `aws_cdk`, so neither
# environment can run them alone — and a suite nothing invokes is silently
# inert.
#
# No extra install: run backend's pytest with infra's site-packages on
# PYTHONPATH. The two venvs are on different Python versions, hence the glob
# rather than a pinned path.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

infra_site=""
for candidate in infra/.venv/lib/python*/site-packages; do
    if [ -d "$candidate" ]; then
        infra_site="$repo_root/$candidate"
        break
    fi
done

if [ -z "$infra_site" ]; then
    echo "No infra/.venv site-packages found. Create it first:" >&2
    echo "  python3 -m venv infra/.venv && infra/.venv/bin/pip install -r infra/requirements.txt" >&2
    exit 1
fi

echo "==> pytest infra/tests (aws_cdk from ${infra_site#"$repo_root/"})"
PYTHONPATH="$infra_site" backend/.venv/bin/python -m pytest infra/tests "$@"
echo "infra tests: OK"
