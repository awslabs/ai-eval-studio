# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Templates ↔ services parity test.

Asserts that adapter and mapper source files in ``app/services/`` are
byte-identical to their mirrors under
``app/templates/harness/src/models/``. Mirror drift causes harness-side
mantle calls to diverge from coordinator-side calls.
Asserted via ``filecmp.cmp(shallow=False)`` so trailing-whitespace and
line-ending differences also fail the build.

After editing any service file in ``PARITY_PAIRS``, copy the new bytes
verbatim to the template path before commit. The test is the canary —
it runs in <50ms (four small file pairs).
"""

import filecmp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SVC = REPO_ROOT / "app" / "services"
TPL = REPO_ROOT / "app" / "templates" / "harness" / "src" / "models"


# (service_filename, template_filename) — names match in every pair
# today, but the tuple shape leaves room for a future mismatch (e.g.
# template-only renames) without restructuring.
PARITY_PAIRS: list[tuple[str, str]] = [
    ("aws_retry.py", "aws_retry.py"),
    ("bedrock_mantle_mappers.py", "bedrock_mantle_mappers.py"),
    ("bedrock_mantle_model.py", "bedrock_mantle_model.py"),
    ("model_descriptors.py", "model_descriptors.py"),
]


def test_mantle_files_byte_identical():
    """All four service/template pairs are byte-identical.

    After editing any source file, re-copy it to the template path
    verbatim before commit. Catches trailing-whitespace and
    line-ending drift in addition to content drift.
    """
    for svc_name, tpl_name in PARITY_PAIRS:
        svc_path = SVC / svc_name
        tpl_path = TPL / tpl_name
        assert svc_path.exists(), f"missing service file: {svc_path}"
        assert tpl_path.exists(), f"missing template mirror: {tpl_path}"
        assert filecmp.cmp(svc_path, tpl_path, shallow=False), (
            f"templates parity drift: {svc_path} != {tpl_path}. "
            f"After editing the service file, re-copy it to the "
            f"template path."
        )


def test_bedrock_model_files_have_intentional_asymmetry():
    """``bedrock_runtime_model.py`` service vs template carries a KNOWN asymmetry.

    The template's ``generate()`` is decorated with ``@_bedrock_retry``
    directly, because it calls Converse itself (and accrues judge usage)
    instead of going through ``generate_with_system``. The service's
    ``generate`` delegates to ``generate_with_system``, which carries the
    decorator, so decorating it as well would double-wrap the retry. The
    mismatch is INTENTIONAL — this test stops a future commit from
    "fixing" it by making the two files identical.

    If this test ever fails, it means someone has aligned the two
    files. UPDATE this test (and document the rationale change in
    the commit message) rather than deleting it — the asymmetry
    record needs to live somewhere.
    """
    svc_path = SVC / "bedrock_runtime_model.py"
    tpl_path = TPL / "bedrock_runtime_model.py"
    assert svc_path.exists() and tpl_path.exists()
    assert not filecmp.cmp(svc_path, tpl_path, shallow=False), (
        "bedrock_runtime_model.py service and template are byte-identical, but "
        "the @_bedrock_retry decorator placement is INTENTIONALLY asymmetric. "
        "If this test fails, "
        "you've removed the asymmetry — re-read the parity rule before "
        "deciding."
    )
