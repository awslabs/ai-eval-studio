#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the security scanners locally, before an external scanner does.
#
# WHY this exists: whether a finding is fixed is a claim about the tracked files,
# and the fastest way to check it is to run the same scanners against those
# files. Reproducing each scanner here means a claim can be checked in a minute
# instead of a review cycle.
#
# Covers bandit, semgrep, the OSV database for Python packages, and npm audit for
# JavaScript. It is NOT a replacement for an external scanner: no secret scanning
# over full history (gitleaks) and no container scanning (grype) run here.
#
# Usage:
#   ./scripts/scan-local.sh            # everything
#   ./scripts/scan-local.sh bandit     # one scanner
#
# Tools are fetched on demand with ``uv tool run``, so nothing needs installing
# first and nothing lands on PATH.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Scanners write caches and version-check files under HOME. Point that at a
# scratch directory: the real HOME may be read-only to this process, and semgrep
# aborts with a PermissionError rather than skipping its version check.
SCAN_TMP="${TMPDIR:-/tmp}/scan-local"
mkdir -p "$SCAN_TMP/home"
export HOME="$SCAN_TMP/home"

WANT="${1:-all}"
rc=0

section() { printf '\n==> %s\n' "$1"; }

# Every scanner runs over TRACKED FILES ONLY, which is what a fresh clone holds.
# Scanning the working tree instead picks up build output: ``infra/cdk/cdk.out``
# carries a full copy of the backend per asset, which multiplies bandit's
# findings many times over and buries the real ones.
tracked_python() {
  git ls-files -- '*.py' ':!:**/cdk.out/**'
}

run_bandit() {
  section "bandit (tracked python, MEDIUM and HIGH)"
  # An array, one element per path, so a path is never word-split or globbed.
  # Built with a read loop because ``mapfile`` is not in the bash macOS ships.
  local files=() path
  while IFS= read -r path; do files+=("$path"); done < <(tracked_python)
  if (( ${#files[@]} == 0 )); then
    echo "  no tracked python files found — refusing to report a clean scan"
    return 1
  fi
  # B101 is ``assert`` and fires thousands of times in the test suite, which is
  # what asserts are for. Severity filtering, not a skip list, keeps the signal.
  #
  # Output goes to a file rather than straight down a pipe: bandit exits 1
  # whenever it reports anything at all, and under ``pipefail`` that status wins
  # over the verdict below — which made a run with no HIGH findings look failed.
  local report="$SCAN_TMP/bandit.json"
  uv tool run --from bandit bandit -q -f json -o "$report" "${files[@]}" >/dev/null 2>&1 || true
  python3 -c '
import json, sys, collections
results = json.load(open(sys.argv[1])).get("results", [])
keep = [r for r in results if r["issue_severity"] in ("MEDIUM", "HIGH")]
counts = collections.Counter((r["issue_severity"], r["test_id"]) for r in keep)
for (sev, test), n in sorted(counts.items(), key=lambda kv: (kv[0][0] != "HIGH", -kv[1])):
    print(f"  {sev:6s} {test} x{n}")
print(f"  total MEDIUM+HIGH: {len(keep)} (of {len(results)} at all severities)")
sys.exit(1 if any(r["issue_severity"] == "HIGH" for r in keep) else 0)
' "$report"
}

run_semgrep() {
  section "semgrep (registry packs)"
  # Community packs, plus the registry's Python security rules. An external
  # scanner may run a wider set still, so a clean result here does not prove a
  # clean result there.
  #
  # ``r/python.lang.security`` is here because the packs above do not carry its
  # audit rules — among them the one that reports an unaudited
  # ``create_subprocess_exec`` as an ERROR. It needs no login, and on this tree
  # it reports nothing: each site it matches carries an audited ``nosemgrep``
  # stating why the match does not apply.
  local report="$SCAN_TMP/semgrep.json"
  uv tool run --from semgrep semgrep \
    --config p/python --config p/javascript --config p/typescript \
    --config p/bandit --config p/secrets --config r/python.lang.security \
    --metrics off --quiet --json -o "$report" \
    --exclude 'cdk.out' --exclude 'node_modules' --exclude 'dist' \
    backend frontend/src infra scripts >/dev/null 2>&1 || true
  python3 -c '
import collections, json, sys

# Dispositioned by RULE SUFFIX. Everything else is reported as NEW and fails the
# gate. Each entry carries its reason beside it — an entry with no reason is
# indistinguishable from a finding somebody silenced to get a green run.
#
# A rule belongs here only when EVERY match is benign by construction. Audit
# rules, which ask for a judgement about one call site, do not: a rule-level
# entry would wave through the next unaudited site as "known". Those are
# suppressed at the site with a ``nosemgrep`` comment that states the audit,
# so a new site is reported as NEW.
DISPOSITIONED = {
    # assert in non-test code: the infra CDK asserts narrow a construct type so a
    # later cast is sound, and they run under cdk synth, never under -O.
    "B101",
    # random for retry jitter, not for anything secret.
    "B311",
    # temp paths are the Fargate worker per-task scratch, or derive an S3 key.
    "B108",
    # container behind an ALB must bind 0.0.0.0.
    "B104",
    # test fixtures, not credentials.
    "B105", "B106",
    # deliberate, with the reason written at each site.
    "B110", "B112",
}

report = json.load(open(sys.argv[1]))
counts = collections.Counter(
    r["check_id"].rsplit(".", 1)[-1] for r in report.get("results", [])
)
new = {rule: n for rule, n in counts.items() if rule not in DISPOSITIONED}
for rule, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    label = "NEW:  " if rule in new else "known:"
    print(f"  {label} {rule} x{n}")
errors = [e for e in report.get("errors", []) if e.get("level") == "error"]
if errors:
    print(f"  scanner errors: {len(errors)} (results may be incomplete)")
print(f"  {sum(counts.values())} finding(s); {sum(new.values())} not yet dispositioned")
sys.exit(1 if new or errors else 0)
' "$report"
}

run_python_deps() {
  section "OSV: python production dependencies"
  # pip-audit is the obvious tool and cannot run here — it builds a venv and
  # upgrades pip inside it, which a sandboxed or offline shell refuses. Querying
  # OSV directly needs only the pinned versions uv already resolved.
  ( cd backend && uv export --no-hashes --no-dev --no-emit-project 2>/dev/null ) \
    | grep -E '^[a-zA-Z0-9]' \
    | python3 -c '
import json, re, sys, urllib.request

pinned = {}
for line in sys.stdin:
    m = re.match(r"^([A-Za-z0-9._-]+)==([^\s;]+)", line.strip())
    if m:
        pinned[m.group(1)] = m.group(2)

queries = [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v}
           for n, v in pinned.items()]
request = urllib.request.Request(
    "https://api.osv.dev/v1/querybatch",
    data=json.dumps({"queries": queries}).encode(),
    headers={"Content-Type": "application/json"},
)
results = json.load(urllib.request.urlopen(request, timeout=120))["results"]

affected = {
    name: [v["id"] for v in (result.get("vulns") or [])]
    for name, result in zip(pinned, results)
    if result.get("vulns")
}
for name, ids in sorted(affected.items()):
    shown = ", ".join(ids[:3])
    print(f"  {name:22s} {pinned[name]:12s} {len(ids)} advisory(ies)  {shown}")
print(f"  affected: {len(affected)} of {len(pinned)} packages")
sys.exit(1 if affected else 0)
'
}

run_js_deps() {
  section "npm audit: javascript production dependencies"
  # ``--cache`` is required, not a nicety: npm cannot write the default cache
  # under a read-only HOME and misreports the EPERM as root-owned cache files.
  local report="$SCAN_TMP/npm-audit.json"
  ( cd frontend && npm audit --omit=dev --json --cache "$SCAN_TMP/npm-cache" 2>/dev/null ) \
    > "$report" || true
  python3 -c '
import json, sys

# Known and dispositioned, by ADVISORY id, never by package name: a package
# entry would pass the next advisory for that package as known. An entry here
# is a finding whose reason is written down, NOT a finding being hidden. A
# package is known only when every advisory reaching it, directly or through a
# dependency, is listed; anything else is NEW and fails the gate.
#
# DOMPurify, through monaco-editor. Monaco imports its own inlined copy at
# esm/vs/base/browser/dompurify/, which is 3.4.15 in monaco 0.57.0 (the newest
# release), so neither an upgrade nor an overrides entry changes the bytes the
# browser runs; an override would only make this report read clean. Each of
# these needs a sanitizer option monaco never sets. Its sanitize path uses
# ALLOWED_TAGS, ALLOWED_ATTR, ALLOW_UNKNOWN_PROTOCOLS, hooks, RETURN_DOM_FRAGMENT
# and RETURN_TRUSTED_TYPE, never IN_PLACE, SAFE_FOR_TEMPLATES or
# PARSER_MEDIA_TYPE. Revisit when monaco vendors 3.4.17 or later.
DISPOSITIONED = {
    "GHSA-6688-9rhm-gjv2": "needs IN_PLACE",
    "GHSA-mrv2-3pqr-j5jr": "needs IN_PLACE",
    "GHSA-p98j-92pf-mc4p": "needs IN_PLACE",
    "GHSA-cg2g-xh64-v4q7": "needs SAFE_FOR_TEMPLATES",
    "GHSA-c6gr-qvv4-5v3f": "needs PARSER_MEDIA_TYPE",
}

text = open(sys.argv[1]).read()
# npm appends notice lines after the JSON on some versions.
report = json.loads(text[: text.rindex("}") + 1])
vulns = report.get("vulnerabilities", {})


def advisories(name, seen=()):
    """Every advisory id reaching ``name``, following package-name ``via`` links."""
    found = set()
    for via in vulns.get(name, {}).get("via", []):
        if isinstance(via, dict):
            found.add(via.get("url", "").rsplit("/", 1)[-1] or via.get("title", "?"))
        elif via not in seen:
            found |= advisories(via, seen + (name,))
    return found


unexpected = {}
for name in sorted(vulns):
    ids = advisories(name)
    open_ids = sorted(ids - set(DISPOSITIONED))
    sev = vulns[name].get("severity")
    if open_ids:
        unexpected[name] = open_ids
        print(f"  NEW:   {name} ({sev}):", ", ".join(open_ids))
    else:
        print(f"  known: {name} ({sev}):", ", ".join(sorted(ids)))
print(f"  total {len(vulns)} package(s); {len(unexpected)} not yet dispositioned")
sys.exit(1 if unexpected else 0)
' "$report"
}

run_vendored() {
  section "vendored copies (inlined libraries)"
  # npm audit reads the dependency GRAPH. Two packages here inline a copy of
  # another library, so the graph and the shipped bytes can disagree — and did.
  # The assertion lives in a test because the version is only legible in the
  # installed package: minification strips the banners from the built bundle.
  ( cd frontend && npx vitest run src/lib/vendored-deps.test.ts 2>&1 ) \
    | grep -E 'Tests |passed|failed' | tail -2
}

case "$WANT" in
  bandit)  run_bandit  || rc=1 ;;
  semgrep) run_semgrep || rc=1 ;;
  python)  run_python_deps || rc=1 ;;
  js)      run_js_deps || rc=1 ;;
  vendored) run_vendored || rc=1 ;;
  all)
    run_bandit      || rc=1
    run_semgrep     || rc=1
    run_python_deps || rc=1
    run_js_deps     || rc=1
    run_vendored    || rc=1
    ;;
  *) echo "unknown scanner: $WANT (bandit|semgrep|python|js|vendored|all)" >&2; exit 2 ;;
esac

section "result"
if [ "$rc" -eq 0 ]; then
  echo "local scan: OK"
else
  echo "local scan: findings above need a disposition"
fi
exit "$rc"
