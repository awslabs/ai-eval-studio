# Naming Conventions

One unified convention for every file in `backend/app/` and `frontend/src/`. Applies universally — no per-folder exceptions. If the convention can't name your file, the file is doing too much; split it.

Read this before adding or renaming any file.

---

## The three-question rule

Every filename answers three questions in order:

> **What domain? What role? What shape?**

- **Domain** — the noun this file belongs to (`eval`, `dataset`, `project`, `chart`, `compare`, `harness`). Always a noun, never a verb or adjective.
- **Role** — what the file *is* from a consumer's perspective, drawn from the closed vocabulary below. No synonyms. No grab bags.
- **Shape** — single file, package, or 1:1 pair. Determined mechanically by the shape rules below.

If any of the three is unclear, the file's responsibility isn't crisp enough to name yet. Fix the code, not the name.

---

## Role vocabulary

One table for the whole repo. Every file adopting a role suffix picks exactly one row. The **test** column is how a reviewer decides deterministically — if two rows could apply, pick the one whose test fits more tightly.

| Role | Test — pick this role when the file… | Backend example | Frontend example |
|---|---|---|---|
| `_runner` / `Runner` | …executes a defined job end-to-end and returns a result | `eval_runner.py` | — |
| `_orchestrator` | …sequences multiple runners/agents, making control-flow decisions | `optimization_orchestrator.py` | — |
| `_tracker` | …owns an in-memory registry of live/recent objects, with eviction | `JobTracker` class inside `services/jobs/`¹ | — |
| `_manager` | …owns a persistent collection and its lifecycle (create/read/update/delete) | `project_manager.py` | — |
| `_state` | …represents a persistable state machine, one instance per entity | `optimization_state.py` | — |
| `_config` | …owns the read path for a configuration source (parse, validate, expose typed) | `harness_config.py` | — |
| `_client` | …wraps an external SDK, API, or subprocess behind a domain interface | `agentcore_relay_client.py` | — |
| `_paths` | …computes or validates filesystem paths for a domain | `project_paths.py` | — |
| `_response` | …constructs protocol-specific response shapes | `mcp_response.py` | — |
| `_prompt` | …builds a prompt string or template for an agent | `coordinator_prompt.py` | — |
| `_analyzer` | …derives structured findings from data; reads OK, no domain-state mutations | `failure_analyzer.py` | — |
| `_optimizer` | …iteratively improves X against a measurable target (has a convergence loop) | `prompt_optimizer.py` (agent) | — |
| `_loader` | …reads data from an external source into in-memory objects (and nothing more) | — (future use) | — |
| `_parser` | …converts one representation into another (string → struct, AST → dataclass); parsing ONLY | `chat_message_parser.py` (harness_config.py also parses an AST, but its role is `_config`) | — |
| `_store` | …abstracts persistent storage of one entity type behind a Protocol; has CRUD + query operations and one or more swappable backends (filesystem/cloud). Distinct from `_manager` because the backend is swappable; distinct from `_loader` because it covers writes too. | `metadata_store.py`, `session_store.py` | — |
| `_authz` | …computes authorization decisions for a specific protocol surface (HTTP routes, MCP tools, queue handlers); pure logic that calls into the shared `Auth` Protocol. Doesn't extract identity (that's the auth middleware) or implement policies (those live on the Auth impl) — just enforces them at one boundary. | `api_authz.py`, `mcp_authz.py` | — |
| `_service` (fallback) | …has multiple small operations on one data type with no single role dominating | `dataset_service.py` | — |
| `Dashboard` | …composes multiple panels into a full workspace view | — | `EvalDashboard.tsx` |
| `Panel` | …owns its own data fetching OR visualization of passed-in data at a full-region scale | — | `AnalyticsPanel.tsx` |
| `Tab` | …is one entry in a tab navigation bar (content lives inside) | — | `SummaryTab.tsx` |
| `Bar` | …is a horizontal strip of status or controls, fixed height | — | `ActiveJobBar.tsx` |
| `Detail` | …shows one entity's full state in a focused pane, driven by a parent selection | — | `CaseDetail.tsx` |
| `<Name>Detail<Shape>` *(stacked)* | …stacks two roles when the combination is genuinely meaningful (e.g. "a drill-down that is also a panel"). Allowed only when a single role loses information. | — | `OptimizationRunDetail.tsx` |
| `Table` / `Grid` | …renders tabular rows; Grid when cells are non-uniform widgets | — | `ResultsTable.tsx`, `ChangesGrid.tsx` |
| `Tree` | …renders hierarchical data with expand/collapse | — | `FileTree.tsx` |
| `Chart` | …renders one specific Plotly/D3 visualization | — | `LatencyHistogram.tsx` |
| `Section` | …is a render fragment reused by multiple parent components; not independently navigable or data-fetching | — | `DetailSection.tsx` |
| `use<Name>` | …is a custom hook; filename == exported function name | — | `useChartTheme.ts` |
| `<name>-utils` | …is pure, stateless helpers whose only shared context is the filename | — | `format-utils.ts` |

¹ The `_tracker` example points *inside* a file (the class) rather than at a file because the `services/jobs/` package houses both the `Job` dataclass and the `JobTracker` class in its `__init__.py`. If a future refactor extracts the tracker into its own file, the example becomes e.g. `job_tracker.py`.

### Bare-noun files (no role suffix)

A file uses only the domain noun when it *is* the domain's identity module — the file name fully describes the file:

```
lib/types.ts          # type definitions for the app
lib/api.ts            # the API client
lib/themes.ts         # theme data
services/agent_turn.py     # the AgentTurn type + its tracker (identity = the type)
agents/data_analyzer.py    # the data analyzer agent (identity = the persona)
```

Rule: if removing a role suffix would make the filename ambiguous, keep the suffix. If the bare noun is already specific, drop it.

Counter-example showing this rule in action: `services/bedrock_runtime_model.py`
was renamed *from* `bedrock_model.py` once a second adapter
(`bedrock_mantle_model.py`) shipped. The bare noun stopped being
specific the moment "bedrock model" had two meanings; the qualifier
became load-bearing.

### Banned roles

These don't describe *what the file is* — they describe either implementation style or "something code-like" (grab bag). Reviewers reject them:

- `_helper`, `_util`, `_tool`, `_handler`, `_processor` — grab-bag invitations
- `_builder`, `_factory`, `_creator` — describe *how*, not *what* (for **service modules** only — see persona exemption below)
- `Component`, `View`, `Container` in `.tsx` filenames — implied by the extension
- `_service` when a more specific role fits (`_loader`, `_runner`, `_manager`, `_client`). The specificity bar: if a single role describes 80%+ of the file, rename. If multiple small operations on one data type have no dominant role, `_service` is the honest answer.

### Persona exemption for `agents/` + `prompts/`

Agent files are named by **persona**, not by service role. Agent filenames describe *who the agent is* the way a job title does — `data_analyzer`, `harness_builder`, `prompt_optimizer`. The closed role vocabulary does NOT apply to `agents/<name>.py` or `prompts/<name>.md` stems. Personas may use words like "builder" or "architect" that are banned for services.

Rule: if it's in `agents/` or `prompts/` and names a persona, the service-vocabulary rules are suspended. Every other convention (1:1 pair matching, snake_case/kebab-case, leading-underscore for fragments) still applies.

---

## Shape rules

Every file is one of three shapes: **single file**, **package**, or **1:1 pair**.

### Single file

The default. Use until one of the trigger conditions for a package is met.

### Package (folder with `__init__.py` / `index.tsx`)

Promote a single file to a package when ALL three are true:

1. The file has grown past ~400 lines AND
2. It has 3+ internal subcomponents / submodules AND
3. The subcomponents are tightly coupled — not independently useful outside the package.

Package folder name: **lowercase noun** — singular for a concept (`compare`), plural for a set (`charts`, `mcp_tools`).

Package boundary rules:
- Only the top-level `index.tsx` / `__init__.py` is importable from outside the package.
- Internal files still follow the role vocabulary (a package does NOT exempt its files from the rules).
- Package tests name the package in a FLAT file, they do not mirror its folders:
  `services/mcp_tools/data.py` → `tests/test_mcp_tools_data.py`. `backend/tests/`
  has no package mirrors — only `fixtures/` and `integration/`, which group by
  kind rather than by source package.

Don't create a package for two files — leave them as siblings. Don't skip `index.tsx` / `__init__.py` (they're the boundary).

### 1:1 pair

Two files representing the same identity in different forms. Stem must match exactly. Current pairs in this repo:

| Left | Right |
|---|---|
| `backend/app/agents/<name>.py` | `backend/app/prompts/<name>.md` (kebab-case stem) |
| `backend/app/services/<module>.py` | `backend/tests/test_<module>.py` |
| `backend/app/services/<pkg>/<file>.py` | `backend/tests/test_<pkg>_<file>.py` (flat — see Package) |
| `frontend/src/hooks/use<Name>.ts` | `function use<Name>` inside (same name) |

### Leading-underscore convention

A leading `_` means **shared internal fragment, not a standalone unit**. Always consumed from another file in the same folder:

```
backend/app/prompts/_rules.md      # shared prompt fragment referenced by agents
```

Not an exception to the naming rules — the `_` is part of the name and signals the role.

---

## Per-folder syntax

The role vocabulary is language-agnostic; only the syntax differs:

| Folder | Pattern | Example |
|---|---|---|
| `backend/app/services/` | `<domain>_<role>.py` | `eval_runner.py` |
| `backend/app/services/<pkg>/` | `<role>.py` (domain implied by package) | `mcp_tools/data.py` |
| `backend/app/agents/` | `<persona>.py` (snake_case) — persona IS the identity | `data_analyzer.py` |
| `backend/app/prompts/` | `<persona>.md` (kebab-case, matching agents stem) | `data-analyzer.md` |
| `backend/app/api/` | `<resource>.py`, named for the resource its routes serve; most routers mount under `/api/projects` | `eval_jobs.py` |
| `backend/app/models/` | `<concern>.py` | `schemas.py` |
| `backend/tests/` | flat `test_<module>.py`; a package module becomes `test_<pkg>_<file>.py` | `test_project_paths.py`, `test_mcp_tools_data.py` |
| `frontend/src/components/<domain>/` | `<Name><Role>.tsx` (PascalCase) | `AnalyticsPanel.tsx` |
| `frontend/src/components/<domain>/<pkg>/` | `<Name><Role>.tsx` (inside pkg boundary) | `compare/SummaryTable.tsx` |
| `frontend/src/components/ui/` | `<name>.tsx` (kebab-case — shadcn) | `stat-item.tsx` |
| `frontend/src/hooks/` | `use<Name>.ts` | `useChartTheme.ts` |
| `frontend/src/lib/` | `<name>.ts` or `<name>-utils.ts` | `format-utils.ts` |

**`lib/` bare name vs `-utils` decision:**

Use `-utils` when the module name is the ONLY thing connecting the exports. Use bare name when the exports describe a single cohesive concept.

- `format-utils.ts` → `formatLabel`, `truncate`, `shortModel`, `fmtNum` — each independent. ✓ `-utils`
- `chart-layout.ts` → `getBaseLayout`, `getHoverStyle` — one concept (chart theming). ✓ bare
- `api.ts` → API client with a base URL and methods — one concept with state. ✓ bare
- `themes.ts` → theme data — one concept, no runtime. ✓ bare

---

## Anti-patterns

These have all appeared in this codebase. Don't repeat them.

### Files named after what they *contain* rather than what they *are*

Pre-refactor `services/eval_tools.py` hit 1,556 lines because "tools" was a grab bag — the suffix named the implementation style ("a collection of tool decorators") instead of the responsibility. Always name by role.

### Files whose names understate the module's purpose

Pre-refactor, `services/mcp_server.py` was a 39-line cache/factory, not a server. It was renamed to `services/mcp_server_cache.py` — a reader looking for "the MCP server" should find the actual server (now the `mcp_tools/` package), not a cache for it. When a file wraps state around something else, name it after the wrapping role (`_cache`, `_pool`, `_registry`), not the underlying thing.

### Two modules with the same base name doing unrelated things

Pre-refactor:
```
agents/prompt_optimizer.py                # the subagent
services/background_prompt_optimizer.py   # the headless runner
```

Both share "prompt_optimizer" but the first is an identity, the second is a client. The fix is to name the second by its role — `prompt_optimizer_client.py` — or, as happened here, to fold it into its only caller: the headless run is now `_run_prompt_optimizer` in `services/optimization_loop_orchestrator.py`, and `agents/prompt_optimizer.py` stays the persona.

### Role suffixes that describe *how*, not *what*

`_builder`, `_factory`, `_creator` describe implementation style. Every constructor builds something; the suffix narrows nothing.

```
# worse
services/coordinator_prompt_builder.py

# better
services/coordinator_prompt.py  # domain + "this is the prompt"
```

### Components whose name repeats the file extension

```
// worse
FileTreeComponent.tsx
CaseDetailView.tsx

// better
FileTree.tsx
CaseDetail.tsx
```

---

## Pre-PR self-check

Before adding or renaming a file:

- [ ] Filename has a clear **domain** (not generic like `utils`, `tools`, `helpers`)
- [ ] Filename has a **role** from the vocabulary above — or is a bare noun that fully describes the file's identity
- [ ] No banned suffix for services (`_builder`, `_factory`, `_creator`, `_helper`, `_util`, `_handler`, `_processor`). `_service` is permitted ONLY when multiple small operations share one data type with no dominant role.
- [ ] If it's an **agent** (`agents/` or `prompts/`), it's named by persona — the banned-role list does not apply. The prompt file stem must match the agent file stem.
- [ ] If it's an **API router**, the filename names the resource its routes serve
- [ ] If it's a **hook**, the filename matches the exported function name
- [ ] If it's a **test**, it is a flat file in `backend/tests` — `test_<module>.py`, or `test_<pkg>_<file>.py` for a module inside a package
- [ ] If it's a **package**, it meets all three trigger conditions (400+ lines, 3+ subcomponents, tightly coupled)
- [ ] No two files in the repo share a base name except intentional 1:1 pairs

If the file fails any check, the fix is usually to split or merge the code — not to invent a new suffix.

### Automating these checks

Two of these rules are mechanical enough to check automatically; neither is automated yet:

1. **Missing-test pair** — every `backend/app/services/<module>.py` must have a matching `backend/tests/test_<module>.py`. Exceptions are only allowed for files with a leading underscore (shared fragments) or listed in an allowlist.
2. **Banned-suffix grep** — fail if any file in `backend/app/services/` matches `_builder|_factory|_creator|_helper|_util|_handler|_processor`. The `agents/` and `prompts/` folders are excluded.

---

## When conventions conflict

If you cannot match any existing pattern:

1. **Check if the file is doing too much.** If two roles fit equally, the file probably has two responsibilities — split it.
2. **`_service` is the honest fallback** when multiple small operations genuinely share a data type and no single operation dominates (see `dataset_service.py`). Justify in the PR description.
3. **Don't fork the vocabulary for one file.** Either the role belongs on the table (add it in a separate PR) or it doesn't apply.
4. **Propose new roles via PR to this doc.** Bar: at least two current files benefit, no existing role fits, and you can write a one-sentence deterministic test for reviewers.
