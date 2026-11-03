// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Configure Monaco to load from the bundled ``monaco-editor`` npm package
 * instead of the default CDN (``cdn.jsdelivr.net``).
 *
 * Two reasons we can't use the default CDN:
 *
 * 1. CSP. Our CloudFront edge attaches ``script-src 'self'`` (see
 *    ``infra/cdk/frontend_stack.py``); the CDN script load is blocked
 *    by the browser before it ever fires. The visible symptom is
 *    ``Monaco initialization: error`` in the console followed by
 *    every editor instance failing to mount.
 * 2. Hermetic deploys. An internal-only deploy shouldn't depend on
 *    a third-party CDN being reachable from the user's browser.
 *
 * Importing the bare ``monaco-editor`` package would pull every
 * basic-language tokenizer (~85 of them, ~470 emitted chunks). Use
 * the slim ``editor.api`` core instead and contribute only the
 * languages the SPA actually requests. The allowlist is derived
 * from the backend's ``EXTENSION_LANGUAGE_MAP`` (``backend/app/api/files.py``)
 * intersected with the SPA's ``BACKEND_TO_MONACO`` table
 * (``monaco-language.ts``, used by ``CodeTab.tsx`` and ``FilePreview.tsx``)
 * plus the hardcoded ``language=`` props in ``PromptsPanel`` and
 * ``EvalDashboard``.
 * Anything outside the intersection falls through to ``plaintext``
 * which Monaco renders without a contribution.
 *
 * Loaded once at app entry from ``main.tsx``. Idempotent — reentry
 * is a no-op because ``loader.config`` just sets a module-level
 * config object the loader reads on first ``loader.init()``.
 */

/* Specifiers here drop the ``esm/vs/`` prefix the older layout used.
 * monaco-editor ships an ``exports`` map that rewrites ``monaco-editor/<p>``
 * to ``./esm/vs/<p>.js``, so a deep path that repeats ``esm/vs`` resolves to
 * ``esm/vs/esm/vs/...`` and fails to resolve at build time. */
import { loader } from "@monaco-editor/react";
// Slim editor core — does NOT bring the basic-language registry with it.
import * as monaco from "monaco-editor/editor/editor.api";

/* Editor features: hover, find, the context menu, quick fixes, code lens and
 * the rest, with the services each one depends on. The slim core above has
 * none of them.
 *
 * This must load here, before any editor exists. The standalone editor fixes
 * its set of services when the first editor is created, but keeps attaching
 * contributions registered later to every new editor. Left alone, this module
 * arrives lazily with the JSON language service when the first JSON model
 * appears, so every editor after that point gets contributions whose services
 * are missing and throws "depends on UNKNOWN service" on mount.
 *
 * It is the module the language services load, rather than a hand-written list
 * of contributions, so the set registered here is the set that would otherwise
 * arrive late. ``monaco-setup.test.ts`` fails if anything still does. */
import "monaco-editor/internal/common/workers";

/* Workers: each Monaco language service has its own dedicated web
 * worker. Vite's ``?worker`` query suffix compiles each one into a
 * same-origin chunk we wire up via ``MonacoEnvironment``.
 *
 * Only JSON gets a full language service (lint, validate,
 * autocomplete) and its own worker. Every other language uses the
 * editor's generic worker for tokenization only, which is all the
 * SPA's read-only viewers of .py / .yaml / .toml / .md / .sh need. */
import EditorWorker from "monaco-editor/editor/editor.worker?worker";
import JsonWorker from "monaco-editor/language/json/json.worker?worker";

/* Language tokenizers. Allowlist only — explicit imports are what makes Vite
 * tree-shake everything else.
 *
 * These are the per-language ``register`` modules, NOT
 * ``basic-languages/monaco.contribution``: that barrel imports all ~90
 * languages, which is the whole registry this file exists to avoid pulling in.
 *
 * ``toml`` is mapped to ``ini`` upstream (see ``BACKEND_TO_MONACO`` in
 * monaco-language.ts) because Monaco doesn't ship a TOML tokenizer.
 * ``csv`` and ``text`` map to ``plaintext`` upstream — no
 * contribution needed. ``jsx``/``tsx`` aren't in the upstream
 * mapping; they fall through to ``plaintext`` today. */
import "monaco-editor/languages/definitions/python/register";
import "monaco-editor/languages/definitions/markdown/register";
import "monaco-editor/languages/definitions/yaml/register";
import "monaco-editor/languages/definitions/ini/register";
import "monaco-editor/languages/definitions/javascript/register";
import "monaco-editor/languages/definitions/typescript/register";
import "monaco-editor/languages/definitions/html/register";
import "monaco-editor/languages/definitions/css/register";
import "monaco-editor/languages/definitions/shell/register";

/* JSON gets the full language service (lint + format); the matching
 * worker is already imported above. */
import "monaco-editor/language/json/monaco.contribution";

declare global {
  interface Window {
    MonacoEnvironment?: { getWorker(workerId: string, label: string): Worker };
  }
}

self.MonacoEnvironment = {
  getWorker(_workerId: string, label: string): Worker {
    if (label === "json") return new JsonWorker();
    return new EditorWorker();
  },
};

/* ``loader.config`` is typed against the FULL monaco module, which carries the
 * ``css``/``html``/``json``/``typescript``/``lsp`` language-service namespaces.
 * The slim ``editor.api`` entry above is the same file at runtime but a
 * different module identity to TypeScript, so it does not pick up that
 * declaration and appears to be missing those namespaces.
 *
 * Widening the import to satisfy the type would pull in every language service,
 * which is the bundle weight this whole file exists to avoid. The loader only
 * stores this object and hands it back to ``@monaco-editor/react``, and the
 * namespaces it does not find are ones nothing here calls — JSON's service is
 * registered by its own contribution import above. */
loader.config({ monaco: monaco as unknown as Parameters<typeof loader.config>[0]["monaco"] });
