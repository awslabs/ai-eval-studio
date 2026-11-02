// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified by Amazon. Original source: https://github.com/assistant-ui/assistant-ui
// Portions Copyright (c) 2026 AgentbaseAI Inc., MIT License. See THIRD-PARTY-LICENSES.

/**
 * Prism syntax highlighter wired for the assistant-ui MarkdownText.
 *
 * Two styles are instantiated (oneLight / oneDark) and the right one is
 * picked based on the current theme mode. Code blocks are readable in
 * both light and dark — the single fixed ``oneDark`` default produced
 * dark-on-light text in light mode.
 *
 * Language registrations are opt-in to keep the bundle small; blocks
 * in an unregistered language render as plain text.
 *
 * Imports ``PrismLight`` directly from ``react-syntax-highlighter``
 * instead of going through ``@assistant-ui/react-syntax-highlighter``.
 * Why: that wrapper's barrel re-exports four highlighter variants
 * (``PrismAsyncLight``, ``LightAsync``, ``PrismLight``, ``Light``).
 * Two of them (``PrismAsyncLight`` and ``LightAsync``) drag in
 * dynamic-import language registries (``async-languages/prism.js``
 * and ``async-languages/hljs.js``) covering every refractor + every
 * highlight.js language. Vite eagerly emits each dynamic-import
 * target as a separate chunk, so even though our code only uses
 * ``PrismLight``, the build produces ~490 single-language ``.js``
 * files in ``dist/assets/``. The BucketDeployment Lambda then has
 * to upload them all, which thrashes its default 128 MB memory and
 * times out the deploy.
 *
 * The wrapper's ``makePrismLightSyntaxHighlighter`` factory is
 * inlined below — it's just a 5-line adapter mapping assistant-ui's
 * ``{Pre, Code, language, code}`` props to ``react-syntax-highlighter``'s
 * ``{PreTag, CodeTag, language, children}`` shape. Inlining costs
 * nothing and severs the dependency that was pulling in the
 * problematic variants.
 *
 * Container uses ``background: transparent`` so the <pre> styling from
 * ``markdown-text.tsx`` (``bg-muted/40``) shows through and matches the
 * theme.
 */
import type { ComponentProps, FC } from "react";
import { PrismLight } from "react-syntax-highlighter";

import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import tsx from "react-syntax-highlighter/dist/esm/languages/prism/tsx";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";

import {
  oneDark,
  oneLight,
} from "react-syntax-highlighter/dist/cjs/styles/prism";

import { useTheme } from "@/lib/theme-provider";

PrismLight.registerLanguage("bash", bash);
PrismLight.registerLanguage("sh", bash);
PrismLight.registerLanguage("shell", bash);
PrismLight.registerLanguage("js", tsx);
PrismLight.registerLanguage("jsx", tsx);
PrismLight.registerLanguage("ts", tsx);
PrismLight.registerLanguage("tsx", tsx);
PrismLight.registerLanguage("json", json);
PrismLight.registerLanguage("markdown", markdown);
PrismLight.registerLanguage("md", markdown);
PrismLight.registerLanguage("python", python);
PrismLight.registerLanguage("py", python);
PrismLight.registerLanguage("yaml", yaml);
PrismLight.registerLanguage("yml", yaml);

const commonCustomStyle: React.CSSProperties = {
  margin: 0,
  width: "100%",
  background: "transparent",
  padding: 0,
  fontSize: "inherit",
};

/** Props assistant-ui's MarkdownText passes to a syntax highlighter. */
interface AuiHighlighterProps {
  components: {
    Pre: React.ElementType;
    Code: React.ElementType;
  };
  language: string;
  code: string;
}

/** Build an assistant-ui-shaped highlighter wrapping a configured PrismLight. */
function makeHighlighter(
  config: Omit<ComponentProps<typeof PrismLight>, "children" | "language" | "PreTag" | "CodeTag">,
): FC<AuiHighlighterProps> {
  const Impl: FC<AuiHighlighterProps> = ({ components: { Pre, Code }, language, code }) => (
    <PrismLight
      {...config}
      PreTag={Pre}
      CodeTag={Code}
      language={language}
    >
      {code}
    </PrismLight>
  );
  Impl.displayName = "PrismSyntaxHighlighter";
  return Impl;
}

const DarkSyntaxHighlighter = makeHighlighter({
  style: oneDark,
  customStyle: commonCustomStyle,
});

const LightSyntaxHighlighter = makeHighlighter({
  style: oneLight,
  customStyle: commonCustomStyle,
});

/**
 * Theme-aware Prism highlighter. Reads the current mode from
 * ``useTheme`` and delegates to the appropriate pre-built highlighter.
 */
export const SyntaxHighlighter: FC<AuiHighlighterProps> = (props) => {
  const { mode } = useTheme();
  const Impl = mode === "dark" ? DarkSyntaxHighlighter : LightSyntaxHighlighter;
  return <Impl {...props} />;
};
