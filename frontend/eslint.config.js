// @ts-check
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ESLint flat config.
 *
 * The load-bearing rule here is `local/no-hardcoded-colors`: this app is
 * fully themable, so colors must come from semantic CSS-variable-backed
 * Tailwind tokens (bg-card, text-muted-foreground, …) — never raw palette
 * classes (bg-blue-500), color literals (#fff, rgb(), oklch()), or
 * text/bg-white/black. Files that legitimately produce concrete colors
 * (theme source, Plotly/Monaco/Prism, theme-preview swatches) are exempted
 * in the final config block.
 */
import tseslint from "typescript-eslint";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";

/** Banned className patterns. */
const PALETTE =
  "(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)";
// Color-bearing utility prefixes. Deliberately excludes gradient/position
// prefixes (from/via/to) since those collide with non-color utilities.
const PREFIX =
  "(?:text|bg|border|ring|fill|stroke|outline|decoration|divide|accent|caret|placeholder|shadow)";
const rePalette = new RegExp(`\\b${PREFIX}-${PALETTE}-\\d{2,3}\\b`);
const reBW = /\b(?:text|bg|border|fill|stroke|ring)-(?:white|black)(?:\/\d{1,3})?\b/;
const reHex = /#[0-9a-fA-F]{3,8}\b/;
const reFunc = /\b(?:rgb|rgba|hsl|hsla|oklch|oklab|lab|lch)\(/;

/**
 * Scan a className string literal and report any banned color usage.
 *
 * @param {import("eslint").Rule.RuleContext} context
 * @param {import("estree").Node} node
 * @param {string} value
 */
function checkClassName(context, node, value) {
  let msg = null;
  if (rePalette.test(value)) msg = "raw Tailwind palette color";
  else if (reBW.test(value)) msg = "text/bg-white|black (use a theme token)";
  else if (reHex.test(value)) msg = "hex color literal";
  else if (reFunc.test(value)) msg = "color function literal (rgb/hsl/oklch)";
  if (msg) {
    context.report({
      node,
      message: `Hardcoded color in className (${msg}). Use a semantic theme token (e.g. bg-card, text-muted-foreground, bg-primary). See docs/frontend-design-system.md.`,
    });
  }
}

/** The custom rule. */
const noHardcodedColors = {
  meta: {
    type: "problem",
    docs: { description: "Disallow hardcoded colors in className." },
    schema: [],
  },
  /** @param {import("eslint").Rule.RuleContext} context */
  create(context) {
    /** True if this attribute/prop is a className/class. */
    const isClassName = (name) => name === "className" || name === "class";

    /** Recursively scan an expression for string literals (cn(...), ternaries, templates). */
    function scanExpr(expr) {
      if (!expr) return;
      switch (expr.type) {
        case "Literal":
          if (typeof expr.value === "string") checkClassName(context, expr, expr.value);
          break;
        case "TemplateLiteral":
          // Static chunks AND interpolated expressions (e.g. a ternary that
          // picks between color classes inside `${...}`).
          expr.quasis.forEach((q) => checkClassName(context, q, q.value.raw));
          expr.expressions.forEach(scanExpr);
          break;
        case "ConditionalExpression":
          scanExpr(expr.consequent);
          scanExpr(expr.alternate);
          break;
        case "LogicalExpression":
          scanExpr(expr.left);
          scanExpr(expr.right);
          break;
        case "CallExpression": // cn("...", cond && "...")
          expr.arguments.forEach(scanExpr);
          break;
        case "ArrayExpression":
          expr.elements.forEach(scanExpr);
          break;
        default:
          break;
      }
    }

    return {
      // <div className="...">
      JSXAttribute(node) {
        if (!node.name || !isClassName(node.name.name)) return;
        const v = node.value;
        if (!v) return;
        if (v.type === "Literal" && typeof v.value === "string") {
          checkClassName(context, v, v.value);
        } else if (v.type === "JSXExpressionContainer") {
          scanExpr(v.expression);
        }
      },
    };
  },
};

export default tseslint.config(
  // Build output, deps, generated typings, and the Node-side theme script are
  // out of scope — this config exists for the themability guardrail on the app.
  { ignores: ["dist", "node_modules", "src/lib/monaco-editor.d.ts", "scripts/**"] },
  {
    files: ["src/**/*.{ts,tsx}"],
    // Report unused disable directives now that the react-hooks rules are ON:
    // a stale `eslint-disable react-hooks/*` is a claim that a violation exists
    // there, and a claim that has stopped being true should be deleted rather
    // than left to mislead the next reader.
    linterOptions: { reportUnusedDisableDirectives: true },
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: { ecmaFeatures: { jsx: true } },
      globals: { ...globals.browser },
    },
    plugins: {
      local: { rules: { "no-hardcoded-colors": noHardcodedColors } },
      "react-hooks": reactHooks,
    },
    rules: {
      // Deliberately narrow — we do NOT adopt the recommended JS/TS rulesets
      // (tsc already gates real errors via the build). The rules below are the
      // ones that catch defects types cannot see.
      "local/no-hardcoded-colors": "error",

      // rules-of-hooks is ERROR: a conditional or out-of-order hook call
      // corrupts React's hook ordering and produces state bleeding between
      // unrelated components — a class of bug that is very hard to diagnose from
      // symptoms and impossible for tsc to see. Enabling it cost nothing: the
      // tree had ZERO violations, so this is pure ratchet.
      "react-hooks/rules-of-hooks": "error",

      // exhaustive-deps is WARN, not error, on purpose. Most of its findings are
      // real (stale closures, identity churn) but a minority are deliberate
      // omissions that are correct — re-running an effect on a 3s poll's fresh
      // array identity would be worse than the stale read. Those are annotated
      // with a targeted eslint-disable and a reason, and
      // reportUnusedDisableDirectives above keeps those annotations honest.
      // Error would force either a bad refactor or a blanket suppression.
      "react-hooks/exhaustive-deps": "warn",
    },
  },
  // Documented intentional exceptions — see docs/frontend-design-system.md.
  {
    files: [
      "src/lib/themes.ts",
      "src/lib/color-utils.ts",
      "src/lib/chart-layout.ts",
      "src/lib/monaco-theme.ts",
      "src/components/assistant-ui/syntax-highlighter.tsx",
      "src/components/ui/plot.tsx",
      "src/pages/SettingsPage.tsx",
    ],
    rules: { "local/no-hardcoded-colors": "off" },
  },
);
