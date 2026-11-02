# Frontend Design System

This app is **fully themable**: all visuals derive from semantic CSS-variable-
backed Tailwind tokens defined in `frontend/src/lib/themes.ts` and applied at
runtime by `theme-provider.tsx`. **Never hardcode a color.** A build-time
ESLint rule (`local/no-hardcoded-colors` in `frontend/eslint.config.js`)
enforces this — `npm run build` fails on a violation.

## Token vocabulary

| Need | Token(s) |
|------|----------|
| Page background | `bg-background` |
| Raised surface (card, panel, dropdown) | `bg-card` / `bg-popover` |
| Secondary/muted surface | `bg-muted`, `bg-secondary`, `bg-accent` |
| Primary text | `text-foreground` |
| Secondary text | `text-muted-foreground` |
| Brand / primary action | `bg-primary text-primary-foreground` |
| Focus ring (controls) | `focus-visible:ring-3 focus-visible:ring-ring/50` |
| Focus ring (rows, cards) | `focus-visible:ring-2 focus-visible:ring-ring` (rows often `ring-1 ring-inset`) |
| Borders & dividers | `border-border` |
| Status (data) | `text-success` / `warning` / `error` / `info` (+ `/10` tints) |
| Destructive **affordance** | `destructive` (Button `variant="danger"` solid; `variant="destructive"` soft) |
| Charts | `chart-1..5`, via `useChartTheme()` |
| Heatmap / data viz spectrum | `data-good/fair/neutral/warn/bad`, `data-scale-low/mid/high` |
| Micro text (~10px) | `text-2xs` (never `text-[10px]`) |
| Elevation | `shadow-sm` (cards), `shadow-md` (popovers/dropdowns), `shadow-lg` (modals) |
| Modal scrim | `bg-overlay` |

## Shared primitives — prefer these over hand-rolling

Hand-rolled interactive elements were the single biggest source of missing
hover/focus states. Use these so the focus contract comes for free:

- `Button` (`components/ui/button.tsx`) — variants: `default`, `outline`,
  `secondary`, `ghost`, `link`, `destructive` (soft danger), `danger` (solid
  danger), `success`, `warning`, `info`. Icon buttons: `size="icon-sm"` etc.
- `Badge` — variants mirror Button's status set (`success`/`warning`/`error`/
  `info`/`destructive`).
- `Tabs` / `TabsList` / `TabsTrigger` — full tab bars with the raised active pill.
- `SegmentedControl` (`components/ui/segmented-control.tsx`) — compact
  single-select view switch with a raised active pill.
- `SelectableRow` + `clickableCardClass` + `onActivateKeyDown`
  (`components/ui/interactive.tsx`) — keyboard-operable list rows and clickable
  cards (use `clickableCardClass` for cards that contain nested buttons).

## Button variant rule (apply everywhere — keeps buttons consistent)

One hierarchy, picked by the button's role — never by aesthetics:

- **Primary CTA** (the main action of a view/dialog/bar) → `default` (solid primary). One per context.
- **Secondary action** beside a primary, or a destructive-confirm alternative → `secondary` (solid grey). Pairs with a primary as a matched set (e.g. Resume = primary, Start New = secondary).
- **Tertiary / toolbar / utility / row / icon actions** (export, clear, expand, copy, download-utility, nav icons) → `ghost`.
- **Export/download utility buttons in a toolbar** → `outline` (CSV/JSON/Export/Download-harness all match).
- **Dialog "Cancel"** → `outline`.
- **Solid destructive confirm** (e.g. dialog "Delete") → `danger`. Inline/soft danger → `destructive`.
- **Toggles** (Light/Dark, segmented): active = `default`/filled, inactive = `outline`/`ghost` — never both solid.

**Every non-disabled interactive control must have a *visibly* distinct hover** — not a near-identical opacity nudge. Solid-fill variants hover to a different token (`secondary`/`outline` → `bg-accent`), not `bg-self/80` (imperceptible in dark mode). Soft-tint variants step `/10`→`/20`. `default` darkens to `/80`.

## Tabs

`Tabs`/`TabsList`/`TabsTrigger`: the list is an inset "well" (`bg-background border`),
the active trigger a lighter `bg-accent` pill with `accent-foreground` + shadow + border
so it clearly stands out in dark mode (a `bg-muted` track sat too close to the pill's
lightness and looked flat). Height is `h-9` to match sibling bars (FilterBar, ActiveJobBar).
Spacing between the tab list and its content is owned by the `Tabs` root (`gap-4`) — do
**not** add `mt-*` on `TabsContent` (that double-stacks the gap).

## Rules

1. **Elevation:** raised surfaces = `bg-card border border-border shadow-sm`.
   Rings are reserved for focus/selection, never for resting elevation.
2. **Every interactive element** has hover + `focus-visible` + (where relevant)
   active/disabled/selected states. Prefer the primitives above — they carry the
   focus contract. A raw `<button>`/`<div onClick>` must add the focus ring
   itself.
3. **`destructive` vs `error`:** `destructive` is for interactive danger
   affordances (delete buttons, stop, destructive-confirm banners). `error`
   (and `success`/`warning`/`info`) is for **data/status** display.
4. **Status from data, not prose:** never infer status by matching English in
   output (e.g. `.includes("error")`). Read a structured flag — for chat tool
   UIs, the `isError` field threaded through `ToolUIProps` from the assistant-ui
   tool-call part.
5. **Type scale:** use `text-2xs / xs / sm / base / lg …`. No `text-[Npx]`.

## Intentional exceptions (exempt from the lint rule)

These legitimately use concrete colors and are allowlisted in
`eslint.config.js`:

- `lib/themes.ts` — the theme source of truth (oklch values).
- `lib/color-utils.ts`, `lib/chart-layout.ts` — Plotly palettes (Plotly can't
  consume oklch CSS vars), fed via `useChartTheme()`. **Includes the heatmap
  red→orange→yellow→green spectrum** — this is deliberate and must stay vivid &
  theme-independent.
- `lib/monaco-theme.ts` — Monaco editor theme (third-party).
- `components/assistant-ui/syntax-highlighter.tsx` — Prism (third-party).
- `components/ui/plot.tsx` — resolves theme CSS vars at render (`useThemeLayout`)
  and passes concrete strings to Plotly, tracking all themes.
- `pages/SettingsPage.tsx` — theme-preview swatches (dynamic per-theme data).

## Adding a theme

Every theme is one of tweakcn's built-in presets, which are Apache-2.0
licensed: utils/theme-presets.ts in the
[tweakcn repository](https://github.com/jnsahaj/tweakcn), and config/theme.ts
there for "Default". To add one, copy the preset's light and dark
values into `themes.ts`, converted to oklch. tweakcn fills any token a preset
leaves out from its default theme, so do the same, or the theme renders with
gaps. Ensure it spreads `...SHADOWS` (or overrides) so elevation tokens
resolve. Every key a theme sets must be listed in `THEME_VARS`.

Do not take themes from tweakcn's community gallery. They carry no license we
have been able to verify, so none may ship in this repository without legal
review. Code copied from any other source needs its license added to
[`THIRD-PARTY-LICENSES`](../THIRD-PARTY-LICENSES) first.

A theme's fonts are part of the theme. Install the font's `@fontsource`
package (most are under the SIL Open Font License), import it in `frontend/src/index.css`, and name the
family in the theme's `font-sans`, `font-mono` or `font-serif`. The
`@fontsource-variable` packages register the family as `'<Family> Variable'`,
so that is the name to use.
