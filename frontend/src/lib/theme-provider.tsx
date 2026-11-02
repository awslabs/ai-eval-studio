// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Theme provider -- manages active theme + light/dark mode.
 *
 * Applies CSS variables to document.documentElement and toggles
 * the `.dark` class. Persists selection in localStorage.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  themes,
  resolveThemeVars,
  DEFAULT_THEME,
  DEFAULT_MODE,
  type ThemeDefinition,
} from "./themes";

type Mode = "light" | "dark";

/** Shape of the theme context exposed to consumers. */
interface ThemeContextValue {
  /** Active theme ID (key in themes map). */
  themeId: string;
  /** Light or dark mode. */
  mode: Mode;
  /** The active theme definition. */
  theme: ThemeDefinition;
  /** Switch to a different theme. */
  setThemeId: (id: string) => void;
  /** Switch light/dark mode. */
  setMode: (mode: Mode) => void;
  /** Toggle between light and dark. */
  toggleMode: () => void;
  /**
   * Resolve a CSS variable to its computed value.
   * Useful for libraries like Recharts that need raw color strings.
   */
  resolveVar: (varName: string) => string;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

const LS_THEME_KEY = "eval-studio-theme";
const LS_MODE_KEY = "eval-studio-mode";

/** Read stored theme ID or fall back to default. */
function getStoredTheme(): string {
  const stored = localStorage.getItem(LS_THEME_KEY);
  return stored && stored in themes ? stored : DEFAULT_THEME;
}

/** Read the stored mode, or fall back to DEFAULT_MODE. */
function getStoredMode(): Mode {
  const stored = localStorage.getItem(LS_MODE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return DEFAULT_MODE;
}

/** Apply CSS variables from a theme definition to document root. */
function applyTheme(themeId: string, mode: Mode) {
  const root = document.documentElement;
  // Single source of truth for the resolved variable map (see
  // resolveThemeVars in themes.ts).
  const vars = resolveThemeVars(themeId, mode);

  for (const [key, value] of Object.entries(vars)) {
    root.style.setProperty(`--${key}`, value);
  }

  if (mode === "dark") {
    root.classList.add("dark");
  } else {
    root.classList.remove("dark");
  }
}

/** Provider that manages theme and mode state for the entire app. */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [themeId, setThemeIdState] = useState(getStoredTheme);
  const [mode, setModeState] = useState(getStoredMode);

  const setThemeId = useCallback((id: string) => {
    if (!(id in themes)) return;
    localStorage.setItem(LS_THEME_KEY, id);
    setThemeIdState(id);
  }, []);

  const setMode = useCallback((m: Mode) => {
    localStorage.setItem(LS_MODE_KEY, m);
    setModeState(m);
  }, []);

  const toggleMode = useCallback(() => {
    setModeState((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      localStorage.setItem(LS_MODE_KEY, next);
      return next;
    });
  }, []);

  const resolveVar = useCallback((varName: string) => {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(`--${varName}`)
      .trim();
  }, []);

  // Apply theme whenever themeId or mode changes.
  useEffect(() => {
    applyTheme(themeId, mode);
  }, [themeId, mode]);

  const value: ThemeContextValue = {
    themeId,
    mode,
    theme: themes[themeId],
    setThemeId,
    setMode,
    toggleMode,
    resolveVar,
  };

  return <ThemeContext value={value}>{children}</ThemeContext>;
}

/** Access the theme context. Must be used within ThemeProvider. */
export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
