/**
 * Theme management hook.
 *
 * Handles theme state with the following priority:
 * 1. Manual override persisted in localStorage (`elspeth_theme`)
 * 2. OS preference via prefers-color-scheme media query
 * 3. Default to dark theme
 *
 * The hook applies `data-theme` attribute to document.documentElement,
 * which CSS selectors use to override the default dark theme tokens.
 */

import { useState, useEffect, useCallback, useMemo } from "react";

const THEME_STORAGE_KEY = "elspeth_theme";
const THEME_CHANGE_EVENT = "elspeth-theme-change";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

interface UseThemeReturn {
  /** Current theme setting: "light", "dark", or "system" */
  theme: Theme;
  /** Resolved actual theme being displayed */
  resolvedTheme: ResolvedTheme;
  /** Set the theme (persists to localStorage) */
  setTheme: (theme: Theme) => void;
  /** Toggle between light and dark (skips system) */
  toggleTheme: () => void;
}

/**
 * Detect OS color scheme preference.
 * Falls back to "dark" if matchMedia is not available (SSR, test environments).
 */
function getSystemTheme(): ResolvedTheme {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return "dark";
  }
  return window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

/**
 * Load theme preference from localStorage.
 */
function loadStoredTheme(): Theme | null {
  if (typeof window === "undefined") return null;
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === "light" || stored === "dark" || stored === "system") {
    return stored;
  }
  return null;
}

/**
 * Apply theme to document by setting data-theme attribute.
 * Always sets the resolved theme so CSS only needs [data-theme="light"]
 * and :root (dark) selectors — no @media duplication needed.
 */
function applyThemeToDocument(_theme: Theme, resolvedTheme: ResolvedTheme): void {
  if (typeof document === "undefined") return;

  document.documentElement.setAttribute("data-theme", resolvedTheme);
  document.documentElement.style.colorScheme = resolvedTheme;
}

export function useTheme(): UseThemeReturn {
  // Initialize from localStorage or default to "system"
  const [theme, setThemeState] = useState<Theme>(() => {
    return loadStoredTheme() ?? "system";
  });

  // Track system preference for resolving "system" theme
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(getSystemTheme);

  // Resolve the actual displayed theme
  const resolvedTheme: ResolvedTheme = useMemo(() => {
    return theme === "system" ? systemTheme : theme;
  }, [theme, systemTheme]);

  // Listen for OS theme changes
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }

    const mediaQuery = window.matchMedia("(prefers-color-scheme: light)");

    const handleChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? "light" : "dark");
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  // Synchronize theme across all hook instances via custom event
  useEffect(() => {
    if (typeof window === "undefined") return;

    const handleThemeChange = (e: Event) => {
      const newTheme = (e as CustomEvent<Theme>).detail;
      setThemeState(newTheme);
    };

    window.addEventListener(THEME_CHANGE_EVENT, handleThemeChange);
    return () => window.removeEventListener(THEME_CHANGE_EVENT, handleThemeChange);
  }, []);

  // Apply theme to document whenever it changes
  useEffect(() => {
    applyThemeToDocument(theme, resolvedTheme);
  }, [theme, resolvedTheme]);

  // Persist theme to localStorage and broadcast to all hook instances
  const setTheme = useCallback((newTheme: Theme) => {
    localStorage.setItem(THEME_STORAGE_KEY, newTheme);
    setThemeState(newTheme);
    // Notify other mounted useTheme() consumers
    window.dispatchEvent(new CustomEvent(THEME_CHANGE_EVENT, { detail: newTheme }));
  }, []);

  // Toggle between light and dark (useful for simple toggle button)
  const toggleTheme = useCallback(() => {
    setTheme(resolvedTheme === "dark" ? "light" : "dark");
  }, [resolvedTheme, setTheme]);

  return { theme, resolvedTheme, setTheme, toggleTheme };
}
