import { readFileSync } from "node:fs";

import { beforeEach, describe, expect, it, vi } from "vitest";

function readThemeInit(): string {
  return readFileSync("public/theme-init.js", "utf8");
}

function readThemeStorageKey(): string {
  const hookSource = readFileSync("src/hooks/useTheme.ts", "utf8");
  const match = /const THEME_STORAGE_KEY = "([^"]+)";/.exec(hookSource);
  if (!match) {
    throw new Error("Could not find THEME_STORAGE_KEY in useTheme.ts");
  }
  return match[1];
}

function runThemeInit(script: string, prefersLight: boolean): void {
  window.matchMedia = vi.fn().mockReturnValue({ matches: prefersLight });
  new Function(script)();
}

describe("theme init script", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
    document.documentElement.style.colorScheme = "";
  });

  it("loads before the React module script", () => {
    const indexHtml = readFileSync("index.html", "utf8");
    const initScriptIndex = indexHtml.indexOf('<script src="/theme-init.js"></script>');
    const reactModuleIndex = indexHtml.indexOf(
      '<script type="module" src="/src/main.tsx"></script>',
    );

    expect(initScriptIndex).toBeGreaterThan(-1);
    expect(reactModuleIndex).toBeGreaterThan(-1);
    expect(initScriptIndex).toBeLessThan(reactModuleIndex);
  });

  it("uses the same localStorage key as useTheme", () => {
    const themeInit = readThemeInit();
    const storageKey = readThemeStorageKey();

    expect(themeInit).toContain(`getItem("${storageKey}")`);
  });

  it("applies a stored theme synchronously before React boots", () => {
    const themeInit = readThemeInit();
    localStorage.setItem("elspeth_theme", "light");

    runThemeInit(themeInit, false);

    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(document.documentElement.style.colorScheme).toBe("light");
  });

  it("resolves system theme from prefers-color-scheme when no explicit theme is stored", () => {
    const themeInit = readThemeInit();

    runThemeInit(themeInit, true);

    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(document.documentElement.style.colorScheme).toBe("light");
  });
});
