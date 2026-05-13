import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const appCss = readFileSync("src/App.css", "utf8");

function extractLightThemeToken(tokenName: string): string {
  const blockMatch = /\[data-theme="light"\]\s*\{([\s\S]*?)\n\}/.exec(appCss);
  if (!blockMatch) {
    throw new Error("Could not find light theme token block in App.css");
  }

  const tokenMatch = new RegExp(`${tokenName}:\\s*(#[0-9a-fA-F]{6})\\s*;`).exec(blockMatch[1]);
  if (!tokenMatch) {
    throw new Error(`Could not find ${tokenName} in light theme token block`);
  }

  return tokenMatch[1];
}

function channelToLinear(value: number): number {
  const normalized = value / 255;
  if (normalized <= 0.04045) {
    return normalized / 12.92;
  }
  return ((normalized + 0.055) / 1.055) ** 2.4;
}

function relativeLuminance(hex: string): number {
  const match = /^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/.exec(hex);
  if (!match) {
    throw new Error(`Expected six-digit hex colour, got ${hex}`);
  }

  const red = channelToLinear(Number.parseInt(match[1], 16));
  const green = channelToLinear(Number.parseInt(match[2], 16));
  const blue = channelToLinear(Number.parseInt(match[3], 16));
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(foreground: string, background: string): number {
  const foregroundLuminance = relativeLuminance(foreground);
  const backgroundLuminance = relativeLuminance(background);
  const lighter = Math.max(foregroundLuminance, backgroundLuminance);
  const darker = Math.min(foregroundLuminance, backgroundLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

describe("light theme colour contrast", () => {
  it("keeps muted body text at WCAG AA contrast against the light background", () => {
    const background = extractLightThemeToken("--color-bg");
    const mutedText = extractLightThemeToken("--color-text-muted");

    expect(contrastRatio(mutedText, background)).toBeGreaterThanOrEqual(4.5);
  });
});
