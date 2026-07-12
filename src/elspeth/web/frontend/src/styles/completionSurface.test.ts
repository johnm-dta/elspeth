import { existsSync, readFileSync } from "node:fs";

describe("guided completion surface", () => {
  it("lets expanded Run Results fill downward while collapsed results stay compact", () => {
    const css = readFileSync("src/components/chat/chat.css", "utf8");
    const rule = css.match(
      /\.chat-panel--completed\s*>\s*\.inline-run-results:not\(\.inline-run-results--collapsed\)\s*\{(?<body>[^}]*)\}/s,
    );

    expect(rule?.groups?.body).toContain("flex: 1 1 auto");
    expect(rule?.groups?.body).toContain("min-height: 6rem");
    expect(rule?.groups?.body).toContain("max-height: none");
    expect(rule?.groups?.body).toContain("margin-top: var(--space-xs)");
    expect(rule?.groups?.body).toContain("margin-bottom: 0");
  });

  it("lets the YAML preview yield height on wide, short completed pages", () => {
    const css = readFileSync("src/components/chat/guided/guided.css", "utf8");
    const completionRule = css.match(
      /\.chat-panel--completed\s*>\s*\.guided-completion\s*\{(?<body>[^}]*)\}/s,
    );
    const yamlRule = css.match(
      /\.chat-panel--completed\s*>\s*\.guided-completion\s+\.guided-completion-yaml-container\s*\{(?<body>[^}]*)\}/s,
    );

    expect(completionRule?.groups?.body).toContain("min-height: 18rem");
    expect(completionRule?.groups?.body).toContain("overflow: hidden");
    expect(yamlRule?.groups?.body).toContain("flex: 1 1 auto");
    expect(yamlRule?.groups?.body).toContain("min-height: 0");
  });

  it("declares the ELSPETH SVG favicon as a same-origin public asset", () => {
    const indexHtml = readFileSync("index.html", "utf8");
    const faviconPath = "public/favicon.svg";

    expect(indexHtml).toContain(
      '<link rel="icon" href="/favicon.svg" type="image/svg+xml" />',
    );
    expect(existsSync(faviconPath)).toBe(true);
    if (!existsSync(faviconPath)) return;

    const favicon = readFileSync(faviconPath, "utf8");
    expect(favicon).toContain('viewBox="0 0 32 32"');
    expect(favicon).toContain('fill="#10272e"');
    expect(favicon).toContain('fill="#68d4df"');
  });
});
