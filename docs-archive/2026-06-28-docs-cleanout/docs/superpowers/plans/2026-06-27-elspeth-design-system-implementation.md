# ELSPETH Design System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the imported "ELSPETH Design System" into the repo — install it as a Skill, build a typed React primitive library (proven by migrating LoginPage), tokenize the font-weight/wordmark values, and stand up the marketing website — all in one worktree off `release/0.7.0`.

**Architecture:** The design system is a faithful mirror of the current frontend, so the token/CSS layer is already in place. The new primitives are thin typed wrappers over the existing `shared.css` classes (additive, no visual change). The website is decoupled static HTML on the token CSS. Source spec: `docs/superpowers/specs/2026-06-27-elspeth-design-system-implementation-design.md`.

**Tech Stack:** React 18 (automatic JSX runtime), TypeScript (strict), Vite, Vitest + @testing-library/react (+ jest-dom matchers). Static HTML/CSS + Lucide (CDN) for the website.

## Global Constraints

- **Worktree:** all work lands on branch `design-system-impl` in `/home/john/elspeth/.worktrees/design-system-impl`; merge `--no-ff` to `release/0.7.0` at the end.
- **No `import React`:** the frontend uses the automatic JSX runtime (see `LoginPage.tsx`). Import only types/hooks from `"react"` (e.g. `import type { ButtonHTMLAttributes, ReactNode } from "react"`). Do NOT copy the DS `.jsx` files' `import React from "react"` line.
- **Barrel imports:** consumers import primitives from `components/ui` (the `index.ts` barrel), never from a component file directly (design-system adherence rule).
- **No token-value changes:** the frontend already matches the design. Task "C" is purely additive (new tokens) + mechanical (replace literals with `var(...)`). Visual output must be unchanged.
- **No emoji** in UI; functional unicode glyphs only (`⚠` `∅`), monochrome, inheriting colour.
- **Port source (in-repo after Task 2):** `.claude/skills/elspeth-design/` holds the DS `.jsx`/`.d.ts`/`.css` to port from. (Pre-Task-2 mirror: `…/scratchpad/elspeth-design-system-latest/`.)
- **Frontend dir:** `src/elspeth/web/frontend` (abbreviated `FE/` below). Run all `npm` commands from there.

---

### Task 1: Worktree prep — install frontend deps, confirm green baseline

**Files:** none changed (environment only).

- [ ] **Step 1: Install frontend deps** (gitignored, absent in a fresh worktree)

Run (from the worktree): `cd src/elspeth/web/frontend && npm ci`
(Alternative if `npm ci` is slow: `ln -s /home/john/elspeth/src/elspeth/web/frontend/node_modules src/elspeth/web/frontend/node_modules` — same `release/0.7.0` lockfile, so safe.)

- [ ] **Step 2: Confirm the suite runs green before any change**

Run: `npm run test && npm run typecheck && npm run lint && npm run lint:css`
Expected: all PASS. (Establishes that later failures are ours.)

No commit (environment setup).

---

### Task 2: A — Install the design system as a Skill (also stages the port source)

**Files:**
- Create: `.claude/skills/elspeth-design/**` (copied from the latest mirror)

**Interfaces:**
- Produces: the DS source tree in-repo (`.claude/skills/elspeth-design/components/**`, `.../ui_kits/website/**`, `.../styles.css`, `.../tokens/**`) used as the port source by Tasks 4–9.

- [ ] **Step 1: Copy the latest mirror verbatim into the skill dir**

Run:
```bash
mkdir -p .claude/skills/elspeth-design
cp -R /tmp/claude-1000/-home-john-elspeth/5153a90c-c45c-4d20-aedf-aa324416684a/scratchpad/elspeth-design-system-latest/. .claude/skills/elspeth-design/
# drop the analysis-only artifact if present
rm -f .claude/skills/elspeth-design/ANALYSIS.md
```

- [ ] **Step 2: Verify the skill is well-formed**

Run: `head -5 .claude/skills/elspeth-design/SKILL.md && ls .claude/skills/elspeth-design/`
Expected: YAML front-matter (`name: elspeth-design`, `user-invocable: true`) and the full tree (`tokens/ components/ guidelines/ ui_kits/ assets/ styles.css SKILL.md readme.md`).

- [ ] **Step 3: Confirm no repo lint tree-walks into `.claude/`** (reality check)

Run: `git -C /home/john/elspeth grep -n "\.claude" -- .pre-commit-config.yaml package.json src/elspeth/web/frontend/package.json 2>/dev/null; echo "checked"`
Expected: no lint/oxlint rule includes `.claude/` (the bundled `_adherence.oxlintrc.json` is inert here). If something does, scope it out.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/elspeth-design
git commit -m "feat(design-system): install ELSPETH design system as a skill"
```

---

### Task 3: C — Tokenize font weights + wordmark tracking

**Files:**
- Modify: `FE/src/styles/tokens.css` (add 5 tokens to `:root`)
- Modify: `FE/src/styles/common.css:83,133,289,373`
- Modify: `FE/src/styles/shared.css:22,80,171,216,295,377,514,533,551`
- Modify: `FE/src/styles/base.css:109`
- Modify: `FE/src/components/header/header.css:86`

**Interfaces:**
- Produces: CSS vars `--font-weight-regular|medium|semibold|bold`, `--tracking-wordmark`.

- [ ] **Step 1: Add the tokens to `tokens.css` `:root`** (place in the Typography block, after the line-height vars, before `/* ── Spacing ── */`)

```css
  /* ── Font weights ────────────────────────────────────────────────────────── */
  --font-weight-regular:  400;
  --font-weight-medium:   500;
  --font-weight-semibold: 600;
  --font-weight-bold:     700;

  /* ── Brand wordmark tracking — JetBrains Mono, uppercase ─────────────────── */
  --tracking-wordmark: 0.18em;
```
Keep `:root` and `[data-theme="light"]` as top-level selectors (parsed by the style tests).

- [ ] **Step 2: Replace the hardcoded weights** — at each line above, map by value:
`font-weight: 400` → `var(--font-weight-regular)`; `500` → `var(--font-weight-medium)`; `600` → `var(--font-weight-semibold)`; `700` → `var(--font-weight-bold)`. (The pinned lines are all `600` except `shared.css:80,171,216,295` which are `500`.)

- [ ] **Step 3: Replace the wordmark tracking** in `header.css:86`: `letter-spacing: 0.18em;` → `letter-spacing: var(--tracking-wordmark);`

- [ ] **Step 4: Verify no behavioural/visual change + lint**

Run: `cd src/elspeth/web/frontend && npm run test -- src/styles && npm run lint:css`
Expected: `colorContrast.test.ts`, `statusBadgeAccessibility.test.ts`, `themeInit.test.ts` PASS; stylelint clean. (Weights are colour-independent; tests unaffected.)

- [ ] **Step 5: Confirm zero remaining hardcoded weights in the touched files**

Run: `rg -n 'font-weight:\s*[0-9]' src/elspeth/web/frontend/src/styles/{common,shared,base}.css; echo done`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/styles/tokens.css src/elspeth/web/frontend/src/styles/common.css src/elspeth/web/frontend/src/styles/shared.css src/elspeth/web/frontend/src/styles/base.css src/elspeth/web/frontend/src/components/header/header.css
git commit -m "refactor(styles): tokenize font weights and wordmark tracking"
```

---

### Task 4: B — Scaffold `components/ui/` + Button (archetype) + barrel

**Files:**
- Create: `FE/src/components/ui/Button.tsx`
- Create: `FE/src/components/ui/Button.test.tsx`
- Create: `FE/src/components/ui/index.ts`

**Interfaces:**
- Produces: `Button` (the pattern every later primitive follows), barrel `components/ui`.

- [ ] **Step 1: Write the failing test** — `FE/src/components/ui/Button.test.tsx`

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Button } from "./Button";

describe("Button", () => {
  it("renders the base .btn class, label, and type=button", () => {
    render(<Button>Run</Button>);
    const btn = screen.getByRole("button", { name: "Run" });
    expect(btn).toHaveClass("btn");
    expect(btn).toHaveAttribute("type", "button");
  });
  it("applies the variant modifier (secondary = no modifier)", () => {
    render(<Button variant="primary">Go</Button>);
    expect(screen.getByRole("button", { name: "Go" })).toHaveClass("btn", "btn-primary");
  });
  it("uses the standalone compact base class", () => {
    render(<Button compact>x</Button>);
    const b = screen.getByRole("button", { name: "x" });
    expect(b).toHaveClass("btn-compact");
    expect(b).not.toHaveClass("btn");
  });
  it("forwards className and DOM props", () => {
    render(<Button className="x" disabled aria-label="lbl" />);
    const b = screen.getByRole("button", { name: "lbl" });
    expect(b).toHaveClass("x");
    expect(b).toBeDisabled();
  });
});
```

- [ ] **Step 2: Run it to verify it fails** — `cd src/elspeth/web/frontend && npm run test -- src/components/ui/Button` → FAIL (`Button` not found).

- [ ] **Step 3: Implement `FE/src/components/ui/Button.tsx`**

```tsx
import type { ButtonHTMLAttributes, ReactNode } from "react";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual style. @default "secondary" */
  variant?: "primary" | "secondary" | "danger" | "ghost";
  /** Use the 36px chrome-row size instead of the 44px default. @default false */
  compact?: boolean;
  iconLeft?: ReactNode;
  iconRight?: ReactNode;
}

export function Button({
  variant = "secondary",
  compact = false,
  type = "button",
  iconLeft,
  iconRight,
  className = "",
  children,
  ...rest
}: ButtonProps) {
  const base = compact ? "btn-compact" : "btn";
  const variantClass =
    variant === "primary" ? "btn-primary"
    : variant === "danger" ? "btn-danger"
    : variant === "ghost" ? "btn-ghost"
    : "";
  const cls = [base, variantClass, className].filter(Boolean).join(" ");
  return (
    <button type={type} className={cls} {...rest}>
      {iconLeft}
      {children}
      {iconRight}
    </button>
  );
}
```

- [ ] **Step 4: Create the barrel** — `FE/src/components/ui/index.ts`

```ts
export { Button } from "./Button";
export type { ButtonProps } from "./Button";
```

- [ ] **Step 5: Run tests + typecheck** — `npm run test -- src/components/ui/Button && npm run typecheck` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/ui/
git commit -m "feat(ui): add Button primitive + components/ui barrel"
```

---

### Task 5: B — Badge primitives: TypeBadge, StatusBadge

**Files:** Create `FE/src/components/ui/{TypeBadge,StatusBadge}.tsx` + `.test.tsx`; extend `index.ts`.
**Port source:** `.claude/skills/elspeth-design/components/core/{TypeBadge,StatusBadge}.{jsx,d.ts}`
**Pattern:** follow Task 4 (no `import React`; forward `...rest`/`className`).

**TypeBadge** — `interface TypeBadgeProps extends HTMLAttributes<HTMLSpanElement> { type?: "source"|"transform"|"gate"|"sink"|"aggregation"|"coalesce"; children?: ReactNode }`. Renders `<span className="type-badge type-badge-{type}">{children ?? type}</span>`, `type` default `"source"`.

**StatusBadge** — `status?: "pending"|"running"|"completed"|"completed_with_failures"|"failed"|"empty"|"cancelled"|"cancelling"` (default `"pending"`). Colour-class maps `completed_with_failures→completed`, `cancelling→cancelled`, else identity → `status-badge status-badge-{colorKey}`. Glyph map `{completed_with_failures:"⚠", empty:"∅"}` rendered as a leading `<span aria-hidden="true">`. Label = `children ?? status`.

```tsx
// StatusBadge.tsx (full — non-obvious mapping)
import type { HTMLAttributes, ReactNode } from "react";

const GLYPH: Partial<Record<string, string>> = {
  completed_with_failures: "⚠",
  empty: "∅",
};

export interface StatusBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  status?:
    | "pending" | "running" | "completed" | "completed_with_failures"
    | "failed" | "empty" | "cancelled" | "cancelling";
  children?: ReactNode;
}

export function StatusBadge({ status = "pending", className = "", children, ...rest }: StatusBadgeProps) {
  const colorKey =
    status === "completed_with_failures" ? "completed"
    : status === "cancelling" ? "cancelled"
    : status;
  const cls = ["status-badge", `status-badge-${colorKey}`, className].filter(Boolean).join(" ");
  const glyph = GLYPH[status];
  return (
    <span className={cls} {...rest}>
      {glyph ? <span aria-hidden="true">{glyph}</span> : null}
      {children ?? status}
    </span>
  );
}
```

- [ ] **Step 1: Failing tests** — `TypeBadge.test.tsx`: renders `type-badge type-badge-gate`, default label = type name, `children` overrides. `StatusBadge.test.tsx`: `completed_with_failures` → has class `status-badge-completed` AND renders `⚠`; `empty` → `status-badge-empty` AND `∅`; `cancelling` → `status-badge-cancelled`; `running` → `status-badge-running`, no glyph.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement both** (TypeBadge per spec above; StatusBadge per the full code above).
- [ ] **Step 4: Extend `index.ts`** with `TypeBadge`/`StatusBadge` (+ their prop types).
- [ ] **Step 5: Run tests + typecheck → PASS.**
- [ ] **Step 6: Commit** — `feat(ui): add TypeBadge and StatusBadge primitives`.

---

### Task 6: B — Form primitives: Input, Textarea

**Files:** Create `FE/src/components/ui/{Input,Textarea}.tsx` + `.test.tsx`; extend `index.ts`.
**Port source:** `.claude/skills/elspeth-design/components/forms/{Input,Textarea}.{jsx,d.ts}`

**Input** — `interface InputProps extends InputHTMLAttributes<HTMLInputElement> { label?: ReactNode; hint?: ReactNode; mono?: boolean }`. Renders (when `label` or `hint` present) a wrapper with a `.field-label` associated to the control and a `.field-hint`; the control is `<input className={["input", mono && "input-mono", className]}>`. **Accessibility requirement:** the label must be associated with the input — use the passed `id` for `htmlFor`/`id`; if no `id` is supplied, generate one with React `useId()`. Forward all DOM props (`value`, `onChange`, `required`, `autoComplete`, `type`, …).

**Textarea** — `interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> { label?: ReactNode; hint?: ReactNode }`. Same label/hint/`useId` pattern; control `<textarea className="textarea">`.

- [ ] **Step 1: Failing tests** — `Input.test.tsx`: renders an `<input>` with class `input`; `label="Username"` → `getByLabelText("Username")` returns the input (association works); `mono` → `input-mono`; forwards `value`/`onChange`/`required`. `Textarea.test.tsx`: renders `textarea` class; label association via `getByLabelText`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement both** (port from the `.jsx`; add `useId()` for label association as above).
- [ ] **Step 4: Extend `index.ts`.**
- [ ] **Step 5: Run tests + typecheck → PASS.**
- [ ] **Step 6: Commit** — `feat(ui): add Input and Textarea form primitives`.

---

### Task 7: B — Container/feedback primitives: Card (+CardHeader), AlertBanner, Tabs, WordMark

**Files:** Create `FE/src/components/ui/{Card,AlertBanner,Tabs,WordMark}.tsx` + `.test.tsx`; extend `index.ts`.
**Port source:** `.claude/skills/elspeth-design/components/core/{Card,AlertBanner,Tabs}.{jsx,d.ts}` and `.../components/composer/WordMark.{jsx,d.ts}`

**Card** — `interface CardProps extends HTMLAttributes<HTMLDivElement> { paper?: boolean; pad?: boolean }` (`pad` default true). `<div className={["card", paper && "card-paper", className]} style={pad === false ? { padding: 0 } : undefined}>`. Also export `CardHeader({ title, actions, eyebrow }: { title: ReactNode; actions?: ReactNode; eyebrow?: ReactNode })` — port the markup from `Card.jsx` (eyebrow = small uppercase label, title row with right-aligned actions).

**AlertBanner** — `interface AlertBannerProps extends Omit<HTMLAttributes<HTMLDivElement>,"role"> { tone?: "error"|"warning"|"info"|"success"; action?: ReactNode; role?: string }` (`tone` default `"error"`). Class `alert-banner` + `alert-banner--{tone}` for non-error tones (error = base). **ARIA:** default `role` = `"alert"` for error, `"status"` otherwise, overridable. Render children, then right-aligned `action` if present.

**Tabs** — `interface TabItem { id: string; label: ReactNode; count?: number }`; `interface TabsProps extends Omit<HTMLAttributes<HTMLDivElement>,"onChange"> { tabs: TabItem[]; value: string; onChange?: (id:string)=>void }`. Port `Tabs.jsx` verbatim (logic preserved): `role="tablist"`, each `<button role="tab" aria-selected>` with `tab-strip-tab`/`tab-strip-tab-active`, optional count pill.

**WordMark** — `interface WordMarkProps extends HTMLAttributes<HTMLElement> { size?: number|string; as?: keyof JSX.IntrinsicElements }` (`size` default 13, `as` default `"span"`). Renders the chosen element with the JetBrains-Mono / uppercase / `var(--tracking-wordmark)` styling and the literal text `ELSPETH`. Port styling from `WordMark.jsx`; use `--tracking-wordmark` (now a token from Task 3).

- [ ] **Step 1: Failing tests** — `Card.test.tsx`: `card` class; `paper` → `card-paper`; `pad={false}` → inline `padding:0`. `AlertBanner.test.tsx`: default tone error → `role="alert"` + `alert-banner`; `tone="warning"` → `alert-banner--warning` + `role="status"`; renders `action`. `Tabs.test.tsx`: renders a tab per item; active tab has `aria-selected=true` + `tab-strip-tab-active`; clicking a tab calls `onChange` with its id; `count` renders. `WordMark.test.tsx`: renders text `ELSPETH`; `as="h1"` → an `<h1>`; applies the wordmark letter-spacing.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement the four** (port per specs above).
- [ ] **Step 4: Extend `index.ts`** (now exports all 9 primitives + `CardHeader` + types).
- [ ] **Step 5: Run the full `components/ui` suite + typecheck → PASS** — `npm run test -- src/components/ui && npm run typecheck`.
- [ ] **Step 6: Commit** — `feat(ui): add Card, AlertBanner, Tabs, WordMark primitives`.

---

### Task 8: B — Exemplar migration: LoginPage adopts the primitives

**Files:**
- Modify: `FE/src/components/auth/LoginPage.tsx`
- Modify: `FE/src/components/auth/LoginPage.test.tsx`

**Interfaces:** Consumes `Button`, `Input`, `AlertBanner` from `components/ui`.

**Note:** `LoginPage` currently uses bespoke inline `style={{}}` for its error box, inputs, and buttons. Migration replaces those with the primitives — a deliberate, minor normalization onto the canonical button/input styling (same tokens, slightly different metrics e.g. 44px button height, 6px radius). The outer centered card + `<h1>Sign in to ELSPETH</h1>` stay as-is (out of primitive scope). `WordMark`/`TypeBadge`/`StatusBadge`/`Card`/`Tabs` are validated by their unit tests, not by this exemplar.

- [ ] **Step 1: Add a failing test** to `LoginPage.test.tsx` — resolve local-auth config and assert the form renders via primitives:

```tsx
it("renders the local-auth form with labelled inputs and a sign-in button", async () => {
  vi.mocked(api.fetchAuthConfig).mockResolvedValue({
    provider: "local", oidc_issuer: null, oidc_client_id: null, authorization_endpoint: null,
  });
  render(<LoginPage />);
  expect(await screen.findByLabelText("Username")).toBeInTheDocument();
  expect(screen.getByLabelText("Password")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run → FAIL** (current inputs aren't associated via `getByLabelText` the same way / button name differs). `npm run test -- src/components/auth/LoginPage`

- [ ] **Step 3: Migrate `LoginPage.tsx`** — import `{ Button, Input, AlertBanner } from "../ui"`. Replace:
  - the `loginError` `<div role="alert" style=…>` → `<AlertBanner tone="error">{loginError}</AlertBanner>`;
  - the two `<label>+<input style=…>` blocks → `<Input label="Username" id="login-username" type="text" autoComplete="username" required value={username} onChange={…} />` and the password equivalent;
  - the SSO `<button style=…>` → `<Button variant="primary" type="button" onClick={handleSsoRedirect} aria-label="Sign in with single sign-on">Sign in with SSO</Button>`;
  - the submit `<button style=…>` → `<Button variant="primary" type="submit" disabled={isSubmitting} aria-label={isSubmitting ? "Signing in" : "Sign in"}>{isSubmitting ? "Signing in…" : "Sign in"}</Button>`.
  Keep the centered wrapper, the card `<div>`, the `<h1>`, and the `configLoading` spinner block unchanged (the existing loading test must still pass).

- [ ] **Step 4: Run LoginPage tests → PASS** (both the existing loading test and the new form test). `npm run test -- src/components/auth/LoginPage`

- [ ] **Step 5: Typecheck + lint** — `npm run typecheck && npm run lint`.

- [ ] **Step 6: Commit** — `refactor(auth): migrate LoginPage to ui primitives (exemplar)`.

---

### Task 9: D — Marketing website at `website/`

**Files:**
- Create: `website/{index,authoring,assurance,use-cases,get-started}.html`, `website/site.css`, `website/site.js`
- Create: `website/styles.css`, `website/tokens/{fonts,colors,typography,layout,base,primitives}.css`
- Create: `website/README.md`

**Port source:** `.claude/skills/elspeth-design/ui_kits/website/*` and `.claude/skills/elspeth-design/{styles.css,tokens/}`

- [ ] **Step 1: Copy the website + its token CSS into `website/`**

```bash
mkdir -p website/tokens
cp .claude/skills/elspeth-design/ui_kits/website/* website/
cp .claude/skills/elspeth-design/styles.css website/styles.css
cp .claude/skills/elspeth-design/tokens/*.css website/tokens/
```

- [ ] **Step 2: Rewrite the token CSS link** in all 5 pages (`../../styles.css` → `styles.css`)

```bash
sed -i 's#href="\.\./\.\./styles\.css"#href="styles.css"#' website/*.html
rg -n 'rel="stylesheet"' website/*.html   # verify: each links styles.css + site.css
```

- [ ] **Step 3: Add `website/README.md`** documenting the surface + the token-duplication note:

```markdown
# ELSPETH marketing website

Standalone static site built on the ELSPETH design tokens. No build step.
Serve `website/` with any static server (or GitHub Pages); open `index.html`.

Pages: index (Home), authoring, assurance, use-cases, get-started.
Shared: `site.css`, `site.js` (Lucide icons + light/dark toggle). Lucide loads
from the unpkg CDN.

**Token source:** `website/tokens/` + `website/styles.css` are a static MIRROR of
the frontend's canonical tokens (`src/elspeth/web/frontend/src/styles/tokens.css`).
They are identical today; if the frontend tokens change, re-copy them here.
```

- [ ] **Step 4: Smoke test — internal links + assets resolve**

Run:
```bash
cd website
node -e '
const fs=require("fs");
const pages=["index","authoring","assurance","use-cases","get-started"].map(p=>p+".html");
let bad=[];
for (const f of pages){
  const h=fs.readFileSync(f,"utf8");
  for (const m of h.matchAll(/(?:href|src)="([^"#:]+\.(?:html|css|js))"/g)){
    if(!fs.existsSync(m[1])) bad.push(f+" -> "+m[1]);
  }
}
if(bad.length){console.error("BROKEN:",bad);process.exit(1);}
console.log("all internal links + assets resolve ("+pages.length+" pages)");
'
```
Expected: "all internal links + assets resolve (5 pages)".

- [ ] **Step 5: Commit**

```bash
git add website
git commit -m "feat(website): add ELSPETH marketing site (static, design-token based)"
```

---

### Task 10: Whole-suite verification + merge

**Files:** none (verification + merge).

- [ ] **Step 1: Full frontend gate** — `cd src/elspeth/web/frontend && npm run test && npm run typecheck && npm run lint && npm run lint:css` → all PASS.
- [ ] **Step 2: Confirm working tree clean** — `git -C /home/john/elspeth/.worktrees/design-system-impl status --short` → empty.
- [ ] **Step 3: Review the branch diff** — `git -C /home/john/elspeth/.worktrees/design-system-impl log --oneline release/0.7.0..HEAD` (expect: spec, skill, tokens, 4× ui, LoginPage, website).
- [ ] **Step 4: Merge `--no-ff` to `release/0.7.0`** (from the main checkout, after confirming with the operator):

```bash
git -C /home/john/elspeth checkout release/0.7.0   # NOTE: operator may have uncommitted WIP — coordinate first
git -C /home/john/elspeth merge --no-ff design-system-impl -m "merge: ELSPETH design system implementation (skill + ui primitives + tokens + website)"
```
> The exemplar (LoginPage) + additive tokens don't touch the front-page/composer files under active edit, so this should be conflict-free. **Merging is an operator-gated step** — surface the diff and confirm before checkout/merge, since `release/0.7.0` may carry uncommitted WIP.

- [ ] **Step 5: Clean up the worktree** (after merge) — `git -C /home/john/elspeth worktree remove .worktrees/design-system-impl`.

---

## Self-review (against the spec)

**Spec coverage:** A → Task 2; B library → Tasks 4–7 (all 9 primitives); B exemplar → Task 8; C → Task 3; D → Task 9; worktree/deps/5a checks → Tasks 1, 2(step 3), 9(step 2); verify+merge → Task 10. No gaps.

**Placeholder scan:** every code step shows real code; component ports name an exact in-repo source file + reproduce the contract + list exact classes — no "TBD"/"similar to"/"add error handling".

**Type consistency:** prop interfaces match the DS `.d.ts` (Button `variant`/`compact`; StatusBadge 8 statuses → 6 colour classes via the documented map; Tabs `TabItem`/`value`/`onChange`; Card `paper`/`pad`; AlertBanner `tone`/`action`/`role`; Input/Textarea `label`/`hint`/`mono`; WordMark `size`/`as`). Barrel exports them all; LoginPage consumes `Button`/`Input`/`AlertBanner` (defined in Tasks 4/6/7, before Task 8).
