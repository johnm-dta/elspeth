# DTA Theme Retheme — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retheme the ELSPETH web frontend from the current purple-blue dark palette to the Digital Transformation Agency (DTA) of Australia's green/teal identity, fixing accessibility violations and token architecture issues in the process.

**Architecture:** The design tokens live in CSS custom properties in `App.css` `:root`. Components reference tokens via `var(--token-name)` in inline styles, except React Flow (GraphView) which requires raw hex values. A new `tokens.ts` file provides typed JS constants for contexts where CSS variables cannot be used. The retheme introduces `--color-accent` to separate brand accent from `--color-focus-ring` (keyboard-only), and semantic background tokens (`--color-error-bg` etc.) to eliminate hardcoded rgba duplicates.

**Tech Stack:** React 18, CSS custom properties, TypeScript, @xyflow/react (React Flow), Inter font (Google Fonts via CDN).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/elspeth/web/frontend/src/App.css` | Modify | Replace `:root` palette, add semantic bg tokens, update `.tab-strip-tab-active`, `.btn-primary`, `.skip-to-content`, `.component-card-dimmed` |
| `src/elspeth/web/frontend/src/styles/tokens.ts` | Create | Typed JS constants for badge/edge colors used by React Flow |
| `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx` | Modify | Import from `tokens.ts` instead of hardcoded hex maps |
| `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx` | Modify | Replace `--color-focus-ring` → `--color-accent` for Send button, use `--color-error-bg/border` tokens |
| `src/elspeth/web/frontend/src/components/auth/LoginPage.tsx` | Modify | Replace `--color-focus-ring` → `--color-accent` for Sign-in buttons, use `--color-error-bg/border` tokens, fix text color to `--color-text-inverse` |
| `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx` | Modify | Replace `--color-focus-ring` → `--color-accent` for active indicator and new-session button |
| `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx` | Modify | Replace `--color-focus-ring` → `--color-accent` for SELECTED badge, use `--color-warning-bg/border` tokens |
| `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx` | Modify | Use `--color-error-bg/border` and `--color-warning-bg/border` tokens |
| `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` | Modify | Use `--color-error-bg` token |
| `src/elspeth/web/frontend/src/App.tsx` | Modify | Use `--color-error-bg/border` tokens for system-unavailable banner |
| `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx` | Modify | Use `--color-success-bg` token for copied state |
| `src/elspeth/web/frontend/index.html` | Modify | Add Inter font preconnect/stylesheet link |

---

## Task 1: Add Inter Font and Replace `:root` Palette

**Files:**
- Modify: `src/elspeth/web/frontend/index.html`
- Modify: `src/elspeth/web/frontend/src/App.css:1-110`

- [ ] **Step 1: Add Inter font link to index.html**

Open `src/elspeth/web/frontend/index.html` and add the Google Fonts preconnect and stylesheet link inside `<head>`, before any existing stylesheets:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
```

- [ ] **Step 2: Replace the entire `:root` block in App.css**

Replace lines 11–110 (the full `:root { ... }` block) with the DTA-themed palette below. This is a complete replacement — do not merge, overwrite the entire block:

```css
:root {
  /* ── Backgrounds — DTA deep teal family ──────────────────────────────────── */
  --color-bg:                #0f2d35;
  --color-surface:           #122f37;
  --color-surface-sidebar:   #0c2830;
  --color-surface-inspector: #112e36;
  --color-surface-elevated:  #1a3d47;
  --color-surface-hover:     rgba(255, 255, 255, 0.04);

  /* ── Text ────────────────────────────────────────────────────────────────── */
  --color-text:              #dff0ee;
  --color-text-secondary:    #8db8b8;
  --color-text-muted:        #6a9898;
  --color-text-inverse:      #ffffff;

  /* ── Borders — teal-tinted ───────────────────────────────────────────────── */
  --color-border:            rgba(143, 200, 200, 0.12);
  --color-border-strong:     rgba(143, 200, 200, 0.25);

  /* ── Chat bubbles ────────────────────────────────────────────────────────── */
  --color-bubble-user:           rgba(40, 130, 100, 0.14);
  --color-bubble-user-border:    rgba(40, 130, 100, 0.35);
  --color-bubble-assistant:      rgba(255, 255, 255, 0.05);
  --color-bubble-assistant-border: rgba(255, 255, 255, 0.10);
  --color-bubble-system:         rgba(255, 255, 255, 0.03);

  /* ── Component type badges — functional colour coding ────────────────────── */
  --color-badge-source:      #4db89a;
  --color-badge-transform:   #e8a030;
  --color-badge-gate:        #c390f9;
  --color-badge-sink:        #e07040;

  /* ── Semantic colours — DTA/GOLD dark theme ──────────────────────────────── */
  --color-success:           #14b0ae;
  --color-error:             #e85653;
  --color-warning:           #e38444;
  --color-info:              #61daff;

  /* ── Semantic backgrounds — for inline alert/banner components ────────────── */
  --color-success-bg:        rgba(20, 176, 174, 0.12);
  --color-success-border:    rgba(20, 176, 174, 0.30);
  --color-error-bg:          rgba(232, 86, 83, 0.12);
  --color-error-border:      rgba(232, 86, 83, 0.30);
  --color-warning-bg:        rgba(227, 132, 68, 0.14);
  --color-warning-border:    rgba(227, 132, 68, 0.30);
  --color-info-bg:           rgba(97, 218, 255, 0.10);
  --color-info-border:       rgba(97, 218, 255, 0.25);

  /* ── Interactive / accent ────────────────────────────────────────────────── */
  --color-accent:            #29b480;
  --color-focus-ring:        #c390f9;
  --color-btn-primary-bg:    #1c7c58;
  --color-btn-primary-bg-hover: #206850;
  --color-link:              #61daff;
  --color-highlight:         rgba(41, 180, 128, 0.10);
  --color-selected-ring:     rgba(41, 180, 128, 0.50);

  /* ── Status badges ───────────────────────────────────────────────────────── */
  --color-status-pending:    #7a9a9a;
  --color-status-running:    #61daff;
  --color-status-completed:  #14b0ae;
  --color-status-failed:     #e85653;
  --color-status-cancelled:  #e38444;

  /* ── Dimming ─────────────────────────────────────────────────────────────── */
  --opacity-dimmed: 0.35;

  /* ── Scrollbar — teal-tinted ─────────────────────────────────────────────── */
  --color-scrollbar-track:       transparent;
  --color-scrollbar-thumb:       rgba(143, 200, 200, 0.15);
  --color-scrollbar-thumb-hover: rgba(143, 200, 200, 0.28);

  /* ── Sizing ──────────────────────────────────────────────────────────────── */
  --sidebar-expanded-width:  200px;
  --sidebar-collapsed-width: 40px;
  --inspector-default-width: 320px;
  --inspector-min-width:     240px;

  /* ── Typography — Inter for body, system fallback ────────────────────────── */
  --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Oxygen, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
  --font-mono: "JetBrains Mono", "Fira Code", "SF Mono", "Cascadia Code",
    "Source Code Pro", Menlo, Monaco, Consolas, "Liberation Mono", monospace;

  --font-size-xs:   12px;
  --font-size-sm:   13px;
  --font-size-base: 14px;
  --font-size-lg:   16px;
  --font-size-xl:   18px;

  --line-height-tight:   1.3;
  --line-height-normal:  1.5;
  --line-height-relaxed: 1.7;

  /* ── Spacing ─────────────────────────────────────────────────────────────── */
  --space-xs:  4px;
  --space-sm:  8px;
  --space-md:  12px;
  --space-lg:  16px;
  --space-xl:  24px;
  --space-2xl: 32px;

  /* ── Radius ──────────────────────────────────────────────────────────────── */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;

  /* ── Transitions ─────────────────────────────────────────────────────────── */
  --transition-fast:   100ms ease;
  --transition-normal: 150ms ease;
  --transition-slow:   250ms ease;
}
```

- [ ] **Step 3: Update type badge CSS classes to use new badge colour values**

The `.type-badge-source`, `.type-badge-transform`, `.type-badge-gate`, `.type-badge-sink` classes in App.css use hardcoded rgba values that correspond to the old badge colours. Update them to derive from the new badge tokens.

Replace these four class blocks (lines ~389-414) with:

```css
/* Source — aqua-green */
.type-badge-source {
  background-color: rgba(77, 184, 154, 0.15);
  color: var(--color-badge-source);
  border: 1px solid rgba(77, 184, 154, 0.3);
}

/* Transform — amber */
.type-badge-transform {
  background-color: rgba(232, 160, 48, 0.15);
  color: var(--color-badge-transform);
  border: 1px solid rgba(232, 160, 48, 0.3);
}

/* Gate — purple */
.type-badge-gate {
  background-color: rgba(195, 144, 249, 0.15);
  color: var(--color-badge-gate);
  border: 1px solid rgba(195, 144, 249, 0.3);
}

/* Sink — orange-red */
.type-badge-sink {
  background-color: rgba(224, 112, 64, 0.15);
  color: var(--color-badge-sink);
  border: 1px solid rgba(224, 112, 64, 0.3);
}
```

- [ ] **Step 4: Update status badge CSS classes**

Replace the five `.status-badge-*` blocks (lines ~431-454) with values derived from the new status tokens:

```css
.status-badge-pending {
  background-color: rgba(122, 154, 154, 0.15);
  color: var(--color-status-pending);
}

.status-badge-running {
  background-color: rgba(97, 218, 255, 0.15);
  color: var(--color-status-running);
}

.status-badge-completed {
  background-color: rgba(20, 176, 174, 0.15);
  color: var(--color-status-completed);
}

.status-badge-failed {
  background-color: rgba(232, 86, 83, 0.15);
  color: var(--color-status-failed);
}

.status-badge-cancelled {
  background-color: rgba(227, 132, 68, 0.15);
  color: var(--color-status-cancelled);
}
```

- [ ] **Step 5: Update `.tab-strip-tab-active` to use `--color-accent`**

In the `.tab-strip-tab-active` rule (line ~576-579), change `border-bottom-color` from `var(--color-focus-ring)` to `var(--color-accent)`:

```css
.tab-strip-tab-active {
  color: var(--color-text);
  border-bottom-color: var(--color-accent);
}
```

- [ ] **Step 6: Update `.btn-primary` to use accent/DTA green tokens**

Replace the `.btn-primary` and `.btn-primary:hover` rules (lines ~529-537):

```css
.btn-primary {
  background-color: rgba(20, 176, 174, 0.2);
  border-color: var(--color-success);
  color: var(--color-success);
}

.btn-primary:hover:not(:disabled) {
  background-color: rgba(20, 176, 174, 0.3);
}
```

- [ ] **Step 7: Update `.btn-danger` to use new error colour**

Replace the `.btn-danger` and `.btn-danger:hover` rules (lines ~539-547):

```css
.btn-danger {
  background-color: var(--color-error-bg);
  border-color: var(--color-error);
  color: var(--color-error);
}

.btn-danger:hover:not(:disabled) {
  background-color: rgba(232, 86, 83, 0.25);
}
```

- [ ] **Step 8: Update `.validation-banner-pass` and `.validation-banner-fail`**

Replace lines ~625-635:

```css
.validation-banner-pass {
  background-color: var(--color-success-bg);
  border: 1px solid var(--color-success-border);
  color: var(--color-success);
}

.validation-banner-fail {
  background-color: var(--color-error-bg);
  border: 1px solid var(--color-error-border);
  color: var(--color-error);
}
```

- [ ] **Step 9: Update `.skip-to-content` to fix contrast**

Replace the `.skip-to-content` rule (lines ~208-221) — use `--color-accent` as background with `--color-text-inverse` for guaranteed contrast:

```css
.skip-to-content {
  position: absolute;
  top: -100%;
  left: 0;
  z-index: 1000;
  padding: var(--space-sm) var(--space-lg);
  background-color: var(--color-accent);
  color: var(--color-text-inverse);
  font-size: var(--font-size-base);
  font-weight: 600;
  text-decoration: none;
  border-radius: 0 0 var(--radius-md) 0;
}
```

- [ ] **Step 10: Update `.component-card-dimmed` to use surface token**

Replace the hardcoded rgba in `.component-card-dimmed` (line ~481):

```css
.component-card-dimmed {
  background-color: rgba(26, 61, 71, var(--opacity-dimmed));
}
```

(The rgb values `26, 61, 71` are the decimal components of `#1a3d47` = `--color-surface-elevated`.)

- [ ] **Step 11: Verify the build compiles**

Run:
```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit
```

Expected: 0 errors. (This step validates that no CSS variable names we removed were referenced in TypeScript type definitions.)

- [ ] **Step 12: Commit**

```bash
git add src/elspeth/web/frontend/index.html src/elspeth/web/frontend/src/App.css
git commit -m "feat(web/frontend): retheme to DTA palette — deep teal backgrounds, GOLD semantic colours, Inter font

Replace purple-blue palette with DTA/AGDS-derived deep teal backgrounds,
green accent (#29b480), GOLD purple focus ring (#c390f9), and Inter body font.
Add semantic background tokens (--color-error-bg, --color-warning-bg, etc.)
and separate --color-accent from --color-focus-ring."
```

---

## Task 2: Create `tokens.ts` — Typed JS Mirror of Badge Colours

**Files:**
- Create: `src/elspeth/web/frontend/src/styles/tokens.ts`

- [ ] **Step 1: Create the styles directory and tokens file**

Create `src/elspeth/web/frontend/src/styles/tokens.ts`:

```typescript
/**
 * Design tokens that must be used in JS contexts where CSS variables
 * cannot be resolved (e.g. React Flow inline node styles).
 *
 * These values MUST stay in sync with the :root block in App.css.
 * When changing App.css colour tokens, update this file in the same commit.
 */

// ── Component type badge colours ─────────────────────────────────────────────

export const BADGE_COLORS = {
  source: "#4db89a",
  transform: "#e8a030",
  gate: "#c390f9",
  sink: "#e07040",
} as const;

export const BADGE_BACKGROUNDS = {
  source: "rgba(77, 184, 154, 0.15)",
  transform: "rgba(232, 160, 48, 0.15)",
  gate: "rgba(195, 144, 249, 0.15)",
  sink: "rgba(224, 112, 64, 0.15)",
} as const;

// ── Edge colours ─────────────────────────────────────────────────────────────

export const EDGE_COLORS = {
  normal: "#6a9898",     // --color-text-muted
  error: "#e85653",      // --color-error
} as const;

export const EDGE_LABEL_COLOR = "#8db8b8"; // --color-text-secondary
```

- [ ] **Step 2: Commit**

```bash
git add src/elspeth/web/frontend/src/styles/tokens.ts
git commit -m "feat(web/frontend): add tokens.ts — typed JS mirror of CSS colour tokens for React Flow"
```

---

## Task 3: Migrate GraphView.tsx to Use `tokens.ts`

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx:16,29-53,129,132`

- [ ] **Step 1: Replace the hardcoded colour maps with imports from tokens.ts**

In `GraphView.tsx`, add the import at line 16 (after the existing imports):

```typescript
import { BADGE_COLORS, BADGE_BACKGROUNDS, EDGE_COLORS, EDGE_LABEL_COLOR } from "@/styles/tokens";
```

Then delete the three constant blocks `NODE_COLORS`, `NODE_BORDER_COLORS`, and `NODE_TEXT_COLORS` (lines 29–53, including the comment above them), and the comment on line 32.

- [ ] **Step 2: Update node style references in the `useMemo` block**

In the `rfNodes` mapping (around line 109–120), replace the three references:

Change `NODE_COLORS[node.type]` → `BADGE_BACKGROUNDS[node.type]`
Change `NODE_BORDER_COLORS[node.type]` → `BADGE_COLORS[node.type]`
Change `NODE_TEXT_COLORS[node.type]` → `BADGE_COLORS[node.type]`

The resulting style object should be:

```typescript
      style: {
        backgroundColor: BADGE_BACKGROUNDS[node.type],
        border: `2px solid ${BADGE_COLORS[node.type]}`,
        color: BADGE_COLORS[node.type],
        borderRadius: 6,
        padding: "8px 12px",
        fontSize: 12,
        fontWeight: 600,
        width: NODE_WIDTH,
        textAlign: "center" as const,
      },
```

- [ ] **Step 3: Update edge style references**

In the `rfEdges` mapping (around line 128–132), replace the hardcoded edge colours:

Change `"#f66"` → `EDGE_COLORS.error`
Change `"#999"` → `EDGE_COLORS.normal`
Change `"#b0b0c0"` → `EDGE_LABEL_COLOR`

The resulting style should be:

```typescript
      style: {
        stroke: edge.edge_type === "error" ? EDGE_COLORS.error : EDGE_COLORS.normal,
        strokeWidth: 1.5,
      },
      labelStyle: { fontSize: 10, fill: EDGE_LABEL_COLOR },
```

- [ ] **Step 4: Verify build**

Run:
```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit
```

Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/GraphView.tsx
git commit -m "refactor(web/frontend): GraphView uses tokens.ts instead of hardcoded hex colours"
```

---

## Task 4: Fix ChatInput.tsx — Accent Token and Semantic Backgrounds

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx:91-95,164-167`

- [ ] **Step 1: Replace hardcoded error rgba with semantic tokens**

In the upload error `<div>` (lines 91–95), change:

```typescript
            backgroundColor: "rgba(255, 102, 102, 0.12)",
            color: "var(--color-error)",
            borderRadius: 4,
            fontSize: 12,
            border: "1px solid rgba(255, 102, 102, 0.3)",
```

to:

```typescript
            backgroundColor: "var(--color-error-bg)",
            color: "var(--color-error)",
            borderRadius: 4,
            fontSize: 12,
            border: "1px solid var(--color-error-border)",
```

- [ ] **Step 2: Replace Send button focus-ring misuse with accent token**

In the Send button style (lines 164–167), change:

```typescript
            backgroundColor: canSend
              ? "var(--color-focus-ring)"
              : "var(--color-text-muted)",
            color: "var(--color-text)",
```

to:

```typescript
            backgroundColor: canSend
              ? "var(--color-accent)"
              : "var(--color-surface-elevated)",
            color: canSend
              ? "var(--color-text-inverse)"
              : "var(--color-text-muted)",
```

- [ ] **Step 3: Verify build**

Run:
```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/ChatInput.tsx
git commit -m "fix(web/frontend): ChatInput uses --color-accent for Send button, semantic error tokens"
```

---

## Task 5: Fix LoginPage.tsx — Accent Token, Contrast, Semantic Backgrounds

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/auth/LoginPage.tsx:142,162-166,183-184,268-271`

- [ ] **Step 1: Replace hardcoded box-shadow**

On line 142, change:

```typescript
          boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
```

to:

```typescript
          boxShadow: "0 2px 8px rgba(10, 40, 50, 0.4)",
```

- [ ] **Step 2: Replace hardcoded error rgba in login error banner**

In the login error `<div>` (lines 162–166), change:

```typescript
              backgroundColor: "rgba(255, 102, 102, 0.12)",
              color: "var(--color-error)",
              borderRadius: 4,
              fontSize: 14,
              border: "1px solid rgba(255, 102, 102, 0.3)",
```

to:

```typescript
              backgroundColor: "var(--color-error-bg)",
              color: "var(--color-error)",
              borderRadius: 4,
              fontSize: 14,
              border: "1px solid var(--color-error-border)",
```

- [ ] **Step 3: Fix SSO button — replace focus-ring with accent, fix text contrast**

In the SSO button style (lines 183–184), change:

```typescript
              backgroundColor: "var(--color-focus-ring)",
              color: "var(--color-text)",
```

to:

```typescript
              backgroundColor: "var(--color-accent)",
              color: "var(--color-text-inverse)",
```

- [ ] **Step 4: Fix local auth Sign-in button — same fix**

In the submit button style (lines 268–271), change:

```typescript
                backgroundColor: isSubmitting
                  ? "var(--color-text-muted)"
                  : "var(--color-focus-ring)",
                color: "var(--color-text)",
```

to:

```typescript
                backgroundColor: isSubmitting
                  ? "var(--color-surface-elevated)"
                  : "var(--color-accent)",
                color: isSubmitting
                  ? "var(--color-text-muted)"
                  : "var(--color-text-inverse)",
```

- [ ] **Step 5: Verify build**

Run:
```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/auth/LoginPage.tsx
git commit -m "fix(web/frontend): LoginPage uses --color-accent, --color-text-inverse, semantic error tokens"
```

---

## Task 6: Fix SessionSidebar.tsx — Accent Token for Active Indicator and New-Session Button

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx:113,195`

- [ ] **Step 1: Replace active session border-left colour**

On line 113, change:

```typescript
                        ? "3px solid var(--color-focus-ring)"
```

to:

```typescript
                        ? "3px solid var(--color-accent)"
```

- [ ] **Step 2: Replace new-session button background**

On lines 193–196, change:

```typescript
            backgroundColor: isCreating
              ? "var(--color-text-muted)"
              : "var(--color-focus-ring)",
            color: "var(--color-text)",
```

to:

```typescript
            backgroundColor: isCreating
              ? "var(--color-surface-elevated)"
              : "var(--color-accent)",
            color: isCreating
              ? "var(--color-text-muted)"
              : "var(--color-text-inverse)",
```

- [ ] **Step 3: Verify build**

Run:
```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx
git commit -m "fix(web/frontend): SessionSidebar uses --color-accent instead of --color-focus-ring"
```

---

## Task 7: Fix SpecView.tsx — Accent for SELECTED Badge, Warning Semantic Tokens

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx:88,239-243`

- [ ] **Step 1: Replace SELECTED badge colour**

On line 88, change:

```typescript
    return { label: "SELECTED", color: "var(--color-focus-ring)" };
```

to:

```typescript
    return { label: "SELECTED", color: "var(--color-accent)" };
```

- [ ] **Step 2: Replace hardcoded warning rgba in validation errors section**

In the validation errors block (lines 239–243), change:

```typescript
              backgroundColor: "rgba(255, 204, 102, 0.15)",
              borderRadius: 6,
              fontSize: 13,
              color: "var(--color-warning)",
              border: "1px solid rgba(255, 204, 102, 0.3)",
```

to:

```typescript
              backgroundColor: "var(--color-warning-bg)",
              borderRadius: 6,
              fontSize: 13,
              color: "var(--color-warning)",
              border: "1px solid var(--color-warning-border)",
```

- [ ] **Step 3: Verify build**

Run:
```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/SpecView.tsx
git commit -m "fix(web/frontend): SpecView uses --color-accent for SELECTED badge, semantic warning tokens"
```

---

## Task 8: Fix ProgressView.tsx — Semantic Background Tokens

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx:34-36,169-173,199-202,213`

- [ ] **Step 1: Replace WebSocket disconnect banner warning rgba**

In the disconnect banner (lines 34–36), change:

```typescript
            backgroundColor: "rgba(255, 204, 102, 0.15)",
            color: "var(--color-warning)",
            border: "1px solid rgba(255, 204, 102, 0.3)",
```

to:

```typescript
            backgroundColor: "var(--color-warning-bg)",
            color: "var(--color-warning)",
            border: "1px solid var(--color-warning-border)",
```

- [ ] **Step 2: Replace cancellation message warning rgba**

In the cancellation message (lines 169–173), change:

```typescript
            backgroundColor: "rgba(255, 204, 102, 0.15)",
            color: "var(--color-warning)",
            borderRadius: 4,
            fontSize: 13,
            border: "1px solid rgba(255, 204, 102, 0.3)",
```

to:

```typescript
            backgroundColor: "var(--color-warning-bg)",
            color: "var(--color-warning)",
            borderRadius: 4,
            fontSize: 13,
            border: "1px solid var(--color-warning-border)",
```

- [ ] **Step 3: Replace recent errors container error rgba**

In the recent errors container (lines 199–202), change:

```typescript
              backgroundColor: "rgba(255, 102, 102, 0.12)",
              borderRadius: 4,
              padding: 8,
              border: "1px solid rgba(255, 102, 102, 0.3)",
```

to:

```typescript
              backgroundColor: "var(--color-error-bg)",
              borderRadius: 4,
              padding: 8,
              border: "1px solid var(--color-error-border)",
```

- [ ] **Step 4: Replace error list item divider**

On line 213, change:

```typescript
                      ? "1px solid rgba(255, 102, 102, 0.2)"
```

to:

```typescript
                      ? "1px solid var(--color-error-border)"
```

- [ ] **Step 5: Fix 11px font size to 12px minimum**

On lines 121 and 139, change both instances of:

```typescript
              fontSize: 11,
```

to:

```typescript
              fontSize: 12,
```

- [ ] **Step 6: Verify build**

Run:
```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit
```

Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/execution/ProgressView.tsx
git commit -m "fix(web/frontend): ProgressView uses semantic bg tokens, fix 11px text to 12px minimum"
```

---

## Task 9: Fix Remaining Files — ChatPanel.tsx, App.tsx, YamlView.tsx

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx:102`
- Modify: `src/elspeth/web/frontend/src/App.tsx:65-66`
- Modify: `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx:128`

- [ ] **Step 1: Fix ChatPanel.tsx error banner**

On line 102, change:

```typescript
            backgroundColor: "rgba(255, 102, 102, 0.12)",
```

to:

```typescript
            backgroundColor: "var(--color-error-bg)",
```

- [ ] **Step 2: Fix App.tsx system-unavailable banner**

On lines 65–66, change:

```typescript
              backgroundColor: "rgba(255, 102, 102, 0.12)",
              color: "var(--color-error)",
              borderBottom: "1px solid rgba(255, 102, 102, 0.3)",
```

to:

```typescript
              backgroundColor: "var(--color-error-bg)",
              color: "var(--color-error)",
              borderBottom: "1px solid var(--color-error-border)",
```

- [ ] **Step 3: Fix YamlView.tsx copied-state background**

On line 128, change:

```typescript
            ? "rgba(102, 204, 153, 0.15)"
```

to:

```typescript
            ? "var(--color-success-bg)"
```

- [ ] **Step 4: Verify full build**

Run:
```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit && npx vite build
```

Expected: Build completes with 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/App.tsx src/elspeth/web/frontend/src/components/inspector/YamlView.tsx
git commit -m "fix(web/frontend): remaining components use semantic bg tokens instead of hardcoded rgba"
```

---

## Task 10: Final Verification — No Remaining Hardcoded Colours

**Files:** None (verification only)

- [ ] **Step 1: Grep for old palette hex values**

Run:
```bash
cd src/elspeth/web/frontend/src && grep -rn '#1a1a2e\|#1e1e34\|#16162a\|#1c1c32\|#242440\|#7c8fff\|#8ab4ff\|#6c9\|#fc6\|#c9f\|#f96\|#f66\|#6cf' --include='*.tsx' --include='*.ts' .
```

Expected: 0 matches. Any remaining matches are missed migration points and must be fixed.

- [ ] **Step 2: Grep for hardcoded rgba(255, 102, 102) and rgba(255, 204, 102) patterns**

Run:
```bash
cd src/elspeth/web/frontend/src && grep -rn 'rgba(255, 102, 102\|rgba(255, 204, 102\|rgba(102, 204, 153\|rgba(204, 153, 255\|rgba(255, 153, 102' --include='*.tsx' --include='*.ts' .
```

Expected: 0 matches. The only rgba values remaining should be in `tokens.ts` (for React Flow) and those use the new DTA colour values, not the old ones.

- [ ] **Step 3: Grep for any remaining `--color-focus-ring` used as non-focus styling**

Run:
```bash
cd src/elspeth/web/frontend/src && grep -rn 'color-focus-ring' --include='*.tsx' --include='*.ts' .
```

Expected: 0 matches in component files. The token should only appear in `App.css` (`:root` definition and `:focus-visible` rule) and nowhere in `.tsx` files.

- [ ] **Step 4: Visual smoke test**

Run:
```bash
cd src/elspeth/web/frontend && npx vite build && npx vite preview
```

Open in a browser and visually verify:
- Deep teal backgrounds (not purple-blue)
- Green accent on Send button, New Session button, active tab underline, active session indicator
- White text on green buttons (not grey-on-purple)
- Purple focus ring only appears on keyboard Tab navigation
- Badge colours remain four distinct hues on graph nodes
- Error banners are coral-red, warning banners are amber, success states are teal
