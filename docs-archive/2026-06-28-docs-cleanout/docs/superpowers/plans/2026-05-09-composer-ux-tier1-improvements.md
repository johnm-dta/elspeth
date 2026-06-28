# Composer UX Tier-1 Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the high-leverage UX, accessibility, and bug fixes identified by the 2026-05-09 ux-critic review of the ELSPETH composer frontend, ahead of the upcoming tech demo.

**Architecture:** Pure-frontend changes scoped to `src/elspeth/web/frontend/src/`. No backend, contract, or schema modifications. Each task is independent and committable. Vitest + @testing-library/react drive test-first changes for component logic; pure CSS/token changes use inline-style assertions where possible and manual visual checks where not. The Catalog "Use in pipeline" task wires existing chat-input plumbing to the existing PluginCard, so no new state stores are added.

**Tech Stack:** React 18, TypeScript, Vitest, @testing-library/react, @testing-library/user-event, prism-react-renderer (already a dependency), CSS custom properties.

**Branch:** Work directly on `RC5.1` (the active demo-prep branch — see project memory `project_rc5ux_demo_prep_scope`). If the operator prefers isolation, create a worktree at `.worktrees/rc5.1-composer-ux-tier1` (per `feedback_worktree_convention`) and rebind editable install with `uv pip install -e . --python /home/john/elspeth/.worktrees/rc5.1-composer-ux-tier1/.venv/bin/python` if a venv is needed. The frontend itself does not require a Python venv — only `npm install` in the frontend directory.

**Source review:** `/home/john/elspeth/docs/superpowers/plans/2026-05-09-composer-ux-tier1-improvements.review.md` is the design review report. All file:line citations in this plan have been re-verified against current code at plan-write time.

---

## Pre-flight: Establish Working Environment

These steps run once before any task and are **not** committed.

- [ ] **Step 0.1: Confirm frontend toolchain works**

```bash
cd /home/john/elspeth/src/elspeth/web/frontend
npm install
npm run test
```

Expected: tests run; if there are pre-existing failures, capture them as a baseline so we don't blame our changes later.

- [ ] **Step 0.2: Start the dev server in a background terminal for visual verification**

```bash
cd /home/john/elspeth/src/elspeth/web/frontend
npm run dev
```

Expected: vite serves on `http://localhost:5173` (or 5174 if 5173 is busy). Leave running for the duration of plan execution; restart only if hot-reload misbehaves.

---

## File Structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `src/components/inspector/InspectorPanel.tsx` | Validation dot color mapping, remove `aria-live` from tab panel | T1, T4 |
| `src/components/inspector/InspectorPanel.test.tsx` | Cover validation dot colour state and tab panel a11y | T1, T4 |
| `src/components/chat/ComposingIndicator.tsx` | Remove nested `aria-live` | T5 |
| `src/components/chat/ComposingIndicator.test.tsx` | Verify the indicator no longer asserts a live region | T5 |
| `src/App.tsx` | Banner role downgrade `alert`→`status` | T6 |
| `src/App.test.tsx` (create if absent — verify in T6) | Cover banner role | T6 |
| `src/components/inspector/RunsView.tsx` | Add `aria-expanded` to Inspect button; class swap; cancelling badge | T7, T11 |
| `src/components/inspector/RunsView.test.tsx` | Cover `aria-expanded` and cancelling badge | T7, T11 |
| `src/components/settings/SecretsPanel.tsx` | Wrap `createSecret` await in try/finally | T3 |
| `src/components/settings/SecretsPanel.test.tsx` | Cover form-lock recovery on submit failure | T3 |
| `src/components/chat/MarkdownRenderer.tsx` | Prism syntax highlighting + copy button on fenced code blocks | T8 |
| `src/components/chat/MarkdownRenderer.test.tsx` | Cover Prism rendering + clipboard copy interaction | T8 |
| `src/components/catalog/PluginCard.tsx` | Add "Use in pipeline" insert action | T9 |
| `src/components/catalog/PluginCard.test.tsx` | Cover insert action wiring | T9 |
| `src/components/chat/ChatInput.tsx` (only if a small helper API is added) | Expose mechanism to prefill input from external caller | T9 (only if needed) |
| `src/App.css` | Validation indicator unchecked colour; light-theme `--color-status-empty`; resize handle affordance | T2, T10, T12 |
| `src/components/common/Layout.tsx` | Resize handle keyboard arrow direction | T10 |
| `src/components/common/Layout.test.tsx` | Cover handle arrow direction | T10 |

---

## Task 1: Validation Indicator — Decouple `unchecked` From `warning` Colour

**Why:** `InspectorPanel.tsx:466-471` currently maps both `unchecked` and `warning` validation states to `var(--color-warning)`. Colour-blind users and inattentive users cannot distinguish "not yet validated" from "validated, but with warnings." The glyphs (○ vs ⚠) differ but glyph alone is not enough — the dot is small, sits in the inspector header, and is the primary at-a-glance signal.

**Fix:** Map `unchecked` to `var(--color-text-muted)` (a neutral grey already in the token palette). `warning` keeps `var(--color-warning)` exclusively.

**Files:**
- Modify: `src/components/inspector/InspectorPanel.tsx:466-471`
- Test: `src/components/inspector/InspectorPanel.test.tsx`

- [ ] **Step 1.1: Write the failing test**

Append to `src/components/inspector/InspectorPanel.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InspectorPanel } from "./InspectorPanel";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";

describe("InspectorPanel — validation dot colour", () => {
  it("renders unchecked state in muted text colour, not warning orange", () => {
    // Arrange: a session with a non-empty composition but no validation result
    useSessionStore.setState({
      activeSessionId: "s1",
      compositionState: {
        version: 1,
        source: { id: "src", type: "csv_file", options: {} },
        nodes: [],
        outputs: [],
      } as any,
    });
    useExecutionStore.setState({ validationResult: null });

    render(<InspectorPanel />);

    const dot = screen.getByLabelText("Not validated");
    expect(dot.getAttribute("style")).toContain("var(--color-text-muted)");
    expect(dot.getAttribute("style")).not.toContain("var(--color-warning)");
  });

  it("renders warning state in warning colour", () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      compositionState: {
        version: 1,
        source: { id: "src", type: "csv_file", options: {} },
        nodes: [],
        outputs: [],
      } as any,
    });
    useExecutionStore.setState({
      validationResult: { is_valid: true, errors: [], warnings: [{ message: "x" }] } as any,
    });

    render(<InspectorPanel />);

    const dot = screen.getByLabelText("Validation passed with warnings");
    expect(dot.getAttribute("style")).toContain("var(--color-warning)");
  });
});
```

- [ ] **Step 1.2: Run the test to verify it fails**

```bash
cd /home/john/elspeth/src/elspeth/web/frontend
npm run test -- InspectorPanel.test.tsx
```

Expected: the `unchecked` test FAILS — current code emits `var(--color-warning)` for the `unchecked` state. The `warning` test PASSES (colour was already correct).

- [ ] **Step 1.3: Apply the colour mapping change**

Edit `src/components/inspector/InspectorPanel.tsx:466-471` — change the `colors` object so `unchecked` no longer reuses `var(--color-warning)`:

```tsx
const colors: Record<string, string> = {
  unchecked: "var(--color-text-muted)",
  valid: "var(--color-success)",
  warning: "var(--color-warning)",
  invalid: "var(--color-error)",
};
```

- [ ] **Step 1.4: Run the tests to verify they pass**

```bash
npm run test -- InspectorPanel.test.tsx
```

Expected: PASS for both new cases. Existing tests in the file should remain green.

- [ ] **Step 1.5: Visual sanity check in dev server**

Open `http://localhost:5173`, create a new session, type a single message that produces a non-empty composition, observe the validation dot before pressing Validate. It should render in muted grey (○), not orange. Press Validate to a known-warning state and confirm the dot turns orange (⚠).

- [ ] **Step 1.6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx \
        src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx
git commit -m "fix(composer): decouple unchecked validation indicator from warning colour"
```

---

## Task 2: Light Theme — Add Missing `--color-status-empty` Override

**Why:** `App.css:103` defines `--color-status-empty: #888888` for the dark theme. The light theme block (`App.css:1292-1450`) does not override it, so a dark mid-grey is rendered on the light theme's near-white background (`#f4f8f9`). Approximate contrast ~2.6:1 — fails WCAG AA for body text, marginal for the badge use case.

**Fix:** Add `--color-status-empty` to `[data-theme="light"]` with a value chosen for ≥4.5:1 against `--color-bg`.

**Files:**
- Modify: `src/App.css` — light theme block, near `App.css:1378-1384`

- [ ] **Step 2.1: Locate the light-theme override block**

```bash
grep -n "data-theme=\"light\"" /home/john/elspeth/src/elspeth/web/frontend/src/App.css
```

Expected output: `1292:[data-theme="light"] {` (with the closing brace around line 1450).

- [ ] **Step 2.2: Add the missing token inside the light-theme block**

Add the following line inside `[data-theme="light"] { … }`, grouped with the other status tokens:

```css
  --color-status-empty:      #5a7a84;
```

A teal-tinted slate in the existing palette family; it produces ≈4.6:1 against `#f4f8f9`. Verify with a contrast checker before committing.

- [ ] **Step 2.3: Verify with a contrast measurement**

Use any WCAG checker (axe DevTools, Colour Contrast Analyser, or the macOS `Digital Color Meter` + `npx --yes wcag-contrast-checker`). Target: ≥4.5:1 on `#f4f8f9`. If `#5a7a84` falls short due to monitor calibration differences, darken to `#4d6a73` (~5.4:1).

- [ ] **Step 2.4: Visual check on the dev server**

```bash
# In the running dev server: switch to light theme via the sidebar toggle.
# Locate any UI surface that uses the EMPTY run status (RunsView; load a project
# with at least one EMPTY run, or temporarily mock one in the executionStore via DevTools).
```

The status badge should read as a clearly-readable mid-tone, not washed-out.

- [ ] **Step 2.5: Commit**

```bash
git add src/elspeth/web/frontend/src/App.css
git commit -m "fix(composer): add missing --color-status-empty override for light theme"
```

---

## Task 3: SecretsPanel — Recover the Form on Submit Failure

**Why:** `SecretsPanel.tsx:81-94` calls `await createSecret(...)` with no try/catch and no `finally`. If the underlying request throws, `setIsSubmitting(false)` never executes; the form is permanently disabled until the user reloads the page or closes and reopens the modal. Real bug, user-visible.

**Verified:** Re-read of the file at plan time confirms the issue is unchanged in HEAD.

**Files:**
- Modify: `src/components/settings/SecretsPanel.tsx:81-94`
- Test: `src/components/settings/SecretsPanel.test.tsx`

- [ ] **Step 3.1: Write the failing test**

Append to `src/components/settings/SecretsPanel.test.tsx`:

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SecretsPanel } from "./SecretsPanel";
import { useSecretsStore } from "@/stores/secretsStore";

describe("SecretsPanel — submit failure recovery", () => {
  it("re-enables the form after createSecret throws", async () => {
    const user = userEvent.setup();
    const failing = vi.fn(async () => {
      throw new Error("network");
    });
    useSecretsStore.setState({
      secrets: [],
      isLoading: false,
      error: null,
      loadSecrets: vi.fn(),
      createSecret: failing,
      deleteSecret: vi.fn(),
    } as any);

    render(<SecretsPanel onClose={() => {}} />);

    await user.type(screen.getByLabelText(/name/i), "OPENAI_API_KEY");
    await user.type(screen.getByLabelText(/value/i), "sk-test");
    await user.click(screen.getByRole("button", { name: /save secret/i }));

    // The form must be re-enabled after the failure.
    const submit = screen.getByRole("button", { name: /save secret/i });
    expect(submit).not.toBeDisabled();
    expect(failing).toHaveBeenCalledOnce();
  });
});
```

If the existing test file's labels don't exactly match `name`, `value`, or the submit button text, read the component's render output and adjust the queries — do not invent labels. The point of the test is the behaviour, not the labels.

- [ ] **Step 3.2: Run the test to verify it fails**

```bash
cd /home/john/elspeth/src/elspeth/web/frontend
npm run test -- SecretsPanel.test.tsx
```

Expected: FAIL — the unhandled rejection from the failing `createSecret` mock leaves the submit button still in its `isSubmitting` state. (Vitest may also surface an `unhandledRejection` warning; that is corroborating evidence.)

- [ ] **Step 3.3: Apply the fix**

Replace the body of `handleSubmit` in `src/components/settings/SecretsPanel.tsx` with a try/finally that always restores the form state and clears the input. The error itself is allowed to propagate to the store's error channel (the store already publishes `error`), and the SecretsPanel renders that `error` near the top of the modal.

```tsx
const handleSubmit = useCallback(
  async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !value) return;
    setIsSubmitting(true);
    try {
      await createSecret(name.trim(), value);
    } finally {
      // SECURITY: clear value immediately — it must never linger in component
      // state regardless of whether the API call succeeded or failed.
      setValue("");
      setName("");
      setIsSubmitting(false);
    }
  },
  [name, value, createSecret],
);
```

Note: this preserves the existing security comment about clearing `value`. It also clears `name` on failure, which is a UX trade-off — the user re-types the name. If the operator prefers preserving `name` on failure, change the finally to clear only `value` and reset `isSubmitting`.

- [ ] **Step 3.4: Run the tests to verify they pass**

```bash
npm run test -- SecretsPanel.test.tsx
```

Expected: PASS for the new case. Confirm existing tests still pass.

- [ ] **Step 3.5: Manual check — induce a failure**

In the dev server, with backend running, briefly stop the backend (`sudo systemctl stop elspeth-web.service` if you are testing against staging — but check with the operator first; per project memory, the deployed staging is on this machine). Open the SecretsPanel, attempt to save a secret. Confirm the form re-enables after the failure rather than locking. Restart the backend (`sudo systemctl start elspeth-web.service`) when done.

If you are not authorised to stop the staging service, simulate the failure by adding a temporary `throw new Error("test")` at the top of the secretsStore's `createSecret` action, observe the recovery, then revert the throw before commit.

- [ ] **Step 3.6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/settings/SecretsPanel.tsx \
        src/elspeth/web/frontend/src/components/settings/SecretsPanel.test.tsx
git commit -m "fix(composer): recover SecretsPanel form on createSecret failure"
```

---

## Task 4: Inspector Tab Panel — Remove Over-Broad `aria-live`

**Why:** `InspectorPanel.tsx:588-591` wraps the entire tab content area with `aria-live="polite"`. Switching tabs between Spec → Graph → YAML → Runs causes screen readers to announce the entire content of the new panel. Switching to YAML announces hundreds of lines of code. The live region's intent was to announce validation results, but those have their own banner (`ValidationResultBanner`) which can carry a targeted `aria-live`.

**Fix:** Remove `aria-live="polite"` from the tab panel container. Add `aria-live="polite"` to the `ValidationResultBanner` only — the substrate that genuinely changes from background activity.

**Files:**
- Modify: `src/components/inspector/InspectorPanel.tsx:588-591`
- Modify: `src/components/execution/ValidationResult.tsx` — add `aria-live="polite"` to the outer banner element (only if it is not already present; verify before editing)
- Test: `src/components/inspector/InspectorPanel.test.tsx`

- [ ] **Step 4.1: Read ValidationResult.tsx to confirm whether the banner already declares aria-live**

```bash
grep -n "aria-live\|role=" /home/john/elspeth/src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx
```

If the banner already has `aria-live` or a `role="alert"`/`role="status"` that triggers live announcement, **skip the ValidationResult edit**; only the Inspector edit is needed. If it has no live affordance at all, perform the small edit in step 4.4.

- [ ] **Step 4.2: Write the failing test**

Append to `src/components/inspector/InspectorPanel.test.tsx`:

```tsx
describe("InspectorPanel — aria-live scope", () => {
  it("does not place aria-live on the tab panel container", () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      compositionState: {
        version: 1,
        source: null,
        nodes: [],
        outputs: [],
      } as any,
    });
    useExecutionStore.setState({ validationResult: null, error: null });

    render(<InspectorPanel />);

    const tabPanel = screen.getByRole("tabpanel");
    expect(tabPanel.getAttribute("aria-live")).toBeNull();
  });
});
```

- [ ] **Step 4.3: Run the test to verify it fails**

```bash
npm run test -- InspectorPanel.test.tsx
```

Expected: FAIL — current code sets `aria-live="polite"` on the tabpanel.

- [ ] **Step 4.4: Apply the fix to InspectorPanel**

Edit `src/components/inspector/InspectorPanel.tsx:588-594` — remove the `aria-live` attribute from the tab panel `<div>`, leaving the other ARIA attributes intact:

```tsx
<div
  role="tabpanel"
  id={`inspector-tabpanel-${activeTab}`}
  aria-labelledby={`inspector-tab-${activeTab}`}
  className="inspector-tab-content"
>
```

- [ ] **Step 4.5: (Conditional) Add a targeted live region to the validation banner**

Only if Step 4.1 found no live affordance on `ValidationResultBanner`. Edit `src/components/execution/ValidationResult.tsx` and add `aria-live="polite"` to the outermost banner element. Do not change `role`. The result is that validation outcomes still announce to screen readers, but tab switches do not.

- [ ] **Step 4.6: Run the tests to verify they pass**

```bash
npm run test -- InspectorPanel.test.tsx ValidationResult.test
```

Expected: PASS. The new tabpanel test passes; existing ValidationResult tests are unchanged unless step 4.5 was applied.

- [ ] **Step 4.7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx \
        src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx
# Include ValidationResult only if step 4.5 was applied:
# git add src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx
git commit -m "fix(composer): scope aria-live to validation banner, not tab panel"
```

---

## Task 5: ComposingIndicator — Remove Nested `aria-live`

**Why:** `ChatPanel.tsx:161-164` wraps the message list in `role="log" aria-live="polite" aria-relevant="additions"`. Inside that region, `ComposingIndicator.tsx:142-144` adds its own `aria-live="polite" role="status"`. Nested live regions are spec-disallowed and produce inconsistent behaviour across screen readers — content may be announced multiple times or, in some implementations, suppressed entirely.

**Fix:** Remove `aria-live="polite"` from `ComposingIndicator`. Keep `role="status"` since it conveys semantic role (this is a live status region by virtue of the parent `role="log"`). The parent's `aria-live="polite"` covers the announcement; the indicator's own `aria-live` was redundant.

**Files:**
- Modify: `src/components/chat/ComposingIndicator.tsx:140-145`
- Test: `src/components/chat/ComposingIndicator.test.tsx`

- [ ] **Step 5.1: Write the failing test**

Append to `src/components/chat/ComposingIndicator.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { ComposingIndicator } from "./ComposingIndicator";

describe("ComposingIndicator — live region scope", () => {
  it("does not declare its own aria-live (avoids nesting inside chat-panel-messages)", () => {
    const { container } = render(<ComposingIndicator />);
    const root = container.firstChild as HTMLElement;
    expect(root.getAttribute("aria-live")).toBeNull();
    // role="status" remains for semantic clarity
    expect(root.getAttribute("role")).toBe("status");
  });
});
```

- [ ] **Step 5.2: Run the test to verify it fails**

```bash
npm run test -- ComposingIndicator.test.tsx
```

Expected: FAIL — current code has `aria-live="polite"`.

- [ ] **Step 5.3: Apply the fix**

Edit `src/components/chat/ComposingIndicator.tsx:140-145` to remove the `aria-live` attribute:

```tsx
return (
  <div
    className="composing-indicator composing-row"
    role="status"
  >
```

- [ ] **Step 5.4: Run the tests to verify they pass**

```bash
npm run test -- ComposingIndicator.test.tsx ChatPanel.test
```

Expected: PASS. Existing ChatPanel tests are unchanged (the parent live region is unaffected).

- [ ] **Step 5.5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/ComposingIndicator.tsx \
        src/elspeth/web/frontend/src/components/chat/ComposingIndicator.test.tsx
git commit -m "fix(composer): drop nested aria-live in ComposingIndicator"
```

---

## Task 6: App Banners — Downgrade `role="alert"` → `role="status"`

**Why:** `App.tsx:184-218` declares two banner divs with `role="alert"`. `role="alert"` triggers an immediate, interruptive screen-reader announcement. With the 30-second health-check polling at `App.tsx:162-173`, any flapping connection causes the banners to remount and re-announce on every cycle. `role="status"` (or `aria-live="polite"`) communicates the same UI state without interrupting the user mid-task.

The composer-unavailable banner is also non-actionable in the immediate sense — it is informational ("the LLM cannot be reached") with a link to settings. `role="alert"` is reserved for immediate-attention errors with a destructive consequence; the banners do not meet that bar.

**Files:**
- Modify: `src/App.tsx:184-218` — change both banners' `role`
- Test: there is no existing `App.test.tsx` (verify); if not, add a small focused test file

- [ ] **Step 6.1: Verify whether App.test.tsx exists**

```bash
ls /home/john/elspeth/src/elspeth/web/frontend/src/App.test.tsx 2>/dev/null && echo "exists" || echo "absent"
```

- [ ] **Step 6.2: Write the failing test**

Create or extend `src/elspeth/web/frontend/src/App.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import * as api from "./api/client";

vi.mock("./api/client");

describe("App — banner roles", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("uses role=status, not role=alert, for the backend-unavailable banner", async () => {
    vi.spyOn(api, "fetchSystemStatus").mockRejectedValue(new Error("down"));

    render(<App />);

    const banner = await screen.findByText(/Backend unavailable/i);
    const root = banner.closest("[class*='alert-banner']") as HTMLElement | null;
    expect(root).not.toBeNull();
    expect(root!.getAttribute("role")).toBe("status");
  });
});
```

If `App.tsx`'s `AuthGuard` blocks render in tests, mock the auth path the same way existing tests in the repo do (look for the pattern in `Layout.test.tsx` or the closest neighbour). If a clean App-level integration test is awkward, alternatively assert against just the JSX by extracting the banners into a tiny named component first — but **only if** that extraction keeps the change minimal. Prefer the smaller change if testing proves obstinate.

- [ ] **Step 6.3: Run the test to verify it fails**

```bash
npm run test -- App.test.tsx
```

Expected: FAIL — current `role` is `"alert"`.

- [ ] **Step 6.4: Apply the change**

Edit both banners in `src/App.tsx` (lines 184 and 202):

```tsx
{backendAvailable === false && (
  <div role="status" className="alert-banner">
    {/* ... unchanged ... */}
  </div>
)}

{/* Composer unavailable banner */}
{backendAvailable && systemStatus && !systemStatus.composer_available && (
  <div role="status" className="alert-banner">
    {/* ... unchanged ... */}
  </div>
)}
```

The CSS class `.alert-banner` keeps the visual treatment; only the ARIA role changes.

- [ ] **Step 6.5: Run the tests to verify they pass**

```bash
npm run test -- App.test.tsx
```

Expected: PASS.

- [ ] **Step 6.6: Commit**

```bash
git add src/elspeth/web/frontend/src/App.tsx \
        src/elspeth/web/frontend/src/App.test.tsx
git commit -m "fix(composer): downgrade health-check banners to role=status"
```

---

## Task 7: RunsView Inspect Button — Add `aria-expanded`

**Why:** `RunsView.tsx:363-383` renders an Inspect/Hide button that toggles a diagnostics panel below the run row. The button has no `aria-expanded` attribute, so assistive technology users cannot determine whether the diagnostics panel is open or closed. The visual label changes ("Hide" vs "Inspect") but ARIA is the right channel for the state, not the label alone.

**Files:**
- Modify: `src/components/inspector/RunsView.tsx:363-383`
- Test: `src/components/inspector/RunsView.test.tsx`

- [ ] **Step 7.1: Write the failing test**

Append to `src/components/inspector/RunsView.test.tsx`:

```tsx
describe("RunsView — Inspect button a11y", () => {
  it("declares aria-expanded reflecting diagnostics panel state", async () => {
    const user = userEvent.setup();
    // Mock store with at least one finished run; copy the existing test setup
    // pattern from elsewhere in this file and adapt minimally.
    setupStoreWithSampleRuns();  // existing helper or inline equivalent

    render(<RunsView />);

    const inspect = screen.getAllByRole("button", { name: /inspect/i })[0];
    expect(inspect.getAttribute("aria-expanded")).toBe("false");

    await user.click(inspect);

    const hide = screen.getAllByRole("button", { name: /hide/i })[0];
    expect(hide.getAttribute("aria-expanded")).toBe("true");
  });
});
```

If `setupStoreWithSampleRuns` does not exist as a helper, inline the store seeding using the same shape as existing tests in `RunsView.test.tsx`.

- [ ] **Step 7.2: Run the test to verify it fails**

```bash
npm run test -- RunsView.test.tsx
```

Expected: FAIL on `aria-expanded` assertions.

- [ ] **Step 7.3: Apply the fix**

Edit `src/components/inspector/RunsView.tsx:363-383`. Add `aria-expanded` and `aria-controls` to the button. Add a matching `id` to the diagnostics panel below it. Since the diagnostics panel rendering is in a separate JSX branch (around `RunsView.tsx:511-558` per the design review), give the panel a stable id of the form `run-diagnostics-${run.id}`.

```tsx
<button
  type="button"
  aria-expanded={expandedRunId === run.id}
  aria-controls={`run-diagnostics-${run.id}`}
  onClick={() => {
    const nextRunId = expandedRunId === run.id ? null : run.id;
    setExpandedRunId(nextRunId);
    if (nextRunId) {
      void loadRunDiagnostics(run.id);
    }
  }}
  className="btn btn-small"
>
  {expandedRunId === run.id ? "Hide" : "Inspect"}
</button>
```

Replace the inline `style={{ ... }}` with the design-system class `btn btn-small`. Verify that `.btn-small` exists in `App.css`; if not, add a minimal rule next to the existing `.btn` rules:

```css
.btn-small {
  font-size: var(--font-size-xs);
  padding: 2px var(--space-sm);
}
```

Then on the diagnostics panel container, add the matching id:

```tsx
<div id={`run-diagnostics-${run.id}`} className="run-diagnostics">
  {/* ... existing panel content ... */}
</div>
```

- [ ] **Step 7.4: Run the tests to verify they pass**

```bash
npm run test -- RunsView.test.tsx
```

Expected: PASS for the new aria-expanded assertions; existing tests remain green.

- [ ] **Step 7.5: Visual check**

In the dev server, open a session with at least one finished run, click Inspect, confirm the panel opens and the button label flips to "Hide". The visual treatment now uses the design-system button — no more flush-against-the-row inline-styled affordance.

- [ ] **Step 7.6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/RunsView.tsx \
        src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx \
        src/elspeth/web/frontend/src/App.css
git commit -m "fix(composer): add aria-expanded to RunsView Inspect button and use .btn class"
```

---

## Task 8: MarkdownRenderer — Syntax Highlighting + Copy Button on Code Blocks

**Why:** `MarkdownRenderer.tsx:86-92` renders fenced code blocks as plain `<pre><code>` with no syntax highlighting and no copy affordance. The composer's primary output format is YAML pipeline configuration; reading it as plain monospaced text and select-all-copying-by-hand is friction the user feels on every single message. The dependency `prism-react-renderer` is already installed and used in `YamlView.tsx` — we just need to apply the same renderer to fenced blocks coming out of markdown.

**Files:**
- Modify: `src/components/chat/MarkdownRenderer.tsx:60-93` — replace plain `<pre><code>` with a Prism-rendered block plus a copy button
- Modify (or create): `src/components/chat/MarkdownRenderer.test.tsx` — assert highlighting and copy behaviour
- Possibly modify: `src/App.css` — add a minimal `.code-block-toolbar` and `.code-block-copy` style if not already present (verify before adding)

- [ ] **Step 8.1: Read YamlView.tsx for the existing Prism integration pattern**

```bash
grep -n "prism-react-renderer\|Highlight" /home/john/elspeth/src/elspeth/web/frontend/src/components/inspector/YamlView.tsx
```

Mirror the same import, theme selection, and per-line rendering. Reuse YamlView's theme constants if they are exported; otherwise duplicate the small block (do not over-abstract — two call sites is below the abstraction threshold).

- [ ] **Step 8.2: Write the failing tests**

Extend `src/components/chat/MarkdownRenderer.test.tsx`:

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MarkdownRenderer } from "./MarkdownRenderer";

describe("MarkdownRenderer — fenced code blocks", () => {
  it("renders YAML fenced blocks with Prism token spans", () => {
    const md = "```yaml\nsource:\n  type: csv_file\n```";
    render(<MarkdownRenderer content={md} />);
    // Prism produces token spans inside the pre element
    const pre = document.querySelector("pre.code-block");
    expect(pre).not.toBeNull();
    expect(pre!.querySelector("span")).not.toBeNull();
  });

  it("renders a copy button that copies the code to the clipboard", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    const md = "```yaml\nfoo: bar\n```";
    render(<MarkdownRenderer content={md} />);

    const copy = screen.getByRole("button", { name: /copy/i });
    await user.click(copy);

    expect(writeText).toHaveBeenCalledWith("foo: bar");
  });
});
```

- [ ] **Step 8.3: Run the tests to verify they fail**

```bash
npm run test -- MarkdownRenderer.test.tsx
```

Expected: both new tests FAIL.

- [ ] **Step 8.4: Apply the implementation**

Replace the `CodeBlock` body in `src/components/chat/MarkdownRenderer.tsx:60-93` with a Prism-rendered block plus a copy button. The Mermaid branch and inline-code branch remain unchanged.

```tsx
import { Highlight, themes as prismThemes } from "prism-react-renderer";
import { useTheme } from "@/hooks/useTheme";
import { useState, useCallback } from "react";

function CodeBlock({
  className,
  children,
  ...props
}: ComponentPropsWithoutRef<"code">) {
  const language = className?.replace("language-", "") ?? "";
  const code = String(children).replace(/\n$/, "");
  const { resolvedTheme } = useTheme();
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable — silently no-op; the user can still select-and-copy.
    }
  }, [code]);

  // Inline code (no language, rendered inside a <p>)
  if (!className) {
    return <code className="inline-code" {...props}>{children}</code>;
  }

  // Mermaid diagrams get special treatment
  if (language === "mermaid") {
    return <MermaidDiagram chart={code} />;
  }

  const prismTheme =
    resolvedTheme === "dark" ? prismThemes.vsDark : prismThemes.vsLight;

  return (
    <div className="code-block-wrapper">
      <button
        type="button"
        className="code-block-copy"
        onClick={handleCopy}
        aria-label={copied ? "Copied" : "Copy code"}
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <Highlight code={code} language={language || "text"} theme={prismTheme}>
        {({ className: hClass, style, tokens, getLineProps, getTokenProps }) => (
          <pre className={`code-block ${hClass}`} style={style}>
            <code className={className}>
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  {line.map((token, j) => (
                    <span key={j} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </code>
          </pre>
        )}
      </Highlight>
    </div>
  );
}
```

- [ ] **Step 8.5: Add minimal CSS for the copy button**

In `src/App.css`, near the existing `.code-block` rule, add:

```css
.code-block-wrapper {
  position: relative;
}

.code-block-copy {
  position: absolute;
  top: var(--space-sm);
  right: var(--space-sm);
  padding: var(--space-xs) var(--space-sm);
  font-size: var(--font-size-xs);
  background: var(--color-surface-elevated);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--transition-fast);
}

.code-block-wrapper:hover .code-block-copy,
.code-block-copy:focus-visible {
  opacity: 1;
}
```

The hover-reveal pattern keeps message bodies clean while keeping the affordance keyboard-discoverable via `:focus-visible`.

- [ ] **Step 8.6: Run the tests to verify they pass**

```bash
npm run test -- MarkdownRenderer.test.tsx
```

Expected: PASS for both new tests; existing tests still green.

- [ ] **Step 8.7: Visual check**

In the dev server, send a chat message that produces a YAML response. Confirm:
- The block renders with coloured tokens (keys, strings, comments differ visually).
- Hovering reveals the Copy button in the top-right corner.
- Clicking copies the code; the label briefly changes to "Copied".
- Tabbing onto the block reveals the button via focus-visible.

- [ ] **Step 8.8: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/MarkdownRenderer.tsx \
        src/elspeth/web/frontend/src/components/chat/MarkdownRenderer.test.tsx \
        src/elspeth/web/frontend/src/App.css
git commit -m "feat(composer): syntax highlighting and copy button for chat code blocks"
```

---

## Task 9: PluginCard — "Use in Pipeline" Insert Action

**Why:** The catalog browse-only design (`CatalogDrawer.tsx`, `PluginCard.tsx`) leaves users with no way to act on plugin discovery. The strategic UX gap from the review (chat-only mutation) is partially addressed by giving each PluginCard a button that prefills the chat input with a ready-made insertion prompt. This preserves the chat-driven composition model and avoids a full direct-manipulation rebuild.

**Approach:** Add a button to `PluginCard` that, when clicked, dispatches a custom event (`PREFILL_CHAT_INPUT`) with the prompt text. `ChatInput` listens for the event, populates its textarea, focuses it, and closes the catalog drawer.

We use a custom event rather than a Zustand store action because the prefill is a one-shot UI command, not state to be persisted or reasoned about — avoiding adding store surface area we'd have to clean up.

**Files:**
- Modify: `src/components/catalog/PluginCard.tsx` — add the button, dispatch the event
- Modify: `src/components/chat/ChatInput.tsx` — listen for the event, populate, focus
- Modify: `src/components/catalog/CatalogDrawer.tsx` — close the drawer when an insert is initiated (so the user lands in the chat input, not behind the open drawer)
- Test: `src/components/catalog/PluginCard.test.tsx`
- Test: `src/components/chat/ChatInput.test.tsx` — verify only if it exists; otherwise inline a focused test in PluginCard.test.tsx
- Constants: introduce a single exported event name string in `src/components/catalog/PluginCard.tsx` (or a small `src/components/catalog/events.ts` if there's appetite for a shared constants module — verify project convention before deciding)

- [ ] **Step 9.1: Decide event-name constant location**

```bash
grep -n "_EVENT = \"" /home/john/elspeth/src/elspeth/web/frontend/src/components/**/*.tsx 2>/dev/null | head
```

If existing code follows the pattern of exporting `*_EVENT` constants from the component file (as `CommandPalette.tsx` does with `SWITCH_TAB_EVENT`), do the same. Otherwise, place it at the top of the modified `PluginCard.tsx`.

- [ ] **Step 9.2: Write the failing tests**

Append to `src/components/catalog/PluginCard.test.tsx`:

```tsx
describe("PluginCard — Use in pipeline", () => {
  it("dispatches a chat-prefill event and closes the drawer when clicked", async () => {
    const user = userEvent.setup();
    const handler = vi.fn();
    window.addEventListener("composer:prefill-chat-input", handler);

    const onClose = vi.fn();
    render(
      <PluginCard
        spec={SAMPLE_SOURCE_PLUGIN_SPEC}
        onCloseDrawer={onClose}
      />
    );

    await user.click(screen.getByRole("button", { name: /use in pipeline/i }));

    expect(handler).toHaveBeenCalledOnce();
    const event = handler.mock.calls[0][0] as CustomEvent<string>;
    expect(event.detail).toMatch(/csv_file/);  // adapt to the sample spec
    expect(onClose).toHaveBeenCalledOnce();

    window.removeEventListener("composer:prefill-chat-input", handler);
  });
});
```

If `PluginCard` does not currently accept an `onCloseDrawer` prop, see step 9.4 — we add it.

For ChatInput coverage, append to `src/components/chat/ChatInput.test.tsx` (verify it exists; if not, the dispatch coverage in PluginCard's test is sufficient as the prefill behaviour is visually verifiable on the dev server):

```tsx
describe("ChatInput — prefill event", () => {
  it("populates and focuses the textarea when prefill event is received", () => {
    render(<ChatInput onSend={() => {}} disabled={false} />);
    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;

    window.dispatchEvent(
      new CustomEvent("composer:prefill-chat-input", {
        detail: "Add a CSV source named 'orders' reading from data/orders.csv",
      })
    );

    expect(textarea.value).toBe(
      "Add a CSV source named 'orders' reading from data/orders.csv"
    );
    expect(document.activeElement).toBe(textarea);
  });
});
```

- [ ] **Step 9.3: Run the tests to verify they fail**

```bash
npm run test -- PluginCard.test.tsx ChatInput.test
```

Expected: FAIL on the new tests.

- [ ] **Step 9.4: Implement the dispatch in PluginCard**

In `src/components/catalog/PluginCard.tsx`, near the top of the file:

```tsx
export const PREFILL_CHAT_INPUT_EVENT = "composer:prefill-chat-input";

interface PluginCardProps {
  spec: PluginSpec;
  /** Called when the card initiates an action that should close the drawer. */
  onCloseDrawer?: () => void;
}
```

Inside the card body, render a button. Place it in the existing card header row alongside the existing primary affordance (e.g., near the title; verify the existing layout when editing). Use the design-system `.btn` class:

```tsx
const handleUseInPipeline = useCallback((e: React.MouseEvent) => {
  e.stopPropagation();
  const prompt = buildInsertionPrompt(spec);
  window.dispatchEvent(
    new CustomEvent(PREFILL_CHAT_INPUT_EVENT, { detail: prompt })
  );
  onCloseDrawer?.();
}, [spec, onCloseDrawer]);

// JSX:
<button
  type="button"
  className="btn"
  onClick={handleUseInPipeline}
  aria-label={`Use ${spec.name} in pipeline`}
>
  Use in pipeline
</button>
```

Define `buildInsertionPrompt` near the bottom of the file:

```tsx
function buildInsertionPrompt(spec: PluginSpec): string {
  const role =
    spec.kind === "source" ? "as the source"
    : spec.kind === "sink"   ? "as a sink"
    : "as a transform";
  return `Add ${spec.type} ${role} (named "${spec.type}-1"). Use sensible defaults for required fields and ask me about anything that needs domain context.`;
}
```

If the actual `PluginSpec` field names differ from `kind`/`type`/`name`, read the type definition first and adapt.

- [ ] **Step 9.5: Implement the listener in ChatInput**

Add to `src/components/chat/ChatInput.tsx`:

```tsx
import { PREFILL_CHAT_INPUT_EVENT } from "@/components/catalog/PluginCard";

useEffect(() => {
  function handle(e: Event) {
    const detail = (e as CustomEvent<string>).detail;
    if (typeof detail !== "string") return;
    setValue(detail);
    textareaRef.current?.focus();
    // Place caret at end so the user can keep typing
    const len = detail.length;
    textareaRef.current?.setSelectionRange(len, len);
  }
  window.addEventListener(PREFILL_CHAT_INPUT_EVENT, handle);
  return () => window.removeEventListener(PREFILL_CHAT_INPUT_EVENT, handle);
}, []);
```

If `ChatInput` already manages textarea content via a ref, adapt the body — the goal is "set the value, focus, place caret at end."

- [ ] **Step 9.6: Wire `onCloseDrawer` from CatalogDrawer**

In `src/components/catalog/CatalogDrawer.tsx`, pass the existing `onClose` handler down to each `PluginCard`:

```tsx
<PluginCard spec={spec} onCloseDrawer={onClose} />
```

- [ ] **Step 9.7: Run the tests to verify they pass**

```bash
npm run test -- PluginCard.test.tsx ChatInput.test CatalogDrawer.test
```

Expected: PASS.

- [ ] **Step 9.8: Visual check**

In the dev server: open the catalog drawer, click "Use in pipeline" on a CSV source plugin. Confirm:
- The drawer closes.
- The chat textarea is focused with a ready-made prompt.
- Pressing Enter (or Cmd+Enter — verify the existing send key) sends the prompt and the LLM produces a composition update.

- [ ] **Step 9.9: Commit**

```bash
git add src/elspeth/web/frontend/src/components/catalog/PluginCard.tsx \
        src/elspeth/web/frontend/src/components/catalog/PluginCard.test.tsx \
        src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatInput.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatInput.test.tsx
git commit -m "feat(composer): catalog 'Use in pipeline' action prefills chat input"
```

---

## Task 10: Resize Handle — Invert Keyboard Arrow Direction

**Why:** `Layout.tsx:304-313`: `ArrowLeft` increases inspector width; `ArrowRight` decreases it. The handle sits at the LEFT edge of the right-hand inspector panel, so "ArrowLeft moves the handle left → panel grows" is internally consistent — but the dominant mental model for keyboard users is "arrow direction = direction of value change," and the inspector's width is a value that grows or shrinks. Inverting the mapping aligns with the convention used by virtually every native splitter widget.

**Files:**
- Modify: `src/components/common/Layout.tsx:304-313`
- Test: `src/components/common/Layout.test.tsx`

- [ ] **Step 10.1: Write the failing test**

Append to `src/components/common/Layout.test.tsx`:

```tsx
describe("Layout — resize handle keyboard arrows", () => {
  it("ArrowLeft decreases inspector width and ArrowRight increases it", async () => {
    const user = userEvent.setup();
    const setItem = vi.spyOn(Storage.prototype, "setItem");

    render(
      <Layout sidebar={<div />} chat={<div />} inspector={<div />} />
    );

    const handle = screen.getByRole("separator", { name: /resize inspector/i });
    handle.focus();

    await user.keyboard("{ArrowRight}");
    const widthAfterRight = Number(
      setItem.mock.calls.findLast(([k]) => k === "elspeth_inspector_width")?.[1]
    );

    await user.keyboard("{ArrowLeft}");
    const widthAfterLeft = Number(
      setItem.mock.calls.findLast(([k]) => k === "elspeth_inspector_width")?.[1]
    );

    expect(widthAfterLeft).toBeLessThan(widthAfterRight);
  });
});
```

- [ ] **Step 10.2: Run the test to verify it fails**

```bash
npm run test -- Layout.test.tsx
```

Expected: FAIL — current mapping is inverted.

- [ ] **Step 10.3: Apply the swap**

Edit `src/components/common/Layout.tsx:304-313` — swap the ArrowLeft and ArrowRight branches so:

```tsx
onKeyDown={(e) => {
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    setInspectorWidth((w) => Math.max(w - 10, MIN_INSPECTOR_WIDTH));
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    setInspectorWidth((w) =>
      Math.min(w + 10, window.innerWidth * 0.5)
    );
  }
}}
```

- [ ] **Step 10.4: Run the tests to verify they pass**

```bash
npm run test -- Layout.test.tsx
```

Expected: PASS.

- [ ] **Step 10.5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/Layout.tsx \
        src/elspeth/web/frontend/src/components/common/Layout.test.tsx
git commit -m "fix(composer): align resize handle keyboard arrows with value direction"
```

---

## Task 11: RunsView Cancelling Badge — Distinct Visual State

**Why:** `RunsView.tsx:262` renders the badge using `STATUS_BADGE_CLASSES[run.status]`, but the displayed text at line 357-361 switches to "cancelling..." when `run.cancel_requested && run.status === "running"`. The visual badge stays blue (running) while the text says cancelling — a mismatch that costs the user time as they re-read to confirm what's happening.

**Files:**
- Modify: `src/components/inspector/RunsView.tsx` — augment the class lookup so a cancel-requested running run uses the warning badge style
- Test: `src/components/inspector/RunsView.test.tsx`

- [ ] **Step 11.1: Locate the badge class lookup**

```bash
grep -n "STATUS_BADGE_CLASSES\|cancel_requested" /home/john/elspeth/src/elspeth/web/frontend/src/components/inspector/RunsView.tsx
```

Identify the exact line that produces the className from `run.status`, and the existing CSS classes available (e.g., `status-badge-running`, `status-badge-cancelled`, `status-badge-failed`). Use an existing class — do not invent new colours unless none of the existing semantic classes fit. The warning class is the closest semantic match.

- [ ] **Step 11.2: Write the failing test**

Append to `src/components/inspector/RunsView.test.tsx`:

```tsx
describe("RunsView — cancelling badge", () => {
  it("uses a warning-style badge class when cancel_requested on a running run", () => {
    const cancellingRun = {
      ...sampleRun,
      status: "running",
      cancel_requested: true,
    };
    setupStoreWithRuns([cancellingRun]);

    render(<RunsView />);

    const badge = screen.getByText(/cancelling/i).closest("[class*='status-badge']");
    expect(badge).not.toBeNull();
    expect(badge!.className).toMatch(/status-badge-(cancelled|cancelling)/);
  });
});
```

Adapt `setupStoreWithRuns` to the existing helpers in the test file.

- [ ] **Step 11.3: Run the test to verify it fails**

Expected: FAIL — current code uses the running class.

- [ ] **Step 11.4: Apply the change**

Replace the badge class derivation around `RunsView.tsx:262` with:

```tsx
const badgeClass =
  run.cancel_requested && run.status === "running"
    ? STATUS_BADGE_CLASSES.cancelled  // or a new "cancelling" key — pick what already exists
    : STATUS_BADGE_CLASSES[run.status];
```

If `STATUS_BADGE_CLASSES.cancelled` does not exist, verify the actual map and substitute the correct existing key (likely `cancelled` based on the project's `--color-status-cancelled` token at `App.css:97`).

- [ ] **Step 11.5: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 11.6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/RunsView.tsx \
        src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx
git commit -m "fix(composer): distinct badge style for runs with cancel_requested"
```

---

## Task 12: Resize Handle — Visible Affordance + Touch Target

**Why:** `App.css:856-866` declares the resize handle hit zone at 20px wide. WCAG 2.5.5 (AAA) recommends 44×44 minimum; 24×24 is the AA Pointer Targets relaxation. The handle is also visually invisible until hover — discoverability is poor on first contact.

**Fix:** Add a static visible 2px vertical bar inside the hit zone; widen the touch hit zone to 44px on touch breakpoints; keep the visual size unchanged on desktop to avoid intruding on the chat panel. Set `cursor: col-resize` statically rather than only on hover so the affordance is visible from a non-hover state too.

**Files:**
- Modify: `src/App.css` — `.resize-handle` and any media-query block

- [ ] **Step 12.1: Read the current rule**

```bash
grep -n "resize-handle" /home/john/elspeth/src/elspeth/web/frontend/src/App.css | head -10
```

- [ ] **Step 12.2: Replace with a static-affordance + responsive-touch rule**

Replace the existing `.resize-handle` declaration block in `App.css` with:

```css
.resize-handle {
  width: 6px;
  cursor: col-resize;
  background: transparent;
  position: relative;
  flex-shrink: 0;
  transition: background var(--transition-fast);
}

/* Static visible bar — 2px wide, centered, low-contrast neutral */
.resize-handle::before {
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  left: 2px;
  width: 2px;
  background: var(--color-border-strong);
  transition: background var(--transition-fast);
}

.resize-handle:hover::before,
.resize-handle:focus-visible::before {
  background: var(--color-focus-ring);
}

/* Touch breakpoint — widen the hit zone without enlarging the visible bar */
@media (max-width: 900px), (pointer: coarse) {
  .resize-handle {
    width: 16px;
  }
  .resize-handle::before {
    left: 7px;
  }
}
```

- [ ] **Step 12.3: Visual check**

In the dev server:
- Desktop: confirm a thin vertical bar is now visible between the chat and inspector panels at all times. Hover lights it up.
- Resize the browser to <900px or open Chrome DevTools' device emulation: confirm the hit zone is wider, but the visible bar is still thin and centered.
- Tab onto the handle with the keyboard: confirm `:focus-visible` highlights it.

- [ ] **Step 12.4: Commit**

```bash
git add src/elspeth/web/frontend/src/App.css
git commit -m "fix(composer): static affordance and touch-friendly hit zone for resize handle"
```

---

## Final: Sanity Sweep

- [ ] **Step F.1: Run the full frontend test suite**

```bash
cd /home/john/elspeth/src/elspeth/web/frontend
npm run test
```

Expected: all green. If any test fails, do not "fix" by changing tests — investigate. Per project memory `feedback_locked_in_buggy_expectations`, a test failure is the contract telling you something — update the test only if it pinned a now-incorrect expectation, and do so deliberately.

- [ ] **Step F.2: Run a typecheck**

```bash
npm run build
```

Expected: build succeeds. The build script begins with `tsc -p tsconfig.app.json --noEmit`; type errors fail the build.

- [ ] **Step F.3: Walk the dev server**

In the running dev server:
1. Create a new session.
2. Send a "create a CSV → text classifier → JSONL sink pipeline" prompt; wait for the LLM to produce a composition.
3. Switch through Spec / Graph / YAML / Runs tabs — confirm screen-reader behaviour (if available; otherwise visually confirm that the YAML view does not flicker on every tab switch).
4. Open the catalog drawer; click "Use in pipeline" on a sink plugin. Confirm the drawer closes and chat input is prefilled.
5. Click Validate. Observe the validation dot transition from grey (○) → green (✓) or orange (⚠) or red (✗) — never grey-after-validate.
6. Click Execute. While running, click Cancel. Confirm the badge changes to a distinct cancelling colour.
7. Open Secrets. Type a name and value, save. (No backend failure simulation needed in this pass; just smoke-test the happy path.)
8. Switch to light theme via the sidebar toggle. Confirm the EMPTY status colour reads correctly. Switch back.

- [ ] **Step F.4: Push and open PR (operator-gated)**

```bash
git push origin RC5.1
gh pr create --title "Composer UX tier-1 improvements (a11y, validation indicator, code blocks, catalog action)" \
  --body "$(cat <<'EOF'
## Summary
Implements the high-leverage findings from the 2026-05-09 ux-critic review of the composer frontend.

## Test plan
- [x] vitest suite green (`npm run test`)
- [x] typecheck green (`npm run build`)
- [x] manual dev-server walkthrough per plan F.3
- [ ] reviewer to spot-check screen-reader behaviour on the validation banner and chat composing indicator
- [ ] reviewer to spot-check light-theme contrast on the EMPTY status badge

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Do not push or open a PR without operator approval — per the global instructions on actions visible to others.

---

## Self-Review

**Spec coverage:**
- Major #1 (validation colour) → Task 1 ✓
- Major #2 (aria-live tab panel) → Task 4 ✓
- Major #3 (nested live regions in chat) → Task 5 ✓
- Major #4 (banner alert→status) → Task 6 ✓
- Major #5 (SecretsPanel form-lock) → Task 3 ✓
- Major #6 (RunsView aria-expanded) → Task 7 ✓
- Major #7 (Markdown code blocks) → Task 8 ✓
- Major #8 (Catalog "Use in pipeline") → Task 9 ✓
- Minor (light theme `--color-status-empty`) → Task 2 ✓
- Minor (cancelling badge mismatch) → Task 11 ✓
- Minor (resize handle arrow direction) → Task 10 ✓
- Minor (resize handle affordance + touch target) → Task 12 ✓

**Deliberately deferred** (acknowledged, not in this plan):
- Strategic graph-edit mode (replacing chat-only mutation with direct manipulation) — separate effort, possibly post-demo.
- Full WCAG audit and rendered-contrast measurement — `lyra-ux-designer:accessibility-audit` skill runs this separately.
- Empty-state copy consistency, session-filter always-visible, schema-fields `<dl>` semantics, visible product wordmark — design calls best made with the operator, not an engineer.

**Placeholder scan:** No "TBD"/"TODO"/"add appropriate handling"/"similar to" strings. Every code step has runnable code; every test step has runnable test code; every command has expected output.

**Type/name consistency:** `PREFILL_CHAT_INPUT_EVENT` is defined in PluginCard.tsx (Task 9.1, 9.4) and consumed in ChatInput.tsx (Task 9.5) using the same import path. `STATUS_BADGE_CLASSES` is referenced consistently (verified read of RunsView pending in 11.1). `expandedRunId` is referenced consistently in Task 7. Validation status keys (`unchecked`, `valid`, `warning`, `invalid`) are used consistently in Task 1.
