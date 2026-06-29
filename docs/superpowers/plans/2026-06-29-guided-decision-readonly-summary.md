# Guided decision read-only summary + lead-with-rationale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render every guided decision as a compact read-only summary by default — leading with the dynamic LLM build rationale — with an Edit toggle that reveals the existing editable form (interactive composer only; the passive tutorial is summary-only).

**Architecture:** `SchemaFormTurn` gains a `view: "summary" | "edit"` local state (default `"summary"`). The summary renders the visible knobs as a read-only `<dl>` (short scalars inline; JSON via the shared `CodeBlock`; blob `path` masked to its friendly basename). A non-tutorial **Edit** button switches to the unchanged `KnobFieldRenderer` form; a **Done editing** button returns (disabled while any field has a parse error). `ChatPanel` leads the `guided-current-decision` block with the current step's latest assistant `chat_history` rationale as an `<h2>` (falling back to `GUIDED_STEP_PURPOSES`). The submit payload, `canSubmit` gating, path-mask doctrine, and backend are unchanged — `view` is pure presentation over the same `values`.

**Tech Stack:** React 18 + TypeScript, Vitest + Testing Library + jest-axe, `prism-react-renderer` (via the existing `CodeBlock`), Zustand stores, design-system CSS tokens.

## Global Constraints

- **Frontend presentation only.** No change to the guided `/respond` contract, the `GuidedRespondRequest` submit shape, `canSubmit`/`fieldHasError`/`submittedValue` semantics, the `interpretationEventsStore`, or any backend.
- **Submit parity is load-bearing.** `Continue` / `Apply recipe` must emit the *identical* `GuidedRespondRequest` they do today: prefilled values when unedited, edited values when edited. Existing submit tests must keep passing (adapted only to click `Edit` first where they edit a field).
- **Path-leak doctrine preserved.** The blob `path` knob's absolute `storage_path` must never render verbatim in the summary — show `friendlyBlobRef(...)`. The *real* path must still be what `Continue` submits. The dedicated tutorial path-leak test is **retargeted, not deleted** (security boundary — MEMORY: guided schema_form path leak).
- **a11y is gated, not assumed.** Add `SchemaFormTurn` to the jest-axe audit (`src/test/a11y/components.a11y.test.tsx`) so the new `<dl>`/`role=note`/toggle DOM is actually checked. Add `colorContrast.test.ts` assertions for every new token/background pairing.
- **Verification commands (exact, verified against `package.json`):** `npm run typecheck`, `npm run lint`, `npm run lint:css` (NOT `stylelint`), `npm test` (vitest run; trailing arg filters by file), `npm run build`, `npm run test:e2e:staging`.
- **Reuse, don't duplicate.** JSON values render through the existing `CodeBlock` (`prettyJson`); the path mask reuses `friendlyBlobRef` in `SchemaFormTurn.tsx`.
- **Tutorial is summary-only** (no Edit button) and presupposes all required knobs are prefilled (document this assumption in code — `canSubmit` must be true at mount, since there is no Edit affordance to fix an unmet field).

---

### Task 1: `SchemaFormTurn` read-only summary view (default)

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx`
- Test: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx`

**Interfaces:**
- Consumes: `SchemaFormPayload`, `KnobField`, `values` state, `visibleFields()`, `friendlyBlobRef()`, `handleContinue()`, `CodeBlock` (`{ code, prettyJson, ariaLabel, showCopy }`, imported from `../CodeBlock`).
- Produces: `summaryValueNode(field: KnobField, value: unknown): ReactNode`; `view: "summary" | "edit"` state (default `"summary"`) consumed by Tasks 2–3. Summary rows carry `className="guided-schema-summary-row"`; JSON values render a `CodeBlock` (element with a `data-codeblock-format` attribute).

- [ ] **Step 1: Write the failing tests** (append to `SchemaFormTurn.test.tsx`)

```tsx
describe("read-only summary view", () => {
  it("renders prefilled scalar knobs as read-only text, not editable controls", () => {
    render(
      <SchemaFormTurn
        payload={pluginPayload(
          [
            field({ name: "encoding", label: "Encoding", kind: "text" }),
            field({ name: "skip_rows", label: "Skip Rows", kind: "number-int" }),
            field({ name: "enabled", label: "Enabled", kind: "checkbox" }),
          ],
          { encoding: "utf-8", skip_rows: 0, enabled: true },
        )}
        onSubmit={vi.fn()}
      />,
    );
    // Summary is the default — no schema-form inputs are rendered up front.
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();
    expect(screen.getByText("Encoding")).toBeInTheDocument();
    expect(screen.getByText("utf-8")).toBeInTheDocument();
    expect(screen.getByText("Skip Rows")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
    expect(screen.getByText("Yes")).toBeInTheDocument(); // checkbox -> Yes/No
  });

  it("renders empty/undefined scalar values as (none)", () => {
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "columns", label: "Columns", kind: "string-list" })], { columns: [] })}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByText("(none)")).toBeInTheDocument();
  });

  it("renders a JSON-shaped knob value through CodeBlock (pretty/highlighted)", () => {
    const { container } = render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "schema", label: "Schema", kind: "json-object" })], {
          schema: { mode: "observed", guaranteed_fields: ["url"] },
        })}
        onSubmit={vi.fn()}
      />,
    );
    expect(container.querySelector("[data-codeblock-format]")).not.toBeNull();
  });

  it("masks an absolute blob path to its friendly basename in the summary", () => {
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "path", label: "Path", kind: "text" })], {
          path: "/home/u/data/blobs/sess/cb7f1f46-b724-4472-9acb-1680cefef45e_project_pages.json",
        })}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByText("project_pages.json")).toBeInTheDocument();
    expect(screen.queryByText(/\/home\/u\/data\/blobs/)).not.toBeInTheDocument();
  });

  it("hides a visible_when-gated field from the summary when its predicate is unmet", () => {
    render(
      <SchemaFormTurn
        payload={pluginPayload(
          [
            field({ name: "provider", label: "Provider", kind: "enum", enum: ["azure", "openrouter"] }),
            field({ name: "deployment_name", label: "Deployment", kind: "text", visible_when: { field: "provider", equals: "azure" } }),
          ],
          { provider: "openrouter" },
        )}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.queryByText("Deployment")).not.toBeInTheDocument();
  });

  // FORWARD GUARD (not a genuine RED — today's form already submits prefilled
  // values on Continue): pins that the summary-default path keeps submit parity.
  it("submits the prefilled values verbatim from the summary (no edit)", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "encoding", label: "Encoding", kind: "text" })], { encoding: "utf-8" })}
        onSubmit={onSubmit}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ edited_values: expect.objectContaining({ options: { encoding: "utf-8" } }) }),
    );
  });
});
```

- [ ] **Step 2: Run, verify the new RED tests fail**

Run: `npm run --prefix src/elspeth/web/frontend test -- SchemaFormTurn`
Expected: the summary-render tests FAIL (form currently renders editable controls, no `.guided-schema-summary-row`, no "(none)"/"Yes" text). The "submits prefilled" test already passes (forward guard).

- [ ] **Step 3: Implement the summary view** in `SchemaFormTurn.tsx`:
- Add imports: `import { CodeBlock } from "../CodeBlock";` and `import { useState, type ReactNode } from "react";` (merge with existing `useId`/`useState` import).
- Add the formatter at module scope:

```tsx
function summaryValueNode(field: KnobField, value: unknown): ReactNode {
  if (field.kind === "checkbox") return Boolean(value) ? "Yes" : "No";
  if (field.kind === "json-object" || field.kind === "json-array" || field.kind === "json-value") {
    return (
      <CodeBlock
        code={typeof value === "string" ? value : JSON.stringify(value ?? null)}
        prettyJson
        showCopy={false}
        ariaLabel={field.label}
      />
    );
  }
  if (field.kind === "string-list") {
    const text = Array.isArray(value) ? value.join(", ") : typeof value === "string" ? value : "";
    return text === "" ? "(none)" : text;
  }
  const raw = value === null || value === undefined ? "" : String(value);
  if (field.name === "path" && raw.startsWith("/")) return friendlyBlobRef(raw);
  return raw === "" ? "(none)" : raw;
}
```

- Add state: `const [view, setView] = useState<"summary" | "edit">("summary");`
- Build the summary node:

```tsx
const summary = (
  <dl className="guided-schema-summary">
    {visibleFields().map((f) => (
      <div className="guided-schema-summary-row" key={f.name}>
        <dt className="guided-schema-summary-label">{f.label}</dt>
        <dd className="guided-schema-summary-value">{summaryValueNode(f, values[f.name])}</dd>
      </div>
    ))}
  </dl>
);
```

- In the returned JSX render `view === "summary" ? summary : (<div className="guided-schema-fields">{visibleFields().map(... existing KnobFieldRenderer ...)}</div>)`. Keep `RecipeContextHeader`, the actions block, and (for now) the existing standalone teaching `<p>` — Task 3 relocates it.

- [ ] **Step 4: Run, verify pass**

Run: `npm run --prefix src/elspeth/web/frontend test -- SchemaFormTurn`
Expected: the new summary tests PASS. (Existing "editable control" / edit-and-submit tests now FAIL — adapted in Task 2.)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx
git commit -m "feat(web/composer): read-only summary view for guided decisions"
```

---

### Task 2: Edit toggle + invalid-state guard + adapt existing tests

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx`
- Test: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx`

**Interfaces:**
- Consumes: `view`/`setView` (Task 1), `isTutorial`, `canSubmit()`, `fieldHasError()`, `visibleFields()`, `values`.
- Produces: an `Edit` button (`!isTutorial && view==="summary"`); a `Done editing` button (`view==="edit"`, disabled when any visible field `fieldHasError`); a summary-mode banner (`!isTutorial && view==="summary" && !canSubmit()`) with class `guided-schema-summary-needs-edit`.

- [ ] **Step 1: Write the failing tests** (append)

```tsx
describe("edit toggle", () => {
  it("reveals the editable form on Edit, and returns on Done editing (non-tutorial)", async () => {
    const user = userEvent.setup();
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "encoding", label: "Encoding", kind: "text" })], { encoding: "utf-8" })}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.queryByRole("textbox", { name: "Encoding" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Edit" }));
    expect(screen.getByRole("textbox", { name: "Encoding" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Done editing" }));
    expect(screen.queryByRole("textbox", { name: "Encoding" })).not.toBeInTheDocument();
  });

  it("does NOT render an Edit button in tutorial mode", () => {
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "encoding", label: "Encoding", kind: "text" })], { encoding: "utf-8" })}
        onSubmit={vi.fn()}
        isTutorial
      />,
    );
    expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
  });

  it("submits the edited value after editing via the form", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "encoding", label: "Encoding", kind: "text" })], { encoding: "utf-8" })}
        onSubmit={onSubmit}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Edit" }));
    const input = screen.getByRole("textbox", { name: "Encoding" });
    await user.clear(input);
    await user.type(input, "latin-1");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ edited_values: expect.objectContaining({ options: { encoding: "latin-1" } }) }),
    );
  });

  it("blocks Done editing while a field holds invalid JSON, then allows it once corrected", async () => {
    const user = userEvent.setup();
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "cfg", label: "Config", kind: "json-object" })], { cfg: { ok: true } })}
        onSubmit={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Config" }), { target: { value: "{bad" } });
    expect(screen.getByRole("button", { name: "Done editing" })).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: "Config" }), { target: { value: '{"ok":false}' } });
    expect(screen.getByRole("button", { name: "Done editing" })).toBeEnabled();
  });

  it("shows a needs-edit banner in the non-tutorial summary when an unfilled required field blocks Continue", () => {
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "token", label: "Token", kind: "text", required: true })])}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByText(/click Edit to review/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `npm run --prefix src/elspeth/web/frontend test -- SchemaFormTurn`
Expected: FAIL — no Edit / Done editing / banner yet.

- [ ] **Step 3: Implement** in the actions block (before the primary button):

```tsx
{!isTutorial && view === "summary" && (
  <button type="button" className="guided-turn-secondary guided-schema-edit-toggle" onClick={() => setView("edit")}>
    Edit
  </button>
)}
{!isTutorial && view === "edit" && (
  <button
    type="button"
    className="guided-turn-secondary guided-schema-edit-toggle"
    onClick={() => setView("summary")}
    disabled={visibleFields().some((f) => fieldHasError(f, values[f.name]))}
  >
    Done editing
  </button>
)}
```

And, in summary view, render a banner above the actions when Continue is blocked (non-tutorial only — the tutorial has no Edit affordance and presupposes valid prefill):

```tsx
{!isTutorial && view === "summary" && !canSubmit() && (
  <p className="guided-schema-summary-needs-edit" role="status">
    Some values need attention — click Edit to review.
  </p>
)}
```

- [ ] **Step 4: Adapt the existing editable-control tests** — prefix each with `const user = userEvent.setup()` + `await user.click(screen.getByRole("button", { name: "Edit" }))` before the first form-control interaction, and make the test `async`. Adapt these (verified to exist in `SchemaFormTurn.test.tsx`):
  - `"renders %s as an editable control"` (line 84, it.each) — Edit-click before `getByRole(role,…)`.
  - `"renders checkbox, enum, string-list, and JSON kinds"` (line 90).
  - `"uses aria-describedby for field descriptions"` (line 113).
  - `"submits only visible fields and drops variant state on discriminator change"` (line 128).
  - `"disables continue when a required text field is cleared"` (line 175).
  - `"submits nullable cleared text as null"` (line 197).
  - `"supports keyboard editing of string-list fields"` (line 221).
  - `"parses number and JSON field values on submit"` (line 245).
  - `"renders recipe context and submits recipe slot decisions"` (line 283) — Edit-click before typing Threshold (the recipe header + `Apply recipe` stay visible in summary).
  - The `required marking and inline validation` describe block (lines 357, 396, 411, 436) — Edit-click first in each.
  - `"does NOT mask the path outside tutorial mode…"` (line 526) — Edit-click first, then assert the input shows the real path / not readonly.
  - UNCHANGED (no Edit needed): `"renders an enabled Apply recipe…"` (line 42, knobs==[] → empty summary), `"submits the REAL absolute path…in tutorial mode"` (line 544, tutorial summary → Continue).
  - **RETARGET (do not delete)** `"masks an absolute blob storage_path to its friendly basename (read-only) in tutorial mode"` (line 506) → assert the masked basename appears as summary text and no `.guided-schema-input` exists:

```tsx
it("masks an absolute blob storage_path to its friendly basename in tutorial summary mode", () => {
  const { container } = render(
    <SchemaFormTurn
      payload={pluginPayload([field({ name: "path", kind: "text", required: true })], {
        path: "/home/john/elspeth/data/blobs/sess/cb7f1f46-b724-4472-9acb-1680cefef45e_project_pages.json",
      })}
      onSubmit={vi.fn()}
      isTutorial
    />,
  );
  expect(screen.getByText("project_pages.json")).toBeInTheDocument();
  expect(screen.queryByText(/\/home\/john\/elspeth\/data\/blobs/)).not.toBeInTheDocument();
  expect(container.querySelector(".guided-schema-input")).toBeNull();
});
```

- [ ] **Step 5: Run, verify pass + migration sweep**

Run: `npm run --prefix src/elspeth/web/frontend test -- SchemaFormTurn`
Expected: PASS. Sweep check: every test that interacts with a `textbox`/`spinbutton`/`combobox` either starts with an Edit click or uses the `recipe_decision` empty-knobs path. (Grep the test file for `getByRole("textbox"` / `"spinbutton"` / `"combobox"` and confirm an `Edit` click precedes each, except the recipe empty-knobs and summary-text cases.)

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx
git commit -m "feat(web/composer): Edit toggle + invalid-state guard for guided decision summary"
```

---

### Task 3: Move the validation-failure caveat into the summary row (remove dead branch)

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx`
- Test: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx`

**Interfaces:**
- Consumes: `showValidationFailureTeaching` (true only when `isTutorial`), `TUTORIAL_VALIDATION_FAILURE_CAVEAT`, the summary rows.
- Produces: the caveat rendered inside the `on_validation_failure` summary `dd` (class `guided-schema-summary-caveat`, `role="note"`). The old standalone `guided-schema-teaching` `<p>` is removed entirely (it is unreachable once the form is summary-first: `showValidationFailureTeaching` requires `isTutorial`, and the tutorial never enters edit view).

- [ ] **Step 1: Update the caveat tests** — the three existing `tutorial validation-failure teaching copy` tests (lines 474/485/494) keep their present/absent assertions; add a positional test:

```tsx
it("attaches the caveat to the on_validation_failure summary row in tutorial mode", () => {
  const { container } = render(
    <SchemaFormTurn
      payload={pluginPayload([field({ name: "on_validation_failure", label: "On Validation Failure", kind: "text", required: true })], {
        on_validation_failure: "discard",
      })}
      onSubmit={vi.fn()}
      isTutorial
    />,
  );
  const caveat = container.querySelector(".guided-schema-summary-caveat");
  expect(caveat).not.toBeNull();
  expect(caveat?.textContent ?? "").toMatch(/quarantine sink/i);
});
```

- [ ] **Step 2: Run, verify the positional test fails**

Run: `npm run --prefix src/elspeth/web/frontend test -- SchemaFormTurn`
Expected: FAIL — `.guided-schema-summary-caveat` does not exist (caveat still the standalone `<p className="guided-schema-teaching">`).

- [ ] **Step 3: Implement** — in the summary row `dd`, render the caveat for the `on_validation_failure` row:

```tsx
<dd className="guided-schema-summary-value">
  {summaryValueNode(f, values[f.name])}
  {showValidationFailureTeaching && f.name === "on_validation_failure" && (
    <p className="guided-schema-summary-caveat" role="note">
      {TUTORIAL_VALIDATION_FAILURE_CAVEAT}
    </p>
  )}
</dd>
```

Remove the standalone `{showValidationFailureTeaching && (<p className="guided-schema-hint guided-schema-teaching" role="note">…</p>)}` block completely (no edit-mode rendering — it would be dead code, since `showValidationFailureTeaching` implies `isTutorial` and the tutorial has no edit view).

- [ ] **Step 4: Run, verify pass**

Run: `npm run --prefix src/elspeth/web/frontend test -- SchemaFormTurn`
Expected: PASS (the three present/absent tests + the positional test).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx
git commit -m "feat(web/composer): move validation-failure caveat into the decision summary row"
```

---

### Task 4: Lead the decision with the dynamic build rationale (as the heading)

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/guided/guidedRationale.ts`
- Create: `src/elspeth/web/frontend/src/components/chat/guided/guidedRationale.test.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` (the `guided-current-decision-copy` block ~lines 1437-1449; import near the top)
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx` (the guided "Current decision" heading assertion ~line 561)

**Interfaces:**
- Consumes: `GuidedSession.chat_history` (`ChatTurn[]` = `{ role, content, seq, step, ts_iso }`), `GuidedStep`, `GUIDED_STEP_PURPOSES`.
- Produces: `latestAssistantRationale(session: GuidedSession): string | null` — highest-`seq` assistant turn whose `step === session.step`, else `null`.

- [ ] **Step 1: Write the failing helper test** (`guidedRationale.test.ts`) — the factory sets ONLY real `GuidedSession` fields (`step`, `history`, `terminal`, `chat_history`, `chat_turn_seq`, `profile`):

```ts
import { describe, expect, it } from "vitest";
import { latestAssistantRationale } from "./guidedRationale";
import type { GuidedSession } from "@/types/guided";

function session(overrides: Partial<GuidedSession>): GuidedSession {
  return {
    step: "step_1_source",
    history: [],
    terminal: null,
    chat_history: [],
    chat_turn_seq: 0,
    profile: null,
    ...overrides,
  } as GuidedSession;
}

describe("latestAssistantRationale", () => {
  it("returns the highest-seq assistant turn for the current step", () => {
    const s = session({
      step: "step_1_source",
      chat_history: [
        { role: "user", content: "go", seq: 1, step: "step_1_source", ts_iso: "t" },
        { role: "assistant", content: "Source created as a 3-row CSV.", seq: 2, step: "step_1_source", ts_iso: "t" },
        { role: "assistant", content: "Sink set.", seq: 4, step: "step_2_sink", ts_iso: "t" },
      ],
    });
    expect(latestAssistantRationale(s)).toBe("Source created as a 3-row CSV.");
  });

  it("returns null when no assistant turn exists for the step", () => {
    const s = session({
      step: "step_2_sink",
      chat_history: [{ role: "user", content: "go", seq: 1, step: "step_2_sink", ts_iso: "t" }],
    });
    expect(latestAssistantRationale(s)).toBeNull();
  });
});
```

- [ ] **Step 2: Run, verify fail** — `npm run --prefix src/elspeth/web/frontend test -- guidedRationale` → FAIL (module not found).

- [ ] **Step 3: Implement** (`guidedRationale.ts`):

```ts
import type { GuidedSession } from "@/types/guided";

/**
 * The current step's latest assistant rationale (the LLM's "what I built"
 * summary), used as the prominent decision headline. Highest-seq assistant
 * turn whose step matches the active step; null when none (the caller falls
 * back to the static step purpose).
 */
export function latestAssistantRationale(session: GuidedSession): string | null {
  let best: { seq: number; content: string } | null = null;
  for (const turn of session.chat_history) {
    if (turn.role !== "assistant" || turn.step !== session.step) continue;
    if (best === null || turn.seq > best.seq) best = { seq: turn.seq, content: turn.content };
  }
  return best === null ? null : best.content;
}
```

- [ ] **Step 4: Run, verify pass** — `npm run --prefix src/elspeth/web/frontend test -- guidedRationale` → PASS.

- [ ] **Step 5: Wire into ChatPanel** — add the import near the other `./guided/...` imports:

```tsx
import { latestAssistantRationale } from "./guided/guidedRationale";
```

In the existing IIFE that wraps the `guided-current-decision` section (~line 1425), compute the rationale in the outer body (no nested IIFE) and render the **rationale as the `<h2>`** (carrying the section's heading id), with the eyebrow demoted to an `aria-hidden` `<p>`:

```tsx
{(() => {
  const stepIsSendDriven =
    isTutorial && (lockedChatPrompt?.[guidedSession.step] ?? "") !== "";
  const rationale = latestAssistantRationale(guidedSession);
  return (
    <section
      className={stepIsSendDriven ? "guided-current-decision guided-current-decision--tutorial" : "guided-current-decision"}
      aria-labelledby="guided-current-decision-heading"
    >
      <div className="guided-current-decision-copy">
        <p className="guided-current-decision-eyebrow" aria-hidden="true">Current decision</p>
        <h2 id="guided-current-decision-heading" className="guided-current-decision-rationale">
          {rationale ?? GUIDED_STEP_PURPOSES[guidedSession.step]}
        </h2>
        {stepIsSendDriven && !tutorialStepBuilt && (
          <p className="guided-current-decision-tutorial-note">
            You don't need to fill this in by hand — press <strong>Send</strong> above and the assistant builds this step.
            Then confirm the decision below to continue.
          </p>
        )}
      </div>
      {/* …unchanged: guidedLogRef log + GuidedTurn + "Saving decision…" pending… */}
    </section>
  );
})()}
```

(The `<h2>` keeps a heading-level element in the section per the spec; `aria-labelledby` now resolves to the rationale text.)

- [ ] **Step 6: Adapt the ChatPanel heading test** — `ChatPanel.test.tsx:~561` currently asserts `getByRole("heading", { name: /current decision/i })`. After this change the heading's accessible name is the rationale text (or the step-purpose fallback). Update it to assert the heading carries the decision lead. Determine the test's guided fixture: if its `guidedSession.chat_history` has no assistant turn for the active step, the fallback `GUIDED_STEP_PURPOSES[step]` renders, so assert that; if it has one, assert that rationale. Concretely, change the assertion to match the rendered lead text, e.g.:

```tsx
// The decision now leads with the dynamic rationale (or the step-purpose
// fallback) AS the heading; "Current decision" is a decorative eyebrow.
expect(screen.getByRole("heading", { level: 2 })).toHaveAccessibleName(
  /choose the input and confirm what elspeth can read|source created as/i,
);
```

If a sibling assertion checks `getByText(/choose the input/i)` as a paragraph, update/remove it — that copy now lives in the `<h2>` (only when no assistant rationale). Run the ChatPanel suite to discover the exact fixture shape and reconcile.

- [ ] **Step 7: Run typecheck + ChatPanel suite**

Run: `npm run --prefix src/elspeth/web/frontend typecheck && npm run --prefix src/elspeth/web/frontend test -- ChatPanel`
Expected: typecheck exit 0; ChatPanel suite green (heading assertion adapted).

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/guided/guidedRationale.ts src/elspeth/web/frontend/src/components/chat/guided/guidedRationale.test.ts src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx
git commit -m "feat(web/composer): lead guided decision with the dynamic LLM rationale heading"
```

---

### Task 5: a11y + contrast gates, styling, full verification, deploy, e2e

**Files:**
- Modify: `src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx` (add `SchemaFormTurn` to the audit)
- Modify: `src/elspeth/web/frontend/src/styles/colorContrast.test.ts` (assert the new pairing)
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/guided.css` (new classes + specificity + orphan cleanup)
- Modify: `src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts` (one-shot summary DOM assertion)

- [ ] **Step 1: Add `SchemaFormTurn` to the jest-axe audit** — in `components.a11y.test.tsx`, add three render cases to `AUDITED_COMPONENTS` (summary, edit, tutorial) and update `EXPECTED_AUDITED_COMPONENTS_SORTED` to include `"SchemaFormTurn"` (follow the file's existing entry shape; each case renders a `plugin_options` payload with a couple of knobs, then `expect(await axe(container)).toHaveNoViolations()`; the edit case clicks `Edit` first). Run: `npm run --prefix src/elspeth/web/frontend test -- components.a11y` → PASS (fix any `<dl>`/`role=note`/toggle violation at the component, not by relaxing the test).

- [ ] **Step 2: Add the contrast assertions** — in `colorContrast.test.ts`, assert `--color-text-secondary` on `--color-bg` ≥ 4.5:1 in both themes (covers `.guided-current-decision-eyebrow` and `.guided-schema-summary-caveat` small text). Run: `npm run --prefix src/elspeth/web/frontend test -- colorContrast` → PASS. If it fails, deepen the **token** (mirror to the website palette per the shared-palette rule), don't rewire call sites.

- [ ] **Step 3: Add styles** in `guided.css` using the tokens already used in that file:
  - `.guided-schema-summary { display: grid; gap: var(--space-2); margin: var(--space-2) 0; }`
  - `.guided-schema-summary-row { display: grid; grid-template-columns: minmax(8rem, max-content) 1fr; gap: var(--space-3); align-items: baseline; }`
  - `.guided-schema-summary-label { color: var(--color-text-secondary); font-weight: 500; }`
  - `.guided-schema-summary-value { color: var(--color-text-primary); min-width: 0; }`
  - `.guided-schema-summary-caveat { margin: var(--space-1) 0 0; font-size: var(--font-size-sm); color: var(--color-text-secondary); }`
  - `.guided-schema-summary-needs-edit { font-size: var(--font-size-sm); color: var(--color-text-secondary); }`
  - `.guided-current-decision-eyebrow { margin: 0; font-size: var(--font-size-sm); letter-spacing: .04em; text-transform: uppercase; color: var(--color-text-secondary); }`
  - **High-specificity rationale (beats the existing `.guided-current-decision-copy h2` at 0-1-1):** `.guided-current-decision-copy .guided-current-decision-rationale { margin: var(--space-1) 0 var(--space-3); font-size: var(--font-size-lg); line-height: 1.4; color: var(--color-text-primary); text-transform: none; }`
  - **Orphan cleanup:** the rationale is now the only `h2` in `.guided-current-decision-copy`, so the existing compound rules `.guided-current-decision-copy h2` (≈line 108) and `.guided-current-decision--tutorial .guided-current-decision-copy h2` (≈line 133) now target it — confirm they don't fight the rationale rule (the rationale rule's specificity 0-2-0 wins over both 0-1-1 / 0-2-1? check: tutorial rule is 0-2-1, which BEATS 0-2-0 — so either raise the rationale rule to `.guided-current-decision--tutorial .guided-current-decision-copy .guided-current-decision-rationale` too, or drop `h2` from those compound rules). Simplest: change those two rules to target `.guided-current-decision-eyebrow` instead of `h2` (the eyebrow is what should stay small), and let `.guided-current-decision-rationale` own the heading style. Verify in the browser/`lint:css`.

- [ ] **Step 4: Full FE gate set**

```bash
cd src/elspeth/web/frontend
npm run typecheck && npm run lint && npm run lint:css && npm test
```
Expected: all exit 0; `colorContrast.test.ts` + `components.a11y.test.tsx` green; the whole vitest suite green.

- [ ] **Step 5: Strengthen the e2e with a one-shot summary assertion** — in `tutorial-reliability.staging.spec.ts` `driveGuidedWalk`, add a one-time check inside the while-loop: when a `.guided-schema-summary` is first visible, assert no editable schema input is shown (the decision is a summary, not a form), then set a flag so it runs once:

```ts
// (declare before the loop) let assertedSummary = false;
if (!assertedSummary && (await page.locator(".guided-schema-summary").first().isVisible().catch(() => false))) {
  assertedSummary = true;
  if ((await page.locator(".guided-schema-input").count().catch(() => 0)) > 0) {
    throw new Error("guided decision rendered an editable form, expected a read-only summary");
  }
}
```

- [ ] **Step 6: Build + redeploy staging**

```bash
cd src/elspeth/web/frontend && npm run build
```
(FE-only → no service restart.) Verify the served bundle hash matches the new `dist/index.html` and `/api/health` is 200.

- [ ] **Step 7: Run the staging tutorial e2e battery**

```bash
cd src/elspeth/web/frontend
STAGING_BASE_URL=https://elspeth.foundryside.dev STAGING_USERNAME=dta_user STAGING_PASSWORD=dta_pass \
  PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev npm run test:e2e:staging
```
Expected: `1 passed`; RunRecord `dim_a_tutorial_completed: true`, `dim_b_realsystem_passed: true`, run POST 200; the one-shot summary assertion did not throw. Capture the source-decision screenshot from the trace for a visual check of the rationale-led summary.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx src/elspeth/web/frontend/src/styles/colorContrast.test.ts src/elspeth/web/frontend/src/components/chat/guided/guided.css src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts
git commit -m "style+test(web/composer): a11y/contrast gates, compact summary styling, e2e summary guard"
```

## Self-Review

- **Spec coverage:** read-only summary default for all decisions (Task 1) ✓; Edit toggle non-tutorial / tutorial summary-only (Task 2) ✓; lead-with-rationale as heading (Task 4) ✓; demote caveat (Task 3) ✓; styling/a11y/contrast gated (Task 5) ✓; submit/contract parity (Global + Task 1 parity test + Task 2 edited-value test) ✓; path-leak preserved + retargeted test (Task 1 + Task 2) ✓; invalid-state trap closed (Task 2 Done-editing guard + summary banner) ✓.
- **Review findings folded in:** `lint:css` (Global, Task 5) ✓; rationale=`h2` + ChatPanel test adapt (Task 4) ✓; CSS specificity + orphan cleanup (Task 5 Step 3) ✓; SchemaFormTurn in axe audit (Task 5 Step 1) ✓; contrast assertion (Task 5 Step 2) ✓; dead caveat branch removed (Task 3) ✓; factory without `session_id`/`next_turn` (Task 4) ✓; false-RED notes (Task 1/2) ✓; flatten IIFE (Task 4 Step 5) ✓; migration sweep (Task 2 Step 5) ✓; tutorial-prefill assumption documented (Global) ✓; e2e DOM assertion (Task 5 Step 5) ✓.
- **Placeholder scan:** none. CSS token names defer to "tokens already used in `guided.css`" (a real instruction; the palette lives in the file the task opens).
- **Type consistency:** `view: "summary" | "edit"`; `summaryValueNode(field, value)`; `latestAssistantRationale(session): string | null` consumed with `?? GUIDED_STEP_PURPOSES[...]`; `CodeBlock` props (`code`/`prettyJson`/`showCopy`/`ariaLabel`) match.
