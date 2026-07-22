# Freeform Operator Primer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brief freeform introduction with the approved static, casework-based primer for non-technical professional operators.

**Architecture:** Keep the existing account-level dismissal behavior unchanged. Expand only `FreeformIntroduction` into semantic, scannable sections and adjust its local CSS so the longer text remains a quiet bounded card at desktop and narrow widths.

**Tech Stack:** React, TypeScript, semantic HTML, CSS design tokens, Vitest, Testing Library, Playwright.

---

### Task 1: Render the operator primer

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/FreeformIntroduction.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/FreeformIntroduction.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/chat.css`

- [ ] **Step 1: Write the failing content and structure test**

Replace the old brief-copy assertion with tests that require both section headings, every defined term, the casework metaphor, and the existing dismissal action:

```tsx
expect(screen.getByRole("heading", { name: "How pipelines work" })).toBeVisible();
expect(screen.getByRole("heading", { name: "The three building blocks" })).toBeVisible();
expect(screen.getByRole("heading", { name: "Wiring the flow" })).toBeVisible();

for (const term of [
  "Sources",
  "Transforms",
  "Sinks",
  "Gate",
  "Fork",
  "Coalesce",
  "Aggregate",
  "Queue",
  "Expand",
]) {
  expect(screen.getByText(term, { selector: "dt" })).toBeVisible();
}

expect(screen.getByText(/each record as a case moving through/i)).toBeVisible();
expect(screen.getByRole("button", { name: "Don’t show this again" })).toBeVisible();
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run src/components/chat/FreeformIntroduction.test.tsx
```

Expected: the new heading and definition-term assertions fail against the current brief introduction.

- [ ] **Step 3: Implement the approved semantic primer**

Keep the existing store selectors, loaded/dismissed gate, error handling, and button behavior. Replace only the rendered content with this structure and exact approved wording:

```tsx
<section
  className="freeform-introduction"
  aria-labelledby="freeform-introduction-title"
>
  <h2 id="freeform-introduction-title">How pipelines work</h2>
  <p className="freeform-introduction-lead">
    A pipeline is a controlled route for information. You choose what enters,
    what happens to it, and where the result goes. ELSPETH records each step so
    you can review how every output was produced.
  </p>

  <section className="freeform-introduction-section" aria-labelledby="pipeline-building-blocks">
    <h3 id="pipeline-building-blocks">The three building blocks</h3>
    <dl className="freeform-introduction-definitions">
      <div><dt>Sources</dt><dd>bring records into the pipeline from files, databases, APIs, or text. ELSPETH tracks each incoming record through the run.</dd></div>
      <div><dt>Transforms</dt><dd>examine or change records. They can clean fields, enrich content, apply an LLM, or prepare data for the next step.</dd></div>
      <div><dt>Sinks</dt><dd>receive records at the end of a route. They can write results to files, data stores, or other configured destinations; records requiring attention can follow a separate route.</dd></div>
    </dl>
  </section>

  <section className="freeform-introduction-section" aria-labelledby="pipeline-wiring-flow">
    <h3 id="pipeline-wiring-flow">Wiring the flow</h3>
    <p>Wiring is the set of connections between these components. A simple pipeline runs from source to transforms to sink. For a more involved flow, think of each record as a case moving through a controlled workplace:</p>
    <dl className="freeform-introduction-definitions">
      <div><dt>Gate</dt><dd>is a sorting desk. It sends each case along the appropriate route according to a stated condition.</dd></div>
      <div><dt>Fork</dt><dd>sends controlled copies of one case to several specialist teams. ELSPETH tracks each parallel path independently.</dd></div>
      <div><dt>Coalesce</dt><dd>waits for the required specialist responses, then combines their findings into one case that can continue.</dd></div>
      <div><dt>Aggregate</dt><dd>brings a group of cases together for batch work, such as producing totals, statistics, or a report.</dd></div>
      <div><dt>Queue</dt><dd>is a shared inbox. It accepts cases from several upstream teams and feeds one next step while keeping every case separate.</dd></div>
      <div><dt>Expand</dt><dd>opens a bundled case into several independently tracked cases.</dd></div>
    </dl>
  </section>

  <p className="freeform-introduction-closing">
    Describe the outcome you need in ordinary language. ELSPETH will propose
    the components and their wiring; review the graph and details before you
    run it.
  </p>
  {/* Keep the existing dismissal Button unchanged apart from adding the local action class. */}
</section>
```

- [ ] **Step 4: Style the longer primer for scanning**

Increase the bounded card width to `48rem`, align prose left, keep the top-level heading centered, and style each definition row as a compact term/description pair. At `max-width: 520px`, stack each term above its description. Use only existing spacing, type, border, surface, and text tokens.

```css
.freeform-introduction-definitions > div {
  display: grid;
  grid-template-columns: minmax(6.5rem, auto) 1fr;
  gap: var(--space-sm);
}

@media (max-width: 520px) {
  .freeform-introduction-definitions > div {
    grid-template-columns: 1fr;
    gap: var(--space-xs);
  }
}
```

- [ ] **Step 5: Run focused component and accessibility tests**

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run src/components/chat/FreeformIntroduction.test.tsx src/components/chat/ChatPanel.test.tsx src/test/a11y/components.a11y.test.tsx
```

Expected: all selected tests pass and axe reports no violations.

- [ ] **Step 6: Commit the primer**

```bash
git add src/elspeth/web/frontend/src/components/chat/FreeformIntroduction.tsx src/elspeth/web/frontend/src/components/chat/FreeformIntroduction.test.tsx src/elspeth/web/frontend/src/components/chat/chat.css
git commit -m "feat(web): expand freeform operator primer"
```

### Task 2: Verify and deploy the copy change

**Files:**
- Verify only; no planned source changes.

- [ ] **Step 1: Run the complete frontend gate**

```bash
cd src/elspeth/web/frontend
npm test
npm run lint
npm run lint:css
npm run build
```

Expected: all tests and linters pass and the Vite production build exits 0. Existing chunk-size warnings remain non-failing.

- [ ] **Step 2: Restart the dev service and wait for health**

```bash
sudo -n /usr/bin/systemctl restart elspeth-web.service
for attempt in $(seq 1 20); do
  curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health && break
  sleep 1
done
```

Expected: the unit is active and the health endpoint returns `{"status":"ok"}`.

- [ ] **Step 3: Verify desktop and narrow presentation with Playwright**

Sign in to the dev instance, open a fresh freeform session whose introduction has not been dismissed, and verify:

- the complete primer is visible without horizontal overflow at desktop and 390px widths;
- headings and definition terms remain scannable;
- the composer remains available below the primer;
- “Don’t show this again” still dismisses the primer and remains dismissed after reload; and
- the browser console has no new errors.
