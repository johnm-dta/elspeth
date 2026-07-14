# Two-LLM Colour Hybrid Demonstration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a live, auditable example in which ten colour rows split across two independently routed LLMs, merge into ten flat typed hybrid rows, and write clean JSON suitable for later statistical aggregation.

**Architecture:** Upload a ten-row CSV, ask the composer in outcome-oriented language for two parallel colour assessments, require both branches to coalesce through a union merge, and project away raw provider metadata before writing JSON. Validate the inferred topology and typed contracts before execution, then independently parse the downloaded output and reconcile run accounting.

**Tech Stack:** ELSPETH web composer at `https://elspeth.foundryside.dev/`, Playwright CLI, OpenRouter-backed LLM transforms, CSV input, fork/coalesce orchestration, typed LLM output fields, JSON sinks, and Python standard-library verification.

---

## Run-sheet boundaries

- This is a live demonstration, not a source-code change.
- Do not place staging credentials in this file, shell history, screenshots, traces, or committed artifacts. Use the operator-provided credentials only in the browser login form.
- The composer prompt may name two LLMs, JSON output, required fields, output fields, and unwanted fields. It must not prescribe composer tool calls or describe their argument payloads.
- Do not repair a rejected composition by importing hand-written YAML. Continue the conversation in ordinary outcome-oriented language so the test still measures the composer's orchestration ability.
- On any error, stop and use `superpowers:systematic-debugging`: capture the exact UI message, console entry, request/response, current state version, and run status before changing anything.

## Files and evidence

**Create during execution:**

- `output/playwright/two-llm-colour-palette.csv` — ten-row source fixture.
- `output/playwright/two-llm-colour-hybrid-graph.png` — validated topology screenshot.
- `output/playwright/two-llm-colour-hybrid-run.png` — completed run/accounting screenshot.
- `output/playwright/two-llm-colour-hybrid-output.json` — downloaded successful output, copied from Playwright's download directory if necessary.
- `.playwright-cli/traces/` — browser trace directory retained as local diagnostic evidence; do not commit it.

**Do not modify:**

- Application source under `src/`.
- Existing composer sessions or artifacts unrelated to this demonstration.
- The approved design at `docs/superpowers/specs/2026-07-13-two-llm-colour-hybrid-pipeline-design.md`.

---

### Task 1: Prepare the browser session and source fixture

**Files:**

- Create: `output/playwright/two-llm-colour-palette.csv`

- [ ] **Step 1: Confirm the Playwright prerequisite**

Run:

```bash
command -v npx
```

Expected: an absolute path to `npx`. The bundled wrapper currently lacks its executable bit in this environment, so invoke it through Bash without changing the shared skill installation:

```bash
export PWCLI="$HOME/.codex/skills/playwright/scripts/playwright_cli.sh"
bash "$PWCLI" --help
```

Expected: Playwright CLI help and exit code 0.

- [ ] **Step 2: Create the exact ten-row input fixture**

Create `output/playwright/two-llm-colour-palette.csv` with exactly:

```csv
color_name,hex
Pure Red,#FF0000
Pure Blue,#0000FF
Purple,#800080
Magenta,#FF00FF
Cyan,#00FFFF
Navy,#000080
Orange,#FF7F00
Teal,#008080
Grey,#808080
White,#FFFFFF
```

Verify:

```bash
uv run --frozen python -c "import csv, pathlib; p=pathlib.Path('output/playwright/two-llm-colour-palette.csv'); rows=list(csv.DictReader(p.open(newline='', encoding='utf-8'))); assert len(rows)==10; assert list(rows[0])==['color_name','hex']; assert len({r['hex'] for r in rows})==10; print('10 unique colour rows ready')"
```

Expected: `10 unique colour rows ready`.

- [ ] **Step 3: Open a new isolated browser session and start tracing**

Run:

```bash
bash "$PWCLI" --session two-llm-colours open https://elspeth.foundryside.dev/
bash "$PWCLI" --session two-llm-colours snapshot
```

Expected: the ELSPETH login page with Username, Password, and Sign in controls.

Fill the operator-provided credentials through refs from the fresh snapshot, sign in, and start tracing:

```bash
bash "$PWCLI" --session two-llm-colours tracing-start
```

Expected: a trace path under `.playwright-cli/traces/`.

- [ ] **Step 4: Create a new empty composer session**

Use the session switcher to create a fresh session rather than overwriting the earlier replication demonstration. Snapshot after navigation.

Expected:

- Composition history is `v1`.
- The side rail says `No pipeline yet`.
- `Run pipeline` and `Export YAML` are disabled.

- [ ] **Step 5: Upload the fixture**

Click `Upload file`, then run:

```bash
bash "$PWCLI" --session two-llm-colours upload /home/john/elspeth/output/playwright/two-llm-colour-palette.csv
```

Expected: the message box acknowledges `two-llm-colour-palette.csv` as the intended pipeline input.

---

### Task 2: Ask the composer for the split/merge pipeline

**Files:** None.

- [ ] **Step 1: Submit this exact outcome-oriented request**

Replace the message box contents with:

```text
I've uploaded two-llm-colour-palette.csv; use it as the input. Build a runnable pipeline that sends every colour row through two independent LLMs in parallel and then merges both completed assessments back into one flat hybrid row.

The blue-assessment LLM must require color_name and hex. It must estimate the perceptible amount of blue and expose three typed output fields: blue_amount as an integer from 0 to 100, blue_confidence as a number from 0 to 1, and blue_reason as one concise sentence. Define 0 as no perceptible blue, 50 as a substantial but balanced blue component, and 100 as overwhelmingly blue.

The red-assessment LLM must independently require color_name and hex. It must estimate the perceptible amount of red and expose red_amount as an integer from 0 to 100, red_confidence as a number from 0 to 1, and red_reason as one concise sentence. Define 0 as no perceptible red, 50 as a substantial but balanced red component, and 100 as overwhelmingly red.

Wait for both LLM branches for each original row, then produce one flat row containing exactly color_name, hex, blue_amount, blue_confidence, blue_reason, red_amount, red_confidence, and red_reason. Remove raw LLM responses, token usage, model metadata, branch bookkeeping, and all other intermediate fields from the successful output. Write the ten successful hybrid rows as one JSON array of objects and route failures to a separate JSON output. Use deterministic sampling where supported and an economical configured model. Make the pipeline runnable now.
```

This prompt intentionally describes the outcome and field contract but contains no composer tool names, call ordering, or tool argument payloads.

- [ ] **Step 2: Wait on composition state, not a fixed sleep**

After sending, wait until the `Stop composing` control is hidden, using a maximum 60-second condition wait per attempt:

```bash
bash "$PWCLI" --session two-llm-colours run-code "async (page) => { await page.getByRole('button', { name: 'Stop composing' }).waitFor({ state: 'hidden', timeout: 60000 }); }"
```

Expected: the composer finishes with either a proposed graph or an actionable review/validation state.

- [ ] **Step 3: Record the session evidence**

Snapshot and record:

- Session UUID from the URL hash.
- Composition version.
- Composer response text.
- Number and names of inferred components.
- Any pending interpretation reviews.

Do not correct anything yet if the topology is wrong; first capture the exact graph and validation evidence.

---

### Task 3: Inspect the inferred topology and contracts

**Files:**

- Create: `output/playwright/two-llm-colour-hybrid-graph.png`

- [ ] **Step 1: Open the graph and verify component topology**

The graph must contain, in source-to-sink flow:

1. One CSV source using the uploaded blob.
2. One two-way fork or equivalent independently routed split.
3. One blue LLM node.
4. One red LLM node.
5. One coalesce node that requires both branches.
6. A union-style flat merge.
7. One deterministic cleanup/projection step after the merge.
8. One successful JSON sink.
9. One failure JSON sink reachable from both LLM branches and any downstream cleanup failure.

Fail this checkpoint if the composer substitutes one LLM node with two internal queries: that does not demonstrate independent branch routing.

- [ ] **Step 2: Inspect the blue branch configuration**

Verify through the graph's node inspector:

- Required input fields are exactly `color_name` and `hex` or a conservative superset containing both.
- The prompt defines the blue rubric from 0 through 100.
- Typed extracted fields include `blue_amount` as integer, `blue_confidence` as number, and `blue_reason` as string.
- The branch preserves `color_name` and `hex` for the merge.

- [ ] **Step 3: Inspect the red branch configuration**

Verify through the graph's node inspector:

- Required input fields are exactly `color_name` and `hex` or a conservative superset containing both.
- The prompt defines the red rubric from 0 through 100.
- Typed extracted fields include `red_amount` as integer, `red_confidence` as number, and `red_reason` as string.
- The branch preserves `color_name` and `hex` for the merge.

- [ ] **Step 4: Inspect merge and cleanup semantics**

Verify:

- Coalesce policy is `require_all` or an equivalent explicit both-branches requirement.
- Merge shape is flat union, not nested and not branch selection.
- Shared `color_name` and `hex` collisions preserve identical source values with auditable lineage.
- Final projection retains exactly the eight approved business fields.
- Successful output is JSON array format; failures use a separate JSON sink.

- [ ] **Step 5: Capture the validated graph screenshot**

Once the topology matches, capture the graph dialog or graph region:

```bash
bash "$PWCLI" --session two-llm-colours screenshot ".graph-modal" --filename output/playwright/two-llm-colour-hybrid-graph.png
```

Expected: a screenshot showing the source, split, two LLM branches, coalesce, cleanup, and both sinks.

- [ ] **Step 6: Correct topology only through ordinary language if needed**

If the graph uses one multi-query LLM or omits the merge, send this correction:

```text
Please revise the topology. I need two separately routed LLM nodes operating on parallel copies of each source row, followed by a real wait-for-both merge into one flat row. A single LLM node with two internal queries does not demonstrate the split and merge lifecycle I need. Keep the exact eight-field final JSON contract and the separate failure output.
```

Then repeat Tasks 2 Step 2 and 3 in full. Do not name or prescribe composer tool calls.

---

### Task 4: Review decisions and validate the composition

**Files:** None.

- [ ] **Step 1: Review every interpretation card**

Inspect before acknowledging:

- Blue and red scale interpretations preserve the exact 0/50/100 anchors.
- Each prompt requests only `amount`, `confidence`, and `reason` JSON keys for its own branch.
- Model choices are economical and available through the configured provider.
- Prompt-injection recommendations correctly describe the uploaded CSV as user-controlled input.

Approve only accurate cards. If a card changes the field names, ranges, or branch independence, use its change path and restate the approved requirement.

- [ ] **Step 2: Run validation after all decisions resolve**

Expected validation checks include:

- `secret_refs`: pass.
- `settings_load`: pass.
- `plugin_instantiation`: pass.
- `graph_structure`: pass.
- `route_target_resolution`: pass.
- `schema_compatibility`: pass.
- `interpretation_review`: no pending reviews.

- [ ] **Step 3: Fail closed on schema drift**

Do not execute if validation reports any missing blue/red field at the coalesce or cleanup boundary. Capture the exact producer, consumer, guaranteed fields, required fields, and missing fields, then ask the composer to restore the approved eight-field hybrid contract in ordinary language.

Expected completion state:

- Audit status is `Audit ready`.
- `Run pipeline` is enabled.
- Graph topology remains unchanged from the reviewed version.

---

### Task 5: Execute the live pipeline

**Files:** None.

- [ ] **Step 1: Start execution and review side effects**

Click `Run pipeline`. Confirm the dialog discloses:

- One CSV source read.
- Two LLM nodes making provider calls.
- One successful JSON output and one failure JSON output.

Confirm the run. If the execute endpoint returns HTTP 428 and the UI opens an LLM fanout review, treat it as the expected precondition handshake, verify it names both LLM nodes, and click `Execute` once.

- [ ] **Step 2: Wait for a terminal state**

Use a condition wait of at most 60 seconds at a time against the visible `Pipeline running.` status. Do not use arbitrary sleeps.

Expected terminal state: `completed`.

If still running after 60 seconds, snapshot run accounting and inspect console/network evidence before waiting again. Do not launch a duplicate run.

- [ ] **Step 3: Reconcile run accounting**

The run must report:

- Source rows: `10`.
- Successful hybrid rows: `10`.
- Failed rows: `0` for a clean demonstration.
- Pending tokens: `0`.
- Audit closure: `closed`.
- Twenty LLM assessments or equivalent per-node call evidence: ten blue plus ten red.
- Ten completed coalesces or equivalent merge evidence.

Record the run UUID from run history.

---

### Task 6: Download and independently verify the hybrid JSON

**Files:**

- Create: `output/playwright/two-llm-colour-hybrid-output.json`
- Create: `output/playwright/two-llm-colour-hybrid-run.png`

- [ ] **Step 1: Download the successful output**

Open Run outputs and download the successful JSON artifact. Copy the downloaded file to:

```text
output/playwright/two-llm-colour-hybrid-output.json
```

Record its UI-reported SHA-256 before parsing.

- [ ] **Step 2: Parse and assert the exact business contract**

Run:

```bash
uv run --frozen python -c "import json, pathlib; p=pathlib.Path('output/playwright/two-llm-colour-hybrid-output.json'); rows=json.loads(p.read_text(encoding='utf-8')); expected={'color_name','hex','blue_amount','blue_confidence','blue_reason','red_amount','red_confidence','red_reason'}; assert isinstance(rows,list) and len(rows)==10, (type(rows),len(rows) if isinstance(rows,list) else None); assert all(set(r)==expected for r in rows); assert all(type(r['blue_amount']) is int and 0<=r['blue_amount']<=100 for r in rows); assert all(type(r['red_amount']) is int and 0<=r['red_amount']<=100 for r in rows); assert all(isinstance(r['blue_confidence'],(int,float)) and not isinstance(r['blue_confidence'],bool) and 0<=r['blue_confidence']<=1 for r in rows); assert all(isinstance(r['red_confidence'],(int,float)) and not isinstance(r['red_confidence'],bool) and 0<=r['red_confidence']<=1 for r in rows); assert all(isinstance(r['blue_reason'],str) and r['blue_reason'].strip() for r in rows); assert all(isinstance(r['red_reason'],str) and r['red_reason'].strip() for r in rows); print('10 aggregation-ready hybrid rows verified')"
```

Expected: `10 aggregation-ready hybrid rows verified`.

- [ ] **Step 3: Verify palette identity and output integrity**

Run:

```bash
uv run --frozen python -c "import csv,json,pathlib,hashlib; source=list(csv.DictReader(pathlib.Path('output/playwright/two-llm-colour-palette.csv').open(newline='',encoding='utf-8'))); p=pathlib.Path('output/playwright/two-llm-colour-hybrid-output.json'); rows=json.loads(p.read_text(encoding='utf-8')); assert {(r['color_name'],r['hex']) for r in rows}=={(r['color_name'],r['hex']) for r in source}; print({'rows':len(rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})"
```

Expected: ten rows and a digest matching the UI-reported SHA-256.

- [ ] **Step 4: Capture final visual evidence**

Capture the run-results region:

```bash
bash "$PWCLI" --session two-llm-colours screenshot "[aria-label='Pipeline run results']" --filename output/playwright/two-llm-colour-hybrid-run.png
```

Expected: terminal completed status, ten source rows, ten successes, zero failures, zero pending tokens, and closed audit accounting.

---

### Task 7: Debugging branches and closeout

**Files:** None unless a defect is promoted through Filigree.

- [ ] **Step 1: Classify every observed error before acting**

For each error, capture:

1. Exact visible message and component name.
2. Browser console entry.
3. Failing network request status and response body.
4. Current session UUID, state ID, composition version, and run UUID if allocated.
5. Whether the failure occurred during authoring, validation, provider execution, coalesce, cleanup, sink write, or preview.

State one root-cause hypothesis and test one variable at a time. Do not import replacement YAML or make source changes during this run sheet.

- [ ] **Step 2: Apply these failure-specific decisions**

- Composer chooses one LLM: correct the outcome in ordinary language and regenerate; do not accept the topology.
- Missing branch field: inspect typed output declarations and guarantee propagation before changing the sink.
- LLM JSON/type failure: preserve the typed contract; inspect the exact provider response/error and prompt schema rather than loosening validation.
- Coalesce remains pending: inspect both branch terminal outcomes and branch names; do not rerun the source.
- HTTP 428 before execution: verify whether it is the expected LLM fanout acknowledgment before classifying it as an error.
- Output file parses but preview is malformed: treat execution data and preview rendering as separate boundaries; promote a preview defect with artifact hash evidence.

- [ ] **Step 3: Track incidental defects**

Create a Filigree observation only for a defect outside this demonstration's core acceptance scope. Include session/run IDs, artifact hash, file path, line, and reproducible browser evidence. Promote clear product defects before session end; do not use observations to hide a failed core acceptance criterion.

- [ ] **Step 4: Stop tracing and close the browser**

Run:

```bash
bash "$PWCLI" --session two-llm-colours tracing-stop
bash "$PWCLI" --session two-llm-colours close
```

Expected: trace saved and browser session closed.

- [ ] **Step 5: Report the demonstration outcome**

The report must include:

- Whether the composer inferred two independent LLM branches without composer tool-call instructions.
- Exact topology and final field contract.
- Session UUID, composition version, and run UUID.
- Source/success/failure/pending counts and audit closure.
- Output artifact name and SHA-256.
- Blue/red value ranges observed in the ten rows.
- Links to the graph and run screenshots.
- Any debugged errors and any Filigree issue IDs created.
- A clear go/no-go statement for adding statistical aggregators next.
