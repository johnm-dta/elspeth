# Execute Audit — Chunk Runner Prompt

This file is a self-contained prompt template. Paste it into a fresh Claude Code session (or send it as the user message), substituting `<<CHUNK_ID>>` with one of the chunk IDs from `docs/audit/test-suite/CHUNKS.md` (e.g., `U-ENGINE-1`, `I-2`, `P-1`).

The prompt produces output in the **same structure** as the existing 3 completed waves (U-CONTRACTS-1, U-CORE-1, I-1) so synthesis stays collatable across the whole audit.

---

## How to use

1. Open a fresh Claude Code session at the repo root (`/home/john/elspeth`).
2. Paste the **prompt block** below as a single user message.
3. Replace `<<CHUNK_ID>>` with the target chunk ID.
4. Approve the agent dispatch and file writes. The session will:
   - Enumerate the chunk's file list from `docs/audit/test-suite/CHUNKS.md`
   - Dispatch 5 specialist agents in parallel
   - Synthesise the findings
   - Write `<chunk>-findings.md` and `<chunk>-raw-reports.md`
   - File new filigree issues for critical gaps; update existing issues if cross-reference revises their scope
   - Update `docs/audit/test-suite/CHUNKS.md` progress table and `docs/audit/test-suite/README.md` cross-chunk patterns

Expected wall-clock: ~5–15 minutes depending on chunk size and agent saturation. Token cost: substantial (5 agents × full chunk read = ~700K–1.5M tokens).

---

## The prompt

```
Execute the test-suite audit for chunk <<CHUNK_ID>>.

This is wave N of an ongoing systematic audit of the ELSPETH test suite. The audit's
goal is to identify pointless, duplicate, low-effort, defective, and underperforming
tests, alongside major coverage gaps. Findings are persisted to `docs/audit/test-suite/` and
critical production-code gaps are filed as P0/P1 filigree issues.

## Step 1 — Orient

Read in this order, and only these files (don't go exploring):

1. `docs/audit/test-suite/CHUNKS.md` — the chunk plan and progress state. Find chunk
   <<CHUNK_ID>> in the tables; note its file scope description.
2. `docs/audit/test-suite/README.md` — the cross-chunk patterns and methodology section.
   Note any patterns flagged as suite-wide so the new wave can confirm or extend them.
3. `docs/audit/test-suite/<most-recent-completed-chunk>-findings.md` — read the most recently
   completed chunk's synthesis to inherit its tone, severity calibration, and
   cross-reference framing. Use it as the structural template for your own synthesis.

If <<CHUNK_ID>> is already marked Done in CHUNKS.md, stop and ask the user whether
to re-run, skip, or do a different chunk.

## Step 2 — Enumerate the file list

From `CHUNKS.md`'s scope description, build the explicit file list with bash. Use
`find` and `ls` (NOT the Glob tool) so the file list is reproducible and visible
in the transcript:

  find tests/<path> -type f -name "test_*.py" -not -path "*/__pycache__/*" | sort

Confirm the count matches the CHUNKS.md sizing (within a few). If the count is
wildly off, sanity-check the scope and ask the user before proceeding.

## Step 3 — Dispatch 5 specialist agents in parallel

Send a single message with 5 Agent tool calls (parallel dispatch). All 5 agents
get the same file list verbatim. Each agent has a distinct charter — do not
collapse charters or use only one agent.

The 5 lenses, in this order:

1. `ordis-quality-engineering:test-suite-reviewer` — anti-patterns
2. `axiom-sdlc-engineering:quality-assurance-analyst` — VER/VAL theatre
3. `axiom-python-engineering:python-code-reviewer` — Python-specific smells
4. `pr-review-toolkit:pr-test-analyzer` — scenario coverage
5. `ordis-quality-engineering:coverage-gap-analyst` — SUT coverage gaps

For each agent, the prompt MUST contain:

- **Project context block** (CLAUDE.md-derived rules relevant to the lens). For example:
  - Tier-1/2/3 trust model: Tier-1 audit data crashes on anomaly, no coercion
  - Defensive programming forbidden in production; `hasattr` unconditionally banned
  - Plugins are system code, not user code — bugs crash, not coerce
  - Audit primacy: Landscape is the legal record; mocked recorders without
    `assert_called_with` mask audit silence
  - Integration tests MUST use `ExecutionGraph.from_plugin_instances()` and
    `instantiate_plugins_from_config()`
  - Hashes survive payload deletion — recording tests without hash↔payload binding
    miss the project's core integrity guarantee
- **The full file list** in a fenced code block — DO NOT shorten with "..." — agents
  must see every file path explicitly
- **The lens charter** — what specific defect classes to look for (see prior-wave
  prompts in transcript history; structure is identical across waves)
- **Out-of-scope clauses** — production bugs are noted briefly only; sibling lenses'
  territories are explicitly off-limits
- **Cross-reference task** (gap-analyst and pr-test-analyzer ONLY): a list of
  prior-wave critical gaps to verify against this chunk. Read the most recent
  findings file's "Filed filigree issues" section to extract the gap list.
- **Output contract** — structured Markdown with exact section headers; severity
  scale (Critical/Major/Minor or Critical/High/Medium/Low); file:line citations
  required for every finding; word limit 1200 (1500 for pr-test-analyzer with
  cross-reference)

If you've not done this before, find the prior wave's dispatch in transcript history
and clone the structure verbatim. The prompts are deliberately uniform across waves
so synthesis stays comparable.

## Step 4 — Synthesise

Once all 5 agents return, write `docs/audit/test-suite/<chunk>-findings.md` with these sections
(in this order — the structure is canonical across the audit):

1. **Header**: Scope, Method, Date.
2. **Verdict** (1 paragraph): overall health (Healthy/Mixed/Concerning/Poor) and
   per-lens one-line summaries.
3. **Convergent findings** (≥2 agents agree): each as a labelled CONV-N entry with
   sites, severity, recommendation. These are highest-confidence — lead with them.
4. **Single-lens findings worth surfacing**: SOLO-N entries for high-severity findings
   only one agent caught. Note which agent.
5. **Cross-reference verdicts** (only if cross-reference task was run): table of
   prior-wave gaps × this layer, with status (open/closed/scope-revised) and evidence.
6. **New gaps surfaced**: critical/high/medium gaps unique to this chunk.
7. **Top deletion candidates**: ranked table with line counts and confidence.
8. **Top "add immediately" candidates**: prioritised list of tests to write.
9. **Notable strengths**: 4–6 bullets on what's genuinely good. Critical to prevent
   future "add more tests" suggestions for areas already strong.
10. **Filed filigree issues**: table populated after Step 5.
11. **Out-of-scope observations**: production-code concerns noted but not analysed.

Then write `docs/audit/test-suite/<chunk>-raw-reports.md` with the verbatim outputs of all
5 agents, in the same numbered order, including their agent IDs (for traceability —
agent IDs are NOT durable across sessions, but recording them documents which run
produced which finding).

## Step 5 — Filigree issues

For each Critical-severity SUT coverage gap (from the gap-analyst's findings), file
a filigree issue. Required fields:

- title: terse, declarative, includes the symbol or module name
- type: bug
- priority: 0 (or 1 if it's a structural epic affecting many files)
- labels: ["P0" or "P1", "from-test-audit", "audit-integrity" or "security" or
  "test-quality" as applicable, "test-gap"]
  NOTE: do NOT include "bug" in labels — it's reserved as a type name; will error.
- description: includes Source pointer (chunk + findings file paths + raw reports
  + which agent discovered it), the gap statement with file:line citations, why
  critical (CLAUDE.md-grounded reasoning), required tests (concrete list), confidence.

Cross-reference revisions: if this wave's cross-reference task downgrades a prior
issue (e.g., shows it's covered at another layer), add a comment to the existing
issue using `mcp__filigree__add_comment` (parameter is `text`, not `body`). State
what was found and recommend a severity adjustment. Do not auto-close — let the
human decide.

Do NOT file issues for test-quality findings (deletable tautologies, fragile mocks,
etc.). Those are handled as a single sweep PR per chunk, not per-test issues.
Per-test issues create tracker pollution and obscure the gap-vs-quality distinction.

## Step 6 — Update progress

Edit `docs/audit/test-suite/CHUNKS.md`:
- Mark the chunk's status from ⏳ Pending to ✅ Done
- Add the synthesis links and filigree issue IDs to the progress table

Edit `docs/audit/test-suite/README.md`:
- Add the chunk's row to the progress table
- If new cross-chunk patterns emerged (i.e., a finding type appearing in 2+ chunks),
  add or extend the relevant bullet in the cross-chunk patterns section
- If a prior issue was scope-revised, add to the "cross-wave verification" section

## Step 7 — Report to the user

Reply with a concise summary:
- Chunk audited, file count, wall-clock and rough token cost
- Top-line verdict (Mixed/Concerning/etc.)
- Critical gaps filed (issue IDs and titles)
- Top 3 deletion candidates by line count
- Cross-reference results (if applicable)
- Recommended next chunk per `CHUNKS.md` dispatch order, with one-line rationale

## Style rules

- Output style: explanatory. Use the `★ Insight ─────────────────────────────────────`
  block for non-obvious points (codebase-specific findings, calibration insights).
  Don't use it for filler.
- Cite file:line for every finding.
- Be willing to call tests "theatre" or "tautology" when they are. The user explicitly
  wants this; performative politeness defeats the audit's purpose.
- Don't propose new tests outside the chunk's SUT footprint.
- Don't auto-close prior issues. Always let the user adjudicate scope revisions.
- Don't fabricate findings to fill quota. If a chunk is genuinely healthy, say so —
  prior waves found chunks ranging from "Concerning" (I-1) to "Mixed with bright spots"
  (U-CORE-1) to "Mixed" (U-CONTRACTS-1). A future "Healthy" verdict is plausible
  and should be reported as such.

## Anti-patterns to avoid

- **Don't run a single agent.** The 5-lens parallel structure is load-bearing —
  prior calibration showed 4 of 9 critical gaps were found by exactly one agent that
  others missed.
- **Don't skip the cross-reference task.** It validates prior findings at this layer
  and surfaces issues that should be downgraded. The 1 scope revision in I-1 saved
  the project from over-prioritising a P0.
- **Don't synthesise before all 5 agents return.** Convergence (≥2 agents agreeing)
  is the highest-confidence signal; you can't compute it from partial returns.
- **Don't use the Glob tool to enumerate files** — use `find` so the count and full
  path list show up in the transcript and can be passed verbatim to agents.
- **Don't read individual test files yourself in Step 2.** That's the agents' job.
  Your role is orchestration, dispatch, and synthesis — not first-pass review.
- **Don't widen scope.** If you find yourself wanting to add files from outside the
  chunk's CHUNKS.md scope, stop and update CHUNKS.md instead.

## If something goes wrong

- **An agent times out or returns empty:** rerun just that one agent (parallel
  dispatch is per-message; you can send a follow-up message with a single Agent call).
- **Two waves disagree on a finding's severity:** trust the more recent wave (it has
  more cross-wave context). Note the discrepancy in the synthesis.
- **A chunk turns out to be much bigger than CHUNKS.md says:** propose a split to
  the user; don't silently audit a 60-file chunk as if it were 25.
- **Filigree label rejection:** "bug" is reserved as a type name and can't be used
  as a label. If you get a validation error, drop "bug" from the labels array; the
  type field is sufficient.

## Done.

Ready to execute. Begin with Step 1.
```

---

## Notes for prompt maintainers

- This template was extracted from the dispatch patterns used in waves 1–3
  (U-CONTRACTS-1, U-CORE-1, I-1) on 2026-05-06.
- The "cross-reference task" was added in wave 3 and proved high-leverage; it's
  baked into Step 3's gap-analyst and pr-test-analyzer charter requirements.
- The "don't auto-close prior issues" rule is from wave 3: the ADR-019 sweep methods
  issue (`elspeth-f6f50e9394`) was scope-revised, not closed, because the unit-layer
  gap is still genuine — only the integration-layer concern was downgraded.
- The "don't use Glob" rule is because Glob doesn't pin a specific file count
  visible in transcript history. `find ... | sort` shows the exact file list and
  count in the transcript, which the agents need to receive verbatim.
- If the lens set ever needs to be compressed: gap-analyst + qa-analyst +
  python-code-reviewer covers ~85% of what 5 agents produce. But the calibration
  showed dropping to <5 agents loses real signal — keep the full 5 unless cost
  pressure forces compression.
