# Design: `codex_panel_review.py` — release-gating SME review fleet

- **Date:** 2026-06-28
- **Status:** Approved (design); ready for implementation planning
- **Branch:** `release/0.7.0`
- **Author:** John Morrissey (with Claude)
- **Relationship to existing tooling:** A **new sibling** to the three Codex hunt
  scripts (`codex_test_defect_hunt.py`, `codex_integration_seam_hunt.py`,
  `codex_exemption_validator.py`). Those are **left untouched**. This tool reuses
  helpers from `codex_audit_common.py` but adds its execution path **additively**
  — it does not modify the shared `run_codex_once` the three scripts depend on.
  Complementary to, not overlapping with, the `bug-sweep` skill (that fans
  *Claude subagents* into Filigree; this drives the *codex CLI* into reports).

## Thesis

The existing Codex scanners each answer **one narrow question** ("is this test
weak?", "is this an integration-seam defect?", "is this exemption valid?"). As
ELSPETH moves into release territory, reviews need to become **broad and
oppressive**: every source file should face a *panel* of subject-matter-expert
reviewers looking not only for bugs but for code smell, inefficiency,
improvement opportunities, and **easy wins we are leaving on the table**.

This tool reviews each file through a routed panel of SME personas, one file at a
time, fully observable, and aggressively prompt-cached so the cost of running the
same large project-context + file-source prefix across many lenses is paid once,
not once per call.

## Positioning and scope

- **In scope:** a new `scripts/codex_panel_review.py` plus a
  `scripts/codex_lenses/` directory of editable persona prompt files; an additive
  streaming/serial execution path reusing the smaller helpers from
  `codex_audit_common.py` (retry, rate-limit, usage parsing, report writers,
  evidence-gate primitives).
- **Out of scope (v1):** Filigree auto-lodge (designed-for, deferred to a
  follow-up); any change to the three existing hunt scripts; any change to the
  shared `run_codex_once` signature/behaviour.
- **Non-goals:** replacing the narrow scanners; replacing the `bug-sweep`
  subagent flow.

## Lens roster and routing

Each lens is a **persona prompt file** (`scripts/codex_lenses/<lens>.md`),
version-controlled and editable. Personas are validated by **running them** on a
sample and reading the output — not by tests asserting prompt text (project
doctrine: skill/LLM prompts are not code).

**Smart routing** — every *applicable* lens runs; inapplicable ones are skipped:

| Lens | Applies to |
|---|---|
| `solution-architect` | every file |
| `systems-thinker` | every file |
| `quality-engineer` | every file |
| `security-architect` | every file |
| `python-engineer` | `*.py` |
| `typescript-engineer` | `*.ts`, `*.tsx` |
| `ux-specialist` | frontend component files (`web/frontend/src/**` `*.tsx`/`*.vue`/component dirs) |
| *(extensible domain slot, e.g. `audit-tier-model`)* | configured globs |

A `--lenses a,b,c` flag overrides the routed set for a run. Routing is table-
driven (a list of `(lens, predicate)`), so adding a lens is one entry plus one
prompt file.

## Execution model — file-level worker pool, serial within a file

- **Worker pool over files.** `--workers N` (default **4**, tune to the
  codex/OpenAI account rate limit). Each worker owns **one file at a time** and
  runs that file's lenses **serially, file-major**, then pulls the next file from
  a shared queue.
- **Parallel by default** because the work is I/O-bound (waiting on the API), so
  speedup is near-linear up to the account's rate/concurrency ceiling — `--workers
  2` ≈ half wall-clock, `--workers 4` ≈ quarter. `--workers 1` restores a single
  pristine one-file-at-a-time narration stream.
- **Full-tree audit is the primary mode** (the headline use-case). A `--since
  <base-ref>` git-diff mode is a first-class secondary entry point for branch/PR
  gating (review only changed files).
- **Resumable, keyed per `(file, lens)`** via the per-lens report files already
  on disk: a restart skips any `(file, lens)` whose report exists, never redoing a
  finished lens. Worker-agnostic, so resume works at any `--workers`.
- **Wall-clock reality:** `src/elspeth` is large (~hundreds of source files × ~5
  lenses ≈ ~1,500 codex calls; each a multi-minute reasoning run). Fully serial is
  ~75–125h; at `--workers 4` ≈ 15–30h. This is an overnight-to-multi-day chore by
  design, which is why scoping, resume, and live progress are first-class.

## Live observability

- **Per-worker dashboard:** one live status line per worker — `current file ·
  lens · findings-so-far` — plus a **global counter**: `[file 12/300 · 47
  findings · elapsed 6h · ETA 22h]`.
- **Per-file logs:** each file's full crawl/finding narration (the *other* files
  each lens reads for investigation, surfaced from the codex event stream;
  findings as they land) is written to a per-file log the operator can `tail`.
- **Streaming, not buffered.** Today's engine buffers all of codex stdout with
  `process.communicate()` and parses usage only at the end. The new path reads the
  `codex exec --json` JSONL event stream **line-by-line as it arrives** and
  surfaces selected events (tool/Read calls = "what it's crawling"; findings =
  "what it's finding"). This is the principal additive engine change.

## Prompt architecture and caching

Codex (OpenAI) prompt caching keys on the **longest identical input prefix** and
discounts cached input tokens heavily; the engine already parses
`cached_input_tokens`. The rule is **stable content first, variable content
last** — the existing scripts violate it by putting `Target file: {path}` in the
first ~30 tokens, making the cacheable prefix ≈ zero.

The new prompt is layered **most-shared → least-shared**:

```
[ project context: skills + CLAUDE.md/AGENTS.md ]   ← shared by EVERY call (all files, all lenses)
[ target file's full inlined source              ]   ← shared by all lenses of THIS file
─────────────────────────────────────────────────
[ lens persona + checklist + output schema       ]   ← the only uncached bytes per call (small)
[ trivial per-call tail (file path, lens id)     ]
```

- **Inline the focus file's source** into the prefix. Each `codex exec` is a
  fresh conversation, so a file the model Reads via the sandbox does **not**
  persist across calls and is **not** cached — only the *input prompt prefix* is.
  Inlining is what makes "cache the code" possible. The model can still Read **any
  other** repo file via the read-only sandbox for investigation.
- **File-major ordering** means a file's first lens warms `[context][source]` and
  the remaining lenses for that file **hit** it — caching the whole file body plus
  all project context, leaving only a small persona suffix uncached. The trade
  (cross-*file* persona caching lost) is favourable because persona text ≪
  context+source.
- **Parallelism is orthogonal to caching.** The per-file `[context][source]`
  cache lives *inside* one worker (still serial-file-major), so it is fully
  preserved. Parallelism only touches the outer `[context]` layer: an N-way herd
  of misses at startup, then steady-state warmth (refresh-on-hit keeps it alive
  while any worker keeps firing inside the TTL).

### Caching is **verified empirically**, not asserted

The per-file caching win depends on **per-call duration vs. the cache TTL
(~5–10 min, refresh-on-hit)**: if one lens call runs longer than the TTL, the
`[context][source]` prefix ages out before the next lens reuses it. This cannot
be settled by reasoning. **Before any full run**, a pilot (one file, all its
lenses; then ~5 files) runs and the caching section's claims are written from the
observed `cached_input_tokens` / hit-rate and measured call durations. The pilot
also tells us the real `--workers` ceiling (first sustained 429s = cap) and
whether file-major ordering beats interleaving in practice.

## Finding schema

Structured JSON sidecar, a **superset** of the existing `findings[]` shape so the
existing report writers (`generate_summary`, `write_findings_index`,
`write_summary_file`) reuse cleanly:

```json
{
  "findings": [
    {
      "lens": "security-architect",
      "category": "security",          // bug | correctness | smell | efficiency | improvement | easy-win | security | design
      "severity": "P1",                // P0 | P1 | P2 | P3
      "confidence": "high",            // SME confidence
      "effort": "small",               // trivial | small | medium | large
      "impact": "…why it matters…",
      "summary": "…",
      "evidence": [{"path": "src/…", "line": 123, "claim": "…"}],
      "suggested_fix": "…",
      "target_file": "src/elspeth/…"
    }
  ]
}
```

`effort` + `impact` are the fields that make "easy wins we're leaving on the
table" actionable (sortable by impact-per-effort).

## Evidence gate — made category-aware

The existing gate auto-downgrades any finding lacking a `file:line` citation.
That is correct for claims that *must* be located (`bug`, `correctness`,
`security`, `smell`) — it suppresses hallucinated defects. But it would **gut the
opportunity categories**, which are legitimately sometimes location-less ("there
is no retry abstraction; add one").

- **Strict** file:line bar for `{bug, correctness, security, smell}` (existing
  behaviour).
- **Relaxed** for `{improvement, easy-win, efficiency}`: require a non-empty
  *rationale* but do **not** downgrade for a missing line number.

Because each lens is its **own** codex call, the existing per-call by-index
evidence-gate alignment (`_apply_structured_evidence_gate`) stays intact within a
call. The cross-lens **merge** happens afterward in our own code, so the
by-index coupling the engine relies on is never spanned across lenses.

## Cross-lens merge + per-file synthesis pass

After a file's lenses all complete:

1. **Deterministic merge** (plain Python, no API call): group near-duplicate
   findings by `(normalized location, category)` and **attribute** them
   (`raised by: security-architect, python-engineer`), keeping distinct findings.
2. **Synthesis pass** (one extra codex call per file): an "editor" prompt inlines
   the **same already-warm file source** plus the merged lens findings and
   produces a **ranked editor's-summary** — the primary per-file artifact. Because
   it rides the warm `[context][source]` cache, its +1 call/file (~+20% call
   count) is mostly *cached* input tokens. Raw per-lens findings are retained
   underneath as evidence/detail.

## Outputs

Reuses the existing report family, written under a new output dir (e.g.
`docs/quality-audit/findings-panel/`):

- per-file ranked synthesis report (`<source-path>.md`) + structured sidecar;
- per-lens raw finding detail retained underneath;
- `SUMMARY.md` (triage dashboard incl. token + cache-hit stats),
  `FINDINGS_INDEX.md` (sortable, now with lens/category/effort/impact columns),
  `RUN_METADATA.md`, and an execution log;
- per-file live logs (see Observability).

**Fail-closed exit code** (existing `exit_code_from_stats`): a partial run (any
`(file, lens)` raised after retries) exits non-zero so a partial scan never
presents as a clean complete run.

Filigree auto-lodge (a `--lodge` flag promoting confirmed findings under one
panel-review sweep tag, deduped) is **designed-for but deferred** to a follow-up.

## Reuse vs. new code (blast-radius ledger)

| Reused from `codex_audit_common.py` (unchanged) | New in this tool |
|---|---|
| retry + rate-limiter, usage parsing, evidence-gate primitives, `load_context`, report writers, `exit_code_from_stats` | streaming JSONL runner; file-level worker pool + serial-file-major orchestration; layered/inlined prompt builder; lens roster + routing; category-aware gate wrapper; cross-lens merge; synthesis pass; live dashboard; `--since` diff selection; persona prompt files |

The shared `run_codex_once` is **not modified**; the streaming runner is a new
function (in the new module, or added additively to common without altering the
existing one), so the three existing scripts are byte-for-byte unaffected.

## Verification plan

1. **Caching/throughput pilot** (above) — writes the empirical caching numbers
   and sets `--workers` ceiling, **before** any full run.
2. **Lens validation** — run each persona on a hand-picked file with a known
   issue; read output for signal/noise; tune the prompt. No prompt-text tests.
3. **Resume correctness** — interrupt a run; confirm restart skips completed
   `(file, lens)` and completes the remainder; confirm fail-closed exit on an
   injected failure.
4. **End-to-end** on a small scoped subtree (e.g. one subsystem) before a
   full-tree launch.

## Open implementation questions (for the plan, not blocking design)

- Exact frontend-component predicate for the `ux-specialist` lens.
- Whether the streaming runner lands in `codex_audit_common.py` (additive) or a
  new `codex_stream.py` module — decided at plan time by what keeps the existing
  scripts provably untouched.
- Per-lens reasoning-effort defaults (e.g. security/architecture = high).
