# Design: `codex_panel_review.py` — pre-1.0 sandblasting SME review fleet

- **Date:** 2026-06-28
- **Status:** Revised design — supersedes the original "release-gating" framing of
  the same date (which oversold a multi-hour audit as a CI gate, and claimed a
  schema/writer reuse that did not hold). Ready for implementation planning.
- **Branch:** `release/0.7.0`
- **Author:** John Morrissey (with Claude)
- **Relationship to existing tooling:** A **new sibling** to the four Codex
  scripts that depend on the shared runner (`codex_test_defect_hunt.py`,
  `codex_integration_seam_hunt.py`, `codex_exemption_validator.py`,
  `codex_tier_model_rejudge.py`). Those are **left untouched**. This tool reuses
  helpers from `codex_audit_common.py` but adds its execution path **additively**
  — it does not modify the shared `run_codex_once` / `run_codex_with_retry_and_logging`
  the four scripts depend on, nor the markdown evidence gate they use. Complementary
  to, not overlapping with, the `bug-sweep` skill (that fans *Claude subagents*
  into Filigree; this drives the *codex CLI* into reports).

## Thesis

The existing Codex scanners each answer **one narrow question** ("is this test
weak?", "is this an integration-seam defect?", "is this exemption valid?"). Before
1.0, ELSPETH needs the opposite: a **pre-1.0 sandblasting pass** — an exhaustive,
broad scour where every in-scope source file faces a *panel* of subject-matter-
expert reviewers looking not only for bugs but for code smell, inefficiency,
improvement opportunities, and **easy wins we are leaving on the table**.

This is an **audit fleet, not a gate.** It does not block merges or PRs; it
produces a ranked body of findings a human (or follow-up agent) sweeps up. It
reviews each file through a routed panel of SME personas, one file at a time,
fully observable, resumable, and aggressively prompt-cached so the cost of running
the same large project-context + file-source prefix across many lenses is paid
once, not once per call.

## Positioning and scope

- **In scope:** a new `scripts/codex_panel_review.py` plus a
  `scripts/codex_lenses/` directory of editable persona prompt files; an additive
  streaming/serial execution path reusing the smaller helpers from
  `codex_audit_common.py` (retry, rate-limit, usage parsing, summary/metadata
  writers, the `has_file_line_evidence` evidence primitive, `exit_code_from_stats`).
- **Out of scope (v1):** Filigree auto-lodge (designed-for, deferred to a
  follow-up — see Outputs); any change to the four existing scripts; any change to
  the shared `run_codex_once` / `run_codex_with_retry_and_logging` signature or
  behaviour; any change to the markdown `apply_evidence_gate`.
- **Non-goals:** acting as a CI/release gate (it cannot — see below); replacing
  the narrow scanners; replacing the `bug-sweep` subagent flow.

### Not a gate

The original framing called this "release-gating." It is not, and the spec no
longer claims to be:

- The primary full-tree pass is an overnight-to-multi-day job (see Wall-clock
  reality). Nothing in a merge pipeline blocks on that.
- Reasoning-model output is non-deterministic; a re-run surfaces a *similar but
  not identical* finding set. That is fine for an audit and disqualifying for a
  gate.
- The **fail-closed exit code** (reused `exit_code_from_stats`) means *"this
  sandblast was partial — some `(file, lens)` failed after retries, so resume
  it,"* **not** *"fail the build."* It exists so a partial scan never presents as
  a complete one, not to block anything.

If a true findings-based CI gate is ever wanted, that is separate, explicit work
(fail on confirmed P0/P1 in changed files) and is not part of this tool.

## Scope selection — subsection focus is first-class

Three selectors, **composable** (intersection when combined), plus a default:

| Selector | Meaning | Typical use |
|---|---|---|
| *(none)* | the full source tree | the occasional big pre-1.0 full sandblast |
| `--path <dir\|glob>` (repeatable) | restrict to a subtree / globs | **the routine driver** — sandblast a subsystem we just reworked |
| `--since <base-ref>` | only files changed vs a git ref | post-merge or branch delta review |
| `--files a,b,c` | an explicit file list | targeted re-run, incl. resuming a failed subset |

- **Subsection focus is the common case, not an afterthought.** Example driving
  this tool's first real use: a major `web/` UX rewrite just landed, so the first
  run is `codex_panel_review.py --path src/elspeth/web` rather than a full-tree
  scour. Full-tree is the heavier, rarer mode.
- **Composition is intersection.** `--since main --path src/elspeth/web` = files
  changed vs `main` **and** under `src/elspeth/web`. `--files` is an explicit set;
  combining it with `--path`/`--since` intersects too.
- `--lenses a,b,c` is **orthogonal** to file selection — it overrides the routed
  lens set for whatever files were selected.

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
- **Pin to a commit.** A subsection or full sandblast can run for hours-to-days
  while `release/0.7.0` keeps moving. The run captures the git rev at start and
  records it in `RUN_METADATA.md`; reviewing file A at hour 1 and file B at hour
  30 against different working-tree states would make the report incoherent. (The
  read-only codex sandbox already runs against `--cd repo_root`; pinning is about
  recording and reasoning against one snapshot.)
- **Resumable, keyed per `(file, lens)`**, on a **completion sentinel, not file
  existence.** A call killed mid-write can leave a report file that *exists* but
  is unfinished/ungated. The "done" marker is the per-call **usage summary** that
  `run_codex_with_retry_and_logging` writes only after success; resume skips a
  `(file, lens)` only when that sentinel is present. Worker-agnostic, so resume
  works at any `--workers`.
- **Wall-clock reality:** `src/elspeth` is large (~hundreds of source files × ~5
  lenses ≈ ~1,500 codex calls; each a multi-minute reasoning run). Fully serial is
  ~75–125h; at `--workers 4` ≈ 15–30h. A full-tree pass is an overnight-to-multi-
  day chore by design — which is exactly why scope selection (run a subsection,
  not the tree), resume, and live progress are first-class. A subsection run
  (e.g. `src/elspeth/web`) is a small fraction of that.

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
  "what it's finding"). Structured output still rides the existing
  `--output-schema` + `--output-last-message` file mechanism (independent of the
  stdout stream), so the streaming runner reads events live *and* gets the
  structured sidecar. This is the principal additive engine change.

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

### Caching is **verified empirically**, not asserted — with a decision branch

The per-file caching win depends on **per-call duration vs. the cache TTL
(~5–10 min, refresh-on-hit)**: if one lens call runs longer than the TTL, the
`[context][source]` prefix ages out before the next lens reuses it. This cannot
be settled by reasoning. **Before any full run**, a pilot (one file, all its
lenses; then ~5 files) runs and the caching section's claims are written from the
observed `cached_input_tokens` / hit-rate and measured call durations. The pilot
also tells us the real `--workers` ceiling (first sustained 429s = cap) and
whether file-major ordering beats interleaving in practice.

**The missing branch, now decided:** if the pilot shows per-call durations
routinely exceed the TTL (plausible at `reasoning_effort=high` with the 1800s
per-call timeout), the per-file cache collapses and the run costs ~3–5× the input
tokens. Because this is a **one-shot** sandblast (not a per-PR cost), the call is:
record the no-cache cost estimate from the pilot, and if it is within the
one-time budget, **run anyway**; only abort if the no-cache projection exceeds
budget. The pilot's job is to produce that number, not to be a hard pass/fail.

### Cost (one-time)

Sandblasting is an occasional spend, not a per-PR one — but it must be sized
before launch. A full tree is ~1,500 lens calls + ~300 synthesis calls (~1,800
multi-minute reasoning calls); a subsection is a small fraction. The pilot writes
the per-call token/cost numbers (cached and uncached) so the operator approves a
concrete dollar projection before a full-tree launch. A subsection run is cheap
enough to launch without that ceremony.

## Finding schema

Structured JSON sidecar, a **true superset** of the existing `findings[]` shape:
it keeps `priority` (the field every shared writer actually reads) and *adds*
fields. It does **not** rename `priority` to `severity` — the original draft did,
which silently bucketed every finding as `unknown` because
`_priority_from_structured_finding` reads `priority`. There is no top-level
`severity` here; the markdown gate's word-`severity` stamp is irrelevant because
this tool uses its own structured gate (below).

```json
{
  "findings": [
    {
      "priority": "P1",                // P0 | P1 | P2 | P3  — REQUIRED; read by every writer
      "lens": "security-architect",    // NEW
      "category": "security",          // NEW: bug | correctness | smell | efficiency | improvement | easy-win | security | design
      "confidence": "high",            // existing — read by writers
      "effort": "small",               // NEW: trivial | small | medium | large
      "impact": "…why it matters…",    // NEW — also serves as the relaxed-gate rationale
      "summary": "…",                  // existing — read by writers
      "evidence": [{"path": "src/…", "line": 123, "claim": "…"}],  // existing shape
      "suggested_fix": "…",            // NEW
      "target_file": "src/elspeth/…"   // existing — read by write_findings_index
    }
  ]
}
```

`effort` + `impact` are the fields that make "easy wins we're leaving on the
table" actionable (sortable by impact-per-effort in the global rollup).

**Reuse check (run as a verification step):** writing a sidecar in this schema and
running `generate_summary` / the index writer over it must bucket a `P1` finding
as `P1` (the original draft's schema produced `unknown`). This is a few lines and
guards the keystone fix.

## Evidence gate — a new, structured-first, category-aware gate

The existing markdown gate (`apply_evidence_gate`) auto-downgrades any report
lacking a `file:line` citation. That is correct for claims that *must* be located
(`bug`, `correctness`, `security`, `smell`) — it suppresses hallucinated defects.
But it would **gut the opportunity categories**, which are legitimately sometimes
location-less ("there is no retry abstraction; add one"). It is also **markdown-
section-coupled** (it parses `## Evidence`/`## Severity` and mirrors to JSON by
index), so making it category-aware in place would invert its data dependency and
risk the four scripts that share it.

So this tool ships a **new** structured-first gate (`apply_panel_evidence_gate`)
that reads/writes the JSON sidecar directly and reuses only the
`has_file_line_evidence` primitive. The existing markdown `apply_evidence_gate`
is **untouched**.

- **Strict** for `{bug, correctness, security, smell}`: require at least one
  `evidence[]` entry with a real `path` + integer `line`; otherwise downgrade
  `priority` to `P3` and annotate "needs verification: no file:line".
- **Relaxed** for `{improvement, efficiency}`: require a non-empty `impact`
  rationale; do **not** downgrade for a missing line.
- **`easy-win`** is relaxed on severity logic but **still requires a code anchor**
  (`evidence[].line`): an easy win you cannot point at is not easy. This is the
  cheapest defense against generic filler in the category most prone to it.

Because each lens is its **own** codex call with its **own** sidecar, the gate
runs per-call in isolation — the by-index markdown↔JSON coupling the engine's old
path relies on is **never used** here. The cross-lens merge happens afterward in
our own code, on already-gated structured findings.

## Cross-lens merge + per-file synthesis pass

After a file's lenses all complete:

1. **Deterministic merge** (plain Python, no API call): group near-duplicate
   findings by `(normalized location, category)` and **attribute** them
   (`raised by: security-architect, python-engineer`), keeping distinct findings.
   This produces the file's **canonical structured sidecar** — and it is the
   **counting source** for all stats (token counts and triage buckets never depend
   on an LLM re-emitting valid JSON).
2. **Synthesis pass** (one extra codex call per file): an "editor" prompt inlines
   the **same already-warm file source** plus the merged lens findings and
   produces a **ranked editor's-summary in prose** — the human-facing primary
   read. Because it rides the warm `[context][source]` cache, its +1 call/file
   (~+20% call count) is mostly *cached* input tokens. The synthesis prose does
   **not** feed the counters (the deterministic merge does); raw per-lens findings
   are retained underneath as evidence/detail.

## Outputs

Output directory (e.g. `docs/quality-audit/findings-panel/`), laid out so the
**shared counters see exactly one finding-set per file** by exploiting the
existing `iter_report_files` exclusion of copy-dirs:

```
findings-panel/
  src/elspeth/foo.py.md                 ← synthesis PROSE (human-facing primary)
  src/elspeth/foo.py.md.structured.json ← deterministic-merge findings  ← counters read THIS
  _lenses/                              ← excluded from iter_report_files (like PRIORITY_COPY_DIR)
    src/elspeth/foo.py.security.md (+ .structured.json)   ← raw per-lens detail
    src/elspeth/foo.py.python.md   (+ .structured.json)
  SUMMARY.md, FINDINGS_INDEX.md, RUN_METADATA.md, run.log
```

- **Reused unchanged:** `generate_summary`, `write_summary_file`,
  `write_run_metadata`, `exit_code_from_stats` — these reuse cleanly *because*
  the schema keeps `priority` and the layout gives them one sidecar per file.
- **New writer:** `write_panel_findings_index` (in the new module) emits the
  `lens / category / priority / effort / impact / confidence` columns, **sortable
  by impact-per-effort**. The shared `write_findings_index` is **not** reused for
  this (it has a fixed column set and reads none of the new fields) and is **not**
  modified.
- **Global ranked rollup is mandatory (not deferred).** Because the output *is*
  the deliverable for a sandblast, `SUMMARY.md` ranks findings by impact-per-
  effort across the whole run — turning thousands of findings into "here are the
  N easy-wins, sorted." It also carries token + cache-hit stats.
- **Fail-closed exit code** (reused `exit_code_from_stats`): a partial run (any
  `(file, lens)` raised after retries) exits non-zero so a partial sandblast never
  presents as a complete one — a **resume** signal, not a build gate.

Filigree auto-lodge (a `--lodge` flag promoting confirmed findings under one
panel-review sweep tag, deduped) is **designed-for but deferred** to a follow-up.
The global ranked `SUMMARY.md` is the v1 funnel; `--lodge` is the v2 funnel.

## Reuse vs. new code (blast-radius ledger)

| Reused from `codex_audit_common.py` (unchanged) | New in this tool |
|---|---|
| `run_codex_once` / retry wrapper, retry + rate-limiter, usage parsing, `load_context`, `generate_summary`, `write_summary_file`, `write_run_metadata`, `exit_code_from_stats`, `has_file_line_evidence` primitive, `iter_report_files` (+ its copy-dir exclusion) | streaming JSONL runner; file-level worker pool + serial-file-major orchestration; commit pinning; completion-sentinel resume; layered/inlined prompt builder; lens roster + routing; scope selection (`--path` / `--since` / `--files`, composable); **structured category-aware gate** (`apply_panel_evidence_gate`); cross-lens deterministic merge; synthesis pass; **panel findings-index writer**; global impact/effort rollup; live dashboard; persona prompt files |

The shared `run_codex_once` / `run_codex_with_retry_and_logging` and the markdown
`apply_evidence_gate` are **not modified**; the streaming runner and the panel
gate are new functions, so the four existing scripts are byte-for-byte unaffected
(verified by an import/smoke check — see Verification plan).

## Verification plan

1. **Schema-roundtrip check** — write a sidecar in the finding schema and run
   `generate_summary` + the panel index writer; confirm a `P1` buckets as `P1`
   (not `unknown`). Guards the keystone fix.
2. **Existing-scripts-unaffected check** — import the four sibling scripts and
   run one on a tiny fixture; confirm byte-for-byte unchanged behaviour after the
   additive engine changes.
3. **Caching/throughput pilot** (above) — writes the empirical caching numbers,
   the no-cache cost projection, and sets the `--workers` ceiling, **before** any
   full run.
4. **Lens validation** — run each persona on a hand-picked file with a known
   issue; read output for signal/noise; tune the prompt. No prompt-text tests.
5. **Resume correctness** — interrupt a run; confirm restart skips only `(file,
   lens)` pairs with the completion sentinel and completes the remainder; confirm
   a mid-write kill is **re-run**, not skipped; confirm fail-closed exit on an
   injected failure.
6. **Scope selection** — confirm `--path`, `--since`, and their intersection
   select the expected file sets (drive the real first use: `--path src/elspeth/web`).
7. **End-to-end** on a small scoped subtree (one subsystem) before any full-tree
   launch.

## Open implementation questions (for the plan, not blocking design)

- Exact frontend-component predicate for the `ux-specialist` lens.
- Whether the streaming runner lands in `codex_audit_common.py` (additive) or a
  new `codex_stream.py` module — decided at plan time by what keeps the existing
  scripts provably untouched.
- Per-lens reasoning-effort defaults (e.g. security/architecture = high).
- Normalized-location key for the deterministic merge (how aggressively to
  collapse near-duplicate findings across lenses).
