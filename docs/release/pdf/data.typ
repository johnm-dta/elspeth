// data.typ — single source of truth for load-bearing figures.
//
// Every number that appears in more than one document, or that risks
// drift between source `.md` and typeset `.typ`, is bound here. Both
// the progress and the velocity documents import this file. The
// executive summary imports a small subset for the at-a-glance
// dashboard.
//
// Numbers are taken from:
//   - docs/release/elspeth-progress-rc1-to-rc5.md (cumulative-commit
//     table, capability list, accretion table)
//   - docs/release/elspeth-velocity-rc1-to-rc5.md (full daily table,
//     phase totals, weekly totals, peak-day attributions)
//
// If you update a number here, also update the source `.md` — and
// vice-versa. The contract is one-to-one.

// ---------------------------------------------------------------------------
// Headline statistics
// ---------------------------------------------------------------------------

#let project-start = "12 January 2026"
#let project-end = "19 May 2026"

// The date the documents bear, used in two roles:
//   1. `doc-date:` argument to `cover-page` — the date in the
//      bibliographic "Document date" row.
//   2. `draft-date:` argument to `document-frame` and `cover-page`
//      — the date stamped inside the DRAFT — AWAITING REVIEW banner
//      of the executive summary.
// These are the same value: the documents are dated to today and
// the draft is stamped today. Update this one place when the
// snapshot rolls to a new date. Companion alias `draft-date` below
// preserves the name used in code paths that emit the DRAFT banner.
#let doc-date = "19 May 2026"
#let draft-date = doc-date

#let calendar-days = 128
#let active-days = 123
#let idle-days = 5
#let total-commits = 4521
#let mean-active-day = 36.8
#let mean-calendar-day = 35.3
#let median-active-day = 27
#let max-daily-commits = 177
#let max-daily-date = "20 January 2026"
#let days-ge-100 = 6
#let days-ge-50 = 31
// Rounding rule: the raw test count is reported to the nearest 100,
// because component sub-counts shift continuously and a rounded
// figure is more honest about precision. `test-count-card` is the
// short visual form used in the dashboard tiles; `test-count-prose`
// is the long form used inside body sentences. Both must be updated
// in lockstep if the raw number crosses a 100-row threshold.
#let test-count = 14106          // 10500 framework + 3159 backend + 447 frontend
#let test-count-card = "~14,100"
#let test-count-prose = "approximately 14,100"

// Pilot-evaluation row count (also reported to the nearest 100).
// Cited in the executive summary's prose; SoT so the figure cannot
// drift between two paragraphs.
#let pilot-rows-prose = "approximately 2,200"

// ---------------------------------------------------------------------------
// Cumulative commits at each release milestone.
// (release, date, theme, cumulative-commits)
// ---------------------------------------------------------------------------

#let release-milestones = (
  ("Project start", "2026-01-12", "Empty scaffold", 0),
  ("RC-1", "2026-01-22", "Auditable SDA framework", 782),
  ("RC-2 (0.1.0)", "2026-02-02", "Telemetry; ChaosLLM; bug burndown", 1593),
  ("RC-2.5", "2026-02-12", "Key Vault, schema contracts, PipelineRow, SQLCipher, declarative DAG", 2120),
  ("RC-3.2 (0.3.0)", "2026-02-22", "Strict typing at audit boundaries", 2294),
  ("RC-3.3 (0.3.3)", "2026-03-02", "Architectural remediation (T10/T17-T19)", 2441),
  ("RC-3.4 (0.3.4)", "2026-03-10", "Systematic hardening; 191-bug triage", 2595),
  ("RC-4.0 (0.4.0)", "2026-03-29", "Dataverse + RAG + output schema contracts", 2921),
  ("RC-4.1 (0.4.1)", "2026-04-02", "ChromaSink, depends_on, commencement gates", 3013),
  ("RC-5 (0.5.0)", "2026-04-03", "Web UX Composer + auth + blob + secret refs + MCP", 3073),
  ("RC-5.1 (0.5.1)", "2026-05-11", "Composer correctness + audit-integrity coverage", 3883),
  ("RC-5.2 (0.5.2)", "2026-05-14", "Guided composer + durable progress + recovery UX", 4210),
  ("RC-5.2 snapshot", "2026-05-19", "Phase 6/7/8 — completion gestures, catalog reshape, polish", 4521),
)

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------
//
// Typst's `str()` of a float strips trailing zeros: `str(19.0)` → `"19"`,
// `str(71.1)` → `"71.1"`. For rate columns where the reader expects a
// consistent one-decimal display, that mixed presentation is jarring.
// `fmt-1dec()` pads integer-valued floats with `.0` so every row in the
// rate column has identical visual weight.
#let fmt-1dec(x) = {
  let s = str(x)
  if s.contains(".") { s } else { s + ".0" }
}

// ---------------------------------------------------------------------------
// Per-phase totals.
// (phase, dates, calendar-days, active-days, total-commits,
//  commits-per-active-day, headline)
// ---------------------------------------------------------------------------

#let phase-totals = (
  ("Pre-RC1 Foundation",
   "2026-01-12 -- 2026-01-22", 11, 11, 782, 71.1,
   "Empty scaffold to working SDA framework"),
  ("RC-1 Hardening",
   "2026-01-23 -- 2026-02-02", 11, 11, 811, 73.7,
   "Telemetry; ChaosLLM; 100+ bugs closed"),
  ("RC-2 Sub-releases",
   "2026-02-03 -- 2026-02-12", 10, 10, 527, 52.7,
   "Key Vault, schema contracts, PipelineRow, WebScrape, SQLCipher, declarative DAG"),
  ("RC-3 Series",
   "2026-02-13 -- 2026-03-10", 26, 25, 475, 19.0,
   "Strict typing at audit boundaries; T10/T17/T18/T19; 191-bug triage"),
  ("RC-4 + RC-5 cut",
   "2026-03-11 -- 2026-04-03", 24, 21, 478, 22.8,
   "Dataverse, RAG, ChromaSink, depends_on, commencement gates, RC-5 web composer cut"),
  ("RC-5.1 Composer Correctness",
   "2026-04-04 -- 2026-05-11", 38, 37, 810, 21.9,
   "Substrate framing; validator hardening; advisor escalation; 10 statistical batch plugins"),
  ("RC-5.2 Composer Maturation",
   "2026-05-12 -- 2026-05-19",  8,  8, 638, 79.8,
   "Guided mode; 4 phases of composer progress persistence; frontend recovery UX"),
)

// ---------------------------------------------------------------------------
// Top 15 peak days.
// (rank, date, commits, attribution)
// ---------------------------------------------------------------------------

#let peak-days = (
  ( 1, "2026-01-20", 177, "Pre-RC1 LLM pooled-execution + batch aggregation. Heaviest single day in project history."),
  ( 2, "2026-01-30", 156, "RC-1 hardening burst: ChaosLLM weighted error selection; AuditIntegrityError introduction; Azure Monitor + Datadog exporters."),
  ( 3, "2026-05-12", 142, "RC-5.2 composer redaction-MANIFEST pass: guided step_2_chosen_plugin; RecipeOfferTurn editable slots; ARG_ERROR canonicalization."),
  ( 4, "2026-01-18", 125, "Pre-RC1 LLM infrastructure: client base classes; pooled-execution scaffolding; reorder buffer."),
  ( 5, "2026-02-03", 125, "RC-2.1 to RC-2.2: Langfuse SDK v3 migration; secret-resolution audit trail; schema-contract propagation."),
  ( 6, "2026-02-02", 118, "RC-2 cutover day. Release Candidate 2 commit + post-cutover cleanup."),
  ( 7, "2026-05-14",  97, "RC-5.2 release-stamp day: changelog finalize; per-step chat merge; phase3 compose-loop persistence merge."),
  ( 8, "2026-05-19",  94, "Today -- Phase 8 polish + Phase 6 completion-gestures merge + CI allowlist burn-down + Phase-5 chat-data-entry."),
  ( 9, "2026-05-18",  94, "Phase 7 catalog reshape merge + fix/catalog-i1-i2-i3 + plugin-coverage gate calibration."),
  (10, "2026-05-09",  93, "RC-5.1 composer-progress-persistence Phase 1B: persist_compose_turn happy path; OperationalError + audit-failure primacy."),
  (11, "2026-05-13",  88, "Phase A coverage gap + per-step chat to RC-5.2 merge + cross-step chat_history accumulation test."),
  (12, "2026-01-21",  87, "Final pre-RC1 hardening day: line-length-140 reformat + bug-burndown sweeps before RC-1 cutover."),
  (13, "2026-01-29",  85, "RC-1 hardening: telemetry property tests; contract tests for keyword filter / content safety / prompt shield / multi-query."),
  (14, "2026-05-17",  84, "Phase 2C composer implementation: ReadinessRowDetail + ExplainDialog real impls; InspectorPanel mount; Validate button deletion."),
  (15, "2026-01-28",  83, "RC-1 hardening: contract boundary tightening; type-soup cleanup."),
)

// ---------------------------------------------------------------------------
// Per-day commit values across all 128 calendar days.
// Each entry is (date, count). Idle days are present with count 0.
// ---------------------------------------------------------------------------

#let daily-commits = (
  ("2026-01-12", 70), ("2026-01-13", 74), ("2026-01-14", 40), ("2026-01-15", 28),
  ("2026-01-16", 51), ("2026-01-17", 54), ("2026-01-18", 125), ("2026-01-19", 59),
  ("2026-01-20", 177), ("2026-01-21", 87), ("2026-01-22", 17), ("2026-01-23", 47),
  ("2026-01-24", 77), ("2026-01-25", 49), ("2026-01-26", 65), ("2026-01-27", 42),
  ("2026-01-28", 83), ("2026-01-29", 85), ("2026-01-30", 156), ("2026-01-31", 31),
  ("2026-02-01", 58), ("2026-02-02", 118), ("2026-02-03", 125), ("2026-02-04", 36),
  ("2026-02-05", 37), ("2026-02-06", 27), ("2026-02-07", 54), ("2026-02-08", 38),
  ("2026-02-09", 43), ("2026-02-10", 35), ("2026-02-11", 54), ("2026-02-12", 78),
  ("2026-02-13", 67), ("2026-02-14", 19), ("2026-02-15", 31), ("2026-02-16", 5),
  ("2026-02-17", 12), ("2026-02-18", 4), ("2026-02-19", 4), ("2026-02-20", 11),
  ("2026-02-21", 11), ("2026-02-22", 10), ("2026-02-23", 0), ("2026-02-24", 3),
  ("2026-02-25", 29), ("2026-02-26", 26), ("2026-02-27", 34), ("2026-02-28", 11),
  ("2026-03-01", 27), ("2026-03-02", 17), ("2026-03-03", 7), ("2026-03-04", 12),
  ("2026-03-05", 24), ("2026-03-06", 23), ("2026-03-07", 21), ("2026-03-08", 24),
  ("2026-03-09", 21), ("2026-03-10", 22), ("2026-03-11", 0), ("2026-03-12", 2),
  ("2026-03-13", 1), ("2026-03-14", 1), ("2026-03-15", 6), ("2026-03-16", 0),
  ("2026-03-17", 0), ("2026-03-18", 2), ("2026-03-19", 21), ("2026-03-20", 28),
  ("2026-03-21", 16), ("2026-03-22", 40), ("2026-03-23", 26), ("2026-03-24", 22),
  ("2026-03-25", 42), ("2026-03-26", 17), ("2026-03-27", 25), ("2026-03-28", 20),
  ("2026-03-29", 57), ("2026-03-30", 30), ("2026-03-31", 12), ("2026-04-01", 13),
  ("2026-04-02", 37), ("2026-04-03", 60), ("2026-04-04", 3), ("2026-04-05", 2),
  ("2026-04-06", 0), ("2026-04-07", 16), ("2026-04-08", 30), ("2026-04-09", 25),
  ("2026-04-10", 28), ("2026-04-11", 9), ("2026-04-12", 21), ("2026-04-13", 9),
  ("2026-04-14", 27), ("2026-04-15", 17), ("2026-04-16", 12), ("2026-04-17", 28),
  ("2026-04-18", 23), ("2026-04-19", 8), ("2026-04-20", 45), ("2026-04-21", 20),
  ("2026-04-22", 5), ("2026-04-23", 5), ("2026-04-24", 2), ("2026-04-25", 2),
  ("2026-04-26", 6), ("2026-04-27", 2), ("2026-04-28", 39), ("2026-04-29", 32),
  ("2026-04-30", 11), ("2026-05-01", 6), ("2026-05-02", 6), ("2026-05-03", 27),
  ("2026-05-04", 20), ("2026-05-05", 7), ("2026-05-06", 77), ("2026-05-07", 31),
  ("2026-05-08", 57), ("2026-05-09", 93), ("2026-05-10", 23), ("2026-05-11", 36),
  ("2026-05-12", 142), ("2026-05-13", 88), ("2026-05-14", 97), ("2026-05-15", 5),
  ("2026-05-16", 34), ("2026-05-17", 84), ("2026-05-18", 94), ("2026-05-19", 94),
)

// ---------------------------------------------------------------------------
// Weekly summary. (week-label, date-range, total, headline)
// ---------------------------------------------------------------------------

#let weekly-summary = (
  ("W01", "01-12 to 01-18", 442, "Project initiation; canonical JSON; LLM scaffolding"),
  ("W02", "01-19 to 01-25", 513, "Peak Pre-RC1 (Jan 20 = 177); LLM batch / pooled; RC-1 cutover"),
  ("W03", "01-26 to 02-01", 520, "RC-1 hardening: telemetry, ChaosLLM, contract boundary tightening"),
  ("W04", "02-02 to 02-08", 435, "RC-2 cutover; RC-2.1 (Key Vault, schema contracts); RC-2.2; RC-2.3 begins"),
  ("W05", "02-09 to 02-15", 327, "RC-2.4 bug sprint; RC-2.5 (SQLCipher + ChaosWeb + declarative DAG)"),
  ("W06", "02-16 to 02-22",  57, "RC-3.2 prep + tag day"),
  ("W07", "02-23 to 03-01", 130, "RC-3.3 architectural-remediation kickoff"),
  ("W08", "03-02 to 03-08", 128, "RC-3.3 release; steady remediation cadence"),
  ("W09", "03-09 to 03-15",  53, "RC-3.4 release; transition to RC-4 planning"),
  ("W10", "03-16 to 03-22", 107, "RC-4 implementation accelerates"),
  ("W11", "03-23 to 03-29", 209, "Dataverse + RAG + output-schema-contracts"),
  ("W12", "03-30 to 04-05", 157, "RC-4.1 (RAG ingestion); RC-5 cut on Apr 3"),
  ("W13", "04-06 to 04-12", 129, "RC-5 settling; exception-hygiene merge"),
  ("W14", "04-13 to 04-19", 124, "Composer hardening; doc refresh sweep"),
  ("W15", "04-20 to 04-26",  85, "Statistical batch plugin design"),
  ("W16", "04-27 to 05-03", 123, "Composer skill-pack updates; Phase plans"),
  ("W17", "05-04 to 05-10", 308, "RC-5.1 closing sprint; Phase 1B persistence"),
  ("W18", "05-11 to 05-17", 486, "RC-5.1 release; RC-5.2 release; progress-persistence Phases 1-4"),
  ("W19", "05-18 to 05-19", 188, "Phase 6/7/8 polish (partial week)"),
)

// ---------------------------------------------------------------------------
// Capability accretion. (capability, first-rc) — sourced from the
// "Capabilities" table in the progress doc.
// ---------------------------------------------------------------------------

#let capability-accretion = (
  ("Auditable SDA pipeline", "RC-1"),
  ("LLM transforms (Azure / OpenRouter / batch / multi-query)", "RC-1"),
  ("Crash-safe checkpoint / resume", "RC-1"),
  ("Telemetry subsystem (OTLP / Azure Monitor / Datadog)", "RC-1 Hardening"),
  ("ChaosLLM stress / fault injection", "RC-1 Hardening"),
  ("Azure Key Vault secrets backend", "RC-2.1"),
  ("Schema contracts (first-row inference, propagation, audit)", "RC-2.1"),
  ("Tier 2 Langfuse tracing (v3)", "RC-2.1 / 2.2"),
  ("Typed PipelineRow end-to-end", "RC-2.3"),
  ("WebScrape transform with SSRF prevention", "RC-2.3"),
  ("DIVERT routing for quarantine / error sinks", "RC-2.3"),
  ("SQLCipher encryption-at-rest", "RC-2.5"),
  ("Declarative DAG wiring", "RC-2.5"),
  ("ChaosWeb fake server", "RC-2.5"),
  ("Strict typing at audit boundaries", "RC-3.2"),
  ("Single configurable LLM transform (T10)", "RC-3.3"),
  ("Layer-enforced 4-tier architecture (L0-L3)", "RC-3.3"),
  ("Repository pattern for Landscape", "RC-3.3"),
  ("Deep-immutability freeze_fields API", "RC-3.4"),
  ("hasattr() ban", "RC-3.4"),
  ("Dataverse plugin (OData v4)", "RC-4.0"),
  ("RAG retrieval transform", "RC-4.0"),
  ("Output schema contract enforcement", "RC-4.0"),
  ("immutability.freeze_guards CI rule", "RC-4.0"),
  ("ChromaSink (vector store population)", "RC-4.1"),
  ("depends_on pipeline sequencing", "RC-4.1"),
  ("Commencement gates", "RC-4.1"),
  ("Readiness contracts on retrieval providers", "RC-4.1"),
  ("elspeth web CLI + React composer", "RC-5.0"),
  ("Three-provider auth (Local / OIDC / Entra)", "RC-5.0"),
  ("Blob storage manager + secret references", "RC-5.0"),
  ("Background pipeline execution with WebSocket progress", "RC-5.0"),
  ("Sink failsink pattern", "RC-5.0"),
  ("elspeth-composer MCP server", "RC-5.0"),
  ("TokenRef type", "RC-5.0"),
  ("Guard-symmetry CI scanner", "RC-5.0"),
  ("Frontend Playwright E2E", "RC-5.1"),
  ("Composer pipeline recipes", "RC-5.1"),
  ("Source-inspection MCP tool", "RC-5.1"),
  ("Forced-repair loop with proof diagnostics", "RC-5.1"),
  ("10 statistical batch plugins", "RC-5.1"),
  ("Composer guided mode", "RC-5.2"),
  ("ComposerLLMCall audit channel", "RC-5.2"),
  ("Composer progress persistence (4 phases)", "RC-5.2"),
  ("38-entry redaction MANIFEST + 2,752-line walker", "RC-5.2"),
  ("Postgres-portability testcontainer lane", "RC-5.2"),
  ("Frontend recovery UX (panel + diff + redacted tool rows)", "RC-5.2"),
  ("Per-step guided chat", "RC-5.2"),
)

// Ordered list of all RC labels used as the column header in the
// accretion matrix.
#let rc-columns = (
  "RC-1", "RC-1 Hardening", "RC-2.1", "RC-2.1 / 2.2",
  "RC-2.3", "RC-2.5", "RC-3.2", "RC-3.3", "RC-3.4",
  "RC-4.0", "RC-4.1", "RC-5.0", "RC-5.1", "RC-5.2",
)
