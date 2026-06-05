// architecture.typ — ELSPETH Architecture Reference (RC-5.2)
// Audience: engineering reviewers, auditors, and integrators evaluating
// ELSPETH's architectural commitments.
//
// This document compresses docs/architecture/{overview,landscape,
// telemetry,token-lifecycle,subsystems,audit-remediation,requirements}.md
// plus the ADR set under docs/architecture/adr/ into a release-PDF
// formatted reference. It is NOT a draft — it describes the
// architecture as currently committed.
//
// Source-of-truth alignment:
//   When an architecture decision changes (new ADR, layer rule
//   amendment, trust-tier policy change, attributability test
//   refinement), update docs/architecture/*.md FIRST, then carry the
//   change here. The markdown set under docs/architecture/ is the
//   working source; this PDF is the release-formatted reading copy.
//
// Visual treatment differs from executive-summary.typ:
//   - draft: false                — established design, not under review
//   - h1-pagebreak: false         — linear reading, not slide-deck
//   - cover-hero-sda()            — appropriate; the SDA model is the
//                                   document's central subject.

#import "tokens.typ": *
#import "theme.typ": *
#import "data.typ" as data

#show: document-frame.with(
  title: "ELSPETH Architecture",
  subtitle: "RC-5.2 -- " + data.doc-date,
  draft: false,
  h1-pagebreak: false,
)

#cover-page(
  title: "Architecture Reference",
  subtitle: "Sense / Decide / Act — auditable by construction.",
  doc-date: data.doc-date,
  version: "RC-5.2",
  author: "John Morrissey, CTO Branch",
  affiliation: "Digital Transformation Agency",
  audience: "Engineering reviewers, auditors, integrators",
  classification: default-classification,
  status: "Reference — current as of " + data.doc-date,
  distribution: "Public evaluation release copy",
  hero: cover-hero-sda(),
)

#outline(
  title: text(font: font-body, size: size-h1, weight: "bold",
    fill: c-navy, "Contents"),
  indent: auto,
  depth: 2,
)
#pagebreak()

// ---------------------------------------------------------------------------
// 1. Framing
// ---------------------------------------------------------------------------

= Framing

ELSPETH is a domain-agnostic Sense / Decide / Act framework for
high-reliability, auditable data processing. The framework makes no
assumption about what a decision step does — LLM, ML model, rules
engine, threshold check — so long as the surrounding pipeline records
enough state for any output to be explained.

The architectural commitments below are the load-bearing ones: every
one of them shows up in code, in tests, and in CI gates. They are
documented in `docs/architecture/overview.md` (canonical), the ADR
set at `docs/architecture/adr/`, and (for the audit subsystem) in
`docs/architecture/landscape.md` and `landscape-entry-points.md`.

== Design principles

#callout(kind: "note", title: "Auditability first")[
  Every decision the system makes must be explainable. The
  attributability test: for any output, `explain(query, data_flow,
  run_id, token_id=…)` proves complete lineage back to the source
  row, the configuration in force, the code version, and any
  external calls. This is the core of the system, not optional
  telemetry.
]

#callout(kind: "note", title: "Domain agnostic")[
  The framework knows nothing about what transforms do. A weather
  monitor, a tender evaluator, and a satellite-anomaly detector use
  the same engine; only the plugins differ. The decision logic is
  pluggable; the audit machinery is not.
]

#callout(kind: "note", title: "Routing as first-class")[
  Rows can be routed to different destinations based on
  classification decisions. A gate transform can `continue`,
  `route_to_sink`, or `fork_to_paths`. Routing semantics are
  declared in configuration and enforced by the orchestrator; they
  are not buried in transform code.
]

#callout(kind: "note", title: "Reliability over performance")[
  Design choices favour correctness over speed and explicit over
  implicit. This is a high-reliability system, not a
  high-throughput system.
]

// ---------------------------------------------------------------------------
// 2. The SDA model
// ---------------------------------------------------------------------------

= The SDA model

#pdf.artifact(align(center, diagram-sda-flow()))

#text(size: size-small, fill: c-ink-soft)[
  *Figure — SDA flow.* External systems supply data to Sense
  (sources); Decide (transforms and gates) processes and classifies;
  Act (sinks) emits results to downstream systems. Every node
  records to the Landscape audit trail; the Landscape's records are
  the legal source of truth.
]

== Plugin primitives

#data-table(
  columns: 3,
  header: ("Primitive", "Role", "State"),
  align-rules: (left, left, left),
  ("Source",    "Get data into the system",           "Stateless"),
  ("Transform", "Do something with data",
   "Stateless between rows (except aggregations)"),
  ("Sink",      "Emit data to downstream systems",    "Stateless"),
)

The three-primitive taxonomy is deliberate. The distinction between
"LLM plugin" and "ML plugin" is artificial — both are transforms that
happen to make external calls. The uniform mental model keeps the
routing model uniform, which keeps the audit trail uniform.

== Transform subtypes

#data-table(
  columns: 3,
  header: ("Type", "Behaviour", "Routing"),
  align-rules: (left, left, left),
  ("Row Transform", "Process one row, emit one row",
   "No routing decision"),
  ("Gate", "Evaluate row, decide destination",
   "Returns continue / route_to_sink / fork_to_paths"),
  ("Aggregation", "Collect N rows until trigger, then emit result",
   "May or may not route"),
  ("Coalesce", "Merge results from parallel paths",
   "No routing decision"),
)

== Plugin ownership

All plugins are system-owned code, not user-provided extensions.
Sources, transforms, aggregations, and sinks are developed, tested,
and deployed with the same rigour as the engine itself. ELSPETH uses
`pluggy` for clean architecture, not to accept arbitrary user plugins.

#callout(kind: "advisory", title: "Implications for error handling")[
  Plugin defects crash the pipeline rather than degrade silently. A
  plugin that throws is a bug in code we control; a defective
  plugin that silently produces wrong results destroys the audit
  trail. Never wrap plugin calls in try / except to "recover" — let
  them crash. User-data defects (Tier 3) quarantine the row and
  continue; plugin defects (our code) stop the run.
]

// ---------------------------------------------------------------------------
// 3. Layer dependency model
// ---------------------------------------------------------------------------

= Layer dependency model

ELSPETH uses a strict four-layer model. Imports flow downward only;
upward imports fail CI. The boundary is enforced by the
`trust_tier.tier_model` `elspeth-lints` rule.

#data-table(
  columns: 3,
  header: ("Layer", "Name", "What it contains"),
  align-rules: (left, left, left),
  ("L0", "contracts/",
   "Shared types, enums, protocols. Imports nothing above."),
  ("L1", "core/",
   "Landscape, DAG, config, canonical JSON. Imports L0 only."),
  ("L2", "engine/",
   "Orchestrator, processors, executors. Imports L0–L1."),
  ("L3", "plugins/, mcp/, tui/, cli, telemetry/, testing/",
   "Application layer. Imports L0–L2."),
)

#callout(kind: "note", title: "Cross-layer resolution priority")[
  When a new cross-layer need arises: (1) move the code down to a
  lower layer if possible, (2) extract the primitive into
  `contracts/`, (3) restructure the caller to use dependency
  injection or a protocol. Never add a lazy import with an
  apologetic comment — that defers the structural fix and the
  pattern recurs.
]

The architecture is also *observable*: `elspeth-lints dump-edges`
emits the intra-layer import graph as JSON, Mermaid, or Graphviz DOT
for refactor planning and dependency-graph diffing across branches.

// ---------------------------------------------------------------------------
// 4. The Landscape (audit backbone)
// ---------------------------------------------------------------------------

= The Landscape

The Landscape is the audit backbone of ELSPETH. It captures every
run with its resolved configuration, every plugin instance registered
for the run, every row loaded from the source, every transform
applied to every row (with before / after state), every external call
made by transforms, every routing decision, and every artefact
produced by sinks.

== The attributability test

Given any output, prove complete lineage:

```python
lineage = landscape.explain(run_id, token_id=token_id, field=field)

assert lineage.source_row is not None
assert len(lineage.node_states) > 0

for state in lineage.node_states:
    assert state.input_hash is not None
    if state.status == "completed":
        assert state.output_hash is not None

if lineage.token.parents:
    for parent_token_id in lineage.token.parents:
        parent_lineage = landscape.explain(
            run_id, token_id=parent_token_id)
        assert parent_lineage is not None
```

If `explain()` cannot prove this chain, the Landscape has failed its
contract. "I don't know what happened" is never an acceptable answer
for any output.

== Core invariants

#data-table(
  columns: 2,
  header: ("Invariant", "Guarantee"),
  align-rules: (left, left),
  ("Run reproducibility",
   "Every run stores resolved config (not just hash)"),
  ("Deterministic linkage",
   "External calls link to spans that exist at call time"),
  ("Strict ordering",
   "Transforms ordered by (sequence, attempt); calls by (state_id, call_index)"),
  ("No orphan records",
   "Foreign keys enforced (PRAGMA foreign_keys=ON in SQLite)"),
  ("Uniqueness",
   "(run_id, row_id) unique; (state_id, call_index) unique"),
  ("Canonical JSON contract",
   "Hash algorithm versioned, never silently changed"),
  ("Token data isolation",
   "Sibling tokens (from fork or expand) have independent row_data"),
)

== Run-level reproducibility grade

Every run is assigned a grade based on its transforms:

#data-table(
  columns: 2,
  header: ("Grade", "Meaning"),
  align-rules: (left, left),
  ("FULL_REPRODUCIBLE",
   "All transforms deterministic — recompute any output from source"),
  ("REPLAY_REPRODUCIBLE",
   "Has non-deterministic calls, payloads retained — replay to identical downstream outputs"),
  ("ATTRIBUTABLE_ONLY",
   "Payloads purged or absent — lineage and hashes exist, cannot replay"),
)

The grade is computed at run completion and stored in run metadata.
It may degrade over time as payloads are purged under the retention
policy.

== Recompute vs replay vs verify

Precise terminology prevents confusion for implementers and auditors:

#data-table(
  columns: 3,
  header: ("Term", "Applies to", "Meaning"),
  align-rules: (left, left, left),
  ("Recompute", "Deterministic transforms",
   "Run the code again; expect identical output hashes"),
  ("Replay", "Non-deterministic calls",
   "Substitute recorded responses; no live calls"),
  ("Verify", "Non-deterministic calls",
   "Run live AND compare to recorded; flag differences"),
)

// ---------------------------------------------------------------------------
// 5. Three-tier trust model
// ---------------------------------------------------------------------------

= Trust tiers

ELSPETH treats data trust as a three-tier model with distinct handling
rules. The tiers are not advisory; they are enforced by code and by
CI rules.

#pdf.artifact(align(center, diagram-trust-tiers()))

#text(size: size-small, fill: c-ink-soft)[
  *Figure — three trust tiers.* External data enters at Tier 3 and
  is coerced, validated, or quarantined at the source boundary.
  Type-valid data flows downward into Tier 2 (pipeline) without
  re-coercion. Tier 1 (audit / Landscape) reads are guarded — a
  read of bad data from our own database raises `TIER_1_ERRORS`
  rather than coercing.
]

#data-table(
  columns: 2,
  header: ("Tier", "Handling rule"),
  align-rules: (left, left),
  ("Tier 1 — Our Data",
   "Must be 100% pristine. Bad data in the audit trail = crash immediately. No coercion, no defaults, no silent recovery."),
  ("Tier 2 — Pipeline Data",
   "Type-valid but potentially operation-unsafe. Transforms / sinks expect conformance — wrong type = upstream plugin bug. Wrap operations on values, not types."),
  ("Tier 3 — External Data",
   "Can be literal trash. Validate at the boundary, coerce where possible (e.g. \"42\" → 42), quarantine rows that can't be coerced or validated. Record absence as None — do not fabricate."),
)

#callout(kind: "advisory", title: "Coercion vs fabrication")[
  Coercion is meaning-preserving (`"42"` → `42`). Fabrication is
  not (`None` → `0` changes "unknown" to "zero"). The test: can
  the downstream consumer distinguish real data from synthetic? If
  not, it is fabrication. Inference from adjacent fields is still
  fabrication.
]

// ---------------------------------------------------------------------------
// 6. Failure semantics
// ---------------------------------------------------------------------------

= Failure semantics

== Token terminal states

Every token ends in a terminal state. There are no silent drops. The
terminal state is *derived* from `node_states`, `routing_events`,
and batch membership — it is not stored as a column.

#data-table(
  columns: 2,
  header: ("Terminal state", "Meaning"),
  align-rules: (left, left),
  ("COMPLETED",         "Reached output sink"),
  ("ROUTED",            "Sent to a named sink by a gate (move mode)"),
  ("FORKED",            "Split into child tokens"),
  ("CONSUMED_IN_BATCH", "Fed into an aggregation batch"),
  ("COALESCED",         "Merged with other tokens"),
  ("QUARANTINED",       "Failed, stored for investigation"),
  ("FAILED",            "Failed, not recoverable"),
  ("EXPANDED",          "Split into child tokens via a 1-to-N expand"),
)

`BUFFERED` is non-terminal — it becomes `COMPLETED` on flush.

== Retry semantics

Retries are explicit attempts with ordering. `(run_id, row_id,
transform_seq, attempt)` is unique; each attempt is recorded
separately; the final outcome indicates which attempt succeeded (or
that all failed). Backoff metadata is captured (delay, reason,
policy).

== Sink idempotency

Sinks receive idempotency keys to prevent duplicate side effects:
`{run_id}:{row_id}:{sink_name}:{artifact_kind}`. The system provides
at-least-once delivery; sinks should be idempotent or explicitly
acknowledge this limitation in configuration validation.

// ---------------------------------------------------------------------------
// 7. Canonical JSON and hashing
// ---------------------------------------------------------------------------

= Canonical JSON

Hashes are only meaningful with deterministic serialisation. Python's
built-in `hash()` is not stable across processes. Canonicalisation
happens in two phases:

#data-table(
  columns: 2,
  header: ("Phase", "What it does"),
  align-rules: (left, left),
  ("1. Normalise",
   "Our code. Convert pandas / numpy types to JSON-safe primitives. Reject NaN and Infinity (invalid input states, not \"missing\")."),
  ("2. Serialise",
   "rfc8785 — RFC 8785 JSON Canonicalisation Scheme. Standards-compliant; we keep our normalisation layer."),
)

The current canonical version is `sha256-rfc8785-v1`. The version is
stored with every run; old hashes remain valid under their recorded
version even if the rules later change.

#callout(kind: "advisory", title: "NaN policy: strict rejection")[
  NaN and Infinity are invalid input states, not "missing". Use
  `None` / `pd.NA` / `pd.NaT` for intentional missing values. The
  normaliser raises `ValueError` on non-finite floats. This
  prevents silent data corruption in audit records.
]

== Secret handling

Secrets are NEVER written to the Landscape. We store an HMAC
fingerprint for "same secret used" verification. Plain hashing
(`sha256(secret)`) creates an offline guessing oracle. HMAC with a
managed key requires the attacker to have both the fingerprint and
the key.

// ---------------------------------------------------------------------------
// 8. Architecture decisions (ADR index)
// ---------------------------------------------------------------------------

= Architecture decisions

The ADR set under `docs/architecture/adr/` is the canonical record of
architectural decisions. As of #data.doc-date, the headline decisions
relevant to a release reader are:

#data-table(
  columns: 2,
  header: ("ADR", "Subject"),
  align-rules: (left, left),
  ("ADR-001", "Plugin-level concurrency"),
  ("ADR-002", "Routing copy-mode limitation"),
  ("ADR-003", "Schema-validation lifecycle"),
  ("ADR-005", "Declarative DAG wiring"),
  ("ADR-006", "Layer-dependency remediation (the L0–L3 model)"),
  ("ADR-007", "Pass-through contract propagation"),
  ("ADR-008", "Runtime contract cross-check"),
  ("ADR-009", "Pass-through pathway fusion"),
  ("ADR-010", "Declaration-trust framework"),
  ("ADR-011", "Declared output-fields contract"),
  ("ADR-012", "Can-drop-rows contract"),
  ("ADR-013", "Declared required-fields contract"),
  ("ADR-014", "Schema config-mode contract"),
  ("ADR-015", "Creates-tokens contract"),
  ("ADR-016", "Source guaranteed-fields contract"),
  ("ADR-017", "Sink required-fields contract"),
  ("ADR-018", "Producer-site outcome discrimination"),
  ("ADR-019", "Two-axis terminal model"),
  ("ADR-020", "Retire batch LLM transforms"),
  ("ADR-021", "Sources and sinks uniformly boundary"),
  ("ADR-022", "Shareable reviews"),
  ("ADR-023", "Custom Python CI analyser"),
  ("ADR-024", "Delivery governance for single-maintainer mode"),
)

For the full text of each ADR (context, decision, consequences),
see `docs/architecture/adr/`.

// ---------------------------------------------------------------------------
// 9. Configuration precedence
// ---------------------------------------------------------------------------

= Configuration

Configurations merge with clear precedence — higher overrides lower:

#data-table(
  columns: 2,
  header: ("Level", "Source"),
  align-rules: (left, left),
  ("1 — Highest", "Runtime overrides (CLI flags, environment variables)"),
  ("2",           "Pipeline configuration (settings.yaml)"),
  ("3",           "Profile configuration (profiles/production.yaml)"),
  ("4",           "Plugin pack defaults (packs/llm/defaults.yaml)"),
  ("5 — Lowest",  "System defaults"),
)

Settings → Runtime contracts are enforced by the configuration layer;
contract drift between the documented settings and the runtime
contract is a build-time failure, not a runtime surprise.

// ---------------------------------------------------------------------------
// 10. Where to read more
// ---------------------------------------------------------------------------

= Reading on

#data-table(
  columns: 2,
  header: ("Document", "Subject"),
  align-rules: (left, left),
  ("docs/architecture/overview.md",
   "Full architecture overview — canonical source for this PDF"),
  ("docs/architecture/landscape.md",
   "Landscape data model in detail"),
  ("docs/architecture/landscape-entry-points.md",
   "Recording API: where audit calls enter the Landscape"),
  ("docs/architecture/token-lifecycle.md",
   "Token identity, fork / join / expand mechanics"),
  ("docs/architecture/subsystems.md",
   "Per-subsystem reference (engine, plugins, telemetry, MCP)"),
  ("docs/architecture/telemetry.md",
   "Telemetry vs Landscape primacy; emission discipline"),
  ("docs/architecture/audit-remediation.md",
   "Audit-trail remediation patterns and replay tooling"),
  ("docs/architecture/requirements.md",
   "Non-functional requirements (reliability, attributability, latency)"),
  ("docs/architecture/adr/",
   "Full ADR set (000-template + ADR-001 to ADR-024 as of " + data.doc-date + ")"),
)
