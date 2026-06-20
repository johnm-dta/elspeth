# Vision — ELSPETH

> **Extensible Layered Secure Pipeline Engine for Transformation and Handling.**
> Bootstrapped from observed reality on 2026-06-14 (see `decisions/0001`).
> Purpose and audience are drawn from `README.md`; assumptions are marked.

## Purpose

ELSPETH is a high-assurance pipeline substrate for consequential workflows —
systems where a wrong output can cause operational, legal, safety, financial, or
security harm. It exists to make the *substrate itself the product*: pipelines
built from declared primitives (sources, transforms, pure-config gates,
aggregations, coalesce points, sinks) that carry schema and semantic contracts,
are validated before they run, and emit a complete, replayable Landscape audit
record. Validation and audit are core product properties, not after-the-fact
diagnostics. The change in the world it exists to make: let people build
consequential data/LLM workflows whose every decision is reviewable and
explainable to an auditor — **without** trading that assurance away for
authoring convenience.

## Who it serves

- **Primary — operators in sensitive/regulated/transactional/operational/
  medical/security/defence-adjacent workflows.** They hand-edit reviewable,
  version-controlled YAML; the pipeline can be read, reviewed, versioned, and
  explained before it runs. ELSPETH is open-source software developed and
  publicly released by DTA staff.
- **Secondary — knowledge workers building document QA, classification,
  routing, extraction, reporting, and review workflows.** They use the
  authenticated Web Composer (an LLM tool loop) — but over the *same* substrate,
  contracts, validation, and audit trail, never a weaker parallel engine. Served,
  but not at the primary's expense.
- **Explicitly not:** teams who need high-throughput ETL, sub-second streaming,
  or simple scripts with no audit requirement. The README routes those to Spark/
  dbt, Flink/Kafka Streams, and plain Python respectively.

## Anti-goals (what it refuses to be)

- **An easy-authoring tool that weakens assurance.** The whole reason the
  composer is a tool loop over contracts — and not a "generate YAML and hope"
  box — is to refuse the LLM-builder bargain of trading provenance for ease.
- **A high-throughput / low-latency data engine.** ELSPETH optimises for
  reviewability and audit integrity, not for ETL throughput or sub-second
  streaming. It will decline to compete there even under pressure.
- **A two-engine product.** The Web Composer is an authoring *surface* over the
  one runtime assurance model — never a second runtime with its own validation.
- **A blind-trust system.** No silent Tier-3 coercion, no un-audited divergence,
  no gate blessed without provenance. (Assumption to confirm: phrasing.)

## Authority grant

Granted by: John Morrissey (john@foundryside.dev)     Last reviewed: 2026-06-14
Review cadence: monthly, or on any vision/strategy change
Status: **confirmed** (operator-confirmed at bootstrap, 2026-06-14)

Autonomous within strategy — the agent MAY, without asking:
  prioritize the backlog, write PRDs, dispatch delivery, accept work against
  falsifiable criteria, reprioritize bets, and kill a failing bet per
  `metrics.md`.

Escalate BEFORE acting — the agent MUST get owner sign-off for:
  - changing this vision / strategy / authority grant;
  - tier-model / HMAC signing and any merge to `main` (operator holds the
    red-gate state deliberately — see project memory);
  - a public release, version stamp, or announcement;
  - deprecating a feature operators depend on;
  - data deletion **beyond** the already-authorized ELSPETH dev/staging session
    & audit DBs (that narrow grant stands separately);
  - anything touching an external party.
  (Taxonomy + rationale: product-ownership-operating-model.md.)
