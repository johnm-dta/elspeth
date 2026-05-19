# ELSPETH Framework — Security Architecture and Assurance Evidence

**Date:** 2026-03-01
**Document classification:** OFFICIAL
**Framework version:** 0.3.3 (RC-3.3)
**Measurements as of:** commit `ab9e2648` (2026-03-01, branch `RC3.3-architectural-remediation`)
**Prepared by:** Architecture analysis, synthesized from engineering specification, requirements, CI tooling, and test suite
**Intended audience:** CISOs and other security evaluators assessing the framework's suitability for use in auditable data processing environments

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Description](#2-system-description)
3. [Threat Model and Trust Boundaries](#3-threat-model-and-trust-boundaries)
4. [Security Controls](#4-security-controls)
   - 4.1 [Three-Tier Data Trust Model](#41-three-tier-data-trust-model)
   - 4.2 [Offensive Error Handling](#42-offensive-error-handling)
   - 4.3 [CI-Enforced Trust Model Compliance](#43-ci-enforced-trust-model-compliance)
   - 4.4 [Architecture Enforcement](#44-architecture-enforcement)
   - 4.5 [Configuration Contract Verification](#45-configuration-contract-verification)
   - 4.6 [Expression Evaluation](#46-expression-evaluation)
   - 4.7 [Audit Trail Integrity](#47-audit-trail-integrity)
   - 4.8 [Secret Management](#48-secret-management)
   - 4.9 [SSRF Prevention](#49-ssrf-prevention)
   - 4.10 [Crash Recovery with Topology Validation](#410-crash-recovery-with-topology-validation)
   - 4.11 [Rate Limiting](#411-rate-limiting)
   - 4.12 [Plugin Trust Surface Reduction](#412-plugin-trust-surface-reduction)
   - 4.13 [Content Safety Screening](#413-content-safety-screening)
5. [Assurance Evidence](#5-assurance-evidence)
   - 5.1 [Test Architecture](#51-test-architecture)
   - 5.2 [Property-Based Fuzzing of Security Boundaries](#52-property-based-fuzzing-of-security-boundaries)
   - 5.3 [Chaos Testing Infrastructure](#53-chaos-testing-infrastructure)
   - 5.4 [Mutation Testing](#54-mutation-testing)
   - 5.5 [Requirements Traceability](#55-requirements-traceability)
   - 5.6 [Architecture Decision Records](#56-architecture-decision-records)
   - 5.7 [Incident-Driven Hardening Cycle](#57-incident-driven-hardening-cycle)
   - 5.8 [Test Infrastructure Audit](#58-test-infrastructure-audit)
   - 5.9 [Engineering Specification](#59-engineering-specification)
6. [Residual Risk](#6-residual-risk)
7. [External Validation Status](#7-external-validation-status)
8. [Annex A: Independent Verification Procedures](#annex-a-independent-verification-procedures)
9. [Annex B: Development Methodology](#annex-b-development-methodology)
10. [Annex C: Comparative Context](#annex-c-comparative-context)

---

## 1. Executive Summary

ELSPETH is a Python framework for auditable Sense/Decide/Act (SDA) data pipelines, designed for environments where every processing decision must be traceable to its source data, configuration, and code version. The framework prioritises audit trail integrity as a first-class architectural concern.

Security posture summary:

- **~80,400 lines** of production code across 243 Python files
- **10,476 automated tests** across 622 test files (231K lines of test code — a 2.9:1 test-to-production ratio). The default CI run executes ~10,400; performance benchmarks and stress tests (71 tests) run on a separate schedule.
- **6-layer test pyramid** including 1,173 property-based (fuzz) tests and bespoke chaos testing infrastructure
- **3-tier data trust model** with distinct error handling strategies per trust level, enforced by custom AST-based static analysis in CI
- **19-table audit database** with HMAC-signed exports, RFC 8785 canonical hashing, and SQLCipher encryption at rest
- **25 frozen (immutable) dataclass types** for all audit records — language-level mutation prevention
- **4-layer architecture model** with 0 violations, enforced by CI on every commit
- **374 tracked requirements** with implementation status and evidence links
- **6 Architecture Decision Records** with rationale, alternatives, and consequences
- **361 individually justified static analysis allowlist entries**, each with owner, reason, safety rationale, and expiration date

No independent penetration test or external security audit has been performed to date. This document presents internal assurance evidence for external evaluation. For protected workloads, the platform is intended to undergo an IRAP assessment when an appropriate candidate workload is identified.

---

## 2. System Description

### Purpose

ELSPETH provides scaffolding for data processing workflows where the "decide" step may be an LLM, ML model, rules engine, or threshold check. It is domain-agnostic — the framework handles pipeline orchestration, audit recording, and lineage tracking while plugins handle domain-specific logic.

### Processing Model

```
SENSE (Sources) → DECIDE (Transforms/Gates) → ACT (Sinks)
```

- **Source**: Loads data from external systems (CSV, API, database, message queue). Exactly one per run.
- **Transform**: Processes or classifies data. Zero or more, executed in DAG order. Includes routing gates for conditional path selection.
- **Sink**: Outputs results to external destinations. One or more named sinks per pipeline.

Pipelines compile to directed acyclic graphs (DAGs). Token identity tracks row instances through forks and joins, providing complete lineage.

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| CLI | Typer |
| Configuration | Dynaconf + Pydantic |
| Plugins | pluggy |
| Data | pandas |
| Database | SQLAlchemy Core (multi-backend) |
| Migrations | Alembic |
| Canonical JSON | `rfc8785` (RFC 8785 / JCS standard) |
| DAG validation | NetworkX |
| Rate limiting | pyrate-limiter |
| LLM integration | LiteLLM (optional) |
| Encryption at rest | SQLCipher (optional) |

### Operational Context

ELSPETH runs as a CLI tool on the operator's infrastructure. It processes data from external sources through a configured pipeline and writes results to configured sinks. The audit database is a local SQLite (or SQLCipher) file co-located with the pipeline execution.

The framework does not operate as a persistent service (server mode is a planned future capability). Each pipeline execution is a discrete run with its own audit trail.

---

## 3. Threat Model and Trust Boundaries

### Assets Under Protection

| Asset | Confidentiality | Integrity | Availability |
|-------|----------------|-----------|--------------|
| **Audit trail** (Landscape database) | Medium — may contain business-sensitive lineage | **Critical** — legal record, must be tamper-evident | High — loss requires re-processing |
| **Pipeline correctness** | N/A | **Critical** — every row must reach exactly one terminal state | High — incorrect routing has downstream impact |
| **Secret material** (API keys, vault credentials) | **Critical** — must never appear in audit trail or logs | High — HMAC fingerprints must be correct | Medium |
| **External boundary** (ingested data) | Low | High — must be validated before entering pipeline | Medium |

### Trust Assumptions

| Actor | Trust Level | Rationale |
|-------|-------------|-----------|
| Pipeline operator (writes YAML config, runs CLI) | Trusted | Operators define what data to process and where to send it. Malicious config = malicious pipeline. |
| ELSPETH codebase (plugins, engine, contracts) | Fully trusted | All code is system-owned, tested, and CI-gated. |
| External data sources (CSV files, API responses, LLM outputs) | Zero trust | Anything from outside the system boundary may be malformed, malicious, or missing. |
| Audit database contents (on read) | Fully trusted | Data was written by the framework. Corruption indicates system failure. |
| Host environment | Trusted | File system, network, and Python runtime are assumed non-hostile. |

### Threat Actors and Scope

| Threat Actor | In Scope | Notes |
|-------------|----------|-------|
| Malformed external data (accidental) | Yes | Primary threat — CSV garbage, API errors, LLM hallucination |
| Malicious external data (deliberate) | Yes | Expression injection, SSRF via crafted URLs, prompt injection via LLM |
| Compromised upstream API | Partially | Validated at Tier 3 boundary; credential theft out of scope |
| Hostile operator with host access | No | Operator is trusted; host compromise is an environment-level concern |
| Supply chain compromise | No | Dependency management is an environment-level concern |
| Network-level attacks | No | Network controls are assumed present in the deployment environment |

### What ELSPETH Does Not Claim

- **Not a sandbox for untrusted code.** All 25 plugins are system-owned. There is no untrusted plugin execution.
- **Not hardened against a hostile operator.** Root access to the host can modify the binary, database, or configuration.
- **Not a network security tool.** SSRF prevention is defense-in-depth, not a substitute for firewalls or egress filtering.

---

## 4. Security Controls

### 4.1 Three-Tier Data Trust Model

Error handling strategy is determined by the trust level of the data being accessed — specifically, by **who authored the data**:

| Tier | Data Origin | Strategy | Violation Response |
|------|------------|----------|-------------------|
| **Tier 1** | Audit database (our data) | Full trust — crash on any anomaly | `AuditIntegrityError` — indicates corruption or tampering |
| **Tier 2** | Pipeline data (post-source-validation) | Elevated trust — types valid, operations may fail | `TransformResult.error()` — quarantine the row, continue processing |
| **Tier 3** | External systems (CSV, APIs, LLM responses) | Zero trust — validate, coerce, quarantine | Coerce at boundary, quarantine failures, record what was received |

Key design principle: **coercion is meaning-preserving; fabrication is not.** Converting `"42"` → `42` preserves the value (coercion, permitted at Tier 3). Converting `None` → `0` changes the meaning from "unknown" to "zero" (fabrication, forbidden at all tiers).

Trust tiers follow **data flows, not plugin types**. A transform that calls an external LLM API creates a Tier 3 boundary within a Tier 2 component. The external response must be validated immediately at the call boundary before being treated as Tier 2 data.

| Plugin Type | Coercion Permitted | Rationale |
|-------------|-------------------|-----------|
| Source | Yes | Normalises external data at ingestion boundary |
| Transform (on pipeline row data) | No | Receives validated data; wrong types = upstream bug |
| Transform (on external call response) | Yes | External response is Tier 3 — validate/coerce immediately |
| Sink | No | Receives validated data; wrong types = upstream bug |

**Enforcement mechanism:** See section 4.3 (CI-enforced compliance).

### 4.2 Offensive Error Handling

The framework explicitly forbids defensive programming patterns against system-owned code and data. Silent error suppression is treated as a category of defect more severe than a crash.

Rationale: in an auditable pipeline, a silent wrong result — data that flows through and gets recorded as "correct" when it isn't — is a worse outcome than a crash. A crash stops the pipeline and triggers investigation. A silent wrong result contaminates the audit trail.

Forbidden patterns include defensive access on typed objects (`.get()`, `getattr()` with defaults, `hasattr()`, `isinstance()` on system-owned code), broad exception suppression, and silent fallback values. These are enforced by the 9-rule CI checker described in section 4.3.

Required patterns:

- Direct attribute access on typed objects (crash on bug)
- `from exc` on all re-raised exceptions (preserve diagnostic chain)
- Explicit `AuditIntegrityError` with context when Tier 1 data is anomalous

The decision test:

| Situation | Response |
|-----------|----------|
| System-owned data is wrong | Crash immediately — this is corruption |
| User-provided data is wrong | Quarantine the row, continue processing |
| External API returned invalid data | Validate at boundary, coerce once, trust thereafter |

### 4.3 CI-Enforced Trust Model Compliance

A custom 1,117-line AST-based static analysis tool (`scripts/cicd/enforce_tier_model.py`) enforces the trust model on every commit. The tool parses Python source code into abstract syntax trees and detects patterns that violate the trust model:

| Rule | Detection |
|------|-----------|
| R1 | `.get()` with default values on system-owned data structures |
| R2 | `getattr(obj, "field", default)` on typed dataclasses |
| R3 | Bare `except Exception` that swallows errors without re-raising |
| R4 | `except BaseException` (overly broad exception handling) |
| R5 | `isinstance()` / `hasattr()` checks on system-owned objects |
| R6 | Silent exception handling with fallback values |
| R7 | `contextlib.suppress()` usage |
| R8 | `dict.setdefault()` on Tier 1 data |
| R9 | `dict.pop(key, default)` with implicit defaults |

Legitimate exceptions (e.g., `isinstance()` in an AST walker that must dispatch on node types) require an explicit allowlist entry with four mandatory fields:

| Field | Purpose |
|-------|---------|
| `key` | Exact file path, rule, symbol context, and code fingerprint |
| `owner` | Who justified the exception (person, team, or defect ID) |
| `reason` | Why the pattern exists at a trust boundary |
| `safety` | How failures are handled — never "it's fine" |
| `expires` | Rolling expiration date — stale entries fail the build |

The project maintains **361 such entries** across 10 per-module YAML files. Each entry is specific to a code location and includes a content-hash fingerprint, meaning code changes invalidate the entry automatically.

Source plugins (which sit at the Tier 3 boundary) are exempted from defensive-pattern rules at the per-file level, because defensive programming is the correct strategy at the external data boundary.

Pre-commit hooks run 12 checks on the full codebase (not just staged files):

- Ruff linting + formatting
- mypy strict type checking
- Tier model enforcement (custom AST analysis)
- Configuration contract verification (custom AST analysis)
- Standard hygiene checks (trailing whitespace, merge conflicts, debug statements, large files, YAML/TOML validity)

### 4.4 Architecture Enforcement

A strict 4-layer dependency model, enforced by CI:

```
L0  contracts/     Leaf — imports nothing above. Shared types, enums, protocols.
L1  core/          Can import L0 only. Landscape, DAG, config, canonical JSON.
L2  engine/        Can import L0, L1. Orchestrator, processors, executors.
L3  plugins/       Can import L0, L1, L2. Sources, transforms, sinks, clients.
    mcp/ tui/ cli* telemetry/ testing/   — also L3 (application layer)
```

The `enforce_tier_model.py` script detects upward imports and fails the build. The project had 10 violations at the time of ADR-006; all 10 were resolved, and CI enforcement has maintained 0 violations since.

When a cross-layer dependency is needed, the resolution priority is:

1. Move the code to the lower layer
2. Extract the type/constant into `contracts/` (L0)
3. Restructure the caller using dependency injection or protocols defined in L0

The leaf module principle (`contracts/` has zero outbound dependencies) enables all layers to share types without creating circular dependencies.

### 4.5 Configuration Contract Verification

Configuration uses a two-layer pattern to prevent silent divergence between what the operator configures and what the engine uses:

1. **Settings classes** (Pydantic) validate operator YAML
2. **Runtime\*Config classes** (frozen dataclasses) are consumed by engine components
3. **Protocol definitions** specify what the engine expects
4. **`from_settings()` methods** explicitly map every field

Three enforcement layers catch gaps:

- **mypy** (structural typing) verifies the dataclass satisfies the protocol
- **Custom AST checker** verifies `from_settings()` uses all Settings fields
- **Alignment tests** verify field mappings are correct and complete

This control exists because of defect P2-2026-01-21: a configuration field (`exponential_base`) was added to the Settings class, validated by Pydantic, and accepted from operators — but silently ignored at runtime because the mapping to the engine was never created.

### 4.6 Expression Evaluation

Routing conditions are parsed using Python's `ast` module into an abstract syntax tree, then evaluated against a restricted visitor. The expression parser (655 lines, `core/expression_parser.py`) uses a whitelist model:

- **Permitted:** Field access (`row['field']`), comparisons, boolean operators, arithmetic, literals, membership tests, ternary expressions
- **Forbidden:** Function calls (except `row.get`), `lambda`, comprehensions, assignment expressions, `await`/`yield`, f-strings, arbitrary attribute access, arbitrary name resolution

No use of `eval()`, `exec()`, or `compile()`.

The parser is property-tested with Hypothesis (see section 5.2) to fuzz for bypass vectors, including attack patterns such as `__import__('os').system(...)`.

### 4.7 Audit Trail Integrity

The Landscape audit subsystem is the architectural backbone:

| Capability | Implementation |
|-----------|---------------|
| Schema | 19 database tables recording complete lineage (runs, nodes, edges, rows, tokens, token parents, outcomes, node states, operations, calls, artifacts, routing events, batches, batch members, batch outputs, validation errors, transform errors, checkpoints, secret resolutions) |
| Complete traceability | `elspeth explain --run <id> --row <id>` reconstructs the full path any row took through the pipeline |
| Tamper-evident exports | HMAC-signed with manifest hash chains |
| Deterministic hashing | RFC 8785 canonical JSON (JSON Canonicalization Scheme). NaN and Infinity are rejected, not silently converted. |
| Encryption at rest | SQLCipher, configured via passphrase or environment variable |
| Secret exclusion | Secrets are never stored; HMAC fingerprints are recorded for verification |
| Immutable records | 25 frozen dataclass types in `contracts/audit.py`. `frozen=True, slots=True` provides language-level mutation prevention — `FrozenInstanceError` on any mutation attempt, `AttributeError` on arbitrary attribute assignment. |
| Read-only forensic tooling | The Landscape MCP server provides read-only access to the audit database. The tool architecturally cannot mutate the evidence it examines. |

Design principle: **"I don't know what happened" is never an acceptable answer for any output.** Every output must be provably traceable to source data, configuration, and code version.

### 4.8 Secret Management

- Secrets are never stored in the audit trail or logs
- HMAC fingerprints are recorded, enabling "was this the same key?" verification without exposing the key
- External secret loading via Azure Key Vault with audit recording of resolution events (vault source, fingerprint, latency)
- **Fail-closed:** Missing fingerprint key + secrets in configuration = startup failure, not silent degradation

### 4.9 SSRF Prevention

URL and IP validation (`core/security/web.py`) before external HTTP calls:

- Blocks private networks (RFC 1918), link-local, loopback
- Covers IPv4-mapped IPv6 bypass attempts (`::ffff:10.0.0.1`)
- Covers zone-scoped IPv6 addresses
- Multi-homed hostname resolution validation
- Property-tested for bypass resistance (see section 5.2)

**Known residual risk:** See section 6 (DNS rebinding TOCTOU).

### 4.10 Crash Recovery with Topology Validation

Checkpointing enables resume of interrupted runs. The resume path validates that the pipeline DAG topology hash matches the checkpoint before resuming. Resuming against a different pipeline configuration is rejected — it would corrupt the audit trail by mixing lineage from two different pipeline definitions.

### 4.11 Rate Limiting

Per-service rate limiting (`core/rate_limit/`) using `pyrate-limiter`:

- All plugins sharing an external service share the rate limit bucket
- Configurable backpressure strategy (block or drop)
- Prevents resource exhaustion against external APIs

### 4.12 Plugin Trust Surface Reduction

All 25 plugins are system-owned code, developed and tested as part of the framework with the same CI rigour as engine code. ELSPETH uses `pluggy` for clean architectural separation, not for accepting arbitrary user-provided code. There is no plugin sandbox because there is no untrusted plugin code.

### 4.13 Content Safety Screening

For LLM-based pipelines, Azure Content Safety and Prompt Shield plugins provide content screening. These were hardened to **fail-closed** after P0 defects were discovered where they originally failed-open (a `None` return from the safety API was being treated as "safe" rather than "unknown").

---

## 5. Assurance Evidence

### 5.1 Test Architecture

A six-layer test pyramid, with each layer serving a specific verification purpose:

| Layer | Count | Purpose |
|-------|-------|---------|
| Unit | ~8,500 | Component-level correctness (includes core alignment / configuration contract verification) |
| Property | 1,173 | Hypothesis-based fuzzing — generates randomised inputs at runtime |
| Integration | ~610 | Cross-subsystem interactions through production code paths |
| End-to-end | ~60 | Full pipeline execution (source → transform → sink) |
| Performance | 71 | Benchmarks, stress tests, scalability, memory profiling (excluded from default CI run) |
| Fixtures | 37 | Test infrastructure verification (factory and fixture correctness) |

**Total:** 10,476 tests across 622 test files (231K lines of test code). The default CI run executes ~10,400 tests; performance benchmarks and stress tests are marked and run on a separate schedule.

Integration tests are required to use production factory methods (`ExecutionGraph.from_plugin_instances()`, `instantiate_plugins_from_config()`). This rule was established after defect BUG-LINEAGE-01, where tests that manually constructed graph objects passed while the production factory method had a different (incorrect) node mapping. The test suite was audited to ensure compliance (see section 5.8).

### 5.2 Property-Based Fuzzing of Security Boundaries

1,173 tests across 83 files use the Hypothesis library to generate randomised inputs. Two security-relevant examples:

**SSRF prevention fuzzing** (`test_ssrf_properties.py`): Generates IP addresses from every blocked range (private networks, link-local, loopback), including IPv4-mapped IPv6 bypass attempts, zone-scoped IPv6, and multi-homed hostname resolution. Verifies the SSRF filter is fail-closed (defaults to blocking when uncertain).

**Expression injection fuzzing** (`test_expression_safety.py`): Generates arbitrary Python expressions including function calls, lambdas, comprehensions, assignment expressions, `await`/`yield`, f-strings, and attribute access. Verifies the AST-based expression parser rejects every forbidden construct while accepting all valid routing conditions.

These are generative tests — they discover attack patterns that were not explicitly anticipated by the developer.

### 5.3 Chaos Testing Infrastructure

External integration testing uses bespoke HTTP servers (~9,500 lines) that simulate realistic failure modes, rather than mock objects that return canned responses:

**ChaosLLM** (`testing/chaosllm/`): A running HTTP server compatible with OpenAI and Azure OpenAI chat completion endpoints.

- Configurable rates of 429 rate limits, 5xx errors, timeouts, mid-response disconnects, malformed JSON
- Latency simulation with burst patterns for rate limiter validation
- SQLite metrics database for query/response pattern analysis

**ChaosWeb** (`testing/chaosweb/`): A running HTTP server for web scraping transform testing.

- HTTP error injection (4xx/5xx, timeouts, connection resets, slow responses)
- Content generation with encoding variations
- Named failure profiles from 5% to 50% error rates

These servers exercise the same production code paths as real external calls — retry logic, rate limiting, timeout handling, error recording, and Tier 3 boundary validation are all tested against realistic failure conditions.

### 5.4 Mutation Testing

Mutation testing (via mutmut) is configured as a weekly CI workflow. It introduces artificial bugs into production code and verifies that the test suite detects them. Target scores by subsystem, prioritised by security impact:

| Subsystem | Target* | Rationale |
|-----------|---------|-----------|
| `canonical.py` | 95%+ | Hash integrity is foundational to audit trail |
| `landscape/` | 90%+ | Audit trail is the legal record |
| `engine/` | 85%+ | Orchestration correctness affects routing and lineage |

*These are targets for the next scheduled run, not achieved scores. The mutation testing framework and CI integration are in place; the run is pending completion of the RC3.3 architectural remediation (mutation testing against code undergoing structural reorganisation produces unreliable results). Achieved scores will be published separately.

### 5.5 Requirements Traceability

A 374-requirement traceability matrix (`docs/architecture/requirements.md`) where each requirement has:

| Field | Purpose |
|-------|---------|
| Requirement ID | Unique identifier with domain prefix (e.g., `CFG-017`, `LND-042`) |
| Requirement | What the system must do (🆕 prefix marks requirements discovered during implementation, not in original spec) |
| Source | Where the requirement originated (specification section, ADR, bug analysis, etc.) |
| Status | Implementation state |
| Evidence | Code path, test file, or configuration reference |

Status breakdown:

| Status | Count | Meaning |
|--------|-------|---------|
| IMPLEMENTED | 327 | Fully built and verified |
| FIXED | 10 | Bug-driven requirements — discovered via defect analysis, implemented and closed |
| PARTIAL | 11 | Partially implemented with known gaps |
| DEFERRED | 6 | Consciously deferred to a later release |
| NOT IMPLEMENTED | 14 | Not yet built |
| DIVERGED | 2 | Implemented differently than specified, with documented justification |
| IMPROVED / EXTENDED / CHANGED | 4 | Variants of DIVERGED — implemented with enhancements beyond the original specification |

The DIVERGED family of statuses (DIVERGED, IMPROVED, EXTENDED, CHANGED) acknowledges that implementation sometimes legitimately departs from the original specification, and requires the departure to be documented rather than silent.

### 5.6 Architecture Decision Records

6 formal ADRs documenting key decisions:

| ADR | Decision | Security Implication |
|-----|----------|---------------------|
| ADR-001 | Plugin-level concurrency (not orchestrator) | Deterministic audit trail ordering |
| ADR-002 | Move-only routing (no copy) | Unambiguous token provenance |
| ADR-003 | Two-phase schema validation at DAG construction | Type mismatches caught before data processing |
| ADR-004 | Explicit sink routing via named edges | Every routing decision auditable |
| ADR-005 | Declarative DAG wiring (`input`/`on_success`) | No implicit routing conventions |
| ADR-006 | Strict 4-layer model with CI enforcement | Prevents dependency cycles and architectural erosion |

### 5.7 Incident-Driven Hardening Cycle

Every significant defect generates a structural countermeasure targeting the entire class of defect, not just the specific instance. The cycle is: defect → root cause analysis → structural countermeasure → CI enforcement.

| Incident | Root Cause | Structural Countermeasure |
|----------|-----------|--------------------------|
| BUG-LINEAGE-01 | Tests used manual graph construction; production factory had different mapping | Rule: integration tests must use production code paths. Test infrastructure audit. |
| P2-2026-01-21 | Settings field accepted by Pydantic, silently ignored at runtime | Configuration contract system: 3 enforcement layers (mypy, AST checker, alignment tests) |
| P0 Content Safety | Safety plugin returned `None` instead of blocking | Fail-closed hardening: `None` treated as unsafe. Bool type validation enforced. |
| P0 Prompt Shield | Same pattern as Content Safety | Same hardening sweep applied across all safety plugins |
| 10 lazy imports | Architecture eroding at ~0.5 violations/day | ADR-006: structural remediation (10→0) + CI gate preventing new violations |

The hardening cycle compounds: each incident reduces the surface area for the next class of defect.

### 5.8 Test Infrastructure Audit

A 763-line audit of the test suite (`docs/audits/test-infrastructure-audit-2026-03-01.md`) was conducted after a test was found constructing a `PluginContext` with an invalid `operation_id`, bypassing the foreign key chain required by the audit database.

The audit identified 900+ violations across 120+ test files where tests constructed objects directly instead of using centralised factories. The remediation plan specifies 5 priority tiers, 6 new factory functions, and a CI enforcement script.

Additionally, the project performed a complete test suite rewrite (v1→v2), deleting 222,000 lines and replacing them with 231,000 lines that conform to the trust model. The old suite was deleted in a single commit — not deprecated or dual-maintained.

### 5.9 Engineering Specification

The `CLAUDE.md` file (897 lines) is a machine-readable engineering specification loaded into the AI development agent's context at the start of every session. It codifies the trust model, error handling rules, coercion rules, and architectural constraints.

This creates three properties relevant to security assurance:

1. The specification is consumed for every code change — it cannot be "not read."
2. Specification updates propagate to all subsequent development sessions immediately.
3. CI independently verifies that the codebase conforms to the specification on every commit.

The specification is version-controlled alongside the code it governs.

---

## 6. Residual Risk

| Risk | Likelihood | Impact | Current Mitigation | Tracking |
|------|-----------|--------|-------------------|----------|
| **DNS rebinding TOCTOU in SSRF prevention** — `validate_url_for_ssrf()` validates the IP, but `httpx` re-resolves the hostname. An attacker controlling DNS could change resolution between validation and connection. | Low (requires DNS infrastructure control) | High (SSRF to internal services) | Property-tested for known bypass vectors. | Tracked, architectural fix requires custom DNS resolver. |
| **Non-atomic file writes** — JSON sink, CSV sink, payload store, and journal use truncate-then-write. Crash during write causes data loss. | Medium (crash during write window) | Medium (data durability, not confidentiality) | Tracked across 4 subsystems. | Under active remediation. |
| **NaN/Infinity in float validation** — Accepted by source validation, undermines RFC 8785 canonicalisation guarantees. | Low (specific float values required) | Medium (hash integrity for affected rows) | Sanitisation layer added for quarantine paths. | Full fix requires source-level rejection. |
| **Untyped dicts at Tier 1 boundary** — 10 open defects where `dict[str, Any]` crosses into the audit trail where frozen dataclasses should be used. | Medium (existing code paths) | Low (data is correct but not type-guaranteed) | Fix pattern established (TokenUsage precedent). | Each being addressed individually. |
| **Unsandboxed Jinja2 templates** — blob_sink template rendering and ChaosLLM test server use unsandboxed Jinja2. | Low (templates are operator-authored or test-only) | Medium (template injection if operator is untrusted — but operator is in trust boundary) | ChaosLLM is testing-only. blob_sink templates are written by trusted operators. | Tracked. |
| **No external security audit** | N/A | N/A | Internal analysis only to date. This document provides assurance evidence for external evaluation. | IRAP assessment intended for protected workloads when candidate workload identified. |
| **Mutation testing scores not yet baselined** | N/A | N/A | Framework and CI integration in place. Run pending completion of RC3.3 remediation. | Scheduled. |

---

## 7. External Validation Status

No independent penetration test or external security audit has been performed to date. The findings in this document are the product of internal analysis:

- Systematic codebase review
- Property-based fuzzing of security boundaries
- Custom static analysis (trust model enforcement)
- Incident-driven hardening cycle (178 defects triaged to date)
- Test infrastructure audit

This document is intended to provide assurance evidence suitable for external evaluation. For protected workloads, the platform is intended to undergo an IRAP (Information Security Registered Assessors Program) assessment when an appropriate candidate workload is identified. The internal controls, verification procedures (Annex A), and residual risk disclosures documented here are designed to support that assessment process.

---

## Annex A: Independent Verification Procedures

The following commands can be executed in the project root to independently verify claims made in this document. All commands assume a configured Python virtual environment (`.venv/`).

| Claim | Verification Command | Expected Output |
|-------|---------------------|-----------------|
| 10,476 automated tests | `.venv/bin/python -m pytest tests/ --co -q -o "addopts=" 2>/dev/null \| tail -3` | 10,476 tests collected (default run without `-o` flag shows ~10,400 after marker deselection) |
| Tier model enforcement, 0 new violations | `python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` | Exit code 0, 0 new findings |
| 361 allowlist entries | `grep -r "^- key:" config/cicd/enforce_tier_model/ \| wc -l` | 361 |
| 25 frozen audit dataclasses | `grep -c "frozen=True" src/elspeth/contracts/audit.py` | 25 |
| Configuration contract alignment | `.venv/bin/python -m scripts.check_contracts` | Exit code 0 |
| No eval()/exec()/compile() in expression parser | `grep -n "eval\|exec\|compile" src/elspeth/core/expression_parser.py` | No matches |
| 4-layer architecture, 0 violations | Same as tier model enforcement command above | 0 layer violations in output |
| ~80,400 lines of production code | `find src/elspeth -name '*.py' \| xargs wc -l \| tail -1` | ~80,400 total |
| RFC 8785 canonical JSON (no custom serialisation) | `grep -r "import rfc8785" src/elspeth/` | Import present in `core/canonical.py` |
| All pre-commit hooks present | `cat .pre-commit-config.yaml` | 12 hook entries visible |
| 374 tracked requirements | `grep -cP '^\| [A-Z]{2,5}-\d{3}' docs/architecture/requirements.md` | 374 |
| 6 ADRs | `ls docs/architecture/adr/00[1-9]*.md \| wc -l` | 6 |
| Property tests exist and pass | `.venv/bin/python -m pytest tests/property/ -q` | ~1,173 tests passed |
| Test-to-production ratio | Compare `find src/elspeth -name '*.py' \| xargs wc -l \| tail -1` with `find tests -name '*.py' \| xargs wc -l \| tail -1` | ~231K test lines / ~80K production lines ≈ 2.9:1 |

---

## Annex B: Development Methodology

ELSPETH is developed with AI assistance (Claude Code) governed by the engineering specification described in section 5.9. This section provides context on the development methodology for evaluators who consider development process relevant to assurance.

The AI development agent operates under the same constraints as the CI pipeline: the trust model, error handling rules, architecture enforcement, and coding standards are loaded into the agent's context for every session. When the agent deviates from the specification, CI blocks the commit.

Properties of this approach relevant to assurance:

1. **Specification consumption is guaranteed** — the specification is a system input, not a document to be read.
2. **Hardening cycle has high iteration velocity** — the project has completed 178 defect triages, a complete test suite rewrite (222K lines deleted and replaced), and a 31-task architectural remediation, each tightening the security posture.
3. **Custom CI tooling is economically feasible** — the 1,117-line AST-based static analysis tool with 361 individually justified allowlist entries was built and iterated as part of normal development.
4. **Institutional memory is persistent** — hard-won lessons from incidents are permanently encoded in the engineering specification.

---

## Annex C: Comparative Context

For readers interested in how ELSPETH's security controls compare to common development methodologies, the following table provides simplified context against two baselines:

- **"Ad-hoc AI-assisted"** (colloquially known as "vibe coding") refers to AI-assisted development without engineering constraints — prompting an LLM to generate code and shipping whatever compiles, with minimal testing or specification. This is the default mode of unstructured AI-assisted development.
- **"Plan-driven (waterfall)"** refers to traditional sequential development. Mature waterfall organisations (CMMI Level 3+) do have configuration management, traceability, and test strategies; the characterisations below reflect common *practice* gaps rather than inherent methodology limitations. In many cases, waterfall can achieve equivalent outcomes — the constraint is typically economic (the human labour cost of sustaining these practices at scale) rather than methodological.

Agile/DevOps is not included as a separate baseline. Mature Agile teams already practice many of the same feedback loops; the meaningful differences narrow to enforcement mechanism (code review vs. CI gates) and specification durability (tribal knowledge vs. machine-consumed specification).

A more detailed comparative analysis is available in the companion document (`security-comparison-blog.md`).

| Security Dimension | Ad-hoc AI-assisted | Plan-driven (waterfall) | ELSPETH |
|-------------------|-------------------|------------------------|---------|
| Data trust model | None | In requirements doc, inconsistently applied | 3-tier model, CI-enforced via custom AST analysis |
| Error handling | Silent suppression | Defensive ("resilient") | Offensive — crash on system bugs, quarantine user data |
| Test strategy | Ad hoc | Separate test team, late | 6-layer pyramid, property fuzzing, chaos testing |
| Architecture enforcement | None | Design doc (drifts) | 4-layer model, CI-enforced, 0 violations |
| Audit trail | Logging | Logging + compliance checklist | 19-table audit DB, HMAC exports, RFC 8785, SQLCipher |
| Configuration safety | No validation | Schema validation | Two-layer settings→runtime with protocol enforcement |
| Expression safety | `eval()` | Restricted `eval()` | AST-parsed, whitelist-based, property-fuzzed |
| Secret handling | Plaintext in config | Env vars or vault | HMAC fingerprinting, Key Vault, fail-closed |
| Pre-commit checks | None or formatting | Linting + maybe types | 12 hooks: lint, types, tier model, contracts, hygiene |
| Requirements traceability | None | RTM (maintained, drifts) | 374 requirements with status and evidence links |
| Test effectiveness verification | None | Coverage metrics | Mutation testing framework (targets set, baseline pending) |
| Bug feedback loop | Fix the instance | Defect tracker, triage | Every defect → structural countermeasure → CI enforcement |
| Known vulnerabilities | Undiscovered | Found by pentest, deferred | Continuously discovered, tracked, prioritised |
| Data immutability | Mutable objects | Immutable by convention | 25 frozen dataclasses — language-level enforcement |
| Engineering specification | None | Written once, drifts | Machine-consumed every session, CI-verified every commit |
