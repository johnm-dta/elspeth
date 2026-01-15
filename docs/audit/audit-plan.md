# ELSPETH Audit Plan and Checklist

## Purpose
Provide a repeatable, evidence-driven audit plan for ELSPETH runs and the
underlying system. This plan focuses on auditability, determinism, and
lineage completeness under extreme scrutiny.

## Scope and Assumptions
- Scope: Core framework behavior, audit trail integrity, and run reproducibility.
- Out of scope: External infrastructure (DB hosting, OS hardening) unless
  explicitly required by a specific audit.
- Assumption: Schema and requirements follow `docs/design/architecture.md` and
  `docs/design/requirements.md`.

## Required Audit Artifacts
- Landscape database for the run (or export archive if configured)
- Settings file(s) and environment variable list used for the run
- Version identifiers (code revision, plugin versions)
- Payload store location (if enabled) and retention policy

## Evidence Collection Workflow
1. Capture run identifiers and settings provenance.
2. Export or snapshot Landscape data for the run.
3. Collect payload store refs for any content hashes used in evidence.
4. Record validation results and test outputs referenced below.
5. Retain all evidence artifacts under an immutable audit archive.

## Global Invariants (All Audits)
- Resolved configuration is stored with the run and is deterministic.
- No silent drops: every token has a terminal disposition derived from records.
- All hashes are generated via canonical JSON with versioned algorithm.
- Every routing decision, external call, and sink artifact is traceable.
- Foreign key integrity is enforced (no orphan records).

## Subsystem Checklists

### Configuration Resolution
- Evidence:
  - `runs.settings_json` and `runs.config_hash`
  - Resolved config provenance (settings file path, env var keys used)
- Checks:
  - [ ] Resolved config is complete and non-empty
  - [ ] Secrets are redacted or HMAC-fingerprinted only
  - [ ] Config hash is stable across identical inputs
- Tests:
  - Unit tests for precedence and redaction
  - Integration test verifying stored config equals resolved config

### Plugin System
- Evidence:
  - `nodes.plugin_version`, `nodes.determinism`, `nodes.schema_hash`
- Checks:
  - [ ] Plugin identity maps to a single class per name/type
  - [ ] Metadata derives from plugin class, not runtime defaults
  - [ ] Lifecycle hooks are enforced as a contract (not optional)
- Tests:
  - Registry tests for metadata extraction and schema hashing
  - Lifecycle conformance tests

### Execution Graph Compiler
- Evidence:
  - `nodes` and `edges` tables for the run
- Checks:
  - [ ] DAG is acyclic and has exactly one source
  - [ ] All route labels resolve to valid edges
  - [ ] Edge labels are deterministic and collision-free
- Tests:
  - DAG validation tests
  - Route resolution tests for gates

### Orchestrator
- Evidence:
  - Run status and node registration sequence
- Checks:
  - [ ] Fail fast if resolved config or metadata is missing
  - [ ] Lifecycle order: source -> transforms -> sinks (start) and reverse on completion
  - [ ] No silent drops of tokens or outcomes
- Tests:
  - Run lifecycle tests, including failure paths
  - Lifecycle ordering tests

### Row Processing and Executors
- Evidence:
  - `node_states`, `routing_events`, `artifacts`
- Checks:
  - [ ] Routing semantics (copy vs move) are honored
  - [ ] Node states are ordered and complete per token
  - [ ] Artifacts recorded for all sink types
- Tests:
  - Routing tests for copy/move
  - Sink execution tests for file and non-file sinks

### Landscape (Audit Store)
- Evidence:
  - Full audit tables for run
- Checks:
  - [ ] Foreign keys and uniqueness constraints are satisfied
  - [ ] Hashes and timestamps are present where required
  - [ ] Explain queries reconstruct complete lineage
- Tests:
  - Recorder unit tests
  - Explain lineage integration tests

### Canonicalization
- Evidence:
  - Hashes in `runs`, `rows`, `node_states`, `artifacts`
- Checks:
  - [ ] RFC 8785 canonicalization applied consistently
  - [ ] NaN/Infinity are rejected, not coerced
  - [ ] Hash versions are recorded and stable
- Tests:
  - Canonicalization unit tests for numpy/pandas edge cases

### Payload Store
- Evidence:
  - Payload refs and retrieved payloads for sampled records
- Checks:
  - [ ] Stored payload matches hash
  - [ ] Retention policy enforced without breaking explain
- Tests:
  - Store/retrieve tests
  - Retention tests with explain fallback to hashes

### Retry and Checkpointing
- Evidence:
  - Attempt records, checkpoint entries
- Checks:
  - [ ] Attempt uniqueness and ordering are preserved
  - [ ] Resume points are deterministic and correct
- Tests:
  - Retry and checkpoint integration tests

### External Calls (Planned)
- Evidence:
  - `calls` records with request/response hashes
- Checks:
  - [ ] Request/response payloads are recorded and hash-verified
  - [ ] Secrets are never stored, only fingerprints
- Tests:
  - Call recorder/replayer tests (planned)

### Export Pipeline
- Evidence:
  - Export files, manifests, signatures
- Checks:
  - [ ] Export contains complete audit records
  - [ ] Signatures and manifests verify integrity
- Tests:
  - Export and signature validation tests

### Schema Enforcement (Planned)
- Evidence:
  - Schema hashes on nodes, validation errors in node states
- Checks:
  - [ ] Compatibility checks are deterministic and accurate
  - [ ] Runtime validation behavior matches policy (warn/error)
- Tests:
  - Config-time and runtime schema validation tests

### Observability and Spans
- Evidence:
  - Span traces (if enabled) and timing metrics
- Checks:
  - [ ] Spans cover source/transform/sink work
  - [ ] Errors are traced, not suppressed
- Tests:
  - Span emission tests or tracer mock assertions

### CLI/TUI
- Evidence:
  - CLI run logs and validation output
- Checks:
  - [ ] CLI passes resolved config into orchestrator
  - [ ] Validate mode matches run-time enforcement
- Tests:
  - CLI integration tests

## Audit Exit Criteria
- All checklist items are verified or have documented exceptions.
- Deviations from requirements are recorded with risk assessment and mitigation.
- Evidence artifacts are archived and linked to the audit report.

## Exceptions Log Template
- Exception ID:
- Subsystem:
- Description:
- Risk Assessment:
- Mitigation/Follow-up:
- Approver:
- Date:
