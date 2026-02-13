# ELSPETH RC-3 Documentation & Project Layout Remediation Plan

**Date:** 2026-02-13
**Status:** IMPLEMENTED (2026-02-13)
**Auditor:** Claude Code (RC3-quality-sprint)
**Scope:** Documentation, project metadata, configuration paths — no runtime code changes.

## Decisions (WI-1, resolved 2026-02-13)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **F-01: Version number** | Bump to `0.3.0` | Reflects RC-3 milestone |
| **F-05: REQUIREMENTS.md** | Delete — pyproject.toml is single source of truth | Eliminates version drift by removing the duplicate |
| **F-16: docs/bugs/** | Update markdown tracker for now; retire shortly | Beads is the active tracker; markdown kept temporarily for continuity |

---

## Context

A full audit of the project layout and documentation was performed on 2026-02-13 as part of the RC-3 transition. The project has evolved significantly since RC-2 (routing trilogy, test suite v2 migration, gate plugin removal, engine package refactoring, graceful shutdown, etc.) but documentation has not kept pace. This plan captures every finding and organizes remediation by dependency order.

**Stats at audit time:**
- Source: ~75,800 lines across 234 Python files in `src/elspeth/`
- Tests: ~207,000 lines
- Branch: `RC3-quality-sprint`

---

## Finding Summary

| # | Finding | Severity | Phase |
|---|---------|----------|-------|
| F-01 | RC-2/RC-2.5 version strings across 10+ files | HIGH | 1 |
| F-02 | CLAUDE.md source layout diagram is outdated | HIGH | 1 |
| F-03 | Gate plugin references persist in contract docs | HIGH | 1 |
| F-04 | mutmut paths in pyproject.toml point to deleted files | HIGH | 1 |
| F-05 | REQUIREMENTS.md dependency versions completely stale | HIGH | 1 |
| F-06 | Broken links in docs/README.md | HIGH | 1 |
| F-07 | Placeholder URLs in pyproject.toml and Dockerfile | MEDIUM | 2 |
| F-08 | docs/README.md ADR list incomplete | MEDIUM | 2 |
| F-09 | Release docs remain RC-2 oriented | MEDIUM | 2 |
| F-10 | Feature inventory is 3 weeks stale | MEDIUM | 2 |
| F-11 | ARCHITECTURE.md LOC statistics stale | MEDIUM | 2 |
| F-12 | pyproject.toml classifier still Alpha | MEDIUM | 2 |
| F-13 | TEST_SYSTEM.md title still says "v2" | LOW | 3 |
| F-14 | RC-3 remediation plan has contradictory FEAT-05 state | LOW | 3 |
| F-15 | Plans index metadata drift | LOW | 3 |
| F-16 | Bug tracking docs internally inconsistent | LOW | 3 |
| F-17 | Example README coverage gaps | LOW | 4 |
| F-18 | Archived docs point to non-existent canonical location | LOW | 4 |
| F-19 | Layout inconsistency for analysis docs | LOW | 4 |
| F-20 | Telemetry OTLP exporter hardcodes version "0.1.0" | LOW | 2 |

---

## Phase 1 — Fix Silent Breakage and Source-of-Truth Errors (P0)

These items cause incorrect behavior or mislead anyone using the docs as a reference.

### F-01: Version/Status String Normalization

All RC-2 and RC-2.5 references must be updated to RC-3.

| File | Line(s) | Current | Target |
|------|---------|---------|--------|
| `CLAUDE.md` | 7 | `RC-2` | `RC-3` |
| `README.md` | 7 | badge `RC-2.5` | badge `RC-3` |
| `ARCHITECTURE.md` | 5 | `RC-2.5 branch` | `RC-3` |
| `ARCHITECTURE.md` | 6 | `0.1.0 (RC-2.5)` | version TBD `(RC-3)` |
| `docs/README.md` | 5 | `RC-2` | `RC-3` |
| `docs/release/guarantees.md` | 3-4 | `RC-2` | `RC-3` |
| `docs/guides/tier2-tracing.md` | 5 | `RC-2` | `RC-3` |
| `docs/architecture/telemetry.md` | 3 | `RC-2` | `RC-3` |
| `pyproject.toml` | 3 | `0.1.0` | Decision needed (see note) |
| `src/elspeth/__init__.py` | 8 | `0.1.0` | Must match pyproject.toml |

**Version number decision:** The semver `0.1.0` has been static since project inception. Options:
- `0.3.0` to reflect RC-3
- `0.1.0-rc3` for pre-release tag
- Keep `0.1.0` until GA

### F-02: CLAUDE.md Source Layout Diagram

The source tree at `CLAUDE.md:~630-660` is significantly outdated. Three single-file modules are now packages, and several new files are missing.

**Replace the current tree with:**

```
src/elspeth/
├── core/
│   ├── landscape/      # Audit trail storage (recorder, exporter, schema)
│   ├── checkpoint/     # Crash recovery checkpoints
│   ├── dag/            # DAG construction, validation, graph models (NetworkX)
│   │   ├── builder.py  # DAG construction from config
│   │   ├── graph.py    # ExecutionGraph with public API
│   │   └── models.py   # Node, Edge, DAG data models
│   ├── rate_limit/     # Rate limiting for external calls
│   ├── retention/      # Payload purge policies
│   ├── security/       # Secret fingerprinting via HMAC, URL/IP validation
│   ├── config.py       # Configuration loading (Dynaconf + Pydantic)
│   ├── canonical.py    # Deterministic JSON hashing (RFC 8785)
│   ├── events.py       # Synchronous event bus for CLI observability
│   ├── identifiers.py  # ID generation utilities
│   ├── logging.py      # Structured logging setup
│   ├── operations.py   # Operation type definitions
│   ├── payload_store.py # Content-addressable storage for large blobs
│   └── templates.py    # Jinja2 field extraction
├── contracts/          # Type contracts, schemas, and protocol definitions
├── engine/
│   ├── orchestrator/   # Full run lifecycle management
│   │   ├── core.py     # Main orchestrator logic
│   │   ├── aggregation.py # Aggregation flush/timeout handling
│   │   ├── export.py   # Post-run export orchestration
│   │   ├── outcomes.py # Token outcome resolution
│   │   ├── types.py    # Orchestrator type definitions
│   │   └── validation.py # Pre-run validation
│   ├── executors/      # Transform, gate, sink, aggregation executors
│   │   ├── transform.py # Transform execution
│   │   ├── gate.py     # Config-driven gate evaluation
│   │   ├── sink.py     # Sink execution
│   │   ├── aggregation.py # Aggregation execution
│   │   └── types.py    # Executor type definitions
│   ├── processor.py    # DAG traversal with work queue
│   ├── dag_navigator.py # DAG path navigation
│   ├── coalesce_executor.py # Fork/join barrier with merge policies
│   ├── batch_adapter.py # Batch windowing logic
│   ├── retry.py        # Tenacity-based retry with backoff
│   ├── tokens.py       # Token identity and lineage management
│   ├── triggers.py     # Aggregation trigger evaluation
│   ├── expression_parser.py # AST-based expression parsing (no eval)
│   ├── clock.py        # Clock abstraction for testing
│   └── spans.py        # Telemetry span management
├── plugins/
│   ├── sources/        # CSVSource, JSONSource, NullSource, AzureBlobSource
│   ├── transforms/     # FieldMapper, Passthrough, Truncate, etc.
│   ├── sinks/          # CSVSink, JSONSink, DatabaseSink, BlobSink
│   ├── llm/            # Azure OpenAI transforms (batch, multi-query)
│   ├── clients/        # HTTP, LLM, Replayer, Verifier clients
│   ├── batching/       # Batch-aware transform adapters
│   └── pooling/        # Thread pool management for plugins
├── telemetry/          # OpenTelemetry exporters and instrumentation
├── testing/            # ChaosLLM, ChaosWeb, ChaosEngine test servers
├── mcp/                # Landscape MCP analysis server
├── tui/                # Terminal UI (Textual) - explain screens and widgets
├── cli.py              # Typer CLI
├── cli_helpers.py      # CLI utility functions
└── cli_formatters.py   # Event formatting for CLI output
```

### F-03: Gate Plugin References in Contract Docs

Gate plugins were deliberately removed (hard prohibition, 2026-02-11). These docs still reference deleted code:

| File | What to fix |
|------|-------------|
| `docs/contracts/system-operations.md:~100-230` | Remove `PluginGateReason` TypedDict, `GateProtocol` section, "Plugin gates" narrative. Update `RoutingReason` union type to remove `PluginGateReason`. |
| `docs/contracts/execution-graph.md:477-478` | Remove `GateExecutor.execute_gate()` reference. Gates are config-driven only. |
| `CLAUDE.md:196` | Change "Sources, Transforms, Gates, Aggregations, Sinks" to "Sources, Transforms, Aggregations, Sinks" — Gates are system operations, not plugins. |

### F-04: mutmut Paths in pyproject.toml

`pyproject.toml:358-361` references files that are now packages. Mutation testing silently finds no code to mutate.

| Current Path | Replacement |
|-------------|-------------|
| `src/elspeth/engine/orchestrator.py` | `src/elspeth/engine/orchestrator/` |
| `src/elspeth/engine/executors.py` | `src/elspeth/engine/executors/` |

### F-05: REQUIREMENTS.md Dependency Version Table

The entire dependency table in `REQUIREMENTS.md` reports January-era minimums. Every row needs updating to match `pyproject.toml` actuals:

| Package | REQUIREMENTS.md (stale) | pyproject.toml (actual) |
|---------|------------------------|------------------------|
| `typer` | `>=0.12` | `>=0.21,<1` |
| `textual` | `>=0.52` | `>=7.2,<8` |
| `pydantic` | `>=2.6` | `>=2.12,<3` |
| `pluggy` | `>=1.4` | `>=1.6,<2` |
| `tenacity` | `>=8.2` | `>=9.0,<10` |
| `networkx` | `>=3.2` | `>=3.6,<4` |
| `structlog` | `>=24.1` | `>=25.0,<26` |
| `pyrate-limiter` | `>=3.1` | `>=3.9,<4` |
| `deepdiff` | `>=7.0` | `>=8.6,<9` |
| `opentelemetry-*` | `>=1.23` | `>=1.39,<2` |
| `litellm` | `>=1.30` | `>=1.81,<2` |
| `openai` | `>=1.0` | `>=2.15,<3` |
| `pytest` | `>=8.0` | `>=9.0,<10` |
| `ruff` | `>=0.3` | `==0.15.0` |
| `mypy` | `>=1.8` | `>=1.19,<2` |

**Recommendation:** Consider whether REQUIREMENTS.md should be auto-generated from pyproject.toml, or whether it should be deleted entirely in favor of pyproject.toml as the single source of truth.

### F-06: Broken Links in docs/README.md

| Line | Link Target | Status | Fix |
|------|-------------|--------|-----|
| 47-51 | `quality-audit/` | MISSING | Repoint to `archive/quality-audit-2026-01-22/` |
| 71 | `plans/RC2-remediation.md` | MISSING | Repoint to `plans/RC3-remediation.md` |

---

## Phase 2 — Release Metadata and Documentation Alignment (P1)

### F-07: Placeholder URLs

`pyproject.toml:196-198` and `Dockerfile:48` use `github.com/your-org/elspeth` while `README.md` uses the real URL `github.com/johnm-dta/elspeth`.

| File | Line(s) | Fix |
|------|---------|-----|
| `pyproject.toml` | 196-198 | Replace `your-org` with `johnm-dta` |
| `Dockerfile` | 48 | Replace `your-org` with `johnm-dta` |
| `scripts/deploy-vm.sh` | 16, 35, 71 | Replace `your-org` with `johnm-dta` |

### F-08: docs/README.md ADR List Incomplete

Line 37-39 lists only ADR 001-003. ADR 004 (Explicit Sink Routing) and ADR 005 (Declarative DAG Wiring) are missing.

Add:
```markdown
  - [ADR-004: Explicit Sink Routing](design/adr/004-adr-explicit-sink-routing.md)
  - [ADR-005: Declarative DAG Wiring](design/adr/005-adr-declarative-dag-wiring.md)
```

### F-09: Release Docs Remain RC-2 Oriented

| File | Issue | Fix |
|------|-------|-----|
| `docs/release/rc2-checklist.md` | RC-2 specific | Keep as historical; create `rc3-checklist.md` |
| `docs/release/guarantees.md` | Versioned as RC-2 | Update version header to RC-3 |
| `docs/release/feature-inventory.md` | Footer says "Next update: After RC-2 release" | Update footer |
| `docs/README.md:75-76` | Links described as "RC-2 checklists and guarantees" | Update description text |

### F-10: Feature Inventory is 3 Weeks Stale

`docs/release/feature-inventory.md` is dated January 29, 2026. Missing features added since:

- Graceful pipeline shutdown (FEAT-05, commit `6286367f`)
- DROP-mode sentinel requeue handling (commit `bbc2f515`)
- ExecutionGraph public API refactoring (TEST-01, commit `1c31869b`)
- Multiple contract and telemetry fixes from RC3-quality-sprint

### F-11: ARCHITECTURE.md LOC Statistics

| Stat | Current | Actual |
|------|---------|--------|
| Line 20: Production LOC | `~74,000` | `~75,800` |
| Line 21: Test LOC | `~201,000 (2.7:1)` | `~207,000 (2.7:1)` |

### F-12: pyproject.toml Classifier

Line 11: `"Development Status :: 3 - Alpha"` — for RC-3, `4 - Beta` is more appropriate.

### F-20: Telemetry OTLP Exporter Hardcoded Version

`src/elspeth/telemetry/exporters/otlp.py:435` hardcodes `version="0.1.0"`. Should reference `elspeth.__version__` instead.

---

## Phase 3 — Internal Consistency Fixes (P2)

### F-13: TEST_SYSTEM.md Title

Line 1 still says "Test Suite v2 Design". The v2 migration is complete and v2 IS the suite now. Update title to "Test System Design" or "Test Suite Architecture".

Also update the intro blurb (lines 3-4) which says "Drop-in replacement for `tests/`. Built one file at a time..." — this is historical migration context that is no longer relevant.

### F-14: RC-3 Remediation Plan Contradictory FEAT-05 State

In `docs/plans/RC3-remediation.md`:
- Line 78 marks FEAT-05 as DONE
- Line 163 still includes FEAT-05 in effort totals
- Line 174 still lists FEAT-05 as must-have

Reconcile by removing FEAT-05 from remaining-work sections and updating totals.

### F-15: Plans Index Metadata Drift

In `docs/plans/README.md`:
- Line 19 says "Active Plans (3)" but lists 4 plan rows
- Line 12 references `cancelled/` as top-level, but actual location is `docs/plans/completed/cancelled`

### F-16: Bug Tracking Docs Internally Inconsistent

- `docs/bugs/open/README.md` claims 1 open bug but the referenced file may not exist
- `docs/bugs/README.md` still recommends urgent P0/P1 work from older snapshot
- `docs/bugs/BUGS.md` may not reflect current state of `.beads/` issue tracker

**Recommendation:** Since the project now uses beads for issue tracking, consider whether the `docs/bugs/` directory should be archived entirely.

---

## Phase 4 — Polish and Coverage (P3) — DONE

### F-17: Example README Coverage Gaps — DONE

12 example directories now have README files. Additionally, a master `examples/README.md` was created with a full index, categorised listings, and a "If you want to see X, look at Y" guide. Coverage gaps (fork/coalesce, checkpoint/resume, rate limiting, database sink) are documented in the master README.

### F-18: Archived Docs Point to Non-Existent Location — DONE

`docs/archive/quality-audit-2026-01-22/README.md` updated to remove stale reference to non-existent `docs/quality-audit/` directory.

### F-19: Layout Inconsistency for Analysis Docs — DONE

`docs/arch-analysis-2026-02-02-1114` deleted entirely (7 files). Will regenerate architecture docs fresh when needed.

---

## Execution Order

```
Phase 1 (P0): Silent breakage — F-01 through F-06
  ├── F-04 first (mutmut — invisible test infrastructure breakage)
  ├── F-03 next (gate references — contradicts hard prohibition)
  ├── F-02 next (CLAUDE.md layout — reference for all future work)
  ├── F-05 next (REQUIREMENTS.md — misleads setup)
  ├── F-01 next (version strings — cosmetic but broad)
  └── F-06 last (broken links)

Phase 2 (P1): Release alignment — F-07 through F-12, F-20
  ├── F-07, F-08, F-09 (metadata and links)
  ├── F-10, F-11 (content freshness)
  └── F-12, F-20 (classifier and version reference)

Phase 3 (P2): Internal consistency — F-13 through F-16

Phase 4 (P3): Polish — F-17 through F-19
```

---

## Verification Checklist

- [ ] `grep -r "RC-2" *.md docs/**/*.md` returns only historical/archive references
- [ ] `grep -r "your-org" pyproject.toml Dockerfile scripts/` returns nothing
- [ ] No references to `GateProtocol`, `BaseGate`, `execute_gate()`, `PluginGateReason` outside of `archive/`
- [ ] `mutmut run --paths-to-mutate` resolves all paths to actual files
- [ ] REQUIREMENTS.md versions match pyproject.toml
- [ ] CLAUDE.md source tree matches `ls -R src/elspeth/` structure
- [ ] All links in `docs/README.md` resolve to existing files
- [ ] `docs/README.md` ADR list includes all 5 ADRs (001-005)
- [ ] `ARCHITECTURE.md` LOC numbers are within 5% of actual
- [ ] `docs/release/` contains RC-3 checklist
- [ ] `docs/TEST_SYSTEM.md` title has no "v2" qualifier

---

## Notes

- This plan is documentation-only except for F-04 (pyproject.toml mutmut paths) and F-20 (telemetry version reference) which touch config/source files but not runtime behavior.
- RC-3 release communication should not proceed until Phase 1 and Phase 2 are complete.
- The REQUIREMENTS.md question (F-05) may warrant a decision to delete the file entirely. `pyproject.toml` is already the canonical source.
- The `docs/bugs/` question (F-16) depends on whether beads has fully replaced the markdown bug tracker.
