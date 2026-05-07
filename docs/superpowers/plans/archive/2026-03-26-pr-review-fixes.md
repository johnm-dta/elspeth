# PR Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all critical, important, and minor findings from the 6-agent PR review plus 3 elevated findings from specialist panel review.

**Architecture:** Option A — move `bootstrap_and_run()` to L3 (`cli_helpers.py`), define `PipelineRunner` Protocol in L0 (`contracts/`), inject runner callback into L2 functions. Fix secret resolution and passphrase gaps by leveraging L3 co-location. Fix readiness audit gap by reordering record-before-raise.

**Tech Stack:** Python 3.11+, frozen dataclasses, Pydantic v2, pluggy, structlog, pytest

---

## Batch 1 — Structural + Error Handling

### Task 1: Define PipelineRunner Protocol in contracts

**Files:**
- Modify: `src/elspeth/contracts/__init__.py`
- Create: `src/elspeth/contracts/pipeline_runner.py`
- Test: `tests/unit/contracts/test_pipeline_runner.py`

- [ ] **Step 1:** Create `contracts/pipeline_runner.py` with Protocol
- [ ] **Step 2:** Export from `contracts/__init__.py`
- [ ] **Step 3:** Write test verifying protocol compliance
- [ ] **Step 4:** Run tests, commit

### Task 2: Move bootstrap_and_run() to L3, wire secrets + passphrase

**Files:**
- Modify: `src/elspeth/cli_helpers.py` (add bootstrap_and_run)
- Modify: `src/elspeth/engine/bootstrap.py` (remove bootstrap_and_run, update resolve_preflight signature)
- Modify: `src/elspeth/engine/dependency_resolver.py` (accept runner callback, wrap exceptions, enum comparison)
- Modify: `tests/unit/engine/test_bootstrap_preflight.py`
- Modify: `tests/unit/engine/test_dependency_resolver.py`
- Modify: `tests/unit/cli/test_cli_preflight.py`

- [ ] **Step 1:** Update `resolve_preflight()` signature to accept probes + runner
- [ ] **Step 2:** Update `resolve_dependencies()` to accept runner callback
- [ ] **Step 3:** Move `bootstrap_and_run()` to `cli_helpers.py` with secret resolution + passphrase
- [ ] **Step 4:** Wire CLI call sites to pass probes and runner
- [ ] **Step 5:** Add DependencyFailedError wrapping for runner exceptions
- [ ] **Step 6:** Fix enum comparison (status.name → RunStatus.COMPLETED)
- [ ] **Step 7:** Update all tests (mock setup, import paths, new runner param)
- [ ] **Step 8:** Run tests, commit

### Task 3: Narrow except Exception in probes and readiness checks

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/probe_factory.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py`
- Modify: `tests/unit/plugins/infrastructure/test_probe_factory.py`
- Modify: `tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py`
- Modify: `tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py`

- [ ] **Step 1:** Narrow ChromaCollectionProbe.probe() — separate import, client construction, and collection access
- [ ] **Step 2:** Narrow ChromaSearchProvider.check_readiness() to chromadb.errors.ChromaError + ConnectionError + OSError
- [ ] **Step 3:** Narrow AzureSearchProvider.check_readiness() to httpx.HTTPError + ConnectionError + OSError
- [ ] **Step 4:** Add crash-through tests (TypeError, AttributeError must not be caught)
- [ ] **Step 5:** Add client-mode probe test (T1 gap)
- [ ] **Step 6:** Run tests, commit

### Task 4: Fix ChromaSink audit asymmetry

**Files:**
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py`
- Modify: `tests/unit/plugins/sinks/test_chroma_sink.py`

- [ ] **Step 1:** Replace slog.debug with AuditIntegrityError on error-path audit failure
- [ ] **Step 2:** Update test_error_path_audit_failure_preserves_original_exception
- [ ] **Step 3:** Run tests, commit

### Task 5: Fix CLI catch-all and readiness audit gap

**Files:**
- Modify: `src/elspeth/cli.py` (remove catch-all except Exception)
- Modify: `src/elspeth/plugins/transforms/rag/transform.py` (reorder record-before-raise)
- Modify: `tests/unit/plugins/transforms/rag/test_transform.py`
- Modify: `tests/unit/cli/test_cli_preflight.py`

- [ ] **Step 1:** Remove CLI except Exception catch-all (lines 518-523), reclassify unexpected errors as exit code 4
- [ ] **Step 2:** Move record_readiness_check() before conditional raise in transform.py
- [ ] **Step 3:** Add test: failed readiness check is recorded in Landscape
- [ ] **Step 4:** Run tests, commit

---

## Batch 2 — Hardening + Cleanup

### Task 6: Dead code removal + comment fixes

**Files:**
- Modify: `src/elspeth/contracts/config/runtime.py` (6 unreachable bounds checks)
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py` (tier classification comment)
- Modify: `src/elspeth/engine/commencement.py` (docstring fixes)
- Modify: `src/elspeth/engine/bootstrap.py` (phase numbering, docstring)
- Modify: `src/elspeth/plugins/infrastructure/probe_factory.py` (Tier 3 comment)
- Modify: `src/elspeth/contracts/export_records.py` (NodeStateExportRecord docstring)
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/connection.py` (module docstring)
- Modify: `src/elspeth/core/dependency_config.py` (PreflightResult docstring)
- Modify: `src/elspeth/engine/dependency_resolver.py` (redundant comment)

- [ ] **Step 1:** Remove 6 unreachable if-checks in runtime.py
- [ ] **Step 2:** Fix all comment issues from comment-reviewer
- [ ] **Step 3:** Run tests, commit

### Task 7: Validation + type hardening

**Files:**
- Modify: `src/elspeth/contracts/probes.py` (__post_init__ validation)
- Modify: `src/elspeth/core/dependency_config.py` (__post_init__ validation)
- Modify: `src/elspeth/contracts/errors.py` (non-empty validation, module-level import)
- Modify: `src/elspeth/contracts/export_records.py` (Literal narrowing)
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/connection.py` (port range)
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py` (port range)
- Modify: `src/elspeth/plugins/infrastructure/probe_factory.py` (dict→Mapping)
- Modify: `tests/unit/contracts/test_probes.py`
- Modify: `tests/unit/core/test_dependency_config.py`
- Modify: `tests/unit/contracts/test_new_errors.py`
- Modify: `tests/unit/contracts/test_export_records.py`

- [ ] **Step 1:** Add __post_init__ to CollectionReadinessResult (non-empty collection, count >= 0)
- [ ] **Step 2:** Add __post_init__ to DependencyRunResult (non-empty name/run_id/settings_hash, duration_ms >= 0)
- [ ] **Step 3:** Add non-empty validation to CommencementGateResult (name, condition)
- [ ] **Step 4:** Add non-empty validation to DependencyFailedError, RetrievalNotReadyError
- [ ] **Step 5:** Move deep_freeze import to module level in CommencementGateFailedError
- [ ] **Step 6:** Add Literal narrowing to all 15 export record TypedDicts
- [ ] **Step 7:** Add port ge=1, le=65535 to ChromaConnectionConfig and ChromaSinkConfig
- [ ] **Step 8:** Fix probe_factory config type: dict→Mapping
- [ ] **Step 9:** Add DuplicateDocumentError empty guard test (T2 gap)
- [ ] **Step 10:** Add commencement gate TOCTOU test (T3 gap)
- [ ] **Step 11:** Write all validation tests
- [ ] **Step 12:** Run tests, commit

### Task 8: Audit snapshot env_keys + flush docstring

**Files:**
- Modify: `src/elspeth/engine/commencement.py` (add env_keys to snapshot)
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py` (flush docstring)
- Modify: `tests/unit/engine/test_commencement.py`

- [ ] **Step 1:** Add env_keys (sorted key names, not values) to audit snapshot
- [ ] **Step 2:** Add docstring to ChromaSink.flush()
- [ ] **Step 3:** Update snapshot test
- [ ] **Step 4:** Run tests, commit
