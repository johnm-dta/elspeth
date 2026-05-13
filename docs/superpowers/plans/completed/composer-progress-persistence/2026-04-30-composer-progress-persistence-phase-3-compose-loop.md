# Composer Progress Persistence — Phase 3: Compose-Loop Persistence + Tool-Call Cap (rev 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Revision history.** Rev 1 (commit `6bcf2f2d`, 2026-05-10) was authored when Phase 2 was planned but unimplemented. Rev 2 (2026-05-12) re-baselined against the Phase 2 implementation at HEAD `f54ee7e8`. Rev 3 (2026-05-14) re-baselined against current `RC5.2` HEAD `47efb073722c7628d7f8d84ca13127cd50f6f6c8` after the four-axis plan review found stale rev-2 assumptions. Rev 4 (2026-05-14) applied the NO_GO_REV_3 feedback: added FastAPI exception-handler task, `SessionServiceProtocol` injection into `ComposerServiceImpl`, persisted assistant IDs through the composer result/carrier contract, literal route-disposition rows, no Task 7 ellipsis, concrete property-test state/injection requirements, `raw_assistant_content=None`, and in-scope Tier-1 runbook. Rev 5 (2026-05-14) applied the NO_GO_REV_4 architecture feedback: changed `ComposerServiceImpl` DI from required keyword-only to sentinel default + first-use guard, added a whole-codebase signature-change sweep rule, and named the exact composer seam-contract docstring in `src/elspeth/web/composer/protocol.py`. Rev 6 (2026-05-14) applied the NO_GO_REV_5_QUALITY feedback: added a quality-axis red-test fixture/helper ground rule, created Task 0 for `_run_one_turn_for_test` plus fixture definitions/citations, required property-machine helpers to wire `sessions_service`, added signature-sweep escalation, and replaced Tier-1 `getattr` handling with direct `exc.failed_turn` access plus an invariant test. Rev 7 (this revision, 2026-05-14) applies the NO_GO_REV_6_SYSTEMS feedback: makes `AuditIntegrityError.failed_turn` typed-nullable with a documented degraded route body for non-compose-loop origins, and adds a systems-axis ground rule requiring cancellation enum/injection-table updates for any new `_compose_loop` await point.

**Goal:** Wire the live compose loop through the Phase 1 persistence primitive so a failed compose turn lands its assistant row, tool rows, and composition state atomically; enforce the per-turn tool-call cap; surface `failed_turn` + `tool_responses_persisted` on 422/500 error bodies; and expose `include_tool_rows=true` on `GET /api/sessions/{sid}/messages` with audit-grade access logging.

**Architecture:** Compose-loop body in `src/elspeth/web/composer/service.py:_compose_loop` is rewritten to the §5.2.1 single-sync-block shape — gather tool outcomes async, redact via the Phase 2 manifest walker, dispatch ONE `persist_compose_turn_async` per turn. Phase 1 owns the schema, the sync primitive, and the advisory lock; Phase 2 owns the redaction manifest; this phase owns the loop + route shape + transcript endpoint. The legacy route-layer tool-row drain is `src/elspeth/web/sessions/routes.py:_persist_tool_invocations`; Task 6 enumerates which of its call sites are removed, retained, or guarded so the compose-loop cutover is atomic rather than half-deleted.

**Tech Stack:** Python 3.13, SQLAlchemy 2.x (sync `Engine`), FastAPI, structlog, OpenTelemetry, Hypothesis (property tests), pytest, testcontainer PostgreSQL on the Docker-enabled CI lane.

**Spec sections:**
§5.2 (insertion sites — loop shape, sync write function, what this design eliminates, cancellation semantics, atomicity grain),
§5.4 (`partial_state` redaction symmetry),
§5.5 (failure-mode interaction table — twelve rows),
§5.7.5 (`MANIFEST` lookup integration),
§6.1 (`_handle_*` `failed_turn` field, read-consistency note),
§6.2 (transcript endpoint + `include_tool_rows`),
§6.3 (audit-grade access logging + `audit_access_log` writes),
§8.2 (CL-PP-1..13 integration tests),
§8.3 (property test — `INV-AUDIT-AHEAD` bidirectional),
§8.5.1 (testcontainer CI lane),
§11 Phase 3 scope + done-when.

**Spec revision:** This plan consumes **revision 5** of the spec (manifest-keyed redaction; `AuditOutcome` two-field shape; `AuditIntegrityError`-raising Tier-1 disposition). Cite the spec body, not the rev-4 §11 supersession block.

**Plan-review posture:** The latest sibling review sidecar reports `NO_GO_REV_6_SYSTEMS`: architecture, quality, and reality axes are green, but the rev-6 handler assumed every `AuditIntegrityError` reaching FastAPI had compose-loop `failed_turn` metadata even though existing non-compose-loop raise sites do not. Rev 7 is a plan repair only. Do not start Phase 3 implementation until this plan is re-reviewed and receives GO.

---

## Dependency posture (rev-3 update — RC5.2 is the baseline)

**This plan executes on `RC5.2` at HEAD `47efb073722c7628d7f8d84ca13127cd50f6f6c8` or later.** All upstream Phase 1A/1B/1C and Phase 2 deliverables are landed. The review packet cited `63b9474cd`; implementers must verify `git rev-parse HEAD` before starting and update only symbol-local references if the branch has advanced again.

- Phase 1 (1A/1B/1C) is **shipped** on this branch. The compose loop calls `persist_compose_turn_async`, `RedactedToolRow`, `StatePayload`, `CompositionStateData`, `AuditOutcome`, `StaleComposeStateError`, and `AuditIntegrityError` — all already importable from `src/elspeth/web/sessions/protocol.py`, `_persist_payload.py`, and `elspeth.contracts.errors`. `_ToolOutcome` exists in `web/sessions/_persist_payload.py` (frozen, `freeze_fields(self, "call", "response")`); Phase 3 imports it, does NOT redefine. The `audit_access_log_table` is defined INERT in `web/sessions/models.py`; Task 11 wires the writer.
- Phase 2 (manifest-keyed dispatch + `Sensitive[T]` promotion wave + adequacy guard + F1–F6 follow-ups) is **shipped**. The Phase 3 plan's red tests against Phase 2 symbols now fail with `AssertionError` (wiring not yet present), not `ImportError`. Specifically, every Phase 2 surface this plan consumes is on disk:
  - `MANIFEST: Mapping[str, ToolRedaction]` in `web/composer/redaction.py` (38 entries — 10 type-driven + 28 declarative per `project_phase2_implementation_complete`).
  - `redact_tool_call_arguments(tool_name, decoded_args, telemetry=...)` in `web/composer/redaction.py`.
  - `redact_tool_call_response(tool_name, response, telemetry=...)` in `web/composer/redaction.py`.
  - `class RedactionTelemetry(Protocol)`, `NoopRedactionTelemetry`, and `OtelRedactionTelemetry` in `web/composer/redaction_telemetry.py`.
  - `_arg_error_payload(exc, tool_name) -> Mapping[str, Any]` in `web/composer/service.py` (F2 — module-tail helper preserving AST fingerprints; populates `validation_errors` field on ARG_ERROR `result_canonical` when `exc.__cause__` is a `pydantic.ValidationError`).
  - `canonicalize_pydantic_cause(exc)` in `web/composer/audit.py` (F2 — leak-safe `loc`/`msg`/`type` canonicalisation, `input`/`url`/`ctx` stripped).

  Phase 3 consumes these symbols verbatim; the walker dispatches through `MANIFEST[tool_name]` (spec §5.7.5); `_arg_error_payload` is the canonical ARG_ERROR factory invoked by both `dispatch_with_audit` and the `except ToolArgumentError` arm in `_compose_loop`.
- Phase 1 also shipped CL-PP-11 (concurrent multi-session writes against testcontainer Postgres — commit `eca88974`). Phase 3 does not re-author that test; it cites the existing test as the CL-PP-11 deliverable in the §11 done-when checklist.
- **Sequencing for Phase 3 execution:** the implementer starts on current `RC5.2`. No upstream phase is pending. Reds in Phase 3 tasks fail because the loop body is unwired, not because imports are unresolved.

---

## File Structure

### Files to modify (existing)

- `src/elspeth/web/composer/service.py` — `_compose_loop` body rewritten to the §5.2.1 shape; per-turn tool-call cap; sentinel-guarded `SessionServiceProtocol` dependency for compose-turn persistence; legacy `BufferingRecorder.add_message` drain inside the loop deleted; tool-result accumulation switched to `_ToolOutcome`.
- `src/elspeth/contracts/errors.py` (or a neutral `src/elspeth/contracts/audit.py` imported by it) — defines `FailedTurnMetadata` and gives route-visible `AuditIntegrityError` a typed nullable `failed_turn: FailedTurnMetadata | None = None` field or constructor argument; handlers branch on this typed field directly, never with `getattr`.
- `src/elspeth/web/composer/protocol.py` — `ComposerConvergenceError.capture` accepts the new `tool_call_cap_exceeded` reason code; `ComposerProgressEvent` reason-code enum gains `tool_call_cap_exceeded`; imports the neutral `FailedTurnMetadata`; `ComposerResult` and the partial-state carrier exceptions expose the persisted assistant message id / failed-turn metadata needed by the route layer; seam contract B docstrings at the current `protocol.py` `Does NOT depend on SessionService` text are updated to the new narrower dependency contract.
- `src/elspeth/web/app.py` — `ComposerServiceImpl` construction passes `session_service`; FastAPI registers handlers for `AuditIntegrityError`, `StaleComposeStateError`, and the dedicated audit-grade access-log write-failure exception.
- `src/elspeth/web/sessions/routes.py` — `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure` each add the `failed_turn` field; `GET /api/sessions/{sid}/messages` gains `include_tool_rows: bool = False` and emits an `audit_access_log` row when `include_tool_rows=true`.
- `src/elspeth/web/sessions/service.py` — adds `count_tool_responses_for_assistant(session_id, assistant_message_id) -> int` and `record_audit_grade_view(session_id, requesting_principal, request_path, query_args, ip_address) -> None`.
- `src/elspeth/web/sessions/protocol.py` — `SessionServiceProtocol` gains `count_tool_responses_for_assistant_async` and `record_audit_grade_view_async`; defines the dedicated `AuditAccessLogWriteError` route-boundary exception if Task 11 chooses that type.
- `src/elspeth/web/sessions/telemetry.py` — `_SessionsTelemetry` extended with `tool_call_cap_exceeded_total`, `audit_grade_view_total`, and `audit_access_log_write_failed_total`; `build_sessions_telemetry` wires all three in both fake and real branches.
- `src/elspeth/web/sessions/schemas.py` — response model for the messages endpoint gains `tool_call_id`, `parent_assistant_id`, `sequence_no` fields (Phase 1 already added the columns; Phase 3 exposes them on the API surface).
- `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-overview.md` — Task 14 below updates the overview to rev-5 wording (currently still says "revision 4" / `Sensitive[T]` in its spec, architecture, phase-table, and dependency prose).
- `tests/unit/web/composer/test_audit_failure_primacy.py` — exists at rev-3 baseline; extend in place for compose-loop route-stack propagation, do not recreate.
- `tests/integration/web/test_inv_audit_ahead_backward.py` — exists at rev-3 baseline; extend in place for cancellation/backward-direction coverage, do not recreate.
- `tests/unit/web/composer/conftest.py`, `tests/unit/web/conftest.py`, `tests/integration/web/conftest.py`, and `tests/property/web/composer/conftest.py` as needed — Task 0 either defines or cites every fixture/helper used by red tests; no red test may reference an undefined fixture.

### Files to create (new)

- `tests/unit/web/composer/test_compose_loop_tool_call_cap.py` — Step 0 unit tests.
- `tests/unit/web/composer/test_compose_loop_persistence.py` — Step 1/2/3 unit tests against in-memory SQLite via `create_session_engine` + `initialize_session_schema`.
- `tests/unit/web/sessions/test_count_tool_responses_for_assistant.py` — read helper.
- `tests/unit/web/sessions/test_record_audit_grade_view.py` — access-log write helper.
- `tests/unit/web/sessions/test_messages_route_include_tool_rows.py` — endpoint behaviour + access-log emission.
- `tests/unit/web/test_composer_exception_handlers.py` — FastAPI exception handler shape for `AuditIntegrityError`, `StaleComposeStateError`, and audit-grade access-log write failures.
- `tests/integration/web/test_compose_loop_failed_turn_field.py` — `failed_turn` response shape across all three `_handle_*` helpers.
- `tests/property/web/composer/test_compose_loop_invariants.py` — Hypothesis property test (§8.3).
- `tests/property/web/composer/strategies.py` — strategy contracts (§8.3.1) if not already in place.

Existing characterization tests in `tests/integration/pipeline/test_composer_llm_eval_characterization.py` are **extended in place** to author/refresh CL-PP-1..10b, 12, 13. CL-PP-11 (commit `eca88974`) is unchanged.

### Files NOT touched in Phase 3

- `src/elspeth/web/composer/redaction.py` — Phase 2 owns this. Phase 3 imports the walker entry points.
- `src/elspeth/web/composer/tools.py` — Phase 2's promotion wave touches it. Phase 3 does not.
- `src/elspeth/web/sessions/models.py` — Phase 1 owns schema. The `audit_access_log` table and the new `chat_messages` columns are already in place.
- `src/elspeth/web/frontend/**` — Phase 4 owns the recovery panel.

### Rev-6 architecture choice

Rev 5 keeps **service-owned compose-turn persistence**: `ComposerServiceImpl` can receive a `SessionServiceProtocol` handle and `_compose_loop` calls `persist_compose_turn_async(...)` directly. To avoid a fourth recurrence of signature-change blast radius, the constructor parameter is optional with a first-use guard rather than required keyword-only. Production app wiring must pass the real `SessionServiceImpl`; existing constructor-only tests that never exercise compose-turn persistence do not need churn. If a test or caller exercises the new persistence path without wiring the service, it fails loudly with `RuntimeError("sessions_service not wired")`. The implementer must update the existing composer seam documentation in `src/elspeth/web/composer/protocol.py` that currently says the composer does not depend on `SessionService`; after Phase 3 the narrower truth is: composer service depends on the `SessionServiceProtocol` persistence boundary for compose-turn audit rows, while route handlers still own user-message insertion, LLM-call sidecars, guided-flow drains, and HTTP response assembly.

---

## Ground rules

- **TDD throughout.** Every task ends in a failing test before any implementation lands. The reds are wiring-level (`AssertionError` / `AttributeError`), not import-level — Phase 2 symbols resolve at current `RC5.2` (see Dependency posture; the rev-1 "import-time RED against Phase 2 symbols" framing is superseded). Do not stub Phase 2 surfaces; they exist.
- **Real databases in tests.** `create_session_engine(..., poolclass=StaticPool)` + `initialize_session_schema()` for SQLite (per spec §8.6); testcontainer PostgreSQL for CL-PP-11 (already wired) and any new PostgreSQL-only assertion. Bare `metadata.create_all()` is banned. Mocking `persist_compose_turn` or any `_insert_*` helper is banned.
- **No defensive programming.** Tier-1 audit data crashes on anomaly. `AuditIntegrityError` is the canonical signal; it is in `TIER_1_ERRORS` and must not be caught by `except Exception`. The compose loop's `except Exception as tool_exc:` branch (the plugin-bug surface) **must** re-raise `AuditIntegrityError` unmodified — capture only via the existing `ComposerPluginCrashError.capture(...)` call site in `src/elspeth/web/composer/service.py`.
- **No legacy code.** The §5.2.1 shape supersedes route-layer persistence of compose-loop tool rows. Task 6 deletes or guards the relevant `routes.py:_persist_tool_invocations` compose/recompose call sites in the same PR; retained call sites must be non-compose-loop surfaces and explicitly justified.
- **Audit primacy.** Logging is permitted only on the unwind-audit-failed path (`AuditOutcome.unwind_audit_failed=True`) where the audit-system is the failing surface; everywhere else, telemetry counters carry the operational signal.
- **`from exc` chaining preserved.** Existing `ComposerPluginCrashError.capture` already sets `__cause__`; the rewrite must keep that chain intact when re-raising after the audit write.
- **Signature-change sweep required.** Any constructor parameter, dataclass field, protocol method, route dependency, or public response-model field added by this phase requires a whole-codebase `rg` sweep before commit. The task must either enumerate and update every call site or deliberately shape the change to tolerate existing callers (for example a defaulted dataclass field or a sentinel constructor dependency with first-use guard). Do not repeat the rev-4 failure mode where one named caller is fixed and adjacent construction sites are left to fail with `TypeError`.
- **Red-test scaffolding must be defined before use.** Every red test in this plan must reference only fixtures and helpers that are already defined in the repo or explicitly created by Task 0 before the red test is committed. For each fixture/helper, Task 0 names the definition path and the contract it satisfies. If a new red test needs a new fixture/helper, add it to Task 0 first or cite the existing `conftest.py` definition; otherwise the RED is invalid because it fails on test infrastructure rather than unimplemented production behavior.
- **Await-point cancellation sweep required.** Any new `await` point added to `_compose_loop` or its same-turn helper path must update the `st_cancellation_arrival_time` enum, Task 12c's injection-mechanism table, and the cancellation property/integration examples in the same commit. If the await is provably outside compose-turn atomicity or cancellation semantics, document that proof next to the await and in Task 12c. Do not let future async additions create unmodeled cancellation windows.
- **Frequent commits.** Each task ends with a commit; tasks with subtasks commit per subtask.

---

## Task 0: Test harness fixtures + compose-loop test driver

Close the quality-axis recurrence before any TDD task starts: red tests must fail for the behavior under test, not because fixtures or helper methods are missing.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (add `_run_one_turn_for_test` test driver)
- Modify: `tests/unit/web/composer/conftest.py`
- Modify: `tests/unit/web/conftest.py`
- Modify: `tests/integration/web/conftest.py`
- Modify or create: `tests/property/web/composer/conftest.py`

- [x] **Step 1: Add `_run_one_turn_for_test` as an explicit test-only driver.**

In `ComposerServiceImpl`, add a narrow helper used only by tests:

```python
def _run_one_turn_for_test(
    self,
    *,
    llm: Any | None = None,
    session_id: str | None = None,
    initial_state: Any | None = None,
    user_message_id: str | None = None,
) -> ComposeLoopTestResult:
    """Drive exactly one compose-loop turn for Phase 3 tests.

    Test-only helper: it bypasses HTTP route setup but exercises the
    same `_compose_loop` body, including `_require_sessions_service()`.
    Missing `sessions_service` must therefore fail with
    RuntimeError("sessions_service not wired"), not AttributeError or a
    constructor TypeError.
    """
```

Define `ComposeLoopTestResult` in the test module or as a private service-side dataclass if the service helper needs a structured return. It must expose only the fields the plan snippets assert: `assistant_message`, `tool_outcomes`, `persisted_assistant_row`, `persisted_assistant_tool_calls`, and `persisted_tool_row_content`. Do not make this helper a second implementation of the loop; it delegates into `_compose_loop` with a one-turn budget / fake LLM surface.

- [x] **Step 2: Define or cite every red-test fixture/helper before use.**

Add the following inventory to the named conftest files. If an existing fixture already exists under the cited path, reuse it and update only the docstring/contract if needed. If it does not exist, create it before any test that consumes it.

| Fixture/helper | Definition path | Contract |
|---|---|---|
| `build_test_sessions_service` | `tests/unit/web/composer/conftest.py` | Builds `SessionServiceImpl` with `create_session_engine("sqlite://", poolclass=StaticPool)` + `initialize_session_schema(engine)` and fake telemetry; no bare `metadata.create_all()`. |
| `composer_service_with_real_sessions` | `tests/unit/web/composer/conftest.py` | Returns `ComposerServiceImpl(..., sessions_service=build_test_sessions_service(...))` with real SQLite-backed sessions service. |
| `composer_service_without_sessions_service` | `tests/unit/web/composer/conftest.py` | Returns `ComposerServiceImpl(...)` without `sessions_service`, used only by the sentinel first-use guard test. |
| `fake_composer_service` | `tests/unit/web/composer/conftest.py` | Lightweight `ComposerServiceImpl` with fake catalog/settings and a wired real or protocol-faithful sessions service when the test reaches persistence. |
| `fake_llm_emitting_n_tool_calls` | `tests/unit/web/composer/conftest.py` | Factory `n -> fake LLM` whose first assistant response emits `n` tool calls and tracks `execute_tool_invocations`. |
| `fake_llm_two_tool_calls` | `tests/unit/web/composer/conftest.py` | Fake LLM for exactly two successful tool calls. |
| `fake_llm_three_tool_calls` | `tests/unit/web/composer/conftest.py` | Fake LLM for exactly three successful tool calls. |
| `fake_llm_tool_argument_error_on_second` | `tests/unit/web/composer/conftest.py` | Second tool raises `ToolArgumentError`; loop continues and records ARG_ERROR payload. |
| `fake_llm_runtime_error_on_second` | `tests/unit/web/composer/conftest.py` | Second tool raises `RuntimeError`; loop captures `ComposerPluginCrashError` and persists unwind rows first. |
| `fake_llm_with_sensitive_tool_call` | `tests/unit/web/composer/conftest.py` | Emits a tool call with arguments covered by Phase 2 manifest redaction. |
| `fake_llm_summarizer_active` | `tests/unit/web/composer/conftest.py` | Emits a tool response that exercises `redact_tool_call_response`. |
| `fake_llm_preflight_rewrites_content` | `tests/unit/web/composer/conftest.py` | Exposes `original_text`; runtime preflight mutates visible assistant content after raw capture. |
| `fake_llm_tool_call_with_no_content` | `tests/unit/web/composer/conftest.py` | Assistant message has `content=None`; persisted raw content must remain NULL. |
| `result_session_id` | `tests/unit/web/composer/conftest.py` | Session id used by `_run_one_turn_for_test` result assertions. |
| `sqlalchemy_event_listener` | `tests/unit/web/composer/conftest.py` | Counts begin/commit/rollback on the real SQLite test engine. |
| `add_message_spy` | `tests/unit/web/composer/conftest.py` | Records caller frames for legacy `add_message` invocations. |
| `inject_commit_OperationalError` | `tests/unit/web/composer/conftest.py` and re-export/cite from `tests/integration/web/conftest.py` | SQLAlchemy event hook that raises `OperationalError` on COMMIT and removes itself after the test. |
| `inject_IntegrityError_on_chat_messages` | `tests/unit/web/composer/conftest.py` | SQLAlchemy hook or monkeypatch that raises `IntegrityError` on assistant insert. |
| `inject_non_compose_loop_AuditIntegrityError` | `tests/unit/web/conftest.py` or `tests/integration/web/conftest.py` | Raises `AuditIntegrityError(failed_turn=None)` from a route-visible non-`_compose_loop` audit boundary to exercise the typed degraded handler body. |
| `inject_audit_access_log_write_failure` | `tests/unit/web/conftest.py` | Forces `record_audit_grade_view` write failure, increments `audit_access_log_write_failed_total`, and raises `AuditAccessLogWriteError`. |
| `test_client` | existing `tests/unit/web/conftest.py` or define there | FastAPI `TestClient` with app state exposing `sessions_service`. |
| `composer_test_client` | existing `tests/integration/web/composer/guided/conftest.py` for guided tests; define/cite in `tests/integration/web/conftest.py` for generic compose tests | FastAPI `TestClient` configured for compose routes and fake LLM injection. |
| `session_with_pending_compose_request` | `tests/integration/web/conftest.py` | Session fixture with owner, user message, and initial composition state ready for `/compose`. |
| `session_with_composer_state` | `tests/integration/web/conftest.py` | Session fixture used by failed-turn route tests. |
| `session_with_user_assistant_tool_rows` | `tests/unit/web/conftest.py` | Session containing user, assistant, and tool rows with monotonic `sequence_no`. |
| `populated_audit_db` | `tests/property/web/composer/conftest.py` or `tests/integration/web/conftest.py` | Real initialized engine populated through the compose loop with mixed tool successes/failures/cancellations. |

Run this preflight before committing Task 0:

```bash
rg -n "def (build_test_sessions_service|composer_service_with_real_sessions|composer_service_without_sessions_service|fake_composer_service|fake_llm_emitting_n_tool_calls|fake_llm_two_tool_calls|fake_llm_three_tool_calls|inject_commit_OperationalError|inject_non_compose_loop_AuditIntegrityError|inject_audit_access_log_write_failure|composer_test_client|session_with_pending_compose_request|populated_audit_db)" tests
```

Expected: every name above resolves to a fixture/helper definition or an explicitly cited existing fixture. If any red-test fixture/helper cannot be classified, stop and surface it to the operator before committing.

- [x] **Step 3: Commit.**

```bash
git commit -am "test(composer): define phase-3 compose-loop test harness fixtures (composer-progress-persistence phase 3)"
```

---

## Task 1: Verify the `_ToolOutcome` payload dataclass

The compose loop's per-iteration outcome record already exists at `src/elspeth/web/sessions/_persist_payload.py` (`_ToolOutcome`). Phase 3 imports it; it does not author or extend it unless a field is missing.

**Files:**
- Read-only: `src/elspeth/web/sessions/_persist_payload.py`

- [x] **Step 1: Confirm the field list matches §5.2.1's loop body.**

Required fields per spec §5.2.1: `call: Any`, `response: Any`, `error_class: str | None`, `error_message: str | None`, `pre_version: int`, `post_version: int`. Verify by reading `_ToolOutcome` directly in `_persist_payload.py`. The dataclass is `frozen=True, slots=True` with a `freeze_fields(self, "call", "response")` post-init guard.

If every field matches, this task is a verification-only no-op and is closed by commenting in the PR description: "Task 1 verified: `_ToolOutcome` fields match spec §5.2.1; no Phase 3 changes required to `_persist_payload.py`." No commit.

If a field is missing or has the wrong type, treat this as a Phase 1 hygiene defect: file a Filigree issue with reproducer steps, link from the PR, and add the missing field on this branch with a per-field test in `tests/unit/web/sessions/test_persist_payload.py` (extending the existing file). Do not add fields that §5.2.1 does not call for.

- [x] **Step 2: Commit (only if Step 1 added a field).**

Commit message: `fix(sessions): add missing _ToolOutcome.<field> per spec §5.2.1 (composer-progress-persistence phase 3)`.

Verification result: `_ToolOutcome` already exposes `call`, `response`, `error_class`, `error_message`, `pre_version`, and `post_version`; no payload change was required.

---

## Task 2: Per-turn tool-call cap configuration + reason code

The §1.4 NFR caps tool calls per assistant turn at 16 (default), env-tunable. The cap raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")` **before** any tool execution. Step 0 of the §5.2.1 loop body depends on this.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (ComposerServiceImpl constructor: accept and store `max_tool_calls_per_turn`)
- Modify: `src/elspeth/web/composer/protocol.py` (`ComposerProgressEvent` reason-code enum gains `tool_call_cap_exceeded`; `ComposerConvergenceError.capture` accepts and propagates the new reason)
- Modify: `src/elspeth/web/sessions/telemetry.py` (`_SessionsTelemetry` gains `tool_call_cap_exceeded_total`; `build_sessions_telemetry` wires it in both fake and real branches)
- Modify: `src/elspeth/config.py` and the matching Settings→Runtime contract (see `config-contracts-guide` skill) for the env-tunable `MAX_TOOL_CALLS_PER_TURN`
- Create: `tests/unit/web/composer/test_compose_loop_tool_call_cap.py`

- [x] **Step 1: Write the failing red test.**

In `tests/unit/web/composer/test_compose_loop_tool_call_cap.py`:

```python
"""Tests for the per-turn tool-call cap (spec §1.4 NFR / §5.2.1 Step 0).

Drives the compose loop with an LLM that emits more tool calls in one
assistant turn than the cap allows. The loop must raise
ComposerConvergenceError(reason="tool_call_cap_exceeded") BEFORE any
tool execution, and increment composer.tool_call_cap_exceeded_total.
"""
from __future__ import annotations

import pytest

from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerProgressEvent,
)
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value


def test_cap_exceeded_raises_before_any_tool_execution(
    fake_composer_service, fake_llm_emitting_n_tool_calls
):
    # Cap is 16 by default; LLM emits 17.
    fake_llm = fake_llm_emitting_n_tool_calls(n=17)
    fake_composer_service._max_tool_calls_per_turn = 16
    fake_composer_service._llm = fake_llm

    with pytest.raises(ComposerConvergenceError) as excinfo:
        fake_composer_service._run_one_turn_for_test()

    # The reason code is the new tool_call_cap_exceeded variant.
    assert excinfo.value.reason == ComposerProgressEvent.Reason.tool_call_cap_exceeded
    assert excinfo.value.evidence["observed"] == 17
    assert excinfo.value.evidence["cap"] == 16

    # No tool execution attempted — execute_tool must not have been
    # called at all. fake_llm tracks calls into execute_tool via a
    # spy installed in the conftest fixture.
    assert fake_llm.execute_tool_invocations == 0


def test_cap_exceeded_increments_counter(
    fake_composer_service, fake_llm_emitting_n_tool_calls
):
    telemetry = build_sessions_telemetry()  # fake counters
    fake_composer_service._telemetry = telemetry
    fake_composer_service._max_tool_calls_per_turn = 16
    fake_composer_service._llm = fake_llm_emitting_n_tool_calls(n=17)

    with pytest.raises(ComposerConvergenceError):
        fake_composer_service._run_one_turn_for_test()

    assert observed_value(telemetry.tool_call_cap_exceeded_total) == 1


def test_cap_not_exceeded_does_not_increment(
    fake_composer_service, fake_llm_emitting_n_tool_calls
):
    telemetry = build_sessions_telemetry()
    fake_composer_service._telemetry = telemetry
    fake_composer_service._max_tool_calls_per_turn = 16
    fake_composer_service._llm = fake_llm_emitting_n_tool_calls(n=16)

    fake_composer_service._run_one_turn_for_test()  # no raise

    assert observed_value(telemetry.tool_call_cap_exceeded_total) == 0
```

Run the test:

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_tool_call_cap.py -v
```

Expected: FAIL — `ComposerProgressEvent.Reason.tool_call_cap_exceeded` does not exist; `_max_tool_calls_per_turn` is not an attribute on the service; `tool_call_cap_exceeded_total` is not a field on `_SessionsTelemetry`.

- [x] **Step 2: Add the reason code + telemetry counter.**

In `src/elspeth/web/composer/protocol.py`, extend `ComposerProgressEvent.Reason` (or its current `Literal[...]` equivalent — verify against the file's current shape) with `"tool_call_cap_exceeded"`. Update `ComposerConvergenceError.capture` to accept the reason and `evidence: Mapping[str, Any]` extension carrying `observed` and `cap`.

In `src/elspeth/web/sessions/telemetry.py` (`_SessionsTelemetry`), append the field after `tool_row_integrity_violation_total`:

```python
tool_call_cap_exceeded_total: _Counter
```

In `build_sessions_telemetry` (both `meter is None` and real branches), add the matching entry:

```python
tool_call_cap_exceeded_total=_FakeCounter(),
# existing fake counters remain unchanged
tool_call_cap_exceeded_total=meter.create_counter(
    "composer.tool_call_cap_exceeded_total"
),
```

Update the `_SessionsTelemetry` docstring comment to drop the "Phase 3 (compose loop + audit-grade view) adds" forward-looking line, since this PR is delivering it.

- [x] **Step 3: Wire `_max_tool_calls_per_turn` into the composer service.**

Add `max_tool_calls_per_turn: int = 16` to `ComposerServiceImpl.__init__`, store as `self._max_tool_calls_per_turn`. Source the value from runtime config via the existing `from_settings(...)` mapping (see `config-contracts-guide` skill): add `MAX_TOOL_CALLS_PER_TURN: int = 16` to the Composer settings dataclass, route it through the contract, and confirm the `check_contracts` script passes:

```bash
.venv/bin/python -m scripts.check_contracts
```

- [x] **Step 4: Re-run the test to verify GREEN.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_tool_call_cap.py -v
```

Expected: PASS.

- [x] **Step 5: Commit.**

```bash
git add -p src/elspeth/web/composer/protocol.py \
            src/elspeth/web/composer/service.py \
            src/elspeth/web/sessions/telemetry.py \
            src/elspeth/config.py \
            src/elspeth/contracts/config/composer.py \
            tests/unit/web/composer/test_compose_loop_tool_call_cap.py
git commit -m "feat(composer): per-turn tool-call cap + tool_call_cap_exceeded reason code (composer-progress-persistence phase 3)"
```

---

## Task 3: Rewrite `_compose_loop` body — Step 0 (cap enforcement)

Implement Step 0 of the §5.2.1 loop body inside `_compose_loop`. The check runs **before** the for-loop over `assistant_message.tool_calls` opens; no tool execution happens for the over-cap turn.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` method body, after the LLM call returns)

- [x] **Step 1: Read the current `_compose_loop` body.**

Read `web/composer/service.py` at the `async def _compose_loop` definition. Identify where `assistant_message` becomes available after the LLM call and where the existing per-tool for-loop begins. Step 0 sits between them.

- [x] **Step 2: Insert the Step 0 check.**

Immediately after `assistant_message` is bound and before any iteration over `assistant_message.tool_calls` begins, insert:

```python
# Step 0 — enforce the per-turn tool-call cap (spec §1.4 NFR / §5.2.1).
# RSK-13: bounds the cost of a single turn that a misbehaving LLM
# might inflate. Raises BEFORE any tool execution; no DB writes for
# the over-cap turn; the existing _handle_convergence_error path
# downstream emits partial_state captured pre-cap.
if len(assistant_message.tool_calls) > self._max_tool_calls_per_turn:
    self._telemetry.tool_call_cap_exceeded_total.add(1)
    raise ComposerConvergenceError.capture(
        state,
        reason=ComposerProgressEvent.Reason.tool_call_cap_exceeded,
        evidence={
            "observed": len(assistant_message.tool_calls),
            "cap": self._max_tool_calls_per_turn,
        },
        initial_version=initial_version,
    )
```

Telemetry increments BEFORE the raise so the counter is incremented even if the exception path is later caught and re-raised differently by `_handle_convergence_error`.

- [x] **Step 3: Re-run Task 2's tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_tool_call_cap.py -v
```

Expected: PASS (this fully closes the cap-exceeded surface).

- [x] **Step 4: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 0 — enforce per-turn tool-call cap (composer-progress-persistence phase 3)"
```

---

## Task 4: Rewrite `_compose_loop` body — Step 1 (tool-execution accumulation)

Convert the existing per-tool-call execution into the §5.2.1 Step 1 shape: accumulate `_ToolOutcome` records, distinguish ToolArgumentError / interpreter-invariant / plugin-bug arms, capture `plugin_crash` without re-raising, and break out of the for-loop on a plugin crash so Step 2 still runs.

The loop body must also track two load-bearing variables for Step 2:

1. **`current_state_id`** — the state id observed BEFORE the LLM call. This is the `expected_current_state_id` argument to `persist_compose_turn_async` (§5.2.1 line 1498). Capture it as a local at the top of the turn (before the LLM call) so it survives the `await` and is passed into Step 2 unchanged.

2. **`derived_from_state_id=None` for per-tool `StatePayload`s** — Phase 1 allocates composition-state IDs inside `persist_compose_turn` under the held write lock, so the async loop does not know the predecessor state ID for a state row that has not been inserted yet. Do not add a `pre_state_id_for(...)` helper in Phase 3; inline `derived_from_state_id=None` with a load-bearing comment citing spec §5.7.1 and `_insert_composition_state`'s version-ordering lineage.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` body)

- [x] **Step 1: Write the failing red test.**

Create `tests/unit/web/composer/test_compose_loop_persistence.py`:

```python
"""Compose-loop Step 1/2/3 unit tests (spec §5.2.1).

These tests drive the loop against an in-memory SQLite SessionsService
built via create_session_engine + initialize_session_schema (per spec
§8.6 — bare metadata.create_all is banned for this surface). They assert
the §5.5 failure-mode rows 1, 2, and 4 individually.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from elspeth.web.sessions.engine import create_session_engine, initialize_session_schema
from sqlalchemy.pool import StaticPool


@pytest.fixture
def sessions_service():
    engine = create_session_engine("sqlite://", poolclass=StaticPool)
    initialize_session_schema(engine)
    return build_test_sessions_service(engine=engine)


def test_step1_three_tools_all_succeed_accumulates_three_outcomes(
    composer_service_with_real_sessions, fake_llm_three_tool_calls
):
    """Step 1 happy path — three successful tools produce three outcomes
    with response set, error_class None, and post_version tracking."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_three_tool_calls
    )
    outcomes = result.tool_outcomes_for_assertion  # test hook
    assert len(outcomes) == 3
    assert all(o.error_class is None for o in outcomes)
    assert all(o.response is not None for o in outcomes)
    # post_version monotonic non-decreasing across outcomes:
    assert outcomes[0].post_version <= outcomes[1].post_version <= outcomes[2].post_version


def test_step1_tool_argument_error_continues_loop(
    composer_service_with_real_sessions, fake_llm_tool_argument_error_on_second
):
    """ToolArgumentError on call 2 of 3 — outcome 2 has
    error_class='ToolArgumentError'; loop continues to call 3."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_tool_argument_error_on_second
    )
    outcomes = result.tool_outcomes_for_assertion
    assert len(outcomes) == 3
    assert outcomes[0].error_class is None
    assert outcomes[1].error_class == "ToolArgumentError"
    assert outcomes[2].error_class is None


def test_step1_assertion_error_reraises_before_persist(
    composer_service_with_real_sessions,
    fake_llm_assertion_error_on_second,
    sessions_service,
    result_session_id,
):
    """AssertionError on call 2 of 3 — re-raised BEFORE Step 2 runs.
    No DB writes happen for the failed turn; route-helper partial-state
    persist covers the snapshot."""
    with pytest.raises(AssertionError):
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_assertion_error_on_second
        )
    # No chat_messages row for this turn — assistant or tool:
    with sessions_service._engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id FROM chat_messages "
                "WHERE session_id = :session_id AND role IN ('assistant', 'tool')"
            ),
            {"session_id": str(result_session_id)},
        ).fetchall()
    assert rows == []


def test_step1_plugin_bug_captures_crash_breaks_loop(
    composer_service_with_real_sessions, fake_llm_runtime_error_on_second
):
    """RuntimeError on call 2 of 3 — outcome 2 has error_class='RuntimeError';
    loop breaks (no call 3); plugin_crash is captured for raise-after-Step-2."""
    with pytest.raises(ComposerPluginCrashError) as excinfo:
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_runtime_error_on_second
        )
    # The audit write DID happen — assistant + 2 tool rows visible:
    # (one normal outcome 1, one error outcome 2; outcome 3 never executed)
    # The captured exception preserves the original cause chain.
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, RuntimeError)
```

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py::test_step1_three_tools_all_succeed_accumulates_three_outcomes -v
```

Expected: FAIL — the loop currently drains to `BufferingRecorder`, not `_ToolOutcome[]`. The test hook `tool_outcomes_for_assertion` does not exist.

- [x] **Step 2: Refactor the per-tool for-loop body to accumulate `_ToolOutcome`.**

In `_compose_loop`, replace the current per-tool record/drain pattern with the §5.2.1 Step 1 shape:

```python
# Step 1 — execute every tool call in async land, accumulating
# _ToolOutcome records. No audit work writes yet; cancellation here
# is safe (the LLM response is in memory, the DB is unchanged).
tool_outcomes: list[_ToolOutcome] = []
plugin_crash: ComposerPluginCrashError | None = None
pre_state_id = current_state_id  # captured before LLM call; load-bearing for Step 2

for tool_call in assistant_message.tool_calls:
    pre_version = state.version
    try:
        response = await execute_tool(tool_call, state)
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=response,
            error_class=None,
            error_message=None,
            pre_version=pre_version,
            post_version=state.version,
        ))
        # If the tool advanced state, the previous state row's id is
        # the new pre_state_id for the next iteration. State id
        # bookkeeping is loop-local — no SessionsService roundtrip
        # because persist_compose_turn re-reads MAX(version) under
        # the held write lock anyway (spec §5.7.1).
        if state.version > pre_version:
            # The new state id will be allocated inside
            # persist_compose_turn (Step 2); for now retain the
            # pre-call state id as the predecessor for this outcome's
            # StatePayload.derived_from_state_id. See Step 2 below.
            pass
    except ToolArgumentError as exc:
        # Tier-3 boundary signal. The loop continues; the LLM gets
        # the error tool row as feedback. Existing contract preserved
        # by the current ToolArgumentError arm and ToolArgumentError
        # protocol docstring.
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=None,
            error_class="ToolArgumentError",
            error_message=str(exc),
            pre_version=pre_version,
            post_version=state.version,
        ))
    except (AssertionError, MemoryError, RecursionError, SystemError):
        # Interpreter-level invariant violations. Not recoverable, not
        # wrappable. Re-raise BEFORE Step 2 runs; route-helper
        # partial-state persist runs from the current _handle_* helpers.
        raise
    except AuditIntegrityError:
        # Tier-1 invariant raised by execute_tool or downstream.
        # Re-raise unmodified — registered in TIER_1_ERRORS so this
        # except Exception block below CANNOT swallow it (the bare
        # AuditIntegrityError clause runs first).
        raise
    except Exception as tool_exc:
        # Plugin-bug surface. Current service.py ComposerPluginCrashError.capture
        # call-site pattern: wrap with
        # ComposerPluginCrashError.capture, break out, raise AFTER
        # the Step-2 audit write. The audit row for the crashing tool
        # is committed in the same transaction as the surviving tool
        # rows; the crash signal propagates only after audit is durable.
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=None,
            error_class=type(tool_exc).__name__,
            error_message=str(tool_exc),
            pre_version=pre_version,
            post_version=state.version,
        ))
        plugin_crash = ComposerPluginCrashError.capture(
            tool_exc,
            state=state,
            initial_version=initial_version,
        )
        break
```

**Critical:** the `except AuditIntegrityError:` clause is positioned BEFORE the catch-all `except Exception as tool_exc:` arm. `AuditIntegrityError` is decorated with `@tier_1_error` (registered in `TIER_1_ERRORS`), but `except Exception` will still catch it without the dedicated clause; the explicit re-raise is the offensive-programming guard. CLAUDE.md no-defensive-programming: do not wrap this in `except (AuditIntegrityError,) as exc: raise`. The bare `raise` is correct.

Delete the now-orphaned `BufferingRecorder.add_message` calls within the loop body. The recorder remains in scope for non-tool events (LLM call telemetry, request-level audit), but `add_message` is no longer the path for per-tool persistence.

- [x] **Step 3: Re-run the Step 1 tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v -k "step1"
```

Expected: PASS for `test_step1_three_tools_all_succeed_accumulates_three_outcomes`, `test_step1_tool_argument_error_continues_loop`, and `test_step1_assertion_error_reraises_before_persist`. The plugin-bug test (`test_step1_plugin_bug_captures_crash_breaks_loop`) still fails because Step 2 isn't wired — it expects the audit row to exist, which only happens after Step 2 lands.

- [x] **Step 4: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 1 — _ToolOutcome accumulation + crash capture (composer-progress-persistence phase 3)"
```

---

## Task 5: `_compose_loop` Step 2 — async-side redaction via the Phase 2 manifest

Build the redacted assistant `tool_calls` tuple and the `RedactedToolRow` tuple in async land, calling Phase 2's `redact_tool_call_arguments` and `redact_tool_call_response`. The walker takes the already-decoded arguments dict and a `RedactionTelemetry` instance; the manifest is module-global.

**Dependency:** Phase 2 must be merged. Until then this task's tests fail at import time on `redact_tool_call_arguments` / `redact_tool_call_response`. That is the expected red.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` body, after the Task 4 Step 1 block)

- [x] **Step 1: Write the failing red test.**

Append to `tests/unit/web/composer/test_compose_loop_persistence.py`:

```python
from elspeth.web.composer.redaction import (
    MANIFEST,
    RedactionTelemetry,
    redact_tool_call_arguments,
    redact_tool_call_response,
)


def test_step2_redacts_via_manifest_walker(
    composer_service_with_real_sessions, fake_llm_with_sensitive_tool_call
):
    """Step 2 — redact_tool_call_arguments produces the canonical
    redacted form before persist_compose_turn_async fires. The persisted
    chat_messages.tool_calls JSON is byte-identical to the walker output."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_with_sensitive_tool_call
    )
    # The persisted assistant row's tool_calls JSON matches the walker.
    expected = tuple(
        redact_tool_call_arguments(
            tc.function.name,
            decoded_args[tc.id],
            telemetry=composer_service_with_real_sessions._redaction_telemetry,
        )
        for tc in result.assistant_message.tool_calls
    )
    persisted = result.persisted_assistant_tool_calls
    assert persisted == expected


def test_step2_redacts_response_with_summarizer(
    composer_service_with_real_sessions, fake_llm_summarizer_active
):
    """The redacted tool-row content matches redact_tool_call_response."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_summarizer_active
    )
    expected_content = redact_tool_call_response(
        tool_name=result.tool_outcomes[0].call.function.name,
        response=result.tool_outcomes[0].response,
        telemetry=composer_service_with_real_sessions._redaction_telemetry,
    )
    assert result.persisted_tool_row_content[0] == expected_content
```

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py::test_step2_redacts_via_manifest_walker -v
```

Expected (rev-3 update — Phase 2 is shipped on `RC5.2`): FAIL with `AssertionError` on `assert persisted == expected` (or similar wiring assertion). Phase 2 imports resolve cleanly; the red is the loop body not yet calling the walker. (Rev-1 said "ImportError until Phase 2 merges"; that expectation is superseded.)

- [x] **Step 2: Wire the redaction step in `_compose_loop`.**

After the Task 4 Step 1 for-loop, before the (Task 6) dispatch:

```python
# Step 2a — redact in async land. All redaction is pure / non-blocking;
# building the redacted payload here keeps the sync worker's wall-clock
# window narrow. Phase 2 owns the walker; Phase 3 calls it via tool name
# (no lookup_tool_class helper — rev-5 §5.7.5 retired that pattern).
redacted_assistant_tool_calls = tuple(
    redact_tool_call_arguments(
        tc.function.name,
        decoded_args[tc.id],  # M3: arguments parsed once at LLM-call time
        telemetry=self._redaction_telemetry,
    )
    for tc in assistant_message.tool_calls
)

redacted_tool_rows = tuple(
    RedactedToolRow(
        tool_call_id=outcome.call.id,
        content=_serialize_response_via_walker(
            outcome, telemetry=self._redaction_telemetry
        ),
        composition_state_payload=(
            StatePayload(
                data=CompositionStateData(
                    source=state.source,
                    nodes=state.nodes,
                    edges=state.edges,
                    outputs=state.outputs,
                    metadata_=state.metadata_,
                    is_valid=state.is_valid,
                    validation_errors=state.validation_errors,
                    composer_meta=state.composer_meta,
                ),
                # Phase 1 inserts composition state rows inside
                # persist_compose_turn under the session write lock and
                # re-derives lineage from per-session version ordering
                # when derived_from_state_id is None (spec §5.7.1).
                # The async loop deliberately does not fabricate a
                # predecessor state id for a row that has not been
                # allocated yet.
                derived_from_state_id=None,
            )
            if outcome.post_version > outcome.pre_version
            else None
        ),
    )
    for outcome in tool_outcomes
)
```

Where `_serialize_response_via_walker` is a private helper on `ComposerServiceImpl` that calls `redact_tool_call_response` on a success outcome and produces the canonical error-tool-row JSON on a failure outcome:

```python
def _serialize_response_via_walker(
    self,
    outcome: _ToolOutcome,
    *,
    telemetry: RedactionTelemetry,
) -> str:
    if outcome.error_class is None:
        redacted = redact_tool_call_response(
            tool_name=outcome.call.function.name,
            response=outcome.response,
            telemetry=telemetry,
        )
        return canonical_json(redacted)  # existing canonical helper
    # Error outcome — the LLM sees a structured error response; the
    # persisted content is the same shape. No redaction needed for the
    # error class/message (no sensitive payload).
    return canonical_json({
        "error_class": outcome.error_class,
        "error_message": outcome.error_message,
    })
```

Do not introduce a module-level `pre_state_id_for(...)` helper in Phase 3. It would be a premature abstraction whose only legal return value is `None`. The inline comment above is load-bearing: it documents that `None` is intentional and delegated to Phase 1's version-ordering lineage, not a missing predecessor lookup.

- [x] **Step 3: Re-run the Step 2 tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v -k "step2"
```

Expected: PASS. (Phase 2 is shipped on `RC5.2`; rev-1's "PASS once Phase 2 is in" is superseded — see Dependency posture.)

- [x] **Step 4: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 2 — async-side manifest redaction (composer-progress-persistence phase 3)"
```

---

## Task 6: `_compose_loop` Step 2 — single sync dispatch via `persist_compose_turn_async`

Replace route-layer persistence of compose-loop tool rows with a single sentinel-guarded `await sessions_service.persist_compose_turn_async(...)` call per turn. Rev-5 baseline check: the legacy helper is `src/elspeth/web/sessions/routes.py:_persist_tool_invocations`, not a function inside `_compose_loop`. Atomic cutover means compose/recompose success and compose-loop carrier exceptions must not also drain the same `tool_invocations` through the route helper after `_compose_loop` has committed them.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` body, after Task 5; add the sentinel-guarded `SessionServiceProtocol` dependency, the single `persist_compose_turn_async` dispatch, and stop returning compose-loop tool rows for route-layer persistence)
- Modify: `src/elspeth/web/composer/protocol.py` (`ComposerResult` / carrier docstrings updated so `tool_invocations` is no longer the route persistence path for compose-loop rows after Phase 3; update seam contract B docstrings at the current text matching `Does NOT depend on SessionService` and `depend on SessionService (seam contract B)` to reflect the new `SessionServiceProtocol` dependency)
- Modify: `src/elspeth/web/composer/audit.py` (`BufferingRecorder` docstring/comment updated: post-Phase-3 tool rows must not use this route-layer drain path; LLM/chat-turn audit buffers remain valid)
- Modify: `src/elspeth/web/sessions/routes.py` (delete/guard compose/recompose `_persist_tool_invocations` call sites listed in Step 3; retain non-compose-loop guided call sites)

- [x] **Step 1: Write the failing red test.**

Append to `tests/unit/web/composer/test_compose_loop_persistence.py`:

```python
def test_step2_dispatches_one_persist_compose_turn_async_per_turn(
    composer_service_with_real_sessions, fake_llm_two_tool_calls,
    sqlalchemy_event_listener,
):
    """Exactly one BEGIN/COMMIT pair per turn — verified via SQLAlchemy
    event listener counting transaction lifecycle events."""
    composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_two_tool_calls
    )
    assert sqlalchemy_event_listener.begin_count == 1
    assert sqlalchemy_event_listener.commit_count == 1
    assert sqlalchemy_event_listener.rollback_count == 0


def test_step2_passes_raw_content_for_B2_attribution(
    composer_service_with_real_sessions, fake_llm_preflight_rewrites_content,
):
    """The pre-redaction LLM output is passed as raw_content; the
    redacted form is passed as assistant_content. B2 audit-attribution
    is preserved when runtime preflight rewrites the visible content."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_preflight_rewrites_content
    )
    persisted_assistant = result.persisted_assistant_row
    assert persisted_assistant.raw_content == fake_llm_preflight_rewrites_content.original_text
    assert persisted_assistant.content != persisted_assistant.raw_content


def test_step2_preserves_absent_raw_content_as_none(
    composer_service_with_real_sessions, fake_llm_tool_call_with_no_content,
):
    """A missing LLM assistant content field is evidence. The audit
    raw_content column stays NULL; it is not fabricated as an empty string."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_tool_call_with_no_content
    )
    persisted_assistant = result.persisted_assistant_row
    assert persisted_assistant.raw_content is None


def test_step2_does_not_call_legacy_add_message_inside_loop(
    composer_service_with_real_sessions, fake_llm_two_tool_calls,
    add_message_spy,
):
    """The legacy BufferingRecorder.add_message drain path inside
    _compose_loop is deleted. add_message is still called for
    non-tool surfaces (system messages, user turns) but never from
    inside _compose_loop's per-turn body."""
    composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_two_tool_calls
    )
    # add_message_spy records callers via inspect.stack(); none should
    # originate inside _compose_loop frames.
    callers = add_message_spy.caller_frames
    assert "_compose_loop" not in {frame.function for frame in callers}
```

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v -k "step2_dispatches or raw_content or legacy"
```

Expected: FAIL — `persist_compose_turn_async` is not yet called from `_compose_loop`; the legacy `add_message` drain path is still present.

- [x] **Step 2: Capture `raw_content` before runtime preflight rewrites assistant content without fabricating empty strings.**

Spec §5.2.1 line 1489 sets `raw_content=raw_assistant_content`. The "raw" is the pre-redaction LLM output; existing routes already pass `raw_content=result.raw_assistant_content` to `add_message` at the route layer. Inside `_compose_loop`, the equivalent value must be captured **before** any runtime preflight rewrite mutates `assistant_message.content`.

Read `web/composer/service.py` around the LLM-call site to confirm the precise location where `assistant_message.content` is first available. Bind:

```python
raw_assistant_content = assistant_message.content
# (any subsequent preflight rewrites mutate assistant_message.content
# but raw_assistant_content retains the original. None means the LLM
# response omitted content; do not collapse that evidence to "".)
```

Hold the variable through Step 1's await boundary so it is available at Step 2 dispatch time. The `chat_messages.content` column remains non-null, so `assistant_content=assistant_message.content or ""` is acceptable for the visible content column only. The audit-attribution field `raw_content` must receive `raw_assistant_content` unchanged (`None` stays `None`).

- [x] **Step 2b: Wire `SessionServiceProtocol` into `ComposerServiceImpl`.**

Because rev 5 chooses service-owned compose-turn persistence without a broad constructor-call-site migration, add a sentinel-default `sessions_service: SessionServiceProtocol | None = None` dependency and a first-use guard. Do **not** add a required keyword-only parameter; current tests and helper factories construct `ComposerServiceImpl(...)` in many non-persistence scenarios. Those are not legacy callers, and they should not fail at construction time.

```python
class ComposerServiceImpl:
    def __init__(
        self,
        catalog: CatalogService,
        settings: ComposerSettings,
        *,
        sessions_service: SessionServiceProtocol | None = None,
        session_engine: Engine | None = None,
        secret_service: Any | None = None,
        runtime_preflight_coordinator: RuntimePreflightCoordinator | None = None,
    ) -> None:
        self._sessions_service = sessions_service
        self._session_engine = session_engine
        # existing constructor assignments remain unchanged

    def _require_sessions_service(self) -> SessionServiceProtocol:
        if self._sessions_service is None:
            raise RuntimeError("sessions_service not wired")
        return self._sessions_service
```

Every `_compose_loop` access uses `self._require_sessions_service()` immediately before the persistence call, then calls the returned protocol object. The guard is offensive-programming-compliant: missing wiring crashes loudly and informatively at the persistence boundary, not at unrelated constructor-only test setup.

Add a red test in `tests/unit/web/composer/test_compose_loop_persistence.py`:

```python
def test_persistence_path_without_sessions_service_crashes_loudly(
    composer_service_without_sessions_service,
    fake_llm_two_tool_calls,
):
    """Constructor-only callers may omit sessions_service, but the
    compose-turn persistence path requires it and fails at first use."""
    with pytest.raises(RuntimeError, match="sessions_service not wired"):
        composer_service_without_sessions_service._run_one_turn_for_test(
            llm=fake_llm_two_tool_calls
        )
```

Update `src/elspeth/web/app.py:create_app` so the existing `session_service = SessionServiceImpl(...)` object is passed into `ComposerServiceImpl(...)`. Tests for app creation must assert `app.state.composer_service` has the service dependency wired without creating a second `SessionServiceImpl`.

Perform the required signature-change sweep before committing:

```bash
rg -n "ComposerServiceImpl\\(" src tests
! rg -n "Does NOT depend on SessionService|depend on SessionService \\(seam contract B\\)" \
  src/elspeth/web/composer/protocol.py
```

The `ComposerServiceImpl(` sweep is not an instruction to edit every caller. Its acceptance criterion is that existing callers either (a) are production/app wiring and pass `sessions_service=...`, (b) are persistence-path tests and use a fixture that wires a real or protocol-faithful service, or (c) are constructor-only/non-persistence tests and rely on the sentinel default intentionally. If a caller cannot be classified into (a), (b), or (c), stop and surface it to the operator before committing. The protocol docstring sweep must be closed by editing `src/elspeth/web/composer/protocol.py` seam contract B text so it no longer claims the composer has no `SessionService` dependency.

- [x] **Step 2c: Make persisted assistant ids available to the route layer mechanically.**

Define a small immutable metadata type in a neutral contracts module, then thread it through the composer result/carrier surface. Do not define this type in `web/composer/protocol.py` and import it from `contracts.errors`; that would invert the dependency. The allowed shapes are either `src/elspeth/contracts/errors.py:FailedTurnMetadata` or `src/elspeth/contracts/audit.py:FailedTurnMetadata` imported by both `contracts.errors` and `web/composer/protocol.py`:

```python
@dataclass(frozen=True, slots=True)
class FailedTurnMetadata:
    assistant_message_id: str | None
    tool_calls_attempted: int
    tool_responses_persisted: int | None = None


@dataclass(frozen=True, slots=True)
class ComposerResult:
    # existing fields remain unchanged
    persisted_assistant_message_id: str | None = None
    persisted_tool_call_turn: bool = False
```

Add `failed_turn: FailedTurnMetadata | None` to `ComposerConvergenceError`, `ComposerPluginCrashError`, and `ComposerRuntimePreflightError` capture paths. The field is populated in `_compose_loop` immediately after `audit_outcome` returns:

```python
failed_turn = FailedTurnMetadata(
    assistant_message_id=audit_outcome.assistant_id,
    tool_calls_attempted=len(assistant_message.tool_calls or ()),
)
```

Add `failed_turn: FailedTurnMetadata | None = None` to route-visible `AuditIntegrityError` in `src/elspeth/contracts/errors.py` or provide an equivalent constructor/factory. `None` is a typed known state meaning "this AuditIntegrityError originated outside `_compose_loop`'s catch-and-annotate scope"; it is not an attribute-absence fallback. Any `AuditIntegrityError` raised by `persist_compose_turn_async` and caught by `_compose_loop` is annotated/re-raised with `FailedTurnMetadata` before it can reach the route stack. Existing non-compose-loop raise sites may leave `failed_turn=None`, and the FastAPI handler emits the documented degraded body for that typed state.

For plugin-crash and runtime-preflight carriers raised after the commit, attach this `failed_turn` object to the captured exception before raising it. For successful terminal results, set `ComposerResult.persisted_assistant_message_id=audit_outcome.assistant_id` and `persisted_tool_call_turn=True` when the turn included tool calls. Route code uses these fields as the concrete assistant-id source and to decide whether terminal `add_message("assistant", ...)` is still responsible for storing the final no-tool answer.

- [x] **Step 3: Replace the legacy persistence with the single dispatch and route cutover.**

After Task 5's redaction step, insert:

```python
# Step 2b — single sync dispatch via the protocol async wrapper.
# The async wrapper opens an asyncio.shield-wrapped worker thread and
# runs the sync primitive under it (commit-wins cancellation contract,
# §5.2.2). Calling self._sessions_service.persist_compose_turn directly
# from this coroutine raises RuntimeError via the async-loop guard.
sessions_service = self._require_sessions_service()
audit_outcome = await sessions_service.persist_compose_turn_async(
    session_id=session_id,
    assistant_content=assistant_message.content or "",
    raw_content=raw_assistant_content,
    redacted_assistant_tool_calls=redacted_assistant_tool_calls,
    redacted_tool_rows=redacted_tool_rows,
    parent_composition_state_id=current_state_id,
    expected_current_state_id=current_state_id,
    writer_principal="compose_loop",
    plugin_crash_pending=plugin_crash is not None,
)
```

Wrap only this await in a narrow `except AuditIntegrityError as exc:` block to attach route diagnostics, then immediately bare-raise the same exception:

```python
try:
    sessions_service = self._require_sessions_service()
    audit_outcome = await sessions_service.persist_compose_turn_async(
        session_id=session_id,
        assistant_content=assistant_message.content or "",
        raw_content=raw_assistant_content,
        redacted_assistant_tool_calls=redacted_assistant_tool_calls,
        redacted_tool_rows=redacted_tool_rows,
        parent_composition_state_id=current_state_id,
        expected_current_state_id=current_state_id,
        writer_principal="compose_loop",
        plugin_crash_pending=plugin_crash is not None,
    )
except AuditIntegrityError as exc:
    exc.failed_turn = FailedTurnMetadata(
        assistant_message_id=None,
        tool_calls_attempted=len(assistant_message.tool_calls or ()),
        tool_responses_persisted=0,
    )
    raise
```

This does not recover, translate, or swallow the Tier-1 failure; it preserves the outward `AuditIntegrityError` signal while giving the FastAPI handler enough context to return the required `failed_turn` diagnostic surface.

`current_state_id` was captured before the LLM call (Task 4 Step 2); it serves as both the parent state id and the stale-state guard. The sync primitive re-reads `MAX(version)` under the session write lock and raises `StaleComposeStateError` if a concurrent writer has advanced the session between LLM call and persist.

Then make the route cutover explicit. This is a literal call-site disposition, not a semantic category list; re-run the `rg` command below and adjust line numbers if `RC5.2` has advanced before implementation:

| Current rev-4 baseline site | Disposition in this PR |
|---|---|
| `routes.py:1364` — `_handle_convergence_error` unwind drain | Delete or guard so it never drains compose-loop carriers after `_compose_loop` has committed. Convergence errors raised after a committed turn surface `exc.failed_turn`; pre-commit convergence errors such as `tool_call_cap_exceeded` must carry no route-persistable `tool_invocations`. |
| `routes.py:1514` — `_handle_plugin_crash` unwind drain | Delete or guard so it never drains compose-loop carriers. Plugin-crash tool rows, including the crashing call, are committed by `persist_compose_turn_async` before `ComposerPluginCrashError` is raised; route code reads `exc.failed_turn`. |
| `routes.py:1737` — `_handle_runtime_preflight_failure` unwind drain | Guard by origin. Compose-loop-origin failures use committed rows and `exc.failed_turn`; retain only the non-loop post-compose/state-save path if current code still has one, with a comment explaining why it is outside `_compose_loop` persistence. |
| `routes.py:3433` — `send_message` assistant `add_message("assistant", result.message, ...)` | Retain only for the terminal assistant response that has no tool rows to persist. It must not be used for assistant messages whose tool-call rows were already committed by `_compose_loop`; the returned `ComposerResult` contract must make this distinction mechanical. |
| `routes.py:3444` — `send_message` normal success `_persist_tool_invocations(... parent_assistant_id=assistant_msg.id, plugin_crash_pending=False)` | Delete or guard. `_compose_loop` owns assistant + tool row persistence for tool-call turns; this call would duplicate rows. |
| `routes.py:3978` — `recompose` assistant `add_message("assistant", result.message, ...)` | Same as `3433`: retain only for terminal no-tool assistant responses, never for already-persisted tool-call assistant turns. |
| `routes.py:3986` — `recompose` normal success `_persist_tool_invocations(... parent_assistant_id=assistant_msg.id, plugin_crash_pending=False)` | Delete or guard. `_compose_loop` owns assistant + tool row persistence for tool-call turns; this call would duplicate rows. |
| `routes.py:4944` — guided endpoint success/`finally` drain | Retain. This is a guided-flow `BufferingRecorder` drain, not the compose-loop persistence path. Add a local comment naming the distinction if the surrounding code is ambiguous. |
| `routes.py:4964` — guided endpoint exception/`finally` drain | Retain for the same guided-flow reason as `4944`. |
| `routes.py:5533` — second guided/terminal endpoint success/`finally` drain | Retain. Different recorder, not covered by `persist_compose_turn_async`. |
| `routes.py:5553` — second guided/terminal endpoint exception/`finally` drain | Retain. Different recorder, not covered by `persist_compose_turn_async`. |

Also retain `_persist_llm_calls` call sites unless a separate task explicitly moves LLM-call sidecars into the Phase 1 primitive. Phase 3's single dispatch covers assistant + tool rows; LLM call audit sidecars remain route-layer sidecars unless separately scoped.

Concrete rev-4 baseline inventory commands:

Confirm by grep:

```bash
grep -n "_persist_tool_invocations\|add_message" src/elspeth/web/composer/service.py
grep -n "_persist_tool_invocations" src/elspeth/web/sessions/routes.py
```

The service grep should show no compose-loop route-drain path. The routes grep should show only retained non-compose-loop/guided call sites or guarded branches whose comments explain why they are not duplicating rows already committed by `persist_compose_turn_async`.

- [x] **Step 4: Re-run the Step 2 tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v
```

Expected: PASS for every Step 1 and Step 2 case including the plugin-crash test from Task 4 (now that the audit write runs before the raise).

- [x] **Step 5: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 2 — single sync dispatch via persist_compose_turn_async; delete legacy drain (composer-progress-persistence phase 3)"
```

---

## Task 7: `_compose_loop` Step 3 — `AuditOutcome` + `AuditIntegrityError` dispatch

The post-dispatch step:

1. If `plugin_crash is not None`, raise the captured `ComposerPluginCrashError`. The audit row has already committed (or `audit_outcome.unwind_audit_failed=True` means the audit failed under primacy and the counter has already incremented inside `persist_compose_turn`). Either way, the plugin-crash signal is the dominant outward signal.
2. If `plugin_crash is None` and `audit_outcome.assistant_id is None`, that means `unwind_audit_failed=True` arose with no in-flight plugin crash — but that combination is structurally impossible: `unwind_audit_failed` is only set when `plugin_crash_pending=True` (spec §5.2.2 line 957–972). Treat the case as a Tier-1 invariant: raise `AuditIntegrityError`.
3. `AuditIntegrityError` raised inside the sync worker (CL-PP-4b, CL-PP-10a) propagates out of `persist_compose_turn_async` unmodified. It is in `TIER_1_ERRORS`; the surrounding `_compose_loop` MUST NOT catch it.

The third historical `_AuditOutcome.tier1_violation` shape is removed (spec §5.2.1 lines 1503–1513 — the rev-4 flag-return pattern is supplanted by raise-from-worker).

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` body, after Task 6)
- Modify: `tests/unit/web/composer/test_audit_failure_primacy.py` (exists at rev-3 baseline; extend in place)
- Create or extend: `tests/integration/web/test_compose_loop_audit_integrity_route_stack.py` (end-to-end FastAPI route propagation)

- [x] **Step 1: Extend the failing red tests.**

Read the existing `tests/unit/web/composer/test_audit_failure_primacy.py` before editing. It already covers `persist_compose_turn` COMMIT-failure primacy with a dialect-level `do_commit` injection. Extend it for the compose-loop dispatch boundary; do not replace it with the older rev-2 draft.

```python
"""Audit-failure primacy disposition (spec §5.2.2 / §8.1 audit-failure
primacy test surface)."""
from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.protocol import ComposerPluginCrashError


def test_tool_success_audit_commit_failure_raises_AuditIntegrityError(
    composer_service_with_real_sessions,
    inject_commit_OperationalError,
    fake_llm_two_tool_calls,
):
    """Tool succeeds, audit COMMIT fails — AuditIntegrityError raised
    from the sync worker; composer.audit.tool_row_tier1_violation_total
    increments; _compose_loop propagates unmodified."""
    inject_commit_OperationalError()  # SQLAlchemy event hook
    with pytest.raises(AuditIntegrityError) as excinfo:
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_two_tool_calls
        )
    # chained from the OperationalError:
    assert isinstance(excinfo.value.__cause__, OperationalError)
    # route-visible Tier-1 errors must carry failed_turn directly; no
    # handler fallback is permitted.
    assert excinfo.value.failed_turn.assistant_message_id is None
    assert excinfo.value.failed_turn.tool_calls_attempted == 2
    assert excinfo.value.failed_turn.tool_responses_persisted == 0
    # counter incremented:
    counter_value = composer_service_with_real_sessions._telemetry.tool_row_tier1_violation_total
    from elspeth.web.sessions.telemetry import observed_value
    assert observed_value(counter_value) == 1


def test_tool_failure_audit_commit_failure_returns_unwind_audit_failed(
    composer_service_with_real_sessions, inject_commit_OperationalError,
    fake_llm_runtime_error_on_second,
):
    """Tool fails AND audit COMMIT fails — AuditOutcome(assistant_id=None,
    unwind_audit_failed=True) returned; unwind counter increments;
    log permitted under primacy (audit-system failure);
    ComposerPluginCrashError propagates (the dominant outward signal)."""
    inject_commit_OperationalError()
    with pytest.raises(ComposerPluginCrashError):
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_runtime_error_on_second
        )
    from elspeth.web.sessions.telemetry import observed_value
    telemetry = composer_service_with_real_sessions._telemetry
    assert observed_value(telemetry.tool_row_persist_failed_during_unwind_total) == 1
    # Tier-1 counter does NOT increment in this branch (primacy disposition):
    assert observed_value(telemetry.tool_row_tier1_violation_total) == 0


def test_audit_IntegrityError_propagates_regardless_of_in_flight_tool_state(
    composer_service_with_real_sessions, inject_IntegrityError_on_chat_messages,
    fake_llm_two_tool_calls,
):
    """IntegrityError on chat_messages always propagates — no primacy
    disposition, no recovery."""
    inject_IntegrityError_on_chat_messages()
    with pytest.raises(IntegrityError):  # NOT AuditIntegrityError
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_two_tool_calls
        )
```

Add an end-to-end route-stack regression:

```python
def test_audit_integrity_error_from_sync_worker_returns_500_without_body_suppression(
    composer_test_client,
    session_with_pending_compose_request,
    inject_commit_OperationalError,
):
    """AuditIntegrityError raised by persist_compose_turn_async propagates
    through _compose_loop and the route helper stack as a 500.

    The response must not be laundered into convergence/preflight shape
    and must not return tool rows whose audit write failed.
    """
    inject_commit_OperationalError()
    response = composer_test_client.post(
        f"/api/sessions/{session_with_pending_compose_request.id}/compose",
        json={"message": "Use the test tool once."},
    )
    assert response.status_code == 500
    body = response.json()
    assert body.get("error_type") == "audit_integrity_error"
    assert body["failed_turn"]["assistant_message_id"] is None
    assert body["failed_turn"]["tool_calls_attempted"] >= 1
    assert body["failed_turn"]["tool_responses_persisted"] == 0
    assert "tool_rows" not in body
```

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_audit_failure_primacy.py -v
.venv/bin/python -m pytest tests/integration/web/test_compose_loop_audit_integrity_route_stack.py -v
```

Expected: FAIL — Step 3 dispatch logic not yet present.

- [x] **Step 2: Insert Step 3 dispatch.**

After Task 6's `audit_outcome = await ...` line:

```python
# Step 3 — dispatch by audit outcome and any pending plugin crash.
#
# AuditOutcome has two valid shapes (spec §5.2.2):
#   (1) success — assistant_id set, unwind_audit_failed=False.
#   (2) tool failed AND audit unwind failed — assistant_id=None,
#       unwind_audit_failed=True. (Only reachable when
#       plugin_crash_pending=True; see spec §5.2.2 line 957.)
#
# A third combination — assistant_id=None with plugin_crash=None — is
# structurally unreachable: persist_compose_turn raises
# AuditIntegrityError on Tier-1 failure, never returns the flag-set
# shape with no plugin crash in flight.
failed_turn = FailedTurnMetadata(
    assistant_message_id=audit_outcome.assistant_id,
    tool_calls_attempted=len(assistant_message.tool_calls or ()),
)
if plugin_crash is not None:
    # The captured plugin-bug exception is the dominant outward signal.
    # Audit either committed (audit_outcome.assistant_id set) or
    # failed under primacy (audit_outcome.unwind_audit_failed=True);
    # in either case the route-helper _handle_plugin_crash runs and
    # produces the 500 response with partial_state + failed_turn.
    assert audit_outcome.assistant_id is not None or audit_outcome.unwind_audit_failed
    plugin_crash.failed_turn = failed_turn
    raise plugin_crash
# Success path: turn committed atomically. Loop continues.
assert audit_outcome.assistant_id is not None
persisted_assistant_message_id = audit_outcome.assistant_id
```

These asserts are intentional Tier-1 invariants, not defensive recovery. They guard future `AuditOutcome` shape changes from silently corrupting the loop and satisfy the plugin-crash branch check requested in rev-3 review. Do not replace them with a catch-and-continue branch.

- [x] **Step 3: Re-run the audit-failure-primacy tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_audit_failure_primacy.py -v
.venv/bin/python -m pytest tests/integration/web/test_compose_loop_audit_integrity_route_stack.py -v
```

Expected: PASS.

- [x] **Step 4: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 3 — AuditOutcome dispatch + plugin-crash raise-after-audit (composer-progress-persistence phase 3)"
```

---

## Task 7b: FastAPI exception handlers for audit and stale-compose failures

`AuditIntegrityError` and `StaleComposeStateError` are now route-visible failures. They must not fall through to FastAPI's default 500 body, and they must not be misclassified as convergence/preflight/plugin failures. `include_tool_rows=true` audit-log write failures also need a fail-closed route contract.

**Files:**
- Modify: `src/elspeth/web/app.py` (register exception handlers in `create_app`)
- Modify: `src/elspeth/web/sessions/protocol.py` (define `AuditAccessLogWriteError` if the helper uses a dedicated exception)
- Modify: `src/elspeth/web/sessions/service.py` (translate audit-access-log write failures to `AuditAccessLogWriteError`, increment the failure counter from Task 11)
- Create: `tests/unit/web/test_composer_exception_handlers.py`

- [x] **Step 1: Write the failing route-handler tests.**

In `tests/unit/web/test_composer_exception_handlers.py`:

```python
"""Route-boundary exception handlers for compose-loop persistence failures."""
from __future__ import annotations


def test_audit_integrity_error_handler_returns_static_500(
    composer_test_client,
    session_with_pending_compose_request,
    inject_commit_OperationalError,
):
    inject_commit_OperationalError()
    response = composer_test_client.post(
        f"/api/sessions/{session_with_pending_compose_request.id}/compose",
        json={"message": "Use the test tool once."},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["error_type"] == "audit_integrity_error"
    assert body["failed_turn"]["assistant_message_id"] is None
    assert body["failed_turn"]["tool_responses_persisted"] == 0
    assert "tool_rows" not in body
    assert "messages" not in body
    assert "OperationalError" not in response.text


def test_audit_integrity_error_handler_returns_typed_degraded_body_without_failed_turn(
    composer_test_client,
    session_with_pending_compose_request,
    inject_non_compose_loop_AuditIntegrityError,
):
    """AuditIntegrityError outside _compose_loop annotation scope has a
    typed failed_turn=None state and still receives a stable body."""
    inject_non_compose_loop_AuditIntegrityError()
    response = composer_test_client.post(
        f"/api/sessions/{session_with_pending_compose_request.id}/compose",
        json={"message": "Use the test tool once."},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["error_type"] == "audit_integrity_error"
    assert body["diagnostic"] == "no_failed_turn_metadata"
    assert body["reason"] == "originated outside compose-loop annotation scope"
    assert "failed_turn" not in body
    assert "messages" not in body


def test_stale_compose_state_error_handler_returns_409(
    composer_test_client,
    session_with_pending_compose_request,
    advance_compose_state_after_llm_before_persist,
):
    advance_compose_state_after_llm_before_persist()
    response = composer_test_client.post(
        f"/api/sessions/{session_with_pending_compose_request.id}/compose",
        json={"message": "Use the test tool once."},
    )
    assert response.status_code == 409
    assert response.json()["error_type"] == "stale_compose_state"


def test_audit_access_log_write_error_handler_returns_fail_closed_500(
    test_client,
    session_with_user_assistant_tool_rows,
    inject_audit_access_log_write_failure,
):
    inject_audit_access_log_write_failure()
    response = test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
        f"?include_tool_rows=true"
    )
    assert response.status_code == 500
    body = response.json()
    assert body["error_type"] == "audit_access_log_write_failed"
    assert "messages" not in body
```

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/test_composer_exception_handlers.py -v
```

Expected: FAIL — the handlers are not registered and audit-access-log write failure is not yet translated.

- [x] **Step 2: Register handlers in `create_app`.**

Add handlers with static, scrubbed bodies:

```python
@app.exception_handler(AuditIntegrityError)
async def _audit_integrity_error_handler(request: Request, exc: AuditIntegrityError) -> JSONResponse:
    failed_turn = exc.failed_turn
    if failed_turn is None:
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "audit_integrity_error",
                "detail": "Audit persistence failed; no audit-grade data returned.",
                "diagnostic": "no_failed_turn_metadata",
                "reason": "originated outside compose-loop annotation scope",
            },
        )
    return JSONResponse(
        status_code=500,
        content={
            "error_type": "audit_integrity_error",
            "detail": "Audit persistence failed; no audit-grade data returned.",
            "failed_turn": {
                "assistant_message_id": failed_turn.assistant_message_id,
                "tool_calls_attempted": failed_turn.tool_calls_attempted,
                "tool_responses_persisted": 0,
                "transcript_url": None,
            },
        },
    )


@app.exception_handler(StaleComposeStateError)
async def _stale_compose_state_error_handler(request: Request, exc: StaleComposeStateError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error_type": "stale_compose_state",
            "detail": "The session changed while the compose turn was running.",
        },
    )


@app.exception_handler(AuditAccessLogWriteError)
async def _audit_access_log_write_error_handler(request: Request, exc: AuditAccessLogWriteError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error_type": "audit_access_log_write_failed",
            "detail": "Audit-grade transcript access could not be recorded.",
        },
    )
```

Do not include `str(exc)`, SQL text, provider payloads, request-id fallback logic, or tool rows in these responses. `AuditIntegrityError.failed_turn is None` is a typed route-boundary state meaning the exception originated outside `_compose_loop`'s catch-and-annotate scope; it is not an attribute-absence fallback. Compose-loop-origin audit failures must populate `FailedTurnMetadata`; non-compose-loop audit failures get the typed degraded body above. Do not use `getattr`, `hasattr`, default dicts, or catch-and-fill behavior in the handler. Logging policy: route handlers may emit telemetry and class-name-only diagnostics for the audit subsystem failure, but must not log row-level content or tool payloads.

- [x] **Step 3: Re-run the handler tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/test_composer_exception_handlers.py -v
```

Expected: PASS.

- [x] **Step 4: Commit.**

```bash
git commit -am "feat(web): add compose persistence exception handlers (composer-progress-persistence phase 3)"
```

---

## Task 8: `failed_turn` response field on `_handle_*` helpers

Add `failed_turn` to the response body of `_handle_convergence_error`, `_handle_plugin_crash`, and `_handle_runtime_preflight_failure` (`src/elspeth/web/sessions/routes.py` helper symbols; do not rely on stale line numbers). The field carries `assistant_message_id`, `tool_calls_attempted`, `tool_responses_persisted`, and `transcript_url`. Phase 4 consumes this to decide whether the recovery panel opens. The same diagnostic surface must also exist for the route-level `AuditIntegrityError` 500 added in Task 7 so a Tier-1 crash does not leave the operator with an empty body.

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py` (three handler functions)
- Create: `tests/integration/web/test_compose_loop_failed_turn_field.py`

- [x] **Step 1: Write the failing red test.**

In `tests/integration/web/test_compose_loop_failed_turn_field.py`:

```python
"""failed_turn response shape across the three _handle_* helpers
(spec §6.1)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_handle_convergence_error_returns_failed_turn(
    composer_test_client,
    session_with_composer_state,
    fake_llm_exceeds_budget,
):
    """Convergence error mid-loop → failed_turn carries
    assistant_message_id, tool_calls_attempted, tool_responses_persisted,
    transcript_url."""
    response = composer_test_client.post(
        f"/api/sessions/{session_with_composer_state.id}/compose",
        json={"message": "Run enough discovery turns to exceed the test budget."},
    )
    assert response.status_code == 422
    body = response.json()
    assert "failed_turn" in body
    ft = body["failed_turn"]
    assert "assistant_message_id" in ft
    assert isinstance(ft["tool_calls_attempted"], int)
    assert isinstance(ft["tool_responses_persisted"], int)
    assert ft["tool_responses_persisted"] <= ft["tool_calls_attempted"]
    assert ft["transcript_url"].endswith("&include_tool_rows=true")


def test_handle_plugin_crash_returns_failed_turn(
    composer_test_client,
    session_with_composer_state,
    fake_llm_plugin_crash,
):
    """Plugin crash mid-loop → 500 response carries failed_turn."""
    response = composer_test_client.post(
        f"/api/sessions/{session_with_composer_state.id}/compose",
        json={"message": "Call the plugin-crash fixture tool."},
    )
    assert response.status_code == 500
    body = response.json()
    assert "failed_turn" in body


def test_handle_runtime_preflight_failure_returns_failed_turn(
    composer_test_client,
    session_with_composer_state,
    fake_llm_preflight_blocks,
):
    """Runtime preflight failure → response carries failed_turn (with
    tool_responses_persisted=0 since no tools executed)."""
    response = composer_test_client.post(
        f"/api/sessions/{session_with_composer_state.id}/compose",
        json={"message": "Preview the pipeline with the failing runtime preflight fixture."},
    )
    body = response.json()
    assert body["failed_turn"]["tool_responses_persisted"] == 0
```

The fixtures above are names for the integration harness implementer to bind to the existing session/client factories. Do not leave literal `...` ellipses in the committed test file.

```bash
.venv/bin/python -m pytest tests/integration/web/test_compose_loop_failed_turn_field.py -v
```

Expected: FAIL — none of the helpers emits `failed_turn` yet.

- [x] **Step 2: Add the field to each helper.**

In each of `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure`, after `partial_state` is computed and before `response_body` is returned:

```python
failed_turn = exc.failed_turn
assistant_message_id = failed_turn.assistant_message_id if failed_turn else None
tool_responses_persisted = await sessions_service.count_tool_responses_for_assistant_async(
    session_id=session_id,
    assistant_message_id=assistant_message_id,  # None if no assistant row landed
)
tool_calls_attempted = failed_turn.tool_calls_attempted if failed_turn else 0

response_body["failed_turn"] = {
    "assistant_message_id": assistant_message_id,
    "tool_calls_attempted": tool_calls_attempted,
    "tool_responses_persisted": tool_responses_persisted,
    "transcript_url": (
        f"/api/sessions/{session_id}/messages"
        f"?since={user_message_id}&include_tool_rows=true"
    ),
}
```

Task 9 below adds `count_tool_responses_for_assistant_async`; this task wires the call site.

- [x] **Step 3: Re-run the test (after Task 9 lands the helper).**

Expected: PASS once Task 9 is in.

- [x] **Step 4: Commit (combined with Task 9's helper).**

This task's commit is paired with Task 9 because the route-layer wiring is meaningless without the helper. See Task 9 Step 5.

---

## Task 9: `count_tool_responses_for_assistant` read helper on SessionsService

`failed_turn.tool_responses_persisted` is computed by a SELECT after the audit writes commit. The route helper does not run inline SQL; the SELECT lives on `SessionsService` as a typed read helper. Spec §6.1 read-consistency note: this SELECT runs after the surrounding compose-loop coroutine has fully awaited, so the writes are durable.

**Files:**
- Modify: `src/elspeth/web/sessions/service.py` (add `count_tool_responses_for_assistant` sync + `count_tool_responses_for_assistant_async` async dispatcher)
- Modify: `src/elspeth/web/sessions/protocol.py` (add the async method to `SessionServiceProtocol`)
- Create: `tests/unit/web/sessions/test_count_tool_responses_for_assistant.py`

- [x] **Step 1: Write the failing red test.**

```python
"""count_tool_responses_for_assistant — read helper used by the route
layer to compute failed_turn.tool_responses_persisted (spec §6.1)."""
from __future__ import annotations

import pytest

from elspeth.web.sessions.engine import (
    create_session_engine, initialize_session_schema,
)
from sqlalchemy.pool import StaticPool


@pytest.fixture
def sessions_service():
    engine = create_session_engine("sqlite://", poolclass=StaticPool)
    initialize_session_schema(engine)
    return build_test_sessions_service(engine=engine)


def test_count_zero_when_no_tool_rows(sessions_service, persisted_assistant_no_tools):
    count = sessions_service.count_tool_responses_for_assistant(
        session_id=persisted_assistant_no_tools.session_id,
        assistant_message_id=persisted_assistant_no_tools.id,
    )
    assert count == 0


def test_count_matches_inserted_tool_rows(sessions_service, persisted_assistant_three_tools):
    count = sessions_service.count_tool_responses_for_assistant(
        session_id=persisted_assistant_three_tools.session_id,
        assistant_message_id=persisted_assistant_three_tools.id,
    )
    assert count == 3


def test_count_none_assistant_id_returns_zero(sessions_service):
    """Assistant row never landed (e.g., CL-PP-4a) → count is 0."""
    count = sessions_service.count_tool_responses_for_assistant(
        session_id="sess-1",
        assistant_message_id=None,
    )
    assert count == 0


@pytest.mark.asyncio
async def test_async_dispatcher_runs_in_worker_thread(sessions_service, persisted_assistant_three_tools):
    count = await sessions_service.count_tool_responses_for_assistant_async(
        session_id=persisted_assistant_three_tools.session_id,
        assistant_message_id=persisted_assistant_three_tools.id,
    )
    assert count == 3
```

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_count_tool_responses_for_assistant.py -v
```

Expected: FAIL — method does not exist.

- [x] **Step 2: Add the sync helper.**

In `src/elspeth/web/sessions/service.py`, after the existing read helpers:

```python
def count_tool_responses_for_assistant(
    self,
    *,
    session_id: str,
    assistant_message_id: str | None,
) -> int:
    """Count role='tool' rows linked to the given assistant message.

    Used by the route layer to compute
    ``failed_turn.tool_responses_persisted`` after the compose-loop
    coroutine has fully awaited (spec §6.1 read-consistency note —
    persist_compose_turn's writes are durable by the time this SELECT
    runs because the route helper awaits the full compose-loop
    coroutine before calling this).

    ``assistant_message_id=None`` (no assistant row landed, e.g.
    CL-PP-4a) returns 0. The check is at the boundary, not inside the
    SQL, to keep the predicate path simple.
    """
    if assistant_message_id is None:
        return 0
    with self._engine.connect() as conn:
        result = conn.execute(
            select(func.count()).select_from(chat_messages_table)
            .where(chat_messages_table.c.session_id == session_id)
            .where(chat_messages_table.c.parent_assistant_id == assistant_message_id)
            .where(chat_messages_table.c.role == "tool")
        ).scalar_one()
    return int(result)
```

- [x] **Step 3: Add the async dispatcher.**

In `src/elspeth/web/sessions/service.py`:

```python
async def count_tool_responses_for_assistant_async(
    self,
    *,
    session_id: str,
    assistant_message_id: str | None,
) -> int:
    return await self._run_sync(
        self.count_tool_responses_for_assistant,
        session_id=session_id,
        assistant_message_id=assistant_message_id,
    )
```

In `src/elspeth/web/sessions/protocol.py`, add the method to `SessionServiceProtocol` so the route layer types it via the protocol, not the concrete class.

- [x] **Step 4: Re-run Task 8 + Task 9 tests.**

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_count_tool_responses_for_assistant.py \
  tests/integration/web/test_compose_loop_failed_turn_field.py -v
```

Expected: PASS.

- [x] **Step 5: Commit (paired with Task 8's wiring).**

```bash
git commit -am "feat(sessions): count_tool_responses_for_assistant + failed_turn response field (composer-progress-persistence phase 3)"
```

---

## Task 10: `include_tool_rows` query parameter on `GET /api/sessions/{sid}/messages`

Extend the existing messages endpoint with the new query parameter and the new response-row fields. Default `include_tool_rows=false` preserves the live chat panel's existing behaviour; `include_tool_rows=true` returns tool rows interleaved by `sequence_no` AND triggers the audit-grade access-log emission in Task 11.

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py` (the messages-list endpoint)
- Modify: `src/elspeth/web/sessions/schemas.py` (message response model — add `tool_call_id`, `parent_assistant_id`, `sequence_no`)
- Create: `tests/unit/web/sessions/test_messages_route_include_tool_rows.py`

- [x] **Step 1: Write the failing red test.**

```python
"""GET /api/sessions/{sid}/messages — include_tool_rows query parameter
(spec §6.2)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_default_excludes_tool_rows(test_client, session_with_user_assistant_tool_rows):
    response = test_client.get(f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages")
    assert response.status_code == 200
    rows = response.json()["messages"]
    assert all(r["role"] in ("user", "assistant", "system") for r in rows)


def test_include_tool_rows_returns_tool_rows_interleaved_by_sequence_no(
    test_client, session_with_user_assistant_tool_rows,
):
    response = test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
        f"?include_tool_rows=true"
    )
    assert response.status_code == 200
    rows = response.json()["messages"]
    sequence_nos = [r["sequence_no"] for r in rows]
    assert sequence_nos == sorted(sequence_nos)  # strictly monotonic
    roles = [r["role"] for r in rows]
    assert "tool" in roles


def test_response_rows_expose_new_columns(
    test_client, session_with_user_assistant_tool_rows,
):
    response = test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
        f"?include_tool_rows=true"
    )
    rows = response.json()["messages"]
    for row in rows:
        assert "sequence_no" in row
        if row["role"] == "tool":
            assert row["tool_call_id"] is not None
            assert row["parent_assistant_id"] is not None
        else:
            assert row.get("tool_call_id") is None
            assert row.get("parent_assistant_id") is None
```

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_messages_route_include_tool_rows.py -v
```

Expected: FAIL.

- [x] **Step 2: Extend the response schema.**

In `src/elspeth/web/sessions/schemas.py`, the existing message response model gains three optional fields:

```python
class MessageResponse(BaseModel):
    # existing fields remain unchanged
    tool_call_id: str | None = None
    parent_assistant_id: str | None = None
    sequence_no: int  # NOT None — every row has one
```

- [x] **Step 3: Extend the endpoint.**

```python
@router.get("/api/sessions/{session_id}/messages")
async def list_messages(
    session_id: str,
    request: Request,
    include_tool_rows: bool = Query(False, description=(
        "When true, role='tool' rows are interleaved by sequence_no. "
        "Triggers audit-grade access logging per spec §6.3."
    )),
    since: str | None = Query(None),
    sessions_service: SessionServiceProtocol = Depends(get_sessions_service),
    principal: AuthPrincipal = Depends(get_current_principal),
) -> MessagesListResponse:
    # Existing ownership check is unchanged.
    await _enforce_session_ownership(session_id=session_id, principal=principal)

    # Task 11 adds the audit-grade-view access-log emission here when
    # include_tool_rows=true. Insertion point marked.

    messages = await sessions_service.list_messages_async(
        session_id=session_id,
        include_tool_rows=include_tool_rows,
        since=since,
    )
    return MessagesListResponse(messages=messages)
```

`SessionServiceImpl.list_messages` already exists for the default case — extend it to accept `include_tool_rows: bool` and conditionally widen the role filter. Default ordering is `(sequence_no ASC)` per spec §6.2.

- [x] **Step 4: Re-run the test.**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_messages_route_include_tool_rows.py -v
```

Expected: PASS for the three default-vs-true tests. The audit-grade-view assertion belongs to Task 11.

- [x] **Step 5: Commit.**

```bash
git commit -am "feat(sessions): include_tool_rows query parameter on messages endpoint (composer-progress-persistence phase 3)"
```

---

## Task 11: `record_audit_grade_view` + `audit_access_log` emission

When `include_tool_rows=true`, emit an `audit_access_log` row (writer_principal='audit_grade_view') before returning rows; increment `composer.audit.audit_grade_view_total`. If the write fails, increment `composer.audit.audit_access_log_write_failed_total`, raise `AuditAccessLogWriteError`, and return no tool rows. The table schema is already in place from Phase 1 (`audit_access_log_table` in `src/elspeth/web/sessions/models.py`).

**Files:**
- Modify: `src/elspeth/web/sessions/service.py` (add `record_audit_grade_view` + async dispatcher)
- Modify: `src/elspeth/web/sessions/protocol.py` (add `record_audit_grade_view_async` to the protocol)
- Modify: `src/elspeth/web/sessions/telemetry.py` (`_SessionsTelemetry` gains `audit_grade_view_total` and `audit_access_log_write_failed_total`; build_sessions_telemetry wires both)
- Modify: `src/elspeth/web/sessions/routes.py` (the messages endpoint emits the row before returning when `include_tool_rows=true`)
- Create: `tests/unit/web/sessions/test_record_audit_grade_view.py`

- [x] **Step 1: Write the failing red test.**

```python
"""record_audit_grade_view — write-helper for the audit_access_log
table (spec §6.3)."""
from __future__ import annotations

import pytest


def test_record_audit_grade_view_writes_row(sessions_service, session_owned_by_alice):
    sessions_service.record_audit_grade_view(
        session_id=session_owned_by_alice.id,
        requesting_principal="alice",
        request_path="/api/sessions/sess-1/messages",
        query_args={"include_tool_rows": "true", "since": "msg-42"},
        ip_address="10.0.0.5",
    )
    rows = sessions_service.list_audit_access_log(session_id=session_owned_by_alice.id)
    assert len(rows) == 1
    row = rows[0]
    assert row.requesting_principal == "alice"
    assert row.writer_principal == "audit_grade_view"
    assert row.request_path == "/api/sessions/sess-1/messages"
    assert row.query_args == {"include_tool_rows": "true", "since": "msg-42"}
    assert row.ip_address == "10.0.0.5"


def test_record_audit_grade_view_increments_counter(sessions_service, session_owned_by_alice):
    from elspeth.web.sessions.telemetry import observed_value
    sessions_service.record_audit_grade_view(
        session_id=session_owned_by_alice.id,
        requesting_principal="alice",
        request_path="/api/sessions/sess-1/messages",
        query_args={},
        ip_address=None,
    )
    assert observed_value(sessions_service._telemetry.audit_grade_view_total) == 1


def test_record_audit_grade_view_writer_principal_is_pinned(sessions_service, session_owned_by_alice):
    """The helper does NOT accept a writer_principal argument — pinned
    to 'audit_grade_view'. Admin-tool writes use a separate code path."""
    import inspect
    sig = inspect.signature(sessions_service.record_audit_grade_view)
    assert "writer_principal" not in sig.parameters


def test_endpoint_emits_audit_log_when_include_tool_rows_true(
    test_client, session_with_user_assistant_tool_rows,
):
    test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
        f"?include_tool_rows=true"
    )
    sessions_service = test_client.app.state.sessions_service
    rows = sessions_service.list_audit_access_log(
        session_id=session_with_user_assistant_tool_rows.id,
    )
    assert len(rows) == 1


def test_endpoint_does_not_emit_audit_log_when_include_tool_rows_false(
    test_client, session_with_user_assistant_tool_rows,
):
    test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
    )
    sessions_service = test_client.app.state.sessions_service
    rows = sessions_service.list_audit_access_log(
        session_id=session_with_user_assistant_tool_rows.id,
    )
    assert rows == []


def test_endpoint_fails_closed_when_audit_access_log_write_fails(
    test_client,
    session_with_user_assistant_tool_rows,
    inject_audit_access_log_write_failure,
):
    """include_tool_rows=true is audit-grade access. If the
    audit_access_log write fails, the endpoint returns 500 and must not
    return tool rows whose access was not recorded."""
    inject_audit_access_log_write_failure()
    response = test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
        f"?include_tool_rows=true"
    )
    assert response.status_code == 500
    body = response.json()
    assert body.get("error_type") == "audit_access_log_write_failed"
    assert "messages" not in body
    sessions_service = test_client.app.state.sessions_service
    from elspeth.web.sessions.telemetry import observed_value
    assert observed_value(
        sessions_service._telemetry.audit_access_log_write_failed_total
    ) == 1
```

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_record_audit_grade_view.py -v
```

Expected: FAIL.

- [x] **Step 2: Add the counter to `_SessionsTelemetry`.**

In `src/elspeth/web/sessions/telemetry.py`, extend the dataclass and the build function:

```python
audit_grade_view_total: _Counter
audit_access_log_write_failed_total: _Counter
# existing counter fields remain unchanged
audit_grade_view_total=_FakeCounter(),
audit_access_log_write_failed_total=_FakeCounter(),
# existing fake-counter assignments remain unchanged
audit_grade_view_total=meter.create_counter("composer.audit.audit_grade_view_total"),
audit_access_log_write_failed_total=meter.create_counter(
    "composer.audit.audit_access_log_write_failed_total"
),
```

Update the `_SessionsTelemetry` docstring to drop the "Phase 3 (compose loop + audit-grade view) adds ... `audit_grade_view_total`" forward-looking line.

- [x] **Step 3: Add `record_audit_grade_view` to SessionsService.**

```python
def record_audit_grade_view(
    self,
    *,
    session_id: str,
    requesting_principal: str,
    request_path: str,
    query_args: Mapping[str, str],
    ip_address: str | None,
) -> None:
    """Append one row to audit_access_log (spec §6.3).

    The writer_principal is pinned to 'audit_grade_view'; admin-tool
    writes go through a separate path. The audit_access_log table is
    append-only — there is no update or delete API exposed by
    SessionsService.

    Called from the messages endpoint when include_tool_rows=true.
    The Phase 1 schema already defines the table
    with the writer_principal CHECK constraint pinning the value to
    one of ('audit_grade_view', 'admin_tool').
    """
    now = self._now()
    with self._engine.begin() as conn:
        conn.execute(
            audit_access_log_table.insert().values(
                id=self._new_id(),
                timestamp=now,
                session_id=session_id,
                requesting_principal=requesting_principal,
                request_path=request_path,
                query_args=dict(query_args),  # JSON-serialisable
                ip_address=ip_address,
                writer_principal="audit_grade_view",
            )
        )
    self._telemetry.audit_grade_view_total.add(1)

async def record_audit_grade_view_async(
    self,
    *,
    session_id: str,
    requesting_principal: str,
    request_path: str,
    query_args: Mapping[str, str],
    ip_address: str | None,
) -> None:
    return await self._run_sync(
        self.record_audit_grade_view,
        session_id=session_id,
        requesting_principal=requesting_principal,
        request_path=request_path,
        query_args=query_args,
        ip_address=ip_address,
    )
```

Wrap only the database write in a narrow `except SQLAlchemyError as exc:` block that increments `audit_access_log_write_failed_total` and raises `AuditAccessLogWriteError from exc`. This is a route-boundary translation, not defensive recovery: the request still fails closed and returns no transcript rows. Add `record_audit_grade_view_async` to `SessionServiceProtocol` so the route layer types via the protocol.

- [x] **Step 4: Wire the emission into the messages endpoint.**

In the `list_messages` route handler from Task 10, at the marked insertion point:

```python
if include_tool_rows:
    # Audit-grade access logging (spec §6.3). Emit BEFORE returning
    # rows so the log row is durable even if the response is dropped
    # mid-stream. Audit primacy: the log fires synchronously; if it
    # fails, the request fails — we do not return tool rows whose
    # access went unrecorded.
    await sessions_service.record_audit_grade_view_async(
        session_id=session_id,
        requesting_principal=principal.subject,
        request_path=request.url.path,
        query_args={k: v for k, v in request.query_params.items()},
        ip_address=request.client.host if request.client else None,
    )
```

No broad `except Exception: pass` is permitted around this call. The fail-closed test above must fail if an implementer swallows the write failure and returns rows anyway.

- [x] **Step 5: Re-run the tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_record_audit_grade_view.py -v
```

Expected: PASS.

- [x] **Step 6: Commit.**

```bash
git commit -am "feat(sessions): record_audit_grade_view + audit-grade access logging (composer-progress-persistence phase 3)"
```

---

## Task 12: Integration test surface — CL-PP-* scenarios

This task closes the §11 done-when checklist by ensuring every CL-PP-* scenario from §8.2 passes against `test_composer_llm_eval_characterization.py` and (for CL-PP-11) the testcontainer PostgreSQL CI lane.

### Task 12a: Re-baseline CL-PP-1..8 for the new response shape

These scenarios exist in the characterization test surface (some in skeletal pre-Phase-3 form). They each need updating to assert the new `failed_turn` field on the response body and the new per-row columns on transcript responses.

- [x] **Step 1: Audit `tests/integration/pipeline/test_composer_llm_eval_characterization.py` for existing CL-PP-1..8 cases.**

```bash
grep -n "CL-PP-1\|CL-PP-2\|CL-PP-3\|CL-PP-4a\|CL-PP-4b\|CL-PP-4c\|CL-PP-4d\|CL-PP-5\|CL-PP-6\|CL-PP-7\|CL-PP-8" tests/integration/pipeline/test_composer_llm_eval_characterization.py
```

For each existing case:

- If it asserts only the legacy response shape, extend it to also assert `failed_turn` per §6.1.
- If the case predates the `composition_states.provenance` discriminator, add the discriminator check.
- If the case is missing (CL-PP-4d, e.g.), author it new.

- [x] **Step 2: Each case follows the same TDD cycle.**

Run the case → confirm RED on the new assertions → update the implementation only if a defect surfaces (the loop should now pass) → confirm GREEN.

- [x] **Step 3: Mechanical assertion-churn gate before any CL-PP commit.**

Before committing any CL-PP-* re-baseline, dump the tests diff and categorize every modified assertion:

```bash
git diff -- tests/
```

Write a one-paragraph note in the implementation log with three buckets:

- `shape-now-correct` — assertion changed because response/schema shape intentionally changed in Phase 3.
- `counter-now-correct` — assertion changed because the telemetry/audit counter contract intentionally changed in Phase 3.
- `other` — any assertion change not explained by the two categories above.

If the `other` bucket is non-empty, stop and surface it to the operator before committing. This is the procedural gate against assertion-churn laundering: the CL-PP re-baseline may update expected shapes and counters, but it must not quietly weaken unrelated behavior.

Implementation log (2026-05-14): the CL-PP-1..8 grep found no labelled cases in `test_composer_llm_eval_characterization.py`; the live Phase 3 surface instead exposed one characterization harness failure where the composition-budget replay reached `_compose_loop` persistence without a wired `SessionServiceProtocol`. The fix wires a real `SessionServiceImpl` with `create_session_engine(..., StaticPool)` and `initialize_session_schema()` for that replay. Assertion-churn buckets from `git diff -- tests/`: `shape-now-correct` = none; `counter-now-correct` = none; `other` = none (no assertions changed, only real-DB harness wiring).

- [x] **Step 4: Commit (one commit per scenario or per coherent batch).**

```bash
git commit -am "test(integration): CL-PP-{N} extends failed_turn assertions (composer-progress-persistence phase 3)"
```

### Task 12b: Author CL-PP-9, 10a, 10b, 12, 13

- [x] **CL-PP-9: Mixed redaction policy (§8.2 line 2583).** A tool whose argument model has both `Sensitive[T]`-annotated and non-sensitive fields. Drive a turn that exercises both. Assert sensitive fields persist as sentinel/summarizer output; non-sensitive fields are byte-identical; structural shape preserved.

- [x] **CL-PP-10a: INSERT succeeded, COMMIT failed (no plugin crash).** Inject `OperationalError` on COMMIT via SQLAlchemy event hook. Assert `AuditIntegrityError` raised (chained from the injected error); `composer.audit.tool_row_tier1_violation_total` increments; caller propagates; no rows visible (transaction rolled back).

- [x] **CL-PP-10b: COMMIT failed (plugin crash in flight).** Same injection plus a `RuntimeError` from the second tool. Assert `AuditOutcome(assistant_id=None, unwind_audit_failed=True)`; `tool_row_persist_failed_during_unwind_total` increments; log entry emitted; caller raises the captured `ComposerPluginCrashError`.

- [x] **CL-PP-10c: `asyncio.CancelledError` during shielded sync dispatch.** Inject cancellation after `persist_compose_turn_async` has entered its shielded worker dispatch but before COMMIT returns. Assert cancellation does not interrupt the commit; assistant/tool rows are durable; after the shield completes the caller observes the cancellation according to the route contract. This closes spec §5.5 rows 5-8 at integration level.

- [x] **CL-PP-10d: `asyncio.CancelledError` after COMMIT before response yield.** Inject cancellation after COMMIT succeeds and before the HTTP response is yielded. Assert the cancellation propagates to the client path without data loss; assistant/tool rows and composition-state rows remain queryable; no duplicate route-layer `_persist_tool_invocations` drain runs. This closes spec §5.5 rows 9-11 at integration level.

- [x] **CL-PP-12: Tool-call cap exceeded.** LLM emits 17 tool calls; loop raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")` BEFORE any tool execution; no DB writes; `composer.tool_call_cap_exceeded_total` increments.

- [x] **CL-PP-13: Unknown response key fail-closed (§8.2 line 2627).** A declarative-manifest-entry tool returns a response containing a key not in `known_response_keys`. Assert the value is replaced with the fixed sentinel `<redacted-unknown-response-key>` (rev-5 form; no length disclosure); `composer.redaction.unknown_response_key_total` increments. The tool call completes successfully.

Each case follows: RED test → run → GREEN. Commit per case.

### Task 12c: Property test extension + schema-level backward-direction test

- [x] **Step 1: Author the strategy contracts at `tests/property/web/composer/strategies.py`.**

Strategies per spec §8.3.1: `st_tool_call`, `st_argument_dict`, `st_redaction_policy`, `st_failure_injection_point` (with `audit_raises_OperationalError_on_commit`, `advisory_lock_unavailable`, `tool_call_cap_exceeded`, `unknown_response_key` arms), `st_cancellation_arrival_time`, `st_session_state`. Use Hypothesis `@example(...)` decorators to guarantee every cancellation/failure branch is reached (closes spec QA F-6). The strategies file's docstring contains a mapping table from §5.5 row numbers to strategy values so future drift is detectable.

Create `tests/property/web/composer/__init__.py` if the directory does not already exist so pytest collection is unambiguous.

The cancellation enum is not enough by itself; every value must have a concrete injection mechanism:

| `st_cancellation_arrival_time` value | Required injection mechanism |
|---|---|
| `before_llm_call` | Cancel the compose task before the fake LLM awaitable is entered; assert no assistant/tool rows. |
| `during_llm_call` | Fake LLM blocks on an `asyncio.Event`; cancel while it is awaiting; assert no audit transaction started. |
| `after_llm_before_tool` | Fake LLM releases a hook after returning tool calls and before dispatch loop begins; cancel at the hook; assert no rows. |
| `during_tool_dispatch` | Fake tool blocks on an `asyncio.Event`; cancel while tool execution is in progress; assert no partial audit rows. |
| `after_tool_before_sync_dispatch` | Hook immediately before `persist_compose_turn_async`; cancel at the hook; assert no rows and no route-layer fallback drain. |
| `during_run_sync_between_insert_and_commit` | SQLAlchemy event hook inside the worker after INSERTs and before COMMIT; cancel the outer coroutine while `asyncio.shield` is active; assert commit wins and rows are durable. |
| `during_advisory_lock_acquisition` | Monkeypatch the advisory-lock acquisition helper to block on an event before the lock is acquired; cancel there; assert no rows and no stale lock. |
| `after_commit_before_response_yielded` | Hook after `persist_compose_turn_async` returns and before `_compose_loop` raises/returns; cancel; assert rows remain queryable and cancellation propagates. |
| `after_response_yielded` | Cancel only after the response has been yielded/observed; assert no duplicate writes and no post-response mutation. |

The machine must contain real state and invariants. A committed test with `pass` invariants or an undefined helper name is a false green and fails this plan review.

```python
from __future__ import annotations

from hypothesis import example, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

CANCELLATION_ARRIVAL_TIMES = (
    "before_llm_call",
    "during_llm_call",
    "after_llm_before_tool",
    "during_tool_dispatch",
    "after_tool_before_sync_dispatch",
    "during_run_sync_between_insert_and_commit",
    "during_advisory_lock_acquisition",
    "after_commit_before_response_yielded",
    "after_response_yielded",
)


class ComposeLoopAuditMachine(RuleBasedStateMachine):
    @initialize()
    def start_session(self) -> None:
        self.session_id = create_real_session_for_machine()
        self.observed_assistant_ids: set[str] = set()
        self.observed_tool_call_ids: set[str] = set()
        self.cancellation_points_seen: set[str] = set()

    @rule(
        tool_call=st_tool_call(),
        argument_dict=st_argument_dict(),
        redaction_policy=st_redaction_policy(),
        failure_injection_point=st_failure_injection_point(),
        cancellation_arrival_time=st_cancellation_arrival_time(),
        session_state=st_session_state(),
    )
    def compose_turn(
        self,
        tool_call,
        argument_dict,
        redaction_policy,
        failure_injection_point,
        cancellation_arrival_time,
        session_state,
    ) -> None:
        outcome = drive_compose_loop_with_injections(
            session_id=self.session_id,
            tool_call=tool_call,
            argument_dict=argument_dict,
            redaction_policy=redaction_policy,
            failure_injection_point=failure_injection_point,
            cancellation_arrival_time=cancellation_arrival_time,
            session_state=session_state,
        )
        self.cancellation_points_seen.add(cancellation_arrival_time)
        self.observed_assistant_ids.update(outcome.assistant_message_ids)
        self.observed_tool_call_ids.update(outcome.tool_call_ids)

    @invariant()
    def audit_rows_are_bidirectional(self) -> None:
        assert_no_tool_row_without_parent_assistant(self.session_id)
        assert_no_tool_state_without_tool_row(self.session_id)

    @invariant()
    def sequence_numbers_are_unique_and_ordered(self) -> None:
        assert_sequence_numbers_unique_and_monotonic(self.session_id)

    @invariant()
    def redaction_contract_holds(self) -> None:
        assert_persisted_tool_payloads_match_manifest(self.session_id)
```

The helper names above are placeholders for real helpers implemented in the test module; do not commit them undefined. `create_real_session_for_machine()` must construct `ComposerServiceImpl` with `sessions_service=` bound to a real or protocol-faithful `SessionService`; the sentinel default is for non-persistence callers only and is not appropriate for the property machine. The committed property test must include nine explicit examples, one per cancellation enum value. Use a parameterized harness such as:

```python
@example(cancellation_arrival_time="before_llm_call")
@example(cancellation_arrival_time="during_llm_call")
@example(cancellation_arrival_time="after_llm_before_tool")
@example(cancellation_arrival_time="during_tool_dispatch")
@example(cancellation_arrival_time="after_tool_before_sync_dispatch")
@example(cancellation_arrival_time="during_run_sync_between_insert_and_commit")
@example(cancellation_arrival_time="during_advisory_lock_acquisition")
@example(cancellation_arrival_time="after_commit_before_response_yielded")
@example(cancellation_arrival_time="after_response_yielded")
@settings(max_examples=200)
def test_compose_loop_audit_machine(cancellation_arrival_time: str) -> None:
    machine = ComposeLoopAuditMachine.TestCase()
    machine.force_cancellation_arrival_time = cancellation_arrival_time
    machine.runTest()
```

If Hypothesis's `RuleBasedStateMachine.TestCase()` shape does not permit passing the forced example cleanly in the current Hypothesis version, implement a small `drive_single_example_trace(cancellation_arrival_time=...)` helper and use the same invariant helpers after the trace. The invariant helpers must be real code in the committed test file, and every cancellation enum above must be mechanically injectable.

- [x] **Step 2: Author the stateful machine at `tests/property/web/composer/test_compose_loop_invariants.py`.**

Uses Hypothesis's `RuleBasedStateMachine`. After each trace, the machine asserts the §8.3.2 post-conditions: forward-direction, backward-direction (via the schema-level SQL predicate), ordering & uniqueness, redaction, cancellation-specific, audit-failure primacy, OTel counter post-conditions.

- [x] **Step 3: Author the schema-level integration test at `tests/integration/web/test_inv_audit_ahead_backward.py`.**

```python
"""INV-AUDIT-AHEAD bidirectional schema-level test (spec §4.1.2 /
§5.3 / §8.3.2 closes QA F-1)."""
from __future__ import annotations

from sqlalchemy import text


def test_no_state_row_without_tool_row(populated_audit_db):
    """The post-condition is now a pure SQL predicate."""
    with populated_audit_db.connect() as conn:
        result = conn.execute(text(
            "SELECT cs.id FROM composition_states cs "
            "LEFT JOIN chat_messages cm "
            "  ON cm.composition_state_id = cs.id AND cm.role = 'tool' "
            "WHERE cs.provenance = 'tool_call' AND cs.version > 0 "
            "  AND cm.id IS NULL"
        ))
        orphans = result.fetchall()
    assert orphans == []
```

The fixture `populated_audit_db` exercises the compose loop end-to-end with mixed tool successes/failures/cancellations so the predicate has meaningful surface to evaluate.

- [x] **Step 4: Run the full property + integration surface.**

```bash
.venv/bin/python -m pytest tests/property/web/composer/ tests/integration/web/test_inv_audit_ahead_backward.py -v
```

Expected: PASS. The property test runs the §8.3.2 OTel-counter post-conditions across the campaign.

- [x] **Step 5: Commit.**

```bash
git commit -am "test(integration): CL-PP-9/10/12/13 + property + schema-level backward-direction (composer-progress-persistence phase 3)"
```

---

## Task 13: OTel alert routes, dashboard, runbook for Tier-1 audit counters

Overview done-when item 5: "before Phase 3 ships the Tier-1 audit counters have an alert route, dashboard visibility, and runbook entry."

`composer.audit.tool_row_tier1_violation_total` is the critical counter — non-zero on any production trace indicates a Tier-1 audit-write failure. `composer.audit.tool_row_integrity_violation_total` is the second-critical counter — non-zero indicates a schema-constraint breach (writer_principal misuse, sequence-no collision, etc.). The three remaining counters (`audit_grade_view_total`, `tool_row_persist_failed_during_unwind_total`, `tool_call_cap_exceeded_total`) are operational signals, not Tier-1 invariants.

**Files:**
- Inspect: `config/` and `infra/` directories for existing alert/dashboard plumbing
- Modify or create: alert config and dashboard config — exact files depend on existing infra
- Create: `docs/runbooks/audit-tier1-violation.md`

- [x] **Step 1: Survey existing alert/dashboard infra.**

```bash
find config infra -type f \( -name '*.yml' -o -name '*.yaml' -o -name '*.json' \) 2>/dev/null \
  | xargs grep -l "composer\.\|sessions\.\|alertmanager\|grafana" 2>/dev/null
ls docs/runbooks/ 2>/dev/null
```

If existing infra is found, append the new alert routes + dashboard panel + runbook entry to the existing files. If no existing infra is found, this task **scopes down**:

  - Wire the counters in `_SessionsTelemetry` (Tasks 2 + 11 already do this — confirm).
  - Create `docs/runbooks/audit-tier1-violation.md` in this Phase 3 PR. The runbook must cover `tool_row_tier1_violation_total`, `tool_row_integrity_violation_total`, and `audit_access_log_write_failed_total`; include operator triage steps, expected response contracts, and the rule that transcript tool rows must not be returned when audit logging fails.
  - File follow-up Filigree tickets only for alert-route and dashboard deliverables if no deployable infra exists in the repo; capture the ticket IDs.
  - Cite those ticket IDs in the Phase 3 PR description with the heading "Pre-merge prerequisites for production deploy" — the operator confirms the tickets are tracked before merge but they do not block the PR review.

The overview done-when requires that the artifacts exist before Phase 3 **ships** (production deploy). The runbook exists in this PR; only alert/dashboard wiring may be ticketed if the repository has no place to land it.

- [x] **Step 2: Commit (or comment in PR if scoped to tickets).**

```bash
git commit -am "ops(audit): alert routes / dashboard / runbook entries for Tier-1 audit counters (composer-progress-persistence phase 3)"
```

Or, if scoped to follow-up tickets, no commit — just the PR-description block citing the ticket IDs.

---

## Task 14: Overview update — rev-5 / manifest-keyed framing (standalone docs commit)

The overview at `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-overview.md` still references rev-4 / `Sensitive[T]` in five places. This is a documentation sync, not an engineering dependency for the compose-loop PR. Run it as a standalone docs-only commit outside the engineering PR after normal docs verification passes.

**Files:**
- Modify: `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-overview.md`

- [x] **Step 1: Update the spec-revision pointer.**

Line 5: "revision 4, 2481 lines" → "revision 5, 3608 lines (manifest-keyed redaction)".

- [x] **Step 2: Update the architecture summary.**

Line 9: "Phase 2 adds the type-driven redaction primitive (`Sensitive[T]`) plus the legacy declarative escape valve" → "Phase 2 adds the manifest-keyed redaction primitive (`ToolRedaction` dataclass + module-level `MANIFEST`), promotes ~6–8 sensitive-touching tools to type-driven `argument_model` entries via `Sensitive[T]` annotations, and retains the declarative `ToolRedactionPolicy` shape for the remaining ~29–31 tools.".

- [x] **Step 3: Update the Phase 2 row in the phase plans table.**

Line 20: rewrite to "`ToolRedaction` manifest dataclass, module-level `MANIFEST` keyed by tool name, `Sensitive[T]` promotion wave for ~6–8 tools, declarative `ToolRedactionPolicy` + `HandlesNoSensitiveDataReason` for the remaining ~29–31 tools, shared traversal iterator, `RedactionTelemetry` Protocol, four-assertion adequacy guard, broadened policy-hash snapshot, label-gate CI step.".

- [x] **Step 4: Update the Phase 3 row.**

Line 21: keep largely as-is; the existing description is rev-5-correct (compose-loop integration, tool-call cap, failed_turn, include_tool_rows). No edit required — confirm and move on.

- [x] **Step 5: Update the cross-phase dependencies.**

Line 27: "Phase 3 depends on Phase 2's redaction primitives (the compose loop uses `redact_tool_call` and `lookup_tool_class`)." → "Phase 3 depends on Phase 2's redaction primitives (the compose loop uses `redact_tool_call_arguments`, `redact_tool_call_response`, and the module-level `MANIFEST`; the rev-4 `lookup_tool_class` helper is removed per rev-5 §5.7.5).".

Line 29: drop the rev-4 supersession-list bullets that referred to fictional rev-4 symbols (`_StatePayload.version`, etc.); they're already handled in the spec body and the Phase 1A/1B/1C plans.

- [x] **Step 6: Commit outside the engineering PR.**

```bash
git commit -am "docs(plan): overview reflects rev-5 manifest-keyed framing + Phase 3 plan landing (composer-progress-persistence phase 3)"
```

---

## Task 15: Session-end follow-up — OQ-3 Filigree ticket for `chat_messages` integrity-hash chain

Spec §10 OQ-3 and §11 cross-phase considerations: file a Filigree ticket for the integrity-hash chain (mechanism sketched in spec §10), cite the ID in the Phase 3 PR description. The integrity-hash chain is out of scope for Phase 3 itself. This is a session-end tracking action, not an implementation task; do it after code gates are green and before the operator PR-open summary.

- [x] **Step 1: File the ticket.**

```bash
filigree create "chat_messages integrity-hash chain — composer-progress-persistence OQ-3" \
  --type=task --priority=2 \
  --label-prefix=composer-progress-persistence
```

The ticket body should reference spec §10 OQ-3 and §11; include a short summary of the mechanism (per-row hash chain seeded by the previous row's hash, anchored at the session row).

- [x] **Step 2: Capture the ticket ID for the PR description.**

```bash
filigree show <new-ticket-id>
```

Cite the ticket ID in the Phase 3 PR description under the "Filed follow-ups" section.

- [x] **Step 3: No commit (Filigree state is outside the repo).**

Filed follow-up: `elspeth-dbeb1fbbe9`.

---

## Task 16: Final Phase 3 CI run + PR

- [x] **Step 1: Run the full Phase 3 test surface.**

```bash
.venv/bin/python -m pytest \
  tests/unit/web/composer/test_compose_loop_tool_call_cap.py \
  tests/unit/web/composer/test_compose_loop_persistence.py \
  tests/unit/web/composer/test_audit_failure_primacy.py \
  tests/unit/web/sessions/test_count_tool_responses_for_assistant.py \
  tests/unit/web/sessions/test_record_audit_grade_view.py \
  tests/unit/web/sessions/test_messages_route_include_tool_rows.py \
  tests/integration/web/test_compose_loop_failed_turn_field.py \
  tests/integration/web/test_inv_audit_ahead_backward.py \
  tests/integration/pipeline/test_composer_llm_eval_characterization.py \
  tests/property/web/composer/ \
  -v
```

Expected: PASS for all. CL-PP-11 (commit `eca88974`) runs only on the Docker-enabled CI lane via the `testcontainer` marker.

Verification run on this branch: the plan's `tests/integration/web/test_compose_loop_failed_turn_field.py` path no longer exists, so the live equivalent used `tests/unit/web/sessions/test_failed_turn_handlers.py` and `tests/unit/web/test_composer_exception_handlers.py` alongside the other listed paths. Result: 74 passed.

- [x] **Step 2: Run static-analysis gates.**

```bash
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m scripts.check_contracts
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check
```

Expected: all green.

Verification run on this branch: mypy, ruff, `scripts.check_contracts`, tier-model enforcement, and freeze-guard enforcement all passed. The freeze-guard command requires the current `--root src/elspeth --allowlist config/cicd/enforce_freeze_guards` arguments.

- [x] **Step 3: Verify counter post-conditions match the §1.4 SLO claims.**

```bash
.venv/bin/python -m pytest tests/property/web/composer/ -v -k "otel_counter_postconditions"
```

Expected: PASS. `composer.audit.tool_row_tier1_violation_total == 0` across the property-test campaign (the counter only increments when the test explicitly injects a Tier-1 fault and asserts the increment).

- [x] **Step 4: Re-run signature-change sweeps before final PR summary.**

```bash
rg -n "ComposerServiceImpl\\(" src tests
rg -n "Does NOT depend on SessionService|depend on SessionService \\(seam contract B\\)" \
  src/elspeth/web/composer/protocol.py
rg -n "class ComposerResult|FailedTurnMetadata|count_tool_responses_for_assistant_async|record_audit_grade_view_async" \
  src tests
rg -n "def (build_test_sessions_service|composer_service_with_real_sessions|composer_service_without_sessions_service|fake_composer_service|fake_llm_emitting_n_tool_calls|fake_llm_two_tool_calls|fake_llm_three_tool_calls|inject_commit_OperationalError|inject_non_compose_loop_AuditIntegrityError|inject_audit_access_log_write_failure|composer_test_client|session_with_pending_compose_request|populated_audit_db)" tests
rg -n "_run_one_turn_for_test" src tests
! rg -n "getattr\\(exc, \"failed_turn\"|hasattr\\(exc, \"failed_turn\"" src tests
```

Expected: no required-constructor-parameter fallout remains, every red-test fixture/helper resolves to a definition, `_run_one_turn_for_test` is defined before use, and there is no defensive `getattr`/`hasattr` for `AuditIntegrityError.failed_turn`. Any remaining `ComposerServiceImpl(` callers without `sessions_service=` are constructor-only/non-persistence callers covered by the sentinel default, or tests that deliberately assert `RuntimeError("sessions_service not wired")` at first persistence use. The seam-contract grep returns no stale "does not depend" wording after Task 6.

- [x] **Step 5: Surface to operator for PR-open decision. Do NOT run `gh pr create`.**

Per Phase 2 rev-5 BLOCKER B4 closure pattern (per `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-2-redaction.md` "Phase 2 done-when removes PR-open from scope; plan rewrite ends at 'gate green; await operator PR-open instruction'"), and per project memory `feedback_default_to_worktree.md` (worktree-default policy revision 2026-05-11) and `project_phase2_plan_review_verdict.md`: Phase 3 implementation ends at "gate green; await operator PR-open instruction." The implementer captures the readiness state in the conversation and stops; the operator decides when (and whether) to open the PR. (Rev-1 of this plan ran `gh pr create` unconditionally here; that was a re-introduction of the Phase 2 rev-1 BLOCKER B4 pattern. Rev-2 removes it.)

The implementer surfaces the following to the operator after the gate runs are green:

> Phase 3 implementation complete on `feat/composer-progress-persistence-1a` at HEAD `<commit-sha>`. Gate state:
>
> - Unit: `<N>k` pass, `<m>` skipped, `<x>` xfailed. Integration: `<M>` pass (Docker-disabled lane). Property: `<P>` pass.
> - mypy / ruff / contracts / tier-model / freeze-guards: all clean.
> - CL-PP-11 testcontainer scenario: gates on Docker-enabled CI lane only; surfaced via `pytest -m testcontainer` for the operator to spot-check locally.
>
> Phase 3 ships:
>
> - Compose-loop rewrite to the §5.2.1 single-sync-block shape: gather tool outcomes async, redact via the Phase 2 manifest walker (`MANIFEST` / `redact_tool_call_arguments` / `redact_tool_call_response` / `RedactionTelemetry`), dispatch ONE `persist_compose_turn_async` per turn; raise `ComposerPluginCrashError` after audit commit; let `AuditIntegrityError` propagate from the sync worker.
> - `ComposerServiceImpl` `SessionServiceProtocol` dependency wired through production app creation with sentinel-default constructor semantics and first-use `RuntimeError("sessions_service not wired")` guard for unconfigured persistence paths.
> - Per-turn tool-call cap (default 16, env-tunable via `MAX_TOOL_CALLS_PER_TURN`); new `tool_call_cap_exceeded` reason code on `ComposerConvergenceError`; new `composer.tool_call_cap_exceeded_total` counter.
> - `failed_turn` field on 422/500 response bodies emitted by `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure`; four-key shape Phase 4 consumes (`assistant_message_id`, `tool_calls_attempted`, `tool_responses_persisted`, `transcript_url`).
> - `include_tool_rows` query parameter on `GET /api/sessions/{sid}/messages`; new response-row fields `tool_call_id`, `parent_assistant_id`, `sequence_no`.
> - Audit-grade access logging via new `record_audit_grade_view` helper writing to the Phase-1-provisioned `audit_access_log` table; new `composer.audit.audit_grade_view_total` counter.
> - `count_tool_responses_for_assistant` read helper on SessionsService for the route-layer SELECT (spec §6.1 read-consistency note).
> - Legacy `BufferingRecorder.add_message` drain inside `_compose_loop` deleted (no-legacy-code policy).
> - Integration tests CL-PP-1..10b, 12, 13 (extending the existing characterization surface; CL-PP-11 already shipped in commit `eca88974`).
> - Property test (§8.3) with full strategy contracts (§8.3.1), §8.3.2 post-conditions, and the schema-level backward-direction predicate.
> - Overview updated to rev-5 / manifest-keyed framing.
>
> Open Phase 3 follow-ups filed during implementation:
>
> - OQ-3 `chat_messages` integrity-hash chain — Filigree ticket: `<id>`.
> - Pre-merge-to-release prerequisites (per overview done-when item 5): alert route, dashboard, runbook entries — Filigree tickets `<id-1>`, `<id-2>`, `<id-3>`.
>
> Ready for operator PR-open decision. The PR description body (below) is pre-drafted from this plan's Summary section; the operator runs `gh pr create` with the desired base and review pool when ready.

**Pre-drafted PR body (for operator use; do NOT invoke `gh pr create` from this step):**

```text
## Summary

Phase 3 of composer-progress-persistence (spec §11):

- Compose-loop rewrite to the §5.2.1 single-sync-block shape: gather tool outcomes async, redact via the Phase 2 manifest walker, dispatch ONE `persist_compose_turn_async` per turn; raise `ComposerPluginCrashError` after audit commit; let `AuditIntegrityError` propagate from the sync worker.
- `ComposerServiceImpl` receives `SessionServiceProtocol` via sentinel-default DI: production app wiring passes the real service; non-persistence constructor callers remain valid; persistence callers without wiring fail at first use with `RuntimeError("sessions_service not wired")`.
- Per-turn tool-call cap (default 16, env-tunable via `MAX_TOOL_CALLS_PER_TURN`); new `tool_call_cap_exceeded` reason code on `ComposerConvergenceError`; new `composer.tool_call_cap_exceeded_total` counter.
- `failed_turn` field on 422/500 response bodies emitted by `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure`; carries `assistant_message_id`, `tool_calls_attempted`, `tool_responses_persisted`, `transcript_url`.
- `include_tool_rows` query parameter on `GET /api/sessions/{sid}/messages`; new response-row fields `tool_call_id`, `parent_assistant_id`, `sequence_no`.
- Audit-grade access logging via new `record_audit_grade_view` helper writing to the Phase-1-provisioned `audit_access_log` table; new `composer.audit.audit_grade_view_total` counter.
- `count_tool_responses_for_assistant` read helper on SessionsService for the route-layer SELECT (spec §6.1 read-consistency note).
- Legacy `BufferingRecorder.add_message` drain inside `_compose_loop` deleted (no-legacy-code policy).
- Integration tests CL-PP-1..10b, 12, 13 (extending the existing characterization surface; CL-PP-11 already shipped in commit `eca88974`).
- Property test (§8.3) with full strategy contracts (§8.3.1), §8.3.2 post-conditions, and the schema-level backward-direction predicate.
- Overview updated to rev-5 / manifest-keyed framing.

## Spec

`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` revision 5. Reviewer-finding traceability table at §12.2.

## Depends on

- Phase 1 PRs (1A, 1B, 1C) — merged on this branch.
- Phase 2 PR (manifest dispatch + `Sensitive[T]` promotion wave + adequacy guard + F1–F6 follow-ups) — landed before current `RC5.2`; rev-3 baseline is `47efb073722c7628d7f8d84ca13127cd50f6f6c8`.

## Out of scope (later phases)

- Frontend recovery panel (Phase 4).
- Integrity-hash chain (OQ-3 follow-up — Filigree ticket cited below).

## Filed follow-ups

- OQ-3 — `chat_messages` integrity-hash chain — Filigree ticket: [cite ID]
- Pre-merge-to-release prerequisites (per overview done-when item 5):
  - Alert route for `composer.audit.tool_row_tier1_violation_total` — Filigree ticket: [cite ID]
  - Grafana dashboard panel for Tier-1 audit counters — Filigree ticket: [cite ID]
  - Runbook: delivered in this PR at `docs/runbooks/audit-tier1-violation.md`

## Test plan

- [x] `test_compose_loop_tool_call_cap.py` — cap enforcement + counter
- [x] `test_compose_loop_persistence.py` — Step 1/2/3 unit surfaces
- [x] `test_audit_failure_primacy.py` — `AuditIntegrityError` vs `ComposerPluginCrashError` disposition
- [x] `test_count_tool_responses_for_assistant.py` — read helper + async dispatcher
- [x] `test_record_audit_grade_view.py` — access-log write + counter + pinned writer_principal
- [x] `test_composer_exception_handlers.py` — `AuditIntegrityError`, `StaleComposeStateError`, and audit-access-log fail-closed route handlers
- [x] `test_messages_route_include_tool_rows.py` — endpoint behaviour + new response columns
- [x] `test_compose_loop_failed_turn_field.py` — `failed_turn` shape across all three `_handle_*` helpers
- [x] `test_inv_audit_ahead_backward.py` — schema-level backward-direction post-condition
- [x] `test_composer_llm_eval_characterization.py` — CL-PP-1..10b, 12, 13
- [x] `test_compose_loop_invariants.py` — property test, §8.3.2 OTel counter post-conditions
- [x] mypy / ruff / tier-model / freeze-guard / contracts CI green
- [x] Docker-enabled CI lane runs CL-PP-11; non-Docker lanes deselect via `-m "not testcontainer"`; `ci-success` aggregation job passes
```

---

## Phase 3 Done When

Engineering tasks 0-13 and 16 are complete, with Task 14 handled as a standalone docs commit and Task 15 handled as a session-end Filigree follow-up. Specifically (closing the spec §11 done-when):

1. ✅ All §8.2 CL-PP-* scenarios pass — CL-PP-1, 2, 3, 4a, 4b, 4c, 4d, 5, 6, 7, 8, 9, 10a, 10b, 12, 13 in `test_composer_llm_eval_characterization.py`; CL-PP-11 in `test_compose_loop_concurrent_sessions.py` (Phase 1, commit `eca88974`).
2. ✅ §8.3 property test passes, including the schema-level backward-direction post-condition and the §8.3.2 OTel counter post-conditions.
3. ✅ `tests/integration/web/test_inv_audit_ahead_backward.py` schema-level backward-direction test passes.
4. ✅ `composer.audit.tool_row_tier1_violation_total == 0` across the property-test campaign (only increments when the test explicitly injects a Tier-1 fault).
5. ✅ Per-turn tool-call cap enforced before any tool execution; `composer.tool_call_cap_exceeded_total` matches injected counterexamples.
6. ✅ `failed_turn` field present on every 422/500 response from the three `_handle_*` helpers; `tool_responses_persisted` matches the actual DB state computed via `count_tool_responses_for_assistant`.
7. ✅ `include_tool_rows=true` triggers `audit_access_log` emission with `writer_principal='audit_grade_view'`; `composer.audit.audit_grade_view_total` counter matches request count.
8. ✅ Legacy `BufferingRecorder.add_message` drain inside `_compose_loop` deleted.
9. ✅ `ComposerServiceImpl` constructor remains safe for existing constructor-only/non-persistence callers: `sessions_service` is sentinel-defaulted, production app wiring passes the real service, persistence-path tests without wiring fail with `RuntimeError("sessions_service not wired")`, and the final signature-change sweep is recorded.
10. ✅ Composer seam contract B documentation in `src/elspeth/web/composer/protocol.py` no longer claims the composer has no `SessionService` dependency.
11. ✅ Every red-test fixture/helper used by this plan is defined or cited by Task 0; `_run_one_turn_for_test` exists and exercises `_require_sessions_service()`.
12. ✅ `AuditIntegrityError.failed_turn` is a typed nullable field; compose-loop-origin failures populate it, non-compose-loop-origin failures return the documented `diagnostic="no_failed_turn_metadata"` degraded body, and no handler path uses `getattr`/`hasattr` fallback for the field.
13. ✅ Any new `_compose_loop` await point added by this phase is represented in the cancellation enum/injection table, or documented as outside compose-turn cancellation semantics.
14. ✅ `enforce_tier_model.py`, `enforce_freeze_guards.py`, `check_contracts`, mypy, ruff CI green.
15. ✅ Overview reflects rev-5 / manifest-keyed framing.
16. ✅ OQ-3 Filigree ticket filed and cited in the PR description.
17. ✅ Tier-1 audit runbook delivered at `docs/runbooks/audit-tier1-violation.md`; alert/dashboard prerequisites filed as Filigree tickets and cited in the PR description unless existing infra was present and updated in this PR.

Phase 4 begins after this PR merges. Phase 4 reads the `failed_turn` field this phase delivered and the `include_tool_rows=true` transcript endpoint this phase opened.

---

## Appendix A — Phase-2-as-shipped traceability (added rev-2)

Phase 2 shipped before the rev-3 `RC5.2` baseline with 39 manifest entries plus the F1–F6 follow-up sweep. The type-driven/declarative split is intentionally approximate in the phase overview unless recounted live during the same task. Three deliverables from that sweep are load-bearing for Phase 3 and not obvious from the spec body — this appendix surfaces them so an implementer at task-execution time doesn't have to reverse-engineer the precedents.

### A.1 F2 — ARG_ERROR `validation_errors` field (commit `70424cc1`)

F2 added two helpers that Phase 3 inherits:

- `canonicalize_pydantic_cause(exc: BaseException | None) -> list[dict[str, Any]] | None` in `src/elspeth/web/composer/audit.py`. Leak-safe Pydantic-`ValidationError` canonicalisation: `loc` / `msg` / `type` only; `input` / `url` / `ctx` stripped (Tier-3 sensitive material). Returns `None` when the cause is not a `ValidationError` (recording an empty list has no audit value).
- `_arg_error_payload(exc: ToolArgumentError, tool_name: str) -> Mapping[str, Any]` in `src/elspeth/web/composer/service.py`. Module-level (not loop-local) so it is testable in isolation. Builds the structured ARG_ERROR payload with two fields: `error` (operator-safe, LLM-facing) plus `validation_errors` (present iff `exc.__cause__` is a `ValidationError`). Populates `result_canonical` for the audit row AND serves as the LLM-facing `role=tool` message content.

**Implications for Phase 3:**

- Task 4's walker call on the ARG_ERROR `_ToolOutcome` payload sees a dict that may include `validation_errors`. The Phase 4 recovery panel will display these field-name details to help operators recover from Pydantic-rejected tool arguments; Phase 3's persistence must preserve the field through the redaction walker.
- The existing `except ToolArgumentError as exc:` arm in `_compose_loop` already calls `_arg_error_payload(exc, tool_name)`; Task 4's accumulator must produce the same payload shape for the `error_class="ToolArgumentError"` `_ToolOutcome` branch, otherwise the persisted record diverges from the LLM-visible one.
- §12.2 BLOCKER_A traceability — the cross-phase contract integrity guarantee at the persistence boundary that F2 closes for ARG_ERROR specifically; Phase 3's walker call is the downstream consumer of F2.

### A.2 F3 — closed-list Sensitive markers

F3 replaced overbroad `Sensitive` markers on `routes` and `trigger` fields with closed-list types. This is a Phase 2 hardening; Phase 3 does not modify or extend it. The implication for Phase 3 is purely citation: when reviewers reference the `Sensitive[T]` annotation surface, the closed-list types are the canonical shape; the older "overbroad" form is no longer in use.

### A.3 AST-fingerprint observation `elspeth-obs-02a0002fae`

`scripts/cicd/enforce_tier_model.py` fingerprints findings by AST `body[N]` index. Any line-shifting edit to `src/elspeth/web/composer/service.py` rotates downstream fingerprints in `config/cicd/enforce_tier_model/`. F2 (commit `70424cc1`) established the mitigation pattern: append new module-level defs at file tail with a "do-not-move" header comment so the AST indices of existing definitions remain stable.

**Implications for Phase 3:**

- Task 3's `_PerTurnPersistContext` dataclass (if introduced) lands at `service.py` tail with the same "do-not-move" header. Zero allowlist refresh.
- Task 10's `record_audit_grade_view` writer lands at `web/sessions/service.py` tail with the same header. Zero refresh.
- Task 11's protocol extension appends at protocol-class tail. Minimal refresh.
- **Tasks 3/4/5/6/7 modify the `_compose_loop` body — these edits CANNOT be tail-appended; they must land where the loop body sits.** Expected allowlist refresh: ~40–70 entries across the loop-body tasks. Refresh is mechanical (regenerate from current findings; same-commit), not investigation; the implementer must distinguish "same finding at a new fingerprint" (refresh) from "new finding" (fix the code).
- Memory: `project_tier_model_python_version` — the worktree venv MUST be Python 3.13 to avoid the ~300 spurious tier-model FPs the version-skew issue triggers.

### A.4 Current symbol anchors (rev-3 update)

Rev-3 deliberately avoids numeric line citations for high-churn files. The implementation baseline is current `RC5.2` HEAD `47efb073722c7628d7f8d84ca13127cd50f6f6c8`; implementers should locate symbols with `rg` immediately before editing rather than trusting plan line numbers.

Use these symbol anchors:

| File | Symbol / surface |
|---|---|
| `src/elspeth/web/composer/service.py` | `ComposerServiceImpl._compose_loop(...)` |
| `src/elspeth/web/composer/service.py` | `_arg_error_payload(exc, tool_name)` helper |
| `src/elspeth/web/composer/service.py` | `ComposerPluginCrashError.capture(...)` call site inside the plugin-bug `except Exception as tool_exc` branch |
| `src/elspeth/web/sessions/service.py` | `SessionServiceImpl.persist_compose_turn(...)` sync primitive |
| `src/elspeth/web/sessions/service.py` | `SessionServiceImpl.persist_compose_turn_async(...)` async wrapper |
| `src/elspeth/web/sessions/_persist_payload.py` | `StatePayload`, `_ToolOutcome`, `RedactedToolRow`, `AuditOutcome` |
| `src/elspeth/web/sessions/models.py` | `audit_access_log_table` and `ck_audit_access_log_writer_principal` |
| `src/elspeth/web/sessions/routes.py` | `_persist_tool_invocations(...)` legacy route-layer helper |
| `src/elspeth/web/sessions/routes.py` | `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure` |
| `src/elspeth/web/sessions/routes.py` | `get_messages` endpoint for `/{session_id}/messages` |

Required pre-edit commands:

```bash
rg -n "async def _compose_loop|ComposerPluginCrashError.capture|def _arg_error_payload" src/elspeth/web/composer/service.py
rg -n "def _persist_tool_invocations|_persist_tool_invocations\\(|async def get_messages" src/elspeth/web/sessions/routes.py
rg -n "class _ToolOutcome|class RedactedToolRow|class AuditOutcome" src/elspeth/web/sessions/_persist_payload.py
```

If any symbol has moved or changed shape, update this plan section before implementation. Do not paste fresh numeric line tables back into the task body.

### A.5 Anticipated open questions (rev-1 inheritance — all resolved by spec rev-5)

The on-disk Phase 3 plan-authoring prompt at `notes/composer-phase-3-plan-prompt-2026-05-10.md` anticipated four "decisions to surface" before drafting. Rev-2 confirms all four are resolved by spec rev-5 and the current code; the implementer does NOT need to bring them back to the operator unless a contradiction surfaces during execution:

1. **`failed_turn.transcript_url` cursor shape.** Resolved: user-message ID. Reference: Phase 4 plan line ~140 (`?since=u_1&include_tool_rows=true`) — Phase 4's tests pin the contract.
2. **`provenance='tool_call'` write timing within the sync block.** Resolved: spec §5.2.2 step list — assistant row first, then per-tool (optional state row first, then tool row). All in one `engine.begin()` context. Implementation already lives in `web/sessions/service.py:SessionServiceImpl.persist_compose_turn`.
3. **Latency NFR vs CI infra.** Resolved: spec §1.4 splits sanity bound (CI, p95 ≤ 250 ms) from tight target (nightly bench, p95 ≤ 25 ms). Phase 3 done-when references the sanity bound only; the tight target is filed as tracking observation per §1.4.
4. **Cancellation arrival-time enum coverage.** Resolved: spec §5.2.4 cancellation table + §8.3.1 `st_cancellation_arrival_time` nine-value enum — exhaustive coverage of the §5.5 failure-mode rows 5–11. Task 12c's strategy contract consumes this directly.

### A.6 Phase 1B `_ToolOutcome` field check (Task 1 disposition)

Task 1 of this plan is "verify `_ToolOutcome` matches spec §5.2.1." Rev-3 confirms via direct inspection at current `RC5.2`:

- `web/sessions/_persist_payload.py` defines `_ToolOutcome(call: Any, response: Any, error_class: str | None, error_message: str | None, pre_version: int, post_version: int)`.
- Frozen, slots, `freeze_fields(self, "call", "response")` in `__post_init__`.
- Matches spec §5.2.1 verbatim. Task 1 is a verify-only no-op at execution time; no commit, no follow-up ticket.

If a future Phase 1 hygiene change adds or removes a field, Task 1's "Step 1: Confirm the field list" branch is the surface to catch it.

---

End of rev-2 plan, composer-progress-persistence Phase 3.
