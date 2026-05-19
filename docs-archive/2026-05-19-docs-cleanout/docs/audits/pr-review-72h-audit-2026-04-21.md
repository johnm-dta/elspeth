# PR Review — 72-hour Window Audit (2026-04-21)

- **Scope:** 70 commits, 359 files, +37621 / −1126 lines
- **Range:** `bd6d95c8^..HEAD` (2026-04-18 → 2026-04-21)
- **Branch:** `RC5-UX`
- **Review method:** 8 specialist agents (code-review × 2, silent-failure × 2, test-coverage × 2, type-design × 2), one pass per epic
- **Reviewer:** Claude (pr-review-toolkit) with advisor steer

The window contains two unrelated epics; findings are triaged separately because they ship on independent timelines.

- **Epic A** — ADR-010 declaration-trust framework landing (~58 commits, 2026-04-19→21). Introduces `AuditEvidenceBase`, `@tier_1_error`, `DeclarationContract` registry, Phase 2B/2C contract adopters, frozen registries at bootstrap, and new CI scanners.
- **Epic B** — Web-server hardening pass (~15 commits, 2026-04-18). Tier-1 strict response models, IDOR / TOCTOU / JWKS / path-traversal fixes, typed exceptions, atomic-rename refactor.

Reviewers were given the regression-chain signal (six fix-forward commits within the window correcting issues introduced earlier in the same window) and the live filigree observations. Scope was filtered to `src/elspeth/**`, `tests/**`, `config/cicd/**`, `scripts/cicd/**`, `.github/workflows/**` — `docs/superpowers/`, `docs/plans/`, and ADR prose were excluded as non-code.

## Reviewer Contradiction (resolved)

Epic A's code-reviewer and silent-failure-hunter reached opposite verdicts on `src/elspeth/engine/processor.py:884`. The code-reviewer called it CRITICAL (regression-chain miss); the silent-failure-hunter called it a false positive ("the severity upgrade is architecturally correct").

**Resolution in favour of the code-reviewer.** The dispositive evidence is commit `76e318ec`, which introduced `LandscapeRecordError` specifically to narrow this exact pattern at the sibling `_record_source_boundary_failure` site (`processor.py:1530`). The silent-failure-hunter evaluated the pattern in isolation without reading the fix commit. Commit-level evidence overrides pattern-in-isolation reasoning. See CR-1.

## Critical — fix before ship

### CR-1. Batch-flush recorder catch — regression-chain miss

- **Location:** `src/elspeth/engine/processor.py:877-893` (`_record_flush_violation`)
- **Commit evidence:** `76e318ec` narrowed the sibling source-boundary path to `except LandscapeRecordError`; batch-flush was not swept.
- **Pattern:**

  ```python
  except Exception as record_failure:
      raise AuditIntegrityError(
          f"Failed to record {type(violation).__name__} FAILED outcome "
          ...
      ) from record_failure
  ```

- **CLAUDE.md rule:** §Plugin Ownership — plugin/framework bugs must crash with their original type; re-wrapping as `AuditIntegrityError` destroys triage signal.
- **Fix:** narrow to `except LandscapeRecordError as record_failure:` (matches `_record_source_boundary_failure` at line 1530).
- **Filigree:** promote `obs-a37d439ab2` to an issue linked to this fix.
- **Confidence:** 95.

### CR-2. `NodeStateGuard.__exit__` — same pre-76e318ec triad, unfixed

- **Location:** `src/elspeth/engine/executors/state_guard.py:140-147` and `178-189`
- **Pattern:**

  ```python
  except contract_errors.TIER_1_ERRORS:
      raise
  except (TypeError, AttributeError, KeyError, NameError):
      raise
  except Exception as db_err:
      raise AuditIntegrityError(...) from db_err
  ```

- **Why it's still wrong:** the hand-rolled filter list is incomplete (no `ValueError`, `RuntimeError`, non-`SQLAlchemyError` DB descendants). Anything outside the filter becomes `AuditIntegrityError`.
- **Fix:** replace both blocks with `except LandscapeRecordError as db_err:`, drop the filter lists. Mirrors `76e318ec`.
- **Confidence:** 90.

### CR-3. Fabricated plugin name in aggregate audit record

- **Location:** `src/elspeth/engine/executors/declaration_dispatch.py:83-91` (`_serialize_plugin_name`)
- **Pattern:** `return name if name else type(plugin).__name__`.
- **CLAUDE.md rule:** §Data Manifesto — absence is evidence; inference is fabrication. An `AggregateDeclarationContractViolation` is Tier-1 audit evidence and must not carry a fabricated identifier.
- **Fix:** raise `FrameworkBugError` on empty `plugin.name`.
- **Confidence:** 85.

### CR-4. `SecretDecryptionError` silently bucketed as "missing secret" — cousin missed by sweep

- **Location:** `src/elspeth/web/secrets/service.py:163`
- **Pattern:** `except (SecretNotFoundError, SecretDecryptionError): return None` — the `FingerprintKeyMissingError` sibling got `_log_fingerprint_missing_rate_limited()` in commit `5f1360a4`; the decryption-failure case did not.
- **Consequence:** a master-key rotation that breaks stored secrets produces an audit trail saying "secret missing" with no operator-visible breadcrumb.
- **Fix:** symmetric rate-limited breadcrumb (mirrors the fingerprint pattern); split the `except` tuple so the breadcrumb fires only for `SecretDecryptionError`.
- **Confidence:** 90.

### CR-5. Pooling executor converts plugin bugs into row errors + leaks exception text

- **Location:** `src/elspeth/plugins/infrastructure/pooling/executor.py:334-344`
- **Pattern:** `except Exception as exc:` builds `TransformResult.error(error=f"{type(exc).__name__}: {exc}")` for anything not in `TIER_1_ERRORS`.
- **Two problems compounded:**
  1. Plugin Ownership violation — plugin bugs (`AttributeError`, `TypeError`, `KeyError`) become distinguishable from Tier-3 row failures only by error-string shape, not by type.
  2. Tier-3 payload leak — `str(exc)` interpolates LiteLLM `__cause__` chains, template-rendered row fields, and tenacity retry chains into the audit-persisted payload.
- **Fix:** drop the broad `except` entirely (typed wrappers already cover legitimate recoverable failures), or redact to `type(exc).__name__` only and narrow the catch.
- **Confidence:** 85.

## Important — should fix this cycle

### Epic A

- **I-1.** `_record_flush_violation` has no `try/except` around `_emit_token_completed` mid-loop; source-boundary sibling does. A telemetry failure aborts the per-token loop, violating the "every row reaches exactly one terminal state" invariant. `src/elspeth/engine/processor.py:894`.
- **I-2.** `_attach_contract_name` / `_attach_by_dispatcher` TOCTOU + spoofing (`src/elspeth/contracts/declaration_contracts.py:665-674, 762-773`). Single-underscore methods let any non-dispatcher caller attach an arbitrary name; the set-once guard is non-atomic. Fix: token-gate via module-private sentinel kwarg.
- **I-3.** `BoundaryInputs.row_data: Any` has no `deep_freeze` (`src/elspeth/contracts/declaration_contracts.py:297-324`). Sink-boundary contracts can receive and stash a mutable dict. Only mutable-payload hole in the new bundle set.
- **I-5.** `process_existing_row` skips source-boundary contracts on resume without documenting inherited-evidence semantics or checking `runtime_val_manifest` drift between run-1 and resume-run. `src/elspeth/engine/processor.py:1689-1730`.
- **I-7.** Hypothesis property tests vary bundle inputs but use hand-coded stub contracts with constant `applies_to` / `runtime_check`. No `@example()` seeds for any in-window regression (`76e318ec`, `08b6d4e2`, `f1a4325a`, `d5cc5b77`, `c81e0a22`). `tests/property/engine/test_declaration_dispatch_properties.py`.

### Epic B

- **I-4.** `CompositionStateResponse` five `Any | None` fields (`source`, `nodes`, `edges`, `outputs`, `metadata`) defeat the `_StrictResponse` Tier-1 contract. Persistence-side models already exist (`CompositionStateRecord.nodes: Sequence[Mapping[str, Any]]`). `src/elspeth/web/sessions/schemas.py:97-107`.
- **I-6.** `RunStatus` Literal declared in six places with partial drift-guard coverage; `LEGAL_RUN_TRANSITIONS` uses raw strings. Extract to one location, derive transitions from the Literal.
- **I-8.** Blob exception family has no common base — five sibling `Exception` subclasses. Secrets family has `SecretsError` as a catchable umbrella; blobs should mirror. `src/elspeth/web/blobs/protocol.py:154-244`.
- **I-9.** `cleanup_run` clears `_terminalized`, enabling a duplicate terminal broadcast from any future code path (stray `_on_pipeline_done` extension, rescued-from-futures, etc.). `src/elspeth/web/execution/progress.py:241-242`. Drop the `discard()`.

## Suggestions

- **Flake risks** (Epic B tests): `tests/unit/web/composer/test_tools.py:3428` `timeout=0.3` may be too tight under xdist load; `tests/unit/web/test_app.py:587` `asyncio.sleep(0.05)` before task-cancel risks scheduler-jitter false-failures.
- **OSError skill filename redaction** lacks a canary test despite the canonical pattern existing for LiteLLM. `src/elspeth/web/composer/service.py`.
- **JWKS cold-start with unreachable IdP** path not explicitly asserted to raise clean `AuthenticationError` rather than programmer-bug from `None.kid`.
- **`DeclarationContract` default no-op dispatch bodies** silently pass when decorator is on a typo'd method name; replace with `raise FrameworkBugError`.
- **`cast(frozenset[str], plugin.X)` across seven adopter contracts** — introduce `ContractablePlugin` Protocol to eliminate the cast and give mypy real coverage.
- **Multi-worker guard `except ValueError: pass`** silently fails open on unparseable `WEB_CONCURRENCY`. `src/elspeth/web/app.py:388-415`. Emit a breadcrumb or fail closed.
- **`CreateSecretResult.available: bool`** is "always True today" — per no-legacy-code policy, delete until a deferred-fingerprint mode exists.
- **Tier-1 response model escape hatch** on `ChatMessageResponse.tool_calls: Any | None` — narrow to `Mapping[str, Any] | None` at minimum.
- **`tier_registry.py:44-45`** docstring says "errors.py re-exports it for back-compat" — phrasing conflicts with §No Legacy Code. The code is a legitimate circular-import break; reword the docstring.

## Positive — strong work worth noting

- **`AuditEvidenceBase` CPython 3.13 ABC-bypass guard** (`src/elspeth/contracts/audit_evidence.py:40-54`) — inline comment names the `BaseException.__new__` fast-path mechanism.
- **`@tier_1_error(caller_module=__name__)` + CI literal-check** closes frame-offset tampering elegantly.
- **Bootstrap set-equality with self-diagnosing drift message** (`src/elspeth/engine/orchestrator/core.py:227-282`).
- **IDOR byte-parity test pattern** — `resp_a.content == resp_b.content` across independently-seeded services is the strongest possible IDOR-oracle shape. `tests/unit/web/execution/test_routes.py`.
- **Path-traversal proof-of-exploit tests** with 18 parametrised malformed IDs including fullwidth unicode homoglyphs and sentinel-file-survival assertions. `tests/unit/composer_mcp/test_session.py`.
- **`_execute_update_blob` atomic-rename refactor** — snapshot-before-mutation, `tempfile.mkstemp` in parent (same-FS `os.replace` atomicity), three typed sentinels with matching rollback discipline, unconditional `finally: unlink(missing_ok=True)`, 40-line rationale comment. `src/elspeth/web/composer/tools.py:2035-2240`.
- **`RunEvent` discriminated union with import-time drift assertion** — couples `_EVENT_TYPE_TO_DATA_TYPE` dict keys to the Literal via `get_args` at module load. Exemplary offensive programming. `src/elspeth/web/execution/schemas.py:221-226`.
- **Allowlist discipline** — every new tier-model entry carries multi-line `reason:`/`safety:` justification; `enforce_tier_1_decoration/errors.yaml` has `allow_classes: []`; `enforce_contract_manifest` has zero allow_hits.
- **ADR-010 invariant harness drives live registry, not a hand-curated list** (`tests/invariants/test_contract_negative_examples_fire.py:26-68`), with a non-empty-at-start guard (`tests/invariants/conftest.py:36-52`) defeating the silent-0-tests vector.
- **JWKS narrowing** (`src/elspeth/web/auth/oidc.py:165-276`) — `(httpx.HTTPError, httpx.InvalidURL, ValueError)` replaces the historical `(HTTPError, KeyError, ValueError, TypeError, AttributeError)` with each catch justified inline, cold-start throttle, shape-validators-before-cache.

## Observation disposition

| Observation | Disposition | Rationale |
|---|---|---|
| `obs-6eb2461e06` SinkRequiredFieldsContract on CSVSource | RESOLVED | Current `plugin_roles.py` uses MRO `write`/`flush` probe; `applies_to(CSVSource)` returns False cleanly. |
| `obs-a37d439ab2` processor.py:884 broad catch | CONFIRMED → promote to issue (CR-1) | Unfixed; regression-chain miss. |
| `obs-92573630e8` property-test field filter | RESOLVED | `_EXCLUDED_ROW_API_NAMES` now includes `contract` and `to_dict`. |
| `obs-15019a87f4` rag test flake under xdist | LIKELY STALE | No shared fixtures or I/O; dismiss unless fresh flap logs surface. |
| `obs-8171cf1c7b` example cleanup path | Unchanged (out of scope) | Still tracked against `elspeth-db55ff6a06`. |
| `obs-689f95d5c9` stale YAML fixtures | Unchanged (out of scope) | Separate pre-existing benchmark failure. |
| `obs-a7476733da` `codex_bug_hunt --branch` UX | Unchanged (out of scope) | Tooling UX, not code review scope. |

## Recommended Action Order

1. **Ship CR-1 + CR-2 together** (same regression-chain pattern, same `LandscapeRecordError` fix). Sweep `src/elspeth/engine/` one more time for `except TIER_1_ERRORS: raise; except (…): raise; except Exception → AuditIntegrityError` — the pre-76e318ec template may exist in one more place.
2. **CR-4 + CR-5** (typed-exception asymmetries in the web stack).
3. **CR-3** (small, localised; raise on empty plugin name).
4. **I-1 / I-2 / I-3 / I-5 / I-7** — ADR-010 follow-up bundle.
5. **I-4 / I-6 / I-8 / I-9** — web-stack polish; opportunistic when touching those files.

Epic A and Epic B findings are on independent code paths and independent release timelines; triage separately.

## Appendix — Commits Reviewed

### Epic A (c633a36b^..HEAD, 58 commits)

Representative: `76e318ec` fix audit and schema contract regressions; `08b6d4e2` fix declaration-trust contract propagation; `f1a4325a` preserve ADR-013 input field attribution; `8c4bfde8` implement phase 2c boundary contracts; `56e003a0` Complete Phase 2B declaration-trust rollout; `85dfa660` land phase 2b bootstrap and contract adopters; `abf4b05b` Hypothesis property tests for ADR-010 dispatch surfaces; `009b6009` ADR-010 §Semantics amendment; `51206684` freeze declaration + tier registries at bootstrap; `4b035157` add DeclarationContract protocol + registry; `9f2cea71` add AuditEvidenceBase nominal ABC; `0d5c9fb1` add @tier_1_error factory; `2ba34c2b` land ADR-009 pass-through pathway fusion; `32921388` close elspeth-87f6d5dea5 pass-through contract propagation.

### Epic B (bd6d95c8^..c633a36b^, 15 commits)

`bd6d95c8` tighten JWKS catch, secret-resolve TOCTOU, async crash persist; `6ec5da51` close update_blob concurrency race + quota rollback divergence; `9b428112` coalesce progress broadcast, bound skill loader, lock in schema detection; `f1953aed` enforce Tier 1 strict response models; `5f1360a4` typed exceptions + eager fingerprint + request-id correlation; `0d325305` couple status Literals to route guard; `872a9af3` close /execute IDOR oracle + five information-leak paths; `302dd34d` close six P2 hardening gaps; `b83dced4` comment rot, test flake, ge=1 JWKS floor, assert→raise; `92f7537a` Tier-1 asserts, plugin catch, fingerprint signal; `5c67ab53` Tier-1 redaction + frozen-field type lies; `ee5126d8` canonical exc_class + regression tests; `186f1b8c` type/immutability tightening + test hardening; `05548311` composer blob tools bypass Tier-1 guards; `e3010bc3` SessionManager path traversal.
