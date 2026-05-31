# RC5.2 Tier 1 Spine — Consolidated Review Findings

**Generated:** 2026-05-30
**Scope:** L0 `contracts/` + L1 `core/` + L2 `engine/` (the audit-critical spine), `main...RC5.2`
**Method:** 14 review agents across 3 clusters — Python engineer, solution architect, systems thinker on every cluster; + type-design (A), silent-failure + embedded-DB (B), silent-failure + determinism (C). Each agent given the full ELSPETH doctrine preamble (trust tiers, no-defensive, crash-loud audit, deep_freeze, layer model, no-legacy).

**Headline:** The spine is **healthy and, in places, hardened by this branch** — the orchestrator decomposition is behavior-preserving and RC5.2 *closes* two pre-existing silent-failure holes (coalesce cleanup, processor terminal-outcome recording). **No P0 found.** The real risks cluster in two places: **(1) resume/replay on fan-out (fork/expand) topologies**, and **(2) a Postgres-only schema-staleness gap**. A handful of doctrine items (no-legacy shims, slog primacy) and one real frozen-dataclass bug round it out.

> Confidence note: every finding below was verified against source by at least one agent; convergence count = how many independent agents flagged it. The advisor (second-model pass) was rate-limited during the run, so severities are the agents' own calibration, reconciled here.

---

## P1 — Address before merge

### F1. Resume re-emits already-completed branches of forked/expanded rows *(silent audit corruption)*
**Where:** `engine/orchestrator/resume.py:171-204`, `engine/processor.py:1962-1978` (`process_existing_row`), driven by `core/checkpoint/recovery.py:327-475` (`get_unprocessed_rows`).
**Flagged by:** determinism-engine (P1). Related root cause to F2.
**What:** A row that forked into N children (or expanded into M outputs) where *some* children recorded terminal outcomes before a crash, but at least one did not, is returned as "unprocessed." `process_existing_row` mints **one fresh token and re-traverses the DAG from the source node** — re-emitting the already-completed branches with new `uuid4` token identities. There is no idempotency guard on `record_token_outcome` or sink writes, so the audit trail silently accumulates duplicate, contradictory lineage under one `row_id`. **Linear pipelines are safe** (single leaf token); the hazard is specific to fork/expand topologies.
**Why it matters:** This is exactly the legal-record corruption the resume contract exists to prevent — an auditor querying that `row_id` sees more terminal outcomes than leaf tokens.
**Severity hinge — RESOLVED 2026-05-30 (precondition CONFIRMED; F1 stands as P1).** Investigated from primary source:
- **Fork+checkpoint coexist by design.** `recovery.py:327-475` (`get_unprocessed_rows`) has an explicit "CORRECT SEMANTICS FOR FORK/AGGREGATION/COALESCE RECOVERY" block and a documented prior fix (`P2-2026-01-29-recovery-skips-partial-forks`) that *deliberately* changed selection to "ALL non-delegation leaf tokens must be terminal" — so a partial-fork row (child A done, child B crashed) **is intentionally returned for reprocessing.** No DAG rule forbids fork+checkpoint.
- **Checkpoints fire after every sink write** (`core.py:578-600`; `_maybe_checkpoint` called from `checkpoint_after_sink`, `frequency` 0/1 both checkpoint per sink token), so a crash-window exists between branch-A's sink+checkpoint and branch-B's completion. (Checkpointing is opt-in — `core.py:563` `checkpoint_config.enabled` — so exposure is limited to checkpoint-enabled fork/expand pipelines that are resumed after a crash.)
- **Reprocessing re-emits completed branches.** `processor.py:1962-1978` (`process_existing_row`) mints **one new token and restarts the whole row from source** (`_record_source_and_start_traversal`) — no logic skips already-terminal branches. The row re-forks to *all* configured branches, so the already-complete branch is re-emitted (new token_id, duplicate sink write) under the same `row_id`.
- **The double-emission is UNTESTED.** `tests/property/audit/test_fork_join_balance.py:608` (`test_partial_fork_detected_by_recovery`) asserts only that the row is *detected* as unprocessed (line 731); it deletes only one branch's outcome (the surviving branch's outcome stays), never resumes the run, and never checks for duplicate outcomes. Detection is blessed; reprocessing semantics are not.

**Net:** real, reachable P1 for checkpoint-enabled fork/expand pipelines resumed after a crash. Not a hypothetical.
**Fix direction:** mid-DAG resume (terminalize orphaned sibling tokens, resume only incomplete sub-paths) **or** per-(row_id, node_id, branch) idempotent outcome/sink writes. Either way, extend `test_partial_fork_detected_by_recovery` to actually resume and assert no duplicate terminal outcomes per row_id.

### F2. Resume-from-audit `rows_processed` diverges from the live path on fan-out
**Where:** `engine/orchestrator/run_status.py:88-135` (`derive_resume_terminal_status_from_audit`).
**Flagged by:** arch-engine (P1).
**What:** The audit-derived resume path increments `rows_processed` **per terminal leaf token**; the live path increments **per source row** (`core.py:2198`). For a fork (1 row → N children) the resumed run records `rows_processed = N` while the original recorded `1`. Terminal *status* is unaffected (`rows_processed` used only as `==0` presence), but `RunResult`/`RunSummary.total_rows` written to the legal record (and shown to operators) differ between a resumed and an uninterrupted run of the same pipeline.
**Shared gap with F1:** `tests/integration/test_adr_019_resume_counter_parity.py` asserts resume↔live parity but **only for gate/error/quarantine/diversion scenarios — no fork, expand, or coalesce case.** The fan-out shapes where both F1 and F2 live are precisely the untested ones.
**Fix direction:** decide the intended semantics of `rows_processed` (source-rows vs terminal-leaves), make both paths agree, and add fork/expand/coalesce scenarios to the parity test.

### F3. `_REQUIRED_COLUMNS` omits the two new NOT NULL `runs` columns → Postgres staleness gap
**Where:** `core/landscape/database.py` `_REQUIRED_COLUMNS`; columns `runs.openrouter_catalog_sha256` / `openrouter_catalog_source` (`schema.py`).
**Flagged by:** arch-core (P1), py-core (P2), db-core (P2) — **3-agent convergence.**
**What:** The two columns are NOT NULL with no `server_default` (correct, per the anti-fabrication doctrine — do *not* add a default). Stale-DB detection for SQLite is handled by the `user_version` epoch bump (9→10). **But the epoch gate is SQLite-only** (`database.py:497` sets `schema_epoch=0` for non-SQLite; `0` is treated as compatible). `postgresql` is a config-reachable audit backend (`config.py:1046`). For a Postgres audit DB, `_REQUIRED_COLUMNS` is the *only* staleness backstop — and these two columns were omitted from it, while their epoch-9 siblings (`llm_call_count`, `seeded_from_cache`, `cache_key`) *were* added. A stale Postgres DB therefore fails at the first `runs` insert with a raw NOT-NULL violation mid-run instead of a clean `SchemaCompatibilityError` at open.
**Fix:** add `("runs","openrouter_catalog_sha256")` and `("runs","openrouter_catalog_source")` to `_REQUIRED_COLUMNS`. (Mechanical, low-risk.) Structural follow-up: the SQLite-only epoch gate leaving Postgres reliant on `_REQUIRED_COLUMNS` is itself an integration-reality gap.

### F4. `begin_run` writes `runs` and `run_attributions` in two separate transactions *(non-atomic)*
**Where:** `core/landscape/run_lifecycle_repository.py:213-242`.
**Flagged by:** db-core (P1).
**What:** Each `execute_insert` opens its own `engine.begin()`. The `runs` insert commits, *then* `run_attributions` runs in a fresh transaction. A crash between the two commits leaves a `runs` row with **no attribution row** in the legal record — an auditor asking "who initiated run X?" gets nothing, though the system knew at write time.
**Fix:** write both rows in one `with self._db.connection() as conn:` transaction, mirroring `write_repository.record_synthesised_run` (which already does runs+nodes+rows atomically).

---

## P2 — Should fix on this branch

### F5. `SlotSpec` frozen-dataclass deep-freeze violation + discarded coercion
**Where:** `contracts/composer_slots.py` `__post_init__`; consumer `web/composer/recipes.py:139`.
**Flagged by:** py-contracts (P1), types-contracts (P2) — **2-agent convergence.**
**What:** `__post_init__` calls `_coerce_default(...)` purely for its raising side-effect and **discards the return value**. So `self.default` keeps the raw input: for a `str_list` slot, a mutable `list` on a `frozen=True, slots=True` dataclass (forbidden deep_freeze anti-pattern), *and* a shape divergence — a supplied value is coerced to `tuple` via `_coerce_slot`, but an absent optional slot returns `spec.default` (the raw `list`) at `recipes.py:139`. The author's own comment at `recipes.py:101-104` anticipates this risk; the mitigation is defeated because the coerced result is never stored.
**Fix:** `object.__setattr__(self, "default", deep_freeze(_coerce_default(...)))` in `__post_init__`.

### F6. Enum ↔ DB CHECK ↔ multi-consumer sync is comment-enforced, not mechanical *(recurring archetype)*
**Where:** `contracts/enums.py`, `contracts/composer_interpretation.py` (and the SQL CHECKs in `web/sessions/models.py`).
**Flagged by:** sys-contracts (P1), arch-contracts (P2) — **2-agent convergence.**
**What:** Closed enums and their DB CHECK constraints / dispatch consumers are kept in sync only by "must change together" comments. This exact class already fired in this project (the `InterpretationKind` / `LLM_MODEL_CHOICE` hello-world deadlock — see memory). Because `contracts/` is L0, drift here silently desyncs every downstream switch arm and the DB's accept-set.
**Leverage fix (highest in the cluster):** one parametrized test per enum that parses the SQL CHECK `IN (...)` list from `models.py` and asserts set-equality with the enum members; a second asserting every `InterpretationKind` is handled by each named consumer. Converts a recurring cross-layer drift class into a build failure.

### F7. Audit journal enrichment hand-parses SQL UPDATE text *(latent fragility)*
**Where:** `core/landscape/journal.py:341-394` (`_parse_update_statement`, `_update_columns_to_values`).
**Flagged by:** py-core (P2), db-core (P2), arch-core (P3), sys-core (P3) — **4-agent convergence.**
**What:** Reconstructs columns by string-splitting compiled SQL and assumes SET-clause bound params come first positionally (`params[:len(columns)]`). Holds for SQLAlchemy today; a SET expression containing a comma/`=` in a literal or function, or a future param-ordering change, would silently mis-map payload refs into a `calls` audit row — or (more likely) silently skip enrichment. This is the **JSONL emergency-backup journal**, not the primary Landscape write, which bounds the blast radius. Note: `journal.py:~365` also has a self-contradictory branch — `if len(params) < len(columns): return dict(zip(..., strict=True))` can only raise, never return (py-core, P3).
**Fix:** drive enrichment from the SQLAlchemy compiled statement's column set rather than re-parsing rendered SQL; at minimum assert the SET/WHERE split and fail loud.

### F8. Back-compat re-export shim violates No-Legacy policy
**Where:** `core/landscape/write_repository.py:20-23` — `SynthesisedNodeSpec` re-exported "for back-compat."
**Flagged by:** arch-core (P2), py-core (P3), sys-core (P3) — **3-agent convergence.** sys-core notes the only importer (`web/composer/tutorial_service.py`) is *new in the same series* — there is no legacy to be compatible with.
**Fix:** repoint the importer to `contracts.synthesised_audit`, delete the re-export.

### F9. `RowProcessorHandle` Protocol typed `*args/**kwargs/Any` — decomposition seam loses type safety
**Where:** `engine/orchestrator/types.py:59-92`.
**Flagged by:** py-engine (P3), arch-engine (P2) — **2-agent convergence.**
**What:** The Protocol introduced to decouple orchestrator helpers from the concrete `RowProcessor` types every method as `(*args: Any, **kwargs: Any) -> Any`, so mypy can no longer check any orchestrator→processor call — exactly the drift a decomposition is most exposed to. The *same diff* does this right elsewhere (`DAGTraversalSnapshot`, `TelemetryManagerProtocol` are fully typed). `RowProcessor` already satisfies the real signatures structurally.
**Fix:** give the Protocol methods real signatures mirroring `RowProcessor`'s public API.

### F10. `monotonic()` clock reset shifts aggregation/coalesce timeout baselines on resume
**Where:** `engine/coalesce_executor.py:321-356`, `engine/executors/aggregation.py:632-633,766-768`.
**Flagged by:** determinism-engine (P2).
**What:** Timeout baselines are reconstructed via `now - elapsed_age_seconds` against the *resuming* process's monotonic clock (not comparable across processes); downtime isn't counted. A time-triggered flush may land at a different row boundary on resume than in an uninterrupted run. Count/condition triggers unaffected.
**Fix:** document the resume timeout semantics explicitly (downtime excluded), or persist a wall-clock anchor.

### F11. `INTERPRETATION_HASH_DOMAIN_V1` retained alongside V2 — operator decision needed
**Where:** `contracts/composer_interpretation.py:337-376`.
**Flagged by:** arch-contracts (P2).
**What:** V1 kept "for historical/read-side compatibility." Either a *live* read path still computes against V1 (then that read path is the real legacy and should be cut over per delete-the-DB) or it's dead and violates No-Legacy. **Needs an explicit operator call** — there may be V1-hashed rows in a staging DB a reader must still verify.

### F12. `security.py` secret fingerprint changed HMAC-SHA256 → PBKDF2-HMAC-SHA256 *(no security agent ran — follow-up)*
**Where:** `contracts/security.py:559-598`.
**Flagged by:** py-contracts (P2, deferred), py-core (caveat). **No security-specialist agent was assigned to the contracts cluster** — this is a gap in the review pass.
**What:** A fingerprint whose purpose is cross-run secret-equality comparison silently changed derivation with no in-band scheme version tag. Under "no users yet" the portability break is acceptable, but a future cross-version comparison would report a mismatch as a secret change. Engineering suggestion: prefix the digest with a scheme tag (`pbkdf2$`).
**Action:** route through a security review (ordis-security-architect) for crypto-suitability adjudication.

---

## P3 — Lower priority (representative; see agent reports for full list)

- **F13.** `AggregationNodeCheckpoint.from_dict` Tier-1 read raises plain `ValueError`/`TypeError` for counter-only corruption instead of `AuditIntegrityError` (the class the rest of the read path uses). Crashes either way; exception-class inconsistency. — py-contracts, arch-contracts.
- **F14.** `checkpoint/manager.py` "large checkpoint" warning uses **stdlib `logging`** (not even `structlog`) for an operational pipeline signal — audit-primacy says telemetry, not logger. — py-core, arch-core, sys-core.
- **F15.** `manager.py` coalesce over-limit raises bare `RuntimeError`; the identical aggregation guard raises `OrchestrationInvariantError`. Inconsistent type for one invariant. — arch-core.
- **F16.** `blobs_inline.py` config walk doesn't recurse into list elements — a `blob_ref` nested in a list-of-dicts is silently never discovered (not resolved, not flagged). Reachability depends on whether any plugin option-schema nests blob-bearing dicts in lists. — sys-core (P2), silentfail-core (P3, Tier-3 boundary caveat).
- **F17.** `record_synthesised_run` writes one identical `source_data_hash` to every row — defensible for cache-replay (one source blob) but per-row hash no longer distinguishes rows; verify `explain()` read-side doesn't rely on per-row uniqueness. — sys-core.
- **F18.** `blobs_inline.py` stale module docstring ("first slice implements pure discovery only" — it now does discovery+validate+fetch+substitute). — arch-core.
- **F19.** `landscape/formatters.py:128` type regression: `LineageResult | None` → `Any | None` (TYPE_CHECKING import deleted); pure static-safety loss. — py-core.
- **F20.** `engine/triggers.py:300-305` branch commented "backward compatibility for checkpoints that predate count fire offsets" — unreachable given the v5.0 hard version-reject; comment misleading (code itself correct). — py-engine, silentfail-engine.

---

## Disagreements reconciled

- **`RunResult` quarantine counter** — sys-contracts flagged the subset guard `rows_quarantined <= rows_failed` as *one-directional* (a future path bumping only `rows_failed` when it should quarantine stays silent); types-contracts verified the guard *is* enforced before the subtraction (no underflow). **Both correct:** the guard prevents underflow but does not structurally enforce that quarantine and failed increments are *paired*. Net: P2 design observation, not a live bug. Highest-leverage fix (sys-contracts): route both increments through one `mark_quarantined()` transition so pairing is structural.
- **F3 severity** P1 (arch-core) vs P2 (py-core/db-core): reconciled to **P1** because Postgres is verified config-reachable and bypasses the epoch gate; SQLite deployments are protected (so P2 for SQLite-only shops). Fix is mechanical regardless.

---

## What RC5.2 does *well* (verified, so it isn't re-litigated)

- **Orchestrator decomposition is behavior-preserving.** Every extracted module (`run_status`, `resume`, `cleanup`, `graph_wiring`, `landscape_registration`, `shutdown`, `runtime_preflight`) is a stateless function taking explicit params — no shared mutable orchestration state threaded across seams, no god-object traded for worse coupling. Removed blocks map to extracted destinations line-for-line. — py-engine, arch-engine, sys-engine, silentfail-engine (4-agent agreement).
- **Two pre-existing silent-failure holes CLOSED:** `coalesce_executor.py:1042` removed a `try/except→slog.error→del pending` that swallowed audit-write failures and dropped consumed tokens; `processor.py` now wraps terminal-outcome recording to raise `AuditIntegrityError` on `LandscapeRecordError`. — silentfail-engine, py-engine.
- **Layer purity preserved/improved:** `node_state_context.py` and `contexts.py` *removed* TYPE_CHECKING upward imports by introducing local L0 protocols (the correct "extract the primitive" resolution, not a lazy import). `config/protocols.py` replaced an L0→L1 import with seven in-L0 `*SettingsProtocol` classes. — sys-contracts, py-contracts.
- **Tier-1 write guards are genuine strengths:** offensive validation in `auth_audit_repository`, `run_lifecycle_repository` (catalog/attribution), `write_repository._validate_node_specs`; NOT NULL without `server_default` is a *correct* anti-fabrication choice; checkpoint integrity uses `hmac.compare_digest` and crashes on version mismatch. — silentfail-core (0 findings), db-core, sys-core.
- **Checkpoint 10MB size guard** removed from two executors was *relocated* (not dropped) to the single persistence boundary `core/checkpoint/manager.py`. — multiple.

---

## Recommended action order
1. **Confirm the F1 precondition** (can fork/expand coexist with checkpointing?) — this single fact sets whether F1 is a pre-merge P1 or a P3 doc note.
2. **F3 + F4** — mechanical/contained audit-integrity fixes (add 2 columns; one transaction).
3. **F2 + the parity-test gap** — make resume↔live `rows_processed` agree and add fork/expand/coalesce scenarios (closes the test blind spot behind both F1 and F2).
4. **F5, F8** — quick correctness/doctrine fixes (deep_freeze write-back; delete re-export).
5. **F6** — add the enum↔CHECK parity tests (prevents a recurring class).
6. **F12** — route the PBKDF2 fingerprint change through a security review (the one specialist lens this pass lacked).
7. Sweep the remaining P2/P3 doctrine items (slog primacy, exception-class consistency, type regressions).

### Agent IDs (resumable via SendMessage for deeper drill-down)
Contracts: py `a5bc54f483a68e8ea` · arch `ab0105e810620ecc4` · sys `aadb39801240770bd` · types `a3faf81023469de2f`
Core: py `a9fbd4642c6951ab2` · arch `a408305f55de18888` · sys `af02a9733b750cfef` · silent `aa21869f45466c669` · db `a2382311c19c4fcb2`
Engine: py `abac978f599e86ddb` · arch `a71873951fc700f2b` · sys `ac0a214304da03e61` · silent `a7788559ef9b47643` · determinism `a996b4a48c843d082`
