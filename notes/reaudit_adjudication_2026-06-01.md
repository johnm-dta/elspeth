# Reaudit adjudication — run 80691748c34c463080d454c8834e9d7a (2026-06-01)

> ⚠ **THESE BUCKETS ARE STALE — adjudicated under the OLD judge prompt
> (`f63d228b…`).** After this analysis the judge "Your role" section was
> rewritten (new hash `08052cb8…`) to make the judge *credit code comments and
> stated facts* as evidence. That changes the answer for many entries: a site
> whose **code comment already states the load-bearing why** (e.g. sink.py
> 533-539, which already names the offensive-crash / Plugin-Ownership why) will
> likely now pass **as-is, with no re-justify**. The "needs re-justify"
> discriminator collapses to: **does the in-code comment already state the
> why?** If yes → probably no work; if the why lives only outside the code →
> the rationale must carry it.
>
> **Do NOT use the buckets below as an action queue.** The real queue comes from
> **re-running the full reaudit sweep under `08052cb8…`** (after the 4/4 smoke
> test passes) and re-bucketing from that run. The analysis below is retained
> as the *reasoning* (per-site doctrine), not the worklist.


Full sweep: 240 outcomes. **204 STILL_AGREES, 25 WAS_ACCEPTED_NOW_BLOCKED,
11 ENTRY_OBSOLETE, 0 WAS_BLOCKED_NOW_ACCEPTED.** Flip rate ≈ 25/229 ≈ 11%,
uniformly stricter. The 11 obsoletes are all `generation.py` (my Finding-1
edit shifted the AST) → swept by `rotate`, not real flips.

## The governing principle (operator doctrine, 2026-06-01)

Three operator clarifications reframed the whole exercise away from "the judge
over-rotated, weaken the prompt":

1. **There must be a place where data is checked before it is declared Tier-2.**
   A rule that bans all `isinstance` is wrong: validation *requires* a shape
   check at the boundary. The discriminator is not "is there an isinstance" but
   **"is the checked type actually guaranteed at runtime, or only by
   annotation?"**
   - Type guaranteed only structurally / by annotation / by a loose callee
     (`Protocol`, unenforced dataclass field, `-> Any` callee, raw external
     value) → the `isinstance` IS the validation/enforcement. Legitimate.
   - Type our own *controlled* code statically guarantees and a wrong value
     can't occur without a bug in code we own → redundant guard. Forbidden.

2. **The judge sees only the code excerpt + comment + stored rationale.** The
   "why" of a legitimate guard often lives *outside* the excerpt (e.g.
   "SinkProtocol is structural, not runtime-enforced"). The rationale must
   *state* it. A thin / boilerplate / wrong-exception rationale earns a BLOCK
   even when the code is correct — and that BLOCK is the judge working as
   designed.

3. **We use an LLM judge instead of a static "no isinstance / no except" rule
   precisely because adjudicating exceptions is required.** A BLOCK on a
   legitimate-but-unjustified guard is not a false positive — it is the judge
   correctly demanding the exception be justified. The default answer to a
   wrong-looking BLOCK is therefore **a better rationale, not a weaker rule.**
   Weakening the prompt is reserved for the rare case where *no honest
   rationale could satisfy the rule* (the rule genuinely forbids a correct
   pattern even when the why is fully stated).

### The bucketing question reduces to one test, applied per site

> Does an honest rationale exist that adjudicates this exception — explaining
> *why* the guard/swallow is legitimate, in terms the judge can credit against
> what it sees?

- **YES, and the current rationale is thin/boilerplate/names-wrong-exception**
  → **RE-JUSTIFY** (IRAP-grade chore; operator HMAC key). The dominant outcome.
- **NO honest rationale exists** (guard truly redundant on controlled data; or
  swallow truly hides a first-party / audit-integrity error) → **CODE CHANGE**.
- **Honest rationale exists but depends on out-of-excerpt facts the judge
  structurally cannot credit even when stated** → the rare **PROMPT GAP**. Try
  the rationale first; only escalate to prompt if a well-stated rationale still
  re-BLOCKs at temp=0.

The old 6-lane taxonomy still maps: lane-1/2 = CODE; lane-4 = RE-JUSTIFY;
lane-3 = PROMPT GAP (now demoted to "rare, prove it with a re-justify attempt
first"). Sentinel-consumption sites are NOT lane-4 re-justify-as-honest-wording
of a swallow — they need a rationale that *names the sentinel* so the judge sees
it as control-flow, not error-handling.

## The 25 flips, bucketed (verified by reading full bodies + callee contracts)

### R5 — `isinstance` on first-party / typed returns (6 sites)

| Site | Verdict | Why |
|---|---|---|
| `engine/executors/sink.py:SinkExecutor.write` (×2 internal: `SinkWriteResult`, `ArtifactDescriptor`) | **RE-JUSTIFY** | `SinkProtocol` is a structural `Protocol` (`plugin_protocols.py:652`) — return type NOT runtime-enforced. `isinstance→raise` is the SOLE enforcement of Plugin-Ownership "wrong type → CRASH". Load-bearing, not redundant. Code comment already says this verbatim; stored rationale must carry the **structural-Protocol** fact (judge can't see the Protocol def). |
| `core/checkpoint/recovery.py:IncompleteTokenSpec.__post_init__` | **RE-JUSTIFY** | Persisted Tier-1 row; dataclass annotation is not runtime-enforced; **NOT NULL ≠ non-empty**. An empty `token_data_ref` is a valid `str` that becomes a *silent wrong payload-store key* — it does NOT crash naturally. This is offensive detection of a **non-crashing** Tier-1 anomaly, the exact case the "seal-check is the natural crash" doctrine does not cover. Rationale must state the non-crashing-anomaly why. |
| `contracts/schema_contract.py:PipelineRow.to_dict` | **RE-JUSTIFY** (or lane-2 retype) | `deep_thaw(self._data)` — `deep_thaw -> Any` (genuinely polymorphic), `self._data` is our Mapping. The `isinstance` narrows `Any→dict` so the `-> dict[str, Any]` annotation is honest, AND asserts. Can't fire in practice but load-bearing for type honesty. Alternative: typed `deep_thaw_mapping` helper, then drop the guard (lane 2). |
| `contracts/schema_contract.py:PipelineRow.to_checkpoint_format` | **RE-JUSTIFY** (or lane-2 retype) | Identical to `to_dict`. |
| `web/composer/tools/sessions.py:_execute_apply_pipeline_recipe` | **JUDGMENT** | `isinstance(result.data, Mapping)` where `result.data: dict|None` is **our own** `_execute_set_pipeline`'s return — code we fully control. Weakest case: closest to a redundant guard (we could audit the return paths instead). Either re-justify ("future helper through same path" risk is real) or CODE (drop `else: raise`, let `{**existing_data}` crash naturally). Operator call. |
| `plugins/transforms/rag/query.py:QueryBuilder._build_field_only` | **DESIGN** | `extracted = row_data[field]` is a Tier-2 row *value*; str-ness is not schema-guaranteed for a dynamically-selected field. `build()` **quarantines** `extracted is None` but this **crashes** on non-str — inconsistent. Either re-justify ("non-str query_field is a config/upstream bug; informative crash is correct; removing it only yields a murkier downstream crash") OR change to quarantine for parity with the None path. Operator call. |

**The R5 cluster is NOT one ratify/overrule decision.** It is 4 re-justify + 2
genuine code/design judgments. None is "weaken R5 in the prompt" — the judge is
correctly demanding each offensive `isinstance→raise` prove its type isn't
already runtime-guaranteed.

### R6 / R9 / R7 — exception handling (~13 sites)

**Substantive — no honest rationale keeps the swallow → CODE CHANGE:**
- `web/secrets/service.py:resolve` (×2): swallows `SecretDecryptionError` /
  `FingerprintKeyMissingError` → `None`. Secrets path = Tier-1-equivalent; a
  decryption failure or missing fingerprint key is an integrity / deployment
  signal silently downgraded to "ref not found." Should surface, not return
  `None`. **Strong judge call.**
- `core/landscape/data_flow_repository.py:record_transform_error`: swallows a
  serialization error at the **Landscape WRITE boundary** and substitutes a
  fabricated repr-based fallback payload, then writes it to the audit trail.
  Audit integrity — must crash, not fabricate. **Strong judge call.** (Verify:
  is the serialized payload Tier-3 transform data? Even so, a write-boundary
  fabrication into our audit DB is the concern.)
- `composer_mcp/server.py:create_server:call_tool`: swallows
  `canonical_json/stable_hash` error on `result_dict` — our own first-party
  dispatch output. Per Plugin Ownership a non-serializable type in our output is
  a bug → crash. **Likely correct.** (Verify result_dict origin.)

**Telemetry-primacy — event swallowed to `slog` that belongs in audit → CODE
(translate to audit/telemetry, not re-justify the slog):**
- `web/execution/service.py:_run_pipeline` (R6): status-update failure → slog
  only. Status is authoritative run data → audit.
- `web/sessions/routes/_helpers.py:_handle_runtime_preflight_failure` (R6): DB
  save failure → slog. Same shape.
- `plugins/infrastructure/batching/mixin.py:_complete_ticket` (R6): dropped row
  result post-timeout → `slog.debug`, no audit. A dropped row is a Landscape
  event.
  (All three are the CLAUDE.md "reviewer recommends slog → it belongs in audit"
  pattern. Verify each isn't already double-recorded before changing.)

**Sentinel / cancellation idiom — judge flagged a non-error signal →
RE-JUSTIFY naming the sentinel (NOT honest-wording-of-a-swallow):**
- `plugins/infrastructure/batching/mixin.py:_release_loop` (R6):
  `except TimeoutError: continue` is a poll sentinel from `wait_for`, not an
  error.
- `web/sessions/routes/composer.py:recompose` (R7):
  `suppress(CancelledError)` after `asyncio.shield` is the cancellation idiom.
- `telemetry/manager.py:close` (R6): `queue.Full`/`queue.Empty` on shutdown
  sentinel insertion — internal plumbing; telemetry-only exemption may apply.
  ⚠ These three risk a wasted operator signing cycle: a thin re-justify will
  re-BLOCK at temp=0. The rationale MUST identify the value as an expected
  sentinel. If a well-stated sentinel rationale still re-BLOCKs → genuine PROMPT
  GAP (R6 conflates sentinel-consumption with error-swallow).

**dict.pop idempotency (R9) — NEEDS CODE READ to settle:**
- `web/execution/service.py:_run_pipeline` (R9) and
  `web/execution/progress.py:cleanup_run` (R9): `dict.pop(key, default)` on
  internal first-party state with an idempotency claim. If two legitimate call
  paths exist → re-justify (idempotent cleanup). If single path → missing key is
  an invariant break → CODE (direct subscript / `del`). Verify call sites.

### R1 — `.get()` / membership on tier-disputed data (3 sites)

- `web/sessions/routes/_helpers.py:_extract_runtime_model_snapshot` (R1):
  `.get('model')` on persisted `composition_state` — **this is the persisted-
  composer-state-is-Tier-3 case we just clarified.** Either the state is
  contract-guaranteed (→ direct subscript, missing key = anomaly) or it is a
  Tier-3 re-read (→ `.get` default risks fabrication; record absence as `None`
  honestly). The rationale currently frames it as Tier-1; pick a lane honestly.
  Likely **CODE**.
- `web/composer/recipes.py:apply_recipe` (R1): `name` from LLM tool-call =
  Tier-3; membership-check-then-raise-with-valid-names is honest. Judge prefers
  direct subscript + converted `KeyError` (project test). Form nuance →
  **RE-JUSTIFY** (membership-raise is at least as honest) or minor refactor.
- `web/catalog/routes.py:get_schema` (R1): `_PLURAL_TO_SINGULAR.get(plugin_type)`
  where `plugin_type` is a Tier-3 URL path param — `.get` with `None` default IS
  the legitimate boundary coercion. Judge wants the `@trust_boundary(tier=3)`
  decorator form. **RE-JUSTIFY / add decorator** — code is fine.

## Recommended operator queue

1. **CODE changes (no honest rationale; real defects surfaced by the stricter
   bar):** secrets/service ×2, data_flow_repository, composer_mcp call_tool, the
   3 telemetry-primacy slog swallows, `_extract_runtime_model_snapshot`. These
   are lane-1/2 — the stricter standard caught genuine silent-degradation /
   audit-integrity issues. Fix, co-land fingerprint baseline regen, then remove
   the (now-stale) allowlist entries.
2. **RE-JUSTIFY (rationale must carry the why; operator HMAC key):** sink.py ×2,
   recovery.py IncompleteTokenSpec, schema_contract ×2, get_schema, apply_recipe.
   Each rationale states the out-of-excerpt fact (structural Protocol /
   non-crashing Tier-1 anomaly / `Any`-narrowing / Tier-3 boundary coercion).
3. **VERIFY THEN DECIDE:** sessions.py `_execute_apply_pipeline_recipe`,
   `_build_field_only` (design: crash vs quarantine), the 2 R9 dict.pop
   idempotency sites. Read call sites; bucket into 1 or 2.
4. **SENTINEL re-justify, watch for prompt gap:** batching `_release_loop`,
   recompose, telemetry close. Name the sentinel; if re-BLOCK at temp=0, that
   trio is the only candidate for a targeted R6 prompt note
   (sentinel-consumption ≠ error-swallow).

## ★ FINAL ACTION QUEUE — re-run under the new prompt (08052cb8), 240/240

Run `a1f5839537714546afb5b397a6ff12a0` (complete): **194 STILL_AGREES, 35 flips
(34 keys), 11 ENTRY_OBSOLETE.** vs old run: 15 NOVEL flips (new prompt stricter),
5 old flips RESOLVED to ACCEPT. The flip count GREW (34 vs 24) — the role
clarification is **sharper, not looser**, and it enforces rationale quality
(most novel flips are `BLOCK-PENDING`, not `GENUINE VIOLATION`).

Disposition the judge itself names is the primary signal: `GENUINE VIOLATION` =
likely real defect; `BLOCK-PENDING` = code may be fine, rationale doesn't engage
the flagged rule → re-justify. This is THE authoritative queue (supersedes the
stale OLD-prompt buckets above).

### LANE A — RESOLVED / PASSES AS-IS (no action) ✅
Comment/code-crediting validated. `sink.py:write` (×4, R5) STILL ACCEPTS as-is —
its stored rationale + comment carry the structural why; **no re-justify needed**
(my earlier RE-JUSTIFY call was over-cautious). Old flips now ACCEPT:
`purge_payloads`, `get_schema` (R1), `_handle_runtime_preflight_failure` (R6),
`recompose` (R7 sentinel). Sentinel + boundary-coercion crediting works.

### LANE B — RE-JUSTIFY (code is/likely fine; rationale must engage the rule). Majority.
- **B1 — telemetry-primacy / best-effort broadcast** (cite: WS/progress broadcast
  is *ephemeral operational visibility, not a Landscape fact*; client-disconnect /
  queue-full is expected): `progress._safe_put`, `service._safe_broadcast`,
  `routes.websocket_run_progress` (×2), `service._run_pipeline` R6,
  `progress.broadcast` R1, `telemetry._drop_oldest_and_enqueue_newest`,
  `telemetry.close`, `web/app._prometheus_metrics` R4.
- **B2 — sentinel / idempotent-cleanup** (name the sentinel / state expected
  absence): `batching._release_loop` (`except TimeoutError: continue` poll
  sentinel), `batching._complete_ticket`, `batching.evict_submission`,
  `progress.cleanup_run` (**zero-subscriber absence — call-graph PROVEN legit;
  judge can't see the subscribe-time setdefault, so STATE it**),
  `progress.unsubscribe`, `runtime_preflight._evict`.
- **B3 — Tier-3 absence/coercion** (state why None/fallback is meaning-preserving
  AND recorded, and which exception is actually caught): `json_source._load_json_array`
  (rationale named UnicodeDecodeError; except catches JSONDecodeError/ValueError),
  `azure_blob._load_csv`, `fanout_guard._estimate_source_rows`,
  `plugin_context.record_validation_error` (repr_hash fallback), `schema_factory.
  _find_non_finite_value_path`, `rag/config._get_providers`,
  `contract_propagation.merge_contract_with_output`,
  `redaction._build_substitute_provider`, `web/blobs.finalize_run_output_blobs`.
- **B4 — type-narrowing on a controlled-but-loose callee** (state the `Any`→T
  narrowing): `schema_contract.to_dict` + `to_checkpoint_format` (deep_thaw→Any;
  ⚠ judge split the near-identical twins — `to_dict`=BLOCK-PENDING,
  `to_checkpoint_format`=GENUINE VIOLATION — model noise on a borderline; decide
  once, apply to both), `recovery.IncompleteTokenSpec` (non-crashing empty-string
  Tier-1 anomaly — the empty `""` is a valid str that becomes a silent wrong
  payload-key, NOT a natural crash; 1 of 2 fps flipped).

### LANE C — CODE CHANGE (genuine; no honest rationale keeps it)
- `composer_mcp.call_tool` (R6) — canonical_json swallow on OUR first-party
  dispatch output → should crash (Plugin Ownership). Judge: GENUINE VIOLATION.
- `web/secrets/service.resolve` (R6) — `SecretDecryptionError` /
  `FingerprintKeyMissingError` silently → `None`; Tier-1-equivalent integrity /
  misconfig downgraded to "not found". Judge: GENUINE VIOLATION. Strong.
- `sessions._execute_apply_pipeline_recipe` (R5) — redundant `isinstance(Mapping)
  + else: raise` on our own controlled `dict|None` return (smoke Case 4 class).
  Drop the guard, keep the `None` dispatch.
- `service._run_pipeline` (R9) — `_shutdown_events.pop(run_id, None)` straggler;
  align with the already-remediated sibling at :665 (NOT a leak — :653 passes
  `str(run_id)`).
- `_build_field_only` (R5) — DESIGN (operator): no upstream str guarantee →
  quarantine non-str for parity with the None path, or declare a typed input
  contract. Judge: GENUINE VIOLATION.
- `_extract_runtime_model_snapshot` (R1) — persisted composer state `.get('model')`;
  pick the tier honestly (direct subscript if guaranteed, else record absence).
- `apply_recipe` (R1) — Tier-3 `name`; judge prefers direct subscript + converted
  `KeyError` over membership-raise. Minor refactor or re-justify.
- `data_flow_repository.record_transform_error` (R6) — fabricated fallback payload
  at the Landscape WRITE boundary. ⚠ VERIFY before coding: confirm whether the
  serialized value is Tier-3 transform data (then a recorded fallback may be ok →
  re-justify) or our authored audit value (then crash → code). 1 of 2 fps flipped.

**Distribution: ~8 LANE-A (done), ~24 LANE-B (re-justify, the cheap IRAP lane),
~8 LANE-C (real code).** The queue did NOT shrink — it shifted toward
rationale-quality work, which is the correct and cheapest outcome.

## VERIFY-THEN-DECIDE — resolved (prompt-independent code facts, 2026-06-01)

These four were the "needs a code read to bucket" set. Resolved by tracing call
graphs / contracts; conclusions DON'T change with the judge prompt.

1. **`web/execution/service.py:_run_pipeline:1634` (R9, `_shutdown_events.pop(run_id, None)`)
   — NOT a bug; CODE-CONSISTENCY fix.** Suspected a key-type leak (`RunRecord.id`
   is `UUID`, dict keyed by `str(run_id)` at :640). But the caller submits
   `_run_pipeline(str(run_id), …)` at :653 and the param is `run_id: str`, so
   inside the finally `run_id` is the str form → the pop matches → no leak. The
   key is registered at :640 *before* `submit`, so it's present at cleanup. The
   sibling cleanup at :665 was already converted to the team's accepted R9
   remediation (`run_key = str(run_id); if run_key in …: del`, with a
   races/double-run comment); :1634 is the lone straggler still on the old
   `pop(…, None)` form. → Align :1634 with :665 (small consistency edit), not a
   re-justify of an already-decided pattern.

2. **`web/execution/progress.py:cleanup_run:253` (R9, `_subscribers.pop(run_id, None)`)
   — legitimate, NOT a bug.** `_subscribers` gets a `run_id` entry only when a WS
   client *subscribes* (`setdefault` at :108). A run with zero connected
   subscribers never adds the key, so `pop(…, None)` correctly no-ops — absence
   is an expected state, not a hidden missing-key bug. → RE-JUSTIFY stating the
   zero-subscriber why (or one docstring line; the comment-crediting prompt would
   then pass it). No code change.

3. **`plugins/transforms/rag/query.py:_build_field_only` (R5, `isinstance(extracted, str)`
   → raise) — DESIGN DECISION (operator).** `query_field` is only a field *name*
   (validated as a non-keyword identifier, `config.py:118`); NOTHING pins the
   *value* at `row_data[query_field]` to `str`. So a non-str is a recoverable
   Tier-2 *value* problem, not a type-contract violation — yet `build()`
   QUARANTINES the `None` case while this CRASHES on non-str. The comment's
   "upstream plugin bug, Tier-2 must not be coerced" framing assumes a str
   guarantee that doesn't exist. Honest options: (a) quarantine non-str (return
   `QueryResult` error) for parity with the None path — simplest, consistent; or
   (b) declare a typed input-field contract so the DAG validates str-ness
   upstream, making the crash a *legitimate* contract violation. Keep-and-justify
   is the weakest. → Operator picks (a)/(b); both are CODE.

4. **`web/composer/tools/sessions.py:_execute_apply_pipeline_recipe` (R5,
   `isinstance(result.data, Mapping)` + `else: raise`) — CODE (drop the redundant
   guard).** `result.data` is `_execute_set_pipeline`'s `dict | None` return —
   code we fully control. The comment itself states "that contract is system
   code, not Tier-3 LLM data." This is the canonical controlled-return case —
   smoke Case 4 confirmed the new prompt BLOCKs exactly this ("re-checking a
   guaranteed shape is the redundant defensive pattern, dressed up as
   offensive"). The `if None / elif Mapping` dispatch is needed (different
   handling); the `else: raise AssertionError` is the redundant R5 target. → Drop
   the `isinstance`/`else: raise`, keep the None dispatch (`else: merged =
   {**existing_data, …}`); let the `dict | None` contract hold.

See [[project_trust_tier_custody_doctrine_2026-06-01]] and
[[feedback_tooling_auditability_irap_not_landscape]].
