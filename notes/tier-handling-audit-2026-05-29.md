# Trust-tier mishandling audit — plugins (2026-05-29)

**Hypothesis (operator):** plugins handle recoverable Tier-2 / Tier-3 failures as
fatal Tier-1 failures (crash the run / lose rows) instead of routing/quarantining.

**Verdict: CONFIRMED.** Real, recurring pattern. 3 HIGH + a systemic sink design
gap + 1 fabrication + 2 low. 5 parallel auditors (opus) over 7 sources, 27
transform files (+ llm/azure/rag subpackages), 6 sinks. High findings
adversarially verified at the crux mechanism below.

## Crux mechanism (verified)
`engine/executors/transform.py:418-447`: an exception raised from
`transform.process()` is recorded `NodeStateStatus.FAILED` then **re-raised**
(line 447). Only a *returned* `TransformResult.error(...)` routes the row to
`on_error`. So a transform that **raises** on a bad row value crashes the run;
it does NOT get per-row routing. (TIER_1_ERRORS re-raise by design — correct.)
Sources: an uncaught exception in `load()` propagates through the orchestrator
(`core.py:~2680` records a phase error then re-raises) — crashes the run, vs the
source's own `SourceRow.quarantined()` per-row path which is graceful.

## HIGH — crash-on-recoverable (Tier-3/Tier-2 handled as Tier-1)

1. **`plugins/sources/csv_source.py:391`** (T3) — `resolve_field_names(raw_headers=…)`
   is called on EXTERNAL header names OUTSIDE any try/except. A header
   normalization *collision* (e.g. `"Case Study"` + `"case-study"` → `case_study`)
   raises `ValueError` (`sources/field_normalization.py` collision check) →
   uncaught → run crashes. Inconsistent with the same file's own
   `UnicodeDecodeError`/`csv.Error`/column-count quarantine handling. Per-row
   resolution in json/dataverse IS wrapped; CSV resolves headers once, up front,
   unguarded.
   **Fix:** wrap the header `resolve_field_names` call; on `ValueError`, record a
   parse-level validation error and fail-the-source-gracefully /
   quarantine-per-policy like the sibling paths (a collision means NO rows are
   parseable, so the correct outcome is a recorded source-level failure, not an
   unrecorded crash).

2. **`plugins/sources/azure_blob_source.py:618`** (T3) — same pattern, same fix.
   (The `columns`/`schema`-name resolve calls at :625/:635 are config-derived —
   crashing there is correct; only the `raw_headers` path at :618 is the bug.)

3. **`plugins/transforms/field_mapper.py:278`** (T2) — `get_nested_field(row_data, source)`
   for a dotted mapping source (e.g. `meta.origin`) raises `TypeError` when a
   PRESENT intermediate value is a non-dict (`infrastructure/utils.py:54-56`;
   a missing key safely returns MISSING — only a present non-dict raises). The
   `TypeError` is unwrapped → executor re-raises → run crashes on one malformed
   nested value. Contracts are flat and default schema mode is `observed`, so a
   non-dict nested value is type-valid Tier-2 data, not an upstream type-contract
   violation.
   **Fix:** wrap `get_nested_field` per mapping; on `TypeError`, return
   `TransformResult.error(...)` so the row routes to `on_error`.

## MEDIUM — sinks crash whole batch on per-row-attributable write failure
Systemic: only `chroma_sink` implements the per-row `on_write_failure` /
`_divert_row` contract (`infrastructure/base.py:1177`). The other five raise on
every write failure and never consult `on_write_failure` in their write paths —
and four advertise per-row routing in composer hints they don't implement
(hint-vs-implementation contradiction, quotable).

4. **`plugins/sinks/dataverse.py:533`** — per-row PATCH loop bare-`raise`s on
   `DataverseClientError`; isolatable + route declared (:244) + self-contradicting.
   Strongest sink finding.
5. **`plugins/sinks/database_sink.py:527`** — whole-batch insert; a per-row
   constraint failure (UNIQUE/NOT-NULL/check) fails the batch and won't survive
   retry; hint (:615) promises per-row routing.
6. **`plugins/sinks/csv_sink.py:413`** — batch staging; one bad row fails the
   batch; hint (:639) promises per-row quarantine.
7. **`plugins/sinks/json_sink.py:413`** — `json.dump(row, …, allow_nan=False)`
   raises on one non-serializable/NaN VALUE → whole batch fails; hint (:498)
   promises per-row.

**Note:** rollback-then-raise for *batch integrity* is correct; the gap is the
absence of the declared per-row escape hatch. Connection/auth/whole-blob failures
that raise ARE correct (not per-row isolatable). No silent_drop in any sink —
every except records an ERROR audit call before raising.

## MEDIUM — fabrication
8. **`plugins/transforms/llm/providers/openrouter.py:441`** (T3) — when the
   provider response omits `model`, the code substitutes the REQUESTED model as
   the model that SERVED the request. The fabricated value flows to the audited
   `calls` row (`LLMCallResponse.model`), the pipeline row `{response_field}_model`,
   and `success_reason.metadata["model"]`, while the recorded `raw_response` has
   no `model` — two contradictory audit sources. Sibling Azure provider
   (`clients/llm.py:223 _validate_provider_response_model`) RAISES on the same
   absence.
   **Fix:** mirror Azure — record + raise a non-retryable `LLMClientError` (route
   the row) instead of inferring the response model.

## LOW
9. **`plugins/sinks/azure_blob_sink.py:563`** — `allow_nan=False` serialization
   crash on one value aborts the cumulative upload; blob immutability makes
   batch-atomic upload legitimately all-or-nothing → low.
10. **`plugins/transforms/batch_classifier_metrics.py:277`** — `_safe_ratio`
    returns `0.0` for undefined precision/recall (denominator 0), while sibling
    `batch_effect_size.py:329` returns `None` for undefined `cohens_d`. Fabrication-
    distinguishability inconsistency; normalize toward `None`. Derived Tier-2
    metric (not source absence) → low.

## Clean (no findings)
json/dataverse/text/null sources + field_normalization; all row transforms except
field_mapper; all 12 batch/aggregation transforms (uniformly hardened: empty-batch
→ error result, single-element stdev guarded, honest `None` for undefined);
web_scrape family; llm/azure/rag (except openrouter `model`); chroma_sink (the
model citizen for per-row diversion).

## Relationship to the demo
The hello-world "2 of 5 rows" symptom is NOT any of these (csv source recorded the
3 malformed rows to `validation_errors` correctly). Its causes: (a) the composer
authored comma-unsafe CSV (system code producing malformed data), and (b)
`on_validation_failure="discard"` records-but-hides. Finding 1 (csv header crash)
is a *different* csv-source bug on the same plugin.

---

## Remediation status (2026-05-29, branch RC5.2)

Operator chose to fix all tracks. TDD (failing-test-first) per finding.

**Fixed + committed (localized tier fixes):**
- **#3 field_mapper** — dotted-path `get_nested_field` TypeError now returns
  `TransformResult.error(reason="type_mismatch")` (routes per-row), not a crash.
  Rationale verified: contracts are flat → a non-dict intermediate is operation-
  unsafe Tier-2 data, not a violated type contract. Two tests that pinned the
  crash were corrected (they had misclassified it as a contract violation).
- **#8 openrouter** — missing/empty response `model` now raises
  `LLMClientError(retryable=False)` (recorded by the provider's own handler, routed
  by the transform), mirroring Azure's `_validate_provider_response_model`. No more
  fabricating the requested model as the serving model.
- **#1/#2 csv + azure header resolution** — new `ExternalHeaderError(ValueError)`
  marks Tier-3 external-header faults (normalize-to-empty, normalization collision,
  duplicate raw headers). Sources catch it *specifically* at the header boundary →
  record + quarantine/discard. Config faults (bad `field_mapping`) stay plain
  `ValueError` → crash. Guard test (`test_field_mapping_config_collision_still_crashes`)
  proves the catch is not over-broadened. Two locked-in "raises on collision" tests
  corrected to assert quarantine.

**Observed (out of scope, filigree MCP was version-skewed v17<v18 so recorded here):**
- `azure_blob_source.py:973` `_validate_and_yield` broad `except ValueError` around
  `_normalize_row_keys` quarantines genuine Tier-3 faults (good) but also absorbs
  Tier-1 config `field_mapping` errors as data (every row quarantined w/ recorded
  reason — non-silent, milder than the crash bug, INVERSE failure mode). Proper fix:
  make the "Expected JSON object" raise a Tier-3 marker and narrow the catch to
  `ExternalHeaderError` + that marker so config errors propagate. Not a crash bug.

**NEEDS OPERATOR DECISION — #10 batch_classifier_metrics `_safe_ratio` (LOW):**
Per-label undefined precision/recall/f1 (denominator 0, e.g. a label never predicted)
is fabricated as `0.0`. Leaf-level fix to `None` is unambiguous (matches sibling
`batch_effect_size` cohens_d). BUT the aggregate metrics (macro/weighted/micro_*)
consume the leaves; switching to `None` forces a statistical-semantics choice that
CHANGES audited numbers (e.g. macro_f1 0.389→0.583, weighted_f1 0.458→0.611 in the
existing test) and has a defensible "deliberate sklearn `zero_division=0` convention"
counter-reading. Proposed rule if approved: per-label undefined→None; macro_* = mean
over defined-only (None if all undefined); weighted_* over defined support; micro_*
None when denom 0; recompute the pinned test numbers. Deferred to operator, NOT folded
silently.

**Sink per-row routing redesign (#4-7, #9):** in progress — separate commit.
