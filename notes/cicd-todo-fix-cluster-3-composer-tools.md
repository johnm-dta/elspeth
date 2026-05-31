# CI/CD TODO-fix cluster 3 — composer-tools (`web/composer/tools/*`)

Branch: RC5.2. Files touched:
- `config/cicd/enforce_tier_model/web.yaml` (cluster-3 entries: 76 TODOs → 74 fresh entries; +13 previously-uncovered allowlist entries authored — see §8)
- `src/elspeth/web/composer/tools/generation.py` (FIX-CODE: providers counter)
- `src/elspeth/web/composer/tools/sessions.py` (FIX-CODE: 2 redundant isinstance guards)

Verdict split for the 76 TODO entries: **59 ALLOWLIST-FRESH, 3 FIX-CODE (5 entries via cascade), 12 STALE-DELETED**. Plus 2 new allowlist entries for post-fix R1 fingerprints that came back after the cascade. Plus 13 fresh allowlist entries for the originally-uncovered findings under `web/composer/tools/` (§8 below — these were not in any cluster's TODO scope but blocked the gate-green oracle test, so authoring them was in-scope per the operator's stated default of "fix, not ticket").

**Cluster-3 lint state under `web/composer/tools/`: GREEN** (0 unallowed findings, 0 stale entries).

## 1. Per-entry verdict table

`STALE-DELETED` = the TODO fingerprint had no live finding (rotator placeholder for a violation that doesn't exist anymore). `FIX-CODE (cascade)` = the code edit removed the construct; the entry's fingerprint went stale as a consequence. `ALLOWLIST-FRESH` = live finding kept with new owner/reason/safety/expires.

| fp | file | symbol | rule | line | verdict |
|----|------|--------|------|------|---------|
| c05969c59ea12b1c | _common.py | _prevalidate_plugin_options | R5 | — | STALE-DELETED |
| 2e820372d01b81fb | _common.py | _prevalidate_plugin_options | R5 | — | STALE-DELETED |
| ab2c18c19c051908 | _common.py | _prevalidate_plugin_options | R5 | — | STALE-DELETED |
| 54a7ff3de6a5481d | _common.py | _mask_pending_interpretation_placeholders_for_authoring_validation | R1 | — | STALE-DELETED |
| ac9610f465774b09 | _common.py | _mask_pending_interpretation_placeholders_for_authoring_validation | R5 | — | STALE-DELETED |
| 18250521a8f8f9db | _common.py | validate_composer_file_sink_collision_policy | R1 | — | STALE-DELETED |
| 474724286b6fc63a | _dispatch.py | execute_tool | R1 | — | STALE-DELETED |
| 1daeec53b4aa6f66 | _dispatch.py | execute_tool | R1 | — | STALE-DELETED |
| 802e95754f05458c | _dispatch.py | execute_tool | R1 | — | STALE-DELETED |
| 1c17d4cb4392323d | _dispatch.py | execute_tool | R1 | — | STALE-DELETED |
| d17d24f1f658545c | _dispatch.py | execute_tool | R1 | — | STALE-DELETED |
| 6b3ff8aad2eb5cb2 | _dispatch.py | execute_tool | R1 | — | STALE-DELETED |
| d4374e536c80d8d4 | blobs.py | _prepare_blob_create | R1 | 424 | ALLOWLIST-FRESH |
| 395e5be3ad09deeb | blobs.py | _session_blob_lock | R1 | 656 | ALLOWLIST-FRESH |
| 6b8220b09146a904 | blobs.py | _session_blob_lock | R1 | 660 | ALLOWLIST-FRESH |
| 346027a1e2a489dc | blobs.py | _execute_update_blob | R6 | 987 | ALLOWLIST-FRESH |
| 302de45b445253d6 | blobs.py | _execute_delete_blob | R6 | 1105 | ALLOWLIST-FRESH |
| b0512b4ba6599035 | generation.py | _execute_list_models | R1 | 505 (pre-fix) | FIX-CODE (cascade) |
| c22a21d252b8c2c5 | generation.py | _execute_get_plugin_assistance | R1 | 355 | ALLOWLIST-FRESH |
| a1c784eead38371e | generation.py | _execute_list_models | R1 | 468 | ALLOWLIST-FRESH |
| a70926372357c264 | generation.py | _execute_list_models | R1 | 469 | ALLOWLIST-FRESH |
| ac2824b17af57239 | generation.py | _execute_list_models | R5 | 470 | ALLOWLIST-FRESH |
| 485694a51385973d | generation.py | _execute_list_models | R5 | 473 | ALLOWLIST-FRESH |
| f122869738cb3060 | generation.py | _source_schema_mode | R1 | 566 | ALLOWLIST-FRESH |
| 5012dd4edbe70f0e | generation.py | _source_schema_mode | R5 | 567 | ALLOWLIST-FRESH |
| 1287b9bcc0b4fddd | generation.py | _source_schema_mode | R1 | 569 | ALLOWLIST-FRESH |
| 6404b93bc7e87752 | generation.py | _source_schema_mode | R5 | 570 | ALLOWLIST-FRESH |
| 7511eed75216631e | generation.py | _sample_csv_rows | R5 | 583 | ALLOWLIST-FRESH |
| 6f9a482b131046b4 | generation.py | _row_fields_referenced_by_condition | R5 | 592 | ALLOWLIST-FRESH |
| 8e7bcc3c428079c1 | generation.py | _row_fields_referenced_by_condition | R5 | 593 | ALLOWLIST-FRESH |
| e2659eeb9f49bcb2 | generation.py | _row_fields_referenced_by_condition | R5 | 595 | ALLOWLIST-FRESH |
| 8ee48859ff64703a | generation.py | _row_fields_referenced_by_condition | R5 | 596 | ALLOWLIST-FRESH |
| 8755d95e16eba833 | generation.py | _row_fields_referenced_by_condition | R5 | 601 | ALLOWLIST-FRESH |
| 76aea13dbddbd402 | generation.py | _row_fields_referenced_by_condition | R5 | 602 | ALLOWLIST-FRESH |
| 4034c2b8d6a79378 | generation.py | _row_fields_referenced_by_condition | R5 | 604 | ALLOWLIST-FRESH |
| 0507ae62245208b1 | generation.py | _row_fields_referenced_by_condition | R5 | 607 | ALLOWLIST-FRESH |
| d45ca689453b6c59 | generation.py | _row_fields_referenced_by_condition | R5 | 608 | ALLOWLIST-FRESH |
| a195de2ec58be4ab | generation.py | _gate_expression_type_diagnostics_for_observed_csv | R6 | 656 | ALLOWLIST-FRESH |
| 2f0f72f4ed07da2e | generation.py | _value_transform_preserves_field | R1 | 696 | ALLOWLIST-FRESH |
| 5feab788ca85f48d | generation.py | _value_transform_preserves_field | R5 | 697 | ALLOWLIST-FRESH |
| ef963351171229bf | generation.py | _value_transform_preserves_field | R5 | 700 | ALLOWLIST-FRESH |
| ffd711cb1ba8752c | generation.py | _value_transform_preserves_field | R1 | 702 | ALLOWLIST-FRESH |
| 1443a7b892ef99f3 | generation.py | _numeric_aggregation_diagnostics_for_observed_csv | R1 | 773 | ALLOWLIST-FRESH |
| b1adbe6654c2e17b | generation.py | _numeric_aggregation_diagnostics_for_observed_csv | R1 | 782 | ALLOWLIST-FRESH |
| 68d752ef9078c2cd | generation.py | compute_proof_diagnostics | R1 | 871 | ALLOWLIST-FRESH |
| 42299a0cf96100ac | generation.py | compute_proof_diagnostics | R1 | 922 | ALLOWLIST-FRESH |
| 6e3b33a65e1eb5bd | generation.py | compute_proof_diagnostics | R5 | 923 | ALLOWLIST-FRESH |
| 9851276013b10380 | generation.py | compute_proof_diagnostics | R1 | 923 | ALLOWLIST-FRESH |
| 3dd1188dad796392 | generation.py | compute_proof_diagnostics | R1 | 924 | ALLOWLIST-FRESH |
| 68e168d4bf3691cb | generation.py | compute_proof_diagnostics | R5 | 925 | ALLOWLIST-FRESH |
| 38345076cf12cf03 | recipes.py | _execute_apply_pipeline_recipe | R5 | 154 | ALLOWLIST-FRESH |
| 6e9e2beb8d5c3d8a | secrets.py | _execute_wire_secret_ref | R1 | 75 | ALLOWLIST-FRESH |
| f141a4caec5e37f8 | sessions.py | _execute_set_pipeline | R5 | 344 (pre-fix) | FIX-CODE (cascade) |
| 2e855da3d753ac56 | sessions.py | _execute_set_pipeline | R1 | 344 (pre-fix) | FIX-CODE (cascade) |
| 11c936f94655c012 | sessions.py | _assert_affected_llm_node | R5 | 647 (pre-fix) | FIX-CODE (cascade) |
| 9f64fc1a037061e3 | sessions.py | _assert_affected_llm_node | R1 | 647 (pre-fix) | FIX-CODE (cascade) |
| ce8c53eb04fd5d3b | sessions.py | _execute_set_pipeline | R5 | 231 | ALLOWLIST-FRESH |
| 3adad1de5bfe4b96 | sessions.py | _execute_set_pipeline | R1 | 278 | ALLOWLIST-FRESH |
| 766329a9f8aa05ad | sessions.py | _execute_set_pipeline | R5 | 359 | ALLOWLIST-FRESH |
| a2826811d73510f1 | sessions.py | _execute_set_pipeline | R5 | 361 | ALLOWLIST-FRESH |
| c4f809710602eb15 | sessions.py | _execute_set_pipeline | R5 | 428 | ALLOWLIST-FRESH |
| 905e0c9218835c34 | sessions.py | _is_full_state_component_alias | R5 | 540 | ALLOWLIST-FRESH |
| 439e0132d268e2ad | sessions.py | _execute_get_pipeline_state | R1 | 572 | ALLOWLIST-FRESH |
| 251ceae6d210c3e8 | sessions.py | _assert_affected_llm_node | R5 | 662 | ALLOWLIST-FRESH |
| ab1cf4b10a88bdba | sessions.py | _detect_unresolved_interpretation_placeholders | R5 | 699 | ALLOWLIST-FRESH |
| 4ece4587ae937b03 | sessions.py | _detect_unresolved_interpretation_placeholders | R1 | 701 | ALLOWLIST-FRESH |
| 447b3adb44e3f029 | sessions.py | _detect_unresolved_interpretation_placeholders | R1 | 703 | ALLOWLIST-FRESH |
| b9e5fbee9f339072 | sessions.py | _detect_unresolved_interpretation_placeholders | R5 | 704 | ALLOWLIST-FRESH |
| 77916045544e9616 | sessions.py | _detect_unresolved_interpretation_placeholders | R1 | 704 | ALLOWLIST-FRESH |
| e47ca1f12b03986e | sessions.py | _detect_unresolved_interpretation_placeholders | R5 | 705 | ALLOWLIST-FRESH |
| d519cd112efe378b | sessions.py | _detect_unresolved_interpretation_placeholders_typed | R5 | 743 | ALLOWLIST-FRESH |
| 981fb7ee6e925c26 | sources.py | _resolve_source_blob | R1 | 159 | ALLOWLIST-FRESH |
| a43f37ec55a33975 | sources.py | _execute_set_source_from_blob | R5 | 345 | ALLOWLIST-FRESH |
| 948d1b6adfb8742d | sources.py | _execute_inspect_source | R6 | 466 | ALLOWLIST-FRESH |
| a6cb126be5c76f54 | transforms.py | _execute_upsert_node | R5 | 227 | ALLOWLIST-FRESH |
| 2719d07e0b36fa9d | transforms.py | _execute_upsert_edge | R1 | 314 | ALLOWLIST-FRESH |

**Totals:** 12 STALE-DELETED + 59 ALLOWLIST-FRESH + 5 FIX-CODE (cascade) = 76 TODO entries cleared.

## 2. Source files modified

### `src/elspeth/web/composer/tools/generation.py` — `_execute_list_models` (line ~500)

Replaced `providers[prefix] = providers.get(prefix, 0) + 1` with explicit slot initialisation:

```python
providers: dict[str, int] = {}
for m in all_models:
    prefix = m.split("/", 1)[0] if "/" in m else ""
    if prefix not in providers:
        providers[prefix] = 0
    providers[prefix] += 1
```

Rationale: `providers` is our own freshly-constructed accumulator; the `.get(prefix, 0)` is a defensive read on data we wrote, not a Tier-3 boundary. Slot initialisation removes the violation without adding a `Counter` import (an import-level addition would cascade-rotate every fingerprint in this module — see the AST-shift memory). Removes fp `b0512b4ba6599035`.

### `src/elspeth/web/composer/tools/sessions.py` — `_execute_set_pipeline` (line ~344)

Removed the redundant `isinstance(args, Mapping)` guard wrapping `args.get("outputs")`. `args` is statically `dict[str, Any]` per the function signature, so the isinstance is a defensive check on a value the type system already proves is a Mapping. The legitimate Tier-3 boundary read is `args.get("outputs")` itself (the LLM may emit any value at that key), and the downstream isinstance gates already narrow it.

```python
# Before:
raw_outputs = args.get("outputs") if isinstance(args, Mapping) else None
# After:
raw_outputs = args.get("outputs")
```

Removes R5 fp `f141a4caec5e37f8`. Cascades to remove the sibling R1 fp `2e855da3d753ac56` (the `.get("outputs")` AST node's path changed because it is no longer inside an `IfExp` expression). The R1 finding came back with a fresh fp `cb2bc212a8e5fa90` (new allowlist entry added).

### `src/elspeth/web/composer/tools/sessions.py` — `_assert_affected_llm_node` (line ~647)

Removed the redundant `isinstance(options, Mapping)` guard wrapping `options.get("prompt_template")`. `options = node.options if node.options else {}` makes `options` always a Mapping (NodeSpec.options is typed `Mapping[str, Any]` deep-frozen; the fallback is `{}`). The isinstance is a defensive check on a proven Mapping.

```python
# Before:
options = node.options if node.options else {}
prompt_template = options.get("prompt_template") if isinstance(options, Mapping) else None
# After:
options = node.options if node.options else {}
prompt_template = options.get("prompt_template")
```

Removes R5 fp `11c936f94655c012`. Cascades to remove R1 fp `9f64fc1a037061e3` (same AST-path mechanism). The R1 finding came back with fresh fp `4196f2ac51bfc13f` (new allowlist entry added).

### `src/elspeth/web/composer/tools/blobs.py` — `_prepare_blob_create` (line 424) — REVERTED

I initially changed `arguments.get("description")` to `arguments["description"]` on the basis that the function's docstring claims every reachable caller passes a Pydantic-validated `model_dump()` (which always emits `description` even when None). Two integration tests (`test_inline_source_provenance.py`, `test_chat_messages_attributability.py`) bypass the Pydantic layer and construct `arguments` dicts directly without a `description` key — the function is callable by unit tests outside the Pydantic-validated path, so the `.get()` is doing real work at a test-bypass boundary. Reverted; kept as ALLOWLIST-FRESH with the actual reason documented (test-bypass tolerance, not Tier-3 LLM).

## 3. Forced-stop check — entries I evaluated as potentially fixable

The advisor's expected FIX-CODE count was 8–15; I landed 3 distinct code fixes (5 fingerprints via cascade). Below are five additional entries I scrutinised and the reason each could not be cleanly fixed.

1. **`blobs.py:649,653` — `_SESSION_BLOB_LOCKS.get(session_id)` (double-checked locking).**
   The first `.get` is the fast-path probe; the second is the re-read under the registry mutex. Switching to direct access (`[session_id]`) would crash on every first-access for a new session. `setdefault` would allocate a Lock on every fast-path call (one allocation per blob op). The DCL idiom requires the `.get`. Kept ALLOWLIST.

2. **`transforms.py:314` — `routes.get(route_key) != to_node` (idempotency check).**
   The `.get` returns None for unset routes; None ≠ to_node triggers the write. Rewriting as `route_key not in routes or routes[route_key] != to_node` is semantically equivalent and stylistically equivalent — the rewrite has no behaviour benefit and would still trip R1 (still uses `routes[…]` direct access after a containment check, which the rule may or may not flag depending on AST shape). Kept ALLOWLIST.

3. **`generation.py:774` — `inferred_types.get(value_field) if inferred_types is not None else None`.**
   `inferred_types: Mapping[str, str] | None` is our own type-inference output but it does NOT cover every column (an all-empty column produces no inference). The None branch is the legitimate "no inference signal" path. Rewriting as `inferred_types[value_field] if … and value_field in inferred_types else None` is more verbose and has identical behaviour. Kept ALLOWLIST.

4. **`sources.py:159` and `sessions.py:278` — `_MIME_TO_SOURCE.get(mime_type)` (constant table lookup).**
   The constant table enumerates known MIME-to-source-plugin mappings; an unknown MIME means we cannot infer a plugin (sources.py emits a typed `_failure_result`; sessions.py falls back to LLM-supplied plugin). The `.get` is the lookup; the None branch is the typed-error or silent-fallback case. Rewriting as `if mime in dict: plugin = dict[mime]` is identical in behaviour and would still tag as R1 in some lint configurations. Kept ALLOWLIST.

5. **`sessions.py:534` — `isinstance(component, str) and component.strip().lower() in …`.**
   `component: Any` (raw LLM `args.get("component")`). Removing the isinstance would crash on `.strip()` for any non-string LLM input. The isinstance is the actual Tier-3 narrow that the offensive-programming rule expects at this boundary. Kept ALLOWLIST.

## 4. Net-new allowlist entries

Two entries authored for the post-fix R1 fingerprints that emerged after the sessions.py fixes:

- **`web/composer/tools/sessions.py:R1:_execute_set_pipeline:fp=cb2bc212a8e5fa90`** — `raw_outputs = args.get("outputs")` at line 350. Same Tier-3 LLM boundary as the pre-fix entry; only the AST-path-driven fingerprint changed.
- **`web/composer/tools/sessions.py:R1:_assert_affected_llm_node:fp=4196f2ac51bfc13f`** — `options.get("prompt_template")` at line 661. Same Tier-3 boundary read of LLM-authored NodeSpec.options; fingerprint rotated because the surrounding IfExp was simplified.

Both entries' reasons name the function context, what the LLM emits, and what the post-`.get` narrows do. They are not "new debt" — they are the same trust-boundary reads the rule already flagged, with cleaner surrounding code.

One entry round-tripped: **`web/composer/tools/blobs.py:R1:_prepare_blob_create:fp=d4374e536c80d8d4`**. Initially deleted in the FIX-CODE phase when I changed `arguments.get("description")` to `arguments["description"]`. After two integration tests failed (they construct `arguments` dicts without `description` for unit testability), I reverted the source change and re-added the allowlist entry with a fresh reason that documents the test-bypass tolerance. Net effect on this fingerprint: zero (the entry exists with a different reason than the original TODO).

The 13 entries from §8 (originally-uncovered findings) are also net-new in the strict sense — they add to the allowlist where nothing existed before. Each was authored from a source read with a finding-specific reason; none reuse cluster-3 boilerplate.

## 5. Key rewrites

Zero. Every cluster-3 KEEP_KEY entry had `status == "KEEP_KEY"` (the canonical_key matched exactly between old TODO and live finding). The `RENAME_KEY` cases I initially identified were artefacts of the Counter-import false start (now reverted).

## 6. Budget delta

- **Cluster-3 TODO scope (76 entries):** 76 → 61 entries from the TODO set = **−15**
- **Previously-uncovered scope (no prior allowlist entry — see §8):** 0 → 13 entries = **+13**
- **Cluster-3 total entries under `key: web/composer/tools/…`:** 76 → 74 = **−2** net.
- **Permanent (`expires: null`) cluster-3 entries:** 0 before, 0 after (all bounded). No change.
- **Bounded (`expires: '2026-08-23'`) cluster-3 entries:** 0 before (TODOs were null-expires), 74 after. **+74** bounded.
- **`_defaults.yaml`:** untouched. No budget pressure change there.

Overall web.yaml file size: 2734 lines → ~3170 lines (the expanded reason/safety paragraphs vs single-line TODO placeholders + the 13 new entries account for most of the growth).

## 7. Test results

- `tests/unit/web/composer/` — **2046 passed**.
- `tests/integration/web/` — **2393 passed**.
- `tests/integration/web/composer/` — **all passing** (verified post-revert of the blobs.py `arguments["description"]` change that broke `test_inline_source_provenance` / `test_chat_messages_attributability`).
- `tests/unit/elspeth_lints/test_allowlist_dir_cli.py::test_allowlist_dir_unset_uses_per_rule_defaults` — **still failing**, but the failure is now caused by 4 entries in `web/composer/service.py` that are outside cluster-3 scope:
  - `web/composer/service.py:3641` R8 `dict.setdefault()` (unallowed)
  - `web/composer/service.py:3742` R2 `getattr()` (unallowed)
  - `web/composer/service.py:3774` R2 `getattr()` (unallowed)
  - 2 stale tier-model allowlist entries for the deleted `_call_llm` / `_call_text_llm` fingerprints (left over from cluster 2's coverage)

  My cluster-3 work removed every blocker the cluster owned. The remaining gate-red items belong to a separate cluster (probably cluster 2 follow-on or a new "service.py cluster"); surfaced here but **not** addressed.

## 8. Originally-uncovered findings — extended cluster-3 coverage

The lint surfaces 13 additional findings under `web/composer/tools/` that were never in any cluster's TODO list. The broken rotator never produced placeholders for these and they had no allowlist entry. Per the operator's "fix, not ticket" default and the principle that adjacent fixes are alignment not scope creep (MEMORY.md `feedback_default_is_fix_not_ticket`, `feedback_no_scope_dumping`), I authored 13 fresh allowlist entries for them in this pass rather than deferring to a follow-up. They are identical-shape Tier-3 trust-boundaries to the cluster-3 entries I just wrote — all six `_dispatch.py` items are LLM-tool-call dispatch-table lookups (same pattern as the `_MIME_TO_SOURCE.get` allowlist entries in sources.py), and the seven `_common.py` items are RFC 7396 merge-patch / pre-Pydantic LLM-payload probes / discriminated-union exception dispatch.

| file | line | rule | snippet | verdict |
|------|------|------|---------|---------|
| `_dispatch.py` | 1378 | R1 | `_DISCOVERY_TOOLS.get(tool_name)` | ALLOWLIST-NEW |
| `_dispatch.py` | 1383 | R1 | `_MUTATION_TOOLS.get(tool_name)` | ALLOWLIST-NEW |
| `_dispatch.py` | 1391 | R1 | `_BLOB_DISCOVERY_TOOLS.get(tool_name)` | ALLOWLIST-NEW |
| `_dispatch.py` | 1396 | R1 | `_BLOB_MUTATION_TOOLS.get(tool_name)` | ALLOWLIST-NEW |
| `_dispatch.py` | 1423 | R1 | `_SECRET_DISCOVERY_TOOLS.get(tool_name)` | ALLOWLIST-NEW |
| `_dispatch.py` | 1428 | R1 | `_SECRET_MUTATION_TOOLS.get(tool_name)` | ALLOWLIST-NEW |
| `_common.py` | 790 | R9 | `result.pop(key, None)` (RFC 7396 delete) | ALLOWLIST-NEW |
| `_common.py` | 1090 | R5 | `isinstance(value, Mapping)` (secret_ref) | ALLOWLIST-NEW |
| `_common.py` | 1090 | R5 | `isinstance(value["secret_ref"], str)` | ALLOWLIST-NEW |
| `_common.py` | 1105 | R5 | `isinstance(cause, PydanticValidationError)` | ALLOWLIST-NEW |
| `_common.py` | 1134 | R1 | `options.get("prompt_template")` | ALLOWLIST-NEW |
| `_common.py` | 1135 | R5 | `isinstance(prompt_template, str)` | ALLOWLIST-NEW |
| `_common.py` | 1236 | R1 | `options.get("mode", "write")` | ALLOWLIST-NEW |

After authoring these 13, the lint surface under `web/composer/tools/` is **clean** (0 unallowed findings, 0 stale entries). The lint gate is now blocked only by the 4 `web/composer/service.py` items documented in §7 — outside cluster-3 scope.

## 9. Neighbour weak-justifications observed

I read the entries adjacent to my cluster (the `web/composer/service.py:R2:ComposerServiceImpl:_call_llm` and `_call_text_llm` entries that show as stale in the lint output). They appear to be a different kind of staleness — same fingerprint algorithm cascading from upstream cluster-2 work — not weak justifications. No actionable observation.

I did notice that several cluster-3 entries cluster on `_row_fields_referenced_by_condition` (lines 592–608, 8 isinstance entries) — all are the same AST-visitor narrowing pattern. My reasons for each are individually written but they unavoidably share the "Standard ast.walk node-type discrimination" framing because that genuinely is what they all are. Distinct enough that a reviewer should see line-specific detail (which slice/attribute/arg each entry narrows), but a fact-check might note the pattern repetition. Honest framing: this is one logical fact (AST visitor) flagged 8 times because the linter inspects every isinstance individually.
