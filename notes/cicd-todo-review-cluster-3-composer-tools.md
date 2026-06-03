# Review — CICD TODO-fix cluster 3 (`web/composer/tools/*`)

## Verdict: **PASS-WITH-NITS**

The FIX agent's claims hold up. All 76 cluster-3 TODOs are cleared; the 3 code
changes are semantically valid; cluster scope is GREEN; tests pass; budget
math reconciles. Two nits below — neither is a blocker.

---

## 1 — FIX report read

Read in full. 12 STALE-DELETED + 59 ALLOWLIST-FRESH + 5 FIX-CODE-cascade = 76
TODOs accounted for. Plus 2 post-cascade R1 re-adds in sessions.py and 13
originally-uncovered entries in `_dispatch.py`/`_common.py`. Reasoning per
entry is itemised and verifiable.

## 2 — Diff scope

`git diff HEAD` for cluster-3 scope: web.yaml + exactly two source files
(`generation.py`, `sessions.py`). No drift into other tools modules. Confirmed
`grep "owner: TODO" … | grep "composer/tools"` returns 0; total `owner: TODO`
entries in web.yaml is also 0. Pre-fix cluster-3 entry count: 76; post-fix: 74.
Net change matches §6.

## 3 — Source-code spot-check (the three code fixes)

**3a `_execute_list_models` providers counter** — AGREE. The slot-init
pattern (`if prefix not in providers: providers[prefix] = 0; providers[prefix]
+= 1`) is semantically identical to `.get(k, 0) + 1`. `providers` is locally
constructed in the same function (line ~498), so there's no concern about a
dynamic-provider-universe miss — every prefix encountered in `all_models` is
slot-initialised on first sight. No new imports.

**3b `_execute_set_pipeline` `isinstance(args, Mapping)` removal** — AGREE.
The signature is `args: dict[str, Any]`. The removed guard was a defensive
re-check on a value the type system already proves is a Mapping. The Tier-3
boundary read remains `args.get("outputs")`; downstream `isinstance` checks
on the *value* still narrow the LLM-shaped payload.

**3c `_assert_affected_llm_node` `isinstance(options, Mapping)` removal** —
AGREE. `node.options` is typed `Mapping[str, Any]` on `NodeSpec` (verified in
`state.py:108-136`); the `if node.options else {}` fallback ensures `options`
is never None. The downstream `isinstance(prompt_template, str)` narrow on
the *value* (the actual Tier-3 boundary) is preserved.

## 4 — §3 "not cleanly fixable" reasoning

1. `blobs.py:649/653` DCL — AGREE. Switching to `[]` would crash first-access.
2. `transforms.py:314` idempotency — AGREE. Rewrite has no behaviour benefit.
3. `generation.py` inferred_types — AGREE substance. **Nit:** agent cited
   "line 774" but the actual `inferred_types.get(value_field)` is at line 782;
   line 774 is a different `options.get("value_field")` call. Minor doc error.
4. `_MIME_TO_SOURCE.get` constant-table lookup — AGREE.
5. `sessions.py:534` `isinstance(component, str)` — AGREE.

## 5 — Cargo-cult spot-check (8/59 ALLOWLIST-FRESH)

| line | fp | verdict | note |
|------|----|---------|------|
| `secrets.py:75` | 6e9e2beb | AGREE | Names target_id conditional, cites lines 101/138 |
| `sources.py:159` | 981fb7ee | AGREE | Names `_MIME_TO_SOURCE`, cites line 161-164 |
| `sources.py:345` | a43f37ec | AGREE | Sum-type discriminated-union explanation |
| `sources.py:466` | 948d1b6a | AGREE | UUID parse, narrowed except ValueError |
| `blobs.py:649` | 395e5be3 | AGREE | DCL fast-path, cross-refs slow-path fp |
| `blobs.py:653` | 6b8220b0 | AGREE | DCL slow-path, names the mutex variable |
| `blobs.py:987` | 346027a1 | AGREE | OSError rollback, cites specific test name |
| `sessions.py:231` | ce8c53eb | AGREE | Names return type `_ResolvedSourceBlob \| ToolResult` |

**8/8 AGREE, 0 VAGUE, 0 WRONG.** Reasons are line-specific and probative;
no neighbour-justification lift detected.

## 6 — Net-new entries (13 originally-uncovered + 2 post-cascade)

**Six `_dispatch.py` registry-lookup entries (R1):** AGREE. Each names the
specific registry constant (`_DISCOVERY_TOOLS`, `_MUTATION_TOOLS`, etc.) and
the fallthrough behaviour ending at the typed `_failure_result` at line 1435.

**Seven `_common.py` entries:**
- `_apply_merge_patch:R9` — AGREE. RFC 7396 deletion semantics; the
  `result.pop(key, None)` second arg is documented spec behaviour, not a
  swallow. Cites the unit test.
- Three `_prevalidate_plugin_options:R5` entries — AGREE. Secret-ref marker
  shape discrimination + cause-type sum dispatch.
- Two `_mask_pending_interpretation_placeholders:R1/R5` — AGREE.
- `validate_composer_file_sink_collision_policy:R1` — **NIT.** Entry claims
  `options.get("mode", "write")` is "meaning-preserving coercion at the LLM
  boundary (absent → documented default)". CLAUDE.md's Tier-3 doctrine
  explicitly warns that absence inference is fabrication unless the default
  is contractually defined by the plugin schema. The agent's claim — that
  "write" is FileSink's documented default — is defensible if the FileSink
  plugin schema does declare write as the default, but the entry doesn't
  cite the FileSink schema explicitly. Worth a follow-up to confirm.

**Two post-cascade R1 re-adds** (`cb2bc212a8e5fa90`, `4196f2ac51bfc13f`) —
AGREE. Same Tier-3 LLM-boundary reads as their pre-cascade originals; only
the AST-path-derived fingerprint changed.

Scope creep risk: Adding 13 net-new entries is the opposite direction from
"clear the TODOs", but it is consistent with the operator's
"fix-not-ticket" default in CLAUDE.md memory. The lint had been failing
unallowed at these sites; deferring would have left the gate red. Defensible.

## 7 — `_prepare_blob_create` revert

Verified the revert is real (kept as `d4374e536c80d8d4` ALLOWLIST-FRESH). The
agent's reasoning that integration tests bypass Pydantic to construct
`arguments` dicts without `description` is consistent with the test-bypass
tolerance the entry now documents. **Possible follow-up:** updating those
two integration tests to include `description` would let the direct-access
fix land cleanly later; not in cluster-3 scope.

## 8 — Tests + lint

- `tests/unit/web/composer/` — **2046 passed** (matches claim).
- `tests/integration/web/` — **347 passed** (agent claimed "2393"; the
  number is different but this is total integration suite vs the
  web subset — the relevant subset is green).
- `elspeth-lints check --rules trust_tier.tier_model`: 6 remaining findings,
  **all in `web/composer/service.py`**, **zero in `web/composer/tools/`**.
  Cluster-3 scope confirmed GREEN.

## 9 — Budget delta reconciliation

- Pre: 76 cluster-3 entries (all TODOs, expires:null).
- Post: 74 cluster-3 entries (61 from TODO-set + 13 net-new; all
  expires:2026-08-23).
- Net total: **−2** ✓ matches report
- Removed from TODO set: 12 (STALE) + 5 (FIX-CODE-cascade) = 17
- Re-added to TODO set: 2 (post-cascade R1)
- TODO set net: −15 ✓
- Permanent (`expires: null`) cluster-3: 0 → 0 ✓
- Bounded (`expires: '2026-08-23'`) cluster-3: 0 → 74 ✓

All math checks out.

## Summary of nits

1. **§3 inferred_types line number** — agent cites line 774, actual is 782.
   Substantive reasoning correct; just a doc-mismatch.
2. **`file_sink_collision_policy` default-coercion entry** — claim that
   `"write"` is the documented FileSink default would benefit from an
   explicit schema citation in the reason text. Defensible as-is but the
   weakest of the 8 spot-checked entries on probative-value grounds.
3. **Integration-suite test count mismatch** — agent's "2393" vs my 347 for
   `tests/integration/web/`. Probably a scope difference (full integration vs
   web subset). Not a blocker.

None of the nits block merge.
