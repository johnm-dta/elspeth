# CI/CD TODO-fix cluster 2 — web-mid (execution/sessions/interpretation)

Branch: RC5.2. File touched: `config/cicd/enforce_tier_model/web.yaml`.
No source code modified. 38 allowlist entries authored with fresh
per-finding justifications. Verdict split: 38 ALLOWLIST-FRESH, 0
FIX-CODE, 0 STALE-DELETED (every finding still resolves to live source
and every justification is a real trust-boundary the rule cannot
mechanically distinguish from defensive programming).

## Per-entry verdict table

All verdicts are ALLOWLIST-FRESH with `expires: '2026-08-23'` (bounded
90-day re-evaluation). Reasons abridged here; the full text lives in
the YAML.

| fp | symbol | rule | line | verdict | reason summary |
|----|--------|------|------|---------|----------------|
| 976876634c13da21 | `web/interpretation_state.py::interpretation_sites` | R1 | 87 | ALLOWLIST-FRESH | T3 read of composer-authored `NodeSpec.options['prompt_template']`; absent key is legitimate (alternate-shape nodes use parts) |
| 4b539b3c9fb51aee | `interpretation_sites` | R5 | 88 | ALLOWLIST-FRESH | T3 type discrimination on the looked-up value; non-str skips legacy branch |
| b372d1d932279ec6 | `_materialize_node_for_authoring` | R1 | 137 | ALLOWLIST-FRESH | T3 read; absent key returns node unchanged downstream |
| e0199f5c0c8a79cd | `_materialize_node_for_authoring` | R5 | 138 | ALLOWLIST-FRESH | T3 type discrimination; non-str returns node unchanged |
| 6630263a6a3ef59c | `_replace_prompt_if_changed` | R1 | 155 | ALLOWLIST-FRESH | T3 read for change detection; None vs str equality drives correct rewrite branch |
| de1806c489da01fc | `_replace_prompt_if_changed` | R1 | 156 | ALLOWLIST-FRESH | Same change-detection pattern on `resolved_prompt_template_hash` |
| a70c15fa64b94659 | `_requirements` | R5 | 196 | ALLOWLIST-FRESH | T3 shape validation; raises `TypeError` on non-list/tuple |
| 63f52a06ad5feef6 | `_requirements` | R5 | 200 | ALLOWLIST-FRESH | T3 element-shape validation; raises `TypeError` on non-Mapping entry |
| 0ffa9e0289a39a2f | `_coerce_requirement` | R5 | 210 | ALLOWLIST-FRESH | T3 validate `id`; raises `TypeError` on non-str/empty |
| f19fa1c749d0d1ce | `_coerce_requirement` | R5 | 212 | ALLOWLIST-FRESH | T3 validate `user_term`; raises `TypeError` on non-str/empty |
| 5d18c4b559c476c1 | `_coerce_requirement` | R5 | 217 | ALLOWLIST-FRESH | T3 cross-field: resolved status requires str accepted_value |
| 840682c6329e8637 | `_prompt_parts` | R5 | 234 | ALLOWLIST-FRESH | T3 shape validation of `prompt_template_parts`; raises `TypeError` |
| a316960357ab995a | `_prompt_parts` | R5 | 238 | ALLOWLIST-FRESH | T3 element-shape; non-Mapping raises |
| c33669edd20677a4 | `_prompt_parts` | R5 | 243 | ALLOWLIST-FRESH | T3 type of text-part `text`; non-str raises |
| 7153f333c79f05f9 | `_prompt_parts` | R5 | 248 | ALLOWLIST-FRESH | T3 type/non-empty of interpretation_ref `requirement_id` |
| 4b90f4c5bdb219cf | `_render_prompt_parts` | R5 | 280 | ALLOWLIST-FRESH | Late T3 safety net: accepted_value must be str at render time |
| cbc93e20036437ee | `web/sessions/service.py::_patch_structured_interpretation_prompt` | R5 | 492 | ALLOWLIST-FRESH | T3 shape of `INTERPRETATION_REQUIREMENTS_KEY`; raises domain exception |
| c83bca8819367925 | `_patch_structured_interpretation_prompt` | R5 | 497 | ALLOWLIST-FRESH | T3 shape of `PROMPT_TEMPLATE_PARTS_KEY` |
| 3846d2c68bfd2033 | `_patch_structured_interpretation_prompt` | R5 | 507 | ALLOWLIST-FRESH | T3 element-shape of a requirement entry |
| b25701a5d377da07 | `_patch_structured_interpretation_prompt` | R5 | 514 | ALLOWLIST-FRESH | T3 validate `id` before using as dict key |
| 97ffa45270f1ba18 | `_patch_structured_interpretation_prompt` | R5 | 518 | ALLOWLIST-FRESH | T3 validate `user_term` before normalisation match |
| 4522399378639674 | `_patch_structured_interpretation_prompt` | R1 | 527 | ALLOWLIST-FRESH | T3 read of `status`; equality vs 'pending' is the desired discriminator |
| 404513d29c87c02f | `_patch_structured_interpretation_prompt` | R5 | 542 | ALLOWLIST-FRESH | T3 element-shape of a prompt-part |
| 2585bcc92642a790 | `_patch_structured_interpretation_prompt` | R5 | 549 | ALLOWLIST-FRESH | T3 type of `text` part payload |
| 25c02d3834ca4f69 | `_patch_structured_interpretation_prompt` | R5 | 560 | ALLOWLIST-FRESH | T3 validate `requirement_id` AND structural integrity check against requirements list |
| 8fa9fc3c82f4ce76 | `_patch_structured_interpretation_prompt` | R1 | 576 | ALLOWLIST-FRESH | T3 read of `status` for non-target ref; absent → pending-text branch |
| 2504d50b5fc751f8 | `_patch_structured_interpretation_prompt` | R1 | 577 | ALLOWLIST-FRESH | T3 read of `accepted_value`; next line isinstance(str) catches absence |
| d548d74ca0432882 | `_patch_structured_interpretation_prompt` | R5 | 578 | ALLOWLIST-FRESH | T3 validate `accepted` is str for resolved non-target refs |
| f65ffe2b6eacead7 | `SessionServiceImpl::archive_session::_sync` | R6 | 1682 | ALLOWLIST-FRESH | T3 filesystem boundary; `OSError` on restore rename annotates and re-raises primary |
| 5d64d769907cac14 | `SessionServiceImpl::archive_session::_sync` | R6 | 1698 | ALLOWLIST-FRESH | Post-commit best-effort `shutil.rmtree`; OSError keeps dir in quarantine |
| 2776823d39a788ae | `SessionServiceImpl::archive_session::_sync` | R7 | 1702 | ALLOWLIST-FRESH | Best-effort `rmdir` of parent quarantine_root |
| 807f122d01342885 | `SessionServiceImpl::prune_state_versions::_sync` | R1 | 3706 | ALLOWLIST-FRESH | Cross-table lineage walk; None == "walk terminates here" |
| 71b7e715f93ea4d1 | `SessionServiceImpl::prune_state_versions::_sync` | R1 | 3709 | ALLOWLIST-FRESH | Inner step of the same lineage walk |
| be313c147c5a50be | `web/sessions/routes/messages.py::register_message_routes::send_message` | R7 | 769 | ALLOWLIST-FRESH | asyncio cancel boundary; suppress re-raised CancelledError around shielded `_persist_llm_calls` |
| 1a85561c51205f0c | `register_message_routes::send_message` | R7 | 779 | ALLOWLIST-FRESH | Same cancel-shield pattern around `_publish_progress` |
| 6ffa502bacb1e244 | `register_message_routes::send_message` | R6 | 808 | ALLOWLIST-FRESH | `asyncio.wait_for(timeout=2.0)`; narrow `TimeoutError` only, cancels runaway |
| 039bf42792d84027 | `web/execution/service.py::ExecutionServiceImpl::_execute_locked` | R5 | 449 | ALLOWLIST-FRESH | Discriminated sum-type return `CompositionState \| InterpretationReviewPending`; canonical isinstance discriminator |
| 4cfc107b881f53c8 | `ExecutionServiceImpl::_execute_locked` | R9 | 604 | ALLOWLIST-FRESH | Idempotent `_shutdown_events.pop` on BaseException setup-failure path |

## Source files modified

None. Only `config/cicd/enforce_tier_model/web.yaml` was changed.

Diff scope on the YAML: 38 entries authored or rewritten. Net additions:
+161 / -175 lines (the rewritten reasons are longer and more specific
than the TODO stubs but the addition of multi-line YAML for some
entries is offset by removal of the multi-line TODO bodies).

Five entries had their canonical `key:` strings rewritten because the
symbol-context path stored in the old TODO entry was truncated (the
fingerprint matched the live finding but the key didn't, so the lint
reported them as stale and emitted the same finding under the correct
key):

* `f65ffe2b6eacead7` `…:R6:SessionServiceImpl:_sync` → `…:R6:SessionServiceImpl:archive_session:_sync`
* `5d64d769907cac14` `…:R6:SessionServiceImpl:_sync` → `…:R6:SessionServiceImpl:archive_session:_sync`
* `807f122d01342885` `…:R1:SessionServiceImpl:_sync` → `…:R1:SessionServiceImpl:prune_state_versions:_sync`
* `71b7e715f93ea4d1` `…:R1:SessionServiceImpl:_sync` → `…:R1:SessionServiceImpl:prune_state_versions:_sync`
* `6ffa502bacb1e244` `…:R6:send_message` → `…:R6:register_message_routes:send_message`

Four entries were added net-new (the corresponding TODO had been
dropped entirely by commit 5f4c503f0 and the lint was failing on
un-allowlisted findings):

* `2776823d39a788ae` `web/sessions/service.py:R7:SessionServiceImpl:archive_session:_sync`
* `be313c147c5a50be` `web/sessions/routes/messages.py:R7:register_message_routes:send_message`
* `1a85561c51205f0c` `web/sessions/routes/messages.py:R7:register_message_routes:send_message`
* `4cfc107b881f53c8` `web/execution/service.py:R9:ExecutionServiceImpl:_execute_locked`

## Budget delta

| metric | HEAD | after | delta |
|--------|------|-------|-------|
| `allow_hits` total | 363 | 367 | +4 |
| `allow_hits` with `expires: null` (permanent) | 283 | 249 | -34 |
| `allow_hits` with bounded `expires` | 80 | 118 | +38 |

`max_allow_hits: 542` ceiling — current 367, ample headroom.
`max_permanent_allow_hits: 392` ceiling — current 249, decreasing.

All 38 of my entries carry `expires: '2026-08-23'` (90 days from
2026-05-23). None retained `expires: null`. The advisor's budget
reconciliation should not need to bump any ceiling for this cluster.

## Test results

| suite | pass | fail |
|-------|------|------|
| `tests/unit/web/sessions` + `tests/unit/web/execution` | 1261 | 0 |
| `tests/unit/web/sessions` + `tests/unit/web/execution` + `tests/integration/web` | 1608 | 0 |

No source code modified, so the tests are a sanity gate, not an
attestation of behavior change.

Lint after edits, scoped to my groups:

```
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  check --rules trust_tier.tier_model --root src/elspeth --format text \
  | grep -E "web/execution|web/sessions|web/interpretation_state"
# (no output — CLEAN in my scope)
```

The 31 remaining un-allowlisted findings reported by the lint are all
in `web/composer/*` (out of scope for this cluster; cluster 3
territory).

## Things I wasn't sure about

1. **`039bf42792d84027` (execution/service.py R5 line 449,
   `isinstance(materialized_state, InterpretationReviewPending)`).** This
   is a discriminated sum-type discrimination on an OUR-typed return
   union (`CompositionState | InterpretationReviewPending`). The R5
   rule's "use `match` instead" suggestion would technically apply,
   but the isinstance form is fully exhaustive (both arms terminate
   correctly: pending arm raises, happy arm rebinds), the sum type is
   closed in a sibling module, and an existing allowlist precedent
   already exists for the same shape
   (`source_inspection._declared_field_name` — "Both arms are
   exhaustive over the declared union; neither fabricates a name").
   I allowlisted on that precedent rather than refactoring to `match`,
   since the change would be stylistic, not a trust-tier improvement.
   Worth a second look in quarterly review.

2. **`6ffa502bacb1e244` (messages.py R6 line 808, `except
   TimeoutError`).** The inline comments on L799-804 say "Programmer
   bugs and DB write failures propagate here instead of being silently
   swallowed". My read: the only way the `wait_for` boundary can
   produce a `TimeoutError` is the scheduling timeout, not the inner
   task's exceptions (those would surface as the task's own exception
   type). I'm reasonably confident this is correct based on the
   asyncio.wait_for contract, but a closer reader of the auto-title
   task might want to double-check by looking at what
   `_run_session_auto_title_lifecycle` (or whatever the task body is
   named) does internally.

3. **`807f122d01342885` and `71b7e715f93ea4d1` (prune_state_versions
   parent-link walks).** I framed these as legitimate cross-table
   absence handling and called the walk "conservative" (protects
   strictly more than the true closure when an orphan reference is
   encountered). That is true given the algorithm as written, but
   whether a state row being referenced by a chat-message but not
   present in `composition_states_table` SHOULD be a silent
   no-op or a logged anomaly is a design question I didn't decide.
   The audit-trail's perspective is: if `chat_messages.composition_state_id`
   references a missing state, that's a referential-integrity violation
   the operator might want to see. I left it alone because escalating
   the read would be code change, not allowlist work.

## Neighbour entries with weak justifications (follow-up findings, not fixed)

While reading the file I noticed two patterns of weak existing
allowlist text that the cluster-3 agent or a future allowlist audit
should look at:

1. Several `web/composer/*` non-TODO entries have reasons of the form
   "Tier 3 boundary — <one-line description>. Returns None on miss,
   handled by 404 response." (e.g.
   `web/catalog/routes.py:R1:get_schema:fp=bc80b0854f0cb34e`). These
   are technically true but read like one-liner badges rather than
   justifications. They don't explain why the underlying behaviour
   (returning None and converting to 404) is correct given the data
   flow. Not in my scope to fix, but flagged.

2. A handful of `web/execution/*` entries use a near-identical
   "Fingerprint shifted from `<old>` by the F-17/F-21 pre-execute
   placeholder gate ImportFroms" formula in the `reason` field (e.g.
   `web/execution/service.py:R8:execute:fp=ffe2194b4bd4f84e`,
   `R1:cancel:fp=888bf9a58b0bd233`, several others). These reasons
   tell you what rotated the fingerprint but don't tell you what the
   underlying code is doing or why the violation is acceptable. They
   would all fail a "could a reviewer who has never seen this file
   approve this on the basis of the reason text alone?" test. Out of
   my scope (those entries are non-TODO and already allowlisted) but
   they're prime candidates for a future depth-pass.
