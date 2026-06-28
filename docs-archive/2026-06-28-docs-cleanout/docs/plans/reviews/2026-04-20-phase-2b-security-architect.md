# Phase 2B Declaration-Trust — Security Architect Review

**Verdict:** APPROVE-WITH-CHANGES

The plan correctly extends the landed four-site audit-complete framework and does not weaken any S2-00x / H5 posture. The hard Spoofing / Repudiation vectors are already closed in code (nominal ABC, `_attach_contract_name`, aggregate sibling, MC3a/b/c scanner, `EXPECTED_CONTRACT_SITES` equality at bootstrap, `_require_pytest_process`). However, the plan under-specifies three security-critical obligations for new adopters: per-violation `payload_schema` TypedDicts (H5 Layer 1), scrubber extensions for new payload fields (H5 Layer 2), and Tier-1 classification justifications for three of the four new violations. It also retires the ADR-009 Clause 3 empty-emission carve-out via `can_drop_rows` without naming the audit-trail state that now covers "legitimate zero emission," leaving a Repudiation gap. Each of these is a required condition for merge, not a nice-to-have.

---

## SEC-1 — Per-violation payload_schema TypedDicts must be explicit

- **Severity**: HIGH
- **STRIDE**: Tampering / Information disclosure
- **Finding**: The plan shows one payload TypedDict (`UnexpectedEmptyEmissionPayload` at plan.md:201-205). `DeclaredOutputFieldsViolation`, `DeclaredRequiredFieldsViolation`, and `SchemaConfigModeViolation` are named but their payload shapes are left undocumented.
- **Evidence**: plan.md:122-174 (Task 2 — no schema shown); plan.md:225-278 (Task 4 — no schema); plan.md:280-320 (Task 5 — no schema); enforced at `src/elspeth/contracts/declaration_contracts.py:479-535` (H5 Layer 1 `_validate_payload_against_schema` rejects undeclared keys).
- **Recommendation**: Each ADR (011/012/013/014) MUST inline the purpose-built `TypedDict payload_schema` with every key and `Required[...]` / `NotRequired[...]` wrapping, BEFORE the first code PR. The ADR is the review surface; a PR introducing a payload schema without ADR pre-approval should be blocked by reviewers. Add a Task 7 checkbox: "every landed adopter's `payload_schema` appears verbatim in its ADR."

## SEC-2 — Secret-scrubber extension obligations unstated

- **Severity**: HIGH
- **STRIDE**: Information disclosure
- **Finding**: `SchemaConfigModeViolation` will likely carry `raw_schema_config` / `mode` / emitted-contract-shape context; `DeclaredOutputFieldsViolation`'s `missing` set is low-risk, but `DeclaredRequiredFieldsViolation` may include field-name previews or sample lookups from plugin config. The scrubber's closed-set defence (`_PATTERNS`, `_SECRET_KEY_NAMES`) assumes every new payload shape is audited against it — the plan never names this obligation.
- **Evidence**: `src/elspeth/contracts/secret_scrub.py:38-86` (closed-set pattern + key-name lists); plan.md:296-306 references "raw plugin config dicts" as something to avoid but does not forbid them; H5 module docstring at `declaration_contracts.py:386-401` explicitly notes `scrub_payload_for_audit` cannot cover every future format.
- **Recommendation**: Add a Task 7 verification item: "for each new violation payload, enumerate which fields could carry string values sourced from plugin config or row data; confirm each source is already covered by `_SECRET_KEY_NAMES` / `_PATTERNS`, or extend the scrubber in the same PR." Forbid `raw_schema_config` / `config_dict` / `options` / `sample_row` as payload keys (they are open-ended mapping sinks). Require payload values to be structural (field-name sets, mode strings, bool flags, counts) — not arbitrary config snapshots.

## SEC-3 — `can_drop_rows` retirement of ADR-009 Clause 3 carve-out creates a Repudiation gap

- **Severity**: HIGH
- **STRIDE**: Repudiation
- **Finding**: The plan correctly replaces the carve-out with a first-class declaration, but does not specify the audit-trail outcome for a *legitimate* zero-emission row (`can_drop_rows=True`, 0 emitted). Under audit-complete posture (dispatcher docstring at `declaration_dispatch.py:22-26`), silence in the audit trail must mean "checked and passed." A legitimately-dropped row must still produce a terminal state the auditor can query; otherwise the audit record is indistinguishable from "transform ran, produced nothing, we forgot to record it."
- **Evidence**: plan.md:176-223 (Task 3 semantics); CLAUDE.md §Critical Implementation Patterns names the terminal states (`COMPLETED`/`ROUTED`/`FORKED`/`CONSUMED_IN_BATCH`/`COALESCED`/`QUARANTINED`/`FAILED`/`EXPANDED`) — "no silent drops."
- **Recommendation**: ADR-012 MUST name the terminal state a legitimately-dropped row takes (new state or existing `CONSUMED_IN_BATCH`-style reuse) and the corresponding Landscape row-level record. If a new terminal state is needed, land it in the same PR as the contract — not as a follow-up. Add a Task 7 test: "a `can_drop_rows=True` transform emitting 0 rows on a row produces a Landscape record whose terminal state is queryable and distinct from `FAILED`."

## SEC-4 — `DeclaredRequiredFieldsContract` Option 1 scope-limit must be mechanically enforced

- **Severity**: MEDIUM
- **STRIDE**: Repudiation
- **Finding**: The plan recommends scoping ADR-013 to non-batch transform execution (plan.md:255-262) because no batch-pre-execution dispatch site exists. If the contract silently skips batch-aware plugins via `applies_to` returning `False`, the audit trail's silence on batch transforms is indistinguishable from "checked and passed" — the same Repudiation surface S2-003 closed for trivial bodies.
- **Evidence**: plan.md:255-262 names the gap but offers no mechanical enforcement; `declaration_dispatch.py:127-131` — `applies_to=False` skips without recording.
- **Recommendation**: If Option 1 is chosen, `applies_to` MUST raise `FrameworkBugError` (Tier-1) when passed a batch-aware transform that declares `declared_required_fields`, not return `False`. An operator configuring a batch-aware transform with required fields should see a bootstrap crash, not a silent skip. Alternatively, reject at plugin construction: a batch-aware transform MUST NOT have `declared_required_fields` populated until a batch-pre-emission site exists. Either approach is acceptable; silent skip is not.

## SEC-5 — Tier-1 classification for new violations is under-argued

- **Severity**: MEDIUM
- **STRIDE**: Elevation of privilege
- **Finding**: Plan asserts Tier-1 for `DeclaredOutputFieldsViolation` (plan.md:135-137). No Tier-1 recommendation for the other three. Tier-1 registration is load-bearing: it prevents `on_error` absorption (`errors.py:849-891`). An under-classified violation can be routed to failsinks, which silently hides evidence.
- **Evidence**: `tier_registry.py:51-56` (module-prefix allowlist); `declaration_contracts.py:593-596` (aggregate is Tier-1); `errors.py:849-891` (F3 two-layer sink invariant — pre-write VAL is dispatcher-owned, commit-boundary is Tier-1).
- **Recommendation**: Each ADR must state Tier-1 classification + justification. My posture:
  - `DeclaredOutputFieldsViolation` — Tier-1 (plan agrees; a misdeclared output schema corrupts downstream lineage).
  - `DeclaredRequiredFieldsViolation` — Tier-1 (a plugin running on input it didn't declare producing outputs means every downstream record is unattributable).
  - `SchemaConfigModeViolation` — Tier-1 (a FIXED-mode transform emitting OBSERVED violates the contract auditors query by; fabrication surface).
  - `CanDropRowsViolation` — Tier-2 (plugin bug, row-level; same posture as base `PluginContractViolation`).
- Verify each `@tier_1_error` call site passes `caller_module=__name__` literal (enforced by TDE2 scanner per `tier_registry.py:73-87`). Forbid any new module-prefix allowlist entry; all four violations belong under `elspeth.contracts.*`.

## SEC-6 — Bootstrap-miss detection is strong but Task 1 needs belt-and-suspenders

- **Severity**: MEDIUM
- **STRIDE**: Spoofing
- **Finding**: A new adopter that forgets to add its import to `declaration_contract_bootstrap.py` will either (a) fail MC2 in CI (manifest entry with no registration scan hit) if the manifest was updated, or (b) fail N1 per-site manifest equality at `prepare_for_run` (`orchestrator/core.py:223-278`). But if the developer *also* forgets the manifest entry, both MC1/MC2 and N1 pass while the contract silently does not register — the bootstrap miss is invisible.
- **Evidence**: plan.md:84-118 (Task 1 bootstrap module); `enforce_contract_manifest.py:597-716` (MC1/MC2/MC3 rules); `orchestrator/core.py:223-278`.
- **Recommendation**: The Task 1 bootstrap test (plan.md:110-112) MUST assert that the set of modules imported by `declaration_contract_bootstrap.py` equals the set of `register_declaration_contract(...)` call sites under `src/elspeth/engine/executors/`. Implement as an AST scan in the test, not a hand-maintained list. A contract module existing without a bootstrap entry is the actual attack surface; a manifest-only check can't see it.

## SEC-7 — `declared_required_fields` uniform runtime attribute is not a new Spoofing surface

- **Severity**: LOW (informational)
- **STRIDE**: Spoofing
- **Finding**: Reviewer flagged: does a uniform `declared_required_fields` on `BaseTransform` enable a mis-annotated plugin to declare the attribute but lie? Answer: no more than today's `passes_through_input` does, and that surface is already mitigated by the runtime cross-check.
- **Evidence**: `plugin_protocols.py:564-568` (sinks already expose `declared_required_fields` and SinkExecutor enforces); `pass_through.py:141-145` (`applies_to` reads plugin attribute directly per CLAUDE.md offensive programming).
- **Recommendation**: No change to the attribute design. But ensure the contract's `runtime_check` compares `plugin.declared_required_fields` against `inputs.effective_input_fields` (the caller-derived set per F1 resolution), not against a plugin-derived view — the plugin cannot be its own witness. This is implicit in plan.md:252-254 but should be stated as an ADR-013 invariant.

## SEC-8 — CreatesTokensContract ADR-first deferral is the correct Repudiation posture

- **Severity**: LOW (informational)
- **STRIDE**: Repudiation
- **Finding**: Reviewer asked whether ADR-first on `CreatesTokensContract` creates a Repudiation surface via semantic drift. It does not: the plan gates production code on ADR-015 resolution (plan.md:323-360), and the tracking ticket will either be retyped/closed (path 1) or re-landed with updated protocol (path 2). Silence on this dispatch site is already safe because no contract is registered for it — the audit-complete posture only applies to registered contracts.
- **Evidence**: plan.md:322-360 (Task 6 is explicitly ADR-first); `declaration_dispatch.py:177-189` (pre-emission site has no adopter yet, and dispatch only runs `applies_to` for registered contracts).
- **Recommendation**: Make ADR-015's "path 1" outcome (retype/close ticket) mandatorily include a regression comment in ADR-010's Adoption State table so a future contributor cannot re-open the ticket against stale semantics. If path 2 is chosen, the protocol-doc update and the `tests/unit/engine/test_processor.py` change MUST land in the same PR as the contract module (paired landing, F4-style).

## SEC-9 — DoS budget regression from four new adopters

- **Severity**: LOW
- **STRIDE**: Denial of service
- **Finding**: `budget_median(N) = 27 + (N-1)*1.5 µs` holds provided each `applies_to` is cheap and each `runtime_check` short-circuits on miss. The plan correctly flags this (plan.md:421). `DeclaredOutputFieldsContract` is a set difference; `CanDropRowsContract` is two attribute reads + int compare; `DeclaredRequiredFieldsContract` is a set subset test; `SchemaConfigModeContract` is the one risk — "runtime contract mode equality" needs to avoid deep attribute walks.
- **Evidence**: plan.md:421 (risk table); `enforce_contract_manifest.py:547-589` already guards MC3c trivial bodies, so "cheap = no-op" is not a shortcut.
- **Recommendation**: Add a Task 7 check: "each new `applies_to` body is O(1) in plugin attribute reads; no regex, no reflection, no nested `getattr` walks." For `SchemaConfigModeContract`, require `applies_to` to read a single flag (`plugin._output_schema_config is not None` or similar); all mode-comparison logic belongs in `post_emission_check` after the applies-to filter has pruned.

---

## Confidence Assessment

**High confidence**: SEC-1, SEC-2, SEC-3, SEC-5. These are direct mappings of already-enforced invariants (H5 Layer 1, H5 Layer 2, audit-complete terminal states, Tier-1 registry) onto the new adopters. The plan leaves them implicit and must make them explicit.

**Medium confidence**: SEC-4, SEC-6. SEC-4 depends on ADR-013's final scope choice (the Option 1 silent-skip hazard is real; the mitigation is one of several acceptable). SEC-6 depends on how the Task 1 bootstrap test is written — my AST recommendation may be overkill if a simpler test hits the same target.

**Lower confidence**: SEC-7, SEC-8, SEC-9. These are either informational confirmations of good plan choices or forward-looking concerns that need benchmark data the plan already commits to collecting.

## Information Gaps

- Exact field-list for each proposed violation payload — must be produced by ADR authors before code PRs.
- Whether a new terminal state is required for legitimately-dropped `can_drop_rows=True` rows, or an existing one can be reused. Requires engine/processor review outside security scope.
- Performance data for `SchemaConfigModeContract.applies_to` — Task 7 benchmark is the correct gate.
- Whether `declared_required_fields` on `BaseTransform` collides with the existing sink attribute of the same name in any edge case (e.g., a sink-configured transform). Probably not, but needs an alignment test.

## Caveats

- Review scope was STRIDE per adopter + framework overview. Did not re-review landed S2-00x / H5 posture (assumed intact per the reviewer's background note; spot-checks at `declaration_contracts.py:479-535`, `tier_registry.py:51-133`, and `secret_scrub.py:38-86` confirmed).
- Did not review ADR-010 Adoption State text updates (Task 7 item). The plan correctly gates those on actual landings.
- `pre_emission_check` site coverage by a first adopter is not itself a security concern — an empty site is safe (zero contracts iterated); the hot-path comment at plan.md:27-29 is a performance/engineering issue, not security.
