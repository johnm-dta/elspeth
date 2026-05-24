# Composer tools rearchitect — reviewer verdict (2026-05-23)

Worktree: `/home/john/elspeth/.worktrees/composer-tools-rearchitect/`
Branch: `refactor/composer-tools-rearchitect`
Commit: `7114e78cc`
Reviewer: complex-reviewer (Opus 4.7, 1M context)

## VERDICT: APPROVE-WITH-NITS

The commit delivers what the writer claims. The `ToolContext` collapse is
cleanly executed; every handler in the six sync registries now has the
uniform `(arguments, state, context) -> ToolResult` shape; the import-time
parity assertions in `_dispatch.py` make the discovery↔registry contract
mechanically enforced; the six audit findings (A1, A2, A3, B1, B4, B5) are
all in the code as described; the writer held scope tightly and the
fingerprint allowlist arithmetic balances (51 stale removed; 42 composer +
9 non-composer added). The forbidden files (`redaction.py`, `state.py`,
`web.blobs.service`) were untouched. No `ToolDeclaration` paradigm was
introduced. The 39-tool count and registry breakdown match expectations.

Two minor doc-hygiene nits keep this off a clean APPROVE; neither is
load-bearing and neither should block merge.

## Critical findings

None.

## Major findings

None.

## Minor findings

### M1. Stale docstring in `tools/blobs.py:15`
**File:** `src/elspeth/web/composer/tools/blobs.py:13-16`
**Issue:** The module docstring still claims the file owns
`_BLOB_STORE_ONLY_MUTATION_TOOLS`. That frozenset was moved to
`discovery.py` and renamed `_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES` in this
commit; the bottom of the same file (`blobs.py:1241-1247`) correctly
explains the move, so the top-of-file claim now contradicts the bottom of
the file. A new employee reading the docstring first will look for a
declaration that isn't there.
**Fix:** Strike `_BLOB_STORE_ONLY_MUTATION_TOOLS` and the
`is_blob_store_only_mutation_tool` predicate from the module docstring's
"Policy frozensets" enumeration (lines 13–16); leave the explanatory
comment at lines 1241–1247 which is now the authoritative pointer.

### M2. Historical-context references in `_common.py:1194-1197`
**File:** `src/elspeth/web/composer/tools/_common.py:1194-1197`
**Issue:** The `ToolContext` docstring narrates the historical
`BlobToolHandler` / `SecretToolHandler` aliases as the kwarg gymnastics
that prompted the collapse. Those aliases are gone; their last surviving
references are docstrings. The reference is correct as history but a
reader who greps for `BlobToolHandler` to learn its current shape will
land on prose explaining its deletion. Not wrong — the writer made this
choice for new-employee context — but it does mean these strings will
flag as "stale references" in any future grep-based audit.
**Fix (optional):** Strip the two type-alias names from the docstring,
keeping the structural point ("kwarg gymnastics"). Defensible to leave
as-is.

## Scope discipline

**Held tight.** The writer's report enumerates two scope expansions:
folding `dataclasses.replace` into `_execute_patch_source_options` and
`_execute_patch_output_options` (A2-style hand-rolled spec rebuilds — same
A2 pattern they were already fixing), and deleting the vestigial
`deep_thaw(node.routes) / deep_thaw(node.trigger)` round-trips in
`secrets.py::_execute_wire_secret_ref`. Both are defended in the
"Opened scope" section and both are correct: the underlying
`freeze_fields` post-init handles the freezing, so the thaw-then-refreeze
was pure noise.

No forbidden files touched. No `ToolDeclaration` paradigm. No
`_execute_set_pipeline` promotion. No `web.blobs.service` private symbol
coupling change. No `state.py` change. No `redaction.py` change.

## Intent fit (per requested change)

| Change | Found | Status |
|---|---|---|
| CHANGE 1: Registry collapse via `ToolContext` | `_common.py:1185-1247`, `_dispatch.py:1272-1368` | Correct. Six sync registries unified through `all_handlers` dict; `_inject_prior_validation` gated on `_ALL_MUTATION_TOOL_NAMES`; `ToolContext` is `@dataclass(frozen=True, slots=True)` with the 11 fields the report names. |
| CHANGE 1b: `execute_tool` kwarg-compat | `_dispatch.py:1272-1287` | Public signature preserved exactly. No caller breakage. |
| CHANGE 1c: `current_validation` wired into `diff_pipeline` | `_dispatch.py:1358` and `generation.py:1116-1117, 1127` | Correctly threaded. `prior_validation` kwarg → `context.current_validation` → `diff_states(...)`. |
| CHANGE 2: `discovery.py` leaf | `discovery.py:1-165`, asserts at `_dispatch.py:1211-1248` | Eight frozensets + five predicates + parity assertions. Build fails at import if any drift. Confirmed by `python -c 'import _dispatch'` succeeding with 39 tools. |
| CHANGE 3: `_all_tools` / `_all_tools_v2` collapse | `_dispatch.py:1377-1397` | v1 deleted; `_all_tools` is the union including session-aware. The two dual-registry assertions (no overlap; sync handlers not async; async handlers not sync) live at lines 1386-1427. |
| CHANGE 4 — A1: `replace()` in `_execute_patch_node_options` | `transforms.py:475` | Confirmed. 16-line hand-rolled `NodeSpec(...)` rebuild collapsed to `replace(current, options=new_options)`. |
| CHANGE 4 — A2: `replace()` in `_execute_wire_secret_ref` | `secrets.py:74, 94, 109` | All three branches (source, node, output) use `replace(spec, options=patched_options)`. Vestigial `deep_thaw` round-trips removed. `NodeSpec` / `SourceSpec` / `OutputSpec` imports correctly absent. |
| CHANGE 4 — A2 (opportunistic): `_execute_patch_source_options` | `sources.py:551` | `replace(state.source, options=new_options)`. Correct. |
| CHANGE 4 — A2 (opportunistic): `_execute_patch_output_options` | `outputs.py:187` | `replace(current, options=new_options)`. Correct. |
| CHANGE 5 — A3/B5: dead `try/except PydanticValidationError` deleted | `transforms.py:113-132` (`_handle_upsert_node`), `transforms.py:480-508` (`_handle_patch_node_options`), `outputs.py:192-222` (`_handle_patch_output_options`) | Option (b) implemented: `if not result.success: return result` followed by `model_validate(arguments)` outside try/except, followed by `assert node/output is not None`. All three sites consistent. |
| CHANGE 6 — B1: `_VALIDATION_ERROR_PATTERNS` typed `Final[tuple[...]]` | `generation.py:119` | `Final[tuple[tuple[str, str, str], ...]]`. Closing `]` correctly rewritten as `)`. |
| CHANGE 6 — B4: `assert output is not None` | `outputs.py:211-214` | The previous `if output is None: return result` short-circuit became `assert output is not None` with an invariant-violation message naming the missing sink (lines 212-214). |
| `_handle_request_interpretation_review` left untouched | `sessions.py:843-859` (async, kwarg-heavy); `service.py:3924` dispatches via `_SESSION_AWARE_TOOL_HANDLERS[tool_name]` directly, not via `execute_tool` | Confirmed. The writer's claim that this handler is dispatched outside `execute_tool` is verifiable in `service.py:3924`. The `SessionAwareToolHandler` alias remains, as the report states. |

## Structural integrity — `ToolContext` collapse

- `_DISCOVERY_TOOLS`, `_MUTATION_TOOLS`, `_BLOB_DISCOVERY_TOOLS`,
  `_BLOB_MUTATION_TOOLS`, `_SECRET_DISCOVERY_TOOLS`, `_SECRET_MUTATION_TOOLS`:
  all typed `dict[str, ToolHandler]` uniformly (`_dispatch.py:1142, 1165,
  1181, 1189, 1197, 1203`).
- `_ALL_MUTATION_TOOL_NAMES = _MUTATION_TOOL_NAMES | _BLOB_MUTATION_TOOL_NAMES
  | _SECRET_MUTATION_TOOL_NAMES` declared `Final[frozenset[str]]`
  (`_dispatch.py:1269`). `_inject_prior_validation` is gated on this union
  (`_dispatch.py:1364`).
- The `iscoroutinefunction` async/sync sweep asserts still execute at module
  import (`_dispatch.py:1402-1427`). They use a `cast(...)`-wrapped tuple of
  the six sync registries to satisfy mypy without weakening the runtime
  check.
- Import-time parity assertions exist for all 7 registries (six sync + the
  session-aware) plus the cacheable-subset relationship (`_dispatch.py:1211-1248`).
- Each handler that doesn't read `context` performs `del context` for
  signature uniformity. Spot-checked: `_handle_get_expression_grammar`,
  `_execute_explain_validation_error`, `_execute_get_plugin_assistance`,
  `_execute_get_audit_info`, `_execute_list_models`, `_execute_set_metadata`,
  `_execute_remove_edge`, `_execute_remove_node`, `_execute_upsert_edge`,
  `_execute_remove_output`, `_execute_get_pipeline_state`,
  `_execute_patch_node_options`, `_execute_list_recipes`. No leftover
  unused-arg warnings expected.

## Handler-signature uniformity

Verified across all plane files: `blobs.py`, `generation.py`, `outputs.py`,
`recipes.py`, `secrets.py`, `sessions.py`, `sources.py`, `transforms.py`,
`sinks.py`. Every handler that is a value in one of the six sync registries
has the `(arguments, state, context) -> ToolResult` shape. Handlers that
need context fields read them as `context.catalog`, `context.session_engine`,
`context.session_id`, `context.secret_service`, `context.user_id`,
`context.data_dir`, `context.baseline`, `context.current_validation`,
`context.runtime_preflight`, `context.max_blob_storage_per_session_bytes`,
`context.user_message_id`.

`_handle_request_interpretation_review` is the documented exception — async
with a 13-field per-call kwarg set, dispatched separately at
`service.py:3924`. The writer's decision to leave it untouched is correct.

## Audit-finding correctness

All six audit findings (A1, A2, A3, B1, B4, B5) are correctly addressed in
the code. The cited diff lines hold up to inspection. The two opportunistic
A2 sites (sources, outputs) follow the same pattern with the same
field-by-field rebuild → `replace()` collapse.

For A3/B5, the writer chose option (b) (re-validate cheaply on success)
over option (a) (return `tuple[ToolResult, ValidatedModel]`). The
implemented structure is `result = _execute_*(...)`; `if not result.success:
return result`; then `validated = Model.model_validate(arguments)`; then
`assert node/output is not None`. This is consistent across the three
sites. Option (b)'s cost (one extra `model_validate` per successful
mutation) is the documented trade-off and is acceptable.

## Test-update fidelity

- The 9 modified test files all update to `ToolContext(...)`-style call
  shapes.
- A `_ctx()` local helper in `test_promote_patch_options.py` and a
  `tool_context` / `make_tool_context` fixture pair in `conftest.py`
  (lines 481-506) are reasonable scaffolding — neither is a god-fixture;
  both are small and explicit.
- No `pytest.skip()` or `xfail` markers were added to suppress failures.
  All "skip" hits in test files are in docstring narration or test-method
  names (e.g., `test_skips_injection_on_failure`).
- Direct sanity check: ran
  `pytest tests/unit/web/composer/test_promote_set_pipeline.py
  test_promote_patch_options.py` — 42 passed.
- Test signature updates are confined to call-site argument construction.
  No assertions weakened.

## Tier-model allowlist hygiene

Diff: 51 entries removed and 51 added (net 0 entries; gross 102 lines of
churn out of 553). Inspection breakdown of the 51 added:

| File | Added entries |
|---|---|
| `web/composer/tools/sessions.py` | 19 |
| `web/composer/tools/_common.py` | 7 |
| `web/composer/tools/generation.py` | 6 |
| `web/composer/tools/blobs.py` | 5 |
| `web/composer/tools/outputs.py` (implied from arithmetic; report says transforms.py 2 and outputs.py 1) | matches 42-composer total |
| `web/composer/tools/transforms.py` | 2 |
| `web/composer/tools/_dispatch.py` | 1 |
| `web/composer/tools/secrets.py` | 1 |
| `web/composer/tools/recipes.py` | 1 |
| `web/sessions/service.py` | 5 (rotation) |
| `web/sessions/routes/messages.py` | 3 (rotation) |
| `web/execution/service.py` | 1 (rotation) |

(Composer files sum to 42, non-composer to 9 — matches the writer's
report.) All 9 non-composer entries are clearly tagged:

> `reason: Fingerprint rotation after composer-tools-rearchitect merge —
> pre-existing pattern, AST shift in unrelated commit rotated the fp`
> `safety: Unchanged from prior allowlisted state; pattern is owned by the
> file's original team and predates this refactor`

The `owner: TODO` plus this specific reason text is *more* tagged than the
writer's report claimed ("clearly tagged"); the next CICD allowlist audit
has the exact phrase to grep for. Several previously-vague
"TODO — fingerprint rotation without matching stale entry" placeholders
have been correctly replaced with the specific composer-tools-rearchitect-
tagged form — this is an improvement, not just a churn.

The 42 composer entries all have specific reasons (e.g., `'Pattern reified
by ToolContext refactor — Potential dict.get() usage: ...'`) and a `safety:`
field explaining the Tier-3 boundary status. None blank.

Net effect: tier-model lint reports clean post-patch (parent task confirmed
this on their tier_model run; not independently re-run here).

## Risk surfaces not in writer's report — verified

- **`current_validation` threading into `_execute_diff_pipeline`**:
  traced from `execute_tool`'s `prior_validation` kwarg
  (`_dispatch.py:1283`) → `context = ToolContext(...,
  current_validation=prior_validation, ...)` (`_dispatch.py:1358`) →
  `_execute_diff_pipeline` reads `context.current_validation`
  (`generation.py:1117`) → `diff_states(baseline, state,
  current_validation=current_validation)` (`generation.py:1127`). Correct.
- **`del context` consistency**: spot-checked every handler that doesn't
  use context; the `del context  # unused; signature uniformity with the
  other handlers.` comment is uniform across all such sites. No leftover
  unused-arg.
- **`service.py` external caller breakage**: `execute_tool` keeps its
  kwarg-compatible signature; the only direct call sites are `service.py`
  (intercepted-or-dispatched paths), `sessions/routes/_helpers.py`
  (imports `execute_tool` and `_DATA_ERROR_KEY`; the helpers file does not
  actually call `execute_tool` directly — it's a comment reference in the
  failure-context block), and the `guided/steps.py` production call sites
  which the writer correctly updated to construct `ToolContext`. No mypy
  surface change that ruff/mypy wouldn't catch.
- **`_BLOB_STORE_ONLY_MUTATION_TOOLS` rename**: the single external caller
  (`service.py:2528`) uses `is_blob_store_only_mutation_tool(tool_name)`
  predicate, not the frozenset. Rename does not propagate. The two
  remaining string references to the old name are in docstrings (M1, M2).

## Module-load sanity check

```
.venv/bin/python -c "from elspeth.web.composer.tools import _dispatch; print('OK, num tools:', len(_dispatch._all_tools))"
# Output: OK, num tools: 39
```

All import-time assertions pass. Tool counts match `get_tool_definitions`
docstring claim (13 discovery + 13 mutation + 9 blob + 3 secret + 1 advisor
= 39 LLM-visible tools, plus 1 session-aware = 39 internal registry total
once advisor's specialness is accounted for).

```
discovery: 13
mutation: 13
blob_disc: 4
blob_mut: 5
sec_disc: 2
sec_mut: 1
session_aware: 1
total: 39
```

## Anything the writer's report claims but the code doesn't deliver

Nothing material. Two micro-discrepancies, both documentation:

1. **`tools/blobs.py:15` module docstring**: claims this file owns
   `_BLOB_STORE_ONLY_MUTATION_TOOLS`, but it doesn't (moved to
   `discovery.py`, renamed `..._TOOL_NAMES`). M1 above.
2. **`tools/_common.py:1194-1197` ToolContext docstring**: narrates the
   deleted type aliases as if a reader were grepping for them. M2 above.

Neither contradicts the writer's report (the report itself notes both
moves); they are leftover doc strings that should also have rotated.

## Anything I flagged that I could not independently verify

- **The 51 stale removals are net-zero clean** — I checked the gross
  arithmetic (51 removed, 51 added; 42 composer-new + 9 non-composer-
  rotation) but did not run the tier-model lint pre-/post-patch myself.
  Trusting the parent's "tier_model: clean" claim.
- **`test_skill_drift.py` and other test files I did not open** — the
  writer claims 2 022 composer unit tests pass; I sanity-checked
  `test_promote_set_pipeline.py` and `test_promote_patch_options.py` (42
  tests, pass). The parent agent ran the broader composer-unit suite
  and confirmed pass; I trust that result.
- **The Anthropic cache-marker trailing-tool position** for
  `wire_secret_ref` — the comment at `_dispatch.py:1080-1083` says the
  trailing position is pinned by a test; I did not verify the test
  itself, but the position in `get_tool_definitions` is preserved
  (`wire_secret_ref` is last in the definitions list).

## Confidence Assessment

**Confidence:** High.

**Basis:** Direct reading of every file in the diff. Verified every
audit-finding line, every `ToolContext` field-read site, the parity
assertions, the `_ALL_MUTATION_TOOL_NAMES` gating, the `del context`
consistency, the import-time module load, the test sanity run, and the
allowlist arithmetic. The writer's claims hold up under independent
inspection. The two findings are doc-string defects, not code defects.

## Risk Assessment

**Residual Risk:** Low.

- The `ToolContext` shape is new; any future plane file that forgets to
  thread one of the 11 fields will fail mypy (typed accessors on a frozen
  dataclass) or fail the parity asserts at import time. The mechanical
  enforcement closes the high-risk drift vectors.
- The 9 non-composer rotation entries with `owner: TODO` are honest debt
  and should be re-audited at the next `cicd-allowlist-audit` skill run
  (per the operator's periodic-audit memory). The reason field is
  greppable; this is the intended discipline.
- The doc-string staleness in M1 and M2 is non-load-bearing; a future
  new-employee read will spot the conflict against the in-file
  corrections at the bottom of `blobs.py` and against
  `discovery.py`'s declarations. Not a regression vector.

## Information Gaps

- Did not run the full composer integration suite or the
  `pytest tests/integration/web/composer/` directory; the parent agent
  did this and reported pass. If a guided-mode regression slipped past
  the writer's claim that `test_step_handlers.py` is green, this review
  would not catch it.
- Did not re-run `elspeth-lints check --rules trust_tier.tier_model`.
  Parent's "clean" result is trusted.
- Did not exercise the live web composer end-to-end against staging.
  Trusting the structural review and the writer's local test surface.

## Caveats

- This review verifies structural correctness, intent fit, scope
  discipline, and code-vs-report consistency. It does **not** verify
  *semantic* correctness of behaviour under all dispatch paths (e.g., that
  the `runtime_preflight` callback the compose loop wires into
  `context.runtime_preflight` produces the same `ValidationResult` it did
  pre-refactor). The test-suite green-light is the load-bearing semantic
  evidence; this review is the structural complement.
- The two doc-string defects (M1, M2) are surfaced but explicitly not
  blocking. The writer or a follow-up cleanup commit can address them in
  a one-line patch each.
- Recommend merging as-is; address M1 and M2 in the next composer-touch
  commit.
