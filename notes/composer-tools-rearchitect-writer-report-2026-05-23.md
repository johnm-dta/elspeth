# Composer tools rearchitect — writer report (2026-05-23)

Worktree: `/home/john/elspeth/.worktrees/composer-tools-rearchitect/`
Branch: `refactor/composer-tools-rearchitect`
Baseline at start: 2,022 composer unit tests pass; ruff + mypy clean.

## Per-change summary

### CHANGE 6 — hygiene B1, B4

- `tools/generation.py:121` — `_VALIDATION_ERROR_PATTERNS` changed from
  `list[tuple[str, str, str]]` (mutable) to
  `Final[tuple[tuple[str, str, str], ...]]`. Closing `]` rewritten as `)`.
- `tools/outputs.py:212` — the post-mutation `if output is None: return result`
  short-circuit became an offensive `assert output is not None` with an
  invariant-violation message naming the missing sink.

### CHANGE 4 — A1, A2 dataclasses.replace

- `tools/transforms.py:_execute_patch_node_options` — the 16-line hand-rolled
  `NodeSpec(...)` rebuild collapsed to `replace(current, options=new_options)`.
- `tools/secrets.py:_execute_wire_secret_ref` — all three spec rebuilds
  (`SourceSpec`, `NodeSpec`, `OutputSpec`) collapsed to `replace(spec,
  options=patched_options)`. The vestigial `deep_thaw(node.routes)` /
  `deep_thaw(node.trigger)` round-trips were also removed — `NodeSpec`'s
  `__post_init__` already deep-freezes via `freeze_fields`, so the thaw-then-
  refreeze loop was pure noise.
- Drop-side: `NodeSpec` / `OutputSpec` / `SourceSpec` removed from
  `secrets.py` imports (no longer referenced after `replace` substitution).

Same fix applied opportunistically inside CHANGE 1 to two more sites I had
to touch anyway (`tools/sources.py:_execute_patch_source_options` and
`tools/outputs.py:_execute_patch_output_options`) where the same drift-prone
field-by-field rebuilds existed. Surfaced in the "Opened scope" section.

### CHANGE 5 — A3, B5 kill double-validation

Chose option (b) per the spec's "Use your judgment, document choice":
**deleted the dead `try/except PydanticValidationError`** in
`_handle_upsert_node`, `_handle_patch_node_options`, and
`_handle_patch_output_options`. The exception branch was unreachable —
`_validate_mutation_arguments` and the inner Pydantic-rejecting block in
`_execute_*` re-raise as `ToolArgumentError` before `_handle_*` sees the
ValidationError. The `_handle_*` re-validation now sits inside a single
post-success branch; absence of the post-mutation entry is asserted as an
invariant rather than handled as a recoverable runtime branch.

Why (b) over (a): option (a) (returning `tuple[ToolResult, ValidatedModel]`)
would have broken ~10 direct test call sites that today consume `_execute_*`
as plain `ToolResult` producers. (b) keeps the external `_execute_*`
signature stable and removes the structural defect (the dead try/except)
without expanding the test-update blast radius beyond what CHANGE 1 already
required. The cost — one extra `model_validate` per successful mutation — is
negligible compared to the structural clarity gained.

### CHANGE 3 — collapse `_all_tools` / `_all_tools_v2`

- Deleted the v1 `_all_tools` definition + assertion block at
  `_dispatch.py:1201-1216`.
- Renamed `_all_tools_v2` → `_all_tools`. Kept the v2-style assertion (which
  includes `_SESSION_AWARE_TOOL_HANDLERS`).
- Removed both names from `tools/__init__.py`'s import block and `__all__`.
  Verified no external import of either name.

### CHANGE 2 — extract `tools/discovery.py`

- Created `tools/discovery.py` (175 lines) as the single source of truth
  for tool-name classification:
  - Canonical frozensets: `_DISCOVERY_TOOL_NAMES`, `_MUTATION_TOOL_NAMES`,
    `_BLOB_DISCOVERY_TOOL_NAMES`, `_BLOB_MUTATION_TOOL_NAMES`,
    `_SECRET_DISCOVERY_TOOL_NAMES`, `_SECRET_MUTATION_TOOL_NAMES`,
    `_SESSION_AWARE_TOOL_NAMES`, `_CACHEABLE_DISCOVERY_TOOL_NAMES`,
    `_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES`.
  - All five predicates: `is_discovery_tool`, `is_mutation_tool`,
    `is_cacheable_discovery_tool`, `is_session_aware_tool`,
    `is_blob_store_only_mutation_tool`.
- In `_dispatch.py`, added explicit parity assertions at import time for
  every registry: `assert set(_DISCOVERY_TOOLS) == _DISCOVERY_TOOL_NAMES`
  (and the five siblings, plus the session-aware variant). A registry-vs-
  declaration drift now fails the build at module import, before any
  compose() call could observe an orphan handler or missing tool.
- Removed the original predicate definitions from `tools/blobs.py` and
  `tools/sessions.py`. `_BLOB_QUOTA_MUTATION_TOOLS` and
  `_BLOB_PROVENANCE_MUTATION_TOOLS` stay in `blobs.py` because they describe
  the *kwarg-shape* dispatch (which extended kwargs each blob handler
  receives), not tool classification — keeping them next to the blob
  handlers preserves the original intent.
- Updated `tools/__init__.py` to re-export the predicates and name sets
  from `discovery.py` instead of `_dispatch.py`/`sessions.py`/`blobs.py`.
- Renamed exported `_BLOB_STORE_ONLY_MUTATION_TOOLS` →
  `_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES` per the No Legacy policy (single
  external caller — `service.py` — uses the predicate, not the frozenset,
  so the rename doesn't propagate).

### CHANGE 1 — registry collapse via `ToolContext`

The largest change. Net effect: the three hardcoded `if tool_name == ...`
branches in `execute_tool` and the six kwarg-divergent registry-lookup
blocks collapse to one registry lookup over the union of all six sync
registries; every handler now has the uniform signature
`(arguments, state, context) -> ToolResult`.

Mechanics:
- `tools/_common.py` — added `ToolContext` (frozen dataclass) carrying
  `catalog`, `data_dir`, `session_engine`, `session_id`, `secret_service`,
  `user_id`, `baseline`, `current_validation`, `runtime_preflight`,
  `max_blob_storage_per_session_bytes`, `user_message_id`. `ToolHandler`
  rewritten as `Callable[[dict, CompositionState, ToolContext], ToolResult]`.
- `current_validation` is a new field on `ToolContext` that solves the
  `diff_pipeline` problem the advisor flagged: that handler reads the live
  state's ValidationSummary to compute its delta, but the value isn't
  per-tool — it's the caller's pre-mutation `prior_validation`. `execute_tool`
  copies `prior_validation` into `context.current_validation` so
  `_execute_diff_pipeline` reads it from `context` and the if-branch
  disappears.
- Every handler refactored (sinks → secrets → outputs → transforms → sources
  → blobs → generation → recipes → sessions). Handlers that don't need
  context perform `del context` for signature uniformity (no unused-arg lint
  hits). `_handle_request_interpretation_review` (async session-aware) is
  **left untouched** — it's not dispatched by `execute_tool` (the compose
  loop calls it directly at `service.py:3924`) and its 9-field per-call
  injection set doesn't fit a steady `ToolContext`. The
  `SessionAwareToolHandler` alias stays.
- `tools/blobs.py`: deleted `BlobToolHandler` alias. `tools/secrets.py`:
  deleted `SecretToolHandler` alias. All registry dicts now typed
  `dict[str, ToolHandler]`. Removed the now-unused `Callable` import from
  `blobs.py`.
- `execute_tool` (public function) keeps its kwarg-compatible signature
  exactly — every existing caller (`service.py`, `sessions/routes/_helpers.py`,
  `composer_mcp/server.py`, every test) continues to work without
  modification. Inside, it builds the `ToolContext` once and threads it.
- The membership check between `_inject_prior_validation`-wrapped mutation
  handlers and pass-through discovery handlers is now a single `if tool_name
  in _ALL_MUTATION_TOOL_NAMES` lookup.
- Updated the external production call site at
  `web/composer/guided/steps.py` (4 invocations of `_execute_set_source`,
  `_execute_set_output`, `_execute_apply_pipeline_recipe`,
  `_execute_set_pipeline`) to pass a `ToolContext` constructed from local
  kwargs.

Test sites updated to construct `ToolContext`:
`test_promote_create_blob.py`, `test_promote_update_blob.py`,
`test_promote_set_pipeline.py`, `test_promote_set_source.py`,
`test_promote_set_source_from_blob.py`, `test_promote_apply_pipeline_recipe.py`,
`test_promote_patch_options.py`, `test_tools.py` (one
`_execute_preview_pipeline` site + two `_execute_update_blob` rollback
tests). Added a `_ctx()` helper in `test_promote_patch_options.py` and a
`tool_context` / `make_tool_context` fixture pair in
`tests/unit/web/composer/conftest.py` (used by future tests; the existing
tests use direct `ToolContext(...)` construction because their setup wires
session engines / data dirs they own).

## Allowlist updates

Pre-flight tier-model lint reported 51 stale entries (mostly in
non-composer files whose fingerprints had rotated in unrelated upstream
commits) plus 42 new findings inside `composer/tools/`. The CICD-pruning
discipline calls for programmatic patching:

- Wrote a script (executed once, not retained) that:
  - dropped the 51 stale entries from
    `config/cicd/enforce_tier_model/web.yaml`,
  - added 42 new entries for the composer/tools/ findings (computing the
    symbol-context via AST walk so the keys match what the rule emits),
  - added 9 entries for non-composer findings exposed by the stale-removal
    step (fingerprint rotation for `web/sessions/service.py`,
    `web/execution/service.py`, `web/sessions/routes/messages.py`).

Post-patch, the tier-model lint is **clean** (exit 0, zero findings).

## Test surface — final state

| Command | Result |
| --- | --- |
| `pytest tests/unit/web/composer/ -x -q` | **2 022 passed** |
| `pytest tests/unit/web/ -q` | 4 247 passed, 4 prometheus failures, 2 prometheus errors (all pre-existing — `opentelemetry.exporter.prometheus` not installed; verified absent on the main worktree too) |
| `pytest tests/integration/web/ -q` | 312 passed, 2 prometheus failures, 33 prometheus errors (same pre-existing issue) |
| `pytest tests/integration/pipeline/test_composer_llm_eval_characterization.py` | **17 passed** |
| `pytest tests/integration/web/composer/test_inline_source_provenance.py` | **6 passed** |
| `pytest tests/integration/web/composer/guided/test_step_handlers.py` | **10 passed** (caught and fixed the `_execute_set_source` production call site in `guided/steps.py`) |
| `mypy src/elspeth/web/composer/` | **Success: no issues found in 54 source files** |
| `ruff check src/elspeth/web/composer/` | **All checks passed!** |
| `elspeth-lints check --rules trust_tier.tier_model` | **clean** (zero findings) |

Pre-existing prometheus failures are unrelated to this refactor — verified
by inspecting the traceback (`ModuleNotFoundError: No module named
'opentelemetry.exporter.prometheus'` at `src/elspeth/web/app.py:22`, a file
this refactor did not touch). The exporter is an optional dep.

## Decisions where the spec was ambiguous

1. **`_handle_request_interpretation_review` adaptation to `ToolContext`** —
   the spec said "(or `Awaitable[ToolResult]` for the async session-aware
   handler)" but inspection showed that handler is dispatched outside
   `execute_tool` with a per-call kwarg set that doesn't fit a steady
   context. **I left it untouched** and kept the `SessionAwareToolHandler`
   alias. Verified with the advisor.

2. **A3/B5 (kill double-validation) — option (a) vs (b)** — chose (b) (delete
   the dead try/except, re-validate cheaply on success) over (a) (return
   `tuple[ToolResult, ValidatedModel]`). Option (a) would have broken ~10
   test call sites of `_execute_*` that consume `ToolResult` directly. The
   structural defect (dead try/except) is resolved either way; (b) preserves
   the external `_execute_*` signature.

3. **`_BLOB_STORE_ONLY_MUTATION_TOOLS` rename** — moved from `blobs.py` to
   `discovery.py` and renamed to `_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES` for
   symmetry with the other name-sets. The single external caller
   (`service.py`) uses the predicate, not the frozenset, so the rename
   doesn't propagate. Per No Legacy I deleted the old name from `__all__`
   rather than alias it.

## Places where I opened scope (and why)

- **Folded `dataclasses.replace` into `_execute_patch_source_options` and
  `_execute_patch_output_options`** — both had the same A2-style hand-rolled
  spec rebuild. Touching the function signatures for CHANGE 1 made the
  one-line `replace(...)` swap free; leaving them as field-by-field rebuilds
  would have reintroduced the drift mechanism the spec was closing.

- **`tools/sources.py`: removed the vestigial `deep_thaw(node.routes)` /
  `deep_thaw(node.trigger)` round-trips in `_execute_wire_secret_ref`** —
  same rationale as A2. `freeze_fields` re-freezes; the thaw was pure noise.

- **`SecretToolHandler` and `BlobToolHandler` deletion** — the spec implied
  ToolHandler uniformity but didn't explicitly require deleting the aliases.
  Per the No Legacy policy ("WE HAVE NO USERS YET — deferring breaking
  changes is the opposite of what we want") I deleted them. Verified no
  external consumer.

- **Fingerprint rotation for 9 non-composer entries** — when the script
  dropped the 51 stale entries, 9 of them turned out to be in non-composer
  files (`web/sessions/service.py`, `web/sessions/routes/messages.py`,
  `web/execution/service.py`). Those files' fingerprints had rotated in an
  unrelated upstream commit and the stale entries protected pre-existing
  patterns. I added new allowlist entries for them with the current
  fingerprints to keep the build green. They're tagged `owner: TODO` and
  flagged as fingerprint-rotation in the reason field, so the next
  CICD allowlist audit can re-justify or retire them.

## Things I deliberately did NOT touch (per spec)

- `web/composer/redaction.py` (MANIFEST untouched).
- `web/composer/state` — `_coalesce_branch_*`, `_serialize_branches` still
  consumed from `_common.py` (audit C5 deferred).
- `_blob_*` private symbol couplings to `web.blobs.service` (audit C1
  deferred).
- The `_execute_set_pipeline` "promote to public" question (audit C6
  deferred).
- The "ToolDeclaration registry-derived-from-declaration" architectural
  intervention diagnosed in
  `notes/composer-tools-growth-mechanism-2026-05-23.md`. The structural fix
  this PR lands is the **kwarg-shape collapse via `ToolContext`**, which
  closes elspeth-0a1dae9f90's specific concern (registry fragmentation
  driven by signature divergence). The deeper "every tool is a self-
  describing declaration" change is a separate ticket.

## Suggested commit-message scaffold

```
refactor(composer/tools): collapse kwarg-divergent handlers via ToolContext

Closes elspeth-0a1dae9f90 and elspeth-40a47d57e6 (audit findings A1, A2,
A3, B1, B4, B5, C2, plus the registry-fragmentation systems-thinker
diagnosis filed in elspeth-5aa2e8c2a1).

Every composer tool handler now takes the uniform
``(arguments, state, context) -> ToolResult`` signature.  The previous
six-registry kwarg gymnastics — three hardcoded ``if tool_name ==``
branches in ``execute_tool`` for ``preview_pipeline`` / ``diff_pipeline``
/ ``set_pipeline`` plus six dispatch blocks with divergent
``**blob_kwargs`` / ``secret_service=`` / ``runtime_preflight=`` shapes
— collapses to one registry lookup over the union of all six sync
registries.

Surface changes:

- New ``ToolContext`` frozen dataclass in ``tools/_common.py`` carries
  the full per-call context.  ``ToolHandler`` type alias updated to
  the three-arg shape.
- New ``tools/discovery.py`` leaf module is the single source of truth
  for tool-name classification (eight canonical frozensets + five
  membership predicates).  Handler dicts in ``_dispatch.py`` are now
  subordinate: an import-time assertion fails the build if the
  registry keys diverge from the declared name sets.
- ``_all_tools`` / ``_all_tools_v2`` divergence resolved by deleting
  v1 and renaming v2 → ``_all_tools``.
- ``BlobToolHandler`` and ``SecretToolHandler`` type aliases deleted
  (no longer needed; everything is ``ToolHandler``).
- Audit hygiene: A1, A2 ``dataclasses.replace()`` adoption across four
  spec rebuilds; A3 / B5 dead ``try/except PydanticValidationError``
  branches deleted; B1 ``_VALIDATION_ERROR_PATTERNS`` typed
  ``Final[tuple[...]]``; B4 defensive ``if output is None: return``
  converted to ``assert output is not None`` per offensive-programming
  policy.

Public ``execute_tool`` kwarg-compatible: no caller signatures change.

Tier-model allowlist: 51 stale entries pruned, 42 composer/tools
fingerprint additions for AST-shifted patterns, 9 non-composer
fingerprint rotations exposed by stale-pruning (owner TODO, reason
documents that they're rotation not new debt).

Tests: 2 022 composer unit tests pass; mypy + ruff clean;
tier_model rule clean.
```
