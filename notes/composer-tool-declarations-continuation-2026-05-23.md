# Composer ToolDeclaration migration — continuation prompt for Steps 3, 4, 5

**Ticket:** elspeth-6c9972ccbf (Composer tools — ToolDeclaration paradigm)
**Date:** 2026-05-23
**Author of this prompt:** claude-opus-tool-decl (session that landed Steps 1 + 2)
**Branch:** `feat/composer-tool-declarations` (local-only; 2 commits ahead of RC5.2)
**Worktree:** `/home/john/elspeth/.worktrees/composer-tool-declarations/`

---

## What's already done

### Step 1 — commit `72d4ef0f6`

`ToolDeclaration` primitive landed. `create_blob` migrated as the falsifiable
exemplar. Key files:

- `src/elspeth/web/composer/tools/declarations.py` — `ToolKind` enum,
  `ToolDeclaration` frozen dataclass, pure `derive_*` helpers
  (`derive_handler_map_for`, `derive_name_set_for`,
  `derive_tool_definitions_by_name`, `derive_blob_quota_names`,
  `derive_blob_provenance_names`, `derive_blob_store_only_names`,
  `derive_cacheable_names`, `assert_unique_names`).
- `src/elspeth/web/composer/tools/blobs.py` — `_CREATE_BLOB_DECLARATION`
  co-located with `_execute_create_blob`; module-level `TOOLS_IN_MODULE`
  tuple.
- `src/elspeth/web/composer/tools/_dispatch.py` — aggregation site for
  `_REGISTERED_TOOLS`; derives `_TOOL_DEFS_BY_NAME`; import-time parity
  assertion that every declaration's handler matches the hand-maintained
  registry for its kind.
- `tests/unit/web/composer/test_tool_declarations.py` — 24 tests
  (byte-identity + every `__post_init__` invariant + every `derive_*`
  helper + `assert_unique_names`).

Dead-code clean-up bundled: `_BLOB_QUOTA_MUTATION_TOOLS` and
`_BLOB_PROVENANCE_MUTATION_TOOLS` in `blobs.py` were deleted — they had
no runtime consumer after the precursor's `ToolContext` collapse.

### Step 2 — commit `10b150d02`

All five blob-mutation tools now declared:

| tool                  | location                | needs_blob_quota | needs_blob_provenance | blob_store_only |
|-----------------------|-------------------------|------------------|-----------------------|-----------------|
| create_blob           | blobs.py                | True             | True                  | True            |
| update_blob           | blobs.py                | True             | False                 | True            |
| delete_blob           | blobs.py                | False            | False                 | True            |
| set_source_from_blob  | sources.py              | False            | False                 | False           |
| apply_pipeline_recipe | sessions.py             | True             | True                  | False           |

Three plane modules now expose `TOOLS_IN_MODULE`: `blobs`, `sources`,
`sessions`. `_dispatch.py:_REGISTERED_TOOLS` aggregates all three.

### Locked design decisions (don't re-litigate)

1. **`ToolDeclaration` does NOT carry a redaction field (option B).** The
   MANIFEST in `redaction.py` and the `HandlesNoSensitiveDataReason` prose
   stay in one auditor-readable surface. The hybrid "None means fall through
   to MANIFEST" was rejected on advisor consultation as Shifting-the-Burden
   inside the intervention. **Consequence:** the Step 2 security-review gate
   the diagnoser originally framed does not trigger — nothing about
   `redaction.py` changes during the migration.
2. **Aggregation lives in `_dispatch.py`, not in `declarations.py`.** A
   decorator-magic or module-level aggregator in `declarations.py` has a
   real circular-import trap. Pure-data leaf + explicit aggregation at the
   consumer site is the only correct shape.
3. **`needs_blob_quota` / `needs_blob_provenance` live on the declaration**
   (override of the precursor's locality choice; locality is preserved
   because declarations live in the plane module next to handlers).

---

## What's still fragmented (the work for Steps 3, 4, 5)

After Step 2 the per-tool fragmentation is:

| Surface | State after Step 2 |
|---------|---------------------|
| Handler function in plane module | Still required (this is the intended locality) |
| Registry dict in `_dispatch.py` (`_DISCOVERY_TOOLS`, `_MUTATION_TOOLS`, `_BLOB_DISCOVERY_TOOLS`, `_BLOB_MUTATION_TOOLS`, `_SECRET_DISCOVERY_TOOLS`, `_SECRET_MUTATION_TOOLS`) | **Still hand-maintained** — Step 4 decommissions these |
| Name frozenset in `discovery.py` | **Still hand-maintained** — Step 4 decommissions these |
| Inline JSON schema in `_dispatch.py:get_tool_definitions()` | Substituted for the 5 blob-mutation tools only; **other 34 tools still inline** |
| `MANIFEST` entry in `redaction.py:2443` | **Intentionally untouched** per (B) decision |
| `__all__` in `tools/__init__.py` | **Still hand-maintained** — Step 4 trims |
| Skill markdown reference | Still hand-maintained; Step 4 derives the tool-inventory bullets only (per Refinement 3) |
| Argument-model class in `redaction.py` | **Intentionally untouched** per (B) decision |

Total tools still to migrate: **34** (13 discovery + 13 standard mutation +
3 secret + 1 session-aware + 4 blob-discovery). The session-aware tier
(`request_interpretation_review`) requires extra care because its handler
is async and dispatched outside `execute_tool`.

---

## Step 3 — Migrate remaining tool tiers

### Scope

Add a `ToolDeclaration` for every tool that doesn't have one yet:

**Discovery tier (13 tools):** `list_sources`, `list_transforms`, `list_sinks`,
`get_plugin_schema`, `get_expression_grammar`, `explain_validation_error`,
`get_plugin_assistance`, `list_models`, `get_audit_info`, `list_recipes`,
`get_pipeline_state`, `preview_pipeline`, `diff_pipeline`. **Cacheable subset
(`_CACHEABLE_DISCOVERY_TOOL_NAMES`)** is `_DISCOVERY_TOOL_NAMES` minus
`{diff_pipeline, get_pipeline_state, preview_pipeline}` — set
`cacheable=True` on the cacheable ones only.

**Standard mutation tier (13 tools):** `set_source`, `upsert_node`,
`upsert_edge`, `remove_node`, `remove_edge`, `set_metadata`, `set_output`,
`remove_output`, `patch_source_options`, `patch_node_options`,
`patch_output_options`, `set_pipeline`, `clear_source`.

**Blob-discovery tier (4 tools):** `list_blobs`, `get_blob_metadata`,
`get_blob_content`, `inspect_source`. Set `kind=ToolKind.BLOB_DISCOVERY` and
all blob-kwarg booleans `False` (discovery tools don't write).

**Secret tier (3 tools):** `list_secret_refs`, `validate_secret_ref` (both
`SECRET_DISCOVERY`); `wire_secret_ref` (`SECRET_MUTATION`). None are
cacheable (secret state can change between calls).

**Session-aware tier (1 tool):** `request_interpretation_review`. **Defer
this one** — its handler is `async` (`_handle_request_interpretation_review`)
and dispatched outside `execute_tool` via the compose loop's
`_SESSION_AWARE_TOOL_HANDLERS` dict. The current `ToolHandler` type alias
in `_common.py` (`Callable[..., ToolResult]` synchronous) does not admit
async handlers. Two design options for this tool:

- (A) Widen `ToolDeclaration.handler` typing to a union of sync + async
  callables; tag with `kind=ToolKind.SESSION_AWARE`; the existing dispatch
  path inspects `kind` and routes accordingly.
- (B) Leave the session-aware handler outside the declaration model entirely;
  document it as a known carve-out in `declarations.py`.

Recommend (A) — uniformity is the whole point of the intervention — but
expect the type story to need care (async handlers' `ToolContext` shape may
differ; check what kwargs `_handle_request_interpretation_review` actually
needs; per the precursor writer report it had 9 fields that didn't fit a
steady context, so the declaration may need a `session_aware_extra_kwargs`
escape hatch). If (A) proves messy, ship (B) with a P3 ticket to revisit.

### Sequencing within Step 3

Migrate tier-by-tier. For each tool:

1. Locate the handler (`_execute_*` or `_handle_*`) in its plane module.
2. Copy its inline JSON schema from `_dispatch.py:get_tool_definitions()`
   into a declaration co-located with the handler. Preserve the original
   description prose exactly.
3. Add the declaration's name to the plane's `TOOLS_IN_MODULE` tuple.
4. In `_dispatch.py:get_tool_definitions()`, replace the inline schema block
   with `_TOOL_DEFS_BY_NAME["<tool_name>"]`.
5. Add a byte-identity test in `test_tool_declarations.py`.

For plane modules without `TOOLS_IN_MODULE` yet (transforms, outputs, secrets,
generation, recipes, sinks), add the import for `ToolDeclaration` + `ToolKind`
and add `TOOLS_IN_MODULE` at module scope.

**Plane modules NOT yet exposing `TOOLS_IN_MODULE`:**
- `transforms.py` (handlers: `_handle_list_transforms`, `_handle_upsert_node`,
  `_handle_upsert_edge`, `_handle_remove_node`, `_handle_remove_edge`,
  `_handle_set_metadata`, `_handle_patch_node_options`)
- `outputs.py` (`_handle_set_output`, `_handle_remove_output`,
  `_handle_patch_output_options`)
- `secrets.py` (`_handle_list_secret_refs`, `_handle_validate_secret_ref`,
  `_execute_wire_secret_ref`)
- `generation.py` (the 13 discovery tools' handlers + `_handle_get_plugin_schema`
  + `_handle_get_expression_grammar`)
- `recipes.py` (`_execute_list_recipes`)
- `sinks.py` (`_handle_list_sinks`)
- `sources.py` already has it (set_source_from_blob); needs extension for
  `_handle_list_sources`, `_handle_set_source`, `_handle_clear_source`,
  `_handle_patch_source_options`, `_execute_inspect_source` (blob-discovery).
- `sessions.py` already has it (apply_pipeline_recipe); needs extension for
  `_execute_get_pipeline_state`, `_handle_set_pipeline`,
  `_handle_request_interpretation_review` (if migrating session-aware).
- `blobs.py` already has it; needs extension for `_handle_list_blobs`,
  `_handle_get_blob_metadata`, `_execute_get_blob_content` (blob-discovery).

### Cacheable-discovery sub-step

When migrating the discovery tier, set `cacheable=True` on:
- `list_sources`, `list_transforms`, `list_sinks`, `get_plugin_schema`,
  `get_expression_grammar`, `explain_validation_error`,
  `get_plugin_assistance`, `list_models`, `get_audit_info`, `list_recipes`

Leave `cacheable=False` on:
- `get_pipeline_state`, `preview_pipeline`, `diff_pipeline` (session-mutable)

The `__post_init__` invariant enforces this for any mutation kind, but the
discovery-tier cacheability is opt-in. After migration is complete the
import-time assertion in `discovery.py` (lines 158, 163, 168) should still
hold via the declaration-derived sets.

### Verification per tool

Same pattern as Step 2:
- `pytest tests/unit/web/composer/test_tool_declarations.py -x`
- `pytest tests/unit/web/composer/`
- `ruff check src/elspeth/web/composer/`
- `mypy src/elspeth/web/composer/`
- `env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`

### Estimated effort

~13 commits if you migrate one tier per commit (discovery → mutation →
blob-discovery → secret → session-aware). Each tier touches one or two
plane modules + `_dispatch.py` + the test file. Expect 1-2 stale fingerprints
per commit from AST shift; rotate using `/tmp/rotate_fps.py` or the inline
Python script template in this session's history.

---

## Step 4 — Decommission hand-maintained surfaces

**Only run Step 4 after Step 3 is fully complete** (every tool except
possibly `request_interpretation_review` carries a declaration).

### What gets decommissioned

1. **`_dispatch.py:_DISCOVERY_TOOLS / _MUTATION_TOOLS / _BLOB_DISCOVERY_TOOLS /
   _BLOB_MUTATION_TOOLS / _SECRET_DISCOVERY_TOOLS / _SECRET_MUTATION_TOOLS`
   handler dicts.** Replace with `derive_handler_map_for(_REGISTERED_TOOLS,
   ToolKind.X)` calls at module scope (or inline at the dispatch site).
2. **`_dispatch.py:get_tool_definitions()` inline schema list.** Replace the
   ~1043-line body with: `return list(_TOOL_DEFS_BY_NAME.values())`. Order
   stability matters — the test in `test_skill_drift.py` likely expects a
   specific ordering, so verify the test still passes after the swap. If
   order changed, decide whether the test's expectation should track the
   declaration-tuple order (recommended) or the file-source-order order
   (require explicit sorting).
3. **`discovery.py:_DISCOVERY_TOOL_NAMES / _MUTATION_TOOL_NAMES / etc.`
   frozensets.** Either delete entirely (consumers grep
   `is_discovery_tool` / etc. predicates, which can be reimplemented as
   one-liners over `_REGISTERED_TOOLS`); or keep as derived constants
   computed from `derive_name_set_for`. Recommend deleting and switching
   the predicates to `derive_name_set_for` derivations.
4. **`discovery.py:_CACHEABLE_DISCOVERY_TOOL_NAMES /
   _SESSION_MUTABLE_DISCOVERY_TOOL_NAMES /
   _BLOB_STORE_ONLY_MUTATION_TOOL_NAMES`.** Same — derive from declarations.
   Keep the cacheable-vs-session-mutable disjointness assertion (move it
   to `_dispatch.py` after the derivation block).
5. **`tools/__init__.py:__all__` machinery.** Audit the list. Many entries
   (e.g. `_BLOB_QUOTA_MUTATION_TOOLS`, several `_TOOL_NAMES` frozensets,
   the registry dicts) become obsolete. Trim to what's actually imported
   by external consumers (`service.py`, `guided/steps.py`, etc.).
6. **`test_skill_drift.py:test_skill_tool_inventory_matches_get_tool_definitions`.**
   Per Refinement 3 (ticket comment 1434): **only the tool-inventory parser
   inside `TestComposerToolNameDrift` becomes obsolete** when the
   `get_tool_definitions()` return is derived. The rest of `test_skill_drift.py`
   remains — it guards drift in the per-tool prose, the worked
   `set_pipeline` example edges, the failsink plugin registry mention, etc.
   **DELETE only the tool-inventory parser and its test**; KEEP everything
   else in that file. Don't rewrite the whole test file.

### Skill-markdown derivation (limited scope)

The skill at `src/elspeth/web/composer/skills/pipeline_composer.md` has 2033
lines. **Only the "Foundation knowledge" section's tool-inventory bullets
(around line 51) are derivable** from `ToolDeclaration.description`. The
rest is behavioural prose that has no machine-readable analogue. Step 4 may
optionally add a build-time generator that produces the inventory bullets
from the declarations, but **do not attempt to derive the prose** — that's
a category error.

If you want the inventory bullets derived: add a small script
`scripts/cicd/generate_skill_inventory.py` that emits the bullets, hook it
into pre-commit, and gate via `--check` mode to fail if the skill drifts.
This replaces the deleted `test_skill_tool_inventory_matches_get_tool_definitions`.

### Verification

After Step 4 the dispatcher's `get_tool_definitions()` body should be a
small handful of lines (a `return list(_TOOL_DEFS_BY_NAME.values())` or
the equivalent). The discovery name-set frozensets either don't exist or
are derived. The composer test suite must still pass; the LLM-facing tool
list must still be byte-identical to the pre-Step-1 shape (the byte-identity
tests guard this per tool).

---

## Step 5 — Falsification measurement

### The test

Add a no-op tool `tool_40` (or any unused name) in a throwaway branch off
the post-Step-4 main. Measure `git diff --stat` for the diff that adds it.

Per Refinement 1 of comment 1434, decompose the file count:

- **(a) Boilerplate registration files** — declaration + handler. Should
  touch ≤2 files. **This is the falsification target.** If it touches
  ≥3 files, the declaration model failed to centralise ownership and the
  intervention is incomplete.
- **(b) Security-required annotation files** — MANIFEST entry in
  `redaction.py`. May legitimately touch 1 additional file (`redaction.py`)
  per the (B) decision. Record both counts.

### What to add

The fake tool can be a trivial discovery:

```python
def _execute_tool_40(arguments, state, context):
    return _discovery_result(state, {"ok": True})

_TOOL_40_DECLARATION = ToolDeclaration(
    name="tool_40",
    handler=_execute_tool_40,
    kind=ToolKind.DISCOVERY,
    description="No-op tool for falsification measurement.",
    json_schema={"type": "object", "properties": {}, "required": []},
    cacheable=False,
)
```

Add it to any plane's `TOOLS_IN_MODULE` (e.g., `sinks.py` or a new
`tools/_falsification.py` file). Also add a redaction MANIFEST entry in
`redaction.py` with `handles_no_sensitive_data=True` and a minimal
`HandlesNoSensitiveDataReason` (this is the security-required (b) touch).

### Record the result

Run `git diff --stat origin/main` and post to the ticket as a comment
*before closing*. Expected shape:

```
src/elspeth/web/composer/tools/<plane>.py  | <small N> +
src/elspeth/web/composer/redaction.py      | <small N> +
2 files changed, ...
```

If the boilerplate count > 2, **do not close the ticket**. Identify what
broke the centralisation and either fix the declaration model or document
the additional surface as a separate ticket.

### Closure criteria recap

Per the ticket description's closure section:

- A `ToolDeclaration` primitive exists, used by ≥1 tool cluster (per
  diagnoser). **Satisfied as of Step 1.**
- The falsification test (add tool #40 touches ≤2 files for boilerplate)
  passes in a throwaway branch and the measurement is recorded as a
  comment on this epic before close.
- The hand-maintained registries / MANIFEST for the migrated tier are
  decommissioned. **MANIFEST is intentionally not decommissioned (per (B));
  the rest is decommissioned at Step 4 completion.**

---

## Practical session-handling notes for the next agent

### Worktree + venv

The worktree's `.venv` is symlinked to main's `.venv`. The editable install
in main's venv points at `/home/john/elspeth/src/`, NOT at the worktree's
`src/`. Pytest's `pyproject.toml` config has `pythonpath = ["src"]` — when
you run pytest with the worktree as CWD, pytest adds `<worktree>/src` to
sys.path FIRST, so tests resolve to the worktree's source. **Always `cd`
to the worktree before running pytest.**

**Never** run `uv pip install` from inside the worktree without an explicit
`--python /path/to/worktree/.venv/bin/python` — the venv-leak memory
(`feedback_uv_venv_leak`) warns you'd clobber main's installation.

### Tier-model fingerprint rotation

Adding declarations + `TOOLS_IN_MODULE` blocks shifts AST positions in the
affected plane modules, which cascades fingerprint rotations downstream in
the file. Step 1 rotated 6; Step 2 rotated 23. Expect similar magnitude
per tier in Step 3.

The mapping is computed by AST-walking each file at the new finding's
line to recover the containing symbol, then pairing 1:1 with stale entries
by `(file, rule, symbol)` preserving source order. See
`/tmp/rotate_fps.py` (if it still exists from this session) or write the
equivalent — the algorithm is short.

After rotation, run:
```
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  check --rules trust_tier.tier_model --root src/elspeth
```

Exit code must be 0 before commit. The pre-commit hook will re-run it and
block the commit if there's drift.

### Filigree claim

The ticket is claimed by `claude-opus-tool-decl`. Filigree's permission
guard requires write operations (comments, status changes) to come from
the assignee. Either re-claim via
`mcp__filigree__start_work issue_id=elspeth-6c9972ccbf assignee=<your-actor>`
(use a distinct actor name from this session) or use the CLI with
`--actor claude-opus-tool-decl` to operate as the existing assignee.

The claim expires `2026-05-25T14:13:58Z`; refresh with `heartbeat_work`
or re-claim if working past that.

### Branch state

`feat/composer-tool-declarations` is local-only. Two commits:
- `72d4ef0f6` — Step 1
- `10b150d02` — Step 2

Don't merge to RC5.2 until Step 5's falsification measurement has landed
and been recorded on the ticket. Operator has not authorised a push or
merge.

### Don't do (lessons from this session)

- Don't put the registry aggregation in `declarations.py` — it creates a
  circular-import trap. Aggregation lives in `_dispatch.py`.
- Don't carry redaction on `ToolDeclaration` — the (B) decision is locked.
- Don't try to derive the LLM-facing JSON schema from the Pydantic argument
  model — they're structurally distinct artifacts; the JSON schema's prose
  is itself security-relevant.
- Don't oversell skill-markdown derivation. Most of the 2033 lines is
  behavioural prose with no machine-readable analogue.
- Don't migrate `request_interpretation_review` (async session-aware) in
  the same commit as another tier — its kwarg surface differs and may
  require type-level work.
