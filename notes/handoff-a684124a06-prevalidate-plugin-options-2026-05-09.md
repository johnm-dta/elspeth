# Handoff — fix `elspeth-a684124a06` (`_prevalidate_plugin_options` silent-None on unknown plugin)

**Date prepared:** 2026-05-09
**Prepared during:** systematic-debugging Phase 1+2 across composer-correctness + composer-llm-eval umbrellas
**Issue:** `elspeth-a684124a06` (P2 bug, parented under `elspeth-e1ab67e55a` composer-correctness umbrella)
**Status when handed off:** Phase 1 (root cause) and Phase 2 (pattern + call-site verification) complete. Phase 3 (hypothesis) is committed. Phase 4 (implementation) is your job.

---

## TL;DR

`src/elspeth/web/composer/tools.py:2147-2149` catches `UnknownPluginTypeError` and `return None`s. Returning `None` from this function means "options are valid" to every caller. The result: unknown plugin names get silently accepted into composition state, the LLM sees `success: True`, and the bad data sits there until engine-time validation crashes far from the cause.

This is structurally identical to the anti-pattern fixed in `elspeth-c5b7343431` (commit `5f30d7c6` on RC5.1, April 2026). The *fix mechanism* differs from c5b7343431 (see "Working reference" below), but the *intent* is the same: convert a silent fail-open into a structured rejection.

The fix is one line plus a comment removal. Three new failing tests prove the regression; existing 20-odd tests in the class must stay green.

---

## Pre-conditions before starting

1. **Verify `composer-ux-tier1` has been merged into RC5.1.** Run from `/home/john/elspeth`:

   ```bash
   git -C /home/john/elspeth log --oneline RC5.1 -- src/elspeth/web/composer/tools.py | head -10
   ```

   You're looking for a recent commit that adds `_prepend_rejection_entry` and a rewritten `_failure_result`. If the latest commit on RC5.1 is older than 2026-05-09 and `git diff RC5.1 -- src/elspeth/web/composer/tools.py` against the `composer-ux-tier1` worktree still shows the `_prepend_rejection_entry` helper as a delta, the merge has not happened yet — **stop and check with the operator**.

2. **Verify no parallel work is touching `_prevalidate_plugin_options`.** As of handoff prep, three worktrees existed:

   - `.worktrees/composer-convergence` (branch `feat/composer-convergence`) — building `compute_proof_diagnostics` + `preview_pipeline` proof step. Touches `composer/tools.py` heavily but **does not modify `_prevalidate_plugin_options`**. Verify still true:

     ```bash
     cd /home/john/elspeth/.worktrees/composer-convergence \
       && git diff main..HEAD -- src/elspeth/web/composer/tools.py | grep -E "_prevalidate_plugin_options|UnknownPluginTypeError"
     ```

     Expected output: nothing. If anything matches, the convergence branch has started touching the same surface and you must coordinate before proceeding.

   - `.worktrees/composer-progress-1a` (branch `feat/composer-progress-persistence-1a`) — sessions Phase 1A/1B. Touches `composer/tools.py` incidentally but not the prevalidate function. Same verification command applies.

3. **Confirm working tree on RC5.1 is clean.**

   ```bash
   git -C /home/john/elspeth status --short
   ```

   The two notes files (`composer-skill-gaps-2026-05-09.md`, `composer-bigbrother-investigation-2026-05-09.md`, plus this handoff) are expected as untracked. Anything else, stop and ask.

4. **Optional but recommended: do this work in a worktree.** Per project memory (`.worktrees/<name>` inside project; main venv is Python 3.13 so the worktree venv must match):

   ```bash
   cd /home/john/elspeth
   git worktree add .worktrees/a684124a06 -b fix/a684124a06-prevalidate-unknown-plugin RC5.1
   cd .worktrees/a684124a06
   uv venv .venv --python 3.13
   uv pip install -e ".[dev]" --python /home/john/elspeth/.worktrees/a684124a06/.venv/bin/python
   ```

   The explicit `--python` is load-bearing — per project memory, omitting it causes uv to clobber the main worktree's `.venv`.

---

## The bug

**File:** `src/elspeth/web/composer/tools.py`
**Function:** `_prevalidate_plugin_options` (defined at line 2094)
**Offending block:** lines 2147-2149

```python
except UnknownPluginTypeError:
    # Plugin name not in registry — let engine validation catch it later.
    return None
```

The function's contract (lines 2100-2106):

> Returns None if valid, or a descriptive error message suitable for returning to the LLM agent.

So `return None` here is contractually equivalent to "the options are valid" — which is false. The plugin name is unknown; the options can't be valid because the plugin doesn't exist. The comment's "let engine validation catch it later" is the c5b7343431 anti-pattern verbatim.

### Why this matters

`_prevalidate_plugin_options` is invoked by 7 mutation tool executors (via three thin wrappers `_prevalidate_source`/`_prevalidate_transform`/`_prevalidate_sink`):

- `_execute_set_source` (tools.py:2370)
- `_execute_set_pipeline` (tools.py:4044, 4073, 4095, 4123)
- `_execute_set_source_from_blob` (tools.py:2620 area — verify line)
- `_execute_upsert_node` (tools.py:2420)
- `_execute_set_output` (tools.py:2620)
- `_execute_patch_source_options` (tools.py:4306)
- `_execute_patch_node_options` (tools.py:4391)
- `_execute_patch_output_options` (tools.py:4453)

All 7 follow this identical pattern:

```python
prevalidation_error = _prevalidate_<kind>(plugin, options, ...)
if prevalidation_error is not None:
    return _failure_result(state, prevalidation_error)
# ... mutation proceeds ...
```

So when the function returns `None` for an unknown plugin, the mutation goes through. The unknown plugin name lands in `CompositionState`. The LLM gets `success: True`. The error doesn't surface until the engine instantiates the pipeline and crashes — or worse, until `/validate` finally rejects it after a turn the LLM thought it had completed.

---

## The fix

Replace the catch with a structured error string. Symmetric with the `ValueError as exc` branch immediately below (lines 2150-2152) which already returns a structured error for the same return channel:

```python
except UnknownPluginTypeError:
    return f"Unknown {plugin_type} plugin '{plugin_name}'. Call list_{plugin_type}s to see available {plugin_type} plugins."
```

Notes on this exact wording:

- The `list_<kind>s` tools exist (`list_sources`, `list_transforms`, `list_sinks`); the LLM already has them in its tool list.
- The error string is written to be *actionable* — it names the next call. This matches the rejection-mutation skill text added by `ae4acc03` ("discourage explain_validation_error on rejected_mutation").
- The error returns through the `str | None` channel of `_prevalidate_plugin_options`, which routes via `_prevalidate_<kind>` wrappers to the call site's `_failure_result(state, prevalidation_error)` — and `_failure_result` (post ux-tier1 merge) prepends a `rejected_mutation` `ValidationEntry`, putting the message at `validation.errors[0]` where the LLM's coached read order finds it first.

**Comment removal:** delete the `# Plugin name not in registry — let engine validation catch it later.` line. Per CLAUDE.md "No Legacy Code Policy" + "Default to writing no comments" — the line is now both obsolete (we no longer defer to engine validation) and self-explanatory at the new code.

**Do not add a comment justifying the new return.** The behaviour is symmetric with the `ValueError` branch four lines below; a future reader can read both branches.

---

## Working reference (c5b7343431, commit 5f30d7c6)

The structural anti-pattern (catch `UnknownPluginTypeError` → return None) was previously fixed in `src/elspeth/web/catalog/service.py:_resolve_config_model`. To read that fix:

```bash
git -C /home/john/elspeth show 5f30d7c6 -- src/elspeth/web/catalog/service.py
```

**Important caveat — the fix mechanism is NOT transferable.** c5b7343431 worked because `_resolve_config_model` is a method on `CatalogServiceImpl` and had access to `self._pm` (a `PluginManager` instance). The fix swapped *global validation dispatch* (`get_*_config_model` from the validation module) for *PluginManager-direct* (`self._pm.get_*_by_name(name).get_config_model()`). That removed the global-dispatch path entirely, which is why dropping the catch was safe.

`_prevalidate_plugin_options` is a module-level function with no PluginManager handle. It uses the global validation dispatch by design — there is no PluginManager to switch to without refactoring callers. So:

- **Don't** try to plumb a PluginManager through. That's scope creep.
- **Do** keep the global-dispatch calls. They're correct; the catch is the bug.
- The reference value of c5b7343431 is the *intent* (silent acceptance of unknown plugin names is the wrong shape), not the *mechanism*.

If reading the c5b7343431 commit makes you want to do something larger, that's a yellow flag — the issue says "one-line fix" and Phase 2 verified that's accurate.

---

## Phase 2 evidence already gathered

The hand-off includes these conclusions so you don't redo them:

1. **No call site of any of the three `_prevalidate_<kind>` wrappers depends on the silent-None behaviour for unknown plugin names.** Every caller already expects the function to return a string for any error and routes the string into `_failure_result`. Returning a string instead of None for the unknown-plugin case slots into the existing rejection path with zero behavioural change for known plugins.

2. **Existing test class `TestPrevalidatePluginOptions` at `tests/unit/web/composer/test_tools.py:6663`** has roughly 20 cases covering valid/invalid options for *known* plugins. None tests the unknown-plugin path. That is the test gap your TDD pass closes.

3. **The post-`3938b6d3` rejection-mutation infrastructure** (lead `validation.errors` with `rejected_mutation` entry) is the channel by which the new error reaches the LLM. After ux-tier1 merges, this is fully wired. Don't re-implement it; just trust that returning a string from `_prevalidate_plugin_options` ends up at `validation.errors[0]`.

---

## TDD plan (Phase 4 step 1 of systematic-debugging)

Use `superpowers:test-driven-development` if your harness supports it; if not, follow the structure below by hand. **Failing tests first; no fix until they fail.**

### Where the new tests live

`tests/unit/web/composer/test_tools.py`, inside the existing `TestPrevalidatePluginOptions` class (search for `class TestPrevalidatePluginOptions:` around line 6663). Add the new methods at the end of the class so the existing tests stay in their current order.

### The three failing tests

```python
def test_unknown_transform_plugin_returns_actionable_error(self) -> None:
    """Regression for elspeth-a684124a06: unknown plugin name must surface
    as a structured error, not silently pass via return None.

    Mirrors elspeth-c5b7343431's fix in _resolve_config_model — same
    anti-pattern (catch UnknownPluginTypeError, return None) in a
    different surface where silent acceptance lets bad plugin names
    into composition state.
    """
    result = _prevalidate_plugin_options(
        "transform",
        "this_plugin_does_not_exist",
        {"some": "options"},
    )
    assert result is not None, (
        "Unknown plugin must surface as error, not pass silently."
    )
    assert "this_plugin_does_not_exist" in result
    assert "transform" in result.lower()
    assert "list_transforms" in result, (
        "Error must name the discovery tool the LLM should call next."
    )

def test_unknown_source_plugin_returns_actionable_error(self) -> None:
    """Source-side variant of test_unknown_transform_plugin_returns_actionable_error.

    The thin wrapper _prevalidate_source routes through the same catch;
    this guards the source-specific call sites (set_source, set_pipeline
    source spec, patch_source_options).
    """
    result = _prevalidate_plugin_options(
        "source",
        "no_such_source_plugin",
        {"path": "/data/blobs/in.csv"},
    )
    assert result is not None
    assert "no_such_source_plugin" in result
    assert "source" in result.lower()
    assert "list_sources" in result

def test_unknown_sink_plugin_returns_actionable_error(self) -> None:
    """Sink-side variant. Guards set_output, set_pipeline output specs,
    and patch_output_options call sites.
    """
    result = _prevalidate_plugin_options(
        "sink",
        "no_such_sink_plugin",
        {"path": "/data/outputs/out.csv"},
    )
    assert result is not None
    assert "no_such_sink_plugin" in result
    assert "sink" in result.lower()
    assert "list_sinks" in result
```

### Run the tests; confirm they fail

From the worktree (or main if not using a worktree):

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py::TestPrevalidatePluginOptions -x -v
```

The three new tests must fail with `assert None is not None` (or similar). The other ~20 tests in the class must pass. If any pre-existing test fails, **stop**: that's a Phase 1 signal that your branch base is dirty or the merge produced a regression. Investigate before proceeding.

---

## Apply the fix

Edit `src/elspeth/web/composer/tools.py:2147-2149`. Replace:

```python
    except UnknownPluginTypeError:
        # Plugin name not in registry — let engine validation catch it later.
        return None
```

With:

```python
    except UnknownPluginTypeError:
        return f"Unknown {plugin_type} plugin '{plugin_name}'. Call list_{plugin_type}s to see available {plugin_type} plugins."
```

(Verify the `plugin_type` variable is in scope — it's the function's first parameter; line 2094.)

---

## Verification (Phase 4 step 3)

Run in this order, stop at the first failure:

```bash
# 1. The new TestPrevalidatePluginOptions cases — must now pass.
.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py::TestPrevalidatePluginOptions -x -v

# 2. The full composer test_tools.py — must stay green; no other test regressed.
.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -x

# 3. Type checking on the changed file.
.venv/bin/python -m mypy src/elspeth/web/composer/tools.py

# 4. Lint on the changed file.
.venv/bin/python -m ruff check src/elspeth/web/composer/tools.py

# 5. Tier-model check on the changed file (project memory: venv must be Python 3.13 or this gives ~300 spurious violations).
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model

# 6. The whole composer test suite, just to be safe.
.venv/bin/python -m pytest tests/unit/web/composer/ -x
```

If any step fails:

- **Steps 1–2 fail:** the fix is wrong or there's an unexpected interaction with another caller. Return to Phase 1; do not attempt a second fix on top.
- **Step 3 fails:** likely a type signature mismatch in your wording. The function signature is `(plugin_type: PluginKind, plugin_name: str, options: dict[str, Any], *, injected_fields: dict[str, Any] | None = None) -> str | None`. The error string satisfies the return type.
- **Step 4 fails:** probably line length. Wrap the f-string across two lines using parens.
- **Step 5 fails:** check your venv's Python version per project memory; if it's 3.13 and tier-model still complains, the issue is real and needs investigation.

If everything passes, stop. Do not commit yet.

---

## Stop-before-commit (CLAUDE.md git-safety)

Per CLAUDE.md "Never run destructive git commands without explicit user permission" and the broader "no commits without explicit user permission" pattern this project follows: present the diff to the operator and wait for the nod.

```bash
git -C /home/john/elspeth/.worktrees/a684124a06 diff
git -C /home/john/elspeth/.worktrees/a684124a06 diff -- tests/
```

Show the operator the two diffs (source + tests) and wait for explicit "commit" before:

```bash
# Only after operator approval:
git -C /home/john/elspeth/.worktrees/a684124a06 add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git -C /home/john/elspeth/.worktrees/a684124a06 commit -m "fix(composer): _prevalidate_plugin_options surfaces unknown plugin as actionable error

The catch on UnknownPluginTypeError silently returned None, equivalent to
'options are valid', letting unknown plugin names land in CompositionState
with success=True. Same anti-pattern as elspeth-c5b7343431 (commit 5f30d7c6)
in a different surface; mechanism differs (no PluginManager handle here)
but intent matches: convert silent fail-open into a structured rejection
that flows through the existing _failure_result -> rejected_mutation
ValidationEntry path.

Closes elspeth-a684124a06.

Co-Authored-By: <your model identity>"
```

Use a HEREDOC for the message per CLAUDE.md commit-formatting guidance if your harness expects it.

**Do not push without explicit operator permission.** Per project memory and CLAUDE.md.

---

## Filigree integration on success

After the commit lands and the operator confirms the change is good:

```bash
filigree --actor <agent-name> close elspeth-a684124a06 \
  --reason "Fixed in commit <sha>. _prevalidate_plugin_options now returns a structured error for unknown plugin names instead of silently returning None. Same anti-pattern as elspeth-c5b7343431 (commit 5f30d7c6) in a different surface; fix mechanism differs (no PluginManager available) but intent matches. Regression tests added in TestPrevalidatePluginOptions covering source/transform/sink unknown-plugin paths."
```

If MCP filigree tools are configured, prefer `mcp__filigree__close_issue` over the CLI per CLAUDE.md.

---

## Out of scope (do not bundle)

Per CLAUDE.md "Don't add features, refactor, or introduce abstractions beyond what the task requires":

- Do not refactor `_prevalidate_plugin_options` to take a PluginManager. Scope creep.
- Do not edit `_prevalidate_source`/`_prevalidate_transform`/`_prevalidate_sink` wrappers. They're fine.
- Do not touch the `ValueError as exc` branch (lines 2150-2152). Already symmetric with the new code.
- Do not add catch-list extensions for `af7c270380`'s 3 hard-mode bugs (`fb00d281c7`, `9286f57d35`, `f221b29bcc`). Those are parked pending the convergence-branch merge — adding them here is parallel work that risks colliding with the convergence-branch's `compute_proof_diagnostics` mechanism.
- Do not add audit-trail comments, no `# Fixes elspeth-a684124a06` markers in code. Per CLAUDE.md "Don't reference the current task, fix, or callers in code comments — those belong in the PR description and rot."

---

## What "done" looks like

- One change to `src/elspeth/web/composer/tools.py` (one block of three lines becomes one line).
- One block of three new test methods appended to `TestPrevalidatePluginOptions` in `tests/unit/web/composer/test_tools.py`.
- All six verification steps above passing.
- Diff presented to operator; commit only on operator nod.
- Filigree issue `elspeth-a684124a06` closed with the structured close-reason above.

Total elapsed time should be ~30-60 minutes, of which most is verification, not editing.

---

## When to stop and escalate

Per the systematic-debugging skill's red-flag list:

- Pre-existing test in `TestPrevalidatePluginOptions` fails before your changes — base is dirty. Stop.
- Step 5 (tier-model check) fails after your one-line edit — that's almost certainly Python-version venv pollution, but verify the change isn't actually triggering it. Stop and check before applying a workaround.
- More than two attempts needed to make the new tests pass — Phase 4.5 of systematic-debugging says question architecture, don't keep fixing on top. Read both branches of the catch logic again.
- Any caller of `_prevalidate_plugin_options` turns out to actually depend on silent-None for a path I didn't surface — escalate. The Phase 2 evidence said no caller does, but if you find a counter-example, that's new information that changes the fix shape.

End of handoff.
