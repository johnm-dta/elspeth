# Composer ToolDeclaration paradigm — (a)-boilerplate falsification measurement

**Date:** 2026-05-24
**Context:** Closing measurement for epic `elspeth-6c9972ccbf` (composer tools — ToolDeclaration paradigm). Two empirical runs were captured on the throwaway branches `experiment/tool-40-falsification` and `experiment/tool-41-falsification`; this note consolidates the data so the branches can be deleted.

## Hypothesis being tested

After migrating composer tools to the ToolDeclaration paradigm + decommissioning the hand-maintained registries (epic `elspeth-6c9972ccbf`), the cost of adding one trivial no-op tool should drop. The pre-refactor cost involved editing multiple hand-maintained name frozensets, the `__all__` list, the redaction policy table, the skill markdown, the discovery registry, and the tier-model fingerprint allowlist — call this the *(a)-boilerplate* surface.

The falsification question: after the refactor, how many files / how many net lines does it actually take to introduce one trivial tool?

## Recovering the commits

If a future session needs the raw experiment diffs:

```
git reflog                                          # find HEAD~ pointers to the deleted branches
git show e08e5b205e644bf662cd3ac2e70facf9922ac3f9   # tool_40
git show 53dec7cec255ec4e63f17e7d590b9aaccb27713a   # tool_41
```

Both commits sit on top of RC5.2 (`e9f35b65f`). They will remain reachable via reflog until git gc prunes them (default ~30–90d depending on config), and can be recovered by SHA from `git fsck --lost-found` for longer.

## Run 1 — tool_40 (baseline)

- **SHA:** `e08e5b205e644bf662cd3ac2e70facf9922ac3f9`
- **Author date:** 2026-05-24 04:12 +1000
- **Setup:** add one no-op tool with `cacheable=False`
- **Net change:** 6 files / +55 / -5

| File | Lines |
|---|---|
| `config/cicd/enforce_tier_model/web.yaml` | +2 / -2 |
| `src/elspeth/web/composer/redaction.py` | +21 / -0 |
| `src/elspeth/web/composer/skills/pipeline_composer.md` | +1 / -1 |
| `src/elspeth/web/composer/tools/recipes.py` | +24 / -1 |
| `tests/data/web/composer/redaction_policy_snapshot.json` | +5 / -0 |
| `tests/unit/web/composer/test_tools.py` | +2 / -1 |

**Interpretive note:** `cacheable=False` routes the new tool into `_SESSION_MUTABLE_DISCOVERY_TOOL_NAMES`, which trips the design-review-checkpoint test that gates the stateful-discovery list. That test's docstring **explicitly frames its literal enumeration as intentional friction** — it forces a human review whenever a new stateful discovery tool is added. So the 6th file (`test_tools.py`) is not generic (a)-boilerplate; it's intentional governance friction, and the tool_41 re-run removes it from the count by switching to the typical no-op shape.

## Run 2 — tool_41 (post-Step-6 re-measurement)

- **SHA:** `53dec7cec255ec4e63f17e7d590b9aaccb27713a`
- **Author date:** 2026-05-24 04:33 +1000
- **Setup:** add one no-op tool with `cacheable=True` (the more typical no-op shape — a no-op IS safely cacheable); also reflects Step-6 closure of the two residual growth surfaces (skill markdown generator + retargeted membership tests).
- **Net change:** 5 files / +59 / -4

| File | Lines |
|---|---|
| `config/cicd/enforce_tier_model/web.yaml` | +2 / -2 |
| `src/elspeth/web/composer/redaction.py` | +21 / -0 |
| `src/elspeth/web/composer/skills/pipeline_composer.md` | +1 / -1 |
| `src/elspeth/web/composer/tools/recipes.py` | +30 / -1 |
| `tests/data/web/composer/redaction_policy_snapshot.json` | +5 / -0 |

**Verification at time of measurement:** 79 `test_tool_declarations.py` pass; 2164 composer unit tests pass; mypy + ruff + `trust_tier.tier_model` all exit 0.

## What the data shows

Adding one trivial no-op tool after the ToolDeclaration paradigm + post-Step-6 closure touches **5 files**, and three of those are file-class boilerplate that exists *by design* and is not amenable to further reduction:

- `web.yaml` — tier-model fingerprint allowlist; rotates on any AST shift, not paradigm-specific
- `redaction.py` — per-tool redaction policy entry; declarative-by-design
- `pipeline_composer.md` — skill markdown; auto-derived (the generator closed in Step-6)
- `redaction_policy_snapshot.json` — golden file mirroring `redaction.py`; required by snapshot tests
- `recipes.py` — the tool definition itself

The (a)-boilerplate hypothesis — "more than the irreducible surface" — was **falsified**: there is no extra registry/frozenset/`__all__`/discovery-list bookkeeping left to edit when adding a tool. The remaining surface is exactly what the paradigm requires.

The one file outside this set (`test_tools.py` in tool_40, +2/-1) is governance friction the system *wants* to impose when a new stateful discovery tool is added, not paradigm boilerplate.

## Why these branches are being deleted

The empirical evidence has been preserved in this note. The branches themselves contain no production-bound code — both add no-op tools that exist solely as measurement instrumentation. Keeping them would (a) clutter `git branch` listings with names whose meaning is non-obvious without context, and (b) risk a future session merging an experimental no-op tool into prod.

Branches removed: 2026-05-24.
