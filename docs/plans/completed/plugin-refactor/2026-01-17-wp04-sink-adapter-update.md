# WP-04: Delete SinkAdapter & SinkLike

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the adapter layer entirely - sinks now implement batch interface directly.

**Rationale:** WP-03 made sinks batch-aware with ArtifactDescriptor returns. SinkAdapter and SinkLike are now redundant indirection layers that served to bridge "Phase 2 row-wise sinks" to "Phase 3B batch interface". With sinks implementing batch directly, these layers add complexity without value.

**Tech Stack:** Python 3.12, file deletion, import cleanup

**Unlocks:** WP-13 (sink test rewrites can proceed knowing no adapter exists)

---

## Task 1: Delete adapters.py and test_adapters.py

**Files:**
- Delete: `src/elspeth/engine/adapters.py`
- Delete: `tests/engine/test_adapters.py`

**Step 1: Verify no other code imports from adapters.py**

```bash
grep -r "from elspeth.engine.adapters import\|from elspeth.engine import.*SinkAdapter" src/ --include="*.py" | grep -v "adapters.py"
```

Expected: Only `cli.py`, `orchestrator.py`, and `__init__.py` import SinkAdapter

**Step 2: Delete the files**

```bash
rm src/elspeth/engine/adapters.py
rm tests/engine/test_adapters.py
```

**Step 3: Commit deletion**

```bash
git add -A && git commit -m "$(cat <<'EOF'
refactor(engine): delete SinkAdapter and test_adapters

SinkAdapter was a bridge between Phase 2 row-wise sinks and Phase 3B
batch interface. Now that sinks implement batch directly (WP-03),
this layer is redundant.

BREAKING: SinkAdapter removed from public API

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Delete SinkLike from executors.py

**Files:**
- Modify: `src/elspeth/engine/executors.py`

**Step 1: Find SinkLike protocol definition**

```bash
grep -n "class SinkLike" src/elspeth/engine/executors.py
```

**Step 2: Delete the SinkLike protocol class**

Remove the entire `SinkLike` class (approximately lines 668-692, verify exact location).

**Step 3: Update SinkExecutor to use SinkProtocol**

Change the type hint in `SinkExecutor.write()`:

```python
# BEFORE
from somewhere import SinkLike

def write(self, sink: SinkLike, ...) -> ...:

# AFTER
from elspeth.plugins.protocols import SinkProtocol

def write(self, sink: SinkProtocol, ...) -> ...:
```

**Step 4: Remove SinkLike from .contracts-whitelist.yaml**

```bash
# Check for SinkLike entries
grep -n "SinkLike" .contracts-whitelist.yaml
```

Remove any lines referencing `SinkLike` - these contracts no longer exist.

**Step 5: Run mypy on executors.py**

```bash
mypy src/elspeth/engine/executors.py --strict
```

**Step 6: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
refactor(executors): delete SinkLike, use SinkProtocol directly

SinkLike was an engine-internal protocol duplicating SinkProtocol.
Now that SinkProtocol is batch-aware, use it directly.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update orchestrator.py to use SinkProtocol

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`

**Step 1: Remove SinkLike import**

```python
# DELETE this line
from elspeth.engine.executors import SinkLike
```

**Step 2: Add SinkProtocol import**

```python
from elspeth.plugins.protocols import SinkProtocol
```

**Step 3: Update PipelineConfig type hint**

```python
@dataclass
class PipelineConfig:
    source: SourceProtocol
    transforms: list[RowPlugin]  # or whatever the current type is
    sinks: dict[str, SinkProtocol]  # Changed from SinkLike
    config: dict[str, Any] = field(default_factory=dict)
```

**Step 4: Remove type: ignore comments**

Search for `type: ignore` comments related to sinks and remove them - types should now align.

**Step 5: Remove stale SinkLike comments**

Search for comments explaining "SinkLike" vs "SinkProtocol" distinction and delete them:

```bash
grep -n "SinkLike" src/elspeth/engine/orchestrator.py
```

For example, line 49 has a comment like:
```python
]  # Engine uses SinkLike (batch write), not SinkProtocol (single row)
```

This is now incorrect - delete it. The engine uses `SinkProtocol` directly.

**Step 6: Run mypy on orchestrator.py**

```bash
mypy src/elspeth/engine/orchestrator.py --strict
```

**Step 7: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
refactor(orchestrator): use SinkProtocol instead of SinkLike

PipelineConfig.sinks now typed as dict[str, SinkProtocol].
Removes stale comments and type: ignore hacks.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.5: Update `_export_landscape` for direct sink usage

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`

**Rationale:** The `_export_landscape` method has its own SinkAdapter dependency that Task 3 doesn't cover. It unwraps SinkAdapter to call row-wise writes, but sinks are now batch-aware.

**Step 1: Remove SinkAdapter import from `_export_landscape`**

Delete the local import around line 668:
```python
# DELETE this line
from elspeth.engine.adapters import SinkAdapter
```

**Step 2: Replace unwrap logic with batched write**

Current code (lines ~666-692):
```python
# Unwrap SinkAdapter if present (adapter expects bulk writes,
# but export writes records individually)
from elspeth.engine.adapters import SinkAdapter

artifact_path: str | None = None
if isinstance(sink, SinkAdapter):
    artifact_path = sink.artifact_path
    raw_sink = sink._sink
else:
    raw_sink = sink

if export_config.format == "csv":
    # ... csv path ...
else:
    # JSON export: write records one at a time to single sink
    for record in exporter.export_run(run_id, sign=export_config.sign):
        raw_sink.write(record, ctx)
    raw_sink.flush()
    raw_sink.close()
```

Replace with:
```python
if export_config.format == "csv":
    # ... csv path unchanged ...
else:
    # JSON export: batch all records, single write
    records = list(exporter.export_run(run_id, sign=export_config.sign))
    if records:
        artifact_descriptor = sink.write(records, ctx)
        # artifact_path available via artifact_descriptor.path_or_uri if needed
    sink.flush()
    sink.close()
```

**Step 3: Update comments referencing SinkAdapter**

Remove stale comments at lines ~581 and ~710 that mention SinkAdapter.

**Step 4: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
refactor(orchestrator): update _export_landscape for direct sink usage

- Remove SinkAdapter unwrapping logic
- Batch export records for single write() call
- Simpler AND more efficient (1 write vs N writes)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update CLI to use sinks directly

**Files:**
- Modify: `src/elspeth/cli.py`

**Step 1: Remove SinkAdapter import**

```python
# DELETE this line
from elspeth.engine.adapters import SinkAdapter
```

**Step 2: Update sink creation to use sinks directly**

```python
# BEFORE (with adapter wrapping)
sinks: dict[str, SinkAdapter] = {}
...
artifact_descriptor = {"kind": "file", "path": sink_options["path"]}
sinks[sink_name] = SinkAdapter(
    raw_sink,
    plugin_name=sink_plugin,
    sink_name=sink_name,
    artifact_descriptor=artifact_descriptor,
)

# AFTER (direct sink usage)
sinks: dict[str, SinkProtocol] = {}  # Update type hint
...
# Remove artifact_descriptor construction entirely
sinks[sink_name] = raw_sink  # That's it - orchestrator sets node_id later
```

**IMPORTANT:** Do NOT set `node_id` in CLI. The orchestrator sets it from DAG registration
(see orchestrator.py:268: `sink.node_id = sink_id_map[sink_name]`).

**Step 3: Remove artifact_descriptor construction**

Delete all `artifact_descriptor = {...}` lines. The adapter needed this to know how to
build ArtifactDescriptor. Sinks now return their own ArtifactDescriptor from write().

**Step 4: Run CLI to verify it works**

```bash
# If you have a test config
python -m elspeth --settings test_settings.yaml --dry-run
```

**Step 5: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
refactor(cli): use sinks directly without adapter wrapper

Sinks are now batch-aware and return ArtifactDescriptor directly.
No adapter layer needed.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Remove SinkAdapter from engine/__init__.py exports

**Files:**
- Modify: `src/elspeth/engine/__init__.py`

**Step 1: Remove SinkAdapter from imports and __all__**

```python
# DELETE these lines
from elspeth.engine.adapters import SinkAdapter

__all__ = [
    ...,
    "SinkAdapter",  # DELETE this
    ...,
]
```

**Step 2: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
refactor(engine): remove SinkAdapter from public exports

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Run full verification

**Step 1: Verify no SinkAdapter references remain**

```bash
grep -r "SinkAdapter" src/ tests/ --include="*.py"
```

Expected: No results

**Step 2: Verify no SinkLike references remain**

```bash
grep -r "SinkLike" src/ tests/ --include="*.py"
```

Expected: No results

**Step 3: Run mypy on engine module**

```bash
mypy src/elspeth/engine/ --strict
```

**Step 4: Run all engine tests**

```bash
pytest tests/engine/ -v
```

Note: Some tests may fail if they depended on SinkAdapter or SinkLike. These failures are expected and will be fixed in WP-13/WP-14.

**Step 5: Run sink tests**

```bash
pytest tests/plugins/sinks/ -v
```

**Step 6: Final commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
chore: verify WP-04 SinkAdapter/SinkLike deletion complete

- adapters.py deleted
- test_adapters.py deleted
- SinkLike deleted from executors.py
- SinkProtocol used directly in orchestrator and executors
- CLI uses sinks directly without wrapper
- No SinkAdapter or SinkLike references remain

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

- [ ] `src/elspeth/engine/adapters.py` deleted
- [ ] `tests/engine/test_adapters.py` deleted
- [ ] `SinkLike` protocol deleted from `executors.py`
- [ ] `SinkExecutor.write()` uses `SinkProtocol` type hint
- [ ] `orchestrator.py` imports `SinkProtocol`, not `SinkLike`
- [ ] `PipelineConfig.sinks` typed as `dict[str, SinkProtocol]`
- [ ] `_export_landscape` uses batched `sink.write()` directly (no SinkAdapter unwrap)
- [ ] `cli.py` creates sinks directly (no `SinkAdapter` wrapper)
- [ ] `engine/__init__.py` does not export `SinkAdapter`
- [ ] `grep -r "SinkAdapter" src/` returns nothing
- [ ] `grep -r "SinkLike" src/` returns nothing
- [ ] `.contracts-whitelist.yaml` has no SinkLike entries
- [ ] No stale comments referencing SinkLike remain in source files
- [ ] `mypy --strict` passes on engine module
- [ ] Sink tests pass

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/engine/adapters.py` | DELETE | Remove entire file |
| `tests/engine/test_adapters.py` | DELETE | Remove entire file |
| `src/elspeth/engine/executors.py` | MODIFY | Delete SinkLike, update SinkExecutor |
| `src/elspeth/engine/orchestrator.py` | MODIFY | Use SinkProtocol, update PipelineConfig, update `_export_landscape`, remove stale comments |
| `.contracts-whitelist.yaml` | MODIFY | Remove SinkLike entries |
| `src/elspeth/engine/__init__.py` | MODIFY | Remove SinkAdapter export |
| `src/elspeth/cli.py` | MODIFY | Use sinks directly |

---

## Dependency Notes

- **Depends on:** WP-03 (sinks must have batch signatures with ArtifactDescriptor)
- **Unlocks:** WP-13 (sink test rewrites)
- **Risk:** Medium - CLI and orchestrator integration needs careful testing

---

## Expected Test Failures

After this WP, some engine tests may fail because they:
1. Import `SinkAdapter` or `SinkLike`
2. Use `MockSink` from `test_adapters.py`
3. Expect adapter behavior

These will be fixed in WP-13 (sink test rewrites) and WP-14 (engine test rewrites).
