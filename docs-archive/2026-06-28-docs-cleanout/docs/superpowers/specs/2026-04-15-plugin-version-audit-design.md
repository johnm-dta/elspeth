# Plugin Version Audit & Source File Hash Enforcement

**Date:** 2026-04-15
**Status:** Approved (revised after panel review)
**Scope:** Plugin versioning discipline, source-file-hash change detection, CI enforcement, landscape integration

## Problem

Every plugin currently declares `plugin_version = "1.0.0"` (25 plugins) or omits it entirely and defaults to `"0.0.0"` (7 plugins). The version string has no mechanical enforcement — developers can change plugin behaviour without bumping the version, and the audit trail records a meaningless constant.

The 7 omitting plugins are the heavyweight ones most likely to have behavioural changes:

| Plugin | Type | File |
|---|---|---|
| `AzureBlobSource` | Source | `sources/azure_blob_source.py` |
| `DataverseSource` | Source | `sources/dataverse.py` |
| `DataverseSink` | Sink | `sinks/dataverse.py` |
| `RAGRetrievalTransform` | Transform | `transforms/rag/transform.py` |
| `LLMTransform` | Transform | `transforms/llm/transform.py` |
<!-- AzureBatchLLMTransform / OpenRouterBatchLLMTransform retired 2026-05-06 per ADR-020. -->

## Design

### Two-channel versioning

Each plugin carries two version signals:

- **`plugin_version`** (human-readable semver) — communicates the nature of a change to operators and auditors. Bumped by developer judgment.
- **`source_file_hash`** (SHA-256 of source file content) — mechanically detects whether the plugin's entry-point source file changed. Computed deterministically; enforced by CI.

```python
class LLMTransform(BaseTransform):
    name = "llm"
    plugin_version = "1.0.0"
    source_file_hash = "sha256:e4d909c290d0fb1c"
```

The name `source_file_hash` (not `code_hash`) is deliberate — it communicates that only the entry-point file is hashed, not the full import closure. See "What source_file_hash Does Not Detect" below.

### Hash format and computation

`sha256:<first-16-hex-chars>` — truncated SHA-256 of the plugin module's file content. 16 hex chars = 64-bit collision space, sufficient for uniqueness across the plugin set.

**Computation rules:**

1. **Read raw bytes** (`file_path.read_bytes()`), not decoded text. This prevents CRLF/BOM platform divergence.
2. **Normalize the `source_file_hash` line to a placeholder before hashing.** The hash must not include its own value (self-referential hash is an impossible fixed point). Before computing, the line matching `source_file_hash = "sha256:..."` is replaced with `source_file_hash = "sha256:0000000000000000"`. Both `check` and `--fix` modes use the identical normalization.
3. **Hash the normalized content.** `sha256(normalized_bytes).hexdigest()[:16]`.

### Protocol and base class changes

**Protocols** (`contracts/plugin_protocols.py`):
- `SourceProtocol`: add `source_file_hash: str | None`
- `TransformProtocol`: add `source_file_hash: str | None`
- `BatchTransformProtocol`: add `source_file_hash: str | None`
- `SinkProtocol`: add `source_file_hash: str | None`

**Base classes** (`plugins/infrastructure/base.py`):
- `BaseSource`: add `source_file_hash: str | None = None`
- `BaseTransform`: add `source_file_hash: str | None = None`
- `BaseSink`: add `source_file_hash: str | None = None`

`None` is the "not declared" sentinel. CI fails on `None`. This aligns the type through the full flow (base class → CI → landscape) with no `""` → `None` coercion.

Note: `@runtime_checkable` protocols check only callable members, not data attributes. The `_: type[Protocol] = Impl` static conformance pattern (enforced by mypy) catches missing `source_file_hash` at type-check time, not runtime.

### Plugin changes (32 plugins)

- 7 plugins that omit `plugin_version`: add explicit `"1.0.0"`
- All 32 plugins: add `source_file_hash = "sha256:<computed>"` with the correct hash

### CI enforcement script

New: `scripts/cicd/enforce_plugin_hashes.py`

**Discovery:** Recursively scans plugin directories (`rglob("*.py")`) rather than mirroring the flat `PLUGIN_SCAN_CONFIG` glob. This ensures subdirectory plugins (`transforms/llm/`, `transforms/rag/`, `transforms/azure/`) are included. The script validates its discovery count against a hardcoded expected plugin count and fails if it finds fewer plugins than expected (guards against silent scan regressions).

**AST extraction:** Reads declared `source_file_hash` via `ast.parse()` to avoid import side effects. Extracts class-level string assignments from `ast.ClassDef` bodies. If zero assignments are found in a recognized plugin class, reports "no source_file_hash at class body level" with file and class name — crashes loudly rather than silently reporting "missing."

**Failure modes:**

| Condition | Result |
|---|---|
| `source_file_hash` is `None` (base class default) | FAIL — "plugin has no source_file_hash declaration" |
| `source_file_hash` != computed hash | FAIL — "stale source_file_hash, expected sha256:abc123" |
| `plugin_version` is `"0.0.0"` | FAIL — "plugin has no version declaration" |
| All match | PASS |

**Subcommands:**
- `check` — verify and fail on mismatch (CI and pre-commit mode)
- `check --fix` — auto-update stale values in-place (manual developer command only)
- `check --root <path> --allowlist <dir>` — standard allowlist for exceptions

**Auto-fix rewrite strategy:** Uses AST to find the exact line number of the `source_file_hash = "sha256:..."` assignment, then replaces that single line via `source_lines[line_number - 1]`. Preserves all formatting, indentation, comments, and surrounding code. No regex (fragile on multi-line strings), no AST round-trip (discards comments).

**Developer workflow:**
1. Change plugin code
2. Pre-commit runs `enforce_plugin_hashes.py check` — fails with correct hash
3. Developer runs `enforce_plugin_hashes.py check --fix`
4. Developer stages the updated file, recommits

**Pre-commit integration:** Added to `.pre-commit-config.yaml` in **check-only mode** (matching the existing convention: "Hooks are CHECK-ONLY — they do NOT auto-fix or modify files"). The `--fix` mode is a developer CLI convenience, not a pre-commit action. This avoids the staged/working-tree divergence footgun.

### Landscape integration

**Audit dataclass** (`contracts/audit.py`): Add `source_file_hash: str | None` to `Node`. Nullable for backwards compatibility.

**Format validation in `__post_init__`:** Non-None values must match `sha256:[0-9a-f]{16}`. Invalid format → crash (Tier 1 crash-on-anomaly, consistent with existing `_validate_enum` pattern).

**Landscape nodes table**: Add `source_file_hash` column (nullable `TEXT`).

**Schema compatibility:** `metadata.create_all()` does not ALTER existing tables. The implementation must handle existing databases that lack the `source_file_hash` column. Options (to be decided during implementation planning):
- Bump schema epoch and add migration logic in `_validate_schema()`
- `NodeLoader` handles missing column gracefully (maps to `None`)
Both approaches must be tested against a real pre-existing audit database.

**`register_node()`** (`core/landscape/data_flow_repository.py`): Add `source_file_hash: str | None = None` parameter.

**Orchestrator** (`engine/orchestrator/core.py`): At registration, reads `plugin.source_file_hash` and passes to `register_node()`. Config gates and coalesce nodes pass `source_file_hash=None` (engine-internal, not versioned plugins).

**Three independent audit dimensions per node:**

| Field | Tracks | Updated by |
|---|---|---|
| `config_hash` | Plugin configuration changes | Automatic (canonical JSON hash) |
| `plugin_version` | Developer-communicated version | Developer judgment |
| `source_file_hash` | Entry-point file changes | Deterministic file hash |

### MCP / TUI visibility

The `source_file_hash` field is included in:
- Landscape MCP analysis server node detail responses
- TUI `explain` node detail panel
- JSON/CSV export node records

When displaying `source_file_hash` comparisons between runs, the UI should note: "Source file hash detects changes to the plugin's entry-point module. Changes to imported helper modules are not captured — see 'What source_file_hash Does Not Detect' below."

## What `source_file_hash` Does Not Detect

`source_file_hash` is a necessary but not sufficient condition for behavioural equivalence between runs. It detects changes to the plugin's own module file. It does NOT detect:

1. **Helper module changes.** `LLMTransform` imports from 12+ sibling modules (`templates.py`, `providers/azure.py`, `multi_query.py`, etc.). A change to retry backoff logic in `providers/azure.py` produces no change to `transforms/llm/transform.py`'s hash, but fundamentally alters runtime behaviour.

2. **Base class changes.** `BaseTransform.__init__()` or `BaseTransform._build_output_schema_config()` affects every transform's behaviour. No plugin's `source_file_hash` changes.

3. **Shared utility changes.** `field_collision.py`, `canonical.py`, or client library modules are imported by multiple plugins. Changes propagate behaviourally but not through the hash.

**Operator guidance:** A matching `source_file_hash` between two runs means the plugin's entry-point file was identical. To confirm full behavioural equivalence, also compare `config_hash`, check git history for changes to the plugin's directory and shared imports, and verify library versions.

**Future work:** Transitive dependency hashing (hashing the full import closure) would close this gap. It is deliberately deferred — the file-level hash is the 80/20 improvement over the current zero-discipline baseline. The honest naming (`source_file_hash`, not `code_hash`) prevents the hash from being mistaken for a behavioural equivalence guarantee.

## Out of scope

- **Transitive dependency hashing:** Deferred as documented above.
- **Semver enforcement:** No CI gate on whether a bump is patch/minor/major. `source_file_hash` handles mechanical change detection; `plugin_version` remains a human communication tool.
- **Existing landscape backfill:** Old runs keep `source_file_hash = NULL`. No backfill.

## Test strategy

### CI script tests (6 cases)

1. **Happy path:** compute hash for a known file, verify match passes
2. **Stale hash detection:** modify file content, verify mismatch detected with correct expected hash
3. **Missing declaration:** plugin with `source_file_hash = None` (base default), verify failure message
4. **Missing `plugin_version`:** plugin with `"0.0.0"`, verify failure message
5. **`--fix` mode correctness:** run `--fix` on a stale file, verify the hash is updated to the correct value
6. **`--fix` mode idempotency:** run `--fix` twice, verify file bytes are identical after both runs

### Hash computation edge cases (4 cases)

7. **Self-referential normalization:** verify that computing the hash with different `source_file_hash` values in the file produces the same result (normalization is working)
8. **Binary stability:** verify `read_bytes()` produces consistent hashes across platforms (no CRLF/BOM sensitivity)
9. **AST extraction variants:** test `source_file_hash = "sha256:..."` (no space), `source_file_hash: str = "sha256:..."` (annotated), both extracted correctly
10. **Multiple classes in one file:** verify the correct class's `source_file_hash` is extracted when a file contains helper classes

### Protocol conformance (1 case)

11. **Static conformance:** mypy enforces `source_file_hash` via `_: type[Protocol] = Impl` pattern. Verify mypy catches a plugin that omits the attribute.

### Landscape integration (4 cases)

12. **Store and retrieve:** `register_node()` with `source_file_hash` stores and reads back correctly
13. **Backward compatibility:** open an existing audit DB lacking the `source_file_hash` column; verify reads return `Node(source_file_hash=None)` without error
14. **Mixed-run query:** compare nodes from an old run (NULL hash) with a new run (non-NULL hash) without error
15. **Format validation:** `Node.__post_init__` crashes on `source_file_hash="invalid"` but accepts `None` and valid `sha256:<16-hex>`

### Orchestrator integration (2 cases)

16. **Plugin nodes:** run a pipeline, verify every plugin node in the landscape has non-None `source_file_hash`
17. **Engine nodes:** verify config gates and coalesce nodes have `source_file_hash=None`

### Regression (2 cases)

18. **Test fixture audit:** identify all mock/fake transforms in tests that need `source_file_hash` added; verify full test suite passes
19. **Initial CI run:** run enforcement script against all 32 plugins with zero false positives on the initial commit
