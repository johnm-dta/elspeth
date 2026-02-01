# Bug Report: Plugin discovery ignores protocol-only implementations

## Summary

- Plugin docs say implementations may “subclass base classes or implement protocols directly.”
- Dynamic discovery only accepts subclasses of `BaseSource`/`BaseTransform`/`BaseSink`.
- A plugin class that implements `SourceProtocol`/`TransformProtocol`/`SinkProtocol` directly (without inheriting the base class) is silently skipped and never registered.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into contents of `src/elspeth/plugins` and create bug tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/plugins/base.py` and `src/elspeth/plugins/discovery.py`

## Steps To Reproduce

1. Create a source class in `src/elspeth/plugins/sources/` that implements `SourceProtocol` but does not inherit `BaseSource`.
2. Ensure it defines a non-empty `name` attribute and implements `load()`/`close()`.
3. Call `discover_all_plugins()`.
4. Observe that the class is not returned or registered.

## Expected Behavior

- Protocol-only plugins should be discoverable if the system advertises protocol-based implementations as supported.

## Actual Behavior

- Discovery filters on `issubclass(obj, BaseSource/BaseTransform/BaseSink)` only, so protocol-only implementations are skipped.

## Evidence

- Base class docs say protocol-only implementations are allowed: `src/elspeth/plugins/base.py`.
- Discovery requires subclassing a base class: `src/elspeth/plugins/discovery.py` (`issubclass(obj, base_class)`).

## Impact

- User-facing impact: valid plugin implementations are never registered, leading to confusing “plugin not found” behavior.
- Data integrity / security impact: low.
- Performance or cost impact: low.

## Root Cause Hypothesis

- Discovery logic was implemented against base classes only and never updated to honor protocol-only implementations.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/discovery.py`: accept classes that satisfy the relevant runtime-checkable protocol (e.g., `issubclass(obj, SourceProtocol)`) in addition to subclassing base classes.
  - Alternatively, update documentation to require subclassing base classes and make discovery failure explicit (raise on protocol-only classes if found).
- Tests to add/update:
  - Add a test plugin that implements `SourceProtocol` without inheriting `BaseSource` and assert it is discovered (or explicitly rejected if policy changes).
- Risks or migration steps:
  - If changing discovery policy, ensure no false positives for helper classes by validating required attributes/methods.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/base.py` module docstring (protocol-only support).
- Observed divergence: discovery only supports base-class inheritance.
- Reason (if known): discovery keyed to base class for simplicity.
- Alignment plan or decision needed: either support protocol-only discovery or update docs to remove the claim.

## Acceptance Criteria

- Protocol-only plugin classes are discoverable (or explicitly rejected with a clear error if policy is updated).

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/plugins/test_discovery.py`
- New tests required: yes (protocol-only plugin discovery)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/plugins/base.py`

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 4

**Current Code Analysis:**

The bug is still present. The discrepancy between documentation and implementation remains:

1. **Documentation claim** (`src/elspeth/plugins/base.py:1-5`):
   ```python
   """Base classes for plugin implementations.

   These provide common functionality and ensure proper interface compliance.
   Plugins can subclass these for convenience, or implement protocols directly.
   ```
   This docstring has been present since commit `06df9e5` (initial implementation) and remains unchanged.

2. **Discovery implementation** (`src/elspeth/plugins/discovery.py:115`):
   ```python
   # Must inherit from base_class (but not BE base_class)
   if not issubclass(obj, base_class) or obj is base_class:
       continue
   ```
   Discovery strictly requires `issubclass(obj, BaseSource/BaseTransform/BaseSink)` to pass.

3. **Technical limitation discovered:**
   The plugin protocols (`SourceProtocol`, `TransformProtocol`, `SinkProtocol`) are marked `@runtime_checkable`, BUT they contain non-method members (`name`, `output_schema`, `determinism`, `plugin_version`, etc.).

   Python's `typing.Protocol` raises `TypeError` when `issubclass()` is called on protocols with non-method members:
   ```
   TypeError: Protocols with non-method members don't support issubclass().
   Non-method members: 'determinism', 'name', 'node_id', 'output_schema', 'plugin_version'.
   ```

   Only `isinstance()` works on protocol instances, not `issubclass()` on protocol classes.

4. **Current reality:**
   - All 21 existing plugins inherit from base classes (verified via grep of all plugin files)
   - No protocol-only plugins exist in the codebase
   - The claim in documentation is aspirational but technically unimplemented

**Git History:**

No commits have addressed this issue. The module docstring has remained unchanged since the original implementation. Recent commits related to protocols:
- `430307d`: Added schema validation to plugin protocols (didn't change discovery)
- `a7c65bd`: Fixed crash on import errors and duplicate names in discovery (didn't add protocol support)
- `2da1747`: Enforced PluginProtocol type contracts via mypy (type-checking only, not runtime discovery)

**Root Cause Confirmed:**

Yes, the bug is confirmed and has two dimensions:

1. **Documentation-implementation mismatch:** The docstring claims protocol-only implementations are supported, but discovery only recognizes base class inheritance.

2. **Technical impossibility with current protocol design:** Even if discovery wanted to check `issubclass(obj, SourceProtocol)`, it cannot due to Python's limitation on protocols with non-method members. The protocols would need to be redesigned (attributes moved to methods or separate protocol) to support `issubclass()`.

**Recommendation:**

**Close as OBE - documentation should be corrected, not implementation changed.**

**Rationale:**

1. **No user impact:** Zero protocol-only plugins exist. All plugins inherit base classes.

2. **Technical correctness:** The base class approach is superior for this architecture:
   - Enforces `_validate_self_consistency()` via `__init_subclass__` hook
   - Provides default implementations of lifecycle methods
   - Works with standard Python `issubclass()` without protocol limitations
   - Follows "Plugin Ownership: System Code, Not User Code" principle (CLAUDE.md)

3. **Protocol design limitation:** Making protocols work with `issubclass()` would require removing attribute declarations and converting them to property methods, which degrades type-checking quality.

4. **Simple fix:** Update module docstring from:
   ```
   Plugins can subclass these for convenience, or implement protocols directly.
   ```
   To:
   ```
   Plugins must subclass these base classes for proper discovery and lifecycle enforcement.
   Protocols define the type contracts but are not sufficient for runtime discovery.
   ```

5. **Alignment with architecture:** Per CLAUDE.md, plugins are "system-owned code, not user-provided extensions." Requiring base class inheritance is appropriate enforcement.

**Suggested action:** Update documentation to remove the misleading claim, then close the bug. This is a documentation defect, not a code defect.

---

## CLOSURE: 2026-01-29

**Status:** FIXED (Documentation)

**Fixed By:** Claude Opus 4.5

**Resolution:**

Updated the module docstring in `src/elspeth/plugins/base.py` to:

1. **Remove the misleading claim** that plugins can "implement protocols directly"
2. **Explain why base class inheritance is required:**
   - Plugin discovery uses `issubclass()` checks against base classes
   - Python's `Protocol` with non-method members cannot support `issubclass()`
   - Base classes enforce self-consistency via `__init_subclass__` hooks
   - Aligns with CLAUDE.md "Plugin Ownership" principle

**Updated docstring:**

```python
"""Base classes for plugin implementations.

These provide common functionality and ensure proper interface compliance.
Plugins MUST subclass these base classes (BaseSource, BaseTransform, BaseSink).

Why base class inheritance is required:
- Plugin discovery uses issubclass() checks against base classes
- Python's Protocol with non-method members (name, determinism, etc.) cannot
  support issubclass() - only isinstance() on already-instantiated objects
- Base classes enforce self-consistency via __init_subclass__ hooks
- Per CLAUDE.md "Plugin Ownership", all plugins are system code, not user extensions

The protocol definitions (SourceProtocol, TransformProtocol, SinkProtocol) exist
for type-checking purposes only - they define the interface contract but cannot
be used for runtime discovery.
...
"""
```

**Classification:** This was a documentation defect, not a code defect. The implementation correctly requires base class inheritance; only the documentation was misleading.

**Verified By:** Claude Opus 4.5 (2026-01-29)
