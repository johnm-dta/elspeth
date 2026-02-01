## Summary

RowProcessor rejects protocol-only plugins by requiring BaseGate/BaseTransform subclasses, contradicting the plugin contract that allows protocol implementations.

## Severity

- Severity: major
- Priority: P1

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [x] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** engine ↔ plugins

**Integration Point:** runtime plugin dispatch/type detection in RowProcessor

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: engine/processor (`src/elspeth/engine/processor.py`:656-662,728-862)

```python
656            # Type-safe plugin detection using base classes
657            if isinstance(transform, BaseGate):
658                # Gate transform
659                outcome = self._gate_executor.execute_gate(
660                    gate=transform,
661                    token=current_token,
662                    ctx=ctx,
...
728            elif isinstance(transform, BaseTransform):
...
861            else:
862                raise TypeError(f"Unknown transform type: {type(transform).__name__}. Expected BaseTransform or BaseGate.")
```

### Side B: plugins/base contract (`src/elspeth/plugins/base.py`:2-6)

```python
2  """Base classes for plugin implementations.
3
4  These provide common functionality and ensure proper interface compliance.
5  Plugins can subclass these for convenience, or implement protocols directly.
6
```

### Coupling Evidence: plugin factory uses protocol types (`src/elspeth/plugins/manager.py`:333-358)

```python
333    def create_transform(self, transform_type: str, config: dict[str, Any]) -> TransformProtocol:
334        """Create transform plugin instance with validated config.
...
355        plugin_cls = self.get_transform_by_name(transform_type)
...
358        return plugin_cls(config)
```

## Root Cause Hypothesis

Engine dispatch logic was implemented around concrete base classes for runtime checks, while the plugin API documentation and factory paths evolved to protocol-based contracts, leaving the processor’s runtime gate out of sync with the plugin contract.

## Recommended Fix

1. Decide a single canonical plugin contract (Base* inheritance vs protocol-only) and document it in one place.
2. If protocol-only is valid, update `src/elspeth/engine/processor.py` to dispatch using `GateProtocol`/`TransformProtocol` (and align other runtime checks) instead of Base* classes.
3. Add an integration test that runs a protocol-only transform/gate through the processor to prevent regressions.

## Impact Assessment

- **Coupling Level:** Medium
- **Maintainability:** Medium
- **Type Safety:** Low
- **Breaking Change Risk:** Medium

## Related Seams

`src/elspeth/engine/orchestrator.py` (BaseTransform check in `_validate_transform_error_sinks`)
`src/elspeth/plugins/protocols.py` (protocol contracts referenced by plugin API)
---
Template Version: 1.0
