## Summary

[One-sentence description of the integration seam defect or architectural anti-pattern]

## Severity

- Severity: [critical|major|minor|trivial]
- Priority: [P0|P1|P2|P3]

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [ ] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** [subsystem A] ↔ [subsystem B]

**Integration Point:** [interface/protocol/data model/exception boundary]

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: [subsystem/file]

```python
[Code showing definition/usage on side A with line numbers]
```

### Side B: [subsystem/file]

```python
[Code showing definition/usage on side B with line numbers]
```

### Coupling Evidence: [translation layer/assumption/shared state]

```python
[Code showing the coupling mechanism - translation functions, adapters, etc.]
```

## Root Cause Hypothesis

[Why did this happen? Usually: independent development, missing shared contract, legacy code, etc.]

## Recommended Fix

[Concrete steps to resolve the seam issue]

1. [Create canonical definition in contracts/]
2. [Remove duplicate definitions]
3. [Update imports]
4. [Remove translation layer]
5. [Add type enforcement]

## Impact Assessment

- **Coupling Level:** [Low|Medium|High] - How tightly are the systems coupled?
- **Maintainability:** [Low|Medium|High] - How hard is it to change one side?
- **Type Safety:** [Low|Medium|High] - Are violations caught at compile time?
- **Breaking Change Risk:** [Low|Medium|High] - Will fixing this break existing code?

## Related Seams

[List other files/interfaces that may have similar issues]

---
Template Version: 1.0
