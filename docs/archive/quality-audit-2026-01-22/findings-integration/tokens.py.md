## Summary

TokenManager persists source row payloads itself and passes `payload_ref` into LandscapeRecorder.create_row, leaking payload storage/canonicalization details across the engine↔landscape boundary instead of encapsulating them in the recorder API.

## Severity

- Severity: major
- Priority: P1

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [x] Leaky Abstraction (implementation details cross boundaries)
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

**Boundary:** engine ↔ core/landscape

**Integration Point:** TokenManager.create_initial_token ↔ LandscapeRecorder.create_row (source row creation + payload persistence)

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: engine/TokenManager (`src/elspeth/engine/tokens.py`)

`src/elspeth/engine/tokens.py:75-90`

```python
75	        # Store payload if payload_store is configured (audit requirement)
76	        payload_ref = None
77	        if self._payload_store is not None:
78	            # Use canonical_json to handle pandas/numpy types, Decimal, datetime, etc.
79	            # This prevents TypeError crashes when source data contains non-primitive types
80	            payload_bytes = canonical_json(row_data).encode("utf-8")
81	            payload_ref = self._payload_store.store(payload_bytes)
82
83	        # Create row record with payload reference
84	        row = self._recorder.create_row(
85	            run_id=run_id,
86	            source_node_id=source_node_id,
87	            row_index=row_index,
88	            data=row_data,
89	            payload_ref=payload_ref,
90	        )
```

### Side B: core/landscape/recorder (`src/elspeth/core/landscape/recorder.py`)

`src/elspeth/core/landscape/recorder.py:725-775`

```python
725	    def create_row(
726	        self,
727	        run_id: str,
728	        source_node_id: str,
729	        row_index: int,
730	        data: dict[str, Any],
731	        *,
732	        row_id: str | None = None,
733	        payload_ref: str | None = None,
734	    ) -> Row:
...
748	        row_id = row_id or _generate_id()
749	        data_hash = stable_hash(data)
...
757	            source_data_ref=payload_ref,
```

### Coupling Evidence: payload persistence split across layers

`src/elspeth/core/landscape/recorder.py:231-239` and `src/elspeth/core/landscape/recorder.py:2072-2082`

```python
231	    def __init__(self, db: LandscapeDB, *, payload_store: Any | None = None) -> None:
...
238	        self._db = db
239	        self._payload_store = payload_store
...
2072	        # Auto-persist request to payload store if available and ref not provided
2073	        # This enables replay/verify modes to retrieve the original request
2074	        if request_ref is None and self._payload_store is not None:
2075	            request_bytes = canonical_json(request_data).encode("utf-8")
2076	            request_ref = self._payload_store.store(request_bytes)
```

## Root Cause Hypothesis

Payload storage was added as an engine-side responsibility for source rows while the recorder already owns payload persistence for other audit records (calls), leaving create_row without a unified façade and forcing engine code to know recorder internals (canonical_json + store).

## Recommended Fix

[Concrete steps to resolve the seam issue]

1. Move source-row payload persistence into `LandscapeRecorder.create_row` using its `_payload_store` (if configured) so recorder owns serialization and storage.
2. Remove `payload_store` handling and `canonical_json` calls from `TokenManager.create_initial_token`, passing only `row_data`.
3. Update call sites constructing `TokenManager` to stop passing `payload_store` once recorder handles persistence.
4. Remove or make internal the `payload_ref` parameter on `create_row` to prevent bypassing recorder-owned persistence.
5. Add tests asserting `create_row` stores `source_data_ref` when payload_store is configured and that `get_row_data` returns `AVAILABLE`.

## Impact Assessment

- **Coupling Level:** High - engine must currently know payload storage details.
- **Maintainability:** Medium - changes to payload storage logic require engine updates.
- **Type Safety:** Low - `payload_store` is `Any` and the contract is implicit.
- **Breaking Change Risk:** Medium - refactoring recorder/create_row and TokenManager affects engine initialization paths.

## Related Seams

`src/elspeth/engine/orchestrator.py`, `src/elspeth/core/checkpoint/recovery.py`, `src/elspeth/core/landscape/recorder.py`
---
Template Version: 1.0
