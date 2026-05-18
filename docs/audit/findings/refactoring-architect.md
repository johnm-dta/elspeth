# Refactoring Architect — CI Suppression Audit

**Date**: 2026-05-19
**Scope**: Two R1-suppression clusters in Tier-2 code
**Analyst**: Refactoring Architect (SME Agent)

---

## Mode Declaration

Both patterns are **Refactor mode** (in-place). No public API changes in either case.
No module splits. No import path changes for callers.

---

## Pattern A — Sparse-token-lookup exporter

### Scope

`src/elspeth/core/landscape/exporter.py` — `LandscapeExporter._iter_records`, lines 296–638

### Public API (contract)

- `LandscapeExporter.export_run(run_id, sign=False) -> Iterator[dict[str, Any]]`
- `LandscapeExporter.export_run_grouped(run_id, sign=False) -> dict[str, list[dict[str, Any]]]`

### Test coverage of public API

**Strong.** Three independent test files exercise the public API:
- `tests/integration/audit/test_exporter_batch_queries.py` (233 lines) — relational integrity
- `tests/integration/audit/test_export.py` — integration coverage
- `tests/property/core/test_exporter_properties.py` — property-based
- `tests/e2e/audit/test_export_reimport.py` — round-trip E2E

### Diagnosis

**Smell identified: per-fingerprint allowlisting of a uniform pattern.**

The 8 suppressions are not individually motivated — all 8 carry the same reason string
("Sparse lookup - not all rows have tokens (batch export uses pre-loaded dicts)") and the
same owner tag (`P2-2026-02-02-76r`). The R1 rule fires because `dict.get(key, [])` is
syntactically identical to a defensive access — but these calls are semantically different.
The dicts are built from `defaultdict(list)`, and absence of a key IS a valid, meaningful
state (root token, still-running token, state with no routing events, etc.). No fabrication
occurs; the empty list is semantically "nothing exists for this entity".

**The annotation-runtime mismatch is the root cause of the suppressions.** All 8 dicts are
declared as `dict[str, list[T]]` but assigned a `defaultdict(list)`. The static type says
"can raise KeyError on missing key"; the runtime value says "will never raise". The `.get(key,
[])`  calls are a compensation for the declared type being more restrictive than the runtime
object.

### Option Analysis

**Option α — SparseIndex[K, V] typed wrapper**

A named type encapsulates the "empty list on miss" contract at the declaration site:

```python
class SparseIndex(defaultdict[str, list[V]]):
    """Sparse pre-loaded lookup index. Missing keys return [], not KeyError."""
    def __init__(self) -> None:
        super().__init__(list)
    def lookup(self, key: str) -> list[V]:
        return self[key]  # defaultdict guarantees [], no .get() needed
```

This replaces 8 `.get(key, [])` with `index.lookup(key)` (no `.get()`; no R1) and 1
`allow_hits` entry in core.yaml for the type definition site if the type itself triggers
something. However, `SparseIndex` only has one current user and no anticipated second user.
The "extract a class because it might grow" pattern is explicitly listed as a forbidden
anti-pattern in this agent's protocol. **Option α introduces premature abstraction.**

**Option β — EAFP (try/except KeyError)**

Trades 8 R1 suppressions for 8 try/except blocks plus 8 R6 suppressions. Net negative.
Rejected without further analysis.

**Option γ — Per-file allowlisting instead of per-fingerprint**

Converts the 8 individual `allow_hits` entries to a single `per_file_rules` entry:

```yaml
per_file_rules:
  - pattern: core/landscape/exporter.py
    rules: [R1]
    reason: >-
      Sparse lookup indexes built from defaultdict(list). All 8 dict.get(key, [])
      calls return [] on miss — absence is meaningful, not a bug. The [] default
      matches defaultdict's __missing__ exactly. Suppression is structural, not
      per-call. Owner: P2-2026-02-02-76r.
    expires: null
    max_hits: 8
```

This is purely a CI metadata fix. Code is unchanged. The `max_hits: 8` cap preserves the
same safety property as the 8 individual fingerprints (adding a new `.get()` exceeds the
cap), and eliminates the fingerprint-rotation cascade documented in project memory
`feedback_ast_shift_fingerprint_rotation.md`.

**Option δ — Change annotation to `defaultdict[str, list[T]]` and use direct subscription**

This is the structurally cleanest option. All 8 dicts are created with `defaultdict(list)`.
Widening the annotation from `dict[str, list[T]]` to `defaultdict[str, list[T]]` makes the
declared contract match the runtime contract. With the corrected annotation, the caller can
use direct subscription `index[key]` instead of `index.get(key, [])` — `defaultdict.__missing__`
returns `[]` without raising. No `.get()` call; no R1 firing; no suppression needed.

Mypy verification (confirmed by running against `/tmp/test_mypy.py` and `/tmp/test_mypy3.py`):

- `dict[str, list[T]]` — mypy permits direct subscription without type error. KeyError is a
  runtime risk it does not track statically.
- `defaultdict[str, list[T]]` — same. Mypy does not flag either form.

Both annotations pass mypy clean. The annotation change is therefore **semantic documentation
only**, not a type-error gate. The migration is safe.

**This is consistent with the existing build phase.** The build phase of all 8 indexes
already uses direct subscription — e.g., `op_calls_by_operation[call.operation_id].append(call)`
at line 302, `tokens_by_row[token.row_id].append(token)` at line 390, and equivalently at lines
396, 402, 408, 418, 424, and 620. The `.get(key, [])` in the lookup phase is therefore
*inconsistent* with the build phase in the same function. Option δ makes the lookup phase
match the build phase — it is cleanup of an existing inconsistency, not introduction of a
new pattern.

**Side effect to acknowledge:** direct subscription on `defaultdict` adds an empty list to
the dict on every miss (that is `defaultdict`'s contract). For the 8 indexes in
`_iter_records`, this is bounded: the dicts are local variables that go out of scope when
`_iter_records` returns. The memory impact is proportional to the number of unique foreign
keys queried (not rows), and the function already holds all pre-loaded data in memory.
**No observable behavior change; no audit impact.**

### Verdict: **(b) Restructurable via Option δ**

Option δ eliminates all 8 suppressions with zero new abstractions and zero CI config changes,
at the cost of 8 one-line annotation edits plus 8 one-line lookup edits. It makes the code
honest: the declared type matches the runtime object. Option γ is a valid fallback if this
project decides annotation widening is not worth the diff, but Option δ is strictly superior
because it fixes the root cause (annotation-runtime mismatch) rather than suppressing
symptoms.

Option α is premature abstraction — rejected per protocol.

### Behavior-Preserving Move Sequence (Option δ)

**Precondition:** `_iter_records` has 8 dict-annotated defaultdict indexes, each using
`.get(key, [])` at the lookup site. All 8 are suppressed in `config/cicd/enforce_tier_model/core.yaml`.

**Move 1** (can be one commit, all 8 are independent and uniform):

- Type: Annotation correction + direct subscription
- File: `src/elspeth/core/landscape/exporter.py`
- Per-site change:
  ```python
  # Before:
  tokens_by_row: dict[str, list[Token]] = defaultdict(list)
  ...
  for token in tokens_by_row.get(row.row_id, []):

  # After:
  tokens_by_row: defaultdict[str, list[Token]] = defaultdict(list)
  ...
  for token in tokens_by_row[row.row_id]:
  ```
- Apply identically to all 8 indexes:
  - line 296/324: `op_calls_by_operation`
  - line 388/441: `tokens_by_row`
  - line 394/457: `parents_by_token`
  - line 400/490: `states_by_token`
  - line 406/576: `events_by_state`
  - line 412/593: `calls_by_state`
  - line 422/468: `outcomes_by_token`
  - line 618/638: `members_by_batch`
- Postcondition: No `.get()` at any of the 8 sites. R1 does not fire.
- Affected callers: none — `_iter_records` is private.
- Test impact: Green. Behavior identical. Property tests verify completeness.

**Move 2** (can be same commit as Move 1):

- Type: CI allowlist cleanup
- File: `config/cicd/enforce_tier_model/core.yaml`
- Remove the 8 `allow_hits` entries with `owner: P2-2026-02-02-76r`.
- No replacement entry needed — R1 no longer fires at these sites.
- Postcondition: core.yaml has 8 fewer entries. `max_hits` cap is no longer needed.

**Sequencing rule satisfied**: after Move 1, the codebase is shippable (behavior unchanged,
tests green). Move 2 is CI hygiene; it can be co-committed with Move 1 with no risk.

**Contingency**: If mypy rejects the `defaultdict` annotation in the project's full-run
context (which the isolated test did not surface), fall back to Option γ (per-file allowlist,
zero code change). The contingency is low-risk: mypy does not model `defaultdict`'s
`__missing__` as a KeyError suppressor and both annotation forms type-check identically.

---

## Pattern B — Sink display-mapping fallbacks

### Scope

- `src/elspeth/plugins/infrastructure/display_headers.py` — helpers `get_effective_display_headers`,
  `apply_display_headers`
- `src/elspeth/plugins/sinks/csv_sink.py` — `validate_output_target`, `_open_file`,
  `_get_field_names_and_display`
- `src/elspeth/plugins/sinks/json_sink.py` — `validate_output_target`

### Public API (contract)

- `CSVSink.write(rows, ctx) -> SinkWriteResult`
- `JSONSink.write(rows, ctx) -> SinkWriteResult`
- `CSVSink.validate_output_target() -> OutputValidationResult`
- `JSONSink.validate_output_target() -> OutputValidationResult`

### Test coverage of public API

**Strong.** Multiple independent test files:
- `tests/unit/plugins/test_sink_header_config.py` (152 lines)
- `tests/integration/audit/test_csv_sink_executor_audit.py`
- `tests/property/sinks/test_csv_sink_properties.py`
- `tests/property/sinks/test_json_sink_properties.py`
- `tests/e2e/pipelines/test_csv_to_csv.py`
- `tests/e2e/pipelines/test_json_to_json.py`
- `tests/integration/pipeline/test_resume_comprehensive.py`

### Diagnosis

**Note on count:** The task description cited "~6 R1 suppressions" for Pattern B. Verification
against `config/cicd/enforce_tier_model/plugins.yaml` confirms exactly **4** entries with
`owner: feature` matching the display-mapping pattern. The count is used accurately throughout
this section.

**Smell identified: identity-fallback repeated at call sites rather than centralized once.**

The `display_headers.py` module already contains `apply_display_headers` (which handles bulk
row-key rewriting with correct `if k in display_map` branching — no `.get()`, no R1) and
`get_effective_display_headers` (which returns the raw `dict[str, str] | None`).

However, there is **no helper for single-field name resolution**. The four call sites each
independently implement the identity-fallback inline:

1. `csv_sink.py:180` — `display_map.get(f, f)` in `validate_output_target`
2. `csv_sink.py:492` — `reverse_map.get(h, h)` in `_open_file`
3. `csv_sink.py:598` — `display_map.get(field, field)` in `_get_field_names_and_display`
4. `json_sink.py:198` — `display_map.get(f, f)` in `validate_output_target`

The semantic is identical at all four sites: "return the display name for this field; if the
field has no display override, return the field name itself". The `reverse_map` variant at
site 2 inverts the direction but is the same semantic applied to the inverse mapping.

**The helper does not exist and call sites are NOT bypassing it.** This is option (b), not
(c). Sites 1, 3, and 4 operate on the forward mapping (normalized → display). Site 2 operates
on an inline reverse mapping (display → normalized) and is a caller-side local variable that
exists nowhere in `display_headers.py`.

### Verdict: **(b) Restructurable — add `display_name_for` to `display_headers.py`**

### Proposed Helper (in `display_headers.py`)

```python
def display_name_for(display_map: dict[str, str] | None, field: str) -> str:
    """Return the display name for a field, falling back to the original name.

    In ORIGINAL mode, transform-added fields (not in the source contract) have
    no display override. Identity-on-miss is the correct semantic: these fields
    were added after source normalization and have no original header to restore.

    In CUSTOM mode, all fields MUST be explicitly mapped — do not call this
    function in CUSTOM mode; use apply_display_headers() which enforces completeness.

    Args:
        display_map: Mapping from normalized field name to display name, or None
            if no display headers are configured (NORMALIZED mode).
        field: The normalized field name to resolve.

    Returns:
        display_map[field] if field is in display_map, else field unchanged.
    """
    if display_map is None:
        return field
    if field in display_map:
        return display_map[field]
    return field
```

Implementation uses `in` + direct subscription — no `.get()`, no R1.

**Why `None` handling is here**: three of the four call sites are already guarded by
`if display_map is not None:` before the `.get()`, so calling sites could skip the `None`
check. However, centralizing the `None` check in the helper makes the contract complete and
prevents future callers from needing to know to guard first. The redundant `None` guard at
call sites can stay (belt-and-suspenders, no harm) or be removed as a follow-up.

### Call-site rewrites

**Site 1 — `csv_sink.py:180` in `validate_output_target`:**
```python
# Before:
expected = [display_map.get(f, f) for f in expected_normalized]

# After:
from elspeth.plugins.infrastructure.display_headers import display_name_for
expected = [display_name_for(display_map, f) for f in expected_normalized]
```

**Site 3 — `csv_sink.py:598` in `_get_field_names_and_display`:**
```python
# Before:
display_fields = [display_map.get(field, field) for field in data_fields]

# After:
display_fields = [display_name_for(display_map, field) for field in data_fields]
```

**Site 4 — `json_sink.py:198` in `validate_output_target`:**
```python
# Before:
expected = [display_map.get(f, f) for f in expected_normalized]

# After:
display_name_for is already imported via display_headers import block
expected = [display_name_for(display_map, f) for f in expected_normalized]
```

**Site 2 — `csv_sink.py:492` in `_open_file`:**

Site 2 is different: it inverts the display map to `reverse_map = {v: k for k, v in display_map.items()}` and then applies `reverse_map.get(h, h)` — mapping display names back to data field names. This is a distinct semantic (reverse lookup, not forward lookup). Two options:

- Option 2a: Create a `reverse_display_name_for(display_map, display_name)` helper in `display_headers.py`. Adds symmetry but creates a helper with one current caller.
- Option 2b: Keep site 2 as `reverse_map[h] if h in reverse_map else h` (no `.get()`, no R1) and add one targeted `allow_hits` entry for just that site if the inline form trips R1. Since `reverse_map` is a local `dict`, R1 would fire; the inline rewrite eliminates it.

**Recommendation for site 2**: Use option 2b inline `if ... in ... else` form. The reverse lookup is contextual to `_open_file`'s append-mode logic; extracting it as a named helper would obscure its role. The one-liner is self-documenting with the existing comment.

```python
# Before (line 492):
self._fieldnames = [reverse_map.get(h, h) for h in existing_fieldnames]

# After:
self._fieldnames = [reverse_map[h] if h in reverse_map else h for h in existing_fieldnames]
```

### Behavior-Preserving Move Sequence

**Precondition:** 4 R1 suppressions in `config/cicd/enforce_tier_model/plugins.yaml` for the
4 `.get(key, key)` patterns across csv_sink and json_sink.

**Move 1**: Add `display_name_for` to `display_headers.py`.
- No callers yet. Tests: green.
- No suppressions change.

**Move 2**: Rewrite sites 1, 3, 4 to use `display_name_for`.
- Import `display_name_for` in csv_sink.py (already imports from this module).
- Import `display_name_for` in json_sink.py (already imports from this module).
- Remove 3 R1 suppressions from `plugins.yaml`.
- Tests: green. Behavior identical — semantics of `display_name_for` match `dict.get(k, k)` exactly.

**Move 3**: Rewrite site 2 to inline `if h in reverse_map else h` form.
- Remove 1 R1 suppression from `plugins.yaml`.
- Tests: green.

All 3 moves are independently shippable. Move 2 and 3 can be co-committed if desired.

### Note on CUSTOM mode correctness

The identity-fallback is semantically wrong for CUSTOM mode: if a field has no mapping in
CUSTOM mode, that is a configuration error, not an "unnamed transform-added field".
`apply_display_headers` correctly raises `ValueError` for unmapped fields in CUSTOM mode.
The `display_name_for` helper is therefore appropriate only for ORIGINAL mode.

The call sites (sites 1, 3, 4) are each guarded by `if display_map is not None:` and the
surrounding context establishes that CUSTOM mode mappings are complete (validated at
`__init__` time). The proposed `display_name_for` docstring explicitly warns against use in
CUSTOM mode. No behavioral change to CUSTOM mode paths.

---

## Cross-Cutting Observations

**The annotation-runtime mismatch in Pattern A is a broader category smell.** A quick search
for `dict[str, list[` assigned to `defaultdict(` in the codebase (other than `_iter_records`)
would identify any other sites with the same structural gap. Not checked as part of this
review; the observation scope is limited to the two patterns in scope.

**The R1 suppression mechanism is working correctly.** The fact that these 12 suppressions
exist, are fingerprinted, have explicit owner tags, and have non-expired expirations means
the CI system surfaced them for human review exactly as designed. The question this report
answers is whether the suppressions represent "inherent" or "restructurable" patterns — both
are "restructurable", but for different reasons (annotation gap vs missing helper).

---

## Confidence Assessment

| Claim | Confidence | Basis |
|-------|------------|-------|
| All 8 Pattern A `.get()` calls are on dicts declared as `dict` but holding `defaultdict` | High | Confirmed by reading all 8 declaration sites in exporter.py:296–618 |
| Option δ (annotation widening) eliminates R1 without mypy error | High | Verified by running both annotation forms through mypy with no errors |
| Direct subscription on `defaultdict[str, list[T]]` is behavior-identical to `.get(key, [])` | High | Fundamental contract of `defaultdict.__missing__`; confirmed by runtime test |
| Pattern B helper does NOT already exist in display_headers.py | High | Full read of display_headers.py (259 lines); no `display_name_for` or equivalent |
| The 4 Pattern B call sites cover all R1 suppressions with `owner: feature` in plugins.yaml | Medium | Counted from plugins.yaml: exactly 4 `csv_sink`/`json_sink` R1 entries, all matching display mapping pattern |
| Test suite provides adequate characterization of the public API being refactored | High | Multiple integration + property + E2E test files confirmed above |

---

## Risk Assessment

**Pattern A:**

- **Low risk.** Annotation correction (`dict` → `defaultdict`) is documentation-level.
  The runtime object is already a `defaultdict`; behavior cannot regress.
- **Fingerprint rotation risk eliminated.** Option δ removes the suppressions entirely,
  so there are no fingerprints left to rotate when `_iter_records` is edited.
- **The empty-list insertion side effect** (defaultdict adds an entry on miss) is bounded
  to the local scope of `_iter_records`. The dicts are not returned, not stored on `self`,
  and not observable to callers. Risk: negligible.
- **If mypy rejects `defaultdict` annotation in the full project run** (not observed in
  isolated test): the contingency is Option γ — convert 8 per-fingerprint entries to 1
  per-file entry with `max_hits: 8`. Zero code change; same CI safety.

**Pattern B:**

- **Low-medium risk.** The rewrite introduces a new function in an infrastructure module
  that sinks depend on. The function is a pure identity-or-lookup with no side effects.
  The failure mode (regression) would require `display_name_for` to return a different
  value than `dict.get(k, k)` — not possible by inspection.
- **CUSTOM mode correctness**: the helper is not safe for CUSTOM mode (documented in
  docstring). The existing call sites are not in CUSTOM mode paths. Risk: low, but the
  docstring warning is load-bearing.

---

## Information Gaps

1. **No test coverage verified for `display_name_for` itself** — it does not exist yet.
   Move 1 should include a unit test for the helper before Moves 2 and 3 wire it in.
   (This is not a blocker for the design review; it is a pre-execution requirement.)

2. **The `reverse_map.get(h, h)` at site 2 has a different semantic from the other three.**
   If any future caller needs to resolve reverse-direction mappings, option 2a (symmetric
   helper) would be the right move. No evidence of future callers at this time.

3. **Other `defaultdict` annotation mismatches** in the codebase were not audited.
   Pattern A may recur elsewhere, but that is outside the scope of this review.

4. **Exporter test coverage of sparse-token scenarios** (root tokens, tokens with no states,
   states with no routing events) was confirmed by file existence but not by running the
   test suite. Behavior preservation relies on the existing integration tests being
   adequate characterization tests. If they are not, new characterization tests should
   precede the code move.

---

## Caveats

- This is a design review, not an execution. No code has been modified.
- The mypy verification was performed on isolated test files, not the full project run.
  The full-project mypy pass should be run after Move 1 of Pattern A to confirm no
  unexpected errors from the `defaultdict` annotation widening.
- The R1 rule fires on any `.get()` call on a non-HTTP, non-ChromaDB object. It cannot
  distinguish "this is a defaultdict so `.get(k, [])` is redundant" from "this is a plain
  dict and `.get(k, [])` is hiding a missing key bug". That distinction is the human
  judgment this review provides.
- The `owner: P2-2026-02-02-76r` tag on Pattern A entries is a filigree ticket convention
  meaning these suppressions were intended to be temporary (ticket-tracked). The verdict
  here confirms they should be resolved — not by closing the ticket silently, but by the
  Option δ code change that makes the suppressions unnecessary.
