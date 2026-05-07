# VerifiedTokenRef Type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the raw `(token_id, run_id)` dual-parameter pattern by introducing a `TokenRef` frozen dataclass that bundles the pair, preventing parameter-order bugs and enforcing coherence at construction time.

**Architecture:** Two-layer approach — `TokenRef` (plain bundle) lives in `contracts/audit.py`; `verify_token_ref()` factory lives in `core/landscape/data_flow_repository.py` where DB access exists. Dataclass fields change from `token_id: str, run_id: str` to `token_ref: TokenRef`. Loaders construct `TokenRef` from DB rows. Public API methods accept `TokenRef` (pre-verified by callers).

**Tech Stack:** Pure Python dataclass, no new dependencies.

**Filigree Issue:** `elspeth-f8fb4f784d` (P2 feature, `proposed` status)

---

## Design Decisions to Resolve

### Issue Description Correction

The issue lists 4 dataclasses: Token, TokenOutcome, Operation, TransformErrorRecord. **Operation does NOT have `token_id`** — it has `(run_id, node_id)`. The actual affected dataclasses are:

| Dataclass | Fields | File:Line |
|-----------|--------|-----------|
| **Token** | `token_id`, `run_id` | `contracts/audit.py:142` |
| **TokenOutcome** | `token_id`, `run_id` | `contracts/audit.py:642` |
| **TransformErrorRecord** | `token_id`, `run_id` | `contracts/audit.py:623` |

3 dataclasses, 3 loaders (TokenLoader, TokenOutcomeLoader, TransformErrorLoader).

### Decision 1: Field replacement vs API-boundary only

**Option A — Replace dataclass fields** (issue intent):
- `Token.token_id` + `Token.run_id` → `Token.token_ref: TokenRef`
- Loaders construct `TokenRef(token_id=row.token_id, run_id=row.run_id)`
- Every access site changes: `token.token_id` → `token.token_ref.token_id`
- DB `to_dict()` methods destructure: `{"token_id": self.token_ref.token_id, ...}`
- **Pro:** Maximum type safety — impossible to have mismatched pairs in the dataclass
- **Con:** High blast radius (~80+ sites), deeper nesting for field access, DB schema divergence

**Option B — API-boundary only** (pragmatic alternative):
- Dataclasses keep `token_id` and `run_id` as separate fields
- Public API methods (`fork_token`, `record_token_outcome`, etc.) accept `TokenRef` instead of `(token_id, run_id)`
- Internally destructure: `ref.token_id`, `ref.run_id`
- **Pro:** Small blast radius (~10 method signatures + call sites), no loader/test changes
- **Con:** Doesn't prevent mismatched pairs inside dataclass construction

**Recommendation:** Option B first — delivers the parameter-safety value with minimal disruption. Option A can follow as a separate issue if the pattern proves valuable.

### Decision 2: Verification at construction

**Option A — Factory-enforced verification:**
- `TokenRef` has `__init__` suppressed (raise in `__init__`, construction via `@classmethod`)
- `TokenRef.verified(token_id, run_id, validate_fn)` creates after validation
- `TokenRef._from_db(token_id, run_id)` for loaders (trusts Tier 1 data)
- **Pro:** Type-level guarantee that pairs are verified
- **Con:** Two construction paths, complexity in testing

**Option B — Convention-enforced verification:**
- `TokenRef` is a plain frozen dataclass (normal constructor)
- `verify_token_ref(ref, repository)` is called at write boundaries
- Tests construct directly without verification
- **Pro:** Simple, Pythonic, matches existing pattern (`_validate_token_run_ownership` at call sites)
- **Con:** No type-level enforcement that verification happened

**Recommendation:** Option B — matches ELSPETH's style. The existing `_validate_token_run_ownership` calls become `verify_token_ref` calls. The type bundles the pair; verification remains explicit at write boundaries.

### Decision 3: Naming

| Name | Semantics |
|------|-----------|
| `TokenRef` | Plain reference to a token-in-run (recommended — honest about what it is) |
| `VerifiedTokenRef` | Implies verification happened (misleading on read path) |
| `TokenRunPair` | Descriptive but verbose |

**Recommendation:** `TokenRef` — it's a reference type, not a verification proof.

---

## Prerequisites

**Filigree dependency:** This issue (`elspeth-f8fb4f784d`) should depend on the guard symmetry scanner issue (`elspeth-5f37dcce91`). The scanner is purely additive and should ship first. Add the dependency via `filigree add-dep elspeth-f8fb4f784d elspeth-5f37dcce91`.

## Acceptance Criteria (for `proposed` → `approved`)

1. `TokenRef` frozen dataclass exists in `contracts/audit.py` with `token_id: str` and `run_id: str`
2. All public API methods in `data_flow_repository.py` that take both `(token_id, run_id)` accept `TokenRef` instead
3. `_validate_token_run_ownership` operates on `TokenRef`
4. `_validate_token_run_ownership` has dedicated unit tests (valid ref, mismatched ref, nonexistent token)
5. All existing tests pass with updated call sites
6. No new test gaps introduced (coverage does not decrease)
7. CI enforcers pass (tier model, freeze guards) — guard symmetry scanner is a separate prerequisite
8. Follow-up issues filed for `coalesce_tokens` TokenRef adoption and Option A (field replacement)

---

## File Map (Option B — API-boundary)

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `src/elspeth/contracts/audit.py` | Add `TokenRef` frozen dataclass |
| Create | `tests/unit/contracts/test_token_ref.py` | Unit tests for `TokenRef` |
| Modify | `src/elspeth/core/landscape/data_flow_repository.py` | Update 5 public methods + validation |
| Modify | `src/elspeth/core/landscape/execution_repository.py` | Update if it calls updated methods (has token_id+run_id at lines 130,132) |
| Modify | Engine callers: `engine/processor.py`, `engine/tokens.py`, `engine/executors/gate.py`, `engine/executors/sink.py`, `engine/executors/transform.py`, `engine/coalesce_executor.py` | Update call sites |
| Modify | Tests that call the updated public API methods | Update call sites |

If Option A (field replacement) is chosen instead, add:

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `src/elspeth/contracts/audit.py` | Change fields in Token, TokenOutcome, TransformErrorRecord |
| Modify | `src/elspeth/core/landscape/model_loaders.py` | Update 3 loaders |
| Modify | `src/elspeth/core/landscape/data_flow_repository.py` | Update construction sites |
| Modify | `src/elspeth/core/landscape/execution_repository.py` | Update if it accesses token_id/run_id |
| Modify | ~12 test files (~48 fixtures) | Update all dataclass construction |

---

## Implementation Tasks (Option B — API-boundary)

### Task 1: TokenRef Type

**Files:**
- Modify: `src/elspeth/contracts/audit.py`
- Create: `tests/unit/contracts/test_token_ref.py`

- [ ] **Step 1: Write failing tests for TokenRef**

```python
# tests/unit/contracts/test_token_ref.py
"""Unit tests for TokenRef — bundled token_id + run_id reference."""

from __future__ import annotations

import pytest

from elspeth.contracts.audit import TokenRef


class TestTokenRef:
    """Tests for TokenRef frozen dataclass."""

    def test_construction(self) -> None:
        ref = TokenRef(token_id="tok-1", run_id="run-1")
        assert ref.token_id == "tok-1"
        assert ref.run_id == "run-1"

    def test_frozen(self) -> None:
        ref = TokenRef(token_id="tok-1", run_id="run-1")
        with pytest.raises(AttributeError):
            ref.token_id = "tok-2"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = TokenRef(token_id="tok-1", run_id="run-1")
        b = TokenRef(token_id="tok-1", run_id="run-1")
        assert a == b

    def test_inequality_token(self) -> None:
        a = TokenRef(token_id="tok-1", run_id="run-1")
        b = TokenRef(token_id="tok-2", run_id="run-1")
        assert a != b

    def test_inequality_run(self) -> None:
        a = TokenRef(token_id="tok-1", run_id="run-1")
        b = TokenRef(token_id="tok-1", run_id="run-2")
        assert a != b

    def test_hashable(self) -> None:
        ref = TokenRef(token_id="tok-1", run_id="run-1")
        s = {ref}
        assert ref in s

    def test_repr(self) -> None:
        ref = TokenRef(token_id="tok-1", run_id="run-1")
        r = repr(ref)
        assert "tok-1" in r
        assert "run-1" in r
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_token_ref.py -v`
Expected: FAIL — `ImportError: cannot import name 'TokenRef'`

- [ ] **Step 3: Add TokenRef to contracts/audit.py**

Add near the top of `contracts/audit.py` (after the imports, before `Run`):

```python
@dataclass(frozen=True, slots=True)
class TokenRef:
    """Bundled reference to a token within a specific run.

    A token_id is meaningless without its run_id — they always travel
    together semantically. This type enforces that coupling at the
    type level, preventing parameter-order bugs and mismatched pairs.

    Construction:
    - Write path: Create via verify_token_ref() in data_flow_repository
      (validates coherence against audit DB before returning)
    - Read path / tests: Construct directly (Tier 1 data is trusted)
    """

    token_id: str
    run_id: str
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_token_ref.py -v`
Expected: ALL PASSED (7 tests)

- [ ] **Step 5: Verify existing tests still pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ -v --timeout=60`
Expected: ALL PASSED (no regressions — TokenRef is additive)

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/audit.py tests/unit/contracts/test_token_ref.py
git commit -m "feat(contracts): add TokenRef — bundled token_id + run_id type

Frozen dataclass that enforces the semantic coupling between token_id
and run_id. Preparation for replacing dual-parameter API methods."
```

---

### Task 2: Update Validation to Use TokenRef (with dedicated tests)

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py`
- Modify: `tests/unit/contracts/test_token_ref.py` (add validation tests)

- [ ] **Step 1: Write dedicated tests for _validate_token_run_ownership with TokenRef**

Add to `tests/unit/contracts/test_token_ref.py`. These tests require a real (or mocked) repository — find the existing test pattern for `DataFlowRepository` in `tests/unit/core/` and follow it:

```python
class TestValidateTokenRunOwnership:
    """Tests for _validate_token_run_ownership accepting TokenRef.

    These test the validation at the point where TokenRef is first verified
    against the audit database. They ensure the cross-run contamination
    check works correctly with the bundled type.
    """

    def test_valid_ref_passes(self, data_flow_repo_with_token):
        """A TokenRef where token belongs to the specified run should pass."""
        repo, token, run_id = data_flow_repo_with_token
        ref = TokenRef(token_id=token.token_id, run_id=run_id)
        # Should not raise
        repo._validate_token_run_ownership(ref)

    def test_mismatched_run_raises_audit_integrity_error(self, data_flow_repo_with_token):
        """A TokenRef with wrong run_id should raise AuditIntegrityError."""
        repo, token, _run_id = data_flow_repo_with_token
        ref = TokenRef(token_id=token.token_id, run_id="wrong-run-id")
        with pytest.raises(AuditIntegrityError, match="Cross-run contamination"):
            repo._validate_token_run_ownership(ref)

    def test_nonexistent_token_raises(self, data_flow_repo):
        """A TokenRef with a token_id not in the DB should raise."""
        ref = TokenRef(token_id="nonexistent-token", run_id="any-run")
        with pytest.raises(AuditIntegrityError):
            repo._validate_token_run_ownership(ref)
```

**Note:** The exact fixture names (`data_flow_repo_with_token`, etc.) must match or adapt the existing test infrastructure. Grep for `_validate_token_run_ownership` in tests/ to find existing test patterns and fixtures.

- [ ] **Step 2: Update _validate_token_run_ownership to accept TokenRef**

In `data_flow_repository.py`, add the import and update the validation method:

```python
from elspeth.contracts.audit import TokenRef

def _validate_token_run_ownership(self, ref: TokenRef) -> None:
    """Validate that a token belongs to the specified run.

    Per Tier 1 trust model: cross-run contamination of audit records is
    a catastrophic integrity violation.

    Args:
        ref: TokenRef to validate — token_id must belong to run_id

    Raises:
        AuditIntegrityError: If token does not belong to the specified run
    """
    _row_id, actual_run_id = self._resolve_token_ownership(ref.token_id)
    if actual_run_id != ref.run_id:
        raise AuditIntegrityError(
            f"Cross-run contamination prevented: token {ref.token_id!r} belongs to "
            f"run {actual_run_id!r}, not {ref.run_id!r}"
        )
```

- [ ] **Step 3: Update the 4 call sites of _validate_token_run_ownership**

Each call site constructs a `TokenRef` from the existing parameters. Update:

1. `fork_token` (~line 481):
```python
# Before
self._validate_token_run_ownership(parent_token_id, run_id)
# After
self._validate_token_run_ownership(TokenRef(token_id=parent_token_id, run_id=run_id))
```

2. `expand_token` (~line 695):
```python
self._validate_token_run_ownership(TokenRef(token_id=parent_token_id, run_id=run_id))
```

3. `record_token_outcome` (~line 830):
```python
self._validate_token_run_ownership(TokenRef(token_id=token_id, run_id=run_id))
```

4. `record_transform_error` (~line 1393):
```python
self._validate_token_run_ownership(TokenRef(token_id=token_id, run_id=run_id))
```

- [ ] **Step 4: Run tests including the new validation tests**

Run: `.venv/bin/python -m pytest tests/ -x --timeout=120`
Expected: ALL PASSED

- [ ] **Step 5: Run mypy to catch type mismatches early**

Run: `.venv/bin/python -m mypy src/elspeth/core/landscape/data_flow_repository.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/core/landscape/data_flow_repository.py tests/unit/contracts/test_token_ref.py
git commit -m "refactor: _validate_token_run_ownership accepts TokenRef

Internal validation method now takes bundled TokenRef instead of separate
token_id + run_id. Call sites construct TokenRef from existing params.
Includes dedicated tests for valid, mismatched, and nonexistent refs.
No public API change yet — preparation for method signature updates."
```

---

### Task 3: Update Public API Methods

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py`
- Modify: Callers of updated methods (engine code)

- [ ] **Step 1: Identify all public methods to update**

These methods currently take both `token_id` and `run_id` as separate parameters:

| Method | Current Signature | Updated |
|--------|-------------------|---------|
| `record_token_outcome` | `(run_id, token_id, outcome, ...)` | `(ref: TokenRef, outcome, ...)` |
| `record_transform_error` | `(run_id, token_id, transform_id, ...)` | `(ref: TokenRef, transform_id, ...)` |
| `fork_token` | `(parent_token_id, row_id, branches, run_id, ...)` | `(parent_ref: TokenRef, row_id, branches, ...)` |
| `expand_token` | `(parent_token_id, row_id, children, run_id, ...)` | `(parent_ref: TokenRef, row_id, children, ...)` |

Note: `coalesce_tokens` takes `parent_token_ids` (multiple), not a single pair — handle separately.

- [ ] **Step 2: Update method signatures one at a time**

For each method:
1. Change the signature to accept `TokenRef`
2. Update internal references from `token_id`/`run_id` to `ref.token_id`/`ref.run_id`
3. Find callers (grep for method name) and update call sites
4. Run tests after each method change

**Example — `record_token_outcome`:**

```python
# Before:
def record_token_outcome(self, run_id: str, token_id: str, outcome: RowOutcome, ...) -> str:
    self._validate_token_run_ownership(token_id, run_id)
    ...

# After:
def record_token_outcome(self, ref: TokenRef, outcome: RowOutcome, ...) -> str:
    self._validate_token_run_ownership(ref)
    ...
    # Internal uses: ref.token_id, ref.run_id
```

- [ ] **Step 3: Update engine callers**

Grep for each method name to find ALL callers. Known engine files with call sites (verify with grep):
- `src/elspeth/engine/processor.py`
- `src/elspeth/engine/tokens.py`
- `src/elspeth/engine/executors/gate.py` (fork_token)
- `src/elspeth/engine/executors/sink.py` (record_token_outcome)
- `src/elspeth/engine/executors/transform.py` (record_transform_error)
- `src/elspeth/engine/coalesce_executor.py` (record_token_outcome)
- `src/elspeth/core/landscape/execution_repository.py` (check if it calls updated methods)

```bash
# Find ALL callers — do not skip any
grep -rn "record_token_outcome\|record_transform_error\|fork_token\|expand_token" src/elspeth/ --include="*.py" | grep -v "def \|#\|test"
```

Update each caller to construct `TokenRef`:

```python
# Before (in engine code):
repository.record_token_outcome(run_id=self.run_id, token_id=token.token_id, ...)

# After:
from elspeth.contracts.audit import TokenRef
repository.record_token_outcome(ref=TokenRef(token_id=token.token_id, run_id=self.run_id), ...)
```

- [ ] **Step 4: Run mypy after EACH method's callers are updated**

Run mypy after each method change, not just at the end. This catches missed call sites immediately rather than producing a pile of errors at the end:

```bash
# After updating record_token_outcome callers:
.venv/bin/python -m mypy src/elspeth/core/landscape/data_flow_repository.py src/elspeth/engine/
# After updating record_transform_error callers:
.venv/bin/python -m mypy src/elspeth/core/landscape/data_flow_repository.py src/elspeth/engine/
# After updating fork_token callers:
.venv/bin/python -m mypy src/elspeth/core/landscape/data_flow_repository.py src/elspeth/engine/
# After updating expand_token callers:
.venv/bin/python -m mypy src/elspeth/core/landscape/data_flow_repository.py src/elspeth/engine/
```

Expected: No errors after each batch. mypy is the primary safety net for missed call sites.

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x --timeout=120`
Expected: ALL PASSED

- [ ] **Step 6: Run CI enforcers**

```bash
python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
```

Expected: Both pass (TokenRef is a frozen dataclass with scalar-only fields — no freeze guard needed, no tier model violations)

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/core/landscape/data_flow_repository.py src/elspeth/engine/ src/elspeth/core/landscape/execution_repository.py
git commit -m "refactor: public API methods accept TokenRef instead of (token_id, run_id)

record_token_outcome, record_transform_error, fork_token, expand_token
now take TokenRef. Callers construct the ref, validation happens once
internally. Prevents parameter-order bugs at the API boundary."
```

---

### Task 4: Update Test Fixtures

**Files:**
- Modify: Test files that call the updated public API methods

- [ ] **Step 1: Find all test call sites**

```bash
grep -rn "record_token_outcome\|record_transform_error\|fork_token\|expand_token" tests/ | grep -v ".pyc"
```

- [ ] **Step 2: Update each test file**

For each call site, import `TokenRef` and update the call:

```python
from elspeth.contracts.audit import TokenRef

# Before:
repo.record_token_outcome(run_id="run-1", token_id="tok-1", outcome=RowOutcome.COMPLETED, ...)

# After:
repo.record_token_outcome(ref=TokenRef(token_id="tok-1", run_id="run-1"), outcome=RowOutcome.COMPLETED, ...)
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x --timeout=120`
Expected: ALL PASSED

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: update fixtures for TokenRef API — all call sites use bundled refs"
```

---

### Task 5: Finalize, File Follow-Ups, and Close

- [ ] **Step 1: Run all CI checks**

```bash
.venv/bin/python -m pytest tests/ --timeout=120
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
```

- [ ] **Step 2: Update tier model allowlist if needed**

If any new tier model findings appear (unlikely — TokenRef is a plain type), add allowlist entries with reasons.

- [ ] **Step 3: File follow-up issue — coalesce_tokens TokenRef adoption**

`coalesce_tokens` takes `parent_token_ids: list[str]` (multiple tokens). It was excluded from this refactor because the multi-token case needs a different approach (e.g., `list[TokenRef]`). File a follow-up issue:

```bash
filigree create "TokenRef adoption for coalesce_tokens — multi-token parameter pattern" \
  --type=task --priority=3
filigree add-comment <new-id> "coalesce_tokens takes parent_token_ids: list[str] without run context. \
Should accept list[TokenRef] to prevent cross-run contamination in multi-parent joins. \
Deferred from elspeth-f8fb4f784d because the multi-token case needs different design."
```

- [ ] **Step 4: File follow-up issue — Option A (field replacement)**

File the deeper refactor as a tracked issue, not just a plan note:

```bash
filigree create "TokenRef field replacement — embed in Token, TokenOutcome, TransformErrorRecord dataclasses" \
  --type=feature --priority=3
filigree add-comment <new-id> "Option A from elspeth-f8fb4f784d: replace separate token_id+run_id fields \
with token_ref: TokenRef in 3 audit dataclasses + 3 loaders + ~48 test fixtures (~80+ sites). \
Option B (API-boundary) shipped first — this is the deeper type-safety follow-up."
```

- [ ] **Step 5: Close the issue**

```
filigree close elspeth-f8fb4f784d --reason="TokenRef type implemented (Option B), API methods updated, all tests pass. Follow-up issues filed for coalesce_tokens and Option A field replacement."
```

---

## Future Work (Option A — Field Replacement)

If the API-boundary approach proves valuable and the team wants deeper type safety, the follow-up issue (filed in Step 4 above) covers:

1. Replace `token_id: str, run_id: str` fields in Token, TokenOutcome, TransformErrorRecord with `token_ref: TokenRef`
2. Update 3 loaders to construct `TokenRef` from DB rows
3. Update all ~48 test fixtures
4. Add `token_id` and `run_id` convenience properties to the dataclasses for backward-compatible access
5. Update DB `to_dict()` methods to destructure

This is a larger refactor (~80+ sites) and should be a separate issue with its own plan.
