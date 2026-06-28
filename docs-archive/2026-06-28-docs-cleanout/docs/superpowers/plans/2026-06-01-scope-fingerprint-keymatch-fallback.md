# Scope-Fingerprint Key-Match Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a tier-model finding's exact allowlist key misses because a module-level statement (an `@trust_boundary` import) shifted its module-rooted `ast_path`, fall back to matching the judge-gated v2 entry whose enclosing scope and within-scope position are unchanged — so signed entries no longer go stale or break on keyless decorator migrations.

**Architecture:** A pure core helper (`find_scope_fallback_entry`) matches a finding against judge-gated **v2** entries by `(file:rule:symbol, scope_fingerprint, within-scope ast_path suffix)`, returning the unique scope-stable entry or `None` (zero or ≥2 candidates → fail closed). The discriminator — the scope-relative `ast_path` suffix `path_stack[K:]`, where `K` is the depth to the enclosing scope — is shift-invariant and recoverable from existing entries with **no new persisted field**. The helper is wired only into tier_model's `_match_finding` after its exact-key loop misses. Match-only: stored key and HMAC signature are never touched, so load-time verification still passes.

**Tech Stack:** Python 3.13, `ast`, `pytest`. Code under `elspeth-lints/src/elspeth_lints/`; tests under `tests/unit/elspeth_lints/`.

**Spec:** `notes/scope-fingerprint-keymatch-fallback-scoping.md` · **Ticket:** elspeth-17322022a7

---

## Background the implementer must hold

- **Key shape:** `file_path:rule_id:symbol:fp=<hash>` where `fp = sha256(rule_id | ast_path | node_dump)[:16]`, `ast_path = "/".join(path_stack)` is module-rooted (`tier_model/rule.py:572-575`). Adding `Module.body[0]` shifts `body[N]→body[N+1]` for everything below → `fp` rotates → exact key misses.
- **Why scope_fingerprint alone is insufficient:** `compute_scope_fingerprint` hashes the **entire** docstring-stripped enclosing-scope body (`tier_model/scope_fingerprint.py:54-77`), so two findings in one function share an identical `scope_fingerprint`. Measured: 39 `(file:rule:symbol, scope_fp)` groups carry >1 entry (up to ×9). A within-scope positional discriminator is **mandatory**.
- **The discriminator — depth math:** `path_stack` and `node_stack` are index-aligned. `_visit_ast_child`/`_visit_ast_list_item` (`rule.py:1122-1136`) push the edge label onto `path_stack` *then* `visit()` pushes the child onto `node_stack`, so `path_stack[i]` is the edge `node_stack[i] → node_stack[i+1]` and `len(path_stack) == len(node_stack) - 1`. For enclosing scope `S = node_stack[K]`, the path components **above** `S` are `path_stack[0:K]` (these shift on a module-body edit) and the **within-scope suffix** is `path_stack[K:]` (invariant). So `K = node_stack.index(S)` and `suffix = ast_path.split("/")[K:]`.
- **Corpus state (measured 2026-06-01):** 242/242 judge entries are v2; the lone module-level judge entry is `__init__.py:R6:_module_` (not in any migration path). **Module-level findings (`K=0`) are intentionally NOT rescued** — their enclosing scope is the module, which genuinely changes when an import is added (its `scope_fingerprint` flips), so re-justify is the correct fail-closed outcome. This is a documented limitation affecting 1/242 entries.
- **Match-only invariant:** the fallback never rewrites `entry.key`, `entry.ast_path`, or the signature. The stored entry keeps its pre-shift key, which still self-verifies at load (`_verify_judge_metadata_signature_at_load` recomputes over `entry.key`). The fallback only lets a *new* finding match an *old* key. Stored `ast_path` thus lags the live node (audit-precision decay) — documented, not auto-refreshed (IRAP-grade tooling auditability).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `elspeth-lints/src/elspeth_lints/core/allowlist.py` | The shared, pure fallback matcher. | **Modify**: add `find_scope_fallback_entry()` + `_key_without_fp()` near the other key helpers (after `_file_path_from_canonical_key`, ~line 543). |
| `elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py` | Finding shape + stamping + tier_model matcher. | **Modify**: add `Finding.scope_depth` field (~line 113); add `_scope_depth_for_current_node()` (after `_scope_fingerprint_for_current_node`, ~line 556); stamp `scope_depth` at the 3 `Finding(...)` construction sites (lines ~665, ~692, ~715); call the fallback in `_match_finding` (~line 2258); import the helper (~line 41). |
| `tests/unit/elspeth_lints/test_allowlist_scope_fallback.py` | Pure-helper unit tests. | **Create** |
| `tests/unit/elspeth_lints/test_finding_scope_depth.py` | End-to-end: finding carries a shift-invariant suffix. | **Create** |
| `tests/unit/elspeth_lints/test_tier_model_scope_fallback_match.py` | End-to-end: import insertion no longer stales a v2 entry; no cross-bind. | **Create** |

**Not touched (named, deliberate):** `core/allowlist.py::Allowlist.match` and `trust_boundary/shared.py` — trust_boundary findings carry no `scope_fingerprint` and its entries are v1, so wiring core `match` now would be dead code (YAGNI). The helper is pure and reusable: when trust_boundary goes v2, that ticket widens `FindingKey` and wires `Allowlist.match`. `rotate` is match-time-orthogonal and untouched. No allowlist YAML schema change — the discriminator is derived from the entry's existing `ast_path`.

---

## Task 1: `Finding.scope_depth` — stamp the shift-invariant scope boundary

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py` (Finding dataclass ~113; new method after `_scope_fingerprint_for_current_node` ~556; 3 construction sites ~665/692/715)
- Test: `tests/unit/elspeth_lints/test_finding_scope_depth.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/elspeth_lints/test_finding_scope_depth.py`:

```python
# test_finding_scope_depth.py
"""Every tier-model Finding carries scope_depth K such that the scope-relative
ast_path suffix (ast_path.split('/')[K:]) is invariant under a module-body shift.

These drive the real scanner end-to-end. The suffix invariance under adding a
module-level import is the whole point of the key-match fallback.
"""

from pathlib import Path

from elspeth_lints.rules.trust_tier.tier_model.rule import scan_directory


def _r1(findings):
    r1 = [f for f in findings if f.rule_id == "R1"]
    assert r1, "expected an R1 dict.get finding"
    return r1[0]


def test_scope_depth_excludes_the_module_body_index(tmp_path: Path) -> None:
    src = "def handler(payload):\n    return payload.get('missing')\n"
    (tmp_path / "mod.py").write_text(src)
    f = _r1(scan_directory(tmp_path))
    # handler is module body[0]; the .get is body[0]/value inside it.
    assert f.ast_path == "body[0]/body[0]/value"
    assert f.scope_depth == 1  # K skips the leading module-body index
    assert f.ast_path.split("/")[f.scope_depth :] == ["body[0]", "value"]


def test_scope_relative_suffix_invariant_under_module_import_insertion(tmp_path: Path) -> None:
    before = "def handler(payload):\n    return payload.get('missing')\n"
    after = "import os\n\ndef handler(payload):\n    return payload.get('missing')\n"

    (tmp_path / "a.py").write_text(before)
    fa = _r1(scan_directory(tmp_path))

    (tmp_path / "a.py").write_text(after)
    fb = _r1(scan_directory(tmp_path))

    # The module-rooted ast_path SHIFTS (body[0] -> body[1]) ...
    assert fa.ast_path != fb.ast_path
    # ... but the scope-relative suffix is INVARIANT.
    assert fa.ast_path.split("/")[fa.scope_depth :] == fb.ast_path.split("/")[fb.scope_depth :]


def test_module_level_finding_has_scope_depth_zero(tmp_path: Path) -> None:
    src = "import os\nVALUE = os.environ.get('X')\n"
    (tmp_path / "mod.py").write_text(src)
    f = _r1(scan_directory(tmp_path))
    assert f.scope_depth == 0  # no enclosing def/class; module IS the scope
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/elspeth_lints/test_finding_scope_depth.py -v`
Expected: FAIL — `AttributeError: 'Finding' object has no attribute 'scope_depth'`.

- [ ] **Step 3: Add the `scope_depth` field to `Finding`**

In `rule.py`, in the `Finding` dataclass, add the field after `scope_fingerprint` (currently line 114):

```python
    ast_path: str = ""
    scope_fingerprint: str = ""
    scope_depth: int = 0
```

Extend the field's docstring block (the class docstring already documents `ast_path` and `scope_fingerprint`) by appending a paragraph before the closing `"""`:

```python
    ``scope_depth`` is K, the count of ``ast_path`` components strictly above
    the enclosing scope (``node_stack.index(enclosing_scope)``). The scope-
    relative suffix ``ast_path.split("/")[K:]`` is invariant under module-body
    shifts (adding/removing a module-level statement moves only the leading
    index, which lives in the first K components). It is the within-scope
    discriminator the allowlist key-match fallback uses to tell two same-rule
    findings in one scope apart. Module-level findings have no def/class scope,
    so K=0 and the fallback does not apply to them.
```

- [ ] **Step 4: Add `_scope_depth_for_current_node`**

In `rule.py`, immediately after `_scope_fingerprint_for_current_node` (ends ~line 556), add:

```python
    def _scope_depth_for_current_node(self) -> int:
        """Return K = the count of ast_path components above the enclosing scope.

        ``path_stack`` and ``node_stack`` are index-aligned: ``path_stack[i]`` is
        the edge ``node_stack[i] -> node_stack[i+1]`` (the descent helpers push the
        edge label, then ``visit`` pushes the child). So for enclosing scope
        ``S = node_stack[K]`` the components strictly above ``S`` are
        ``path_stack[0:K]`` (these shift when a module-level statement is
        added/removed) and the within-scope suffix is ``path_stack[K:]`` (stable).

        Returns 0 when the current node has no enclosing def/class (module level):
        the whole path is "scope-relative" and the fallback does not apply.
        """
        scope = enclosing_scope_node(list(reversed(self.node_stack)))
        if scope is None:
            return 0
        for index, node in enumerate(self.node_stack):
            if node is scope:
                return index
        # enclosing_scope_node only ever returns a node drawn from node_stack.
        raise AssertionError("enclosing scope not found in node_stack")
```

- [ ] **Step 5: Stamp `scope_depth` at all three Finding construction sites**

In `_add_finding` (~line 665), add the field after `scope_fingerprint=...`:

```python
                ast_path="/".join(self.path_stack) or "<module-root>",
                scope_fingerprint=self._scope_fingerprint_for_current_node(),
                scope_depth=self._scope_depth_for_current_node(),
            )
```

Apply the identical addition in `_add_suppressed_boundary_observation` (~line 693) and `_add_boundary_diagnostic` (~line 715), each immediately after its `scope_fingerprint=...` line.

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/elspeth_lints/test_finding_scope_depth.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py \
        tests/unit/elspeth_lints/test_finding_scope_depth.py
git commit -m "feat(tier-model): stamp scope_depth on findings (shift-invariant scope-relative ast_path suffix)"
```

---

## Task 2: `find_scope_fallback_entry` — the pure scope-stable matcher

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/allowlist.py` (add helpers after `_file_path_from_canonical_key`, ~line 543)
- Test: `tests/unit/elspeth_lints/test_allowlist_scope_fallback.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/elspeth_lints/test_allowlist_scope_fallback.py`:

```python
# test_allowlist_scope_fallback.py
"""Pure-helper tests for the scope-stable allowlist key-match fallback.

The fallback rescues a judge-gated v2 entry whose module-rooted ast_path drifted
(a module-level statement shifted the leading index) but whose enclosing scope
and within-scope position are unchanged. It MUST fail closed on real edits,
depth changes, ambiguity, v1 entries, and un-stamped findings.
"""

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    JudgeVerdict,
    find_scope_fallback_entry,
)

SCOPE = "a" * 64


def _v2(*, key: str, ast_path: str, scope_fingerprint: str = SCOPE) -> AllowlistEntry:
    return AllowlistEntry(
        key=key,
        owner="t",
        reason="r",
        safety="s",
        expires=None,
        ast_path=ast_path,
        scope_fingerprint=scope_fingerprint,
        judge_signature_version=2,
        judge_verdict=JudgeVerdict.ACCEPTED,
    )


def _v1(*, key: str, ast_path: str) -> AllowlistEntry:
    return AllowlistEntry(
        key=key,
        owner="t",
        reason="r",
        safety="s",
        expires=None,
        ast_path=ast_path,
        file_fingerprint="b" * 64,
        judge_verdict=JudgeVerdict.ACCEPTED,
    )


def test_matches_scope_stable_drift() -> None:
    # Stored at body[0]/...; live finding drifted to body[1]/... after an import.
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry],
        canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint=SCOPE,
        ast_path="body[1]/body[0]/value",
        scope_depth=1,
    )
    assert got is entry


def test_two_same_rule_findings_in_one_scope_bind_to_their_own_entry() -> None:
    # Identical scope_fingerprint (whole-body hash), distinct within-scope suffix.
    a = _v2(key="m.py:R6:C:f:fp=a", ast_path="body[0]/body[0]/value")
    b = _v2(key="m.py:R6:C:f:fp=b", ast_path="body[0]/body[1]/value")
    entries = [a, b]
    got_a = find_scope_fallback_entry(
        entries, canonical_key="m.py:R6:C:f:fp=x",
        scope_fingerprint=SCOPE, ast_path="body[1]/body[0]/value", scope_depth=1,
    )
    got_b = find_scope_fallback_entry(
        entries, canonical_key="m.py:R6:C:f:fp=y",
        scope_fingerprint=SCOPE, ast_path="body[1]/body[1]/value", scope_depth=1,
    )
    assert got_a is a
    assert got_b is b


def test_scope_body_edit_fails_closed() -> None:
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry], canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint="c" * 64,  # scope body changed
        ast_path="body[1]/body[0]/value", scope_depth=1,
    )
    assert got is None


def test_depth_change_fails_closed() -> None:
    # Wrapping the function in a class changes path depth.
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry], canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint=SCOPE,
        ast_path="body[0]/body[0]/body[0]/value", scope_depth=2,
    )
    assert got is None


def test_within_scope_transplant_fails_closed() -> None:
    # A different within-scope position (different suffix) must NOT match —
    # this is the fallback-path replacement for the ast_path transplant defence.
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry], canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint=SCOPE,
        ast_path="body[1]/body[7]/value", scope_depth=1,  # suffix body[7] != body[0]
    )
    assert got is None


def test_ambiguity_fails_closed() -> None:
    # Two stored entries satisfy the predicate for one live finding -> no bind.
    a = _v2(key="m.py:R6:C:f:fp=a", ast_path="body[0]/body[0]/value")
    b = _v2(key="m.py:R6:C:f:fp=b", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [a, b], canonical_key="m.py:R6:C:f:fp=x",
        scope_fingerprint=SCOPE, ast_path="body[1]/body[0]/value", scope_depth=1,
    )
    assert got is None


def test_v1_entry_skipped() -> None:
    entry = _v1(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry], canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint=SCOPE, ast_path="body[1]/body[0]/value", scope_depth=1,
    )
    assert got is None


def test_empty_finding_scope_fingerprint_skipped() -> None:
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry], canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint="", ast_path="body[1]/body[0]/value", scope_depth=1,
    )
    assert got is None


def test_different_symbol_skipped() -> None:
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry], canonical_key="m.py:R6:C:g:fp=new",  # symbol g != f
        scope_fingerprint=SCOPE, ast_path="body[1]/body[0]/value", scope_depth=1,
    )
    assert got is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/elspeth_lints/test_allowlist_scope_fallback.py -v`
Expected: FAIL — `ImportError: cannot import name 'find_scope_fallback_entry'`.

- [ ] **Step 3: Implement the helper**

In `core/allowlist.py`, after `_file_path_from_canonical_key` (ends ~line 542), add:

```python
def _key_without_fp(key: str) -> str:
    """Return the ``file_path:rule_id:symbol`` prefix of a canonical key.

    The canonical shape is ``<file>:<rule>:<symbol>:fp=<hash>``. The fingerprint
    is hex and the symbol is a ``:``-joined tuple of Python identifiers, neither
    of which can contain ``:fp=``, so a single split is unambiguous.
    """
    if ":fp=" not in key:
        raise ValueError(f"allowlist key is not in canonical form (missing ':fp=' suffix): {key!r}")
    return key.split(":fp=", 1)[0]


def find_scope_fallback_entry(
    entries: list[AllowlistEntry],
    *,
    canonical_key: str,
    scope_fingerprint: str,
    ast_path: str,
    scope_depth: int,
) -> AllowlistEntry | None:
    """Match a finding whose exact key missed against a scope-stable v2 entry.

    Rescues a judge-gated **v2** entry whose module-rooted ``ast_path`` drifted
    (a module-level statement shifted the leading index) but whose enclosing
    scope and within-scope position are unchanged — the same suppression,
    relocated by an unrelated module-body edit.

    An entry is a candidate iff ALL hold:

    1. same ``file_path:rule_id:symbol`` as the finding;
    2. it is judge-gated and v2 (carries ``scope_fingerprint``); v1 entries bind
       ``file_fingerprint`` and are skipped;
    3. its persisted ``scope_fingerprint`` equals the finding's (the enclosing
       scope body is byte-identical — a real body edit fails here);
    4. its ``ast_path`` has the same component count (a depth-changing relocation
       fails here);
    5. its within-scope suffix ``ast_path.split("/")[scope_depth:]`` equals the
       finding's (the within-scope position is identical — the fallback-path
       replacement for the exact ``ast_path`` transplant defence).

    Returns the unique candidate, or ``None`` when there are zero or **two or
    more** (ambiguity must never silently bind — fail closed). An empty
    ``scope_fingerprint`` on the finding (un-stamped / non-tier_model finding)
    yields ``None``: there is nothing to bind against.

    Match-only: this never rewrites the entry's key, ast_path, or signature. The
    matched entry keeps its pre-shift key (still self-consistent at load); only
    a *new* finding is allowed to match an *old* key.
    """
    if not scope_fingerprint:
        return None
    finding_prefix = _key_without_fp(canonical_key)
    live_components = ast_path.split("/")
    live_suffix = live_components[scope_depth:]
    candidates: list[AllowlistEntry] = []
    for entry in entries:
        if entry.judge_verdict is None or entry.scope_fingerprint is None or entry.ast_path is None:
            continue
        if _key_without_fp(entry.key) != finding_prefix:
            continue
        if entry.scope_fingerprint != scope_fingerprint:
            continue
        stored_components = entry.ast_path.split("/")
        if len(stored_components) != len(live_components):
            continue
        if stored_components[scope_depth:] != live_suffix:
            continue
        candidates.append(entry)
    if len(candidates) == 1:
        return candidates[0]
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/elspeth_lints/test_allowlist_scope_fallback.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/allowlist.py \
        tests/unit/elspeth_lints/test_allowlist_scope_fallback.py
git commit -m "feat(allowlist): scope-stable key-match fallback helper (fail-closed, match-only)"
```

---

## Task 3: Wire the fallback into `_match_finding` + end-to-end reproduction

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py` (import ~41; `_match_finding` ~2258)
- Test: `tests/unit/elspeth_lints/test_tier_model_scope_fallback_match.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/elspeth_lints/test_tier_model_scope_fallback_match.py`:

```python
# test_tier_model_scope_fallback_match.py
"""End-to-end: a module-level import insertion no longer stales a v2 entry.

Reproduces the ticket's headline case (generation.py::_csv_source_delimiter, 44
stale entries) in miniature: scan a file, build the matching v2 allowlist entry
from the real finding, insert a module-level import (shifting every downstream
ast_path), re-scan, and assert the drifted finding still matches its entry via
the scope-stable fallback (so it is suppressed, and the entry is not stale).
"""

from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, AllowlistEntry, JudgeVerdict
from elspeth_lints.rules.trust_tier.tier_model.rule import (
    _match_finding,
    scan_directory,
)

BEFORE = "def handler(payload):\n    return payload.get('missing')\n"
AFTER = "from x import trust_boundary\n\ndef handler(payload):\n    return payload.get('missing')\n"


def _r1(findings):
    r1 = [f for f in findings if f.rule_id == "R1"]
    assert r1
    return r1[0]


def _entry_from(finding) -> AllowlistEntry:
    return AllowlistEntry(
        key=finding.canonical_key,
        owner="t",
        reason="r",
        safety="s",
        expires=None,
        ast_path=finding.ast_path,
        scope_fingerprint=finding.scope_fingerprint,
        judge_signature_version=2,
        judge_verdict=JudgeVerdict.ACCEPTED,
    )


def test_import_insertion_does_not_stale_a_v2_entry(tmp_path: Path) -> None:
    src = tmp_path / "a.py"
    src.write_text(BEFORE)
    before = _r1(scan_directory(tmp_path))
    allowlist = Allowlist(entries=[_entry_from(before)])

    # Sanity: exact match before any drift.
    assert _match_finding(allowlist, before) is allowlist.entries[0]

    # Insert a module-level import -> every downstream ast_path shifts.
    src.write_text(AFTER)
    after = _r1(scan_directory(tmp_path))
    assert after.canonical_key != before.canonical_key  # exact key drifted

    matched = _match_finding(allowlist, after)
    assert matched is allowlist.entries[0]      # rescued by the fallback
    assert allowlist.entries[0].matched is True  # not reported stale


def test_real_body_edit_still_stales(tmp_path: Path) -> None:
    src = tmp_path / "a.py"
    src.write_text(BEFORE)
    before = _r1(scan_directory(tmp_path))
    allowlist = Allowlist(entries=[_entry_from(before)])

    # Change the scope body itself -> scope_fingerprint flips -> no fallback.
    src.write_text("def handler(payload):\n    x = 1\n    return payload.get('missing')\n")
    after = _r1(scan_directory(tmp_path))
    assert _match_finding(allowlist, after) is None
    assert allowlist.entries[0].matched is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/elspeth_lints/test_tier_model_scope_fallback_match.py -v`
Expected: FAIL — `test_import_insertion_does_not_stale_a_v2_entry` fails at `assert matched is allowlist.entries[0]` (currently returns `None`; the fallback is not wired).

- [ ] **Step 3: Import the helper**

In `rule.py`, extend the existing import block (currently lines 41-45):

```python
from elspeth_lints.core.allowlist import (
    FindingKey,
    find_scope_fallback_entry,
    load_allowlist,
    verify_entry_binding_against_finding,
)
```

- [ ] **Step 4: Call the fallback in `_match_finding`**

In `rule.py`, replace the final `return None` of `_match_finding` (currently line 2258) — the line reached after the exact-key `for entry in allowlist.entries` loop — with:

```python
    # Exact key missed. A judge-gated v2 entry whose module-rooted ast_path
    # drifted (an unrelated module-level statement shifted the leading index)
    # but whose enclosing scope + within-scope position are unchanged is the
    # same suppression, relocated. The fallback's predicate (scope_fingerprint
    # equality + within-scope ast_path suffix equality) IS its binding check and
    # is at least as strong as the exact-match transplant defence, so we do NOT
    # also call verify_entry_binding_against_finding (which asserts ast_path
    # equality and would crash by construction on the fallback path).
    fallback = find_scope_fallback_entry(
        allowlist.entries,
        canonical_key=finding.canonical_key,
        scope_fingerprint=finding.scope_fingerprint,
        ast_path=finding.ast_path,
        scope_depth=finding.scope_depth,
    )
    if fallback is not None:
        fallback.matched = True
        return fallback
    return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/elspeth_lints/test_tier_model_scope_fallback_match.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py \
        tests/unit/elspeth_lints/test_tier_model_scope_fallback_match.py
git commit -m "feat(tier-model): match scope-stable drifted v2 entries via key-match fallback (elspeth-17322022a7)"
```

---

## Task 4: Regression — full elspeth-lints suite + live tier-model gate

**Files:** none modified — this task verifies the change is non-regressive and the real gate is green.

- [ ] **Step 1: Run the full elspeth-lints unit suite**

Run: `.venv/bin/python -m pytest tests/unit/elspeth_lints/ -q`
Expected: PASS. (If `test_baseline_capture_is_self_consistent` is red, that is the known pre-existing `fingerprint_baseline.json` drift from 92bc9fb2f/49925bd79 — unrelated to this change; confirm it fails identically on `HEAD~4` before discounting it.)

- [ ] **Step 2: Run the existing binding tests explicitly (transplant defence intact)**

Run: `.venv/bin/python -m pytest tests/unit/elspeth_lints/test_allowlist_match_binding.py tests/unit/elspeth_lints/test_finding_scope_fingerprint.py tests/unit/elspeth_lints/test_reaudit.py -v`
Expected: PASS — the exact-match C8-3 path and reaudit are unaffected (acceptance criteria 5 & 6).

- [ ] **Step 3: Run the live tier-model gate over the real source tree**

Run: `env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`
Expected: exit 0, no new stale entries, no new violations. (This is the production gate; a green run proves the fallback did not change matching of any currently-exact entry — trust_boundary and unsigned entries included.)

- [ ] **Step 4: Lint + type-check the changed files**

Run: `.venv/bin/python -m ruff check elspeth-lints/src/elspeth_lints/core/allowlist.py elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py && PYTHONPATH=elspeth-lints/src .venv/bin/python -m mypy elspeth-lints/src/elspeth_lints/core/allowlist.py elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py`
Expected: clean.

- [ ] **Step 5: Acceptance-criteria checklist (map tests → ticket criteria)**

Confirm each ticket acceptance criterion has a green test:
1. Reproduction (no-stale on import insertion) → `test_import_insertion_does_not_stale_a_v2_entry`
2. No collapse of two same-scope findings → `test_two_same_rule_findings_in_one_scope_bind_to_their_own_entry`
3. Fail closed on body edit / depth change → `test_real_body_edit_still_stales`, `test_scope_body_edit_fails_closed`, `test_depth_change_fails_closed`
4. Fail closed on ambiguity → `test_ambiguity_fails_closed`
5. Signature intact at load (match-only) → covered by unchanged `test_allowlist_judge_metadata_integrity.py` in Step 1 + design (no re-key)
6. Transplant still caught → `test_within_scope_transplant_fails_closed` + unchanged `test_allowlist_match_binding.py`
7. trust_boundary byte-identical → not wired; Step 3 live gate green proves it
8. rotate untouched → no change to rotate code (grep confirms)

- [ ] **Step 6: Commit (if any lint/type fixups were needed; otherwise skip)**

```bash
git add -A && git commit -m "chore(tier-model): lint/type fixups for scope key-match fallback"
```

---

## Known limitation (documented, not a defect)

**Module-level signed entries (`scope_depth == 0`) are not rescued.** Their enclosing scope is the module, whose `scope_fingerprint` genuinely changes when an import is added, so the fallback correctly declines (fail closed → re-justify). The corpus has exactly one such entry (`__init__.py:R6:_module_`), not in any `@trust_boundary` migration path. If a future migration needs to add an import to a file carrying a module-level signed entry, that entry needs an operator re-justify — unchanged from today.

**Stored `ast_path` lags after a fallback match.** The matched entry keeps its pre-shift key/ast_path (match-only — required to preserve the HMAC). The audit-precision cost (the stored address no longer points at the live node) is IRAP-grade acceptable. A verdict-preserving re-key verb to refresh it is explicitly out of scope (file as a follow-up if the decay is judged worth closing).

---

## Self-Review

- **Spec coverage:** §4.1 discriminator → Task 1; §4.2 predicate → Task 2; §4.3 binding-on-fallback-path (skip strict verify) → Task 3 Step 4; §4.4 match-only → asserted by Task 4 Step 1 (`test_allowlist_judge_metadata_integrity`) + no re-key code; §5 touch points → all covered, with core `match`/trust_boundary deliberately deferred (documented); §6 all 8 acceptance criteria → Task 4 Step 5 map; §2a module-level limitation → "Known limitation".
- **Placeholder scan:** none — every code step shows complete code; every run step shows the command and expected result.
- **Type/name consistency:** `find_scope_fallback_entry` / `_key_without_fp` / `Finding.scope_depth` / `_scope_depth_for_current_node` used identically across Tasks 1-3. The helper's keyword args (`canonical_key`, `scope_fingerprint`, `ast_path`, `scope_depth`) match the call site in Task 3 Step 4. `enclosing_scope_node` is already imported in `rule.py` (line 55) — no new import needed for Task 1.
