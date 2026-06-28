# Judge Scope-Fingerprint Binding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the whole-file `file_fingerprint` binding on judge-gated tier-model allowlist entries with an enclosing-scope AST fingerprint, so editing code *near* (but not *at*) a signed suppression no longer forces an operator-only re-signing.

**Architecture:** A new `scope_fingerprint` binds each judge-gated entry to the AST-content of its innermost enclosing `FunctionDef`/`AsyncFunctionDef`/`ClassDef` (module-level fallback otherwise). It is computed **forward** by the scanner/visitor — never reverse-resolved from `ast_path` — stamped onto every `Finding`, carried into the HMAC-signed payload as **signature version 2**, and verified at **match time** (reusing the finding the matcher already produced). Version 1 (`file_fingerprint`, verified at load) runs in parallel until a keyed operator session migrates the 221 live entries and deletes the v1 path. Load-time keeps only a cheap *file-exists* check for v2.

**Tech Stack:** Python 3.13, `ast`, `hashlib`/`hmac`, dataclasses, pytest. All code lives under `elspeth-lints/src/elspeth_lints/`. Run tests from `/home/john/elspeth/elspeth-lints` with the repo venv: `../.venv/bin/python -m pytest`.

---

## Operator Boundary (read before starting)

This plan has a **hard line** between agent-buildable work and operator-only work, because the HMAC signing key (`ELSPETH_JUDGE_METADATA_HMAC_KEY`) is **operator-only and MUST NOT be in any agent's environment** (CLAUDE.md "CICD Judge Gate: HMAC Key Custody").

- **Tasks 1–11 (agent-buildable):** all v2 code lands *alongside* v1. Unit tests inject a throwaway test key via the existing `hmac_key=` parameter on `compute_judge_metadata_signature` — they never read the production key. Commits stay green because the 221 live entries in `config/cicd/enforce_tier_model/` remain v1 and load under the retained v1 path.
- **Tasks 12–13 (OPERATOR ONLY):** running the batch migration, re-justifying stragglers, and the final commit that deletes the v1 field/path. These require the key and are explicitly flagged. An agent may *prepare* and *propose* them but must not execute them.

> **Conscious deviation from spec §4.2.** The spec described a single operator session with "no intermediate dual-version state ever committed." This plan instead commits a **dual-version state** at the end of Task 11 (v2 code present, v1 path retained, all entries still v1). That is unavoidable for an agent-buildable increment: the agent cannot run the keyed migration, so the v1 path must survive in committed code until the operator session. This is a deliberate, flagged choice — not a silent slip.

> **Coordinated sibling feature (v2 payload is shared).** The Claude-Agent-SDK judge transport (`docs/superpowers/plans/2026-05-31-agent-sdk-judge-transport.md`) adds one more field — `judge_transport` — to **this plan's v2 signed payload**, and is sequenced to land **after this plan's Tasks 1–11 but before its operator Task 12**. When that plan has landed, the v2 payload includes `judge_transport`, `_validate_judge_metadata_atomic` requires it on v2 entries, and **`migrate-judge-scope` (Task 10) backfills `judge_transport="openrouter"`** as part of the single v1→v2 re-sign — so the operator migration (Task 12) produces complete v2 entries (`scope_fingerprint` + `judge_transport`) in one pass. If that sibling plan has *not* landed when you run this one, the v2 payload is scope-fingerprint-only and the transport feature reconciles to it later; the two must never produce two independent v2 payload schemas (transport-plan §5).

---

## Reconciliation Against Live Code (READ FIRST — corrects Tasks 1–11)

This plan was authored in a prior session; a reality pass on RC5.2 (2026-05-31)
found the **production-source anchors accurate** (every `allowlist.py` / `rule.py`
/ `cli.py` symbol and line number is right to ±a few lines) but the **test-layer
anchors systematically wrong**. Apply these corrections to every task below;
they override the per-task `Run:` / `Test:` / `git add` paths wherever they
conflict.

**R-1 — Test root (affects EVERY task).** There is **no `elspeth-lints/tests/`
directory**, and no `tests/core/` or `tests/rules/` tree. The elspeth-lints
tests live at **`tests/unit/elspeth_lints/`** rooted at the repo
(`/home/john/elspeth`). pytest `rootdir` is the repo root (`testpaths =
["tests"]`). Translate every command:
- `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/<X>`
  → `cd /home/john/elspeth && .venv/bin/python -m pytest tests/unit/elspeth_lints/<X>`
  (both spellings point at the same venv, `/home/john/elspeth/.venv`).
- `git add elspeth-lints/tests/<X>` → `git add tests/unit/elspeth_lints/<X>`.
- mypy/ruff/`check` commands (which target `src/`) are unaffected.

**R-2 — New tier-model test files go FLAT (Tasks 1, 2).** Do **not** create a
`tests/.../rules/trust_tier/tier_model/` subdir — the existing tier-model rule
tests (`test_trust_tier_model_rule.py`, `test_tier_model_decorator_suppression.py`,
`test_rotate_tier_model.py`) sit flat at `tests/unit/elspeth_lints/`. Place the
new tests there too, matching the sibling convention (no `__init__.py` needed at
that level):
- Task 1 → `tests/unit/elspeth_lints/test_scope_fingerprint.py`
- Task 2 → `tests/unit/elspeth_lints/test_finding_scope_fingerprint.py`
The Task 2 import `from elspeth_lints.rules.trust_tier.tier_model.rule import
scan_directory` is **correct** — `scan_directory(root, exclude_patterns=None)`
exists (rule.py:1431); the single-positional-arg call in the test works.

**R-3 — Real test-file homes (B2).** Re-point each task's `Test:` line and `-k`
command to the real file (all under `tests/unit/elspeth_lints/`):

| Task | Plan's (wrong) file | Real file to extend / create |
|------|---------------------|------------------------------|
| 3 (schema/loader) | `test_allowlist_schema.py` | extend `test_allowlist_loader_unification.py` |
| 4 (signing) | `test_allowlist_signing.py` | extend `test_allowlist_judge_metadata_integrity.py` (note: it hand-builds `f"hmac-sha256:v1:{digest}"` at ~:68 rather than calling the signer — add v2 fixtures consistently) |
| 5 (load binding) | `test_allowlist_load_binding.py` | extend `test_allowlist_judge_metadata_integrity.py` |
| 6 (match binding) | `test_allowlist_match_binding.py` | **create** `test_allowlist_match_binding.py` (flat) |
| 7 (justify) | `test_cli_justify.py` | extend `test_justify.py` |
| 8 (coverage) | `test_judge_coverage.py` | extend `test_judge_coverage.py` (exists) |
| 8 (sidecar) | `test_reaudit_sidecar.py` | **create** `test_reaudit_sidecar.py` — `_entry_to_dict`/`_entry_from_dict` have no current test coverage |
| 8 (rotate) | `test_rotate*.py` | extend `test_rotate_tier_model.py` |
| 9 (reaudit) | `test_reaudit*.py` | extend `test_reaudit.py` / `test_reaudit_multi_rule.py` |
| 10 (migrate) | `test_cli_migrate_judge_scope.py` | **create** `test_cli_migrate_judge_scope.py` (flat) |

**R-4 — Task 6 `shared.py` call site is `protocols.Finding`, which has NO
`scope_fingerprint` field (MAJOR).** `shared.py:36` imports `Finding` from
`core/protocols.py`, whose `Finding` (`:43`) defines `ast_path` (`:55`) but not
`scope_fingerprint`. The literal code block in Task 6 Step 4 (`scope_fingerprint=
finding.scope_fingerprint`) would `AttributeError` at runtime. Use the
**getattr-fallback form from that step's prose-note** instead —
`scope_fingerprint=getattr(finding, "scope_fingerprint", "")` with the comment
explaining trust_boundary findings are v1-only today and a future v2
trust_boundary entry would (correctly) fail-closed at match time. The
tier-model `rule.py:2201` site keeps the real `finding.scope_fingerprint`
(tier-model `Finding` gets the field in Task 2).

**R-5 — Task 7 justify-v2 coverage is tier-model-only (MAJOR).**
`_scan_single_file_findings_for_justify` (cli.py:1625) returns tier-model
`rule.Finding` (has `scope_fingerprint`) for `trust_tier.tier_model`, but
`protocols.Finding` (no `scope_fingerprint`) for `trust_boundary.*` rules.
`_finding_scope_fingerprint` raising on a missing/empty value is therefore
**correct fail-closed behaviour for tier-model**, but would raise on a
`justify` of a trust_boundary rule. Task 7's claim that the finding "always
carries scope_fingerprint — confirmed" holds only for tier-model. Keep the
raise (do not fabricate a value); state in the accessor's comment that v2
justify is tier-model-only until trust_boundary's scanner stamps the field.

**R-6 — Minor anchor fixes.**
- Task 8 rotate.py: the `_JUDGE_METADATA_SIGNATURE_PREFIX` *usages* are at
  rotate.py **:833 and :835** (definition at :88), not the `:96`/`:820` the task
  text implies. Grep `_JUDGE_METADATA_SIGNATURE_PREFIX` in rotate.py before
  editing.
- Task 8 sidecar: `reaudit_sidecar.py` already has `_optional_int` (:1011) and
  `_optional_str` (:1002). Use the existing `_optional_int` — do **not** add a
  new `_optional_int_field` helper.
- Task 4 Step 4: `_verify_judge_metadata_signature_at_load` has a missing-key
  early-return (`_can_skip_judge_metadata_hmac_recompute_for_missing_key()`,
  ~:660) *before* the recompute call (:662). **Preserve it** when rewriting the
  recompute; the plan's snippet shows only the recompute and must not drop the
  guard above it.

Everything else in the plan (production symbol names, signatures, invariant
numbers 4 and 8, the two `verify_entry_binding_against_finding` call sites,
the `compute_judge_metadata_signature` direct callers) was verified accurate.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `rules/trust_tier/tier_model/scope_fingerprint.py` | **New.** Pure function: given an enclosing-scope `ast.AST` node (or `None` for module-level), return its normative scope fingerprint. The single source of truth for the hash, shared by the visitor and any future scanner. | Create |
| `rules/trust_tier/tier_model/rule.py` | Tier-model scanner/visitor + `Finding` + matcher. Compute the enclosing-scope node from `node_stack`, stamp `Finding.scope_fingerprint`, thread it into the match-time verifier. | Modify |
| `core/allowlist.py` | `AllowlistEntry` schema, loader, atomic validator, signature compute/verify, load-time + match-time binding checks. v2 payload + version dispatch; load-time file-exists; match-time scope check. | Modify |
| `core/cli.py` | `justify` write path (`_build_yaml_entry_text`) writes v2; **new** `migrate-judge-scope` subcommand. | Modify |
| `core/judge_coverage.py` | Source-binding identity tuple for the coverage diff. Retarget `file_fingerprint` → `scope_fingerprint`. | Modify |
| `core/reaudit_sidecar.py` | Sidecar round-trip of an `AllowlistEntry`. Add the two new fields. | Modify |
| `core/reaudit.py` | Re-judge sweep. Restore pre-re-judge tamper detection for v2 by matching+verifying entries against the findings it already scans. | Modify |
| `rules/trust_tier/tier_model/rotate.py` | Rotation refusal of judge-gated entries + raw-entry source-binding. Retarget the `file_fingerprint` key read. | Modify |

The new hash lives in its own module (not buried in `rule.py`) specifically so the justify-write path, the match-verify path, and the migration command all import **one** definition — divergence between them is the single highest-severity failure mode (one byte of difference crashes every v2 entry at match time).

---

## Determinism Rules (normative — from spec §3.1, frozen here)

The scope fingerprint MUST be byte-reproducible at justify-write time and match-verify time. The rules:

1. **Scope node** = the innermost enclosing `ast.FunctionDef`, `ast.AsyncFunctionDef`, or `ast.ClassDef` of the suppressed node.
2. **Module-level fallback** = if there is no enclosing def, the scope is the whole `ast.Module`.
3. **Hash input** = `ast.dump(scope_node, include_attributes=False, annotate_fields=True)` over the scope node **with its leading docstring removed** (see rule 4). This naturally includes `name`, `args` (functions), `bases`/`keywords` (classes), `decorator_list`, and `body`.
4. **Docstring excluded:** if `scope_node.body` is non-empty and `body[0]` is an `ast.Expr` wrapping an `ast.Constant` whose value is a `str`, that element is dropped from the dumped body. A docstring edit must not force a security re-sign.
5. **No `ast_path` prefix.** Unlike `_fingerprint_node`, the scope hash omits the path prefix, so inserting a *sibling* def above does not change a suppression's scope hash. This is the relief mechanism.
6. **Renaming the enclosing def changes the hash** (the `name` is in the dump). Acceptable: a rename is a genuine identity change worth a re-justify.
7. **Output shape** = `sha256(...).hexdigest()` — full 64-char lowercase hex (distinct from `_fingerprint_node`'s 16-char truncation; this is a binding primitive, not a collision-tolerant short id).

---

## Task 1: The scope-fingerprint primitive

**Files:**
- Create: `elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/scope_fingerprint.py`
- Test: `elspeth-lints/tests/rules/trust_tier/tier_model/test_scope_fingerprint.py`

- [ ] **Step 1: Write the failing tests**

```python
# test_scope_fingerprint.py
import ast

import pytest

from elspeth_lints.rules.trust_tier.tier_model.scope_fingerprint import (
    compute_scope_fingerprint,
    enclosing_scope_node,
)


def _func(src: str) -> ast.AST:
    """Return the first FunctionDef/AsyncFunctionDef/ClassDef in src."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return node
    raise AssertionError("no scope node in source")


def test_hash_is_64_char_lowercase_hex() -> None:
    fp = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    assert len(fp) == 64
    assert fp == fp.lower()
    bytes.fromhex(fp)  # raises if not hex


def test_reformatting_and_comments_are_free() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def f(x):  # a comment\n        return x.get('a')\n"))
    assert a == b


def test_docstring_edit_is_free() -> None:
    a = compute_scope_fingerprint(_func('def f(x):\n    "old doc"\n    return x.get("a")\n'))
    b = compute_scope_fingerprint(_func('def f(x):\n    "completely different doc"\n    return x.get("a")\n'))
    assert a == b


def test_adding_a_docstring_is_free() -> None:
    a = compute_scope_fingerprint(_func('def f(x):\n    return x.get("a")\n'))
    b = compute_scope_fingerprint(_func('def f(x):\n    "now documented"\n    return x.get("a")\n'))
    assert a == b


def test_body_change_changes_hash() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def f(x):\n    return x.get('b')\n"))
    assert a != b


def test_parameter_change_changes_hash() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def f(x, y):\n    return x.get('a')\n"))
    assert a != b


def test_decorator_change_changes_hash() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("@retry\ndef f(x):\n    return x.get('a')\n"))
    assert a != b


def test_rename_changes_hash() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def g(x):\n    return x.get('a')\n"))
    assert a != b


def test_class_scope_first_string_stmt_is_treated_as_docstring() -> None:
    # The class docstring is dropped; a method body change is not.
    a = compute_scope_fingerprint(_func('class C:\n    "doc"\n    def m(self):\n        return 1\n'))
    b = compute_scope_fingerprint(_func('class C:\n    "other"\n    def m(self):\n        return 1\n'))
    assert a == b


def test_module_level_fallback_uses_whole_module() -> None:
    tree = ast.parse("import os\nX = os.environ.get('A')\n")
    fp_a = compute_scope_fingerprint(None, module=tree)
    fp_b = compute_scope_fingerprint(None, module=ast.parse("import os\nX = os.environ.get('B')\n"))
    assert len(fp_a) == 64
    assert fp_a != fp_b


def test_module_fallback_requires_module_arg() -> None:
    with pytest.raises(ValueError, match="module"):
        compute_scope_fingerprint(None)


def test_enclosing_scope_node_finds_innermost_def() -> None:
    src = "class C:\n    def m(self, x):\n        return x.get('a')\n"
    tree = ast.parse(src)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    # ancestors innermost-first: the Call's enclosing scope is method m, not class C.
    ancestors = _ancestors_of(tree, call)
    scope = enclosing_scope_node(ancestors)
    assert isinstance(scope, ast.FunctionDef)
    assert scope.name == "m"


def _ancestors_of(tree: ast.AST, target: ast.AST) -> list[ast.AST]:
    """Return [target, parent, ..., module] — innermost first."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    chain: list[ast.AST] = [target]
    cur = target
    while id(cur) in parents:
        cur = parents[id(cur)]
        chain.append(cur)
    return chain
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/rules/trust_tier/tier_model/test_scope_fingerprint.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named '...scope_fingerprint'`.

- [ ] **Step 3: Implement the module**

```python
# scope_fingerprint.py
"""Enclosing-scope AST fingerprint for judge-gated tier-model suppressions.

This is the v2 binding primitive (replaces the whole-file ``file_fingerprint``).
It binds a judge-gated allowlist entry to the AST content of the *innermost
enclosing scope* of the suppressed node, so editing an unrelated scope in the
same file no longer invalidates the entry's HMAC signature.

The hash MUST be byte-reproducible at justify-write time and match-verify time.
Both call sites import :func:`compute_scope_fingerprint` from here — there is
deliberately only one definition. See the design doc
``docs/superpowers/specs/2026-05-31-judge-scope-fingerprint-design.md`` §3.1 for
the normative determinism rules.
"""

from __future__ import annotations

import ast
import hashlib

_ScopeNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def enclosing_scope_node(ancestors: list[ast.AST]) -> _ScopeNode | None:
    """Return the innermost enclosing def/class from an innermost-first ancestor list.

    ``ancestors`` is the suppressed node followed by each parent up to the
    module, innermost first (the shape produced by the visitor's
    ``node_stack`` reversed). Returns ``None`` when the node is at module
    level (no enclosing def/class), which the caller maps to the
    whole-module fallback.
    """
    for node in ancestors:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return node
    return None


def _strip_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Return ``body`` without a leading docstring statement, if present.

    A docstring is an ``ast.Expr`` whose value is an ``ast.Constant`` holding
    a ``str``. Editing or adding a docstring must not change the fingerprint
    (rule 4): documentation is not the code the judge reasoned about.
    """
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[1:]
    return body


def compute_scope_fingerprint(scope_node: _ScopeNode | None, *, module: ast.Module | None = None) -> str:
    """Return the 64-char hex scope fingerprint for ``scope_node``.

    When ``scope_node`` is ``None`` (module-level suppression), ``module``
    must be provided and the whole module is fingerprinted instead. The
    leading docstring of the scope (or module) is excluded.
    """
    if scope_node is None:
        if module is None:
            raise ValueError("compute_scope_fingerprint: module is required when scope_node is None (module-level fallback)")
        target: ast.AST = ast.Module(body=_strip_leading_docstring(list(module.body)), type_ignores=list(module.type_ignores))
    else:
        # Shallow-copy the scope node and replace only its body, so the original
        # AST (still being walked by the visitor) is never mutated.
        import copy

        target = copy.copy(scope_node)
        target.body = _strip_leading_docstring(list(scope_node.body))  # type: ignore[union-attr]
    dump = ast.dump(target, include_attributes=False, annotate_fields=True)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/rules/trust_tier/tier_model/test_scope_fingerprint.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Type-check and lint the new module**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m mypy src/elspeth_lints/rules/trust_tier/tier_model/scope_fingerprint.py && ../.venv/bin/python -m ruff check src/elspeth_lints/rules/trust_tier/tier_model/scope_fingerprint.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/scope_fingerprint.py elspeth-lints/tests/rules/trust_tier/tier_model/test_scope_fingerprint.py
git commit -m "feat(tier-model): add enclosing-scope fingerprint primitive"
```

---

## Task 2: Stamp `scope_fingerprint` onto every `Finding`

The visitor maintains `self.node_stack` (every ancestor of the current node, pushed in `visit()` at rule.py:513–519). At finding-emit time we reverse it to get an innermost-first ancestor list, locate the enclosing scope, and compute the fingerprint. The `Finding` dataclass gets a new field; **every** construction site must populate it (advisor lock #3 — an unpopulated site defaults to `""` and would silently skip verification, so v2 verify rejects empty; see Task 6).

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py` (`Finding` at :76–100; `_add_finding` at :618; the layer-import finding emitter; the visitor needs the module root)
- Test: `elspeth-lints/tests/rules/trust_tier/tier_model/test_finding_scope_fingerprint.py`

- [ ] **Step 1: Write the failing test**

```python
# test_finding_scope_fingerprint.py
import ast
from pathlib import Path

from elspeth_lints.rules.trust_tier.tier_model.rule import scan_directory
from elspeth_lints.rules.trust_tier.tier_model.scope_fingerprint import compute_scope_fingerprint


def test_finding_carries_scope_fingerprint_of_enclosing_function(tmp_path: Path) -> None:
    src = "def handler(payload):\n    return payload.get('missing')\n"
    f = tmp_path / "mod.py"
    f.write_text(src)
    findings = scan_directory(tmp_path)
    r1 = [x for x in findings if x.rule_id == "R1"]
    assert r1, "expected an R1 dict.get finding"
    expected = compute_scope_fingerprint(ast.parse(src).body[0])  # the handler FunctionDef
    assert r1[0].scope_fingerprint == expected


def test_module_level_finding_uses_module_fallback(tmp_path: Path) -> None:
    src = "import os\nVALUE = os.environ.get('X')\n"
    f = tmp_path / "mod.py"
    f.write_text(src)
    findings = scan_directory(tmp_path)
    r1 = [x for x in findings if x.rule_id == "R1"]
    assert r1
    expected = compute_scope_fingerprint(None, module=ast.parse(src))
    assert r1[0].scope_fingerprint == expected
```

> NOTE before writing the test: confirm the public scan entry point name and signature with `grep -n "def scan_directory" src/elspeth_lints/rules/trust_tier/tier_model/rule.py`. If it requires arguments beyond the path (e.g. a root or config), pass the minimal real values — do not stub. If `R1` is not emitted for a bare module file in this codebase's config, switch the fixture to a rule that is (`R6` silent-except is reliably emitted) and recompute `expected` against the same enclosing scope.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/rules/trust_tier/tier_model/test_finding_scope_fingerprint.py -q`
Expected: FAIL with `AttributeError: 'Finding' object has no attribute 'scope_fingerprint'`.

- [ ] **Step 3: Add the field to `Finding`**

In `rule.py`, add to the `Finding` dataclass (after `ast_path: str = ""` at :100):

```python
    ast_path: str = ""
    scope_fingerprint: str = ""
```

Extend the class docstring (after the `ast_path` paragraph) with:

```
    ``scope_fingerprint`` is the v2 binding primitive: the 64-char hex
    fingerprint of the innermost enclosing scope (FunctionDef /
    AsyncFunctionDef / ClassDef, or the whole module at module level) of
    the finding's subject node. It replaces the whole-file ``file_fingerprint``
    for judge-gated entries — see
    ``elspeth_lints.rules.trust_tier.tier_model.scope_fingerprint``. It is
    computed forward by the visitor (never reverse-resolved from ``ast_path``)
    and verified at match time in ``verify_entry_binding_against_finding``.
```

- [ ] **Step 4: Compute the fingerprint in the visitor**

First, ensure the visitor can reach the module root and its ancestor chain. Add the import near the top of `rule.py`:

```python
from elspeth_lints.rules.trust_tier.tier_model.scope_fingerprint import (
    compute_scope_fingerprint,
    enclosing_scope_node,
)
```

Add a helper method to the visitor class (near `_fingerprint_node` at :533):

```python
    def _scope_fingerprint_for_current_node(self) -> str:
        """Return the enclosing-scope fingerprint for the node being visited.

        ``self.node_stack`` holds every ancestor of the current node,
        outermost-first (the module is index 0). Reversed it is innermost-
        first, the shape ``enclosing_scope_node`` expects. When there is no
        enclosing def/class the module root (``node_stack[0]``) is the
        fallback target.
        """
        ancestors = list(reversed(self.node_stack))
        scope = enclosing_scope_node(ancestors)
        if scope is not None:
            return compute_scope_fingerprint(scope)
        module = self.node_stack[0]
        assert isinstance(module, ast.Module)  # node_stack[0] is always the module root
        return compute_scope_fingerprint(None, module=module)
```

In `_add_finding` (:618), populate the field on the constructed `Finding`. Locate the `Finding(...)` construction inside `_add_finding` and add `scope_fingerprint=self._scope_fingerprint_for_current_node()` to it.

> Before editing, run `grep -n "Finding(" src/elspeth_lints/rules/trust_tier/tier_model/rule.py` to enumerate **every** construction site. For each AST-derived finding, add the `scope_fingerprint=` argument computed from the visitor context. For the **layer-import** finding emitter (findings with `ast_path` of the form `import:<module>` and no AST subject), the enclosing scope is the module — set `scope_fingerprint=compute_scope_fingerprint(None, module=<module tree for that file>)`. If the layer-import emitter does not have the module tree in scope, pass the module AST it parsed for that file; if it genuinely has none (path-only import graph), set `scope_fingerprint=""` **and** add a code comment noting these findings are never judge-gated (layer-import suppressions use per-file rules / TC warnings, not signed entries) so the empty value is intentional, not a missed site.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/rules/trust_tier/tier_model/test_finding_scope_fingerprint.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full tier-model rule test module to confirm no regressions**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/rules/trust_tier/tier_model/ -q`
Expected: PASS (existing tests unaffected — `scope_fingerprint` defaults to `""` where not asserted).

- [ ] **Step 7: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py elspeth-lints/tests/rules/trust_tier/tier_model/test_finding_scope_fingerprint.py
git commit -m "feat(tier-model): stamp enclosing-scope fingerprint onto every Finding"
```

---

## Task 3: Add `scope_fingerprint` + `judge_signature_version` to `AllowlistEntry` and the loader

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/allowlist.py` (`AllowlistEntry` :135–154; `_parse_allow_hits` :439–465)
- Test: `elspeth-lints/tests/core/test_allowlist_schema.py` (extend; confirm exact path with `ls tests/core | grep allowlist`)

- [ ] **Step 1: Write the failing test**

```python
def test_entry_parses_scope_fingerprint_and_version() -> None:
    from elspeth_lints.core.allowlist import _parse_allow_hits

    data = {
        "allow_hits": [
            {
                "key": "core/x.py:R6:C:m:fp=abc",
                "owner": "tester",
                "reason": "trust boundary on external call",
                "safety": "row quarantined on failure",
                "judge_signature_version": 2,
                "scope_fingerprint": "a" * 64,
                "ast_path": "body[0]/body[0]",
                # ... full judge cluster omitted here; this test only asserts the two new fields parse
            }
        ]
    }
    # Parse without source_root so no live-source verification runs (schema-only).
    # NOTE: a real judge entry needs the full cluster to pass _validate_judge_metadata_atomic;
    # extend `data` with judge_verdict/judge_recorded_at/judge_model/judge_policy_hash/judge_rationale
    # /judge_metadata_signature from an existing fixture in this file before running.
    entries = _parse_allow_hits(data, source_file="x.yaml", source_root=None)
    assert entries[0].scope_fingerprint == "a" * 64
    assert entries[0].judge_signature_version == 2
```

> Use an existing fully-populated judge-entry fixture in this test file as the base and add the two new keys — that keeps the atomic validator satisfied. The assertion is only about the two new fields round-tripping.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_schema.py -k scope_fingerprint_and_version -q`
Expected: FAIL — `AllowlistEntry.__init__() got an unexpected keyword argument 'scope_fingerprint'` (or the parser ignores the unknown YAML key and the assertion fails).

- [ ] **Step 3: Add the dataclass fields**

In `allowlist.py`, in `AllowlistEntry` (after `ast_path: str | None = None` at :141):

```python
    ast_path: str | None = None
    scope_fingerprint: str | None = None
    judge_signature_version: int | None = None
```

- [ ] **Step 4: Parse the fields in `_parse_allow_hits`**

In `_parse_allow_hits` (the `AllowlistEntry(...)` construction at :439), add after `ast_path=...` (:446):

```python
            ast_path=_optional_string(entry, "ast_path", context=ctx),
            scope_fingerprint=_optional_string(entry, "scope_fingerprint", context=ctx),
            judge_signature_version=_optional_signature_version(entry, "judge_signature_version", context=ctx),
```

Add the parse helper near `_optional_int` (:1147):

```python
def _optional_signature_version(data: dict[str, Any], key: str, *, context: str) -> int | None:
    """Parse the optional judge signature version (1 or 2).

    Absent / null yield ``None`` (the pre-judge era and v1 legacy entries
    written before the version field existed are treated as v1 at the
    dispatch site). Any present value must be exactly 1 or 2 — an unknown
    version is corruption and crashes on load per Tier-1 doctrine.
    """
    value = _optional_int(data, key, context=context)
    if value is not None and value not in (1, 2):
        raise ValueError(f"{context}.{key} must be 1 or 2; got {value!r}")
    return value
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_schema.py -k scope_fingerprint_and_version -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/allowlist.py elspeth-lints/tests/core/test_allowlist_schema.py
git commit -m "feat(allowlist): add scope_fingerprint + judge_signature_version fields"
```

---

## Task 4: Version-2 signature payload + atomic-validator dispatch

The signed payload must bind `scope_fingerprint` for v2 instead of `file_fingerprint`, and the version must live *inside* the signed payload so a v1↔v2 flip is unforgeable without the key (spec §4.1). The atomic validator must require the correct binding field per version.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/allowlist.py` (`compute_judge_metadata_signature` :547–598; `_verify_judge_metadata_signature_at_load` :641–681; `_validate_judge_metadata_atomic` invariant 8 :948–962; signature shape validator :684–695)
- Test: `elspeth-lints/tests/core/test_allowlist_signing.py` (confirm filename with `ls tests/core | grep -i sign`)

- [ ] **Step 1: Write the failing tests**

```python
import hmac

from elspeth_lints.core.allowlist import compute_judge_metadata_signature
# import JudgeVerdict and a datetime fixture as the existing signing tests do

_TEST_KEY = b"x" * 32


def _sig(**overrides):
    base = dict(
        key="core/x.py:R6:C:m:fp=abc",
        ast_path="body[0]/body[0]",
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=_AWARE_DT,           # reuse this file's fixture
        judge_model="anthropic/claude-opus",
        judge_rationale="external call boundary",
        judge_policy_hash="sha256:" + "0" * 64,
        hmac_key=_TEST_KEY,
    )
    base.update(overrides)
    return compute_judge_metadata_signature(**base)


def test_v2_signature_binds_scope_fingerprint() -> None:
    sig = _sig(signature_version=2, scope_fingerprint="a" * 64)
    assert sig.startswith("hmac-sha256:v2:")


def test_v1_signature_binds_file_fingerprint_and_keeps_v1_prefix() -> None:
    sig = _sig(signature_version=1, file_fingerprint="b" * 64)
    assert sig.startswith("hmac-sha256:v1:")


def test_v1_and_v2_signatures_differ_for_same_logical_entry() -> None:
    v1 = _sig(signature_version=1, file_fingerprint="b" * 64)
    v2 = _sig(signature_version=2, scope_fingerprint="a" * 64)
    assert not hmac.compare_digest(v1, v2)


def test_v2_requires_scope_fingerprint() -> None:
    with pytest.raises(ValueError, match="scope_fingerprint"):
        _sig(signature_version=2, scope_fingerprint=None)


def test_v1_requires_file_fingerprint() -> None:
    with pytest.raises(ValueError, match="file_fingerprint"):
        _sig(signature_version=1, file_fingerprint=None)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_signing.py -k "v2 or v1_signature or v1_and_v2 or requires" -q`
Expected: FAIL — `compute_judge_metadata_signature() got an unexpected keyword argument 'signature_version'`.

- [ ] **Step 3: Make the signature versioned**

Replace the signature of `compute_judge_metadata_signature` (:547) — make `file_fingerprint` optional, add `scope_fingerprint` and `signature_version`:

```python
def compute_judge_metadata_signature(
    *,
    key: str,
    ast_path: str,
    judge_verdict: JudgeVerdict,
    judge_recorded_at: datetime,
    judge_model: str,
    judge_rationale: str,
    judge_policy_hash: str,
    signature_version: int = 1,
    file_fingerprint: str | None = None,
    scope_fingerprint: str | None = None,
    judge_model_verdict: JudgeVerdict | None = None,
    judge_confidence: float | None = None,
    judge_excerpt_redactions: tuple[RedactionRecord, ...] = (),
    hmac_key: bytes | None = None,
) -> str:
```

Replace the payload/prefix construction (the body at :572–598) with version dispatch:

```python
    if hmac_key is None:
        hmac_key = _judge_metadata_hmac_key()
    if signature_version == 2:
        if scope_fingerprint is None:
            raise ValueError("compute_judge_metadata_signature: scope_fingerprint is required for signature_version 2")
        binding: dict[str, Any] = {"scope_fingerprint": scope_fingerprint}
        prefix = _JUDGE_METADATA_SIGNATURE_PREFIX_V2
    elif signature_version == 1:
        if file_fingerprint is None:
            raise ValueError("compute_judge_metadata_signature: file_fingerprint is required for signature_version 1")
        binding = {"file_fingerprint": file_fingerprint}
        prefix = _JUDGE_METADATA_SIGNATURE_PREFIX_V1
    else:
        raise ValueError(f"compute_judge_metadata_signature: unknown signature_version {signature_version!r}")
    payload = {
        "version": signature_version,
        "key": key,
        **binding,
        "ast_path": ast_path,
        "judge_verdict": judge_verdict.value,
        "judge_model_verdict": judge_model_verdict.value if judge_model_verdict is not None else None,
        "judge_recorded_at": judge_recorded_at.isoformat(),
        "judge_model": judge_model,
        "judge_rationale": judge_rationale,
        "judge_policy_hash": judge_policy_hash,
        "judge_excerpt_redactions": [
            {"pattern": r.pattern_name, "byte_count": r.byte_count, "redacted_hash": r.redacted_hash}
            for r in judge_excerpt_redactions
        ],
    }
    if judge_confidence is not None:
        payload["judge_confidence"] = judge_confidence
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hmac.new(hmac_key, canonical, hashlib.sha256).hexdigest()
    return f"{prefix}{digest}"
```

Replace the module constant (:26):

```python
_JUDGE_METADATA_SIGNATURE_PREFIX_V1 = "hmac-sha256:v1:"
_JUDGE_METADATA_SIGNATURE_PREFIX_V2 = "hmac-sha256:v2:"
_JUDGE_METADATA_SIGNATURE_PREFIXES = (_JUDGE_METADATA_SIGNATURE_PREFIX_V1, _JUDGE_METADATA_SIGNATURE_PREFIX_V2)
```

> Search the module for the old name `_JUDGE_METADATA_SIGNATURE_PREFIX` and update every reference (shape validator at :685–687, and the rotate.py copy at rotate.py:88 — Task 8).

Update `_validate_judge_metadata_signature_shape` (:684) to accept either prefix:

```python
def _validate_judge_metadata_signature_shape(signature: str, *, context: str) -> None:
    if not signature.startswith(_JUDGE_METADATA_SIGNATURE_PREFIXES):
        raise ValueError(f"{context}: judge_metadata_signature must start with one of {_JUDGE_METADATA_SIGNATURE_PREFIXES}; got {signature!r}")
    for prefix in _JUDGE_METADATA_SIGNATURE_PREFIXES:
        if signature.startswith(prefix):
            digest = signature.removeprefix(prefix)
            break
    if len(digest) != 64:
        raise ValueError(f"{context}: judge_metadata_signature digest must be 64 lowercase hex characters")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError(f"{context}: judge_metadata_signature digest must be valid hex") from exc
    if digest.lower() != digest:
        raise ValueError(f"{context}: judge_metadata_signature digest must use lowercase hex")
```

- [ ] **Step 4: Dispatch verification by version at load**

In `_verify_judge_metadata_signature_at_load` (:662), replace the `compute_judge_metadata_signature(...)` recompute call so it passes the entry's version and the correct binding field:

```python
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    expected = compute_judge_metadata_signature(
        key=entry.key,
        ast_path=entry.ast_path,
        judge_verdict=entry.judge_verdict,
        judge_model_verdict=entry.judge_model_verdict,
        judge_recorded_at=entry.judge_recorded_at,
        judge_model=entry.judge_model,
        judge_rationale=entry.judge_rationale,
        judge_policy_hash=entry.judge_policy_hash,
        judge_excerpt_redactions=entry.judge_excerpt_redactions,
        judge_confidence=entry.judge_confidence,
        signature_version=version,
        file_fingerprint=entry.file_fingerprint,
        scope_fingerprint=entry.scope_fingerprint,
    )
```

> The pre-recompute `assert entry.file_fingerprint is not None` at :653 must change: a v2 entry has `file_fingerprint is None` and `scope_fingerprint is not None`. Replace the binding asserts with a version-aware guard:
> ```python
>     if version == 2:
>         assert entry.scope_fingerprint is not None
>     else:
>         assert entry.file_fingerprint is not None
>     assert entry.ast_path is not None
> ```

- [ ] **Step 5: Make invariant 8 version-aware**

In `_validate_judge_metadata_atomic`, invariant 8 (:948–962) currently requires `file_fingerprint` + `ast_path` for every judge-gated entry. Replace the binding-presence block so the binding field required depends on the version:

```python
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    missing_binding: list[str] = []
    if entry.ast_path is None:
        missing_binding.append("ast_path")
    if version == 2:
        if entry.scope_fingerprint is None:
            missing_binding.append("scope_fingerprint")
        if entry.file_fingerprint is not None:
            raise ValueError(
                f"{context}: judge_signature_version is 2 but file_fingerprint is present; "
                "v2 entries bind via scope_fingerprint and must not carry the v1 file_fingerprint."
            )
    else:
        if entry.file_fingerprint is None:
            missing_binding.append("file_fingerprint")
        if entry.scope_fingerprint is not None:
            raise ValueError(
                f"{context}: judge_signature_version is absent/1 but scope_fingerprint is present; "
                "v1 entries bind via file_fingerprint. A scope_fingerprint on a v1 entry is corruption."
            )
    if missing_binding:
        raise ValueError(
            f"{context}: judge_verdict is set ({entry.judge_verdict.value!r}) but required "
            f"binding fields are missing ({', '.join(missing_binding)})."
        )
```

> Also update invariant 4 (:889–916, the pre-judge stray-field check) to add `scope_fingerprint` and `judge_signature_version` to the `stray` list checks, so a verdict-less entry carrying either is rejected as corruption:
> ```python
>         if entry.scope_fingerprint is not None:
>             stray.append("scope_fingerprint")
>         if entry.judge_signature_version is not None:
>             stray.append("judge_signature_version")
> ```

- [ ] **Step 6: Run the signing tests + the full allowlist suite**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_signing.py tests/core/test_allowlist_schema.py -q`
Expected: PASS. **The default `signature_version=1` is deliberate for the dual-version window** (advisor fix B): existing direct callers that omit the version keep getting a v1 signature and stay green untouched — including `_build_yaml_entry_text`, which still calls `compute_judge_metadata_signature(..., file_fingerprint=...)` until Task 7. If the default were `2`, justify would raise "scope_fingerprint is required" across Tasks 4–6 and the full suite (notably `test_cli_justify`) would be red between commits — violating the green-between-commits guarantee. Task 7 passes `signature_version=2` explicitly; Task 13 makes v2 the only path and flips/removes the default.

- [ ] **Step 7: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/allowlist.py elspeth-lints/tests/core/test_allowlist_signing.py
git commit -m "feat(allowlist): version-2 signature payload binding scope_fingerprint"
```

---

## Task 5: Load-time — file-exists for v2, byte-hash for v1 only

Under v2 there is no whole-file hash to verify at load. The load-time check splits into a version-independent *file-exists* guard (always) and the v1-only byte-hash.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/allowlist.py` (`_verify_file_fingerprint_at_load` :713–745; its call site :469)
- Test: `elspeth-lints/tests/core/test_allowlist_load_binding.py` (confirm/extend the file that tests `_verify_file_fingerprint_at_load`)

- [ ] **Step 1: Write the failing tests**

```python
def test_v2_entry_loads_when_file_present_even_if_bytes_changed(tmp_path) -> None:
    # A v2 entry binds by scope, not whole-file bytes. Editing an unrelated
    # line in the file must NOT crash the load (the relief this change delivers).
    # Build a source tree + a signed v2 entry whose scope is unchanged, mutate an
    # unrelated line, and assert load_allowlist(..., source_root=...) does not raise.
    ...


def test_v2_entry_crashes_at_load_when_source_file_missing(tmp_path) -> None:
    # The file-exists guard still fires for v2.
    with pytest.raises(ValueError, match="does not exist"):
        ...


def test_v1_entry_still_crashes_on_whole_file_byte_drift(tmp_path) -> None:
    # The retained v1 path is unchanged.
    with pytest.raises(ValueError, match="file_fingerprint mismatch"):
        ...
```

> Build these with the real loader and a real on-disk source file + signed entry (test key via the `hmac_key` path is not available through `load_allowlist`, which calls `_judge_metadata_hmac_key()` — so set `ELSPETH_JUDGE_METADATA_HMAC_KEY` to a 32-byte test value via `monkeypatch.setenv` for these load tests, and sign the fixture entries with the same value). This mirrors how the existing load-binding tests construct signed fixtures — copy that setup.

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_load_binding.py -q`
Expected: FAIL — the v2 entry currently has no `file_fingerprint`, so invariant 8 (now version-aware after Task 4) passes, but `_verify_file_fingerprint_at_load` still asserts `entry.file_fingerprint is not None` and raises.

- [ ] **Step 3: Split the load-time check**

Rename and restructure `_verify_file_fingerprint_at_load` (:713) into a version-aware guard:

```python
def _verify_source_binding_at_load(entry: AllowlistEntry, *, source_root: Path, context: str) -> None:
    """Verify a judge-gated entry's source binding at load, dispatched by version.

    All versions: the source file the entry's key points at must exist
    (a binding to a deleted file is audit-broken). v1 additionally
    recomputes the whole-file byte hash (its binding primitive). v2's
    binding primitive (scope_fingerprint) is parse-dependent and is
    verified at match time in ``verify_entry_binding_against_finding``,
    reusing the scanner's parse — not re-parsed here.
    """
    file_path = _file_path_from_canonical_key(entry.key)
    source_path = source_root / file_path
    if not source_path.exists():
        raise ValueError(
            f"{context}: judge-gated entry binds to {file_path!r} which does not exist "
            f"under {source_root}; either the source file was removed without removing "
            f"the dependent allowlist entry, or the entry's key was transplanted from a "
            f"different repository layout. Refusing to load."
        )
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    if version == 2:
        # scope_fingerprint is verified at match time; nothing more to do at load.
        return
    live_fingerprint = _compute_file_fingerprint(source_path)
    assert entry.file_fingerprint is not None  # invariant 8 guarantees presence for v1
    if entry.file_fingerprint != live_fingerprint:
        raise ValueError(
            f"{context}: file_fingerprint mismatch for {file_path!r}: persisted "
            f"{entry.file_fingerprint!r} but live source hashes to {live_fingerprint!r}. "
            f"Either the source was modified after judgment (re-justify required) or the "
            f"quartet was transplanted from a different file (corruption / tampering)."
        )
```

Update the call site (:469):

```python
            _verify_source_binding_at_load(allowlist_entry, source_root=source_root, context=ctx)
            _verify_judge_metadata_signature_at_load(allowlist_entry, context=ctx)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_load_binding.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/allowlist.py elspeth-lints/tests/core/test_allowlist_load_binding.py
git commit -m "feat(allowlist): v2 load-time is file-exists only; v1 byte-hash retained"
```

---

## Task 6: Match-time scope verification

`verify_entry_binding_against_finding` (allowlist.py:748) already checks `ast_path`. For v2 it must also assert the finding's live `scope_fingerprint` equals the entry's. Critically (advisor lock #3), v2 verification **must reject an empty/None finding scope_fingerprint** so that any un-stamped finding construction site crashes loudly rather than silently passing on `"" == ""`.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/allowlist.py` (`verify_entry_binding_against_finding` :748–780)
- Modify: `elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py` (`_match_finding` call site :2201)
- Modify: `elspeth-lints/src/elspeth_lints/rules/trust_boundary/shared.py` (call site :428 — pass the new arg)
- Test: `elspeth-lints/tests/core/test_allowlist_match_binding.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_v2_match_passes_when_scope_fingerprint_matches() -> None:
    entry = _v2_entry(scope_fingerprint="a" * 64, ast_path="body[0]/body[0]")
    verify_entry_binding_against_finding(
        entry, file_path="core/x.py", ast_path="body[0]/body[0]", scope_fingerprint="a" * 64
    )  # no raise


def test_v2_match_crashes_on_scope_drift() -> None:
    entry = _v2_entry(scope_fingerprint="a" * 64, ast_path="body[0]/body[0]")
    with pytest.raises(ValueError, match="scope_fingerprint"):
        verify_entry_binding_against_finding(
            entry, file_path="core/x.py", ast_path="body[0]/body[0]", scope_fingerprint="b" * 64
        )


def test_v2_match_crashes_on_empty_finding_scope_fingerprint() -> None:
    # An un-stamped finding (default "") must NOT silently pass.
    entry = _v2_entry(scope_fingerprint="a" * 64, ast_path="body[0]/body[0]")
    with pytest.raises(ValueError, match="scope_fingerprint"):
        verify_entry_binding_against_finding(
            entry, file_path="core/x.py", ast_path="body[0]/body[0]", scope_fingerprint=""
        )


def test_v1_match_ignores_scope_fingerprint_and_checks_ast_path() -> None:
    entry = _v1_entry(file_fingerprint="b" * 64, ast_path="body[0]/body[0]")
    verify_entry_binding_against_finding(
        entry, file_path="core/x.py", ast_path="body[0]/body[0]", scope_fingerprint="anything"
    )  # v1 has no scope binding; ast_path still enforced
    with pytest.raises(ValueError, match="ast_path mismatch"):
        verify_entry_binding_against_finding(
            entry, file_path="core/x.py", ast_path="body[9]", scope_fingerprint=""
        )
```

> `_v2_entry` / `_v1_entry` are tiny local builders for an `AllowlistEntry` with `judge_verdict=ACCEPTED` and the right binding fields/version. No signature needed — `verify_entry_binding_against_finding` does not check the HMAC (that is the load-time path).

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_match_binding.py -q`
Expected: FAIL — `verify_entry_binding_against_finding() got an unexpected keyword argument 'scope_fingerprint'`.

- [ ] **Step 3: Add the scope check**

Replace `verify_entry_binding_against_finding` (:748). Add the `scope_fingerprint` parameter and the v2 branch:

```python
def verify_entry_binding_against_finding(
    entry: AllowlistEntry, *, file_path: str, ast_path: str, scope_fingerprint: str
) -> None:
    """Assert a matched judge-gated entry still binds to the live finding.

    Checks ``ast_path`` (all versions — in-file transplant defence) and,
    for v2 entries, ``scope_fingerprint`` (the enclosing-scope content the
    judge read). The live ``scope_fingerprint`` is the one the scanner
    stamped onto the finding; an empty value means the finding was emitted
    by a construction site that did not compute it, which for a v2 entry is
    a defect (we must never pass a v2 binding on an unverifiable empty
    value) and crashes here.

    Pre-judge entries (``judge_verdict is None``) carry no binding and are
    not checked.
    """
    if entry.judge_verdict is None:
        return
    assert entry.ast_path is not None
    if entry.ast_path != ast_path:
        raise ValueError(
            f"ast_path mismatch on judge-gated entry for {file_path!r}: persisted "
            f"{entry.ast_path!r} but live finding's ast_path is {ast_path!r}. Entry key: {entry.key!r}."
        )
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    if version == 2:
        assert entry.scope_fingerprint is not None  # invariant 8 (v2) guarantees presence
        if not scope_fingerprint:
            raise ValueError(
                f"scope_fingerprint missing on the live finding for judge-gated v2 entry "
                f"{entry.key!r} ({file_path!r}); the scanner must stamp scope_fingerprint on "
                "every finding. An empty value cannot verify a v2 binding."
            )
        if entry.scope_fingerprint != scope_fingerprint:
            raise ValueError(
                f"scope_fingerprint mismatch on judge-gated entry for {file_path!r}: persisted "
                f"{entry.scope_fingerprint!r} but the live enclosing scope hashes to "
                f"{scope_fingerprint!r}. The function/class the judge inspected changed; "
                f"re-justify is required. Entry key: {entry.key!r}."
            )
```

- [ ] **Step 4: Update both call sites**

In `rule.py` `_match_finding` (:2201):

```python
            verify_entry_binding_against_finding(
                entry,
                file_path=finding.file_path,
                ast_path=finding.ast_path,
                scope_fingerprint=finding.scope_fingerprint,
            )
```

In `trust_boundary/shared.py` (:428):

```python
        verify_entry_binding_against_finding(
            matched, file_path=finding.file_path, ast_path=finding.ast_path, scope_fingerprint=finding.scope_fingerprint,
        )
```

> The trust_boundary `Finding` type may differ from tier_model's. Confirm with `grep -n "scope_fingerprint\|class Finding\|ast_path" src/elspeth_lints/rules/trust_boundary/shared.py` (and wherever its Finding is defined). If trust_boundary findings have **no** `scope_fingerprint` attribute, two options: (a) trust_boundary entries are all v1 today (no judge entries on disk per the audit) so pass `scope_fingerprint=getattr(finding, "scope_fingerprint", "")` and the v1 branch ignores it; **but** a future v2 trust_boundary entry would then crash with the "missing on live finding" error, which is the *correct* fail-closed behaviour and a clear signal to wire trust_boundary's scanner. Add a code comment to that effect. Do **not** silently default to a fabricated value.

- [ ] **Step 5: Run the match-binding tests + both rule suites**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_match_binding.py tests/rules/trust_tier/tier_model/ tests/rules/trust_boundary/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/allowlist.py elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py elspeth-lints/src/elspeth_lints/rules/trust_boundary/shared.py elspeth-lints/tests/core/test_allowlist_match_binding.py
git commit -m "feat(allowlist): match-time scope_fingerprint verification for v2 entries"
```

---

## Task 7: `justify` writes v2 entries

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/cli.py` (`_build_yaml_entry_text` :1830–1939; the justify call site :1390–1408)
- Test: `elspeth-lints/tests/core/test_cli_justify.py` (confirm name with `ls tests/core | grep -i justify`)

- [ ] **Step 1: Write the failing test**

```python
def test_justify_writes_v2_entry_with_scope_fingerprint(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", "x" * 32)
    # Drive _run_justify (or the dry-run path) against a real source file containing
    # an R6 suppression with a known enclosing scope. Assert the emitted YAML text:
    #   - contains "judge_signature_version: 2"
    #   - contains "scope_fingerprint: <64 hex>"
    #   - does NOT contain "file_fingerprint:"
    #   - judge_metadata_signature starts with "hmac-sha256:v2:"
    # and that re-loading the written YAML with source_root verifies clean.
    ...
```

> Mirror the existing justify CLI test setup (it mocks/stubs the judge transport to return an ACCEPTED `JudgeResponse`). Reuse that harness; only the assertions about the new fields are new.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_cli_justify.py -k v2 -q`
Expected: FAIL — output still contains `file_fingerprint:` and a `v1` signature.

- [ ] **Step 3: Thread `scope_fingerprint` to the writer**

At the justify call site (cli.py:1390), the `finding` already carries `scope_fingerprint` (it comes from `_scan_single_file_findings_for_justify` → the visitor — confirmed). Replace the `build_signed_yaml_entry` call (:1393) to pass scope instead of file fingerprint:

```python
    scope_fingerprint = _finding_scope_fingerprint(finding)
    target_yaml = _suggest_yaml_target(finding=finding, allowlist_dir=allowlist_dir)

    def build_signed_yaml_entry() -> str:
        return _build_yaml_entry_text(
            key=finding_key,
            owner=args.owner,
            reason=args.rationale,
            verdict=write_verdict,
            recorded_at=response.recorded_at,
            model_id=response.model_id,
            judge_rationale=response.judge_rationale,
            judge_confidence=response.confidence,
            policy_hash=response.policy_hash,
            model_verdict=model_verdict,
            scope_fingerprint=scope_fingerprint,
            ast_path=_finding_ast_path(finding),
            excerpt_redactions=safe_excerpt.redactions,
        )
```

Add the accessor near `_finding_ast_path` (:1801):

```python
def _finding_scope_fingerprint(finding: Any) -> str:
    scope_fingerprint = getattr(finding, "scope_fingerprint", "")
    if not isinstance(scope_fingerprint, str) or not scope_fingerprint:
        raise ValueError(
            f"finding {_finding_canonical_key(finding)} has no scope_fingerprint; "
            "judge-gated v2 entries must bind to the enclosing scope the judge inspected. "
            "The scanner must stamp scope_fingerprint on every finding."
        )
    return scope_fingerprint
```

> The `file_fingerprint = safe_excerpt.file_fingerprint` line at :1390 and the comment block above it (:1377–1389) describe the v1 binding; remove the now-unused `file_fingerprint` local and rewrite the comment to describe the scope binding. `safe_excerpt.file_fingerprint` is still used as the scrubber salt for `RedactionRecord` — do **not** remove that usage inside `extract_safe_excerpt`; only the *binding* use here goes away.

- [ ] **Step 4: Rewrite `_build_yaml_entry_text` for v2**

Change its signature (:1830): replace `file_fingerprint: str` with `scope_fingerprint: str`. Update the `compute_judge_metadata_signature` call (:1884) to pass `signature_version=2, scope_fingerprint=scope_fingerprint` (drop `file_fingerprint`). Replace the binding-field emission (:1918–1925):

```python
    lines.append(f"  judge_signature_version: 2")
    lines.append(f"  scope_fingerprint: {_yaml_inline_scalar(scope_fingerprint)}")
    lines.append(f"  ast_path: {_yaml_inline_scalar(ast_path)}")
    lines.append(f"  judge_metadata_signature: {_yaml_inline_scalar(judge_metadata_signature)}")
```

> Confirm the YAML emits `judge_signature_version: 2` as an integer (no quotes). `_yaml_inline_scalar` is for strings; an int literal is emitted directly via the f-string as shown. Verify the loader's `_optional_signature_version` reads it back as `int` 2 (PyYAML parses unquoted `2` as int) — covered by Task 3's parser.

- [ ] **Step 5: Run the justify test + reload round-trip**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_cli_justify.py -q`
Expected: PASS, including the written-then-reloaded verification.

- [ ] **Step 6: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/cli.py elspeth-lints/tests/core/test_cli_justify.py
git commit -m "feat(cli): justify writes v2 entries binding scope_fingerprint"
```

---

## Task 8: Retarget the `file_fingerprint` consumers

Three readers reference `file_fingerprint` and must understand v2 entries: the coverage-diff identity tuple, the sidecar round-trip, and the rotation refusal/raw-entry read.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/judge_coverage.py` (:398–423)
- Modify: `elspeth-lints/src/elspeth_lints/core/reaudit_sidecar.py` (`_entry_to_dict` :898–917; `_entry_from_dict` :944+)
- Modify: `elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rotate.py` (:88, :96, :820)
- Test: extend the existing tests for each (`test_judge_coverage*`, `test_reaudit_sidecar*`, `test_rotate*`)

- [ ] **Step 1: Write failing tests**

```python
# judge_coverage: a v2 entry's binding identity must include scope_fingerprint
def test_judge_binding_identity_uses_scope_for_v2() -> None:
    from elspeth_lints.core.judge_coverage import _judge_binding_identity
    entry = _v2_entry(scope_fingerprint="a" * 64, ast_path="p")
    ident = _judge_binding_identity(entry)
    assert "a" * 64 in ident  # scope_fingerprint participates in the identity

# sidecar: round-trip preserves the two new fields
def test_sidecar_round_trips_scope_and_version() -> None:
    from elspeth_lints.core.reaudit_sidecar import _entry_to_dict, _entry_from_dict
    entry = _v2_entry(scope_fingerprint="a" * 64, judge_signature_version=2)
    restored = _entry_from_dict(_entry_to_dict(entry), sidecar_path=Path("s"), line_no=1)
    assert restored.scope_fingerprint == "a" * 64
    assert restored.judge_signature_version == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_judge_coverage.py tests/core/test_reaudit_sidecar.py -k "scope or version" -q`
Expected: FAIL.

- [ ] **Step 3: judge_coverage — include both binding primitives**

In `_judge_binding_identity` (:398) and `_judge_metadata_payload` (:420), add `scope_fingerprint` (and keep `file_fingerprint` so v1 entries' identity is unchanged):

```python
def _judge_binding_identity(entry: AllowlistEntry) -> tuple[str, str | None, str | None, str | None]:
    """Return source-binding fields a fresh ``justify`` run may legitimately change."""
    return (entry.key, entry.file_fingerprint, entry.scope_fingerprint, entry.ast_path)
```

In `_judge_metadata_payload` (:403), add `entry.scope_fingerprint` and `entry.judge_signature_version` to the returned tuple (after `entry.file_fingerprint`), and add `entry.scope_fingerprint is None` to the pre-judge `None`-guard at :411.

- [ ] **Step 4: sidecar — round-trip the new fields**

In `_entry_to_dict` (:898) add after `"file_fingerprint": entry.file_fingerprint,`:

```python
        "scope_fingerprint": entry.scope_fingerprint,
        "judge_signature_version": entry.judge_signature_version,
```

In `_entry_from_dict` (:944) add to the `AllowlistEntry(...)` construction after `ast_path=...`:

```python
        scope_fingerprint=_optional_str(payload, "scope_fingerprint", sidecar_path, line_no),
        judge_signature_version=_optional_int_field(payload, "judge_signature_version", sidecar_path, line_no),
```

> If `reaudit_sidecar` has no `_optional_int_field` helper, add a minimal one mirroring `_optional_str` that accepts `int | None` and rejects `bool`. Confirm with `grep -n "_optional_int\|_optional_str" src/elspeth_lints/core/reaudit_sidecar.py`.

- [ ] **Step 5: rotate.py — retarget the prefix constant + raw read**

At rotate.py:88, update the duplicated prefix constant to import or mirror the new names:

```python
from elspeth_lints.core.allowlist import _JUDGE_METADATA_SIGNATURE_PREFIXES  # if exported; else mirror both literals
```

In `_REJUDGE_REQUIRED_FIELDS` (:89) the set lists the v1 binding field `file_fingerprint`. A v2 entry has `scope_fingerprint` instead. Make the required-fields check version-aware where it is consumed, or broaden the set to treat *either* binding field as satisfying the requirement. Inspect the consumer of `_REJUDGE_REQUIRED_FIELDS` (`grep -n "_REJUDGE_REQUIRED_FIELDS" src/...`) and adjust so a v2 entry with `scope_fingerprint` (and no `file_fingerprint`) is not flagged as missing a required field.

At rotate.py:820, the raw-entry read `raw_entry["file_fingerprint"]` will `KeyError` on a v2 YAML entry. Change `source_binding` construction to read whichever binding key is present:

```python
        binding_value = raw_entry.get("scope_fingerprint", raw_entry.get("file_fingerprint"))
        source_binding=(key, binding_value, raw_entry["ast_path"]),
```

> `_refuse_rotation_of_judge_gated_entry` (:131) refuses purely on `judge_verdict is not None` — it does **not** read `file_fingerprint`, so it needs no change; its docstring mentions the field but the logic is version-agnostic. Update the docstring text to say "the persisted binding fields" rather than naming `file_fingerprint`, to avoid documentation rot.

- [ ] **Step 6: Run the three consumer suites**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_judge_coverage.py tests/core/test_reaudit_sidecar.py tests/rules/trust_tier/tier_model/test_rotate*.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/judge_coverage.py elspeth-lints/src/elspeth_lints/core/reaudit_sidecar.py elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rotate.py elspeth-lints/tests/
git commit -m "feat: retarget file_fingerprint consumers to understand v2 scope binding"
```

---

## Task 9: Restore reaudit's pre-re-judge tamper check for v2

`reaudit.py` loads with `source_root` purely to trigger the *load-time* `file_fingerprint` tamper check (its comment at :680–683), then re-judges. Under v2 the load-time binding is only file-exists, so reaudit loses the "weren't tampered since the original verdict" check for v2 entries. reaudit already scans findings per file (`findings_cache`, `_scan_findings_for_file`); we match each entry to its live finding and call the binding verifier before re-judging.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/reaudit.py` (the per-entry loop that uses `findings_cache`, after :716)
- Test: `elspeth-lints/tests/core/test_reaudit*.py`

- [ ] **Step 1: Write the failing test**

```python
def test_reaudit_crashes_on_v2_scope_drift_before_rejudging(monkeypatch, tmp_path) -> None:
    # Build a v2 signed entry whose enclosing scope has since changed (the
    # entry's key/fp still match the finding because the suppressed node is
    # unchanged, but the surrounding function body changed). reaudit must
    # raise the scope_fingerprint mismatch before dispatching to the judge.
    ...
```

> Locate reaudit's per-entry dispatch and how it gets a finding for an entry (`grep -n "findings_cache\|_scan_findings_for_file\|_finding_key_for\|canonical_key" src/elspeth_lints/core/reaudit.py`). The match is by canonical key; once you have the matching live finding for the entry, call `verify_entry_binding_against_finding(entry, file_path=..., ast_path=finding.ast_path, scope_fingerprint=finding.scope_fingerprint)`.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_reaudit.py -k scope_drift -q`
Expected: FAIL — reaudit re-judges without raising.

- [ ] **Step 3: Add the match-time check into reaudit's scan loop**

In the per-entry processing (where `findings_cache[file_path]` is consulted), after obtaining the entry's matching finding, insert:

```python
from elspeth_lints.core.allowlist import verify_entry_binding_against_finding

# Restore the pre-re-judge binding check the v1 load-time file_fingerprint
# gate used to provide. For v2 entries the binding is scope-scoped and
# parse-dependent, so it is verified here against the finding we just
# scanned — not at load. A drifted scope crashes before we spend a judge call.
if entry.judge_verdict is not None and matching_finding is not None:
    verify_entry_binding_against_finding(
        entry,
        file_path=matching_finding.file_path,
        ast_path=matching_finding.ast_path,
        scope_fingerprint=matching_finding.scope_fingerprint,
    )
```

> If reaudit's design intentionally re-judges entries whose *finding no longer fires* (stale-but-present, to record a SUPERSEDED outcome), do not crash when there is no matching finding — the binding check only applies when a finding matched. Preserve reaudit's existing handling of the no-finding case; only add the check on the has-finding branch.

- [ ] **Step 4: Run the reaudit suite**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_reaudit*.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/reaudit.py elspeth-lints/tests/core/
git commit -m "feat(reaudit): verify v2 scope binding against live finding before re-judging"
```

---

## Task 10: `migrate-judge-scope` CLI command (code only; operator runs it)

A keyed batch command migrates every **currently-valid v1** entry to v2: load v1 → run the scan → for each entry that matches a live finding, take that finding's `scope_fingerprint` and re-sign as v2, carrying verdict/rationale forward unchanged. Entries that match no finding (already stale, e.g. today's 7) are **refused** and reported — they need genuine `justify`/`reaudit` against the changed source. This reuses the matcher (advisor lock #1): no `ast_path`→node reverse-resolver, and the matched/unmatched split *is* the valid/stale split.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/cli.py` (argparse subcommand registration + a new `_run_migrate_judge_scope`)
- Test: `elspeth-lints/tests/core/test_cli_migrate_judge_scope.py`

- [ ] **Step 1: Write the failing test**

```python
def test_migrate_rewrites_valid_v1_entry_as_v2(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", "x" * 32)
    # 1. Build a source tree with an R6 suppression and a *valid* v1 signed entry
    #    (file_fingerprint matches live bytes).
    # 2. Run _run_migrate_judge_scope(args) over it.
    # 3. Assert the YAML entry is now v2: judge_signature_version: 2, scope_fingerprint present,
    #    file_fingerprint absent, signature hmac-sha256:v2:, verdict/rationale/recorded_at unchanged.
    # 4. Assert the migrated allowlist reloads clean with source_root and the finding still matches.
    ...


def test_migrate_refuses_and_reports_already_stale_v1_entry(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", "x" * 32)
    # A v1 entry whose suppressed node changed (no live finding matches its key).
    # Assert: the entry is left untouched (still v1) and the command's report
    # lists it under "refused / needs re-justify", and the exit code is non-zero.
    ...
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_cli_migrate_judge_scope.py -q`
Expected: FAIL — command does not exist.

- [ ] **Step 3: Implement the command**

Add a subcommand registration mirroring `justify`/`reaudit` (find their `add_parser` blocks with `grep -n "add_parser" src/elspeth_lints/core/cli.py`), wiring `--root`, `--allowlist-dir`, `--dry-run`, `--owner`. Implement:

```python
def _run_migrate_judge_scope(args: argparse.Namespace) -> int:
    """Migrate currently-valid v1 judge-gated entries to v2 scope binding.

    For each v1 entry whose canonical key matches a live finding (proving the
    suppressed node — and thus what the judge inspected — is unchanged), re-sign
    it as v2 using that finding's scope_fingerprint, carrying the existing
    verdict / rationale / recorded_at forward unchanged (mechanical; no judge
    re-review). v1 entries with no matching live finding are already stale and
    are refused (they need justify/reaudit against the changed source). Requires
    the operator HMAC key.
    """
    # Load WITHOUT source_root to read raw entries without the v1
    # file_fingerprint *live-source* gate firing on entries we are about to
    # migrate (byte-drift elsewhere in the file must not block migration —
    # that is the whole point). BUT loading without source_root ALSO skips the
    # HMAC signature check (allowlist.py:468 gates both on source_root), so this
    # command MUST verify each v1 signature itself before re-signing (see the
    # integrity note below) — otherwise it would launder tampering into a clean
    # v2 signature.
    # Then scan the source tree once, build a {canonical_key: finding} map,
    # and for each v1 judge-gated entry:
    #   - if key in finding_map (suppressed node unchanged) AND the entry's v1
    #     signature verifies: recompute the v2 signature with
    #     finding.scope_fingerprint and rewrite the YAML entry in place
    #     (drop file_fingerprint, add scope_fingerprint +
    #     judge_signature_version: 2 + v2 signature), carrying verdict /
    #     rationale / recorded_at forward unchanged.
    #   - else: append to `refused` with the reason (no matching finding =
    #     stale → re-justify; signature mismatch = tampering → STOP), leave
    #     untouched.
    # Emit a report (migrated count, refused list with reasons). Return 1 if any
    # refused, else 0. In --dry-run, compute and report but do not write.
    ...
```

> Implementation notes for the engineer:
> - Reuse `_scan_single_file_findings` / the tier_model scanner the justify path uses to build the finding map; key each finding by its `canonical_key`.
> - **Two independent gates, do not conflate them (advisor fix A):**
>   1. **Integrity gate (always):** before re-signing, call `_verify_judge_metadata_signature_at_load(entry, context=...)` so the entry's existing v1 signature is verified with the operator key. This is the property §6 promises to keep — without it, a key-holder running migrate would mint a clean v2 signature over content whose v1 signature was never checked, laundering any keyless tamper (e.g. a hand-flipped verdict with a recomputed, publicly-computable `file_fingerprint`). A signature mismatch here is **tampering → refuse and STOP**, not a routine stale entry.
>   2. **Relevance gate:** does the entry's `canonical_key` match a live finding? Yes → the suppressed node is unchanged → migrate. No → already stale → refuse with "re-justify required."
> - **Do NOT gate on byte-freshness** (`file_fingerprint == live bytes`). That would refuse exactly the byte-drifted-but-scope-stable entries migration exists to relieve. `signature_version in (None, 1)` selects v1 entries; the integrity gate proves they weren't tampered; the relevance gate proves the node is unchanged. Byte equality is irrelevant.
> - Rewrite YAML by locating the entry block by `key:` line and replacing its `file_fingerprint:` line with the three v2 lines, recomputing the signature. Reuse `_build_yaml_entry_text` where practical by reconstructing the entry's fields, OR do a targeted line surgery — pick whichever keeps the rest of the entry byte-identical (prefer surgery to avoid reflowing rationale block scalars). Add a test asserting the non-binding fields are byte-identical after migration.
> - This command writes signed metadata, so it carries the same custody constraint as `justify`: it calls `_judge_metadata_hmac_key()` and fails without the key. Document in `--help` that it is operator-only.

- [ ] **Step 4: Run the migrate tests**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_cli_migrate_judge_scope.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/cli.py elspeth-lints/tests/core/test_cli_migrate_judge_scope.py
git commit -m "feat(cli): migrate-judge-scope command (v1→v2 re-sign of valid entries)"
```

---

## Task 11: Full-suite green-gate + CLAUDE.md / skill doc updates

**Files:**
- Modify: `CLAUDE.md` (the "CICD Judge Gate" section references; the dev-commands block if it names `file_fingerprint`)
- Modify: any `tier-model-deep-dive` / `engine-patterns-reference` skill text describing the binding primitives
- Modify: `docs/elspeth-lints/rationale.md` if it documents `file_fingerprint`

- [ ] **Step 1: Run the entire elspeth-lints test suite**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest -q`
Expected: PASS (full suite). Investigate and fix any v1-expectation tests that need to name their version explicitly (the locked-in-expectations pattern).

- [ ] **Step 2: Run the tier-model gate against the real source tree (keyless shape-only — agent context)**

Run:
```bash
cd /home/john/elspeth && env PYTHONPATH=elspeth-lints/src ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing \
  .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
```
Expected: the 221 live v1 entries load + verify-shape clean under the retained v1 path; gate result unchanged from before this work (the v2 code is dormant until migration). The one prominent shape-only downgrade warning is expected.

- [ ] **Step 3: mypy + ruff across the touched modules**

Run:
```bash
cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m mypy src/elspeth_lints && ../.venv/bin/python -m ruff check src/elspeth_lints
```
Expected: no errors.

- [ ] **Step 4: Update docs**

Update CLAUDE.md and the relevant skill text so the binding primitives are described as `scope_fingerprint` (v2) with `file_fingerprint` (v1) noted as the migrating-away legacy. Do not delete the v1 description yet — it is still live until Task 13.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: describe v2 scope_fingerprint binding (v1 file_fingerprint migrating out)"
```

> **End of agent-buildable work.** The repo now has v2 code alongside v1, all 221 entries still v1, all gates green. The remaining two tasks require the operator HMAC key.

---

## Task 12 — OPERATOR ONLY: run the migration

> **Requires `ELSPETH_JUDGE_METADATA_HMAC_KEY` (operator-held). An agent must NOT execute this.** Do it in a worktree, with the key in the environment for the duration of the session only.

- [ ] **Step 1: Dry-run the migration to preview the partition**

```bash
cd /home/john/elspeth && env ELSPETH_JUDGE_METADATA_HMAC_KEY=<key> PYTHONPATH=elspeth-lints/src \
  .venv/bin/python -m elspeth_lints.core.cli migrate-judge-scope \
  --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --dry-run
```
Expected: a report of how many entries migrate vs. are refused (stale). The refused set should include the 7 entries from `notes/F1-F2-merge-handoff-2026-05-31.md` if those are still unresolved.

- [ ] **Step 2: Apply the migration**

Re-run without `--dry-run`. The valid v1 entries are rewritten as v2 in `config/cicd/enforce_tier_model/*.yaml`.

- [ ] **Step 3: Resolve the refused (stale) entries**

For each refused entry, `reaudit` then `justify` (or `justify` directly) against the current source — these need genuine re-review because their suppressed code or scope actually changed. This subsumes the 7 entries the F1/F2 merge hand-off already flagged.

- [ ] **Step 4: Verify with the key present**

```bash
cd /home/john/elspeth && env ELSPETH_JUDGE_METADATA_HMAC_KEY=<key> PYTHONPATH=elspeth-lints/src \
  .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
```
Expected: all entries now v2, full HMAC verification passes (no shape-only downgrade).

- [ ] **Step 5: Commit the migrated allowlists**

```bash
git add config/cicd/enforce_tier_model/
git commit -m "chore(cicd): migrate judge-gated tier-model entries to v2 scope binding"
```

---

## Task 13 — OPERATOR ONLY: delete the v1 path (No Legacy Code)

> Only after Task 12 leaves **zero** v1 entries on disk. This satisfies the No Legacy Code policy: the dual-version state was a transient migration scaffold, not a kept compatibility shim.

- [ ] **Step 0: DECISION — the redaction salt (advisor note C; resolve before deleting the field)**

`source_excerpt.py:757` salts `RedactionRecord.redacted_hash` with the whole-file `file_fingerprint`, and its comments (:122–129) state that verifying a persisted redaction record needs **both** the `redacted_hash` AND the `file_fingerprint`. Deleting `file_fingerprint` from `AllowlistEntry` drops, for any entry that carries `judge_excerpt_redactions`, the salt an auditor needs to verify those records. This is rare (only when the judge prompt had secrets scrubbed) and was considered by neither the spec nor the original plan.

Decide consciously, do not delete blindly:
- **(a) Accept the loss:** if no live entry carries `judge_excerpt_redactions` (check: `grep -rl "judge_excerpt_redactions" config/cicd/enforce_tier_model/`), the salt was never load-bearing on disk; document that and proceed.
- **(b) Retain the salt:** keep the whole-file digest on redaction-bearing entries under a dedicated, **unsigned**, clearly-named field (e.g. `redaction_salt`) so the binding semantics (now scope-scoped) and the audit-verification semantics (still whole-file) are not conflated. This is *not* the binding primitive — it must not re-introduce the re-sign tax.

Record the choice in the commit message. The rest of Task 13 assumes (a) unless the operator selects (b).

- [ ] **Step 1: Delete the v1 field + payload branch + byte-hash load path**

Remove from `core/allowlist.py`: the `file_fingerprint` field on `AllowlistEntry`; the v1 branch in `compute_judge_metadata_signature` (and the `signature_version`/`file_fingerprint` params — v2 becomes the only path); `_compute_file_fingerprint`; the v1 byte-hash branch in `_verify_source_binding_at_load`; the v1 prefix constant and the dual-prefix shape acceptance; the v1 branches in `_validate_judge_metadata_atomic` invariant 8 and the version dispatch in `_verify_judge_metadata_signature_at_load`. Make `judge_signature_version` required (must be 2) for judge-gated entries.

Remove `file_fingerprint` from: `_build_yaml_entry_text`, the justify call site, `judge_coverage._judge_binding_identity`/`_judge_metadata_payload`, `reaudit_sidecar` round-trip, `rotate.py` raw read and `_REJUDGE_REQUIRED_FIELDS`. The `migrate-judge-scope` command itself can be retained (harmless, idempotent) or removed — operator's choice; if removed, delete its tests too.

- [ ] **Step 2: Delete or update the v1-specific tests**

The v1-path tests written in Tasks 4–6 (explicit `signature_version=1`, v1 byte-drift crash) now test deleted code. Delete them. Keep the v2 tests.

- [ ] **Step 3: Full suite + gate + types + lint**

```bash
cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest -q && ../.venv/bin/python -m mypy src/elspeth_lints && ../.venv/bin/python -m ruff check src/elspeth_lints
cd /home/john/elspeth && env ELSPETH_JUDGE_METADATA_HMAC_KEY=<key> PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
grep -rn "file_fingerprint" elspeth-lints/src/  # expect: only source_excerpt.py's scrubber-salt usage remains
```
Expected: green suite, green gate, and `file_fingerprint` surviving **only** as the `source_excerpt` scrubber salt (which was never the binding primitive — confirm each remaining hit is salt-related, not binding-related).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(allowlist): delete v1 file_fingerprint binding (all entries migrated to v2)"
```

---

## Self-Review

**Spec coverage** (against `2026-05-31-judge-scope-fingerprint-design.md`):
- §3 `scope_fingerprint` replaces `file_fingerprint` → Tasks 1–7. ✓
- §3.1 determinism rules (innermost scope; module fallback; name/args/decorator/body in; docstring out; no ast_path prefix; rename sensitive) → Task 1 (frozen as the "Determinism Rules" section + tests). ✓
- §3.2 verification at match time, file-exists at load, confirm coverage vs `_verify_file_fingerprint_at_load` → Tasks 5, 6, 9 (reaudit gap closed). ✓
- §4.1 versioning self-protecting inside payload → Task 4 (`"version"` in signed payload + `judge_signature_version` stored field). ✓
- §4.2 eager-atomic-mechanical migration → Task 10 (command) + Task 12 (operator run); the matched/stale split = valid/refused. ✓
- §5 relief calibration / §6 security delta → encoded in behaviour (match-time fires on enclosing-scope change only); no code task, documented in §3.2/Task 11 docs. ✓
- §7 affected code → Tasks map 1:1 to the listed files. ✓
- §8 testing (determinism, fallback, version self-protection, migrate valid/stale, deletion leaves no refs) → Tasks 1, 4, 10, 13. ✓
- §9 separate tracks (keyless RC5.2 debt; today's 7; ast_path Phase 2) → explicitly out of scope; today's 7 surface naturally as Task 12 refusals. ✓

**Advisor lock-ins (pre-write):** #1 single `_scope_fingerprint`, justify-from-scan, migrate-by-matcher → Tasks 1/7/10. #2 reaudit gap → Task 9. #3 stamp every Finding site + v2 rejects empty → Tasks 2/6. #4 operator boundary hard line + flagged dual-version deviation → Operator Boundary section + Tasks 12/13. ✓

**Advisor pass (post-write):** **A** — migrate must verify the v1 signature before re-signing (else it launders tampering into a clean v2 sig) and must NOT gate on byte-freshness → Task 10 integrity/relevance gates rewritten. **B** — `compute_judge_metadata_signature` defaults to `signature_version=1` during the dual-version window so justify stays green across Tasks 4–6 → Task 4 Step 3/Step 6. **C** — deleting `file_fingerprint` drops the redaction-record salt for entries with `judge_excerpt_redactions` → Task 13 Step 0 decision gate. ✓

**Type consistency:** `compute_scope_fingerprint(scope_node, *, module=None)`, `enclosing_scope_node(ancestors)`, `verify_entry_binding_against_finding(entry, *, file_path, ast_path, scope_fingerprint)`, `compute_judge_metadata_signature(..., signature_version=2, file_fingerprint=None, scope_fingerprint=None)`, `Finding.scope_fingerprint`, `AllowlistEntry.scope_fingerprint`/`.judge_signature_version` — names used identically across all tasks. ✓

**Known soft spots the engineer must resolve in-task (flagged inline, not placeholders):** exact test filenames under `tests/core/` (confirm with `ls`); the public scan entry-point name in Task 2; the layer-import finding's module availability; the reaudit per-entry finding-match shape; trust_boundary `Finding` attribute presence. Each has a concrete `grep` to run and a stated fallback.
