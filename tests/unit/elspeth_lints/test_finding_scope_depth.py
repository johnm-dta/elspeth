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
