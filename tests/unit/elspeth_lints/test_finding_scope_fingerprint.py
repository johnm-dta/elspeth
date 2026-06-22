# test_finding_scope_fingerprint.py
"""Every tier-model ``Finding`` must carry the enclosing-scope fingerprint.

These tests drive the real scanner end-to-end (not a re-implemented
producer): they assert that the value the visitor stamps onto a finding
equals the fingerprint computed forward from the same enclosing scope. The
v2 match-verifier later rejects any judge-gated entry whose live finding
carries an empty ``scope_fingerprint``, so this is the highest-stakes
correctness property of the feature.
"""

import ast
import hashlib
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
    handler = ast.parse(src).body[0]
    assert isinstance(handler, ast.FunctionDef)
    expected = compute_scope_fingerprint(handler)
    assert r1[0].scope_fingerprint == expected


def test_finding_carries_file_fingerprint_of_scanned_bytes(tmp_path: Path) -> None:
    src = "def handler(payload):\n    return payload.get('missing')\n"
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")
    findings = scan_directory(tmp_path)
    r1 = [x for x in findings if x.rule_id == "R1"]
    assert r1, "expected an R1 dict.get finding"
    assert r1[0].file_fingerprint == hashlib.sha256(src.encode("utf-8")).hexdigest()


def test_module_level_finding_uses_module_fallback(tmp_path: Path) -> None:
    src = "import os\nVALUE = os.environ.get('X')\n"
    f = tmp_path / "mod.py"
    f.write_text(src)
    findings = scan_directory(tmp_path)
    r1 = [x for x in findings if x.rule_id == "R1"]
    assert r1
    expected = compute_scope_fingerprint(None, module=ast.parse(src))
    assert r1[0].scope_fingerprint == expected
