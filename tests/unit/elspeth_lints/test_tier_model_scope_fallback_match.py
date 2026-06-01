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
    Finding,
    _match_finding,
    scan_directory,
)

BEFORE = "def handler(payload):\n    return payload.get('missing')\n"
AFTER = "from x import trust_boundary\n\ndef handler(payload):\n    return payload.get('missing')\n"


def _r1(findings: list[Finding]) -> Finding:
    r1 = [f for f in findings if f.rule_id == "R1"]
    assert r1
    return r1[0]


def _entry_from(finding: Finding) -> AllowlistEntry:
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

    # Reset so the post-fallback `matched is True` assertion below actually
    # tests that the FALLBACK path sets it (the sanity match above set it True).
    allowlist.entries[0].matched = False

    # Insert a module-level import -> every downstream ast_path shifts.
    src.write_text(AFTER)
    after = _r1(scan_directory(tmp_path))
    assert after.canonical_key != before.canonical_key  # exact key drifted
    assert after.scope_fingerprint == before.scope_fingerprint  # the scope is unchanged — this is why the fallback fires

    matched = _match_finding(allowlist, after)
    assert matched is allowlist.entries[0]  # rescued by the fallback
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
