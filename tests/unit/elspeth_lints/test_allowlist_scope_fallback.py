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
    a = _v2(key="m.py:R6:C:f:fp=a", ast_path="body[0]/body[0]/value")
    b = _v2(key="m.py:R6:C:f:fp=b", ast_path="body[0]/body[1]/value")
    entries = [a, b]
    got_a = find_scope_fallback_entry(
        entries,
        canonical_key="m.py:R6:C:f:fp=x",
        scope_fingerprint=SCOPE,
        ast_path="body[1]/body[0]/value",
        scope_depth=1,
    )
    got_b = find_scope_fallback_entry(
        entries,
        canonical_key="m.py:R6:C:f:fp=y",
        scope_fingerprint=SCOPE,
        ast_path="body[1]/body[1]/value",
        scope_depth=1,
    )
    assert got_a is a
    assert got_b is b


def test_scope_body_edit_fails_closed() -> None:
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry],
        canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint="c" * 64,
        ast_path="body[1]/body[0]/value",
        scope_depth=1,
    )
    assert got is None


def test_depth_change_fails_closed() -> None:
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry],
        canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint=SCOPE,
        ast_path="body[0]/body[0]/body[0]/value",
        scope_depth=2,
    )
    assert got is None


def test_within_scope_transplant_fails_closed() -> None:
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry],
        canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint=SCOPE,
        ast_path="body[1]/body[7]/value",
        scope_depth=1,
    )
    assert got is None


def test_ambiguity_fails_closed() -> None:
    a = _v2(key="m.py:R6:C:f:fp=a", ast_path="body[0]/body[0]/value")
    b = _v2(key="m.py:R6:C:f:fp=b", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [a, b],
        canonical_key="m.py:R6:C:f:fp=x",
        scope_fingerprint=SCOPE,
        ast_path="body[1]/body[0]/value",
        scope_depth=1,
    )
    assert got is None


def test_v1_entry_skipped() -> None:
    entry = _v1(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry],
        canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint=SCOPE,
        ast_path="body[1]/body[0]/value",
        scope_depth=1,
    )
    assert got is None


def test_empty_finding_scope_fingerprint_skipped() -> None:
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry],
        canonical_key="m.py:R6:C:f:fp=new",
        scope_fingerprint="",
        ast_path="body[1]/body[0]/value",
        scope_depth=1,
    )
    assert got is None


def test_different_symbol_skipped() -> None:
    entry = _v2(key="m.py:R6:C:f:fp=old", ast_path="body[0]/body[0]/value")
    got = find_scope_fallback_entry(
        [entry],
        canonical_key="m.py:R6:C:g:fp=new",
        scope_fingerprint=SCOPE,
        ast_path="body[1]/body[0]/value",
        scope_depth=1,
    )
    assert got is None
