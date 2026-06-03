"""Match-time binding verification for ``verify_entry_binding_against_finding``.

Covers the v1/v2 split of the C8-3 in-file transplant defence: v1 entries bind
via ``ast_path`` only at match time (whole-file ``file_fingerprint`` is a
load-time concern), and v2 entries additionally verify the enclosing-scope
``scope_fingerprint`` the judge inspected. v2 must reject an empty (un-stamped)
live ``scope_fingerprint`` rather than let it silently pass.
"""

import pytest

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    JudgeVerdict,
    verify_entry_binding_against_finding,
)


def _v2_entry(*, scope_fingerprint: str, ast_path: str) -> AllowlistEntry:
    # Minimal v2 judge-gated entry. No signature needed:
    # verify_entry_binding_against_finding does NOT check the HMAC (that is the
    # load-time path). Build only what the matcher reads.
    return AllowlistEntry(
        key="core/x.py:R6:C:m",
        owner="t",
        reason="r",
        safety="s",
        expires=None,
        ast_path=ast_path,
        scope_fingerprint=scope_fingerprint,
        judge_signature_version=2,
        judge_verdict=JudgeVerdict.ACCEPTED,
    )


def _v1_entry(*, file_fingerprint: str, ast_path: str) -> AllowlistEntry:
    return AllowlistEntry(
        key="core/x.py:R6:C:m",
        owner="t",
        reason="r",
        safety="s",
        expires=None,
        ast_path=ast_path,
        file_fingerprint=file_fingerprint,
        judge_verdict=JudgeVerdict.ACCEPTED,
    )


def test_v2_match_passes_when_scope_fingerprint_matches() -> None:
    entry = _v2_entry(scope_fingerprint="a" * 64, ast_path="body[0]/body[0]")
    verify_entry_binding_against_finding(
        entry,
        file_path="core/x.py",
        ast_path="body[0]/body[0]",
        scope_fingerprint="a" * 64,
    )  # no raise


def test_v2_match_crashes_on_scope_drift() -> None:
    entry = _v2_entry(scope_fingerprint="a" * 64, ast_path="body[0]/body[0]")
    with pytest.raises(ValueError, match="scope_fingerprint"):
        verify_entry_binding_against_finding(
            entry,
            file_path="core/x.py",
            ast_path="body[0]/body[0]",
            scope_fingerprint="b" * 64,
        )


def test_v2_match_crashes_on_empty_finding_scope_fingerprint() -> None:
    # advisor lock #3: an un-stamped finding (default "") must NOT silently pass
    # a v2 binding.
    entry = _v2_entry(scope_fingerprint="a" * 64, ast_path="body[0]/body[0]")
    with pytest.raises(ValueError, match="scope_fingerprint"):
        verify_entry_binding_against_finding(
            entry,
            file_path="core/x.py",
            ast_path="body[0]/body[0]",
            scope_fingerprint="",
        )


def test_v1_match_ignores_scope_fingerprint_and_checks_ast_path() -> None:
    entry = _v1_entry(file_fingerprint="b" * 64, ast_path="body[0]/body[0]")
    # v1 has no scope binding at match time; ast_path is enforced, scope ignored.
    verify_entry_binding_against_finding(
        entry,
        file_path="core/x.py",
        ast_path="body[0]/body[0]",
        scope_fingerprint="anything",
    )  # no raise
    with pytest.raises(ValueError, match="ast_path mismatch"):
        verify_entry_binding_against_finding(
            entry,
            file_path="core/x.py",
            ast_path="body[9]",
            scope_fingerprint="",
        )
