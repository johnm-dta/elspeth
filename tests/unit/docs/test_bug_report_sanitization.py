"""Documentation must not encourage users to disclose raw composer history."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _doc_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _single_line(text: str) -> str:
    return " ".join(text.split())


def test_guided_mode_bug_reports_do_not_request_raw_chat_history() -> None:
    troubleshooting = _single_line(_doc_text("docs/guides/troubleshooting.md"))

    assert "chat history attached" not in troubleshooting
    assert "sanitized reproduction" in troubleshooting
    assert "Do not attach raw composer chat history" in troubleshooting


def test_user_manual_bug_report_pointer_requires_sanitization() -> None:
    user_manual = _single_line(_doc_text("docs/guides/user-manual.md"))

    assert "Do not attach raw composer chat history" in user_manual
    assert "secrets, tokens, PII, blob contents" in user_manual
