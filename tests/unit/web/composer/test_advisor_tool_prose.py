"""Phase P5.9 — the advisor tool description no longer claims 'Disabled by default'."""

from __future__ import annotations

from elspeth.web.composer.tools._dispatch import _REQUEST_ADVISOR_HINT_DEFINITION


def test_advisor_tool_prose_not_stale() -> None:
    description = _REQUEST_ADVISOR_HINT_DEFINITION["description"]
    assert isinstance(description, str)
    assert "Disabled by default" not in description
    # The mandatory END sign-off is profile-gated and runs independently of the
    # on-demand escape budget; the corrected prose says so.
    assert "operator-configured" in description
