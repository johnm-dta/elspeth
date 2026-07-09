"""Session default-title minting and recognition (sessions/titles.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.web.sessions.titles import (
    abandoned_tutorial_session_title,
    format_default_session_title,
    is_default_session_title,
    mint_default_session_title,
)

_JUL_2 = datetime(2026, 7, 2, 10, 30, tzinfo=UTC)


class TestFormatDefaultSessionTitle:
    def test_format(self) -> None:
        assert format_default_session_title(_JUL_2) == "Session — 2 Jul 2026"

    def test_no_zero_padded_day(self) -> None:
        assert format_default_session_title(datetime(2026, 12, 9, tzinfo=UTC)) == "Session — 9 Dec 2026"


class TestMintDefaultSessionTitle:
    def test_first_of_the_day_is_undisambiguated(self) -> None:
        assert mint_default_session_title(_JUL_2, []) == "Session — 2 Jul 2026"

    def test_collision_appends_counter(self) -> None:
        existing = ["Session — 2 Jul 2026"]
        assert mint_default_session_title(_JUL_2, existing) == "Session — 2 Jul 2026 (2)"

    def test_counter_skips_taken_variants(self) -> None:
        existing = [
            "Session — 2 Jul 2026",
            "Session — 2 Jul 2026 (2)",
            "Session — 2 Jul 2026 (3)",
        ]
        assert mint_default_session_title(_JUL_2, existing) == "Session — 2 Jul 2026 (4)"

    def test_gap_in_counters_is_reused(self) -> None:
        existing = ["Session — 2 Jul 2026", "Session — 2 Jul 2026 (3)"]
        assert mint_default_session_title(_JUL_2, existing) == "Session — 2 Jul 2026 (2)"

    def test_other_titles_do_not_collide(self) -> None:
        existing = ["Weather Data Enrichment", "Session — 1 Jul 2026"]
        assert mint_default_session_title(_JUL_2, existing) == "Session — 2 Jul 2026"


class TestIsDefaultSessionTitle:
    @pytest.mark.parametrize(
        "title",
        [
            "Session — 2 Jul 2026",
            "Session — 2 Jul 2026 (2)",
            "Session — 31 Dec 2027 (14)",
            # Legacy defaults from the pre-convention frontend/backends.
            "New session",
            "Untitled",
        ],
    )
    def test_defaults_recognised(self, title: str) -> None:
        assert is_default_session_title(title)

    @pytest.mark.parametrize(
        "title",
        [
            "Weather Data Enrichment",
            "Session — 2 Jul 2026 extras",
            "prefix Session — 2 Jul 2026",
            "Session — 2 Julember 2026",
            "First-run tutorial",
            "",
        ],
    )
    def test_user_titles_not_recognised(self, title: str) -> None:
        assert not is_default_session_title(title)

    def test_minted_titles_round_trip(self) -> None:
        minted = mint_default_session_title(_JUL_2, ["Session — 2 Jul 2026"])
        assert is_default_session_title(minted)


class TestAbandonedTutorialSessionTitle:
    def test_human_register(self) -> None:
        assert abandoned_tutorial_session_title(_JUL_2) == "First-run tutorial (abandoned 2 Jul 2026)"

    def test_not_a_default_title(self) -> None:
        # The abandoned title must not be eligible for content-derived
        # auto-titling — it is a terminal, human-facing marker.
        assert not is_default_session_title(abandoned_tutorial_session_title(_JUL_2))
