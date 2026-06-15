from pathlib import Path

HARNESS_PLAN = Path("docs/superpowers/plans/2026-06-06-tutorial-reliability-harness.md")


def test_tutorial_reliability_harness_plan_does_not_embed_staging_credentials() -> None:
    text = HARNESS_PLAN.read_text(encoding="utf-8")

    assert "dta_user" not in text
    assert "dta_pass" not in text
