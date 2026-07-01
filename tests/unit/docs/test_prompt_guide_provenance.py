"""Regression checks for prompt-guide citation/provenance hygiene."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_GUIDE = REPO_ROOT / "docs-archive" / "2026-06-28-docs-cleanout" / "docs" / "assets" / "prompt-guide.md"


def test_prompt_guide_has_no_transcript_citation_placeholders() -> None:
    text = PROMPT_GUIDE.read_text(encoding="utf-8")

    assert "filecite" not in text
    assert "cite" not in text
    assert "turn0file" not in text
    assert "turn7view" not in text


def test_prompt_guide_records_provenance_and_bibliography() -> None:
    text = PROMPT_GUIDE.read_text(encoding="utf-8")

    assert "## Provenance, Date, And Scope" in text
    assert "Source prompt reviewed: `src/elspeth/web/composer/skills/pipeline_composer.md`" in text
    assert "Review artifact remediated: 2026-05-20" in text
    assert "## Bibliography" in text
    assert "https://platform.openai.com/docs/guides/prompt-engineering" in text
    assert "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview" in text
    assert "https://ai.google.dev/gemini-api/docs/long-context" in text
