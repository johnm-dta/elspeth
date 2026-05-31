"""Tests for the tutorial-run flat-file cache."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretBytes, ValidationError

from elspeth.web.preferences.tutorial_cache import (
    CANONICAL_SEED_PROMPT,
    TutorialCache,
    TutorialCacheCorruptError,
    TutorialCacheEntry,
)

_CANONICAL_PIPELINE_YAML = """\
source:
  type: inline_blob
  rows:
    - url: ato.gov.au
transforms:
  - type: web_scrape
  - type: llm_rate
sink:
  type: tutorial_summary
"""


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tutorial_cache"
    d.mkdir()
    return d


@pytest.fixture
def cache(cache_dir: Path) -> TutorialCache:
    return TutorialCache(cache_dir=cache_dir)


def test_canonical_seed_prompt_constant_is_exact() -> None:
    """The seed prompt must match the tutorial canonical prompt verbatim."""
    assert CANONICAL_SEED_PROMPT == (
        "Create a data source with URLs for five public government agency web pages "
        "that you choose. Use abuse contact noreply@dta.gov.au and "
        "scraping reason 'DTA technical demonstration'. Read the HTML for each "
        "page, have an LLM identify the primary colours for each government agency. "
        "Remove the HTML and save the rest to a json file."
    )


def test_lookup_returns_none_on_miss(cache: TutorialCache) -> None:
    assert cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7") is None


def test_lookup_returns_entry_on_hit(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[{"url": "ato.gov.au", "score": 5, "rationale": "clear nav"}],
        source_data_hash="a7f3e2deadbeef",
        llm_call_count=5,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )
    cache.store(entry)

    got = cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")

    assert got is not None
    assert got.canonical_prompt == CANONICAL_SEED_PROMPT
    assert got.model_id == "claude-opus-4-7"
    assert got.source_data_hash == "a7f3e2deadbeef"
    assert got.llm_call_count == 5
    assert got.rows[0]["url"] == "ato.gov.au"
    assert got.pipeline_yaml == _CANONICAL_PIPELINE_YAML


def test_entry_rejects_identity_fields() -> None:
    """Foreign run/session identity must not enter the shared cache."""
    with pytest.raises(ValidationError):
        TutorialCacheEntry.model_validate(
            {
                "canonical_prompt": CANONICAL_SEED_PROMPT,
                "model_id": "claude-opus-4-7",
                "cached_at": "2026-05-15T00:00:00+00:00",
                "rows": [],
                "source_data_hash": "hash",
                "llm_call_count": 0,
                "pipeline_yaml": _CANONICAL_PIPELINE_YAML,
                "run_id": "abc-123",
            }
        )


def test_lookup_misses_on_different_model(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[],
        source_data_hash="hash",
        llm_call_count=0,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )
    cache.store(entry)

    assert cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-8") is None


def test_lookup_misses_on_different_prompt(cache: TutorialCache) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[],
        source_data_hash="hash",
        llm_call_count=0,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )
    cache.store(entry)

    edited = CANONICAL_SEED_PROMPT + " and also rate accessibility"
    assert cache.lookup(edited, "claude-opus-4-7") is None


def test_store_and_lookup_round_trip(cache: TutorialCache, cache_dir: Path) -> None:
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[{"url": "example.gov.au", "score": 3}],
        source_data_hash="hash",
        llm_call_count=5,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )
    cache.store(entry)

    files = list(cache_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"
    raw = json.loads(files[0].read_text(encoding="utf-8"))
    assert raw["canonical_prompt"] == CANONICAL_SEED_PROMPT
    assert raw["model_id"] == "claude-opus-4-7"
    assert "run_id" not in raw
    assert "session_id" not in raw
    assert "interpretation_event_id" not in raw


def test_corrupt_file_crashes_lookup(cache: TutorialCache, cache_dir: Path) -> None:
    """A present-but-unparseable file is a fault, not a miss."""
    from elspeth.web.preferences.tutorial_cache import _compute_key

    key = _compute_key(CANONICAL_SEED_PROMPT, "claude-opus-4-7")
    (cache_dir / f"{key}.json").write_text("this is not json", encoding="utf-8")

    with pytest.raises(TutorialCacheCorruptError, match="not valid JSON"):
        cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")


def test_file_with_mismatched_prompt_crashes_lookup(cache: TutorialCache, cache_dir: Path) -> None:
    """A file whose contents disagree with its key is corrupt."""
    from elspeth.web.preferences.tutorial_cache import _compute_key

    key = _compute_key(CANONICAL_SEED_PROMPT, "claude-opus-4-7")
    bad_entry = {
        "canonical_prompt": "a different prompt",
        "model_id": "claude-opus-4-7",
        "cached_at": "2026-05-15T00:00:00+00:00",
        "rows": [],
        "source_data_hash": "hash",
        "llm_call_count": 0,
        "pipeline_yaml": _CANONICAL_PIPELINE_YAML,
    }
    (cache_dir / f"{key}.json").write_text(json.dumps(bad_entry), encoding="utf-8")

    with pytest.raises(TutorialCacheCorruptError, match="prompt mismatch"):
        cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7")


def test_store_leaves_exactly_one_json_file(cache: TutorialCache, cache_dir: Path) -> None:
    """Successful store leaves no observable tempfiles."""
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[{"url": "example.gov.au"}],
        source_data_hash="hash",
        llm_call_count=0,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )
    cache.store(entry)

    files = list(cache_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"


def test_store_atomic_under_oserror(cache: TutorialCache, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If rename fails mid-write, no .json cache entry is left behind."""
    entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="claude-opus-4-7",
        cached_at=datetime(2026, 5, 15, tzinfo=UTC),
        rows=[{"url": "example.gov.au"}],
        source_data_hash="hash",
        llm_call_count=0,
        pipeline_yaml=_CANONICAL_PIPELINE_YAML,
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("os.replace", boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        cache.store(entry)

    assert list(cache_dir.glob("*.json")) == []


def test_tutorial_cache_dir_defaults_to_data_dir_subdir(tmp_path: Path) -> None:
    """Default tutorial_cache_dir lives under data_dir, no extra env var required."""
    from elspeth.web.config import WebSettings

    settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=SecretBytes(b"\x00" * 32),
    )

    assert settings.tutorial_cache_dir == tmp_path / "tutorial_cache"
    cache_dir = settings.tutorial_cache_dir
    assert cache_dir is not None
    cache = TutorialCache(cache_dir=cache_dir)
    assert cache.lookup(CANONICAL_SEED_PROMPT, "claude-opus-4-7") is None
