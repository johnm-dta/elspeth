"""p1 Task 5 — a driver-produced URL-row source routes web_scrape into transforms."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_1_source_chat
from elspeth.web.composer.guided.recipe_match import match_recipe
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved


def _fake_url_source_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="resolve_source",
                                arguments=json.dumps(
                                    {
                                        "resolution": "source",
                                        "plugin": "json",
                                        "filename": "urls.json",
                                        "mime_type": "application/json",
                                        "content": json.dumps(
                                            [
                                                {"url": "https://example.test/project-1.html"},
                                                {"url": "https://example.test/project-2.html"},
                                            ]
                                        ),
                                        "options": {"schema": {"mode": "observed"}},
                                        "observed_columns": ["url"],
                                        "sample_rows": [{"url": "https://example.test/project-1.html"}],
                                        "assistant_message": "I set up a URL list source.",
                                    }
                                ),
                            )
                        )
                    ],
                )
            )
        ]
    )


@pytest.mark.asyncio
async def test_url_source_driver_output_matches_web_scrape_recipe() -> None:
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=_fake_url_source_response()),
    ):
        resolution = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="scrape these project pages and pull out the name and top risk",
            plugin_hint="json",
            current_source=None,
            temperature=None,
            seed=None,
        )
    assert resolution is not None
    assert resolution.plugin == "json"
    assert "url" in resolution.observed_columns

    # The driver's output, once blob-backed, is the URL-row source the
    # web_scrape recipe routes. _web_scrape_predicate keys on the json/csv
    # plugin + an observed `url` column + blob_ref in options + a single
    # json sink. Simulate the post-blob source shape:
    source = SourceResolved(
        plugin=resolution.plugin,
        options={**dict(resolution.options), "blob_ref": "blob-123"},
        observed_columns=resolution.observed_columns,
        sample_rows=resolution.sample_rows,
    )
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={"path": "out.jsonl"},
                required_fields=(),
                schema_mode="observed",
            ),
        )
    )
    match = match_recipe(source, sink)
    assert match is not None
    assert match.recipe_name == "web-scrape-llm-rate-jsonl"
