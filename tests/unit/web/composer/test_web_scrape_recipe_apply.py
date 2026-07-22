"""End-to-end apply_pipeline_recipe proof for the web-scrape recipe (D11/P4.3).

Drives the canonical web-scrape recipe through the public
``execute_tool('apply_pipeline_recipe', ...)`` dispatcher with a seeded
URL-row blob and ``create_catalog_service()`` (the real catalog), for both a
JSON and a CSV URL source, and asserts the resulting state preserves the
re-polarized shield polarity:

  * NO unbuildable azure_prompt_shield hard node, AND
  * the medium-severity prompt-shield ADVISORY survives the real apply path.

Apply is compose-time only: no real fetch/LLM round-trip occurs (consistent
with the P4.4 zero-LLM gate). This proves the JSON/CSV resolver path and that
set_pipeline prevalidation accepts the web_scrape + llm + field_mapper options.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.blobs.service import content_hash as _content_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import execute_tool
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.interpretation_state import prompt_shield_recommendation_warning_pairs
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema

_BASE_SLOTS = {
    "profile": "tutorial-default",
    "abuse_contact": "web-scrape-contact@dta.gov.au",
    "scraping_reason": "Tutorial exercise: fetch public pages for rating",
    "output_path": "outputs/ratings.jsonl",
}


class _NoSecrets:
    def has_server_ref(self, name: str) -> bool:
        return False

    def has_user_ref(self, principal: str, name: str) -> bool:
        return False

    def has_ref(self, principal: str, name: str) -> bool:
        return False


def _policy_context() -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    settings = WebSettings(
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        llm_profiles={
            "tutorial-default": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            }
        },
        tutorial_llm_profile="tutorial-default",
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    catalog = create_catalog_service()
    snapshot = build_plugin_snapshot(
        policy=policy,
        catalog=catalog,
        profiles=profiles,
        principal_scope="local:test-user",
        secret_inventory=_NoSecrets(),
        generation_key=b"web-scrape-recipe-test-generation-key",
    )
    return PolicyCatalogView(catalog, snapshot, profiles), snapshot


def _seed_blob(tmp_path, *, filename: str, mime_type: str, body: bytes):
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="test-user",
                auth_provider_type="local",
                title="Test",
                created_at=now,
                updated_at=now,
            )
        )
    blob_id = str(uuid4())
    storage_dir = tmp_path / "blobs" / session_id
    storage_dir.mkdir(parents=True)
    storage_path = storage_dir / f"{blob_id}_{filename}"
    storage_path.write_bytes(body)
    with engine.begin() as conn:
        conn.execute(
            blobs_table.insert().values(
                id=blob_id,
                session_id=session_id,
                filename=filename,
                mime_type=mime_type,
                size_bytes=len(body),
                content_hash=_content_hash(body),
                storage_path=str(storage_path),
                created_at=now,
                created_by="user",
                source_description=None,
                status="ready",
            )
        )
    return engine, session_id, blob_id


@pytest.fixture
def _seeded_url_blob_json(tmp_path):
    return _seed_blob(
        tmp_path,
        filename="urls.json",
        mime_type="application/json",
        body=b'[{"url": "https://www.dta.gov.au"}]',
    )


@pytest.fixture
def _seeded_url_blob_csv(tmp_path):
    return _seed_blob(
        tmp_path,
        filename="urls.csv",
        mime_type="text/csv",
        body=b"url\nhttps://www.dta.gov.au\n",
    )


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def test_apply_web_scrape_recipe_json_preserves_shield_advisory(_seeded_url_blob_json) -> None:
    engine, session_id, blob_id = _seeded_url_blob_json
    catalog, snapshot = _policy_context()
    result = execute_tool(
        "apply_pipeline_recipe",
        {
            "recipe_name": "web-scrape-llm-rate-jsonl",
            "slots": {
                "source_blob_id": blob_id,
                "source_plugin": "json",
                **_BASE_SLOTS,
            },
        },
        _empty_state(),
        catalog,
        plugin_snapshot=snapshot,
        session_engine=engine,
        session_id=session_id,
    )
    assert result.success, getattr(result, "data", result)
    state = result.updated_state
    assert state.sources["source"].plugin == "json"
    # Re-polarized shield: no hard node, advisory survives.
    assert all(node.plugin != "azure_prompt_shield" for node in state.nodes)
    warning_pairs = prompt_shield_recommendation_warning_pairs(state)
    assert warning_pairs, "prompt-shield advisory must remain warning/advisory"
    assert any(component == "node:rate_pages" for component, _ in warning_pairs)


def test_apply_web_scrape_recipe_csv_preserves_shield_advisory(_seeded_url_blob_csv) -> None:
    engine, session_id, blob_id = _seeded_url_blob_csv
    catalog, snapshot = _policy_context()
    result = execute_tool(
        "apply_pipeline_recipe",
        {
            "recipe_name": "web-scrape-llm-rate-jsonl",
            "slots": {
                "source_blob_id": blob_id,
                "source_plugin": "csv",
                **_BASE_SLOTS,
            },
        },
        _empty_state(),
        catalog,
        plugin_snapshot=snapshot,
        session_engine=engine,
        session_id=session_id,
    )
    assert result.success, getattr(result, "data", result)
    state = result.updated_state
    # The resolved source_plugin slot preserves the csv source materialisation.
    assert state.sources["source"].plugin == "csv"
    assert all(node.plugin != "azure_prompt_shield" for node in state.nodes)
    warning_pairs = prompt_shield_recommendation_warning_pairs(state)
    assert warning_pairs, "prompt-shield advisory must remain warning/advisory"
    assert any(component == "node:rate_pages" for component, _ in warning_pairs)
