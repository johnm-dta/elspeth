"""p4 Task 8a — STEP_2.5 accept-seam SSRF ``allowed_hosts`` injection (tutorial-gated).

The tutorial passive auto-drive accepts the web-scrape recipe at STEP_2.5. The
``web_scrape.allowed_hosts`` egress allowlist is an SSRF control: it must be the
server-side deterministic resolver value, NEVER a client/LLM-authored value.
These tests pin the dispatcher seam (``_dispatch_guided_respond`` accept branch):

  * TUTORIAL session + loopback base -> the committed ``web_scrape`` node carries
    the tight loopback CIDR list, overriding any client-supplied ``allowed_hosts``.
  * TUTORIAL session + public base   -> ``allowed_hosts`` is OMITTED (the web_scrape
    field default ``public_only`` applies), stripping any client value.
  * NON-tutorial (live) session      -> the accept path is UNCHANGED; a
    client-supplied ``allowed_hosts`` flows through verbatim (no injection).

The accept is driven directly through the dispatcher (the same seam the HTTP
route calls), with a deliberately MALICIOUS client ``allowed_hosts`` in the
submitted slots to prove the client cannot override the server's value.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool

from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.recipe_match import RecipeMatch
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.config import WebSettings
from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
from tests.fixtures.stores import MockPayloadStore

WEB_SCRAPE_RECIPE = "web-scrape-llm-rate-jsonl"
LOOPBACK_CIDRS = ["127.0.0.1/32", "::1/128"]
# A deliberately wide-open allowlist a malicious client would smuggle in.
MALICIOUS_CLIENT_HOSTS = ["0.0.0.0/0"]
# A benign, clearly client-chosen value distinct from both resolver outcomes.
CLIENT_CHOSEN_HOSTS = ["203.0.113.0/24"]


def _settings(tmp_path: Path, *, base_url: str | None) -> WebSettings:
    return WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
        tutorial_sample_base_url=base_url,
    )


def _real_catalog() -> Any:
    """Real PluginManager so set_pipeline prevalidation sees authentic schemas."""
    from elspeth.plugins.infrastructure.manager import PluginManager
    from elspeth.web.catalog.service import CatalogServiceImpl

    pm = PluginManager()
    pm.register_builtin_plugins()
    return CatalogServiceImpl(pm)


@pytest.fixture
def seeded(tmp_path: Path) -> tuple[Any, str, str]:
    """Seed a minimal session DB with one url-column CSV blob.

    Returns ``(engine, session_id, blob_id)``. ``_execute_apply_pipeline_recipe``
    resolves ``source_blob_id`` against this DB, so the blob row must exist.
    """
    from elspeth.web.blobs.service import content_hash as _content_hash
    from elspeth.web.sessions.engine import create_session_engine
    from elspeth.web.sessions.models import blobs_table, sessions_table
    from elspeth.web.sessions.schema import initialize_session_schema

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
    storage_path = storage_dir / f"{blob_id}_data.csv"
    body = b"url\nhttp://127.0.0.1/tutorial-site/project-1.html\n"
    storage_path.write_bytes(body)
    with engine.begin() as conn:
        conn.execute(
            blobs_table.insert().values(
                id=blob_id,
                session_id=session_id,
                filename="data.csv",
                mime_type="text/csv",
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


def _offered_match(blob_id: str) -> RecipeMatch:
    """The server-staged web-scrape offer. Prefilled slots are binding-checked."""
    return RecipeMatch(
        recipe_name=WEB_SCRAPE_RECIPE,
        slots={
            "source_blob_id": blob_id,
            "source_plugin": "csv",
            "output_path": "outputs/ratings.jsonl",
        },
        unsatisfied_slots={},
    )


def _accept_turn(blob_id: str, *, client_allowed_hosts: list[str] | None) -> dict[str, Any]:
    slots: dict[str, Any] = {
        # Echo the prefilled (server-authored) slots exactly — binding check.
        "source_blob_id": blob_id,
        "source_plugin": "csv",
        "output_path": "outputs/ratings.jsonl",
        # Operator-filled unsatisfied slots.
        "model": "anthropic/claude-sonnet-4.6",
        "api_key_secret": "OPENROUTER_API_KEY",
        "abuse_contact": "ops@example.gov.au",
        "scraping_reason": "synthetic tutorial scrape",
    }
    if client_allowed_hosts is not None:
        # The SSRF vector: a client smuggling allowed_hosts over the wire.
        slots["allowed_hosts"] = client_allowed_hosts
    return {
        "chosen": ["accept"],
        "edited_values": {"recipe_name": WEB_SCRAPE_RECIPE, "slots": slots},
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }


async def _drive_accept(
    *,
    profile: Any,
    settings: WebSettings | None,
    request_origin: str | None,
    seeded: tuple[Any, str, str],
    client_allowed_hosts: list[str] | None,
) -> CompositionState:
    engine, session_id, blob_id = seeded
    state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    guided = replace(
        GuidedSession.initial(profile=profile),
        step=GuidedStep.STEP_2_5_RECIPE_MATCH,
        step_2_5_recipe_offer=_offered_match(blob_id),
    )
    new_state, _new_guided, _next_turn = await _dispatch_guided_respond(
        state=state,
        guided=guided,
        current_step=GuidedStep.STEP_2_5_RECIPE_MATCH,
        current_turn_type=TurnType.RECIPE_OFFER,
        turn_response=_accept_turn(blob_id, client_allowed_hosts=client_allowed_hosts),
        catalog=_real_catalog(),
        recorder=BufferingRecorder(),
        user_id="test-user",
        data_dir=None,
        session_engine=engine,
        session_id=session_id,
        blob_service=None,
        payload_store=MockPayloadStore(),
        model="test-model",
        temperature=None,
        seed=None,
        settings=settings,
        request_origin=request_origin,
    )
    return new_state


def _web_scrape_node(state: CompositionState) -> Any:
    for node in state.nodes:
        if node.plugin == "web_scrape":
            return node
    raise AssertionError(f"no web_scrape node committed; plugins present: {[n.plugin for n in state.nodes]}")


def _node_allowed_hosts(node: Any) -> Any:
    """The web_scrape SSRF allowlist lives under the ``http`` sub-config."""
    return node.options.get("http", {}).get("allowed_hosts")


# The recipe build (recipes.py) nests web_scrape's allowed_hosts under ``http``
# (the plugin's WebScrapeHTTPConfig field) — these pins apply end-to-end through
# WebScrapeConfig (extra:forbid), so they catch nesting/extra-forbid regressions,
# not merely the dict the builder emitted.


# --- Seam gating pins ---------------------------------------------------------


@pytest.mark.asyncio
async def test_tutorial_public_base_omits_allowed_hosts_stripping_client(seeded) -> None:
    """TUTORIAL + public base -> allowed_hosts OMITTED (public_only), client value stripped.

    Recipe-fix-independent: stripping means no allowed_hosts is ever added, so
    the recipe applies regardless of the nesting bug. Proves the seam strips a
    client-smuggled value on a public base.
    """
    state = await _drive_accept(
        profile=TUTORIAL_PROFILE,
        settings=_settings(Path("/unused"), base_url="https://elspeth.foundryside.dev"),
        request_origin="http://127.0.0.1",  # ignored: configured setting wins
        seeded=seeded,
        client_allowed_hosts=MALICIOUS_CLIENT_HOSTS,
    )
    node = _web_scrape_node(state)
    assert _node_allowed_hosts(node) is None
    assert "allowed_hosts" not in node.options  # not stranded at the top level either


@pytest.mark.asyncio
async def test_non_tutorial_no_client_hosts_does_not_inject(seeded) -> None:
    """LIVE (non-tutorial) session + no client hosts -> the seam injects nothing.

    Recipe-fix-independent: with no allowed_hosts the recipe applies cleanly.
    Pins that the injection is strictly tutorial-gated (the seam does not touch
    a non-tutorial accept).
    """
    state = await _drive_accept(
        profile=EMPTY_PROFILE,
        settings=_settings(Path("/unused"), base_url="http://127.0.0.1:8000"),
        request_origin="http://127.0.0.1",
        seeded=seeded,
        client_allowed_hosts=None,
    )
    node = _web_scrape_node(state)
    assert _node_allowed_hosts(node) is None
    assert "allowed_hosts" not in node.options


# --- mustFix CIDR-injection pins (end-to-end through WebScrapeConfig) ----------


@pytest.mark.asyncio
async def test_tutorial_loopback_base_injects_loopback_cidr_over_client(seeded) -> None:
    """TUTORIAL + loopback base -> web_scrape carries the loopback CIDR, NOT the client value."""
    state = await _drive_accept(
        profile=TUTORIAL_PROFILE,
        settings=_settings(Path("/unused"), base_url="http://127.0.0.1:8000"),
        request_origin="http://example.gov.au",  # ignored: configured setting wins
        seeded=seeded,
        client_allowed_hosts=MALICIOUS_CLIENT_HOSTS,
    )
    node = _web_scrape_node(state)
    assert list(_node_allowed_hosts(node)) == LOOPBACK_CIDRS
    assert "0.0.0.0/0" not in _node_allowed_hosts(node)


@pytest.mark.asyncio
async def test_tutorial_request_origin_fallback_resolves_host_class(seeded) -> None:
    """No configured base -> request_origin is the fallback; a loopback origin -> CIDR."""
    state = await _drive_accept(
        profile=TUTORIAL_PROFILE,
        settings=_settings(Path("/unused"), base_url=None),
        request_origin="http://127.0.0.1:9000",
        seeded=seeded,
        client_allowed_hosts=None,
    )
    node = _web_scrape_node(state)
    assert list(_node_allowed_hosts(node)) == LOOPBACK_CIDRS


@pytest.mark.asyncio
async def test_non_tutorial_accept_flows_client_hosts_through(seeded) -> None:
    """LIVE session -> no injection; a client-supplied allowed_hosts flows through verbatim.

    A non-tutorial caller may EXPLICITLY supply a CIDR allowed_hosts; the slot
    routes it to http.allowed_hosts (the pre-existing patch_node_options
    exposure class — the enforcement boundary is unchanged). Pins that the seam
    leaves the non-tutorial value untouched (no override, no strip).
    """
    state = await _drive_accept(
        profile=EMPTY_PROFILE,
        settings=_settings(Path("/unused"), base_url="http://127.0.0.1:8000"),
        request_origin="http://127.0.0.1",
        seeded=seeded,
        client_allowed_hosts=CLIENT_CHOSEN_HOSTS,
    )
    node = _web_scrape_node(state)
    assert list(_node_allowed_hosts(node)) == CLIENT_CHOSEN_HOSTS
