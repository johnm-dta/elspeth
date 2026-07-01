"""Tests for the OpenRouter model catalog reader and live-prime path.

The catalog reader has two states:

* Unprimed (default at import time, also the state CLI / non-web entry
  points stay in): :func:`_read_openrouter_catalog` returns the bundled
  litellm OpenRouter slice. This preserves the offline-friendly default
  and keeps existing CLI/test behaviour unchanged.

* Primed (web-server lifespan called :func:`prime_openrouter_catalog_from_live`
  with a working HTTP getter): :func:`_read_openrouter_catalog` returns
  the live snapshot taken from OpenRouter's ``/models`` endpoint. This is
  the path that closes the validate/runtime drift bug: composer LLM picks
  from the live set, the value-source compliance walker validates against
  the live set, and the runtime preflight no longer 404s on identifiers
  retired from OpenRouter but still in the bundled litellm catalog.

The prime call is graceful: transport errors and malformed responses
leave the catalog in its unprimed state (bundled fallback) and the boot
sequence continues.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from elspeth.plugins.transforms.llm import model_catalog


@pytest.fixture(autouse=True)
def _reset_live_catalog() -> Iterator[None]:
    """Reset the module-level live catalog before and after every test.

    The catalog reader keeps a module-global :data:`_LIVE_CATALOG` snapshot
    populated by :func:`prime_openrouter_catalog_from_live`. Tests
    assertion-check the *transition* between unprimed and primed states,
    so each test must start from unprimed regardless of prior test order.
    """
    model_catalog.reset_live_openrouter_catalog()
    yield
    model_catalog.reset_live_openrouter_catalog()


def _make_response(payload: Any, *, status_code: int = 200) -> httpx.Response:
    """Build an httpx.Response carrying ``payload`` as JSON.

    Matches the shape ``prime_openrouter_catalog_from_live`` expects from
    its injected ``http_get`` callable. Keeping this helper local to the
    test module keeps the fixture surface narrow.
    """
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("GET", "https://openrouter.ai/api/v1/models"),
    )


async def _prime_live_catalog(ids: list[str]) -> None:
    async def fake_get(url: str) -> httpx.Response:
        assert url == model_catalog.OPENROUTER_MODELS_URL
        return _make_response({"data": [{"id": model_id} for model_id in ids]})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=fake_get)
    assert primed is True
    assert model_catalog._read_openrouter_catalog() == frozenset(ids)
    _sha, source = model_catalog.read_openrouter_catalog_snapshot_id()
    assert source == "live"


def _assert_bundled_catalog_is_active() -> None:
    assert model_catalog._read_openrouter_catalog() == model_catalog._bundled_openrouter_slice()
    sha, source = model_catalog.read_openrouter_catalog_snapshot_id()
    assert source == "bundled"
    assert sha == model_catalog._bundled_openrouter_slice_sha256()


def test_read_openrouter_catalog_returns_bundled_slice_when_unprimed() -> None:
    """Unprimed reader = bundled litellm slice (pre-fix behaviour preserved).

    CLI invocations and unit-test runs never call the lifespan prime, so
    the existing offline behaviour must remain intact: the bundled
    litellm OpenRouter entries are returned with the ``openrouter/``
    routing prefix stripped.
    """
    catalog = model_catalog._read_openrouter_catalog()
    bundled = {
        m[len(model_catalog.OPENROUTER_LITELLM_PREFIX) :]
        for m in model_catalog.read_litellm_model_list()
        if m.startswith(model_catalog.OPENROUTER_LITELLM_PREFIX)
    }
    assert catalog == frozenset(bundled)


@pytest.mark.asyncio
async def test_prime_openrouter_catalog_excludes_retired_models() -> None:
    """Live prime overrides the bundled slice with the upstream-current set.

    Pins the actual bug: ``anthropic/claude-3.5-sonnet`` is in litellm's
    bundled catalog but retired from OpenRouter's live ``/models``. After
    prime, the catalog must reflect what OpenRouter actually serves, not
    what litellm bundled at a snapshot in time.
    """
    live_payload = {
        "data": [
            {"id": "anthropic/claude-3.5-haiku"},
            {"id": "anthropic/claude-sonnet-4.6"},
            {"id": "openai/gpt-4o"},
        ],
    }

    async def fake_get(url: str) -> httpx.Response:
        assert url == model_catalog.OPENROUTER_MODELS_URL
        return _make_response(live_payload)

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=fake_get)
    assert primed is True

    catalog = model_catalog._read_openrouter_catalog()
    assert "anthropic/claude-3.5-sonnet" not in catalog
    assert "anthropic/claude-3.5-haiku" in catalog
    assert "anthropic/claude-sonnet-4.6" in catalog
    assert "openai/gpt-4o" in catalog


@pytest.mark.asyncio
async def test_prime_openrouter_catalog_propagates_live_additions() -> None:
    """Models OpenRouter adds *after* litellm's bundle ship in the live set.

    Validates the symmetric case to retirement: any id present on
    OpenRouter's live endpoint must appear in the post-prime catalog,
    even if litellm has never seen it. Otherwise we'd silently exclude
    new models the operator picks from the composer dropdown.
    """
    novel_id = "future-vendor/never-seen-before-model"
    bundled = model_catalog.read_litellm_model_list()
    assert f"openrouter/{novel_id}" not in bundled, "test fixture invalid: id is in bundle"

    async def fake_get(url: str) -> httpx.Response:
        return _make_response({"data": [{"id": novel_id}]})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=fake_get)
    assert primed is True
    assert novel_id in model_catalog._read_openrouter_catalog()


@pytest.mark.asyncio
async def test_prime_openrouter_catalog_falls_back_on_transport_error() -> None:
    """Transport error during prime is graceful: bundled fallback, no raise.

    The lifespan caller logs a structured warning and continues. The
    catalog reader must continue answering — staging boot must not block
    on OpenRouter availability.
    """

    async def failing_get(url: str) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated transport failure")

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=failing_get)
    assert primed is False

    catalog = model_catalog._read_openrouter_catalog()
    bundled = {
        m[len(model_catalog.OPENROUTER_LITELLM_PREFIX) :]
        for m in model_catalog.read_litellm_model_list()
        if m.startswith(model_catalog.OPENROUTER_LITELLM_PREFIX)
    }
    assert catalog == frozenset(bundled)


@pytest.mark.asyncio
async def test_failed_transport_reprime_after_success_clears_live_snapshot() -> None:
    await _prime_live_catalog(["vendor/live-only-model"])

    async def failing_get(url: str) -> httpx.Response:
        assert url == model_catalog.OPENROUTER_MODELS_URL
        raise httpx.ConnectTimeout("simulated re-prime transport failure")

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=failing_get)

    assert primed is False
    _assert_bundled_catalog_is_active()


@pytest.mark.parametrize("failure_kind", ("http_status", "json_decode", "malformed_top_level", "empty_data"))
@pytest.mark.asyncio
async def test_failed_early_return_reprime_after_success_clears_live_snapshot(failure_kind: str) -> None:
    await _prime_live_catalog([f"vendor/live-only-model-{failure_kind}"])

    class _BadJsonResponse:
        status_code = 200

        def json(self) -> Any:
            raise ValueError("not json")

    async def failing_get(url: str) -> Any:
        assert url == model_catalog.OPENROUTER_MODELS_URL
        if failure_kind == "http_status":
            return _make_response("Service Unavailable", status_code=503)
        if failure_kind == "json_decode":
            return _BadJsonResponse()
        if failure_kind == "malformed_top_level":
            return _make_response({"unexpected": "shape"})
        if failure_kind == "empty_data":
            return _make_response({"data": []})
        raise AssertionError(f"unhandled failure kind: {failure_kind}")

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=failing_get)

    assert primed is False
    _assert_bundled_catalog_is_active()


@pytest.mark.asyncio
async def test_prime_openrouter_catalog_rejects_malformed_response() -> None:
    """Malformed top-level shape falls back; no exception propagates.

    Tier-3 trust boundary: the OpenRouter response is external data. A
    response missing ``data`` (e.g. an OpenRouter outage page returning
    a JSON error envelope, or an upstream API contract change) must not
    crash the boot. Same fallback semantics as a transport error.
    """

    async def malformed_get(url: str) -> httpx.Response:
        return _make_response({"unexpected": "shape"})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=malformed_get)
    assert primed is False

    catalog = model_catalog._read_openrouter_catalog()
    bundled = {
        m[len(model_catalog.OPENROUTER_LITELLM_PREFIX) :]
        for m in model_catalog.read_litellm_model_list()
        if m.startswith(model_catalog.OPENROUTER_LITELLM_PREFIX)
    }
    assert catalog == frozenset(bundled)


@pytest.mark.asyncio
async def test_prime_falls_back_on_http_non_200() -> None:
    """5xx (or any non-200) response: graceful fallback to bundled, no raise.

    Mirrors the most common OpenRouter transient outage shape: the
    endpoint is reachable but the upstream returns a service-unavailable
    response. Boot must continue with the bundled fallback in effect.
    """

    async def fake_get(url: str) -> httpx.Response:
        return _make_response("Service Unavailable", status_code=503)

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=fake_get)
    assert primed is False
    assert model_catalog._read_openrouter_catalog() == model_catalog._bundled_openrouter_slice()


@pytest.mark.asyncio
async def test_prime_falls_back_on_json_decode_error() -> None:
    """200 OK with a body that fails JSON decoding: graceful fallback.

    OpenRouter could in principle ship an HTML error page with HTTP 200
    (e.g. a CDN-level outage page). The reader must not crash, must not
    raise, and must leave the catalog in its unprimed state.
    """

    class _BadJsonResponse:
        status_code = 200

        def json(self) -> Any:
            raise ValueError("not json")

    async def fake_get(url: str) -> Any:
        return _BadJsonResponse()

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=fake_get)
    assert primed is False
    assert model_catalog._read_openrouter_catalog() == model_catalog._bundled_openrouter_slice()


@pytest.mark.asyncio
async def test_prime_falls_back_on_200_empty_data() -> None:
    """200 OK with valid shape but empty ``data`` array: graceful fallback.

    This is the most likely real-world failure mode: a transient
    OpenRouter incident returns the canonical shape but no model
    entries. Replacing the bundled catalog with an empty set would
    break every value-source compliance check downstream, so the
    prime explicitly refuses to publish an empty live snapshot.
    """

    async def fake_get(url: str) -> httpx.Response:
        return _make_response({"data": []})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=fake_get)
    assert primed is False
    assert model_catalog._read_openrouter_catalog() == model_catalog._bundled_openrouter_slice()


@pytest.mark.asyncio
async def test_prime_skips_malformed_entries_without_failing() -> None:
    """Mixed-shape ``data`` entries: per-entry coercion, valid ids retained.

    OpenRouter occasionally ships entries with non-standard fields; the
    prime tolerates that by skipping entries that fail the per-entry
    shape check (not a dict, no string id, empty id, non-string id),
    rather than aborting the entire prime. Valid entries are retained
    in the published snapshot.
    """
    payload = {
        "data": [
            {"id": "valid/one"},
            "not_a_dict",
            {"id": None},
            {"id": ""},
            {"id": 42},
            {"id": "valid/two"},
        ],
    }

    async def fake_get(url: str) -> httpx.Response:
        return _make_response(payload)

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=fake_get)
    assert primed is True
    catalog = model_catalog._read_openrouter_catalog()
    assert "valid/one" in catalog
    assert "valid/two" in catalog
    # The 4 malformed entries (non-dict, None id, empty id, int id) are
    # all skipped; only the two valid ids survive.  The exact skip count
    # is not asserted here — the contract is "valid ids retained, no
    # crash" — but the catalog cardinality pins the per-entry filter.
    assert len(catalog) == 2


# ---------------------------------------------------------------------------
# Catalog snapshot id (B2.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_catalog_snapshot_id_is_canonical_over_input_order() -> None:
    """Two primes against the same id set produce the same sha, any order.

    The canonical sha-recipe (sorted utf-8 ids joined on ``\\n``) is the
    audit-trail anchor: two boots that see the same OpenRouter catalog
    must record the same sha regardless of the JSON traversal order or
    Python's set hashing.
    """
    ids = ["openai/gpt-4o", "anthropic/claude-3.5-haiku", "openai/gpt-4o-mini"]

    async def first_order(url: str) -> httpx.Response:
        return _make_response({"data": [{"id": i} for i in ids]})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=first_order)
    assert primed is True
    sha_first, source_first = model_catalog.read_openrouter_catalog_snapshot_id()
    assert source_first == "live"

    model_catalog.reset_live_openrouter_catalog()

    async def reverse_order(url: str) -> httpx.Response:
        return _make_response({"data": [{"id": i} for i in reversed(ids)]})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=reverse_order)
    assert primed is True
    sha_second, _ = model_catalog.read_openrouter_catalog_snapshot_id()
    assert sha_second == sha_first


@pytest.mark.asyncio
async def test_catalog_snapshot_id_changes_on_set_change() -> None:
    """Adding or removing an id changes the sha — the snapshot is bound to membership."""
    base_ids = [{"id": "a/one"}, {"id": "a/two"}]

    async def base_get(url: str) -> httpx.Response:
        return _make_response({"data": base_ids})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=base_get)
    assert primed is True
    sha_base, _ = model_catalog.read_openrouter_catalog_snapshot_id()

    model_catalog.reset_live_openrouter_catalog()

    async def added_get(url: str) -> httpx.Response:
        return _make_response({"data": [*base_ids, {"id": "a/three"}]})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=added_get)
    assert primed is True
    sha_added, _ = model_catalog.read_openrouter_catalog_snapshot_id()
    assert sha_added != sha_base

    model_catalog.reset_live_openrouter_catalog()

    async def removed_get(url: str) -> httpx.Response:
        return _make_response({"data": [{"id": "a/one"}]})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=removed_get)
    assert primed is True
    sha_removed, _ = model_catalog.read_openrouter_catalog_snapshot_id()
    assert sha_removed != sha_base
    assert sha_removed != sha_added


@pytest.mark.asyncio
async def test_catalog_snapshot_source_reflects_prime_state() -> None:
    """``source`` is ``"bundled"`` when unprimed, ``"live"`` after a prime.

    The audit row records which catalog blessed a run's decisions; the
    source discriminator must accurately reflect whether the live probe
    succeeded.
    """
    sha_bundled, source_bundled = model_catalog.read_openrouter_catalog_snapshot_id()
    assert source_bundled == "bundled"
    assert sha_bundled, "bundled sha must be non-empty"

    async def fake_get(url: str) -> httpx.Response:
        return _make_response({"data": [{"id": "openai/gpt-4o"}]})

    primed = await model_catalog.prime_openrouter_catalog_from_live(http_get=fake_get)
    assert primed is True
    sha_live, source_live = model_catalog.read_openrouter_catalog_snapshot_id()
    assert source_live == "live"
    assert sha_live != sha_bundled
