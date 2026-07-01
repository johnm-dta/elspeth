"""Tests for the authority-free review bundle schema + serializer.

A review bundle is the agent->operator handoff artifact. It is a *claim*,
never an authority: it must never serialize an HMAC signature, every staged
preview must be non-authoritative, and a malformed action must fail fast at
load (before any tree walk) so ``verify_bundle_against_tree`` only ever sees
structurally valid actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from elspeth_lints.core.review_bundle import (
    ActionPreview,
    BundleAction,
    RekeyPlan,
    ReviewBundle,
    dump_bundle,
    load_bundle,
    read_bundle,
    write_bundle,
)


def _new_judgment_action(**overrides: Any) -> BundleAction:
    fields: dict[str, Any] = {
        "lane": "new_judgment",
        "kind": "justify",
        "key": "plugins/widget.py:R1:Widget:lookup:fp=abc123",
        "file_path": "plugins/widget.py",
        "symbol": "Widget.lookup",
        "rule": "R1",
        "fingerprint": "abc123",
        "draft_rationale": "payload is Tier-3 external data",
        "preview": ActionPreview(
            verdict="ACCEPTED",
            rationale="preview says genuine",
            model="claude-opus-4-7",
            transport="claude_agent_sdk",
        ),
    }
    fields.update(overrides)
    return BundleAction(**fields)


def _bundle(actions: tuple[BundleAction, ...], *, rekey: RekeyPlan | None = None) -> ReviewBundle:
    return ReviewBundle(
        bundle_id="sample-bundle",
        schema_version=1,
        created_at="2026-06-28T00:00:00+00:00",
        staged_by="agent-x",
        root="src/elspeth",
        allowlist_dir="config/cicd/enforce_tier_model",
        source_rev="deadbeef",
        source_dirty=True,
        actions=actions,
        rekey=rekey,
    )


def test_preview_authoritative_true_is_rejected() -> None:
    with pytest.raises(ValueError):
        ActionPreview(
            verdict="ACCEPTED",
            rationale="x",
            model="m",
            transport="claude_agent_sdk",
            authoritative=True,
        )


def test_bundle_roundtrip_has_no_signature() -> None:
    bundle = _bundle((_new_judgment_action(),))
    text = dump_bundle(bundle)
    assert "judge_metadata_signature" not in text
    assert "hmac-sha256:" not in text
    assert load_bundle(text) == bundle


def test_bundle_roundtrip_with_rekey_plan() -> None:
    rekey = RekeyPlan(
        old_key_env="OLD_KEY_ENV",
        new_key_env="NEW_KEY_ENV",
        keys=("a:fp=1", "b:fp=2"),
        broken_keys=("c:fp=3",),
    )
    bundle = _bundle(
        (BundleAction(lane="resign", kind="stale_delete", key="z:fp=9", source_file="plugins.yaml"),),
        rekey=rekey,
    )
    restored = load_bundle(dump_bundle(bundle))
    assert restored == bundle
    # tuple-typed RekeyPlan fields must survive the JSON list round-trip.
    assert isinstance(restored.rekey.keys, tuple)
    assert isinstance(restored.rekey.broken_keys, tuple)
    assert isinstance(restored.actions, tuple)


def test_load_bundle_rejects_malformed_action_per_kind() -> None:
    # drift_repair without diagnosis_status.
    drift = {
        "lane": "resign",
        "kind": "drift_repair",
        "key": "plugins/widget.py:R1:Widget:lookup:fp=abc123",
    }
    with pytest.raises(ValueError):
        load_bundle(_serialize_with_actions([drift]))

    # justify (new_judgment) without fingerprint.
    justify = {
        "lane": "new_judgment",
        "kind": "justify",
        "key": "plugins/widget.py:R1:Widget:lookup:fp=abc123",
        "file_path": "plugins/widget.py",
        "symbol": "Widget.lookup",
    }
    with pytest.raises(ValueError):
        load_bundle(_serialize_with_actions([justify]))


def test_load_bundle_rejects_incoherent_lane_kind() -> None:
    incoherent = {
        "lane": "new_judgment",
        "kind": "rotation",
        "key": "plugins/widget.py:R1:Widget:lookup:fp=abc123",
        "source_file": "plugins.yaml",
    }
    with pytest.raises(ValueError):
        load_bundle(_serialize_with_actions([incoherent]))


def test_load_bundle_rejects_unknown_key() -> None:
    bundle = _bundle((_new_judgment_action(),))
    data = json.loads(dump_bundle(bundle))
    data["surprise"] = "not a field"
    with pytest.raises(ValueError):
        load_bundle(json.dumps(data))


def test_write_then_read_bundle(tmp_path: Path) -> None:
    bundle = _bundle((_new_judgment_action(),))
    staged_dir = tmp_path / "staged-reviews"
    path = write_bundle(bundle, staged_dir=staged_dir)
    assert path == staged_dir / "sample-bundle.json"
    assert path.parent == staged_dir
    assert read_bundle(path) == bundle


def test_read_malformed_bundle_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    # A JSON object missing schema_version is malformed.
    path.write_text(json.dumps({"bundle_id": "x", "actions": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_bundle(path)


def _serialize_with_actions(action_dicts: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "bundle_id": "sample-bundle",
            "schema_version": 1,
            "created_at": "2026-06-28T00:00:00+00:00",
            "staged_by": "agent-x",
            "root": "src/elspeth",
            "allowlist_dir": "config/cicd/enforce_tier_model",
            "source_rev": None,
            "source_dirty": False,
            "actions": action_dicts,
            "rekey": None,
        }
    )
