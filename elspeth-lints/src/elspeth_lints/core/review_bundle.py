"""Authority-free review bundle schema, serializer, and on-disk writer/reader.

A *review bundle* is the agent->operator handoff artifact for the turnkey
judge/signature stage. The agent (key-free, via the ``elspeth-judge`` MCP
surface) assembles it; the operator fires it with ``elspeth-lints sign-bundle``.

The bundle carries *claims*, never authority:

* it never serializes an HMAC signature (no ``judge_metadata_signature`` key,
  no ``hmac-sha256:`` value);
* every staged ``ActionPreview`` is non-authoritative
  (``authoritative`` is structurally always ``False``);
* the HMAC key bytes never enter a bundle -- a ``RekeyPlan`` records only the
  *names* of the env vars holding the old/new keys.

``dump_bundle``/``load_bundle`` round-trip losslessly and ``load_bundle`` is
strict: unknown keys raise, ``schema_version`` is checked, and a malformed
action fails fast at parse (per-kind required fields + lane<->kind coherence)
so the from-tree verify only ever sees structurally valid actions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elspeth_lints.core.atomic_io import atomic_update_text

SCHEMA_VERSION = 1

# A bundle ``kind`` fully determines its ``lane`` -- the two can never drift.
_KIND_TO_LANE: dict[str, str] = {
    "justify": "new_judgment",
    "drift_repair": "resign",
    "rotation": "resign",
    "stale_delete": "resign",
}

# Per-kind required fields (``key`` is required for every kind, checked
# separately). A malformed action raises a typed ``ValueError`` at construction
# so ``load_bundle`` rejects it before any tree walk.
_REQUIRED_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "justify": ("file_path", "symbol", "fingerprint"),
    "drift_repair": ("diagnosis_status",),
    "rotation": ("source_file",),
    "stale_delete": ("source_file",),
}

_PREVIEW_FIELDS = ("verdict", "rationale", "model", "transport", "authoritative")
_ACTION_FIELDS = (
    "lane",
    "kind",
    "key",
    "file_path",
    "symbol",
    "rule",
    "fingerprint",
    "scope_fingerprint",
    "ast_path",
    "draft_rationale",
    "diagnosis_status",
    "source_file",
    "preview",
)
_REKEY_FIELDS = ("old_key_env", "new_key_env", "keys", "broken_keys")
_BUNDLE_FIELDS = (
    "bundle_id",
    "schema_version",
    "created_at",
    "staged_by",
    "root",
    "allowlist_dir",
    "source_rev",
    "source_dirty",
    "actions",
    "rekey",
)


@dataclass(frozen=True, slots=True)
class ActionPreview:
    """Non-authoritative preview verdict for a ``new_judgment`` action.

    Produced by the key-free agent preview judge. The ``authoritative`` flag is
    structurally always ``False``: ``__post_init__`` rejects ``True`` so a
    bundle can never carry an authoritative agent-produced verdict ([O1]).
    """

    verdict: str
    rationale: str
    model: str
    transport: str
    authoritative: bool = False

    def __post_init__(self) -> None:
        if self.authoritative:
            raise ValueError("ActionPreview.authoritative must be False; a bundle preview is never authoritative")


@dataclass(frozen=True, slots=True)
class BundleAction:
    """One staged repair/judgment action.

    ``kind`` fully determines ``lane`` (``__post_init__`` rejects an incoherent
    pair) and which fields are required.
    """

    lane: str
    kind: str
    key: str
    file_path: str | None = None
    symbol: str | None = None
    rule: str | None = None
    fingerprint: str | None = None
    scope_fingerprint: str | None = None
    ast_path: str | None = None
    draft_rationale: str | None = None
    diagnosis_status: str | None = None
    source_file: str | None = None
    preview: ActionPreview | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("BundleAction.key must be a non-empty canonical allowlist key")
        expected_lane = _KIND_TO_LANE.get(self.kind)
        if expected_lane is None:
            raise ValueError(f"BundleAction.kind={self.kind!r} is not a known kind (expected one of {sorted(_KIND_TO_LANE)})")
        if self.lane != expected_lane:
            raise ValueError(
                f"BundleAction.lane={self.lane!r} is incoherent with kind={self.kind!r} "
                f"(kind {self.kind!r} requires lane {expected_lane!r})"
            )
        missing = [name for name in _REQUIRED_FIELDS_BY_KIND[self.kind] if not getattr(self, name)]
        if missing:
            raise ValueError(f"BundleAction kind={self.kind!r} is missing required field(s): {missing}")


@dataclass(frozen=True, slots=True)
class RekeyPlan:
    """Advisory provenance for a key rotation.

    Records only the *names* of the env vars holding the old/new HMAC keys --
    never the key bytes. ``rekey`` re-derives the judge-gated set from the tree
    at fire time and treats this plan as advisory.
    """

    old_key_env: str
    new_key_env: str
    keys: tuple[str, ...]
    broken_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReviewBundle:
    """The full authority-free handoff artifact."""

    bundle_id: str
    schema_version: int
    created_at: str
    staged_by: str
    root: str
    allowlist_dir: str
    source_rev: str | None
    source_dirty: bool
    actions: tuple[BundleAction, ...]
    rekey: RekeyPlan | None = None


def dump_bundle(bundle: ReviewBundle) -> str:
    """Serialize a bundle to deterministic JSON.

    Only the known, non-signature-bearing fields are emitted, so the result is
    structurally signature-free.
    """
    return json.dumps(_bundle_to_dict(bundle), indent=2, sort_keys=True) + "\n"


def load_bundle(text: str) -> ReviewBundle:
    """Parse a bundle from JSON text (strict).

    Unknown keys raise, ``schema_version`` is checked, and each action is
    validated per-kind via ``BundleAction.__post_init__``.
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"review bundle must be a JSON object; got {type(data).__name__}")
    _reject_unknown_keys(data, _BUNDLE_FIELDS, "bundle")

    schema_version = data.get("schema_version")
    if schema_version is None:
        raise ValueError("review bundle is missing required field 'schema_version'")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"review bundle schema_version={schema_version!r}; this build understands {SCHEMA_VERSION}")

    raw_actions = _require(data, "actions", "bundle")
    if not isinstance(raw_actions, list):
        raise ValueError(f"review bundle 'actions' must be a list; got {type(raw_actions).__name__}")
    actions = tuple(_action_from_dict(item) for item in raw_actions)

    raw_rekey = data.get("rekey")
    rekey = _rekey_from_dict(raw_rekey) if raw_rekey is not None else None

    return ReviewBundle(
        bundle_id=_require_str(data, "bundle_id", "bundle"),
        schema_version=schema_version,
        created_at=_require_str(data, "created_at", "bundle"),
        staged_by=_require_str(data, "staged_by", "bundle"),
        root=_require_str(data, "root", "bundle"),
        allowlist_dir=_require_str(data, "allowlist_dir", "bundle"),
        source_rev=data.get("source_rev"),
        source_dirty=bool(_require(data, "source_dirty", "bundle")),
        actions=actions,
        rekey=rekey,
    )


def write_bundle(bundle: ReviewBundle, *, staged_dir: Path) -> Path:
    """Atomically write ``bundle`` to ``<staged_dir>/<bundle_id>.json``.

    Creates ``staged_dir`` if absent (``.elspeth/staged-reviews/`` is already
    gitignored).
    """
    path = Path(staged_dir) / f"{bundle.bundle_id}.json"
    atomic_update_text(path, lambda _current: dump_bundle(bundle), create_parent=True)
    return path


def read_bundle(path: Path) -> ReviewBundle:
    """Read + strictly parse a bundle file; refuses malformed input."""
    return load_bundle(Path(path).read_text(encoding="utf-8"))


def _bundle_to_dict(bundle: ReviewBundle) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        "schema_version": bundle.schema_version,
        "created_at": bundle.created_at,
        "staged_by": bundle.staged_by,
        "root": bundle.root,
        "allowlist_dir": bundle.allowlist_dir,
        "source_rev": bundle.source_rev,
        "source_dirty": bundle.source_dirty,
        "actions": [_action_to_dict(action) for action in bundle.actions],
        "rekey": _rekey_to_dict(bundle.rekey) if bundle.rekey is not None else None,
    }


def _action_to_dict(action: BundleAction) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "lane": action.lane,
        "kind": action.kind,
        "key": action.key,
        "file_path": action.file_path,
        "symbol": action.symbol,
        "rule": action.rule,
        "fingerprint": action.fingerprint,
        "scope_fingerprint": action.scope_fingerprint,
        "ast_path": action.ast_path,
        "draft_rationale": action.draft_rationale,
        "diagnosis_status": action.diagnosis_status,
        "source_file": action.source_file,
        "preview": _preview_to_dict(action.preview) if action.preview is not None else None,
    }
    return payload


def _preview_to_dict(preview: ActionPreview) -> dict[str, Any]:
    return {
        "verdict": preview.verdict,
        "rationale": preview.rationale,
        "model": preview.model,
        "transport": preview.transport,
        "authoritative": preview.authoritative,
    }


def _rekey_to_dict(rekey: RekeyPlan) -> dict[str, Any]:
    return {
        "old_key_env": rekey.old_key_env,
        "new_key_env": rekey.new_key_env,
        "keys": list(rekey.keys),
        "broken_keys": list(rekey.broken_keys),
    }


def _action_from_dict(data: Any) -> BundleAction:
    if not isinstance(data, dict):
        raise ValueError(f"bundle action must be a JSON object; got {type(data).__name__}")
    _reject_unknown_keys(data, _ACTION_FIELDS, "action")
    raw_preview = data.get("preview")
    preview = _preview_from_dict(raw_preview) if raw_preview is not None else None
    return BundleAction(
        lane=_require_str(data, "lane", "action"),
        kind=_require_str(data, "kind", "action"),
        key=_require_str(data, "key", "action"),
        file_path=data.get("file_path"),
        symbol=data.get("symbol"),
        rule=data.get("rule"),
        fingerprint=data.get("fingerprint"),
        scope_fingerprint=data.get("scope_fingerprint"),
        ast_path=data.get("ast_path"),
        draft_rationale=data.get("draft_rationale"),
        diagnosis_status=data.get("diagnosis_status"),
        source_file=data.get("source_file"),
        preview=preview,
    )


def _preview_from_dict(data: Any) -> ActionPreview:
    if not isinstance(data, dict):
        raise ValueError(f"action preview must be a JSON object; got {type(data).__name__}")
    _reject_unknown_keys(data, _PREVIEW_FIELDS, "preview")
    return ActionPreview(
        verdict=_require_str(data, "verdict", "preview"),
        rationale=_require_str(data, "rationale", "preview"),
        model=_require_str(data, "model", "preview"),
        transport=_require_str(data, "transport", "preview"),
        authoritative=bool(data.get("authoritative", False)),
    )


def _rekey_from_dict(data: Any) -> RekeyPlan:
    if not isinstance(data, dict):
        raise ValueError(f"rekey plan must be a JSON object; got {type(data).__name__}")
    _reject_unknown_keys(data, _REKEY_FIELDS, "rekey")
    return RekeyPlan(
        old_key_env=_require_str(data, "old_key_env", "rekey"),
        new_key_env=_require_str(data, "new_key_env", "rekey"),
        keys=tuple(_require_str_list(data, "keys", "rekey")),
        broken_keys=tuple(_require_str_list(data, "broken_keys", "rekey")),
    )


def _reject_unknown_keys(data: dict[str, Any], allowed: tuple[str, ...], context: str) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(f"review {context} has unknown key(s): {sorted(unknown)}")


def _require(data: dict[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise ValueError(f"review {context} is missing required field {key!r}")
    return data[key]


def _require_str(data: dict[str, Any], key: str, context: str) -> str:
    value = _require(data, key, context)
    if not isinstance(value, str):
        raise ValueError(f"review {context} field {key!r} must be a string; got {type(value).__name__}")
    return value


def _require_str_list(data: dict[str, Any], key: str, context: str) -> list[str]:
    value = _require(data, key, context)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"review {context} field {key!r} must be a list of strings")
    return value
