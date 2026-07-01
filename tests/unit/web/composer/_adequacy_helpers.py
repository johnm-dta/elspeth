"""Adequacy guard helpers — registry name collection (§4.4.1) and policy-hash
snapshot utilities (§4.4.3).

The hash utilities are shared between the test file and the bootstrap script
(``scripts/cicd/bootstrap_redaction_snapshot.py``) to prevent logic
duplication.  The bootstrap script computes ``sys.path`` relative to its own
location to import from this module without installing the tests package.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from elspeth.web.composer.redaction import (
    HandlesNoSensitiveDataReason,
    ToolRedaction,
    TraversalNode,
    _SensitiveMarker,
    walk_model_schema,
)
from elspeth.web.composer.tools import (
    _BLOB_DISCOVERY_TOOLS,
    _BLOB_MUTATION_TOOLS,
    _DISCOVERY_TOOLS,
    _MUTATION_TOOLS,
    _SECRET_DISCOVERY_TOOLS,
    _SECRET_MUTATION_TOOLS,
    _SESSION_AWARE_TOOL_HANDLERS,
)


def collect_registry_names() -> frozenset[str]:
    """Union of every tool name dispatchable through the composer.

    Spec §4.4.1: the manifest's set-equality contract is keyed on this
    set. Adding a tool to a dispatch dict without a manifest entry —
    or removing a manifest entry without removing the dispatch dict
    entry — must fail this guard at CI.

    The six dispatch dicts (tools.py lines 5274-5340) are unioned here.
    preview_pipeline, diff_pipeline, and set_pipeline already appear in
    _DISCOVERY_TOOLS / _MUTATION_TOOLS; their extended-signature
    special-casing in execute_tool() is an implementation detail, not a
    separate registration.

    request_advisor_hint is intercepted at service.py:2070 before
    execute_tool() is called, so it never appears in the six dispatch
    dicts. It requires a MANIFEST entry and is enumerated here explicitly.

    Phase 5b Task 5: ``_SESSION_AWARE_TOOL_HANDLERS`` is the seventh
    dispatch dict — session-aware async tools that are intercepted in
    the compose loop ahead of ``execute_tool`` (same pattern as
    ``request_advisor_hint``). Every name here requires a MANIFEST
    entry; the redaction policy applies regardless of which dispatch
    surface the tool lands on.
    """
    return frozenset(
        set(_DISCOVERY_TOOLS)
        | set(_MUTATION_TOOLS)
        | set(_BLOB_DISCOVERY_TOOLS)
        | set(_BLOB_MUTATION_TOOLS)
        | set(_SECRET_DISCOVERY_TOOLS)
        | set(_SECRET_MUTATION_TOOLS)
        | set(_SESSION_AWARE_TOOL_HANDLERS)
        | {"request_advisor_hint"}  # advisor escape hatch — intercepted at service.py
    )


# ---------------------------------------------------------------------------
# Policy-hash snapshot utilities (spec §4.4.3).
#
# These are shared between the adequacy-guard test suite and the bootstrap
# script so neither side duplicates the canonical encoding logic.
# ---------------------------------------------------------------------------


def _canonicalise_node(node: TraversalNode) -> dict[str, Any]:
    """Produce a deterministic representation of a TraversalNode for hashing.

    Spec §4.4.3: ``{"path": ..., "type_name": ..., "metadata": [...marker
    types and summarizer fully-qualified names...]}``.

    ``type_name`` is the ``__name__`` attribute for concrete types, or
    ``repr()`` for parameterised generics (e.g., ``"dict[str, Any]"``).
    Using ``repr()`` for generics is deterministic for a fixed Python version
    and produces human-readable output without importing ``typing``'s
    internal machinery.

    ``metadata`` encodes every ``_SensitiveMarker`` in the node's metadata
    tuple.  Non-marker metadata entries (e.g., ``pydantic.FieldInfo``) are
    intentionally ignored — they are not part of the redaction policy.

    For each ``_SensitiveMarker``:
      - ``"marker": "_SensitiveMarker"`` (the class name; literals are stable).
      - ``"summarizer"``: the summarizer's fully-qualified name as
        ``f"{func.__module__}.{func.__qualname__}"`` when non-None, or
        ``null`` when None.

    Using fully-qualified name instead of ``id()`` ensures the hash is stable
    across process restarts and module reloads (spec §4.4.3, line 1234).
    Known false-negative: in-place mutation of closure-captured state does
    not change the function's ``__qualname__``; see the test
    ``test_closure_state_mutation_false_negative`` and spec §4.4.3.
    """
    # Compute type_name: prefer __name__ (concrete type), fall back to repr.
    try:
        type_name: str = node.field_type.__name__
    except AttributeError:
        type_name = repr(node.field_type)

    # Encode only _SensitiveMarker entries; other metadata is not policy.
    encoded_metadata: list[dict[str, Any]] = []
    for m in node.metadata:
        if isinstance(m, _SensitiveMarker):
            if m.summarizer is not None:
                summarizer_fqn: str | None = f"{m.summarizer.__module__}.{m.summarizer.__qualname__}"
            else:
                summarizer_fqn = None
            encoded_metadata.append(
                {
                    "marker": "_SensitiveMarker",
                    "summarizer": summarizer_fqn,
                }
            )

    return {
        "path": node.path,
        "type_name": type_name,
        "metadata": encoded_metadata,
    }


def _reason_text_hash(reason: HandlesNoSensitiveDataReason) -> str:
    """Stable SHA-256 over the text fields of a HandlesNoSensitiveDataReason.

    Used by ``_entry_hash`` for the ``reason_text_hash`` field of declarative
    entries so that changes to ``why_arguments_safe`` or ``why_responses_safe``
    flip the snapshot hash.  The ``sensitive_data_locations`` tuple is also
    included so location changes are detected.
    """
    content = {
        "sensitive_data_locations": sorted(reason.sensitive_data_locations),
        "why_arguments_safe": reason.why_arguments_safe,
        "why_responses_safe": reason.why_responses_safe,
    }
    canon = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _entry_hash(name: str, entry: ToolRedaction) -> dict[str, Any]:
    """Compute a content-keyed snapshot record for a manifest entry (spec §4.4.3).

    Returns a dict with:
      - ``"hash"``: stable SHA-256 hex (deterministic across runs).
      - ``"shape"``: ``"type_driven"`` | ``"declarative"``.
      - ``"sensitive_path_count"``: int — Task 18 uses this for the
        direction-aware label gate without needing a live Python run.

    **Type-driven entries** (``argument_model`` is not None): hash over the
    schema-walk produced by the shared iterator.  Every node in the walk is
    included (sensitive and non-sensitive alike) so that adding, removing, or
    renaming a field — not only changing a Sensitive annotation — flips the
    hash.  ``response_model`` (if set) is walked and appended.
    ``sensitive_path_count`` counts nodes whose metadata contains at least one
    ``_SensitiveMarker``.

    **Declarative entries** (``policy`` is not None): hash over the policy's
    key-sets, summarizer keys, and reason text.  Summarizer *values* (callable
    identity) are not hashed for declarative entries because the declarative
    shape uses ``argument_summarizers`` keyed by string; key-set changes are
    the detectable unit.  A declared ``known_argument_keys`` allowlist is
    included because it changes whether unknown LLM-supplied arguments fail
    closed before persistence.
    ``sensitive_path_count`` is
    ``len(sensitive_argument_keys) + len(sensitive_response_keys)`` without
    deduplication (each named key contributes once per field surface).
    """
    if entry.argument_model is not None:
        shape = "type_driven"
        nodes: list[TraversalNode] = list(walk_model_schema(entry.argument_model))
        if entry.response_model is not None:
            nodes.extend(walk_model_schema(entry.response_model))

        canon_payload = [_canonicalise_node(n) for n in nodes]
        sensitive_path_count = sum(1 for n in nodes if any(isinstance(m, _SensitiveMarker) for m in n.metadata))
    else:
        assert entry.policy is not None  # ToolRedaction invariant
        shape = "declarative"
        policy = entry.policy
        canon_payload = {  # type: ignore[assignment]  # dict intentional here
            "sensitive_argument_keys": sorted(policy.sensitive_argument_keys),
            "sensitive_response_keys": sorted(policy.sensitive_response_keys),
            "known_response_keys": sorted(policy.known_response_keys),
            "summarizer_keys": sorted(policy.argument_summarizers.keys()),
            "handles_no_sensitive_data": policy.handles_no_sensitive_data,
            "reason_text_hash": (
                _reason_text_hash(policy.handles_no_sensitive_data_reason_struct)
                if policy.handles_no_sensitive_data_reason_struct is not None
                else None
            ),
        }
        if policy.known_argument_keys or policy.redact_unknown_argument_keys:
            canon_payload["known_argument_keys"] = sorted(policy.known_argument_keys)
        if policy.redact_unknown_argument_keys:
            canon_payload["redact_unknown_argument_keys"] = True
        sensitive_path_count = len(policy.sensitive_argument_keys) + len(policy.sensitive_response_keys)

    canon = json.dumps(canon_payload, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()

    return {
        "hash": h,
        "shape": shape,
        "sensitive_path_count": sensitive_path_count,
    }


def compute_manifest_snapshot(
    manifest: dict[str, ToolRedaction] | Any,
) -> dict[str, dict[str, Any]]:
    """Compute the full snapshot dict for every entry in ``manifest``.

    ``manifest`` is any ``Mapping[str, ToolRedaction]`` (e.g., ``MANIFEST``
    or a plain dict built for testing).  Returns a plain dict sorted by tool
    name for deterministic JSON serialisation.
    """
    return {name: _entry_hash(name, entry) for name, entry in sorted(manifest.items())}
