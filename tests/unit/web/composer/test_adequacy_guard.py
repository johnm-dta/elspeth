"""Adequacy guard — manifest-level structural assertions (spec §4.4).

Assertion 1 (registry-manifest set equality, §4.4.1) is xfail until the
manifest reaches parity with the dispatch registry (Tasks 13-16).

Assertions 2 (per-entry shape walk), 5 (extra="forbid"), 6
(sensitive_response_keys ⊆ known_response_keys), and the walker-completeness
floor-check (rev-3 M1) are enforced unconditionally at every collection of
this file. They trivially pass while MANIFEST holds only set_source; they
begin enforcing real coverage as Tasks 13-16 populate the manifest.

The adequacy guard works at MANIFEST-level only. AST inspection of handler
source is forbidden (rev-2 M.3): any handler using ``args = arguments;
args['x']``, destructuring, or helper delegation evades the scan. Tools
requiring mechanical key-coverage guarantees MUST be promoted to type-driven
Pydantic argument models with ``extra="forbid"``.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal, get_origin
from uuid import UUID

import pytest

from elspeth.web.composer.redaction import (
    MANIFEST,
    _SensitiveMarker,
    walk_model_schema,
)

from ._adequacy_helpers import collect_registry_names

# Closed list of non-redaction-eligible scalar types (spec §4.4.2 disposition).
#
# A TraversalNode whose ``field_type`` is one of these is treated as
# structurally incapable of carrying sensitive data without an explicit
# Sensitive() marker on its annotation. The list is intentionally narrow:
#  * Built-in primitives that cannot hold arbitrary nested values.
#  * Standard-library scalar value types (datetime, date, UUID, Decimal).
#  * NoneType (Optional unwrap residue).
#
# Conspicuously absent: ``Any``, ``object``, and bare ``dict``/``list``/
# ``tuple``. These are inspection-resistant escape hatches; if a field is
# typed that way and not marked Sensitive, the policy author MUST either
# declare it Sensitive or use a more specific type. The adequacy guard
# fails closed on them.
_NON_REDACTION_ELIGIBLE_SCALARS: frozenset[type] = frozenset(
    {
        str,
        int,
        float,
        bool,
        bytes,
        type(None),
        datetime,
        date,
        UUID,
        Decimal,
    }
)


def _is_non_redaction_eligible_scalar(field_type: object) -> bool:
    """True iff ``field_type`` is a closed-list scalar requiring no Sensitive marker.

    Accepts:
      * Members of ``_NON_REDACTION_ELIGIBLE_SCALARS`` (str, int, ..., UUID).
      * ``Enum`` subclasses (closed sets of named members, structurally scalar).
      * ``Literal[...]`` types (closed sets of literal values).

    Rejects ``Any``, ``object``, parameterised containers, and anything else.
    The rejection is fail-closed: a node that falls through this check and
    has no Sensitive marker triggers the assertion.
    """
    if field_type in _NON_REDACTION_ELIGIBLE_SCALARS:
        return True
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return True
    return get_origin(field_type) is Literal


# ---------------------------------------------------------------------------
# Assertion 1 (§4.4.1, Task 9). Remains xfail until Tasks 13-16 land the
# remaining manifest entries.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Manifest is populated by Tasks 13-16; this assertion is "
        "intentionally red until then. Remove this marker when the manifest "
        "reaches parity with collect_registry_names() — Task 16's final "
        "completion gate. If pytest reports XPASS, the marker must come off "
        "(the suite is now correctly green)."
    ),
)
def test_manifest_keys_equal_registry_names() -> None:
    """Set-equality contract: MANIFEST.keys() == collect_registry_names()."""
    registry = collect_registry_names()
    manifest_keys = frozenset(MANIFEST.keys())
    assert manifest_keys == registry, (
        f"Manifest/registry parity broken. "
        f"Missing from manifest: {sorted(registry - manifest_keys)!r}. "
        f"Extra in manifest (not in registry): {sorted(manifest_keys - registry)!r}."
    )


# ---------------------------------------------------------------------------
# Assertion 2 (§4.4.2 per-entry shape walk).
# ---------------------------------------------------------------------------


def test_per_entry_shape_walk_yields_only_sensitive_or_scalar_fields() -> None:
    """Every node walked on a type-driven entry is Sensitive or a closed-list scalar.

    Spec §4.4.2: for each manifest entry, walk ``argument_model`` (and
    ``response_model`` if present) via the shared iterator; assert each
    ``TraversalNode`` either carries a ``_SensitiveMarker`` in its metadata
    OR has a ``field_type`` from the closed list of non-redaction-eligible
    scalars (``_NON_REDACTION_ELIGIBLE_SCALARS`` + ``Enum`` + ``Literal``).

    Fails closed on ``Any``, ``object``, and any other inspection-resistant
    type: an un-annotated ``Any``-typed field is a redaction-bypass surface
    the policy author must address explicitly (mark Sensitive or replace
    with a typed scalar).

    Declarative entries are validated transitively: ``ToolRedactionPolicy``
    enforces its own invariants in ``__post_init__`` (orphan-summarizer
    rejection, handles_no_sensitive_data + reason struct consistency, and
    known_response_keys non-emptiness when sensitive data is handled). The
    adequacy guard provides a CI-level cross-check by virtue of importing
    the manifest at collection time — any policy that fails its invariants
    raises at module import, failing this test file before assertions run.
    """
    for tool_name, entry in MANIFEST.items():
        # Type-driven entries: walk argument_model (and response_model).
        if entry.argument_model is not None:
            models_to_walk: list[tuple[str, type]] = [("argument_model", entry.argument_model)]
            if entry.response_model is not None:
                models_to_walk.append(("response_model", entry.response_model))
            for surface_label, model_cls in models_to_walk:
                for node in walk_model_schema(model_cls):
                    has_sensitive = any(isinstance(m, _SensitiveMarker) for m in node.metadata)
                    if has_sensitive:
                        continue
                    if _is_non_redaction_eligible_scalar(node.field_type):
                        continue
                    raise AssertionError(
                        f"Tool {tool_name!r} {surface_label} {model_cls.__name__}: "
                        f"TraversalNode at path {node.path!r} has field_type "
                        f"{node.field_type!r} which is neither marked Sensitive() "
                        f"nor in the closed list of non-redaction-eligible scalars "
                        f"({sorted(t.__name__ for t in _NON_REDACTION_ELIGIBLE_SCALARS)!r} "
                        f"+ Enum subclasses + Literal[...]). "
                        f"Either annotate the field with Sensitive() (and a summarizer "
                        f"if needed) or replace the type with a closed-list scalar. "
                        f"Spec §4.4.2 fails closed on Any/object/parameterised "
                        f"containers without explicit Sensitive marking."
                    )
        # Declarative entries: ToolRedactionPolicy.__post_init__ enforces
        # internal consistency at construction time. Importing this test
        # file imports the manifest, which constructs every policy; any
        # invariant violation would already have raised. No further
        # assertion is required here.


# ---------------------------------------------------------------------------
# Assertion 5 (rev-2 M.2, §4.4.2). extra="forbid" on type-driven entries.
# ---------------------------------------------------------------------------


def test_type_driven_entries_use_extra_forbid_model_config() -> None:
    """Every type-driven argument_model declares ``extra="forbid"`` (rev-2 M.2).

    Without ``extra="forbid"``, the model silently accepts keys the walker
    has no record of, breaking the manifest/canonical-arguments parity
    invariant the adequacy guard relies on. See spec §4.4.2 and the
    ``SetSourceArgumentsModel`` docstring (``redaction.py``) for the
    canonical justification.

    The assertion uses ``model_config["extra"] == "forbid"`` (literal
    string, not enum) — pydantic 2.x exposes ``model_config`` as a plain
    dict-like mapping on the model class.
    """
    for tool_name, entry in MANIFEST.items():
        if entry.argument_model is None:
            continue
        extra = entry.argument_model.model_config.get("extra")
        assert extra == "forbid", (
            f"Tool {tool_name!r}: argument_model "
            f"{entry.argument_model.__name__} has model_config['extra']={extra!r}; "
            f"spec §4.4.2 (rev-2 M.2) requires the literal value 'forbid'. "
            f"Without extra='forbid', LLM-supplied keys outside the model's "
            f"declared field set silently slip past the walker and land in "
            f"chat_messages.tool_calls JSON unredacted."
        )


# ---------------------------------------------------------------------------
# Assertion 6 (rev-5 quality W7-PARTIAL).
# sensitive_response_keys ⊆ known_response_keys for declarative entries.
# ---------------------------------------------------------------------------


def test_declarative_sensitive_response_keys_subset_of_known_response_keys() -> None:
    """For each declarative entry with handles_no_sensitive_data=False,
    every key in ``sensitive_response_keys`` is also in ``known_response_keys``.

    Closes rev-5 quality W7-PARTIAL: the dual-typo silent-pass where
    ``sensitive_response_keys=("contents",)`` (misspelled) AND
    ``known_response_keys`` contains the correctly-spelled ``"content"``.
    The Task 16-final-a runtime smoke test cannot catch this without
    payloads exercising the real key; the construction-time adequacy
    assertion catches it without needing any runtime payload.

    The same discipline does not apply to ``sensitive_argument_keys``
    because there is no ``known_argument_keys`` analogue.
    """
    for tool_name, entry in MANIFEST.items():
        if entry.policy is None:
            continue
        policy = entry.policy
        if policy.handles_no_sensitive_data:
            continue
        orphans = set(policy.sensitive_response_keys) - set(policy.known_response_keys)
        assert not orphans, (
            f"Tool {tool_name!r}: sensitive_response_keys contains keys not in "
            f"known_response_keys: {sorted(orphans)!r}. Either the sensitive key "
            f"is misspelled (compare against the actual handler response shape) "
            f"or it must be added to known_response_keys. A 'sensitive' key that "
            f"is not in the known set is a typo silently producing the wrong "
            f"redaction path."
        )


# ---------------------------------------------------------------------------
# Walker-completeness floor-check (rev-3 M1).
# ---------------------------------------------------------------------------


def test_walker_emits_node_for_every_top_level_field() -> None:
    """Floor check: walk_model_schema must yield at least one node whose path
    root matches every top-level field in model_fields. Catches the failure
    mode where the iterator silently drops a field due to an unhandled
    annotation form — a class of bug the shared iterator pattern does NOT
    protect against (it forecloses guard/walker disagreement, not shared
    blind spots). Closes rev-3 M1.
    """

    def _root(path: str) -> str:
        # Strip container-descent suffixes to get the top-level field name.
        return path.split(".")[0].split("[")[0].split("{")[0]

    for tool_name, entry in MANIFEST.items():
        if entry.argument_model is None:
            continue
        expected = set(entry.argument_model.model_fields.keys())
        actual = {_root(n.path) for n in walk_model_schema(entry.argument_model)}
        assert expected <= actual, (
            f"Iterator dropped fields for {tool_name}: missing={expected - actual!r}; "
            f"this means walk_model_schema does not understand a field annotation "
            f"on {entry.argument_model.__name__}. Adequacy guard cannot detect this "
            f"because the runtime walker shares the same blind spot."
        )
