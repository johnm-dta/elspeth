"""Adequacy guard — manifest-level structural assertions (spec §4.4).

Assertion 1 (registry-manifest set equality, §4.4.1) is xfail until the
manifest reaches parity with the dispatch registry (Tasks 13-16).

Assertions 2 (per-entry shape walk), 5 (extra="forbid"), 6
(sensitive_response_keys ⊆ known_response_keys), and the walker-completeness
floor-check (rev-3 M1) are enforced unconditionally at every collection of
this file. They trivially pass while MANIFEST holds only set_source; they
begin enforcing real coverage as Tasks 13-16 populate the manifest.

Assertion 3 (mass-copy uniqueness, §4.4.4, Task 11) detects copy-paste
rationale in declarative manifest entries. Two tools may not share an
exact-match ``why_arguments_safe`` (or ``why_responses_safe``) string after
whitespace normalisation. ``sensitive_data_locations`` duplication is
intentionally allowed — some tools genuinely share a location string like
"server-side secret resolver".

The adequacy guard works at MANIFEST-level only. AST inspection of handler
source is forbidden (rev-2 M.3): any handler using ``args = arguments;
args['x']``, destructuring, or helper delegation evades the scan. Tools
requiring mechanical key-coverage guarantees MUST be promoted to type-driven
Pydantic argument models with ``extra="forbid"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal, get_origin
from uuid import UUID

import pytest

from elspeth.web.composer.redaction import (
    MANIFEST,
    HandlesNoSensitiveDataReason,
    ToolRedaction,
    ToolRedactionPolicy,
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


# ---------------------------------------------------------------------------
# Assertion 3 (§4.4.4, Task 11). Mass-copy uniqueness of declarative reasons.
# ---------------------------------------------------------------------------


def _why_text_collisions(
    manifest: Mapping[str, ToolRedaction],
) -> list[tuple[str, str, str]]:
    """Return collisions on ``why_arguments_safe`` or ``why_responses_safe``
    across declarative manifest entries whose policy carries a
    ``HandlesNoSensitiveDataReason`` struct.

    Each element of the returned list is a 3-tuple:
      ``(tool_a, tool_b, normalised_text)``
    where ``tool_a < tool_b`` (lexicographic) so a given pair appears exactly
    once regardless of iteration order.

    Whitespace is normalised with ``" ".join(text.split())`` before
    comparison: this collapses runs of internal whitespace to a single space
    AND strips leading/trailing whitespace in one operation (the standard
    Python idiom, spec §4.4.4).

    Checks BOTH text fields: ``why_arguments_safe`` and ``why_responses_safe``.
    A collision on either field is a finding.

    ``sensitive_data_locations`` is intentionally NOT checked here. Duplicate
    location strings across entries are allowed — some tools genuinely share a
    location (e.g., "server-side secret resolver") and there is no policy value
    in forbidding that coincidence.

    Returns ALL collisions (not just the first pair) so an operator receives
    the full list to fix in one pass.
    """
    # Accumulate, per (field_label, normalised_text), the list of tool names
    # that use that text.  Two lists with > 1 name → collision.
    seen: dict[tuple[str, str], list[str]] = {}

    for tool_name, entry in manifest.items():
        if entry.policy is None:
            continue
        struct = entry.policy.handles_no_sensitive_data_reason_struct
        if struct is None:
            continue
        for field_label, raw_text in (
            ("why_arguments_safe", struct.why_arguments_safe),
            ("why_responses_safe", struct.why_responses_safe),
        ):
            key = (field_label, " ".join(raw_text.split()))
            if key not in seen:
                seen[key] = []
            seen[key].append(tool_name)

    collisions: list[tuple[str, str, str]] = []
    for (_field_label, normalised_text), names in seen.items():
        if len(names) > 1:
            # Emit every pair for that text so the caller sees the full
            # picture; sort to guarantee lexicographic ordering tool_a < tool_b.
            sorted_names = sorted(names)
            for i in range(len(sorted_names)):
                for j in range(i + 1, len(sorted_names)):
                    collisions.append((sorted_names[i], sorted_names[j], normalised_text))

    return collisions


def test_mass_copy_uniqueness_passes_for_live_manifest() -> None:
    """Assertion 3 (§4.4.4): no two declarative entries in MANIFEST share an
    exact-match ``why_arguments_safe`` or ``why_responses_safe`` text (after
    whitespace normalisation).

    This test passes trivially while MANIFEST contains only ``set_source``
    (a type-driven entry with no ``HandlesNoSensitiveDataReason``). It becomes
    a real gate as Tasks 13-16 populate MANIFEST with declarative entries — any
    copy-paste across those entries will be caught here at CI.

    The test is kept unconditional (not xfail) because the spec §4.4.4 contract
    is already valid on a single-entry manifest: zero declarative entries produce
    zero possible collisions, so the assertion passes correctly.
    """
    collisions = _why_text_collisions(MANIFEST)
    assert not collisions, (
        "Mass-copy detected in handles_no_sensitive_data reasons (spec §4.4.4). "
        "The following tool pairs share identical why_arguments_safe or "
        "why_responses_safe text after whitespace normalisation. "
        "Each entry must provide a distinct, concrete justification — "
        "copy-paste rationale defeats the audit trail and indicates a "
        "bulk-placeholder migration. Fix by rewriting each tool's reason "
        "in terms of that tool's specific argument/response surface. "
        f"Collisions: {collisions!r}"
    )


def _make_declarative_entry(
    why_arguments_safe: str,
    why_responses_safe: str,
    sensitive_data_locations: tuple[str, ...] | None = None,
) -> ToolRedaction:
    """Build a minimal declarative ToolRedaction entry for local-fixture tests.

    ``known_response_keys`` must be non-empty when ``handles_no_sensitive_data``
    is False (ToolRedactionPolicy invariant).  We set
    ``handles_no_sensitive_data=True`` with a full ``HandlesNoSensitiveDataReason``
    so the ``known_response_keys`` constraint does not apply and the
    construction stays focused on the uniqueness surface.

    ``sensitive_data_locations`` defaults to a single generic entry so
    ``HandlesNoSensitiveDataReason.__post_init__`` passes its non-empty check.
    """
    locations: tuple[str, ...] = (
        sensitive_data_locations if sensitive_data_locations is not None else ("no LLM-supplied inputs reach this tool",)
    )
    reason = HandlesNoSensitiveDataReason(
        sensitive_data_locations=locations,
        why_arguments_safe=why_arguments_safe,
        why_responses_safe=why_responses_safe,
    )
    policy = ToolRedactionPolicy(
        handles_no_sensitive_data=True,
        handles_no_sensitive_data_reason_struct=reason,
    )
    return ToolRedaction(policy=policy)


def test_mass_copy_uniqueness_flags_exact_match_why_arguments_safe() -> None:
    """Two declarative entries with identical ``why_arguments_safe`` text are
    flagged by ``_why_text_collisions`` (spec §4.4.4).

    ``why_responses_safe`` intentionally differs between the two entries so
    only the argument field triggers the collision — confirming the check
    targets the right text field and does not require both fields to collide.
    """
    shared_args_text = (
        "All arguments are validated against a Pydantic model with extra=forbid; "
        "every field is structural pipeline metadata with no user-supplied content."
    )
    fake_manifest: Mapping[str, ToolRedaction] = {
        "tool_alpha": _make_declarative_entry(
            why_arguments_safe=shared_args_text,
            why_responses_safe=("tool_alpha returns only a boolean success flag; no payload content is present in the response."),
        ),
        "tool_beta": _make_declarative_entry(
            why_arguments_safe=shared_args_text,
            why_responses_safe=("tool_beta returns only a session identifier; no user-supplied data appears in the response shape."),
        ),
    }

    collisions = _why_text_collisions(fake_manifest)

    assert len(collisions) == 1, f"Expected exactly one collision pair, got {len(collisions)}: {collisions!r}"
    tool_a, tool_b, norm_text = collisions[0]
    assert {tool_a, tool_b} == {"tool_alpha", "tool_beta"}, (
        f"Expected the offending pair to be tool_alpha + tool_beta; got {tool_a!r}, {tool_b!r}"
    )
    assert norm_text == " ".join(shared_args_text.split()), f"Normalised collision text mismatch: {norm_text!r}"


def test_mass_copy_uniqueness_flags_whitespace_only_difference_why_arguments_safe() -> None:
    """Two entries whose ``why_arguments_safe`` texts differ only by trailing
    whitespace (or internal extra spaces) are still flagged — the check
    normalises whitespace with ``" ".join(text.split())`` before comparing
    (spec §4.4.4).

    This verifies the normalisation step is applied, not just raw string
    equality: a copy-paste that adds a trailing newline or double-space does
    not bypass the assertion.
    """
    canonical = (
        "All arguments are structural metadata accepted by the Pydantic model; "
        "no user-supplied text or sensitive identifiers appear in any field."
    )
    # Add trailing whitespace to one entry's text — post-normalisation these
    # two strings are identical and must collide.
    with_trailing = canonical + "   \n  "

    fake_manifest: Mapping[str, ToolRedaction] = {
        "tool_gamma": _make_declarative_entry(
            why_arguments_safe=canonical,
            why_responses_safe=(
                "tool_gamma responses contain only a pipeline state summary; no verbatim user data or secret material is returned."
            ),
        ),
        "tool_delta": _make_declarative_entry(
            why_arguments_safe=with_trailing,
            why_responses_safe=(
                "tool_delta responses expose only structural metadata fields; "
                "the response shape is fully known and contains no sensitive content."
            ),
        ),
    }

    collisions = _why_text_collisions(fake_manifest)

    assert len(collisions) == 1, f"Expected exactly one collision (whitespace-normalised match); got {len(collisions)}: {collisions!r}"
    tool_a, tool_b, norm_text = collisions[0]
    assert {tool_a, tool_b} == {"tool_gamma", "tool_delta"}, (
        f"Expected the offending pair to be tool_gamma + tool_delta; got {tool_a!r}, {tool_b!r}"
    )
    # The returned text must be the normalised form (trailing whitespace stripped).
    assert norm_text == " ".join(canonical.split()), f"Normalised text should equal the stripped canonical form; got {norm_text!r}"


def test_mass_copy_uniqueness_flags_collisions_on_why_responses_safe() -> None:
    """Two declarative entries with identical ``why_responses_safe`` text are
    flagged by ``_why_text_collisions`` (spec §4.4.4 — the check applies to
    BOTH ``why_arguments_safe`` AND ``why_responses_safe``).

    ``why_arguments_safe`` intentionally differs between the two entries so
    only the response field triggers the collision — confirming both directions
    are checked independently, not only the argument field.
    """
    shared_resp_text = (
        "Response contains only a boolean 'ok' field and an integer 'count' field; "
        "neither carries user-supplied content or secret material."
    )
    fake_manifest: Mapping[str, ToolRedaction] = {
        "tool_epsilon": _make_declarative_entry(
            why_arguments_safe=(
                "tool_epsilon arguments are a session_id string and an integer limit; "
                "both are structural references with no sensitive content."
            ),
            why_responses_safe=shared_resp_text,
        ),
        "tool_zeta": _make_declarative_entry(
            why_arguments_safe=(
                "tool_zeta accepts only a plugin name string validated against a closed enumeration; no user data appears in this field."
            ),
            why_responses_safe=shared_resp_text,
        ),
    }

    collisions = _why_text_collisions(fake_manifest)

    assert len(collisions) == 1, f"Expected exactly one collision pair on the response field; got {len(collisions)}: {collisions!r}"
    tool_a, tool_b, norm_text = collisions[0]
    assert {tool_a, tool_b} == {"tool_epsilon", "tool_zeta"}, (
        f"Expected the offending pair to be tool_epsilon + tool_zeta; got {tool_a!r}, {tool_b!r}"
    )
    assert norm_text == " ".join(shared_resp_text.split()), f"Normalised collision text mismatch: {norm_text!r}"


def test_mass_copy_uniqueness_allows_identical_sensitive_data_locations() -> None:
    """Identical ``sensitive_data_locations`` values across entries are NOT
    flagged as mass-copy collisions (spec §4.4.4, ``_why_text_collisions``
    docstring).

    Some tools genuinely share a location description — e.g., both may
    declare ``("server-side secret resolver",)`` because that is literally
    where sensitive material lives for both tools. Forbidding this coincidence
    would produce false positives and mislead policy authors into writing
    artificially distinct location strings.

    The uniqueness constraint applies only to the *reasoning* fields
    (``why_arguments_safe``, ``why_responses_safe``); the *location* field
    describes factual topology and permits coincidence.
    """
    shared_location = ("server-side secret resolver",)
    fake_manifest: Mapping[str, ToolRedaction] = {
        "tool_eta": _make_declarative_entry(
            why_arguments_safe=(
                "tool_eta arguments are a read-only filter expression; "
                "the expression is validated against a schema with no string-value "
                "interpolation and cannot encode secret material."
            ),
            why_responses_safe=(
                "tool_eta returns a list of matching session IDs; session IDs are opaque structural references, not sensitive content."
            ),
            sensitive_data_locations=shared_location,
        ),
        "tool_theta": _make_declarative_entry(
            why_arguments_safe=(
                "tool_theta accepts a pipeline name and an integer step index; both are structural coordinates with no sensitivity surface."
            ),
            why_responses_safe=(
                "tool_theta response is a status enum and a timestamp; neither field contains user-supplied data or secret material."
            ),
            sensitive_data_locations=shared_location,
        ),
    }

    collisions = _why_text_collisions(fake_manifest)

    assert not collisions, (
        "Expected zero collisions when only sensitive_data_locations is shared "
        "(identical locations are allowed by spec §4.4.4); "
        f"got {collisions!r}"
    )
