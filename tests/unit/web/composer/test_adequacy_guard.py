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

import json
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, get_origin
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from elspeth.web.composer.redaction import (
    MANIFEST,
    HandlesNoSensitiveDataReason,
    Sensitive,
    ToolRedaction,
    ToolRedactionPolicy,
    _SensitiveMarker,
    walk_model_schema,
)

from ._adequacy_helpers import (
    _entry_hash,
    collect_registry_names,
    compute_manifest_snapshot,
)

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
# Assertion 1 (§4.4.1, Task 9).
#
# The xfail marker that bracketed this assertion through Tasks 13-16 was
# removed at the end of Task 16 (commit 16-final-b) — the manifest now
# reaches parity with ``collect_registry_names()``.  A regression that
# removes an entry without removing the dispatch dict (or vice versa) must
# fail this assertion at CI.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Assertion 4 (§4.4.3, Task 12). Policy-hash snapshot equality.
# ---------------------------------------------------------------------------


def test_redaction_policy_snapshot_matches_live_manifest() -> None:
    """The committed snapshot file must match the live MANIFEST hash.

    Bootstrap the snapshot via:
        .venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py --write

    Whenever a manifest entry changes (Tasks 13-16 add new entries; Task 19
    is the gate), this test will fail until the snapshot is regenerated and
    the diff is reviewed.

    The snapshot includes a ``sensitive_path_count`` field per entry so the
    Task 18 label gate can read it without a live Python run.
    """
    snapshot_path = Path(__file__).parent / "redaction_policy_snapshot.json"
    expected = json.loads(snapshot_path.read_text())
    actual = compute_manifest_snapshot(MANIFEST)
    assert actual == expected, (
        "Redaction policy snapshot drift. Run "
        "'scripts/cicd/bootstrap_redaction_snapshot.py --write' to regenerate, "
        "and review the diff against your manifest changes before merging."
    )


# ---------------------------------------------------------------------------
# Hash-semantics tests (rev-2 BLOCKER_C, §4.4.3). These exercise the four
# properties of _entry_hash: removed Sensitive flips hash, renamed declarative
# key flips hash, replaced summarizer flips hash, closure-state mutation does
# NOT flip hash (documented false negative).
# ---------------------------------------------------------------------------


def _make_type_driven_entry_with_sensitive() -> ToolRedaction:
    """Minimal type-driven manifest entry with one Sensitive-annotated field."""

    class _ModelWithSensitive(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        secret: Annotated[str, Sensitive()]

    return ToolRedaction(argument_model=_ModelWithSensitive)


def _make_type_driven_entry_without_sensitive() -> ToolRedaction:
    """Identical model shape but WITHOUT the Sensitive annotation on ``secret``."""

    class _ModelWithoutSensitive(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        secret: str  # same field name, same type, no Sensitive annotation

    return ToolRedaction(argument_model=_ModelWithoutSensitive)


def test_removed_sensitive_annotation_flips_hash_for_type_driven_entry() -> None:
    """Removing a ``Sensitive()`` annotation flips the snapshot hash (spec §4.4.3).

    Construct two type-driven manifest entries that are identical in structure
    except that one has ``secret: Annotated[str, Sensitive()]`` and the other
    has ``secret: str``.  The ``_entry_hash`` result must differ between the
    two entries so that a PR that removes a Sensitive marker is detected and
    triggers the label-gate CI step (Task 18).
    """
    entry_with = _make_type_driven_entry_with_sensitive()
    entry_without = _make_type_driven_entry_without_sensitive()

    hash_with = _entry_hash("_test_with_sensitive", entry_with)
    hash_without = _entry_hash("_test_without_sensitive", entry_without)

    assert hash_with["hash"] != hash_without["hash"], (
        "Expected _entry_hash to produce different hashes when the Sensitive() "
        "annotation is present vs absent; got identical hashes. This means "
        "_canonicalise_node is not encoding marker presence — the snapshot "
        "would miss Sensitive() removals and BLOCKER_C is not closed."
    )
    assert hash_with["sensitive_path_count"] == 1
    assert hash_without["sensitive_path_count"] == 0


def _make_declarative_entry_with_key(key: str) -> ToolRedaction:
    """Minimal declarative manifest entry with one sensitive argument key."""
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=(key,),
        known_response_keys=("ok",),
    )
    return ToolRedaction(policy=policy)


def test_renamed_key_in_sensitive_argument_keys_flips_hash_for_declarative_entry() -> None:
    """Renaming a key in ``sensitive_argument_keys`` flips the snapshot hash (§4.4.3).

    A policy with ``sensitive_argument_keys=("storage_path",)`` and one with
    ``sensitive_argument_keys=("connection_string",)`` must produce different
    hashes.  This ensures that renaming a key in the declarative policy surface
    — which changes what data is redacted — is detectable via the snapshot.
    """
    entry_original = _make_declarative_entry_with_key("storage_path")
    entry_renamed = _make_declarative_entry_with_key("connection_string")

    hash_original = _entry_hash("_test_original_key", entry_original)
    hash_renamed = _entry_hash("_test_renamed_key", entry_renamed)

    assert hash_original["hash"] != hash_renamed["hash"], (
        "Expected _entry_hash to produce different hashes when "
        "sensitive_argument_keys differs; got identical hashes. "
        "Renaming a sensitive key changes the redaction surface and must be "
        "visible in the snapshot."
    )


def _summarizer_alpha(v: Any) -> str:
    """First summarizer — distinct from _summarizer_beta."""
    return f"alpha:{v!r}"


def _summarizer_beta(v: Any) -> str:
    """Second summarizer — distinct from _summarizer_alpha."""
    return f"beta:{v!r}"


def _make_type_driven_entry_with_summarizer(
    summarizer_func: Any,
) -> ToolRedaction:
    """Type-driven entry whose Sensitive field carries the given summarizer."""

    class _ModelWithSummarizer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        payload: Annotated[str, Sensitive(summarizer=summarizer_func)]

    return ToolRedaction(argument_model=_ModelWithSummarizer)


def test_replacing_summarizer_with_new_function_flips_hash() -> None:
    """Replacing the summarizer callable with a different function flips the hash (§4.4.3).

    ``_entry_hash`` encodes summarizer identity by fully-qualified name
    (``f"{func.__module__}.{func.__qualname__}"``), not by function ``id()``.
    Two distinct function objects at different fully-qualified names MUST
    produce different hashes so that registering a new summarizer in place of
    an existing one is detectable.
    """
    entry_alpha = _make_type_driven_entry_with_summarizer(_summarizer_alpha)
    entry_beta = _make_type_driven_entry_with_summarizer(_summarizer_beta)

    hash_alpha = _entry_hash("_test_alpha_summarizer", entry_alpha)
    hash_beta = _entry_hash("_test_beta_summarizer", entry_beta)

    assert hash_alpha["hash"] != hash_beta["hash"], (
        "Expected _entry_hash to produce different hashes when the summarizer "
        "function differs; got identical hashes. Replacing a summarizer is a "
        "policy change that must be visible in the snapshot (§4.4.3 / BLOCKER_C)."
    )


def test_closure_state_mutation_false_negative() -> None:
    """DOCUMENTED FALSE-NEGATIVE: closure-state mutation does NOT flip the hash.

    This is a known false-negative class; see spec §4.2.3 / §4.4.3.
    In-place mutation of closure-captured state does not flip the hash.

    A summarizer that closes over a module-level mutable variable (here
    simulated via a mutable container) will produce an unchanged snapshot
    hash even if the closure's effective behaviour changes.  The hash
    captures the summarizer's fully-qualified name; it cannot detect
    mutations to captured objects without inspecting bytecode or executing
    the function — neither of which the snapshot mechanism does.

    The ELSPETH no-legacy-code + offensive-programming disciplines make
    module-level mutable state a code smell that MUST be caught in review.
    Implementers MUST NOT introduce module-level state that summarizers
    close over (spec §4.4.3, §4.2.3).

    This test pins the false-negative explicitly so future reviewers
    understand the limitation and do not rely on the snapshot to catch
    this class of behavioural drift.
    """
    # Use a mutable container to simulate a closed-over module-level variable.
    _closed_state: dict[str, str] = {"prefix": "v1"}

    def _closure_summarizer(v: Any) -> str:
        return f"{_closed_state['prefix']}:{v!r}"

    class _ModelWithClosure(BaseModel):
        model_config = ConfigDict(extra="forbid")
        data: Annotated[str, Sensitive(summarizer=_closure_summarizer)]

    entry = ToolRedaction(argument_model=_ModelWithClosure)

    hash_before = _entry_hash("_test_closure", entry)

    # Mutate the closed-over state — _closure_summarizer now behaves differently.
    _closed_state["prefix"] = "v2"

    hash_after = _entry_hash("_test_closure", entry)

    # The hash is UNCHANGED — this is the documented false-negative.
    assert hash_before["hash"] == hash_after["hash"], (
        "hash_before and hash_after should be identical because in-place "
        "mutation of the closed-over state does not change _closure_summarizer's "
        "fully-qualified name. If they differ, the hash mechanism has changed "
        "and this documented false-negative must be re-evaluated against §4.4.3."
    )
