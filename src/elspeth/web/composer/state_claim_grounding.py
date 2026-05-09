"""State-claim grounding: detect un-grounded prose claims and emit corrections.

Path 3 of the demo-window mitigation for issue ``elspeth-c028f7d186``
(composer state-introspection narration unreliable across turns). The
composer LLM has a documented failure mode where prose contradicts
persisted state in either direction:

- Forward contradiction: prose claims a field still has its old value
  while state has been mutated to the new value (T4 case).
- Backward contradiction: prose claims a fresh action ("I just fixed")
  while state was unchanged this turn (T5 case).

Both shapes corrupt operator trust because amateur personas (compliance
officer, marketing-ops) read prose as authoritative without inspecting
state. This module detects those un-grounded claims and produces an
``[ELSPETH-SYSTEM]`` correction suffix that the synthesizer appends to
the assistant prose.

Policy departure: ``service.py:_finalize_no_tool_response`` documents
that "Gate logic (no regex on natural-language text)". That policy
applies to *routing decisions* (which exit shape to take). This module
does not make routing decisions — it produces an additive correction
suffix on the existing happy-path. Prose pattern matching here is
content-grounding, not gate routing. The augmentation contract
(``augmented.startswith(content)``) is preserved.

Scope: v1 targets operator-trust-relevant scalar fields where
misclaims are demo-blockers — source-level ``on_validation_failure``,
``plugin``, ``on_success``; output-level ``on_write_failure``,
``plugin``. Node-level fields (``policy``, ``merge``, ``condition``)
are deferred because disambiguating "which node id?" from natural
language has unacceptable false-positive cost without more structure.

Action-claim detection is intentionally conservative — only "I just
<verb>" / "I'm <verb>ing now" patterns are matched, not bare past
tense. Bare "I changed X" is ambiguous in multi-turn dialogue
(could refer to an earlier turn) and a false-positive correction
would itself corrupt the audit trail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from elspeth.web.composer.state import CompositionState

# ---------------------------------------------------------------------------
# Field catalogue
#
# Each entry is (field_name, allowed_value_tokens). The allowed_value_tokens
# tuple acts as a closed-list of values we will recognise in prose; values
# outside this list are not extracted (avoids false positives from arbitrary
# prose tokens following a field name).
# ---------------------------------------------------------------------------

# Source-level on_validation_failure: the field that triggered the panel-evals
# finding. Allowed values are the engine's documented enum for source
# rejection routing — see SourceSpec.on_validation_failure docstring.
_SOURCE_ON_VALIDATION_FAILURE_VALUES: Final[tuple[str, ...]] = (
    "discard",
    "rejected_records",
    "quarantine",
)

# Source-level on_success: the upstream connection name. Values are
# arbitrary identifiers (operator-chosen), so we cannot use a closed list.
# We DO NOT attempt to extract claimed values for arbitrary-identifier
# fields — too noisy.
#
# Source/output ``plugin``: deferred to a follow-up. Plugin names appear
# in prose much more often in non-state-claim contexts (listing options,
# describing capabilities) than in state-claim contexts, so a closed-list
# match would have unacceptable false-positive cost without additional
# context anchoring.

# Output-level on_write_failure: same shape as on_validation_failure
# (engine-documented routing values) but for sink writes.
_ON_WRITE_FAILURE_VALUES: Final[tuple[str, ...]] = (
    "discard",
    "quarantine",
)

# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------


def _build_field_value_pattern(field_name: str, allowed_values: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a regex matching ``<field_name>[separator]<one of allowed_values>``.

    Separator handles colon, equals, "is", "uses", "is set to", with optional
    backticks/quotes/punctuation around the value. Case-insensitive so prose
    paraphrases like ``On_Validation_Failure`` or ``ON_VALIDATION_FAILURE``
    are matched.

    The pattern is anchored on the field name as a literal word to avoid
    matching substrings (e.g., ``custom_on_validation_failure`` won't match).
    """
    field_re = re.escape(field_name)
    values_alt = "|".join(re.escape(v) for v in allowed_values)
    # Field-name word boundary, optional separators, optional surrounding
    # punctuation/quotes, then a value word.
    #
    # Verbal-separator alternation order matters: ``re`` alternation is
    # leftmost-first, not longest-match. ``is`` must come AFTER the
    # phrases that begin with ``is`` (``is set to``, ``is configured to``)
    # so the engine doesn't commit to the bare ``is`` branch and then
    # fail when the value token turns out to be ``set``.
    pattern = (
        rf"\b{field_re}\b"
        r"\s*"
        r"(?:[`'\"]?\s*(?::|=|->|→)\s*[`'\"]?"
        r"|\s+(?:"
        r"is\s+set\s+to"
        r"|is\s+configured\s+(?:to|as|with)"
        r"|set\s+to"
        r"|configured\s+(?:to|as|with)"
        r"|uses"
        r"|is"
        r")\s+[`'\"]?)"
        rf"({values_alt})"
        r"\b[`'\"]?"
    )
    return re.compile(pattern, re.IGNORECASE)


def _build_value_then_field_pattern(field_name: str, allowed_values: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a regex matching ``<value> for <field_name>`` style paraphrases.

    Catches "uses discard for on_validation_failure" or "set on_validation_failure
    to discard" — natural prose orderings the field-value pattern misses.
    """
    field_re = re.escape(field_name)
    values_alt = "|".join(re.escape(v) for v in allowed_values)
    pattern = (
        rf"\b({values_alt})\b"
        r"\s+(?:for|as\s+the|on)\s+"
        rf"(?:the\s+)?\b{field_re}\b"
    )
    return re.compile(pattern, re.IGNORECASE)


# Compile once at module import. Tuple of (field_name, scope, pattern, value_index).
# scope identifies whether the field lives on source or output(s) for the
# verifier to look up the actual state value.
_FIELD_PATTERNS: Final[tuple[tuple[str, str, re.Pattern[str], int], ...]] = (
    (
        "on_validation_failure",
        "source",
        _build_field_value_pattern("on_validation_failure", _SOURCE_ON_VALIDATION_FAILURE_VALUES),
        1,
    ),
    (
        "on_validation_failure",
        "source",
        _build_value_then_field_pattern("on_validation_failure", _SOURCE_ON_VALIDATION_FAILURE_VALUES),
        1,
    ),
    (
        "on_write_failure",
        "outputs",
        _build_field_value_pattern("on_write_failure", _ON_WRITE_FAILURE_VALUES),
        1,
    ),
    (
        "on_write_failure",
        "outputs",
        _build_value_then_field_pattern("on_write_failure", _ON_WRITE_FAILURE_VALUES),
        1,
    ),
)

# Action-claim pattern. Conservative — requires "just <verb>" or "now"
# modifier to scope the claim to *this* turn.
_ACTION_CLAIM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bI(?:'ve|\s+have)?\s+just\s+"
    r"(?P<verb>fixed|changed|updated|set|configured|switched|toggled|applied|added|removed|adjusted|modified|corrected|repaired)"
    r"\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Claim and violation dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StateClaim:
    """A field=value claim extracted from assistant prose.

    All fields are immutable scalars; ``frozen=True`` is sufficient.
    """

    field_name: str
    scope: str
    claimed_value: str
    span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ActionClaim:
    """A 'just <verb>' completion claim extracted from assistant prose.

    All fields are immutable scalars; ``frozen=True`` is sufficient.
    """

    verb: str
    span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class GroundingViolation:
    """A claim that contradicts persisted state.

    For state-claim violations, ``actual_value`` is the value found in
    state for the claimed field. For action-claim violations,
    ``actual_value`` is ``None`` (the violation is about the action
    not having occurred this turn, not about a specific field).
    """

    kind: str
    field_name: str | None
    scope: str | None
    claimed_value: str | None
    actual_value: str | None
    explanation: str


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_state_claims(prose: str) -> tuple[StateClaim, ...]:
    """Extract field=value claims from assistant prose.

    Returns a tuple of ``StateClaim`` for each match against the field-value
    or value-then-field patterns. Duplicate claims (same field, same value,
    overlapping spans) are deduplicated — a prose excerpt that matches
    both pattern shapes for the same (field, value) pair counts once.
    """
    claims: list[StateClaim] = []
    seen: set[tuple[str, str, str]] = set()
    for field_name, scope, pattern, value_group in _FIELD_PATTERNS:
        for match in pattern.finditer(prose):
            claimed = match.group(value_group).lower()
            key = (field_name, scope, claimed)
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                StateClaim(
                    field_name=field_name,
                    scope=scope,
                    claimed_value=claimed,
                    span=match.span(),
                )
            )
    return tuple(claims)


def extract_action_claims(prose: str) -> tuple[ActionClaim, ...]:
    """Extract 'just <verb>' completion claims from assistant prose.

    Conservative: requires the "just" modifier (or present-progressive
    "now" form, which the regex does not currently support — bare
    "I'm fixing it" without explicit time anchor is too ambiguous).
    """
    claims: list[ActionClaim] = []
    for match in _ACTION_CLAIM_PATTERN.finditer(prose):
        claims.append(
            ActionClaim(
                verb=match.group("verb").lower(),
                span=match.span(),
            )
        )
    return tuple(claims)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _lookup_source_field(state: CompositionState, field_name: str) -> str | None:
    """Read a scalar field off ``state.source``. Returns ``None`` if no source."""
    if state.source is None:
        return None
    if field_name == "on_validation_failure":
        return state.source.on_validation_failure
    if field_name == "plugin":
        return state.source.plugin
    if field_name == "on_success":
        return state.source.on_success
    return None


def _lookup_output_field_values(state: CompositionState, field_name: str) -> tuple[str, ...]:
    """Read a scalar field off every output. Returns the set of distinct values.

    A claim about an output-level field is satisfied if the claimed value
    appears in *any* output's actual value (multi-output pipelines may have
    different routing per sink). Returns the distinct actual values across
    outputs for verifier inspection.
    """
    values: list[str] = []
    for output in state.outputs:
        if field_name == "on_write_failure":
            values.append(output.on_write_failure)
        elif field_name == "plugin":
            values.append(output.plugin)
    return tuple(dict.fromkeys(values))


def verify_state_claims(
    claims: tuple[StateClaim, ...],
    state: CompositionState,
) -> tuple[GroundingViolation, ...]:
    """Compare extracted state-claims against persisted state.

    A violation is emitted when the claimed value cannot be found in any
    in-scope state field. For source-scoped claims, the source's actual
    value must match. For output-scoped claims, the claimed value must
    appear in at least one output's actual value.

    When state has no source (source-scoped claim) or no outputs
    (output-scoped claim), no violation is emitted — the model is
    discussing a hypothetical configuration, not contradicting state.
    The empty-state finalize paths in ``service.py`` cover this case
    via a different exit shape.
    """
    violations: list[GroundingViolation] = []
    for claim in claims:
        if claim.scope == "source":
            actual = _lookup_source_field(state, claim.field_name)
            if actual is None:
                continue
            if actual.lower() != claim.claimed_value.lower():
                violations.append(
                    GroundingViolation(
                        kind="state_claim",
                        field_name=claim.field_name,
                        scope=claim.scope,
                        claimed_value=claim.claimed_value,
                        actual_value=actual,
                        explanation=(f"Prose claims source.{claim.field_name} is {claim.claimed_value!r}, but state has {actual!r}."),
                    )
                )
        elif claim.scope == "outputs":
            actuals = _lookup_output_field_values(state, claim.field_name)
            if not actuals:
                continue
            if not any(claim.claimed_value.lower() == a.lower() for a in actuals):
                actual_repr = ", ".join(sorted(actuals)) if len(actuals) > 1 else actuals[0]
                violations.append(
                    GroundingViolation(
                        kind="state_claim",
                        field_name=claim.field_name,
                        scope=claim.scope,
                        claimed_value=claim.claimed_value,
                        actual_value=actual_repr,
                        explanation=(
                            f"Prose claims an output's {claim.field_name} is "
                            f"{claim.claimed_value!r}, but configured outputs use {actual_repr!r}."
                        ),
                    )
                )
    return tuple(violations)


def verify_action_claims(
    claims: tuple[ActionClaim, ...],
    *,
    mutation_success_seen: bool,
    state_changed: bool,
) -> tuple[GroundingViolation, ...]:
    """Verify "I just <verb>" claims against this turn's mutation activity.

    A violation is emitted when an action-claim is present but neither a
    successful mutation tool was called this turn nor did state actually
    change. Both signals are checked to avoid edge cases where a mutation
    tool succeeded but produced an identity transformation.

    Per the panel-evals T5 case, the LLM claimed "I just fixed the
    workflow behavior" while state was unchanged from T4 (mutation
    happened on T4, not T5). This detector flags exactly that shape.
    """
    if mutation_success_seen or state_changed:
        return ()
    violations: list[GroundingViolation] = []
    seen_verbs: set[str] = set()
    for claim in claims:
        if claim.verb in seen_verbs:
            continue
        seen_verbs.add(claim.verb)
        violations.append(
            GroundingViolation(
                kind="action_claim",
                field_name=None,
                scope=None,
                claimed_value=claim.verb,
                actual_value=None,
                explanation=(
                    f"Prose claims 'I just {claim.verb}' but no successful mutation tool was called this turn and state did not change."
                ),
            )
        )
    return tuple(violations)


# ---------------------------------------------------------------------------
# Top-level entry point and correction formatting
# ---------------------------------------------------------------------------


def check_state_claim_grounding(
    *,
    prose: str,
    state: CompositionState,
    mutation_success_seen: bool,
    state_changed: bool,
) -> tuple[GroundingViolation, ...]:
    """Run all grounding checks against a single assistant turn's prose.

    Returns the combined tuple of violations from both the state-claim
    detector and the action-claim detector. An empty tuple means the
    prose is grounded against state and no correction is needed.

    Args:
        prose: The assistant's prose for this turn (raw model output,
            before any synthesizer post-processing).
        state: The composition state after this turn's tool calls
            have been applied.
        mutation_success_seen: Whether any mutation tool succeeded
            this turn.
        state_changed: Whether ``state.version`` advanced this turn.
            Together with ``mutation_success_seen`` this gates the
            action-claim detector.

    Returns:
        Tuple of GroundingViolation. Caller should append a correction
        suffix when this is non-empty.
    """
    if not prose.strip():
        return ()
    state_claims = extract_state_claims(prose)
    action_claims = extract_action_claims(prose)
    state_violations = verify_state_claims(state_claims, state)
    action_violations = verify_action_claims(
        action_claims,
        mutation_success_seen=mutation_success_seen,
        state_changed=state_changed,
    )
    return state_violations + action_violations


_CORRECTION_HEADER: Final[str] = (
    "[ELSPETH-SYSTEM] The composer's prose above contradicts the actual "
    "pipeline state. The state below is authoritative; the prose may be "
    "stale or refer to an earlier turn."
)


def format_grounding_correction(violations: tuple[GroundingViolation, ...]) -> str:
    """Build the ``[ELSPETH-SYSTEM]`` correction suffix.

    The suffix is intended to be appended to the assistant's prose,
    separated by a horizontal rule, so the augmentation contract
    (``augmented.startswith(content)``) is satisfied at the call site.

    Returns the empty string when there are no violations.
    """
    if not violations:
        return ""
    lines: list[str] = ["", "", "---", "", _CORRECTION_HEADER, ""]
    for v in violations:
        lines.append(f"- {v.explanation}")
    lines.append("")
    lines.append("Re-read the actual state via `get_pipeline_state` before making further claims about pipeline configuration.")
    return "\n".join(lines)


def compose_grounded_message(
    *,
    prose: str,
    violations: tuple[GroundingViolation, ...],
) -> str:
    """Compose ``prose + correction`` such that the result starts with ``prose``.

    Used by the finalizer's grounding-correction exit shape. The augmentation
    prefix invariant — ``augmented.startswith(prose)`` — is the audit-integrity
    contract enforced at the service.py call site.
    """
    suffix = format_grounding_correction(violations)
    if not suffix:
        return prose
    return prose + suffix
