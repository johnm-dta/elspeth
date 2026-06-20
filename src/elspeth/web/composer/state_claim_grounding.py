"""State-claim grounding: detect un-grounded prose claims and emit corrections.

Path 3 of the demo-window mitigation for issues ``elspeth-c028f7d186``
(composer state-introspection narration unreliable across turns) and
``elspeth-905fe2a3d8`` (verbal acknowledgement without state mutation).
The composer LLM has a documented failure mode where prose contradicts
persisted state in three shapes:

- Forward contradiction: prose claims a field still has its old value
  while state has been mutated to the new value (T4 case).
- Backward contradiction: prose claims a fresh action ("I just fixed")
  while state was unchanged this turn (T5 case).
- Verbal acknowledgement without action: prose agrees with the user
  ("you're right, I'll change that") but no mutation tool was called
  this turn (cells #2/#4 of panel-smoke-2026-05-10).

All three shapes corrupt operator trust because amateur personas
(compliance officer, marketing-ops) read prose as authoritative without
inspecting state. This module detects those un-grounded claims and
produces an ``[ELSPETH-SYSTEM]`` correction suffix that the synthesizer
appends to the assistant prose.

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

Action-claim detection uses four conservative pattern categories,
each with its own anchoring beyond a bare verb to bound the
false-positive cost in multi-turn dialogue (a legitimate reference
to an earlier-turn action paired with a no-mutation turn would
otherwise produce a spurious correction):

- ``_ACTION_CLAIM_PATTERN``: requires the literal token "just"
  ("I just fixed", "I've just changed").
- ``_AGREEMENT_PROMISE_PATTERN``: agreement opener + first-person +
  action verb ("you're right, I'll change that", "yes, I can fix it").
- ``_PRESENT_PERFECT_ACTION_PATTERN``: present perfect
  ("I've fixed", "I have changed") with negation/progressive
  excluded by negative lookahead.
- ``_BARE_PAST_WITH_CONSEQUENCE_PATTERN``: bare simple past followed
  by a consequence clause ("I fixed X so Y is no longer Z"). The
  consequence clause is the disambiguator — a reference to an
  earlier-turn action would not narrate a fresh present-tense
  consequence.

The verifier gate (``if mutation_success_seen or state_changed:
return ()``) is the second line of defence — a regex match on a turn
that legitimately mutated state produces no violation regardless of
the prose pattern that matched.
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

# Action-verb tokens. Two parallel forms:
#  - past tense / past participle: used by the past-tense patterns
#    (``_ACTION_CLAIM_PATTERN``, ``_PRESENT_PERFECT_ACTION_PATTERN``,
#    ``_BARE_PAST_WITH_CONSEQUENCE_PATTERN``).
#  - base form: used by the future/agreement pattern
#    (``_AGREEMENT_PROMISE_PATTERN``).
#
# Verb pairs are kept synchronised by index so the violation explanation
# can always refer to the past-tense form regardless of which pattern
# matched (avoids surfacing tense-inconsistent narration to operators).
_ACTION_VERBS_PAST: Final[tuple[str, ...]] = (
    "fixed",
    "changed",
    "updated",
    "set",
    "configured",
    "switched",
    "toggled",
    "applied",
    "added",
    "removed",
    "adjusted",
    "modified",
    "corrected",
    "repaired",
    # Epistemic completion verb. Issue elspeth-905fe2a3d8 cites
    # "I confirmed" as one of the three example phrases the detector
    # must catch. Treated symmetrically with the action verbs because
    # the operator-trust harm of a false "I confirmed X" is the same
    # ("the change is in place") regardless of whether the model
    # claims to have written X or to have verified X.
    "confirmed",
)
_ACTION_VERBS_BASE: Final[tuple[str, ...]] = (
    "fix",
    "change",
    "update",
    "set",
    "configure",
    "switch",
    "toggle",
    "apply",
    "add",
    "remove",
    "adjust",
    "modify",
    "correct",
    "repair",
    "confirm",
)
_BASE_TO_PAST: Final[dict[str, str]] = dict(zip(_ACTION_VERBS_BASE, _ACTION_VERBS_PAST, strict=True))


def _verb_alternation(verbs: tuple[str, ...]) -> str:
    """Compile a regex alternation of verbs. Tokens are escaped to be safe
    against regex metacharacters even though all current verbs are
    plain alphabetic — the helper documents the contract for future
    additions."""
    return "|".join(re.escape(v) for v in verbs)


# Existing pattern: requires the literal token "just" to anchor the
# claim to this turn. Most precise of the four; lowest false-positive
# cost. Kept for the original T5 case.
_ACTION_CLAIM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bI(?:'ve|\s+have)?\s+just\s+" rf"(?P<verb>{_verb_alternation(_ACTION_VERBS_PAST)})" r"\b",
    re.IGNORECASE,
)

# Agreement opening followed by a first-person commitment to act.
# Matches the panel-evals "you're right, I'll change that to
# rejected_records" / "yes, I can fix that" shape — the model agrees
# verbally without calling any mutation tool. The verb is captured in
# its base form (e.g. ``change``) and rewritten to past tense
# (``changed``) by ``extract_action_claims`` so violation explanations
# stay in one tense.
#
# Conservative anchoring: requires (1) an agreement opener token,
# (2) a connector (comma, dash, colon, em-dash, or whitespace), and
# (3) ``I`` followed optionally by a future/modal auxiliary
# (``'ll``, ``will``, ``can``, ``should``) before the verb. A bare
# "I change" (present tense, no auxiliary) is unusual in agreement
# context but still matches — the verifier gate then decides.
_AGREEMENT_OPENER_ALTERNATION: Final[str] = (
    r"you(?:'re|\s+are)\s+right"
    r"|yes"
    r"|sure"
    r"|absolutely"
    r"|of\s+course"
    r"|got\s+it"
    r"|good\s+catch"
    r"|right(?=\s*[,—\-])"
    r"|as\s+you\s+(?:asked|requested)"
    r"|per\s+your\s+(?:request|ask)"
)
_AGREEMENT_PROMISE_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:{_AGREEMENT_OPENER_ALTERNATION})"
    r"[\s,.\-—:;]+"
    r"I(?:[\s']*(?:'ll|\s+will|\s+can|\s+should))?\s+"
    rf"(?P<verb>{_verb_alternation(_ACTION_VERBS_BASE)})"
    r"\b",
    re.IGNORECASE,
)

# Present-perfect completion claim. "I've fixed", "I have changed".
# Less ambiguous than bare simple past in multi-turn dialogue —
# present perfect English usage strongly implies "before now, possibly
# just now" without explicit prior-turn anchoring.
#
# Negative lookahead excludes:
#  - "I have not <verb>" / "I haven't <verb>" — negation
#  - "I have been <verb>" / "I've been <verb>" — progressive
#    (refers to ongoing/intermittent action across turns)
_PRESENT_PERFECT_ACTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bI(?:'ve|\s+have)\s+"
    r"(?!(?:not|n't|been)\b)"
    rf"(?P<verb>{_verb_alternation(_ACTION_VERBS_PAST)})"
    r"\b",
    re.IGNORECASE,
)

# Bare simple past followed by a present-tense consequence clause.
# Matches the T5 corpus prose ("I fixed the workflow behavior so
# source validation is no longer silently dropping rows from the
# record set"). The consequence clause is the disambiguator — a
# reference to an earlier-turn action would say "I fixed that
# earlier" or "as I changed before", not narrate a fresh
# present-tense consequence.
#
# Anchoring requirements:
#  - sentence start (start-of-string or after ``.``/``!``/``?``)
#  - first-person ``I``
#  - one of the action verbs from the past-tense list
#  - a determiner (``the``/``that``/``this``/``it``/``your``)
#    targeting an object — bare "I fixed everything" doesn't match
#  - a consequence connector within 150 characters
#    (``so``, ``so that``, ``and now``, ``now``, ``is no longer``,
#    ``are no longer``, ``will no longer``, ``is now``, ``are now``,
#    ``will now``)
_BARE_PAST_WITH_CONSEQUENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[.!?]\s+)I\s+"
    rf"(?P<verb>{_verb_alternation(_ACTION_VERBS_PAST)})"
    r"\s+(?:the|that|this|it|your)"
    r"\b[^.!?]{0,150}?"
    r"\b(?:so\s+that|so|and\s+now|is\s+no\s+longer|are\s+no\s+longer|"
    r"will\s+no\s+longer|is\s+now|are\s+now|will\s+now|now)\b",
    re.IGNORECASE,
)


# Each entry: (pattern, value_form). value_form is "past" or "base" —
# matches in base form are rewritten to past tense via _BASE_TO_PAST
# so violation explanations stay in one tense regardless of which
# pattern matched.
_ACTION_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (_ACTION_CLAIM_PATTERN, "past"),
    (_AGREEMENT_PROMISE_PATTERN, "base"),
    (_PRESENT_PERFECT_ACTION_PATTERN, "past"),
    (_BARE_PAST_WITH_CONSEQUENCE_PATTERN, "past"),
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
    """Extract action claims from assistant prose.

    Iterates all four conservative pattern categories defined in this
    module (see module docstring). Base-form verbs from the
    agreement-promise pattern (``change``/``fix``/...) are rewritten
    to their past-tense form (``changed``/``fixed``/...) so violation
    explanations stay in one tense regardless of which pattern matched.

    Spans across patterns may overlap; the verifier
    (``verify_action_claims``) deduplicates on verb so multiple
    matches for the same verb produce at most one violation per turn.
    """
    claims: list[ActionClaim] = []
    for pattern, verb_form in _ACTION_PATTERNS:
        for match in pattern.finditer(prose):
            verb_token = match.group("verb").lower()
            normalised_verb = _BASE_TO_PAST[verb_token] if verb_form == "base" else verb_token
            claims.append(
                ActionClaim(
                    verb=normalised_verb,
                    span=match.span(),
                )
            )
    return tuple(claims)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _lookup_source_field_values(state: CompositionState, field_name: str) -> tuple[str, ...]:
    """Read a scalar field off every source. Returns distinct values."""
    values: list[str] = []
    for source in state.sources.values():
        if field_name == "on_validation_failure":
            values.append(source.on_validation_failure)
        elif field_name == "plugin":
            values.append(source.plugin)
        elif field_name == "on_success":
            values.append(source.on_success)
    return tuple(dict.fromkeys(values))


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
            actuals = _lookup_source_field_values(state, claim.field_name)
            if not actuals:
                continue
            if not any(claim.claimed_value.lower() == actual.lower() for actual in actuals):
                actual_repr = ", ".join(sorted(actuals)) if len(actuals) > 1 else actuals[0]
                violations.append(
                    GroundingViolation(
                        kind="state_claim",
                        field_name=claim.field_name,
                        scope=claim.scope,
                        claimed_value=claim.claimed_value,
                        actual_value=actual_repr,
                        explanation=(
                            f"Prose claims a source's {claim.field_name} is "
                            f"{claim.claimed_value!r}, but configured sources use {actual_repr!r}."
                        ),
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
    """Verify action claims against this turn's mutation activity.

    A violation is emitted when an action-claim is present but neither a
    successful mutation tool was called this turn nor did state actually
    change. Both signals are checked to avoid edge cases where a mutation
    tool succeeded but produced an identity transformation.

    Per the panel-evals T5 case, the LLM claimed "I just fixed the
    workflow behavior" while state was unchanged from T4 (mutation
    happened on T4, not T5). Per the panel-evals cells #2/#4 cases, the
    LLM said "you're right, I'll change that" without calling any
    mutation tool. This detector flags both shapes via the four
    pattern categories in ``extract_action_claims``.
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
                    f"Prose narrates an action ({claim.verb!r}) but no mutation tool succeeded this turn and pipeline state did not change."
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
