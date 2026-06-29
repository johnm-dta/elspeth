"""Composer-interpretation event contract (L0).

Records the user's discrete decision about an LLM-surfaced interpretation
during composer chat — e.g., the LLM proposes a definition of "cool" for a
rating transform, and the user accepts, amends, opts out of future
surfacings, or abandons the session before responding.

Three structural row shapes exist, distinguished by :class:`InterpretationSource`:

* ``user_approved`` — the LLM surfaced the term, presented a draft, and the
  user resolved it (accepted / amended / abandoned). All interpretation-
  surface fields are populated.
* ``auto_interpreted_opt_out`` — the user clicked "stop asking" before any
  surfacing occurred for this term. Session-level marker rows have no LLM
  consultation; provenance, kind, and surface fields are NULL. Later
  surface-specific opt-out rows may carry kind, surface fields, accepted
  value, and a V2 argument hash when the LLM auto-interprets because the
  session is opted out.
* ``auto_interpreted_no_surfaces`` — rate cap exhausted; the LLM baked the
  interpretation into the prompt itself without surfacing it for review.
  Interpretation-surface fields are NULL (no draft existed), but kind and
  LLM provenance MUST be populated — the LLM *was* consulted, and the audit
  trail records which model produced the silent interpretation.

Layer: L0 (contracts). Imports nothing above. Canonical-JSON serialization
and SHA-256 hashing happen at L3 construction sites (recorders/dispatchers),
not from this module — that keeps the leaf clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class InterpretationChoice(StrEnum):
    """The user's resolution of an interpretation surface.

    CLOSED LIST — adding a value requires (a) amending the interpretation-event contract,
    (b) extending this enum, (c) updating the closed-enum tests, and
    (d) a writer-path audit.

    PENDING               — surfaced but not yet resolved (working state).
    ACCEPTED_AS_DRAFTED   — user accepted the LLM's draft verbatim.
    AMENDED               — user edited the draft and accepted the edit.
    OPTED_OUT             — user clicked "stop asking" for this session.
    ABANDONED             — session ended without resolution (page close,
                            timeout). The audit trail records the absence
                            of decision; it is not a synonym for ``pending``.
    """

    PENDING = "pending"
    ACCEPTED_AS_DRAFTED = "accepted_as_drafted"
    AMENDED = "amended"
    OPTED_OUT = "opted_out"
    ABANDONED = "abandoned"


class InterpretationSource(StrEnum):
    """Structural source of an interpretation event row.

    Closed enum. Adding a value requires (a) amending the interpretation-event
    contract, (b) extending InterpretationSource here, (c) updating the closed-enum tests, and (d) a
    writer-path audit. NO SILENT EXTENSION. See models.py governance block.
    """

    USER_APPROVED = "user_approved"
    AUTO_INTERPRETED_OPT_OUT = "auto_interpreted_opt_out"
    AUTO_INTERPRETED_NO_SURFACES = "auto_interpreted_no_surfaces"


class InterpretationKind(StrEnum):
    """Class of LLM-authored assumption surfaced for review.

    CLOSED LIST - adding a value requires contract amendment, schema update,
    closed-enum tests, and writer-path audit.
    """

    VAGUE_TERM = "vague_term"
    INVENTED_SOURCE = "invented_source"
    LLM_PROMPT_TEMPLATE = "llm_prompt_template"
    PIPELINE_DECISION = "pipeline_decision"
    LLM_MODEL_CHOICE = "llm_model_choice"


_INTERPRETATION_SURFACE_FIELDS: tuple[str, ...] = (
    "composition_state_id",
    "affected_node_id",
    "tool_call_id",
    "user_term",
    "llm_draft",
)
_INTERPRETATION_KIND_FIELD: tuple[str, ...] = ("kind",)
_INTERPRETATION_LLM_PROVENANCE_FIELDS: tuple[str, ...] = (
    "model_identifier",
    "model_version",
    "provider",
    "composer_skill_hash",
)
_INTERPRETATION_SHAPE_FIELDS: tuple[str, ...] = (
    *_INTERPRETATION_SURFACE_FIELDS,
    *_INTERPRETATION_KIND_FIELD,
    *_INTERPRETATION_LLM_PROVENANCE_FIELDS,
)
_INTERPRETATION_SHAPE_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    *_INTERPRETATION_SHAPE_FIELDS,
    "accepted_value",
    "arguments_hash",
    "hash_domain_version",
)
_CHOICES_WITH_ACCEPTED_VALUE: frozenset[InterpretationChoice] = frozenset(
    {
        InterpretationChoice.ACCEPTED_AS_DRAFTED,
        InterpretationChoice.AMENDED,
    }
)
_AUTO_INTERPRETED_SOURCES: frozenset[InterpretationSource] = frozenset(
    {
        InterpretationSource.AUTO_INTERPRETED_OPT_OUT,
        InterpretationSource.AUTO_INTERPRETED_NO_SURFACES,
    }
)


def _validate_enum_member(value: object, enum_type: type[StrEnum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be {enum_type.__name__}, got {type(value).__name__}: {value!r}")


def _shape_violation_message(
    source: InterpretationSource,
    *,
    missing_required_fields: list[str],
    non_null_fields: list[str],
) -> str:
    offender_names = set(missing_required_fields) | set(non_null_fields)
    ordered_offenders = [name for name in _INTERPRETATION_SHAPE_DIAGNOSTIC_FIELDS if name in offender_names]
    detail_parts: list[str] = []
    if missing_required_fields:
        detail_parts.append(f"missing required fields: {', '.join(missing_required_fields)}")
    if non_null_fields:
        detail_parts.append(f"fields must be None: {', '.join(non_null_fields)}")
    return (
        f"InterpretationEventRecord {source.value} violates row shape; "
        f"offending fields: {', '.join(ordered_offenders)}; "
        f"{'; '.join(detail_parts)}"
    )


@dataclass(frozen=True, slots=True)
class InterpretationEventRecord:
    """A discrete user decision about an LLM-surfaced interpretation.

    Tier-1 read-side record: every field is required-or-explicitly-None
    per the schema. Constructors crash loudly on any anomaly.

    Three row shapes exist (see InterpretationSource enum):

    user_approved rows — LLM surfaced the term; user approved or amended:
        composition_state_id -> pipeline-state reference (NOT NULL)
        affected_node_id     -> the LLM-transform node this binds into (NOT NULL)
        tool_call_id         -> provider tool_call_id from the LLM (NOT NULL)
        user_term            -> the original user-provided term, e.g. "cool" (NOT NULL)
        kind                 -> class of LLM-authored assumption (NOT NULL)
        llm_draft            -> the LLM's draft interpretation (NOT NULL)
        accepted_value       -> the user-approved string (None until resolved)
        arguments_hash       -> rfc8785 hash over required fields; None until resolved
        model_identifier     -> e.g., "anthropic/claude-opus-4-7" (NOT NULL)
        model_version        -> provider's reported version string (NOT NULL)
        provider             -> "anthropic", "openai", etc. (NOT NULL)
        composer_skill_hash  -> SHA-256 of pipeline_composer.md content (NOT NULL)

    auto_interpreted_opt_out rows — user clicked "stop asking":
        session-level marker shape:
        (all ten fields above are NULL — no LLM surfacing occurred)
        choice = 'opted_out'; interpretation_source = 'auto_interpreted_opt_out'
        resolved_at records the opt-out timestamp

        surface-specific opt-out shape:
        kind, surface/provenance fields, accepted_value, arguments_hash, and
        hash_domain_version are populated because the LLM auto-interpreted
        after the session-level opt-out. No pending human surface exists; the
        row is born resolved with choice='opted_out'.

    auto_interpreted_no_surfaces rows — rate cap exhausted; LLM baked it in:
        composition_state_id, affected_node_id, tool_call_id, user_term,
        llm_draft are NULL (no surfacing occurred — the rejected request
        never produced a draft or a composition-state binding)
        kind MUST be populated (the rejected review request still had a
        closed interpretation class)
        model_identifier, model_version, provider, composer_skill_hash MUST be
        populated (the LLM was consulted; provenance is required — read from
        the compose-loop snapshot, same source as user_approved rows).
        Asymmetry: interpretation surface fields are NULL; LLM provenance
        is required. See ck_interpretation_events_no_surfaces_shape.
        choice = 'opted_out' semantics; interpretation_source = 'auto_interpreted_no_surfaces'

    Fields always present:
        id, session_id, choice, created_at, resolved_at, actor,
        interpretation_source

    Per the auditability standard (design doc 06 §"Recording the
    interpretation"), all ten of: user_term, kind, llm_draft, accepted_value,
    created_at, actor, composition_state_id, model_identifier, model_version,
    composer_skill_hash are required for user_approved rows. They are
    intentionally NULL for session-level auto_interpreted_opt_out marker rows
    (F-1: no LLM surfacing).

    interpretation_source is the structural mechanism (closed enum) that
    produced the row: USER_APPROVED, AUTO_INTERPRETED_OPT_OUT, or
    AUTO_INTERPRETED_NO_SURFACES.
    """

    id: UUID
    session_id: UUID
    composition_state_id: UUID | None  # None for opted_out rows
    affected_node_id: str | None  # None for opted_out rows
    tool_call_id: str | None  # None for opted_out rows
    user_term: str | None  # None for opted_out rows
    kind: InterpretationKind | None
    llm_draft: str | None  # None for opted_out rows
    accepted_value: str | None  # None until resolved (or for opted_out)
    choice: InterpretationChoice
    created_at: datetime
    resolved_at: datetime | None
    actor: str  # user identity at resolution
    # F-1: audit provenance fields are nullable — NULL for
    # auto_interpreted_opt_out rows (no LLM was consulted).
    model_identifier: str | None  # e.g., "anthropic/claude-opus-4-7"
    model_version: str | None  # provider-reported version string
    provider: str | None  # "anthropic", "openai", etc.
    composer_skill_hash: str | None  # SHA-256 of pipeline_composer.md
    arguments_hash: str | None  # rfc8785 hash over required fields; None until resolved
    hash_domain_version: str | None  # 'v2' once resolved by current writers; None for opt-out/pending
    interpretation_source: InterpretationSource
    # F-19: runtime model snapshot at resolve time (may differ from composer model).
    runtime_model_identifier_at_resolve: str | None
    runtime_model_version_at_resolve: str | None
    # Cross-DB hash anchor (Option A). NULL until resolved; NULL for
    # auto_interpreted_opt_out rows (no prompt template is patched). For
    # resolved user_approved rows, this is the SHA-256 of the resolved
    # prompt-template string, computed at resolve time using stable_hash()
    # from contracts/hashing.py. NOT part of INTERPRETATION_HASH_DOMAIN_V2.
    resolved_prompt_template_hash: str | None

    def __post_init__(self) -> None:
        """Validate Tier-1 row-shape invariants at construction time."""
        _validate_enum_member(self.choice, InterpretationChoice, "choice")
        _validate_enum_member(self.interpretation_source, InterpretationSource, "interpretation_source")
        if self.kind is not None:
            _validate_enum_member(self.kind, InterpretationKind, "kind")
        self._validate_source_shape()
        self._validate_auto_source_choice()
        self._validate_surface_opt_out_hash_domain_version()
        self._validate_choice_status_fields()
        self._validate_hash_domain_pair()

    def _validate_source_shape(self) -> None:
        missing_required_fields: list[str] = []
        non_null_fields: list[str] = []

        if self.interpretation_source is InterpretationSource.USER_APPROVED:
            missing_required_fields = [name for name in _INTERPRETATION_SHAPE_FIELDS if getattr(self, name) is None]
        elif self.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT:
            if self.kind is None:
                marker_null_fields = (
                    *_INTERPRETATION_SURFACE_FIELDS,
                    *_INTERPRETATION_LLM_PROVENANCE_FIELDS,
                    "accepted_value",
                    "arguments_hash",
                    "hash_domain_version",
                )
                non_null_fields = [name for name in marker_null_fields if getattr(self, name) is not None]
            else:
                surface_opt_out_required_fields = (
                    *_INTERPRETATION_SURFACE_FIELDS,
                    *_INTERPRETATION_LLM_PROVENANCE_FIELDS,
                    "accepted_value",
                    "arguments_hash",
                    "hash_domain_version",
                )
                missing_required_fields = [name for name in surface_opt_out_required_fields if getattr(self, name) is None]
        elif self.interpretation_source is InterpretationSource.AUTO_INTERPRETED_NO_SURFACES:
            non_null_fields = [name for name in _INTERPRETATION_SURFACE_FIELDS if getattr(self, name) is not None]
            missing_required_fields = [
                name for name in (*_INTERPRETATION_KIND_FIELD, *_INTERPRETATION_LLM_PROVENANCE_FIELDS) if getattr(self, name) is None
            ]

        if missing_required_fields or non_null_fields:
            raise ValueError(
                _shape_violation_message(
                    self.interpretation_source,
                    missing_required_fields=missing_required_fields,
                    non_null_fields=non_null_fields,
                )
            )

    def _validate_auto_source_choice(self) -> None:
        if self.interpretation_source in _AUTO_INTERPRETED_SOURCES and self.choice is not InterpretationChoice.OPTED_OUT:
            raise ValueError(f"InterpretationEventRecord {self.interpretation_source.value} rows must use choice='opted_out'")

    def _validate_surface_opt_out_hash_domain_version(self) -> None:
        if (
            self.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT
            and self.kind is not None
            and self.hash_domain_version != "v2"
        ):
            raise ValueError("InterpretationEventRecord surface-specific auto_interpreted_opt_out rows must use hash_domain_version='v2'")

    def _validate_choice_status_fields(self) -> None:
        if (self.choice is InterpretationChoice.PENDING) != (self.resolved_at is None):
            raise ValueError(
                "InterpretationEventRecord choice/resolved_at invariant violated: "
                "choice='pending' must have resolved_at=None and all other choices must have resolved_at populated"
            )

        requires_accepted_value = self.choice in _CHOICES_WITH_ACCEPTED_VALUE or (
            self.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT and self.kind is not None
        )
        if requires_accepted_value != (self.accepted_value is not None):
            raise ValueError(
                "InterpretationEventRecord choice/accepted_value invariant violated: "
                "accepted_value is populated only for accepted_as_drafted/amended choices "
                "or surface-specific auto_interpreted_opt_out rows"
            )

    def _validate_hash_domain_pair(self) -> None:
        if (self.arguments_hash is None) != (self.hash_domain_version is None):
            raise ValueError(
                "InterpretationEventRecord arguments_hash/hash_domain_version invariant violated: "
                "arguments_hash and hash_domain_version must both be None or both be populated"
            )


# INTERPRETATION_HASH_DOMAIN_V1 is retained for historical/read-side
# compatibility only. New writer paths use INTERPRETATION_HASH_DOMAIN_V2.
INTERPRETATION_HASH_DOMAIN_V1: frozenset[str] = frozenset(
    {
        "session_id",
        "composition_state_id",
        "affected_node_id",
        "tool_call_id",
        "user_term",
        "llm_draft",
        "accepted_value",
        "actor",
        "model_identifier",
        "model_version",
        "provider",
        "composer_skill_hash",
    }
)

# INTERPRETATION_HASH_DOMAIN_V2: the closed active field set used to compute
# arguments_hash for interpretation events. The hashing function reads from
# this constant ONLY — adding a field to InterpretationEventRecord without
# adding it here leaves the new field out of the hash silently.
INTERPRETATION_HASH_DOMAIN_V2: frozenset[str] = frozenset(
    {
        "session_id",
        "composition_state_id",
        "affected_node_id",
        "tool_call_id",
        "user_term",
        "kind",
        "llm_draft",
        "accepted_value",
        "actor",
        "model_identifier",
        "model_version",
        "provider",
        "composer_skill_hash",
    }
)
