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
  surfacing occurred for this term. No LLM was consulted; provenance and
  surface fields are NULL.
* ``auto_interpreted_no_surfaces`` — rate cap exhausted; the LLM baked the
  interpretation into the prompt itself without surfacing it for review.
  Interpretation-surface fields are NULL (no draft existed), but LLM
  provenance MUST be populated — the LLM *was* consulted, and the audit
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

    CLOSED LIST — adding a value requires (a) amending the Phase 5b plan,
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

    Closed enum. Adding a value requires (a) amending this plan, (b) extending
    InterpretationSource here, (c) updating the closed-enum tests, and (d) a
    writer-path audit. NO SILENT EXTENSION. See models.py governance block.
    """

    USER_APPROVED = "user_approved"
    AUTO_INTERPRETED_OPT_OUT = "auto_interpreted_opt_out"
    AUTO_INTERPRETED_NO_SURFACES = "auto_interpreted_no_surfaces"


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
        llm_draft            -> the LLM's draft interpretation (NOT NULL)
        accepted_value       -> the user-approved string (None until resolved)
        arguments_hash       -> rfc8785 hash over required fields; None until resolved
        model_identifier     -> e.g., "anthropic/claude-opus-4-7" (NOT NULL)
        model_version        -> provider's reported version string (NOT NULL)
        provider             -> "anthropic", "openai", etc. (NOT NULL)
        composer_skill_hash  -> SHA-256 of pipeline_composer.md content (NOT NULL)

    auto_interpreted_opt_out rows — user clicked "stop asking":
        (all nine fields above are NULL — no LLM surfacing occurred)
        choice = 'opted_out'; interpretation_source = 'auto_interpreted_opt_out'
        resolved_at records the opt-out timestamp

    auto_interpreted_no_surfaces rows — rate cap exhausted; LLM baked it in:
        composition_state_id, affected_node_id, tool_call_id, user_term,
        llm_draft are NULL (no surfacing occurred — the rejected request
        never produced a draft or a composition-state binding)
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
    interpretation"), all nine of: user_term, llm_draft, accepted_value,
    created_at, actor, composition_state_id, model_identifier, model_version,
    composer_skill_hash are required for user_approved rows. They are
    intentionally NULL for auto_interpreted_opt_out rows (F-1: no LLM surfacing).

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
    hash_domain_version: str | None  # 'v1' once resolved; None for opt-out/pending
    interpretation_source: InterpretationSource
    # F-19: runtime model snapshot at resolve time (may differ from composer model).
    runtime_model_identifier_at_resolve: str | None
    runtime_model_version_at_resolve: str | None
    # Cross-DB hash anchor (Option A). NULL until resolved; NULL for
    # auto_interpreted_opt_out rows (no prompt template is patched). For
    # resolved user_approved rows, this is the SHA-256 of the resolved
    # prompt-template string, computed at resolve time using stable_hash()
    # from contracts/hashing.py. NOT part of INTERPRETATION_HASH_DOMAIN_V1.
    resolved_prompt_template_hash: str | None


# INTERPRETATION_HASH_DOMAIN_V1: the closed set of fields used to compute
# arguments_hash for interpretation events. This constant is the source of
# truth for the v1 hash domain. The hashing function reads from this
# constant ONLY — adding a field to InterpretationEventRecord without
# adding it here leaves the new field out of the hash silently.
#
# To add a field: (1) add it here, (2) bump hash_domain_version to 'v2',
# (3) add a CI test that the new field is in the hash domain.
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
