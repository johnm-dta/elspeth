"""Shared input validation utilities for the web layer.

Unicode visibility checks are used by both auth models (identity validation)
and secret schemas (invisible-value rejection). Secret-name constants and
validation also live here so config loading, request validation, and runtime
secret stores share one contract instead of drifting independently.

Phase 5b (F-34) — the interpretation-event helpers
(``_validate_accepted_value_content``, ``_reject_credential_shaped_content``,
``_warn_pii_shaped_content``) live in this module rather than a new
``_validation_helpers.py`` so that
``elspeth.web.sessions.schemas`` (request schema validation) and
``elspeth.web.composer.tools`` (tool-boundary validation) share a single
import path with no peer cross-imports between the two consumer modules.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

# Major Unicode categories that produce visible glyphs.  A string composed
# entirely of characters outside these categories (whitespace, control chars,
# zero-width joiners, format chars, etc.) is "invisible" and rejected.
#
# "M" (Mark/combining) is excluded — combining marks modify a base character
# but a string of only combining marks has no visible base and should be
# rejected.
_VISIBLE_CATEGORIES = frozenset({"L", "N", "P", "S"})

SECRET_NAME_MAX_LENGTH = 256
SECRET_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"
_SECRET_NAME_RE = re.compile(SECRET_NAME_PATTERN)
SERVER_SECRET_RESERVED_PREFIX = "ELSPETH_"
INTERPRETATION_PLACEHOLDER_RE = re.compile(r"\{\{\s*interpretation\s*:\s*([^{}]+?)\s*\}\}")


def has_visible_content(s: str) -> bool:
    """Return True if *s* contains at least one visible character.

    Catches zero-width spaces (U+200B), BOM (U+FEFF), soft hyphens,
    and other invisible characters that ``str.strip()`` does not remove.
    """
    return any(unicodedata.category(c)[0] in _VISIBLE_CATEGORIES for c in s)


def validate_secret_name(name: str, *, field_name: str = "Secret name") -> str:
    """Validate a secret name against the shared web secret contract.

    The contract intentionally matches the existing user-secret API shape:
    a non-empty identifier-style name that fits into audit metadata without
    exceeding the declared schema width.
    """
    if not name:
        raise ValueError(f"{field_name} must not be empty")
    if len(name) > SECRET_NAME_MAX_LENGTH:
        raise ValueError(f"{field_name} must be <= {SECRET_NAME_MAX_LENGTH} characters")
    if _SECRET_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"{field_name} must match {SECRET_NAME_PATTERN}")
    return name


def is_reserved_server_secret_name(name: str) -> bool:
    """Return True when a server-secret name targets ELSPETH internals."""
    return name.startswith(SERVER_SECRET_RESERVED_PREFIX)


# ---------------------------------------------------------------------------
# Phase 5b Task 3 / Task 5 — interpretation-event content validators (F-34)
# ---------------------------------------------------------------------------
#
# These helpers are the boundary-layer guards for two distinct sources of
# untrusted input feeding the interpretation-event audit trail:
#
# * ``amended_value`` — user-typed text submitted via the resolve route.
#   Validated at the schema layer (``InterpretationResolveRequest``).
# * ``user_term`` and ``llm_draft`` — fields drawn from the LLM's tool-call
#   arguments at the ``request_interpretation_review`` tool boundary
#   (Task 5).  Validated as defense-in-depth against prompt-injection
#   payloads embedded in the LLM's draft (F-2).
#
# Both consumers import these helpers from a single module so a future
# audit can confirm the boundary is uniformly enforced.  Tests for
# ``InterpretationResolveRequest`` (Task 3) and the tool boundary
# (Task 5) exercise the same regex set.
#
# Phase 11 deferred: name/address PII detection.  Documenting the deferral
# here so a future agent reading the prefilter does not assume the absent
# pattern is an oversight.


# Credential-shape rejection patterns.  Documented in Phase 5b backend spec
# lines 2073-2082.  Each entry's name is used in unit-test diagnostics and
# in any future telemetry signal.
#
# NOTE on the JWT pattern: the contiguous match (`{4,}\.{4,}\.{4,}` with no
# whitespace in the character class) is deliberately strict.  Benign prose
# with periods ("appealing, well-organized, and easy to use.") contains
# whitespace around each period and therefore cannot match (F-32 negative
# test).  A weaker pattern that allowed whitespace inside the segments
# would produce false positives on every multi-sentence amendment.
_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{40,}")),
    # OpenAI keys (including ``sk-proj-`` and similar prefixes).  The
    # Anthropic pattern is matched first so the more specific prefix wins
    # — that ordering matters only for telemetry-name attribution; both
    # patterns trigger rejection regardless.
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{40,}")),
    (
        "jwt",
        re.compile(r"\b[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b"),
    ),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)

# Credit-card-shaped pattern.  Matched separately so the LUHN check can be
# applied before flagging (avoids false positives on date-like strings of
# the form ``2024-01-02-1234``).
_CREDIT_CARD_RE: re.Pattern[str] = re.compile(r"\b(\d{4})[\s-](\d{4})[\s-](\d{4})[\s-](\d{4})\b")


def _luhn_check(digits: str) -> bool:
    """Return True if ``digits`` (a string of digit characters) is LUHN-valid."""
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# PII patterns that warn (telemetry) but do not reject.  Surfaced for the
# tool-boundary helper in Task 5; defined here so Task 3 and Task 5 share
# a single regex set.  The schema layer (``amended_value``) does NOT emit
# the warning — telemetry is only fired at the tool boundary where the
# enclosing handler has session context for the signal.
_PII_WARNING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")),
    (
        "phone",
        re.compile(r"(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})"),
    ),
    ("ssn_like", re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")),
)


_CREDENTIAL_REJECTION_MESSAGE = "That looks like a credential — please re-enter without secrets."


def _reject_credential_shaped_content(value: str) -> None:
    """Raise ``ValueError`` if ``value`` matches any credential-shape regex.

    Used at two boundaries:
    - ``InterpretationResolveRequest`` schema validator (Task 3): rejects
      a user-typed amendment.
    - ``request_interpretation_review`` tool handler (Task 5): rejects an
      LLM-supplied ``user_term`` or ``llm_draft``.

    The error message is intentionally constant ("That looks like a
    credential — please re-enter without secrets.") so that automated
    log scanners can spot rejection events without parsing the field
    label.  Both consumers wrap or surface this message verbatim.

    Credit-card rejection runs a LUHN check before flagging to avoid
    false positives on date-like strings.  All other patterns are
    structural matches.
    """
    for _name, pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(value):
            raise ValueError(_CREDENTIAL_REJECTION_MESSAGE)
    cc_match = _CREDIT_CARD_RE.search(value)
    if cc_match is not None:
        digits = "".join(cc_match.groups())
        if _luhn_check(digits):
            raise ValueError(_CREDENTIAL_REJECTION_MESSAGE)


@dataclass(frozen=True, slots=True)
class PIIWarning:
    """Operational-telemetry payload for a PII-shaped value detected in input.

    Carries the pattern name (``email`` / ``phone`` / ``ssn_like``) but
    explicitly NOT the matched value — telemetry MUST NOT carry PII even
    when the source field has already been written to the audit DB.  The
    audit DB is the legal record; telemetry is operational visibility and
    carries minimal context per the audit-primacy contract.
    """

    pattern_name: str


def _warn_pii_shaped_content(value: str) -> Iterable[PIIWarning]:
    """Yield a ``PIIWarning`` for each PII pattern that matches ``value``.

    Does NOT raise.  Callers (Task 5 tool boundary) emit a telemetry
    signal per warning and continue processing the request.  The schema
    layer (Task 3) does not call this — telemetry has no handler in
    schema validation and the value is already rejected if it matches a
    credential pattern.
    """
    for name, pattern in _PII_WARNING_PATTERNS:
        if pattern.search(value):
            yield PIIWarning(pattern_name=name)


# Permitted whitespace: horizontal tab (\t), line feed (\n), carriage
# return (\r). All other ASCII control characters (NUL through ETB except
# TAB, then VT, FF, then SI through US, plus DEL) remain rejected.
# Newlines must be permitted because three of the four interpretation
# kinds carry inherently multi-line content: invented_source drafts are
# CSV/JSONL artifacts, llm_prompt_template drafts are templates with
# multi-line instructions, and pipeline_decision drafts are decision
# rationales that span lines. The 1024-character cap is applied per line
# (see below) so the original protection against unbounded single-line
# pastes is preserved.
_FORBIDDEN_CONTROL_CHARS_RE: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _validate_accepted_value_content(value: str) -> None:
    """Validate user/LLM-supplied interpretation-value content.

    Raises ``ValueError`` with an actionable per-condition message on:

    * Jinja-style template metacharacters (``{{`` or ``}}``) — would
      corrupt the placeholder substitution at the consumer.
    * Non-whitespace control characters (NUL, BEL, VT, FF, ESC, DEL,
      etc.). Horizontal tab, line feed, and carriage return are
      explicitly permitted.
    * Any single line longer than 1024 characters — pathological and a
      likely sign of an automated paste-of-large-data. The outer
      8192-character cap is enforced by the field's ``max_length`` on
      the schema; this is the per-line cap.
    * Credential-shaped content (delegates to
      :func:`_reject_credential_shaped_content`).

    Applied at two distinct boundaries (defense-in-depth for F-2):

    * Schema layer — ``InterpretationResolveRequest.amended_value``.
    * Tool boundary — ``request_interpretation_review`` against
      ``llm_draft`` for vague_term and pipeline_decision kinds (Task 5).
    """
    if "{{" in value or "}}" in value:
        raise ValueError("accepted_value must not contain template metacharacters {{ or }}")
    if _FORBIDDEN_CONTROL_CHARS_RE.search(value):
        raise ValueError(
            "accepted_value must not contain non-printable control characters (horizontal tab, newline, and carriage return are permitted)"
        )
    for line in value.splitlines() or [value]:
        if len(line) > 1024:
            raise ValueError("accepted_value has a line exceeding the 1024-character per-line limit")
    _reject_credential_shaped_content(value)
