"""Plugin-declared semantic contracts.

L0 module (contracts layer). Imports nothing above L0.

Vocabulary is intentionally CLOSED. Additions require design review and
a plan amendment — adding enum values lazily is exactly how the project
ends up rebuilding ad hoc runtime validation as expanding prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from elspeth.contracts.freeze import freeze_fields

_FIELD_SEMANTIC_SEVERITIES: frozenset[str] = frozenset({"high", "medium", "low"})


def _require_non_empty_str(value: object, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}: {value!r}")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_optional_non_empty_str(value: object, field_name: str) -> None:
    if value is None:
        return
    _require_non_empty_str(value, field_name)


def _require_string_tuple(value: object, field_name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be tuple[str, ...], got {type(value).__name__}: {value!r}")
    for idx, item in enumerate(value):
        _require_non_empty_str(item, f"{field_name}[{idx}]")


def _require_enum_member(value: object, enum_type: type[StrEnum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be {enum_type.__name__}, got {type(value).__name__}: {value!r}")


def _require_enum_frozenset(value: object, enum_type: type[StrEnum], field_name: str) -> None:
    if type(value) is not frozenset:
        raise TypeError(f"{field_name} must be frozenset[{enum_type.__name__}], got {type(value).__name__}: {value!r}")
    for item in value:
        _require_enum_member(item, enum_type, f"{field_name} item")


def _require_record_tuple(value: object, record_type: type[object], field_name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be tuple[{record_type.__name__}, ...], got {type(value).__name__}: {value!r}")
    for idx, item in enumerate(value):
        if type(item) is not record_type:
            raise TypeError(f"{field_name}[{idx}] must be {record_type.__name__}, got {type(item).__name__}: {item!r}")


def _require_field_semantic_severity(value: object, field_name: str) -> None:
    _require_non_empty_str(value, field_name)
    if value not in _FIELD_SEMANTIC_SEVERITIES:
        raise ValueError(f"{field_name} must be one of {sorted(_FIELD_SEMANTIC_SEVERITIES)!r}, got {value!r}")


class ContentKind(StrEnum):
    """The kind of content a field carries."""

    UNKNOWN = "unknown"
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML_RAW = "html_raw"
    JSON_STRUCTURED = "json_structured"
    BINARY = "binary"


class TextFraming(StrEnum):
    """How a text-bearing field is framed for downstream line operations."""

    UNKNOWN = "unknown"
    NOT_TEXT = "not_text"
    COMPACT = "compact"
    NEWLINE_FRAMED = "newline_framed"
    LINE_COMPATIBLE = "line_compatible"


class SemanticValueType(StrEnum):
    """Semantic runtime shape of a field value.

    This is intentionally narrower than schema-contract python_type. Schema
    contracts preserve checkpoint-safe primitive field types; semantic value
    type describes plugin-level structural promises such as "this field is a
    real list-shaped value, not JSON-looking text".
    """

    UNKNOWN = "unknown"
    STR = "str"
    LIST = "list"


class UnknownSemanticPolicy(StrEnum):
    """How a consumer treats an UNKNOWN producer fact for a required field.

    Phase 1 line_explode uses FAIL — every producer that semantically
    feeds it must declare semantics. WARN and ALLOW are present for
    future consumers but are not used in Phase 1.
    """

    ALLOW = "allow"
    WARN = "warn"
    FAIL = "fail"


class SemanticOutcome(StrEnum):
    """Result of comparing producer facts to a consumer requirement."""

    SATISFIED = "satisfied"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FieldSemanticFacts:
    """Structured facts a producer declares about a field it emits.

    All container fields are tuples / enum values. ``configured_by``
    names option paths that influenced this fact; it MUST contain only
    safe option names, never values, URLs, headers, prompts, row data,
    or exception text.
    """

    field_name: str
    content_kind: ContentKind
    text_framing: TextFraming = TextFraming.UNKNOWN
    value_type: SemanticValueType = SemanticValueType.UNKNOWN
    fact_code: str = "field_semantics"
    configured_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        freeze_fields(self, "configured_by")
        _require_non_empty_str(self.field_name, "field_name")
        _require_enum_member(self.content_kind, ContentKind, "content_kind")
        _require_enum_member(self.text_framing, TextFraming, "text_framing")
        _require_enum_member(self.value_type, SemanticValueType, "value_type")
        _require_non_empty_str(self.fact_code, "fact_code")
        _require_string_tuple(self.configured_by, "configured_by")


@dataclass(frozen=True, slots=True)
class OutputSemanticDeclaration:
    """A producer's full semantic facts across the fields it emits."""

    fields: tuple[FieldSemanticFacts, ...] = ()

    def __post_init__(self) -> None:
        freeze_fields(self, "fields")
        _require_record_tuple(self.fields, FieldSemanticFacts, "fields")


@dataclass(frozen=True, slots=True)
class FieldSemanticRequirement:
    """Structured requirements a consumer declares for a field it consumes."""

    field_name: str
    accepted_content_kinds: frozenset[ContentKind]
    accepted_text_framings: frozenset[TextFraming]
    requirement_code: str
    accepted_value_types: frozenset[SemanticValueType] = frozenset()
    severity: str = "high"
    unknown_policy: UnknownSemanticPolicy = UnknownSemanticPolicy.FAIL
    configured_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # accepted_*_kinds/framings annotated as frozenset, but callers can
        # pass set/list. freeze_fields coerces set -> frozenset and
        # list -> tuple. Identity-preserving when already deeply frozen.
        freeze_fields(
            self,
            "accepted_content_kinds",
            "accepted_text_framings",
            "accepted_value_types",
            "configured_by",
        )
        _require_non_empty_str(self.field_name, "field_name")
        _require_enum_frozenset(self.accepted_content_kinds, ContentKind, "accepted_content_kinds")
        _require_enum_frozenset(self.accepted_text_framings, TextFraming, "accepted_text_framings")
        _require_non_empty_str(self.requirement_code, "requirement_code")
        _require_enum_frozenset(self.accepted_value_types, SemanticValueType, "accepted_value_types")
        _require_field_semantic_severity(self.severity, "severity")
        _require_enum_member(self.unknown_policy, UnknownSemanticPolicy, "unknown_policy")
        _require_string_tuple(self.configured_by, "configured_by")


@dataclass(frozen=True, slots=True)
class InputSemanticRequirements:
    """A consumer's full semantic requirements across the fields it consumes."""

    fields: tuple[FieldSemanticRequirement, ...] = ()

    def __post_init__(self) -> None:
        freeze_fields(self, "fields")
        _require_record_tuple(self.fields, FieldSemanticRequirement, "fields")


@dataclass(frozen=True, slots=True)
class SemanticEdgeContract:
    """Per-edge result of comparing producer facts to consumer requirement.

    consumer_plugin is REQUIRED — assistance lookup MUST address a
    specific plugin class, not iterate every registered transform.
    producer_plugin is optional because some producers (e.g., source)
    are not registered transform classes.
    """

    from_id: str
    to_id: str
    consumer_plugin: str
    producer_plugin: str | None
    producer_field: str
    consumer_field: str
    producer_facts: FieldSemanticFacts | None
    requirement: FieldSemanticRequirement
    outcome: SemanticOutcome

    def __post_init__(self) -> None:
        _require_non_empty_str(self.from_id, "from_id")
        _require_non_empty_str(self.to_id, "to_id")
        _require_non_empty_str(self.consumer_plugin, "consumer_plugin")
        _require_optional_non_empty_str(self.producer_plugin, "producer_plugin")
        _require_non_empty_str(self.producer_field, "producer_field")
        _require_non_empty_str(self.consumer_field, "consumer_field")
        if self.producer_facts is not None and type(self.producer_facts) is not FieldSemanticFacts:
            raise TypeError(
                f"producer_facts must be FieldSemanticFacts or None, got {type(self.producer_facts).__name__}: {self.producer_facts!r}"
            )
        if type(self.requirement) is not FieldSemanticRequirement:
            raise TypeError(f"requirement must be FieldSemanticRequirement, got {type(self.requirement).__name__}: {self.requirement!r}")
        _require_enum_member(self.outcome, SemanticOutcome, "outcome")


def compare_semantic(
    facts: FieldSemanticFacts | None,
    requirement: FieldSemanticRequirement,
) -> SemanticOutcome:
    """Compare producer facts to a consumer requirement.

    Empty accepted sets mean "dimension unconstrained". For constrained
    dimensions, concrete mismatches beat UNKNOWN: a producer that declares
    value_type=str is definitively incompatible with a consumer requiring
    value_type=list even if its content_kind/text_framing are unknown.
    """
    if facts is None:
        return SemanticOutcome.UNKNOWN

    dimension_outcomes: list[SemanticOutcome] = []
    if requirement.accepted_content_kinds:
        if facts.content_kind is ContentKind.UNKNOWN:
            dimension_outcomes.append(SemanticOutcome.UNKNOWN)
        elif facts.content_kind not in requirement.accepted_content_kinds:
            dimension_outcomes.append(SemanticOutcome.CONFLICT)
        else:
            dimension_outcomes.append(SemanticOutcome.SATISFIED)

    if requirement.accepted_text_framings:
        if facts.text_framing is TextFraming.UNKNOWN:
            dimension_outcomes.append(SemanticOutcome.UNKNOWN)
        elif facts.text_framing not in requirement.accepted_text_framings:
            dimension_outcomes.append(SemanticOutcome.CONFLICT)
        else:
            dimension_outcomes.append(SemanticOutcome.SATISFIED)

    if requirement.accepted_value_types:
        if facts.value_type is SemanticValueType.UNKNOWN:
            dimension_outcomes.append(SemanticOutcome.UNKNOWN)
        elif facts.value_type not in requirement.accepted_value_types:
            dimension_outcomes.append(SemanticOutcome.CONFLICT)
        else:
            dimension_outcomes.append(SemanticOutcome.SATISFIED)

    if SemanticOutcome.CONFLICT in dimension_outcomes:
        return SemanticOutcome.CONFLICT
    if SemanticOutcome.UNKNOWN in dimension_outcomes:
        return SemanticOutcome.UNKNOWN
    return SemanticOutcome.SATISFIED
