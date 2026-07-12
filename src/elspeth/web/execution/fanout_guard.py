"""Launch-time guard for LLM fanout and provider spend risk.

The guard evaluates the Tier-1 composition snapshot immediately before run
creation.  When an LLM transform is downstream of a fanout producer, execution
must pause until the caller acknowledges the provider-call risk with the
deterministic token returned by this module.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from elspeth.contracts.freeze import freeze_fields
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.composer.state import CompositionState, NodeSpec, SourceSpec
from elspeth.web.paths import resolve_data_path

FANOUT_GUARD_ERROR_TYPE = "execution_fanout_ack_required"
FANOUT_GUARD_AUDIT_COMMENT = "elspeth_execution_fanout_guard"
LLM_FANOUT_HIGH_CALL_THRESHOLD = 100

_RiskLevel = Literal["medium", "high"]
_ProducerKind = Literal["source", "node"]


class ExecutionFanoutRiskPayload(TypedDict):
    """Transport shape for one execution fanout risk."""

    node_id: str
    provider: str
    model: str | None
    credential_ref: str | None
    estimated_provider_calls: int | None
    provider_calls_per_row: int
    upstream_fanout: list[str]
    risk_level: _RiskLevel
    message: str


class ExecutionFanoutGuardPayload(TypedDict):
    """Transport shape for an execution fanout acknowledgement guard."""

    ack_token: str
    risk_level: _RiskLevel
    summary: str
    risks: list[ExecutionFanoutRiskPayload]


@dataclass(frozen=True, slots=True)
class ExecutionFanoutRisk:
    """One LLM launch risk requiring operator acknowledgement."""

    node_id: str
    provider: str
    model: str | None
    credential_ref: str | None
    estimated_provider_calls: int | None
    provider_calls_per_row: int
    upstream_fanout: Sequence[str]
    risk_level: _RiskLevel
    message: str

    def __post_init__(self) -> None:
        # ``upstream_fanout`` declared as Sequence[str]; producers may
        # pass a list literal which is mutable through the attribute
        # reference without a freeze guard, defeating ``frozen=True``.
        # All scalars are immutable; only the sequence needs the guard.
        freeze_fields(self, "upstream_fanout")

    def to_dict(self) -> ExecutionFanoutRiskPayload:
        return {
            "node_id": self.node_id,
            "provider": self.provider,
            "model": self.model,
            "credential_ref": self.credential_ref,
            "estimated_provider_calls": self.estimated_provider_calls,
            "provider_calls_per_row": self.provider_calls_per_row,
            "upstream_fanout": list(self.upstream_fanout),
            "risk_level": self.risk_level,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ExecutionFanoutGuard:
    """Structured precondition response for high-fanout LLM execution."""

    ack_token: str
    risk_level: _RiskLevel
    summary: str
    risks: Sequence[ExecutionFanoutRisk]

    def __post_init__(self) -> None:
        # ``risks`` declared as Sequence[ExecutionFanoutRisk]. The risk
        # elements are themselves frozen (with their own freeze guards
        # above), but the outer sequence may be a mutable list at the
        # call site — without deep_freeze the guard's ``frozen=True``
        # claim leaks ``risks.append(...)`` mutability.
        freeze_fields(self, "risks")

    def to_dict(self) -> ExecutionFanoutGuardPayload:
        return {
            "ack_token": self.ack_token,
            "risk_level": self.risk_level,
            "summary": self.summary,
            "risks": [risk.to_dict() for risk in self.risks],
        }


class ExecutionFanoutGuardRequired(Exception):
    """Raised when a run requires an LLM fanout acknowledgement."""

    def __init__(self, guard: ExecutionFanoutGuard) -> None:
        self.guard = guard
        super().__init__(guard.summary)


@dataclass(frozen=True, slots=True)
class _Producer:
    kind: _ProducerKind
    node: NodeSpec | None = None
    source_name: str | None = None
    source: SourceSpec | None = None


def _producer_key(producer: _Producer) -> str:
    """Stable identity for cycle guards and deterministic predecessor order.

    Keyed on producer identity (``source:<name>`` / ``node:<id>``), never on a
    connection name — two producers may publish the same declared queue name.
    """
    if producer.kind == "source":
        return f"source:{producer.source_name}"
    if producer.node is None:
        raise RuntimeError("Node producer missing node reference")
    return f"node:{producer.node.id}"


@dataclass(frozen=True)
class _ProducerIndex:
    """Queue-aware producer resolution.

    ``by_connection`` is the ordinary single-producer map (with each declared
    queue installed as the canonical producer of its own id).
    ``queue_predecessors`` holds, per queue id, the distinct upstream producers
    that publish into that queue — in deterministic producer-id order.
    """

    by_connection: Mapping[str, _Producer]
    queue_predecessors: Mapping[str, tuple[_Producer, ...]]


@dataclass(frozen=True, slots=True)
class _FanoutTrace:
    markers: tuple[str, ...]
    source_markers: tuple[str, ...]
    source_estimated_rows: int | None
    has_unknown_cardinality: bool


def evaluate_execution_fanout_guard(
    state: CompositionState,
    *,
    data_dir: str | Path,
) -> ExecutionFanoutGuard | None:
    """Return a guard when the composition can fan out into LLM calls.

    Direct source-to-LLM pipelines are allowed when the source cardinality is
    known and below ``LLM_FANOUT_HIGH_CALL_THRESHOLD``.  Any token-creating
    transform, transform-mode aggregation, or fork gate upstream of an LLM is
    treated as unbounded unless a later implementation can prove a tighter
    source-to-output multiplier.
    """

    producers = _build_producer_index(state)
    nodes_by_id = {node.id: node for node in state.nodes}
    risks: list[ExecutionFanoutRisk] = []

    for node in state.nodes:
        if node.node_type != "transform" or node.plugin != "llm":
            continue

        trace = _trace_upstream_fanout(
            input_label=node.input,
            producers=producers,
            nodes_by_id=nodes_by_id,
            data_dir=Path(data_dir),
        )
        provider_calls_per_row = _provider_calls_per_row(node.options)
        estimated_provider_calls = (
            trace.source_estimated_rows * provider_calls_per_row if trace.source_estimated_rows is not None and not trace.markers else None
        )

        requires_guard = (
            trace.has_unknown_cardinality
            or bool(trace.markers)
            or (estimated_provider_calls is not None and estimated_provider_calls > LLM_FANOUT_HIGH_CALL_THRESHOLD)
        )
        if not requires_guard:
            continue

        risk_level: _RiskLevel = (
            "high"
            if trace.markers or estimated_provider_calls is None or estimated_provider_calls >= LLM_FANOUT_HIGH_CALL_THRESHOLD * 10
            else "medium"
        )
        provider = _string_option(node.options, "provider") or "unknown"
        model = _string_option(node.options, "model") or _string_option(node.options, "deployment_name")
        risks.append(
            ExecutionFanoutRisk(
                node_id=node.id,
                provider=provider,
                model=model,
                credential_ref=_credential_ref(node.options),
                estimated_provider_calls=estimated_provider_calls,
                provider_calls_per_row=provider_calls_per_row,
                upstream_fanout=trace.markers if trace.markers else trace.source_markers,
                risk_level=risk_level,
                message=_risk_message(
                    node_id=node.id,
                    provider=provider,
                    model=model,
                    estimated_provider_calls=estimated_provider_calls,
                    markers=trace.markers,
                ),
            )
        )

    if not risks:
        return None

    risk_dicts = [risk.to_dict() for risk in risks]
    ack_token = stable_hash(
        {
            "kind": "execution_fanout_guard_v1",
            "composition_state": state.to_dict(),
            "risks": risk_dicts,
        }
    )[:32]
    return ExecutionFanoutGuard(
        ack_token=ack_token,
        risk_level="high" if any(risk.risk_level == "high" for risk in risks) else "medium",
        summary=_guard_summary(risks),
        risks=tuple(risks),
    )


def annotate_pipeline_yaml_with_fanout_guard(
    pipeline_yaml: str,
    guard: ExecutionFanoutGuard,
) -> str:
    """Persist an accepted launch guard in the run's YAML launch record."""

    payload = {
        "kind": "execution_fanout_guard_v1",
        "accepted": True,
        "ack_token": guard.ack_token,
        "risk_level": guard.risk_level,
        "summary": guard.summary,
        "risks": [risk.to_dict() for risk in guard.risks],
    }
    return f"# {FANOUT_GUARD_AUDIT_COMMENT}: {canonical_json(payload)}\n{pipeline_yaml}"


def _build_producer_index(state: CompositionState) -> _ProducerIndex:
    queue_ids = {node.id for node in state.nodes if node.node_type == "queue"}
    by_connection: dict[str, _Producer] = {}
    queue_predecessors: dict[str, dict[str, _Producer]] = {queue_id: {} for queue_id in queue_ids}

    def register(label: str, producer: _Producer) -> None:
        # A declared queue accepts many producers under its id: route them into
        # its predecessor set instead of the single-producer map, keyed on
        # stable producer identity so duplicates dedupe deterministically.
        if label in queue_ids and _producer_key(producer) != f"node:{label}":
            queue_predecessors[label].setdefault(_producer_key(producer), producer)
            return
        by_connection[label] = producer

    for source_name, source in state.sources.items():
        if source.on_success != "discard":
            register(source.on_success, _Producer(kind="source", source_name=source_name, source=source))

    for node in state.nodes:
        for label in (node.on_success, node.on_error):
            if label is not None and label != "discard":
                register(label, _Producer(kind="node", node=node))
        if node.routes is not None:
            for label in node.routes.values():
                if label != "discard":
                    register(label, _Producer(kind="node", node=node))
        if node.fork_to is not None:
            for label in node.fork_to:
                if label != "discard":
                    register(label, _Producer(kind="node", node=node))

    # Install each declared queue as the canonical producer of its own id.
    for node in state.nodes:
        if node.node_type == "queue":
            by_connection[node.id] = _Producer(kind="node", node=node)

    frozen_predecessors = {
        queue_id: tuple(predecessors[key] for key in sorted(predecessors)) for queue_id, predecessors in queue_predecessors.items()
    }
    return _ProducerIndex(by_connection=by_connection, queue_predecessors=frozen_predecessors)


def _trace_upstream_fanout(
    *,
    input_label: str,
    producers: _ProducerIndex,
    nodes_by_id: Mapping[str, NodeSpec],
    data_dir: Path,
) -> _FanoutTrace:
    markers: list[str] = []
    source_markers: list[str] = []
    source_estimated_rows: int | None = None
    unknown_source_seen = False
    has_unknown_cardinality = False
    # Cycle guard keyed on stable producer identity, NOT connection name — a
    # queue's predecessors are distinct producers that share the queue's name.
    visited: set[str] = set()

    def record_source(producer: _Producer) -> None:
        nonlocal source_estimated_rows, unknown_source_seen, has_unknown_cardinality
        if producer.source is None or producer.source_name is None:
            raise RuntimeError("Source producer missing source reference")
        estimated_rows = _estimate_source_rows(producer.source, data_dir=data_dir)
        marker_prefix = f"source:{producer.source_name}:{producer.source.plugin}:estimated_rows="
        if estimated_rows is None:
            # An unknown-cardinality source keeps the sum unknowable and pins
            # the risk high; it is never allowed to silently vanish.
            unknown_source_seen = True
            has_unknown_cardinality = True
            source_estimated_rows = None
            source_markers.append(f"{marker_prefix}unknown")
            return
        if unknown_source_seen:
            source_estimated_rows = None
        elif source_estimated_rows is not None:
            source_estimated_rows += estimated_rows
        else:
            source_estimated_rows = estimated_rows
        source_markers.append(f"{marker_prefix}{estimated_rows}")

    def walk_label(label: str) -> None:
        producer = producers.by_connection.get(label)
        if producer is not None:
            walk_producer(producer)

    def walk_producer(producer: _Producer) -> None:
        key = _producer_key(producer)
        if key in visited:
            return
        visited.add(key)

        if producer.kind == "source":
            record_source(producer)
            return

        node = producer.node
        if node is None:
            raise RuntimeError("Node producer missing node reference")

        if node.node_type == "queue":
            # A queue fans in every predecessor; traverse them all so no
            # upstream cardinality or token-creating path is lost behind it.
            for predecessor in producers.queue_predecessors.get(node.id, ()):
                walk_producer(predecessor)
            return

        marker = _fanout_marker_for_node(node)
        if marker is not None:
            markers.append(marker)

        if node.node_type == "coalesce":
            if node.input:
                walk_label(node.input)
            for branch in node.branches or ():
                walk_label(branch)
            return

        walk_label(node.input)

    walk_label(input_label)
    return _FanoutTrace(
        markers=tuple(dict.fromkeys(markers)),
        source_markers=tuple(dict.fromkeys(source_markers)),
        source_estimated_rows=source_estimated_rows,
        has_unknown_cardinality=has_unknown_cardinality,
    )


def _fanout_marker_for_node(node: NodeSpec) -> str | None:
    if node.node_type == "transform":
        if node.plugin is None:
            raise RuntimeError(f"Transform node {node.id!r} has no plugin")
        transform_cls = get_shared_plugin_manager().get_transform_by_name(node.plugin)
        if transform_cls.creates_tokens:
            return f"transform:{node.id}:{node.plugin}"
        return None

    if node.node_type == "aggregation" and node.output_mode == "transform":
        return f"aggregation:{node.id}:output_mode=transform"

    if node.node_type == "gate" and node.fork_to is not None and len(node.fork_to) > 1:
        return f"gate:{node.id}:fork_to={len(node.fork_to)}"

    return None


def _provider_calls_per_row(options: Mapping[str, Any]) -> int:
    queries = options["queries"] if "queries" in options else None
    if queries is None:
        return 1
    if isinstance(queries, Mapping):
        return max(len(queries), 1)
    if isinstance(queries, Sequence) and not isinstance(queries, str | bytes | bytearray):
        return max(len(queries), 1)
    return 1


def _estimate_source_rows(source: SourceSpec, *, data_dir: Path) -> int | None:
    path = _source_path(source, data_dir=data_dir)
    if path is None:
        return _remote_source_limit(source)

    try:
        if source.plugin == "text":
            return _count_text_source_rows(path, source.options)
        if source.plugin == "csv":
            return _count_csv_source_rows(path, source.options)
        if source.plugin == "json":
            return _count_json_source_rows(path, source.options)
    except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError):
        return None
    return None


def _source_path(source: SourceSpec, *, data_dir: Path) -> Path | None:
    raw_path = source.options["path"] if "path" in source.options else source.options["file"] if "file" in source.options else None
    if not isinstance(raw_path, str):
        return None
    return resolve_data_path(raw_path, str(data_dir))


def _remote_source_limit(source: SourceSpec) -> int | None:
    for key in ("top", "limit", "max_rows"):
        raw_value = source.options[key] if key in source.options else None
        if isinstance(raw_value, int):
            return raw_value
    return None


def _count_text_source_rows(path: Path, options: Mapping[str, Any]) -> int:
    encoding = _string_option(options, "encoding") or "utf-8"
    skip_blank_lines = bool(options["skip_blank_lines"]) if "skip_blank_lines" in options else True
    strip_whitespace = bool(options["strip_whitespace"]) if "strip_whitespace" in options else True
    count = 0
    with path.open(encoding=encoding, errors="surrogateescape", newline="") as handle:
        for raw_line in handle:
            value = raw_line.rstrip("\r\n")
            if strip_whitespace:
                value = value.strip()
            if skip_blank_lines and value == "":
                continue
            count += 1
    return count


def _count_csv_source_rows(path: Path, options: Mapping[str, Any]) -> int | None:
    encoding = _string_option(options, "encoding") or "utf-8"
    delimiter = _string_option(options, "delimiter") or ","
    skip_rows = options["skip_rows"] if "skip_rows" in options and isinstance(options["skip_rows"], int) else 0
    columns = options["columns"] if "columns" in options else None
    with path.open(encoding=encoding, errors="surrogateescape", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, strict=True)
        for _ in range(skip_rows):
            next(reader, None)
        if columns is None and next(reader, None) is None:
            return 0
        return sum(1 for _ in reader)


def _count_json_source_rows(path: Path, options: Mapping[str, Any]) -> int | None:
    encoding = _string_option(options, "encoding") or "utf-8"
    raw_format = _string_option(options, "format")
    fmt = raw_format or ("jsonl" if path.suffix == ".jsonl" else "json")
    data_key = _string_option(options, "data_key")

    if fmt == "jsonl":
        with path.open(encoding=encoding) as handle:
            return sum(1 for line in handle if line.strip())

    with path.open(encoding=encoding) as handle:
        payload = json.load(handle)
    if data_key is not None:
        if not isinstance(payload, Mapping):
            return None
        payload = payload[data_key] if data_key in payload else None
    if isinstance(payload, list):
        return len(payload)
    return None


def _string_option(options: Mapping[str, Any], key: str) -> str | None:
    value = options[key] if key in options else None
    if isinstance(value, str) and value.strip():
        return value
    return None


def _credential_ref(options: Mapping[str, Any]) -> str | None:
    raw_api_key = options["api_key"] if "api_key" in options else None
    if isinstance(raw_api_key, Mapping):
        secret_ref = raw_api_key["secret_ref"] if "secret_ref" in raw_api_key else None
        if isinstance(secret_ref, str) and secret_ref.strip():
            return f"secret_ref:{secret_ref}"
    if isinstance(raw_api_key, str) and raw_api_key.strip():
        return "inline_api_key"
    return None


def _risk_message(
    *,
    node_id: str,
    provider: str,
    model: str | None,
    estimated_provider_calls: int | None,
    markers: Sequence[str],
) -> str:
    provider_label = _provider_label(provider)
    model_text = f" model {model}" if model is not None else ""
    if estimated_provider_calls is None:
        source_text = " after fanout" if markers else ""
        return f"LLM transform '{node_id}' may make an unknown number of {provider_label}{model_text} calls{source_text}."
    return f"LLM transform '{node_id}' may make {estimated_provider_calls} {provider_label}{model_text} call(s)."


def _guard_summary(risks: Sequence[ExecutionFanoutRisk]) -> str:
    if len(risks) == 1:
        risk = risks[0]
        calls = (
            "an unknown number of provider calls"
            if risk.estimated_provider_calls is None
            else f"{risk.estimated_provider_calls} provider call(s)"
        )
        model = f" model {risk.model}" if risk.model is not None else ""
        return f"Confirm LLM fanout before execution: node '{risk.node_id}' uses {risk.provider}{model} and may make {calls}."
    return f"Confirm LLM fanout before execution: {len(risks)} LLM nodes may make high-cardinality provider calls."


def _provider_label(provider: str) -> str:
    if provider == "unknown":
        return "provider"
    return provider
