"""Operation outcomes and results.

These types answer: "What did an operation produce?"

IMPORTANT:
- TransformResult.status uses Literal["success", "error"], NOT an enum
- TransformResult and GateResult KEEP audit fields (input_hash, output_hash, duration_ms)
- ArtifactDescriptor matches architecture schema (artifact_type, content_hash REQUIRED, size_bytes REQUIRED)
- FailureInfo provides type-safe error details for RowResult
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, quote, urlparse

from elspeth.contracts.freeze import freeze_fields, require_int
from elspeth.contracts.url import SanitizedDatabaseUrl, SanitizedWebhookUrl, _extract_known_webhook_path_secret

if TYPE_CHECKING:
    from elspeth.contracts.errors import MaxRetriesExceeded
    from elspeth.contracts.node_state_context import NodeStateContext
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract

from elspeth.contracts.enums import _LEGAL_TERMINAL_PAIRS, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    FrameworkBugError,
    OrchestrationInvariantError,
    PluginContractViolation,
    TransformErrorReason,
    TransformSuccessReason,
)
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.routing import RoutingAction


def _require_pipeline_row(value: object, *, location: str) -> PipelineRow:
    """Reject non-PipelineRow output before it can masquerade as valid data."""
    from elspeth.contracts.schema_contract import PipelineRow

    if not isinstance(value, PipelineRow):
        raise PluginContractViolation(
            f"{location} must be a PipelineRow, got {type(value).__name__}. "
            "Build transform output with PipelineRow(data, contract) before returning it."
        )
    return value


def _require_shared_contract_identity(rows: Sequence[PipelineRow], *, location: str) -> None:
    """Reject multi-row outputs whose rows do not share one contract object."""
    if len(rows) < 2:
        return

    first_contract = rows[0].contract
    for i in range(1, len(rows)):
        if rows[i].contract is not first_contract:
            other_contract = rows[i].contract
            raise PluginContractViolation(
                f"{location} received rows with inconsistent contracts: "
                f"row 0 has {first_contract.mode if first_contract else None} contract "
                f"with {len(first_contract.fields) if first_contract else 0} fields, "
                f"but row {i} has {other_contract.mode if other_contract else None} contract "
                f"with {len(other_contract.fields) if other_contract else 0} fields. "
                f"All rows in a multi-row result must share the same contract instance."
            )


_ARTIFACT_TYPES = frozenset({"file", "database", "webhook"})
_ARTIFACT_HASH_RE = re.compile(r"[0-9a-fA-F]+")
_SENSITIVE_ARTIFACT_URI_PARAMS = frozenset(
    {
        "access_token",
        "api_key",
        "api_secret",
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "client_secret",
        "credential",
        "credentials",
        "key",
        "password",
        "secret",
        "sig",
        "signature",
        "token",
        "x-api-key",
    }
)


def _require_non_empty_str(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}: {value!r}")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _base_artifact_param_name(key: str) -> str:
    bracket = key.find("[")
    if bracket != -1:
        key = key[:bracket]
    dot = key.find(".")
    if dot != -1:
        key = key[:dot]
    return key


def _artifact_uri_candidates(path_or_uri: str) -> tuple[str, ...]:
    candidates = [path_or_uri]
    if path_or_uri.startswith("webhook://"):
        candidates.append(path_or_uri.removeprefix("webhook://"))
    if path_or_uri.startswith("db://"):
        _, separator, nested_uri = path_or_uri.removeprefix("db://").partition("@")
        if separator:
            candidates.append(nested_uri)
    return tuple(candidates)


def _require_no_artifact_uri_credentials(path_or_uri: str) -> None:
    for candidate in _artifact_uri_candidates(path_or_uri):
        parsed = urlparse(candidate)
        if parsed.password is not None:
            raise ValueError("path_or_uri must not contain raw URL credentials")
        if parsed.scheme in {"http", "https"} and parsed.username is not None:
            raise ValueError("path_or_uri must not contain raw URL credentials")

        for section_name, encoded_params in (("query", parsed.query), ("fragment", parsed.fragment)):
            sensitive_keys = [
                key
                for key in parse_qs(encoded_params, keep_blank_values=True)
                if _base_artifact_param_name(key.lower()) in _SENSITIVE_ARTIFACT_URI_PARAMS
            ]
            if sensitive_keys:
                raise ValueError(f"path_or_uri must not contain sensitive {section_name} parameters: {sensitive_keys}")
        if _extract_known_webhook_path_secret(parsed) is not None:
            raise ValueError("path_or_uri must not contain known webhook path secrets")


def require_no_artifact_uri_credentials(path_or_uri: str) -> None:
    """Reject artifact paths/URIs that would persist raw credential material."""
    _require_no_artifact_uri_credentials(path_or_uri)


def _require_artifact_hash(content_hash: object) -> None:
    value = _require_non_empty_str(content_hash, "content_hash")
    if _ARTIFACT_HASH_RE.fullmatch(value) is None:
        raise ValueError("content_hash must be a non-empty hex string")


def _require_artifact_metadata(value: object, field_name: str = "metadata") -> None:
    if value is None:
        return
    if not isinstance(value, MappingProxyType):
        raise TypeError(f"{field_name} must be a frozen mapping, got {type(value).__name__}: {value!r}")
    for key, item in value.items():
        if type(key) is not str:
            raise TypeError(f"{field_name} key must be str, got {type(key).__name__}: {key!r}")
        if isinstance(item, MappingProxyType):
            _require_artifact_metadata(item, f"{field_name}[{key!r}]")


@dataclass(frozen=True, slots=True)
class ExceptionResult:
    """Wrapper for exceptions that should propagate through async pattern.

    When a worker thread encounters an uncaught exception (plugin bug),
    it wraps the exception in this container. The waiter then re-raises
    the original exception in the orchestrator thread, ensuring plugin
    bugs crash the pipeline as intended.

    Frozen: exception wrappers are immutable evidence — the captured
    exception and traceback must not be modified after construction.

    Used by:
    - engine/batch_adapter.py: Wraps exceptions in worker threads
    - plugins/batching/mixin.py: Creates ExceptionResult on worker failure
    - plugins/batching/ports.py: Type hint in BatchOutputPort protocol
    """

    exception: BaseException
    traceback: str


@dataclass(frozen=True, slots=True)
class FailureInfo:
    """Type-safe error details for RowResult.

    Captures structured failure information for FAILED outcomes.
    Use factory methods for common error types.

    Frozen: failure info is immutable evidence captured at the point of
    failure. Modifying it after construction would compromise audit integrity.

    Fields:
        exception_type: The exception class name (required)
        message: Human-readable error message (required)
        attempts: Number of retry attempts (optional, for retry failures)
        last_error: The underlying error message (optional)
    """

    exception_type: str
    message: str
    attempts: int | None = None
    last_error: str | None = None

    @classmethod
    def from_max_retries_exceeded(cls, e: MaxRetriesExceeded) -> FailureInfo:
        """Create FailureInfo from MaxRetriesExceeded exception.

        Args:
            e: The MaxRetriesExceeded exception

        Returns:
            FailureInfo with all retry details
        """
        return cls(
            exception_type="MaxRetriesExceeded",
            message=str(e),
            attempts=e.attempts,
            last_error=str(e.last_error),
        )


@dataclass
class TransformResult:
    """Result of a transform operation.

    Use the factory methods to create instances.

    IMPORTANT: status uses Literal["success", "error"], NOT enum, per architecture.
    Audit fields (input_hash, output_hash, duration_ms) are populated by executors.

    Multi-row output:
    - Single-row: success(row) sets row=row, rows=None
    - Multi-row: success_multi(rows) sets row=None, rows=rows
    - Empty success: success_empty() sets row=None, rows=()
    - Use is_multi_row property to distinguish
    - Use has_output_data property to check if ANY explicit output carrier exists
    """

    status: Literal["success", "error"]
    row: PipelineRow | None
    reason: TransformErrorReason | None
    retryable: bool = False
    rows: tuple[PipelineRow, ...] | None = None

    # Success metadata - REQUIRED for success results, None for error results
    # Invariant: status="success" implies success_reason is not None
    success_reason: TransformSuccessReason | None = None

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    # Context snapshot for audit trail (optional)
    # Contains operational metadata like pool stats, ordering info
    # Enables pool metadata to flow to context_after_json
    context_after: NodeStateContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate invariants - success and error results MUST satisfy their contracts."""
        if self.status not in ("success", "error"):
            raise ValueError("TransformResult status must be 'success' or 'error'.")
        if self.status == "success" and self.success_reason is None:
            raise ValueError(
                "TransformResult with status='success' MUST provide success_reason. "
                "Use TransformResult.success(row, success_reason={'action': '...'}) "
                "to create success results. Missing success_reason is a plugin bug."
            )
        if self.status == "success":
            success_reason = self.success_reason
            assert success_reason is not None
            try:
                action = success_reason.get("action")
            except AttributeError as exc:
                raise ValueError(
                    "TransformResult with status='success' MUST provide success_reason as a mapping. "
                    "Use TransformResult.success(row, success_reason={'action': '...'}) "
                    "to create success results. Invalid success_reason is a plugin bug."
                ) from exc
            if not isinstance(action, str):
                raise ValueError(
                    "TransformResult with status='success' MUST include success_reason['action'] as a str. "
                    "Use TransformResult.success(row, success_reason={'action': '...'}) "
                    "to create success results. Missing success_reason['action'] is a plugin bug."
                )
        if self.status == "success" and self.row is None and self.rows is None:
            raise ValueError(
                "TransformResult with status='success' MUST have output data (row or rows). "
                "Use TransformResult.success(row, ...) or TransformResult.success_multi(rows, ...) "
                "to create success results. Missing output data is a plugin bug."
            )
        if self.status == "success" and self.row is not None and self.rows is not None:
            raise ValueError(
                "TransformResult with status='success' MUST provide exactly one of row or rows, not both. "
                "Use TransformResult.success(row, ...) for single-row or "
                "TransformResult.success_multi(rows, ...) for multi-row output."
            )
        if self.status == "success" and self.row is not None:
            _require_pipeline_row(self.row, location="TransformResult.row")
        if self.status == "success" and self.rows is not None:
            for i, row in enumerate(self.rows):
                _require_pipeline_row(row, location=f"TransformResult.rows[{i}]")
            _require_shared_contract_identity(self.rows, location="TransformResult.rows")
        if self.status == "error":
            if self.reason is None:
                raise ValueError(
                    "TransformResult with status='error' MUST provide reason. "
                    "Use TransformResult.error({'reason': '...'}) to create error results. "
                    "Missing reason is a plugin bug."
                )
            if "reason" not in self.reason:
                raise ValueError(
                    "TransformResult with status='error' MUST include reason['reason']. "
                    "Use TransformResult.error({'reason': '...'}) to create error results. "
                    "Missing reason['reason'] is a plugin bug."
                )
            if self.row is not None or self.rows is not None:
                raise ValueError(
                    "TransformResult with status='error' MUST NOT include output data (row or rows). "
                    "Error results carry reason only, not data. This is a plugin bug."
                )
            if self.success_reason is not None:
                raise ValueError(
                    "TransformResult with status='error' MUST NOT include success_reason. "
                    "Error results carry reason only. This is a plugin bug."
                )

    @property
    def is_multi_row(self) -> bool:
        """True if this result contains multiple output rows."""
        return self.rows is not None

    @property
    def has_output_data(self) -> bool:
        """True if this result has any explicit output carrier (row or rows)."""
        return self.row is not None or self.rows is not None

    @classmethod
    def success(
        cls,
        row: PipelineRow,
        *,
        success_reason: TransformSuccessReason,
        context_after: NodeStateContext | None = None,
    ) -> TransformResult:
        """Create successful result with single output row.

        Args:
            row: The transformed row data as a PipelineRow wrapping
                 the output dict with its schema contract.
            success_reason: REQUIRED metadata about what the transform did.
                           Must include at least 'action' field.
                           See TransformSuccessReason for available fields.
            context_after: Optional operational metadata for audit trail
                          (e.g., pool stats, ordering info).

        Returns:
            TransformResult with status="success" and the provided row.

        Example:
            return TransformResult.success(
                PipelineRow(output_dict, contract),
                success_reason={"action": "processed", "fields_modified": ["amount"]}
            )
        """
        row = _require_pipeline_row(row, location="TransformResult.success(row)")
        return cls(
            status="success",
            row=row,
            reason=None,
            rows=None,
            success_reason=success_reason,
            context_after=context_after,
        )

    @classmethod
    def success_multi(
        cls,
        rows: Sequence[PipelineRow],
        *,
        success_reason: TransformSuccessReason,
        context_after: NodeStateContext | None = None,
    ) -> TransformResult:
        """Create successful result with multiple output rows.

        Args:
            rows: List of PipelineRow instances (must not be empty).
            success_reason: REQUIRED metadata about what the transform did.
                           Must include at least 'action' field.
                           See TransformSuccessReason for available fields.
            context_after: Optional operational metadata for audit trail
                          (e.g., pool stats, ordering info).

        Returns:
            TransformResult with status="success", row=None, rows=rows

        Raises:
            ValueError: If rows is empty

        Example:
            return TransformResult.success_multi(
                [PipelineRow(r, contract) for r in output_rows],
                success_reason={"action": "split", "fields_added": ["row_index"]}
            )
        """
        output_rows = tuple(rows)
        if not output_rows:
            raise ValueError("success_multi requires at least one row")
        for i, row in enumerate(output_rows):
            _require_pipeline_row(row, location=f"TransformResult.success_multi(rows[{i}])")
        _require_shared_contract_identity(output_rows, location="success_multi()")
        return cls(
            status="success",
            row=None,
            reason=None,
            rows=output_rows,
            success_reason=success_reason,
            context_after=context_after,
        )

    @classmethod
    def success_empty(
        cls,
        *,
        success_reason: TransformSuccessReason,
        context_after: NodeStateContext | None = None,
    ) -> TransformResult:
        """Create successful result with explicit zero-row emission.

        This is distinct from ``success_multi([])`` which remains invalid.
        ``success_empty()`` is the auditable "the transform intentionally
        emitted nothing" shape used by the engine's zero-emission pathways.
        """
        return cls(
            status="success",
            row=None,
            reason=None,
            rows=(),
            success_reason=success_reason,
            context_after=context_after,
        )

    @classmethod
    def error(
        cls,
        reason: TransformErrorReason,
        *,
        retryable: bool = False,
        context_after: NodeStateContext | None = None,
    ) -> TransformResult:
        """Create error result with structured reason.

        Args:
            reason: Error details with required 'reason' field from
                    TransformErrorCategory (compile-time validated).
                    See TransformErrorReason for all available context fields.
            retryable: Whether the error is transient and should be retried.
            context_after: Optional operational metadata for audit trail
                          (e.g., pool stats from partial execution).

        Returns:
            TransformResult with status="error" and the provided reason.
            Error results never carry contracts (contract=None).
        """
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
            rows=None,
            context_after=context_after,
        )


@dataclass
class GateResult:
    """Result of a config-driven gate evaluation.

    Contains the (possibly modified) row and routing action.
    Constructed entirely by GateExecutor — gates are config-driven,
    not plugin-based.
    """

    row: dict[str, Any]
    action: RoutingAction

    # Schema contract for output (optional)
    # Enables conversion to PipelineRow via to_pipeline_row()
    contract: SchemaContract | None = field(default=None, repr=False)

    # Audit fields - set by GateExecutor
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    def to_pipeline_row(self) -> PipelineRow:
        """Convert to PipelineRow for downstream processing.

        Returns:
            PipelineRow wrapping row data with contract

        Raises:
            ValueError: If contract is None
        """
        from elspeth.contracts.schema_contract import PipelineRow

        if self.contract is None:
            raise FrameworkBugError(
                "GateResult has no contract - cannot create PipelineRow. "
                "The engine must set contract on GateResult before calling to_pipeline_row()."
            )
        return PipelineRow(self.row, self.contract)


@dataclass(frozen=True, slots=True)
class RowResult:
    """Final result of processing a row through the pipeline (ADR-019 two-axis).

    Frozen to prevent post-construction mutation of outcome/sink_name,
    which would bypass __post_init__ invariant checks.

    Fields:
        token: Token identity for this row instance
        final_data: Final row data as PipelineRow (may be original if failed early)
        outcome: Lifecycle answer (None for non-terminal BUFFERED rows)
        path: Provenance answer (always populated)
        sink_name: For paths that reach a sink, the destination sink name
        error: For ON_ERROR_ROUTED, type-safe error details for audit
        scheduler_pending_sink: True only after the durable scheduler row for
            this exact token has been transitioned to PENDING_SINK.
        authoritative_error_hash: For ON_ERROR_ROUTED results REBUILT from a
            persisted pending sink (crash-recovery replay), the ORIGINAL
            audited error hash. The outcome accumulator prefers this over
            recomputing from the synthetic replay FailureInfo, so the replayed
            audit record correlates with the pre-crash one
            (filigree elspeth-d74d19f901). None for live results.
    """

    token: TokenInfo
    final_data: PipelineRow
    outcome: TerminalOutcome | None
    path: TerminalPath
    sink_name: str | None = None
    error: FailureInfo | None = None
    scheduler_pending_sink: bool = False
    authoritative_error_hash: str | None = None

    def __post_init__(self) -> None:
        if type(self.scheduler_pending_sink) is not bool:
            raise OrchestrationInvariantError(
                f"RowResult.scheduler_pending_sink must be bool, got {type(self.scheduler_pending_sink).__name__}"
            )
        if self.authoritative_error_hash is not None:
            if type(self.authoritative_error_hash) is not str or not self.authoritative_error_hash:
                raise OrchestrationInvariantError("RowResult.authoritative_error_hash must be a non-empty string when set")
            if self.path != TerminalPath.ON_ERROR_ROUTED:
                raise OrchestrationInvariantError(
                    f"RowResult.authoritative_error_hash is only valid for ON_ERROR_ROUTED results, got path={self.path!r}"
                )
        if self.outcome is not None and (self.outcome, self.path) not in _LEGAL_TERMINAL_PAIRS:
            raise OrchestrationInvariantError(f"RowResult: illegal (outcome, path) pair: ({self.outcome!r}, {self.path!r})")
        if self.outcome is None and self.path != TerminalPath.BUFFERED:
            raise OrchestrationInvariantError(f"RowResult: outcome=None requires path=BUFFERED, got path={self.path!r}")
        if self.outcome is not None and self.path == TerminalPath.BUFFERED:
            raise OrchestrationInvariantError(f"RowResult: path=BUFFERED requires outcome=None, got outcome={self.outcome!r}")
        if self.outcome is None and self.path == TerminalPath.BUFFERED and self.sink_name is not None:
            raise OrchestrationInvariantError("RowResult: BUFFERED rows must not set sink_name before terminal recording")

        if self.path == TerminalPath.DEFAULT_FLOW and self.sink_name is None:
            raise OrchestrationInvariantError("(SUCCESS, DEFAULT_FLOW) outcome requires sink_name to be set")
        if self.path == TerminalPath.GATE_ROUTED and self.sink_name is None:
            raise OrchestrationInvariantError("(SUCCESS, GATE_ROUTED) outcome requires sink_name to be set")
        if self.path == TerminalPath.ON_ERROR_ROUTED:
            if self.sink_name is None:
                raise OrchestrationInvariantError("(FAILURE, ON_ERROR_ROUTED) outcome requires sink_name to be set")
            if self.error is None:
                raise OrchestrationInvariantError(
                    "(FAILURE, ON_ERROR_ROUTED) outcome requires error (FailureInfo) to be set — "
                    "the originating transform error must be captured on the outcome "
                    "record for single-hop audit attributability."
                )
            if not isinstance(self.error, FailureInfo):
                raise OrchestrationInvariantError("(FAILURE, ON_ERROR_ROUTED) outcome requires error to be a FailureInfo instance")
        if self.path == TerminalPath.COALESCED and self.sink_name is None:
            raise OrchestrationInvariantError("(SUCCESS, COALESCED) outcome requires sink_name to be set")


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Descriptor for an artifact written by a sink.

    Matches architecture artifacts table schema:
    - artifact_type: NOT NULL (matches DB column name)
    - content_hash: NOT NULL (REQUIRED for audit integrity)
    - size_bytes: NOT NULL (REQUIRED for verification)

    Factory methods provide convenient construction for each artifact type.
    """

    artifact_type: Literal["file", "database", "webhook"]
    path_or_uri: str
    content_hash: str  # REQUIRED - audit integrity
    size_bytes: int  # REQUIRED - verification
    metadata: MappingProxyType[str, object] | None = None

    def __post_init__(self) -> None:
        freeze_fields(self, "metadata")
        artifact_type = _require_non_empty_str(self.artifact_type, "artifact_type")
        if artifact_type not in _ARTIFACT_TYPES:
            raise ValueError(f"artifact_type must be one of {sorted(_ARTIFACT_TYPES)}, got {artifact_type!r}")
        path_or_uri = _require_non_empty_str(self.path_or_uri, "path_or_uri")
        _require_no_artifact_uri_credentials(path_or_uri)
        _require_artifact_hash(self.content_hash)
        require_int(self.size_bytes, "size_bytes", min_value=0)
        _require_artifact_metadata(self.metadata)

    @classmethod
    def for_file(
        cls,
        path: str,
        content_hash: str,
        size_bytes: int,
    ) -> ArtifactDescriptor:
        """Create descriptor for file-based artifacts."""
        return cls(
            artifact_type="file",
            path_or_uri=f"file://{quote(path, safe='/:')}",
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    @classmethod
    def for_database(
        cls,
        url: SanitizedDatabaseUrl,
        table: str,
        content_hash: str,
        payload_size: int,
        row_count: int,
    ) -> ArtifactDescriptor:
        """Create descriptor for database artifacts.

        URL must be pre-sanitized using SanitizedDatabaseUrl.from_raw_url().
        This ensures credentials are never stored in the audit trail.
        """
        # Type safety: enforce SanitizedDatabaseUrl, not duck-typed objects
        if not isinstance(url, SanitizedDatabaseUrl):
            raise TypeError(
                "url must be a SanitizedDatabaseUrl instance. Use SanitizedDatabaseUrl.from_raw_url(url) to sanitize raw database URLs."
            )

        metadata: dict[str, object] = {"table": table, "row_count": row_count}
        if url.fingerprint:
            metadata["url_fingerprint"] = url.fingerprint

        return cls(
            artifact_type="database",
            path_or_uri=f"db://{table}@{url.sanitized_url}",
            content_hash=content_hash,
            size_bytes=payload_size,
            metadata=MappingProxyType(metadata),
        )

    @classmethod
    def for_webhook(
        cls,
        url: SanitizedWebhookUrl,
        content_hash: str,
        request_size: int,
        response_code: int,
    ) -> ArtifactDescriptor:
        """Create descriptor for webhook artifacts.

        URL must be pre-sanitized using SanitizedWebhookUrl.from_raw_url().
        This removes supported token forms before storing the URL in the audit
        trail: userinfo, sensitive query/fragment parameters, and known Slack
        incoming-webhook path tokens. Other path-borne secrets are not
        generically redacted.
        """
        # Type safety: enforce SanitizedWebhookUrl, not duck-typed objects
        if not isinstance(url, SanitizedWebhookUrl):
            raise TypeError(
                "url must be a SanitizedWebhookUrl instance. Use SanitizedWebhookUrl.from_raw_url(url) to sanitize raw webhook URLs."
            )

        metadata: dict[str, object] = {"response_code": response_code}
        if url.fingerprint:
            metadata["url_fingerprint"] = url.fingerprint

        return cls(
            artifact_type="webhook",
            path_or_uri=f"webhook://{url.sanitized_url}",
            content_hash=content_hash,
            size_bytes=request_size,
            metadata=MappingProxyType(metadata),
        )


@dataclass(frozen=True, slots=True)
class SourceRow:
    """Result from source loading - either valid data or quarantined invalid data.

    ALL rows from sources MUST be wrapped in SourceRow:
    - Valid rows: SourceRow.valid(row_dict, contract=contract, source_row_index=index)
    - Invalid rows: SourceRow.quarantined(row_data, error, destination, source_row_index=index)

    This makes source outcomes first-class engine concepts:
    - All rows get proper token_id for lineage
    - Metrics include both valid and quarantine counts
    - Audit trail shows complete source output
    - Quarantine sinks receive invalid data for investigation

    Example usage in a source:
        try:
            validated = schema.model_validate(row)
            yield SourceRow.valid(
                validated.to_row(),
                contract=contract,
                source_row_index=source_row_index,
            )
        except ValidationError as e:
            if on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row,
                    error=str(e),
                    destination=on_validation_failure,
                    source_row_index=source_row_index,
                )
            # else: don't yield, row is intentionally discarded
    """

    # Note: row is Any (not dict) because quarantined rows from external data
    # may not be dicts (e.g., JSON arrays containing primitives like numbers).
    # Valid rows are always dicts (they passed schema validation).
    row: Any
    is_quarantined: bool
    quarantine_error: str | None = None
    quarantine_destination: str | None = None
    contract: SchemaContract | None = None
    source_row_index: int | None = None

    def __post_init__(self) -> None:
        """Validate quarantine field invariants.

        Quarantined rows MUST have error and destination (where to route them).
        Non-quarantined rows MUST NOT have quarantine fields set (prevents
        accidental misuse where quarantine metadata is silently ignored).
        """
        if self.is_quarantined:
            if self.source_row_index is None:
                raise ValueError("Quarantined SourceRow must have source_row_index. Pass source_row_index= to SourceRow.quarantined().")
            require_int(self.source_row_index, "SourceRow.source_row_index", min_value=0)
            if self.quarantine_error is None:
                raise ValueError("Quarantined SourceRow must have quarantine_error")
            _require_non_empty_str(self.quarantine_error, "quarantine_error")
            if self.quarantine_destination is None:
                raise ValueError("Quarantined SourceRow must have quarantine_destination")
        else:
            if self.source_row_index is None:
                raise ValueError("Valid SourceRow must have source_row_index. Pass source_row_index= to SourceRow.valid().")
            require_int(self.source_row_index, "SourceRow.source_row_index", min_value=0)
            if self.quarantine_error is not None:
                raise ValueError(f"Non-quarantined SourceRow must not have quarantine_error, got: {self.quarantine_error!r}")
            if self.quarantine_destination is not None:
                raise ValueError(f"Non-quarantined SourceRow must not have quarantine_destination, got: {self.quarantine_destination!r}")
            # Valid rows MUST have a contract — the engine requires it at
            # tokenization. Catching it here prevents a misleading crash later.
            if self.contract is None:
                raise ValueError("Valid SourceRow must have a contract. Pass contract= to SourceRow.valid().")

    @classmethod
    def valid(
        cls,
        row: dict[str, Any],
        *,
        contract: SchemaContract,
        source_row_index: int,
    ) -> SourceRow:
        """Create a valid source row.

        Args:
            row: Validated row data
            contract: Schema contract for the row
            source_row_index: Source-authored row position within its emission stream

        Returns:
            SourceRow with is_quarantined=False
        """
        return cls(row=row, is_quarantined=False, contract=contract, source_row_index=source_row_index)

    @classmethod
    def quarantined(
        cls,
        row: Any,
        error: str,
        destination: str,
        *,
        source_row_index: int,
    ) -> SourceRow:
        """Create a quarantined row result.

        Args:
            row: The original row data (before validation). May be non-dict
                 for malformed external data (e.g., JSON primitives).
            error: The validation error message
            destination: The sink name to route this row to
            source_row_index: Source-authored row position for emitted
                quarantined rows.
        """
        return cls(
            row=row,
            is_quarantined=True,
            quarantine_error=error,
            quarantine_destination=destination,
            contract=None,  # Quarantined rows don't have contracts
            source_row_index=source_row_index,
        )

    def to_pipeline_row(self) -> PipelineRow:
        """Convert to PipelineRow for processing.

        Returns:
            PipelineRow wrapping row data with contract

        Raises:
            ValueError: If row is quarantined
            FrameworkBugError: If contract is None (engine must set it)
        """
        from elspeth.contracts.schema_contract import PipelineRow

        if self.is_quarantined:
            raise ValueError("Cannot convert quarantined row to PipelineRow")
        if self.contract is None:
            raise FrameworkBugError(
                "SourceRow has no contract — cannot create PipelineRow. "
                "The engine must set contract on SourceRow before calling to_pipeline_row()."
            )

        return PipelineRow(self.row, self.contract)
