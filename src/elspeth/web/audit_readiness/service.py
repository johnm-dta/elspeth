"""ReadinessService — pure aggregation of audit signals into a snapshot.
No new validation logic. Layer: L3 (application).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.audit_readiness.models import (
    AuditReadinessSnapshot,
    ReadinessRow,
)
from elspeth.web.audit_readiness.trust import (
    PluginKind,
    PluginTrust,
    classify_plugin,
    is_registered_plugin,
)
from elspeth.web.composer.state import CompositionState
from elspeth.web.execution.schemas import (
    CHECK_OUTCOME_SECRET_REFS_NO_REFS,
    CHECK_OUTCOME_SECRET_REFS_RESOLVED,
    CHECK_OUTCOME_SECRET_REFS_SKIPPED_NO_SERVICE,
    CHECK_OUTCOME_SECRET_REFS_UNRESOLVED,
    CHECK_OUTCOME_SKIPPED_AFTER_FAILURE,
    ValidationResult,
)
from elspeth.web.sessions.converters import (
    state_from_record as _default_state_from_record,
)

# Mirror of validation.py's private constant — duplicated to keep the
# dependency unidirectional (audit_readiness depends on the result shape,
# not on validation's internal naming).
_CHECK_IDENTITY_NODE_ADVISORY = "identity_node_advisory"


class CompositionStateNotFoundError(LookupError):
    """Raised when a session has no current composition state."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        super().__init__("No composition state for this session")


class _ExecutionServiceLike(Protocol):
    # session_id is UUID to match ExecutionServiceImpl.validate (web/execution/service.py:638);
    # the route layer constructs it from the FastAPI path-param parser.
    async def validate(self, session_id: UUID, *, user_id: str | None = None) -> ValidationResult: ...


class _SessionServiceLike(Protocol):
    # session_id is UUID to match SessionServiceImpl.get_current_state (web/sessions/service.py:1884).
    async def get_current_state(self, session_id: UUID) -> Any: ...


class _SecretServiceLike(Protocol):
    # Matches ScopedSecretResolver.list_refs (service.py:312) — no auth_provider_type.
    # The scoped resolver already has auth_provider_type baked in at construction
    # (app.py:470: ScopedSecretResolver(service, settings.auth_provider)).
    # Inject app.state.scoped_secret_resolver, not app.state.secret_service.
    def list_refs(self, user_id: str) -> list[SecretInventoryItem]: ...  # sync; caller uses run_sync_in_worker


class _SettingsLike(Protocol):
    # ``WebSettings`` declares this as a Pydantic ``Field`` with a default
    # (web/config.py:107), which mypy sees as a read-only ``int`` attribute
    # on the model class.  Mirror the read-only declaration so the Protocol
    # accepts the model without a write-vs-read variance complaint.
    @property
    def payload_store_retention_days(self) -> int: ...


class ReadinessService:
    """Compose audit-readiness signals into a snapshot for the panel."""

    def __init__(
        self,
        *,
        execution_service: _ExecutionServiceLike,
        session_service: _SessionServiceLike,
        scoped_secret_resolver: _SecretServiceLike,
        settings: _SettingsLike,
        state_from_record: Callable[..., CompositionState] | None = None,
    ) -> None:
        self._execution_service = execution_service
        self._session_service = session_service
        self._scoped_secret_resolver = scoped_secret_resolver
        self._settings = settings
        self._state_from_record: Callable[..., CompositionState] = (
            state_from_record if state_from_record is not None else _default_state_from_record
        )

    async def compute_snapshot(self, *, session_id: UUID, user_id: str) -> AuditReadinessSnapshot:
        """Return the six-row snapshot.

        Raises:
            CompositionStateNotFoundError: when the session has no composition
                state. The route layer translates this into a 404.
        """
        checked_at = datetime.now(UTC)
        record = await self._session_service.get_current_state(session_id)
        if record is None:
            raise CompositionStateNotFoundError(session_id)
        state: CompositionState = self._state_from_record(record)
        validation = await self._execution_service.validate(session_id, user_id=user_id)
        inventory = await run_sync_in_worker(
            self._scoped_secret_resolver.list_refs,
            user_id,  # scoped_secret_resolver.list_refs takes user_id only
        )
        rows: tuple[ReadinessRow, ...] = (
            _build_validation_row(validation),
            _build_plugin_trust_row(state),
            _build_provenance_row(validation),
            _build_retention_row(self._settings.payload_store_retention_days),
            _build_llm_interpretations_row(state),
            _build_secrets_row(validation, inventory),
        )
        # ``session_id`` is rendered as ``str`` in the JSON envelope; the
        # model's pydantic ``Field(min_length=1)`` accepts the canonical
        # 36-char UUID representation.
        return AuditReadinessSnapshot(
            session_id=str(session_id),
            composition_version=state.version,
            checked_at=checked_at,
            rows=rows,
        )


# ── Row projections ───────────────────────────────────────────────


def _build_validation_row(result: ValidationResult) -> ReadinessRow:
    if result.is_valid:
        return ReadinessRow(
            id="validation",
            label="Validation",
            status="ok",
            summary="All checks pass",
            detail=None,
            component_ids=(),
        )
    component_ids = tuple(sorted({err.component_id for err in result.errors if err.component_id is not None}))
    summary = f"{len(result.errors)} errors — see details" if len(result.errors) != 1 else "1 error — see details"
    detail = "\n".join(f"[{err.component_type or 'unknown'}] {err.component_id or 'unknown'}: {err.message}" for err in result.errors)
    return ReadinessRow(
        id="validation",
        label="Validation",
        status="error",
        summary=summary,
        detail=detail,
        component_ids=component_ids,
    )


def _build_plugin_trust_row(state: CompositionState) -> ReadinessRow:
    """Classify every plugin in the composition (boundary vs internal).
    Panel uses "Plugin trust" vocabulary; tier numbers belong in the Explain view.
    """
    boundary: list[tuple[str, str, str]] = []
    unknown: list[tuple[str, str]] = []

    def _record(kind: PluginKind, component_id: str, name: str | None) -> None:
        if name is None or not is_registered_plugin(kind, name):
            unknown.append((kind, component_id))
            return
        # TODO(phase-7): delete trust.py and replace classify_plugin() callers per Phase 7 plan
        trust = classify_plugin(kind, name)
        if trust is PluginTrust.BOUNDARY:
            boundary.append((kind, component_id, name))

    if state.source is not None:
        _record("source", "source", state.source.plugin)
    for node in state.nodes:
        if node.node_type == "transform":
            _record("transform", node.id, node.plugin)
    for output in state.outputs:
        _record("sink", output.name, output.plugin)

    if unknown:
        ids = tuple(sorted({cid for _, cid in unknown}))
        return ReadinessRow(
            id="plugin_trust",
            label="Plugin trust",
            status="error",
            summary="Unknown plugin in composition",
            detail=("The composition references plugin names not in the registered catalog. Validation will block execution."),
            component_ids=ids,
        )

    if not boundary:
        return ReadinessRow(
            id="plugin_trust",
            label="Plugin trust",
            status="ok",
            summary="All plugins operate on pipeline data",
            detail=None,
            component_ids=(),
        )

    # Boundary plugins are recorded as ok with the boundaries named in
    # detail. The "sensitivity-vs-tier mismatch" warning case needs a
    # user-stated sensitivity surface that does not yet exist (roadmap §G2).
    detail = "\n".join(f"- [{kind}] {cid} ({name}) — crosses an external boundary" for kind, cid, name in boundary)
    return ReadinessRow(
        id="plugin_trust",
        label="Plugin trust",
        status="ok",
        summary=f"{len(boundary)} external-boundary plugin(s) recorded",
        detail=detail,
        component_ids=tuple(cid for _, cid, _ in boundary),
    )


def _build_provenance_row(result: ValidationResult) -> ReadinessRow:
    """Project identity_node_advisory checks into the provenance row.

    Node ids come from check.affected_nodes (structured tuple added by
    Finding 2). No prose parsing of the detail field.
    """
    advisory_checks = [c for c in result.checks if c.name == _CHECK_IDENTITY_NODE_ADVISORY]
    advisories = [c for c in advisory_checks if c.passed]
    if not advisories:
        skipped = [c for c in advisory_checks if not c.passed]
        if skipped:
            return ReadinessRow(
                id="provenance",
                label="Provenance",
                status="not_applicable",
                summary="Provenance check did not run",
                detail="\n".join(c.detail for c in skipped),
                component_ids=(),
            )
        return ReadinessRow(
            id="provenance",
            label="Provenance",
            status="ok",
            summary="All paths record provenance",
            detail=None,
            component_ids=(),
        )
    component_ids = tuple(
        node_id
        for c in advisories
        for node_id in c.affected_nodes  # structured field; no prose parse
    )
    return ReadinessRow(
        id="provenance",
        label="Provenance",
        status="warning",
        summary=f"{len(advisories)} identity passthrough(s) — provenance gap",
        detail="\n".join(c.detail for c in advisories),
        component_ids=component_ids,
    )


def _build_retention_row(retention_days: int) -> ReadinessRow:
    """System-configured; no user requirement to compare against."""
    return ReadinessRow(
        id="retention",
        label="Retention",
        status="not_applicable",
        summary=f"System retention: {retention_days} days",
        detail=("Per-composition retention configuration is not yet exposed; configured retention applies to all payloads."),
        component_ids=(),
    )


def _build_llm_interpretations_row(state: CompositionState) -> ReadinessRow:
    """Always not_applicable in Phase 2A; Phase 5b implements the real signal."""
    has_llm = any(n.node_type == "transform" and n.plugin == "llm" for n in state.nodes)
    summary = "Interpretation surface not yet available" if has_llm else "No LLM transforms in this composition"
    return ReadinessRow(
        id="llm_interpretations",
        label="LLM interpretations",
        status="not_applicable",
        summary=summary,
        detail=None,
        component_ids=(),
    )


# Producer is web/execution/validation.py — the ValidationError.error_code
# values emitted at lines 727 ("missing_secret_ref"), 748 ("fabricated_secret"),
# and 770 ("disallowed_secret_ref"). The plan's draft for this constant used
# "fabricated_secret_ref", but the producer emits the shorter form; match the
# producer so real secret-fabrication errors flip the secrets row to error.
_SECRET_ERROR_CODES: frozenset[str] = frozenset({"missing_secret_ref", "fabricated_secret", "disallowed_secret_ref"})


def _build_secrets_row(validation: ValidationResult, inventory: list[SecretInventoryItem]) -> ReadinessRow:
    """error/ok/not_applicable per secret ref resolution.
    Keyed on ValidationError.error_code, not message substring.
    """
    secret_errors = [err for err in validation.errors if err.error_code in _SECRET_ERROR_CODES]
    if secret_errors:
        return ReadinessRow(
            id="secrets",
            label="Secrets",
            status="error",
            summary="Secret references unresolved",
            detail="\n".join(err.message for err in secret_errors),
            component_ids=tuple(err.component_id for err in secret_errors if err.component_id is not None),
        )
    secret_check = next(
        (c for c in validation.checks if c.name == "secret_refs"),
        None,
    )
    if secret_check is not None and secret_check.outcome_code == CHECK_OUTCOME_SECRET_REFS_NO_REFS:
        return ReadinessRow(
            id="secrets",
            label="Secrets",
            status="not_applicable",
            summary="No secret references in this composition",
            detail=None,
            component_ids=(),
        )
    if secret_check is None and not inventory:
        return ReadinessRow(
            id="secrets",
            label="Secrets",
            status="not_applicable",
            summary="No secret references in this composition",
            detail=None,
            component_ids=(),
        )
    if secret_check is None or secret_check.outcome_code in {
        CHECK_OUTCOME_SECRET_REFS_SKIPPED_NO_SERVICE,
        CHECK_OUTCOME_SKIPPED_AFTER_FAILURE,
    }:
        return ReadinessRow(
            id="secrets",
            label="Secrets",
            status="not_applicable",
            summary="Secret reference check did not run",
            detail=secret_check.detail if secret_check is not None else "Secret reference check was not recorded",
            component_ids=(),
        )
    if secret_check.outcome_code == CHECK_OUTCOME_SECRET_REFS_UNRESOLVED or not secret_check.passed:
        return ReadinessRow(
            id="secrets",
            label="Secrets",
            status="error",
            summary="Secret reference check failed",
            detail=secret_check.detail,
            component_ids=(),
        )
    if secret_check.outcome_code != CHECK_OUTCOME_SECRET_REFS_RESOLVED:
        raise RuntimeError(f"Unhandled secret_refs outcome_code: {secret_check.outcome_code}")
    return ReadinessRow(
        id="secrets",
        label="Secrets",
        status="ok",
        summary="All secret references resolve",
        detail=(f"{len(inventory)} secret(s) in your inventory" if inventory else "Composition references no secrets"),
        component_ids=(),
    )
