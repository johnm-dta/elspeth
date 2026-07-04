"""ReadinessService — pure aggregation of audit signals into a snapshot.
No new validation logic. Layer: L3 (application).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationEventRecord,
    InterpretationSource,
)
from elspeth.contracts.enums import Determinism
from elspeth.contracts.plugin_protocols import SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.audit_readiness.models import (
    AuditReadinessSnapshot,
    ReadinessRow,
)
from elspeth.web.catalog.schemas import PluginKind
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


@lru_cache(maxsize=1)
def _plugin_catalog_snapshot() -> dict[PluginKind, dict[str, type[Any]]]:
    """Build the live builtin-plugin catalog once and freeze it.

    Returns a per-kind mapping ``plugin_name → plugin_class``. Both
    ``_is_registered_plugin`` (membership query) and
    ``_get_plugin_class_for_kind`` (class resolution) read from this
    single snapshot, so the two helpers cannot disagree about what is
    and isn't in the catalog. The previous implementation built two
    independent ``PluginManager`` instances and relied on import-time
    stability to keep them in sync — that "shared snapshot" assumption
    is now mechanically enforced rather than load-bearing-but-documented.

    Inlined from the now-deleted ``elspeth.web.audit_readiness.trust``
    module (Phase 7A No-Legacy commitment). Layer: L3.
    """
    from elspeth.plugins.infrastructure.manager import PluginManager

    manager = PluginManager()
    manager.register_builtin_plugins()
    return {
        "source": {cls.name: cls for cls in manager.get_sources()},
        "transform": {cls.name: cls for cls in manager.get_transforms()},
        "sink": {cls.name: cls for cls in manager.get_sinks()},
    }


def _is_registered_plugin(kind: PluginKind, name: str) -> bool:
    """Return True when ``name`` exists in the live catalog for ``kind``.

    Raises:
        ValueError: when ``kind`` is not one of "source", "transform",
            or "sink". Unreachable under the ``PluginKind`` Literal type;
            retained as an offensive-programming guard so a non-typed
            caller crashes loudly rather than returning a misleading
            False.
    """
    snapshot = _plugin_catalog_snapshot()
    if kind not in snapshot:
        raise ValueError(f"unknown plugin kind: {kind!r}")
    return name in snapshot[kind]


def _get_plugin_class_for_kind(kind: PluginKind, name: str) -> type[SourceProtocol] | type[TransformProtocol] | type[SinkProtocol]:
    """Return the registered plugin class for (kind, name).

    Layer: L3. Callers MUST guard with ``_is_registered_plugin()``
    first (as ``_record()`` does). The two helpers share a single
    ``_plugin_catalog_snapshot()`` so under correct usage the
    RuntimeError branch below is unreachable.

    Raises:
        ValueError: when ``kind`` is not one of "source", "transform",
            or "sink". Mirrors ``_is_registered_plugin``.
        RuntimeError: when ``name`` is not in the catalog for ``kind``.
            Indicates a contract violation in this module's callers —
            either the ``_is_registered_plugin`` guard was skipped, or
            the snapshot was rebuilt between the guard call and this
            resolution call (which the ``lru_cache`` rules out under
            normal use). Replaces the bare ``StopIteration`` that the
            previous ``next(...)`` form would have raised — that
            exception type carried no diagnostic.
    """
    snapshot = _plugin_catalog_snapshot()
    if kind not in snapshot:
        raise ValueError(f"unknown plugin kind: {kind!r}")
    catalog = snapshot[kind]
    try:
        plugin_cls = catalog[name]
    except KeyError as exc:
        raise RuntimeError(
            f"plugin {kind!r}/{name!r} not in catalog snapshot — caller "
            f"must guard with _is_registered_plugin() first. The two "
            f"helpers share a single _plugin_catalog_snapshot(), so "
            f"reaching this branch means the guard was skipped. This "
            f"is an audit_readiness contract violation, not a plugin "
            f"issue."
        ) from exc
    return cast(
        "type[SourceProtocol] | type[TransformProtocol] | type[SinkProtocol]",
        plugin_cls,
    )


class CompositionStateNotFoundError(LookupError):
    """Raised when a session has no current composition state."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        super().__init__("No composition state for this session")


class _ExecutionServiceLike(Protocol):
    async def validate_state(
        self,
        state: CompositionState,
        *,
        user_id: str | None = None,
        session_id: UUID | None = None,
    ) -> ValidationResult: ...


class _SessionServiceLike(Protocol):
    # session_id is UUID to match SessionServiceImpl.get_current_state (web/sessions/service.py:1884).
    async def get_current_state(self, session_id: UUID) -> Any: ...

    # Mirror of SessionServiceProtocol.list_interpretation_events
    # (web/sessions/protocol.py:757). Used by the llm_interpretations
    # row to count pending/resolved events and to detect a session-wide
    # opt-out (via the AUTO_INTERPRETED_OPT_OUT source filter).
    async def list_interpretation_events(
        self,
        session_id: UUID,
        *,
        status: Literal["pending", "all"] = "all",
        composition_state_id: UUID | None = None,
        sources: Sequence[InterpretationSource] | None = None,
    ) -> list[InterpretationEventRecord]: ...


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
        # Tier-1 read: record.id is UUID per CompositionStateRecord
        # (web/sessions/protocol.py:369). It is the composition-state-id
        # that the llm_interpretations row scopes its event lookup to.
        composition_state_id: UUID = record.id
        state: CompositionState = self._state_from_record(record)
        validation = await self._execution_service.validate_state(state, user_id=user_id, session_id=session_id)
        inventory = await run_sync_in_worker(
            self._scoped_secret_resolver.list_refs,
            user_id,  # scoped_secret_resolver.list_refs takes user_id only
        )
        # Pre-fetch interpretation-event signal for the llm_interpretations
        # row. Two separate reads because:
        #
        #   (a) The opt-out indicator is an AUTO_INTERPRETED_OPT_OUT row
        #       written with composition_state_id=NULL
        #       (web/sessions/service.py:2188). A WHERE clause filtering
        #       by composition_state_id would never match it.
        #
        #   (b) The pending/resolved counts are scoped to the CURRENT
        #       composition-state so a stale resolved event from a prior
        #       state cannot flip the row from "no events yet" to "ok".
        #
        # Both reads are skipped when the composition has no LLM
        # transforms — the row short-circuits to not_applicable and
        # nothing further is queried.
        has_llm = _composition_has_llm_transform(state)
        interpretation_events: tuple[InterpretationEventRecord, ...] = ()
        opted_out = False
        if has_llm:
            opt_out_rows = await self._session_service.list_interpretation_events(
                session_id,
                status="all",
                composition_state_id=None,
                sources=(InterpretationSource.AUTO_INTERPRETED_OPT_OUT,),
            )
            opted_out = bool(opt_out_rows)
            if not opted_out:
                interpretation_events = tuple(
                    await self._session_service.list_interpretation_events(
                        session_id,
                        status="all",
                        composition_state_id=composition_state_id,
                    )
                )
        rows: tuple[ReadinessRow, ...] = (
            _build_validation_row(validation),
            _build_plugin_trust_row(state),
            _build_provenance_row(validation),
            _build_retention_row(self._settings.payload_store_retention_days),
            _build_llm_interpretations_row(
                has_llm=has_llm,
                opted_out=opted_out,
                events=interpretation_events,
            ),
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
            validation_result=validation,
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
    summary = "1 problem to fix — see details" if len(result.errors) == 1 else f"{len(result.errors)} problems to fix — see details"
    # Plain message text only — the engineer-grade "[component_type] component_id:"
    # prefix (which rendered "[unknown] unknown: …" for settings-level findings)
    # is a Tier-3 lingo leak on a novice surface (elspeth-901a404926). The web
    # panel re-humanises each finding through the shared humaniser and resolves
    # the component_ids list to plain step phrases; this joined text is the
    # fallback body for non-web consumers (read-only export, MCP audit info).
    detail = "\n".join(err.message for err in result.errors)
    return ReadinessRow(
        id="validation",
        label="Validation",
        status="error",
        summary=summary,
        detail=detail,
        component_ids=component_ids,
    )


# Determinism values that classify a Transform as INTERNAL — i.e. NOT
# audit-flagged. The transform-boundary predicate is derived by
# exclusion: every Determinism value NOT in this set is audit-flagged.
#
# The set is declared as the explicit exclusion list (rather than as
# the audit-flagged inclusion list) so that adding a new Determinism
# value to ``contracts/enums.py`` forces an explicit decision rather
# than silently inheriting "internal" by omission:
#
#   - Inclusive form (rejected): a new ``Determinism.GPU_ACCELERATED``
#     added without updating an inclusive ``_AUDIT_FLAGGED_DETERMINISMS``
#     set silently classifies as INTERNAL. The auditor never sees a
#     new audit-relevant signal that the language already named.
#
#   - Exclusive form (chosen): a new value defaults to BOUNDARY unless
#     the author explicitly adds it to ``_INTERNAL_TRANSFORM_DETERMINISMS``.
#     Audit-relevance becomes the default classification.
#
# The "force a conscious decision" property of this code path is
# completed by the per-plugin parity test
# (``tests/unit/web/audit_readiness/test_boundary_predicate_parity.py``):
# when a new Determinism value is added AND used on a builtin plugin,
# the per-name expected-determinism map in
# ``boundary_expectations.py`` no longer matches, the parity test
# fails, and the author must either pin the new declared value in the
# expectations map (acknowledging the new audit-relevant signal) or
# add the value to ``_INTERNAL_TRANSFORM_DETERMINISMS`` (declaring it
# reproducibility-clean). The exclusion form alone biases the default;
# the parity test forces the decision to be recorded.
#
# A defensive parity test asserts the exclusion set is a subset of
# ``Determinism``; if a future rename drops a value from the enum the
# test surfaces immediately.
#
# Current internal classifications (every other Determinism value
# automatically classifies as audit-flagged for Transforms):
#
#   DETERMINISTIC — pure functions: replay reproduces output from input.
#   SEEDED         — pseudo-random, seed captured: replay reproducible.
#   IO_READ        — Transforms only; reads from filesystem/env. Sources
#                    short-circuit boundary via the kind check, so a
#                    rare IO_READ Transform is internal here.
#   IO_WRITE       — Transforms only; writes to filesystem/env. Sinks
#                    short-circuit boundary via the kind check, so a
#                    rare IO_WRITE Transform is internal here.
#
# The complement (currently {EXTERNAL_CALL, NON_DETERMINISTIC}) carries
# two audit semantics that happen to share a signal:
#
#   EXTERNAL_CALL — crosses an external trust boundary via network
#                   request (web_scrape, RAG, Azure moderation). The
#                   audit trail records request + response so replay
#                   is reproducible.
#
#   NON_DETERMINISTIC — output is not reproducible from inputs alone
#                   (LLM completions). The audit trail records the
#                   verbatim output. A hypothetical future plugin that
#                   is NON_DETERMINISTIC without making an external
#                   call (e.g. unseeded randomness) would also
#                   classify as audit-flagged — correct under
#                   ELSPETH's auditability standard.
_INTERNAL_TRANSFORM_DETERMINISMS: frozenset[Determinism] = frozenset(
    {
        Determinism.DETERMINISTIC,
        Determinism.SEEDED,
        Determinism.IO_READ,
        Determinism.IO_WRITE,
    },
)

_AUDIT_FLAGGED_DETERMINISMS: frozenset[Determinism] = frozenset(Determinism) - _INTERNAL_TRANSFORM_DETERMINISMS

# Module-load assertion: the exclusion list must be a subset of the
# live enum. A drop or rename of a Determinism member that leaves a
# stale name in ``_INTERNAL_TRANSFORM_DETERMINISMS`` would otherwise
# silently shrink ``_AUDIT_FLAGGED_DETERMINISMS`` without test signal.
# This assertion runs at import time so a stale exclusion surfaces
# before any test exercises the predicate.
assert frozenset(Determinism) >= _INTERNAL_TRANSFORM_DETERMINISMS, (
    "_INTERNAL_TRANSFORM_DETERMINISMS contains values not in Determinism; "
    f"stale members: {sorted(_INTERNAL_TRANSFORM_DETERMINISMS - frozenset(Determinism))}. "
    "Remove the stale entries or restore the missing enum members."
)

# Version identifier for the boundary-classification rule encoded by
# ``_AUDIT_FLAGGED_DETERMINISMS`` and ``_build_plugin_trust_row``. The
# rule is presently code-only: an auditor at T+12 months has no
# in-snapshot way to reconstruct which rule produced a verdict shown in
# the readiness panel. That is acceptable today because the panel is a
# UX surface, not a legal record. If a future phase persists derived
# audit characteristics into the Landscape (the legal record), the
# version pin recorded here MUST be stamped alongside each persisted
# verdict so historical rows remain reproducible.
#
# Bump on every semantic change to either the predicate or the
# ``_AUDIT_FLAGGED_DETERMINISMS`` set (membership, name, or wire-format
# meaning). The pin is opaque to consumers — they record it verbatim and
# never parse the version string.
_BOUNDARY_RULE_VERSION = "phase-7a-v1"


def _build_plugin_trust_row(state: CompositionState) -> ReadinessRow:
    """Classify every plugin in the composition (boundary vs internal).

    A plugin is surfaced in the plugin-trust row when:
      - it is a Source — by definition reads external data into the pipeline
      - it is a Sink — by definition emits pipeline data to an external
        destination (file, database, blob store, downstream service)
      - it is a Transform whose declared determinism is in
        ``_AUDIT_FLAGGED_DETERMINISMS`` (EXTERNAL_CALL or NON_DETERMINISTIC)

    The predicate is derived from (kind, determinism) so any future plugin
    is classified correctly at registration time without a separate
    declared attribute.

    The rule version encoded by this predicate and
    ``_AUDIT_FLAGGED_DETERMINISMS`` is ``_BOUNDARY_RULE_VERSION``
    (currently ``"phase-7a-v1"``). Bump that constant on any semantic
    change here. See its module-level docstring for the
    persistence-vs-UX rationale.

    Cross-reference: ``elspeth.web.catalog.service._derive_audit_characteristics``
    consumes the same ``determinism`` attribute to compose display chips
    for the catalog card; the two surfaces deliberately differ in their
    treatment of kind-default determinism — see that function's docstring
    for the divergence rationale.
    """
    boundary: list[tuple[str, str, str]] = []
    unknown: list[tuple[str, str]] = []

    def _record(kind: PluginKind, component_id: str, name: str | None) -> None:
        if name is None or not _is_registered_plugin(kind, name):
            unknown.append((kind, component_id))
            return
        plugin_cls = _get_plugin_class_for_kind(kind, name)
        if kind in ("source", "sink") or plugin_cls.determinism in _AUDIT_FLAGGED_DETERMINISMS:
            boundary.append((kind, component_id, name))

    for source_name, source in state.sources.items():
        component_id = "source" if source_name == "source" else f"source:{source_name}"
        _record("source", component_id, source.plugin)
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
        if not result.is_valid:
            return ReadinessRow(
                id="provenance",
                label="Provenance",
                status="not_applicable",
                summary="Provenance check did not run",
                detail="Validation failed before provenance advisory analysis could run",
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


def _composition_has_llm_transform(state: CompositionState) -> bool:
    """Predicate: does this composition contain at least one ``llm`` transform?

    Drives the short-circuit in ``compute_snapshot`` that skips the
    interpretation-event reads when no LLM transforms are present —
    a composition without LLM transforms cannot have interpretation
    events bound to its nodes.
    """
    return any(n.node_type == "transform" and n.plugin == "llm" for n in state.nodes)


def _build_llm_interpretations_row(
    *,
    has_llm: bool,
    opted_out: bool,
    events: Sequence[InterpretationEventRecord],
) -> ReadinessRow:
    """Project interpretation-event state into the panel row.

    Phase 5b Task 10 (18a-phase-5b-backend.md §Task 10). Status mapping:

      - No LLM transforms in composition → not_applicable
      - LLM transforms present, session opted out → not_applicable
        (with an opt-out note in the summary)
      - LLM transforms present, no interpretation events for the
        current composition state → not_applicable (the surfacing has
        simply not been triggered yet; the row flips to ``warning`` on
        the first ``request_interpretation_review`` call)
      - Any PENDING event for the current composition state → warning
      - All events resolved, at least one present → ok

    Auto-interpreted-no-surfaces rows (rate-cap-baked-in interpretation,
    structurally distinct from session-wide opt-out per
    ``contracts/composer_interpretation.py``) count as resolved
    contributors — they are not session-wide opt-outs and they are
    not PENDING.

    The ``component_ids`` projection lists the LLM-transform node ids
    referenced by events with that node populated, deduplicated and
    sorted; opt-out rows have ``affected_node_id=None`` and contribute
    nothing.
    """
    if not has_llm:
        return ReadinessRow(
            id="llm_interpretations",
            label="LLM interpretations",
            status="not_applicable",
            summary="No LLM transforms in this composition",
            detail=None,
            component_ids=(),
        )
    if opted_out:
        return ReadinessRow(
            id="llm_interpretations",
            label="LLM interpretations",
            status="not_applicable",
            summary="Session opted out of interpretation review",
            detail=(
                "The user has clicked 'stop asking' for this session. "
                "No further interpretation surfacings will occur and "
                "this row is informational only."
            ),
            component_ids=(),
        )
    if not events:
        return ReadinessRow(
            id="llm_interpretations",
            label="LLM interpretations",
            status="not_applicable",
            summary="No interpretation events yet for this composition",
            detail=None,
            component_ids=(),
        )
    pending = [e for e in events if e.choice == InterpretationChoice.PENDING]
    component_ids = tuple(sorted({e.affected_node_id for e in events if e.affected_node_id is not None}))
    if pending:
        return ReadinessRow(
            id="llm_interpretations",
            label="LLM interpretations",
            status="warning",
            summary=(f"{len(pending)} pending interpretation review(s)" if len(pending) != 1 else "1 pending interpretation review"),
            detail="\n".join(
                f"- {e.user_term} (node {e.affected_node_id})"
                for e in pending
                # affected_node_id and user_term are NOT NULL for
                # PENDING rows (interpretation_source=USER_APPROVED is
                # the only source that produces a PENDING row, and the
                # source-conditional CHECK constraint requires both
                # fields to be populated). Guard anyway so a future
                # source that emits PENDING rows with NULL fields would
                # be filtered out rather than rendered as "None".
                if e.affected_node_id is not None and e.user_term is not None
            )
            or None,
            component_ids=component_ids,
        )
    # ``model_identifier`` is the composer model that drafted the
    # interpretation surface; ``runtime_model_identifier_at_resolve`` is the
    # pipeline model that will execute the resolved prompt. They are different
    # roles, so comparing them would produce false "rotated model" warnings.
    # A future same-role drift signal must capture the runtime model at both
    # surfacing and resolve time before warning here.
    return ReadinessRow(
        id="llm_interpretations",
        label="LLM interpretations",
        status="ok",
        summary=(f"{len(events)} interpretation(s) resolved" if len(events) != 1 else "1 interpretation resolved"),
        detail=None,
        component_ids=component_ids,
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
