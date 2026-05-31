from __future__ import annotations

from typing import Literal

from ._helpers import (
    UUID,
    APIRouter,
    AuditIntegrityError,
    Depends,
    HTTPException,
    InterpretationChoice,
    InterpretationEventAlreadyResolvedError,
    InterpretationEventNotFoundError,
    InterpretationNodeMissingError,
    InterpretationNodePluginMutatedError,
    InterpretationOptOutResponse,
    InterpretationPlaceholderConsumedError,
    InterpretationResolveRequest,
    InterpretationResolveResponse,
    InterpretationSource,
    InterpretationUnsupportedChoiceError,
    ListInterpretationEventsResponse,
    OptOutSummaryResponse,
    Request,
    SessionServiceProtocol,
    UserIdentity,
    _extract_runtime_model_snapshot,
    _interpretation_event_response,
    _state_response,
    _verify_session_ownership,
    get_current_user,
)


def register_interpretation_routes(router: APIRouter) -> None:

    # ------------------------------------------------------------------ #
    # Interpretation events
    # ------------------------------------------------------------------ #
    #
    # DI / telemetry note:
    #   * Routes use ``request.app.state.session_service`` directly (the
    #     project's convention; no ``Depends(get_session_service)``).
    #   * ``_verify_session_ownership`` raises 404 on cross-session access,
    #     matching the IDOR-safe response of every other session-scoped
    #     route.  Operational IDOR telemetry is not emitted at the route
    #     level — the spec calls for it but the existing routes.py pattern
    #     (24+ call sites) lets the verifier raise without a dedicated
    #     signal.  Adopting that pattern here keeps the route surface
    #     consistent; a future IDOR-signal helper added to the verifier
    #     itself would apply to all sites at once.
    #   * User-decision audit writes (resolve, opt-out) are audit-primary:
    #     the ``interpretation_events`` row IS the record (F-15).  No
    #     ephemeral telemetry signal is emitted at the route.

    @router.post(
        "/{session_id}/interpretations/{event_id}/resolve",
        response_model=InterpretationResolveResponse,
    )
    async def resolve_interpretation(
        session_id: UUID,
        event_id: UUID,
        body: InterpretationResolveRequest,
        raw_request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> InterpretationResolveResponse:
        """User-driven resolve of a pending interpretation event.

        Tier-3 boundary: request body validated by Pydantic; ``choice`` /
        ``amended_value`` consistency enforced by the model validator on
        :class:`InterpretationResolveRequest`.  Service-side semantic
        constraints (event must be pending; node must still exist;
        prompt-template patch must succeed) are enforced inside
        :meth:`SessionServiceProtocol.resolve_interpretation_event`.

        F-14 business-rule split: the route passes only ``choice`` and
        ``amended_value``; the service computes ``accepted_value`` from
        the pending row's ``llm_draft`` when
        ``choice == 'accepted_as_drafted'``.  Do NOT compute
        ``accepted_value`` here — a second caller (CLI, admin route)
        would need the same branch.

        F-19 runtime-model snapshot: the route extracts
        ``runtime_model_identifier_at_resolve`` and
        ``runtime_model_version_at_resolve`` from the affected LLM
        transform's config on the current composition state, passing
        them to the service for inclusion on the UPDATE.  These columns
        are nullable; if the node has no explicit ``model`` /
        ``model_version`` config, ``None`` is recorded — the
        audit-readiness panel surfaces NULL distinctly from "drift" so
        operators see "no runtime model pinned" rather than a
        fabricated default.

        Status-code mapping is driven by typed service exceptions. Unknown
        event / cross-session remains 404 and double-resolve remains 409;
        prompt-patch failures are 422 so an existing event is never laundered
        as "not found". Tier-1 audit anomalies surface as 500.
        """
        await _verify_session_ownership(session_id, user, raw_request)  # 404 on IDOR
        service: SessionServiceProtocol = raw_request.app.state.session_service
        actor = f"user:{user.user_id}"

        # F-19 extraction.  Performed BEFORE the service call so the
        # service receives the audit values it should write on the
        # UPDATE.  Two reads (pending event + current state) outside
        # the resolve transaction — accepted because the model snapshot
        # is informational, and any race window between extraction and
        # the service's internal state-fetch only affects which "live"
        # model identifier is recorded, not transactional integrity.
        # If the pending row does not exist for this session, the
        # extraction returns ``(None, None)`` and the resolve call
        # raises ``ValueError`` (mapped to 404 below).
        interpretation_events = await service.list_interpretation_events(session_id, status="all")
        pending_event = next((e for e in interpretation_events if e.id == event_id), None)
        affected_node_id = pending_event.affected_node_id if pending_event is not None else None
        current_state = await service.get_current_state(session_id)
        if current_state is None:
            runtime_model_identifier, runtime_model_version = None, None
        else:
            runtime_model_identifier, runtime_model_version = _extract_runtime_model_snapshot(current_state, affected_node_id)

        try:
            event, new_state = await service.resolve_interpretation_event(
                session_id=session_id,
                event_id=event_id,
                choice=InterpretationChoice(body.choice),
                amended_value=body.amended_value,
                actor=actor,
                runtime_model_identifier=runtime_model_identifier,
                runtime_model_version=runtime_model_version,
            )
        except InterpretationEventAlreadyResolvedError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "interpretation_already_resolved",
                    "message": "Interpretation event is already resolved.",
                },
            ) from exc
        except InterpretationEventNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "interpretation_event_not_found",
                    "message": "Interpretation event not found.",
                },
            ) from exc
        except InterpretationNodeMissingError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "interpretation_node_missing",
                    "message": "The affected LLM node is no longer present in the current composition state.",
                },
            ) from exc
        except InterpretationNodePluginMutatedError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "interpretation_node_mutated",
                    "message": "The affected node is no longer an LLM transform.",
                },
            ) from exc
        except InterpretationPlaceholderConsumedError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "interpretation_placeholder_unavailable",
                    "message": "The affected LLM prompt no longer contains the expected interpretation placeholder.",
                },
            ) from exc
        except InterpretationUnsupportedChoiceError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "interpretation_resolution_unsupported",
                    "message": "This interpretation kind does not support inline amendment in this release.",
                },
            ) from exc
        except AuditIntegrityError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "interpretation_audit_integrity_error",
                    "message": "Interpretation resolution hit an audit-integrity anomaly.",
                },
            ) from exc

        return InterpretationResolveResponse(
            event=_interpretation_event_response(event),
            new_state=_state_response(new_state),
        )

    @router.get(
        "/{session_id}/interpretations",
        response_model=ListInterpretationEventsResponse,
    )
    async def list_interpretations(
        session_id: UUID,
        raw_request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        status: Literal["pending", "all"] = "all",
    ) -> ListInterpretationEventsResponse:
        """List interpretation events for the session.

        Used by the frontend on session reload to rehydrate pending
        review affordances, and by the audit-readiness panel for counts.
        """
        await _verify_session_ownership(session_id, user, raw_request)  # 404 on IDOR
        service: SessionServiceProtocol = raw_request.app.state.session_service
        events = await service.list_interpretation_events(session_id, status=status)
        return ListInterpretationEventsResponse(
            events=[_interpretation_event_response(e) for e in events],
        )

    @router.post(
        "/{session_id}/interpretations/opt_out",
        response_model=InterpretationOptOutResponse,
    )
    async def opt_out_of_interpretations(
        session_id: UUID,
        raw_request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> InterpretationOptOutResponse:
        """Record the per-session 'stop asking about interpretations' decision.

        Single transaction (enforced inside
        :meth:`SessionServiceProtocol.record_session_interpretation_opt_out`):
        flip ``sessions.interpretation_review_disabled`` to ``True`` and
        write an ``interpretation_events`` row with ``choice='opted_out'``
        and ``interpretation_source='auto_interpreted_opt_out'``.  Routes
        to ``interpretation_events_table``, NOT ``proposal_events_table``.

        F-29 idempotency: a second POST for the same session returns the
        existing opted-out row (FIRST opt-out timestamp is authoritative)
        without inserting a duplicate.
        """
        await _verify_session_ownership(session_id, user, raw_request)  # 404 on IDOR
        service: SessionServiceProtocol = raw_request.app.state.session_service
        actor = f"user:{user.user_id}"
        record = await service.record_session_interpretation_opt_out(
            session_id=session_id,
            actor=actor,
        )
        # The opt-out row's ``resolved_at`` carries the opt-out timestamp.
        # The service guarantees a non-NULL value on every opt-out row
        # (idempotent path returns the existing row; insertion path uses
        # ``now``); a NULL here would be a Tier-1 anomaly.
        if record.resolved_at is None:
            raise AuditIntegrityError(
                f"Tier 1 audit anomaly: opt-out interpretation event "
                f"{record.id!r} has NULL resolved_at; the writer guarantees "
                f"a non-NULL timestamp on every opted_out row."
            )
        return InterpretationOptOutResponse(
            session_id=record.session_id,
            interpretation_review_disabled=True,
            opted_out_at=record.resolved_at,
        )

    @router.get(
        "/{session_id}/interpretations/opt_out_summary",
        response_model=OptOutSummaryResponse,
    )
    async def opt_out_summary(
        session_id: UUID,
        raw_request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> OptOutSummaryResponse:
        """Retroactive audit of auto-baked interpretations (F-22).

        After a session opts out of interpretation review, the
        composer-LLM continues to auto-bake interpretations.  This
        surface returns every ``interpretation_events`` row whose
        ``interpretation_source`` is ``auto_interpreted_opt_out`` or
        ``auto_interpreted_no_surfaces``, ordered by ``created_at``.
        ``user_approved`` rows are excluded — the standard
        ``GET /interpretations`` route is the right surface for those.

        Closes the "click opt-out once, dozens of auto-interpretations
        accumulate invisibly" audit gap by giving the user one place to
        browse what the LLM baked during the opted-out portion of the
        session.
        """
        await _verify_session_ownership(session_id, user, raw_request)  # 404 on IDOR
        service: SessionServiceProtocol = raw_request.app.state.session_service
        events = await service.list_interpretation_events(
            session_id,
            sources=(
                InterpretationSource.AUTO_INTERPRETED_OPT_OUT,
                InterpretationSource.AUTO_INTERPRETED_NO_SURFACES,
            ),
        )
        return OptOutSummaryResponse(
            events=[_interpretation_event_response(e) for e in events],
        )
