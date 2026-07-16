"""Web-safe sink-effect recovery diagnostics."""

from __future__ import annotations

from elspeth.contracts.sink_effects import SinkEffectAttemptAction, SinkEffectAttemptRequest
from elspeth.web.audit_readiness.service import load_sink_effect_diagnostic
from tests.fixtures.landscape import make_factory, make_landscape_db
from tests.unit.core.landscape.test_sink_effect_finalization import _prepared


def test_web_diagnostic_exposes_no_publication_and_response_loss_without_raw_bodies() -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        effect, _members, lease = _prepared(factory, count=1, replacing_target=True)
        attempt = factory.execution.sink_effects.begin_attempt(
            SinkEffectAttemptRequest(
                effect_id=effect.effect_id,
                member_ordinal=None,
                generation=lease.generation,
                action=SinkEffectAttemptAction.COMMIT,
                request_hash="f" * 64,
            )
        )
        factory.execution.sink_effects.mark_response_lost(attempt.attempt_id)

        diagnostic = load_sink_effect_diagnostic(db, effect.effect_id)

        assert diagnostic is not None
        assert diagnostic.effect_id == effect.effect_id
        assert diagnostic.state == "in_flight"
        assert diagnostic.lease_generation == lease.generation
        assert diagnostic.member_progress == {"prepared": 1}
        assert diagnostic.response_lost_attempts == 1
        assert diagnostic.operator_guidance.startswith("Do not retry publication speculatively")
        payload = diagnostic.model_dump(mode="json")
        assert "target_json" not in payload
        assert "plan_json" not in payload
        assert "evidence_json" not in payload["attempts"][0]
    finally:
        db.close()
