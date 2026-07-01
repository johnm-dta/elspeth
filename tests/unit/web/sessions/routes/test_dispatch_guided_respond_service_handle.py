"""Phase P5.4 — the guided dispatcher accepts a ComposerService handle + pass budget."""

from __future__ import annotations

import inspect

from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond


def test_dispatcher_accepts_composer_service_kwarg() -> None:
    sig = inspect.signature(_dispatch_guided_respond)
    assert "composer_service" in sig.parameters
    param = sig.parameters["composer_service"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    # Safe compatibility default so pre-P5 callers can omit this kwarg.
    # P5.6 fails closed when a tutorial/advisor-checkpoint profile sees None.
    assert param.default is None


def test_dispatcher_accepts_advisor_checkpoint_max_passes_kwarg() -> None:
    sig = inspect.signature(_dispatch_guided_respond)
    assert "advisor_checkpoint_max_passes" in sig.parameters
    param = sig.parameters["advisor_checkpoint_max_passes"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.annotation == "int | None"
    # Safe compatibility default only; tutorial/advisor-checkpoint profiles
    # fail closed if the route/test did not thread a concrete positive budget.
    assert param.default is None
