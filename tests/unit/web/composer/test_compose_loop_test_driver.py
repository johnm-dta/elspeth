"""Phase 3 compose-loop test-driver contract tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

from elspeth.web.composer.service import ComposerServiceImpl


@pytest.mark.asyncio
async def test_run_one_turn_for_test_requires_wired_sessions_service(
    composer_service_without_sessions_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    """The Task 0 driver must fail loudly when sessions persistence is absent."""

    driver = cast(Any, composer_service_without_sessions_service)
    with pytest.raises(RuntimeError, match="sessions_service not wired"):
        await driver._run_one_turn_for_test(
            session_id=result_session_id,
        )
