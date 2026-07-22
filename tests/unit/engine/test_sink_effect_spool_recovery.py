"""Re-drives must recover when the effect body spool is lost (elspeth-501ce2e9e9).

The durable-effect protocol stores the PREPARED plan and member payloads in
the landscape DB, but the staged body lives in a filesystem spool. Host
reboots, container restarts, and /tmp age-cleaners can remove that spool while
the ledger survives. Commit must then re-derive the body from durable member
payloads and verify it against the plan-sealed staged hash instead of raising
``RemoteObjectPreconditionError`` on every re-drive and permanently wedging
the sink stream and its successors.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from elspeth.engine.executors.sink_effects import (
    SinkEffectCoordinator,
    SinkEffectExecutionSeam,
    SinkEffectInjectedFault,
)
from tests.fixtures.landscape import make_factory, make_landscape_db
from tests.fixtures.stores import MockPayloadStore
from tests.unit.core.landscape.test_sink_effect_reservation import _pipeline_members
from tests.unit.engine.test_sink_effect_executor import _execution_request
from tests.unit.plugins.sinks.test_remote_object_sink_effects import _s3, _S3Store

_SPOOL = "spool"


@pytest.fixture(autouse=True)
def _isolated_spool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELSPETH_EFFECT_SPOOL_DIR", str(tmp_path / _SPOOL))


def _crash_at(target: SinkEffectExecutionSeam):
    def hook(seam: SinkEffectExecutionSeam) -> None:
        if seam is target:
            raise SinkEffectInjectedFault(seam)

    return hook


def test_redrive_republishes_after_spool_loss(tmp_path: Path) -> None:
    """Crash after PREPARE, lose the spool, re-drive: the effect must publish."""
    db = make_landscape_db()
    try:
        payload_store = MockPayloadStore()
        factory = make_factory(db, payload_store=payload_store)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        store = _S3Store()

        # Attempt 1 dies at the seam before any provider I/O: the plan and
        # member payloads are durable, the staged body sits in the spool, and
        # nothing has been published.
        first = SinkEffectCoordinator(
            factory=factory,
            worker_id="worker-a",
            fault_hook=_crash_at(SinkEffectExecutionSeam.BEFORE_EFFECT),
        )
        with pytest.raises(SinkEffectInjectedFault):
            first.execute(_execution_request(run_id, sink_id, members), _s3(store))
        assert store.value is None

        # The host reboots: the landscape DB survives, the spool does not.
        shutil.rmtree(tmp_path / _SPOOL)

        # Same worker identity restarts after the reboot; a different worker
        # would (correctly) wait out the TTL fence before taking over.
        second = SinkEffectCoordinator(
            factory=make_factory(db, payload_store=payload_store),
            worker_id="worker-a",
        )
        result = second.execute(_execution_request(run_id, sink_id, members), _s3(store))

        assert result.effect.publication_performed is True
        assert store.value is not None
        assert json.loads(store.value.body) == [{"ordinal": 0}]
    finally:
        db.close()
