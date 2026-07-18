from __future__ import annotations

import asyncio
import gc
import weakref

from elspeth.web.sessions.routes._helpers import _SessionComposeLockRegistry


def test_cleanup_session_lock_weakly_reclaims_idle_base_and_namespaced_locks() -> None:
    async def exercise() -> None:
        registry = _SessionComposeLockRegistry()
        session_id = "11111111-1111-4111-8111-111111111111"
        other_session_id = "22222222-2222-4222-8222-222222222222"
        base = await registry.get_lock(session_id)
        admission = await registry.get_lock(f"{session_id}:guided-respond-admission")
        other = await registry.get_lock(other_session_id)
        base_ref = weakref.ref(base)
        admission_ref = weakref.ref(admission)

        await registry.cleanup_session_lock(session_id)

        assert await registry.get_lock(session_id) is base
        assert await registry.get_lock(f"{session_id}:guided-respond-admission") is admission
        assert await registry.get_lock(other_session_id) is other
        del base, admission
        gc.collect()
        assert base_ref() is None
        assert admission_ref() is None
        assert await registry.get_lock(session_id) is not None
        assert await registry.get_lock(f"{session_id}:guided-respond-admission") is not None

    asyncio.run(exercise())


def test_cleanup_session_lock_preserves_held_admission_lock_until_waiters_drain() -> None:
    async def exercise() -> None:
        registry = _SessionComposeLockRegistry()
        key = "11111111-1111-4111-8111-111111111111:guided-respond-admission"
        admission = await registry.get_lock(key)
        await admission.acquire()
        waiter = asyncio.create_task(admission.acquire())
        await asyncio.sleep(0)

        await registry.cleanup_session_lock(key.split(":", maxsplit=1)[0])

        assert await registry.get_lock(key) is admission
        admission.release()
        await waiter
        assert await registry.get_lock(key) is admission
        admission.release()
        admission_ref = weakref.ref(admission)
        del admission, waiter
        gc.collect()
        assert admission_ref() is None
        assert await registry.get_lock(key) is not None

    asyncio.run(exercise())
