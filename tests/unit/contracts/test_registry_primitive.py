"""Shared frozen-registry primitive tests."""

from __future__ import annotations

from threading import Event, Thread

import pytest

from elspeth.contracts.registry_primitive import FrozenRegistry


def test_write_unfrozen_raises_with_caller_error_after_freeze() -> None:
    registry: FrozenRegistry[str, dict[str, str]] = FrozenRegistry(
        name="test-registry",
        auxiliary={},
    )

    registry.freeze()

    with pytest.raises(RuntimeError, match="test-registry sealed"), registry.write_unfrozen(lambda: RuntimeError("test-registry sealed")):
        registry.items.append("unreachable")


def test_write_unfrozen_raises_default_runtime_error_after_freeze() -> None:
    registry: FrozenRegistry[str, dict[str, str]] = FrozenRegistry(
        name="test-registry",
        auxiliary={},
    )

    registry.freeze()

    with pytest.raises(RuntimeError, match="test-registry registry is frozen"), registry.write_unfrozen():
        registry.items.append("unreachable")


def test_write_unfrozen_allows_reentrant_mutations_before_freeze() -> None:
    registry: FrozenRegistry[str, dict[str, str]] = FrozenRegistry(
        name="test-registry",
        auxiliary={},
    )

    with registry.write_unfrozen():
        registry.items.append("outer")
        with registry.write_unfrozen():
            registry.items.append("inner")

    assert registry.items == ["outer", "inner"]


def test_freeze_is_idempotent() -> None:
    registry: FrozenRegistry[str, dict[str, str]] = FrozenRegistry(
        name="test-registry",
        auxiliary={},
    )

    registry.freeze()
    registry.freeze()

    assert registry.is_frozen()


def test_freeze_waits_for_in_flight_mutation_lock() -> None:
    registry: FrozenRegistry[str, dict[str, str]] = FrozenRegistry(
        name="test-registry",
        auxiliary={},
    )
    mutation_entered = Event()
    release_mutation = Event()
    freeze_finished = Event()

    def hold_mutation_lock() -> None:
        with registry.write_unfrozen(lambda: RuntimeError("unexpected freeze")):
            mutation_entered.set()
            release_mutation.wait(timeout=1.0)

    def freeze_registry() -> None:
        registry.freeze()
        freeze_finished.set()

    mutation_thread = Thread(target=hold_mutation_lock)
    mutation_thread.start()
    assert mutation_entered.wait(timeout=1.0)

    freeze_thread = Thread(target=freeze_registry)
    freeze_thread.start()
    assert not freeze_finished.wait(timeout=0.05), "freeze() completed while a mutation held the registry lock"

    release_mutation.set()
    mutation_thread.join(timeout=1.0)
    freeze_thread.join(timeout=1.0)

    assert not mutation_thread.is_alive()
    assert not freeze_thread.is_alive()
    assert registry.is_frozen()
