"""Unit tests for ``elspeth.contracts.trust_boundary``.

Covers the decorator's metadata attachment, signature validation,
tier restriction, and ``functools.wraps`` semantics. No runtime
behavioural enforcement is tested because the decorator deliberately
performs none — those concerns belong to the static analyzer and
companion CI gates.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest

from elspeth.contracts.trust_boundary import (
    TrustBoundaryMetadata,
    trust_boundary,
)


def test_decorator_passes_through_call_and_attaches_metadata() -> None:
    """Wrapper invokes the original function and exposes ``__trust_boundary__``."""

    @trust_boundary(
        tier=3,
        source="LLM tool-call arguments",
        source_param="arguments",
        suppresses=("R1", "R5"),
        invariant="raises ToolArgumentError on shape mismatch",
        test_ref="tests/unit/example.py::test_rejects_malformed",
        test_fingerprint="abc123",
    )
    def handler(arguments: dict[str, Any]) -> str:
        return f"processed:{len(arguments)}"

    assert handler({"a": 1, "b": 2}) == "processed:2"

    metadata = handler.__trust_boundary__  # type: ignore[attr-defined]
    assert isinstance(metadata, TrustBoundaryMetadata)
    assert metadata.tier == 3
    assert metadata.source == "LLM tool-call arguments"
    assert metadata.source_param == "arguments"
    assert metadata.suppresses == ("R1", "R5")
    assert metadata.invariant == "raises ToolArgumentError on shape mismatch"
    assert metadata.test_ref == "tests/unit/example.py::test_rejects_malformed"
    assert metadata.test_fingerprint == "abc123"
    assert metadata.qualname == handler.__wrapped__.__qualname__  # type: ignore[attr-defined]


def test_decorator_works_on_bound_method_with_self_excluded_from_check() -> None:
    """``source_param`` validation matches the function's real signature, not just self."""

    class Service:
        @trust_boundary(
            tier=3,
            source="external request body",
            source_param="payload",
            suppresses=("R5",),
            invariant="raises ValueError on shape mismatch",
        )
        def handle(self, payload: dict[str, Any]) -> int:
            return len(payload)

    service = Service()
    assert service.handle({"x": 1, "y": 2}) == 2

    metadata = Service.handle.__trust_boundary__  # type: ignore[attr-defined]
    assert metadata.source_param == "payload"
    # ``self`` is in the signature but is not the source_param; the
    # decorator should not have rejected the configuration.
    assert "self" in inspect.signature(Service.handle).parameters


def test_tier_one_raises_typeerror_at_decoration() -> None:
    """Tier-1 must crash, not suppress; decorator refuses tier=1."""
    with pytest.raises(TypeError) as exc:

        @trust_boundary(  # type: ignore[arg-type]
            tier=1,  # type: ignore[arg-type]
            source="audit row",
            source_param="row",
            suppresses=(),
            invariant="ignored",
        )
        def _f(row: dict[str, Any]) -> None:
            return None

    assert "tier=3" in str(exc.value)
    assert "Tier-1 and Tier-2 must crash, not suppress" in str(exc.value)


def test_tier_two_raises_typeerror_at_decoration() -> None:
    """Tier-2 must crash, not suppress; decorator refuses tier=2."""
    with pytest.raises(TypeError) as exc:

        @trust_boundary(  # type: ignore[arg-type]
            tier=2,  # type: ignore[arg-type]
            source="pipeline row",
            source_param="row",
            suppresses=(),
            invariant="ignored",
        )
        def _f(row: dict[str, Any]) -> None:
            return None

    assert "tier=3" in str(exc.value)


def test_source_param_not_in_signature_raises_typeerror() -> None:
    """``source_param`` typo fails at decoration time, not at call time."""
    with pytest.raises(TypeError) as exc:

        @trust_boundary(
            tier=3,
            source="external arguments",
            source_param="arguments",
            suppresses=("R1",),
            invariant="raises on bad shape",
        )
        def _f(payload: dict[str, Any]) -> None:  # parameter is ``payload``, not ``arguments``
            return None

    message = str(exc.value)
    assert "source_param='arguments'" in message
    assert "payload" in message


def test_stacked_trust_boundary_decorators_raise_typeerror() -> None:
    """A second ``@trust_boundary`` must not overwrite the first boundary claim."""
    with pytest.raises(TypeError, match="cannot be stacked"):

        @trust_boundary(
            tier=3,
            source="outer feed",
            source_param="raw",
            suppresses=("R1",),
            invariant="outer invariant",
        )
        @trust_boundary(
            tier=3,
            source="inner feed",
            source_param="raw",
            suppresses=("R5",),
            invariant="inner invariant",
        )
        def _handler(raw: dict[str, Any]) -> int:
            return len(raw)


def test_empty_suppresses_tuple_is_accepted() -> None:
    """An empty suppresses tuple is structurally valid; honesty gate is elsewhere."""

    @trust_boundary(
        tier=3,
        source="external feed",
        source_param="raw",
        suppresses=(),
        invariant="raises ValueError on bad shape",
    )
    def handler(raw: dict[str, Any]) -> int:
        return len(raw)

    assert handler({"a": 1}) == 1
    metadata = handler.__trust_boundary__  # type: ignore[attr-defined]
    assert metadata.suppresses == ()


def test_metadata_is_frozen() -> None:
    """``TrustBoundaryMetadata`` rejects field reassignment after construction."""

    @trust_boundary(
        tier=3,
        source="external feed",
        source_param="raw",
        suppresses=("R1",),
        invariant="raises ValueError on bad shape",
    )
    def handler(raw: dict[str, Any]) -> None:
        return None

    metadata = handler.__trust_boundary__  # type: ignore[attr-defined]
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.tier = 1  # type: ignore[misc]


def test_functools_wraps_preserves_name_and_doc() -> None:
    """``functools.wraps`` semantics — name, qualname, doc, module, __wrapped__."""

    @trust_boundary(
        tier=3,
        source="external feed",
        source_param="raw",
        suppresses=("R1",),
        invariant="raises ValueError on bad shape",
    )
    def documented_handler(raw: dict[str, Any]) -> int:
        """Original docstring stays put."""
        return len(raw)

    assert documented_handler.__name__ == "documented_handler"
    assert documented_handler.__doc__ == "Original docstring stays put."
    assert documented_handler.__module__ == __name__
    # ``__wrapped__`` is the @functools.wraps escape hatch pointing at the
    # original function.
    assert documented_handler.__wrapped__ is not documented_handler  # type: ignore[attr-defined]
    assert documented_handler.__wrapped__.__name__ == "documented_handler"  # type: ignore[attr-defined]


def test_signature_preserved_after_decoration() -> None:
    """``inspect.signature`` returns the same parameters and return type."""

    def original(arguments: dict[str, Any], *, mode: str = "strict") -> bool:
        return bool(arguments) and mode == "strict"

    decorated = trust_boundary(
        tier=3,
        source="external arguments",
        source_param="arguments",
        suppresses=("R1",),
        invariant="raises on shape mismatch",
    )(original)

    assert inspect.signature(decorated) == inspect.signature(original)


def test_metadata_func_field_is_original_function() -> None:
    """``metadata.func`` is the wrapped function, for AST introspection by the analyzer."""

    def original(payload: dict[str, Any]) -> int:
        return len(payload)

    decorated = trust_boundary(
        tier=3,
        source="external feed",
        source_param="payload",
        suppresses=(),
        invariant="raises on bad shape",
    )(original)

    metadata = decorated.__trust_boundary__  # type: ignore[attr-defined]
    assert metadata.func is original


def test_decorator_return_type_preserved_at_runtime() -> None:
    """Smoke test that the wrapper returns the original return value, typed correctly.

    The static type-preservation property (``Callable[P, R]`` in, ``Callable[P, R]``
    out) is exercised by the mypy verification step on the source module; here
    we only verify the runtime value pass-through.
    """

    @trust_boundary(
        tier=3,
        source="external feed",
        source_param="raw",
        suppresses=("R1",),
        invariant="raises on bad shape",
    )
    def returns_complex(raw: dict[str, Any]) -> tuple[int, str]:
        return (len(raw), "ok")

    result = returns_complex({"a": 1, "b": 2})
    assert result == (2, "ok")


def test_non_raising_defaults_false_and_records_true() -> None:
    """``non_raising`` defaults to False and is recorded on the metadata when set."""

    @trust_boundary(
        tier=3,
        source="LLM tool-call arguments",
        source_param="arguments",
        suppresses=("R5",),
        invariant="returns None on malformed input; never raises on arguments",
    )
    def default_boundary(arguments: dict[str, Any]) -> Any:
        return arguments.get("x")

    @trust_boundary(
        tier=3,
        source="LLM tool-call arguments",
        source_param="arguments",
        suppresses=("R5",),
        invariant="returns None on malformed input; never raises on arguments",
        non_raising=True,
    )
    def nonraising_boundary(arguments: dict[str, Any]) -> Any:
        return arguments.get("x")

    assert default_boundary.__trust_boundary__.non_raising is False  # type: ignore[attr-defined]
    assert nonraising_boundary.__trust_boundary__.non_raising is True  # type: ignore[attr-defined]


def test_non_raising_rejects_test_ref() -> None:
    """``non_raising=True`` is mutually exclusive with test_ref/test_fingerprint."""
    with pytest.raises(TypeError, match="mutually exclusive with test_ref"):

        @trust_boundary(
            tier=3,
            source="x",
            source_param="arguments",
            suppresses=("R5",),
            invariant="returns None on malformed input",
            non_raising=True,
            test_ref="tests/unit/example.py::test_rejects_malformed",
        )
        def contradictory(arguments: dict[str, Any]) -> Any:
            return arguments.get("x")
