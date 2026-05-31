"""Tests for F2: canonicalize Pydantic ``__cause__`` errors in ARG_ERROR audits.

Disposition: spec §4.2.6 documents that promoted-handler ``ToolArgumentError``
sites raise with ``from pydantic.ValidationError``. The ``__cause__`` chain
carries field-name detail (``loc``/``msg``/``type``) that is auditably
valuable for recovery flows — but the ``input`` / ``url`` / ``ctx`` fields on
each error are leak vectors (``input`` carries the rejected value verbatim).

Option (a) chosen: persist canonicalized cause errors (loc/msg/type tuples
only, no values) into ``result_canonical`` via the ARG_ERROR payload factory.

These tests pin:

1. ``canonicalize_pydantic_cause`` helper produces leak-safe output.
2. The module-level ``_arg_error_payload`` factory threads
   ``validation_errors`` through when the ``__cause__`` is a Pydantic
   ``ValidationError``.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from elspeth.web.composer.audit import canonicalize_pydantic_cause
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.service import _arg_error_payload


class _IntFieldModel(BaseModel):
    """Two-field model: ``x`` (required ``int``) for ``missing`` / ``int_parsing``."""

    x: int


class _ListIntModel(BaseModel):
    """Single ``list[int]`` field for forcing an int-loc element (list index)."""

    items: list[int]


def _make_int_parsing_error() -> ValidationError:
    """Force a ``ValidationError`` with ``loc=("x",)`` and ``type="int_parsing"``."""
    try:
        _IntFieldModel.model_validate({"x": "not-an-int"})
    except ValidationError as exc:
        return exc
    raise AssertionError("model_validate should have raised")


def _make_missing_error() -> ValidationError:
    """Force a ``ValidationError`` with ``loc=("x",)`` and ``type="missing"``."""
    try:
        _IntFieldModel.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("model_validate should have raised")


def _make_list_index_error() -> ValidationError:
    """Force a ``ValidationError`` with a list-index int in ``loc``."""
    try:
        _ListIntModel.model_validate({"items": [1, "bad", 3]})
    except ValidationError as exc:
        return exc
    raise AssertionError("model_validate should have raised")


# ---------------------------------------------------------------------------
# Helper unit tests.
# ---------------------------------------------------------------------------


def test_canonicalize_pydantic_cause_returns_none_for_none() -> None:
    """``None`` in → ``None`` out (no chained cause to canonicalize)."""
    assert canonicalize_pydantic_cause(None) is None


def test_canonicalize_pydantic_cause_returns_none_for_non_pydantic() -> None:
    """Non-Pydantic exceptions yield ``None`` — the helper opts out cleanly."""
    assert canonicalize_pydantic_cause(ValueError("plain old error")) is None
    assert canonicalize_pydantic_cause(KeyError("missing")) is None
    assert canonicalize_pydantic_cause(RuntimeError("runtime")) is None


def test_canonicalize_pydantic_cause_strips_input_url_ctx() -> None:
    """``input`` / ``url`` / ``ctx`` MUST NOT appear in the canonicalized output.

    ``input`` is the primary leak vector — it carries the rejected
    (LLM-supplied, Tier-3) value verbatim. ``url`` is Pydantic's
    documentation URL (not load-bearing for audit). ``ctx`` may carry the
    rejected value in its context dict. The helper strips all three.
    """
    exc = _make_int_parsing_error()
    result = canonicalize_pydantic_cause(exc)
    assert result is not None
    assert len(result) == 1
    entry = result[0]
    assert set(entry.keys()) == {"loc", "msg", "type"}
    assert "input" not in entry
    assert "url" not in entry
    assert "ctx" not in entry


def test_canonicalize_pydantic_cause_preserves_loc_msg_type() -> None:
    """``loc`` / ``msg`` / ``type`` are preserved with their Pydantic semantics."""
    exc = _make_missing_error()
    result = canonicalize_pydantic_cause(exc)
    assert result is not None
    assert len(result) == 1
    entry = result[0]
    assert entry["loc"] == ["x"]
    assert entry["type"] == "missing"
    assert isinstance(entry["msg"], str)
    assert entry["msg"]  # non-empty


def test_canonicalize_pydantic_cause_int_parsing_type() -> None:
    """``int_parsing`` flows through as the canonicalized ``type`` string."""
    exc = _make_int_parsing_error()
    result = canonicalize_pydantic_cause(exc)
    assert result is not None
    assert result[0]["type"] == "int_parsing"
    assert result[0]["loc"] == ["x"]


def test_canonicalize_pydantic_cause_stringifies_non_str_loc() -> None:
    """List-index ``int`` elements in ``loc`` MUST be stringified.

    Pydantic produces ``loc=("items", 1)`` for the second element of a
    ``list[int]`` field. Canonical JSON (rfc8785) accepts mixed-type
    tuples in principle, but the helper coerces every loc element to
    ``str`` so the recorded shape is uniform and future-proof against
    audit consumers that expect ``list[str]``.
    """
    exc = _make_list_index_error()
    result = canonicalize_pydantic_cause(exc)
    assert result is not None
    # At least one entry should have a stringified index in loc.
    list_index_entries = [e for e in result if "items" in e["loc"]]
    assert list_index_entries, f"expected items-loc entry, got {result}"
    entry = list_index_entries[0]
    # Every loc element is a str.
    for piece in entry["loc"]:
        assert isinstance(piece, str), f"loc element {piece!r} is not str"
    # The list index was stringified — "1", not 1.
    assert "1" in entry["loc"]


# ---------------------------------------------------------------------------
# Factory integration test.
# ---------------------------------------------------------------------------


def test_arg_error_payload_factory_threads_validation_errors() -> None:
    """``_arg_error_payload`` includes ``validation_errors`` when ``__cause__`` is Pydantic."""
    cause = _make_int_parsing_error()
    arg_err = ToolArgumentError(argument="x", expected="an integer", actual_type="str")
    arg_err.__cause__ = cause
    payload = _arg_error_payload(arg_err, "set_metadata")
    assert "error" in payload
    assert "Tool 'set_metadata' failed" in payload["error"]
    assert "validation_errors" in payload
    assert isinstance(payload["validation_errors"], list)
    assert len(payload["validation_errors"]) == 1
    assert payload["validation_errors"][0]["type"] == "int_parsing"
    assert payload["validation_errors"][0]["loc"] == ["x"]


def test_arg_error_payload_factory_omits_validation_errors_for_non_pydantic_cause() -> None:
    """A non-Pydantic ``__cause__`` (or no cause) yields no ``validation_errors`` key.

    Recording ``validation_errors: []`` (or ``validation_errors: None``)
    has no audit value — the absence of the key is the signal.
    """
    arg_err = ToolArgumentError(argument="x", expected="an integer", actual_type="str")
    arg_err.__cause__ = ValueError("not pydantic")
    payload = _arg_error_payload(arg_err, "set_metadata")
    assert "validation_errors" not in payload

    arg_err_no_cause = ToolArgumentError(argument="y", expected="a string", actual_type="int")
    payload_no_cause = _arg_error_payload(arg_err_no_cause, "set_metadata")
    assert "validation_errors" not in payload_no_cause


def test_arg_error_payload_factory_strips_leak_vectors_end_to_end() -> None:
    """End-to-end leak check: rejected value is NOT in the factory output."""
    cause = _make_int_parsing_error()
    # Confirm the rejected value lives on the cause's errors() output.
    raw_errors = cause.errors()
    assert raw_errors[0]["input"] == "not-an-int"
    arg_err = ToolArgumentError(argument="x", expected="an integer", actual_type="str")
    arg_err.__cause__ = cause
    payload = _arg_error_payload(arg_err, "set_metadata")
    # Walk the payload exhaustively for the rejected value.
    import json

    serialized = json.dumps(payload, default=str)
    assert "not-an-int" not in serialized, f"rejected value leaked into payload: {serialized}"
