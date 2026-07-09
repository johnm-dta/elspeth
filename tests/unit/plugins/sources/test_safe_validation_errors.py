"""Input-free rendering of Pydantic validation failures (elspeth-a300402c58).

``str(ValidationError)`` echoes the offending INPUT VALUE by default. Source
plugins sit on the Tier-3 boundary and their error text lands verbatim in
``node_states.error_json``, DIVERT routing reasons, and audit exports — so an
invalid row carrying a secret/PII value would surface it outside the
designated quarantine sink. ``safe_validation_error_text`` renders
loc/msg/type only.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from elspeth.plugins.sources._safe_validation_errors import safe_validation_error_text

SECRET = "SECRET-abc123"


class _Model(BaseModel):
    amount: int
    name: str


def _validation_error(payload: dict[str, object]) -> ValidationError:
    try:
        _Model.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("model_validate unexpectedly succeeded")


class TestSafeValidationErrorText:
    def test_premise_str_echoes_input_value(self) -> None:
        """Guard the premise: str(ValidationError) DOES echo the input value.

        If Pydantic ever stops echoing input by default this boundary can be
        revisited — this pin makes that change visible.
        """
        assert SECRET in str(_validation_error({"amount": SECRET, "name": "ok"}))

    def test_safe_text_excludes_input_value(self) -> None:
        text = safe_validation_error_text(_validation_error({"amount": SECRET, "name": "ok"}))
        assert SECRET not in text

    def test_safe_text_keeps_triage_fields(self) -> None:
        text = safe_validation_error_text(_validation_error({"amount": SECRET, "name": "ok"}))
        assert "amount" in text, "field loc must survive for triage"
        assert "int_parsing" in text, "error type code must survive for triage"
        assert "valid integer" in text, "human message must survive for triage"

    def test_multiple_errors_all_rendered(self) -> None:
        text = safe_validation_error_text(_validation_error({"amount": SECRET}))
        assert "amount" in text
        assert "name" in text
        assert text.startswith("2 validation errors:")

    def test_root_level_error_renders_placeholder_loc(self) -> None:
        try:
            _Model.model_validate(SECRET)
        except ValidationError as exc:
            text = safe_validation_error_text(exc)
        assert SECRET not in text
        assert "<root>" in text
