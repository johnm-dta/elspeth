"""Tests for construction-time validations on plugin infrastructure types.

Covers: ValidationError, CapacityError (__init__), ThrottleConfig, RowTicket,
RowBufferEntry, RowContext, LLMResponse, OpenAIResponse, WebResponse.
"""

import pytest


class TestValidationErrorPostInit:
    def test_rejects_empty_field(self) -> None:
        from elspeth.plugins.infrastructure.validation import ValidationError

        with pytest.raises(ValueError, match="field must not be empty"):
            ValidationError(field="", message="bad", value=42)

    def test_rejects_empty_message(self) -> None:
        from elspeth.plugins.infrastructure.validation import ValidationError

        with pytest.raises(ValueError, match="message must not be empty"):
            ValidationError(field="name", message="", value=42)

    def test_accepts_valid(self) -> None:
        from elspeth.plugins.infrastructure.validation import ValidationError

        e = ValidationError(field="name", message="required", value=None)
        assert e.field == "name"


class TestCapacityErrorInit:
    def test_rejects_invalid_status_code(self) -> None:
        from elspeth.contracts.errors import CapacityError

        with pytest.raises(ValueError, match="valid HTTP status"):
            CapacityError(status_code=0, message="bad")

    def test_rejects_negative_status_code(self) -> None:
        from elspeth.contracts.errors import CapacityError

        with pytest.raises(ValueError, match="valid HTTP status"):
            CapacityError(status_code=-1, message="bad")

    def test_rejects_above_599(self) -> None:
        from elspeth.contracts.errors import CapacityError

        with pytest.raises(ValueError, match="valid HTTP status"):
            CapacityError(status_code=600, message="bad")

    def test_accepts_429(self) -> None:
        from elspeth.contracts.errors import CapacityError

        e = CapacityError(status_code=429, message="rate limit")
        assert e.status_code == 429

    def test_accepts_boundary_100(self) -> None:
        from elspeth.contracts.errors import CapacityError

        e = CapacityError(status_code=100, message="continue")
        assert e.status_code == 100

    def test_accepts_boundary_599(self) -> None:
        from elspeth.contracts.errors import CapacityError

        e = CapacityError(status_code=599, message="network error")
        assert e.status_code == 599

    def test_rejects_boundary_99(self) -> None:
        from elspeth.contracts.errors import CapacityError

        with pytest.raises(ValueError, match="valid HTTP status"):
            CapacityError(status_code=99, message="bad")


class TestThrottleConfigPostInit:
    def test_rejects_negative_min_delay(self) -> None:
        from elspeth.plugins.infrastructure.pooling.throttle import ThrottleConfig

        with pytest.raises(ValueError, match="min_dispatch_delay_ms must be non-negative"):
            ThrottleConfig(min_dispatch_delay_ms=-1)

    def test_rejects_max_less_than_min(self) -> None:
        from elspeth.plugins.infrastructure.pooling.throttle import ThrottleConfig

        with pytest.raises(ValueError, match=r"max_dispatch_delay_ms.*must be >= min_dispatch_delay_ms"):
            ThrottleConfig(min_dispatch_delay_ms=100, max_dispatch_delay_ms=50)

    def test_rejects_backoff_lte_one(self) -> None:
        from elspeth.plugins.infrastructure.pooling.throttle import ThrottleConfig

        with pytest.raises(ValueError, match=r"backoff_multiplier must be > 1\.0"):
            ThrottleConfig(backoff_multiplier=1.0)

    def test_rejects_negative_recovery_step(self) -> None:
        from elspeth.plugins.infrastructure.pooling.throttle import ThrottleConfig

        with pytest.raises(ValueError, match="recovery_step_ms must be non-negative"):
            ThrottleConfig(recovery_step_ms=-1)

    def test_rejects_nan_backoff_multiplier(self) -> None:
        from elspeth.plugins.infrastructure.pooling.throttle import ThrottleConfig

        with pytest.raises(ValueError, match=r"backoff_multiplier must be > 1\.0 and finite"):
            ThrottleConfig(backoff_multiplier=float("nan"))

    def test_rejects_inf_backoff_multiplier(self) -> None:
        from elspeth.plugins.infrastructure.pooling.throttle import ThrottleConfig

        with pytest.raises(ValueError, match=r"backoff_multiplier must be > 1\.0 and finite"):
            ThrottleConfig(backoff_multiplier=float("inf"))

    def test_accepts_defaults(self) -> None:
        from elspeth.plugins.infrastructure.pooling.throttle import ThrottleConfig

        config = ThrottleConfig()
        assert config.min_dispatch_delay_ms == 0
        assert config.max_dispatch_delay_ms == 5000


class TestRowTicketPostInit:
    def test_rejects_negative_sequence(self) -> None:
        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import RowTicket

        with pytest.raises(ValueError, match="sequence must be non-negative"):
            RowTicket(sequence=-1, row_id="r1", submitted_at=1.0)

    def test_rejects_empty_row_id(self) -> None:
        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import RowTicket

        with pytest.raises(ValueError, match="row_id must not be empty"):
            RowTicket(sequence=0, row_id="", submitted_at=1.0)

    def test_accepts_valid(self) -> None:
        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import RowTicket

        t = RowTicket(sequence=0, row_id="r1", submitted_at=1.0)
        assert t.sequence == 0


class TestRowBufferEntryPostInit:
    def test_rejects_negative_sequence(self) -> None:
        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import RowBufferEntry

        with pytest.raises(ValueError, match="sequence must be non-negative"):
            RowBufferEntry(sequence=-1, row_id="r1", result="ok", submitted_at=1.0, completed_at=2.0, buffer_wait_ms=0.5)

    def test_rejects_empty_row_id(self) -> None:
        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import RowBufferEntry

        with pytest.raises(ValueError, match="row_id must not be empty"):
            RowBufferEntry(sequence=0, row_id="", result="ok", submitted_at=1.0, completed_at=2.0, buffer_wait_ms=0.5)

    def test_rejects_negative_buffer_wait_ms(self) -> None:
        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import RowBufferEntry

        with pytest.raises(ValueError, match="buffer_wait_ms must be non-negative"):
            RowBufferEntry(sequence=0, row_id="r1", result="ok", submitted_at=1.0, completed_at=2.0, buffer_wait_ms=-0.1)

    def test_accepts_zero_buffer_wait_ms(self) -> None:
        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import RowBufferEntry

        entry = RowBufferEntry(sequence=0, row_id="r1", result="ok", submitted_at=1.0, completed_at=2.0, buffer_wait_ms=0.0)
        assert entry.buffer_wait_ms == 0.0

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import RowBufferEntry

        entry = RowBufferEntry(sequence=0, row_id="r1", result="ok", submitted_at=1.0, completed_at=2.0, buffer_wait_ms=0.5)
        with pytest.raises(FrozenInstanceError):
            entry.sequence = 99  # type: ignore[misc]


class TestRowContextPostInit:
    def test_rejects_empty_state_id(self) -> None:
        from elspeth.plugins.infrastructure.pooling.executor import RowContext

        with pytest.raises(ValueError, match="state_id must not be empty"):
            RowContext(row={"a": 1}, state_id="", row_index=0)

    def test_rejects_negative_row_index(self) -> None:
        from elspeth.plugins.infrastructure.pooling.executor import RowContext

        with pytest.raises(ValueError, match="row_index must be non-negative"):
            RowContext(row={"a": 1}, state_id="s1", row_index=-1)

    def test_accepts_valid(self) -> None:
        from elspeth.plugins.infrastructure.pooling.executor import RowContext

        ctx = RowContext(row={"a": 1}, state_id="s1", row_index=0)
        assert ctx.state_id == "s1"


class TestLLMResponsePostInit:
    def test_rejects_negative_latency(self) -> None:
        from elspeth.plugins.infrastructure.clients.llm import LLMResponse

        with pytest.raises(ValueError, match="latency_ms must be non-negative"):
            LLMResponse(content="hi", model="gpt-4", latency_ms=-1.0)

    def test_rejects_nan_latency(self) -> None:
        from elspeth.plugins.infrastructure.clients.llm import LLMResponse

        with pytest.raises(ValueError, match="latency_ms must be non-negative and finite"):
            LLMResponse(content="hi", model="gpt-4", latency_ms=float("nan"))

    def test_rejects_inf_latency(self) -> None:
        from elspeth.plugins.infrastructure.clients.llm import LLMResponse

        with pytest.raises(ValueError, match="latency_ms must be non-negative and finite"):
            LLMResponse(content="hi", model="gpt-4", latency_ms=float("inf"))

    def test_accepts_zero_latency(self) -> None:
        from elspeth.plugins.infrastructure.clients.llm import LLMResponse

        r = LLMResponse(content="hi", model="gpt-4", latency_ms=0.0)
        assert r.latency_ms == 0.0


class TestOpenAIResponsePostInit:
    def test_rejects_negative_prompt_tokens(self) -> None:
        from errorworks.llm.response_generator import OpenAIResponse

        with pytest.raises(ValueError, match="prompt_tokens must be non-negative"):
            OpenAIResponse(
                id="fake-1",
                object="chat.completion",
                created=1000,
                model="gpt-4",
                content="hi",
                prompt_tokens=-1,
                completion_tokens=10,
                finish_reason="stop",
            )

    def test_rejects_negative_completion_tokens(self) -> None:
        from errorworks.llm.response_generator import OpenAIResponse

        with pytest.raises(ValueError, match="completion_tokens must be non-negative"):
            OpenAIResponse(
                id="fake-1",
                object="chat.completion",
                created=1000,
                model="gpt-4",
                content="hi",
                prompt_tokens=10,
                completion_tokens=-5,
                finish_reason="stop",
            )

    def test_rejects_negative_created(self) -> None:
        from errorworks.llm.response_generator import OpenAIResponse

        with pytest.raises(ValueError, match="created must be non-negative"):
            OpenAIResponse(
                id="fake-1",
                object="chat.completion",
                created=-1,
                model="gpt-4",
                content="hi",
                prompt_tokens=10,
                completion_tokens=10,
                finish_reason="stop",
            )

    def test_accepts_valid(self) -> None:
        from errorworks.llm.response_generator import OpenAIResponse

        r = OpenAIResponse(
            id="fake-1",
            object="chat.completion",
            created=1000,
            model="gpt-4",
            content="hello world",
            prompt_tokens=5,
            completion_tokens=10,
            finish_reason="stop",
        )
        assert r.content == "hello world"
        assert r.prompt_tokens == 5


class TestWebResponsePostInit:
    def test_rejects_invalid_status_code(self) -> None:
        from errorworks.web.content_generator import WebResponse

        with pytest.raises(ValueError, match="valid HTTP status"):
            WebResponse(content="<html>", content_type="text/html", status_code=0)

    def test_rejects_empty_content_type(self) -> None:
        from errorworks.web.content_generator import WebResponse

        with pytest.raises(ValueError, match="content_type must not be empty"):
            WebResponse(content="<html>", content_type="")

    def test_accepts_valid(self) -> None:
        from errorworks.web.content_generator import WebResponse

        r = WebResponse(content="<html>", content_type="text/html")
        assert r.status_code == 200
