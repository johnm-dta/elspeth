"""Hash stability + construction tests for call data frozen dataclasses.

These tests guard the critical invariant: ``to_dict()`` must produce
byte-identical JSON (and therefore identical ``stable_hash()``) to the
old inline dict construction that it replaces.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.call_data import (
    CallPayload,
    HTTPCallError,
    HTTPCallRequest,
    HTTPCallResponse,
    LLMCallError,
    LLMCallRequest,
    LLMCallResponse,
    RawCallPayload,
)
from elspeth.contracts.token_usage import TokenUsage
from elspeth.core.canonical import stable_hash

# ---------------------------------------------------------------------------
# RawCallPayload
# ---------------------------------------------------------------------------


class TestRawCallPayload:
    """RawCallPayload wraps pre-serialized dicts and satisfies CallPayload."""

    def test_frozen(self) -> None:
        obj = RawCallPayload({"k": "v"})
        with pytest.raises(AttributeError):
            obj.data = {"other": "value"}  # type: ignore[misc]

    def test_to_dict_returns_copy(self) -> None:
        obj = RawCallPayload({"k": "v"})
        returned = obj.to_dict()
        returned["injected"] = "mutation"
        assert "injected" not in obj.to_dict()

    def test_to_dict_matches_input(self) -> None:
        original = {"type": "TestError", "message": "something failed"}
        obj = RawCallPayload(original)
        assert obj.to_dict() == original

    def test_satisfies_call_payload_protocol(self) -> None:
        assert isinstance(RawCallPayload({"k": "v"}), CallPayload)

    def test_hash_stability(self) -> None:
        original_dict = {"type": "TestError", "message": "something failed"}
        payload = RawCallPayload(original_dict)
        assert stable_hash(payload.to_dict()) == stable_hash(original_dict)


# ---------------------------------------------------------------------------
# LLMCallRequest
# ---------------------------------------------------------------------------


class TestLLMCallRequest:
    """LLMCallRequest.to_dict() produces identical hashes to old dict."""

    def test_basic_request_hash_stability(self) -> None:
        old_dict: dict[str, object] = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.0,
            "provider": "openai",
        }
        new_dict = LLMCallRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.0,
            provider="openai",
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_request_with_max_tokens_hash_stability(self) -> None:
        old_dict: dict[str, object] = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "provider": "azure",
        }
        old_dict["max_tokens"] = 100

        new_dict = LLMCallRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            provider="azure",
            max_tokens=100,
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_request_with_extra_kwargs_hash_stability(self) -> None:
        kwargs = {"top_p": 0.9, "frequency_penalty": 0.5}
        old_dict: dict[str, object] = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.0,
            "provider": "openai",
            **kwargs,
        }
        new_dict = LLMCallRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.0,
            provider="openai",
            extra_kwargs=kwargs,
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_max_tokens_none_excluded(self) -> None:
        d = LLMCallRequest(
            model="gpt-4",
            messages=[],
            temperature=0.0,
            provider="openai",
            max_tokens=None,
        ).to_dict()
        assert "max_tokens" not in d

    def test_frozen(self) -> None:
        obj = LLMCallRequest(
            model="gpt-4",
            messages=[],
            temperature=0.0,
            provider="openai",
        )
        with pytest.raises(AttributeError):
            obj.model = "gpt-3.5"  # type: ignore[misc]

    def test_pre_tupled_messages_inner_dicts_frozen(self) -> None:
        """Messages passed as a tuple must still have inner dicts deep-frozen."""
        from types import MappingProxyType

        inner = {"role": "user", "content": "Hello"}
        obj = LLMCallRequest(
            model="gpt-4",
            messages=(inner,),
            temperature=0.0,
            provider="openai",
        )
        # Inner message should be MappingProxyType, not a mutable dict
        assert isinstance(obj.messages[0], MappingProxyType)
        # Original dict should not have been mutated into proxy
        assert isinstance(inner, dict)

    def test_extra_kwargs_collision_raises(self) -> None:
        with pytest.raises(ValueError, match="reserved key"):
            LLMCallRequest(
                model="gpt-4",
                messages=[],
                temperature=0.0,
                provider="openai",
                extra_kwargs={"model": "overridden"},
            )

    def test_extra_kwargs_collision_multiple_keys(self) -> None:
        with pytest.raises(ValueError, match="reserved key"):
            LLMCallRequest(
                model="gpt-4",
                messages=[],
                temperature=0.0,
                provider="openai",
                extra_kwargs={"temperature": 99, "provider": "evil"},
            )

    def test_extra_kwargs_non_reserved_accepted(self) -> None:
        obj = LLMCallRequest(
            model="gpt-4",
            messages=[],
            temperature=0.0,
            provider="openai",
            extra_kwargs={"top_p": 0.9, "frequency_penalty": 0.5},
        )
        d = obj.to_dict()
        assert d["top_p"] == 0.9
        assert d["frequency_penalty"] == 0.5


# ---------------------------------------------------------------------------
# LLMCallResponse
# ---------------------------------------------------------------------------


class TestLLMCallResponse:
    """LLMCallResponse.to_dict() produces identical hashes to old dict."""

    def test_hash_stability(self) -> None:
        usage = TokenUsage.known(prompt_tokens=10, completion_tokens=20)
        raw = {"id": "chatcmpl-123", "choices": []}
        old_dict = {
            "content": "Hello world",
            "model": "gpt-4",
            "usage": usage.to_dict(),
            "raw_response": raw,
        }
        new_dict = LLMCallResponse(
            content="Hello world",
            model="gpt-4",
            usage=usage,
            raw_response=raw,
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_unknown_usage(self) -> None:
        usage = TokenUsage.unknown()
        d = LLMCallResponse(
            content="test",
            model="gpt-4",
            usage=usage,
            raw_response={},
        ).to_dict()
        assert d["usage"] == {}

    def test_frozen(self) -> None:
        obj = LLMCallResponse(
            content="test",
            model="gpt-4",
            usage=TokenUsage.unknown(),
            raw_response={},
        )
        with pytest.raises(AttributeError):
            obj.content = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMCallError
# ---------------------------------------------------------------------------


class TestLLMCallError:
    """LLMCallError.to_dict() produces identical hashes to old dict."""

    def test_retryable_error_hash_stability(self) -> None:
        old_dict = {
            "type": "RateLimitError",
            "message": "429 Too Many Requests",
            "retryable": True,
        }
        new_dict = LLMCallError(
            type="RateLimitError",
            message="429 Too Many Requests",
            retryable=True,
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_non_retryable_error_hash_stability(self) -> None:
        old_dict = {
            "type": "ContentPolicyError",
            "message": "Content policy violation",
            "retryable": False,
        }
        new_dict = LLMCallError(
            type="ContentPolicyError",
            message="Content policy violation",
            retryable=False,
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_frozen(self) -> None:
        obj = LLMCallError(type="err", message="msg", retryable=False)
        with pytest.raises(AttributeError):
            obj.type = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HTTPCallRequest
# ---------------------------------------------------------------------------


class TestHTTPCallRequest:
    """HTTPCallRequest.to_dict() produces identical hashes to old dict."""

    def test_standard_post_hash_stability(self) -> None:
        old_dict: dict[str, object] = {
            "method": "POST",
            "url": "https://api.example.com/v1/data",
            "headers": {"Content-Type": "application/json"},
        }
        old_dict["json"] = {"key": "value"}

        new_dict = HTTPCallRequest(
            method="POST",
            url="https://api.example.com/v1/data",
            headers={"Content-Type": "application/json"},
            json={"key": "value"},
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_standard_get_hash_stability(self) -> None:
        old_dict: dict[str, object] = {
            "method": "GET",
            "url": "https://api.example.com/v1/data",
            "headers": {"Accept": "application/json"},
        }
        old_dict["params"] = {"page": 1}

        new_dict = HTTPCallRequest(
            method="GET",
            url="https://api.example.com/v1/data",
            headers={"Accept": "application/json"},
            params={"page": 1},
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_ssrf_request_hash_stability(self) -> None:
        expected = {
            "method": "GET",
            "url": "https://example.com/page",
            "resolved_ip": "93.184.216.34",
            "headers": {"Host": "example.com"},
            "params": None,  # GET always emits params for hash stability
        }
        actual = HTTPCallRequest(
            method="GET",
            url="https://example.com/page",
            headers={"Host": "example.com"},
            resolved_ip="93.184.216.34",
        ).to_dict()
        assert actual == expected
        assert stable_hash(actual) == stable_hash(expected)

    def test_redirect_hop_hash_stability(self) -> None:
        expected = {
            "method": "GET",
            "url": "https://new.example.com/page",
            "resolved_ip": "1.2.3.4",
            "hop_number": 1,
            "redirect_from": "https://old.example.com/page",
            "headers": {"Host": "new.example.com"},
            "params": None,  # GET always emits params for hash stability
        }
        actual = HTTPCallRequest(
            method="GET",
            url="https://new.example.com/page",
            headers={"Host": "new.example.com"},
            resolved_ip="1.2.3.4",
            hop_number=1,
            redirect_from="https://old.example.com/page",
        ).to_dict()
        assert actual == expected
        assert stable_hash(actual) == stable_hash(expected)

    def test_standard_post_always_includes_json(self) -> None:
        d = HTTPCallRequest(
            method="POST",
            url="https://example.com",
            headers={},
            json=None,
        ).to_dict()
        assert "json" in d
        assert d["json"] is None

    def test_standard_get_always_includes_params(self) -> None:
        d = HTTPCallRequest(
            method="GET",
            url="https://example.com",
            headers={},
            params=None,
        ).to_dict()
        assert "params" in d
        assert d["params"] is None

    def test_put_with_json_includes_payload(self) -> None:
        """PUT with json body must include it in to_dict() — not silently drop.

        Regression test for elspeth-c083584007: only POST got json serialized,
        PUT/PATCH/DELETE payloads were silently dropped from audit records.
        """
        d = HTTPCallRequest(
            method="PUT",
            url="https://api.example.com/v1/resource/42",
            headers={"Content-Type": "application/json"},
            json={"name": "updated"},
        ).to_dict()
        assert "json" in d
        assert d["json"] == {"name": "updated"}

    def test_patch_with_json_includes_payload(self) -> None:
        """PATCH with json body must include it in to_dict()."""
        d = HTTPCallRequest(
            method="PATCH",
            url="https://api.example.com/v1/resource/42",
            headers={"Content-Type": "application/json"},
            json={"status": "active"},
        ).to_dict()
        assert "json" in d
        assert d["json"] == {"status": "active"}

    def test_delete_with_params_includes_payload(self) -> None:
        """DELETE with query params must include them in to_dict()."""
        d = HTTPCallRequest(
            method="DELETE",
            url="https://api.example.com/v1/resource/42",
            headers={},
            params={"force": "true"},
        ).to_dict()
        assert "params" in d
        assert d["params"] == {"force": "true"}

    def test_ssrf_get_includes_params_for_hash_stability(self) -> None:
        """GET with resolved_ip still emits params (None) for hash stability."""
        d = HTTPCallRequest(
            method="GET",
            url="https://example.com",
            headers={},
            resolved_ip="1.2.3.4",
        ).to_dict()
        assert "json" not in d  # GET doesn't emit json unless explicitly set
        assert d["params"] is None  # GET always emits params for hash stability

    def test_ssrf_post_includes_json(self) -> None:
        """POST with resolved_ip must not silently drop json body."""
        d = HTTPCallRequest(
            method="POST",
            url="https://example.com/api",
            headers={"Content-Type": "application/json"},
            json={"query": "test data"},
            resolved_ip="1.2.3.4",
        ).to_dict()
        assert d["json"] == {"query": "test data"}
        assert d["resolved_ip"] == "1.2.3.4"
        assert "params" not in d  # POST doesn't emit params unless explicitly set

    def test_audit_metadata_included_when_present(self) -> None:
        """Audit-only metadata must survive serialization without changing request json."""
        d = HTTPCallRequest(
            method="POST",
            url="https://example.com/api",
            headers={"Content-Type": "application/json"},
            json={"query": "test data"},
            audit_metadata={"variables_hash": "abc123"},
        ).to_dict()
        assert d["json"] == {"query": "test data"}
        assert d["audit_metadata"] == {"variables_hash": "abc123"}

    def test_frozen(self) -> None:
        obj = HTTPCallRequest(method="GET", url="https://x.com", headers={})
        with pytest.raises(AttributeError):
            obj.method = "POST"  # type: ignore[misc]

    def test_hop_number_without_resolved_ip_raises(self) -> None:
        with pytest.raises(ValueError, match="hop_number without resolved_ip requires redirect_from"):
            HTTPCallRequest(
                method="GET",
                url="https://example.com",
                headers={},
                hop_number=1,
            )

    def test_redirect_from_without_hop_number_raises(self) -> None:
        with pytest.raises(ValueError, match="redirect_from requires hop_number"):
            HTTPCallRequest(
                method="GET",
                url="https://example.com",
                headers={},
                resolved_ip="1.2.3.4",
                redirect_from="https://old.example.com",
            )

    def test_valid_redirect_hop_accepted(self) -> None:
        obj = HTTPCallRequest(
            method="GET",
            url="https://new.example.com",
            headers={},
            resolved_ip="1.2.3.4",
            hop_number=1,
            redirect_from="https://old.example.com",
        )
        assert obj.hop_number == 1

    def test_blocked_redirect_attempt_allows_hop_tracking_without_resolved_ip(self) -> None:
        """Blocked redirects still need hop lineage even when SSRF resolution never completes."""
        obj = HTTPCallRequest(
            method="GET",
            url="http://169.254.169.254/latest/meta-data/",
            headers={"Host": "169.254.169.254"},
            hop_number=1,
            redirect_from="https://example.com/start",
        )
        assert obj.hop_number == 1
        assert obj.resolved_ip is None


# ---------------------------------------------------------------------------
# HTTPCallResponse
# ---------------------------------------------------------------------------


class TestHTTPCallResponse:
    """HTTPCallResponse.to_dict() produces identical hashes to old dict."""

    def test_standard_response_hash_stability(self) -> None:
        old_dict = {
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "body_size": 42,
            "body": {"result": "ok"},
        }
        new_dict = HTTPCallResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body_size=42,
            body={"result": "ok"},
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_ssrf_response_with_redirects_hash_stability(self) -> None:
        old_dict: dict[str, object] = {
            "status_code": 200,
            "headers": {"content-type": "text/html"},
            "body_size": 1024,
            "body": "<html>...</html>",
        }
        old_dict["redirect_count"] = 2

        new_dict = HTTPCallResponse(
            status_code=200,
            headers={"content-type": "text/html"},
            body_size=1024,
            body="<html>...</html>",
            redirect_count=2,
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_redirect_hop_response_hash_stability(self) -> None:
        old_dict = {
            "status_code": 301,
            "headers": {"location": "https://new.example.com"},
        }
        new_dict = HTTPCallResponse(
            status_code=301,
            headers={"location": "https://new.example.com"},
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_redirect_count_zero_excluded(self) -> None:
        d = HTTPCallResponse(
            status_code=200,
            headers={},
            body_size=0,
            body="",
        ).to_dict()
        assert "redirect_count" not in d

    def test_body_fields_excluded_when_body_size_none(self) -> None:
        d = HTTPCallResponse(
            status_code=301,
            headers={},
        ).to_dict()
        assert "body_size" not in d
        assert "body" not in d

    def test_frozen(self) -> None:
        obj = HTTPCallResponse(status_code=200, headers={})
        with pytest.raises(AttributeError):
            obj.status_code = 404  # type: ignore[misc]

    def test_rejects_bool_status_code(self) -> None:
        with pytest.raises(TypeError, match="status_code must be int"):
            HTTPCallResponse(status_code=True, headers={})

    def test_rejects_bool_body_size(self) -> None:
        with pytest.raises(TypeError, match="body_size must be int"):
            HTTPCallResponse(status_code=200, headers={}, body_size=False)

    def test_rejects_bool_redirect_count(self) -> None:
        with pytest.raises(TypeError, match="redirect_count must be int"):
            HTTPCallResponse(status_code=200, headers={}, redirect_count=True)

    def test_rejects_status_code_below_100(self) -> None:
        with pytest.raises(ValueError, match="status_code must be >= 100"):
            HTTPCallResponse(status_code=99, headers={})

    def test_rejects_negative_body_size(self) -> None:
        with pytest.raises(ValueError, match="body_size must be >= 0"):
            HTTPCallResponse(status_code=200, headers={}, body_size=-1)

    def test_body_size_none_accepted(self) -> None:
        """None body_size is valid when body is also None (redirect hop)."""
        obj = HTTPCallResponse(status_code=200, headers={}, body_size=None)
        assert obj.body_size is None

    def test_body_without_body_size_rejected(self) -> None:
        """body present but body_size None would silently drop body in to_dict()."""
        with pytest.raises(ValueError, match="body requires body_size"):
            HTTPCallResponse(
                status_code=200,
                headers={},
                body={"data": "value"},
                body_size=None,
            )

    def test_body_str_without_body_size_rejected(self) -> None:
        """String body without body_size is also rejected."""
        with pytest.raises(ValueError, match="body requires body_size"):
            HTTPCallResponse(
                status_code=200,
                headers={},
                body="<html>hello</html>",
                body_size=None,
            )

    def test_body_none_with_body_size_accepted(self) -> None:
        """body_size present but body None is valid — response was received but body not captured."""
        obj = HTTPCallResponse(status_code=200, headers={}, body_size=42, body=None)
        assert obj.body_size == 42
        assert obj.body is None


# ---------------------------------------------------------------------------
# HTTPCallError
# ---------------------------------------------------------------------------


class TestHTTPCallError:
    """HTTPCallError.to_dict() produces identical hashes to old dict."""

    def test_http_error_hash_stability(self) -> None:
        old_dict = {
            "type": "HTTPError",
            "message": "HTTP 404",
            "status_code": 404,
        }
        new_dict = HTTPCallError(
            type="HTTPError",
            message="HTTP 404",
            status_code=404,
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_network_error_hash_stability(self) -> None:
        old_dict = {
            "type": "ConnectTimeout",
            "message": "Connection timed out",
        }
        new_dict = HTTPCallError(
            type="ConnectTimeout",
            message="Connection timed out",
        ).to_dict()
        assert new_dict == old_dict
        assert stable_hash(new_dict) == stable_hash(old_dict)

    def test_status_code_none_excluded(self) -> None:
        d = HTTPCallError(
            type="TimeoutError",
            message="timed out",
        ).to_dict()
        assert "status_code" not in d

    def test_frozen(self) -> None:
        obj = HTTPCallError(type="err", message="msg")
        with pytest.raises(AttributeError):
            obj.type = "other"  # type: ignore[misc]


class TestHTTPCallResponseListBodyFreeze:
    """JSON array bodies must be deeply frozen."""

    def test_list_body_frozen_to_tuple(self) -> None:
        body = [{"id": 1}, {"id": 2}]
        resp = HTTPCallResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body_size=20,
            body=body,  # type: ignore[arg-type]
        )
        body.append({"id": 3})
        assert isinstance(resp.body, tuple)
        assert len(resp.body) == 2

    def test_list_body_nested_dicts_frozen(self) -> None:
        from types import MappingProxyType

        body = [{"nested": {"key": "value"}}]
        resp = HTTPCallResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body_size=30,
            body=body,  # type: ignore[arg-type]
        )
        assert isinstance(resp.body, tuple)
        assert isinstance(resp.body[0], MappingProxyType)

    def test_list_body_round_trips_via_to_dict(self) -> None:
        resp = HTTPCallResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body_size=20,
            body=[{"id": 1}, {"id": 2}],  # type: ignore[arg-type]
        )
        d = resp.to_dict()
        assert isinstance(d["body"], list)
        assert d["body"] == [{"id": 1}, {"id": 2}]


class TestHTTPCallResponseTupleBodyFreeze:
    """Tuple bodies containing mutable dicts must be deeply frozen."""

    def test_tuple_body_dicts_frozen(self) -> None:
        """tuple[dict, ...] body has inner dicts frozen to MappingProxyType."""
        from types import MappingProxyType

        mutable_dict = {"mutable": "value"}
        resp = HTTPCallResponse(
            status_code=200,
            headers={},
            body_size=10,
            body=(mutable_dict,),
        )
        # Inner dict should be frozen — mutation of original must not affect body
        mutable_dict["injected"] = "bad"
        assert isinstance(resp.body, tuple)
        assert isinstance(resp.body[0], MappingProxyType)
        assert "injected" not in resp.body[0]

    def test_tuple_body_round_trips_via_to_dict(self) -> None:
        resp = HTTPCallResponse(
            status_code=200,
            headers={},
            body_size=10,
            body=({"id": 1}, {"id": 2}),
        )
        d = resp.to_dict()
        assert isinstance(d["body"], list)
        assert d["body"] == [{"id": 1}, {"id": 2}]
