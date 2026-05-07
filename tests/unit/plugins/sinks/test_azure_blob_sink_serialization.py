"""Tests for AzureBlobSink JSON serialization boundary.

Verifies that non-finite floats are rejected at serialization time
rather than producing non-standard JSON blobs.
"""

import pytest


class TestAzureBlobSinkNonFiniteRejection:
    """Non-finite floats must be rejected at the JSON/JSONL serialization boundary."""

    @pytest.fixture
    def sink(self):
        """Create sink with minimal config, no Azure connection needed."""
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
        from tests.fixtures.base_classes import inject_write_failure

        return inject_write_failure(
            AzureBlobSink(
                {
                    "container": "test-container",
                    "blob_path": "test.json",
                    "format": "json",
                    "connection_string": "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net",
                    "schema": {"mode": "observed"},
                }
            )
        )

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_serialize_json_rejects_non_finite(self, sink, bad_value: float) -> None:
        with pytest.raises(ValueError, match="Out of range float values"):
            sink._serialize_json([{"id": 1, "value": bad_value}])

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_serialize_jsonl_rejects_non_finite(self, sink, bad_value: float) -> None:
        with pytest.raises(ValueError, match="Out of range float values"):
            sink._serialize_jsonl([{"id": 1, "value": bad_value}])

    def test_serialize_json_accepts_finite_floats(self, sink) -> None:
        result = sink._serialize_json([{"id": 1, "value": 3.14}])
        assert b"3.14" in result

    def test_serialize_jsonl_accepts_finite_floats(self, sink) -> None:
        result = sink._serialize_jsonl([{"id": 1, "value": 3.14}])
        assert b"3.14" in result


class TestAzureBlobSinkCloseResourceRelease:
    """close() must release Azure SDK resources, not just null the reference."""

    def test_close_calls_client_close(self) -> None:
        from unittest.mock import MagicMock

        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
        from tests.fixtures.base_classes import inject_write_failure

        sink = inject_write_failure(
            AzureBlobSink(
                {
                    "container": "test-container",
                    "blob_path": "test.json",
                    "format": "json",
                    "connection_string": "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net",
                    "schema": {"mode": "observed"},
                }
            )
        )

        mock_client = MagicMock()
        sink._container_client = mock_client

        sink.close()

        mock_client.close.assert_called_once()
        assert sink._container_client is None

    def test_close_without_client_is_safe(self) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
        from tests.fixtures.base_classes import inject_write_failure

        sink = inject_write_failure(
            AzureBlobSink(
                {
                    "container": "test-container",
                    "blob_path": "test.json",
                    "format": "json",
                    "connection_string": "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net",
                    "schema": {"mode": "observed"},
                }
            )
        )

        sink.close()  # Should not raise when _container_client is None
