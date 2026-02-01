"""Tests for AzureBlobSink resume capability (NOT supported)."""

import pytest


def test_azure_blob_sink_does_not_support_resume():
    """AzureBlobSink should declare supports_resume=False."""
    from elspeth.plugins.azure.blob_sink import AzureBlobSink

    assert AzureBlobSink.supports_resume is False


def test_azure_blob_sink_configure_for_resume_raises():
    """AzureBlobSink.configure_for_resume should raise NotImplementedError."""
    from elspeth.plugins.azure.blob_sink import AzureBlobSink

    # Create minimal sink - we just need to test the method, not actual Azure connection
    # Use a mock connection string format that passes validation
    sink = AzureBlobSink(
        {
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
            "container": "test-container",
            "blob_path": "test/output.csv",
            "schema": {"fields": "dynamic"},
        }
    )

    with pytest.raises(NotImplementedError) as exc_info:
        sink.configure_for_resume()

    assert "AzureBlobSink" in str(exc_info.value)
    assert "immutable" in str(exc_info.value).lower() or "append" in str(exc_info.value).lower()
