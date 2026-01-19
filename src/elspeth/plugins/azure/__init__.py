"""Azure plugin pack for ELSPETH.

Provides sources and sinks for Azure Blob Storage integration.
"""

from elspeth.plugins.azure.blob_sink import AzureBlobSink
from elspeth.plugins.azure.blob_source import AzureBlobSource

__all__ = ["AzureBlobSink", "AzureBlobSource"]
