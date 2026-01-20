"""Azure plugin pack for ELSPETH.

Provides sources and sinks for Azure Blob Storage integration.
Supports multiple authentication methods:
- Connection string
- Managed Identity (for Azure-hosted workloads)
- Service Principal (for automated/CI scenarios)
"""

from elspeth.plugins.azure.auth import AzureAuthConfig
from elspeth.plugins.azure.blob_sink import AzureBlobSink
from elspeth.plugins.azure.blob_source import AzureBlobSource

__all__ = ["AzureAuthConfig", "AzureBlobSink", "AzureBlobSource"]
