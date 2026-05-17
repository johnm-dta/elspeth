"""Completeness guard: every plugin previously classified BOUNDARY by
trust.py must declare data_trust_tier == 3 now that trust.py is gone.

This test is written BEFORE deleting trust.py. It fails red until all
14 remaining EXTERNAL_BOUNDARY plugins have data_trust_tier = 3.
Once green, trust.py and test_trust.py can be deleted in the same commit.

Do NOT use getattr() to read data_trust_tier — the field is a ClassVar
declared on BaseSource / BaseTransform / BaseSink; direct attribute
access is correct and will AttributeError loudly on missing fields,
which is the desired behaviour.
"""

from __future__ import annotations

import pytest

# Sinks — named EXTERNAL_BOUNDARY_SINKS allowlist from trust.py:73-80.
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from elspeth.plugins.sinks.chroma_sink import ChromaSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sinks.dataverse import DataverseSink

# Sources — all sources are BOUNDARY unconditionally per trust.py:117-118.
from elspeth.plugins.sources.azure_blob_source import AzureBlobSource
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.sources.dataverse import DataverseSource
from elspeth.plugins.sources.json_source import JSONSource
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.sources.text_source import TextSource

# Transforms — named EXTERNAL_BOUNDARY_TRANSFORMS allowlist from trust.py:61-69.
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

_BOUNDARY_PLUGIN_CLASSES = [
    # Sources
    CSVSource,
    JSONSource,
    TextSource,
    AzureBlobSource,
    DataverseSource,
    NullSource,
    # Transforms
    LLMTransform,
    WebScrapeTransform,
    RAGRetrievalTransform,
    AzureContentSafety,
    AzurePromptShield,
    # Sinks
    AzureBlobSink,
    ChromaSink,
    DatabaseSink,
    DataverseSink,
]


@pytest.mark.parametrize("plugin_cls", _BOUNDARY_PLUGIN_CLASSES, ids=lambda c: c.name)
def test_boundary_plugin_has_data_trust_tier_three(plugin_cls) -> None:
    """Every previously-BOUNDARY plugin must declare data_trust_tier == 3.

    This test gates the trust.py deletion in Task 6. It must be fully
    green before the deletion commit proceeds.
    """
    assert plugin_cls.data_trust_tier == 3, (
        f"{plugin_cls.__name__} (name={plugin_cls.name!r}) has "
        f"data_trust_tier={plugin_cls.data_trust_tier!r}; expected 3. "
        f"Author 'data_trust_tier: ClassVar[int | None] = 3' in the class body."
    )
