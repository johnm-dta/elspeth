"""Built-in source plugins for ELSPETH.

Sources load data into the pipeline. Exactly one source per run.
"""

from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.sources.json_source import JSONSource
from elspeth.plugins.sources.null_source import NullSource

__all__ = ["CSVSource", "JSONSource", "NullSource"]
