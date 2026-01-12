"""Built-in source plugins for ELSPETH.

Sources load data into the pipeline. Exactly one source per run.
"""

from elspeth.plugins.sources.csv_source import CSVSource

__all__ = ["CSVSource"]
