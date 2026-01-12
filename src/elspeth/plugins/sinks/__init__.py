"""Built-in sink plugins for ELSPETH.

Sinks output data to destinations. Multiple sinks per run.
"""

from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sinks.json_sink import JSONSink

__all__ = ["CSVSink", "DatabaseSink", "JSONSink"]
