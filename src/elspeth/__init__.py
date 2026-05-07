"""
Elspeth: Auditable Sense/Decide/Act pipelines for high-reliability systems.

A framework for building data processing workflows where every decision
must be traceable to its source.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("elspeth")
except PackageNotFoundError:
    # Dev-only: editable install not yet run. Production installs always have
    # metadata. "UNKNOWN-dev-uninstalled" in a dev audit trail is acceptable —
    # it honestly represents the state rather than crashing.
    __version__ = "UNKNOWN-dev-uninstalled"
