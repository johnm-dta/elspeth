class BaseSource:
    """Structural stand-in for the runtime plugin base (fixtures are AST-scanned, never imported)."""


class StaleHashSource(BaseSource):
    name = "stale-hash"
    plugin_version = "1.0.0"
    source_file_hash = "sha256:0000000000000000"
