class BaseSource:
    """Structural stand-in for the runtime plugin base (fixtures are AST-scanned, never imported)."""


class MissingHashSource(BaseSource):
    name = "missing-hash"
    plugin_version = "1.0.0"
