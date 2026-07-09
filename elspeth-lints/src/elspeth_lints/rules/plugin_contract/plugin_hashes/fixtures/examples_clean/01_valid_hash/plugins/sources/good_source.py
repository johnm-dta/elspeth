class BaseSource:
    """Structural stand-in for the runtime plugin base (fixtures are AST-scanned, never imported)."""


class GoodSource(BaseSource):
    name = "good"
    plugin_version = "1.0.0"
    source_file_hash = "sha256:300892c38800c51b"
