class BaseSource:
    """Structural stand-in for the runtime plugin base (fixtures are AST-scanned, never imported)."""


class MissingVersionSource(BaseSource):
    name = "missing-version"
    source_file_hash = "sha256:f4a070a1c743ba10"
