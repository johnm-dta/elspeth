"""Factory for constructing collection probes from explicit config declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from elspeth.contracts.probes import CollectionProbe, CollectionReadinessResult
from elspeth.core.dependency_config import CollectionProbeConfig
from elspeth.plugins.infrastructure.clients.retrieval.connection import ChromaConnectionConfig


class ChromaCollectionProbe:
    """Probes a ChromaDB collection for readiness.

    Provider config is Tier 3 data (operator-authored YAML). Connection fields
    are validated at construction time by delegating to ChromaConnectionConfig
    (the same pattern used by ChromaSinkConfig and ChromaSearchProviderConfig).
    This ensures config errors surface as clear validation failures, not as raw
    KeyErrors during probe().
    """

    def __init__(self, collection: str, config: Mapping[str, Any]) -> None:
        self.collection_name = collection

        # Validate AND retain the model — probe() reads from the validated
        # instance so that Pydantic defaults (port=8000, ssl=True) are available
        # even when the operator's config omits them.
        try:
            self._conn = ChromaConnectionConfig(collection=collection, **config)
        except ValidationError as exc:
            # Re-raise as ValueError so callers see a clean config error, not a
            # Pydantic internal. First error message is the most specific.
            first_error = exc.errors()[0]["msg"]
            raise ValueError(f"Invalid provider_config for collection {collection!r}: {first_error}") from exc

    def probe(self) -> CollectionReadinessResult:
        """Check collection existence and document count."""
        import chromadb  # ImportError crashes — missing package is a config bug, not "unreachable"

        try:
            # Client construction CAN fail for infrastructure reasons (server down,
            # TLS errors, path permissions) — caught below as "unreachable".
            if self._conn.mode == "persistent":
                # persist_directory guaranteed non-None by validate_mode_fields
                assert self._conn.persist_directory is not None
                client = chromadb.PersistentClient(path=self._conn.persist_directory)
            else:
                # host guaranteed non-None by validate_mode_fields
                assert self._conn.host is not None
                client = chromadb.HttpClient(
                    host=self._conn.host,
                    port=self._conn.port,
                    ssl=self._conn.ssl,
                )

            collection = client.get_collection(self.collection_name)
            count = collection.count()
            return CollectionReadinessResult(
                collection=self.collection_name,
                reachable=True,
                count=count,
                message=(
                    f"Collection '{self.collection_name}' has {count} documents"
                    if count > 0
                    else f"Collection '{self.collection_name}' is empty"
                ),
            )
        except chromadb.errors.NotFoundError:
            # Collection doesn't exist — server reachable, collection absent.
            # count=None: we reached the server but the collection is absent,
            # so the count is unknown (not zero — zero means "empty").
            return CollectionReadinessResult(
                collection=self.collection_name,
                reachable=True,
                count=None,
                message=f"Collection '{self.collection_name}' not found",
            )
        except (chromadb.errors.ChromaError, ConnectionError, OSError) as exc:
            # Infrastructure failures: server down, auth errors, TLS failures,
            # path permission errors, connection refused, etc.
            return CollectionReadinessResult(
                collection=self.collection_name,
                reachable=False,
                count=None,
                message=f"Collection '{self.collection_name}' unreachable: {type(exc).__name__}: {exc}",
            )


_PROBE_REGISTRY: dict[str, type] = {
    "chroma": ChromaCollectionProbe,
}


def build_collection_probes(
    configs: list[CollectionProbeConfig],
) -> list[CollectionProbe]:
    """Construct probes from explicit config declarations."""
    probes: list[CollectionProbe] = []
    for config in configs:
        if config.provider not in _PROBE_REGISTRY:
            raise ValueError(f"Unknown collection probe provider: {config.provider!r}. Available: {sorted(_PROBE_REGISTRY)}")
        probe_cls = _PROBE_REGISTRY[config.provider]
        probes.append(probe_cls(config.collection, config.provider_config))
    return probes
