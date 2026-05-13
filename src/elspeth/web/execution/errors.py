"""Structured execution-layer exceptions.

SemanticContractViolationError carries the same structured records as
the /validate endpoint surfaces, so callers of /execute that need to
render structured errors (frontend banner, MCP error payload) can do
so instead of falling back to string parsing.

Subclassing ValueError preserves backward compatibility for any caller
catching ValueError today; new callers should catch the specific type
to access entries and contracts.
"""

from __future__ import annotations

from elspeth.contracts.plugin_semantics import SemanticEdgeContract
from elspeth.web.composer.state import ValidationEntry


class SemanticContractViolationError(ValueError):
    """Raised when /execute pre-run semantic validation rejects the pipeline.

    Subclasses ValueError so existing ``except ValueError`` paths still
    catch it; new code should catch ``SemanticContractViolationError``
    directly to access the structured payload.
    """

    def __init__(
        self,
        *,
        entries: tuple[ValidationEntry, ...],
        contracts: tuple[SemanticEdgeContract, ...],
    ) -> None:
        self.entries = entries
        self.contracts = contracts
        message = "; ".join(entry.message for entry in entries)
        super().__init__(message)


class ExecuteRequestValidationError(ValueError):
    """Caller-authored /execute request data failed validation.

    Subclasses ValueError for compatibility with older service-level tests
    and callers, but route handlers should catch this specific base class
    and return HTTP 400 rather than conflating malformed input with 404
    not-found/IDOR responses.
    """


class PathAllowlistViolationError(ExecuteRequestValidationError):
    """Raised when a source or sink path escapes the configured allowlist."""


class MalformedBlobRefError(ExecuteRequestValidationError):
    """Raised when caller-supplied blob_ref is not a UUID."""


class BlobSourcePathMismatchError(Exception):
    """Tier 1 audit-integrity violation — composer-stored blob source path
    diverges from the referenced blob's canonical ``storage_path``.

    Raised at /execute time when ``composition_states.source.options.path``
    does not equal the canonical ``BlobRecord.storage_path`` for the bound
    ``blob_ref``.  This is our own audit data (Tier 1 in the trust model);
    a mismatch indicates a bug in composer persistence rather than user
    input we should coerce.  Crash informatively so the operator sees a
    structured error citing the divergence rather than a downstream
    ``FileNotFoundError`` from the source plugin.

    See elspeth-07089fbaa3 for the original defect that motivated the guard.
    """

    def __init__(
        self,
        *,
        stored_path: str | None,
        canonical_path: str,
        blob_id: str,
        session_id: str,
    ) -> None:
        self.stored_path = stored_path
        self.canonical_path = canonical_path
        self.blob_id = blob_id
        self.session_id = session_id
        super().__init__(
            f"Composer-stored blob source path does not match canonical "
            f"storage_path for blob {blob_id} (session {session_id}). "
            f"stored={stored_path!r} canonical={canonical_path!r}. "
            f"Expected canonical shape "
            f"<data_dir>/blobs/<session_id>/<blob_id>_<filename>. "
            f"This indicates a bug in composer persistence, not user input "
            f"to coerce. See elspeth-07089fbaa3."
        )
