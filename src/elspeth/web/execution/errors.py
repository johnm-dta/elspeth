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

from typing import TYPE_CHECKING

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.plugin_semantics import SemanticEdgeContract
from elspeth.web.composer.state import ValidationEntry
from elspeth.web.interpretation_state import InterpretationReviewSite

if TYPE_CHECKING:
    from elspeth.web.execution.schemas import ValidationError, ValidationReadiness


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


class PipelineValidationError(ValueError):
    """Raised when /execute pre-run dry-run validation (``validate_pipeline``)
    rejects the composed pipeline BEFORE a run is created.

    Closes the "detect != enforce" gap (notes/composer-advisor-surface-map-2026-06-08.md):
    ``validate_pipeline`` already detects graph / value-source / generic plugin-config
    contract violations, but ``execute()`` previously created a run and let it fail
    opaquely (``status=failed``, ``rows_processed=0``). This fails CLOSED with the
    structured ``ValidationError`` payload — mirroring ``SemanticContractViolationError``'s
    422 contract — and also closes the tutorial bypass (``tutorial_service`` calls
    ``execute()`` directly with no pre-run validation).

    Not every failure class is preflightable: ``SchemaConfigModeViolation`` is a
    post-emission row check and the Chroma SSRF check is deliberately deferred network
    I/O — those remain runtime failures by design.

    Subclasses ValueError for back-compat with existing ``except ValueError`` paths;
    new callers catch this type to access ``errors`` / ``readiness``.
    """

    def __init__(
        self,
        *,
        errors: tuple[ValidationError, ...],
        readiness: ValidationReadiness | None = None,
    ) -> None:
        errors = tuple(errors)
        if not errors:
            # Offensive guard: is_valid=False ALWAYS carries >=1 error. Constructing
            # this with no errors means the caller raised without checking
            # ValidationResult.is_valid — a control-flow bug worth crashing for.
            raise ValueError(
                "PipelineValidationError requires at least one ValidationError — "
                "caller must check ValidationResult.is_valid before raising."
            )
        self.errors = errors
        self.readiness = readiness
        message = "; ".join(e.message for e in errors)
        super().__init__(f"Pipeline failed pre-run validation: {message}")


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


class UnresolvedInterpretationPlaceholderError(Exception):
    """Raised when /execute encounters an LLM transform whose prompt_template
    still carries one or more unresolved ``{{interpretation:<term>}}``
    placeholders (F-17 / F-21 — Phase 5b Task 5 follow-on).

    The compose-loop is expected to call ``request_interpretation_review``
    for each such placeholder during composition; the placeholder is then
    replaced with the user-accepted concrete value, and the audit-primary
    record is the ``interpretation_events`` row.  If a placeholder survives
    to /execute, the LLM under-fired the review tool — typically after a
    model upgrade where the skill's prompt no longer reliably triggers the
    review path.  We refuse to run the pipeline with the literal
    ``{{interpretation:…}}`` string flowing into the LLM transform (which
    would produce a useless or surprising response) and surface a
    user-actionable error.

    NOT a subclass of ``ExecuteRequestValidationError``: that base maps to
    400 in the route catch order, but this site maps to 422 (Unprocessable
    Entity) to mirror the ``SemanticContractViolationError`` precedent —
    the request was syntactically valid but the composition state is not
    yet executable until the operator resolves the surfaced placeholders.

    The ``placeholders`` field carries ``(node_id, term)`` tuples — NOT
    the ``prompt_template`` value (which may include user-supplied
    content; PII risk).  The route handler renders the same shape into the
    HTTP 422 payload so the frontend banner can list every unresolved
    site without parsing the message string.
    """

    def __init__(
        self,
        *,
        placeholders: tuple[tuple[str, str], ...] | None = None,
        sites: tuple[InterpretationReviewSite, ...] | None = None,
    ) -> None:
        if sites is None:
            if not placeholders:
                # Offensive guard: callers must not raise this exception with
                # an empty placeholders tuple.  An empty list means the gate
                # passed and execution should proceed; constructing the
                # exception in that case is a control-flow bug worth crashing
                # for rather than silently producing an actionable error with
                # an empty body.
                raise ValueError(
                    "UnresolvedInterpretationPlaceholderError requires at "
                    "least one (node_id, term) tuple — caller must check the "
                    "detector's return value before raising."
                )
            sites = tuple(
                InterpretationReviewSite(
                    component_id=node_id,
                    component_type="transform",
                    user_term=term,
                    kind=InterpretationKind.VAGUE_TERM,
                )
                for node_id, term in placeholders
            )
        elif not sites:
            # Offensive guard: callers must not raise this exception with
            # an empty sites tuple.  An empty list means the gate
            # passed and execution should proceed; constructing the
            # exception in that case is a control-flow bug worth crashing
            # for rather than silently producing an actionable error with
            # an empty body.
            raise ValueError(
                "UnresolvedInterpretationPlaceholderError requires at "
                "least one interpretation-review site — caller must check the "
                "detector's return value before raising."
            )
        self.sites = sites
        self.placeholders = tuple(
            (site.component_id, site.user_term)
            for site in sites
            if site.component_type == "transform" and site.kind is InterpretationKind.VAGUE_TERM
        )
        if placeholders is not None and self.placeholders != placeholders:
            raise ValueError("placeholders must match transform/vague_term interpretation-review sites")
        # User-actionable message: list every unresolved (node, term) so
        # the operator sees all sites at once.  Single-site messages read
        # naturally; multi-site messages join with "; " to stay on one
        # line for the frontend banner.
        rendered_sites = "; ".join(_render_interpretation_site(site) for site in sites)
        message = (
            f"Unresolved interpretation review(s) — {rendered_sites}. "
            f"Resolve via request_interpretation_review (compose loop) and "
            f"the /interpretations/<event_id>/resolve endpoint before "
            f"running the pipeline."
        )
        super().__init__(message)


def _render_interpretation_site(site: InterpretationReviewSite) -> str:
    if site.component_type == "transform" and site.kind is InterpretationKind.VAGUE_TERM:
        return f"{{{{interpretation:{site.user_term}}}}} in LLM transform '{site.component_id}'"
    return f"{site.kind.value} review for {site.component_type} '{site.component_id}': {site.user_term}"


class RunSessionIntegrityError(Exception):
    """Tier 1 audit-integrity violation — a run row references a session id
    that has no matching ``sessions`` row.

    Raised by ``ExecutionService.verify_run_ownership`` when
    ``get_run(run_id)`` succeeds but the subsequent
    ``get_session(run.session_id)`` raises ``SessionNotFoundError``.  The run
    exists, so the ``run_id`` was neither malformed nor absent (those are the
    legitimate Tier-3 not-found cases that map to a 4004 close); a *dangling*
    parent-session reference is referential corruption of our own sessions DB,
    not hostile client input we should coerce into "Run not found".

    NOT a subclass of ``ValueError``: the WebSocket / REST ownership paths
    catch broad ``ValueError`` to convert Tier-3 malformed/not-found run ids
    into IDOR-safe not-found closes.  ``SessionNotFoundError`` is itself a
    ``ValueError`` subclass, so re-raising the corruption as a plain
    ``Exception`` is what keeps it from being silently collapsed into that
    benign path; the route surfaces it via an internal-error close (1011) and
    operator-channel logging instead.  Mirrors ``BlobSourcePathMismatchError``.
    """

    def __init__(self, *, run_id: str, session_id: str) -> None:
        self.run_id = run_id
        self.session_id = session_id
        super().__init__(
            f"Run {run_id} references session {session_id}, which has no "
            f"matching sessions row. This is referential corruption of the "
            f"sessions DB (a run with a dangling parent-session FK), not user "
            f"input to coerce into a not-found response."
        )


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
