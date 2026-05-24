"""Type contracts for composer progress events.

Defines the public discriminator Literal (`ComposerProgressReason`) and the
sink callable shape (`ComposerProgressSink`) used by the composer progress
surface. Lives at L0 so `web/composer/protocol.py` can reference these
without importing `web/composer/progress.py`, which would form a cycle:

    web/composer/progress -> web/composer/tools -> web/composer/protocol -> ...

The TYPE_CHECKING import of `ComposerProgressEvent` below is architecturally
impure (an L0 module type-references an L3 symbol) but creates no runtime
coupling. See CLAUDE.md "Layer Dependency Rules → TYPE_CHECKING imports".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from elspeth.web.composer.progress import ComposerProgressEvent

# No ``__all__`` declared. The two public names below are PEP 695 ``type``
# aliases (``ComposerProgressReason``, ``ComposerProgressSink``); listing
# them in ``__all__`` trips CodeQL ``py/undefined-export`` because that
# rule (as of 2026-05) does not model PEP 695 type-alias bindings as
# definitions. The project's convention for type-alias modules
# (``contracts/declaration_contracts.py``, ``contracts/enums.py``,
# ``plugins/transforms/*.py``) is to omit ``__all__`` entirely — every
# consumer imports by name (``from elspeth.contracts.composer_progress
# import ComposerProgressReason``), no wildcard imports exist anywhere
# in the project, so ``__all__`` carries no functional load on these
# modules.


# Stable machine-readable reason codes for composer progress events.
#
# Public taxonomy distinct from ComposerConvergenceError.budget_exhausted —
# the exception models which budget tripped (a private engine concept), this
# Literal is the public-facing UX/observability discriminator. They map but
# they are not the same enum: the convergence error contributes three of
# these codes; the others come from sibling exception classes or from
# success/idle sentinels.
#
# Required when phase == "failed" (enforced by the model_validator on
# ComposerProgressEvent) so a new failure site cannot ship without carrying
# a stable code. The frontend, structured logs, and the 422 response body
# all branch on this value.
type ComposerProgressReason = Literal[
    # Convergence sub-causes — split out from the single
    # ComposerConvergenceError class via its budget_exhausted discriminator.
    "convergence_composition_budget",
    "convergence_discovery_budget",
    "convergence_wall_clock_timeout",
    "tool_call_cap_exceeded",
    # Provider-side failures — LiteLLM exception families.
    "provider_auth_failed",
    "provider_unavailable",
    # Server-side plugin bug escaping execute_tool.
    "plugin_crash",
    # Runtime preflight failure (cached path-1 or post-compose path-2 —
    # users cannot act on the path distinction, so a single code).
    "runtime_preflight_failed",
    # Generic ComposerServiceError — prompt prep / availability / catch-all.
    "service_setup_failed",
    # Client closed the HTTP connection or operator cancelled the request
    # before the composer returned. Distinct from convergence_wall_clock_timeout
    # (server budget exceeded) so dashboards and audit can tell apart "the
    # client gave up" from "the server gave up". Required when phase ==
    # "cancelled" by the same model_validator that requires it on "failed".
    "client_cancelled",
    # Non-failure sentinels — every snapshot carries a code so observability
    # and the SPA never have to special-case None.
    "composer_idle",
    "composer_complete",
]


type ComposerProgressSink = Callable[["ComposerProgressEvent"], Awaitable[None]]
