"""Transform runtime-preflight execution at run-init.

This module owns the orchestrator's run-init step that drives each transform's
*declared* external-readiness check (``transform.runtime_preflight(ctx)``) and
records each as an audited operation before the source begins loading rows.

**Distinct from ``preflight.py``.** That sibling module is the static,
composer/service-shared pipeline-assembly + route-target validation contract
(no I/O, no audit recording, deliberately primitive so it can be called from
both the composer ``/validate`` endpoint and the execution service). This
module is orchestrator-run-only: it performs live external readiness checks
with full ``track_operation`` audit recording. Same word, different concern —
they intentionally do not share a module.

Pure delegation target for the Orchestrator (same pattern as
graph_wiring.py / run_status.py): it holds no orchestrator state and operates
only on the parameters passed in.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, cast

from elspeth.contracts import errors as contract_errors
from elspeth.contracts.errors import OrchestrationInvariantError, PluginRetryableError
from elspeth.core.operations import track_operation

if TYPE_CHECKING:
    from elspeth.contracts import TransformProtocol
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.orchestrator.types import PipelineConfig
    from elspeth.engine.retry import RetryManager


def _runtime_preflight_is_retryable(exc: BaseException) -> bool:
    if issubclass(type(exc), PluginRetryableError):
        return cast(PluginRetryableError, exc).retryable
    if issubclass(type(exc), contract_errors.RuntimePreflightFailedError):
        return cast(contract_errors.RuntimePreflightFailedError, exc).retryable
    return False


class _RuntimePreflightRetryMetadata:
    """Collect type-only retry metadata for the runtime preflight operation."""

    def __init__(self) -> None:
        self._retry_errors: list[dict[str, object]] = []

    def record_retry(self, attempt: int, error: BaseException) -> None:
        self._retry_errors.append({"attempt": attempt, "error_type": type(error).__name__})

    def output_data(self) -> dict[str, object]:
        return {
            "attempt_count": len(self._retry_errors) + 1,
            "retry_count": len(self._retry_errors),
            "retry_errors": list(self._retry_errors),
        }


def run_transform_runtime_preflights(
    factory: RecorderFactory,
    run_id: str,
    config: PipelineConfig,
    ctx: PluginContext,
    *,
    retry_manager: RetryManager | None = None,
    shutdown_event: threading.Event | None = None,
) -> None:
    """Run transform-declared external readiness checks before source load."""
    for transform in config.transforms:
        if not transform.requires_runtime_preflight:
            continue
        if transform.node_id is None:
            raise OrchestrationInvariantError(f"Transform {transform.name!r} requires runtime preflight before its node_id was assigned")

        previous_node_id = ctx.node_id
        ctx.node_id = transform.node_id
        try:
            retry_metadata = _RuntimePreflightRetryMetadata()
            with track_operation(
                recorder=factory.execution,
                run_id=run_id,
                node_id=transform.node_id,
                operation_type="runtime_preflight",
                ctx=ctx,
                input_data={"transform_plugin": transform.name},
            ) as handle:
                try:
                    if retry_manager is None:
                        transform.runtime_preflight(ctx)
                    else:
                        # Bind transform via default-argument capture (B023): the
                        # closure is invoked synchronously within this iteration,
                        # but the loop reassigns the name on each pass.
                        def run_runtime_preflight(_transform: TransformProtocol = transform) -> None:
                            _transform.runtime_preflight(ctx)

                        retry_manager.execute_with_retry(
                            run_runtime_preflight,
                            is_retryable=_runtime_preflight_is_retryable,
                            on_retry=retry_metadata.record_retry,
                            shutdown_event=shutdown_event,
                        )
                finally:
                    handle.output_data = retry_metadata.output_data()
        finally:
            ctx.node_id = previous_node_id
