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

from typing import TYPE_CHECKING

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.operations import track_operation

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.orchestrator.types import PipelineConfig


def run_transform_runtime_preflights(
    factory: RecorderFactory,
    run_id: str,
    config: PipelineConfig,
    ctx: PluginContext,
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
            with track_operation(
                recorder=factory.execution,
                run_id=run_id,
                node_id=transform.node_id,
                operation_type="runtime_preflight",
                ctx=ctx,
                input_data={"transform_plugin": transform.name},
            ):
                transform.runtime_preflight(ctx)
        finally:
            ctx.node_id = previous_node_id
