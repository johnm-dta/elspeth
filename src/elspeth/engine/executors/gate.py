"""GateExecutor - wraps config-driven gates with audit recording and routing."""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from elspeth.contracts import (
    ConfigGateReason,
    GateResult,
    RouteDestination,
    RouteDestinationKind,
    RoutingAction,
    RoutingReason,
    RoutingSpec,
    TokenInfo,
)
from elspeth.contracts.enums import (
    NodeStateStatus,
    RoutingMode,
)
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.node_state_context import GateEvaluationContext
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.secret_scrub import scrub_text_for_audit
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.canonical import stable_hash
from elspeth.core.config import GateSettings
from elspeth.core.expression_parser import (
    ExpressionEvaluationError,
    ExpressionParser,
    ExpressionSecurityError,
    ExpressionSyntaxError,
)
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine.executors.state_guard import NodeStateGuard
from elspeth.engine.executors.types import GateOutcome, MissingEdgeError
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.tokens import TokenManager

logger = logging.getLogger(__name__)
slog = structlog.get_logger(__name__)

_GATE_VALUE_PREVIEW_CHARS = 80


def _describe_untrusted_gate_value(value: Any) -> str:
    """Return bounded metadata for row-derived gate expression results."""
    if isinstance(value, str):
        raw_text = value
    else:
        try:
            raw_text = repr(value)
        except Exception:
            raw_text = f"<unrepresentable {type(value).__name__}>"
    scrubbed = scrub_text_for_audit(raw_text)
    if len(scrubbed) > _GATE_VALUE_PREVIEW_CHARS:
        preview = scrubbed[:_GATE_VALUE_PREVIEW_CHARS] + "..."
    else:
        preview = scrubbed
    digest = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"type={type(value).__name__}, length={len(raw_text)}, sha256={digest}, preview={preview!r}"


@dataclass(frozen=True, slots=True)
class _RouteDispatchOutcome:
    """Internal routing dispatch result used by gate executors."""

    action: RoutingAction
    child_tokens: tuple[TokenInfo, ...] = ()
    sink_name: str | None = None
    next_node_id: NodeID | None = None
    discarded: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_tokens", tuple(self.child_tokens))
        if self.sink_name is not None and self.next_node_id is not None:
            raise ValueError(
                f"_RouteDispatchOutcome invariant violation: sink_name={self.sink_name!r} and "
                f"next_node_id={self.next_node_id!r} are mutually exclusive."
            )
        if self.discarded and (self.sink_name is not None or self.next_node_id is not None):
            raise ValueError(
                f"_RouteDispatchOutcome invariant violation: discarded=True cannot be combined with "
                f"sink_name={self.sink_name!r} or next_node_id={self.next_node_id!r}."
            )


class GateExecutor:
    """Executes config-driven gates with audit recording and routing.

    Evaluates gate conditions via ExpressionParser and:
    1. Records node state start
    2. Times the operation
    3. Populates audit fields in result
    4. Records routing events
    5. Creates child tokens for fork operations
    6. Records node state completion
    7. Emits OpenTelemetry span

    CRITICAL: Status is always "completed" for successful gate execution.
    Terminal state (ROUTED, FORKED) is DERIVED from routing_events/token_parents,
    NOT stored in node_states.status.

    Example:
        executor = GateExecutor(execution, span_factory, step_resolver, edge_map)
        outcome = executor.execute_config_gate(
            gate_config=gate_settings,
            node_id=node_id,
            token=token,
            ctx=ctx,
            token_manager=manager,  # Required for fork destinations
        )
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        span_factory: SpanFactory,
        step_resolver: StepResolver,
        edge_map: dict[tuple[NodeID, str], str] | None = None,
        route_resolution_map: dict[tuple[NodeID, str], RouteDestination] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            execution: Execution repository for audit trail
            span_factory: Span factory for tracing
            step_resolver: Resolves NodeID to 1-indexed audit step position
            edge_map: Maps (node_id, label) -> edge_id for routing
            route_resolution_map: Maps (node_id, label) -> resolved route destination
        """
        self._execution = execution
        self._spans = span_factory
        self._step_resolver = step_resolver
        self._edge_map = edge_map or {}
        self._route_resolution_map = route_resolution_map or {}

    def _resolve_route_destination(self, *, node_id: str, route_label: str) -> RouteDestination:
        """Resolve route label to concrete destination or fail closed."""
        try:
            return self._route_resolution_map[(NodeID(node_id), route_label)]
        except KeyError as exc:
            raise MissingEdgeError(node_id=NodeID(node_id), label=route_label) from exc

    def _dispatch_resolved_destination(
        self,
        *,
        state_id: str,
        node_id: str,
        route_label: str,
        destination: RouteDestination,
        token: TokenInfo,
        ctx: PluginContext,
        token_manager: "TokenManager | None",
        reason: RoutingReason | None,
        mode: RoutingMode,
        fork_branches: list[str] | None,
        continue_as_route: bool,
    ) -> _RouteDispatchOutcome:
        """Dispatch CONTINUE/FORK/SINK/PROCESSING_NODE destinations."""
        if destination.kind == RouteDestinationKind.CONTINUE:
            if continue_as_route:
                action = RoutingAction.route("continue", mode=mode, reason=reason)
            else:
                action = RoutingAction.continue_(reason=reason)
            self._record_routing(
                state_id=state_id,
                node_id=node_id,
                action=action,
            )
            return _RouteDispatchOutcome(action=action)

        if destination.kind == RouteDestinationKind.FORK:
            if fork_branches is None:
                raise OrchestrationInvariantError(
                    f"Gate {node_id} route '{route_label}' resolved to fork but no fork branches are configured"
                )
            if token_manager is None:
                raise OrchestrationInvariantError(
                    f"Gate {node_id} routes to fork but no TokenManager provided. "
                    "Cannot create child tokens - audit integrity would be compromised."
                )

            action = RoutingAction.fork_to_paths(fork_branches, reason=reason)
            child_tokens, _fork_group_id = token_manager.fork_token(
                parent_token=token,
                branches=fork_branches,
                node_id=NodeID(node_id),
                run_id=ctx.run_id,
                row_data=token.row_data,
            )
            self._record_routing(
                state_id=state_id,
                node_id=node_id,
                action=action,
            )
            return _RouteDispatchOutcome(action=action, child_tokens=tuple(child_tokens))

        if destination.kind == RouteDestinationKind.DISCARD:
            return _RouteDispatchOutcome(
                action=RoutingAction.route(route_label, mode=mode, reason=reason),
                discarded=True,
            )

        route_action = RoutingAction.route(route_label, mode=mode, reason=reason)
        self._record_routing(
            state_id=state_id,
            node_id=node_id,
            action=route_action,
        )
        if destination.kind == RouteDestinationKind.SINK:
            return _RouteDispatchOutcome(action=route_action, sink_name=destination.sink_name)
        if destination.kind == RouteDestinationKind.PROCESSING_NODE:
            return _RouteDispatchOutcome(action=route_action, next_node_id=destination.next_node_id)

        raise OrchestrationInvariantError(f"Unsupported route destination kind '{destination.kind}' for gate {node_id}")

    def execute_config_gate(
        self,
        gate_config: GateSettings,
        node_id: str,
        token: TokenInfo,
        ctx: PluginContext,
        token_manager: "TokenManager | None" = None,
    ) -> GateOutcome:
        """Execute a config-driven gate using ExpressionParser.

        Evaluates the gate condition directly using the expression parser.
        The condition expression is evaluated against the token's row_data.

        Route Resolution:
        - If condition returns a string, it's used as the route label directly
        - If condition returns a boolean, it's converted to "true"/"false" label
        - The label is then looked up in gate_config.routes to get the destination

        The step position in the DAG is resolved internally via StepResolver
        using node_id, rather than being passed as a parameter.

        Args:
            gate_config: Gate configuration with condition and routes
            node_id: Node ID assigned by orchestrator
            token: Current token with row data
            ctx: Plugin context
            token_manager: TokenManager for fork operations (required for fork destinations)

        Returns:
            GateOutcome with result, updated token, and routing info

        Raises:
            MissingEdgeError: If routing refers to an unregistered edge
            ValueError: If condition result doesn't match any route label
            RuntimeError: If fork destination without token_manager
        """
        # Resolve step position from node_id (injected StepResolver)
        step = self._step_resolver(NodeID(node_id))

        # Extract dict from PipelineRow for hashing and Landscape recording
        # Landscape stores raw dicts, not PipelineRow objects
        input_dict = token.row_data.to_dict()
        input_hash = stable_hash(input_dict)

        # Set ctx.contract for plugins that use fallback access (dual-name resolution)
        ctx.contract = token.row_data.contract

        # NodeStateGuard guarantees the node state reaches terminal status.
        # If any unhandled exception occurs before guard.complete() is called,
        # the guard auto-completes the state as FAILED in __exit__.
        with NodeStateGuard(
            self._execution,
            token_id=token.token_id,
            node_id=node_id,
            run_id=ctx.run_id,
            step_index=step,
            input_data=input_dict,
        ) as guard:
            # Create parser and evaluate condition
            with self._spans.gate_span(
                gate_config.name,
                node_id=node_id,
                input_hash=input_hash,
                token_id=token.token_id,
            ):
                start = time.perf_counter()
                try:
                    parser = ExpressionParser(gate_config.condition)
                    # Pass PipelineRow directly - it implements __getitem__ and .get()
                    # This preserves dual-name access (normalized and original field names)
                    eval_result = parser.evaluate(token.row_data)
                    duration_ms = (time.perf_counter() - start) * 1000
                except (ExpressionEvaluationError, ExpressionSecurityError, ExpressionSyntaxError):
                    duration_ms = (time.perf_counter() - start) * 1000
                    raise

            # Convert evaluation result to route label
            if isinstance(eval_result, bool):
                route_label = "true" if eval_result else "false"
            elif isinstance(eval_result, str):
                route_label = eval_result
            else:
                raise TypeError(
                    f"Gate '{gate_config.name}' expression returned unsupported route value "
                    f"({_describe_untrusted_gate_value(eval_result)}), expected bool or str. "
                    f"Expression: {gate_config.condition}"
                )

            # Look up destination in routes config
            if route_label not in gate_config.routes:
                raise ValueError(
                    f"Gate '{gate_config.name}' condition returned unconfigured route label "
                    f"({_describe_untrusted_gate_value(route_label)}); configured routes: {list(gate_config.routes.keys())}"
                )

            # Build routing action and process based on destination
            reason: ConfigGateReason = {"condition": gate_config.condition, "result": route_label}
            destination = self._resolve_route_destination(node_id=node_id, route_label=route_label)
            dispatch = self._dispatch_resolved_destination(
                state_id=guard.state_id,
                node_id=node_id,
                route_label=route_label,
                destination=destination,
                token=token,
                ctx=ctx,
                token_manager=token_manager,
                reason=reason,
                mode=RoutingMode.MOVE,
                fork_branches=gate_config.fork_to,
                continue_as_route=False,
            )
            action = dispatch.action
            child_tokens = dispatch.child_tokens
            sink_name = dispatch.sink_name
            next_node_id = dispatch.next_node_id
            discarded = dispatch.discarded

            # Create GateResult for audit fields
            # Config gates don't modify data, so use input dict as output
            result = GateResult(
                row=input_dict,
                action=action,
                contract=token.row_data.contract,  # Preserve contract reference
            )
            result.input_hash = input_hash
            result.output_hash = input_hash  # Gates don't modify data — output equals input
            result.duration_ms = duration_ms

            # Complete node state - always "completed" for successful execution
            # Terminal state is DERIVED from routing_events, not stored here
            gate_context = GateEvaluationContext(
                condition=gate_config.condition,
                result=str(eval_result),
                route_label=route_label,
            )
            guard.complete(
                NodeStateStatus.COMPLETED,
                output_data=input_dict,  # Landscape stores dict, not PipelineRow
                duration_ms=duration_ms,
                context_after=gate_context,
            )

        # Token row_data is unchanged (config gates don't modify data)
        # PipelineRow is already set on token, so just preserve it
        updated_token = token.with_updated_data(token.row_data)

        return GateOutcome(
            result=result,
            updated_token=updated_token,
            child_tokens=child_tokens,
            sink_name=sink_name,
            next_node_id=next_node_id,
            discarded=discarded,
        )

    def _record_routing(
        self,
        state_id: str,
        node_id: str,
        action: "RoutingAction",
    ) -> None:
        """Record routing events for a routing action.

        Raises:
            MissingEdgeError: If any destination has no registered edge.
        """
        typed_node_id = NodeID(node_id)
        if len(action.destinations) == 1:
            dest = action.destinations[0]
            try:
                edge_id = self._edge_map[(typed_node_id, dest)]
            except KeyError as exc:
                raise MissingEdgeError(node_id=typed_node_id, label=dest) from exc

            self._execution.record_routing_event(
                state_id=state_id,
                edge_id=edge_id,
                mode=action.mode,
                reason=action.reason,
            )
        else:
            # Multiple destinations (fork)
            routes = []
            for dest in action.destinations:
                try:
                    edge_id = self._edge_map[(typed_node_id, dest)]
                except KeyError as exc:
                    raise MissingEdgeError(node_id=typed_node_id, label=dest) from exc
                routes.append(RoutingSpec(edge_id=edge_id, mode=action.mode))

            self._execution.record_routing_events(
                state_id=state_id,
                routes=routes,
                reason=action.reason,
            )
