"""Work queue cursor objects and construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from elspeth.contracts import TokenInfo
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import CoalesceName, NodeID


class WorkItemNavigation(Protocol):
    """Navigator surface needed to resolve cursor metadata."""

    def resolve_coalesce_node(self, coalesce_name: CoalesceName) -> NodeID: ...
    def resolve_coalesce_name(self, coalesce_node_id: NodeID) -> CoalesceName: ...
    def resolve_plugin_for_node(self, node_id: NodeID) -> object | None: ...
    def resolve_next_node(self, node_id: NodeID) -> NodeID | None: ...
    def resolve_branch_first_node(self, branch_name: str) -> NodeID: ...
    def is_fork_gate_node(self, node_id: NodeID) -> bool: ...


@dataclass(frozen=True, slots=True)
class WorkItem:
    """Item in the work queue for DAG processing."""

    token: TokenInfo
    current_node_id: NodeID | None
    coalesce_node_id: NodeID | None = None
    coalesce_name: CoalesceName | None = None
    on_success_sink: str | None = None

    def __post_init__(self) -> None:
        has_id = self.coalesce_node_id is not None
        has_name = self.coalesce_name is not None
        if has_id != has_name:
            raise OrchestrationInvariantError(
                f"WorkItem coalesce fields must be both set or both None: "
                f"coalesce_node_id={self.coalesce_node_id}, coalesce_name={self.coalesce_name}"
            )


@dataclass(frozen=True, slots=True)
class WorkItemFactory:
    """Build WorkItem cursors using topology resolver helpers."""

    navigation: WorkItemNavigation

    def create(
        self,
        *,
        token: TokenInfo,
        current_node_id: NodeID | None,
        coalesce_name: CoalesceName | None = None,
        coalesce_node_id: NodeID | None = None,
        on_success_sink: str | None = None,
    ) -> WorkItem:
        """Create a WorkItem, resolving one missing coalesce half when possible."""
        resolved_coalesce_node_id = coalesce_node_id
        resolved_coalesce_name = coalesce_name

        if resolved_coalesce_node_id is None and resolved_coalesce_name is not None:
            resolved_coalesce_node_id = self.navigation.resolve_coalesce_node(resolved_coalesce_name)
        elif resolved_coalesce_node_id is not None and resolved_coalesce_name is None:
            resolved_coalesce_name = self.navigation.resolve_coalesce_name(resolved_coalesce_node_id)

        return WorkItem(
            token=token,
            current_node_id=current_node_id,
            coalesce_node_id=resolved_coalesce_node_id,
            coalesce_name=resolved_coalesce_name,
            on_success_sink=on_success_sink,
        )

    def create_continuation(
        self,
        *,
        token: TokenInfo,
        current_node_id: NodeID,
        coalesce_name: CoalesceName | None = None,
        on_success_sink: str | None = None,
    ) -> WorkItem:
        """Create a child item that continues after current node or resumes at coalesce."""
        if coalesce_name is not None:
            coalesce_node_id = self.navigation.resolve_coalesce_node(coalesce_name)

            # Fork children route to the first processing node in their branch.
            # Non-fork continuations are already mid-branch and advance normally.
            if self.navigation.is_fork_gate_node(current_node_id):
                branch_name = token.branch_name
                if branch_name is None:
                    raise OrchestrationInvariantError(
                        f"Token '{token.token_id}' has coalesce_name='{coalesce_name}' but branch_name is None. "
                        "Fork children must have branch_name set."
                    )
                return self.create(
                    token=token,
                    current_node_id=self.navigation.resolve_branch_first_node(branch_name),
                    coalesce_name=coalesce_name,
                    coalesce_node_id=coalesce_node_id,
                    on_success_sink=on_success_sink,
                )

            return self.create(
                token=token,
                current_node_id=self.navigation.resolve_next_node(current_node_id),
                coalesce_name=coalesce_name,
                coalesce_node_id=coalesce_node_id,
                on_success_sink=on_success_sink,
            )

        return self.create(
            token=token,
            current_node_id=self.navigation.resolve_next_node(current_node_id),
            on_success_sink=on_success_sink,
        )
