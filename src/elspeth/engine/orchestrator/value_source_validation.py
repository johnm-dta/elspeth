"""Value-source validation reports and errors for preflight consumers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueSourceFinding:
    """Structured per-field violation report from the value-source walker.

    Each finding pairs the offending ``component_id`` (the operator-facing
    transform name, e.g. ``openrouter_llm_node_1``) with the ``field_name``
    that violated its declaration and a human-readable ``reason``.

    Carrying the three fields directly - rather than encoding them into a
    formatted string and reverse-parsing at the consumer - eliminates the
    silent-attribution failure mode where a future format change would
    have produced ``ValidationError(component_id=None)`` records the
    composer UI cannot tie back to a specific node.

    All fields are scalars (per CLAUDE.md "Scalar-Only Fields Need No
    Guard"); ``frozen=True, slots=True`` is sufficient - no freeze guard
    is required.
    """

    component_id: str
    field_name: str
    reason: str

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("ValueSourceFinding.component_id must be non-empty")
        if not self.field_name:
            raise ValueError("ValueSourceFinding.field_name must be non-empty")
        if not self.reason:
            raise ValueError("ValueSourceFinding.reason must be non-empty")

    def format(self) -> str:
        """Render as a human-readable string for log/check-detail surfaces.

        The single point of stringification - anything wanting a flat
        message synthesises it here. Keeps the format coupled to the
        finding's own fields rather than scattering ``f"component '{...}'"``
        templates across the codebase.
        """
        return f"component '{self.component_id}' field '{self.field_name}': {self.reason}"


class ValueSourceValidationError(Exception):
    """Raised when a plugin-config field violates its value-source declaration.

    Examples:
    - An OpenRouter LLM transform's ``model`` field is set to a string
      that does not appear in the registered catalog.
    - An Azure LLM transform's ``model`` field has been overridden to a
      value that does not match its ``deployment_name`` sibling.

    Like :class:`RouteValidationError`, this error fires at pipeline
    initialization (pre-token), so the failure is per-pipeline rather
    than per-row.

    ``findings`` carries one :class:`ValueSourceFinding` per offending
    field. Consumers (e.g. the composer ``/validate`` path) read
    ``finding.component_id`` directly to attribute each violation to its
    node - no string parsing.
    """

    def __init__(
        self,
        message: str,
        *,
        findings: tuple[ValueSourceFinding, ...] = (),
    ) -> None:
        super().__init__(message)
        self.findings = findings
