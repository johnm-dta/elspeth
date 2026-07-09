"""Commencement gate evaluation — pre-flight go/no-go checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from elspeth.contracts.errors import CommencementGateFailedError
from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.preflight import CommencementGateResult
from elspeth.core.commencement_gate_expression import (
    COMMENCEMENT_GATE_ALLOWED_NAMES,
    validate_commencement_gate_condition,
)
from elspeth.core.dependency_config import CommencementGateConfig
from elspeth.core.expression_parser import ExpressionParser
from elspeth.engine.error_boundary import reraise_if_engine_crash_through


def build_preflight_context(
    *,
    dependency_results: dict[str, Any],
    collection_probes: dict[str, Any],
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the pre-flight context dict for gate expression evaluation.

    Returns a namespace dict with three keys accessible in gate expressions:
    - ``dependency_runs``: {name: {run_id, settings_hash, duration_ms, indexed_at}} for each dependency
    - ``collections``: {name: {reachable, count}} for each probed collection
    - ``env``: explicitly supplied non-secret gate environment values
    """
    return {
        "dependency_runs": dependency_results,
        "collections": collection_probes,
        "env": env if env is not None else {},
    }


def _build_audit_snapshot(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build a frozen context snapshot for audit, excluding env values.

    Env values are excluded because they may contain secrets (API keys, tokens).
    Env key names are included so auditors know which variables were available
    during gate evaluation without exposing their values.
    """
    env = context["env"]
    frozen: Mapping[str, Any] = deep_freeze(
        {
            "dependency_runs": context["dependency_runs"],
            "collections": context["collections"],
            "env_keys": sorted(env.keys()),
        }
    )
    return frozen


def validate_gate_expressions(gates: list[CommencementGateConfig]) -> None:
    """Validate all gate expressions at config time, before any side effects.

    ExpressionParser validates syntax and security at construction time.
    Calling this before dependency resolution ensures malformed expressions
    are rejected before sub-pipelines run and mutate external state.

    Raises:
        ExpressionSecurityError: If expression contains forbidden constructs
        ExpressionSyntaxError: If expression is not valid Python syntax
    """
    for gate in gates:
        validate_commencement_gate_condition(gate.condition)


def evaluate_commencement_gates(
    gates: list[CommencementGateConfig],
    context: dict[str, Any],
) -> list[CommencementGateResult]:
    """Evaluate gates sequentially. Raises CommencementGateFailedError on failure.

    Context should be a namespace dict with keys from COMMENCEMENT_GATE_ALLOWED_NAMES
    (collections, dependency_runs, env). Unknown keys are not rejected here —
    the ExpressionParser restricts name access during evaluation.
    The entire context dict (including explicit Tier 3 env values) is deep-frozen before evaluation.
    """
    # Deep-freeze entire context for expression evaluation (Tier 3 boundary for explicit env)
    frozen_context = deep_freeze(context)
    # Build audit snapshot from the frozen context (same object used for evaluation) to ensure the snapshot reflects exactly what the gate saw.
    audit_snapshot = _build_audit_snapshot(frozen_context)

    results: list[CommencementGateResult] = []
    for gate in gates:
        try:
            parser = ExpressionParser(
                gate.condition,
                allowed_names=list(COMMENCEMENT_GATE_ALLOWED_NAMES),
            )
            result = parser.evaluate(frozen_context)
            if not isinstance(result, bool):
                raise CommencementGateFailedError(
                    gate_name=gate.name,
                    condition=gate.condition,
                    reason=(
                        # NB: do NOT include {result!r} — for a gate like env['API_KEY']
                        # the result IS the raw secret value, and this reason bypasses the
                        # env-scrubbed audit snapshot (elspeth-83261b699c). The type alone
                        # is the actionable diagnostic.
                        f"Gate expression returned {type(result).__name__}, "
                        f"not bool. Commencement gates must evaluate to True or False — "
                        f"use a comparison (e.g., '> 0', '== \"expected\"') instead of "
                        f"relying on Python truthiness."
                    ),
                    context_snapshot=audit_snapshot,
                )
            passed = result
        except CommencementGateFailedError:
            raise
        except BaseException as exc:
            reraise_if_engine_crash_through(exc)
            if not isinstance(exc, Exception):
                raise
            raise CommencementGateFailedError(
                gate_name=gate.name,
                condition=gate.condition,
                reason=f"Expression raised {type(exc).__name__}",
                context_snapshot=audit_snapshot,
            ) from exc

        if not passed:
            raise CommencementGateFailedError(
                gate_name=gate.name,
                condition=gate.condition,
                reason="Condition evaluated to falsy",
                context_snapshot=audit_snapshot,
            )

        results.append(
            CommencementGateResult(
                name=gate.name,
                condition=gate.condition,
                result=True,
                context_snapshot=audit_snapshot,
            )
        )
    return results
