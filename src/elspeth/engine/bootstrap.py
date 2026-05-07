"""Programmatic pipeline bootstrap — resolve_preflight for dependency resolution and gates."""

from __future__ import annotations

from pathlib import Path

from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.pipeline_runner import PipelineRunner
from elspeth.contracts.probes import CollectionProbe
from elspeth.core.config import ElspethSettings
from elspeth.core.dependency_config import (
    CommencementGateResult,
    DependencyRunResult,
    PreflightResult,
)


def resolve_preflight(
    config: ElspethSettings,
    settings_path: Path,
    *,
    probes: list[CollectionProbe] | None = None,
    runner: PipelineRunner | None = None,
) -> PreflightResult | None:
    """Run dependency resolution and commencement gates if configured.

    This is the pre-execution phase that resolves ``depends_on`` sub-pipelines
    and evaluates ``commencement_gates``. Extracted so both the CLI and
    programmatic callers share the same codepath.

    Args:
        config: Validated pipeline settings (must already be loaded).
        settings_path: Path to the settings YAML — needed for cycle detection
            and dependency resolution (sub-pipelines resolve relative to this).
        probes: Pre-built collection probes for commencement gates. Required
            when ``commencement_gates`` is configured (raises ``FrameworkBugError``
            if None). Caller constructs these at L3 (where probe_factory lives).
        runner: Callback to execute sub-pipelines. Required when ``depends_on``
            is configured (raises ``FrameworkBugError`` if None). Injected from
            L3 to avoid L2→L3 import.

    Returns:
        PreflightResult if any dependencies or gates were configured, else None.
    """
    dependency_results: list[DependencyRunResult] = []
    gate_results: list[CommencementGateResult] = []

    # Validate gate expressions early — before any dependency resolution.
    # ExpressionParser validates syntax and security at construction time.
    # If a gate condition is malformed, we reject it here rather than after
    # dependency pipelines have already run and mutated external state.
    if config.commencement_gates:
        from elspeth.engine.commencement import validate_gate_expressions

        validate_gate_expressions(config.commencement_gates)

    # Dependency resolution (if configured)
    if config.depends_on:
        from elspeth.engine.dependency_resolver import detect_cycles, resolve_dependencies

        if runner is None:
            raise FrameworkBugError("runner is required when depends_on is configured — caller must inject a PipelineRunner callback")

        # Validate dependency name uniqueness BEFORE any execution.
        dep_names = [dep.name for dep in config.depends_on]
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in dep_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"Duplicate dependency names: {duplicates}. Each depends_on entry must have a unique name.")

        detect_cycles(settings_path)

        # Run dependencies sequentially — each calls runner() recursively.
        # Design note: sub-pipelines that declare their own depends_on will also
        # resolve their dependencies recursively. Cycle detection prevents infinite
        # loops. Diamond dependencies (A→B, A→C, B→D, C→D) cause D to run twice —
        # acceptable for correctness since each run produces its own audit trail.
        dependency_results = resolve_dependencies(
            depends_on=config.depends_on,
            parent_settings_path=settings_path,
            runner=runner,
        )

    # Commencement gates (if configured)
    if config.commencement_gates:
        from elspeth.engine.commencement import build_preflight_context, evaluate_commencement_gates

        if probes is None:
            raise FrameworkBugError(
                "probes is required when commencement_gates is configured — caller must inject pre-built CollectionProbe instances"
            )

        # Execute probes (caller provides pre-built probes from L3)
        probe_results = {}
        for probe in probes:
            result = probe.probe()
            if result.collection in probe_results:
                raise FrameworkBugError(
                    f"Duplicate collection probe name '{result.collection}' — "
                    f"earlier probe result would be silently overwritten. "
                    f"Each probe must target a unique collection name."
                )
            probe_results[result.collection] = {
                "reachable": result.reachable,
                "count": result.count,
            }

        # Convert dependency results to gate-accessible dict
        dep_run_dict = {
            r.name: {
                "run_id": r.run_id,
                "settings_hash": r.settings_hash,
                "duration_ms": r.duration_ms,
                "indexed_at": r.indexed_at,
            }
            for r in dependency_results
        }

        context = build_preflight_context(
            dependency_results=dep_run_dict,
            collection_probes=probe_results,
        )
        gate_results = evaluate_commencement_gates(config.commencement_gates, context)

    # Build pre-flight result for audit recording (passed through to orchestrator).
    # Guard on config presence, not result emptiness — a PreflightResult with empty
    # tuples means "preflight was configured and ran but produced no results," which
    # is an auditable fact distinct from "preflight was not configured" (None).
    if config.depends_on or config.commencement_gates:
        return PreflightResult(
            dependency_runs=tuple(dependency_results),
            gate_results=tuple(gate_results),
        )
    return None
