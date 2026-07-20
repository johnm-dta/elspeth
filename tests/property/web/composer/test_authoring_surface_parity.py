"""Generated-DAG cross-surface authoring parity (Plan 05 Task 4).

Task 3 proved parity on nine hand-authored canonical fixtures. This module
raises the bar to *generated* pipelines: a Hypothesis strategy assembles bounded
policy-valid canonical DAGs by composing corpus-grounded building blocks, and
every generated case is driven — independently — through the same three
production authoring surfaces (freeform, guided-full, guided-staged) reused from
the Task 3 real-path adapters, then compared by graph isomorphism to a single
shared committed reference. Cross-surface parity is transitive: every surface is
anchored to the same per-case reference committed graph, exactly as in Task 3.

Why compose from corpus fragments rather than generate arbitrary node/edge
soup
-----------------------------------------------------------------------------
Every emitted case must not only pass the *boundary* (``SetPipelineArgumentsModel``
+ trained-operator/web plugin policy) but also survive the full audited
``set_pipeline`` commit (``validate_composition_state`` — connection
completeness, gate route parity, batch-aware placement, queue topology, runtime
route destinations, semantic field contracts) on THREE independent code paths.
Arbitrary DAG generation makes that intractable; the plan's own instruction is to
"ground shapes in the 9-fixture corpus". So the strategy draws a compact,
bounded ``_Spec`` and a deterministic builder assembles it from fragments each
lifted verbatim (modulo connection renaming) from a corpus fixture that already
commits and is isomorphism-checked in Task 3:

* source stage — one csv source, or two named csv sources fanning into one
  explicit queue (``multi_source_queue``);
* optional aggregation block — ``batch_stats`` alone, or ``batch_stats`` →
  ``batch_replicate`` (``aggregation`` / ``row_expansion``), always followed by a
  transform, mirroring the corpus (an aggregation never feeds a gate/sink
  directly there);
* a 1-2 node transform chain (``linear_transform``; ``type_coerce`` from
  ``conditional_gate`` / ``error_routing``);
* a terminal — single sink (``linear_transform``), a true/false gate split
  (``conditional_gate``), error routing (``error_routing``), a require-all fork /
  coalesce (``fork_coalesce``), or two outputs with a cross-sink write-failure
  fallback (``multi_output``).

LLM nodes are deliberately NOT generated: Task 4's shape list omits them, and the
``structured_llm`` capability stays covered as a Task 3 fixed fixture — this
avoids re-deriving the Task 1 typed-query / operator-profile machinery here while
losing no coverage.

Guided-staged gap awareness (the false-green this file must NOT introduce)
--------------------------------------------------------------------------
Two topology shapes are code-proven UN-authorable through the guided-staged stage
protocol at HEAD (Task 3 excludes ``multi_output`` and ``fork_coalesce`` for the
same reason, tracked as elspeth-b83b5b3204 and elspeth-93dd908354):

* a **require-all coalesce** — the guided consumer model (``NodeSpec`` has a
  single ``input``) cannot represent a multi-branch coalesce, so a branch
  connection is structurally orphaned in the wire projection;
* a **cross-sink ``on_write_failure`` fallback** — the structured sink review
  locks ``on_write_failure`` to ``discard`` unless it is a visible knob field
  (it is not for the json/csv sinks), so a priority→standard fallback is
  unauthorable.

The strategy generates BOTH shapes (they are real capabilities freeform and
guided-full author correctly) but the guided-staged comparison is GATED to the
supported subset. The gate is a *structural* predicate on the built graph —
``_guided_staged_authorable`` excludes a case iff it contains a coalesce node OR
any output whose ``on_write_failure`` is not ``discard`` — which maps EXACTLY to
those two tracked gaps and nothing else. Freeform and guided-full are compared on
the FULL generated space (all five terminals); only guided-staged drops the two
gap shapes. This is a proven-capability exclusion, not a "shrink the space until
it is green" exclusion: any guided-staged failure on a case that is NOT a
coalesce / cross-sink-fallback shape is a genuine regression (or a newly
discovered gap) and must fail this test, never be papered over by widening the
exclusion. See the module docstring of ``parity/test_fixture_matrix.py`` for the
per-mechanism code trace.

Determinism
-----------
``@settings(max_examples=50, deadline=None, derandomize=True)`` pins exactly 50
deterministic examples regardless of ``HYPOTHESIS_PROFILE`` (the loaded ``ci``
profile would otherwise run 100 non-derandomized examples); ``phases`` drops
``reuse`` so a stale ``.hypothesis`` example database cannot replay extra stored
counterexamples on top of the 50. The single shared ``parity_env`` production
stack is intentionally reused across all 50 examples — each ``drive`` /
``reference_state`` call creates a fresh session id, so isolation is per-session,
not per-example (that is why ``HealthCheck.function_scoped_fixture`` is
suppressed). The test body is synchronous and drives the async adapters through
``asyncio.run`` per example, following the repo's async-Hypothesis precedent
(``test_compose_loop_invariants.py``).

On failure the report names the intent, the full canonical payload, the surface,
and the first differing semantic attribute (``assert_isomorphic`` isolates the
last one).
"""

from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import dataclass
from typing import Any

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st
from tests.helpers.composer_graphs import assert_isomorphic

# The parity production stack + surface adapters live in the Task 3 parity
# conftest, which is a sibling scope not inherited by this property package.
# Importing the fixture by name makes it resolvable here — the exact pattern the
# parity conftest itself uses to borrow the guided suite's HTTP client fixture.
from tests.integration.web.composer.parity.conftest import (  # noqa: F401
    ParityEnv,
    parity_env,
)

# --------------------------------------------------------------------------- #
# Corpus-grounded option fragments (lifted verbatim from the fixture corpus)   #
# --------------------------------------------------------------------------- #


def _json_output(name: str, path: str, on_write_failure: str = "discard") -> dict[str, Any]:
    return {
        "sink_name": name,
        "plugin": "json",
        "options": {
            "path": path,
            "format": "json",
            "schema": {"mode": "observed"},
            "mode": "write",
            "collision_policy": "auto_increment",
        },
        "on_write_failure": on_write_failure,
    }


def _csv_output(name: str, path: str, on_write_failure: str = "discard") -> dict[str, Any]:
    return {
        "sink_name": name,
        "plugin": "csv",
        "options": {"path": path, "schema": {"mode": "observed"}, "mode": "write", "collision_policy": "auto_increment"},
        "on_write_failure": on_write_failure,
    }


def _transform_options(plugin: str) -> dict[str, Any]:
    if plugin == "type_coerce":
        return {"schema": {"mode": "observed"}, "conversions": [{"field": "amount", "to": "int"}]}
    return {"schema": {"mode": "observed"}}


# --------------------------------------------------------------------------- #
# Generation spec + deterministic builder                                     #
# --------------------------------------------------------------------------- #

_SOURCES = ("single", "queue")
_AGGREGATIONS = ("none", "stats", "stats_replicate")
_TRANSFORMS = ("passthrough", "type_coerce")
_TERMINALS = ("single_sink", "gate_split", "error_routing", "fork_coalesce", "multi_output")


@dataclass(frozen=True)
class _Spec:
    """A compact, bounded description of one generated canonical DAG.

    Its ``repr`` (surfaced by Hypothesis on shrink) names every generation
    decision, so a shrunk counterexample reports the exact shape.
    """

    source: str
    aggregation: str
    transforms: tuple[str, ...]
    terminal: str


def _build_case(spec: _Spec) -> dict[str, Any]:
    """Assemble a corpus-grounded ``{class, intent, canonical_arguments}`` case.

    The result is shaped exactly like a Task 3 corpus fixture so it drives
    unchanged through ``ParityEnv.reference_state`` and ``ParityEnv.drive``.
    Every connection name is minted uniquely; every node is a proven fragment.
    """
    conns = itertools.count()
    node_ids = itertools.count()

    def conn() -> str:
        return f"gc{next(conns)}"

    def node_id() -> str:
        return f"gn{next(node_ids)}"

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    # --- source stage -------------------------------------------------------
    source_field: dict[str, Any] | None
    sources_field: dict[str, Any] | None
    if spec.source == "queue":
        sources_field = {
            "orders": {
                "plugin": "csv",
                "on_success": "inbound",
                "options": {"path": "orders.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "refunds": {
                "plugin": "csv",
                "on_success": "inbound",
                "options": {"path": "refunds.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
        }
        source_field = None
        # Queue node: id == input, no plugin/routing, description-only options.
        nodes.append({"id": "inbound", "node_type": "queue", "input": "inbound", "options": {"description": "Fan-in point"}})
        head = "inbound"
        head_node = "inbound"
    else:
        source_field = {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": "input.csv", "schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
        }
        sources_field = None
        head = "rows"
        head_node = "source"

    # --- optional aggregation block (always followed by a transform below) ---
    if spec.aggregation in ("stats", "stats_replicate"):
        stats_out = conn()
        stats_id = node_id()
        nodes.append(
            {
                "id": stats_id,
                "node_type": "aggregation",
                "plugin": "batch_stats",
                "input": head,
                "on_success": stats_out,
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}, "value_field": "value"},
                "trigger": {"count": 10},
                "output_mode": "transform",
            }
        )
        head, head_node = stats_out, stats_id
        if spec.aggregation == "stats_replicate":
            rep_out = conn()
            rep_id = node_id()
            nodes.append(
                {
                    "id": rep_id,
                    "node_type": "aggregation",
                    "plugin": "batch_replicate",
                    "input": head,
                    "on_success": rep_out,
                    "on_error": "discard",
                    "options": {"schema": {"mode": "observed"}, "copies_field": "copies", "default_copies": 2},
                    "trigger": {"count": 1},
                    "output_mode": "transform",
                }
            )
            head, head_node = rep_out, rep_id

    # --- transform chain (1..2 nodes) ---------------------------------------
    for plugin in spec.transforms:
        t_out = conn()
        t_id = node_id()
        nodes.append(
            {
                "id": t_id,
                "node_type": "transform",
                "plugin": plugin,
                "input": head,
                "on_success": t_out,
                "on_error": "discard",
                "options": _transform_options(plugin),
            }
        )
        head, head_node = t_out, t_id

    tail_node = nodes[-1]  # the last spine node; terminals consume `head` off it

    # --- terminal -----------------------------------------------------------
    if spec.terminal == "single_sink":
        # Rewire the spine tail's on_success straight into one json sink.
        tail_node["on_success"] = "gsink"
        outputs.append(_json_output("gsink", "outputs/gsink.json"))

    elif spec.terminal == "gate_split":
        gate_id = node_id()
        nodes.append(
            {
                "id": gate_id,
                "node_type": "gate",
                "input": head,
                "condition": "row['value'] >= 0.0",
                "routes": {"true": "ghigh", "false": "glow"},
            }
        )
        outputs.append(_json_output("ghigh", "outputs/ghigh.json"))
        outputs.append(_json_output("glow", "outputs/glow.json"))

    elif spec.terminal == "error_routing":
        # error_routing is generated single-source only (matching the fixture):
        # a queue's per-source validation-failure fan-out is not proven.
        assert source_field is not None, "error_routing must be single-source"
        source_field["on_validation_failure"] = "grejected"
        coerce_id = node_id()
        nodes.append(
            {
                "id": coerce_id,
                "node_type": "transform",
                "plugin": "type_coerce",
                "input": head,
                "on_success": "gclean",
                "on_error": "gerrors",
                "options": {"schema": {"mode": "observed"}, "conversions": [{"field": "amount", "to": "int"}]},
            }
        )
        outputs.append(_json_output("gclean", "outputs/gclean.json"))
        outputs.append(_json_output("gerrors", "outputs/gerrors.json"))
        outputs.append(_json_output("grejected", "outputs/grejected.json"))

    elif spec.terminal == "fork_coalesce":
        gate_id = node_id()
        merge_id = node_id()
        finalize_id = node_id()
        nodes.append(
            {
                "id": gate_id,
                "node_type": "gate",
                "input": head,
                "condition": "True",
                "routes": {"true": "gfork", "false": "gfork"},
                "fork_to": ["gpath_a", "gpath_b"],
            }
        )
        nodes.append(
            {
                "id": merge_id,
                "node_type": "coalesce",
                "input": "gpath_a",
                "branches": {"gpath_a": "gpath_a", "gpath_b": "gpath_b"},
                "policy": "require_all",
                "merge": "union",
            }
        )
        nodes.append(
            {
                "id": finalize_id,
                "node_type": "transform",
                "plugin": "passthrough",
                "input": merge_id,
                "on_success": "gmerged",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            }
        )
        edges = [
            {"id": "ge1", "from_node": head_node, "to_node": gate_id, "edge_type": "on_success", "label": None},
            {"id": "ge2", "from_node": gate_id, "to_node": merge_id, "edge_type": "fork", "label": "gpath_a"},
            {"id": "ge3", "from_node": gate_id, "to_node": merge_id, "edge_type": "fork", "label": "gpath_b"},
            {"id": "ge4", "from_node": merge_id, "to_node": finalize_id, "edge_type": "on_success", "label": None},
        ]
        outputs.append(_json_output("gmerged", "outputs/gmerged.json"))

    elif spec.terminal == "multi_output":
        gate_id = node_id()
        nodes.append(
            {
                "id": gate_id,
                "node_type": "gate",
                "input": head,
                "condition": "row['value'] >= 0.0",
                "routes": {"true": "gpriority", "false": "gstandard"},
            }
        )
        # Cross-sink write-failure fallback: priority failures rejoin standard.
        outputs.append(_json_output("gpriority", "outputs/gpriority.json", on_write_failure="gstandard"))
        outputs.append(_csv_output("gstandard", "outputs/gstandard.csv"))

    else:  # pragma: no cover - strategy only emits the terminals above
        raise AssertionError(f"unknown terminal {spec.terminal!r}")

    canonical: dict[str, Any] = {"nodes": nodes, "edges": edges, "outputs": outputs}
    if sources_field is not None:
        canonical["sources"] = sources_field
    else:
        canonical["source"] = source_field
    canonical["metadata"] = {"name": "generated-dag", "description": _spec_summary(spec)}

    return {
        "class": f"generated:{spec.terminal}",
        "intent": _intent_for(spec),
        "canonical_arguments": canonical,
    }


def _spec_summary(spec: _Spec) -> str:
    return f"source={spec.source} aggregation={spec.aggregation} transforms={list(spec.transforms)} terminal={spec.terminal}"


def _intent_for(spec: _Spec) -> str:
    """A freeform-mutation-triggering intent that also names the drawn shape.

    Must trip ``user_request_expects_pipeline_mutation`` (a build phrase) so the
    freeform surface enters the empty-build planner path; the scripted completion
    emits the built pipeline regardless of the prose.
    """
    src = "two named CSV sources fanning into one shared queue" if spec.source == "queue" else "one CSV source"
    mid = []
    if spec.aggregation == "stats":
        mid.append("a batch-statistics aggregation")
    elif spec.aggregation == "stats_replicate":
        mid.append("a batch-statistics aggregation then a row-expanding replicate")
    mid.append(f"{len(spec.transforms)} ordered transform(s)")
    term = {
        "single_sink": "writing to one JSON output",
        "gate_split": "routing rows through a true/false gate into two outputs",
        "error_routing": "routing validation and coercion failures to explicit error sinks",
        "fork_coalesce": "forking every row into two branches merged by a require-all coalesce",
        "multi_output": "writing to two outputs with a cross-sink write-failure fallback",
    }[spec.terminal]
    return f"Build a pipeline that reads {src}, passes rows through {', '.join(mid)}, and finishes by {term}."


def _guided_staged_authorable(case: dict[str, Any]) -> bool:
    """True iff the built graph avoids BOTH proven guided-staged capability gaps.

    Structural, not terminal-name based, so it maps to the mechanism rather than
    the label: a require-all (any) coalesce node, or any output carrying a
    non-``discard`` ``on_write_failure``. These are exactly elspeth-93dd908354
    (multi-branch coalesce) and elspeth-b83b5b3204 (cross-sink write-failure
    fallback). Every other shape guided-staged authors correctly (Task 3 proves
    seven of nine corpus fixtures drive guided-staged, including gates, queues,
    aggregation, row expansion, and source-validation error routing).
    """
    args = case["canonical_arguments"]
    has_coalesce = any(node.get("node_type") == "coalesce" for node in args["nodes"])
    has_cross_sink_fallback = any((output.get("on_write_failure") or "discard") != "discard" for output in args["outputs"])
    return not (has_coalesce or has_cross_sink_fallback)


@st.composite
def _specs(draw: st.DrawFn) -> _Spec:
    """Draw a bounded, buildable ``_Spec``.

    Constraints are applied at draw time (not via ``assume``) so every emitted
    case builds into a policy-valid DAG and no example is silently discarded:
    ``error_routing`` is single-source only (its source-validation fan-out to a
    reject sink is proven single-source in the corpus).
    """
    terminal = draw(st.sampled_from(_TERMINALS))
    source = "single" if terminal == "error_routing" else draw(st.sampled_from(_SOURCES))
    aggregation = draw(st.sampled_from(_AGGREGATIONS))
    transforms = tuple(draw(st.lists(st.sampled_from(_TRANSFORMS), min_size=1, max_size=2)))
    return _Spec(source=source, aggregation=aggregation, transforms=transforms, terminal=terminal)


# --------------------------------------------------------------------------- #
# The property                                                                #
# --------------------------------------------------------------------------- #


def _failure_report(spec: _Spec, case: dict[str, Any], surface: str, detail: str) -> str:
    payload = json.dumps(case["canonical_arguments"], indent=2, sort_keys=True)
    return (
        "generated-DAG parity mismatch\n"
        f"  spec:    {spec!r}\n"
        f"  surface: {surface}\n"
        f"  intent:  {case['intent']}\n"
        f"  detail:  {detail}\n"
        f"  payload: {payload}"
    )


async def _drive_and_compare(env: ParityEnv, spec: _Spec, case: dict[str, Any], reference: Any) -> None:
    surfaces = ["freeform", "guided_full"]
    if _guided_staged_authorable(case):
        surfaces.append("guided_staged")
    for surface in surfaces:
        committed = await env.drive(surface, case)
        graph_left = f"{surface}:{case['class']}"
        try:
            assert_isomorphic(committed, reference, left=graph_left, right="reference")
        except AssertionError as exc:
            raise AssertionError(_failure_report(spec, case, surface, str(exc))) from exc


@settings(
    max_examples=50,
    deadline=None,
    derandomize=True,
    phases=[Phase.explicit, Phase.generate, Phase.shrink],
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(spec=_specs())
def test_generated_dag_surface_parity(parity_env: ParityEnv, spec: _Spec) -> None:  # noqa: F811
    """Every generated policy-valid DAG derives one isomorphic graph on all surfaces.

    freeform + guided-full are compared on the full generated space; guided-staged
    is compared only on cases it can author (the require-all-coalesce and
    cross-sink-write-failure shapes are the two tracked gaps). The shared
    reference anchors cross-surface parity transitively.
    """
    case = _build_case(spec)

    async def _run() -> None:
        # reference_state is synchronous; build it inside the loop so any deep
        # commit failure surfaces with the same generated case for diagnosis.
        reference = parity_env.reference_state(case)
        await _drive_and_compare(parity_env, spec, case, reference)

    asyncio.run(_run())
