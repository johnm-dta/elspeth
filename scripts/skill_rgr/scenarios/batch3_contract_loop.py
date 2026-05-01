"""Batch 3 scenario — schema-contract violation loop.

This scenario tests the failure mode the static review of
``pipeline_composer.md`` predicted: that under an unsatisfied edge
contract, the LLM oscillates between patching the wrong sides of an
edge because the skill's prose conflates five distinct concepts under
the single word "schema":

1. ``schema:`` — the YAML config block
2. consumer-side validation (``required_fields``)
3. producer-side guarantee (``guaranteed_fields``)
4. consumer-side validation on a sink
5. pass-through propagation across an intermediate transform
   (``participates_in_propagation``, canonical predicate per
   ADR-009 §Clause 1, amending ADR-007).

The skill at ~line 271-280 has an arbitrary "stop after 2 attempts and
explain the limitation" guardrail — a working safety net that masks
the underlying vocabulary gap. This scenario is the empirical RED
that must precede any rewrite of the schema/violation-fixing sections.

Pipeline shape under test
-------------------------

The user starts with a *pre-built* pipeline (constructed via
``set_source`` / ``upsert_node`` / ``set_output`` in earlier turns;
the scenario replays the resulting state via ``get_pipeline_state``)::

    csv source --> clean transform --> use_text sink (output:main)

* ``csv`` source declares ``mode: fixed, fields: [text: str]`` —
  truthful producer guarantee.
* ``clean`` transform has **no schema declaration** — schema-less
  pass-through. This is the chain-breaking case: even though the data
  flows through unchanged, ``clean`` reports zero
  ``producer_guarantees`` to its downstream because it has no
  ``participates_in_propagation`` evidence.
* ``use_text`` sink declares ``required_input_fields: [text]`` —
  truthful consumer requirement.

``preview_pipeline`` returns ``edge_contracts`` showing
``clean -> output:main`` as ``satisfied: false`` with
``consumer_requires: [text]`` and ``producer_guarantees: []``.

The runtime-correct fix is to patch ``clean`` with
``{"schema": {"mode": "flexible", "fields": ["text: str"]}}`` so the
intermediate transform declares what it passes through. Patching the
source again is a no-op (it's already correct); patching the sink to
relax its requirement loses contract evidence.

Stateful preview_pipeline stub
------------------------------

The default harness stub returns an empty ``edge_contracts`` and
``is_valid: true``. That makes the scenario unanswerable — the LLM
cannot know whether its patch worked.

This scenario installs a ``ContractLoopStub`` state machine that:

1. Persists across the run (one instance per scenario invocation).
2. Tracks every ``patch_source_options`` / ``patch_node_options`` /
   ``patch_output_options`` call with its arguments.
3. Recomputes ``edge_contracts`` after each patch based on the
   resulting schema configuration.
4. Returns ``satisfied: true`` only when ``clean``'s declared schema
   includes ``text`` as a guaranteed field, OR when the sink's
   required fields no longer include ``text`` (the consumer-relax
   workaround). The two paths are observably distinct in the post-run
   stub state — predicates use that distinction.

The stub deliberately does **not** simulate runtime preflight. The
edge_contract simulation is enough for the LLM to observe whether its
patch closed the violation, which is the behaviour under test.

RED predicates (current skill)
------------------------------

RED 1 — *vocabulary absence*. The LLM never uses any of the runtime
concept names by name in free-text:
``guaranteed_fields``, ``required_input_fields``, ``required_fields``,
``participates_in_propagation``, ``pass-through``,
``intermediate transform``, ``audit_fields``. This is the most
diagnostic predicate — it fails when the LLM has only the overloaded
word "schema" to work with.

RED 2 — *wrong-side first patch*. The first ``patch_*_options`` call
targets ``source`` or the sink (``output:main``) rather than the
intermediate node ``clean``. This catches the LLM going for the
prominent endpoints while ignoring the actual fault.

RED 3 — *oscillation*. Three or more ``patch_*_options`` calls
without resolution. The skill's existing 2-attempt guardrail caps
this at 2 in practice, but the guardrail surrenders rather than
solving — RED 3 fires when the LLM gives up *or* when it would
otherwise loop. Detected as: ≥3 patches issued, OR ≥2 patches and
the LLM's final message admits the limitation without a substantive
fix recommendation.

GREEN predicates (post-rewrite skill)
-------------------------------------

GREEN 1 — vocabulary present. At least one of the runtime concept
names appears in free-text.

GREEN 2 — first patch is on ``clean``. The intermediate node, not
the source or sink.

GREEN 3 — final preview shows ``satisfied: true`` AND the resolution
came from a producer-side patch (``clean.schema.fields`` declares
``text``), not from relaxing the sink's ``required_input_fields``.

GREEN 4 — free-text mentions ``intermediate`` or ``pass-through`` or
``participates_in_propagation`` when describing the fix. Positive
evidence the rewrite's pass-through-propagation framing landed.

Out of scope (deferred to REFACTOR variants)
--------------------------------------------

- *insist* — user pushes back: "no, just patch the source".
- *override* — user demands ignoring the runtime concepts.
- *consumer-relax* — user wants the sink relaxed instead.

Each gets its own scenario file once GREEN holds on the bootstrap
case. Authoring four variants without baseline empirical signal is
the same premature-bulletproofing the ``batch3_bootstrap`` docstring
warns against.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (
    Scenario,
)

# ---------------------------------------------------------------------------
# Stateful stub
# ---------------------------------------------------------------------------


class ContractLoopStub:
    """In-memory schema state that ``preview_pipeline`` reads from.

    A single instance is constructed per scenario run and shared
    across all stub callbacks via closures. The harness creates one
    Scenario object globally, so we expose factory functions
    ``stubs_for_run()`` that build a fresh stub-and-callback set
    each time ``run_scenario`` is invoked.

    Why a class rather than module-level dict: the harness calls each
    stub independently and we need consistent state across calls in
    one run *without* leaking state into a subsequent run. A class
    instance per ``stubs_for_run()`` call gives that isolation
    cleanly.
    """

    INITIAL_NODE_ID = "clean"
    INITIAL_OUTPUT_NAME = "main"
    REQUIRED_FIELD = "text"

    def __init__(self) -> None:
        # Source schema: csv with truthful guarantee.
        self.source_schema: dict[str, object] = {
            "mode": "fixed",
            "fields": ["text: str"],
        }
        # Intermediate transform: no schema declared (the chain breaker).
        # Stored as None to model "schema-less" rather than empty.
        self.clean_schema: dict[str, object] | None = None
        # Sink: requires text.
        self.sink_required_fields: list[str] = ["text"]
        # Patch history — ordered list of (target, args).
        self.patches: list[dict[str, object]] = []

    # ----- helpers -----

    def _fields_of(self, schema: dict[str, object] | None) -> list[str]:
        """Extract field names from a schema dict's ``fields`` list.

        Field-spec strings look like ``"text: str"`` or
        ``"price: float?"``. The portion before the colon is the
        field name. Mirrors the production parser in
        ``contracts/schema.py::FieldDefinition.from_spec`` enough for
        the stub to compute guaranteed_fields.
        """
        if not schema:
            return []
        fields = schema.get("fields") or []
        names: list[str] = []
        for f in fields:
            if isinstance(f, str):
                name = f.split(":", 1)[0].strip()
                if name:
                    names.append(name)
            elif isinstance(f, dict) and "name" in f:
                names.append(str(f["name"]))
        return names

    def _node_guarantees(self) -> list[str]:
        """What ``clean`` reports as producer_guarantees on its outgoing edge.

        Empty when ``clean`` has no schema (initial state). When
        ``clean.schema.mode`` is ``fixed`` or ``flexible`` with typed
        fields, those fields are implicitly guaranteed (mirrors
        ``SchemaConfig.has_effective_guarantees``).
        """
        s = self.clean_schema
        if not s:
            return []
        mode = s.get("mode")
        if mode in ("fixed", "flexible"):
            return self._fields_of(s)
        # observed mode without explicit guaranteed_fields contributes none.
        gf = s.get("guaranteed_fields")
        if isinstance(gf, list):
            return [str(x) for x in gf]
        return []

    def _source_guarantees(self) -> list[str]:
        return self._fields_of(self.source_schema)

    # ----- tool callbacks -----

    def get_pipeline_state(self, _args: dict) -> dict:
        """Mirror the shape ``_execute_get_pipeline_state`` returns."""
        return {
            "source": {
                "plugin": "csv",
                "on_success": self.INITIAL_NODE_ID,
                "options": {"path": "data.csv", "schema": self.source_schema},
            },
            "nodes": [
                {
                    "id": self.INITIAL_NODE_ID,
                    "node_type": "transform",
                    "plugin": "passthrough",
                    "options": ({"schema": self.clean_schema} if self.clean_schema is not None else {}),
                    "on_success": f"output:{self.INITIAL_OUTPUT_NAME}",
                }
            ],
            "outputs": {
                self.INITIAL_OUTPUT_NAME: {
                    "plugin": "json",
                    "options": {
                        "path": "out.json",
                        "required_input_fields": list(self.sink_required_fields),
                    },
                }
            },
            "version": 1 + len(self.patches),
        }

    def preview_pipeline(self, _args: dict) -> dict:
        """Compute edge_contracts based on current schema state.

        Two edges:

        * ``source -> clean`` — always satisfied here (csv guarantees
          ``text``, and ``clean`` does not declare any
          ``required_fields``).
        * ``clean -> output:main`` — the failing edge initially.
          Becomes satisfied when ``clean``'s schema declares ``text``
          as a fixed/flexible field, OR when the sink's
          ``required_input_fields`` no longer includes ``text``.
        """
        clean_guarantees = self._node_guarantees()
        sink_required = list(self.sink_required_fields)
        missing = sorted(set(sink_required) - set(clean_guarantees))
        edge_satisfied = len(missing) == 0

        edge_contracts = [
            {
                "from": "source",
                "to": self.INITIAL_NODE_ID,
                "producer_guarantees": self._source_guarantees(),
                "consumer_requires": [],
                "missing_fields": [],
                "satisfied": True,
            },
            {
                "from": self.INITIAL_NODE_ID,
                "to": f"output:{self.INITIAL_OUTPUT_NAME}",
                "producer_guarantees": list(clean_guarantees),
                "consumer_requires": list(sink_required),
                "missing_fields": missing,
                "satisfied": edge_satisfied,
            },
        ]

        return {
            "is_valid": edge_satisfied,
            "errors": (
                []
                if edge_satisfied
                else [
                    {
                        "code": "edge_contract_unsatisfied",
                        "message": (
                            f"Edge {self.INITIAL_NODE_ID} -> "
                            f"output:{self.INITIAL_OUTPUT_NAME} requires "
                            f"{missing} but producer guarantees "
                            f"{clean_guarantees}"
                        ),
                    }
                ]
            ),
            "warnings": [],
            "suggestions": [],
            "edge_contracts": edge_contracts,
            "semantic_contracts": [],
            "source": {"plugin": "csv", "on_success": self.INITIAL_NODE_ID},
            "node_count": 1,
            "output_count": 1,
            "nodes": [
                {
                    "id": self.INITIAL_NODE_ID,
                    "node_type": "transform",
                    "plugin": "passthrough",
                }
            ],
            "outputs": [{"name": self.INITIAL_OUTPUT_NAME, "plugin": "json"}],
        }

    def patch_source_options(self, args: dict) -> dict:
        """Apply a shallow merge-patch to source options.

        Mirrors the production semantics: top-level keys are
        replaced atomically (no deep merge into ``schema``).
        """
        patch = args.get("patch") or {}
        if "schema" in patch:
            self.source_schema = dict(patch["schema"])
        self.patches.append({"target": "source", "args": dict(args)})
        return {"success": True, "version": 1 + len(self.patches)}

    def patch_node_options(self, args: dict) -> dict:
        node_id = args.get("node_id")
        patch = args.get("patch") or {}
        if node_id == self.INITIAL_NODE_ID and "schema" in patch:
            self.clean_schema = dict(patch["schema"])
        self.patches.append({"target": f"node:{node_id}", "args": dict(args)})
        return {"success": True, "version": 1 + len(self.patches)}

    def patch_output_options(self, args: dict) -> dict:
        sink_name = args.get("sink_name")
        patch = args.get("patch") or {}
        if sink_name == self.INITIAL_OUTPUT_NAME:
            if "required_input_fields" in patch:
                rif = patch["required_input_fields"]
                self.sink_required_fields = list(rif) if isinstance(rif, list) else []
            elif "schema" in patch:
                # Some skill prose suggests relaxing the sink schema
                # itself to ``observed``. We treat that as a
                # consumer-relax: clear required fields entirely so the
                # edge becomes satisfied.
                schema = patch["schema"]
                if isinstance(schema, dict) and schema.get("mode") == "observed":
                    self.sink_required_fields = []
        self.patches.append({"target": f"output:{sink_name}", "args": dict(args)})
        return {"success": True, "version": 1 + len(self.patches)}

    # ----- introspection used by predicates -----

    def first_patch_target(self) -> str | None:
        if not self.patches:
            return None
        return str(self.patches[0]["target"])

    def num_patches(self) -> int:
        return len(self.patches)

    def resolved_via_producer_side(self) -> bool:
        """True iff the resolution came from declaring fields on ``clean``."""
        return self.REQUIRED_FIELD in self._node_guarantees()

    def resolved_via_consumer_relax(self) -> bool:
        """True iff the sink's required fields no longer include ``text``."""
        return self.REQUIRED_FIELD not in self.sink_required_fields


# Module-level box so predicates and stubs share the *same* instance for
# a given run. ``stubs_for_run`` rebinds this for each invocation.
_CURRENT_STUB: ContractLoopStub | None = None


def _ensure_stub() -> ContractLoopStub:
    global _CURRENT_STUB
    if _CURRENT_STUB is None:
        _CURRENT_STUB = ContractLoopStub()
    return _CURRENT_STUB


def reset_stub() -> ContractLoopStub:
    """Force a fresh stub for the next run; called by the test driver."""
    global _CURRENT_STUB
    _CURRENT_STUB = ContractLoopStub()
    return _CURRENT_STUB


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


_VOCAB_MARKERS: tuple[str, ...] = (
    "guaranteed_fields",
    "required_input_fields",
    "required_fields",
    "participates_in_propagation",
    "pass-through",
    "pass through",
    "intermediate transform",
    "audit_fields",
    "edgecontract",
    "producer_guarantees",
    "consumer_requires",
)


def _all_assistant_text(transcript: list[dict]) -> str:
    """Concatenate every assistant free-text content into one lowered string.

    Vocabulary use can appear in *any* assistant turn — the model may
    explain its reasoning before issuing the patch and never
    re-mention the concept later. Looking at the last message only
    (as ``batch1_pressured`` does for its single-question prompts)
    would miss that. Concatenation gives a single haystack and
    matches the predicate's intent: "did the model name the concept
    at any point during this scenario?"
    """
    parts: list[str] = []
    for entry in transcript:
        if entry.get("role") != "assistant":
            continue
        content = entry.get("content") or ""
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts).lower()


def vocabulary_absent(transcript: list[dict]) -> bool:
    text = _all_assistant_text(transcript)
    return not any(marker in text for marker in _VOCAB_MARKERS)


def vocabulary_present(transcript: list[dict]) -> bool:
    return not vocabulary_absent(transcript)


def first_patch_on_wrong_side(transcript: list[dict]) -> bool:
    stub = _ensure_stub()
    target = stub.first_patch_target()
    if target is None:
        return False  # No patches at all — different failure, not "wrong side".
    return target != f"node:{ContractLoopStub.INITIAL_NODE_ID}"


def first_patch_on_clean(transcript: list[dict]) -> bool:
    stub = _ensure_stub()
    return stub.first_patch_target() == f"node:{ContractLoopStub.INITIAL_NODE_ID}"


def oscillation_or_giveup(transcript: list[dict]) -> bool:
    """RED 3 — three+ patches OR two patches followed by surrender prose.

    The skill's existing 2-attempt guardrail caps actual loops at 2,
    but the guardrail's "explain the limitation" exit is itself a
    failure mode for our purposes: it's a structural surrender that
    a vocabulary-armed LLM would not need.
    """
    stub = _ensure_stub()
    if stub.num_patches() >= 3:
        return True
    if stub.num_patches() >= 2:
        last_text = ""
        for entry in reversed(transcript):
            if entry.get("role") == "assistant":
                content = entry.get("content")
                if isinstance(content, str) and content.strip():
                    last_text = content.lower()
                    break
        surrender_phrases = (
            "cannot be verified",
            "cannot verify",
            "limitation",
            "structural rather than",
            "won't satisfy",
            "wont satisfy",
            "won't propagate",
            "wont propagate",
            "give up",
            "stop patching",
            "explain the limitation",
        )
        # Surrender-after-2 only counts as RED if the resolution
        # was not in fact reached via producer-side fix.
        if any(p in last_text for p in surrender_phrases) and not stub.resolved_via_producer_side():
            return True
    return False


def converged_via_producer_side(transcript: list[dict]) -> bool:
    stub = _ensure_stub()
    return stub.resolved_via_producer_side()


def converged_via_consumer_relax(transcript: list[dict]) -> bool:
    stub = _ensure_stub()
    return stub.resolved_via_consumer_relax()


def free_text_names_propagation_concept(transcript: list[dict]) -> bool:
    text = _all_assistant_text(transcript)
    keywords = (
        "intermediate",
        "pass-through",
        "pass through",
        "participates_in_propagation",
    )
    return any(k in text for k in keywords)


# ---------------------------------------------------------------------------
# Stub registration
# ---------------------------------------------------------------------------


def stubs_for_run() -> dict[str, callable]:
    """Build a fresh stub instance and return its tool callbacks.

    Called once per ``run_scenario`` invocation by the harness via
    the ``Scenario.stubs`` attribute. We can't lazy-build the stubs
    from inside the harness loop, but we *can* reset the module-level
    ``_CURRENT_STUB`` here so a re-run starts clean.
    """
    stub = reset_stub()
    return {
        "get_pipeline_state": stub.get_pipeline_state,
        "preview_pipeline": stub.preview_pipeline,
        "patch_source_options": stub.patch_source_options,
        "patch_node_options": stub.patch_node_options,
        "patch_output_options": stub.patch_output_options,
    }


# ---------------------------------------------------------------------------
# Scenario object
# ---------------------------------------------------------------------------


_USER_PROMPT = (
    "I built a small pipeline and ran preview_pipeline — it's reporting "
    "an unsatisfied edge contract. Here's what I see:\n\n"
    "Pipeline:\n"
    "  source: csv (truthful schema: mode=fixed, fields=['text: str'])\n"
    "  -> transform: clean (plugin=passthrough, no schema declared)\n"
    "  -> output: main (plugin=json, required_input_fields=['text'])\n\n"
    "preview_pipeline returns:\n"
    "  edge_contracts: [\n"
    "    {from: source, to: clean, satisfied: true, "
    "producer_guarantees: ['text'], consumer_requires: []},\n"
    "    {from: clean, to: output:main, satisfied: false, "
    "producer_guarantees: [], consumer_requires: ['text'], "
    "missing_fields: ['text']}\n"
    "  ]\n\n"
    "The data flows through 'clean' unchanged — it's a passthrough. "
    "But the contract check is failing. Please diagnose what's wrong "
    "and fix it. Use the composer tools (get_pipeline_state, "
    "preview_pipeline, patch_source_options, patch_node_options, "
    "patch_output_options) and call preview_pipeline at the end to "
    "confirm the edge is satisfied. Explain your reasoning as you go."
)


# Build stubs at module load — the harness calls each stub during the
# run, but they all reference the module-level _CURRENT_STUB which we
# reset via the harness wrapper at run start.
BATCH3_CONTRACT_LOOP = Scenario(
    name="batch3_contract_loop",
    user_prompt=_USER_PROMPT,
    stubs=stubs_for_run(),
    max_turns=15,
    red_predicates=[
        vocabulary_absent,
        first_patch_on_wrong_side,
        oscillation_or_giveup,
    ],
    green_predicates=[
        vocabulary_present,
        first_patch_on_clean,
        converged_via_producer_side,
        free_text_names_propagation_concept,
    ],
)
