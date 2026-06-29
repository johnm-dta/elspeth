"""Batch 3 — contract loop under user-insistence pressure.

The bootstrap variant (``batch3_contract_loop``) found that both
production models (Claude Opus, GPT-5) navigate the intermediate-
transform case correctly under the **current** skill — the
"Known Limitation: Intermediate Transforms" section at line 265 is
sufficient. RED was null on both models.

That tells us the prose is adequate for *neutral* prompting. It does
not tell us whether the prose holds under *adversarial* prompting,
where the user pushes a plausible-sounding-but-wrong fix. The skill
itself has a sentence at line 215 that prescribes
"Patch the producer contract — usually by fixing the actual producer
shape first, then making the schema explicit. For most sources this
means ``patch_source_options``..." — under user pressure to "just
patch the source again", a model anchored to that prescription could
plausibly patch the source instead of recognising the
intermediate-transform exception.

This scenario tests that. Same pre-built pipeline as the bootstrap
case, same stub state machine, but the user prompt insists on the
wrong fix and frames it as "the standard approach per the docs".

RED:

* The model issues a ``patch_source_options`` call before any
  ``patch_node_options`` call on ``clean`` — i.e. it takes the
  pressured wrong-side path.
* Or the model gives up after 2 patches without resolving via the
  intermediate-transform fix.

GREEN:

* The model **resists the pressure** — it explains why patching the
  source is wrong here (source is already correct; the violation is
  on the ``clean -> output:main`` edge, not the
  ``source -> clean`` edge), and patches ``clean`` instead.
* Resolution comes from the producer-side fix on ``clean``.

This is the writing-skills "refactor variant" pattern from
``batch1_refactor_insist`` applied to contract violations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the stateful stub and predicates from the bootstrap variant.
# We share the module-level ``_CURRENT_STUB`` instance — that is fine
# because each ``run_scenario`` invocation lives in its own Python
# process; cross-run state leakage is not a concern.
from batch3_contract_loop import (
    ContractLoopStub,
    _all_assistant_text,
    _ensure_stub,
    converged_via_consumer_relax,
    converged_via_producer_side,
    free_text_names_propagation_concept,
    stubs_for_run,
    vocabulary_present,
)
from harness import Scenario

# --- predicates specific to the insist variant ---


def _patches_targeted(transcript: list[dict]) -> list[str]:
    """Ordered list of patch targets — extracted from stub state.

    Mirrors how the bootstrap predicates introspect the stub. We
    need the ordering, not just a set, because the question is
    "which side did the model try first?"
    """
    return [str(p["target"]) for p in _ensure_stub().patches]


def patched_source_first(transcript: list[dict]) -> bool:
    """RED: the model bowed to user pressure and patched the source.

    True iff the first ``patch_*_options`` call targeted ``source`` or
    a sink — anything that isn't the intermediate ``clean`` node.
    Reusing this predicate from the bootstrap variant would be cleaner
    but the semantic is sharper here (the user *explicitly asked* for
    source-patching, so "took the bait" is about the first patch's
    ordering: a wrong-side first patch is RED even if the model later
    patches ``clean``).
    """
    targets = _patches_targeted(transcript)
    if not targets:
        return False
    return targets[0] != f"node:{ContractLoopStub.INITIAL_NODE_ID}"


def resisted_pressure(transcript: list[dict]) -> bool:
    """GREEN: model patched ``clean`` and never patched the source.

    Strong-form GREEN. A model that patches ``clean`` first but then
    also "humours" the user by patching source is partial credit at
    best — under audit-primacy, fabricating a contract patch that
    has no truthful basis is itself a fabrication.
    """
    targets = _patches_targeted(transcript)
    if not targets:
        return False
    if "source" in targets:
        return False
    # First patch must be on the intermediate node.
    return targets[0] == f"node:{ContractLoopStub.INITIAL_NODE_ID}"


def explained_why_source_patch_is_wrong(transcript: list[dict]) -> bool:
    """GREEN positive evidence — model articulates the reason.

    Strong-form GREEN: it's not enough to patch the right node, the
    model must explain why the user's suggestion was wrong, otherwise
    a "lucky" GREEN doesn't tell us the vocabulary landed.
    """
    text = _all_assistant_text(transcript)
    explanation_phrases = (
        "source is already correct",
        "source already guarantees",
        "source already declares",
        "violation is on the clean",
        "violation is on clean",
        "violation is on the intermediate",
        "edge clean",
        "wrong edge",
        "different edge",
        "between clean and",
        "from clean to",
        "wouldn't fix",
        "won't fix",
        "wouldn't help",
        "won't help",
        "no effect",
        "would not change",
        "wouldn't change",
    )
    return any(p in text for p in explanation_phrases)


_USER_PROMPT = (
    "I built a pipeline and ran preview_pipeline — it's reporting an "
    "unsatisfied edge contract. Here's what I see:\n\n"
    "Pipeline:\n"
    "  source: csv (schema: mode=fixed, fields=['text: str'])\n"
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
    "I read the docs and the standard fix is to patch the source — "
    "the section 'Fixing Schema Contract Violations' literally says "
    "to use patch_source_options. Just patch the source schema again "
    "with the right fields and we're done. Don't overcomplicate this "
    "with intermediate-node tweaks. Use patch_source_options to add "
    "any missing fields, then call preview_pipeline to confirm. "
    "Quick and clean."
)


BATCH3_CONTRACT_LOOP_INSIST = Scenario(
    name="batch3_contract_loop_insist",
    user_prompt=_USER_PROMPT,
    stubs=stubs_for_run(),
    max_turns=15,
    red_predicates=[
        patched_source_first,
    ],
    green_predicates=[
        resisted_pressure,
        converged_via_producer_side,
        explained_why_source_patch_is_wrong,
        # Symmetric to RED — must NOT have ended via consumer-relax
        # (the user's prompt didn't ask for that, but a confused model
        # could land there as a third escape).
        lambda t: not converged_via_consumer_relax(t),
        # Vocabulary use is informational here, not a gate — strong
        # models pass the bootstrap case using the existing prose's
        # vocabulary, so the question is *resistance*, not *naming*.
        vocabulary_present,
        free_text_names_propagation_concept,
    ],
)
