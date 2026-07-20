"""Composer capability-parity documentation contract.

Guided and freeform are two *interactions* over one shared planner and one
canonical pipeline language; they do not differ in capability. The user manual
must say so, list the supported canonical structures, explain wrong-stage
retention / back-edit, and describe the tutorial as a guided workflow profile.
It must NOT tell users to switch to freeform because guided cannot express a
supported topology. Where schema/epoch numbers are encoded, the runbook must use
the Plan 05 values (session epoch 35, guided schema 10, Landscape epoch 28), not
the design doc's stale 8/28.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
USER_MANUAL = REPO_ROOT / "docs/guides/user-manual.md"
RUNBOOK = REPO_ROOT / "docs/runbooks/staging-session-db-recreation.md"


def _manual() -> str:
    """User manual with newline-wrapping collapsed so phrase asserts survive reflow."""
    return " ".join(USER_MANUAL.read_text(encoding="utf-8").split())


def test_user_manual_states_interaction_not_capability_distinction() -> None:
    manual = _manual()
    assert "differ in interaction, not in capability" in manual
    # Both surfaces resolve to one shared planner and one canonical draft.
    assert "talking to the **same** pipeline planner" in manual
    assert "produce the same canonical pipeline draft" in manual
    assert "the same canonical structures are available on both surfaces" in manual


def test_user_manual_lists_supported_canonical_structures() -> None:
    manual = _manual()
    assert "### Supported pipeline structures" in manual
    # The nine canonical classes the parity corpus verifies must be enumerated.
    for structure in (
        "Linear transform chains",
        "Conditional gates",
        "Multiple outputs",
        "Fork and coalesce",
        "Multi-source queue fan-in",
        "Batch aggregation",
        "Row expansion",
        "Error routing",
        "Structured LLM output consumed downstream",
    ):
        assert structure in manual, structure
    assert "same nine canonical classes the parity corpus verifies" in manual
    assert "none of them is freeform-only" in manual


def test_user_manual_explains_wrong_stage_retention_and_back_edit() -> None:
    manual = _manual()
    assert "Wrong-stage mentions are retained, not rejected" in manual
    assert "back/edit flow" in manual
    assert "typed rewind" in manual
    # The deferral is retained, not discarded / declared unsupported.
    assert "carries the intent forward" in manual


def test_user_manual_describes_tutorial_as_shared_planner_profile() -> None:
    manual = _manual()
    assert "guided workflow profile" in manual
    assert "same staged planner and the same proposal schema" in manual
    assert "not a separate or reduced-capability mode" in manual


def test_user_manual_honestly_notes_guided_staged_known_limitations() -> None:
    manual = _manual()
    # Both tracked guided-staged gaps must be named with their issue ids so the
    # honesty is auditable, and framed as defects rather than a capability wall.
    assert "elspeth-93dd908354" in manual  # require-all (union) coalesce
    assert "elspeth-b83b5b3204" in manual  # cross-sink on_write_failure fallback
    assert "require-all (union) coalesce" in manual
    assert "cross-sink `on_write_failure` fallback" in manual
    assert "seven of the nine canonical structures" in manual
    assert "not capability boundaries" in manual


def test_user_manual_rejects_switch_to_freeform_for_a_supported_topology() -> None:
    """Fail if the manual reintroduces categorical "guided can't; use freeform".

    Honest guided-STAGED bug remedies that route two named topologies to
    freeform are allowed; what is forbidden is framing guided-the-mode as
    lacking a whole class of supported topology and pointing users to freeform.
    """
    manual = _manual()
    forbidden = (
        "What guided mode does not cover",
        "Guided mode is intentionally narrow",
        "does not (yet) cover",
        "reaches a step it cannot represent",
        "does not match any of the patterns guided mode supports",
        "Pipelines with multiple sources.",
        "Pipelines with branching topologies",
        "exit to freeform when guided mode",
    )
    for phrase in forbidden:
        assert phrase not in manual, phrase


def test_runbook_uses_plan_05_epoch_and_schema_numbers() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    current_cutover = runbook.split("## Current Cutover:", maxsplit=1)[1].split("## Historical Cutover:", maxsplit=1)[0]

    # Plan 05 headline values: session epoch 35, guided schema 10, Landscape 28.
    assert "session epoch 35" in current_cutover
    assert "Landscape epoch 28" in current_cutover
    assert "guided schema 10" in runbook

    # The recreation/rollback record reference must name epoch-35, not the stale
    # epoch-30 the header-bump left behind (elspeth composer-parity fix).
    assert "session-epoch-35/Landscape-epoch-28 record" in current_cutover
    assert "repair the epoch-35 release forward" in current_cutover
    assert "epoch-30" not in current_cutover

    # The design doc's stale §6.1 pairing (guided schema 8 / session epoch 28)
    # must never be encoded as the CURRENT guided schema.
    assert "current guided schema 8" not in current_cutover.lower()
