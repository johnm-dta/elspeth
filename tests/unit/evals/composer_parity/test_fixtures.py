"""Loader/validator for the composer capability-parity fixture corpus.

Plan 05 Task 2. This is the TDD anchor for the corpus: every canonical fixture
must validate here before it is admitted. Two structural checks run per fixture,
matching the plan's declared Task 2 scope:

1. The ``canonical_arguments`` payload validates against
   ``SetPipelineArgumentsModel.model_validate``
   (``src/elspeth/web/composer/redaction.py``) — the same argument model the
   audited ``set_pipeline`` dispatch validates LLM-supplied arguments with.
2. Every plugin the payload references (sources, plugin-bearing nodes, and
   sinks) is available to a trained operator, checked through
   ``PolicyCatalogView.for_trained_operator`` and its availability lookups
   (``unavailable_reason``). This is what catches a fixture that names a plugin
   that is not installed (e.g. a typo, or a plugin that was removed).

Full committed-graph validation (``validate_composition_state``) is the costlier
alternative and is DEFERRED to Task 3's real-path matrix: the
args -> ``CompositionState`` conversion lives only inside the session-bound
``_execute_set_pipeline`` handler (``tools/sessions.py``), so reusing it here
would duplicate Task 3 and pull in session/engine wiring. Task 2 gates argument
shape + plugin availability; Task 3 gates the committed graph.

The two colour-scenario files are hashed against the pinned byte form recorded
in the eval ``README.md`` so the fixture corpus cannot silently drift.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.redaction import SetPipelineArgumentsModel
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId

# test_fixtures.py -> composer_parity -> evals -> unit -> tests -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_DIR = REPO_ROOT / "evals" / "composer-parity"
FIXTURES_DIR = CORPUS_DIR / "fixtures"
README = CORPUS_DIR / "README.md"

# The nine canonical capability classes (design §8.1). Exactly one JSON fixture
# per class; a glob miss or a stray/renamed fixture fails loudly here rather
# than passing vacuously.
EXPECTED_CLASSES = frozenset(
    {
        "linear_transform",
        "conditional_gate",
        "multi_output",
        "fork_coalesce",
        "multi_source_queue",
        "aggregation",
        "row_expansion",
        "error_routing",
        "structured_llm",
    }
)

COLOUR_FILES = ("two_llm_colour.csv", "two_llm_colour_request.txt")


def _json_fixture_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.json"))


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _referenced_plugins(args: SetPipelineArgumentsModel) -> set[PluginId]:
    """Return every (kind, plugin) identity the payload binds.

    Sources bind ``source`` plugins; sinks bind ``sink`` plugins. Plugin-bearing
    nodes (``transform`` and ``aggregation`` node types) bind ``transform``
    plugins — aggregation plugins are validated against the transform catalog by
    the real handler (``tools/sessions.py`` ``_validate_plugin_name(..., "transform", ...)``).
    Structural nodes (gate / coalesce / queue) carry ``plugin=None`` and bind
    nothing.
    """
    refs: set[PluginId] = set()
    if args.source is not None:
        refs.add(PluginId("source", args.source.plugin))
    if args.sources is not None:
        for named in args.sources.values():
            refs.add(PluginId("source", named.plugin))
    for node in args.nodes:
        if node.plugin is not None:
            refs.add(PluginId("transform", node.plugin))
    for output in args.outputs:
        refs.add(PluginId("sink", output.plugin))
    return refs


@pytest.fixture(scope="module")
def policy_view() -> PolicyCatalogView:
    """A trained-operator catalog projection over the real installed plugins.

    ``for_trained_operator`` is the explicit full-catalog projection: every
    installed plugin is available, so the availability check's real job is to
    reject a fixture that references a plugin that does not exist.
    """
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return PolicyCatalogView.for_trained_operator(catalog, snapshot)


def _readme_hashes() -> dict[str, str]:
    """Parse the two recorded SHA-256 values out of the eval README.

    The README is the single source of truth for the pinned hashes; the test
    reads them from there so the two cannot drift.
    """
    text = README.read_text(encoding="utf-8")
    hashes: dict[str, str] = {}
    for fname in COLOUR_FILES:
        match = re.search(re.escape(fname) + r"[^\n]*?([0-9a-f]{64})", text)
        assert match is not None, f"README records no SHA-256 for {fname}"
        hashes[fname] = match.group(1)
    return hashes


def test_corpus_has_exactly_nine_class_fixtures() -> None:
    """Exactly nine JSON fixtures, one per declared capability class."""
    paths = _json_fixture_paths()
    assert len(paths) == len(EXPECTED_CLASSES), f"expected {len(EXPECTED_CLASSES)} json fixtures, found {[p.name for p in paths]}"
    classes = {_load_fixture(path)["class"] for path in paths}
    assert classes == EXPECTED_CLASSES, f"fixture classes {sorted(classes)} != {sorted(EXPECTED_CLASSES)}"
    # Filename stem should match the declared class for discoverability.
    for path in paths:
        assert _load_fixture(path)["class"] == path.stem, f"{path.name} declares class {_load_fixture(path)['class']!r}"


@pytest.mark.parametrize("fixture_path", _json_fixture_paths(), ids=lambda p: p.stem)
def test_fixture_structure_and_arguments_validate(fixture_path: Path, policy_view: PolicyCatalogView) -> None:
    """Each fixture is well-formed and its canonical arguments validate."""
    fixture = _load_fixture(fixture_path)

    # Corpus record shape.
    assert isinstance(fixture.get("intent"), str) and fixture["intent"].strip(), "intent must be a non-empty string"
    assert isinstance(fixture.get("semantic_expectations"), dict) and fixture["semantic_expectations"], "semantic_expectations required"
    assert isinstance(fixture.get("runtime_assertions"), list) and fixture["runtime_assertions"], "runtime_assertions required"

    # (1) Structural validation against the canonical argument model.
    validated = SetPipelineArgumentsModel.model_validate(fixture["canonical_arguments"])

    # A genuine pipeline always declares at least one node and one output.
    assert validated.nodes, f"{fixture_path.stem}: canonical_arguments declares no nodes"
    assert validated.outputs, f"{fixture_path.stem}: canonical_arguments declares no outputs"
    assert validated.source is not None or validated.sources, f"{fixture_path.stem}: no source or sources declared"

    # (2) Plugin availability against the trained-operator projection.
    for plugin_id in _referenced_plugins(validated):
        reason = policy_view.unavailable_reason(plugin_id)
        assert reason is None, f"{fixture_path.stem}: plugin {plugin_id} unavailable ({reason})"


def test_colour_files_match_recorded_hashes() -> None:
    """The two colour-scenario files hash to the values recorded in README."""
    recorded = _readme_hashes()
    for fname in COLOUR_FILES:
        data = (FIXTURES_DIR / fname).read_bytes()
        assert b"\r" not in data, f"{fname}: contains CR; must be LF-only"
        assert not data.startswith(b"\xef\xbb\xbf"), f"{fname}: has a UTF-8 BOM"
        assert data.endswith(b"\n") and not data.endswith(b"\n\n"), f"{fname}: must end with exactly one trailing newline"
        digest = hashlib.sha256(data).hexdigest()
        assert digest == recorded[fname], f"{fname}: sha256 {digest} != README {recorded[fname]}"
