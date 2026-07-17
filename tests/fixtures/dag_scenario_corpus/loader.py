"""Strict loading and semantic validation for the maintained DAG scenario corpus."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import yaml

from tests.fixtures.dag_scenario_corpus.schema import (
    EXPECTED_DIMENSIONS,
    EXPECTED_SCENARIOS,
    EvidenceReference,
    HarnessCaseSpec,
    ScenarioManifest,
    ScenarioSpec,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "docs/architecture/dag/scenario-corpus/v1/manifest.yaml"
FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures/dag_scenario_corpus/v1"

_CRITERIA_REF = "docs/architecture/dag/completeness-criteria.md"
_DECISION_LOCATOR = re.compile(r"^elspeth-[0-9a-f]{10}$")
_HARNESS_LOCATOR = re.compile(r"^[a-z0-9][a-z0-9-]*:[A-Za-z0-9][A-Za-z0-9._-]*$")


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> ScenarioManifest:
    """Load the v1 manifest and fail closed on schema or semantic drift."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"DAG scenario manifest must be a YAML mapping: {path}")
    manifest = ScenarioManifest.model_validate(raw)
    _validate_exact_inventory(manifest)
    _validate_evidence_references(manifest)
    _validate_case_paths(manifest)
    return manifest


def iter_harness_cases(manifest: ScenarioManifest) -> tuple[tuple[ScenarioSpec, HarnessCaseSpec], ...]:
    """Return cases in authoritative scenario and per-scenario declaration order."""

    return tuple((scenario, case) for scenario in manifest.scenarios for case in scenario.cases)


def resolve_fixture_path(relative_path: str) -> Path:
    """Resolve one real, regular fixture file contained below ``FIXTURE_ROOT``."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Fixture escapes DAG scenario fixture root: {relative_path}")

    unresolved = FIXTURE_ROOT / relative
    cursor = FIXTURE_ROOT
    for component in relative.parts:
        if component in ("", "."):
            continue
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"DAG scenario fixture must not be a symlink: {relative_path}")

    fixture_root = FIXTURE_ROOT.resolve()
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(fixture_root)
    except ValueError as exc:
        raise ValueError(f"Fixture escapes DAG scenario fixture root: {relative_path}") from exc
    if not resolved.exists():
        raise ValueError(f"DAG scenario fixture does not exist: {relative_path}")
    if not resolved.is_file():
        raise ValueError(f"DAG scenario fixture must be a regular file: {relative_path}")
    return resolved


def _validate_exact_inventory(manifest: ScenarioManifest) -> None:
    if manifest.criteria_ref != _CRITERIA_REF:
        raise ValueError(f"DAG scenario criteria_ref must be {_CRITERIA_REF!r}")

    scenario_ids = tuple(scenario.id for scenario in manifest.scenarios)
    duplicate_ids = tuple(identifier for identifier, count in Counter(scenario_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"DAG scenario manifest contains duplicate scenario id(s): {', '.join(duplicate_ids)}")

    expected_ids = tuple(identifier for identifier, _title in EXPECTED_SCENARIOS)
    if scenario_ids != expected_ids:
        raise ValueError(f"DAG scenario IDs/order mismatch: expected {expected_ids!r}, got {scenario_ids!r}")

    for expected_ordinal, (scenario, (_expected_id, expected_title)) in enumerate(
        zip(manifest.scenarios, EXPECTED_SCENARIOS, strict=True),
        start=1,
    ):
        if scenario.ordinal != expected_ordinal:
            raise ValueError(f"DAG scenario ordinal mismatch for {scenario.id}: expected {expected_ordinal}, got {scenario.ordinal}")
        if scenario.title != expected_title:
            raise ValueError(f"DAG scenario title mismatch for {scenario.id}: expected {expected_title!r}, got {scenario.title!r}")
        dimension_keys = tuple(scenario.dimensions)
        if dimension_keys != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"DAG scenario dimension keys/order mismatch for {scenario.id}: expected {EXPECTED_DIMENSIONS!r}, got {dimension_keys!r}"
            )


def _validate_evidence_references(manifest: ScenarioManifest) -> None:
    evidence_ids = tuple(reference.id for reference in manifest.evidence)
    duplicate_evidence_ids = tuple(identifier for identifier, count in Counter(evidence_ids).items() if count > 1)
    if duplicate_evidence_ids:
        raise ValueError(f"DAG scenario manifest contains duplicate evidence id(s): {', '.join(duplicate_evidence_ids)}")

    evidence_by_id = {reference.id: reference for reference in manifest.evidence}
    for reference in manifest.evidence:
        _validate_evidence_locator(reference)

    registered_case_locators: list[str] = []
    for scenario in manifest.scenarios:
        case_ids = tuple(case.id for case in scenario.cases)
        duplicate_case_ids = tuple(identifier for identifier, count in Counter(case_ids).items() if count > 1)
        if duplicate_case_ids:
            duplicate_locators = ", ".join(f"{scenario.id}:{identifier}" for identifier in duplicate_case_ids)
            raise ValueError(f"DAG scenario manifest contains duplicate case id(s): {duplicate_locators}")
        registered_case_locators.extend(f"{scenario.id}:{case.id}" for case in scenario.cases)

    registered_cases = set(registered_case_locators)
    harness_locators = {reference.locator for reference in manifest.evidence if reference.kind == "harness"}
    unknown_harness_locators = sorted(harness_locators - registered_cases)
    if unknown_harness_locators:
        raise ValueError(f"DAG scenario manifest contains unknown harness locator(s): {', '.join(unknown_harness_locators)}")
    missing_harness_locators = sorted(registered_cases - harness_locators)
    if missing_harness_locators:
        raise ValueError("DAG scenario harness case(s) " + ", ".join(missing_harness_locators) + " lack a matching evidence locator")

    for scenario in manifest.scenarios:
        for dimension, cell in scenario.dimensions.items():
            unknown_ids = tuple(evidence_id for evidence_id in cell.evidence if evidence_id not in evidence_by_id)
            if unknown_ids:
                raise ValueError(f"DAG scenario {scenario.id}.{dimension} references unknown evidence id(s): {', '.join(unknown_ids)}")
            if cell.status == "pass" and not any(evidence_by_id[evidence_id].executable for evidence_id in cell.evidence):
                raise ValueError(f"DAG scenario pass cell {scenario.id}.{dimension} references only document/decision evidence")


def _validate_evidence_locator(reference: EvidenceReference) -> None:
    if reference.kind == "pytest":
        _validate_pytest_locator(reference.locator)
    elif reference.kind == "harness":
        if _HARNESS_LOCATOR.fullmatch(reference.locator) is None:
            raise ValueError(f"Invalid DAG scenario harness locator: {reference.locator}")
    elif reference.kind == "document":
        _validate_document_locator(reference.locator)
    elif _DECISION_LOCATOR.fullmatch(reference.locator) is None:
        raise ValueError(f"Invalid DAG scenario decision locator: {reference.locator}")


def _validate_pytest_locator(locator: str) -> None:
    parts = locator.split("::")
    relative_file = Path(parts[0])
    if (
        not parts[0].startswith("tests/")
        or relative_file.is_absolute()
        or relative_file.suffix != ".py"
        or ".." in relative_file.parts
        or any(character.isspace() for character in locator)
        or any(not selector for selector in parts[1:])
    ):
        raise ValueError(f"Pytest evidence must use a repository-relative pytest locator under tests/: {locator}")

    resolved_file = (REPOSITORY_ROOT / relative_file).resolve()
    try:
        resolved_file.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Pytest evidence locator escapes the repository: {locator}") from exc
    if not resolved_file.is_file():
        raise ValueError(f"pytest locator file does not exist: {locator}")
    if len(parts) == 1:
        return

    tree = ast.parse(resolved_file.read_text(encoding="utf-8"), filename=str(resolved_file))
    scope = tree.body
    for raw_selector in parts[1:]:
        selector = raw_selector.partition("[")[0]
        selected = next(
            (node for node in scope if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == selector),
            None,
        )
        if selected is None:
            raise ValueError(f"Pytest locator does not select pytest node {raw_selector!r}: {locator}")
        scope = selected.body if isinstance(selected, ast.ClassDef) else []


def _validate_document_locator(locator: str) -> None:
    relative_file = Path(locator)
    if not locator.startswith("docs/") or relative_file.is_absolute() or ".." in relative_file.parts:
        raise ValueError(f"Document evidence must use a repository-relative path under docs/: {locator}")
    resolved_file = (REPOSITORY_ROOT / relative_file).resolve()
    try:
        resolved_file.relative_to(REPOSITORY_ROOT / "docs")
    except ValueError as exc:
        raise ValueError(f"Document evidence locator escapes docs/: {locator}") from exc
    if not resolved_file.is_file():
        raise ValueError(f"Document evidence locator does not exist: {locator}")


def _validate_case_paths(manifest: ScenarioManifest) -> None:
    for _scenario, case in iter_harness_cases(manifest):
        resolve_fixture_path(case.fixture)
        resolve_fixture_path(case.input_fixture)
