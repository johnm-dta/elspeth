# Maintained DAG Scenario Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the versioned fifteen-scenario DAG manifest, strict common evidence schema, and production-path config-to-runtime-to-audit/recovery harness approved in the design.

**Architecture:** Keep the authoritative scenario inventory under the permanent DAG documentation area and the executable fixtures/support code under `tests/fixtures/dag_scenario_corpus/`. Strict Pydantic models fail closed on manifest drift; ordinary unit tests gate the complete inventory, while table-driven integration tests execute registered cases through the real config loader, runtime plugin factory, graph builder, orchestrator, Landscape audit exporter, file-backed database reopen, and public resume API.

**Tech Stack:** Python 3.12/3.13, Pydantic v2, PyYAML, pytest, SQLAlchemy, Elspeth production configuration/DAG/orchestrator/checkpoint/audit APIs, Markdown.

---

## File map

| File | Responsibility |
|---|---|
| `docs/architecture/dag/scenario-corpus/README.md` | Evergreen operating instructions and evidence-promotion rules. |
| `docs/architecture/dag/scenario-corpus/v1/manifest.yaml` | Authoritative closed inventory of scenarios, dimensions, evidence, statuses, fixtures, owners, and exit gates. |
| `docs/architecture/dag/README.md` | Links the permanent hub to the live corpus. |
| `tests/fixtures/dag_scenario_corpus/schema.py` | Strict manifest models plus immutable observed-run evidence models. |
| `tests/fixtures/dag_scenario_corpus/loader.py` | Repository-root resolution, safe YAML loading, reference validation, fixture containment, and executable-case selection. |
| `tests/fixtures/dag_scenario_corpus/plugins.py` | Built-in-plus-fixture plugin registry and deterministic fail-once EOF aggregation transform. |
| `tests/fixtures/dag_scenario_corpus/harness.py` | Fixture rendering, production build, run/audit, file-backed reopen, resume, and evidence collection. |
| `tests/fixtures/dag_scenario_corpus/v1/linear/happy-path.yaml` | Built-in CSV -> passthrough -> JSONL production fixture. |
| `tests/fixtures/dag_scenario_corpus/v1/linear/input.csv` | Stable three-row happy-path input. |
| `tests/fixtures/dag_scenario_corpus/v1/checkpoint-deterministic-resume/reopen-resume.yaml` | CSV -> fail-once EOF aggregation -> JSONL recovery fixture. |
| `tests/fixtures/dag_scenario_corpus/v1/checkpoint-deterministic-resume/input.csv` | Stable three-row recovery input. |
| `tests/unit/architecture/test_dag_scenario_corpus_contract.py` | Schema negatives, exact-set gate, reference/fixture parity, and derived-verdict tests. |
| `tests/integration/core/dag/test_dag_scenario_production_path.py` | Table-driven executable production and recovery cases. |

## Normative v1 inventory

The manifest must reproduce the current assessment status matrix exactly when
first introduced. `P`, `△`, `F`, `U`, and `N/A` map to `pass`, `partial`,
`fail`, `unknown`, and `not_applicable`.

| Scenario ID | Config | Build | Contracts | Runtime | Audit | Recovery | Concurrency | Freeform | Guided | Round-trip | Scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `linear` | P | P | P | △ | △ | △ | U | P | △ | △ | △ |
| `multiple-independent-sources` | P | P | P | △ | △ | △ | U | P | F | △ | U |
| `multi-source-queue-fan-in` | P | P | P | △ | △ | U | U | P | F | △ | U |
| `conditional-routing` | P | P | P | △ | △ | U | U | P | F | △ | U |
| `fork-multiple-terminals-partial-failure` | P | P | P | △ | △ | U | U | P | F | U | U |
| `fork-coalesce-policies` | P | P | △ | △ | △ | △ | △ | P | F | △ | U |
| `sequential-nested-fork-coalesce` | P | P | △ | U | U | U | U | P | F | U | U |
| `parallel-coalesces` | P | △ | △ | U | U | U | U | P | F | U | U |
| `aggregation-immutable-batch` | P | P | △ | △ | △ | △ | U | P | F | U | U |
| `row-expansion-parent-child-recovery` | P | P | △ | △ | △ | F | U | P | F | U | U |
| `row-union-interleave` | F | F | F | F | N/A | N/A | N/A | F | F | N/A | N/A |
| `retry-quarantine-discard-routed-errors` | P | P | △ | △ | △ | U | U | P | F | △ | U |
| `sink-write-pending-redrive` | P | P | P | △ | △ | △ | △ | P | △ | △ | U |
| `checkpoint-deterministic-resume` | P | P | △ | △ | △ | △ | U | N/A | N/A | N/A | U |
| `multi-worker-lease-reclaim-late-completion` | N/A | P | △ | △ | △ | △ | △ | N/A | N/A | N/A | U |

Use the exact titles and ordinals from
`docs/architecture/dag/completeness-criteria.md:105-124`. Known failures use
their specific owner (`elspeth-a25e9c009e` for expansion recovery and
`elspeth-a5b86149d4` for row union); guided gaps use
`elspeth-7e2dd67275`; browser/authoring evidence uses
`elspeth-7cf763da7c`; remaining evidence gaps use the corpus owner
`elspeth-ef29ef6ba4`. Every actionable non-pass cell gets an observable exit
gate phrased as a concrete passing corpus assertion. Narrow `not_applicable`
cells use the reasons already recorded at
`docs/architecture/dag/assessments/2026-07-17-1739/02-scorecard-and-scenario-matrix.md:77-82`.

### Task 1: Define the strict manifest and observed-evidence models

**Files:**

- Create: `tests/fixtures/dag_scenario_corpus/__init__.py`
- Create: `tests/fixtures/dag_scenario_corpus/schema.py`
- Create: `tests/unit/architecture/test_dag_scenario_corpus_contract.py`

- [ ] **Step 1: Write failing model-contract tests**

Add tests that instantiate the models directly and pin the cross-field rules:

```python
from pydantic import ValidationError

from tests.fixtures.dag_scenario_corpus.schema import EvidenceCell


def test_pass_cell_requires_executable_evidence() -> None:
    with pytest.raises(ValidationError, match="pass.*executable evidence"):
        EvidenceCell.model_validate({"status": "pass", "evidence": []})


@pytest.mark.parametrize("status", ["partial", "fail", "unknown"])
def test_actionable_non_pass_requires_owned_exit_gate(status: str) -> None:
    with pytest.raises(ValidationError, match="reason.*owner_issue.*exit_gate"):
        EvidenceCell.model_validate({"status": status})


def test_not_applicable_requires_reason_and_rejects_owner() -> None:
    with pytest.raises(ValidationError, match="applicability reason"):
        EvidenceCell.model_validate({"status": "not_applicable"})
    with pytest.raises(ValidationError, match="must not carry an owner"):
        EvidenceCell.model_validate(
            {
                "status": "not_applicable",
                "reason": "Worker multiplicity is runtime configuration.",
                "owner_issue": "elspeth-ef29ef6ba4",
            }
        )
```

- [ ] **Step 2: Run the model tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/unit/architecture/test_dag_scenario_corpus_contract.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'tests.fixtures.dag_scenario_corpus'`.

- [ ] **Step 3: Implement closed schema types**

Implement these exact public types in `schema.py`:

```python
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
IssueId = Annotated[str, StringConstraints(strict=True, pattern=r"^elspeth-[0-9a-f]{10}$")]
CellStatus = Literal["pass", "partial", "fail", "unknown", "not_applicable"]
Dimension = Literal[
    "config", "build", "contracts", "runtime", "audit", "recovery",
    "concurrency", "freeform", "guided", "round_trip", "scale",
]
EvidenceKind = Literal["harness", "pytest", "document", "decision"]
Stage = Literal["config", "build", "runtime", "audit", "recovery"]
Workflow = Literal["run", "recovery"]

EXPECTED_DIMENSIONS: tuple[Dimension, ...] = (
    "config", "build", "contracts", "runtime", "audit", "recovery",
    "concurrency", "freeform", "guided", "round_trip", "scale",
)
EXPECTED_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("linear", "Linear source → transform → sink"),
    ("multiple-independent-sources", "Multiple independent sources"),
    ("multi-source-queue-fan-in", "Multi-source queue fan-in"),
    ("conditional-routing", "Conditional routing, including missing and error destinations"),
    ("fork-multiple-terminals-partial-failure", "Fork to multiple terminals with partial failure"),
    ("fork-coalesce-policies", "Fork and coalesce across every completion policy and merge strategy"),
    ("sequential-nested-fork-coalesce", "Sequential or nested forks and coalesces"),
    ("parallel-coalesces", "Parallel coalesces"),
    ("aggregation-immutable-batch", "Aggregation, batch closure, and immutable membership"),
    ("row-expansion-parent-child-recovery", "Row expansion with parent/child identity and recovery"),
    ("row-union-interleave", "Row union or interleave, whether supported or consistently rejected"),
    ("retry-quarantine-discard-routed-errors", "Retry, quarantine, discard, and routed error handling"),
    ("sink-write-pending-redrive", "Sink write and pending-sink redrive"),
    ("checkpoint-deterministic-resume", "Checkpoint and deterministic resume"),
    ("multi-worker-lease-reclaim-late-completion", "Multi-worker execution, lease expiry, reclaim, and late completion"),
)


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceReference(ClosedModel):
    id: NonEmpty
    kind: EvidenceKind
    locator: NonEmpty
    claim: NonEmpty
    stages: tuple[Stage, ...] = ()

    @property
    def executable(self) -> bool:
        return self.kind in {"harness", "pytest"}


class EvidenceCell(ClosedModel):
    status: CellStatus
    evidence: tuple[NonEmpty, ...] = ()
    reason: NonEmpty | None = None
    owner_issue: IssueId | None = None
    exit_gate: NonEmpty | None = None

    @model_validator(mode="after")
    def validate_status_shape(self) -> "EvidenceCell":
        if self.status == "pass":
            if not self.evidence:
                raise ValueError("pass cell requires executable evidence")
            if any(value is not None for value in (self.reason, self.owner_issue, self.exit_gate)):
                raise ValueError("pass cell must not carry gap metadata")
        elif self.status in {"partial", "fail", "unknown"}:
            if None in (self.reason, self.owner_issue, self.exit_gate):
                raise ValueError("actionable non-pass requires reason, owner_issue, and exit_gate")
        else:
            if self.reason is None:
                raise ValueError("not_applicable cell requires an applicability reason")
            if self.owner_issue is not None or self.exit_gate is not None:
                raise ValueError("not_applicable cell must not carry an owner or exit gate")
            if self.evidence:
                raise ValueError("not_applicable cell must not carry evidence")
        return self


class RunExpectation(ClosedModel):
    status: Literal["completed", "completed_with_failures", "empty"]
    output_rows: int = Field(strict=True, ge=0)
    required_audit_record_types: tuple[NonEmpty, ...]


class HarnessCaseSpec(ClosedModel):
    id: NonEmpty
    workflow: Workflow
    fixture: NonEmpty
    input_fixture: NonEmpty
    expected: RunExpectation


class ScenarioSpec(ClosedModel):
    id: NonEmpty
    ordinal: int = Field(strict=True, ge=1, le=15)
    title: NonEmpty
    cases: tuple[HarnessCaseSpec, ...] = ()
    dimensions: dict[Dimension, EvidenceCell]


class ScenarioManifest(ClosedModel):
    schema_version: Literal[1]
    criteria_ref: NonEmpty
    evidence: tuple[EvidenceReference, ...]
    scenarios: tuple[ScenarioSpec, ...]

    @property
    def verdict(self) -> Literal["complete", "not_complete"]:
        return "complete" if all(
            cell.status in {"pass", "not_applicable"}
            for scenario in self.scenarios
            for cell in scenario.dimensions.values()
        ) else "not_complete"
```

Define the observed evidence models in the same file with these exact fields:

```python
Count = Annotated[int, Field(strict=True, ge=0)]


class ConfigEvidence(ClosedModel):
    loaded: bool = Field(strict=True)
    settings_sha256: NonEmpty


class GraphEvidence(ClosedModel):
    accepted: bool = Field(strict=True)
    node_count: Count | None = None
    edge_count: Count | None = None
    topology_hash: NonEmpty | None = None
    rejection_type: NonEmpty | None = None
    rejection_message: NonEmpty | None = None

    @model_validator(mode="after")
    def validate_build_shape(self) -> "GraphEvidence":
        built = (self.node_count, self.edge_count, self.topology_hash)
        rejected = (self.rejection_type, self.rejection_message)
        if self.accepted and (None in built or any(value is not None for value in rejected)):
            raise ValueError("accepted graph requires counts/hash and forbids rejection evidence")
        if not self.accepted and (any(value is not None for value in built) or None in rejected):
            raise ValueError("rejected graph requires rejection evidence and forbids graph facts")
        return self


class RuntimeEvidence(ClosedModel):
    attempted: bool = Field(strict=True)
    run_id: NonEmpty | None = None
    status: NonEmpty | None = None
    rows_processed: Count = 0
    rows_succeeded: Count = 0
    rows_failed: Count = 0
    output_rows: Count = 0

    @model_validator(mode="after")
    def validate_runtime_shape(self) -> "RuntimeEvidence":
        if self.attempted and (self.run_id is None or self.status is None):
            raise ValueError("attempted runtime requires run_id and status")
        if not self.attempted and (self.run_id is not None or self.status is not None):
            raise ValueError("unattempted runtime must not carry run_id or status")
        if not self.attempted and any((self.rows_processed, self.rows_succeeded, self.rows_failed, self.output_rows)):
            raise ValueError("unattempted runtime must not carry counters")
        return self


class AuditRecordCount(ClosedModel):
    record_type: NonEmpty
    count: Count


class AuditEvidence(ClosedModel):
    attempted: bool = Field(strict=True)
    total_records: Count
    record_counts: tuple[AuditRecordCount, ...]
    source_operation_count: Count

    @model_validator(mode="after")
    def validate_audit_shape(self) -> "AuditEvidence":
        if not self.attempted and (self.total_records or self.record_counts or self.source_operation_count):
            raise ValueError("unattempted audit must not carry records")
        return self


class RecoveryEvidence(ClosedModel):
    attempted: bool = Field(strict=True)
    database_reopened: bool = Field(strict=True)
    checkpoint_id: NonEmpty | None = None
    checkpoint_sequence: Count | None = None
    can_resume: bool = Field(strict=True)
    source_replayed: bool = Field(strict=True)
    checkpoint_removed: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_recovery_shape(self) -> "RecoveryEvidence":
        if self.attempted and (self.checkpoint_id is None or self.checkpoint_sequence is None):
            raise ValueError("attempted recovery requires checkpoint identity")
        if not self.attempted and (self.checkpoint_id is not None or self.checkpoint_sequence is not None):
            raise ValueError("unattempted recovery must not carry checkpoint identity")
        if not self.attempted and any((self.database_reopened, self.can_resume, self.source_replayed, self.checkpoint_removed)):
            raise ValueError("unattempted recovery must not carry recovery results")
        return self


class ScenarioRunEvidence(ClosedModel):
    schema_version: Literal[1]
    scenario_id: NonEmpty
    case_id: NonEmpty
    fixture_sha256: NonEmpty
    config: ConfigEvidence
    graph: GraphEvidence
    runtime: RuntimeEvidence
    audit: AuditEvidence
    recovery: RecoveryEvidence
    completed_stages: tuple[Stage, ...]
```

- [ ] **Step 4: Run model tests and verify GREEN**

Run the Task 1 pytest command. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the schema contract**

```bash
git add tests/fixtures/dag_scenario_corpus/__init__.py \
  tests/fixtures/dag_scenario_corpus/schema.py \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py
git commit -m "test: define DAG scenario evidence schema"
```

### Task 2: Add the authoritative manifest and strict loader

**Files:**

- Create: `docs/architecture/dag/scenario-corpus/v1/manifest.yaml`
- Create: `tests/fixtures/dag_scenario_corpus/loader.py`
- Modify: `tests/unit/architecture/test_dag_scenario_corpus_contract.py`

- [ ] **Step 1: Write failing exact-inventory and corruption tests**

Import `EXPECTED_SCENARIOS` and `EXPECTED_DIMENSIONS` from `schema.py`. Add
tests that call `load_manifest()` and
assert exact ID/ordinal/title order, exact dimension keys, verdict
`not_complete`, and the status matrix above. Add temporary-file negatives for:

```python
def valid_manifest_dict() -> dict[str, object]:
    loaded = yaml.safe_load(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def write_manifest(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_manifest_rejects_unknown_evidence_reference(tmp_path: Path) -> None:
    raw = valid_manifest_dict()
    raw["scenarios"][0]["dimensions"]["config"]["evidence"] = ["missing-id"]
    path = write_manifest(tmp_path, raw)
    with pytest.raises(ValueError, match="unknown evidence id.*missing-id"):
        load_manifest(path)


def test_manifest_rejects_fixture_escape(tmp_path: Path) -> None:
    raw = valid_manifest_dict()
    raw["evidence"].append(
        {
            "id": "linear:escape",
            "kind": "harness",
            "locator": "linear:escape",
            "claim": "Containment-negative fixture",
            "stages": ["config"],
        }
    )
    raw["scenarios"][0]["cases"] = [
        {
            "id": "escape",
            "workflow": "run",
            "fixture": "../../outside.yaml",
            "input_fixture": "linear/input.csv",
            "expected": {
                "status": "completed",
                "output_rows": 0,
                "required_audit_record_types": ["run"],
            },
        }
    ]
    path = write_manifest(tmp_path, raw)
    with pytest.raises(ValueError, match="escapes DAG scenario fixture root"):
        load_manifest(path)
```

- [ ] **Step 2: Run loader tests and verify RED**

Run the Task 1 pytest command. Expected: failure because `loader.py` and the manifest do not exist.

- [ ] **Step 3: Implement loader and semantic validation**

Implement:

```python
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "docs/architecture/dag/scenario-corpus/v1/manifest.yaml"
FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures/dag_scenario_corpus/v1"


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> ScenarioManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"DAG scenario manifest must be a YAML mapping: {path}")
    manifest = ScenarioManifest.model_validate(raw)
    _validate_exact_inventory(manifest)
    _validate_evidence_references(manifest)
    _validate_case_paths(manifest)
    return manifest


def iter_harness_cases(manifest: ScenarioManifest) -> tuple[tuple[ScenarioSpec, HarnessCaseSpec], ...]:
    return tuple((scenario, case) for scenario in manifest.scenarios for case in scenario.cases)


def resolve_fixture_path(relative_path: str) -> Path:
    unresolved = FIXTURE_ROOT / relative_path
    if unresolved.is_symlink():
        raise ValueError(f"DAG scenario fixture must not be a symlink: {relative_path}")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(FIXTURE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Fixture escapes DAG scenario fixture root: {relative_path}") from exc
    if not resolved.is_file():
        raise ValueError(f"DAG scenario fixture does not exist: {relative_path}")
    return resolved
```

`_validate_exact_inventory` compares IDs, ordinals, titles, and dimension sets
against immutable tuples in `schema.py`. `_validate_evidence_references`
rejects duplicate evidence IDs, unknown cell references, pass cells whose
references are all documentary/decision evidence, duplicate case IDs, unknown
`harness` locators, and harness cases without a matching evidence locator of
`<scenario-id>:<case-id>`. `_validate_case_paths` resolves fixture and input
paths under `FIXTURE_ROOT` and rejects missing files, directories, symlinks,
and containment escapes.

- [ ] **Step 4: Create the complete v1 manifest**

Write all fifteen scenarios and all 165 dimension cells using the normative
matrix in this plan. Define reusable evidence IDs for the exact executed suites
in the current assessment, including:

- `core-builder-schema-plural-sources` -> the exact pytest selections under
  “Core builder, schema, and plural sources”;
- `yaml-importer-generator` -> both composer YAML modules;
- `composer-runtime-agreement` -> the exact integration selections;
- `cardinality-identity` -> the exact cardinality pytest selections;
- `runtime-disposition-drains`, `focused-crash-restart`, and
  `direct-contention-fencing` -> their exact reassessment selections.

Use exact repository-relative pytest locators, not prose command names. A pass
cell must reference at least one of these executable records. Do not promote
the assessment statuses merely because a new harness case exists: the new
cases strengthen partial cells but do not prove every production-support arm.
Task 2 intentionally leaves `cases` empty. Task 3 adds the two harness cases
and their evidence records atomically with the fixture files, so path and
registration validation never observes a half-created corpus.

- [ ] **Step 5: Run loader tests and verify GREEN**

Run the Task 1 pytest command. Expected: exact inventory, negative validation, and verdict tests all pass.

- [ ] **Step 6: Commit the manifest and loader**

```bash
git add docs/architecture/dag/scenario-corpus/v1/manifest.yaml \
  tests/fixtures/dag_scenario_corpus/loader.py \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py
git commit -m "test: add maintained DAG scenario manifest"
```

### Task 3: Add shared fixtures and deterministic recovery plugin

**Files:**

- Modify: `docs/architecture/dag/scenario-corpus/v1/manifest.yaml`
- Create: `tests/fixtures/dag_scenario_corpus/plugins.py`
- Create: `tests/fixtures/dag_scenario_corpus/v1/linear/happy-path.yaml`
- Create: `tests/fixtures/dag_scenario_corpus/v1/linear/input.csv`
- Create: `tests/fixtures/dag_scenario_corpus/v1/checkpoint-deterministic-resume/reopen-resume.yaml`
- Create: `tests/fixtures/dag_scenario_corpus/v1/checkpoint-deterministic-resume/input.csv`
- Modify: `tests/unit/architecture/test_dag_scenario_corpus_contract.py`

- [ ] **Step 1: Write failing fixture/registry tests**

Pin that `make_corpus_plugin_manager()` includes built-in `csv`, `passthrough`,
and `json` plus `dag_corpus_fail_once_eof_batch`. Instantiate the custom
transform with a temporary marker, call it with a list of three typed
`PipelineRow`s, assert the first call raises `RuntimeError("injected DAG corpus
EOF flush crash")` and creates the marker, then construct a fresh transform
and assert it emits one row with the sum and count.

- [ ] **Step 2: Run fixture tests and verify RED**

Run the Task 1 pytest command. Expected: import failure for `plugins.py`.

- [ ] **Step 3: Implement the deterministic plugin registry**

`CorpusFailOnceEOFBatchTransform` subclasses `BaseTransform`, explicitly
declares `determinism = Determinism.DETERMINISTIC`, `is_batch_aware = True`,
typed input/output schemas, and `name = "dag_corpus_fail_once_eof_batch"`.
Its config contains `fault_marker_path`; on the first list-shaped `process`
call it atomically creates that file and raises the exact injected error. A
fresh instance sees the marker and returns `TransformResult.success()` with
`{"value": sum, "count": len(rows)}` and an observed locked contract. Scalar
calls return success unchanged so the aggregation processor can buffer them.

`make_corpus_plugin_manager()` creates `PluginManager()`, calls
`register_builtin_plugins()`, then registers the custom transform through
`create_dynamic_hookimpl`. `install_corpus_plugin_manager(monkeypatch)` patches
`elspeth.plugins.infrastructure.manager.get_shared_plugin_manager` to return
that complete manager.

Use this implementation shape:

```python
class CorpusInputSchema(PluginSchema):
    id: int
    value: int


class CorpusOutputSchema(PluginSchema):
    value: int
    count: int


class CorpusFailOnceEOFBatchTransform(BaseTransform):
    name = "dag_corpus_fail_once_eof_batch"
    determinism = Determinism.DETERMINISTIC
    input_schema = CorpusInputSchema
    output_schema = CorpusOutputSchema
    is_batch_aware = True
    on_error = "discard"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        marker = config.get("fault_marker_path")
        if not isinstance(marker, str) or not marker:
            raise ValueError("fault_marker_path must be a non-empty string")
        self._fault_marker = Path(marker)

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        del ctx
        if not isinstance(row, list):
            return TransformResult.success(row, success_reason={"action": "buffer"})
        self._fault_marker.parent.mkdir(parents=True, exist_ok=True)
        if not self._fault_marker.exists():
            try:
                self._fault_marker.touch(exist_ok=False)
            except FileExistsError:
                pass
            else:
                raise RuntimeError("injected DAG corpus EOF flush crash")
        total = sum(int(member["value"]) for member in row)
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(normalized_name="value", original_name="value", python_type=int, required=False, source="inferred"),
                FieldContract(normalized_name="count", original_name="count", python_type=int, required=False, source="inferred"),
            ),
            locked=True,
        )
        return TransformResult.success(
            PipelineRow({"value": total, "count": len(row)}, contract),
            success_reason={"action": "batch_sum"},
        )


def make_corpus_plugin_manager() -> PluginManager:
    manager = PluginManager()
    manager.register_builtin_plugins()
    manager.register(create_dynamic_hookimpl([CorpusFailOnceEOFBatchTransform], "elspeth_get_transforms"))
    return manager


def install_corpus_plugin_manager(monkeypatch: pytest.MonkeyPatch) -> PluginManager:
    manager = make_corpus_plugin_manager()
    monkeypatch.setattr(
        "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
        lambda: manager,
    )
    return manager
```

- [ ] **Step 4: Add exact YAML and CSV fixtures**

Happy path YAML uses `${input_csv}` and `${output_jsonl}` placeholders and the
built-in plugins:

```yaml
sources:
  primary:
    plugin: csv
    on_success: inbound
    options:
      path: ${input_csv}
      on_validation_failure: discard
      schema: {mode: fixed, fields: ["id: int", "value: int"]}
queues: {inbound: {}}
transforms:
  - name: pass_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: discard
    options: {schema: {mode: observed}}
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: ${output_jsonl}
      format: jsonl
      schema: {mode: observed}
```

Recovery YAML uses the same CSV/JSON endpoints and one aggregation named
`eof_sum`, plugin `dag_corpus_fail_once_eof_batch`, input `batch_in`, count
trigger 100, `output_mode: transform`, and option
`fault_marker_path: ${fault_marker}`. Both CSV files contain exactly:

```csv
id,value
1,10
2,20
3,30
```

In the same edit, add manifest cases `linear:happy-path` and
`checkpoint-deterministic-resume:reopen-resume`, plus `harness` evidence
records with stages `(config, build, runtime, audit)` and
`(config, build, runtime, audit, recovery)` respectively. Reference those
records from the applicable partial cells without changing their status.

- [ ] **Step 5: Run fixture tests and verify GREEN**

Run the Task 1 pytest command. Expected: plugin registry, fail-once behavior, and loader fixture-containment tests pass.

- [ ] **Step 6: Commit fixtures and plugin**

```bash
git add docs/architecture/dag/scenario-corpus/v1/manifest.yaml \
  tests/fixtures/dag_scenario_corpus/plugins.py \
  tests/fixtures/dag_scenario_corpus/v1 \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py
git commit -m "test: add DAG scenario production fixtures"
```

### Task 4: Implement config/build/runtime/audit harness

**Files:**

- Create: `tests/fixtures/dag_scenario_corpus/harness.py`
- Create: `tests/integration/core/dag/test_dag_scenario_production_path.py`

- [ ] **Step 1: Write the failing table-driven happy-path test**

```python
MANIFEST = load_manifest()
CASES = [
    pytest.param(scenario, case, id=f"{scenario.id}:{case.id}")
    for scenario, case in iter_harness_cases(MANIFEST)
    if case.workflow == "run"
]


@pytest.mark.parametrize(("scenario", "case"), CASES)
def test_declared_run_case_uses_complete_production_path(
    scenario: ScenarioSpec,
    case: HarnessCaseSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_corpus_plugin_manager(monkeypatch)
    evidence = run_scenario_case(scenario, case, tmp_path)
    assert evidence.completed_stages == ("config", "build", "runtime", "audit")
    assert evidence.runtime.status == case.expected.status
    assert evidence.runtime.output_rows == case.expected.output_rows
    observed_types = {record.record_type for record in evidence.audit.record_counts}
    assert set(case.expected.required_audit_record_types) <= observed_types
    assert evidence.recovery.attempted is False
```

- [ ] **Step 2: Run the happy-path row and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/integration/core/dag/test_dag_scenario_production_path.py -k happy
```

Expected: collection fails because `harness.py` does not exist.

- [ ] **Step 3: Implement safe rendering and production build**

Add frozen internal `RenderedScenario` and `BuiltScenario` dataclasses. Implement:

```python
@dataclass(frozen=True, slots=True)
class RenderedScenario:
    settings: ElspethSettings
    settings_yaml: str
    settings_sha256: str
    fixture_sha256: str
    output_path: Path
    fault_marker: Path


@dataclass(frozen=True, slots=True)
class BuiltScenario:
    rendered: RenderedScenario
    bundle: PluginBundle
    graph: ExecutionGraph
    config: PipelineConfig
    graph_evidence: GraphEvidence


def render_settings(case: HarnessCaseSpec, tmp_path: Path) -> RenderedScenario:
    fixture_path = resolve_fixture_path(case.fixture)
    input_path = resolve_fixture_path(case.input_fixture)
    output_path = tmp_path / "output.jsonl"
    fault_marker = tmp_path / "fault-triggered.marker"
    fixture_bytes = fixture_path.read_bytes()
    input_bytes = input_path.read_bytes()
    rendered = Template(fixture_bytes.decode("utf-8")).substitute(
        input_csv=str(input_path),
        output_jsonl=str(output_path),
        fault_marker=str(fault_marker),
    )
    if "${" in rendered:
        raise ValueError(f"Unresolved DAG scenario template variable in {fixture_path}")
    return RenderedScenario(
        settings=load_settings_from_yaml_string(rendered),
        settings_yaml=rendered,
        settings_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        fixture_sha256=hashlib.sha256(fixture_bytes + b"\0" + input_bytes).hexdigest(),
        output_path=output_path,
        fault_marker=fault_marker,
    )


def build_scenario(rendered: RenderedScenario) -> BuiltScenario:
    settings = rendered.settings
    bundle = instantiate_plugins_from_config(settings, preflight_mode=True)
    execution_sinks = execution_sinks_for_runtime(settings, bundle.sinks)
    graph = ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=execution_sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        coalesce_settings=list(settings.coalesce) if settings.coalesce else None,
        queues=settings.queues,
    )
    graph.validate()
    graph.validate_edge_compatibility()
    config = assemble_and_validate_pipeline_config(
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    graph_evidence = GraphEvidence(
        accepted=True,
        node_count=len(graph.get_nodes()),
        edge_count=len(graph.get_edges()),
        topology_hash=CheckpointCompatibilityValidator().compute_full_topology_hash(graph),
    )
    return BuiltScenario(rendered, bundle, graph, config, graph_evidence)
```

`Template.substitute()` rejects unknown variables and the explicit `${` check
rejects unresolved variables before loading. The build path calls, in order:

1. `instantiate_plugins_from_config(settings, preflight_mode=True)`;
2. `execution_sinks_for_runtime(settings, bundle.sinks)`;
3. `ExecutionGraph.from_plugin_instances(...)`, including queues and coalesces;
4. `graph.validate()` and `graph.validate_edge_compatibility()`; and
5. `assemble_and_validate_pipeline_config(...)` with the same bundle and graph.

Capture node count, edge count, and
`CheckpointCompatibilityValidator().compute_full_topology_hash(graph)` in
`GraphEvidence`.

- [ ] **Step 4: Implement run and audit evidence collection**

For run cases, create `LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")` and
`FilesystemPayloadStore(tmp_path / "payloads")`; call
`Orchestrator(db).run(config, graph=graph, settings=settings,
payload_store=store)`. Parse JSONL output and collect
`Counter(record["record_type"] for record in
LandscapeExporter(db).export_run(result.run_id))`. Build
`ScenarioRunEvidence` using `result.to_dict()`, the fixture SHA-256, output row
count, and exact audit counts. Close the DB in a `finally` block.

Use these helpers so recovery can reuse the same evidence projection:

```python
def _audit_evidence(records: list[dict[str, Any]]) -> AuditEvidence:
    counts = Counter(str(record["record_type"]) for record in records)
    return AuditEvidence(
        attempted=True,
        total_records=len(records),
        record_counts=tuple(
            AuditRecordCount(record_type=record_type, count=count)
            for record_type, count in sorted(counts.items())
        ),
        source_operation_count=sum(
            1 for record in records
            if record.get("record_type") == "operation" and record.get("operation_type") == "source_load"
        ),
    )


def _run_case(scenario: ScenarioSpec, case: HarnessCaseSpec, tmp_path: Path) -> ScenarioRunEvidence:
    rendered = render_settings(case, tmp_path)
    built = build_scenario(rendered)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        result = Orchestrator(db).run(
            built.config,
            graph=built.graph,
            settings=built.rendered.settings,
            payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
        )
        output_rows = len(rendered.output_path.read_text(encoding="utf-8").splitlines())
        audit = _audit_evidence(list(LandscapeExporter(db).export_run(result.run_id)))
        return ScenarioRunEvidence(
            schema_version=1,
            scenario_id=scenario.id,
            case_id=case.id,
            fixture_sha256=rendered.fixture_sha256,
            config=ConfigEvidence(loaded=True, settings_sha256=rendered.settings_sha256),
            graph=built.graph_evidence,
            runtime=RuntimeEvidence(
                attempted=True,
                run_id=result.run_id,
                status=result.status.value,
                rows_processed=result.rows_processed,
                rows_succeeded=result.rows_succeeded,
                rows_failed=result.rows_failed,
                output_rows=output_rows,
            ),
            audit=audit,
            recovery=RecoveryEvidence(
                attempted=False,
                database_reopened=False,
                can_resume=False,
                source_replayed=False,
                checkpoint_removed=False,
            ),
            completed_stages=("config", "build", "runtime", "audit"),
        )
    finally:
        db.close()


def run_scenario_case(scenario: ScenarioSpec, case: HarnessCaseSpec, tmp_path: Path) -> ScenarioRunEvidence:
    if case.workflow == "run":
        return _run_case(scenario, case, tmp_path)
    return _run_recovery_case(scenario, case, tmp_path)
```

- [ ] **Step 5: Run the happy-path row and verify GREEN**

Run the Task 4 pytest command. Expected: one parameterized happy-path case passes with config/build/runtime/audit evidence.

- [ ] **Step 6: Commit the production run harness**

```bash
git add tests/fixtures/dag_scenario_corpus/harness.py \
  tests/integration/core/dag/test_dag_scenario_production_path.py
git commit -m "test: add DAG production-path evidence harness"
```

### Task 5: Extend the harness through file-backed reopen and recovery

**Files:**

- Modify: `tests/fixtures/dag_scenario_corpus/harness.py`
- Modify: `tests/integration/core/dag/test_dag_scenario_production_path.py`

- [ ] **Step 1: Write the failing recovery-row test**

```python
RECOVERY_CASES = [
    pytest.param(scenario, case, id=f"{scenario.id}:{case.id}")
    for scenario, case in iter_harness_cases(MANIFEST)
    if case.workflow == "recovery"
]


@pytest.mark.parametrize(("scenario", "case"), RECOVERY_CASES)
def test_declared_recovery_case_reopens_and_resumes_publicly(
    scenario: ScenarioSpec,
    case: HarnessCaseSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_corpus_plugin_manager(monkeypatch)
    evidence = run_scenario_case(scenario, case, tmp_path)
    assert evidence.completed_stages == ("config", "build", "runtime", "audit", "recovery")
    assert evidence.recovery.attempted is True
    assert evidence.recovery.database_reopened is True
    assert evidence.recovery.can_resume is True
    assert evidence.recovery.source_replayed is False
    assert evidence.recovery.checkpoint_removed is True
    assert evidence.runtime.status == case.expected.status
    assert evidence.runtime.output_rows == case.expected.output_rows
```

- [ ] **Step 2: Run the recovery row and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/integration/core/dag/test_dag_scenario_production_path.py -k reopen_resume
```

Expected: failure because the recovery workflow is not implemented.

- [ ] **Step 3: Execute the deterministic EOF crash and capture pre-reopen evidence**

For `workflow == "recovery"`, construct `CheckpointManager`, derive
`RuntimeCheckpointConfig.from_settings(settings.checkpoint)`, and pass both to
`Orchestrator`. Assert `run()` raises the exact injected EOF error. Query the
single `run_id`, assert the run is failed, assert every `run_sources` lifecycle
is `exhausted`, and capture `CheckpointManager.get_latest_checkpoint(run_id)`.
Require a non-null checkpoint and preserve its ID, sequence, and topology hash.
Close the DB and discard all plugin/config/graph/orchestrator objects.

- [ ] **Step 4: Reopen, rebuild fresh objects, and resume publicly**

Reopen with:

```python
reopened_db = LandscapeDB.from_url(db_url, create_tables=False)
reopened_store = FilesystemPayloadStore(payload_root)
reopened_checkpoint_manager = CheckpointManager(reopened_db)
fresh_rendered = render_settings(case, tmp_path)
fresh = build_scenario(fresh_rendered)
recovery = RecoveryManager(reopened_db, reopened_checkpoint_manager)
check = recovery.can_resume(run_id, fresh.graph)
assert check.can_resume, check.reason
resume_point = recovery.get_resume_point(run_id, fresh.graph)
assert resume_point is not None
result = Orchestrator(
    reopened_db,
    checkpoint_manager=reopened_checkpoint_manager,
    checkpoint_config=checkpoint_config,
).resume(
    resume_point,
    fresh.config,
    fresh.graph,
    payload_store=reopened_store,
    settings=fresh.settings,
)
```

Assert the checkpoint ID before resume equals the reopened checkpoint, the
fault marker proves a fresh transform observed the prior fault, JSONL contains
exactly one `{value: 60, count: 3}` record, all exported token outcomes are
terminal, node-state attempts include resume evidence, and the latest
checkpoint is `None`. Record `source_replayed=False` from the fact that resume
uses persisted rows and no second source operation/load audit record appears.

- [ ] **Step 5: Run both table-driven cases and verify GREEN**

Run:

```bash
.venv/bin/pytest -q tests/integration/core/dag/test_dag_scenario_production_path.py
```

Expected: happy-path and reopen/resume cases both pass without skips or xfails.

- [ ] **Step 6: Commit recovery support**

```bash
git add tests/fixtures/dag_scenario_corpus/harness.py \
  tests/integration/core/dag/test_dag_scenario_production_path.py
git commit -m "test: prove DAG corpus audit recovery path"
```

### Task 6: Publish corpus operating documentation and hub navigation

**Files:**

- Create: `docs/architecture/dag/scenario-corpus/README.md`
- Modify: `docs/architecture/dag/README.md:40-78`
- Modify: `tests/unit/architecture/test_dag_scenario_corpus_contract.py`

- [ ] **Step 1: Write failing documentation-link tests**

Assert the DAG hub contains a repository-relative link to
`scenario-corpus/README.md`, the corpus README links `v1/manifest.yaml`, the
criteria, and the active Filigree issue, and every repository-relative path in
those two documents resolves from its containing directory.

- [ ] **Step 2: Run documentation contract tests and verify RED**

Run the Task 1 pytest command. Expected: missing corpus README/hub link failure.

- [ ] **Step 3: Write the operating README and update the hub**

Document the manifest/schema authority boundary, the exact status vocabulary,
why unknowns are not skipped, how executable evidence is registered, how to
promote a cell in the same commit as its passing evidence, how immutable dated
assessments cite the evergreen corpus, and the focused unit/integration
commands from this plan. Extend the root directory model with
`scenario-corpus/` and add the live corpus under “Start here”.

- [ ] **Step 4: Run documentation tests and verify GREEN**

Run the Task 1 pytest command. Expected: corpus and link tests pass.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/architecture/dag/README.md \
  docs/architecture/dag/scenario-corpus/README.md \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py
git commit -m "docs: publish DAG scenario corpus workflow"
```

### Task 7: Requirement-level verification and tracker handoff

**Files:**

- Verify all files listed in the file map.
- Update: Filigree issue `elspeth-ef29ef6ba4` by comment; close it only if its
  full issue description, not merely this foundation package, is satisfied.

- [ ] **Step 1: Run focused tests**

```bash
.venv/bin/pytest -q \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py \
  tests/integration/core/dag/test_dag_scenario_production_path.py
```

Expected: all tests pass, with exactly two executable harness cases collected and no skips/xfails.

- [ ] **Step 2: Run adjacent production-path and recovery regressions**

```bash
.venv/bin/pytest -q \
  tests/integration/test_multisource_provenance_proof.py \
  tests/integration/pipeline/test_eof_resume_proof.py \
  tests/e2e/recovery/test_crash_and_resume.py
```

Expected: all selected regressions pass.

- [ ] **Step 3: Run static and repository checks**

```bash
.venv/bin/ruff check tests/fixtures/dag_scenario_corpus \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py \
  tests/integration/core/dag/test_dag_scenario_production_path.py
.venv/bin/ruff format --check tests/fixtures/dag_scenario_corpus \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py \
  tests/integration/core/dag/test_dag_scenario_production_path.py
.venv/bin/mypy tests/fixtures/dag_scenario_corpus \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py \
  tests/integration/core/dag/test_dag_scenario_production_path.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 4: Audit the two active-goal requirements**

Verify from current files and test output that:

1. the authoritative manifest contains exactly fifteen scenarios, eleven cells
   per scenario, strict evidence references, owners/exit gates, and a derived
   verdict; and
2. executable cases demonstrably traverse config, production build/validation,
   runtime, Landscape audit export, file-backed close/reopen, and public
   recovery with fresh objects.

Treat missing or indirect evidence as incomplete and correct it before the
final commit.

- [ ] **Step 5: Commit any verification-only corrections**

```bash
git status --short
git add docs/architecture/dag docs/superpowers/plans/2026-07-17-dag-scenario-corpus.md \
  tests/fixtures/dag_scenario_corpus \
  tests/unit/architecture/test_dag_scenario_corpus_contract.py \
  tests/integration/core/dag/test_dag_scenario_production_path.py
git commit -m "test: finalize DAG scenario corpus foundation"
```

Skip this commit only when `git status --short` is empty after verification.

- [ ] **Step 6: Update Filigree with exact evidence**

Add a comment naming the branch, commits, focused/adjacent/static commands,
observed counts, the two executable cases, and the remaining non-pass corpus
cells. Keep `elspeth-ef29ef6ba4` in progress unless all applicable cells in its
full fifteen-scenario acceptance matrix are executable and passing.
