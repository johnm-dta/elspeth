# Sink Effect Exactly-Once Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reserve and reconcile one durable sink effect before publication so every primary, diversion, and failsink retry converges on one external result and one effect-linked artifact.

**Architecture:** Epoch 26 adds target streams, effect groups, ordered members, and intended external attempts. The executor reserves identity, performs an explicit read-only inspect, completes an immutable plan, acquires a generation-fenced lease, and invokes built-in sink adapters whose commit/reconcile operations are convergent. Replacing targets serialize through a predecessor chain; SQL uses a target-side marker transaction; record APIs persist member sub-effects.

**Tech Stack:** Python 3.13, SQLAlchemy 2, SQLite/SQLCipher, PostgreSQL 16 testcontainers, Pydantic, botocore/boto3, Azure Blob SDK, requests/Dataverse, ChromaDB, pytest, mypy strict, Ruff, pre-commit.

---

## Execution posture

This is one coupled safety change, not a set of independently shippable
subprojects: the executor cannot claim the boundary until the ledger,
artifact linkage, adapters, and recovery tests agree. Implementation stays in
`/home/john/elspeth/.worktrees/safety-74a343d5ad` on
`codex/safety-74a343d5ad`. Use `superpowers:executing-plans` inline because the
integration owner already selected same-worktree execution. Keep Filigree
`elspeth-74a343d5ad` open for independent review/integration.

Every production-code task below follows red-green-refactor and ends in a
small commit. Do not weaken a failing test to match current behavior. Do not
use Loomweave in this worktree.

## Integration prerequisites

- The integration branch is `codex/release-0.7.1-worktree`; its approved
  direct base for this implementation is currently `b84147e57`. That head
  includes `b33a26050`, batch
  membership, call-index recovery, and the 4003 outcome lock-order/atomicity
  work that epoch 26 must extend.
- Rebase on `b84147e57` before adding production code, and use
  `codex/release-0.7.1-worktree` (or a later exact integration HEAD supplied by
  the integration owner) for every subsequent rebase and changed-file diff;
  never rebase this work directly onto the older `release/0.7.1` branch;
  preserve commits `76117fb6b`, `4c3dae2ec`, `d9b324d64`, `37249411c`,
  `0373d6c31`, and `e0dca4dd4`.
- If the rebase changes the 4003 repository API or global order, update the
  approved design and this plan before implementation rather than guessing.

## File map

### New focused modules

- `src/elspeth/contracts/sink_effects.py` — closed enums and immutable protocol
  values shared by executor, persistence, and adapters.
- `src/elspeth/core/landscape/execution/sink_effect_identity.py` — bounded
  lineage resolution and deterministic stream/effect/artifact IDs.
- `src/elspeth/core/landscape/execution/sink_effect_reservation.py` — ordered
  member claim, overlap partitioning, stream tail allocation, and lock order.
- `src/elspeth/core/landscape/execution/sink_effect_lifecycle.py` — inspect
  evidence, immutable plan CAS, leases, attempts, response loss, and takeover.
- `src/elspeth/core/landscape/execution/sink_effect_finalization.py` — global
  token/state/effect/artifact/operation finalization transaction.
- `src/elspeth/core/landscape/execution/sink_effects.py` — narrow repository
  facade composing the three persistence units.
- `src/elspeth/engine/executors/sink_effects.py` — caller-level effect
  coordinator and recovery state machine.
- `src/elspeth/plugins/infrastructure/sink_effects.py` — capability validation,
  restricted contexts, plan/evidence validation, and shared adapter helpers.
- `src/elspeth/plugins/sinks/_local_file_effects.py` — bounded streamed staging,
  file identity, advisory lock, atomic replace, and reconciliation.
- `tests/fixtures/sink_effects.py` — effect-capable duplicate-observable fake and
  deterministic fault seams.

### Existing modules with bounded changes

- `src/elspeth/core/landscape/schema.py`, `database.py`, `factory.py`,
  `execution_repository.py`, `execution/__init__.py`, `model_loaders.py` —
  epoch 26, repository wiring, and loaders.
- `src/elspeth/contracts/audit.py`, `results.py`, `export_records.py`,
  `plugin_protocols.py`, `__init__.py` — artifact XOR and public effect API.
- `src/elspeth/core/landscape/execution/artifacts.py`, `operations.py`,
  `calls.py`, `exporter.py`, `export_mappers.py`, `reproducibility.py`,
  `query_repository.py` — effect-linked audit/export/read behavior.
- `src/elspeth/engine/executors/sink.py`,
  `engine/orchestrator/sink_flush.py`, `engine/orchestrator/preflight.py`,
  `plugins/infrastructure/base.py` — replace the write/flush/legacy boundary.
- `src/elspeth/plugins/sinks/{csv_sink,json_sink,text_sink,aws_s3_sink,azure_blob_sink,database_sink,dataverse,chroma_sink}.py`
  — first-party effect adapters.
- `src/elspeth/mcp/types.py`, MCP analyzer/query mapping, and web audit schemas —
  producer-kind and publication-evidence visibility.
- `docs/architecture/token-scheduler-state-engine.md`,
  `docs/contracts/plugin-protocol.md`, `docs/runbooks/`,
  `docs/operator/migrations/`, `docs/release/`, and
  `docs/reference/configuration.md` — operator and release contract.

## Task 1: Rebase and capture the approved baseline

**Files:**

- Verify: branch history and clean worktree
- Verify: 4003 outcome-composition repository and tests

- [x] **Step 1: Wait for the integration dependency and inspect it**

Run:

```bash
git fetch --all --prune
git log --oneline --decorate -20 codex/release-0.7.1-worktree
git merge-base --is-ancestor b84147e57 codex/release-0.7.1-worktree
git status --short
```

Expected: the integration log contains `b84147e57`, the ancestry command exits
0, and status is clean.

- [x] **Step 2: Rebase without dropping the existing safety commits**

Run:

```bash
git rebase b84147e57
git log --oneline --decorate -12
```

Expected: the seven listed commit subjects/content remain in order above
`b84147e57`, including the three approved design commits; rebased commit hashes
may change.

- [x] **Step 3: Run the dependency-sensitive baseline**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_artifact_idempotency_migration.py tests/unit/engine/test_sink_executor_diversion.py tests/unit/engine/orchestrator/test_pending_sink_grouping.py
.venv/bin/mypy src/elspeth/core/landscape/execution/artifacts.py src/elspeth/engine/executors/sink.py
git diff --check
```

Expected: all selected tests pass, mypy reports success, and the diff check is
clean. The known signed-tree failure is not part of this focused baseline.

## Task 1a: Prelock complete bulk-state sets in canonical order

This integration prerequisite closes the one structural gap found by the
Task 1 lock-order audit. It must be complete, committed, and independently
reviewed before Task 2 begins.

**Files:**

- Modify: `src/elspeth/core/landscape/execution/node_states.py`
- Test: `tests/unit/core/landscape/test_execution_repository.py`
- Test: `tests/testcontainer/core/test_token_outcome_atomicity_postgres.py`

- [x] **Step 1: Write failing sorted/deduplicated prelock tests**

Add a unit test that passes reversed completions with a duplicate state ID,
captures the lock acquisitions, and proves the complete unique set is locked
in ascending `state_id` order before the first pre-read or update. Exercise
both caller-owned and repository-owned transaction paths.

Add a real PostgreSQL test using two distinct engines/connections and reversed
caller order. Install a deterministic seam immediately after each contender's
first state lock; hold the first contender there until the second is waiting,
then release both. Assert captured ascending acquisition order, bounded
completion, no `40P01`, and exact terminal rows.

- [x] **Step 2: Run the tests and prove the structural gap**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_execution_repository.py -k 'complete_node_states_completed_many and prelock'
.venv/bin/pytest -q -m testcontainer tests/testcontainer/core/test_token_outcome_atomicity_postgres.py -k 'bulk_state and lock_order'
```

Expected: FAIL because `complete_node_states_completed_many()` currently
performs unlocked pre-reads and caller-ordered executemany updates.

- [x] **Step 3: Implement the explicit state prelock primitive**

Before the existing before-row reads, deduplicate the complete state ID set,
sort it ascending, and acquire every state row explicitly with `FOR UPDATE` in
that order. Missing rows are still reported through the existing error
taxonomy. The primitive must run inside the supplied connection when
`conn is not None` and inside the method's owned write transaction otherwise.
Caller-owned primary-sink composition therefore remains sorted token -> sorted
state; repository-owned calls legitimately begin at the state class. Do not
use caller, driver/executemany, or planner order as the contract.

- [x] **Step 4: Run the unit and composed PostgreSQL proofs**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_execution_repository.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/core/test_token_outcome_atomicity_postgres.py
.venv/bin/mypy src/elspeth/core/landscape/execution/node_states.py
git diff --check
```

Expected: unit behavior remains green; the complete existing 4003 PostgreSQL
composition suite and new deterministic reversed-order proof pass.

- [x] **Step 5: Commit and review the prerequisite**

```bash
git add src/elspeth/core/landscape/execution/node_states.py tests/unit/core/landscape/test_execution_repository.py tests/testcontainer/core/test_token_outcome_atomicity_postgres.py
git commit -m "fix(landscape): prelock bulk state completion"
```

Request independent spec review, then independent quality review. Any
blocking finding must be fixed and re-reviewed before Task 2.

## Task 2: Add closed sink-effect contracts and fail-closed capability preflight

**Files:**

- Create: `src/elspeth/contracts/sink_effects.py`
- Modify: `src/elspeth/contracts/plugin_protocols.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/plugins/infrastructure/base.py`
- Modify: `src/elspeth/engine/orchestrator/preflight.py`
- Modify: `src/elspeth/engine/orchestrator/run_context_factory.py`
- Test: `tests/unit/contracts/test_sink_effect_contract.py`
- Test: `tests/unit/engine/test_sink_effect_preflight.py`

- [ ] **Step 1: Write contract-shape and closed-vocabulary tests**

Add tests with these assertions:

```python
def test_reconcile_result_is_closed_and_exact_descriptor_is_required() -> None:
    exact = SinkEffectReconcileResult.applied(EXACT_DESCRIPTOR, evidence=SAFE_EVIDENCE)
    assert exact.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert exact.descriptor == EXACT_DESCRIPTOR
    with pytest.raises(ValueError, match="descriptor"):
        SinkEffectReconcileResult(kind=SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR)


def test_unknown_cannot_carry_permission_to_commit() -> None:
    result = SinkEffectReconcileResult.unknown(evidence=SAFE_EVIDENCE)
    assert result.kind is SinkEffectReconcileKind.UNKNOWN
    assert result.may_commit is False


def test_preflight_rejects_legacy_sink_before_lifecycle_or_io() -> None:
    sink = LegacyObservableSink()
    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        validate_sink_effect_capability(sink, mode="write")
    assert sink.on_start_calls == 0
    assert sink.write_calls == 0
```

Add fresh-run and resume ordering regressions around
`RunContextFactory.initialize_run_context()` proving collection validation
runs after resolved sink instances/configured modes are available but before
node assignment, restricted-context construction, any plugin `on_start`,
reservation, inspection, credential resolution, or I/O. Add a non-run
construction/settings-stub regression proving composer/config assembly does
not invoke the production capability gate.

- [ ] **Step 2: Run the tests and confirm the API is absent**

Run:

```bash
.venv/bin/pytest -q tests/unit/contracts/test_sink_effect_contract.py tests/unit/engine/test_sink_effect_preflight.py
```

Expected: collection fails because `elspeth.contracts.sink_effects` and
`validate_sink_effect_capability` do not exist.

- [ ] **Step 3: Implement immutable values and the opt-in protocol**

Define these exact public names and fields:

```python
SINK_EFFECT_PROTOCOL_VERSION: Final = "sink-effect-v1"


class SinkEffectRole(StrEnum):
    PRIMARY = "primary"
    FAILSINK = "failsink"


class SinkEffectState(StrEnum):
    RESERVED = "reserved"
    PREPARED = "prepared"
    IN_FLIGHT = "in_flight"
    FINALIZED = "finalized"


class SinkEffectDescriptorMode(StrEnum):
    PRECOMPUTED = "precomputed"
    RESULT_DERIVED = "result_derived"
    NO_PUBLICATION = "no_publication"


class SinkEffectInspectionMode(StrEnum):
    INSPECTED = "inspected"
    NO_INSPECTION_REQUIRED = "no_inspection_required"


class SinkEffectReconcileKind(StrEnum):
    NOT_APPLIED = "not_applied"
    APPLIED_WITH_EXACT_DESCRIPTOR = "applied_with_exact_descriptor"
    UNKNOWN = "unknown"


class SinkEffectAttemptAction(StrEnum):
    INSPECT = "inspect"
    COMMIT = "commit"
    RECONCILE = "reconcile"


class SinkEffectAttemptState(StrEnum):
    INTENT = "intent"
    RETURNED = "returned"
    RESPONSE_LOST = "response_lost"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SinkEffectMember:
    ordinal: int
    token_id: str
    row_id: str
    ingest_sequence: int
    lineage_key: str
    payload_hash: str
    row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SinkEffectInspectionRequest:
    effect_id: str
    target: str
    predecessor_descriptor: ArtifactDescriptor | None


@dataclass(frozen=True, slots=True)
class SinkEffectInspection:
    mode: SinkEffectInspectionMode
    reference: str
    evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SinkEffectPrepareRequest:
    effect_id: str
    members: Sequence[SinkEffectMember]
    snapshot_members: Sequence[SinkEffectMember]
    inspection: SinkEffectInspection


@dataclass(frozen=True, slots=True)
class SinkEffectPlan:
    effect_id: str
    protocol_version: str
    descriptor_mode: SinkEffectDescriptorMode
    inspection_mode: SinkEffectInspectionMode
    target: str
    plan_hash: str
    payload_hash: str
    expected_descriptor: ArtifactDescriptor | None
    safe_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SinkEffectCommitResult:
    descriptor: ArtifactDescriptor
    evidence: Mapping[str, object]
    accepted_ordinals: Sequence[int]
    diverted_ordinals: Sequence[int]


@dataclass(frozen=True, slots=True)
class SinkEffectReconcileResult:
    kind: SinkEffectReconcileKind
    descriptor: ArtifactDescriptor | None = None
    evidence: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def may_commit(self) -> bool:
        return self.kind is SinkEffectReconcileKind.NOT_APPLIED

    @classmethod
    def applied(
        cls,
        descriptor: ArtifactDescriptor,
        *,
        evidence: Mapping[str, object],
    ) -> "SinkEffectReconcileResult":
        return cls(
            kind=SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR,
            descriptor=descriptor,
            evidence=MappingProxyType(dict(evidence)),
        )

    @classmethod
    def not_applied(cls, *, evidence: Mapping[str, object]) -> "SinkEffectReconcileResult":
        return cls(
            kind=SinkEffectReconcileKind.NOT_APPLIED,
            evidence=MappingProxyType(dict(evidence)),
        )

    @classmethod
    def unknown(cls, *, evidence: Mapping[str, object]) -> "SinkEffectReconcileResult":
        return cls(
            kind=SinkEffectReconcileKind.UNKNOWN,
            evidence=MappingProxyType(dict(evidence)),
        )


@dataclass(frozen=True, slots=True)
class RestrictedSinkEffectContext:
    run_id: str
    run_started_at: datetime
    operation_id: str
    sink_node_id: str
```

Validate exact descriptor presence, bounded evidence, credential-free target,
and enum consistency in `__post_init__`. Add `effect_protocol_version: ClassVar[str
| None] = None` to `BaseSink`; first-party adapters opt in only when their task
is complete. Add `inspect_effect`, `prepare_effect`, `commit_effect`, and
`reconcile_effect` to `SinkEffectProtocol` without claiming all legacy
`SinkProtocol` implementations satisfy it.

Use these method signatures on the opt-in protocol:

```python
class SinkEffectProtocol(SinkProtocol, Protocol):
    effect_protocol_version: ClassVar[str]
    supported_effect_modes: ClassVar[frozenset[str]]

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        raise NotImplementedError

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        raise NotImplementedError

    def commit_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        raise NotImplementedError

    def reconcile_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectReconcileResult:
        raise NotImplementedError
```

- [ ] **Step 4: Implement local/declarative preflight**

`validate_sink_effect_capability(sink, mode)` must require the exact protocol
version and an adapter-declared supported mode. It must not call `on_start`,
resolve credentials, inspect a target, or write an audit row. Wire it into run
preflight before plugin lifecycle startup. The collection-level gate belongs
at the shared fresh/resume `RunContextFactory.initialize_run_context()`
boundary, before node assignment or context/lifecycle construction. Do not
wire it into `assemble_and_validate_pipeline_config()` or add a compatibility
bypass for composer/settings stubs; those are non-run construction surfaces.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/contracts/test_sink_effect_contract.py tests/unit/engine/test_sink_effect_preflight.py tests/unit/plugins/test_base_sink_contract.py
.venv/bin/mypy src/elspeth/contracts/sink_effects.py src/elspeth/contracts/plugin_protocols.py src/elspeth/engine/orchestrator/preflight.py src/elspeth/engine/orchestrator/run_context_factory.py
```

Expected: all tests pass and mypy is clean.

Commit:

```bash
git add src/elspeth/contracts src/elspeth/plugins/infrastructure/base.py src/elspeth/engine/orchestrator/preflight.py src/elspeth/engine/orchestrator/run_context_factory.py tests/unit/contracts/test_sink_effect_contract.py tests/unit/engine/test_sink_effect_preflight.py
git commit -m "feat(contracts): define sink effect protocol"
```

## Task 3: Define epoch-26 schema and mechanical invariants

**Files:**

- Modify: `src/elspeth/core/landscape/schema.py`
- Modify: `src/elspeth/contracts/audit.py`
- Modify: `src/elspeth/core/landscape/model_loaders.py`
- Test: `tests/unit/core/landscape/test_sink_effect_schema.py`
- Test: `tests/unit/core/landscape/test_schema_epoch_and_required_columns.py`
- Test: `tests/testcontainer/web/test_schema_probe_postgres.py`

- [ ] **Step 1: Write failing SQLite metadata and CHECK/FK tests**

Create effects through raw SQL and assert each invalid shape is rejected:

```python
@pytest.mark.parametrize(
    ("state", "plan_hash", "lease_owner", "finalized_at"),
    [
        ("reserved", "a" * 64, None, None),
        ("prepared", None, None, None),
        ("in_flight", "a" * 64, None, None),
        ("finalized", "a" * 64, None, None),
    ],
)
def test_effect_state_completeness_is_enforced(
    connection: Connection,
    state: str,
    plan_hash: str | None,
    lease_owner: str | None,
    finalized_at: datetime | None,
) -> None:
    with pytest.raises(IntegrityError):
        connection.execute(sink_effects_table.insert().values(EFFECT_BASE | VALUES))


def test_member_token_row_run_ownership_is_enforced(connection: Connection) -> None:
    with pytest.raises(IntegrityError):
        connection.execute(sink_effect_members_table.insert().values(CROSS_RUN_MEMBER))
```

Also assert the exclusive artifact producer CHECK and the
`operations.sink_effect_id -> sink_effects.effect_id` unique FK.

- [ ] **Step 2: Run the schema tests and verify epoch 26 is absent**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_schema.py tests/unit/core/landscape/test_schema_epoch_and_required_columns.py
```

Expected: FAIL because the four effect tables, new columns, and epoch 26 do not
exist.

- [ ] **Step 3: Add the four tables and artifact/operation linkage**

Add metadata tables with these keys and constraints:

```python
sink_effect_streams_table = Table(
    "sink_effect_streams", metadata,
    Column("stream_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False),
    Column("sink_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("role", String(16), nullable=False),
    Column("requested_target_hash", String(64), nullable=False),
    Column("resolved_target", String(512)),
    Column("next_sequence", Integer, nullable=False),
    Column("tail_effect_id", String(64)),
    Column("head_effect_id", String(64)),
    Column("head_descriptor_hash", String(64)),
    UniqueConstraint("run_id", "sink_node_id", "role", "requested_target_hash"),
    CheckConstraint("role IN ('primary','failsink')"),
    CheckConstraint("next_sequence >= 0"),
    ForeignKeyConstraint(["sink_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)
```

Define `sink_effects`, `sink_effect_members`, and `sink_effect_attempts` with
the approved design's lifecycle CHECKs, composite same-run FKs, stream
sequence/predecessor shape, member ownership, closed states, and unique member
binding. Add the stream/effect tail/head FKs after both tables exist. Change
artifacts to producer XOR plus non-null publication-performed/evidence-kind
columns, and operations to nullable unique effect linkage.
Set `SQLITE_SCHEMA_EPOCH = 26`; do not add any epoch-27 symbol.

- [ ] **Step 4: Add immutable loader models**

Add `SinkEffectStream`, `SinkEffect`, `SinkEffectMemberRecord`, and
`SinkEffectAttempt` dataclasses to `contracts/audit.py` and corresponding
strict loaders. Loader enum conversion and nullable-state validation must
raise on malformed DB rows.

- [ ] **Step 5: Run SQLite and real PostgreSQL schema probes**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_schema.py tests/unit/core/landscape/test_schema_epoch_and_required_columns.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/web/test_schema_probe_postgres.py
```

Expected: metadata invariants pass on SQLite and PostgreSQL 16.

- [ ] **Step 6: Commit the metadata boundary**

```bash
git add src/elspeth/core/landscape/schema.py src/elspeth/contracts/audit.py src/elspeth/core/landscape/model_loaders.py tests/unit/core/landscape/test_sink_effect_schema.py tests/unit/core/landscape/test_schema_epoch_and_required_columns.py tests/testcontainer/web/test_schema_probe_postgres.py
git commit -m "feat(landscape): define epoch 26 sink effects"
```

## Task 4: Implement the exact SQLite 25-to-26 migration

**Files:**

- Modify: `src/elspeth/core/landscape/database.py`
- Test: `tests/unit/core/landscape/test_sink_effect_migration.py`
- Modify: `tests/unit/core/landscape/test_database_compatibility_guards.py`
- Modify: `tests/unit/core/landscape/test_artifact_idempotency_migration.py`

- [ ] **Step 1: Write failing populated-epoch migration tests**

Build a genuine epoch-25 database using the existing helper, insert a legacy
state-linked artifact and operation, then open it through current
`LandscapeDB`:

```python
def test_epoch_25_migrates_to_26_and_preserves_legacy_artifact(tmp_path: Path) -> None:
    path = build_exact_epoch_25_database(tmp_path)
    db = LandscapeDB.from_path(path)
    with db.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 26
        artifact = conn.exec_driver_sql(
            "SELECT produced_by_state_id, sink_effect_id FROM artifacts WHERE artifact_id='artifact-legacy'"
        ).one()
        assert artifact == ("state-legacy", None)


def test_epoch_25_to_26_rolls_back_every_object_when_artifact_rebuild_fails(tmp_path: Path) -> None:
    corrupt_epoch_25_path = build_corrupt_epoch_25_database(tmp_path)
    with pytest.raises(AuditIntegrityError, match="epoch-26"):
        LandscapeDB.from_path(corrupt_epoch_25_path)
    assert read_user_version(corrupt_epoch_25_path) == 25
    assert read_table_names(corrupt_epoch_25_path) == EPOCH_25_TABLES
```

Cover exact 23 -> 24 -> 25 -> 26 ordering, `BEGIN IMMEDIATE` contention,
malformed predecessor refusal, duplicate key refusal, close/invalidation
failure, and successful reopen.

- [ ] **Step 2: Run the migration tests and confirm epoch 25 is rejected**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_migration.py tests/unit/core/landscape/test_artifact_idempotency_migration.py
```

Expected: FAIL because database synchronization has no 25 -> 26 verb.

- [ ] **Step 3: Add ordered migration dispatch**

Extend `_sync_sqlite_schema_epoch()` so only these populated predecessors are
accepted:

```python
if epoch == 23:
    self._migrate_sqlite_epoch_23_to_24()
    epoch = 24
if epoch == 24:
    self._migrate_sqlite_epoch_24_to_25()
    epoch = 25
if epoch == 25:
    self._migrate_sqlite_epoch_25_to_26()
```

`_migrate_sqlite_epoch_25_to_26()` must validate the exact epoch-25 physical
shape before and after `BEGIN IMMEDIATE`, create effect tables in FK-safe
order, rebuild artifacts and operations with data-preserving `INSERT SELECT`,
run `PRAGMA foreign_key_check`, stamp 26 inside the transaction, and apply the
existing rollback/invalidate/close discipline on every exception path.
The artifacts rebuild also adds non-null `publication_performed` and
`publication_evidence_kind`; legacy epoch-25 rows backfill `true` and
`legacy_returned` so old evidence is not misclassified as no-publication.

- [ ] **Step 4: Run migration and compatibility suites**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_migration.py tests/unit/core/landscape/test_artifact_idempotency_migration.py tests/unit/core/landscape/test_database_compatibility_guards.py tests/unit/core/landscape/test_write_intent_begin.py
```

Expected: all migration paths pass, with epoch 27 still rejected.

- [ ] **Step 5: Commit the ordered migration**

```bash
git add src/elspeth/core/landscape/database.py tests/unit/core/landscape/test_sink_effect_migration.py tests/unit/core/landscape/test_artifact_idempotency_migration.py tests/unit/core/landscape/test_database_compatibility_guards.py
git commit -m "feat(landscape): migrate sink effect epoch 26"
```

## Task 5: Migrate artifact and operation APIs to effect linkage

**Files:**

- Modify: `src/elspeth/contracts/audit.py`
- Modify: `src/elspeth/contracts/export_records.py`
- Modify: `src/elspeth/core/landscape/model_loaders.py`
- Modify: `src/elspeth/core/landscape/execution/artifacts.py`
- Modify: `src/elspeth/core/landscape/execution/operations.py`
- Modify: `src/elspeth/core/landscape/exporter.py`
- Modify: `src/elspeth/core/landscape/export_mappers.py`
- Modify: `src/elspeth/core/landscape/reproducibility.py`
- Modify: `src/elspeth/core/landscape/query_repository.py`
- Modify: `src/elspeth/mcp/types.py` and the artifact analyzer mapping
- Modify: web audit artifact response models
- Test: `tests/unit/core/landscape/test_effect_linked_artifacts.py`
- Test: `tests/unit/core/landscape/test_exporter.py`
- Test: `tests/unit/core/landscape/test_reproducibility.py`
- Test: `tests/unit/web/test_aws_ecs_acceptance.py`

- [ ] **Step 1: Write failing XOR and backward-consumer tests**

```python
def test_artifact_requires_exactly_one_producer_link() -> None:
    with pytest.raises(ValueError, match="exactly one producer"):
        make_artifact(produced_by_state_id=None, sink_effect_id=None)
    with pytest.raises(ValueError, match="exactly one producer"):
        make_artifact(produced_by_state_id="state", sink_effect_id="effect")


def test_export_contains_legacy_and_effect_producer_kinds(factory: RecorderFactory) -> None:
    legacy = register_legacy_artifact(factory)
    effect = register_effect_artifact(factory, publication_evidence_kind="reconciled")
    records = export_artifacts(factory)
    assert records[legacy.artifact_id]["producer_kind"] == "node_state"
    assert records[effect.artifact_id]["producer_kind"] == "sink_effect"
    assert records[effect.artifact_id]["publication_performed"] is True
```

Add round-trip, reproducibility, MCP/web serialization, AWS acceptance, and
`NO_PUBLICATION` inherited/virtual evidence assertions.

- [ ] **Step 2: Run focused consumers and observe non-null assumptions**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_effect_linked_artifacts.py tests/unit/core/landscape/test_exporter.py tests/unit/core/landscape/test_reproducibility.py tests/unit/web/test_aws_ecs_acceptance.py
```

Expected: FAIL at current `Artifact.produced_by_state_id: str` and serializers
that do not expose producer kind.

- [ ] **Step 3: Implement the XOR public model and repository API**

Change `Artifact` to:

```python
@dataclass(frozen=True, slots=True)
class Artifact:
    artifact_id: str
    run_id: str
    sink_node_id: str
    artifact_type: str
    path_or_uri: str
    content_hash: str
    size_bytes: int
    created_at: datetime
    produced_by_state_id: str | None = None
    sink_effect_id: str | None = None
    idempotency_key: str | None = None
    publication_performed: bool = True
    publication_evidence_kind: str = "returned"

    def __post_init__(self) -> None:
        if (self.produced_by_state_id is None) == (self.sink_effect_id is None):
            raise ValueError("Artifact requires exactly one producer link")
        require_int(self.size_bytes, "size_bytes", min_value=0)
```

Make `register_artifact` accept keyword-only `state_id` and `sink_effect_id`,
enforce XOR before SQL, include the effect field in idempotent divergence
checks, and retain legacy reads unchanged.

- [ ] **Step 4: Update every read/export consumer**

Add explicit `producer_kind`, both nullable links,
`publication_performed`, and closed evidence kind to export records, MCP/web
models, query mappings, and reproducibility input. Never infer a publication
from a non-null artifact ID.

- [ ] **Step 5: Run all artifact consumers and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_effect_linked_artifacts.py tests/unit/core/landscape/repository_integration/test_recorder_artifacts.py tests/unit/core/landscape/test_exporter.py tests/unit/core/landscape/test_reproducibility.py tests/unit/web/test_aws_ecs_acceptance.py
.venv/bin/mypy src/elspeth/contracts/audit.py src/elspeth/core/landscape/execution/artifacts.py src/elspeth/core/landscape/exporter.py src/elspeth/mcp
```

Expected: tests pass and nullable producer links are handled explicitly.

Commit:

```bash
git add src/elspeth/contracts src/elspeth/core/landscape src/elspeth/mcp src/elspeth/web tests/unit/core/landscape tests/unit/web/test_aws_ecs_acceptance.py
git commit -m "feat(landscape): link artifacts to sink effects"
```

## Task 6: Implement bounded lineage and deterministic effect identity

**Files:**

- Create: `src/elspeth/core/landscape/execution/sink_effect_identity.py`
- Test: `tests/unit/core/landscape/test_sink_effect_identity.py`
- Test: `tests/property/core/test_sink_effect_identity_properties.py`

- [ ] **Step 1: Write failing ordering, corruption, and identity tests**

```python
def test_member_order_uses_ingest_then_recursive_parent_ordinals(factory: RecorderFactory) -> None:
    tokens = build_fork_expand_coalesce_graph(factory)
    ordered = resolve_sink_effect_members(factory, reversed(tokens))
    assert [member.token_id for member in ordered] == EXPECTED_LOGICAL_ORDER
    assert [member.ordinal for member in ordered] == list(range(len(tokens)))


@pytest.mark.parametrize("corruption", ["cycle", "missing_parent", "duplicate_ordinal", "cross_run"])
def test_lineage_corruption_fails_before_effect_creation(
    corruption: str,
    factory: RecorderFactory,
) -> None:
    corrupt_lineage(corruption)
    with pytest.raises(AuditIntegrityError):
        resolve_sink_effect_members(factory, TOKENS)
    assert count_sink_effects(factory) == 0


def test_state_attempt_ids_do_not_change_effect_identity(factory: RecorderFactory) -> None:
    first = build_identity_input(factory, current_state_ids=("state-attempt-0",))
    second = build_identity_input(factory, current_state_ids=("state-attempt-1",))
    assert compute_effect_identity(first.members) == compute_effect_identity(second.members)
```

Property tests generate DAG-shaped lineage within limits and prove reversed
arrival produces the same ordered membership/effect ID.

- [ ] **Step 2: Run tests and verify resolver absence**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_identity.py tests/property/core/test_sink_effect_identity_properties.py
```

Expected: FAIL because the identity module does not exist.

- [ ] **Step 3: Implement bounded canonical lineage**

Use these code-owned limits:

```python
MAX_LINEAGE_DEPTH: Final = 256
MAX_LINEAGE_NODES_PER_MEMBER: Final = 4_096
MAX_LINEAGE_PARENTS: Final = 1_024
MAX_LINEAGE_EVIDENCE_BYTES: Final = 64 * 1024
```

Resolve parents ordered by durable ordinal, encode a root as `[]` and a child
as a list of `[ordinal, parent_structure]` entries, detect the approved corruption cases,
and order members by `(ingest_sequence, lineage_structure, token_id)`. Hash
canonical row payload and the complete ordered identity with the protocol
version. Derive `effect_id`, `artifact_id`, `artifact_idempotency_key`,
`stream_id`, and member sub-effect IDs from labeled hashes.

- [ ] **Step 4: Run unit/property tests and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_identity.py tests/property/core/test_sink_effect_identity_properties.py
.venv/bin/mypy src/elspeth/core/landscape/execution/sink_effect_identity.py
```

Expected: all examples pass within the configured Hypothesis budget.

Commit:

```bash
git add src/elspeth/core/landscape/execution/sink_effect_identity.py tests/unit/core/landscape/test_sink_effect_identity.py tests/property/core/test_sink_effect_identity_properties.py
git commit -m "feat(landscape): derive ordered sink effect identity"
```

## Task 7: Reserve effects, partition overlap, and serialize target streams

**Files:**

- Create: `src/elspeth/core/landscape/execution/sink_effect_reservation.py`
- Create: `src/elspeth/core/landscape/execution/sink_effects.py`
- Modify: `src/elspeth/core/landscape/execution/__init__.py`
- Modify: `src/elspeth/core/landscape/execution_repository.py`
- Modify: `src/elspeth/core/landscape/factory.py`
- Test: `tests/unit/core/landscape/test_sink_effect_reservation.py`
- Test: `tests/testcontainer/core/test_sink_effect_lock_order_postgres.py`

- [ ] **Step 1: Write failing reservation/overlap tests**

```python
def test_two_contenders_reserve_one_effect_and_one_order(factory: RecorderFactory) -> None:
    first, second = run_concurrently(lambda: reserve(factory, MEMBERS), lambda: reserve(factory, reversed(MEMBERS)))
    assert first.effect_id == second.effect_id
    assert first.members == second.members == EXPECTED_MEMBERS


def test_overlap_partitions_final_inflight_and_unbound(factory: RecorderFactory) -> None:
    finalized = reserve_and_finalize(factory, TOKENS[:2])
    in_flight = reserve_and_lease(factory, TOKENS[2:3])
    result = reserve(factory, TOKENS)
    assert result.finalized_effect_ids == (finalized.effect_id,)
    assert result.open_effect_ids == (in_flight.effect_id,)
    assert result.new_effect.members == (TOKENS[3],)


def test_disjoint_replacing_groups_form_one_predecessor_chain(factory: RecorderFactory) -> None:
    one, two = concurrent_reservations(GROUP_A, GROUP_B)
    assert {one.stream_sequence, two.stream_sequence} == {0, 1}
    successor = max((one, two), key=attrgetter("stream_sequence"))
    predecessor = min((one, two), key=attrgetter("stream_sequence"))
    assert successor.predecessor_effect_id == predecessor.effect_id
```

Add immutable finalized-membership and divergent-payload refusal tests.

- [ ] **Step 2: Write a real PostgreSQL reservation-vs-outcome race**

Create two engines/connections, assert distinct `pg_backend_pid()`, pause one
after sorted token locks and the other after its first lock, then release both.
Capture both complete token and current-state lock query results and assert
ascending IDs before either contender reaches a stream. Assert bounded
completion, no `40P01`, and one exact effect/member set.

- [ ] **Step 3: Run focused tests and confirm no repository exists**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_reservation.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/core/test_sink_effect_lock_order_postgres.py -k reservation
```

Expected: FAIL because reservation/factory surfaces are absent.

- [ ] **Step 4: Implement token-first reservation**

Expose:

```python
@dataclass(frozen=True, slots=True)
class SinkEffectReservationRequest:
    run_id: str
    sink_node_id: str
    role: SinkEffectRole
    requested_target_hash: str
    members: Sequence[SinkEffectMember]
    config_hash: str
    replacing_target: bool
    primary_effect_id: str | None


@dataclass(frozen=True, slots=True)
class SinkEffectReservationResult:
    finalized_effect_ids: Sequence[str]
    open_effect_ids: Sequence[str]
    new_effect: SinkEffect | None


class SinkEffectRepository:
    def reserve(
        self,
        *,
        run_id: str,
        sink_node_id: str,
        role: SinkEffectRole,
        requested_target_hash: str,
        members: Sequence[SinkEffectMember],
        config_hash: str,
        replacing_target: bool,
        primary_effect_id: str | None = None,
    ) -> SinkEffectReservationResult:
        request = SinkEffectReservationRequest(
            run_id=run_id,
            sink_node_id=sink_node_id,
            role=role,
            requested_target_hash=requested_target_hash,
            members=tuple(members),
            config_hash=config_hash,
            replacing_target=replacing_target,
            primary_effect_id=primary_effect_id,
        )
        return self._reservation.reserve(request)
```

The implementation must optimistic-read IDs, lock every token ascending,
revalidate row/run/payload and existing binding, lock required current states,
then insert/select-lock the stream, allocate tail/sequence, lock existing
effects ascending, and insert effect/members/operation. Use backend-native
conflict-safe insertion and exact winner comparison. The method must never
hold a stream while waiting for token/state locks. The complete required
current-state witness set must be deduplicated and locked in ascending
`state_id` order after tokens and before the stream; caller or planner order is
not acceptable.

- [ ] **Step 5: Wire the repository and run SQLite/PostgreSQL tests**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_reservation.py tests/unit/core/landscape/test_factory.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/core/test_sink_effect_lock_order_postgres.py -k reservation
.venv/bin/mypy src/elspeth/core/landscape/execution/sink_effect_reservation.py src/elspeth/core/landscape/execution/sink_effects.py src/elspeth/core/landscape/factory.py
```

Expected: all tests pass; PostgreSQL connections are distinct and no deadlock
is retried away.

- [ ] **Step 6: Commit reservation and stream ordering**

```bash
git add src/elspeth/core/landscape/execution src/elspeth/core/landscape/execution_repository.py src/elspeth/core/landscape/factory.py tests/unit/core/landscape/test_sink_effect_reservation.py tests/testcontainer/core/test_sink_effect_lock_order_postgres.py
git commit -m "feat(landscape): reserve ordered sink effects"
```

## Task 8: Persist inspection, immutable plans, leases, and attempts

**Files:**

- Create: `src/elspeth/core/landscape/execution/sink_effect_lifecycle.py`
- Modify: `src/elspeth/core/landscape/execution/sink_effects.py`
- Modify: `src/elspeth/core/landscape/execution/calls.py`
- Modify: `src/elspeth/core/landscape/execution/operations.py`
- Test: `tests/unit/core/landscape/test_sink_effect_lifecycle.py`
- Test: `tests/testcontainer/core/test_sink_effect_lock_order_postgres.py`

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_reserved_effect_cannot_lease_before_complete_plan(repo: SinkEffectRepository) -> None:
    effect = reserve_effect(repo)
    with pytest.raises(LandscapeRecordError, match="prepared"):
        repo.acquire_lease(effect.effect_id, owner="worker-a", ttl=timedelta(seconds=30))


def test_concurrent_plan_cas_accepts_equal_and_rejects_divergent(repo: SinkEffectRepository) -> None:
    effect = reserve_effect(repo)
    assert repo.complete_plan(effect.effect_id, PLAN) == repo.complete_plan(effect.effect_id, PLAN)
    with pytest.raises(AuditIntegrityError, match="divergent plan"):
        repo.complete_plan(effect.effect_id, DIFFERENT_PLAN)


def test_abandoned_commit_intent_becomes_response_lost_before_reconcile(
    repo: SinkEffectRepository,
) -> None:
    attempt = repo.begin_attempt(EFFECT, action="commit", request_hash=PLAN.plan_hash)
    recovered = repo.mark_response_lost(attempt.attempt_id)
    assert recovered.state is SinkEffectAttemptState.RESPONSE_LOST
    assert operation_calls(EFFECT.operation_id) == [("error", "response_lost")]
```

Cover `NO_INSPECTION_REQUIRED`, returned inspect evidence, bounded/redacted
evidence, expiry/takeover generation, stale write rejection, stable call
indices, and no transaction spanning external I/O.

- [ ] **Step 2: Run tests and observe missing lifecycle verbs**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_lifecycle.py
```

Expected: FAIL because inspect/plan/lease/attempt methods are absent.

- [ ] **Step 3: Implement lifecycle CAS methods**

Expose these exact operations:

```python
@dataclass(frozen=True, slots=True)
class SinkEffectAttemptRequest:
    effect_id: str
    member_ordinal: int | None
    generation: int
    action: SinkEffectAttemptAction
    request_hash: str


@dataclass(frozen=True, slots=True)
class SinkEffectAttemptResult:
    attempt_id: str
    evidence: Mapping[str, object]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class SinkEffectLease:
    effect_id: str
    owner: str
    generation: int
    expires_at: datetime


class SinkEffectRepository:
    def begin_attempt(self, request: SinkEffectAttemptRequest) -> SinkEffectAttempt:
        return self._lifecycle.begin_attempt(request)

    def record_attempt_result(self, result: SinkEffectAttemptResult) -> SinkEffectAttempt:
        return self._lifecycle.record_attempt_result(result)

    def mark_response_lost(self, attempt_id: str) -> SinkEffectAttempt:
        return self._lifecycle.mark_response_lost(attempt_id)

    def complete_plan(self, effect_id: str, plan: SinkEffectPlan) -> SinkEffect:
        return self._lifecycle.complete_plan(effect_id, plan)

    def acquire_lease(self, effect_id: str, *, owner: str, ttl: timedelta) -> SinkEffectLease:
        return self._lifecycle.acquire_lease(effect_id, owner=owner, ttl=ttl)

    def heartbeat_lease(
        self,
        effect_id: str,
        *,
        owner: str,
        generation: int,
        ttl: timedelta,
    ) -> SinkEffectLease:
        return self._lifecycle.heartbeat_lease(
            effect_id,
            owner=owner,
            generation=generation,
            ttl=ttl,
        )

    def takeover_expired(self, effect_id: str, *, owner: str, ttl: timedelta) -> SinkEffectLease:
        return self._lifecycle.takeover_expired(effect_id, owner=owner, ttl=ttl)
```

Each method uses a short transaction, validates generation, obeys
stream/effect/operation order, and inserts the call row at returned or
response-lost classification time. Evidence validation must occur before SQL.

- [ ] **Step 4: Add takeover-vs-finalization PostgreSQL interleaving**

Force takeover to pause after effect lock and finalization to approach through
sorted token/state locks. Assert one legal winner: either finalization commits
and takeover observes FINALIZED, or takeover increments generation and stale
finalization is rejected. Assert no deadlock.

- [ ] **Step 5: Run lifecycle/PG tests and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_lifecycle.py tests/unit/core/landscape/test_call_recording.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/core/test_sink_effect_lock_order_postgres.py -k 'takeover or stream'
.venv/bin/mypy src/elspeth/core/landscape/execution/sink_effect_lifecycle.py src/elspeth/core/landscape/execution/calls.py src/elspeth/core/landscape/execution/operations.py
```

Expected: all tests pass.

Commit:

```bash
git add src/elspeth/core/landscape/execution tests/unit/core/landscape/test_sink_effect_lifecycle.py tests/testcontainer/core/test_sink_effect_lock_order_postgres.py
git commit -m "feat(landscape): fence sink effect lifecycle"
```

## Task 9: Finalize effect, artifact, states, and outcomes in global order

**Files:**

- Create: `src/elspeth/core/landscape/execution/sink_effect_finalization.py`
- Modify: `src/elspeth/core/landscape/execution/sink_effects.py`
- Modify: `src/elspeth/core/landscape/data_flow/outcomes.py`
- Test: `tests/unit/core/landscape/test_sink_effect_finalization.py`
- Test: `tests/testcontainer/core/test_sink_effect_lock_order_postgres.py`

- [ ] **Step 1: Write failing exact-finalization tests**

```python
def test_finalization_is_one_transaction_and_retry_returns_winner(
    repo: SinkEffectRepository,
) -> None:
    effect = prepared_applied_effect()
    first = repo.finalize(FINALIZE_REQUEST)
    second = repo.finalize(FINALIZE_REQUEST)
    assert first == second
    assert count_artifacts(effect.effect_id) == 1
    assert all_states_completed(effect.members)
    assert all_outcomes_exact(effect.members)


def test_new_attempt_state_witness_is_resolved_not_part_of_identity(
    repo: SinkEffectRepository,
) -> None:
    effect = reserve_with_state_attempt(0)
    replace_open_state_with_attempt(1)
    result = repo.finalize(finalize_request(effect, generation=1))
    assert result.effect_id == effect.effect_id
    assert result.artifact.sink_effect_id == effect.effect_id
```

Cover descriptor mismatch, stale generation, missing/current witness mismatch,
stream head CAS, result-derived Database evidence, no-publication artifact,
primary/failsink linkage, and finalization response loss.

- [ ] **Step 2: Run tests and verify no atomic effect finalizer exists**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_finalization.py
```

Expected: FAIL at missing finalization API.

- [ ] **Step 3: Implement token/state-first finalization**

`finalize(request)` must optimistic-resolve witnesses; lock sorted tokens;
re-resolve and lock sorted states; lock stream and linked effects sorted;
validate exact result/member evidence; CAS head; register the reserved artifact
ID/key against `sink_effect_id`; complete the stable operation; write routing,
outcomes, scheduler closes, and effect FINALIZED; then commit once. A witness
change triggers a bounded restart before any later-class lock.

Use the Task 1a bulk-state primitive for the complete state-witness set; the
finalizer must not add an effect-specific state-lock path.

- [ ] **Step 4: Add composed PostgreSQL races**

Force finalization versus state/outcome mutation and effect-linked/legacy
artifact mutation/read interleavings on distinct backends. Assert the legal
winner plus no deadlock and no partial audit state. The Task 1a test remains
the canonical proof of the shared bulk-state acquisition primitive.

- [ ] **Step 5: Run finalization, outcome, and PG tests**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_finalization.py tests/unit/core/landscape/test_data_flow_live_buffered_outcomes.py tests/unit/core/landscape/repository_integration/test_recorder_artifacts.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/core/test_sink_effect_lock_order_postgres.py
.venv/bin/mypy src/elspeth/core/landscape/execution/sink_effect_finalization.py src/elspeth/core/landscape/data_flow/outcomes.py
```

Expected: all tests pass, including distinct-backend lock assertions.

- [ ] **Step 6: Commit atomic finalization**

```bash
git add src/elspeth/core/landscape/execution/sink_effect_finalization.py src/elspeth/core/landscape/execution/sink_effects.py src/elspeth/core/landscape/data_flow/outcomes.py tests/unit/core/landscape/test_sink_effect_finalization.py tests/testcontainer/core/test_sink_effect_lock_order_postgres.py
git commit -m "feat(landscape): finalize sink effects atomically"
```

## Task 10: Build the caller-level coordinator with crash seams

**Files:**

- Create: `src/elspeth/engine/executors/sink_effects.py`
- Create: `tests/fixtures/sink_effects.py`
- Modify: `src/elspeth/engine/executors/sink.py`
- Modify: `src/elspeth/engine/orchestrator/sink_flush.py`
- Test: `tests/unit/engine/test_sink_effect_executor.py`
- Test: `tests/integration/pipeline/test_sink_effect_recovery.py`

- [ ] **Step 1: Write the duplicate-observable fault-seam tests**

```python
@pytest.mark.parametrize(
    "fault",
    [
        SinkEffectFault.BEFORE_EFFECT,
        SinkEffectFault.AFTER_EFFECT_BEFORE_RETURN,
        SinkEffectFault.AFTER_RETURN_BEFORE_FINALIZE,
        SinkEffectFault.AFTER_FINALIZE_BEFORE_RESPONSE,
    ],
)
def test_fresh_executor_retry_publishes_once(fault: SinkEffectFault, landscape: LandscapeDB) -> None:
    sink = DuplicateObservableSink(fault=fault)
    run_once_expect_fault(landscape, sink)
    recovered = run_with_fresh_executor_and_sink(landscape, sink.external_target)
    assert sink.external_target.publication_count == 1
    assert recovered.effect_id == sink.external_target.effect_id
    assert recovered.artifact.content_hash == sink.external_target.content_hash
```

Add `UNKNOWN` never-commit, exact NOT_APPLIED retry, equal APPLIED finalize,
predecessor wait, disjoint batch convergence, and finalization-response winner
tests.

- [ ] **Step 2: Run tests against the old write/flush boundary**

Run:

```bash
.venv/bin/pytest -q tests/unit/engine/test_sink_effect_executor.py tests/integration/pipeline/test_sink_effect_recovery.py
```

Expected: FAIL because production calls `write()` before durable reservation
and duplicate publication is observed.

- [ ] **Step 3: Implement the coordinator state machine**

Implement this order without a legacy branch:

```python
reservation = effects.reserve(request.members)
for existing in reservation.open_effects:
    recover(existing)
effect = reservation.new_effect
wait_for_predecessor(effect)
inspection = inspect_or_use_typed_sentinel(effect)
plan = prepare_and_complete_plan(effect, inspection)
if plan.descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION:
    return finalize_no_publication(effect, plan)
lease = effects.acquire_lease(effect.effect_id, owner=worker_id, ttl=lease_ttl)
reconcile = reconcile_after_abandoned_intent_or_takeover(effect, lease)
if reconcile.kind is SinkEffectReconcileKind.UNKNOWN:
    raise SinkEffectUnknownError(effect.effect_id)
result = reconcile if reconcile.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR else commit_with_intent(effect, lease, plan)
return effects.finalize(finalize_request(effect, lease, result))
```

`commit_effect` subsumes durability. Delete every production `flush()` after
commit and the arbitrary write-only/test-double branch. Preserve the public
`SinkExecutor.write` facade while routing all behavior through the coordinator.

- [ ] **Step 4: Run caller-level and existing sink/diversion tests**

Run:

```bash
.venv/bin/pytest -q tests/unit/engine/test_sink_effect_executor.py tests/integration/pipeline/test_sink_effect_recovery.py tests/unit/engine/test_sink_executor_diversion.py tests/unit/engine/orchestrator/test_pending_sink_grouping.py
.venv/bin/mypy src/elspeth/engine/executors/sink_effects.py src/elspeth/engine/executors/sink.py src/elspeth/engine/orchestrator/sink_flush.py
```

Expected: fault seams publish once and current grouping/diversion behavior
remains semantically correct.

- [ ] **Step 5: Commit the production boundary**

```bash
git add src/elspeth/engine/executors src/elspeth/engine/orchestrator/sink_flush.py tests/fixtures/sink_effects.py tests/unit/engine/test_sink_effect_executor.py tests/integration/pipeline/test_sink_effect_recovery.py
git commit -m "feat(engine): coordinate durable sink effects"
```

## Task 11: Implement bounded local-file effects for CSV, JSON, and Text

**Files:**

- Create: `src/elspeth/plugins/sinks/_local_file_effects.py`
- Modify: `src/elspeth/plugins/sinks/csv_sink.py`
- Modify: `src/elspeth/plugins/sinks/json_sink.py`
- Modify: `src/elspeth/plugins/sinks/text_sink.py`
- Test: `tests/unit/plugins/sinks/test_local_file_sink_effects.py`
- Modify: `tests/unit/plugins/sinks/test_csv_sink.py`
- Modify: `tests/unit/plugins/sinks/test_json_sink.py`
- Modify: `tests/unit/plugins/sinks/test_text_sink.py`
- Modify: `tests/integration/plugins/sinks/test_durability.py`

- [ ] **Step 1: Write failing real-file effect tests**

```python
def test_abandoned_atomic_replace_reconciles_by_staged_file_identity(tmp_path: Path) -> None:
    adapter = make_csv_effect_adapter(tmp_path)
    plan = prepare(adapter, effect_id="effect-a", rows=ROWS)
    adapter.commit_effect(plan, fail_after_replace=True)
    result = fresh_adapter(tmp_path).reconcile_effect(plan)
    assert result.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert result.descriptor == plan.expected_descriptor


def test_equal_bytes_from_unrelated_inode_are_unknown(tmp_path: Path) -> None:
    plan = prepare(make_text_effect_adapter(tmp_path), effect_id="effect-a", rows=ROWS)
    plan.target_path.write_bytes(plan.staged_path.read_bytes())
    result = fresh_adapter(tmp_path).reconcile_effect(plan)
    assert result.kind is SinkEffectReconcileKind.UNKNOWN


def test_disjoint_effects_publish_predecessor_then_successor_without_loss(tmp_path: Path) -> None:
    first, second = reserve_disjoint_csv_effects(tmp_path)
    publish(first)
    publish(second)
    assert read_csv_rows(TARGET) == [*FIRST_ROWS, *SECOND_ROWS]
```

Cover streamed byte limits, row limits, lock timeout, unsupported filesystem
identity, staging hash/path mismatch, parent fsync, append baseline, collision
policy, no-publication inherited/virtual descriptors, and cleanup.

- [ ] **Step 2: Run tests and observe missing effect methods**

Run:

```bash
.venv/bin/pytest -q tests/unit/plugins/sinks/test_local_file_sink_effects.py
```

Expected: FAIL because local sinks expose only write/flush.

- [ ] **Step 3: Implement the shared bounded staging primitive**

Create `LocalFileEffectPlanEvidence` with normalized target/staging paths,
predecessor hash/size/file ID, staged hash/size/file ID, encoding/format, and
stream sequence. Stream serialization directly to an effect-addressed
same-directory file while hashing and enforcing limits. `commit_local_effect`
must acquire the bounded advisory lock, verify predecessor/head and exact
pre-image, fsync staging, `os.replace`, fsync parent, and verify target file ID.
`reconcile_local_effect` returns only the closed three results.

- [ ] **Step 4: Convert CSV, JSON, and Text without legacy flush**

Each sink declares `sink-effect-v1`, implements inspect/prepare/commit/reconcile,
and reuses its existing row validation/header/display semantics. Prepare emits
accepted/diverted decisions without publication. CSV/JSON/Text cumulative
snapshots are streamed from the validated baseline plus finalized/current
members in stored order. Their effect commit never keeps an open writer and
their old `flush()` is lifecycle-only compatibility, never called by the
effect executor.

- [ ] **Step 5: Run local adapter and durability suites**

Run:

```bash
.venv/bin/pytest -q tests/unit/plugins/sinks/test_local_file_sink_effects.py tests/unit/plugins/sinks/test_csv_sink.py tests/unit/plugins/sinks/test_json_sink.py tests/unit/plugins/sinks/test_text_sink.py tests/unit/plugins/sinks/test_csv_sink_append.py tests/unit/plugins/sinks/test_json_sink_resume.py tests/integration/plugins/sinks/test_durability.py
.venv/bin/mypy src/elspeth/plugins/sinks/_local_file_effects.py src/elspeth/plugins/sinks/csv_sink.py src/elspeth/plugins/sinks/json_sink.py src/elspeth/plugins/sinks/text_sink.py
```

Expected: all tests pass with real fsync/replace behavior.

- [ ] **Step 6: Commit local adapters**

```bash
git add src/elspeth/plugins/sinks/_local_file_effects.py src/elspeth/plugins/sinks/csv_sink.py src/elspeth/plugins/sinks/json_sink.py src/elspeth/plugins/sinks/text_sink.py tests/unit/plugins/sinks tests/integration/plugins/sinks/test_durability.py
git commit -m "feat(sinks): make local files effect-safe"
```

## Task 12: Implement conditional S3 and Azure Blob effects

**Files:**

- Modify: `src/elspeth/plugins/sinks/aws_s3_sink.py`
- Modify: `src/elspeth/plugins/sinks/azure_blob_sink.py`
- Test: `tests/unit/plugins/sinks/test_aws_s3_sink.py`
- Test: `tests/integration/plugins/sinks/test_aws_s3_sink_botocore.py`
- Test: `tests/unit/plugins/sinks/test_azure_blob_sink.py`
- Test: `tests/unit/plugins/sinks/test_azure_blob_sink_serialization.py`

- [ ] **Step 1: Write failing S3 conditional/recovery tests**

```python
def test_s3_inspect_plan_commit_and_response_loss_reconcile(s3_client: FakeS3Client) -> None:
    plan = inspect_and_prepare_s3(EFFECT, s3_client)
    s3_client.lose_response_after_put = True
    with pytest.raises(OutcomeUnknownError):
        commit_s3(plan)
    result = reconcile_s3(plan, s3_client)
    assert result.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert s3_client.put_calls[0]["IfNoneMatch"] == "*"
    assert s3_client.put_calls[0]["ChecksumSHA256"] == plan.checksum_sha256_b64


def test_s3_never_overwrites_unrelated_object_with_local_effect_id(
    s3_client: FakeS3Client,
) -> None:
    result = reconcile_s3(PLAN, s3_with_divergent_metadata())
    assert result.kind is SinkEffectReconcileKind.UNKNOWN
    assert s3_client.put_call_count == 0
```

Assert successor uses predecessor ETag through `IfMatch`, metadata contains
protocol/effect/plan/content hashes, HEAD evidence is bounded/redacted, stable
run-start timestamp renders the key, and a condition loser reconciles exact.

- [ ] **Step 2: Write Azure ETag/property equivalents**

Assert initial `if_none_match="*"`, successor `if_match=prior_etag`, exact
effect metadata/checksum/length, response-loss property reconciliation, and
unrelated blob refusal.

- [ ] **Step 3: Run remote adapter tests and see old unconditional paths fail**

Run:

```bash
.venv/bin/pytest -q tests/unit/plugins/sinks/test_aws_s3_sink.py tests/integration/plugins/sinks/test_aws_s3_sink_botocore.py tests/unit/plugins/sinks/test_azure_blob_sink.py
```

Expected: new assertions fail because current adapters publish during write
and hold only process-local ETag/upload state.

- [ ] **Step 4: Implement read-only inspect and conditional commit/reconcile**

S3 inspect performs HEAD/non-existence under coordinator intent and returns
typed version/ETag/checksum metadata. Commit sends deterministic cumulative
bytes plus `IfNoneMatch` or `IfMatch`, `ChecksumSHA256`, and bounded metadata.
Azure uses `get_blob_properties`, native match conditions, content settings,
and equivalent metadata. Remove `_buffered_rows`, retry-wall-clock target
rendering, `_has_uploaded`, and process-local remote authority.

- [ ] **Step 5: Run S3/Azure suites and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/plugins/sinks/test_aws_s3_sink.py tests/integration/plugins/sinks/test_aws_s3_sink_botocore.py tests/unit/plugins/sinks/test_azure_blob_sink.py tests/unit/plugins/sinks/test_azure_blob_sink_serialization.py
.venv/bin/mypy src/elspeth/plugins/sinks/aws_s3_sink.py src/elspeth/plugins/sinks/azure_blob_sink.py
```

Expected: all contract-faithful tests pass; live-gated S3 remains separately
selectable and is not required for the commit.

Commit:

```bash
git add src/elspeth/plugins/sinks/aws_s3_sink.py src/elspeth/plugins/sinks/azure_blob_sink.py tests/unit/plugins/sinks/test_aws_s3_sink.py tests/integration/plugins/sinks/test_aws_s3_sink_botocore.py tests/unit/plugins/sinks/test_azure_blob_sink.py tests/unit/plugins/sinks/test_azure_blob_sink_serialization.py
git commit -m "feat(sinks): reconcile conditional object effects"
```

## Task 13: Implement the Database target-side effect ledger

**Files:**

- Modify: `src/elspeth/plugins/sinks/database_sink.py`
- Modify: Database sink config/reference schema
- Test: `tests/unit/plugins/sinks/test_database_sink.py`
- Test: `tests/integration/plugins/sinks/test_database_sink_effects.py`
- Test: `tests/testcontainer/plugins/test_database_sink_effects_postgres.py`

- [ ] **Step 1: Write failing config/preflight tests**

```python
def test_database_requires_declared_effect_ledger_before_reservation() -> None:
    sink = DatabaseSink(CONFIG_WITHOUT_EFFECT_LEDGER)
    with pytest.raises(SinkEffectCapabilityError, match="target-side effect ledger"):
        validate_sink_effect_capability(sink, mode="append")
    assert landscape_effect_count() == 0


def test_inspect_is_read_only_and_rejects_missing_ledger(
    sink: DatabaseSink,
    target_engine: Engine,
) -> None:
    before = list_target_tables()
    with pytest.raises(DatabaseEffectLedgerError, match="provision"):
        sink.inspect_effect(INSPECT_REQUEST, RESTRICTED_CONTEXT)
    assert list_target_tables() == before
```

Add deterministic unsupported replace/dialect errors before I/O.

- [ ] **Step 2: Write failing atomic marker/result-derived tests**

```python
def test_marker_and_accepted_rows_commit_once_with_constraint_diversion(
    sink: DatabaseSink,
    context: RestrictedSinkEffectContext,
) -> None:
    first = sink.commit_effect(RESULT_DERIVED_PLAN, context)
    second = fresh_sink().reconcile_effect(RESULT_DERIVED_PLAN, context)
    assert second == first
    assert target_row_count() == first.descriptor.metadata["row_count"]
    assert marker(first.effect_id).accepted_ordinals == first.accepted_ordinals
    assert marker(first.effect_id).diverted_ordinals == first.diverted_ordinals
```

Inject loss after target transaction commit and assert retry reads the marker
without another insert.

- [ ] **Step 3: Run SQLite and PostgreSQL tests against current append**

Run:

```bash
.venv/bin/pytest -q tests/unit/plugins/sinks/test_database_sink.py tests/integration/plugins/sinks/test_database_sink_effects.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/plugins/test_database_sink_effects_postgres.py
```

Expected: FAIL because no target marker or RESULT_DERIVED contract exists.

- [ ] **Step 4: Implement explicit ledger inspection and transaction**

Add a config capability that names the operator-provisioned ledger and
expected schema version. Inspect verifies dialect/table/columns/permissions by
read-only SQL. Commit starts one target transaction, checks/inserts unique
effect marker, attempts accepted rows with current savepoint diversion logic,
stores accepted/diverted ordinals and derived descriptor evidence in the
marker, then commits. Reconcile reads and validates the marker only. DDL/table
replace is allowed only for dialect/mode combinations proven transactional;
all others fail preflight.

- [ ] **Step 5: Run Database suites and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/plugins/sinks/test_database_sink.py tests/integration/plugins/sinks/test_database_sink_effects.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/plugins/test_database_sink_effects_postgres.py
.venv/bin/mypy src/elspeth/plugins/sinks/database_sink.py
```

Expected: SQLite and real PostgreSQL marker/row transactions pass.

Commit:

```bash
git add src/elspeth/plugins/sinks/database_sink.py tests/unit/plugins/sinks/test_database_sink.py tests/integration/plugins/sinks/test_database_sink_effects.py tests/testcontainer/plugins/test_database_sink_effects_postgres.py
git commit -m "feat(sinks): transact database effect markers"
```

## Task 14: Implement durable Dataverse and Chroma member sub-effects

**Files:**

- Modify: `src/elspeth/plugins/sinks/dataverse.py`
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py`
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py` config validation
- Test: `tests/unit/plugins/sinks/test_dataverse_sink.py`
- Test: `tests/integration/plugins/test_dataverse_pipeline.py`
- Test: `tests/unit/plugins/sinks/test_chroma_sink.py`
- Test: `tests/integration/plugins/sinks/test_chroma_sink_pipeline.py`

- [ ] **Step 1: Write failing Dataverse partial-member recovery tests**

```python
def test_dataverse_partial_batch_persists_exact_and_missing_members(
    effect_repo: SinkEffectRepository,
) -> None:
    client = FakeDataverseClient(fail_after_patch_ordinal=1)
    run_effect_once(client)
    states = load_member_effect_states()
    assert [state.status for state in states] == [FINALIZED, IN_FLIGHT, RESERVED]
    recover_with_fresh_sink(client)
    assert client.patch_count_by_key == {"a": 1, "b": 1, "c": 1}
```

Assert GET exact/missing/divergent maps to the closed member result and group
finalization waits for all exact.

- [ ] **Step 2: Write failing Chroma overwrite/rejection tests**

```python
def test_chroma_overwrite_member_recovery_upserts_each_id_once(
    effect_repo: SinkEffectRepository,
) -> None:
    run_with_response_loss_after_first_upsert()
    recover_with_fresh_chroma_sink()
    assert collection.documents == EXPECTED_DOCUMENTS
    assert member_effects_are_finalized()


@pytest.mark.parametrize("mode", ["skip", "error"])
def test_chroma_non_reconcilable_modes_fail_before_reservation(mode: str) -> None:
    with pytest.raises(SinkEffectCapabilityError, match="on_duplicate=overwrite"):
        preflight_chroma(mode)
    assert landscape_effect_count() == 0
```

- [ ] **Step 3: Run tests against process-local batch behavior**

Run:

```bash
.venv/bin/pytest -q tests/unit/plugins/sinks/test_dataverse_sink.py tests/integration/plugins/test_dataverse_pipeline.py tests/unit/plugins/sinks/test_chroma_sink.py tests/integration/plugins/sinks/test_chroma_sink_pipeline.py
```

Expected: new partial-response tests fail and Chroma skip/error still pass
preflight.

- [ ] **Step 4: Implement member APIs and exact reconciliation**

Dataverse prepares one stable alternate-key member plan, PATCHes under an
attempt, and GET-compares every mapped field. Chroma overwrite prepares one
document-ID plan, upserts under an attempt, and GET-compares document and
metadata. The coordinator persists member state after every call, commits only
missing members, treats divergence as UNKNOWN, and finalizes the group only
when every member is exact. Declare Chroma `skip/error` unsupported with the
approved actionable message.

- [ ] **Step 5: Run record-adapter suites and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/plugins/sinks/test_dataverse_sink.py tests/integration/plugins/test_dataverse_pipeline.py tests/unit/plugins/sinks/test_chroma_sink.py tests/integration/plugins/sinks/test_chroma_sink_pipeline.py
.venv/bin/mypy src/elspeth/plugins/sinks/dataverse.py src/elspeth/plugins/sinks/chroma_sink.py
```

Expected: member recovery passes and unsupported modes fail before I/O.

Commit:

```bash
git add src/elspeth/plugins/sinks/dataverse.py src/elspeth/plugins/sinks/chroma_sink.py tests/unit/plugins/sinks/test_dataverse_sink.py tests/integration/plugins/test_dataverse_pipeline.py tests/unit/plugins/sinks/test_chroma_sink.py tests/integration/plugins/sinks/test_chroma_sink_pipeline.py
git commit -m "feat(sinks): persist remote member effects"
```

## Task 15: Close primary, diversion, failsink, and zero-publication paths

**Files:**

- Modify: `src/elspeth/engine/executors/sink_effects.py`
- Modify: `src/elspeth/engine/executors/sink.py`
- Modify: `src/elspeth/engine/orchestrator/sink_flush.py`
- Modify: `src/elspeth/engine/orchestrator/preflight.py`
- Modify: `src/elspeth/plugins/infrastructure/runtime_factory.py`
- Test: `tests/unit/engine/test_sink_executor_diversion.py`
- Test: `tests/unit/engine/test_failsink_validation.py`
- Test: `tests/integration/pipeline/orchestrator/test_sink_diversion_counters.py`
- Test: `tests/integration/pipeline/test_sink_effect_recovery.py`

- [ ] **Step 1: Write failing linked-effect and zero-publication tests**

```python
def test_primary_finalizes_once_while_diverted_token_waits_for_failsink(
    landscape: LandscapeDB,
) -> None:
    fail_after_primary_before_failsink()
    assert primary_effect().state is SinkEffectState.FINALIZED
    assert failsink_effect().state is SinkEffectState.RESERVED
    assert diverted_token_outcome() is None
    recover()
    assert primary_publication_count() == 1
    assert failsink_publication_count() == 1
    assert diverted_token_outcome().artifact_id == failsink_effect().artifact_id


def test_all_diverted_primary_registers_inherited_no_publication_artifact(
    landscape: LandscapeDB,
) -> None:
    result = execute_all_diverted_batch()
    assert result.primary_artifact.publication_performed is False
    assert result.primary_artifact.publication_evidence_kind == "inherited"
    assert primary_sink_publication_count() == 0
```

Cover initial virtual-empty, Database zero-row marker, discard, failsink CSV,
failsink JSON, response loss between linked effects, scheduler terminalization,
and exact diversion counters.

- [ ] **Step 2: Run diversion/failsink tests**

Run:

```bash
.venv/bin/pytest -q tests/unit/engine/test_sink_executor_diversion.py tests/unit/engine/test_failsink_validation.py tests/integration/pipeline/orchestrator/test_sink_diversion_counters.py tests/integration/pipeline/test_sink_effect_recovery.py
```

Expected: linked/no-publication assertions fail until the coordinator owns
both effects.

- [ ] **Step 3: Implement independent primary and linked failsink finalization**

Prepare primary dispositions before publication. Finalize its accepted effect
when exact. Reserve one linked FAILSINK effect for diverted members, retain
reason/path/error-hash, and delay only those token outcomes/routing/scheduler
closes until the failsink is exact. For discard, record the no-external-effect
terminal evidence in the token-first transaction. Use `NO_PUBLICATION` for
prepare-known zero accepted groups and the target marker for result-derived
Database zero accepted groups.

- [ ] **Step 4: Prove every built-in mode passes preflight and every legacy sink fails**

Parametrize the installed first-party sink inventory and supported mode
matrix. Assert each declares the protocol; Chroma skip/error and unsupported
Database modes fail with remediation; a third-party write/flush-only sink
fails before reservation, lifecycle, or I/O.

- [ ] **Step 5: Run full sink-flow tests and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/engine/test_sink_executor_diversion.py tests/unit/engine/test_failsink_validation.py tests/unit/engine/test_sink_effect_executor.py tests/integration/pipeline/orchestrator/test_sink_diversion_counters.py tests/integration/pipeline/test_sink_effect_recovery.py tests/unit/plugins/sinks/test_sink_protocol_compliance.py
.venv/bin/mypy src/elspeth/engine/executors/sink_effects.py src/elspeth/engine/executors/sink.py src/elspeth/engine/orchestrator/sink_flush.py
```

Expected: all primary/failsink/discard/zero-publication tests pass.

Commit:

```bash
git add src/elspeth/engine src/elspeth/plugins/infrastructure/runtime_factory.py tests/unit/engine tests/integration/pipeline tests/unit/plugins/sinks/test_sink_protocol_compliance.py
git commit -m "feat(engine): recover primary and failsink effects"
```

## Task 16: Export and diagnose complete effect/attempt history

**Files:**

- Modify: `src/elspeth/core/landscape/exporter.py`
- Modify: `src/elspeth/core/landscape/export_mappers.py`
- Modify: `src/elspeth/core/landscape/reproducibility.py`
- Modify: `src/elspeth/mcp/types.py`
- Modify: `src/elspeth/mcp/analyzers/queries.py`
- Modify: `src/elspeth/mcp/analyzer.py`
- Modify: web audit-readiness service and schemas
- Test: `tests/unit/core/landscape/test_sink_effect_export.py`
- Test: `tests/unit/core/landscape/test_reproducibility.py`
- Test: MCP analyzer query tests
- Test: web audit-readiness tests

- [ ] **Step 1: Write failing export/diagnostic tests**

```python
def test_export_preserves_abandoned_intent_before_recovery(
    factory: RecorderFactory,
) -> None:
    begin_commit_intent_without_result()
    export = export_run()
    assert export["sink_effect_attempts"][0]["state"] == "intent"
    assert export["sink_effect_attempts"][0]["request_hash"] == PLAN_HASH


def test_no_publication_is_visible_in_mcp_and_web(
    factory: RecorderFactory,
) -> None:
    artifact = finalize_virtual_empty_effect()
    assert mcp_artifact(artifact)["publication_performed"] is False
    assert web_artifact(artifact)["publication_evidence_kind"] == "virtual_empty"
```

Assert bounded evidence exports hashes/typed metadata without raw target
credentials or provider bodies.

- [ ] **Step 2: Run tests and observe missing effect aggregates**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_export.py tests/unit/core/landscape/test_reproducibility.py tests/unit/mcp tests/unit/web/audit_readiness
```

Expected: effect streams/members/attempts are absent from export and
diagnostics.

- [ ] **Step 3: Add ordered effect export/import/reproducibility records**

Export streams by stream ID, effects by stream sequence/effect ID, members by
ordinal, and attempts by operation call index. Include closed enums, hashes,
safe evidence, predecessor/linkage, and publication metadata. Add these
records to reproducibility hashing and import validation; reject malformed XOR
artifact producers and credential-bearing evidence.

- [ ] **Step 4: Add MCP/web diagnostic surfaces**

Expose effect state, predecessor, lease generation/expiry, reconcile result,
member progress, response-lost attempts, exact descriptor, and operator-safe
UNKNOWN guidance. Keep raw payloads and credentials out of response types.

- [ ] **Step 5: Run audit consumer suites and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_export.py tests/unit/core/landscape/test_exporter.py tests/unit/core/landscape/test_reproducibility.py tests/unit/mcp tests/unit/web/audit_readiness
.venv/bin/mypy src/elspeth/core/landscape/exporter.py src/elspeth/core/landscape/reproducibility.py src/elspeth/mcp src/elspeth/web/audit_readiness
```

Expected: all consumer tests pass with safe complete history.

Commit:

```bash
git add src/elspeth/core/landscape src/elspeth/mcp src/elspeth/web/audit_readiness tests/unit/core/landscape tests/unit/mcp tests/unit/web/audit_readiness
git commit -m "feat(audit): expose sink effect recovery history"
```

## Task 17: Update operator, migration, release, and protocol documentation

**Files:**

- Modify: `docs/architecture/token-scheduler-state-engine.md`
- Modify: `docs/contracts/plugin-protocol.md`
- Create: `docs/runbooks/sink-effect-recovery.md`
- Create: `docs/operator/migrations/epoch-26-sink-effects.md`
- Modify: `docs/reference/configuration.md`
- Modify: current release notes under `docs/release/`
- Test: documentation contract/link tests

- [ ] **Step 1: Write failing documentation assertions**

Add doc tests that require these exact concepts:

```python
REQUIRED_EFFECT_DOC_TERMS = {
    "RESERVED -> PREPARED -> IN_FLIGHT -> FINALIZED",
    "NOT_APPLIED",
    "APPLIED_WITH_EXACT_DESCRIPTOR",
    "UNKNOWN",
    "NO_INSPECTION_REQUIRED",
    "publication_performed",
    "epoch 26",
}
```

Assert the runbook links from the architecture/protocol/migration documents
and no link is broken.

- [ ] **Step 2: Run doc tests and observe missing runbook**

Run:

```bash
.venv/bin/pytest -q tests/unit/docs
```

Expected: new contract test fails because the runbook/migration note is absent.

- [ ] **Step 3: Document the exact operational contract**

The runbook must explain inspection versus reconcile, target predecessor
queues, lease takeover, response-lost attempts, UNKNOWN diagnosis, staging
cleanup, target ledger provisioning, safe repair without speculative commit,
and how to identify a blocked successor. The migration/release note must call
out third-party fail-closed behavior, Chroma skip/error reduction, Database
ledger/DDL restrictions, unsupported filesystems, artifact producer XOR, and
epoch 25 -> 26 behavior. Configuration docs must list byte/row/staging/lock/
network limits.

- [ ] **Step 4: Correct the old architecture claim**

Replace the statement that failure after sink I/O can repeat I/O with the
effect reservation/inspect/plan/commit/reconcile sequence and the remaining
UNKNOWN operator boundary. Document that no production flush follows commit.

- [ ] **Step 5: Run doc/link checks and commit**

Run:

```bash
.venv/bin/pytest -q tests/unit/docs
git diff --check
```

Expected: documentation tests and whitespace checks pass.

Commit:

```bash
git add docs tests/unit/docs
git commit -m "docs: document sink effect recovery"
```

## Task 18: Full verification, independent reviews, and Filigree handoff

**Files:**

- Verify: every file changed since the recorded integration merge-base
- Update: Filigree `elspeth-74a343d5ad` comment only

- [ ] **Step 1: Run focused Landscape/effect tests**

```bash
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_schema.py tests/unit/core/landscape/test_sink_effect_migration.py tests/unit/core/landscape/test_effect_linked_artifacts.py tests/unit/core/landscape/test_sink_effect_identity.py tests/unit/core/landscape/test_sink_effect_reservation.py tests/unit/core/landscape/test_sink_effect_lifecycle.py tests/unit/core/landscape/test_sink_effect_finalization.py
```

Expected: all focused persistence tests pass.

- [ ] **Step 2: Run focused executor and adapter tests**

```bash
.venv/bin/pytest -q tests/unit/engine/test_sink_effect_executor.py tests/unit/engine/test_sink_executor_diversion.py tests/integration/pipeline/test_sink_effect_recovery.py tests/unit/plugins/sinks tests/integration/plugins/sinks
```

Expected: all sink, recovery, diversion, and first-party adapter tests pass.

- [ ] **Step 3: Run every real PostgreSQL proof**

```bash
.venv/bin/pytest -q -m testcontainer tests/testcontainer/core/test_sink_effect_lock_order_postgres.py tests/testcontainer/plugins/test_database_sink_effects_postgres.py tests/testcontainer/web/test_schema_probe_postgres.py
```

Expected: schema parity, target marker, and all forced lock interleavings pass
on PostgreSQL 16 with distinct backend PIDs.

- [ ] **Step 4: Run broad regressions**

```bash
.venv/bin/pytest -q tests/unit/core/landscape tests/unit/engine tests/unit/plugins/sinks tests/integration/checkpoint tests/integration/pipeline tests/integration/plugins/sinks
.venv/bin/pytest -q
```

Expected: the broad selected suite passes. The full default suite may report
only the already-accepted signed-tree failure caused by this branch's real
changes; any other failure blocks completion.

- [ ] **Step 5: Run static and repository gates**

```bash
INTEGRATION_BRANCH=codex/release-0.7.1-worktree
INTEGRATION_BASE=$(git merge-base HEAD "$INTEGRATION_BRANCH")
CHANGED_PY=$(git diff --name-only "$INTEGRATION_BASE"..HEAD -- '*.py')
.venv/bin/mypy $CHANGED_PY
.venv/bin/ruff check $CHANGED_PY
.venv/bin/ruff format --check $CHANGED_PY
.venv/bin/pre-commit run --all-files
git diff --check "$INTEGRATION_BASE"..HEAD
git status --short
```

Expected: mypy, Ruff, pre-commit, and diff checks pass; status is clean.

- [ ] **Step 6: Request independent spec review**

Give the reviewer the approved design, complete branch diff from the recorded
integration merge-base, issue
`elspeth-74a343d5ad`, and exact verification output. Require a requirement-by-
requirement verdict covering crash seams, every built-in, zero-publication,
artifact XOR, migration, and lock races. Fix every blocking finding in its own
TDD commit and rerun the affected gates.

- [ ] **Step 7: Request independent quality review**

Give a separate reviewer the post-spec-review HEAD. Require focus on global
lock order, transaction boundaries, credential-safe evidence, resource bounds,
adapter behavior, and test realism. Fix blocking findings and rerun exact plus
broad gates.

- [ ] **Step 8: Rebase once more on the integration tip and rerun critical proofs**

```bash
git fetch --all --prune
git rebase codex/release-0.7.1-worktree
INTEGRATION_BASE=$(git merge-base HEAD codex/release-0.7.1-worktree)
.venv/bin/pytest -q tests/unit/core/landscape/test_sink_effect_finalization.py tests/unit/engine/test_sink_effect_executor.py tests/integration/pipeline/test_sink_effect_recovery.py
.venv/bin/pytest -q -m testcontainer tests/testcontainer/core/test_sink_effect_lock_order_postgres.py
```

Expected: rebase is clean or conflicts are resolved without weakening the
approved contract; the recorded merge-base is the current integration tip,
and critical SQLite/executor/PostgreSQL proofs pass.

- [ ] **Step 9: Add a complete Filigree evidence comment and keep the issue open**

Run `filigree comment --help`, then use its documented comment verb with actor
`codex-74a` to record commits, exact test counts/commands, the accepted signed-
tree failure, PostgreSQL backend proof, capability reductions, and independent
review verdicts. Do not run `filigree close`; integration/independent acceptance
owns closure.

## Plan completion checklist

- [ ] Every approved design section maps to a task above.
- [ ] Every production change first appears behind a failing test.
- [ ] No sink retains an unfenced production write/flush bypass.
- [ ] Epoch 26 is the only new epoch; epoch 27 remains unused.
- [ ] Current release and 4003 integration dependencies are present.
- [ ] All first-party supported modes have exact commit/reconcile proof.
- [ ] Unsupported third-party/Chroma/Database modes fail before reservation/I/O.
- [ ] Independent spec and quality reviews approve the final HEAD.
- [ ] Filigree remains open with complete evidence.
