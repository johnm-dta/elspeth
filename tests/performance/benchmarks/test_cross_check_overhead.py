"""NFR gate for the pass-through runtime cross-check (ADR-008 / ADR-009).

Measures the per-row overhead of the live
``engine.executors.pass_through.verify_pass_through`` function on a 200-field
input row. Budget: median ≤ 25 µs, P99 (mean+3*stddev proxy) ≤ 50 µs.

A second benchmark covers the batch-flush path (ADR-009 §Clause 2) —
``RowProcessor._cross_check_flush_output`` calls ``verify_pass_through`` once
per emitted row with a batch-homogeneous ``input_fields``. The budget scales
with batch size: median ≤ 1500 µs, P99 ≤ 3000 µs for a 64-row batch.

Both benchmarks import and invoke the real function, not an inlined copy. An
inlined benchmark could pass while the live function regresses — the whole
point of this gate is to catch drift in the live code path.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

# Importing pass_through ensures PassThroughDeclarationContract registers
# itself via its module-level side-effect before any benchmark runs.
import elspeth.engine.executors.pass_through  # noqa: F401
from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    DeclarationContractViolation,
    ExampleBundle,
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    implements_dispatch_site,
    register_declaration_contract,
    registered_declaration_contracts,
)
from elspeth.contracts.errors import PassThroughContractViolation
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.engine.executors.declaration_dispatch import run_post_emission_checks
from elspeth.engine.executors.pass_through import (
    PassThroughDeclarationContract,
    verify_pass_through,
)
from elspeth.testing import make_field


def _stable_tail_bound_sec(stats: Any) -> float:
    """Return a robust upper-tail bound for microsecond benchmarks.

    Host scheduler spikes can add millisecond-scale single outliers to an
    otherwise stable microbenchmark. ``q3 + 3*IQR`` still fails when the normal
    tail widens, but does not let one unrelated host spike dominate the gate.
    """
    return stats["q3"] + 3.0 * stats["iqr"]


def _build_wide_contract(n_fields: int = 200) -> SchemaContract:
    return SchemaContract(
        fields=tuple(
            make_field(
                f"field_{i}",
                python_type=str,
                original_name=f"field_{i}",
                required=True,
                source="declared",
            )
            for i in range(n_fields)
        ),
        mode="FLEXIBLE",
        locked=True,
    )


@pytest.mark.performance
def test_cross_check_p99_within_budget(benchmark: pytest.FixtureRequest) -> None:
    """Median ≤ 25 µs, P99 ≤ 50 µs on a 200-field row — single-token path.

    Calls the live ``verify_pass_through`` function (not an inlined copy) so
    the gate actually guards the code executed in production. Inputs are
    constructed so the contract holds and the function returns None —
    measuring the happy path, which is the dominant case under
    ``passes_through_input=True`` transforms that honour their contract.
    """
    input_contract = _build_wide_contract(200)
    output_contract = _build_wide_contract(200)

    input_row = PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, input_contract)
    output_row = PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, output_contract)
    input_fields = frozenset(fc.normalized_name for fc in input_row.contract.fields)
    emitted_rows = [output_row]

    def run_cross_check() -> None:
        verify_pass_through(
            input_fields=input_fields,
            emitted_rows=emitted_rows,
            can_drop_rows=False,
            static_contract=input_fields,
            transform_name="bench",
            transform_node_id="bench_node",
            run_id="bench_run",
            row_id="bench_row",
            token_id="bench_token",
        )

    benchmark(run_cross_check)

    stats = benchmark.stats
    median_sec = stats["median"]
    mean_sec = stats["mean"]
    stddev_sec = stats["stddev"]
    p99_bound = mean_sec + 3 * stddev_sec
    assert median_sec < 25e-6, f"Median {median_sec * 1e6:.1f}us exceeds 25us budget"
    assert p99_bound < 50e-6, (
        f"Mean+3*stddev {p99_bound * 1e6:.1f}us exceeds 50us budget (mean={mean_sec * 1e6:.1f}us, stddev={stddev_sec * 1e6:.1f}us)"
    )


@pytest.mark.performance
def test_cross_check_raises_on_drop(benchmark: pytest.FixtureRequest) -> None:
    """Sanity: the live function detects a dropped field.

    Exercises the payload-drop vector (contract reused, payload shrunk by one
    field). The buggy-plugin vector — contract still declares field_199 but
    payload omits it. ``verify_pass_through`` raises
    ``PassThroughContractViolation``; the benchmark catches it to include the
    raise path in the timing. The assertion confirms the violation's
    ``divergence_set`` is exactly the dropped field.
    """
    input_contract = _build_wide_contract(200)
    input_row = PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, input_contract)
    # Output reuses the contract but drops field_199 from payload.
    output_row = PipelineRow({f"field_{i}": f"v{i}" for i in range(199)}, input_contract)
    input_fields = frozenset(fc.normalized_name for fc in input_row.contract.fields)
    emitted_rows = [output_row]

    captured: dict[str, frozenset[str]] = {}

    def run_cross_check() -> None:
        try:
            verify_pass_through(
                input_fields=input_fields,
                emitted_rows=emitted_rows,
                can_drop_rows=False,
                static_contract=input_fields,
                transform_name="bench",
                transform_node_id="bench_node",
                run_id="bench_run",
                row_id="bench_row",
                token_id="bench_token",
            )
        except PassThroughContractViolation as violation:
            captured["divergence"] = violation.divergence_set

    benchmark(run_cross_check)
    assert captured.get("divergence") == frozenset({"field_199"})


@pytest.mark.performance
def test_batch_flush_cross_check_within_budget(benchmark: pytest.FixtureRequest) -> None:
    """Batch-flush path: median ≤ 1500 µs, P99 ≤ 3000 µs for 64-row batch.

    Models the ADR-009 §Clause 2 batch-aware path: ``verify_pass_through`` is
    invoked once per emitted row with ``input_fields`` computed as the
    intersection of all buffered input contracts (batch-homogeneous). The
    benchmark includes the intersection computation plus the per-row
    verification — the full flush-path cross-check cost for the operator.

    Budget rationale: single-token happy path ≤ 25 µs median; 64 rows ≈
    1600 µs upper bound with small intersection cost on top. Setting the
    budget at 1500 µs median / 3000 µs P99 gives headroom for the
    intersection (cheap) and any scheduler noise.
    """
    input_contract = _build_wide_contract(200)
    output_contract = _build_wide_contract(200)

    batch_size = 64
    buffered_rows = [PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, input_contract) for _ in range(batch_size)]
    emitted_rows = [PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, output_contract) for _ in range(batch_size)]
    static_contract = frozenset(fc.normalized_name for fc in input_contract.fields)

    def run_batch_cross_check() -> None:
        # Batch-homogeneous input_fields — intersection across every
        # buffered input contract. Measures the primitive cost; the live
        # ``_cross_check_flush_output`` path additionally routes through
        # ``run_runtime_checks`` and ``PassThroughDeclarationContract`` (ADR-010
        # §Decision 3), which adds the dispatcher-overhead benchmark's
        # measured overhead (~15 µs median) on top — well within the 1500 µs
        # budget for a 64-row batch.
        per_input_field_sets = [frozenset(fc.normalized_name for fc in row.contract.fields) for row in buffered_rows]
        input_fields = frozenset.intersection(*per_input_field_sets)
        verify_pass_through(
            input_fields=input_fields,
            emitted_rows=emitted_rows,
            can_drop_rows=False,
            static_contract=static_contract,
            transform_name="bench_batch",
            transform_node_id="bench_batch_node",
            run_id="bench_run",
            row_id="bench_row",
            token_id="bench_token",
        )

    benchmark(run_batch_cross_check)

    stats = benchmark.stats
    median_sec = stats["median"]
    mean_sec = stats["mean"]
    stddev_sec = stats["stddev"]
    p99_bound = mean_sec + 3 * stddev_sec
    assert median_sec < 1500e-6, f"Batch-flush median {median_sec * 1e6:.1f}us exceeds 1500us budget"
    assert p99_bound < 3000e-6, (
        f"Batch-flush mean+3*stddev {p99_bound * 1e6:.1f}us exceeds 3000us budget "
        f"(mean={mean_sec * 1e6:.1f}us, stddev={stddev_sec * 1e6:.1f}us)"
    )


@pytest.mark.performance
@pytest.mark.benchmark(group="dispatcher-overhead")
def test_dispatcher_overhead_vs_direct_verify_pass_through(benchmark: pytest.FixtureRequest) -> None:
    """Reviewer O2/O7: quantify per-row dispatcher overhead against the live registry.

    The dispatcher adds one ``registered_declaration_contracts()`` call + one
    ``applies_to()`` short-circuit + one ``runtime_check()`` invocation on top
    of the direct ``verify_pass_through`` call.

    The budget is derived from the current production registry cardinality:

        budget_median(N) = 27 us (N=1 baseline) + (N - 1) * 1.5 us

    where N is ``len(registered_declaration_contracts())`` at runtime. This
    keeps the benchmark honest as production adopters land. The parametrised
    ``test_dispatcher_overhead_scales_with_n`` companion below still probes the
    wider N∈{1,2,4,8,16} envelope.

    Setup mirrors the direct-call benchmark: a 200-field input row with a
    matching 200-field output row so the pass-through contract holds and the
    happy path is exercised. The plugin stub has ``passes_through_input=True``
    so ``PassThroughDeclarationContract.applies_to`` returns True and
    ``runtime_check`` is invoked.
    """

    class _PassThroughPlugin:
        """Minimal plugin stub satisfying PassThroughDeclarationContract.applies_to."""

        name = "bench_dispatcher"
        node_id = "bench_dispatcher_node"
        passes_through_input = True
        can_drop_rows = False
        declared_output_fields = frozenset()
        declared_input_fields = frozenset()
        is_batch_aware = False
        _output_schema_config = None

    input_contract = _build_wide_contract(200)
    output_contract = _build_wide_contract(200)

    input_row = PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, input_contract)
    output_row = PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, output_contract)
    static_contract = frozenset(fc.normalized_name for fc in input_contract.fields)

    plugin = _PassThroughPlugin()
    effective_input_fields = frozenset(fc.normalized_name for fc in input_row.contract.fields)
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id,
        run_id="bench_run",
        row_id="bench_row",
        token_id="bench_token",
        input_row=input_row,
        static_contract=static_contract,
        effective_input_fields=effective_input_fields,
    )
    outputs = PostEmissionOutputs(emitted_rows=(output_row,))

    def run_dispatcher() -> None:
        run_post_emission_checks(inputs=inputs, outputs=outputs)

    benchmark(run_dispatcher)

    stats = benchmark.stats
    median_sec = stats["median"]
    mean_sec = stats["mean"]
    stddev_sec = stats["stddev"]
    p99_bound = mean_sec + 3 * stddev_sec
    live_contract_count = len(registered_declaration_contracts())
    median_budget_us = 27.0 + (live_contract_count - 1) * 1.5
    p99_budget_us = median_budget_us * 2.0
    assert median_sec * 1e6 < median_budget_us, (
        f"Dispatcher median {median_sec * 1e6:.1f}us exceeds {median_budget_us:.1f}us budget at live N={live_contract_count}"
    )
    assert p99_bound * 1e6 < p99_budget_us, (
        f"Dispatcher mean+3*stddev {p99_bound * 1e6:.1f}us exceeds {p99_budget_us:.1f}us "
        f"budget at live N={live_contract_count} "
        f"(mean={mean_sec * 1e6:.1f}us, stddev={stddev_sec * 1e6:.1f}us)"
    )


# =============================================================================
# H1 — dispatcher-overhead scales with N (issue elspeth-5dae105959)
# =============================================================================
#
# The original dispatcher-overhead benchmark measured the live registry
# (N=1 at the time). The first N-scaling fix only added N-1 skipped contracts,
# which guarded skip-cost but still hid the real production shape: several
# contracts can apply to the same row and each applicable contract performs
# its own runtime observation work.
#
# The parametrised test below varies BOTH N_registered and M_applicable. It
# registers one real PassThroughDeclarationContract, M-1 additional applicable
# contracts that perform production-like contract/payload field observation,
# and N-M skipped contracts. The budget scales separately for apply and skip
# paths so the NFR fails when either law drifts.


class _NoopPayload(TypedDict):
    reason: str


class _SkipContract(DeclarationContract):
    """applies_to-returns-False stub for dispatcher scaling benchmarks.

    Ties up one registry slot without contributing any post_emission_check
    cost so the benchmark isolates the per-registered-contract dispatch-loop
    cost (tuple iteration + applies_to attribute read) from
    verify_pass_through's O(fields) work.
    """

    payload_schema: type = _NoopPayload

    def __init__(self, name: str) -> None:
        self.name = name

    def applies_to(self, plugin: Any) -> bool:
        return False

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        # Unreachable because applies_to is always False; kept for ABC
        # compliance.
        raise NotImplementedError

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        raise NotImplementedError

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        raise NotImplementedError


class _ApplicablePayload(TypedDict):
    reason: str
    missing: list[str]


class _ApplicableViolation(DeclarationContractViolation):
    payload_schema: type = _ApplicablePayload


class _ApplicableContract(DeclarationContract):
    """applies_to-True contract for dispatcher apply-cost benchmarks.

    This deliberately does real runtime observation work (contract fields ∩
    payload keys) over the same 200-field row as PassThroughDeclarationContract.
    The happy path returns normally; the violation path is present so the
    contract has the same audit-evidence shape as production declaration
    contracts if the benchmark fixture drifts.
    """

    payload_schema: type = _ApplicablePayload

    def __init__(self, name: str) -> None:
        self.name = name

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        for emitted in outputs.emitted_rows:
            runtime_contract_fields = frozenset(fc.normalized_name for fc in emitted.contract.fields)
            runtime_payload_fields = frozenset(emitted.keys())
            runtime_observed = runtime_contract_fields & runtime_payload_fields
            missing = inputs.effective_input_fields - runtime_observed
            if missing:
                raise _ApplicableViolation(
                    plugin=inputs.plugin.name,
                    node_id=inputs.node_id,
                    run_id=inputs.run_id,
                    row_id=inputs.row_id,
                    token_id=inputs.token_id,
                    payload={
                        "reason": "applicable benchmark contract observed missing fields",
                        "missing": sorted(missing),
                    },
                    message=f"Applicable benchmark contract {self.name!r} observed missing fields {sorted(missing)!r}.",
                )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        raise NotImplementedError

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        raise NotImplementedError


@pytest.mark.performance
@pytest.mark.benchmark(group="dispatcher-overhead-scaling")
@pytest.mark.parametrize(
    "n_registered,m_applicable",
    [
        (1, 1),
        (2, 1),
        (4, 2),
        (8, 4),
        (16, 5),
    ],
)
def test_dispatcher_overhead_scales_with_registered_and_applicable_contracts(
    benchmark: pytest.FixtureRequest,
    n_registered: int,
    m_applicable: int,
) -> None:
    """NFR derivation (ADR-010 §Consequences, issue elspeth-5dae105959 / H1):

    per-row dispatcher overhead at (N registered, M applicable) contracts is
    bounded by

        budget_median(N, M) =
            27 us pass-through baseline
          + (M - 1) * 25 us per additional applicable contract
          + (N - M) * 1.5 us per skipped contract

    where 1.5 us is the applies_to short-circuit cost per skipped contract
    with generous headroom for CI jitter on CPython 3.13 / Linux x86-64, and
    25 us is the established pass-through happy-path budget for one
    applicable 200-field contract. The tail proxy is q3 + 3*IQR so isolated
    host scheduler spikes do not masquerade as dispatcher overhead.

    The fixed 2 µs at N=1 is the dispatcher's own structural cost —
    ``registered_declaration_contracts()`` tuple materialisation + loop
    enter — and does not scale with N.

    The benchmark registers one real PassThroughDeclarationContract, M-1
    additional applicable contracts that perform 200-field runtime
    observation, and N-M skipped contracts.

    Registry state is snapshot/restored so the benchmark does not leak into
    sibling tests.
    """
    assert 1 <= m_applicable <= n_registered

    class _PassThroughPlugin:
        name = "bench_dispatcher_scaling"
        node_id = "bench_dispatcher_scaling_node"
        passes_through_input = True
        can_drop_rows = False
        declared_output_fields = frozenset()
        declared_input_fields = frozenset()
        is_batch_aware = False
        _output_schema_config = None

    input_contract = _build_wide_contract(200)
    output_contract = _build_wide_contract(200)

    input_row = PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, input_contract)
    output_row = PipelineRow({f"field_{i}": f"v{i}" for i in range(200)}, output_contract)
    static_contract = frozenset(fc.normalized_name for fc in input_contract.fields)

    plugin = _PassThroughPlugin()
    effective_input_fields = frozenset(fc.normalized_name for fc in input_row.contract.fields)
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id,
        run_id="bench_run",
        row_id="bench_row",
        token_id="bench_token",
        input_row=input_row,
        static_contract=static_contract,
        effective_input_fields=effective_input_fields,
    )
    outputs = PostEmissionOutputs(emitted_rows=(output_row,))

    snapshot = _snapshot_registry_for_tests()
    try:
        _clear_registry_for_tests()
        # Real pass-through contract — the one that applies + invokes
        # post_emission_check. The 200-field verify_pass_through dominates
        # the measurement and anchors the N=1 baseline.
        register_declaration_contract(PassThroughDeclarationContract())
        # M-1 applicable contracts — exercise the runtime-check path instead
        # of only the applies_to short-circuit.
        for i in range(m_applicable - 1):
            register_declaration_contract(_ApplicableContract(name=f"applicable_scaling_{i}"))
        # N-M skipped contracts — exercise only the applies_to short-circuit
        # path. Names are distinct to satisfy registry uniqueness.
        for i in range(n_registered - m_applicable):
            register_declaration_contract(_SkipContract(name=f"skip_scaling_{i}"))

        def run_dispatcher() -> None:
            run_post_emission_checks(inputs=inputs, outputs=outputs)

        benchmark(run_dispatcher)
    finally:
        _restore_registry_snapshot_for_tests(snapshot)

    stats = benchmark.stats
    median_sec = stats["median"]
    stable_tail_bound_sec = _stable_tail_bound_sec(stats)

    # Scaling formula (see docstring): separate apply-cost from skip-cost so
    # the benchmark cannot pass by measuring only cheap applies_to=False slots.
    median_budget_us = 27.0 + (m_applicable - 1) * 25.0 + (n_registered - m_applicable) * 1.5
    tail_budget_us = median_budget_us * 2.0
    assert median_sec * 1e6 < median_budget_us, (
        f"Dispatcher median {median_sec * 1e6:.1f}us exceeds {median_budget_us:.1f}us "
        f"budget at N_registered={n_registered}, M_applicable={m_applicable} "
        f"(27us baseline + {m_applicable - 1} * 25us per-apply + "
        f"{n_registered - m_applicable} * 1.5us per-skip; ADR-010 §NFR derivation)"
    )
    assert stable_tail_bound_sec * 1e6 < tail_budget_us, (
        f"Dispatcher stable tail {stable_tail_bound_sec * 1e6:.1f}us exceeds {tail_budget_us:.1f}us "
        f"tail budget at N_registered={n_registered}, M_applicable={m_applicable} "
        f"(q3={stats['q3'] * 1e6:.1f}us, iqr={stats['iqr'] * 1e6:.1f}us)"
    )
