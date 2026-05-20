"""CI workflow invariants for pytest parallel execution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from elspeth.testing.pytest_xdist_auto import pytest_cmdline_main

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yaml"
JUDGE_GATES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "enforce-allowlist-judge-gates.yaml"


def _workflow(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), f"{path.name} workflow YAML root must be a mapping"
    return raw


def _ci_workflow() -> dict[str, Any]:
    return _workflow(CI_WORKFLOW)


def _step_run(job: dict[str, Any], step_name: str) -> str:
    step = _step(job, step_name)
    run = step.get("run")
    assert isinstance(run, str), f"{step_name!r} must have a shell run block"
    return run


def _step(job: dict[str, Any], step_name: str) -> dict[str, Any]:
    for step in job["steps"]:
        if step.get("name") == step_name:
            assert isinstance(step, dict), f"{step_name!r} must be a mapping"
            return step
    raise AssertionError(f"Missing CI step {step_name!r}")


def test_python_matrix_ci_does_not_hard_disable_xdist() -> None:
    """Remote Python test lanes must leave xdist available instead of forcing ``-n0``."""
    workflow = _ci_workflow()
    test_job = workflow["jobs"]["test"]

    coverage_run = _step_run(test_job, "Run tests with coverage")
    no_coverage_run = _step_run(test_job, "Run tests without coverage")

    assert "-n0" not in coverage_run
    assert "--numprocesses=0" not in coverage_run
    assert "-n0" not in no_coverage_run
    assert "--numprocesses=0" not in no_coverage_run


def test_python_matrix_documents_coverage_lane_choice() -> None:
    """The 3.12/3.13 coverage split must carry its rationale in the workflow."""
    workflow_text = CI_WORKFLOW.read_text(encoding="utf-8")
    normalized = " ".join(workflow_text.split())

    assert "Coverage runs on Python 3.13 only" in normalized
    assert "Python 3.12 lane remains" in normalized
    assert "dependency-compatibility signal" in normalized


def test_judge_gates_workflow_mirrors_ci_concurrency_policy() -> None:
    """Policy-gate workflow must not race push and PR runs for one ref."""
    ci_workflow = _ci_workflow()
    judge_workflow = _workflow(JUDGE_GATES_WORKFLOW)

    assert judge_workflow["concurrency"] == ci_workflow["concurrency"]


def test_judge_gates_workflow_has_bounded_job_timeouts() -> None:
    """Judge-gate jobs must not inherit GitHub's six-hour default timeout."""
    judge_workflow = _workflow(JUDGE_GATES_WORKFLOW)

    for job_name in ("check-judge-coverage", "check-override-rate"):
        job = judge_workflow["jobs"][job_name]
        assert job["timeout-minutes"] == 15


def test_judge_coverage_workflow_resolves_true_merge_base() -> None:
    """C1 must diff against the actual PR merge-base, not the PR base tip SHA."""
    judge_workflow = _workflow(JUDGE_GATES_WORKFLOW)
    job = judge_workflow["jobs"]["check-judge-coverage"]

    run = _step_run(job, "Resolve baseline ref")

    assert "git merge-base" in run
    assert "github.event.pull_request.base.sha" not in run


def test_judge_coverage_workflow_runs_on_pushes_with_before_sha_baseline() -> None:
    """C1 must not disappear on direct pushes to protected release branches."""
    judge_workflow = _workflow(JUDGE_GATES_WORKFLOW)
    job = judge_workflow["jobs"]["check-judge-coverage"]

    assert "if" not in job
    runs_on = str(job["runs-on"])
    assert "github.event_name != 'pull_request'" in runs_on

    run = _step_run(job, "Resolve baseline ref")

    assert "github.event.before" in run
    assert "github.event_name" in run
    assert "git merge-base" in run


def test_override_rate_workflow_pins_threshold_policy() -> None:
    """C3 threshold is CI policy and must be explicit in workflow YAML."""
    judge_workflow = _workflow(JUDGE_GATES_WORKFLOW)
    job = judge_workflow["jobs"]["check-override-rate"]

    run = _step_run(job, "Run check-override-rate")

    assert "--max-rate 0.10" in run


def test_override_rate_workflow_surfaces_pass_notice_in_step_summary() -> None:
    """C3 PASS/insufficient-data notices must be visible outside raw job logs."""
    judge_workflow = _workflow(JUDGE_GATES_WORKFLOW)
    job = judge_workflow["jobs"]["check-override-rate"]

    run = _step_run(job, "Run check-override-rate")

    assert "GITHUB_STEP_SUMMARY" in run
    assert "Override-rate drift gate" in run


def test_judge_coverage_workflow_summarizes_trust_boundary_decorator_additions() -> None:
    """PRs adding @trust_boundary decorators must surface them in the step summary."""
    judge_workflow = _workflow(JUDGE_GATES_WORKFLOW)
    job = judge_workflow["jobs"]["check-judge-coverage"]

    run = _step_run(job, "Summarize trust-boundary decorator additions")

    assert "check-trust-boundary-diff" in run
    assert "GITHUB_STEP_SUMMARY" in run
    assert '--baseline-ref "$BASELINE_REF"' in run
    assert "--root src/elspeth" in run


def test_judge_coverage_workflow_checks_rotation_audit_manifest() -> None:
    """Fingerprint rotations in allowlist YAML must have a rotations.log record."""
    judge_workflow = _workflow(JUDGE_GATES_WORKFLOW)
    job = judge_workflow["jobs"]["check-judge-coverage"]

    run = _step_run(job, "Run rotation audit manifest gate")

    assert "check-rotation-audit" in run
    assert '--baseline-ref "$BASELINE_REF"' in run
    assert "--rotation-log .elspeth/rotations.log" in run


def test_integration_job_runs_on_rc_branch_pushes() -> None:
    """RC branch pushes must not skip the integration lane."""
    workflow = _ci_workflow()
    integration_job = workflow["jobs"]["integration"]

    condition = integration_job["if"]

    assert "github.event_name == 'push'" in condition
    assert "refs/heads/main" in condition
    assert "startsWith(github.ref, 'refs/heads/RC')" in condition


def test_xdist_auto_defaults_to_parallel_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local pytest runs default to xdist when no process count is explicit."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses == "auto"


def test_xdist_auto_noops_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI controllers stay sequential for clearer failure output."""
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses is None


def test_xdist_auto_stays_sequential_for_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage jobs stay sequential because pytest-cov owns worker coordination."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(cov_source=["src/elspeth"], numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses is None


def test_xdist_auto_noops_inside_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers must not recursively auto-enable xdist."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses is None


def test_xdist_auto_preserves_explicit_process_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit pytest ``-n`` choices remain authoritative."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=4))

    pytest_cmdline_main(config)  # type: ignore[arg-type]

    assert config.option.numprocesses == 4


def test_postgres_testcontainer_ci_stays_sequential() -> None:
    """The Docker-backed lane is intentionally not fanned out across xdist workers."""
    workflow = _ci_workflow()
    testcontainer_job = workflow["jobs"]["test-postgres-testcontainer"]

    run = _step_run(testcontainer_job, "Run testcontainer tests")

    assert "-n auto" not in run


def test_postgres_testcontainer_lane_gates_ci_success() -> None:
    """Postgres testcontainer regressions must fail the aggregate CI result."""
    workflow = _ci_workflow()
    ci_success = workflow["jobs"]["ci-success"]

    assert "test-postgres-testcontainer" in ci_success["needs"]

    run = _step_run(ci_success, "Check all jobs passed")

    assert "needs.test-postgres-testcontainer.result" in run
    assert "Postgres testcontainer job failed" in run


def test_static_analysis_runs_composer_skill_inventory_drift_gate() -> None:
    """Generated composer skill inventory must be checked in CI, not only pre-commit."""
    workflow = _ci_workflow()
    static_analysis = workflow["jobs"]["static-analysis"]

    run = _step_run(static_analysis, "Check composer skill tool inventory")

    assert "scripts/cicd/generate_skill_inventory.py --check" in run


def test_static_analysis_signed_allowlist_steps_handle_trusted_and_fork_prs() -> None:
    """Signed allowlist loaders verify with secrets when available and degrade for forks."""
    workflow = _ci_workflow()
    static_analysis = workflow["jobs"]["static-analysis"]
    expected_secret = "${{ secrets.ELSPETH_JUDGE_METADATA_HMAC_KEY }}"

    for step_name in (
        "Run trust-tier elspeth-lints rule",
        "Run trust-boundary honesty-gate elspeth-lints rules",
        "Emit elspeth-lints trust-tier SARIF artifact",
    ):
        step = _step(static_analysis, step_name)
        env = step.get("env")
        assert isinstance(env, dict), f"{step_name!r} must define step env"
        assert env.get("ELSPETH_JUDGE_METADATA_HMAC_KEY") == expected_secret
        verify_mode = env.get("ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE")
        assert isinstance(verify_mode, str), f"{step_name!r} must define signature verification mode"
        assert "github.event_name == 'pull_request'" in verify_mode
        assert "github.event.pull_request.head.repo.full_name != github.repository" in verify_mode
        assert "shape-only-when-key-missing" in verify_mode
        assert "required" in verify_mode


def test_trust_tier_ci_failure_points_to_signature_diagnosis_command() -> None:
    """Signed allowlist failures should point operators at the repair triage command."""
    workflow = _ci_workflow()
    static_analysis = workflow["jobs"]["static-analysis"]

    trust_tier_run = _step_run(static_analysis, "Run trust-tier elspeth-lints rule")
    sarif_run = _step_run(static_analysis, "Emit elspeth-lints trust-tier SARIF artifact")

    for run in (trust_tier_run, sarif_run):
        assert "diagnose-judge-signatures --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model" in run
        assert "sign-judge-signatures --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model" in run
        assert "--env-file /path/to/operator.env --owner" in run
        assert "judge_metadata_signature" in run
        assert "scope_fingerprint" in run
