"""CI workflow invariants for pytest parallel execution."""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from elspeth.testing.pytest_xdist_auto import pytest_cmdline_main

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yaml"
ACTIONLINT_CONFIG = REPO_ROOT / ".github" / "actionlint.yaml"
JUDGE_GATES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "enforce-allowlist-judge-gates.yaml"
_SHELL_CONTROL_TOKENS = frozenset({"&&", "||", ";", "|"})


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


def _step_index(job: dict[str, Any], step_name: str) -> int:
    for index, step in enumerate(job["steps"]):
        if step.get("name") == step_name:
            return index
    raise AssertionError(f"Missing CI step {step_name!r}")


def _pytest_args(run: str) -> list[str]:
    lexer = shlex.shlex(run.replace("\\\n", " "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    for index in range(len(tokens)):
        if tokens[index : index + 3] == ["uv", "run", "pytest"]:
            start = index + 3
        elif tokens[index : index + 5] == ["uv", "run", "python", "-m", "pytest"]:
            start = index + 5
        else:
            continue

        args: list[str] = []
        for token in tokens[start:]:
            if token in _SHELL_CONTROL_TOKENS:
                break
            args.append(token)
        return args
    raise AssertionError("Missing pytest invocation")


def _pytest_numprocesses_values(run: str) -> list[str]:
    values: list[str] = []
    args = _pytest_args(run)
    for index, arg in enumerate(args):
        if arg == "-n" and index + 1 < len(args):
            values.append(args[index + 1])
        elif arg.startswith("-n") and arg != "-n":
            values.append(arg[2:])
        elif arg == "--numprocesses" and index + 1 < len(args):
            values.append(args[index + 1])
        elif arg.startswith("--numprocesses="):
            values.append(arg.split("=", 1)[1])
    return values


@pytest.mark.parametrize(
    ("flag_args", "expected"),
    (
        ("-n0", "0"),
        ("-n 0", "0"),
        ("--numprocesses 0", "0"),
        ("--numprocesses=0", "0"),
        ("-nauto", "auto"),
        ("-n auto", "auto"),
        ("--numprocesses auto", "auto"),
        ("--numprocesses=auto", "auto"),
    ),
)
def test_pytest_numprocesses_values_tokenizes_supported_cli_forms(flag_args: str, expected: str) -> None:
    run = f"uv run pytest tests/ {flag_args}"

    assert _pytest_numprocesses_values(run) == [expected]


@pytest.mark.parametrize(
    ("run", "expected_args"),
    (
        ("uv run pytest tests/ -n 0|| status=$?", ["tests/", "-n", "0"]),
        ("uv run pytest tests/ -n auto&& echo done", ["tests/", "-n", "auto"]),
        ("uv run pytest tests/ --numprocesses=auto; echo done", ["tests/", "--numprocesses=auto"]),
    ),
)
def test_pytest_args_stop_at_attached_shell_control_operators(run: str, expected_args: list[str]) -> None:
    assert _pytest_args(run) == expected_args


def test_python_matrix_ci_does_not_hard_disable_xdist() -> None:
    """Remote Python test lanes must leave xdist available instead of forcing ``-n 0``."""
    workflow = _ci_workflow()
    test_job = workflow["jobs"]["test"]

    coverage_run = _step_run(test_job, "Run tests with coverage")
    no_coverage_run = _step_run(test_job, "Run tests without coverage")

    assert "0" not in _pytest_numprocesses_values(coverage_run)
    assert "0" not in _pytest_numprocesses_values(no_coverage_run)


def test_integration_lane_does_not_force_parallel_xdist() -> None:
    """Integration lane stays sequential by omitting explicit xdist process flags."""
    workflow = _ci_workflow()
    integration_job = workflow["jobs"]["integration"]

    run = _step_run(integration_job, "Run integration tests")

    assert _pytest_numprocesses_values(run) == []


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

    for job_name in ("check-override-rate",):
        job = judge_workflow["jobs"][job_name]
        assert job["timeout-minutes"] == 15


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


def test_integration_job_runs_on_rc_branch_pushes() -> None:
    """RC branch pushes must not skip the integration lane."""
    workflow = _ci_workflow()
    integration_job = workflow["jobs"]["integration"]

    condition = integration_job["if"]

    assert "github.event_name == 'push'" in condition
    assert "refs/heads/main" in condition
    assert "startsWith(github.ref, 'refs/heads/RC')" in condition


def test_integration_lane_fails_closed_on_real_test_failures() -> None:
    """A real integration failure must fail the lane.

    The historical ``... || echo "Integration tests skipped (no API keys)"``
    swallowed *every* non-zero pytest exit — assertion regressions, collection
    errors, import failures, and infra faults all left the job green, and
    ``build-push.yaml`` would then build an image off a broken CI run. The lane
    must propagate real failures and tolerate only pytest's exit code 5 ("no
    tests collected").
    """
    workflow = _ci_workflow()
    integration_job = workflow["jobs"]["integration"]
    run = _step_run(integration_job, "Run integration tests")

    # The blanket failure-swallow must be gone.
    assert "|| echo" not in run
    # Real failures propagate via the captured status.
    assert 'exit "$status"' in run
    # Only "no tests collected" (pytest exit 5) is tolerated as a skip.
    assert "-eq 5" in run


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


def test_static_analysis_runs_composer_skill_inventory_drift_gate() -> None:
    """Generated composer skill inventory must be checked in CI, not only pre-commit."""
    workflow = _ci_workflow()
    static_analysis = workflow["jobs"]["static-analysis"]

    run = _step_run(static_analysis, "Check composer skill tool inventory")

    assert "scripts/cicd/generate_skill_inventory.py --check" in run


def test_static_analysis_runs_actionlint_with_repo_policy_config() -> None:
    """Workflow syntax and self-hosted runner labels must be checked in CI."""
    workflow = _ci_workflow()
    static_analysis = workflow["jobs"]["static-analysis"]

    step = _step(static_analysis, "Check GitHub workflows (actionlint)")
    assert step["shell"] == "bash"
    env = step.get("env")
    assert isinstance(env, dict), "actionlint step must pin version and checksum"
    assert env["ACTIONLINT_VERSION"] == "1.7.12"
    assert env["ACTIONLINT_SHA256"] == "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"

    run = _step_run(static_analysis, "Check GitHub workflows (actionlint)")
    assert "sha256sum -c -" in run
    assert "-config-file .github/actionlint.yaml" in run
    assert ".github/workflows/*.yml" in run
    assert ".github/workflows/*.yaml" in run


def test_actionlint_policy_declares_self_hosted_runner_labels() -> None:
    policy = _workflow(ACTIONLINT_CONFIG)

    labels = policy["self-hosted-runner"]["labels"]

    assert {"nyx-ci", "trusted"} <= set(labels)


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


def test_static_analysis_fork_prs_reject_unverified_signed_allowlist_edits() -> None:
    """Fork PRs must not use keyless shape-only CI to forge signed allowlist entries."""
    workflow = _ci_workflow()
    static_analysis = workflow["jobs"]["static-analysis"]

    step = _step(static_analysis, "Reject unverified fork PR signed allowlist edits")
    assert (
        step.get("if") == "${{ github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name != github.repository }}"
    )

    run = _step_run(static_analysis, "Reject unverified fork PR signed allowlist edits")
    assert "git fetch --no-tags --depth=1 origin ${{ github.event.pull_request.base.sha }}" in run
    assert "check-judge-coverage" in run
    assert "--forbid-unverified-judge-metadata" in run
    assert "--allowlist-root config/cicd/enforce_tier_model" in run
    assert "--allowlist-root config/cicd/enforce_trust_boundary_honesty" in run
    assert "--baseline-ref ${{ github.event.pull_request.base.sha }}" in run

    gate_index = _step_index(static_analysis, "Reject unverified fork PR signed allowlist edits")
    assert gate_index < _step_index(static_analysis, "Run trust-tier elspeth-lints rule")
    assert gate_index < _step_index(static_analysis, "Run trust-boundary honesty-gate elspeth-lints rules")


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
