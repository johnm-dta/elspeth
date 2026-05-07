"""Shell-level regressions for the composer eval harness helpers."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
AGENT_PROMPT = REPO_ROOT / "evals" / "composer-harness" / "AGENT_PROMPT.md"
AGGREGATE_SCRIPT = REPO_ROOT / "evals" / "composer-harness" / "hardmode" / "aggregate.sh"
FINALIZE_SCRIPT = REPO_ROOT / "evals" / "composer-harness" / "hardmode" / "finalize_scenario.sh"
HARNESS_SCRIPT = REPO_ROOT / "evals" / "composer-harness" / "hardmode" / "harness.sh"
POST_MESSAGE_SCRIPT = REPO_ROOT / "evals" / "composer-harness" / "hardmode" / "post_message.sh"
SWEEP_SCRIPT = REPO_ROOT / "evals" / "composer-harness" / "hardmode" / "sweep_simplified.sh"
LEGACY_BASIC_FINALIZE_SCRIPT = REPO_ROOT / "evals" / "2026-05-03-composer" / "basic" / "finalize_scenario.sh"
LEGACY_HARDMODE_FINALIZE_SCRIPT = REPO_ROOT / "evals" / "2026-05-03-composer" / "hardmode" / "finalize_scenario.sh"


def _write_valid_jwt(path: Path) -> None:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 4_102_444_800}).encode()).decode().rstrip("=")
    path.write_text(f"header.{payload}.signature")


def _write_fake_curl(bin_dir: Path) -> None:
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

out=""
url=""
while (( $# > 0 )); do
  case "$1" in
    -o)
      out="$2"
      shift 2
      ;;
    -w|-X|-H|--max-time|--data|--data-binary|-d)
      shift 2
      ;;
    --*)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      url="$1"
      shift
      ;;
  esac
done

if [[ -z "$out" ]]; then
  echo "fake curl expected -o <path>" >&2
  exit 2
fi

case "$url" in
  */validate)
    printf '{"detail":"expired token"}' > "$out"
    printf '401'
    ;;
  */state/yaml)
    printf '{}' > "$out"
    printf '200'
    ;;
  */messages)
    printf '[]' > "$out"
    printf '200'
    ;;
  *)
    printf '{}' > "$out"
    printf '200'
    ;;
esac
"""
    )
    fake_curl.chmod(0o755)


def _write_fake_finalize_curl(
    bin_dir: Path,
    *,
    state_yaml_http: str = "404",
    valid: bool = False,
    llm_audit_cost: str | None = None,
) -> None:
    fake_curl = bin_dir / "curl"
    validate_body = (
        '{"is_valid": true, "checks": []}'
        if valid
        else '{"is_valid": false, "checks": [{"name": "state_exists", "passed": false, "detail": "No state"}], "errors": []}'
    )
    messages_body = (
        f'[{{"role":"user","content":"hello"}},{{"role":"tool","content":"llm audit","tool_calls":[{{"_kind":"llm_call_audit","call":{{"model_requested":"openrouter/openai/gpt-5-mini","prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cached_prompt_tokens":3,"provider_cost":{llm_audit_cost},"provider_cost_source":"response_usage.cost"}}}}]}}]'
        if llm_audit_cost is not None
        else '[{"role":"user","content":"hello"}]'
    )
    fake_curl.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

out=""
url=""
while (( $# > 0 )); do
  case "$1" in
    -o)
      out="$2"
      shift 2
      ;;
    -w|-X|-H|--max-time|--data|--data-binary|-d)
      shift 2
      ;;
    --*)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      url="$1"
      shift
      ;;
  esac
done

if [[ -z "$out" ]]; then
  echo "fake curl expected -o <path>" >&2
  exit 2
fi

case "$url" in
  */validate)
    printf '{validate_body}' > "$out"
    printf '200'
    ;;
  */execute)
    printf '{{"run_id":"run-1"}}' > "$out"
    printf '202'
    ;;
  */runs/run-1/diagnostics)
    printf '{{"tokens":[{{"states":[{{"success_reason":{{"metadata":{{"model":"openai/gpt-5-mini","prompt_tokens":10,"cached_prompt_tokens":3,"completion_tokens":5,"total_tokens":15}}}}}}]}}]}}' > "$out"
    printf '200'
    ;;
  */runs/run-1)
    printf '{{"status":"completed","rows_processed":2,"rows_succeeded":2,"rows_routed_success":2,"rows_routed_failure":0,"rows_failed":0,"rows_quarantined":0,"error":null,"landscape_run_id":"run-1"}}' > "$out"
    printf '200'
    ;;
  */state/yaml)
    printf '{{"detail":"yaml unavailable"}}' > "$out"
    printf '{state_yaml_http}'
    ;;
  */messages*)
    printf '{messages_body}' > "$out"
    printf '200'
    ;;
  *)
    printf '{{}}' > "$out"
    printf '200'
    ;;
esac
"""
    )
    fake_curl.chmod(0o755)


def _write_fake_legacy_curl(bin_dir: Path) -> None:
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

out=""
url=""
while (( $# > 0 )); do
  case "$1" in
    -o)
      out="$2"
      shift 2
      ;;
    -w|-X|-H|--max-time|--data|--data-binary|-d)
      shift 2
      ;;
    --*)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      url="$1"
      shift
      ;;
  esac
done

write_out() {
  if [[ -n "$out" ]]; then
    printf '%s' "$1" > "$out"
  else
    printf '%s' "$1"
  fi
}

case "$url" in
  */validate)
    write_out '{"is_valid": true}'
    ;;
  */execute)
    write_out '{"run_id":"run-1"}'
    printf '202'
    ;;
  */sessions/*/runs)
    write_out '{"runs":[{"run_id":"run-1"}]}'
    ;;
  */state/yaml)
    write_out '{}'
    ;;
  */messages)
    write_out '[]'
    ;;
  */runs/run-1/diagnostics)
    write_out '{}'
    ;;
  */runs/run-1/outputs/art-a/content)
    write_out 'artifact-a'
    ;;
  */runs/run-1/outputs/art-b/content)
    write_out 'artifact-b'
    ;;
  */runs/run-1/outputs)
    write_out '{"artifacts":[{"artifact_id":"art-a","sink_node_id":"sink-a","path_or_uri":"file:///tmp/a/results.json","size_bytes":10,"exists_now":true,"content_hash":"sha256:a"},{"artifact_id":"art-b","sink_node_id":"sink-b","path_or_uri":"file:///tmp/b/results.json","size_bytes":10,"exists_now":true,"content_hash":"sha256:b"}]}'
    ;;
  */runs/run-1)
    write_out '{"status":"completed"}'
    ;;
  *)
    write_out '{}'
    ;;
esac
"""
    )
    fake_curl.chmod(0o755)

    fake_sleep = bin_dir / "sleep"
    fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_sleep.chmod(0o755)


def _write_fake_bootstrap_curl(bin_dir: Path) -> None:
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

out=""
url=""
while (( $# > 0 )); do
  case "$1" in
    -o)
      out="$2"
      shift 2
      ;;
    -w|-X|-H|--max-time|--data|--data-binary|-d)
      shift 2
      ;;
    --*)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      url="$1"
      shift
      ;;
  esac
done

write_body() {
  if [[ -n "$out" ]]; then
    printf '%s' "$1" > "$out"
  else
    printf '%s' "$1"
  fi
}

case "$url" in
  */api/auth/login)
    write_body '{"access_token":"header.eyJleHAiOjQxMDI0NDQ4MDB9.signature"}'
    printf '\\n200'
    ;;
  */api/sessions)
    write_body '{"id":"session-1"}'
    printf '201'
    ;;
  *)
    write_body '{}'
    printf '200'
    ;;
esac
"""
    )
    fake_curl.chmod(0o755)


def _write_fake_post_message_curl(bin_dir: Path, *, message_http: str = "502") -> None:
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

out=""
url=""
while (( $# > 0 )); do
  case "$1" in
    -o)
      out="$2"
      shift 2
      ;;
    -w|-X|-H|--max-time|--data|--data-binary|-d)
      shift 2
      ;;
    --*)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      url="$1"
      shift
      ;;
  esac
done

write_body() {{
  if [[ -n "$out" ]]; then
    printf '%s' "$1" > "$out"
  else
    printf '%s' "$1"
  fi
}}

case "$url" in
  */api/sessions/session-1/messages)
    write_body '{{"detail":"upstream unavailable"}}'
    printf '{message_http} 0.03\\n'
    ;;
  */api/sessions/session-1/composer-progress)
    write_body '{{"events":[]}}'
    printf '200'
    ;;
  */api/sessions/session-1/state)
    write_body '{{"version":1,"is_valid":false}}'
    printf '200'
    ;;
  *)
    write_body '{{}}'
    printf '200'
    ;;
esac
"""
    )
    fake_curl.chmod(0o755)


def test_harness_writes_suite_and_run_manifests(tmp_path: Path) -> None:
    """Bootstrap should leave machine-readable evidence about the scenario contract."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_bootstrap_curl(fake_bin)

    runs_root = tmp_path / "runs"
    env = os.environ.copy()
    env.update(
        {
            "ELSPETH_EVAL_BASE_URL": "https://example.invalid",
            "ELSPETH_EVAL_USER": "eval-user",
            "ELSPETH_EVAL_PASS": "eval-pass",
            "ELSPETH_EVAL_RUNS_DIR": str(runs_root),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(HARNESS_SCRIPT), "p1_t3_limit"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    suite_manifest = json.loads((runs_root / "suite_manifest.json").read_text())
    assert suite_manifest["suite"] == "hardmode"
    assert suite_manifest["dispatch_contract"] == "fresh_persona_subagent_for_turns_2_plus"

    run_manifest = json.loads((runs_root / "p1_t3_limit" / "run_manifest.json").read_text())
    assert run_manifest["scenario_id"] == "p1_t3_limit"
    assert run_manifest["session_id"] == "session-1"
    assert run_manifest["scenario_fixture_sha256"]
    assert run_manifest["persona_spec_sha256"]
    assert run_manifest["message_budget_user_turns"] == 5


def test_post_message_records_turn_manifest_and_non_2xx_status(tmp_path: Path) -> None:
    """A composer POST transport failure must be first-class evidence, not a normal turn."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_post_message_curl(fake_bin, message_http="502")

    runs_root = tmp_path / "runs"
    scenario_dir = runs_root / "s1"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "sid.txt").write_text("session-1")
    (scenario_dir / "turn1.user.txt").write_text("hello")
    _write_valid_jwt(scenario_dir / "jwt.txt")

    env = os.environ.copy()
    env.update(
        {
            "ELSPETH_EVAL_BASE_URL": "https://example.invalid",
            "ELSPETH_EVAL_USER": "eval-user",
            "ELSPETH_EVAL_PASS": "eval-pass",
            "ELSPETH_EVAL_RUNS_DIR": str(runs_root),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(POST_MESSAGE_SCRIPT), "s1", "1", str(scenario_dir / "turn1.user.txt")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((scenario_dir / "metrics.t1.json").read_text())
    assert metrics["turn_status"] == {
        "status": "transport_error",
        "http_ok": False,
        "http_code": "502",
        "transport_error": "post_message HTTP 502",
    }
    turn_manifest = json.loads((scenario_dir / "turn1.manifest.json").read_text())
    assert turn_manifest["turn"] == 1
    assert turn_manifest["status"] == metrics["turn_status"]
    assert turn_manifest["persona_dispatch"] == {
        "required": False,
        "verified_by_harness": False,
        "evidence": None,
    }


def test_sweep_simplified_help_exits_before_side_effects(tmp_path: Path) -> None:
    """`--help` must not be interpreted as a run label and start a sweep."""

    harness_root = tmp_path / "composer-harness"
    hardmode_dir = harness_root / "hardmode"
    scenarios_dir = harness_root / "scenarios" / "hardmode"
    hardmode_dir.mkdir(parents=True)
    scenarios_dir.mkdir(parents=True)

    script = hardmode_dir / "sweep_simplified.sh"
    script.write_text(SWEEP_SCRIPT.read_text())
    script.chmod(0o755)

    harness_called = tmp_path / "harness-called"
    (hardmode_dir / "harness.sh").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$1" > {harness_called}
mkdir -p "$ELSPETH_EVAL_RUNS_DIR/$1"
printf '{{"opening_prompt":"hello"}}' > "$ELSPETH_EVAL_RUNS_DIR/$1/scenario.json"
"""
    )
    (hardmode_dir / "harness.sh").chmod(0o755)
    (hardmode_dir / "post_message.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n")
    (hardmode_dir / "post_message.sh").chmod(0o755)
    (scenarios_dir / "p1_t1_happy_categorize.json").write_text(json.dumps({"opening_prompt": "hello"}))

    result = subprocess.run(
        [str(script), "--help"],
        cwd=harness_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
    assert not harness_called.exists()
    assert not (harness_root / "runs").exists()


def test_finalize_scenario_exits_74_when_validate_http_fails(tmp_path: Path) -> None:
    """A validation transport/auth failure is infrastructure failure, not invalid YAML."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_curl(fake_bin)

    runs_root = tmp_path / "runs"
    scenario_dir = runs_root / "s1"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "sid.txt").write_text("session-1")
    (scenario_dir / "scenario.json").write_text(json.dumps({"persona": "tester"}))
    (scenario_dir / "run.json").write_text(json.dumps({"status": "completed"}))
    _write_valid_jwt(scenario_dir / "jwt.txt")

    env = os.environ.copy()
    env.update(
        {
            "ELSPETH_EVAL_BASE_URL": "https://example.invalid",
            "ELSPETH_EVAL_USER": "eval-user",
            "ELSPETH_EVAL_PASS": "eval-pass",
            "ELSPETH_EVAL_RUNS_DIR": str(runs_root),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(FINALIZE_SCRIPT), "s1"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 74
    assert "validate failed" in result.stderr
    assert not (scenario_dir / "ledger.json").exists()


def test_finalize_writes_ledger_when_invalid_state_yaml_is_unavailable(tmp_path: Path) -> None:
    """Invalid scenarios are data; a missing YAML artifact must not erase the ledger."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_finalize_curl(fake_bin, state_yaml_http="404", valid=False)

    runs_root = tmp_path / "runs"
    scenario_dir = runs_root / "s1"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "sid.txt").write_text("session-1")
    (scenario_dir / "scenario.json").write_text(json.dumps({"persona": "tester", "task_class": "limit", "task_summary": "invalid state"}))
    (scenario_dir / "metrics.t1.json").write_text(
        json.dumps({"turn": 1, "wall_seconds": 1.25, "tool_call_count": 0, "in_loop_recovery_count": 0})
    )
    (scenario_dir / "run.json").write_text(json.dumps({"status": "completed", "rows_processed": 99}))
    (scenario_dir / "diagnostics.json").write_text(
        json.dumps({"metadata": {"model": "stale/model", "prompt_tokens": 99, "completion_tokens": 99, "total_tokens": 198}})
    )
    _write_valid_jwt(scenario_dir / "jwt.txt")

    env = os.environ.copy()
    env.update(
        {
            "ELSPETH_EVAL_BASE_URL": "https://example.invalid",
            "ELSPETH_EVAL_USER": "eval-user",
            "ELSPETH_EVAL_PASS": "eval-pass",
            "ELSPETH_EVAL_RUNS_DIR": str(runs_root),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(FINALIZE_SCRIPT), "s1"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    ledger = json.loads((scenario_dir / "ledger.json").read_text())
    assert ledger["is_valid"] is False
    assert ledger["ran_engine"] is False
    assert ledger["run_status"] is None
    assert ledger["rows_processed"] is None
    assert ledger["provider_usage"]["total_tokens"] == 0
    assert ledger["provider_usage"]["models"] == []
    assert ledger["provider_usage"]["token_usage_available"] is False
    assert ledger["provider_usage"]["source"] == "not_available"
    assert ledger["failed_validate_checks"] == ["state_exists"]
    assert ledger["artifact_collection_errors"] == [{"artifact": "final_yaml", "http_code": "404", "path": "final_yaml.json"}]
    assert json.loads((scenario_dir / "final_yaml.json").read_text()) == {}


def test_finalize_preserves_completed_run_when_post_run_yaml_export_fails(tmp_path: Path) -> None:
    """Post-run artifact collection failures are metadata, not run outcome overrides."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_finalize_curl(fake_bin, state_yaml_http="409", valid=True, llm_audit_cost="0.0037")

    runs_root = tmp_path / "runs"
    scenario_dir = runs_root / "s1"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "sid.txt").write_text("session-1")
    (scenario_dir / "scenario.json").write_text(json.dumps({"persona": "tester", "task_class": "happy", "task_summary": "valid state"}))
    (scenario_dir / "metrics.t1.json").write_text(
        json.dumps({"turn": 1, "wall_seconds": 2.0, "tool_call_count": 1, "in_loop_recovery_count": 0})
    )
    _write_valid_jwt(scenario_dir / "jwt.txt")

    env = os.environ.copy()
    env.update(
        {
            "ELSPETH_EVAL_BASE_URL": "https://example.invalid",
            "ELSPETH_EVAL_USER": "eval-user",
            "ELSPETH_EVAL_PASS": "eval-pass",
            "ELSPETH_EVAL_RUNS_DIR": str(runs_root),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(FINALIZE_SCRIPT), "s1"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    ledger = json.loads((scenario_dir / "ledger.json").read_text())
    assert ledger["run_status"] == "completed"
    assert ledger["rows_processed"] == 2
    assert ledger["rows_succeeded"] == 2
    assert ledger["artifact_collection_errors"] == [{"artifact": "final_yaml", "http_code": "409", "path": "final_yaml.json"}]
    assert ledger["provider_usage"] == {
        "prompt_tokens": 20,
        "cached_prompt_tokens": 6,
        "completion_tokens": 10,
        "total_tokens": 30,
        "models": ["openai/gpt-5-mini", "openrouter/openai/gpt-5-mini"],
        "token_usage_available": True,
        "source": "diagnostics+llm_call_audit",
    }
    assert ledger["cost"] == {
        "actual_usd": 0.0037,
        "source": "composer_llm_audit.response_usage.cost",
        "cost_available": True,
        "costed_call_count": 1,
    }


def test_aggregate_reports_artifact_errors_and_provider_usage_without_fake_cost(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    scenario_dir = runs_root / "s1"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "ledger.json").write_text(
        json.dumps(
            {
                "scenario_id": "s1",
                "persona": "tester",
                "task_class": "happy",
                "user_turn_count": 1,
                "is_valid": True,
                "ran_engine": True,
                "run_status": "completed",
                "poll_status": "ok",
                "rows_processed": 2,
                "rows_succeeded": 2,
                "rows_routed_success": 2,
                "total_wall_seconds": 2.0,
                "total_tool_calls": 1,
                "total_in_loop_recoveries": 0,
                "clarifying_keyword_turns": [],
                "limit_keyword_turns": [],
                "failed_validate_checks": [],
                "run_error": None,
                "artifact_collection_errors": [{"artifact": "final_yaml", "http_code": "409", "path": "final_yaml.json"}],
                "provider_usage": {
                    "prompt_tokens": 10,
                    "cached_prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "models": ["openai/gpt-5-mini"],
                    "token_usage_available": True,
                    "source": "diagnostics",
                },
                "cost": {
                    "actual_usd": 0.0037,
                    "source": "composer_llm_audit.response_usage.cost",
                    "cost_available": True,
                    "costed_call_count": 1,
                },
            }
        )
    )
    legacy_dir = runs_root / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "ledger.json").write_text(
        json.dumps(
            {
                "scenario_id": "legacy",
                "persona": "tester",
                "task_class": "limit",
                "user_turn_count": 1,
                "is_valid": False,
                "ran_engine": False,
                "run_status": None,
                "poll_status": "ok",
                "rows_processed": None,
                "rows_succeeded": None,
                "total_wall_seconds": 1.0,
                "total_tool_calls": 0,
                "total_in_loop_recoveries": 0,
                "clarifying_keyword_turns": [],
                "limit_keyword_turns": [],
                "failed_validate_checks": ["state_exists"],
                "run_error": None,
            }
        )
    )

    result = subprocess.run(
        [str(AGGREGATE_SCRIPT), str(runs_root)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    aggregate = json.loads((runs_root / "aggregate.json").read_text())
    by_id = {row["scenario_id"]: row for row in aggregate}
    assert by_id["s1"]["artifact_collection_error_count"] == 1
    assert by_id["s1"]["provider_usage"]["total_tokens"] == 15
    assert by_id["s1"]["provider_usage"]["token_usage_available"] is True
    assert by_id["s1"]["cost"]["actual_usd"] == 0.0037
    assert by_id["s1"]["cost"]["source"] == "composer_llm_audit.response_usage.cost"
    assert by_id["legacy"]["provider_usage"]["token_usage_available"] is False
    summary = json.loads((runs_root / "aggregate_summary.json").read_text())
    assert summary["provider_usage"]["token_usage_available_scenarios"] == 1
    assert summary["provider_usage"]["token_usage_unavailable_scenarios"] == 1
    assert summary["provider_usage"]["total_tokens"] == 15
    assert summary["cost"]["actual_usd"] == 0.0037
    assert summary["cost"]["cost_available_scenarios"] == 1
    assert summary["cost"]["cost_unavailable_scenarios"] == 1
    scorecard = (runs_root / "SCORECARD.md").read_text()
    assert "Artifact errors" in scorecard
    assert "Tokens" in scorecard
    assert "Cost" in scorecard
    assert "| legacy | tester | limit | 1 | 1.0 | — | — |" in scorecard


def test_aggregate_reports_malformed_ledger_errors(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    good_dir = runs_root / "good"
    bad_dir = runs_root / "bad"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir()
    (good_dir / "ledger.json").write_text(
        json.dumps(
            {
                "scenario_id": "good",
                "persona": "tester",
                "task_class": "happy",
                "user_turn_count": 1,
                "is_valid": True,
                "ran_engine": False,
                "run_status": None,
                "poll_status": "ok",
                "total_wall_seconds": 1.0,
                "total_tool_calls": 0,
                "total_in_loop_recoveries": 0,
                "clarifying_keyword_turns": [],
                "limit_keyword_turns": [],
                "failed_validate_checks": [],
                "run_error": None,
            }
        )
    )
    (bad_dir / "ledger.json").write_text("{not json")

    result = subprocess.run(
        [str(AGGREGATE_SCRIPT), str(runs_root)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((runs_root / "aggregate_summary.json").read_text())
    assert summary["scenario_count"] == 1
    assert summary["aggregate_error_count"] == 1
    errors = json.loads((runs_root / "aggregate_errors.json").read_text())
    assert errors[0]["kind"] == "malformed_ledger"
    assert errors[0]["path"] == "bad/ledger.json"
    scorecard = (runs_root / "SCORECARD.md").read_text()
    assert "Aggregate errors" in scorecard


def test_agent_prompt_does_not_estimate_provider_cost_from_wall_time() -> None:
    prompt = AGENT_PROMPT.read_text()

    assert "total_wall_seconds across ledgers" not in prompt
    assert "$0.05/sec" not in prompt
    assert "do not estimate dollars from wall time" in prompt


@pytest.mark.parametrize(
    ("script_source", "mode"),
    [
        (LEGACY_BASIC_FINALIZE_SCRIPT, "basic"),
        (LEGACY_HARDMODE_FINALIZE_SCRIPT, "hardmode"),
    ],
)
def test_legacy_finalizers_keep_duplicate_artifact_basenames_unique(
    tmp_path: Path,
    script_source: Path,
    mode: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_legacy_curl(fake_bin)

    root = tmp_path / mode
    root.mkdir()
    script = root / "finalize_scenario.sh"
    script.write_text(script_source.read_text())
    script.chmod(0o755)

    scenario_dir = root / "s1" if mode == "basic" else root / "results" / "s1"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "sid.txt").write_text("session-1")
    if mode == "hardmode":
        _write_valid_jwt(scenario_dir / "jwt.txt")
        (scenario_dir / "scenario.json").write_text(json.dumps({"persona": "tester"}))

    env = os.environ.copy()
    env.update(
        {
            "JWT": "token",
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(script), "s1"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (scenario_dir / "outputs" / "art-a__results.json").read_text() == "artifact-a"
    assert (scenario_dir / "outputs" / "art-b__results.json").read_text() == "artifact-b"
