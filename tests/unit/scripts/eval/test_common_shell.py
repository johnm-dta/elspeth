"""Shell-level regressions for the composer eval harness helpers."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FINALIZE_SCRIPT = REPO_ROOT / "evals" / "composer-harness" / "hardmode" / "finalize_scenario.sh"
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
