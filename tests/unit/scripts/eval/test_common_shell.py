"""Shell-level regressions for the composer eval harness helpers."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
FINALIZE_SCRIPT = REPO_ROOT / "evals" / "composer-harness" / "hardmode" / "finalize_scenario.sh"


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
