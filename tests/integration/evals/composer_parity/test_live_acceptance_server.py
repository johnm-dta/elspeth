"""Server-side deterministic integration for the live-acceptance oracle.

Plan 05 Task 5. "Server-side" here is the *deployed-invocation boundary*: the
oracle is driven through its CLI entrypoint (:func:`la.main`) against a real
on-disk evidence directory with real permissions, symlinks, and process exit
codes — the shape Task 6's staging deploy uses when it runs
``live_acceptance.py verify`` on exported evidence, and the collect -> redact ->
persist -> verify shape ``run`` uses. It deliberately does NOT spin up an
elspeth web server or execute the engine (that is Task 3 / Task 6 territory) and
performs no provider network call.

The env-gated live test re-verifies operator-exported REAL evidence and SKIPS
when the gate env is absent, mirroring
``tests/integration/web/composer/test_bedrock_live_smoke.py``: real provider
runs are produced separately by the operator / main loop; this test only
re-verifies their exported evidence with the same oracle.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from tests.unit.evals.composer_parity._live_acceptance_fixtures import (
    REVISION,
    build_valid_evidence,
    la,
    write_evidence,
)

pytestmark = [pytest.mark.integration]


def _cli_verify(evidence_dir: Path, revision: str, *, surface: str | None = None) -> int:
    argv = ["verify", "--evidence-dir", str(evidence_dir), "--revision", revision]
    if surface is not None:
        argv += ["--surface", surface]
    return la.main(argv)


# --------------------------------------------------------------------------- #
# Deterministic offline: the CLI + filesystem boundary
# --------------------------------------------------------------------------- #


def test_cli_accepts_exported_evidence_dir(tmp_path: Path) -> None:
    write_evidence(tmp_path, REVISION, build_valid_evidence())
    assert _cli_verify(tmp_path, REVISION) == 0


def test_cli_accepts_with_matching_surface(tmp_path: Path) -> None:
    write_evidence(tmp_path, REVISION, build_valid_evidence(surface="freeform"))
    assert _cli_verify(tmp_path, REVISION, surface="freeform") == 0


def test_cli_rejects_surface_mismatch(tmp_path: Path) -> None:
    write_evidence(tmp_path, REVISION, build_valid_evidence(surface="freeform"))
    assert _cli_verify(tmp_path, REVISION, surface="guided_full") == 1


def test_cli_rejects_fake_provider_evidence(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    # Manifest still declares mode == "live"; the run call betrays a mock double.
    doc[la.RUN_LLM_CALLS_FILE][0]["model_returned"] = "mock-echo-model"
    write_evidence(tmp_path, REVISION, doc)
    assert _cli_verify(tmp_path, REVISION) == 1


def test_cli_rejects_symlinked_evidence_file(tmp_path: Path) -> None:
    revision_dir = write_evidence(tmp_path, REVISION, build_valid_evidence())
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (revision_dir / la.GRAPH_FILE).unlink()
    (revision_dir / la.GRAPH_FILE).symlink_to(outside)
    assert _cli_verify(tmp_path, REVISION) == 1


def test_cli_rejects_lax_permissions(tmp_path: Path) -> None:
    revision_dir = write_evidence(tmp_path, REVISION, build_valid_evidence())
    revision_dir.chmod(0o777)
    assert _cli_verify(tmp_path, REVISION) == 1


def test_cli_rejects_sensitive_content(tmp_path: Path) -> None:
    doc = build_valid_evidence()
    doc[la.MANIFEST_FILE]["provider"]["authorization"] = "Bearer sk-should-not-be-here"
    write_evidence(tmp_path, REVISION, doc)
    assert _cli_verify(tmp_path, REVISION) == 1


# --------------------------------------------------------------------------- #
# Deterministic offline: run -> redact -> persist -> CLI verify (no provider)
# --------------------------------------------------------------------------- #


def test_run_collect_redact_persist_then_cli_verify(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "output" / "composer-parity"

    def collector(*, base_url: str, api_key: str, surface: str, fixture: Path, intent: Path, revision: str) -> dict[str, Any]:
        doc = build_valid_evidence(surface=surface, revision=revision)
        # A collector that failed to strip a credential at the source: the
        # export path must redact it before it ever lands on disk.
        doc[la.RUN_LLM_CALLS_FILE][0]["authorization"] = "Bearer sk-secret"
        return doc

    report = la.run_live(
        base_url="https://staging.example",
        api_key="dummy-not-a-real-key",
        surface="guided_staged",
        fixture=tmp_path / "colours.csv",
        intent=tmp_path / "request.txt",
        revision=REVISION,
        evidence_dir=evidence_dir,
        collector=collector,
    )
    assert report.surface == "guided_staged"

    persisted = evidence_dir / REVISION / la.RUN_LLM_CALLS_FILE
    calls = json.loads(persisted.read_text(encoding="utf-8"))
    assert "authorization" not in calls[0]
    file_mode = stat.S_IMODE(persisted.lstat().st_mode)
    assert file_mode & (stat.S_IWGRP | stat.S_IWOTH) == 0

    # The redacted, persisted evidence re-verifies through the plain CLI path.
    assert _cli_verify(evidence_dir, REVISION, surface="guided_staged") == 0


# --------------------------------------------------------------------------- #
# Env-gated live re-verification (SKIPS when the key/env is absent)
# --------------------------------------------------------------------------- #


def test_live_acceptance_reverifies_operator_exported_evidence() -> None:
    if os.environ.get("ELSPETH_RUN_COMPOSER_PARITY_LIVE") != "1":
        pytest.skip("set ELSPETH_RUN_COMPOSER_PARITY_LIVE=1 to verify operator-exported live evidence")

    evidence_dir = os.environ.get("ELSPETH_EVAL_EVIDENCE_DIR")
    revision = os.environ.get("ELSPETH_EVAL_REVISION")
    if not evidence_dir or not revision:
        pytest.fail("ELSPETH_EVAL_EVIDENCE_DIR and ELSPETH_EVAL_REVISION are required for the live re-verify")

    # No provider call here: the operator/main loop produced this evidence from a
    # REAL run; the oracle proves liveness from intrinsic per-call evidence.
    report = la.verify_evidence_dir(evidence_dir, revision)
    assert report.checks
    assert "live_provider" in report.checks
