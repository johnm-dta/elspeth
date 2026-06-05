from __future__ import annotations

from pathlib import Path

from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file


def test_execution_service_recovery_handlers_are_explicit_results() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source_file = repo_root / "src/elspeth/web/execution/service.py"

    findings = [
        finding
        for finding in scan_file(source_file, repo_root / "src/elspeth")
        if finding.rule_id == "R6"
        and (
            "except (SQLAlchemyError, OSError):" in finding.message
            or "except GracefulShutdownError as gse" in finding.message
            or "except (SQLAlchemyError, OSError) as probe_err" in finding.message
            or "except (SQLAlchemyError, OSError) as status_err" in finding.message
            or "except self._FINALIZE_SUPPRESSED as blob_err" in finding.message
            or "except RuntimeError as broadcast_err" in finding.message
        )
    ]

    assert findings == []
