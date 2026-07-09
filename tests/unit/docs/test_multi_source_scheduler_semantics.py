from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_multi_source_scheduler_docs_name_sequential_source_iteration_contract() -> None:
    adr_025 = " ".join((REPO_ROOT / "docs/architecture/adr/025-multi-source-ingestion.md").read_text(encoding="utf-8").split())
    adr_026 = " ".join((REPO_ROOT / "docs/architecture/adr/026-durable-token-scheduler.md").read_text(encoding="utf-8").split())
    # The sequential multi-source ingest contract comment lives with the
    # source-loop sequencing, which moved core.py -> leader_drain.py in the
    # orchestrator decomposition (elspeth-9e71ae82a4).
    leader_drain = " ".join((REPO_ROOT / "src/elspeth/engine/orchestrator/leader_drain.py").read_text(encoding="utf-8").split())

    required_phrases = (
        "sequential multi-source ingest",
        "YAML declaration order is the determinism anchor",
        "not concurrent source iteration",
    )

    for phrase in required_phrases:
        assert phrase in adr_025
        assert phrase in adr_026
        assert phrase in leader_drain
