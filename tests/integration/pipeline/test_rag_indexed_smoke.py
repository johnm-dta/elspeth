"""End-to-end smoke test: depends_on -> commencement gate -> RAG retrieval.

Exercises the full RAG ingestion story:
1. Query pipeline declares depends_on for the indexing pipeline
2. Indexing pipeline runs first, populating ChromaDB via ChromaSink
3. Commencement gate verifies collection has documents
4. RAG transform readiness check passes
5. Questions are retrieved against the populated collection
6. Landscape records dependency correlation, gate results, and readiness check

Uses tmp_path for all output to avoid polluting the examples/ directory.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import chromadb
import pytest


@pytest.fixture
def example_dir(tmp_path: Path) -> Path:
    """Create a self-contained copy of the example in tmp_path."""
    example_src = Path("examples/chroma_rag_indexed")
    example_dst = tmp_path / "chroma_rag_indexed"
    example_dst.mkdir()

    # Copy data files
    shutil.copy(example_src / "documents.csv", example_dst / "documents.csv")
    shutil.copy(example_src / "questions.csv", example_dst / "questions.csv")

    # Create output directories
    (example_dst / "output").mkdir()
    (example_dst / "runs").mkdir()

    # Paths used in generated YAML (all absolute for tmp_path isolation)
    chroma_dir = example_dst / "chroma_data"
    audit_db = example_dst / "runs" / "audit.db"

    index_yaml = example_dst / "index_pipeline.yaml"
    index_yaml.write_text(
        f"""\
source:
  plugin: csv
  on_success: output
  options:
    path: {example_dst / "documents.csv"}
    schema:
      mode: fixed
      fields:
      - 'doc_id: str'
      - 'text_content: str'
      - 'topic: str'
      - 'subtopic: str'
    on_validation_failure: discard

sinks:
  output:
    plugin: chroma_sink
    on_write_failure: discard
    options:
      collection: smoke-test-facts
      mode: persistent
      persist_directory: {chroma_dir}
      distance_function: cosine
      field_mapping:
        document_field: text_content
        id_field: doc_id
        metadata_fields:
        - topic
        - subtopic
      on_duplicate: overwrite
      schema:
        mode: fixed
        fields:
        - 'doc_id: str'
        - 'text_content: str'
        - 'topic: str'
        - 'subtopic: str'

landscape:
  url: sqlite:///{audit_db}
"""
    )

    query_yaml = example_dst / "query_pipeline.yaml"
    query_yaml.write_text(
        f"""\
depends_on:
- name: index_corpus
  settings: ./index_pipeline.yaml

collection_probes:
- collection: smoke-test-facts
  provider: chroma
  provider_config:
    mode: persistent
    persist_directory: {chroma_dir}

commencement_gates:
- name: corpus_ready
  condition: "collections['smoke-test-facts']['count'] > 0"

source:
  plugin: csv
  on_success: rag_in
  options:
    path: {example_dst / "questions.csv"}
    schema:
      mode: fixed
      fields:
      - 'id: int'
      - 'question: str'
    on_validation_failure: discard

transforms:
- name: retrieve
  plugin: rag_retrieval
  input: rag_in
  on_success: output
  on_error: quarantine
  options:
    query_field: question
    output_prefix: sci
    provider: chroma
    provider_config:
      collection: smoke-test-facts
      mode: persistent
      persist_directory: {chroma_dir}
      distance_function: cosine
    top_k: 3
    min_score: 0.0
    on_no_results: quarantine
    context_format: numbered
    schema:
      mode: flexible
      fields:
      - 'id: int'
      - 'question: str'

sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: {example_dst / "output" / "results.jsonl"}
      schema:
        mode: observed
      format: jsonl
  quarantine:
    plugin: json
    on_write_failure: discard
    options:
      path: {example_dst / "output" / "quarantined.jsonl"}
      schema:
        mode: observed
      format: jsonl

landscape:
  url: sqlite:///{audit_db}
"""
    )

    return example_dst


class TestRAGIndexedSmoke:
    """Full end-to-end smoke test for the RAG ingestion pipeline."""

    def test_full_pipeline_sequence(self, example_dir: Path) -> None:
        """Run query pipeline — depends_on triggers indexing first."""
        from elspeth.cli import bootstrap_and_run

        query_yaml = example_dir / "query_pipeline.yaml"
        result = bootstrap_and_run(query_yaml)

        # Pipeline completed successfully
        assert result.status.name == "COMPLETED"
        assert result.rows_processed > 0

        # ChromaDB collection was populated by the dependency
        chroma_dir = example_dir / "chroma_data"
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("smoke-test-facts")
        assert collection.count() == 10  # 10 documents from CSV

        # Output file was written
        results_path = example_dir / "output" / "results.jsonl"
        assert results_path.exists()
        lines = results_path.read_text().strip().split("\n")
        assert len(lines) == 5  # 5 questions

    def test_landscape_records_preflight_results(self, example_dir: Path) -> None:
        """Verify the Landscape records dependency_runs and commencement_gates.

        Pre-flight results are stored in the ``preflight_results`` table
        (not as a JSON blob on ``runs``).  Each result has its own row with
        ``result_type``, ``name``, and ``result_json`` (canonical JSON).
        """
        from elspeth.cli import bootstrap_and_run

        query_yaml = example_dir / "query_pipeline.yaml"
        result = bootstrap_and_run(query_yaml)

        # Query the preflight_results table — linked to the run via run_id FK
        audit_db = example_dir / "runs" / "audit.db"
        conn = sqlite3.connect(str(audit_db))
        cursor = conn.execute(
            "SELECT result_type, name, result_json FROM preflight_results WHERE run_id = ? ORDER BY created_at",
            (result.run_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) >= 2  # At least one dependency + one gate

        # --- dependency_run recorded ---
        dep_rows = [(name, json.loads(rj)) for rt, name, rj in rows if rt == "dependency_run"]
        assert len(dep_rows) == 1
        dep_name, dep_data = dep_rows[0]
        assert dep_name == "index_corpus"
        assert "run_id" in dep_data
        assert "settings_hash" in dep_data
        assert "indexed_at" in dep_data

        # --- commencement_gate recorded ---
        gate_rows = [(name, json.loads(rj)) for rt, name, rj in rows if rt == "commencement_gate"]
        assert len(gate_rows) == 1
        gate_name, gate_data = gate_rows[0]
        assert gate_name == "corpus_ready"
        assert gate_data["result"] is True
        assert "condition" in gate_data
        # env excluded from context_snapshot (Tier 3 data, not persisted)
        assert "context_snapshot" in gate_data
        assert "env" not in gate_data["context_snapshot"]

        # --- readiness_check recorded (from RAG transform on_start) ---
        readiness_rows = [(name, json.loads(rj)) for rt, name, rj in rows if rt == "readiness_check"]
        assert len(readiness_rows) >= 1  # At least one from the RAG transform
