# RAG Ingestion Sub-plan 5: End-to-End Example + Smoke Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a complete example showing the full RAG ingestion story: an indexing pipeline populates a ChromaDB collection via `ChromaSink`, and a query pipeline uses `depends_on` + commencement gates + RAG retrieval to query it. Plus an integration smoke test that exercises all three mechanisms end-to-end.

**Architecture:** Assembly only — no new production code. Two pipeline YAML configs, sample CSV data, and one integration smoke test that verifies the full sequence fires correctly with Landscape assertions.

**Tech Stack:** YAML configs, CSV test data, pytest with real ephemeral ChromaDB

**Spec:** `docs/superpowers/specs/2026-03-25-rag-ingestion-pipeline-design.md` (End-to-End Example section)

**Depends on:** Sub-plans 1, 2, 3, and 4 (all merged).

**Risk:** LOW — assembly and verification only.

---

## Review Notes from Sub-plan 1 (read before implementing)

1. **ExpressionParser multi-name mode.** When writing commencement gate YAML expressions, the allowed names are `collections`, `dependency_runs`, and `env`. The expression `row['x']` is **not** valid in gate conditions — `row` is excluded from gate expressions by design (gates run before row processing). See sub-plan 3 review notes for the full calling convention.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `examples/chroma_rag_indexed/documents.csv` | Reference corpus (same documents as `seed_collection.py`, now as CSV) |
| Create | `examples/chroma_rag_indexed/questions.csv` | Questions to answer (reuse from `chroma_rag` example) |
| Create | `examples/chroma_rag_indexed/index_pipeline.yaml` | Indexing pipeline: CSV → ChromaSink |
| Create | `examples/chroma_rag_indexed/query_pipeline.yaml` | Query pipeline: depends_on + gate + RAG retrieval |
| Create | `examples/chroma_rag_indexed/README.md` | Usage instructions |
| Create | `tests/integration/engine/test_rag_indexed_smoke.py` | Full end-to-end smoke test |

---

### Task 1: Sample Data Files

**Files:**
- Create: `examples/chroma_rag_indexed/documents.csv`
- Create: `examples/chroma_rag_indexed/questions.csv`

- [ ] **Step 1: Create the documents CSV**

The documents come from the existing `seed_collection.py` — same content, now as a CSV that the indexing pipeline reads.

```csv
doc_id,text_content,topic,subtopic
doc-diabetes-overview,"Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels. Type 1 diabetes is an autoimmune condition where the immune system attacks insulin-producing beta cells in the pancreas. Type 2 diabetes involves insulin resistance, where cells don't respond effectively to insulin. Common symptoms include excessive thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision.",medicine,diabetes
doc-diabetes-management,"Managing diabetes involves regular blood glucose monitoring, medication or insulin therapy, dietary management, and physical activity. HbA1c tests measure average blood sugar over 2-3 months. Complications of unmanaged diabetes include neuropathy, retinopathy, kidney disease, and cardiovascular problems.",medicine,diabetes
doc-photosynthesis,"Photosynthesis is the process by which green plants convert light energy into chemical energy. It occurs primarily in chloroplasts, using chlorophyll to absorb sunlight. The light-dependent reactions split water molecules, releasing oxygen and producing ATP and NADPH. The Calvin cycle then uses these products to fix carbon dioxide into glucose.",biology,botany
doc-earthquakes,"Earthquakes occur when tectonic plates along fault lines suddenly slip past each other, releasing stored elastic energy as seismic waves. The point of rupture underground is the focus (hypocenter), while the point directly above on the surface is the epicenter. Earthquakes are measured using the moment magnitude scale (Mw).",geology,seismology
doc-mitochondria,"Mitochondria are double-membrane organelles found in most eukaryotic cells. They are often called the powerhouse of the cell because they generate most of the cell's supply of adenosine triphosphate (ATP) through oxidative phosphorylation. The inner membrane is folded into cristae.",biology,cell_biology
doc-vaccines,"Vaccines work by training the immune system to recognize and fight specific pathogens. They contain weakened or inactivated forms of a pathogen, or parts of it. When administered, the immune system produces antibodies and memory cells.",medicine,immunology
doc-climate-oceans,"Climate change affects ocean levels through two primary mechanisms: thermal expansion of seawater as it warms, and the melting of land-based ice sheets and glaciers. Current projections estimate sea levels could rise 0.3 to 1.0 meters by 2100.",earth_science,climate
doc-machine-learning,"Machine learning differs from traditional programming in a fundamental way. In traditional programming, developers write explicit rules. In machine learning, algorithms learn patterns from data to make predictions or decisions.",computer_science,ai
doc-dna,"DNA (deoxyribonucleic acid) is a double-helix molecule composed of two polynucleotide chains. Each nucleotide consists of a phosphate group, a deoxyribose sugar, and one of four nitrogenous bases: adenine, thymine, guanine, or cytosine.",biology,genetics
doc-immune-system,"The human immune system has two main branches: innate immunity and adaptive immunity. Innate immunity provides immediate, non-specific defense. Adaptive immunity develops over time and provides specific, long-lasting protection through T cells and B cells.",medicine,immunology
```

- [ ] **Step 2: Create the questions CSV**

Reuse from the existing `chroma_rag` example:

```csv
id,question
1,What are the main symptoms of diabetes?
2,How does photosynthesis work in plants?
3,What causes earthquakes and how are they measured?
4,What is the role of mitochondria in cells?
5,How do vaccines help the immune system?
```

- [ ] **Step 3: Commit**

```bash
git add examples/chroma_rag_indexed/documents.csv examples/chroma_rag_indexed/questions.csv
git commit -m "feat: add sample data for chroma_rag_indexed example"
```

---

### Task 2: Indexing Pipeline Config

**Files:**
- Create: `examples/chroma_rag_indexed/index_pipeline.yaml`

- [ ] **Step 1: Write the indexing pipeline YAML**

```yaml
# No transforms — documents are written directly from source to ChromaSink.
# A chunking transform would be added here for long-document corpora.
source:
  plugin: csv
  on_success: output
  options:
    path: examples/chroma_rag_indexed/documents.csv
    schema:
      mode: fixed
      fields:
        - "doc_id: str"
        - "text_content: str"
        - "topic: str"
        - "subtopic: str"
    on_validation_failure: discard

sinks:
  output:
    plugin: chroma_sink
    options:
      collection: science-facts-indexed
      mode: persistent
      persist_directory: examples/chroma_rag_indexed/chroma_data
      distance_function: cosine
      field_mapping:
        document: text_content
        id: doc_id
        metadata:
          - topic
          - subtopic
      on_duplicate: overwrite
      schema:
        mode: fixed
        fields:
          - "doc_id: str"
          - "text_content: str"
          - "topic: str"
          - "subtopic: str"

landscape:
  url: sqlite:///examples/chroma_rag_indexed/runs/audit.db
```

- [ ] **Step 2: Validate the config**

Run: `elspeth validate --settings examples/chroma_rag_indexed/index_pipeline.yaml`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add examples/chroma_rag_indexed/index_pipeline.yaml
git commit -m "feat: add indexing pipeline config for chroma_rag_indexed example"
```

---

### Task 3: Query Pipeline Config

**Files:**
- Create: `examples/chroma_rag_indexed/query_pipeline.yaml`

- [ ] **Step 1: Write the query pipeline YAML with depends_on and commencement gate**

```yaml
depends_on:
  - name: index_corpus
    settings: ./index_pipeline.yaml

collection_probes:
  - collection: science-facts-indexed
    provider: chroma
    provider_config:
      mode: persistent
      persist_directory: examples/chroma_rag_indexed/chroma_data

commencement_gates:
  - name: corpus_ready
    condition: "collections['science-facts-indexed']['count'] > 0"
    on_fail: abort

source:
  plugin: csv
  on_success: retrieve
  options:
    path: examples/chroma_rag_indexed/questions.csv
    schema:
      mode: fixed
      fields:
        - "id: int"
        - "question: str"
    on_validation_failure: discard

transforms:
  - name: retrieve
    plugin: rag_retrieval
    input: retrieve
    on_success: output
    on_error: quarantine
    options:
      query_field: question
      output_prefix: sci
      provider: chroma
      provider_config:
        collection: science-facts-indexed
        mode: persistent
        persist_directory: examples/chroma_rag_indexed/chroma_data
        distance_function: cosine
      top_k: 3
      min_score: 0.0
      on_no_results: quarantine
      context_format: numbered
      schema:
        mode: flexible
        fields:
          - "id: int"
          - "question: str"

sinks:
  output:
    plugin: json
    options:
      path: examples/chroma_rag_indexed/output/results.jsonl
      schema:
        mode: observed
      format: jsonl
  quarantine:
    plugin: json
    options:
      path: examples/chroma_rag_indexed/output/quarantined.jsonl
      schema:
        mode: observed
      format: jsonl

landscape:
  url: sqlite:///examples/chroma_rag_indexed/runs/audit.db
```

- [ ] **Step 2: Validate the config**

Run: `elspeth validate --settings examples/chroma_rag_indexed/query_pipeline.yaml`
Expected: PASS (validation checks dependency file exists and gate expression parses)

- [ ] **Step 3: Commit**

```bash
git add examples/chroma_rag_indexed/query_pipeline.yaml
git commit -m "feat: add query pipeline config with depends_on and commencement gate"
```

---

### Task 4: Integration Smoke Test

**Files:**
- Create: `tests/integration/engine/test_rag_indexed_smoke.py`

- [ ] **Step 1: Write the end-to-end smoke test**

```python
# tests/integration/engine/test_rag_indexed_smoke.py
"""End-to-end smoke test: depends_on → commencement gate → RAG retrieval.

Exercises the full RAG ingestion story:
1. Query pipeline declares depends_on for the indexing pipeline
2. Indexing pipeline runs first, populating ChromaDB via ChromaSink
3. Commencement gate verifies collection has documents
4. RAG transform readiness check passes
5. Questions are retrieved against the populated collection
6. Landscape records dependency correlation and gate results

Uses tmp_path for all output to avoid polluting the examples/ directory.
"""

from __future__ import annotations

import csv
import shutil
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

    # Write pipeline configs with tmp_path-relative paths
    chroma_dir = example_dst / "chroma_data"
    audit_db = example_dst / "runs" / "audit.db"

    index_yaml = example_dst / "index_pipeline.yaml"
    index_yaml.write_text(f"""\
source:
  plugin: csv
  on_success: output
  options:
    path: {example_dst / "documents.csv"}
    schema:
      mode: fixed
      fields:
        - "doc_id: str"
        - "text_content: str"
        - "topic: str"
        - "subtopic: str"
    on_validation_failure: discard

sinks:
  output:
    plugin: chroma_sink
    options:
      collection: smoke-test-facts
      mode: persistent
      persist_directory: {chroma_dir}
      distance_function: cosine
      field_mapping:
        document: text_content
        id: doc_id
        metadata:
          - topic
          - subtopic
      on_duplicate: overwrite
      schema:
        mode: fixed
        fields:
          - "doc_id: str"
          - "text_content: str"
          - "topic: str"
          - "subtopic: str"

landscape:
  url: sqlite:///{audit_db}
""")

    query_yaml = example_dst / "query_pipeline.yaml"
    query_yaml.write_text(f"""\
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
    on_fail: abort

source:
  plugin: csv
  on_success: retrieve
  options:
    path: {example_dst / "questions.csv"}
    schema:
      mode: fixed
      fields:
        - "id: int"
        - "question: str"
    on_validation_failure: discard

transforms:
  - name: retrieve
    plugin: rag_retrieval
    input: retrieve
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
          - "id: int"
          - "question: str"

sinks:
  output:
    plugin: json
    options:
      path: {example_dst / "output" / "results.jsonl"}
      schema:
        mode: observed
      format: jsonl
  quarantine:
    plugin: json
    options:
      path: {example_dst / "output" / "quarantined.jsonl"}
      schema:
        mode: observed
      format: jsonl

landscape:
  url: sqlite:///{audit_db}
""")

    return example_dst


class TestRAGIndexedSmoke:
    """Full end-to-end smoke test for the RAG ingestion pipeline."""

    def test_full_pipeline_sequence(self, example_dir: Path) -> None:
        """Run query pipeline — depends_on triggers indexing first."""
        from elspeth.engine.bootstrap import bootstrap_and_run

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
        import json
        import sqlite3

        from elspeth.engine.bootstrap import bootstrap_and_run

        query_yaml = example_dir / "query_pipeline.yaml"
        result = bootstrap_and_run(query_yaml)

        # Query the preflight_results table — linked to the run via run_id FK
        audit_db = example_dir / "runs" / "audit.db"
        conn = sqlite3.connect(str(audit_db))
        cursor = conn.execute(
            "SELECT result_type, name, result_json FROM preflight_results "
            "WHERE run_id = ? ORDER BY created_at",
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
        assert "env" not in gate_data.get("context_snapshot", {})

        # --- readiness_check recorded (from RAG transform on_start) ---
        readiness_rows = [(name, json.loads(rj)) for rt, name, rj in rows if rt == "readiness_check"]
        assert len(readiness_rows) >= 1  # At least one from the RAG transform
```

- [ ] **Step 2: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/integration/engine/test_rag_indexed_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/engine/test_rag_indexed_smoke.py
git commit -m "test: add end-to-end smoke test for RAG indexing pipeline"
```

---

### Task 5: Example README

**Files:**
- Create: `examples/chroma_rag_indexed/README.md`

- [ ] **Step 1: Write the README**

```markdown
# ChromaDB RAG with Automated Indexing

This example demonstrates the full RAG ingestion story in ELSPETH:

1. **Indexing pipeline** (`index_pipeline.yaml`): Reads documents from CSV and writes them to a ChromaDB collection via `ChromaSink`.
2. **Query pipeline** (`query_pipeline.yaml`): Uses `depends_on` to run the indexing pipeline first, a commencement gate to verify the collection is populated, then retrieves context via RAG for each question.

## Usage

Run the query pipeline — it automatically indexes the corpus first:

    elspeth run --settings examples/chroma_rag_indexed/query_pipeline.yaml --execute

The indexing pipeline runs as a dependency before the query pipeline starts.

## What Happens

1. `depends_on` triggers `index_pipeline.yaml`
2. CSV source reads `documents.csv` (10 science/health documents)
3. `ChromaSink` upserts documents into ChromaDB collection `science-facts-indexed`
4. Commencement gate checks `collections['science-facts-indexed']['count'] > 0`
5. RAG retrieval transform's readiness check verifies collection has documents
6. Questions from `questions.csv` are augmented with retrieved context
7. Results written to `output/results.jsonl`

## Files

- `documents.csv` — Reference corpus (10 documents)
- `questions.csv` — Questions to answer (5 questions)
- `index_pipeline.yaml` — CSV → ChromaSink indexing pipeline
- `query_pipeline.yaml` — depends_on + gate + RAG retrieval query pipeline
- `chroma_data/` — Persistent ChromaDB store (created on first run)
- `runs/audit.db` — Landscape audit trail (shared by both pipelines)
- `output/results.jsonl` — Retrieved context results
```

- [ ] **Step 2: Commit**

```bash
git add examples/chroma_rag_indexed/README.md
git commit -m "docs: add README for chroma_rag_indexed example"
```

---

### Task 6: Manual Run and Final Verification

- [ ] **Step 1: Run the full example manually**

Run: `elspeth run --settings examples/chroma_rag_indexed/query_pipeline.yaml --execute`
Expected: Both pipelines run. Output written to `examples/chroma_rag_indexed/output/results.jsonl`.

- [ ] **Step 2: Verify the output**

Check that `results.jsonl` contains 5 rows, each with `sci__rag_context`, `sci__rag_score`, `sci__rag_count`, and `sci__rag_sources` fields.

- [ ] **Step 3: Run the full integration test suite**

Run: `.venv/bin/python -m pytest tests/integration/ -x -q`
Expected: PASS

- [ ] **Step 4: Clean up example output (don't commit generated files)**

```bash
rm -rf examples/chroma_rag_indexed/chroma_data examples/chroma_rag_indexed/runs examples/chroma_rag_indexed/output
```

Add to `.gitignore` if not already covered:

```
examples/chroma_rag_indexed/chroma_data/
examples/chroma_rag_indexed/runs/
examples/chroma_rag_indexed/output/
```

- [ ] **Step 5: Final commit**

```bash
git add .gitignore
git commit -m "chore: gitignore generated output for chroma_rag_indexed example"
```
