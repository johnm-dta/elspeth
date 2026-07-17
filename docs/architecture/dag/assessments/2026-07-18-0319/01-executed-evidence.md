# DAG integration delta executed evidence

All commands ran from
`/home/john/elspeth/.worktrees/dag-scenario-corpus`. Production behavior was
assessed at merge commit `0235739274b534bd9e4e2b859bdd94a0b6a09651`; the
manifest and documentation contracts were then rerun against the uncommitted
reconciliation working tree.

## Evidence ledger

| Command or probe | Result | Establishes | Does not establish |
| --- | --- | --- | --- |
| `git merge-tree --write-tree --name-only --messages --merge-base 32c723c182f284cdb08d0bfae853e66f271cdf6a release/0.7.1 codex/dag-scenario-corpus` | Exit 0; tree `40071f019230564b00e7e9785aebcd380ea85cae`; no conflict messages | The two committed branches merge textually without conflict. | Runtime compatibility or semantic correctness. |
| `.venv/bin/pytest -q tests/unit/architecture/test_dag_scenario_corpus_contract.py tests/integration/core/dag/test_dag_scenario_production_path.py` | **147 passed** in 8.98s | The strict inventory, evidence contracts, two registered production-path cases, audit export, database reopen, and public resume remain green after integration. | The thirteen scenarios without executable harness cases. |
| [Pre-delta selector command](#pre-delta-selector-command) | **195 passed**, 13 warnings, in 10.81s from 47 locators at the clean merge baseline | Every previously registered pytest evidence selector executes successfully on the merge. | PostgreSQL testcontainer evidence, which the default marker expression excludes. |
| `.venv/bin/pytest -q tests/unit/architecture/test_dag_scenario_corpus_contract.py tests/integration/core/dag/test_dag_scenario_production_path.py` on the final reconciliation working tree | **153 passed**, exit 0 | The reconciled status matrix, evidence inventory, independent-validation record, assessment links, and production harness agree. | Unregistered scenarios and whole-repository behavior. |
| [Final non-testcontainer selector command](#final-non-testcontainer-selector-command) | **197 passed**, 13 warnings, in 11.02s from 49 locators on the reconciliation working tree | Every final non-testcontainer evidence selector executes successfully, including the two new SQLite expansion locators. | The separately opted-in PostgreSQL locator. |
| `.venv/bin/pytest -q tests/integration/test_multisource_provenance_proof.py tests/integration/pipeline/test_eof_resume_proof.py tests/e2e/recovery/test_crash_and_resume.py` | **17 passed** in 4.14s | Adjacent multi-source, EOF resume, and crash/restart behavior did not regress. | The complete crash-seam matrix. |
| `.venv/bin/pytest -q tests/unit/core/landscape/test_graph_recording.py tests/unit/core/landscape/test_journal.py tests/unit/core/landscape/repository_integration/test_recorder_tokens.py tests/unit/engine/test_processor.py tests/unit/engine/test_executors.py tests/unit/engine/test_token_traversal_characterization.py` | **606 passed** in 5.22s | The merged SQLite/unit surfaces for expansion, output-contract CAS, sidecar outbox, shutdown diversion, and bounded branch-loss reasons are green. | Cross-backend behavior by itself. |
| `.venv/bin/pytest -q -n 0 tests/unit/core/landscape/test_token_recording.py::TestExpandToken::test_batch_expansion_claim_is_scoped_to_batch_not_selected_parent tests/unit/architecture/test_dag_scenario_corpus_contract.py::test_row_expansion_delta_is_backed_by_repaired_cross_backend_evidence` | **2 passed** in 0.96s | Sequential replay through a different batch member is refused and the exact locator is bound into the live manifest. | Database reopen, process death, or multi-process contention. |
| `.venv/bin/pytest -q tests/testcontainer/core/test_token_outcome_atomicity_postgres.py::test_postgres_batch_expansion_claims_batch_once_under_contention tests/testcontainer/core/test_output_contract_concurrency_postgres.py tests/testcontainer/core/test_journal_postgres.py` | Exit 5; **5 deselected, no tests collected** | The default suite intentionally excludes `testcontainer`. This result was diagnostic only. | Any product claim. |
| `.venv/bin/pytest -q -n 0 -m testcontainer tests/testcontainer/core/test_token_outcome_atomicity_postgres.py::test_postgres_batch_expansion_claims_batch_once_under_contention tests/testcontainer/core/test_output_contract_concurrency_postgres.py tests/testcontainer/core/test_journal_postgres.py` | **5 passed** in 10.68s | PostgreSQL contention admits one batch expansion, output-contract writers converge or fail closed, and sidecar publication follows committed durable state. | Other PostgreSQL or multi-process scenarios. |
| `.venv/bin/ruff check tests/fixtures/dag_scenario_corpus tests/unit/architecture/test_dag_scenario_corpus_contract.py tests/integration/core/dag/test_dag_scenario_production_path.py` | Exit 0, all checks passed | Changed corpus Python satisfies Ruff. | Type correctness or runtime behavior. |
| `.venv/bin/ruff format --check tests/fixtures/dag_scenario_corpus tests/unit/architecture/test_dag_scenario_corpus_contract.py tests/integration/core/dag/test_dag_scenario_production_path.py` | Exit 0, 7 files already formatted | Changed corpus Python satisfies formatting. | Runtime behavior. |
| `.venv/bin/mypy tests/fixtures/dag_scenario_corpus tests/unit/architecture/test_dag_scenario_corpus_contract.py tests/integration/core/dag/test_dag_scenario_production_path.py` | Exit 0, no issues in 7 files | Changed corpus Python satisfies the focused type gate. | Whole-repository typing. |
| `/home/john/.local/bin/uv lock --check` using uv 0.10.2 | Exit 0; 213 packages resolved | The integrated lockfile is current. | Dependency installation on every supported platform. |
| `git diff --cached --check` after staging all reconciliation files | Exit 0 | The complete staged reconciliation delta, including newly added assessment files, contains no whitespace errors. | Whitespace in the already-committed merge or test behavior. |

## Exact selector commands

### Pre-delta selector command

The pre-delta command ran while the merge commit's manifest was still the
working-tree copy and contained 47 pytest references:

```bash
mapfile -t locators < <(.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

raw = yaml.safe_load(Path("docs/architecture/dag/scenario-corpus/v1/manifest.yaml").read_text())
for item in raw["evidence"]:
    if item["kind"] == "pytest":
        print(item["locator"])
PY
)
.venv/bin/pytest -q "${locators[@]}"
```

### Final non-testcontainer selector command

The final reconciliation command separated the explicitly opted-in testcontainer
locator from the 49 default-lane references:

```bash
mapfile -t locators < <(.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

raw = yaml.safe_load(Path("docs/architecture/dag/scenario-corpus/v1/manifest.yaml").read_text())
for item in raw["evidence"]:
    if item["kind"] == "pytest" and not item["locator"].startswith("tests/testcontainer/"):
        print(item["locator"])
PY
)
.venv/bin/pytest -q "${locators[@]}"
```

The first lock check attempted `.venv/bin/uv` and failed because this
worktree-local virtual environment does not install the `uv` executable. The
project command resolved to `/home/john/.local/bin/uv`; rerunning the same
read-only gate there succeeded. This was an invocation-path correction, not a
lockfile failure.

## Evidence interpretation

The results justify removing the three repaired defects from the hard-gate
list. They do not justify broad scenario promotion:

- the expansion fix proves atomic parent/batch consumption, sequential replay
  refusal, and selected cross-backend contention, but not the child-enqueue
  process-death seam;
- output-contract serialization and sidecar publication are repaired local
  invariants, while the wider atomicity matrix remains incomplete; and
- shutdown diversion, branch-loss redaction, exports, and URL projection
  strengthen adjacent evidence without completing their mandatory scenario
  rows.
