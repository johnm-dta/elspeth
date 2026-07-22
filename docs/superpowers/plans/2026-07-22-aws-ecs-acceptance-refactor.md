# AWS ECS Acceptance Controller Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `src/elspeth/web/aws_ecs_acceptance.py` from a 9,731-line monolith into focused private modules while preserving its module path, 24-command CLI, runbook invocation compatibility, fail-closed security contracts, and signed CI enforcement.

**Architecture:** Keep `elspeth.web.aws_ecs_acceptance` as the permanent executable facade containing `build_parser()`, `main()`, output helpers, compatibility re-exports, and the `__main__` guard. Move implementation into `elspeth.web._aws_ecs_acceptance` in dependency order: contracts and protected I/O first, pure validators second, domain services third, and evidence/cleanup orchestration last. Shared AWS-region validation belongs in `contracts.py`; S3, Bedrock, and telemetry must not import one another. Do not mix behavioral fixes into mechanical extraction commits.

**Tech Stack:** Python 3.13, argparse, httpx, AWS clients, SQLAlchemy/SQLite, pytest, Ruff, mypy, uv, Loomweave, Filigree, elspeth-lints, and elspeth-judge.

**Current readiness:** **NO-GO pending a green target base and repin.** Live verification on 2026-07-23 found four unrelated repo-wide gates red at the pinned `BASE_SHA`; the fixed scope below forbids repairing them in this refactor.

**Prerequisites:**
- This plan is pinned to `TARGET_RELEASE_BRANCH=release/0.7.2` at `BASE_SHA=59af9eab0c96c5faaecfe54be97c8b12a2a74e17`. Do not recompute the base with `merge-base`. If the intended base changes, amend the literal SHA everywhere and re-run plan review before coding.
- Before Task 0, repair the following failures on `TARGET_RELEASE_BRANCH` in separately tracked work, then update every `BASE_SHA` literal to that green commit and re-run this plan review. No implementation task may start from the currently pinned red base:
  - `scripts/check_contracts.py`: 14 unrelated Composer `dict[str, Any]` violations; existing tracker issue `elspeth-a16b05298a` must be refreshed to the live scope and closed with its fix commit;
  - `plugin_contract.component_type,plugin_contract.plugin_hashes`: stale `JSONSink.source_file_hash` in `plugins/sinks/json_sink.py`;
  - `immutability.freeze_guards,immutability.frozen_annotations`: 14 unrelated Composer FG2/FG3 findings; and
  - `trust_boundary.tests,trust_boundary.scope,trust_boundary.tier`: the `chat_solver.py:1411` test fingerprint records `283f5a4c...` but resolves to `f780f406...`.
- The repinned commit must pass every source-sensitive command listed in Task 15 before worktree creation. Pre-existing failures are not waivers: live CI runs these repository-wide, and this plan's fixed scope does not authorize their repair.
- Fingerprint-baseline issue `elspeth-18fe6e759e` is explicitly incorporated into Task 16. It may remain open while Tasks 0-15 execute, but no new baseline may be generated and neither the refactor parent nor a release milestone may close until the operator signs the frozen tree, regenerates the baseline, records the resulting commit on that P0, and closes it.
- Execute in a clean dedicated worktree created from the pinned `BASE_SHA`. The current root checkout contains unrelated user-owned work; never stage or alter it.
- The plan and review JSON are currently untracked and therefore are not present in a worktree created from `BASE_SHA`. Until the developer commits them and repins the base, the executor must use this absolute plan path read-only: `/home/john/elspeth/docs/superpowers/plans/2026-07-22-aws-ecs-acceptance-refactor.md`. Never add the review JSON to an implementation commit.
- Before coding, verify the key-free `elspeth-judge` MCP exposes `verify_signatures`, `stage_scan`, and `stage_status` from the dedicated worktree. If those tools are absent, reconnect or repair the worktree-local MCP registration before starting; do not postpone discovery until the source is frozen.
- Create one Filigree parent task for this plan and one child task per numbered implementation task before coding. Do not attach the work to a release milestone unless the developer explicitly includes the refactor in that release.
- Build a complete worktree-local Python 3.13 environment with `env -u VIRTUAL_ENV uv sync --frozen --all-extras`.
- Do not edit `uv.lock` or another lock file. This refactor adds no dependency.
- Treat approximately 650 Judge decisions as accepted finalization cost. Do not weaken rules, add blanket suppressions, move code outside `src/elspeth`, or exclude the controller.

---

## Fixed scope and compatibility contract

Allowed implementation files:

- `src/elspeth/web/aws_ecs_acceptance.py`
- new modules under `src/elspeth/web/_aws_ecs_acceptance/`
- `tests/unit/web/test_aws_ecs_acceptance.py`
- new tests under `tests/unit/web/aws_ecs_acceptance/`
- `tests/unit/web/test_aws_ecs_runbook_contract.py` only for additional assertions
- `tests/unit/web/test_landscape_access_guard.py` for reviewed source-path movement
- `tests/unit/architecture/test_aws_ecs_acceptance_dependencies.py`
- `tests/unit/elspeth_lints/fixtures/fingerprint_baseline.json` after signed enforcement is green
- affected `config/cicd/enforce_tier_model/*.yaml` files written by operator tooling

Non-goals:

- no edit to `docs/runbooks/aws-ecs-deployment.md`;
- no CLI rename, option change, output-shape change, or error-class change;
- no change to AWS ownership, cleanup ordering, retry budgets, protected-file permissions, receipt schemas, approval semantics, telemetry identities, or redaction;
- no generalized framework for other acceptance controllers;
- no conversion of `aws_ecs_acceptance.py` into a same-named package;
- no unrelated defect, lint, typing, dependency, or release-process work.

### Runbook compatibility claim

This refactor makes a no-regression claim about the tracked runbook, not a fresh operational certification of that runbook:

- `docs/runbooks/aws-ecs-deployment.md` remains byte-for-byte unchanged from `BASE_SHA`;
- its 55 textual `python -m elspeth.web.aws_ecs_acceptance` invocation sites remain present; and
- the characterized parser and dispatcher remain compatible with the unchanged invocation text.

This does not claim that the complete runbook is currently safe to execute. Its Terraform package is external to this repository, and its hard-coded schema-epoch assumptions must be reconciled before use. Ordinary deployment of an existing acceptance service uses the `operating-aws-ecs-container` skill instead.

The operational boundary remains:

```text
python -m elspeth.web.aws_ecs_acceptance COMMAND [OPTIONS]
```

The exact command set remains:

```python
EXPECTED_COMMANDS = {
    "capture", "provision-storage", "scenario-namespace", "verify-api",
    "verify-payloads", "verify-local-auth", "verify-s3", "verify-bedrock",
    "verify-bedrock-guardrails", "verify-connection-budget",
    "verify-operator-telemetry", "extract-exec-receipt", "sanitize-evidence",
    "control-manifest", "gate-ledger", "receipt-store", "approval-verify",
    "approval-require-current", "scenario-load", "validate-task-definition-policy",
    "compatibility-record-validate", "orphan-sweep", "cleanup-evidence-finalize",
    "evidence-export-receipt",
}
```

`SCENARIO_ASSIGNMENT_NAMES` remains directly importable. Preserve original non-private symbols as facade re-exports for this refactor; deprecation is separate work. Migrate tests away from patching incidental imported dependencies through the facade.

## Target structure and dependency direction

```text
src/elspeth/web/aws_ecs_acceptance.py
src/elspeth/web/_aws_ecs_acceptance/
├── __init__.py
├── contracts.py
├── secure_documents.py
├── state.py
├── http_client.py
├── capture.py
├── receipt_contracts.py
├── s3.py
├── bedrock.py
├── operator_telemetry.py
├── manifest_schema.py
├── scenario_inventory.py
├── manifest.py
├── task_definition.py
├── orphan_sweep.py
├── receipt_store.py
├── approvals.py
├── gate_ledger.py
├── evidence.py
├── cleanup.py
└── control_service.py
```

```text
contracts
  -> secure_documents / state / http_client / receipt_contracts
  -> capture / s3 / bedrock / operator_telemetry
  -> manifest_schema / scenario_inventory / gate_ledger
  -> manifest / task_definition / orphan_sweep / receipt_store / approvals / evidence
  -> cleanup / control_service
  -> aws_ecs_acceptance facade
```

An arrow means the module on the left may be imported by the module on the right. Lower layers must not import the facade. Schema modules must not import mutation services. `gate_ledger.py` must not import `evidence.py`, `cleanup.py`, or `control_service.py`. `manifest.py` owns low-level mutations and must not import `receipt_store.py`, `approvals.py`, `evidence.py`, `cleanup.py`, or `control_service.py`. `control_service.py` is the high-level owner permitted to compose manifest, receipts, approvals, evidence, and ledger services. `cleanup.py` must not import `control_service.py`.

`contracts.py` owns shared AWS-region validation. `s3.py`, `bedrock.py`, and `operator_telemetry.py` import `contracts._resolve_aws_region` directly. No domain module may own, duplicate, or import this helper from another domain module.

## Verification and commit discipline for Tasks 2-14

After each task's focused command, run this exact extraction gate before committing:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance \
  tests/unit/web/test_aws_ecs_runbook_contract.py \
  tests/unit/web/test_landscape_access_guard.py \
  tests/unit/architecture/test_aws_ecs_acceptance_dependencies.py \
  tests/unit/architecture/test_sink_publication_callers.py -q
env -u VIRTUAL_ENV uv run --frozen ruff check \
  src/elspeth/web/aws_ecs_acceptance.py \
  src/elspeth/web/_aws_ecs_acceptance \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance
env -u VIRTUAL_ENV uv run --frozen mypy \
  src/elspeth/web/aws_ecs_acceptance.py \
  src/elspeth/web/_aws_ecs_acceptance
git diff --check
```

Expected: all four commands exit zero. Then use `git status --short`, stage only paths listed in the current task, run `git diff --cached --check`, and commit with the exact message stated by that task. Never stack a new task on a red gate.

Tasks 2-15 have code-only rollback: revert the current extraction commit, restore its child issue to working status, and correct the plan or implementation before continuing. Task 16 is the exception because signed metadata, the fingerprint fixture, and the exact source tree form one coupled state:

- A staging or dry-run abort writes no authority; stage fresh.
- A non-zero fire may have written earlier accepted actions. Do not regenerate the baseline. Keep source frozen and either stage fresh to complete the same tree or restore the entire operator-written allowlist directory to `SIGNING_START_SHA` before changing source or abandoning the attempt. Never hand-edit individual signature rows.
- If regeneration fails after writing the fixture, restore both the allowlist directory and fingerprint fixture to `SIGNING_START_SHA`, then diagnose before retrying.
- After the Task 16 commit, never revert only source commits or only the signing/baseline commit. Roll them back together on a dedicated rollback branch, rerun Task 15 on the restored layout, then execute a fresh complete Task 16. Reopen `elspeth-18fe6e759e` or create a linked operator blocker before fresh signing. Because that P0 predates this refactor, merely reverting the signing commit returns to a known-stale baseline and is not a completed rollback.

---

### Task 0: Create the isolated execution baseline

**Files:**
- Read: `AGENTS.md`
- Read: `src/elspeth/web/aws_ecs_acceptance.py`
- Read: `tests/unit/web/test_aws_ecs_acceptance.py`
- Read: `docs/judge-signature-handoff.md`

- [ ] **Step 1: Create and enter the worktree**

```bash
TARGET_RELEASE_BRANCH=release/0.7.2
BASE_SHA=59af9eab0c96c5faaecfe54be97c8b12a2a74e17
IMPLEMENTATION_BRANCH=refactor/aws-ecs-acceptance
WORKTREE=/home/john/elspeth-aws-ecs-acceptance-refactor
test "$(git rev-parse "${TARGET_RELEASE_BRANCH}^{commit}")" = "$BASE_SHA"
test ! -e "$WORKTREE"
! git show-ref --verify --quiet "refs/heads/$IMPLEMENTATION_BRANCH"
git worktree add "$WORKTREE" -b "$IMPLEMENTATION_BRANCH" "$BASE_SHA"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$BASE_SHA"
test -z "$(git status --porcelain)"
env -u VIRTUAL_ENV uv sync --frozen --all-extras
env -u VIRTUAL_ENV uv pip install --python .venv/bin/python \
  -e './elspeth-lints[mcp,judge-agent]'
.venv/bin/python -c 'import claude_agent_sdk, mcp, elspeth_lints.mcp'
git diff --exit-code -- uv.lock
```

Expected: the ref fails closed if it moved since review; otherwise the branch starts at the exact pinned SHA with a complete local `.venv`, including the MCP/read-only agent Judge runtime, and no source edits or lock-file change. Root `[all]` does not install `claude-agent-sdk`, so the explicit editable `elspeth-lints[mcp,judge-agent]` install is required.

- [ ] **Step 2: Verify the baseline owner**

Use Filigree MCP `issue_get` for `elspeth-18fe6e759e`; CLI fallback:

```bash
filigree show elspeth-18fe6e759e
```

Expected: either already closed with a commit anchor, or still operator-owned with the signing/baseline scope stated in the prerequisites. If its scope, owner, or resolution changed, stop and amend this plan rather than generating a competing baseline.

- [ ] **Step 3: Verify the key-free Judge handoff before source work**

From an agent session whose working directory is this dedicated worktree, inspect MCP `tools/list` and require the `elspeth-judge` tools `verify_signatures`, `stage_scan`, and `stage_status`. Confirm the server environment does not contain `ELSPETH_JUDGE_METADATA_HMAC_KEY`.

For Codex, verify or create a registration whose executable and all relative inputs resolve inside the dedicated worktree:

```bash
test -z "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}"
judge_root=$(pwd -P)
if ! codex mcp get elspeth-judge >/dev/null 2>&1; then
  codex mcp add \
    --env "PYTHONPATH=$judge_root/elspeth-lints/src" \
    elspeth-judge -- \
    "$judge_root/.venv/bin/python" -m elspeth_lints.mcp \
    --root "$judge_root/src/elspeth" \
    --allowlist-dir "$judge_root/config/cicd/enforce_tier_model" \
    --staged-dir "$judge_root/.elspeth/staged-reviews"
fi
codex mcp get elspeth-judge --json
```

Inspect the JSON and stop if any path points at another checkout. Start a fresh Codex session from the dedicated worktree when required for the newly registered tools to appear.

Expected: all five staging tools (`verify_signatures`, `stage_scan`, `stage_status`, `stage_preview`, and `stage_rekey`) are callable from this worktree and remain key-free. If absent, reconnect or repair registration now; do not use an ad hoc signing script or move the operator key into the agent environment.

- [ ] **Step 4: Record the green baseline**

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/test_aws_ecs_runbook_contract.py \
  tests/unit/web/test_landscape_access_guard.py \
  tests/unit/architecture/test_sink_publication_callers.py -q
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/test_aws_ecs_runbook_contract.py -q \
  --cov=elspeth.web.aws_ecs_acceptance --cov-branch --cov-report=
env -u VIRTUAL_ENV uv run --frozen coverage report \
  --include='src/elspeth/web/aws_ecs_acceptance.py' --fail-under=73
```

Expected: PASS. Record counts and the branch-inclusive controller coverage in the Filigree parent; current acceptance collection is 273 nodes, the runbook contract is 41 nodes, and their measured controller coverage is 73.480342% (4,781 statements and 2,112 branches). The 73% floor is a breadth backstop, not proof of assertion strength.

- [ ] **Step 5: Record the pure CLI probe**

```bash
env -u VIRTUAL_ENV uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
  scenario-namespace \
  --acceptance-run-id 00000000-0000-4000-8000-000000000001 \
  --scenario-id A
```

Expected stdout: `a-f8b447a7b5b51f38c800`.

**Definition of Done:** exact pinned worktree, complete environment, incorporated baseline owner, usable key-free Judge handoff, green tests, and recorded CLI output. No commit.

---

### Task 1: Add permanent facade characterization coverage

**Files:**
- Create: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`
- Create: `tests/unit/web/aws_ecs_acceptance/fixtures/facade_contract.json`
- Create: `tests/unit/web/aws_ecs_acceptance/fixtures/collected_ids.txt`
- Create: `tests/unit/architecture/test_aws_ecs_acceptance_dependencies.py`

- [ ] **Step 1: Capture the original parameterized test identities**

Run the current monolith's collection and normalize away only the source file path:

```bash
set -o pipefail
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py --collect-only -q \
  | rg '::test_' | sed -E 's#^[^:]+::##' | sort -u
```

Expected: one stable line per original test function/parameter ID. Add the complete output to `fixtures/collected_ids.txt` with `apply_patch`; do not use shell redirection to write repository files.

- [ ] **Step 2: Add the recursive facade snapshot**

In `test_facade_contract.py`, implement `_parser_surface(parser)` to recurse through every `argparse._SubParsersAction` and return a sorted JSON-safe object with `actions` and `mutually_exclusive_groups`. Every action record must contain command path, action class, `dest`, sorted option strings, `required`, `nargs`, sorted choices, `repr(default)`, `repr(const)`, `repr(metavar)`, and stable callable identity for `type` such as `builtins.int`. Every mutually-exclusive-group record must contain command path, group `required`, and sorted member records containing each member's `dest` and option strings. Group membership cannot be inferred from action-level `required`: the two `receipt-store` actions are individually optional while their group is required. Implement `_defined_public_symbols(source)` with `ast.parse` so it captures module-defined non-private functions, async functions, classes, assignments, and annotated assignments while excluding imported names.

Run the helpers against the untouched module, inspect the JSON, and add it to `fixtures/facade_contract.json` using this schema:

```json
{
  "defined_public_symbols": ["AWSOperatorMetricEmitter", "AWSOperatorTelemetryQueries"],
  "parser_surface": {
    "actions": [
      {
        "action": "_StoreAction",
        "choices": null,
        "command_path": ["verify-connection-budget"],
        "const": "None",
        "default": "None",
        "dest": "approved_budget",
        "metavar": "None",
        "nargs": null,
        "option_strings": ["--approved-budget"],
        "required": true,
        "type": "builtins.int"
      }
    ],
    "mutually_exclusive_groups": [
      {
        "command_path": ["receipt-store"],
        "members": [
          {"dest": "receipt_file", "option_strings": ["--receipt-file"]},
          {"dest": "receipt_stdin", "option_strings": ["--receipt-stdin"]}
        ],
        "required": true
      }
    ]
  }
}
```

The shown arrays are schema examples, not the complete fixture; the committed fixture must contain the complete mechanically captured lists from the untouched source.

- [ ] **Step 3: Add the contract test**

```python
from __future__ import annotations

import argparse
import subprocess
import sys

from elspeth.web import aws_ecs_acceptance as acceptance
from elspeth.web.aws_ecs_acceptance import SCENARIO_ASSIGNMENT_NAMES


def test_module_entrypoint_and_import_surface_remain_stable() -> None:
    parser = acceptance.build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(action.choices) == EXPECTED_COMMANDS
    assert "ACTIVE_SCENARIO_ID" in SCENARIO_ASSIGNMENT_NAMES

    completed = subprocess.run(
        [sys.executable, "-m", "elspeth.web.aws_ecs_acceptance", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "python -m elspeth.web.aws_ecs_acceptance" in completed.stdout
```

Define `EXPECTED_COMMANDS` locally using the exact set above. Load `facade_contract.json`, assert `_parser_surface(acceptance.build_parser())` equals the complete stored parser surface, and assert every stored defined public symbol exists on the facade.

Add a table-driven `main(argv)` contract using the real parser. Each row supplies valid CLI arguments, patches the facade global through which `main()` deliberately dispatches, and asserts exact positional arguments, keyword arguments, `Path` conversion, integer/boolean coercion, return code, stdout, and stderr. Patch domain owners in domain tests, but patch facade globals here because dispatch intentionally resolves the facade-bound compatibility aliases. Require the table's path-to-target mapping to equal this complete 36-leaf mapping:

```python
EXPECTED_DISPATCH_TARGETS = {
    ("approval-require-current",): "approval_require_current",
    ("approval-verify",): "approval_verify",
    ("capture",): "capture",
    ("cleanup-evidence-finalize",): "cleanup_evidence_finalize",
    ("compatibility-record-validate",): "validate_compatibility_record",
    ("control-manifest", "bind-retained-evidence"): "control_manifest_bind_retained_evidence",
    ("control-manifest", "bind-scenario"): "control_manifest_bind_scenario",
    ("control-manifest", "checkpoint-operator-evidence"): "control_manifest_checkpoint_operator_evidence",
    ("control-manifest", "get"): "control_manifest_get",
    ("control-manifest", "init"): "control_manifest_init",
    ("control-manifest", "load-cleanup"): "control_manifest_load_cleanup",
    ("control-manifest", "update"): "control_manifest_update",
    ("control-manifest", "validate"): "control_manifest_validate",
    ("evidence-export-receipt",): "create_evidence_export_receipt",
    ("extract-exec-receipt",): "extract_exec_receipt",
    ("gate-ledger", "bind-candidate"): "gate_ledger_bind_candidate",
    ("gate-ledger", "finalize"): "gate_ledger_finalize",
    ("gate-ledger", "get"): "gate_ledger_get",
    ("gate-ledger", "init"): "gate_ledger_init",
    ("gate-ledger", "record"): "gate_ledger_record",
    ("gate-ledger", "record-cleanup"): "gate_ledger_record_cleanup",
    ("orphan-sweep",): "orphan_sweep",
    ("provision-storage",): "provision_storage",
    ("receipt-store",): "receipt_store",
    ("sanitize-evidence",): "sanitize_evidence",
    ("scenario-load",): "scenario_load",
    ("scenario-namespace",): "scenario_resource_namespace",
    ("validate-task-definition-policy",): "validate_task_definition_policy_binding",
    ("verify-api",): "verify_api",
    ("verify-bedrock",): "verify_bedrock",
    ("verify-bedrock-guardrails",): "run_bedrock_guardrails_live",
    ("verify-connection-budget",): "verify_connection_budget_live",
    ("verify-local-auth",): "verify_local_auth",
    ("verify-operator-telemetry",): "verify_operator_telemetry_live",
    ("verify-payloads",): "verify_payloads",
    ("verify-s3",): "verify_s3",
}
```

The cases must supply bounded text/byte stdin for `extract-exec-receipt`, `receipt-store --receipt-stdin`, `sanitize-evidence`, and task-definition validation; cover both receipt input alternatives; use an async spy for `verify-bedrock`; stub `_suppress_process_output`, `resolve_exec_receipt_env`, and `encode_exec_receipt` for all five live verifier paths; assert successful no-output commands leave both streams empty; and assert JSON/line-output commands emit exactly one safe payload.

Add parser-negative cases for neither/both receipt inputs, invalid integer input, and a missing nested action; each must exit 2. Add dispatcher-negative cases for mismatched cleanup flags and missing guardrail policy binding, then inject `AcceptanceCheckError`, each of `AcceptanceHttpError`, `AcceptanceInputError`, `AcceptanceStateError`, and `OperatorTelemetryAcceptanceError`, plus an unexpected exception whose message contains `secret-sentinel`. For every caught runtime failure assert return code 1, empty stdout, one exact newline-terminated JSON object on stderr, and no message/sentinel leakage. The expected envelopes are:

```text
{"check":"sentinel","error_class":"AcceptanceCheckError"}
{"error_class":"AcceptanceHttpError"}
{"error_class":"AcceptanceInputError"}
{"error_class":"AcceptanceStateError"}
{"error_class":"OperatorTelemetryAcceptanceError"}
{"error_class":"AcceptanceInternalError"}
```

Add an `EXPECTED_OWNERS: dict[str, str]` mapping. Each later extraction task must add every moved public symbol to this mapping and the test must assert:

```python
from importlib import import_module

for symbol, module_name in EXPECTED_OWNERS.items():
    owner = import_module(module_name)
    assert getattr(acceptance, symbol) is getattr(owner, symbol)
```

This makes compatibility exhaustive without preserving incidental imported dependencies.

- [ ] **Step 4: Add the private-package architecture guard**

Create an AST-based test that rejects every import of `elspeth.web.aws_ecs_acceptance` from the private package and checks private imports against this approved matrix:

```python
ALLOWED_INTERNAL_IMPORTS = {
    "__init__": set(),
    "contracts": set(),
    "secure_documents": {"contracts"},
    "state": {"contracts", "secure_documents"},
    "http_client": {"contracts"},
    "capture": {"contracts", "http_client", "state"},
    "receipt_contracts": {"contracts", "secure_documents"},
    "s3": {"contracts", "receipt_contracts"},
    "bedrock": {"contracts", "receipt_contracts"},
    "operator_telemetry": {"capture", "contracts", "receipt_contracts", "state"},
    "gate_ledger": {"contracts", "secure_documents"},
    "manifest_schema": {"contracts", "receipt_contracts", "secure_documents"},
    "scenario_inventory": {"contracts", "manifest_schema", "secure_documents"},
    "manifest": {"contracts", "gate_ledger", "manifest_schema", "receipt_contracts", "scenario_inventory", "secure_documents"},
    "task_definition": {"contracts", "manifest_schema", "scenario_inventory"},
    "orphan_sweep": {"contracts", "manifest_schema", "receipt_contracts", "scenario_inventory", "task_definition"},
    "receipt_store": {"contracts", "manifest", "manifest_schema", "receipt_contracts", "scenario_inventory", "secure_documents"},
    "approvals": {"contracts", "manifest_schema", "secure_documents"},
    "evidence": {"contracts", "gate_ledger", "manifest", "manifest_schema", "receipt_store", "secure_documents"},
    "cleanup": {"contracts", "evidence", "gate_ledger", "manifest", "manifest_schema", "receipt_store", "secure_documents"},
    "control_service": {"approvals", "contracts", "evidence", "gate_ledger", "manifest", "manifest_schema", "receipt_store", "scenario_inventory", "secure_documents"},
}
```

The test must parse both `Import` and `ImportFrom`, fail on an unknown private module or unapproved private dependency, and pass when the private directory does not yet exist.

- [ ] **Step 5: Run the characterization and architecture tests**

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/aws_ecs_acceptance/test_facade_contract.py \
  tests/unit/architecture/test_aws_ecs_acceptance_dependencies.py -q
```

Expected: PASS before production changes. If it fails, correct the test rather than production.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/web/aws_ecs_acceptance/test_facade_contract.py \
  tests/unit/web/aws_ecs_acceptance/fixtures \
  tests/unit/architecture/test_aws_ecs_acceptance_dependencies.py
git commit -m "test(web): pin AWS acceptance facade contract"
```

**Definition of Done:** all 24 top-level commands, all 36 dispatch leaves, parser coercion/default/const/metavar semantics, required mutually-exclusive membership, direct constant import, executable path, facade re-export identity, and static/redacted error mapping are pinned before extraction.

---

### Task 2: Extract contracts, protected documents, and state

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/__init__.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/contracts.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/secure_documents.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/state.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:568-931,1153-1210,1608-1611,2105-2117,3942-4067`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`

- [ ] Add failing identity assertions:

```python
from elspeth.web._aws_ecs_acceptance import contracts, state


def test_foundational_symbols_are_reexported_by_identity() -> None:
    assert acceptance.AcceptanceCheckError is contracts.AcceptanceCheckError
    assert acceptance.AcceptanceHttpError is contracts.AcceptanceHttpError
    assert acceptance.AcceptanceInputError is contracts.AcceptanceInputError
    assert acceptance.AcceptanceStateError is contracts.AcceptanceStateError
    assert acceptance.AcceptanceCredentials is contracts.AcceptanceCredentials
    assert acceptance.AcceptanceState is state.AcceptanceState
```

Expected before extraction: `ModuleNotFoundError`.

- [ ] Create side-effect-free `__init__.py` containing only the package docstring.
- [ ] Move to `contracts.py`: four acceptance errors, `AcceptanceCredentials`, origin normalization, mapping/string/UUID/SHA helpers, `_sha256`, `_utc_timestamp`, a new exception-neutral `_parse_utc_timestamp`, `_resolve_aws_region`, `_task_definition_family`, a pure `_validate_sanitized_resource_identity_fields`, shared budgets, shared identity regexes, and shared closed field sets. `_parse_utc_timestamp` must raise only `TypeError`/`ValueError`; domain wrappers retain their existing static error classes and messages.
- [ ] Move `_resolve_aws_region` without changing its fail-closed rules or caller-supplied static check identifier. S3, Bedrock/plugin-policy, operator telemetry, and connection-budget code must import this single implementation from `contracts.py`.
- [ ] Move to `secure_documents.py`: protected stat/destination/parent validation, bounded protected read/write, `_control_path`, and `_control_timestamp`. Implement `_control_timestamp` over `contracts._parse_utc_timestamp` and continue mapping invalid input to `AcceptanceCheckError("control_manifest_schema")`.
- [ ] Move to `state.py`: `AcceptanceState`, state field/timestamp helpers, `write_acceptance_state`, and `read_acceptance_state`. Keep `_parse_state_timestamp` as the state-domain wrapper over `contracts._parse_utc_timestamp`, preserving `AcceptanceStateError("acceptance state schema is invalid")`.
- [ ] Import/re-export moved symbols from the facade. Define each exception once.
- [ ] Patch state/protected-document owner modules in tests, including the interrupted manifest-write test.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance/test_facade_contract.py -q
env -u VIRTUAL_ENV uv run --frozen ruff check \
  src/elspeth/web/aws_ecs_acceptance.py src/elspeth/web/_aws_ecs_acceptance \
  tests/unit/web/aws_ecs_acceptance/test_facade_contract.py
env -u VIRTUAL_ENV uv run --frozen mypy \
  src/elspeth/web/aws_ecs_acceptance.py src/elspeth/web/_aws_ecs_acceptance
```

Expected: PASS, including identity assertions.

- [ ] Run the exact Tasks 2-14 extraction gate, including `test_aws_ecs_acceptance_dependencies.py` now that the private package exists.

- [ ] Commit:

```bash
git add src/elspeth/web/aws_ecs_acceptance.py src/elspeth/web/_aws_ecs_acceptance \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance/test_facade_contract.py
git commit -m "refactor(web): extract AWS acceptance foundations"
```

**Definition of Done:** one exception identity, unchanged protected-file behavior, unchanged state serialization, and no upward import.

---

### Task 3: Extract HTTP, capture, and local verification

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/http_client.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/capture.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:637-712,934-1150,1614-2081`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:137-1103`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`
- Modify: `tests/unit/web/test_landscape_access_guard.py`

- [ ] Add failing identity assertions for `AcceptanceHttpClient`, `capture`, `verify_api`, `verify_local_auth`, `provision_storage`, and `verify_payloads`.
- [ ] Move `AcceptanceHttpClient` and only request/authentication helpers to `http_client.py`.
- [ ] Move pipeline YAML builders, run/artifact selectors, `capture`, plugin-policy HTTP verification, `verify_api`, `verify_local_auth`, `provision_storage`, and `verify_payloads` to `capture.py`.
- [ ] Keep `scenario_resource_namespace` and `plugin_policy_binding_sha256` in `contracts.py` because later domains consume them.
- [ ] Change tests to patch `capture.settings_from_env`, `capture.LandscapeDB`, `capture.RecorderFactory`, and `capture.FilesystemPayloadStore` explicitly.
- [ ] Change one reviewed `LandscapeDB.from_url` tuple from `src/elspeth/web/aws_ecs_acceptance.py` to `src/elspeth/web/_aws_ecs_acceptance/capture.py`; leave the other three on the facade until Tasks 6 and 7.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "http or state or capture or verify_api or verify_local_auth or provision_storage or verify_payloads or fixed_pipeline" -q
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance/test_facade_contract.py -q
```

Expected: PASS with unchanged static errors and bounded responses.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract AWS acceptance HTTP and capture lanes`.

**Definition of Done:** HTTP/capture behavior has focused owners and tests patch the runtime owner.

---

### Task 4: Consolidate execution-receipt contracts

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/receipt_contracts.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:1213-1605,7616-8038`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:2724-3136,5394-5703`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`

- [ ] Add failing identity assertions for `resolve_exec_receipt_env`, `encode_exec_receipt`, and `extract_exec_receipt`.
- [ ] Move every receipt field set, pure `_validate_*_receipt*` function, bounded receipt parser, `_receipt_number`, Terraform/event-canary validators, `_validate_stored_receipt`, exec receipt encode/extract logic, `_expected_schema_facts`, candidate/rollback package versions, rollback schema epochs, structural-change facts, and the canonical operator receipt metric/trace identity constants into `receipt_contracts.py`.
- [ ] Make `_validate_exec_receipt_schema` call `_validate_connection_budget_receipt` in the same module. The module must import neither `manifest.py` nor `receipt_store.py`.
- [ ] Make receipt timestamp validators translate `contracts._parse_utc_timestamp` failures to their existing `AcceptanceCheckError` check IDs. Validate operator receipt resource fields through `contracts._validate_sanitized_resource_identity_fields`; do not import the later `operator_telemetry.py` class.
- [ ] Preserve prefix, maximum sizes, closed fields, static check names, candidate/task/scenario binding, guardrail policy binding, and telemetry identity.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "exec_receipt or compatibility or event_canary or terraform_receipt" -q
env -u VIRTUAL_ENV uv run --frozen pytest tests/unit/web/test_aws_ecs_acceptance.py -q
```

Expected: PASS without circular imports.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): consolidate AWS acceptance receipt contracts`.

**Definition of Done:** receipt schemas are a lower-level leaf and the forward dependency is removed.

---

### Task 5: Extract the S3 live lane

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/s3.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:2084-2376`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:1100-1456`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`

- [ ] Add failing identity assertions for `verify_s3` and `_S3_ACCEPTANCE_BYTES`.
- [ ] Move `_S3AcceptanceContext`, S3-specific input resolution, not-found classification, source hashing, effect identity, `_drive_s3_acceptance_effect`, `verify_s3`, and S3-only constants to `s3.py` without changing bodies. Import `contracts._resolve_aws_region`; do not move or duplicate it.
- [ ] Preserve injected clients, default-chain restrictions, sink-effect protocol use, collision rejection, cleanup-after-failure ordering, and static/redacted errors.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k "verify_s3 or s3_receipt" -q
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/architecture/test_sink_publication_callers.py -q
```

Expected: PASS, including the sink-publication guard.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract AWS S3 acceptance lane`.

**Definition of Done:** S3 has one owner and retains effect/cleanup invariants.

---

### Task 6: Extract Bedrock, guardrail, and plugin-policy acceptance

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/bedrock.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:2380-2946`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:1457-2049`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`
- Modify: `tests/unit/web/test_landscape_access_guard.py`

- [ ] Add failing identity assertions for `verify_bedrock`, `build_plugin_policy_acceptance`, `verify_bedrock_guardrails`, and `run_bedrock_guardrails_live`.
- [ ] Move fd-output suppression, response projection, Bedrock verification, profile registry, guardrail inputs/secret inventory, plugin-policy acceptance, guardrail verification, telemetry-manager construction, and live orchestration to `bedrock.py`.
- [ ] Import `contracts._resolve_aws_region`; preserve each caller's existing static check ID and exception behavior.
- [ ] Patch `bedrock.asyncio.wait_for` rather than `acceptance.asyncio.wait_for`.
- [ ] Change one remaining reviewed `LandscapeDB.from_url` tuple from the facade to `src/elspeth/web/_aws_ecs_acceptance/bedrock.py`.
- [ ] Preserve timeout mapping, fd suppression, safe/attack pairing, audit-first persistence, four-call ordering, policy hashes, and redaction.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "bedrock or guardrail or plugin_policy_acceptance" -q
```

Expected: PASS, including fd-capture assertions.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract Bedrock acceptance lanes`.

**Definition of Done:** Bedrock/guardrail behavior has one owner and unchanged audit/redaction ordering.

---

### Task 7: Extract operator telemetry and connection-budget verification

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/operator_telemetry.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:2949-3939`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:2050-2723`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`
- Modify: `tests/unit/web/test_landscape_access_guard.py`

- [ ] Add failing identity assertions for every moved public symbol: `OperatorTelemetryAcceptanceError`, `AuditSentinel`, `TelemetrySentinelEmitter`, `TelemetryQueries`, `AWSOperatorTelemetryQueries`, `SanitizedResourceIdentity`, `AcceptancePolicy`, `OperatorTelemetryEvidence`, `OperatorTelemetryOutageEvidence`, `PublicApiLifecycleAudit`, `ExistingLandscapeLifecycleAudit`, `AWSOperatorMetricEmitter`, `operator_metric_dimensions`, `xray_trace_id`, the three `verify_operator_telemetry*` functions, and `verify_connection_budget_live`.
- [ ] Move telemetry errors/protocols/queries, identities/evidence models, lifecycle audits, emitter, AWS observability construction, receipt construction, positive/outage verification, PostgreSQL limit reading, and connection-budget verification to `operator_telemetry.py`.
- [ ] Import `contracts._resolve_aws_region`; preserve each caller's existing static check ID and exception behavior.
- [ ] Implement `SanitizedResourceIdentity.__post_init__` through `contracts._validate_sanitized_resource_identity_fields`, so construction and stored-receipt validation share the same lower-level rule without a receipt/telemetry cycle.
- [ ] Import the canonical receipt metric/trace identity constants from `receipt_contracts.py`; do not duplicate them in telemetry construction.
- [ ] Change the final two reviewed `LandscapeDB.from_url` tuples from the facade to `src/elspeth/web/_aws_ecs_acceptance/operator_telemetry.py`.
- [ ] Preserve metric dimensions, trace correlation, retryable absence, malformed-provider static errors, collector degradation, external-stop ownership, and complete minute-grid validation.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "operator_telemetry or connection_budget or xray_trace_id" -q
```

Expected: PASS for positive, outage, repeated-window, sparse-grid, and provider-failure cases.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract operator telemetry acceptance`.

**Definition of Done:** telemetry protocols, adapters, and services are cohesive and fail closed.

---

### Task 8: Extract gate-ledger, control-manifest, and scenario schemas

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/gate_ledger.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/manifest_schema.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/scenario_inventory.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:4070-5252,8841-8951`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:3137-4630`

- [ ] Add direct owner-module tests for pure validators while keeping facade behavior tests unchanged.
- [ ] Move gate-record hashing, record-stream validation, ledger schema validation, `_read_gate_ledger`, and gate-order constants to `gate_ledger.py`. Leave mutation operations in the facade until Task 12.
- [ ] Move `_validate_control_manifest`, `_read_control_manifest`, `_scenario_inventory_hash`, and control field/order constants to `manifest_schema.py`.
- [ ] Move orphan inventory validation, Terraform binding validation, listener ARN derivation, resource-binding validation, resolved-value validation, scenario inventory validation/isolation, pre-apply/bound loading, and retained-evidence validation/loading to `scenario_inventory.py`.
- [ ] Use the neutral `contracts._task_definition_family` for task-definition ARN family projection; `scenario_inventory.py` must not import the later `task_definition.py` or `orphan_sweep.py` modules.
- [ ] Permit `manifest_schema.py` to import only contracts, secure documents, and receipt contracts. It must not import `gate_ledger.py` or another mutation service; current control-manifest validation/read paths do not require ledger helpers.
- [ ] Preserve closed schemas, printable-ASCII Terraform version, listener/rule binding, account/region isolation, image/guardrail identity, deadlines, and monotonic retained evidence.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "control_manifest or scenario_inventory or retained_evidence or tf_binding" -q
```

Expected: PASS for validation and drift cases.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract AWS acceptance manifest schemas`.

**Definition of Done:** ledger and manifest schema reads are available before services need them, validation is side-effect free, and mutation remains separate.

---

### Task 9: Extract low-level manifest operations and task-definition policy

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/manifest.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/task_definition.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:5268-6710`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:3535-4630`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`

- [ ] Add failing identity assertions for `control_manifest_bind_retained_evidence`, `control_manifest_checkpoint_operator_evidence`, `control_manifest_init`, `control_manifest_bind_scenario`, `control_manifest_get`, and `validate_task_definition_policy_binding`.
- [ ] Move retained-evidence binding, operator checkpointing, manifest init, scenario binding, and manifest get to `manifest.py`.
- [ ] Leave `control_manifest_validate`, `control_manifest_update`, `control_manifest_load_cleanup`, and `scenario_load` in the facade until `control_service.py` is created in Task 13; they require receipt, approval, evidence, and ledger services that do not all exist yet.
- [ ] Move secret detection, Secrets Manager inventory binding, and task-definition policy validation to `task_definition.py`.
- [ ] Reuse `contracts._task_definition_family`; do not recreate or take ownership of the family parser in this higher layer.
- [ ] Preserve manifest idempotency, deadline behavior, cleanup-only resume, pinned candidate/rollback images, non-root one-shot entrypoint, secret selector closure, and rollback digest identity.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "control_manifest or scenario_load or task_definition_policy or evidence_export_receipt" -q
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_landscape_access_guard.py -q
```

Expected: PASS, including the already-current reviewed database-access map.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract AWS acceptance manifest services`.

**Definition of Done:** low-level manifest mutation and task policy have distinct owners; high-level orchestration remains in the facade without an upward private import.

---

### Task 10: Extract orphan lifecycle ownership

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/orphan_sweep.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:6713-7590`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:4631-5269`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`

- [ ] Add failing identity assertions for `OrphanSweepClients` and `orphan_sweep`.
- [ ] Move the client model/builder, AWS error classification, bounded/paged calls, inventory projections, task-definition ownership, transaction-search projection, and orphan sweep to `orphan_sweep.py`.
- [ ] Import `_validate_bounded_receipt_document` from lower-level `receipt_contracts.py` and `_task_definition_family` from `contracts.py`; never reach upward through the facade or `scenario_inventory.py` for either helper.
- [ ] Preserve exact-once client closure, repeated-token rejection, prefix-collision rejection, already-removed acceptance, retained telemetry identity, ECR tag deletion, task-definition deregistration, survivor counting, and endpoint-override rejection.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "orphan_sweep or transaction_search_projection" -q
```

Expected: PASS for all orphan lifecycle cases.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract AWS acceptance orphan lifecycle`.

**Definition of Done:** the 619-line hotspot has a focused owner and unchanged cleanup coverage.

---

### Task 11: Extract receipt persistence and approval verification

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/receipt_store.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/approvals.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:5704-5734,7769-8411,8547-8576`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:5270-5921`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`

- [ ] Add failing identity assertions for `validate_compatibility_record`, `receipt_store`, `approval_verify`, and `approval_require_current`.
- [ ] Move compatibility orchestration, stored receipt validation/persistence, `_verify_stored_receipts`, and manifest checkpointing to `receipt_store.py`; reuse pure validation from `receipt_contracts.py`.
- [ ] Move `_require_current_approval`, base64url decoding, configured signature verifier construction, approval verification, and current-approval enforcement to `approvals.py`.
- [ ] Preserve keyring protection, injected verifier support, plan/receipt/run/scenario/authority binding, expiry, signatures, canonical sanitized content, logical identity, and content-addressed storage.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "receipt_store or compatibility or approval" -q
```

Expected: PASS for persistence, binding, keyring, and fail-closed cases.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract AWS acceptance receipts and approvals`.

**Definition of Done:** schema validation, persistence, and authority are distinct and acyclic.

---

### Task 12: Extract the gate-ledger state machine

**Files:**
- Modify: `src/elspeth/web/_aws_ecs_acceptance/gate_ledger.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:8954-9240`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:6513-6705`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`

- [ ] Add failing identity assertions for every `gate_ledger_*` function and `_gate_ledger_records_hash`.
- [ ] Move the remaining init/get/bind/record/record-cleanup/finalize mutation operations to the `gate_ledger.py` created in Task 8.
- [ ] Keep writes through `secure_documents._write_protected_document`. Do not import evidence or cleanup services.
- [ ] Preserve candidate binding, check order, idempotent replay, conflict rejection, cleanup phase boundaries, finalized checksum, and secret-shaped field rejection.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k "gate_ledger" -q
```

Expected: PASS for all ledger state-machine cases.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract AWS acceptance gate ledger`.

**Definition of Done:** the ledger is independent and has no upward import.

---

### Task 13: Extract evidence, final cleanup, and high-level control service

**Files:**
- Create: `src/elspeth/web/_aws_ecs_acceptance/evidence.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/cleanup.py`
- Create: `src/elspeth/web/_aws_ecs_acceptance/control_service.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py:5663-5701,5737-6450,8412-8838`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py:5922-6512`
- Modify: `tests/unit/web/aws_ecs_acceptance/test_facade_contract.py`

- [ ] Add failing identity assertions for `sanitize_evidence`, `create_evidence_export_receipt`, `cleanup_evidence_finalize`, `control_manifest_validate`, `control_manifest_update`, `control_manifest_load_cleanup`, and `scenario_load`.
- [ ] Move `_validate_evidence_export_receipt`, `_reverify_bound_evidence_export_receipt`, safe value/log projection, sanitization, final cleanup receipt construction/verification, `create_evidence_export_receipt`, and evidence-export projection to `evidence.py`. Reuse `_verify_stored_receipts` from `receipt_store.py`.
- [ ] Validate projected log timestamps with `contracts._parse_utc_timestamp`; retain the current best-effort omission of invalid timestamps without importing `state.py` or catching `AcceptanceStateError`.
- [ ] Move two-phase cleanup orchestration and phase helpers to `cleanup.py`; it may import manifest, receipt, evidence, and ledger services.
- [ ] Move `control_manifest_validate`, `control_manifest_update`, `control_manifest_load_cleanup`, and `scenario_load` to `control_service.py`. Decompose `control_manifest_update` only through named pure projection/validation helpers; preserve its signature and one protected-document commit at the existing success boundary.
- [ ] Permit `control_service.py` to import low-level manifest/schema/inventory services, `receipt_store.py`, `approvals.py`, `evidence.py`, and `gate_ledger.py`. Forbid `cleanup.py` and `control_service.py` from importing each other.
- [ ] Preserve closed evidence kinds, removal of free-form content, initial/final export distinction, hashes, prepare/commit recovery, pending-surface refusal, failed-deadline terminal state, and cleanup-required clearing only after every surface verifies.
- [ ] Run:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py -k \
  "sanitize_evidence or evidence_export or cleanup_evidence_finalize" -q
```

Expected: PASS for sanitization, export, interruption recovery, and terminal states.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): extract AWS acceptance evidence cleanup`.

**Definition of Done:** cleanup and control orchestration are separate top-level services, neither imports the other, and all former facade orchestration has legal downward dependencies.

---

### Task 14: Split tests and reduce the permanent facade

**Files:**
- Create: `tests/unit/web/aws_ecs_acceptance/helpers.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_http_state_capture.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_s3.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_bedrock_guardrails.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_operator_telemetry.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_exec_receipts.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_control_manifest.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_orphan_sweep.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_receipts_approvals.py`
- Create: `tests/unit/web/aws_ecs_acceptance/test_evidence_cleanup_ledger.py`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py`

- [ ] Use the Task 0 base commit to recover the original name-to-domain inventory; the line ranges below are discovery aids only and must not be applied to the edited file:

```bash
base_sha=59af9eab0c96c5faaecfe54be97c8b12a2a74e17
test "$(git cat-file -t "$base_sha")" = commit
test "$(git merge-base HEAD "$base_sha")" = "$base_sha"
git show "$base_sha:tests/unit/web/test_aws_ecs_acceptance.py" \
  | rg -n '^(async )?def test_'
```

- [ ] Move each currently defined test by function name and parameterization into its owning domain, preserving every body and decorator. Use this original grouping:

```text
1-1103    -> test_http_state_capture.py; retain facade-only tests in the original file
1104-1456 -> test_s3.py
1457-2049 -> test_bedrock_guardrails.py
2050-2723 -> test_operator_telemetry.py
2724-3136 -> test_exec_receipts.py
3137-4630 -> test_control_manifest.py
4631-5269 -> test_orphan_sweep.py
5270-5921 -> test_receipts_approvals.py
5922-6705 -> test_evidence_cleanup_ledger.py
```

- [ ] Move only these cross-suite helpers to `helpers.py`: `_TelemetryAudit`, `_TelemetryEmitter`, `_empty_orphan_clients`, `_init_control_manifest`, `_guardrail_receipt_details`, `_s3_receipt_details`, and `_terraform_receipt`. Keep other helpers with their sole consumer.
- [ ] Import owner modules in domain tests and patch globals on those owners. Keep facade tests limited to CLI construction/dispatch, re-export identity, static error mapping, and executable behavior.
- [ ] Reduce `aws_ecs_acceptance.py` to imports/re-exports, `build_parser`, `_print_json`, `_print_error`, `_write_stdout_line`, `main`, and the `__main__` guard. Do not move parser or dispatch.
- [ ] Prove the split itself preserved every original test decorator and body. Before the Task 14 commit, `HEAD` is the Task 13 commit, so compare its monolith AST to the current split files while ignoring only source locations and new tests:

```bash
split_base_sha=$(git rev-parse HEAD)
env -u VIRTUAL_ENV uv run --frozen python - "$split_base_sha" <<'PY'
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SOURCE = "tests/unit/web/test_aws_ecs_acceptance.py"


class Tests(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.rows: dict[str, str] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        key = "::".join([*self.stack, node.name])
        if node.name.startswith("test_"):
            if key in self.rows:
                raise AssertionError(f"duplicate test identity: {key}")
            self.rows[key] = ast.dump(node, include_attributes=False)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function


def collect(text: str) -> dict[str, str]:
    visitor = Tests()
    visitor.visit(ast.parse(text))
    return visitor.rows


before_text = subprocess.check_output(
    ["git", "show", f"{sys.argv[1]}:{SOURCE}"], text=True
)
before = collect(before_text)
after: dict[str, str] = {}
paths = [Path(SOURCE), *sorted(Path("tests/unit/web/aws_ecs_acceptance").glob("test_*.py"))]
for path in paths:
    for key, value in collect(path.read_text(encoding="utf-8")).items():
        if key in after:
            raise AssertionError(f"duplicate post-split test identity: {key}")
        after[key] = value

missing = sorted(set(before) - set(after))
changed = sorted(key for key in before.keys() & after.keys() if before[key] != after[key])
assert not missing, f"missing tests: {missing}"
assert not changed, f"changed decorators/bodies: {changed}"
print(f"preserved {len(before)} original test definitions")
PY
```

Expected: the script reports the exact number of pre-split test definitions with no missing or changed entry. Owner-patch changes must occur in their earlier extraction task, not be hidden inside the mechanical test split.
- [ ] Verify no private module imports the facade:

```bash
rg -n "from elspeth\.web import aws_ecs_acceptance|from elspeth\.web\.aws_ecs_acceptance" \
  src/elspeth/web/_aws_ecs_acceptance
```

Expected: no matches.

- [ ] Verify collection and execution:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance \
  tests/unit/web/test_aws_ecs_runbook_contract.py \
  tests/unit/web/test_landscape_access_guard.py \
  tests/unit/architecture/test_sink_publication_callers.py --collect-only -q
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance \
  tests/unit/web/test_aws_ecs_runbook_contract.py \
  tests/unit/web/test_landscape_access_guard.py \
  tests/unit/architecture/test_sink_publication_callers.py -q
```

Expected: every selected node passes.

- [ ] Prove no original test or parameter case disappeared by normalizing the post-split collected IDs and comparing them to the committed baseline:

```bash
set -o pipefail
collected_now=$(mktemp)
missing_ids=$(mktemp)
trap 'rm -f "$collected_now" "$missing_ids"' EXIT
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance --collect-only -q \
  | rg '::test_' | sed -E 's#^[^:]+::##' | sort -u > "$collected_now"
comm -23 \
  tests/unit/web/aws_ecs_acceptance/fixtures/collected_ids.txt \
  "$collected_now" > "$missing_ids"
test ! -s "$missing_ids"
```

Expected: `missing_ids` is empty and `test ! -s` exits zero. New facade and architecture nodes may increase total collection but can never conceal a missing original ID.

- [ ] Have a separate specification-conformance reviewer inspect the Task 14 test diff with moved-code highlighting:

```bash
git diff --color-moved=dimmed-zebra "$split_base_sha" -- \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance
```

Expected: apart from imports and source-path placement, every decorator, parameterization, setup, call, and assertion is unchanged from the Task 13 tree, as independently enforced by the AST comparison. Removed or weakened assertions block completion even when node IDs and coverage remain green.

- [ ] Run the exact Tasks 2-14 extraction gate, then commit as `refactor(web): split AWS acceptance controller tests`.

**Definition of Done:** production and tests share domain boundaries, no test count is lost, and the facade is approximately 550 lines.

---

### Task 15: Verify packaging, archaeology, and source freeze

**Files:**
- Verify: `src/elspeth/web/aws_ecs_acceptance.py`
- Verify: `src/elspeth/web/_aws_ecs_acceptance/`
- Verify: `tests/unit/web/aws_ecs_acceptance/`

- [ ] Run focused verification:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance \
  tests/unit/web/test_aws_ecs_runbook_contract.py \
  tests/unit/web/test_landscape_access_guard.py \
  tests/unit/architecture/test_aws_ecs_acceptance_dependencies.py \
  tests/unit/architecture/test_sink_publication_callers.py -q
env -u VIRTUAL_ENV uv run --frozen ruff format --check \
  src/elspeth/web/aws_ecs_acceptance.py src/elspeth/web/_aws_ecs_acceptance \
  tests/unit/web/aws_ecs_acceptance tests/unit/web/test_aws_ecs_acceptance.py
env -u VIRTUAL_ENV uv run --frozen ruff check \
  src/elspeth/web/aws_ecs_acceptance.py src/elspeth/web/_aws_ecs_acceptance \
  tests/unit/web/aws_ecs_acceptance tests/unit/web/test_aws_ecs_acceptance.py
env -u VIRTUAL_ENV uv run --frozen mypy \
  src/elspeth/web/aws_ecs_acceptance.py src/elspeth/web/_aws_ecs_acceptance
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/web/test_aws_ecs_acceptance.py \
  tests/unit/web/aws_ecs_acceptance \
  tests/unit/web/test_aws_ecs_runbook_contract.py -q \
  --cov=elspeth.web.aws_ecs_acceptance \
  --cov=elspeth.web._aws_ecs_acceptance \
  --cov-branch --cov-report=
env -u VIRTUAL_ENV uv run --frozen coverage report \
  --include='src/elspeth/web/aws_ecs_acceptance.py,src/elspeth/web/_aws_ecs_acceptance/*' \
  --fail-under=73
```

Expected: every command passes without exclusions, and the same focused acceptance/runbook population used at Task 0 retains at least 73% branch-inclusive controller/package coverage.

- [ ] Prove the tracked runbook did not change and retains the characterized invocation count:

```bash
base_sha=59af9eab0c96c5faaecfe54be97c8b12a2a74e17
git diff --exit-code "$base_sha" -- docs/runbooks/aws-ecs-deployment.md
test "$(
  rg -o 'python -m elspeth\.web\.aws_ecs_acceptance' \
    docs/runbooks/aws-ecs-deployment.md | wc -l
)" -eq 55
```

Expected: no runbook diff and exactly 55 invocation sites. This is static parser/dispatcher compatibility evidence only.

- [ ] Run the five focused PostgreSQL container proofs required for the AWS startup/readiness surface:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest -m testcontainer \
  tests/testcontainer/web/test_doctor_aws_ecs_postgres.py \
  tests/testcontainer/web/test_schema_probe_postgres.py \
  tests/testcontainer/web/test_aws_ecs_validate_only_startup.py \
  tests/testcontainer/web/test_aws_ecs_readiness_postgres.py \
  tests/testcontainer/web/test_landscape_write_gate_postgres.py -q
```

Expected: all five files pass against real PostgreSQL. A missing Docker daemon is an unsatisfied verification prerequisite, not a skip or a clean result.

- [ ] Build and execute the installed wheel:

```bash
task_tmp=$(mktemp -d)
trap 'rm -rf "$task_tmp"' EXIT
env -u VIRTUAL_ENV uv build --out-dir "$task_tmp/dist"
uv venv "$task_tmp/venv" --python 3.13
wheel_path=$(find "$task_tmp/dist" -maxdepth 1 -type f -name '*.whl' -print -quit)
test -n "$wheel_path"
uv pip install --python "$task_tmp/venv/bin/python" \
  "${wheel_path}[webui,llm,aws,postgres]"
"$task_tmp/venv/bin/python" -c \
  'import boto3, psycopg; import elspeth.web.aws_ecs_acceptance'
"$task_tmp/venv/bin/python" -m elspeth.web.aws_ecs_acceptance --help
"$task_tmp/venv/bin/python" -m elspeth.web.aws_ecs_acceptance \
  scenario-namespace \
  --acceptance-run-id 00000000-0000-4000-8000-000000000001 \
  --scenario-id A
```

Expected: the wheel resolves from publishable runtime extras only, imports the acceptance module plus AWS/PostgreSQL dependencies, help succeeds, and the scenario probe prints `a-f8b447a7b5b51f38c800`. Do not install `[all]`: it contains the workspace-only, unpublished `elspeth-lints` package and cannot be resolved in an isolated wheel environment.

- [ ] Build and smoke a local container from the exact frozen source:

```bash
candidate_sha=$(git rev-parse HEAD)
local_image="elspeth:aws-ecs-acceptance-refactor-${candidate_sha:0:12}"
test -z "$(git status --porcelain)"
docker buildx build \
  --build-arg INSTALL_EXTRAS="webui llm aws postgres" \
  --label "org.opencontainers.image.revision=$candidate_sha" \
  --load \
  -t "$local_image" \
  .
test "$(
  docker image inspect "$local_image" \
    --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
)" = "$candidate_sha"
docker run --rm "$local_image" --version
docker run --rm --entrypoint python "$local_image" -c '
from pathlib import Path
import boto3
import psycopg
import elspeth.web
import elspeth.web.aws_ecs_acceptance

root = Path(elspeth.web.__file__).parent
assert (root / "frontend" / "dist" / "index.html").is_file()
print("container smoke passed")
'
```

Expected: the image is stamped with the exact frozen SHA, the CLI starts, the acceptance/AWS/PostgreSQL imports succeed, and the built frontend exists. This native-platform image is a local packaging smoke, not a release artifact and not evidence that ECS was deployed; release publication must rebuild for the CPU architecture discovered from the live task definition.

- [ ] Refresh Loomweave with MCP `analyze_start`/`analyze_status_get`, then run `module_circular_import_list`. CLI fallback: `loomweave analyze .` followed by equivalent queries.

Expected: fresh index, zero circular imports under `_aws_ecs_acceptance`, and `main` remains the sole CLI entrypoint. The AST architecture test remains authoritative for acyclic but forbidden upward imports.

- [ ] Run the primary local lint/type gates:

```bash
env -u VIRTUAL_ENV uv run --frozen ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
env -u VIRTUAL_ENV uv run --frozen ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
env -u VIRTUAL_ENV uv run --frozen mypy src/ elspeth-lints/src/
git diff --exit-code -- uv.lock
```

Expected: all four commands exit zero.

- [ ] Run every source-sensitive custom static gate from `.github/workflows/ci.yaml` except `trust_tier.tier_model`, which is deliberately deferred to Task 16:

```bash
env -u VIRTUAL_ENV uv run --frozen python scripts/cicd/check_slot_type_cross_language.py
env -u VIRTUAL_ENV uv run --frozen python scripts/cicd/generate_skill_inventory.py --check
env -u VIRTUAL_ENV uv run --frozen python scripts/check_contracts.py
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules plugin_contract.options_metadata --root .
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules plugin_contract.component_type,plugin_contract.plugin_hashes --root src/elspeth
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules immutability.freeze_guards,immutability.frozen_annotations --root src/elspeth
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules audit_evidence.nominal_base,audit_evidence.tier_1_decoration,audit_evidence.guard_symmetry,audit_evidence.gve_attribution --root src/elspeth
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules 'composer/*' --root src/elspeth
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules 'contract_invariants/*' --root src/elspeth
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules contract_invariants.session_engine_factory --root .
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules manifest.contract_manifest --root src/elspeth
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules manifest.symbol_inventory,manifest.test_to_source_mapping --root .
ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python -m elspeth_lints.core.cli check --rules meta.no-new-bespoke-cicd-enforcer --root .
env -u VIRTUAL_ENV uv run --frozen python scripts/cicd/enforce_adapter_budget.py
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python scripts/cicd/parity_harness.py --manifest config/cicd/lint_migration_status.yaml --root .
```

Expected: every command exits zero. Fix any source-sensitive failure before freezing or signing.

- [ ] Run Python 3.12 compatibility and the canonical Python 3.13 coverage lane, excluding only the intentionally stale fingerprint fixture:

```bash
env -u VIRTUAL_ENV uv run --isolated --python 3.12 --frozen --all-extras pytest tests/ \
  -v -m "not slow and not stress and not performance and not testcontainer and not fingerprint_baseline"
env -u VIRTUAL_ENV uv run --frozen pytest tests/ \
  --cov=src/elspeth \
  --cov-report=xml \
  --cov-report=term-missing \
  --cov-fail-under=85 \
  -v -m "not slow and not stress and not performance and not testcontainer and not fingerprint_baseline"
coverage_status=0
env -u VIRTUAL_ENV uv run --frozen coverage report \
  --include='src/elspeth/web/aws_ecs_acceptance.py,src/elspeth/web/_aws_ecs_acceptance/*' \
  --fail-under=73 || coverage_status=1
env -u VIRTUAL_ENV uv run --frozen coverage report \
  --include="src/elspeth/core/landscape/*" --fail-under=92 || coverage_status=1
env -u VIRTUAL_ENV uv run --frozen coverage report \
  --include="src/elspeth/core/canonical.py" --fail-under=99 || coverage_status=1
env -u VIRTUAL_ENV uv run --frozen coverage report \
  --include="src/elspeth/engine/orchestrator/*" --fail-under=90 || coverage_status=1
env -u VIRTUAL_ENV uv run --frozen coverage report \
  --include="src/elspeth/contracts/*" --fail-under=62 || coverage_status=1
test "$coverage_status" -eq 0
```

Expected: both lanes exit zero, aggregate, controller/package non-regression coverage, and all four subsystem coverage floors pass, and no non-baseline/non-testcontainer test is excluded. The 73% controller floor is the measured branch-coverage baseline from the untouched monolith's 314 acceptance/runbook tests. The fingerprint-baseline test is deliberately deferred until Task 16 because source movement makes it red before signing and regeneration; testcontainers were run explicitly above. Record commands, pass counts, and SHA in Filigree.
- [ ] Commit only a necessary verification-test correction separately. Then freeze `src/elspeth`: no source edit before Judge staging.

**Definition of Done:** source frozen, installed-wheel behavior preserved, local container packaging smoke green, focused PostgreSQL proofs and required local gates green, controller coverage does not regress, and archaeology confirms the dependency shape.

This refactor plan does not mutate or deploy the live ECS service. See the conditional release-acceptance section after Task 16 if the signed commit is selected for deployment.

---

### Task 16: Stage Judge work, sign, and regenerate the fingerprint baseline

**Files:**
- Modify via operator tooling only: `config/cicd/enforce_tier_model/*.yaml`
- Modify via canonical generator only: `tests/unit/elspeth_lints/fixtures/fingerprint_baseline.json`
- Create as gitignored handoff: one runtime bundle under `.elspeth/staged-reviews/`

Task 16 owns the incorporated acceptance criteria of P0 `elspeth-18fe6e759e`. Record the exact frozen source SHA and do not reuse any older staged bundle:

```bash
BASE_SHA=59af9eab0c96c5faaecfe54be97c8b12a2a74e17
SIGNING_START_SHA=$(git rev-parse HEAD)
FROZEN_SOURCE_SHA=$SIGNING_START_SHA
test "$(git merge-base "$FROZEN_SOURCE_SHA" "$BASE_SHA")" = "$BASE_SHA"
test -z "$(git status --porcelain)"
test -z "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}"
```

Record `BASE_SHA`, `FROZEN_SOURCE_SHA`, and the new bundle ID in the Filigree parent. No source edit is permitted until signing completes or the signing attempt is explicitly rolled back.

- [ ] **Step 1: Stage from the frozen tree without the HMAC key**

Use the key-free elspeth-judge MCP in order: `verify_signatures`, `stage_scan` with the executing identity as `staged_by`, then `stage_status` with the returned bundle ID. Stage from the frozen tree even if an older bundle exists; prior bundles are stale by construction after this refactor.

Expected: the complete lane counts. A count near or above 650 is accepted. Do not run `stage_preview` across the full queue unless the operator requests the extra non-authoritative judge cost.

- [ ] **Step 2: Dry-run in the operator-keyed shell**

Run the emitted command's arguments in an operator-controlled shell where `ELSPETH_JUDGE_METADATA_HMAC_KEY` has already been loaded securely, or use an operator-controlled environment file. Never substitute raw key bytes onto a command line or place them in shell history. Retain the emitted `--owner` identity and add `--judge-transport agent --judge-tools readonly --dry-run`.

Before both dry-run and fire:

```bash
: "${FROZEN_SOURCE_SHA:?set from the exact SHA recorded by the key-free staging step}"
test -n "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}"
test "$(git rev-parse HEAD)" = "$FROZEN_SOURCE_SHA"
test -z "$(git status --porcelain)"
```

The key-bearing operator shell must not invoke the key-free MCP tools.

Expected: verification succeeds, counts match, and no writes or judge calls occur.

- [ ] **Step 3: Fire once after count review**

Run the same emitted command for the same bundle without `--dry-run`, keeping `--judge-transport agent --judge-tools readonly` and adding `--yes` only after the operator accepts the displayed counts.

Expected: accepted entries are signed; blocked entries remain unsuppressed and make the run non-zero. On staleness or duplicate-key abort, do not force: return to Step 1 and stage fresh. A non-zero fire may already have written earlier accepted actions: keep the source frozen and either stage fresh to complete the same tree or restore the entire operator-written allowlist directory to `SIGNING_START_SHA` before changing source. Never hand-edit individual signature rows. If a judge blocks because the moved code is defective, restore the allowlist directory, return to the owning implementation task, add a regression, fix it, rerun Task 15, and stage again from the new frozen tree.

- [ ] **Step 4: Verify signed enforcement before baseline regeneration**

Steps 4-5 remain in the operator-controlled shell holding `ELSPETH_JUDGE_METADATA_HMAC_KEY`; do not return those steps to the key-free agent environment. The canonical generator's own bounded self-consistency subprocess is part of this reviewed operator step. Do not run the general repository suites while the key is present.

```bash
ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=required \
PYTHONPATH=elspeth-lints/src \
  .venv/bin/python -m elspeth_lints.core.cli check \
  --rules trust_tier.tier_model \
  --root src/elspeth
baseline_ref=59af9eab0c96c5faaecfe54be97c8b12a2a74e17
test "$(git merge-base HEAD "$baseline_ref")" = "$baseline_ref"
PYTHONPATH=elspeth-lints/src \
  .venv/bin/python -m elspeth_lints.core.cli check-judge-coverage \
  --baseline-ref "$baseline_ref" \
  --allowlist-root config/cicd/enforce_tier_model \
  --repo-root .
```

Expected: both commands exit zero with authoritative HMAC verification. Do not pass `--forbid-unverified-judge-metadata`: that keyless-fork option deliberately rejects every fresh judged record, even when the operator shell can verify its HMAC. A failure blocks baseline regeneration.

- [ ] **Step 5: Run the canonical gated baseline generator**

```bash
.venv/bin/python scripts/cicd/regen_fingerprint_baseline.py
```

Expected: signed enforcement verifies, the fixture is regenerated, and self-consistency passes. Do not use `--commit`; review signing and baseline changes together. If regeneration fails after writing the fixture, restore both `config/cicd/enforce_tier_model/` and the fingerprint fixture to `SIGNING_START_SHA`, diagnose the failure, and stage fresh before retrying.

- [ ] **Step 6: Run enforcement gates**

End the key-bearing operator shell after Step 5. Resume in a separate key-free agent shell on the same frozen checkout, copy `FROZEN_SOURCE_SHA` from the recorded Filigree value, and inspect `git status --short`: only operator-written `config/cicd/enforce_tier_model/*.yaml` files and the canonical fingerprint fixture may differ from `FROZEN_SOURCE_SHA`.

```bash
: "${FROZEN_SOURCE_SHA:?set from the exact SHA recorded by the key-free staging step}"
test -z "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}"
test "$(git rev-parse HEAD)" = "$FROZEN_SOURCE_SHA"
unexpected_path=0
while IFS= read -r changed_path; do
  case "$changed_path" in
    config/cicd/enforce_tier_model/*.yaml|tests/unit/elspeth_lints/fixtures/fingerprint_baseline.json) ;;
    *) printf 'unexpected post-signing path: %s\n' "$changed_path" >&2; unexpected_path=1 ;;
  esac
done < <(
  git diff --name-only "$FROZEN_SOURCE_SHA"
  git ls-files --others --exclude-standard
)
test "$unexpected_path" -eq 0
env -u VIRTUAL_ENV uv run --frozen pytest \
  tests/unit/elspeth_lints/test_allowlist_loader_unification.py \
  -k test_baseline_capture_is_self_consistent -q
PYTHONPATH=elspeth-lints/src env -u VIRTUAL_ENV uv run --frozen python \
  -m elspeth_lints.core.cli check-override-rate \
  --allowlist-root config/cicd \
  --max-rate 0.10
env -u VIRTUAL_ENV uv run --frozen pytest tests/ \
  -v -m "not slow and not stress and not performance and not testcontainer"
env -u VIRTUAL_ENV uv run --frozen pytest tests/testcontainer/ \
  -v -m testcontainer
git diff --exit-code -- uv.lock
```

Expected: every command passes in the key-free environment, including the now-current fingerprint baseline, complete local PostgreSQL testcontainer suite, and unchanged lock file. No general pytest, testcontainer, lint, or project command receives the HMAC key. A red Step 6 blocks commit and P0 closure: keep source frozen while diagnosing; if correction requires any source edit, restore both the operator-written allowlist directory and fingerprint fixture to `SIGNING_START_SHA`, rerun Task 15, and execute Task 16 again from a fresh bundle. `check-judge-quality` remains a trusted CI verification gate and never signs.

- [ ] **Step 7: Commit only generated enforcement changes**

```bash
git diff --check
git add config/cicd/enforce_tier_model \
  tests/unit/elspeth_lints/fixtures/fingerprint_baseline.json
git diff --cached --check
git commit -m "chore(cicd): sign AWS acceptance refactor fingerprints"
```

Expected: staged-review bundles, source, lock files, runbook, and unrelated files are absent from the commit.

- [ ] **Step 8: Require repository CI on the exact signed commit**

```bash
SIGNING_COMMIT=$(git rev-parse HEAD)
```

Push or open the implementation PR through the developer's normal workflow and require the branch-protection check named `CI Success` to report success for `SIGNING_COMMIT`. This aggregate includes static analysis, Python test lanes, the complete PostgreSQL testcontainer suite, supply-chain audit, frontend E2E, and frontend unit/type checks. Require the configured Judge/override verification checks as applicable; they verify but never sign.

Expected: all required checks are green for the exact `SIGNING_COMMIT`. If they have not run, report implementation complete but merge acceptance pending; focused local tests must not be described as full repository gates.

- [ ] **Step 9: Close the incorporated operator blocker**

The operator records the fresh bundle ID, `FROZEN_SOURCE_SHA`, signing counts, signed-gate result, baseline self-consistency result, Judge coverage result, required CI results, and `SIGNING_COMMIT` on `elspeth-18fe6e759e`. Advance the bug through its valid workflow and close it with the feature-branch commit anchor only after every Task 16 gate passes. Do not force-close it from `confirmed`; agents must not reclaim the operator-owned issue.

**Definition of Done:** authoritative signatures bind the final source layout, the raw baseline matches it, local and required CI enforcement is green for the exact signed commit, the incorporated P0 is closed with that anchor, the agent never handled the HMAC key, and signing occurred after source freeze.

---

## Conditional existing-service ECS release acceptance

Tasks 0-16 establish source, package, local container-packaging, signed-enforcement, and CI acceptance. They do not publish an image, register a task definition, or mutate an ECS service.

This section becomes required only when the developer selects the final signed commit for deployment to the existing AWS acceptance service. Execute it as a separate Filigree release-acceptance issue using the `operating-aws-ecs-container` skill; do not use the exhaustive disposable-environment runbook for an ordinary redeploy.

Required evidence:

- exact clean candidate Git SHA plus explicit AWS profile and region;
- live discovery of account, cluster, service, previous task definition, web container, target group, network configuration, CPU architecture, log group, and ECR repository;
- an image rebuilt for the discovered architecture with `INSTALL_EXTRAS="webui llm aws postgres"`;
- immutable ECR `repository@sha256:...` identity and matching image revision label;
- successful one-shot read-only `doctor aws-ecs --json` before service mutation;
- one completed primary deployment using the exact candidate task-definition ARN, desired/running/pending `1/1/0`, one running task, matching running image digest, and one healthy target;
- exact-200 `/api/health` with `status: ok`, exact-200 `/api/ready` with `ready: true`, expected `/api/system/status` identity/policy facts, and no new unhandled runtime failure in a bounded log window;
- applicable in-task `verify-s3`, `verify-bedrock`, `verify-bedrock-guardrails`, and `verify-operator-telemetry` checks using the deployed task role—an unavailable or unconfigured capability must be recorded explicitly and must not be represented as passing; and
- an authenticated workflow appropriate to the release change.

Capture the previous task definition before mutation. Roll back only if schema compatibility is proved and all previous image digests and configuration remain available; otherwise fix forward. Host-side mocks and the unchanged exhaustive runbook do not replace these candidate checks.

---

## Final acceptance checklist

- [ ] Original import and executable module paths work.
- [ ] All 24 commands and nested surfaces are unchanged.
- [ ] Installed-wheel scenario probe returns the exact original value.
- [ ] The runbook is byte-identical to `BASE_SHA`, its 55 invocation sites remain present, and the refactor introduces no parser/dispatcher regression; no live-runbook certification is implied.
- [ ] Exception classes are defined once and re-exported by identity.
- [ ] Protected modes, bounded reads, atomic writes, static errors, and redaction are unchanged.
- [ ] AWS lanes preserve client ownership and cleanup.
- [ ] Manifest, receipt, approval, evidence, cleanup, and ledger transitions remain fail closed and resume safe.
- [ ] No private module imports the facade and no circular import exists.
- [ ] Acceptance/runbook node counts do not decrease.
- [ ] Ruff, mypy, focused tests, focused and complete PostgreSQL testcontainers, wheel smoke, local container packaging, and required `CI Success` pass for the exact signed commit.
- [ ] Judge staging occurs after source freeze; restaging occurs only for the documented staleness, duplicate-key, or genuine-defect loop. Operator judging uses agent transport with read-only tools.
- [ ] Signed enforcement passes before baseline regeneration.
- [ ] Live ECS release acceptance is either explicitly out of scope or linked through a separate issue with immutable candidate evidence; it is not inferred from local tests.
- [ ] `uv.lock`, runbook, unrelated source, and user-owned dirty files remain untouched.

## Execution handoff

Preferred execution is subagent-driven in the dedicated worktree, one task and commit at a time, with specification-conformance review followed by code-quality review after each task. Use `superpowers:subagent-driven-development`. For inline execution, use `superpowers:executing-plans` and stop after each numbered task for verification.

If a host-local `elspeth-judge` registration points at the disposable worktree, remove it or repoint it to the merged checkout before deleting the worktree. Do not leave a registered MCP command targeting a removed `.venv`.
