# AWS S3 Web-Authorship Endpoint Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Follow the Filigree split below; document number is not execution order.

**Goal:** Reject every non-null `endpoint_url` on web-authored `aws_s3` source and sink configuration while preserving the field for CLI/batch pipelines.

## Review and execution status

**Plan verdict:** APPROVED after the repairs recorded in the sibling review artifact. Implementation is currently blocked by the explicit gate-baseline prerequisite below; this is a live verification dependency, not document-number sequencing.

| Slice | Filigree | Executable now? | Scope | Blocking edges |
|---|---|---:|---|---|
| Gate baseline | `elspeth-8166b310e7` | **Yes** | Signed-tier repair + pre-existing `state.py` trust-boundary evidence repair + authoritative non-inert Wardline analysis | none |
| 08A | `elspeth-c0103e6c88` | **No** | Tasks 1-3: predicate, load-bearing validation gate, every composer mutation backstop | gate baseline `elspeth-8166b310e7` |
| 08B | `elspeth-a342f333a4` | **No** | Task 4 only: guided-source prompt parity | Plan 06 `elspeth-7fe6aa531f` and 08A |

Do not claim 08A until the gate-baseline issue is closed, and start/close 08A and 08B separately. The dependency shape is `(gate baseline -> 08A) || 02`, then `08A + 02 -> 06 -> 07`; 08B follows `06 + 08A`. Review approval does not bypass those live edges.

## Architecture

`web_aws_s3_endpoint_url_policy_error()` is a pure Tier-3 policy predicate in `src/elspeth/web/provider_config_policy.py`. It returns the same static error for `plugin == "aws_s3"` whenever `options.get("endpoint_url") is not None`; omission and explicit `null` are equivalent and allowed. The real Plan-06/07 config models must expose only that canonical top-level spelling, declare no alias or nested boto client-config escape hatch, and forbid extras.

The load-bearing enforcement point is `validate_pipeline()` in `src/elspeth/web/execution/validation.py`, after `llm_base_url_policy` and before settings loading/plugin instantiation. `ExecutionServiceImpl._execute_locked()` calls it unconditionally before `create_run`, so even copied legacy/reverted/forked state cannot execute with an endpoint override. Compose/import/seed persistence computes the same preflight when structural authoring is valid and no result was supplied. Invalid YAML/seed state may be persisted for repair with `is_valid=False`; it is never execution-ready. Revert/fork copy prior validity metadata rather than revalidating at copy time, so the unconditional execution gate is the final backstop.

The web-authoring inventory is deliberately broader than direct `SourceSpec`/`OutputSpec` constructors:

| Surface | Live construction/mutation | Plan-08 disposition |
|---|---|---|
| `set_source`, `patch_source_options` | `composer/tools/sources.py` | immediate Task-3 rejection + central gate |
| `set_source_from_blob` | raw caller options and resolved `SourceSpec` in `sources.py` | immediate raw and post-resolution Task-3 rejection + central gate |
| `set_output`, `patch_output_options` | `composer/tools/outputs.py` | immediate Task-3 rejection + central gate |
| `set_pipeline` named sources, legacy source, outputs | three constructors in `composer/tools/sessions.py` | immediate rejection at all three branches + central gate |
| `apply_pipeline_recipe` | delegates to `_execute_set_pipeline` | delegation regression + inherited immediate gate |
| `wire_secret_ref` source/output | option-only mutation in `composer/tools/secrets.py` | immediate rejection; `endpoint_url` also remains absent from plugin credential fields |
| `wire_blob_inline_ref` source/output | option-only mutation in `composer/tools/blobs.py` | immediate rejection of an `endpoint_url` field path + central gate |
| YAML import source/sink and imported-blob rewrite | `yaml_importer.py` and `sessions/routes/composer/state.py` | source and sink route regressions; persisted invalid, never executable |
| E2E state seed | `seed_state_for_e2e` | source and sink route regressions; persisted invalid, never executable |
| guided source/output flows | delegate to mutation tools | inherited gate; Task 4 adds source guidance after registration |
| revert/fork/legacy state | copies existing state and validity metadata | unconditional execution-service regression is the backstop |

CLI/batch separation is structural: no core/CLI module imports the web policy. Plans 06 and 07 own the positive acceptance tests that load and instantiate CLI source/sink YAML containing `endpoint_url`; Plan 08 owns an AST import-boundary guard and the negative web half.

**Tech stack:** Python, pydantic, pytest; no new dependency or lockfile edit.

---

### Task 0: Restore and prove the verification baseline

**Owner:** Filigree `elspeth-8166b310e7`; verification-baseline work, not AWS S3 feature implementation. The agent owns the evidence and Wardline bridge changes below; the operator alone owns signed metadata repair.

**Baseline-owned files:**

- Modify: `src/elspeth/web/sessions/routes/composer/state.py`
- Modify: `tests/unit/web/sessions/routes/composer/test_state_boundaries.py`
- Create: `config/wardline/elspeth_pack.py`
- Modify: `weft.toml`
- Create: `config/wardline/verify_elspeth_pack_contract.py`
- Modify after operator signing: individually diagnosed `config/cicd/enforce_tier_model/*.yaml` files, bounded to the operator-reviewed final repair set

The live review found three real baseline failures: signed binding drift; `_reject_disallowed_source_paths` fails the direct-test/fingerprint contract; and the current Wardline analysis is non-authoritative. An earlier review recorded `WLN-ENGINE-PYDANTIC-DISCOVERY-LIMIT` when Pydantic discovery exceeded its work budget (`367820/367776`); current live analysis no longer reports that capacity defect, but it must prove both adequate engine capacity and non-inert coverage through Elspeth's trust-boundary pack. None of these failures may be hidden by an allowlist, baseline, waiver, narrowed scan, or inert successful exit.

- [ ] Run the initial read-only signed-entry diagnosis before changing any baseline-owned file. Record the result for comparison, but do not ask the operator to repair rows yet:

  ```bash
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing \
    uv run elspeth-lints diagnose-judge-signatures \
      --root src/elspeth \
      --allowlist-dir config/cicd/enforce_tier_model \
      --format text
  ```

- [ ] Repair the pre-existing trust-boundary evidence on `_reject_disallowed_source_paths`. Add `tests/unit/web/sessions/routes/composer/test_state_boundaries.py::test_reject_disallowed_source_paths_raises_400_on_escaped_path`; its own body must call the helper directly under `pytest.raises(HTTPException)` with an escaped path and assert the raised exception's `status_code == 400`. Retarget the decorator's `test_ref` to that exact node.
- [ ] Finalize the test body with the decorator's fingerprint omitted, run the trust-boundary rule once, and copy only the emitted canonical fingerprint into `test_fingerprint`. Then rerun both the focused test and the complete trust-boundary gate to green; never guess, precompute, or copy an unrelated fingerprint:

  ```bash
  uv run pytest tests/unit/web/sessions/routes/composer/test_state_boundaries.py::test_reject_disallowed_source_paths_raises_400_on_escaped_path -q
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
  ```

- [ ] Add Elspeth's Wardline pack bridge in `config/wardline/elspeth_pack.py` and declare `packs = ["config.wardline.elspeth_pack"]` under `[wardline]` in `weft.toml`. The runtime loader extends the default grammar itself, so the pack must not call `default_grammar()` or include built-ins. Use the required imports and exact contract shape:

  ```python
  from collections.abc import Mapping

  from wardline.core.taints import TaintState
  from wardline.scanner.grammar import BoundaryType, TrustGrammar
  from wardline.scanner.taint.provider import FunctionTaint


  def _seed_elspeth_trust_boundary(_levels: Mapping[str, TaintState]) -> FunctionTaint:
      return FunctionTaint(TaintState.EXTERNAL_RAW, TaintState.EXTERNAL_RAW)


  ELSPETH_TRUST_BOUNDARY = BoundaryType(
      canonical_name="trust_boundary",
      module_prefix="elspeth.contracts.trust_boundary",
      group=1,
      level_args=(),
      seed=_seed_elspeth_trust_boundary,
      builtin=False,
  )

  grammar = TrustGrammar(boundary_types=(ELSPETH_TRUST_BOUNDARY,), rules=())
  ```

  The conservative seed is deliberate. Elspeth's decorator covers both raising validators and `non_raising=True` predicates, so mapping every decorated result to `ASSURED` would create known false positives and over-trust return values. `elspeth-lints` owns decorator-contract honesty; Wardline conservatively anchors the external-origin flow.
- [ ] Add `config/wardline/verify_elspeth_pack_contract.py` as a focused external-tool contract verifier, not a pytest module. Ordinary project CI intentionally does not install Wardline, so this verifier must stay outside `tests/` and the default pytest collection. The verifier imports the pack with Wardline's own pinned interpreter using the canonical `importlib.import_module("config.wardline.elspeth_pack")` path and asserts that lowercase `grammar` is a `TrustGrammar` with exactly one boundary and no rules; assert exact canonical name, module prefix, `group == 1`, `level_args == ()`, `builtin is False`, and `FunctionTaint` seed states `body_taint is TaintState.EXTERNAL_RAW` and `return_taint is TaintState.EXTERNAL_RAW`. Resolve the installed `wardline` executable, derive its interpreter from the executable's shebang, reject a missing or non-executable interpreter, and never hardcode a user-specific tool path. Run the verifier directly from the repository root with that root on `PYTHONPATH`, and require exit 0:

  ```bash
  WARDLINE_BIN="$(command -v wardline)"
  WARDLINE_PYTHON="$(sed -n '1s/^#!//p' "$WARDLINE_BIN")"
  test -n "$WARDLINE_PYTHON" && test -x "$WARDLINE_PYTHON"
  PYTHONPATH="$PWD" "$WARDLINE_PYTHON" config/wardline/verify_elspeth_pack_contract.py
  ```

- [ ] Require the exact installed Wardline 1.3.1 loader/schema contract, then run the authoritative full-root gate and immediately validate the same agent-summary output:

  ```bash
  test "$(wardline --version)" = "wardline, version 1.3.1"
  PYTHONPATH="$PWD" wardline scan . \
    --trust-pack config.wardline.elspeth_pack \
    --allow-custom-packs \
    --fail-on ERROR \
    --fail-on-unanalyzed \
    --format agent-summary \
    --output /tmp/aws-wardline-agent-summary.json
  jq -e '
    .schema == "wardline-agent-summary-1" and
    .gate.verdict == "PASSED" and
    .gate.tripped == false and
    .gate.exit_class == 0 and
    .summary.active_defects == 0 and
    .summary.unanalyzed == 0 and
    .resolution.inert == false and
    .resolution.recognized_boundaries > 0 and
    all(.engine_facts[]?;
      .rule_id != "WLN-ENGINE-FUNCTION-SKIPPED" and
      .rule_id != "WLN-ENGINE-PYDANTIC-DISCOVERY-LIMIT")
  ' /tmp/aws-wardline-agent-summary.json >/dev/null
  ```

  The exact version check stops loader/schema drift for re-review. Exit 0 is necessary but not sufficient; the jq assertion is the Wardline 1.3.1 complete-analysis contract beyond the CLI's unanalyzed sub-gate. `INERT`, zero recognized boundaries, unanalyzed source units/files, either named member of Wardline 1.3.1's formal `INCOMPLETE_ANALYSIS_RULE_IDS` set, missing metrics, an untrusted or missing pack, or an active ERROR+ defect blocks the baseline. Conservative resolution facts outside that formal set, including flow-insensitive fallback, remain visible and are recorded by count but do not imply a false clean result. Exit 1 requires explanation and repair; exit 2 also blocks. A subdirectory or `--affected` scan is advisory only. Do not add a baseline, waiver, allowlist, or scope narrowing. The `/tmp` summary is diagnostic only; durable evidence records only sanitized pass/fail counts, never raw findings or paths.
- [ ] Only after all agent-owned state/pack/weft/test work and the focused trust, pack-contract, version, and authoritative Wardline checks above are complete, run the final read-only signed-entry diagnosis into a mode-`0600` JSON receipt and derive the exact unique YAML repair paths from that receipt. Exit 0 means no repair paths; exit 1 is permitted only when the validated receipt says `requires_action == true`. Reject every unsafe/non-basename `source_file` before prefixing it with the allowlist directory:

  ```bash
  umask 077
  SIGNED_RECEIPT=/tmp/aws-final-signed-diagnosis.json
  SIGNED_PATHS=/tmp/aws-final-signed-yaml-paths.txt
  set +e
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing \
    uv run elspeth-lints diagnose-judge-signatures \
      --root src/elspeth \
      --allowlist-dir config/cicd/enforce_tier_model \
      --format json >"$SIGNED_RECEIPT"
  SIGNED_STATUS=$?
  set -e
  test "$SIGNED_STATUS" -eq 0 -o "$SIGNED_STATUS" -eq 1
  jq -e '
    (.entries | type == "array") and
    (.requires_action | type == "boolean") and
    all(.entries[]; (.requires_action | type == "boolean")) and
    all(.entries[] | select(.requires_action); .source_file | test("^[A-Za-z0-9_.-]+[.]yaml$")) and
    (($status == 0 and .requires_action == false) or
     ($status == 1 and .requires_action == true))
  ' --argjson status "$SIGNED_STATUS" "$SIGNED_RECEIPT" >/dev/null
  jq -r '.entries[] | select(.requires_action) | .source_file' "$SIGNED_RECEIPT" \
    | LC_ALL=C sort -u \
    | sed 's#^#config/cicd/enforce_tier_model/#' >"$SIGNED_PATHS"
  chmod 600 "$SIGNED_RECEIPT" "$SIGNED_PATHS"
  ```

  This receipt is the operator's repair set. At this point the signing freeze begins: through operator repair, post-repair diagnosis, baseline gates, staging, baseline commit, and baseline close, no agent or sibling lane may edit code or configuration. The only permitted config mutation is the operator's exact reviewed signed-YAML repair; all other permitted actions are read-only verification, the one coordinator-owned Filigree structured-field update that records the receipt-listed YAML paths, staging/commit, and closure. The tracker update does not alter the signed source tree and must finish before operator repair. Preserve queued sibling work, commits, and checkpoints intact but paused.
- [ ] If the final diagnosis contains anything except `OK`/`OK_SHAPE_ONLY`, stop for the operator. The agent must not run `justify`, `sign-judge-signatures`, `rotate`, or any other signing/write path. The operator first reviews a dry run, then repairs only the final diagnosed existing rows using operator-held credentials, for example:

  ```bash
  uv run elspeth-lints sign-judge-signatures \
    --root src/elspeth \
    --allowlist-dir config/cicd/enforce_tier_model \
    --env-file <operator-env-file> \
    --owner <operator-id> \
    --dry-run
  # Operator removes --dry-run only after reviewing the exact repair plan.
  ```

  Follow the diagnostic classification: remove genuinely stale `NO_MATCHING_FINDING` rows; use the emitted re-justify path for scope/AST binding drift; never hand-edit judge metadata. Do not commit `*.bak-*` files.
- [ ] After the operator pass, make no further code/config edits. Re-run diagnosis and require only `OK`/`OK_SHAPE_ONLY`; run the shape-only tier-model gate, focused state test, complete trust-boundary gate, focused pack-contract verifier, exact Wardline version preflight, authoritative scan, and exact jq assertion above on the unchanged tree. Any failure exits the freeze for rework and requires a new final diagnosis before signing resumes.
- [ ] Stage and commit one coherent baseline change. Stage the five fixed source/test/pack/config paths exactly, then stage each operator-reviewed signed YAML output individually from the final diagnosis/receipt; never stage the allowlist directory or use a glob:

  ```bash
  git add src/elspeth/web/sessions/routes/composer/state.py
  git add tests/unit/web/sessions/routes/composer/test_state_boundaries.py
  git add config/wardline/elspeth_pack.py
  git add weft.toml
  git add config/wardline/verify_elspeth_pack_contract.py
  mapfile -t SIGNED_YAML_PATHS <"$SIGNED_PATHS"
  if test "${#SIGNED_YAML_PATHS[@]}" -gt 0; then
    git add -- "${SIGNED_YAML_PATHS[@]}"
  fi
  {
    printf '%s\n' \
      src/elspeth/web/sessions/routes/composer/state.py \
      tests/unit/web/sessions/routes/composer/test_state_boundaries.py \
      config/wardline/elspeth_pack.py \
      weft.toml \
      config/wardline/verify_elspeth_pack_contract.py
    cat "$SIGNED_PATHS"
  } | LC_ALL=C sort -u >/tmp/aws-baseline-expected-staged-paths.txt
  git diff --cached --name-only | LC_ALL=C sort -u >/tmp/aws-baseline-actual-staged-paths.txt
  cmp -s /tmp/aws-baseline-expected-staged-paths.txt /tmp/aws-baseline-actual-staged-paths.txt
  if git diff --cached --name-only | rg '(^|/)[^/]*\.bak-'; then exit 1; fi
  git commit -m "chore(security): restore authoritative verification baseline"
  ```

  Before committing, require the staged-name review to contain only those five fixed paths plus the exact individually approved YAML repair set; reject every `*.bak-*` or unrelated path.
- [ ] Close `elspeth-8166b310e7` only when all baseline gates pass. Then start 08A atomically. No AWS S3 feature implementation begins against a known-unverifiable baseline.

---

### Task 1: Add the pure endpoint policy predicate

**08A files:**

- Modify: `src/elspeth/web/provider_config_policy.py`
- Create: `tests/unit/web/test_provider_config_policy.py`

- [ ] Write `TestWebAwsS3EndpointUrlPolicy` first. Cover a different plugin, `None` plugin, omitted key, explicit `None`, loopback HTTP, arbitrary HTTPS, and a non-string non-null value. Every non-null value for `aws_s3` must return `AWS_S3_ENDPOINT_URL_POLICY_ERROR`; the static error must not contain the supplied endpoint.
- [ ] Add an AST-based `test_core_config_has_no_web_runtime_import`. Parse `src/elspeth/core/config.py`, inspect both `ast.Import` and `ast.ImportFrom`, and fail on `elspeth.web` or descendants. Do not use `inspect.getsource()` substring matching: comments can false-fail it and alternate import forms can evade it.
- [ ] Run the new tests and observe the missing-symbol failure:

  ```bash
  uv run pytest tests/unit/web/test_provider_config_policy.py -q
  ```

- [ ] Update the module docstring from transform-only wording to general web-authored provider configuration. Add `AWS_S3_ENDPOINT_URL_POLICY_ERROR` as static prose that names the field and risk but never interpolates an endpoint, credential, bucket, key, or provider exception.
- [ ] Add the predicate with the same boundary posture as its peers:

  ```python
  @trust_boundary(
      tier=3,
      source="web-authored aws_s3 source/sink options (untrusted author-supplied mapping)",
      source_param="options",
      suppresses=("R1", "R5"),
      invariant=(
          "non-aws_s3 plugins are ignored; omitted or explicit-null endpoint_url is allowed; "
          "every non-null aws_s3 endpoint_url returns the static policy error; never raises"
      ),
      non_raising=True,
  )
  def web_aws_s3_endpoint_url_policy_error(
      plugin: str | None,
      options: Mapping[str, Any],
  ) -> str | None:
      if plugin != "aws_s3":
          return None
      if options.get("endpoint_url") is None:
          return None
      return AWS_S3_ENDPOINT_URL_POLICY_ERROR
  ```

- [ ] Re-run the focused test, then stage only these files and commit:

  ```bash
  uv run pytest tests/unit/web/test_provider_config_policy.py -q
  git add src/elspeth/web/provider_config_policy.py tests/unit/web/test_provider_config_policy.py
  git commit -m "feat(web): add aws_s3 endpoint policy predicate"
  ```

---

### Task 2: Add the load-bearing validation and persistence gate

**08A files:**

- Modify: `src/elspeth/web/execution/schemas.py`
- Modify: `src/elspeth/web/execution/validation.py`
- Modify: `tests/unit/web/execution/test_validation.py`
- Modify: `tests/unit/web/execution/test_service.py`
- Modify: `tests/unit/web/sessions/test_routes.py`

- [ ] Extend `ValidationCheckName`, add `CHECK_AWS_S3_ENDPOINT_URL_POLICY`, and insert it in `VALIDATION_BLOCKING_CHECK_NAMES` immediately after `CHECK_LLM_BASE_URL_POLICY`. Import/alias it in `validation.py`; `_ALL_CHECKS` derives from that tuple.
- [ ] Extend `_make_source()` with `plugin: str = "csv"` and `_make_state()` with `source_plugin: str = "csv"`. Keep every required `SourceSpec`, `OutputSpec`, `CompositionState`, `WebSettings`, and `YamlGenerator` field in the fixtures.
- [ ] Add direct red tests for an `aws_s3` source and sink carrying a non-null endpoint. Assert exact check name `aws_s3_endpoint_url_policy`, exact error code `aws_s3_endpoint_url_not_allowed`, source/sink component attribution, blocked readiness, and no call to settings loading/plugin instantiation. The policy is physically after YAML generation, so do not falsely assert that `YamlGenerator.generate_yaml()` was skipped.
- [ ] Add omission and explicit-null source/sink acceptance cases. Add a redaction case with an endpoint sentinel and assert it is absent from serialized checks, errors, readiness, logs, `str`, and `repr`.
- [ ] Insert source/output policy loops after the existing `llm_base_url_policy` pass record and before `# Step 3: Settings loading`. Return the first deterministic violation using the existing fail-fast policy pattern, then append one passing check when neither loop finds a violation. Never include the raw endpoint in detail, error, suggestion, or readiness.
- [ ] Update the live success assertion from 16 to 17 checks, add the new check to the skipped-after-earlier-failure regression, and keep the order-sync/physical-emission-order tests green.
- [ ] Add an execution-service regression using a real endpoint-invalid `CompositionState` (not a mocked invalid `ValidationResult`). Assert `PipelineValidationError`, no `create_run`, no plugin instantiation, and no provider/network call. This pins the final backstop used by revert/fork/legacy state.
- [ ] Parameterize YAML-import route coverage for an invalid `aws_s3` source and invalid `aws_s3` sink. Use a structurally valid source-to-sink YAML document so authoring validation reaches runtime preflight. POST it, then assert the response and persisted record have `is_valid=False` and `validation_errors` contains the static `aws_s3`/`endpoint_url` policy message. `CompositionStateResponse` does not expose validation check names; exact check-name assertions belong in the direct validation tests.
- [ ] Add the same source/sink parameterization for `POST /{session_id}/state/e2e-seed` with `e2e_state_seed_enabled=True`. Assert invalid persistence and unchanged static-message/redaction behavior.
- [ ] Run the red tests, implement the gate, and run the full owning files:

  ```bash
  uv run pytest tests/unit/web/execution/test_validation.py -k 'AwsS3EndpointUrlPolicy or validation_emits_checks_in_declared_order or valid_pipeline_returns_all_checks_passed' -q
  uv run pytest tests/unit/web/execution/test_service.py -k aws_s3_endpoint_url -q
  uv run pytest tests/unit/web/sessions/test_routes.py -k aws_s3_endpoint_url -q
  uv run pytest tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_service.py tests/unit/web/sessions/test_routes.py -q
  ```

- [ ] Stage only Task-2 files and commit:

  ```bash
  git add src/elspeth/web/execution/schemas.py src/elspeth/web/execution/validation.py tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_service.py tests/unit/web/sessions/test_routes.py
  git commit -m "feat(web): enforce aws_s3 endpoint policy in validation"
  ```

---

### Task 3: Reject the endpoint at every composer mutation surface

**08A files:**

- Modify: `src/elspeth/web/composer/tools/sources.py`
- Modify: `src/elspeth/web/composer/tools/outputs.py`
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/composer/tools/secrets.py`
- Modify: `src/elspeth/web/composer/tools/blobs.py`
- Modify after operator signing: `config/cicd/enforce_tier_model/web.yaml` (only the diagnosed existing signed rows; never hand-edit)
- Modify: `tests/unit/web/composer/test_tools.py`
- Modify: `tests/unit/web/composer/test_blob_inline_tools.py`

The central validation check remains authoritative. These tool checks are defense-in-depth that leave state unchanged and give the author immediate repair feedback.

- [ ] Add rejection tests for `set_source`, `patch_source_options`, `set_output`, and `patch_output_options`. Assert `success=False`, exact static endpoint-policy prose, unchanged state/version, and no raw endpoint sentinel.
- [ ] Add separate `set_pipeline` regressions for the named `sources` mapping, legacy single `source`, and output loop. Add one `apply_pipeline_recipe` regression proving delegation cannot bypass `_execute_set_pipeline`.
- [ ] Cover `set_source_from_blob` twice: reject explicit `plugin="aws_s3"` plus caller `endpoint_url` before blob lookup, and reject a synthetic resolved `aws_s3` result carrying the field before `SourceSpec` construction. Check both raw validated options and `resolved.plugin/resolved.options`; this protects future inference/merge changes.
- [ ] Add source and output tests for `wire_secret_ref` targeting `endpoint_url`. Reject before mutation even if a marker ref exists. Plans 06/07 separately own explicit `allowed_secret_ref_fields("source"/"sink", "aws_s3")` assertions proving `endpoint_url` never becomes a credential field after registration.
- [ ] Add source and output tests for `wire_blob_inline_ref` whose canonical field path ends in `.options.endpoint_url`. After applying the candidate marker to a temporary state, run the predicate on the affected component and return `_failure_result` with the original state when it fails. Do not permit an inline blob marker to become an endpoint value.
- [ ] Import and call `web_aws_s3_endpoint_url_policy_error` at the live bound-variable seams: `plugin/options`, current plugin plus merged patch, all three `_execute_set_pipeline` branches, raw/resolved blob source, both secret-ref component arms, and both inline-blob component arms. The policy check must precede plugin option prevalidation wherever possible so 08A remains independently testable before registration.
- [ ] Run the focused and owning suites:

  ```bash
  uv run pytest tests/unit/web/composer/test_tools.py tests/unit/web/composer/test_blob_inline_tools.py -k 'aws_s3 and endpoint_url' -q
  uv run pytest tests/unit/web/composer/test_tools.py tests/unit/web/composer/test_blob_inline_tools.py -q
  ```

- [ ] Run the complete 08A handoff gates:

  ```bash
  uv sync --frozen --all-extras
  uv run pytest tests/unit/web/test_provider_config_policy.py tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_service.py tests/unit/web/sessions/test_routes.py tests/unit/web/composer/test_tools.py tests/unit/web/composer/test_blob_inline_tools.py -q
  uv run pytest tests/unit/web -q
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing uv run elspeth-lints diagnose-judge-signatures --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --format text
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing uv run elspeth-lints check --rules trust_tier.tier_model --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
  git diff --check
  test "$(wardline --version)" = "wardline, version 1.3.1"
  PYTHONPATH="$PWD" wardline scan . --trust-pack config.wardline.elspeth_pack --allow-custom-packs --fail-on ERROR --fail-on-unanalyzed --format agent-summary --output /tmp/aws-wardline-agent-summary.json
  jq -e '
    .schema == "wardline-agent-summary-1" and
    .gate.verdict == "PASSED" and .gate.tripped == false and .gate.exit_class == 0 and
    .summary.active_defects == 0 and .summary.unanalyzed == 0 and
    .resolution.inert == false and .resolution.recognized_boundaries > 0 and
    all(.engine_facts[]?;
      .rule_id != "WLN-ENGINE-FUNCTION-SKIPPED" and
      .rule_id != "WLN-ENGINE-PYDANTIC-DISCOVERY-LIMIT")
  ' /tmp/aws-wardline-agent-summary.json >/dev/null
  ```

  Wardline exit 1 requires `wardline explain-taint <fingerprint> --chain`, a fix at the authoring/config boundary, and a rescan. Exit 2 blocks handoff. Exit 0 remains subject to Task 0's non-inert rule: current engine metrics and recognized boundary count greater than zero are mandatory; `INERT`, unanalyzed source units/files, either formal incomplete-analysis engine fact, missing metrics, or an active ERROR+ defect blocks. Conservative resolution facts outside Wardline 1.3.1's formal incomplete-analysis set remain visible but do not imply incomplete analysis. Do not baseline, waive, allowlist, or narrow away a finding. The authoritative scan supplements rather than replaces the explicit security regressions.

  Task 3 edits scopes that already own signed tier-model entries (`_execute_set_source_from_blob` and `_execute_set_pipeline`), so signature drift is expected after the code change. The implementing agent reruns read-only diagnosis and stops. The operator follows the Task-0 repair workflow for only the diagnosed changed rows; the agent then reruns diagnosis plus the shape-only gate. The HMAC key stays operator-only and must never enter the agent environment.

- [ ] Stage only Task-3 files, commit, and hand 08A to the integration coordinator:

  ```bash
  git add src/elspeth/web/composer/tools/sources.py src/elspeth/web/composer/tools/outputs.py src/elspeth/web/composer/tools/sessions.py src/elspeth/web/composer/tools/secrets.py src/elspeth/web/composer/tools/blobs.py tests/unit/web/composer/test_tools.py tests/unit/web/composer/test_blob_inline_tools.py config/cicd/enforce_tier_model/web.yaml
  git commit -m "feat(web): gate aws_s3 endpoints in composer mutations"
  ```

  If a gate required changes to Task-1/2 files, explicitly stage those
  corrections too and restart all 08A gates. Never use `git add -A`. The
  worker does not close from the feature branch: the coordinator integrates,
  reruns the handoff gates, then closes 08A with the integrated release SHA.
  Plan 06/07 become startable only after that close.

---

### Task 4: Add guided-source parity after Plan 06

**08B prerequisite:** Plan 06 and 08A are closed. Claim `elspeth-a342f333a4` atomically only then.

**08B files:**

- Modify: `src/elspeth/web/composer/guided/chat_solver.py`
- Modify: `tests/unit/web/composer/guided/test_chat_solver.py`

- [ ] Add a focused prompt regression around `_build_step_1_source_dynamic_block()`. Assert the Step-1 prompt includes `aws_s3` in the allowed source inventory, names `endpoint_url`, and says web authors must never set it. This assertion is load-bearing; the existing guided integration suite does not pin the new text.
- [ ] Update the live hardcoded source list to include `aws_s3`, and append concise guidance that `endpoint_url` is CLI/batch-only and rejected in web-authored configuration. Do not add access-key, secret-key, or session-token fields.
- [ ] Run focused, guided, static, trust, and Wardline gates:

  ```bash
  uv run pytest tests/unit/web/composer/guided/test_chat_solver.py -q
  uv run pytest tests/integration/web/composer/guided/ -q
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
  git diff --check
  test "$(wardline --version)" = "wardline, version 1.3.1"
  PYTHONPATH="$PWD" wardline scan . --trust-pack config.wardline.elspeth_pack --allow-custom-packs --fail-on ERROR --fail-on-unanalyzed --format agent-summary --output /tmp/aws-wardline-agent-summary.json
  jq -e '
    .schema == "wardline-agent-summary-1" and
    .gate.verdict == "PASSED" and .gate.tripped == false and .gate.exit_class == 0 and
    .summary.active_defects == 0 and .summary.unanalyzed == 0 and
    .resolution.inert == false and .resolution.recognized_boundaries > 0 and
    all(.engine_facts[]?;
      .rule_id != "WLN-ENGINE-FUNCTION-SKIPPED" and
      .rule_id != "WLN-ENGINE-PYDANTIC-DISCOVERY-LIMIT")
  ' /tmp/aws-wardline-agent-summary.json >/dev/null
  ```

  Exit 0 remains subject to Task 0's non-inert rule: current engine metrics and recognized boundary count greater than zero are mandatory; `INERT`, unanalyzed source units/files, either formal incomplete-analysis engine fact, missing metrics, an untrusted or missing pack, or an active ERROR+ defect blocks 08B. Conservative resolution facts outside Wardline 1.3.1's formal incomplete-analysis set remain visible but do not imply incomplete analysis. Do not baseline, waive, allowlist, or narrow the scan.

- [ ] Stage exactly, commit, and hand 08B to the integration coordinator:

  ```bash
  git add src/elspeth/web/composer/guided/chat_solver.py tests/unit/web/composer/guided/test_chat_solver.py
  git commit -m "feat(web): guide aws_s3 authoring behind endpoint gate"
  ```

  The worker reports evidence but does not close from its feature branch. The
  coordinator integrates, reruns the 08B gates, and closes with the integrated
  release SHA before Plan 10/12 may consume this slice.

## Cross-plan acceptance

- Plan 06 must keep `endpoint_url` canonical/top-level, reject aliases/nested client config/extras, assert it is absent from source `allowed_secret_ref_fields`, and prove a CLI source config survives both settings load and plugin instantiation.
- Plan 07 must mirror that contract and non-secret assertion for the sink and prove CLI sink acceptance.
- Plan 10 consumes both closed 08 slices; Plan 12 re-runs the full suite, trust/static/authoritative non-inert Wardline gates, and live S3 acceptance. No web test may skip/xfail the endpoint rejection.

**Accepted limitation:** Plan 08 deliberately rejects every non-null web value rather than attempting URL allowlisting. Operator-controlled endpoints remain a CLI/batch concern.
