# AWS S3 Source Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add an `aws_s3` source plugin (CSV/JSON/JSONL, AWS default credential chain, `max_object_bytes` guard) plus the shared `build_s3_client` helper and `aws` packaging extra it depends on.

**Architecture:** `src/elspeth/plugins/aws_s3_common.py` provides `build_s3_client()`, a lazy-import boto3 factory reused by the sibling `aws_s3` sink. `src/elspeth/plugins/sources/aws_s3_source.py` mirrors `AzureBlobSource` (`azure_blob_source.py`) almost line-for-line — same schema/field-normalization/quarantine/contract machinery — swapping blob download for a `head_object`-then-capped-stream S3 download. Registration is automatic (discovery.py scans `plugins/sources/*.py` for a `name` attribute — no registry edit).

**Tech Stack:** boto3/botocore (new `aws` extra), pydantic v2, existing `field_normalization`/`schema_contract_factory` infra.

**Depends on:** `2026-07-08-aws-ecs-08-s3-endpoint-gate.md` — `aws_s3` must not reach a web-catalog-reachable deployment before plan 08's `endpoint_url` gate lands (auto-discovery registers it into the web composer catalog on merge, so registration = web-reachable). This plan's tasks have no other dependency and may be built in parallel; the release/deploy checklist must not let this reach staging or prod ahead of plan 08.

**Global Constraints** (verbatim from spec):
- "`max_object_bytes` guard, defaulting to 256 MiB. An object over the limit fails closed with a clear error instead of being read fully into memory."
- Config: `bucket`, `key`, `format`, CSV/JSON options equivalent to Azure source, optional `region_name`/`endpoint_url`. "must not include AWS access-key, secret-key, or session-token fields."
- Audit: "provider `aws_s3`... operation, bucket, key, byte count, content hash where available... latency, and sanitized error class... not record credentials, presigned URLs, raw secret-bearing endpoint strings, or unbounded provider error bodies."
- Packaging: `aws` extra for boto3/botocore.
- Known gap, intentionally left alone: `chat_solver.py:444`'s guided-mode valid-source list omits `aws_s3`. Do not add it here — that would steer web authors to a source this plan's header says isn't web-safe yet. Follow-up: update alongside plan 08.

### Task 1: `aws` packaging extra + shared `build_s3_client` helper

**Files:**
- Modify: `pyproject.toml:140-141` (new `aws = [...]` block after the `azure` extra's closing `]`, before `mcp = [` at :142) and the `all` extra's dependency list (~line 210, add under a new `# aws dependencies` comment).
- Modify: `uv.lock` (regenerated here — Task 1 owns this per the cross-plan uv.lock-ownership decision; plan 10's frozen `uv sync --frozen --all-extras` references this precondition, it does not re-own the regenerate step).
- Create: `src/elspeth/plugins/aws_s3_common.py`
- Test: `tests/unit/plugins/test_aws_s3_common.py`

**Interfaces:**
- Produces: `build_s3_client(region_name: str | None, endpoint_url: str | None) -> Any` (pinned name/signature — the sibling sink plan imports this).

```python
"""Shared AWS S3 client builder — AWS default credential chain, no credential
params. boto3/botocore are imported lazily (mirrors azure_auth.py's deferred
SDK import) so plugin discovery succeeds without the `aws` extra installed."""
from __future__ import annotations

from typing import Any


def build_s3_client(region_name: str | None, endpoint_url: str | None) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise ImportError("boto3 is required for aws_s3 plugins. Install with: uv pip install boto3") from e
    # boto3 defaults (60s/60s) let a stalled endpoint hang a worker for minutes; bound it.
    config = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 3})
    return boto3.client("s3", region_name=region_name, endpoint_url=endpoint_url, config=config)
```

**Steps:**
- [ ] Write `test_build_s3_client_passes_region_and_endpoint` (patch `boto3.client`, assert `region_name=`/`endpoint_url=`/a timeout-bearing `config=`, no credential kwargs) and `test_build_s3_client_none_args_pass_through`. Run `pytest tests/unit/plugins/test_aws_s3_common.py -x` → expect `ModuleNotFoundError`.
- [ ] Add `aws = ["boto3>=1.40,<2", "botocore>=1.40,<2"]` to `pyproject.toml` (extra block + `all` rollup); implement `aws_s3_common.py` above. Rerun → PASS.
- [ ] Regenerate the lockfile so CI's `uv sync --frozen --all-extras` (`ci.yaml`, multiple jobs) doesn't break on the new extra: `uv sync --extra dev --extra aws`, then verify `python -c "import boto3, botocore"` (mirrors plan 02's Task 1 pattern for `postgres`).
- [ ] `git add pyproject.toml uv.lock src/elspeth/plugins/aws_s3_common.py tests/unit/plugins/test_aws_s3_common.py && git commit -m "feat(plugins): add aws packaging extra and shared S3 client builder"`

### Task 2: `AwsS3SourceConfig` + registration + catalog hints

**Files:**
- Create: `src/elspeth/plugins/sources/aws_s3_source.py`
- Test: `tests/unit/plugins/sources/test_aws_s3_source.py`
- Create: `tests/golden/web/catalog/knob_schema/source__aws_s3.json`
- Modify: `src/elspeth/web/audit_readiness/boundary_expectations.py:124` (add `"aws_s3": Determinism.IO_READ` to `EXPECTED_SOURCE_DETERMINISMS` **before** `"azure_blob"` — dict is alphabetical, `aws_s3` sorts first; enforced by `test_boundary_predicate_parity.py`)
- Modify: `tests/unit/plugins/test_discovery.py:252` (bump `EXPECTED_SOURCE_COUNT = 6` → `7` + inline comment; discovery globs `plugins/sources/*.py` non-recursively so the new file registers a 7th source, and this test runs unmarked in default `pytest tests/`)

**Interfaces:**
- Consumes: `build_s3_client` (Task 1); `BaseSource` (`infrastructure/base.py:1276`) / `DataPluginConfig` (`config_base.py:268`); `reject_operator_required_placeholder_value` (used at `azure_blob_source.py:308,316`); local `CSVOptions`/`JSONOptions` mirrored verbatim from `:67-112`; `__init__`'s schema/contract-deferral setup mirrored from `azure_blob_source.py:387-438` (CSV defers the `ContractBuilder`; JSON/JSONL create it eagerly when the contract locks).
- Produces: `name = "aws_s3"` (auto-discovered — `docs/contracts/plugin-protocol.md` "Built-In Discovery Is Dynamic", no registry edit).

```python
class AwsS3SourceConfig(DataPluginConfig):
    _plugin_component_type: ClassVar[str | None] = "source"
    bucket: str = Field(...)
    key: str = Field(...)
    format: Literal["csv", "json", "jsonl"] = Field(default="csv")
    csv_options: CSVOptions = Field(default_factory=CSVOptions)
    json_options: JSONOptions = Field(default_factory=JSONOptions)
    columns: list[str] | None = Field(default=None)
    field_mapping: dict[str, str] | None = Field(default=None)
    on_validation_failure: str = Field(...)
    region_name: str | None = Field(default=None)  # falls back to default credential chain
    endpoint_url: str | None = Field(default=None)  # CLI/batch authorship only (gate: sibling plan 08)
    # le=: defensive ceiling, not a spec number — web-authorable like endpoint_url.
    max_object_bytes: int = Field(default=256 * 1024 * 1024, gt=0, le=1024 * 1024 * 1024)

    @field_validator("bucket")
    @classmethod
    def _validate_bucket(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("bucket cannot be empty")
        return reject_operator_required_placeholder_value(v, field_name="bucket")

    @field_validator("key")
    @classmethod
    def _validate_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("key cannot be empty")
        return reject_operator_required_placeholder_value(v, field_name="key")
```

No `AwsAuthConfig` — no auth fields, unlike Azure. Class attributes `name = "aws_s3"`, `determinism = Determinism.IO_READ`, `plugin_version = "1.0.0"`, `config_model = AwsS3SourceConfig` mirror `azure_blob_source.py:362-366` (`BaseSource.__init_subclass__`, `base.py:1429-1450`, requires `determinism` redeclared per-class). `_get_s3_client()` calls `build_s3_client(self._region_name, self._endpoint_url)` lazily, cached on `self._s3_client`, mirroring `_get_blob_client` (`:440-459`). `validate_field_normalization_options` (`:250-284`) and `columns`/`field_mapping` carry over unchanged. `get_agent_assistance` mirrors `:369-385`: `bucket`+`key` required, no credential fields, `endpoint_url` CLI/batch-only, `max_object_bytes` capped at 1 GiB.

**Steps:**
- [ ] Write, in `TestAwsS3SourceConfig`, in the style of `TestAzureBlobSourceConfig` (`:143-293`) minus auth tests: `test_empty_bucket_raises`, `test_placeholder_bucket_raises` (`:217-221`), `test_plain_placeholder_words_can_be_bucket_names` (counter-test, `:224-228`, so the heuristic doesn't over-match ordinary words), the same empty/placeholder/counter trio for `key`, `test_max_object_bytes_default_256mib`, `test_max_object_bytes_must_be_positive`, `test_max_object_bytes_rejects_over_1gib`, `test_credential_fields_rejected` (`aws_access_key_id` in config dict → `PluginConfigError` via `extra="forbid"`, `config_base.py:164`), and `test_fixed_schema_creates_locked_contract_for_json`/`test_observed_schema_defers_contract_builder_until_json_field_resolution`/`test_csv_defers_contract_until_load` ported verbatim from `:276-292` (construction-only, no aws_s3-specific behavior). Run `pytest tests/unit/plugins/sources/test_aws_s3_source.py -x` → expect `ModuleNotFoundError`.
- [ ] Implement `AwsS3SourceConfig` + `AwsS3Source` skeleton: class attributes above, `__init__` per `:387-438`, `_get_s3_client`, `close()` → `self._s3_client = None` (plus `test_close_nulls_client`/`test_close_idempotent`, `:495-513`), `get_agent_assistance`, `load()` stub raising `NotImplementedError`. Set `source_file_hash: str | None = "sha256:0000000000000000"` as an explicit placeholder (so the later hash-fix step has a line to rewrite). Rerun → PASS.
- [ ] Generate the golden knob-schema snapshot: `python -c "import json; from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager; from elspeth.web.catalog.service import CatalogServiceImpl; svc = CatalogServiceImpl(get_shared_plugin_manager()); info = svc._schema_cache[('source', 'aws_s3')]; open('tests/golden/web/catalog/knob_schema/source__aws_s3.json', 'w').write(json.dumps({'plugin_kind': 'source', 'plugin_name': 'aws_s3', 'knob_schema': info.knob_schema}, indent=2, sort_keys=True) + '\n')"`. Run `pytest tests/unit/web/catalog/test_knob_schema_golden.py -x` → PASS.
- [ ] Add `"aws_s3": Determinism.IO_READ` to `EXPECTED_SOURCE_DETERMINISMS`, before `"azure_blob"`. Run `pytest tests/unit/web/audit_readiness/test_boundary_predicate_parity.py::test_every_source_classifies_as_boundary -x` → PASS.
- [ ] Bump `EXPECTED_SOURCE_COUNT = 6` → `7` in `test_discovery.py:252` (update the inline plugin-name comment too). Run `pytest tests/unit/plugins/test_discovery.py::test_discovery_finds_expected_plugin_counts -x` → PASS.
- [ ] Fix the source-file hash for real: `python -c "from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash; from pathlib import Path; p = Path('src/elspeth/plugins/sources/aws_s3_source.py'); fix_source_file_hash(p, 'AwsS3Source', compute_source_file_hash(p))"` (in-place rewrite — avoids the self-referential mismatch trap of hand-pasting a printed value).
- [ ] `git add src/elspeth/plugins/sources/aws_s3_source.py tests/unit/plugins/sources/test_aws_s3_source.py tests/golden/web/catalog/knob_schema/source__aws_s3.json src/elspeth/web/audit_readiness/boundary_expectations.py tests/unit/plugins/test_discovery.py && git commit -m "feat(plugins): add aws_s3 source config and registration"`

### Task 3: CSV/JSON/JSONL `load()` with guarded download + audit

**Files:**
- Modify: `src/elspeth/plugins/sources/aws_s3_source.py`
- Test: `tests/unit/plugins/sources/test_aws_s3_source.py` (append)
- Test: `tests/property/plugins/sources/test_aws_s3_source_properties.py`
- Test: `tests/integration/cli/test_aws_s3_endpoint_url_accepted.py` (new — closes the named cross-plan obligation from `2026-07-08-aws-ecs-08-s3-endpoint-gate.md`'s `Depends on:`; this plan owns it, not plan 07)

**Interfaces:** Mirror `_load_csv`, `_load_json_array`, `_load_jsonl`, `_normalize_row_keys`, `_resolve_json_field_names`, `_validate_and_yield`, `get_field_resolution` (`azure_blob_source.py:553-1190`) verbatim except `self._container`→`self._bucket`, `self._blob_path`→`self._key`, raw-row dict keys `"container"/"blob_path"`→`"bucket"/"key"`, schema class name `AwsS3RowSchema`, provider string `"aws_s3"`.

Novel replacement for `blob_client.download_blob().readall()` (`azure_blob_source.py:486` — the unbounded read this guard must not replicate):

```python
def _download_object(self) -> tuple[bytes, str]:
    client = self._get_s3_client()
    head = client.head_object(Bucket=self._bucket, Key=self._key)
    content_length = head.get("ContentLength")
    if content_length is not None and content_length > self._max_object_bytes:
        raise RuntimeError(
            f"S3 object '{self._key}' in bucket '{self._bucket}' is {content_length} bytes, "
            f"exceeding max_object_bytes={self._max_object_bytes}. Refusing to load."
        )
    response = client.get_object(Bucket=self._bucket, Key=self._key)
    body = response["Body"]
    chunks: list[bytes] = []
    total = 0
    try:
        while chunk := body.read(65536):
            total += len(chunk)
            if total > self._max_object_bytes:
                raise RuntimeError(
                    f"S3 object '{self._key}' in bucket '{self._bucket}' exceeded "
                    f"max_object_bytes={self._max_object_bytes} while streaming "
                    f"(Content-Length absent or understated). Refusing to load."
                )
            chunks.append(chunk)
    finally:
        body.close()  # unconditional: also reached if body.read() itself raises mid-stream
    obj_bytes = b"".join(chunks)
    content_hash = hashlib.sha256(obj_bytes).hexdigest()
    return obj_bytes, content_hash
```

`load()` wraps this like `:481-535`'s ladder (ImportError re-raised; `TypeError`/`AttributeError`/`KeyError`/`NameError`/`ValueError` re-raised as bugs; a nested `try/except` around the success-path `ctx.record_call` raises `AuditIntegrityError` on an audit-write failure so it's never misattributed as a download failure, mirroring `:490-508`; other `Exception` records an ERROR call then raises `RuntimeError`). Divergences: `request_data={"operation": "read_object", "bucket": ..., "key": ...}` (never `endpoint_url`; labelled `read_object`, not `get_object`, since `_download_object()` makes two real calls and a `head_object`-stage failure — including the guard's own oversized-object `RuntimeError` — would otherwise be filed under a label that never ran); `response_data={"size_bytes": len(obj_bytes), "content_hash": content_hash}`, where `content_hash` is a SHA-256 over the downloaded bytes (not the S3 `ETag`, unreliable for multipart uploads); `provider="aws_s3"`; audit `error=` is `{"type": type(e).__name__}` only — no `"message": str(e)` (spec bars unbounded provider error bodies from audit; the raised `RuntimeError` still carries `str(e)` for the operator). Add `import hashlib`.

**Steps:**
- [ ] Write tests in the style of `TestAzureBlobSourceCSV`/`JSON`/`JSONL`/`AuditAndErrors` (`:300-1125`, through `AuditAndErrors`; stops before `SchemaValidation`/`FieldResolutionUnion`/`SparseFieldMapping` at `:838`+, not mirrored here) against a fake boto3 client exposing `head_object`/`get_object`. Include aws_s3 equivalents of `test_download_failure_raises_runtime_error`, `test_import_error_propagated`, `test_programming_errors_crash_directly`, `test_audit_integrity_error_on_record_call_failure` (`:1042` — fake `ctx.record_call` raises after success, assert `AuditIntegrityError` not `RuntimeError`), `test_success_paths_record_audit_without_normal_info_logs` (`:1069` — exact-match `request_data`, catching any `endpoint_url`/credential leak). Add `test_max_object_bytes_head_object_rejects_oversized` (large `ContentLength` → `RuntimeError`, `get_object` never called), `test_max_object_bytes_streamed_read_rejects_when_content_length_absent` (no `ContentLength`, oversized stream → `RuntimeError`, fake body's `close()` called), `test_download_body_closed_on_mid_stream_read_error` (fake `read()` raises → `close()` still called via `finally`), `test_audit_error_records_type_only_not_message`, `test_content_hash_is_sha256_of_downloaded_bytes`. Run `pytest tests/unit/plugins/sources/test_aws_s3_source.py -x` → expect `NotImplementedError`.
- [ ] Implement `load()`/`_download_object()` + mirrored parsing methods. Rerun → PASS.
- [ ] Port `TestAzureBlobSourceCSVProperties`/`JSONProperties`/`QuarantineProperties` (`test_azure_blob_source_properties.py`) to `test_aws_s3_source_properties.py` against the fake client. Run `pytest tests/property/plugins/sources/test_aws_s3_source_properties.py -x` → PASS.
- [ ] Write `tests/integration/cli/test_aws_s3_endpoint_url_accepted.py::test_cli_aws_s3_endpoint_url_accepted` — the named cross-plan obligation plan 08 assigns to this plan (its `Depends on:` section). Build a minimal pipeline YAML in the style of `_build_yaml_with_model` (`tests/integration/cli/test_instantiate_plugins_value_source.py:42-52`): `sources.primary` with `plugin: aws_s3`, `on_success: output`, `options: {bucket: "test-bucket", key: "data.csv", endpoint_url: "http://localhost:4566", on_validation_failure: "discard", schema: {mode: observed}}`, wired directly to a `plugin: csv` sink — no transform needed, a bare source→sink DAG is valid (`ElspethSettings`, mirrors `test_minimal_valid_config`, `tests/unit/core/test_config.py:79-86`). Call `config = load_settings_from_yaml_string(yaml_text)`, then `bundle = instantiate_plugins_from_config(config)`; assert it does not raise and `bundle.sources["primary"]._endpoint_url == "http://localhost:4566"`. `instantiate_plugins_from_config` is the real acceptance gate, not `load_settings_from_yaml_string` alone: `SourceSettings.options` is an untyped `dict[str, Any]` (`core/config.py:1061`) that accepts any key, so only `manager.get_source_by_name("aws_s3")` constructing `AwsS3Source(dict(source_config.options))` (`runtime_factory.py:76-78`) actually runs `AwsS3SourceConfig`'s field validation — and this is also where the CLI path diverges from the web path, since it never touches `elspeth.web.provider_config_policy`/`validate_pipeline`. No client mocking needed: `_get_s3_client()` builds the boto3 client lazily (Task 2), so construction alone never calls AWS. Run `pytest tests/integration/cli/test_aws_s3_endpoint_url_accepted.py -x` → PASS.
- [ ] Full slice check: `pytest tests/unit/plugins/sources/test_aws_s3_source.py tests/property/plugins/sources/test_aws_s3_source_properties.py tests/integration/cli/test_aws_s3_endpoint_url_accepted.py tests/unit/web/catalog/test_knob_schema_golden.py tests/integration/web/test_catalog_discovery.py tests/unit/web/audit_readiness/test_boundary_predicate_parity.py tests/unit/plugins/test_aws_s3_common.py tests/unit/plugins/test_discovery.py tests/unit/scripts/cicd/test_plugin_hash.py -q` → all PASS.
- [ ] `git commit -m "feat(plugins): implement aws_s3 source load with max_object_bytes guard"`

**Deviations:** `content_hash` is a SHA-256 of the downloaded bytes, not the S3 `ETag` — new surface (Azure's mirror has no `content_hash` field at all, `:499`), not a renamed pinned interface; `ETag` was rejected as unreliable for multipart-uploaded objects.
