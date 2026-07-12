# AWS S3 Source Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-safe `aws_s3` source for CSV, JSON arrays, and JSONL using the AWS default credential chain, bounded spooled downloads, incremental parsing/hashing, redacted audit failures, and a shared S3 client builder for Plan 07.

**Architecture:** `src/elspeth/plugins/aws_s3_common.py` owns the lazy boto3 client factory. `src/elspeth/plugins/sources/aws_s3_source.py` first lands as an unregistered config/download module, then atomically adds the complete `AWSS3Source` registration only after all parsers and tests exist. Raw S3 bytes stream into a small-memory `SpooledTemporaryFile` while SHA-256 and size limits are enforced; CSV/JSONL parse from the spool and JSON arrays use `ijson` one item at a time, preserving Azure source schema/normalization/quarantine semantics without retaining the entire object twice in memory.

**Tech Stack:** Python 3.13, boto3/botocore, ijson, Pydantic v2, tempfile spooling, existing source-contract and audit infrastructure, pytest/Hypothesis.

**Hard dependencies and ordering:**

- Plan 02 / Filigree `elspeth-9070fb0a45` lands first. Plan 06 rebases onto its `postgres` extra and regenerated `uv.lock`, then regenerates the one authoritative combined postgres+AWS lock. Never text-merge `uv.lock`.
- Plan 08A Tasks 1-3 / Filigree `elspeth-c0103e6c88` land before the `AWSS3Source` registration commit. Tasks 1-2 are the load-bearing validation gate and Task 3 completes immediate composer-mutation defenses. Auto-discovery makes registration web-catalog-reachable; the endpoint gate test is deliberately red without Plan 08A.
- Plan 06 implementation issue: `elspeth-7fe6aa531f`. This plan repair does not claim it.
- Plan 08B Task 4 still owns guided-mode prompt parity after the plugin exists. Plan 07 consumes `build_s3_client`, the `aws` extra, and the house-form `AWSS3*` naming established here.

**Global constraints:**

- The public class names are `AWSS3Source` and `AWSS3SourceConfig`; the schema class is `AWSS3RowSchema`. This matches `AWSS3Sink`, `CSVSink`, `JSONSource`, and the live Filigree implementation note.
- Pipeline config contains only `bucket`, `key`, format/parser/schema options, `region_name`, `endpoint_url`, `max_object_bytes`, and `max_record_chars`. It has no access-key, secret-key, session-token, credential-provider, nested client-config, or passthrough kwargs surface.
- CLI/batch `endpoint_url` remains supported, but validation requires a bounded HTTP(S) URL with a hostname, no userinfo, query, fragment, control characters, or whitespace. Plain HTTP is retained for localhost/LocalStack. Web-authored non-null values are rejected by Plan 08; an omitted field or explicit `null` is harmless and accepted.
- `max_object_bytes` defaults to 256 MiB, is positive, and has a 1 GiB defensive ceiling. `max_record_chars` defaults to 1,000,000 decoded characters and has an 8,000,000-character ceiling; this bounds a single parser record independently of variable-width source encodings.
- Downloads use `SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")`; raw object bytes above 8 MiB spill to a private temporary file. No list of chunks, `b"".join`, full-object `bytes`, or full-document decoded string is retained.
- HEAD and GET `ContentLength` values must be exact non-boolean integers `>= 0`. HEAD supplies an ETag that is an exact `str`, 1–1024 UTF-8 bytes, and printable ASCII only (`0x20`–`0x7e`); GET uses that exact value as `IfMatch`. GET length must match HEAD and final bytes read. Missing/malformed metadata, `Body`, or body methods fail closed.
- Unsupported non-empty `ContentEncoding` values fail closed before parsing. This milestone does not transparently decompress S3 objects, so the byte limit always measures the bytes parsed.
- Every S3 body and spool closes under success, parse failure, audit failure, generator close, and source close. A body-close failure never masks an earlier size/read failure; its sanitized class is retained as bounded cleanup metadata. `AWSS3Source.close()` calls the cached boto client's `close()` exactly once and clears the reference even when close raises.
- Provider exception messages, response bodies, endpoint URLs, and raw causes never enter public exceptions, audit, telemetry, or logs. Public S3 exceptions are purpose-built, static/class-only, and raised outside the provider `except` block so `__cause__`/`__context__` do not retain the raw exception. Controlled size failures may expose only numeric observed/limit facts.
- The source module has no top-level boto3, botocore, or ijson import. Built-in discovery must still import cleanly without the optional `aws` extra; missing SDK/parser dependencies fail actionably only when the corresponding S3 operation runs.
- Both success- and failure-path `ctx.record_call` writes are audit-primary. Existing Tier-1 errors propagate; any other audit write failure becomes a redacted `AuditIntegrityError` raised outside the raw recorder exception handler, with the recorder class normalized to `[A-Za-z_][A-Za-z0-9_]{0,127}` and no raw cause/context. Error audit payloads contain a similarly normalized provider class plus bounded integers only.
- CSV, JSON, and JSONL retain Azure parity for schema construction, field normalization, sparse-field union, quarantine/discard, non-finite rejection, surrogate handling, contract locking, and validation-error safety. Empty CSV/JSON are structural parse failures subject to quarantine/discard; empty JSONL yields no rows.
- No normal success/info logging is added. Probative download facts live in the Landscape call record; operational cleanup/audit failures follow the repository logging/telemetry policy.

---

### Task 1: Add the ordered AWS dependency slice and deterministic client builder

**Files:**

- Modify: `pyproject.toml` (`aws` extra, `all` rollup, stubless mypy override)
- Modify: `uv.lock` (regenerate only after rebasing onto completed Plan 02)
- Create: `src/elspeth/plugins/aws_s3_common.py`
- Create: `tests/unit/plugins/test_aws_s3_common.py`

**Produces:**

```python
def build_s3_client(region_name: str | None, endpoint_url: str | None) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError(
            'boto3 is required for aws_s3 plugins; install Elspeth with the "aws" extra'
        ) from exc

    config = Config(
        connect_timeout=10,
        read_timeout=30,
        retries={"mode": "standard", "total_max_attempts": 3},
    )
    return boto3.client(
        "s3",
        region_name=region_name,
        endpoint_url=endpoint_url,
        config=config,
    )
```

HEAD and GET each have their own maximum three-total-attempt SDK budget. A `StreamingBody.read()` failure after GET returns is not replayed by this helper; the source fails the logical read and records one safe failure.

- [ ] Write `test_build_s3_client_passes_region_endpoint_and_exact_config`, `test_none_args_pass_through`, `test_no_credential_kwargs`, and `test_missing_sdk_error_names_the_aws_extra`. Inspect the captured `Config` and require `connect_timeout == 10`, `read_timeout == 30`, `retries == {"mode": "standard", "total_max_attempts": 3}`. Run `uv run pytest tests/unit/plugins/test_aws_s3_common.py -x`; expect import failure before the module exists.
- [ ] Add `aws = ["boto3>=1.40,<2", "botocore>=1.40,<2", "ijson>=3.3,<4"]` after `azure`, add those packages to `all`, and add `boto3.*`, `botocore.*`, and `ijson.*` to the existing documented stubless mypy override. Do not add boto3 type-stub packages or credentials.
- [ ] Implement `aws_s3_common.py` exactly at the lazy-import boundary above. Importing the module must succeed without the `aws` extra; only calling the builder may raise its actionable `ImportError`.
- [ ] Regenerate from the Plan-02 tree and prove the combined lock, never a text merge:

  ```bash
  uv lock
  uv lock --check
  uv sync --frozen --all-extras
  uv run python -c "import boto3, botocore, ijson, psycopg"
  ```

  Expected: all four commands exit 0 and `uv.lock` contains both the Plan-02 PostgreSQL slice and this AWS slice.
- [ ] Prove both optional-dependency boundaries in fresh isolated environments rather than relying on the all-extras development environment:

  ```bash
  uv run --isolated --no-dev --frozen python -c "import elspeth.plugins.aws_s3_common; from elspeth.plugins.aws_s3_common import build_s3_client;
try:
    build_s3_client(None, None)
except ImportError as e:
    assert 'aws' in str(e).lower()
else:
    raise AssertionError('base install unexpectedly supplied boto3')"
  uv run --isolated --no-dev --frozen --extra aws python -c "import boto3, botocore, ijson, elspeth.plugins.aws_s3_common"
  ```

  The first command proves the shared module imports without the optional SDK and fails actionably only when its client builder runs. The second proves the standalone `aws` extra is sufficient. Registration itself does not exist until Task 3 and is checked in Task 4's isolated commands. Both commands here must exit 0.
- [ ] Run `uv run pytest tests/unit/plugins/test_aws_s3_common.py -q`, then commit only the Task-1 files:

  ```bash
  git add pyproject.toml uv.lock src/elspeth/plugins/aws_s3_common.py tests/unit/plugins/test_aws_s3_common.py
  git commit -m "feat(plugins): add bounded shared S3 client support"
  ```

---

### Task 2: Land validated config and bounded download primitives without registering a plugin

**Files:**

- Create: `src/elspeth/plugins/sources/aws_s3_source.py` (config/options/download helpers only; no `BaseSource` subclass yet)
- Create: `tests/unit/plugins/sources/test_aws_s3_source.py`

`AWSS3SourceConfig(DataPluginConfig)` declares `_plugin_component_type = "source"` and every top-level field uses `Field` with a concrete non-empty description so `plugin_contract.options_metadata` passes once registration lands:

```python
bucket: str = Field(..., description="S3 bucket name or access-point identifier")
key: str = Field(..., description="S3 object key")
format: Literal["csv", "json", "jsonl"] = Field(default="csv", description="S3 object data format")
csv_options: CSVOptions = Field(default_factory=CSVOptions, description="CSV parsing options")
json_options: JSONOptions = Field(default_factory=JSONOptions, description="JSON and JSONL parsing options")
columns: list[str] | None = Field(default=None, description="Explicit columns for headerless CSV")
field_mapping: dict[str, str] | None = Field(default=None, description="Overrides for normalized source fields")
on_validation_failure: str = Field(..., description="Quarantine sink name or explicit discard")
region_name: str | None = Field(default=None, description="AWS signing region override")
endpoint_url: str | None = Field(default=None, description="CLI/batch-only S3-compatible HTTP endpoint")
max_object_bytes: int = Field(default=256 * 1024 * 1024, gt=0, le=1024 * 1024 * 1024, description="Maximum S3 object bytes accepted")
max_record_chars: int = Field(default=1_000_000, gt=0, le=8_000_000, description="Maximum decoded characters in one source record")
```

Bucket/key validators retain the operator-placeholder checks and add bounded shape: bucket/access-point text is nonblank and at most 2048 characters; key is nonblank, contains no NUL/control character, and is at most 1024 UTF-8 bytes. `region_name`, when present, is a nonblank maximum-64-character `[A-Za-z0-9-]+` value. The endpoint validator uses `urlsplit`, limits the original string to 2048 characters, permits only `http`/`https`, requires a hostname, and rejects username/password, query, fragment, whitespace, and control characters without resolving DNS or making a request.

`_download_s3_object(client, *, bucket, key, max_object_bytes) -> _DownloadedObject` is independently testable before registration. `_DownloadedObject` owns the spooled binary handle plus `size_bytes`, `content_hash`, and bounded audit metadata, and implements idempotent `close()`/context-manager cleanup.

Download algorithm (define `_MAX_ETAG_BYTES = 1024` beside the other resource constants):

1. `head_object`, then validate `ContentLength`, `ETag`, and `ContentEncoding`; reject an over-limit HEAD before GET.
2. `get_object(..., IfMatch=head_etag)`, validate the GET length/encoding and exact `Body` interface, and reject length mismatch before reading.
3. Read at most 64 KiB per iteration, increment SHA-256 and byte count, fail as soon as `total > max_object_bytes`, and write each accepted chunk to the spool. Require final `total == ContentLength`.
4. Close `Body` in all paths. If a primary failure exists, preserve it and attach only the close exception's sanitized class; otherwise a close failure becomes the primary safe provider failure.
5. On any failure, close the spool and raise a purpose-built `S3SourceReadError` or `S3ObjectSizeLimitError` outside the raw exception handler with no raw cause/context. Normalize every captured exception class through the closed identifier rule above. On success, rewind and transfer spool ownership to `_DownloadedObject`.

- [ ] Write config tests for all Azure-equivalent delimiter/encoding/columns/field-mapping combinations, placeholder counterexamples, exact object-byte and decoded-record-character limits, region shape, and endpoint URL accept/reject cases (HTTPS, LocalStack HTTP, oversized, userinfo, query, fragment, non-HTTP, malformed, whitespace/control). Parameterize forbidden config over `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, `access_key`, `secret_key`, `session_token`, `credentials`, `client_config`, and `client_kwargs`; assert these names are also absent from `AWSS3SourceConfig.model_fields`.
- [ ] Write download tests for HEAD max-1/max/max+1, GET max-1/max/max+1, negative/bool/string/missing lengths, missing/malformed ETag and Body, ETag byte lengths 1023/1024/1025 plus blank/non-string/control/non-ASCII cases, HEAD/GET mismatch, exact `IfMatch`, unsupported `ContentEncoding`, absent encoding, chunk partitions, short and overlong bodies, mid-stream failure, close failure on success, close failure plus primary failure, spool rollover beyond 8 MiB, incremental SHA-256, cleanup, and safe exception surfaces. Credential/endpoint/provider-body sentinels must be absent from `str`, `repr`, `__cause__`, and `__context__`.
- [ ] Implement `CSVOptions`, `JSONOptions`, `AWSS3SourceConfig`, safe exception types, `_DownloadedObject`, and `_download_s3_object`. Do **not** define any `BaseSource` subclass, `name = "aws_s3"`, `source_file_hash`, or registration-visible class in this task.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/plugins/sources/test_aws_s3_source.py -k "Config or Download" -q
  uv run pytest tests/unit/plugins/test_discovery.py::TestDiscoverAllPlugins::test_discovery_finds_expected_plugin_counts -q
  ```

  Expected: both pass and discovery still reports six sources; no incomplete `aws_s3` plugin is selectable.
- [ ] Commit the unregistered foundation:

  ```bash
  git add src/elspeth/plugins/sources/aws_s3_source.py tests/unit/plugins/sources/test_aws_s3_source.py
  git commit -m "feat(plugins): add validated S3 source primitives"
  ```

---

### Task 3: Atomically add complete parsing, audit, registration, and conformance coverage

**Files:**

- Modify: `src/elspeth/plugins/sources/aws_s3_source.py`
- Modify: `tests/unit/plugins/sources/test_aws_s3_source.py`
- Create: `tests/property/plugins/sources/test_aws_s3_source_properties.py`
- Create: `tests/integration/plugins/sources/test_aws_s3_source_botocore.py`
- Create: `tests/integration/cli/test_aws_s3_endpoint_url_accepted.py`
- Create: `tests/integration/plugins/sources/test_aws_s3_source_live.py` (explicit slow real-AWS acceptance owned downstream by Plan 12)
- Create: `tests/golden/web/catalog/knob_schema/source__aws_s3.json`
- Modify: `src/elspeth/web/audit_readiness/boundary_expectations.py`
- Modify: `tests/unit/plugins/test_discovery.py`

`AWSS3Source(BaseSource)` lands complete in this task with `name = "aws_s3"`, `determinism = Determinism.IO_READ`, `plugin_version = "1.0.0"`, `config_model = AWSS3SourceConfig`, and an initial `source_file_hash = "sha256:0000000000000000"`. Its `get_agent_assistance(issue_code=None)` returns a non-empty summary plus short (at most 280-character) imperative hints covering the default credential chain, CLI-only endpoint override, format options, real pre-existing S3 objects, field mapping, and quarantine; other issue codes may return `None`, matching the Azure source. It lazily caches `build_s3_client` and owns `_active_download: _DownloadedObject | None` plus a closed flag. `load()` rejects reuse after close and a second concurrent load with static exceptions; once download succeeds it installs the active object before parsing. Its generator checks the closed flag before consuming or yielding each next row and, in `finally`, detaches only its own still-current object and closes it idempotently. `close()` marks the source closed, detaches and closes the active download first, then detaches and closes the cached client exactly once. A suspended generator resumed after external close terminates cleanly without touching the closed spool; a client-close exception cannot leak a provider message, repeat cleanup, or prevent spool closure.

Parsing contract:

- CSV reads the spool through `TextIOWrapper` and `csv.reader(strict=True)`. Call `readline(max_record_chars + 1)` through a counting decoded-line iterator, reject an overlong physical line immediately, accumulate decoded characters across quoted physical lines, and reset only after `csv.reader` yields one logical record. Preserve Azure header/headerless/columns, normalization, bounded raw preview, quarantine, and contract-lock behavior.
- JSONL uses the same size-limited decoded-line iterator, rejects a logical line over `max_record_chars` before `json.loads`, skips blank lines, and preserves Azure non-finite/surrogate/normalization/quarantine behavior. This keeps every supported codec path incremental without pretending decoded character count equals encoded byte count.
- JSON arrays wrap the spool in `TextIOWrapper(encoding=json_options.encoding, errors="strict")`, then in a `_BoundedJSONTokenReader` that returns at most 64 KiB per `read()` and lexically scans decoded chunks before releasing them to the parser. The scanner carries string/escape and primitive-token state across chunks and rejects any string, number, literal, or map-key token before its decoded length can exceed `max_record_chars`; this applies to selected rows and ignored siblings alike, so ijson never gets the opportunity to allocate an arbitrarily large scalar. The module then lazily imports ijson and uses `ijson.parse(..., use_float=False)`, building only one selected array item at a time with `_MAX_JSON_DEPTH = 64`. The event-driven item builder separately maintains a conservative aggregate materialized-character budget before retaining each map key, scalar, or container edge; it aborts before a many-small-fields item can exceed `max_record_chars`. After an item completes, compact `json.dumps(..., ensure_ascii=False, allow_nan=False)` supplies the exact final cap check. This text-stream seam preserves Azure's UTF-8/UTF-16/UTF-32 and other registered-codec behavior instead of handing non-UTF-8 bytes to ijson. Convert finite `Decimal` values recursively to ordinary floats before source validation while preserving arbitrary-size integers, matching stdlib JSON value shapes; reject a finite Decimal such as `1e9999` if conversion produces a non-finite float. Support either a root array or one exact top-level `json_options.data_key` whose value must be an array; compare `map_key` events directly so dots in literal keys are not interpreted as ijson path syntax. Reject trailing/multiple roots, non-array targets, non-finite values, excessive nesting, and over-budget items. Never call `json.load`, `json.loads` on the whole document, or `.read()` without a size.
- `_normalize_row_keys`, `_resolve_json_field_names`, `_validate_and_yield`, schema construction, sparse-field union, and `get_field_resolution` preserve the live Azure implementation's behavior with `bucket`/`key` replacing `container`/`blob_path` and `AWSS3RowSchema` as the generated class name.

Audit contract:

- One logical call uses `provider="aws_s3"`, `call_type=HTTP`, and `request_data={"operation": "read_object", "bucket": ..., "key": ...}`; it never includes endpoint/config/credentials.
- Success records `size_bytes` and SHA-256.
- Failure records only `type`, `bytes_read`, `max_object_bytes`, and an optional sanitized `cleanup_error_type`. No message/reason/body/URL is persisted.
- A shared `_record_download_call` helper applies Tier-1 propagation and redacted `AuditIntegrityError` wrapping identically to success and failure; non-Tier-1 recorder failures are converted and raised only after leaving their `except` block so the raw recorder object is unreachable from the public exception chain.
- Public failure remains the safe `S3SourceReadError`/`S3ObjectSizeLimitError`; do not reconstruct it with provider text and do not chain the provider exception.

- [ ] Port the complete Azure source conformance surface, not a subset: `TestAzureBlobSourceCSV`, `JSON`, `JSONL`, `SchemaValidation`, `FieldResolutionUnion`, `SparseFieldMapping`, and relevant config/audit cases. Rename only provider/location facts. Preserve format-specific empty behavior: CSV and JSON create a parse validation failure and quarantine unless configured to discard; JSONL yields zero rows. In every case the successful download audit contains SHA-256 of the empty bytes.
- [ ] Add S3-specific tests for all three parsers over a rolled-to-disk spool, maximum record boundaries, huge single CSV/JSONL records, deep/large JSON items, huge selected and ignored-sibling JSON scalars rejected by the lexical reader before ijson receives the over-limit chunk, a many-small-fields JSON item rejected during event construction, UTF-8/UTF-16/UTF-32 JSON streams, JSON float/scientific/large-integer parity including `1e9999` rejection after float conversion, lazy missing-ijson failure, generator early-close cleanup, second-load/reuse-after-close rejection, source close while iteration is suspended, clean termination when that generator resumes, client close/close failure/idempotence, exact audit success/error payloads, and failure-path audit-write failure becoming `AuditIntegrityError`. Pin token and aggregate limits at max-1/max/max+1 and instrument the reader/builder so the tests prove the over-limit scalar/item was not retained. Feed provider/endpoint/credential/body sentinels through head, get, read, close, and audit failures; assert absence from the raised error, `PhaseError.from_exception`, captured logs, and recorded calls.
- [ ] Port all three Azure Hypothesis suites and add chunk-partition/metadata properties across max-1/max/max+1. Properties must assert identical rows/contracts regardless of transport chunking and no named spool survives any failure.
- [ ] Add an offline real-SDK integration using actual botocore `Stubber` and `StreamingBody`. Set dummy standard AWS credential environment variables and disable EC2 metadata; call the real `build_s3_client` with no credential kwargs; stub HEAD/GET including `IfMatch`; exercise body lifecycle and the three formats without network.
- [ ] Add the CLI acceptance test using `load_settings_from_yaml_string` **and** `elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config`. A minimal source→CSV-sink YAML with `endpoint_url: http://localhost:4566` must construct `AWSS3Source` lazily without contacting AWS. This proves CLI retention, not merely untyped settings acceptance.
- [ ] Add `test_aws_s3_endpoint_url_is_not_a_secret_ref_field`: assert `"endpoint_url" not in allowed_secret_ref_fields("source", "aws_s3")`. This keeps `wire_secret_ref` from becoming an alternate non-null web endpoint authoring path after registration.
- [ ] Add `test_registered_aws_s3_source_is_endpoint_url_gated` with the complete Plan-08 fixture shape: `SourceSpec` supplies `plugin`, `on_success`, `options`, and `on_validation_failure`; `CompositionState` supplies that source plus `nodes=()`, `edges=()`, outputs, metadata, and version; `validate_pipeline` receives real `WebSettings` and a `MagicMock(spec=YamlGenerator)`. Assert the exact `aws_s3_endpoint_url_policy` failed check and `aws_s3_endpoint_url_not_allowed` error before settings generation. Add the explicit-null counterpart and assert it is accepted, matching Plan 08's `options.get("endpoint_url") is None` contract. Never skip/xfail these tests; without Plan 08 core the registration commit must remain red.
- [ ] Add source boundary/discovery/catalog coverage atomically with the class: insert `"aws_s3": Determinism.IO_READ` before `azure_blob`; change the source count `6 -> 7`; generate the golden through the test's `_stable_json` helper; run the corrected discovery node ID plus catalog/assistance gates.
- [ ] Add the real-AWS slow test. It requires `ELSPETH_TEST_S3_BUCKET`, uses no `endpoint_url` and no explicit credential kwargs, writes UUID-scoped CSV/JSON/JSONL fixtures with the default-chain client, loads them through `AWSS3Source`, verifies rows/hash/audit, and deletes every object in `finally`. It must fail rather than skip when explicitly selected without its bucket; ordinary suites deselect it via `slow`.
- [ ] After **every** source edit is complete, compute the final hash once:

  ```bash
  uv run python -c "from pathlib import Path; from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash; p = Path('src/elspeth/plugins/sources/aws_s3_source.py'); fix_source_file_hash(p, 'AWSS3Source', compute_source_file_hash(p))"
  ```

- [ ] Run the atomic registration slice:

  ```bash
  uv run pytest tests/unit/plugins/sources/test_aws_s3_source.py tests/property/plugins/sources/test_aws_s3_source_properties.py tests/integration/plugins/sources/test_aws_s3_source_botocore.py tests/integration/cli/test_aws_s3_endpoint_url_accepted.py tests/unit/web/catalog/test_knob_schema_golden.py tests/integration/web/test_catalog_discovery.py tests/unit/contracts/test_plugin_assistance_coverage.py tests/unit/web/audit_readiness/test_boundary_predicate_parity.py tests/unit/plugins/test_aws_s3_common.py tests/unit/plugins/test_discovery.py -q
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.options_metadata --root .
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.component_type,plugin_contract.plugin_hashes --root src/elspeth
  ```

  Expected: every command exits 0; the complete plugin becomes visible only in this green commit.
- [ ] Stage every Task-3 file explicitly and commit:

  ```bash
  git add src/elspeth/plugins/sources/aws_s3_source.py tests/unit/plugins/sources/test_aws_s3_source.py tests/property/plugins/sources/test_aws_s3_source_properties.py tests/integration/plugins/sources/test_aws_s3_source_botocore.py tests/integration/plugins/sources/test_aws_s3_source_live.py tests/integration/cli/test_aws_s3_endpoint_url_accepted.py tests/golden/web/catalog/knob_schema/source__aws_s3.json src/elspeth/web/audit_readiness/boundary_expectations.py tests/unit/plugins/test_discovery.py
  git commit -m "feat(plugins): add bounded audited aws_s3 source"
  ```

---

### Task 4: Run Plan 06 handoff gates

**Files:** Verify only the files touched by Tasks 1-3. No implementation edits are expected unless a gate exposes a scoped defect.

- [ ] Re-run dependency and focused behavior gates from the committed tree:

  ```bash
  uv lock --check
  uv sync --frozen --all-extras
  uv run python -c "import boto3, botocore, ijson, psycopg"
  uv run --isolated --no-dev --frozen python -c "from elspeth.plugins.infrastructure.discovery import discover_all_plugins; from elspeth.plugins.aws_s3_common import build_s3_client; d = discover_all_plugins(); assert any(c.name == 'aws_s3' for c in d['sources']);
try:
    build_s3_client(None, None)
except ImportError as e:
    assert 'aws' in str(e).lower()
else:
    raise AssertionError('base install unexpectedly supplied boto3')"
  uv run --isolated --no-dev --frozen --extra aws python -c "import boto3, botocore, ijson; from elspeth.plugins.infrastructure.discovery import discover_all_plugins; assert any(c.name == 'aws_s3' for c in discover_all_plugins()['sources'])"
  uv run pytest tests/unit/plugins/test_aws_s3_common.py tests/unit/plugins/sources/test_aws_s3_source.py tests/property/plugins/sources/test_aws_s3_source_properties.py tests/integration/plugins/sources/test_aws_s3_source_botocore.py tests/integration/cli/test_aws_s3_endpoint_url_accepted.py tests/unit/web/catalog/test_knob_schema_golden.py tests/integration/web/test_catalog_discovery.py tests/unit/contracts/test_plugin_assistance_coverage.py tests/unit/web/audit_readiness/test_boundary_predicate_parity.py tests/unit/plugins/test_discovery.py -q
  ```

- [ ] Run repository static and load-bearing plugin-contract gates:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.options_metadata --root .
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.component_type,plugin_contract.plugin_hashes --root src/elspeth
  git diff --check
  ```

  Expected: every command exits 0.
- [ ] If any Task-4 correction touches `aws_s3_source.py`, recompute `AWSS3Source.source_file_hash` with the Task-3 command, then restart **all** Task-4 gates from the first frozen-lock command before staging. A gate-driven source edit with an old hash is never committable.
- [ ] Because this is a new network/external-data and author-controlled CLI endpoint boundary, use the `wardline-gate` skill and run `wardline scan . --fail-on ERROR`. On an active defect, explain the fresh fingerprint, fix at ingress/validation, and rescan. Require exit 0 with no new waiver/baseline. The repository currently lacks Wardline trust decorators, so do not misreport an inert scan as decorator coverage; the parser/config/audit boundary tests above remain mandatory.
- [ ] Record exact downstream handoffs:

  - Plan 03B requires the registered source plus boto3/ijson imports in doctor integration.
  - Plan 07 imports `build_s3_client`, reuses `AWSS3*` naming, and must meet this plan's exact client retry/detach-before-close/redaction semantics before it can land after Plan 06. Plan 06 approval establishes that interface; it does **not** approve Plan 07's current skeleton-first task breakdown or public-error language, which remain for the dedicated Plan-07 review.
  - Plan 08 core remains the load-bearing web gate; Task 4 owns guided prompt parity after registration.
  - Plan 10's lean `aws` image must import boto3, botocore, and ijson.
  - Plan 12 explicitly runs `tests/integration/plugins/sources/test_aws_s3_source_live.py` against a disposable real bucket with the default credential chain and zero skips/failures.
- [ ] Create a final commit only if these gates required a scoped correction. Stage exact files; do not create an empty verification commit.

**Accepted limitation:** this plan does not transparently decompress objects and does not use S3 ETag as a content hash. ETag is used only as a conditional-read identity between HEAD and GET; the audit content hash is SHA-256 over the exact parsed bytes.
