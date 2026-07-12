# Plan 15A Text Sink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Also use
> superpowers:test-driven-development and wardline-gate. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Add a strict, auditable, resumable `sink:text` so CSV, JSON, and text
are all complete source/sink pairs before the universal web plugin policy makes
them mandatory core.

**Architecture:** Implement a line-oriented local-file sink that writes one
configured string field per row with canonical LF separators. Follow the CSV
sink's staged-write, collision, append, rollback, hashing, and preflight
patterns; reject rather than coerce non-strings or embedded record separators.
Dynamic discovery registers the new class without a manual registry edit.

**Tech Stack:** Python 3.12+, Pydantic v2, ELSPETH sink contracts,
binary same-directory staging, SHA-256 artifacts, pytest, Hypothesis, Ruff, mypy, ELSPETH
plugin-contract lints.

**Depends on:** shared signed-tier/trust-boundary/Wardline baseline
`elspeth-8166b310e7`.

**Blocks:** Plan 15B universal web plugin policy.

---

### Task 0: Claim the exact slice and establish a clean baseline

**Files:**

- Read: `docs/superpowers/specs/2026-07-12-universal-web-plugin-policy-design.md`
- Read: `src/elspeth/plugins/sources/text_source.py`
- Read: `src/elspeth/plugins/sinks/csv_sink.py`
- Read: `src/elspeth/plugins/sinks/json_sink.py`
- Read: `src/elspeth/plugins/infrastructure/config_base.py`
- Read: `src/elspeth/contracts/sink.py`

- [ ] **Step 1: Create an isolated implementation worktree**

```bash
git status --short
git check-ignore -q .worktrees
BASE_SHA="$(git rev-parse release/0.7.1)"
git worktree add .worktrees/aws-ecs-15a-text-sink -b feat/aws-ecs-15a-text-sink "$BASE_SHA"
cd .worktrees/aws-ecs-15a-text-sink
```

Expected: `.worktrees/` is ignored, the original worktree remains untouched,
and the new worktree starts at the then-current integrated `release/0.7.1` tip.

- [ ] **Step 2: Atomically start the Filigree step**

```bash
filigree start-work elspeth-130dc48252 --assignee codex --actor codex
```

Expected: the step enters `in_progress`. Do not use claim-plus-status.

- [ ] **Step 3: Record the baseline and run the nearest existing tests**

```bash
git rev-parse HEAD
uv run pytest \
  tests/unit/plugins/sources/test_text_source.py \
  tests/unit/plugins/sinks/test_csv_sink.py \
  tests/unit/contracts/test_sink_capabilities.py \
  tests/unit/plugins/test_discovery.py -q
```

Expected: all selected tests pass before edits.

Plan 15A may overlap Plan 08A/07 test files but not their production ownership.
Before merge, rebase on their current integrated tip when applicable and rerun
the combined discovery, boundary-parity, composer-tool, and golden suites; do
not merge independently generated stale goldens.

---

### Task 1: Define the strict text-sink configuration contract

**Files:**

- Create: `src/elspeth/plugins/sinks/text_sink.py`
- Modify: `src/elspeth/plugins/infrastructure/config_base.py`
- Create: `tests/unit/plugins/sinks/test_text_sink.py`
- Modify: `tests/unit/plugins/test_config_base.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests that pin `path`, repository-required `schema`, `field`, encoding,
write mode, and the closed encoding set:

```python
def _config(path: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "path": str(path),
        "field": "line_text",
        "encoding": "utf-8",
        "mode": "write",
        "schema": {"mode": "observed"},
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("encoding", ["utf-8", "ascii", "latin-1", "cp1252"])
def test_text_sink_accepts_only_supported_ascii_compatible_encodings(encoding: str) -> None:
    parsed = TextSinkConfig.from_dict(_config(Path("out.txt"), encoding=encoding), plugin_name="text")
    assert parsed.encoding == encoding
```

Also test invalid/non-identifier `field`, every encoding outside the closed set
(including valid-but-stateful codecs such as UTF-16), unknown codecs, and the
absence of `headers` from the schema. Strict row/diversion tests are deliberately
deferred until Task 2, where the real write lifecycle and orchestrator failure
policy injection exist.

- [ ] **Step 2: Run the tests and verify the expected failure**

```bash
uv run pytest tests/unit/plugins/sinks/test_text_sink.py -q
```

Expected: collection fails because `TextSink` and `TextSinkConfig` do not exist.

- [ ] **Step 3: Extract a header-free local-file sink configuration base**

Do not inherit text from `SinkPathConfig`: that would publish a meaningless
`headers` knob. Move only `collision_policy` into a new base and leave current
CSV/JSON behavior unchanged:

```python
class LocalFileSinkConfig(PathConfig):
    """Base for local file sinks with schema, path, and collision policy."""

    _plugin_component_type: ClassVar[str | None] = "sink"
    collision_policy: OutputCollisionPolicy | None = Field(
        default=None,
        description="Local output collision policy.",
    )


class SinkPathConfig(LocalFileSinkConfig):
    """Local file sink config that additionally supports display headers."""
```

Delete the duplicate `collision_policy` declaration from `SinkPathConfig`.
Add tests proving existing CSV/JSON config schemas retain collision policy and
`TextSinkConfig` has no `headers` property.

- [ ] **Step 4: Implement the configuration and immutable plugin metadata**

Create `TextSinkConfig` and the initial `TextSink` skeleton:

```python
class TextSinkConfig(LocalFileSinkConfig):
    field: str = Field(description="String field written as one line per row.")
    encoding: Literal["utf-8", "ascii", "latin-1", "cp1252"] = Field(default="utf-8")
    mode: Literal["write", "append"] = Field(default="write")

    @field_validator("field")
    @classmethod
    def _validate_field(cls, value: str) -> str:
        if not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError(f"field {value!r} must be a non-keyword Python identifier")
        return value

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        try:
            canonical = codecs.lookup(value).name
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {value!r}") from exc
        if canonical not in {"utf-8", "ascii", "iso8859-1", "cp1252"}:
            raise ValueError("encoding must be one of utf-8, ascii, latin-1, or cp1252")
        return value

    @model_validator(mode="after")
    def _validate_collision_mode(self) -> "TextSinkConfig":
        validate_output_collision_policy_mode(
            plugin_name="TextSink",
            mode=self.mode,
            collision_policy=self.collision_policy,
        )
        return self


class TextSink(BaseSink):
    name = "text"
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    # Placeholder is required so the hash helper can normalize this field.
    # The first green lifecycle commit replaces it with the computed value.
    source_file_hash: str | None = "sha256:0000000000000000"
    config_model = TextSinkConfig
    supports_resume = True
```

In `__init__`, derive the requested/resolved path, configured field and
encoding, collision policy, strict sink schema (`allow_coercion=False`), and
`declared_required_fields = schema_required_fields | {field}`. Initialize the
file handle and incremental hasher to `None`. This skeleton remains abstract
and unregistered until Task 2 implements `write`, `flush`, and `close`; do not
run repository-wide discovery/hash gates or commit this intermediate state.

- [ ] **Step 5: Run the focused configuration tests**

```bash
uv run pytest \
  tests/unit/plugins/sinks/test_text_sink.py \
  tests/unit/plugins/test_config_base.py \
  -k "config or encoding or local_file" -q
```

Expected: the selected config tests pass. Continue directly to Task 2 without
committing: the first green commit is the complete concrete lifecycle and a
valid source hash.

---

### Task 2: Add collision, append/resume, rollback, and artifact integrity

**Files:**

- Modify: `src/elspeth/plugins/sinks/text_sink.py`
- Modify: `tests/unit/plugins/sinks/test_text_sink.py`
- Create: `tests/property/plugins/sinks/test_text_sink_properties.py`
- Modify: `tests/integration/config/test_cli_resume_sink_capability.py`
- Modify: `tests/integration/config/test_cli_resume_schema_validation.py`

- [ ] **Step 1: Write failing write/append/rollback tests**

Cover all of these exact cases:

```python
def test_text_sink_writes_canonical_lf_and_whole_artifact_hash(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path)))

    result = sink.write([{"line_text": "alpha"}, {"line_text": ""}, {"line_text": "omega"}], _sink_context())
    sink.close()

    expected = b"alpha\n\nomega\n"
    assert path.read_bytes() == expected
    assert result.artifact.content_hash == hashlib.sha256(expected).hexdigest()
    assert result.artifact.size_bytes == len(expected)


def test_append_failure_rolls_back_to_original_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path, mode="append", collision_policy="append_or_create")))
    sink._artifact_stat = _raise_after_write

    with pytest.raises(OSError):
        sink.write([{"line_text": "new"}], _sink_context())

    assert path.read_bytes() == b"existing\n"
```

Every test expecting a diversion must wrap the sink with
`inject_write_failure(...)`. Add a separate fail-closed test, without injection,
that proves `_divert_row()` raises `FrameworkBugError`. Add tests for non-string
values, missing fields, embedded CR/LF, sanitized encoding diversion reasons
that contain neither the row value nor `UnicodeEncodeError.__str__`, deferred
`fail_if_exists`, `auto_increment`, `append_or_create`,
preflight no-mutation, multi-batch hashing, existing append bytes included in
the hash, missing terminal LF rejected, invalid existing encoding rejected,
CRLF or interior CR rejected, resume switching to append, empty input, close
idempotence, and rollback when flush/hash/stat fails. Use sink-local fault seams;
do not monkeypatch `Path.stat` globally.

Pin empty-batch semantics separately: no target is created or mutated. If a
target already exists, return its current exact byte/hash descriptor. If no
target exists, return the contract's virtual zero-row descriptor (requested
path, zero size, empty SHA-256) without creating a file, matching the existing
CSV zero-row precedent; downstream verification must treat that descriptor as
no-primary-write rather than attempt a filesystem read. The same rule applies
to a non-empty batch in which all rows divert. Tests cover both absent and
pre-existing targets and document this virtual-artifact consequence.

- [ ] **Step 2: Run the lifecycle tests and verify they fail**

```bash
uv run pytest tests/unit/plugins/sinks/test_text_sink.py -k "write or append or collision or rollback or resume or preflight" -q
```

Expected: failures identify missing lifecycle methods and artifact handling.

- [ ] **Step 3: Implement byte staging, lazy target claim, and target validation**

Stage accepted rows as bytes, never through a persistent text encoder:

```python
def _stage_rows(self, rows: list[dict[str, Any]]) -> bytes:
    staged = io.BytesIO()
    for row_index, row in enumerate(rows):
        value = row.get(self._field, _MISSING)
        if type(value) is not str:
            self._divert_row(row, row_index=row_index, reason=f"Text field {self._field!r} must be a string")
            continue
        if "\r" in value or "\n" in value:
            self._divert_row(row, row_index=row_index, reason="Text values cannot contain CR or LF record separators")
            continue
        try:
            staged.write((value + "\n").encode(self._encoding))
        except UnicodeEncodeError:
            self._divert_row(row, row_index=row_index, reason=f"Text value is not representable in configured codec {self._encoding}")
    return staged.getvalue()
```

Do not call `str(value)`. An empty string stages exactly the bytes for LF in the
configured ASCII-compatible codec.

Use the shared collision resolver only at the first real write, never during
preflight construction. In append mode, validate an existing non-empty target
before mutation:

This lazy first-write claim is an intentional TextSink-only safety improvement
for this release. It does not silently change existing CSV/JSON constructor
timing; the difference is documented and pinned until a separately reviewed
shared-sink refactor owns parity.

The claim itself must be race-safe. For `fail_if_exists`, reserve the requested
path with `os.open(..., O_CREAT|O_EXCL|O_WRONLY)`; for `auto_increment`, loop
candidate names and reserve the winner the same way. A losing process retries
or fails without replacing the winner. Track ownership of the zero-byte
reservation and remove only that owned placeholder on pre-commit failure.
`collision_policy=None` deliberately retains overwrite semantics. Append uses
an atomic append/create open after target validation.

```python
def configure_for_resume(self) -> None:
    self._mode = "append"
    self._collision_policy = "append_or_create"


def validate_output_target(self) -> OutputValidationResult:
    if not self._path.exists() or self._path.stat().st_size == 0:
        return OutputValidationResult.success()
    try:
        saw_content = False
        final_character = ""
        with open(self._path, encoding=self._encoding, newline="") as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), ""):
                if chunk:
                    saw_content = True
                    if "\r" in chunk:
                        return OutputValidationResult.failure(message="Existing text output contains non-canonical CR separators")
                    final_character = chunk[-1]
    except UnicodeError:
        return OutputValidationResult.failure(message=f"Existing text output is not valid {self._encoding}")
    if saw_content and final_character != "\n":
        return OutputValidationResult.failure(message="Existing text output does not end at an LF record boundary")
    return OutputValidationResult.success(target_fields=[self._field])
```

Open append targets in binary mode and capture byte offsets with
`os.fstat(handle.fileno()).st_size` after flush. Initialize the hasher from all
existing bytes in append mode. The closed encoding set is intentionally
ASCII-compatible and stateless for LF, so byte offsets, rollback, hashing, and
record boundaries are one coherent contract.

- [ ] **Step 4: Implement atomic write-mode replacement and append rollback**

For write mode, never open the target with `"w"`. On the first batch, write
only staged bytes to a same-directory temporary file (the configured overwrite
semantics intentionally exclude pre-run target bytes). On every later batch
owned by the same sink instance, copy the complete current target bytes into a
new same-directory temp and append the newly staged bytes. Flush and fsync the
temp, hash/stat its complete bytes, then atomically `os.replace()` it into the
resolved target and fsync the parent directory. This preserves all earlier
batches while a pre-existing pre-run target remains intact through every
pre-commit failure. If failure occurs after replace, treat replacement as the
commit point and report the post-commit durability/audit failure without
pretending the old target can be recovered; tests pin this boundary.

Add a two-batch write-mode regression proving the second temp starts from the
first committed artifact, plus multiprocessing/race tests proving two
`fail_if_exists` writers cannot both win and two `auto_increment` writers never
replace one another's reservation.

For append mode, capture the flushed binary byte offset, append staged bytes,
flush/fsync/hash/stat, and on any pre-commit failure seek/truncate to the byte
offset, flush/fsync, close the binary handle, rebuild the hasher from preserved
bytes, and reopen lazily on the next write. If rollback itself fails, close the
handle and raise `RuntimeError` naming only the byte offset. Test failed append
followed by retry. No stateful `TextIOWrapper` or opaque text `tell()` cookie is
permitted.

Return:

```python
return SinkWriteResult(
    artifact=ArtifactDescriptor.for_file(
        path=str(self._path),
        content_hash=self._hasher.hexdigest(),
        size_bytes=self._path.stat().st_size,
    ),
    diversions=self._get_diversions(),
)
```

- [ ] **Step 5: Add property tests for byte and rejection invariants**

```python
@given(st.lists(st.text(alphabet=st.characters(blacklist_characters="\r\n")), min_size=1, max_size=50))
def test_text_sink_bytes_and_hash_match_all_lines(tmp_path: Path, values: list[str]) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path)))
    result = sink.write([{"line_text": value} for value in values], _sink_context())
    sink.close()

    expected = "".join(f"{value}\n" for value in values).encode("utf-8")
    assert path.read_bytes() == expected
    assert result.artifact.content_hash == hashlib.sha256(expected).hexdigest()
```

Add a property that generated CR/LF-bearing strings and non-string primitives
never appear in output bytes and always create a diversion. Add a distinct
empty-list property proving no filesystem mutation and the zero descriptor.

- [ ] **Step 6: Extend CLI resume capability and target-validation tests**

Assert `sink:text` reports `supports_resume=True`, switches to append through
`configure_for_resume`, accepts a decodable LF-terminated target, and rejects
an undecodable or unterminated target before writing.

- [ ] **Step 7: Run the complete sink lifecycle suite**

```bash
uv run pytest \
  tests/unit/plugins/sinks/test_text_sink.py \
  tests/unit/plugins/test_config_base.py \
  tests/property/plugins/sinks/test_text_sink_properties.py \
  tests/integration/config/test_cli_resume_sink_capability.py \
  tests/integration/config/test_cli_resume_schema_validation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit the lifecycle slice**

Before committing, compute and verify the plugin source hash so the discovery
and plugin-hash gates can safely import the newly concrete class:

```bash
uv run python - <<'PY'
from pathlib import Path
from scripts.cicd.plugin_hash import compute_source_file_hash, extract_plugin_attributes, fix_source_file_hash

path = Path("src/elspeth/plugins/sinks/text_sink.py")
fix_source_file_hash(path, "TextSink", compute_source_file_hash(path))
declared = next(item.source_file_hash for item in extract_plugin_attributes(path) if item.class_name == "TextSink")
computed = compute_source_file_hash(path)
assert declared == computed, (declared, computed)
PY
```

```bash
git add \
  src/elspeth/plugins/sinks/text_sink.py \
  src/elspeth/plugins/infrastructure/config_base.py \
  tests/unit/plugins/sinks/test_text_sink.py \
  tests/unit/plugins/test_config_base.py \
  tests/property/plugins/sinks/test_text_sink_properties.py \
  tests/integration/config/test_cli_resume_sink_capability.py \
  tests/integration/config/test_cli_resume_schema_validation.py
git commit -m "feat(sinks): make text output resumable and auditable"
```

---

### Task 3: Register text as a first-class sink and prove round-trip behavior

**Files:**

- Modify: `src/elspeth/contracts/sink.py`
- Modify: `src/elspeth/web/audit_readiness/boundary_expectations.py`
- Modify: `src/elspeth/web/composer/tools/_common.py`
- Modify: `tests/unit/plugins/test_discovery.py`
- Modify: `tests/unit/plugins/test_builtin_plugin_metadata.py`
- Modify: `tests/unit/contracts/test_sink_capabilities.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/web/composer/test_tools.py`
- Modify: `tests/unit/web/audit_readiness/test_boundary_predicate_parity.py`
- Create: `tests/e2e/pipelines/test_text_to_text.py`
- Create: `tests/golden/web/catalog/knob_schema/sink__text.json`
- Modify: `docs/reference/configuration.md`

- [ ] **Step 1: Write failing discovery, capability, boundary, and E2E tests**

Pin the new sink's semantics:

```python
def test_discover_all_sinks_includes_text() -> None:
    names = {plugin.name for plugin in discover_all_plugins()["sinks"]}
    assert "text" in names


def test_text_sink_is_file_sink_but_not_lossless_failure_sink() -> None:
    capability = SINK_CAPABILITIES_BY_PLUGIN["text"]
    assert capability.requires_path_option is True
    assert capability.default_file_extension == "txt"
    assert capability.eligible_as_failsink is False
    assert capability.local_recovery_file is False
```

Add an exact composer repair regression beside the JSON repair test:

```python
def test_set_pipeline_missing_text_output_options_returns_runnable_repair_hint() -> None:
    result = execute_tool(
        "set_pipeline",
        _pipeline_with_text_output_without_options(),
        _empty_state(),
        _catalog_with_text_sink(),
    )

    assert result.success is False
    assert '\"plugin\": \"text\"' in result.data["error"]
    assert '\"field\": \"line_text\"' in result.data["error"]
    assert '\"mode\": \"write\"' in result.data["error"]
```

Also pin `TextSink.plugin_version == "1.0.0"` and direct catalog
summary/schema discovery of `sink:text`.

The E2E test runs `TextSource -> passthrough -> TextSink` through the real
orchestrator with `strip_whitespace=False` and `skip_blank_lines=False`, then
asserts exact output bytes and terminal Landscape outcomes.

- [ ] **Step 2: Run the cross-cutting tests and verify they fail**

```bash
uv run pytest \
  tests/unit/plugins/test_discovery.py \
  tests/unit/plugins/test_builtin_plugin_metadata.py \
  tests/unit/contracts/test_sink_capabilities.py \
  tests/unit/web/catalog/test_service.py \
  tests/unit/web/composer/test_tools.py \
  tests/unit/web/audit_readiness/test_boundary_predicate_parity.py \
  tests/e2e/pipelines/test_text_to_text.py -q
```

Expected: failures name the absent sink capability/boundary rows and E2E
fixture.

- [ ] **Step 3: Add the sink capability and audit-boundary entries**

Add:

```python
"text": SinkCapabilities(
    requires_path_option=True,
    default_file_extension="txt",
    eligible_as_failsink=False,
    local_recovery_file=False,
)
```

to `SINK_CAPABILITIES_BY_PLUGIN`, and add:

```python
"text": Determinism.IO_WRITE,
```

to `EXPECTED_SINK_DETERMINISMS`. Update exact derived FILE_SINK messages from
`csv/json` to `csv/json/text`; failure-sink messages remain `csv/json`.

Update `_missing_output_options_repair_error` with a text-specific branch
before the generic file-sink branch. Its emitted runnable shape must include
`path`, `schema`, `field: "line_text"`, `mode: "write"`, and
`collision_policy: "auto_increment"`; the prose tells the composer to replace
`line_text` with the actual selected string field.

- [ ] **Step 4: Add plugin assistance and generate the catalog golden**

Implement `get_agent_assistance` with exact guidance: one configured field,
string-only values, embedded newline rejection, and append/resume behavior.
Generate rather than hand-edit the golden using the live catalog serialization
helper used by `test_knob_schema_golden.py`, then inspect the resulting
`sink__text.json` for `path`, `schema`, `field`, `encoding`, `mode`, and
collision-policy knobs.

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.service import CatalogServiceImpl

info = CatalogServiceImpl(get_shared_plugin_manager()).get_schema("sink", "text")
payload = {"plugin_kind": "sink", "plugin_name": "text", "knob_schema": info.knob_schema}
Path("tests/golden/web/catalog/knob_schema/sink__text.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
uv run pytest tests/unit/web/catalog/test_knob_schema_golden.py -q
git diff -- tests/golden/web/catalog/knob_schema/sink__text.json
```

Expected: exactly one new text-sink golden plus any mechanically expected
catalog inventory change; no unrelated golden churn.

- [ ] **Step 5: Finalize the plugin source hash**

```bash
uv run python - <<'PY'
from pathlib import Path
from scripts.cicd.plugin_hash import compute_source_file_hash, extract_plugin_attributes, fix_source_file_hash

path = Path("src/elspeth/plugins/sinks/text_sink.py")
fix_source_file_hash(path, "TextSink", compute_source_file_hash(path))
declared = next(item.source_file_hash for item in extract_plugin_attributes(path) if item.class_name == "TextSink")
computed = compute_source_file_hash(path)
assert declared == computed, (declared, computed)
PY
```

Expected: `TextSink.source_file_hash` becomes `sha256:` plus 16 lowercase hex
characters and recomputation is stable.

- [ ] **Step 6: Document the operator contract**

Update the sink inventory and local collision-policy sections with `text`.
Document the required `field`, schema, exact-string rule, CR/LF rejection,
canonical LF, encoding, append/resume, and why text is not an eligible generic
failure sink.

- [ ] **Step 7: Run the full focused acceptance set**

```bash
uv run pytest \
  tests/unit/plugins/sinks/test_text_sink.py \
  tests/property/plugins/sinks/test_text_sink_properties.py \
  tests/e2e/pipelines/test_text_to_text.py \
  tests/integration/config/test_cli_resume_sink_capability.py \
  tests/integration/config/test_cli_resume_schema_validation.py \
  tests/unit/plugins/test_discovery.py \
  tests/unit/plugins/test_builtin_plugin_metadata.py \
  tests/unit/contracts/test_sink_capabilities.py \
  tests/unit/contracts/test_plugin_assistance_coverage.py \
  tests/integration/web/test_catalog_discovery.py \
  tests/unit/web/catalog/test_service.py \
  tests/unit/web/composer/test_tools.py \
  tests/unit/web/catalog/test_knob_schema_golden.py \
  tests/unit/web/audit_readiness/test_boundary_predicate_parity.py -q
```

Expected: all selected tests pass with no skips.

- [ ] **Step 8: Run static and plugin-contract gates**

```bash
uv run ruff check \
  src/elspeth/plugins/sinks/text_sink.py \
  tests/unit/plugins/sinks/test_text_sink.py \
  tests/property/plugins/sinks/test_text_sink_properties.py \
  tests/e2e/pipelines/test_text_to_text.py
uv run ruff format --check \
  src/elspeth/plugins/sinks/text_sink.py \
  tests/unit/plugins/sinks/test_text_sink.py \
  tests/property/plugins/sinks/test_text_sink_properties.py \
  tests/e2e/pipelines/test_text_to_text.py
uv run mypy src/elspeth/plugins/sinks/text_sink.py
PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check \
  --rules plugin_contract.component_type,plugin_contract.plugin_hashes \
  --root src/elspeth
wardline scan . --fail-on ERROR
```

Expected: every command exits zero.

- [ ] **Step 9: Commit and close the slice**

```bash
git add \
  src/elspeth/plugins/sinks/text_sink.py \
  src/elspeth/plugins/infrastructure/config_base.py \
  src/elspeth/contracts/sink.py \
  src/elspeth/web/audit_readiness/boundary_expectations.py \
  src/elspeth/web/composer/tools/_common.py \
  tests/unit/plugins/test_discovery.py \
  tests/unit/plugins/test_builtin_plugin_metadata.py \
  tests/unit/plugins/test_config_base.py \
  tests/unit/contracts/test_sink_capabilities.py \
  tests/unit/web/catalog/test_service.py \
  tests/unit/web/composer/test_tools.py \
  tests/unit/web/audit_readiness/test_boundary_predicate_parity.py \
  tests/e2e/pipelines/test_text_to_text.py \
  tests/golden/web/catalog/knob_schema/sink__text.json \
  docs/reference/configuration.md
git commit -m "feat(sinks): register text output as a core format"
```

Expected: the worker reports its implementation commit and exact gate evidence
but does not close Filigree from the feature worktree. The integration
coordinator rebases/merges into `release/0.7.1`, reruns the handoff gates, then
closes `elspeth-130dc48252` with the integrated `release/0.7.1@<sha>` anchor.
Only then may Plan 15B treat `sink:text` as an integrated startup-core invariant.
