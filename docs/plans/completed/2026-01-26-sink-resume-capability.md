# Sink Resume Capability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add polymorphic resume capability to sink plugins so the CLI doesn't need to know sink-specific configuration details.

**Architecture:** Each sink declares whether it supports resume and implements `configure_for_resume()` to self-configure. The CLI queries this capability instead of hardcoding sink-type-specific logic. This respects the plugin boundary and scales to future sink types.

**Tech Stack:** Python 3.12, Pydantic v2, Typer CLI, ABC base classes

**Bug Reference:** BUG-CLI-02 (P0) - Resume Forces mode=append For All Sinks

---

## Task 1: Add Resume Capability to SinkProtocol

**Files:**
- Modify: `src/elspeth/plugins/protocols.py:375-470`
- Test: `tests/contracts/sink_contracts/test_sink_protocol.py`

**Step 1: Write the failing test for supports_resume property**

Add to `tests/contracts/sink_contracts/test_sink_protocol.py`:

```python
def test_sink_protocol_requires_supports_resume():
    """SinkProtocol must declare supports_resume attribute."""
    from elspeth.plugins.protocols import SinkProtocol

    # Verify protocol has supports_resume in annotations
    assert hasattr(SinkProtocol, '__protocol_attrs__') or 'supports_resume' in dir(SinkProtocol)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/contracts/sink_contracts/test_sink_protocol.py::test_sink_protocol_requires_supports_resume -v`
Expected: FAIL (supports_resume not in protocol)

**Step 3: Add supports_resume to SinkProtocol**

In `src/elspeth/plugins/protocols.py`, add after line 412 (after `plugin_version: str`):

```python
    # Resume capability (Phase 5 - Checkpoint/Resume)
    supports_resume: bool  # Can this sink append to existing output on resume?
```

**Step 4: Add configure_for_resume method to SinkProtocol**

In `src/elspeth/plugins/protocols.py`, add after `on_complete` method (after line 468):

```python
    def configure_for_resume(self) -> None:
        """Configure sink for resume mode (append instead of truncate).

        Called by engine when resuming a run. Sinks that support resume
        (supports_resume=True) MUST implement this to switch to append mode.

        Sinks that don't support resume (supports_resume=False) will never
        have this called - the CLI will reject resume before execution.

        Raises:
            NotImplementedError: If sink cannot be resumed despite claiming support.
        """
        ...
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/contracts/sink_contracts/test_sink_protocol.py::test_sink_protocol_requires_supports_resume -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/plugins/protocols.py tests/contracts/sink_contracts/test_sink_protocol.py
git commit -m "feat(protocol): add supports_resume and configure_for_resume to SinkProtocol

Adds resume capability declaration to sink protocol:
- supports_resume: bool - declares if sink can append on resume
- configure_for_resume() - self-configure for append mode

Part of BUG-CLI-02 fix.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Resume Capability to BaseSink

**Files:**
- Modify: `src/elspeth/plugins/base.py:201-281`
- Test: `tests/plugins/test_base_sink.py` (create if needed)

**Step 1: Write the failing test for BaseSink.supports_resume**

Create `tests/plugins/test_base_sink.py`:

```python
"""Tests for BaseSink resume capability."""

import pytest
from elspeth.plugins.base import BaseSink


def test_base_sink_supports_resume_default_false():
    """BaseSink.supports_resume should default to False."""
    assert BaseSink.supports_resume is False


def test_base_sink_configure_for_resume_raises_not_implemented():
    """BaseSink.configure_for_resume should raise NotImplementedError by default."""

    class TestSink(BaseSink):
        name = "test"
        input_schema = None  # type: ignore

        def write(self, rows, ctx):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    sink = TestSink({})

    with pytest.raises(NotImplementedError) as exc_info:
        sink.configure_for_resume()

    assert "TestSink" in str(exc_info.value)
    assert "resume" in str(exc_info.value).lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_base_sink.py -v`
Expected: FAIL (supports_resume not defined, configure_for_resume not defined)

**Step 3: Add supports_resume and configure_for_resume to BaseSink**

In `src/elspeth/plugins/base.py`, add after line 235 (after `plugin_version: str = "0.0.0"`):

```python
    # Resume capability (Phase 5 - Checkpoint/Resume)
    # Default: sinks don't support resume. Override in subclasses that can append.
    supports_resume: bool = False

    def configure_for_resume(self) -> None:
        """Configure sink for resume mode (append instead of truncate).

        Called by engine when resuming a run. Override in sinks that support
        resume to switch from truncate mode to append mode.

        Default implementation raises NotImplementedError. Subclasses that
        set supports_resume=True MUST override this method.

        Raises:
            NotImplementedError: If sink cannot be resumed.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support resume. "
            f"To make this sink resumable, set supports_resume=True and "
            f"implement configure_for_resume()."
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/test_base_sink.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/base.py tests/plugins/test_base_sink.py
git commit -m "feat(base): add resume capability to BaseSink

- supports_resume: bool = False (default: no resume support)
- configure_for_resume(): raises NotImplementedError by default

Sinks that support resume must:
1. Set supports_resume = True
2. Implement configure_for_resume() to switch to append mode

Part of BUG-CLI-02 fix.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement Resume Capability in CSVSink

**Files:**
- Modify: `src/elspeth/plugins/sinks/csv_sink.py:35-95`
- Test: `tests/plugins/sinks/test_csv_sink_resume.py` (create)

**Step 1: Write the failing test for CSVSink resume**

Create `tests/plugins/sinks/test_csv_sink_resume.py`:

```python
"""Tests for CSVSink resume capability."""

import pytest
from elspeth.plugins.sinks.csv_sink import CSVSink


def test_csv_sink_supports_resume():
    """CSVSink should declare supports_resume=True."""
    assert CSVSink.supports_resume is True


def test_csv_sink_configure_for_resume_sets_append_mode():
    """CSVSink.configure_for_resume should set mode to append."""
    sink = CSVSink({
        "path": "/tmp/test.csv",
        "schema": {"fields": "dynamic"},
        "mode": "write",  # Explicit write mode
    })

    assert sink._mode == "write"

    sink.configure_for_resume()

    assert sink._mode == "append"


def test_csv_sink_configure_for_resume_idempotent():
    """Calling configure_for_resume multiple times should be safe."""
    sink = CSVSink({
        "path": "/tmp/test.csv",
        "schema": {"fields": "dynamic"},
    })

    sink.configure_for_resume()
    sink.configure_for_resume()  # Second call

    assert sink._mode == "append"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/sinks/test_csv_sink_resume.py -v`
Expected: FAIL (supports_resume not True, configure_for_resume not implemented)

**Step 3: Add resume capability to CSVSink**

In `src/elspeth/plugins/sinks/csv_sink.py`, add after line 64 (after `plugin_version = "1.0.0"`):

```python
    # Resume capability: CSV can append to existing files
    supports_resume: bool = True

    def configure_for_resume(self) -> None:
        """Configure CSV sink for resume mode.

        Switches from truncate mode to append mode so resume operations
        add to existing output instead of overwriting.
        """
        self._mode = "append"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/sinks/test_csv_sink_resume.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/sinks/csv_sink.py tests/plugins/sinks/test_csv_sink_resume.py
git commit -m "feat(csv-sink): implement resume capability

- supports_resume = True
- configure_for_resume() sets _mode = 'append'

CSVSink can now be resumed by calling configure_for_resume().

Part of BUG-CLI-02 fix.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement Resume Capability in DatabaseSink

**Files:**
- Modify: `src/elspeth/plugins/sinks/database_sink.py:48-111`
- Test: `tests/plugins/sinks/test_database_sink_resume.py` (create)

**Step 1: Write the failing test for DatabaseSink resume**

Create `tests/plugins/sinks/test_database_sink_resume.py`:

```python
"""Tests for DatabaseSink resume capability."""

import os
import pytest
from elspeth.plugins.sinks.database_sink import DatabaseSink


@pytest.fixture(autouse=True)
def allow_raw_secrets():
    """Allow raw secrets for testing."""
    os.environ["ELSPETH_ALLOW_RAW_SECRETS"] = "true"
    yield
    os.environ.pop("ELSPETH_ALLOW_RAW_SECRETS", None)


def test_database_sink_supports_resume():
    """DatabaseSink should declare supports_resume=True."""
    assert DatabaseSink.supports_resume is True


def test_database_sink_configure_for_resume_sets_append():
    """DatabaseSink.configure_for_resume should set if_exists to append."""
    sink = DatabaseSink({
        "url": "sqlite:///:memory:",
        "table": "test_table",
        "schema": {"fields": "dynamic"},
        "if_exists": "replace",  # Explicit replace mode
    })

    assert sink._if_exists == "replace"

    sink.configure_for_resume()

    assert sink._if_exists == "append"


def test_database_sink_configure_for_resume_idempotent():
    """Calling configure_for_resume multiple times should be safe."""
    sink = DatabaseSink({
        "url": "sqlite:///:memory:",
        "table": "test_table",
        "schema": {"fields": "dynamic"},
    })

    sink.configure_for_resume()
    sink.configure_for_resume()  # Second call

    assert sink._if_exists == "append"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/sinks/test_database_sink_resume.py -v`
Expected: FAIL (supports_resume not True, configure_for_resume not implemented)

**Step 3: Add resume capability to DatabaseSink**

In `src/elspeth/plugins/sinks/database_sink.py`, add after line 75 (after `plugin_version = "1.0.0"`):

```python
    # Resume capability: Database can append to existing tables
    supports_resume: bool = True

    def configure_for_resume(self) -> None:
        """Configure database sink for resume mode.

        Switches from replace mode to append mode so resume operations
        add to existing table instead of dropping and recreating.
        """
        self._if_exists = "append"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/sinks/test_database_sink_resume.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/sinks/database_sink.py tests/plugins/sinks/test_database_sink_resume.py
git commit -m "feat(database-sink): implement resume capability

- supports_resume = True
- configure_for_resume() sets _if_exists = 'append'

DatabaseSink can now be resumed by calling configure_for_resume().

Part of BUG-CLI-02 fix.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add mode Field to JSONSinkConfig and Implement Resume

**Files:**
- Modify: `src/elspeth/plugins/sinks/json_sink.py:22-90` and `141-149`
- Test: `tests/plugins/sinks/test_json_sink_resume.py` (create)

**Step 1: Write the failing tests for JSONSink resume**

Create `tests/plugins/sinks/test_json_sink_resume.py`:

```python
"""Tests for JSONSink resume capability."""

import pytest
from elspeth.plugins.sinks.json_sink import JSONSink


class TestJSONSinkResumeCapability:
    """Tests for JSONSink resume declaration."""

    def test_jsonl_sink_supports_resume(self):
        """JSONL format should support resume."""
        sink = JSONSink({
            "path": "/tmp/test.jsonl",
            "schema": {"fields": "dynamic"},
            "format": "jsonl",
        })
        assert sink.supports_resume is True

    def test_json_array_sink_does_not_support_resume(self):
        """JSON array format should NOT support resume."""
        sink = JSONSink({
            "path": "/tmp/test.json",
            "schema": {"fields": "dynamic"},
            "format": "json",
        })
        assert sink.supports_resume is False

    def test_json_sink_auto_detect_jsonl_supports_resume(self):
        """Auto-detected JSONL format should support resume."""
        sink = JSONSink({
            "path": "/tmp/test.jsonl",  # .jsonl extension
            "schema": {"fields": "dynamic"},
            # No format specified - auto-detect
        })
        assert sink.supports_resume is True

    def test_json_sink_auto_detect_json_does_not_support_resume(self):
        """Auto-detected JSON array format should NOT support resume."""
        sink = JSONSink({
            "path": "/tmp/test.json",  # .json extension
            "schema": {"fields": "dynamic"},
            # No format specified - auto-detect
        })
        assert sink.supports_resume is False


class TestJSONSinkConfigureForResume:
    """Tests for JSONSink configure_for_resume behavior."""

    def test_jsonl_configure_for_resume_sets_append_mode(self):
        """JSONL sink configure_for_resume should set append mode."""
        sink = JSONSink({
            "path": "/tmp/test.jsonl",
            "schema": {"fields": "dynamic"},
            "format": "jsonl",
            "mode": "write",  # Explicit write mode
        })

        assert sink._mode == "write"

        sink.configure_for_resume()

        assert sink._mode == "append"

    def test_json_array_configure_for_resume_raises(self):
        """JSON array sink configure_for_resume should raise NotImplementedError."""
        sink = JSONSink({
            "path": "/tmp/test.json",
            "schema": {"fields": "dynamic"},
            "format": "json",
        })

        with pytest.raises(NotImplementedError) as exc_info:
            sink.configure_for_resume()

        assert "JSON array" in str(exc_info.value)
        assert "jsonl" in str(exc_info.value).lower()

    def test_jsonl_configure_for_resume_idempotent(self):
        """Calling configure_for_resume multiple times should be safe."""
        sink = JSONSink({
            "path": "/tmp/test.jsonl",
            "schema": {"fields": "dynamic"},
            "format": "jsonl",
        })

        sink.configure_for_resume()
        sink.configure_for_resume()  # Second call

        assert sink._mode == "append"


class TestJSONSinkModeField:
    """Tests for JSONSink mode configuration field."""

    def test_json_sink_mode_default_is_write(self):
        """JSONSink should default to mode='write'."""
        sink = JSONSink({
            "path": "/tmp/test.jsonl",
            "schema": {"fields": "dynamic"},
        })
        assert sink._mode == "write"

    def test_json_sink_respects_append_mode(self):
        """JSONSink should respect mode='append' config."""
        sink = JSONSink({
            "path": "/tmp/test.jsonl",
            "schema": {"fields": "dynamic"},
            "mode": "append",
        })
        assert sink._mode == "append"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/sinks/test_json_sink_resume.py -v`
Expected: FAIL (mode field missing, supports_resume not property, configure_for_resume not implemented)

**Step 3: Add mode field to JSONSinkConfig**

In `src/elspeth/plugins/sinks/json_sink.py`, replace lines 22-32 with:

```python
class JSONSinkConfig(PathConfig):
    """Configuration for JSON sink plugin.

    Inherits from PathConfig, which requires schema configuration.
    """

    format: Literal["json", "jsonl"] | None = None
    indent: int | None = None
    encoding: str = "utf-8"
    validate_input: bool = False  # Optional runtime validation of incoming rows
    mode: Literal["write", "append"] = "write"  # "write" (truncate) or "append"
```

**Step 4: Update JSONSink.__init__ to store mode**

In `src/elspeth/plugins/sinks/json_sink.py`, add after line 64 (after `self._validate_input = cfg.validate_input`):

```python
        self._mode = cfg.mode
```

**Step 5: Add supports_resume property to JSONSink**

In `src/elspeth/plugins/sinks/json_sink.py`, add after line 55 (after `plugin_version = "1.0.0"`):

```python
    @property
    def supports_resume(self) -> bool:
        """JSONL format supports resume (append), JSON array does not.

        JSON array format rewrites the entire file on each write (seek(0) + truncate),
        so it cannot append to existing output. JSONL writes line-by-line and can
        append to existing files.
        """
        return self._format == "jsonl"

    def configure_for_resume(self) -> None:
        """Configure JSON sink for resume mode.

        Only JSONL format supports resume. JSON array format rewrites the
        entire file on each write and cannot append.

        Raises:
            NotImplementedError: If format is JSON array (not JSONL).
        """
        if self._format != "jsonl":
            raise NotImplementedError(
                f"JSONSink with format='{self._format}' does not support resume. "
                f"JSON array format rewrites the entire file and cannot append. "
                f"Use format='jsonl' for resumable JSON output."
            )
        self._mode = "append"
```

**Step 6: Update _write_jsonl_batch to respect mode**

In `src/elspeth/plugins/sinks/json_sink.py`, replace the `_write_jsonl_batch` method (lines 141-149) with:

```python
    def _write_jsonl_batch(self, rows: list[dict[str, Any]]) -> None:
        """Write rows as JSONL.

        Uses write mode (truncate) or append mode based on self._mode.
        Append mode is used during resume to add to existing output.
        """
        if self._file is None:
            file_mode = "a" if self._mode == "append" else "w"
            self._file = open(self._path, file_mode, encoding=self._encoding)  # noqa: SIM115

        for row in rows:
            json.dump(row, self._file)
            self._file.write("\n")
```

**Step 7: Run test to verify it passes**

Run: `pytest tests/plugins/sinks/test_json_sink_resume.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add src/elspeth/plugins/sinks/json_sink.py tests/plugins/sinks/test_json_sink_resume.py
git commit -m "feat(json-sink): implement resume capability for JSONL format

- Add mode field to JSONSinkConfig ('write' or 'append')
- Add supports_resume property (True for JSONL, False for JSON array)
- Add configure_for_resume() - sets append mode for JSONL, raises for JSON array
- Update _write_jsonl_batch to use 'a' mode when mode='append'

JSON array format cannot support resume (rewrites entire file).
JSONL format can append line-by-line.

Part of BUG-CLI-02 fix.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Declare AzureBlobSink Does NOT Support Resume

**Files:**
- Modify: `src/elspeth/plugins/azure/blob_sink.py:203-280`
- Test: `tests/plugins/azure/test_blob_sink_resume.py` (create)

**Step 1: Write the failing test for AzureBlobSink resume**

Create `tests/plugins/azure/test_blob_sink_resume.py`:

```python
"""Tests for AzureBlobSink resume capability (NOT supported)."""

import pytest
from unittest.mock import patch, MagicMock


def test_azure_blob_sink_does_not_support_resume():
    """AzureBlobSink should declare supports_resume=False."""
    from elspeth.plugins.azure.blob_sink import AzureBlobSink

    assert AzureBlobSink.supports_resume is False


def test_azure_blob_sink_configure_for_resume_raises():
    """AzureBlobSink.configure_for_resume should raise NotImplementedError."""
    from elspeth.plugins.azure.blob_sink import AzureBlobSink

    # Mock Azure SDK to avoid real connections
    with patch("elspeth.plugins.azure.blob_sink.BlobServiceClient"):
        sink = AzureBlobSink({
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test==;EndpointSuffix=core.windows.net",
            "container": "test-container",
            "blob_path": "test/output.csv",
            "schema": {"fields": "dynamic"},
        })

    with pytest.raises(NotImplementedError) as exc_info:
        sink.configure_for_resume()

    assert "AzureBlobSink" in str(exc_info.value)
    assert "immutable" in str(exc_info.value).lower() or "append" in str(exc_info.value).lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/azure/test_blob_sink_resume.py -v`
Expected: FAIL (supports_resume not explicitly False at class level)

**Step 3: Add explicit supports_resume=False to AzureBlobSink**

In `src/elspeth/plugins/azure/blob_sink.py`, add after line 207 (after `plugin_version = "1.0.0"`):

```python
    # Resume capability: Azure Blobs are immutable - cannot append
    supports_resume: bool = False

    def configure_for_resume(self) -> None:
        """Azure Blob sink does not support resume.

        Azure Blobs are immutable - once uploaded, they cannot be appended to.
        A new blob would need to be created with combined content, which is
        not supported in the resume flow.

        Raises:
            NotImplementedError: Always, as Azure Blobs cannot be appended.
        """
        raise NotImplementedError(
            "AzureBlobSink does not support resume. "
            "Azure Blobs are immutable and cannot be appended to. "
            "Consider using a different blob_path template (e.g., '{{ run_id }}/output.csv') "
            "to create unique blobs per run, or use a local file sink for resumable pipelines."
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/azure/test_blob_sink_resume.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/azure/blob_sink.py tests/plugins/azure/test_blob_sink_resume.py
git commit -m "feat(azure-blob-sink): explicitly declare no resume support

- supports_resume = False (Azure Blobs are immutable)
- configure_for_resume() raises NotImplementedError with helpful message

Azure Blobs cannot be appended to after creation.

Part of BUG-CLI-02 fix.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update CLI Resume to Use Polymorphic Capability

**Files:**
- Modify: `src/elspeth/cli.py:1430-1447`
- Test: `tests/integration/test_cli_resume_sink_capability.py` (create)

**Step 1: Write the failing integration test**

Create `tests/integration/test_cli_resume_sink_capability.py`:

```python
"""Integration tests for CLI resume sink capability validation."""

import pytest
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from elspeth.cli import app


runner = CliRunner()


class TestCLIResumeSinkCapability:
    """Tests for CLI resume command sink capability validation."""

    def test_resume_rejects_non_resumable_sink(self, tmp_path):
        """Resume should reject sinks that don't support resume."""
        # Create a mock settings file that references a non-resumable sink
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
landscape:
  database_url: "sqlite:///:memory:"
source:
  plugin: csv
  options:
    path: /tmp/input.csv
    schema:
      fields: dynamic
    on_validation_failure: discard
sinks:
  output:
    plugin: json
    options:
      path: /tmp/output.json
      format: json
      schema:
        fields: dynamic
""")

        # Mock the database and run existence
        with patch("elspeth.cli._get_landscape_db") as mock_db:
            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            result = runner.invoke(app, [
                "resume", "fake-run-id",
                "--settings", str(settings_file),
                "--execute"
            ])

        # Should fail with clear error about non-resumable sink
        assert result.exit_code != 0
        assert "json" in result.output.lower() or "resume" in result.output.lower()


class TestCLIResumeCallsConfigureForResume:
    """Tests that CLI resume properly calls configure_for_resume on sinks."""

    def test_resume_calls_configure_for_resume_on_resumable_sinks(self):
        """Resume should call configure_for_resume on each sink."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        # Create sink and verify configure_for_resume changes mode
        sink = CSVSink({
            "path": "/tmp/test.csv",
            "schema": {"fields": "dynamic"},
            "mode": "write",
        })

        assert sink._mode == "write"
        assert sink.supports_resume is True

        sink.configure_for_resume()

        assert sink._mode == "append"
```

**Step 2: Run test to verify baseline**

Run: `pytest tests/integration/test_cli_resume_sink_capability.py -v`
Expected: Test setup works (may pass or fail depending on mock setup)

**Step 3: Update CLI resume to use polymorphic capability**

In `src/elspeth/cli.py`, replace lines 1430-1447 with:

```python
        # CRITICAL: Validate and configure sinks for resume mode
        # Each sink declares whether it supports resume and self-configures
        manager = _get_plugin_manager()
        resume_sinks = {}

        for sink_name, sink_config in settings_config.sinks.items():
            sink_cls = manager.get_sink_by_name(sink_config.plugin)
            sink_options = dict(sink_config.options)

            # Instantiate sink to check resume capability
            try:
                sink = sink_cls(sink_options)
            except Exception as e:
                typer.echo(f"Error creating sink '{sink_name}': {e}", err=True)
                raise typer.Exit(1) from None

            # Check if sink supports resume
            if not sink.supports_resume:
                typer.echo(
                    f"Error: Cannot resume with sink '{sink_name}' (plugin: {sink_config.plugin}). "
                    f"This sink does not support resume/append mode.\n"
                    f"Hint: Use a different sink type or start a new run.",
                    err=True,
                )
                raise typer.Exit(1)

            # Configure sink for resume (switches to append mode)
            try:
                sink.configure_for_resume()
            except NotImplementedError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1) from None

            resume_sinks[sink_name] = sink

        # Override source with NullSource for resume (data comes from payloads)
        null_source = NullSource({})
        resume_plugins = {
            **plugins,
            "source": null_source,
            "sinks": resume_sinks,  # Use resume-configured sinks
        }
```

**Step 4: Run tests to verify behavior**

Run: `pytest tests/integration/test_cli_resume_sink_capability.py -v`
Expected: PASS

**Step 5: Run existing resume tests to ensure no regression**

Run: `pytest tests/integration/test_cli_resume_sink_append.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/cli.py tests/integration/test_cli_resume_sink_capability.py
git commit -m "fix(cli): use polymorphic resume capability instead of hardcoded mode

Replace blind 'mode=append' injection with polymorphic approach:
1. Check sink.supports_resume before attempting resume
2. Call sink.configure_for_resume() to self-configure
3. Reject non-resumable sinks with clear error message

This fixes BUG-CLI-02 where resume crashed for non-CSV sinks because
they don't have a 'mode' field.

Benefits:
- CLI no longer needs sink-specific knowledge
- New sinks work without CLI changes
- Clear error messages for non-resumable sinks

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update Existing Documentation Test

**Files:**
- Modify: `tests/integration/test_cli_resume_sink_append.py`

**Step 1: Update the documentation test to reflect new approach**

Replace content of `tests/integration/test_cli_resume_sink_append.py` with:

```python
"""Unit tests for sink resume capability.

These tests verify that sinks properly declare and implement resume capability,
which is used by the CLI resume command to configure sinks for append mode.
"""

from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
import os
import pytest


class TestSinkResumeCapabilityDeclarations:
    """Verify sinks correctly declare their resume capability."""

    def test_csv_sink_supports_resume(self):
        """CSVSink declares supports_resume=True."""
        assert CSVSink.supports_resume is True

    def test_database_sink_supports_resume(self):
        """DatabaseSink declares supports_resume=True."""
        assert DatabaseSink.supports_resume is True

    def test_jsonl_sink_supports_resume(self):
        """JSONSink with JSONL format supports resume."""
        sink = JSONSink({
            "path": "/tmp/test.jsonl",
            "schema": {"fields": "dynamic"},
            "format": "jsonl",
        })
        assert sink.supports_resume is True

    def test_json_array_sink_does_not_support_resume(self):
        """JSONSink with JSON array format does NOT support resume."""
        sink = JSONSink({
            "path": "/tmp/test.json",
            "schema": {"fields": "dynamic"},
            "format": "json",
        })
        assert sink.supports_resume is False


class TestSinkConfigureForResume:
    """Verify sinks properly implement configure_for_resume."""

    def test_csv_sink_configure_for_resume(self):
        """CSVSink.configure_for_resume sets mode to append."""
        sink = CSVSink({
            "path": "/tmp/test.csv",
            "schema": {"fields": "dynamic"},
            "mode": "write",
        })
        assert sink._mode == "write"

        sink.configure_for_resume()

        assert sink._mode == "append"

    @pytest.fixture(autouse=True)
    def allow_raw_secrets(self):
        """Allow raw secrets for database testing."""
        os.environ["ELSPETH_ALLOW_RAW_SECRETS"] = "true"
        yield
        os.environ.pop("ELSPETH_ALLOW_RAW_SECRETS", None)

    def test_database_sink_configure_for_resume(self):
        """DatabaseSink.configure_for_resume sets if_exists to append."""
        sink = DatabaseSink({
            "url": "sqlite:///:memory:",
            "table": "test",
            "schema": {"fields": "dynamic"},
            "if_exists": "replace",
        })
        assert sink._if_exists == "replace"

        sink.configure_for_resume()

        assert sink._if_exists == "append"

    def test_jsonl_sink_configure_for_resume(self):
        """JSONSink JSONL format configure_for_resume sets mode to append."""
        sink = JSONSink({
            "path": "/tmp/test.jsonl",
            "schema": {"fields": "dynamic"},
            "format": "jsonl",
            "mode": "write",
        })
        assert sink._mode == "write"

        sink.configure_for_resume()

        assert sink._mode == "append"

    def test_json_array_sink_configure_for_resume_raises(self):
        """JSONSink JSON array format configure_for_resume raises NotImplementedError."""
        sink = JSONSink({
            "path": "/tmp/test.json",
            "schema": {"fields": "dynamic"},
            "format": "json",
        })

        with pytest.raises(NotImplementedError) as exc_info:
            sink.configure_for_resume()

        assert "jsonl" in str(exc_info.value).lower()
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/integration/test_cli_resume_sink_append.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_cli_resume_sink_append.py
git commit -m "test: update resume sink tests for polymorphic capability

Replace documentation-style tests with proper capability tests:
- Test supports_resume declarations for all sink types
- Test configure_for_resume implementations
- Test error handling for non-resumable sinks

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Run Full Test Suite and Fix Any Regressions

**Step 1: Run all sink-related tests**

Run: `pytest tests/plugins/sinks/ tests/integration/test_cli_resume*.py tests/contracts/sink_contracts/ -v`
Expected: All PASS

**Step 2: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All PASS (or identify any regressions)

**Step 3: Fix any regressions (if needed)**

If tests fail, investigate and fix without breaking the new functionality.

**Step 4: Final commit (if regression fixes needed)**

```bash
git add -A
git commit -m "fix: address test regressions from resume capability changes

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Files Modified | Tests Added |
|------|----------------|-------------|
| 1 | `protocols.py` | 1 |
| 2 | `base.py` | 2 |
| 3 | `csv_sink.py` | 3 |
| 4 | `database_sink.py` | 3 |
| 5 | `json_sink.py` | 9 |
| 6 | `blob_sink.py` | 2 |
| 7 | `cli.py` | 2 |
| 8 | Test update | 8 |
| **Total** | **8 files** | **30 tests** |

**Estimated Time:** 3-4 hours

**Key Architectural Benefits:**
1. CLI no longer contains sink-specific knowledge
2. New sink types work without CLI modification
3. Resume capability is discoverable via `sink.supports_resume`
4. Clear error messages when resume is not supported
5. Respects plugin boundary (sinks self-configure)
