# tests/unit/plugins/test_assert_to_raise.py
"""Verify assert-to-raise conversions across plugin files.

18 assert statements were replaced with explicit RuntimeError raises across
8 plugin files.  These tests cover the three most common guard patterns:

1. ``_recorder is None`` before ``_create_provider()`` fires
   (LLMTransform unified guard, and the same pattern in AzureContentSafety)

2. ``_writer / _file is None`` invariants inside CSVSink.write()
   (guards that protect against internal state bugs)

3. ``connect_output() already called`` double-initialisation guard
   (AzureContentSafety — LLMTransform already had this covered)

Each test constructs the *minimum* plugin instance required to reach the
guard, leaves the precondition violated, and asserts RuntimeError is raised
with the expected message substring.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory

# ---------------------------------------------------------------------------
# Shared config helpers
# ---------------------------------------------------------------------------

_LLM_AZURE_CONFIG = {
    "provider": "azure",
    "deployment_name": "test-deployment",
    "endpoint": "https://test.openai.azure.com",
    "api_key": "test-key",
    "template": "Classify: {{ row.text }}",
    "schema": {"mode": "observed"},
    "required_input_fields": [],
}

_CONTENT_SAFETY_CONFIG = {
    "endpoint": "https://test.cognitiveservices.azure.com",
    "api_key": "test-key",
    "fields": ["content"],
    "thresholds": {
        "hate": 2,
        "violence": 2,
        "sexual": 2,
        "self_harm": 0,
    },
    "schema": {"mode": "observed"},
}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestAssertToRaiseConversions:
    """One representative test per converted guard pattern."""

    # ------------------------------------------------------------------
    # Pattern 1: LLMTransform — _recorder not initialized
    # ------------------------------------------------------------------

    def test_llm_transform_recorder_not_initialized_raises(self) -> None:
        """_create_provider() raises RuntimeError when _recorder is None.

        Before the assert-to-raise conversion this was:
            assert self._recorder is not None

        After conversion:
            raise RuntimeError(
                "_recorder not initialized — _create_provider called before on_start()"
            )

        The guard fires when the transform is used without on_start() having
        been called (i.e. without the engine setting up the recorder).
        """
        from elspeth.plugins.transforms.llm.transform import LLMTransform

        transform = LLMTransform(_LLM_AZURE_CONFIG)

        # _recorder is None by default — on_start() was never called.
        assert transform._recorder is None

        # _create_provider must raise, not silently proceed with a None recorder.
        with pytest.raises(RuntimeError, match="recorder"):
            transform._create_provider()

    # ------------------------------------------------------------------
    # Pattern 2: AzureContentSafety — _recorder not initialized
    # ------------------------------------------------------------------

    def test_content_safety_recorder_not_initialized_raises(self) -> None:
        """_get_http_client() raises RuntimeError when _recorder is None.

        Same conversion pattern as the LLM transform but in the
        AzureContentSafety transform's HTTP client cache method.

        Guard message: "_recorder not initialized — _get_http_client called
        before begin_run()"
        """
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(_CONTENT_SAFETY_CONFIG)

        # _recorder is None by default — on_start() was never called.
        assert transform._recorder is None

        # _get_http_client must raise, not silently create a client with None recorder.
        with pytest.raises(RuntimeError, match="recorder"):
            transform._get_http_client("some-state-id")

    # ------------------------------------------------------------------
    # Pattern 3: AzureContentSafety — connect_output() already called
    # ------------------------------------------------------------------

    def test_content_safety_connect_output_twice_raises(self) -> None:
        """connect_output() raises RuntimeError when called a second time.

        Guard message: "connect_output() already called"

        The first call initialises the batch processing infrastructure.
        A second call must be rejected — the BatchTransformMixin is already
        running, and reinitialisation would corrupt internal state.
        """
        from elspeth.plugins.infrastructure.batching.ports import CollectorOutputPort
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(_CONTENT_SAFETY_CONFIG)
        collector = CollectorOutputPort()

        # First call succeeds and sets _batch_initialized = True.
        transform.connect_output(collector, max_pending=5)

        try:
            with pytest.raises(RuntimeError, match="already"):
                transform.connect_output(collector, max_pending=5)
        finally:
            transform.close()

    # ------------------------------------------------------------------
    # Pattern 4: CSVSink — internal invariant guards inside write()
    # ------------------------------------------------------------------

    def test_csv_sink_writer_not_initialized_raises(self) -> None:
        """CSVSink.write() raises RuntimeError when _writer is None after _open_file.

        The guard:
            if file is None or writer is None:
                raise RuntimeError("CSVSink writer not initialized - this is a bug")

        This protects the invariant that _file and _writer are always set
        together.  We trigger it by patching _open_file to leave _writer=None
        while letting the rows-non-empty branch pass.

        Uses append mode because write mode stages before opening the file
        and uses a separate code path (_open_file_write_mode).  The guard
        under test lives in the append/subsequent-call branch.
        """
        import tempfile
        from pathlib import Path

        from elspeth.plugins.sinks.csv_sink import CSVSink

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.csv"
            sink = inject_write_failure(
                CSVSink(
                    {
                        "path": str(output_path),
                        "schema": {"mode": "observed"},
                        "mode": "append",
                    }
                )
            )
            factory = make_factory()
            ctx = make_context(run_id="test-run", landscape=factory.plugin_audit_writer())

            # Patch _open_file so it opens _file but leaves _writer as None.
            # This simulates partial initialisation — the exact state the guard
            # protects against.
            def broken_open_file(rows: list) -> None:  # type: ignore[type-arg]
                # Open a real file handle (so the file exists for stat() calls),
                # but deliberately leave _writer=None.
                sink._file = open(str(output_path), "w")  # noqa: SIM115
                sink._writer = None
                sink._fieldnames = list(rows[0].keys())
                sink._hasher = __import__("hashlib").sha256()

            with (
                patch.object(sink, "_open_file", side_effect=broken_open_file),
                pytest.raises(RuntimeError, match="writer"),
            ):
                sink.write([{"col": "value"}], ctx)

            # Clean up the file handle left open by broken_open_file
            if sink._file is not None:
                sink._file.close()

    def test_csv_sink_fieldnames_not_set_raises(self) -> None:
        """CSVSink.write() raises RuntimeError when _fieldnames is None.

        The guard:
            if fieldnames is None:
                raise RuntimeError(
                    "write() called before _fieldnames set by _write_header()"
                )

        We trigger it by patching _open_file to set _file and _writer but
        leave _fieldnames=None.

        Uses append mode because write mode stages before opening the file
        and uses a separate code path.  The guard under test lives in the
        append/subsequent-call branch.
        """
        import tempfile
        from pathlib import Path

        from elspeth.plugins.sinks.csv_sink import CSVSink

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.csv"
            sink = inject_write_failure(
                CSVSink(
                    {
                        "path": str(output_path),
                        "schema": {"mode": "observed"},
                        "mode": "append",
                    }
                )
            )
            factory = make_factory()
            ctx = make_context(run_id="test-run", landscape=factory.plugin_audit_writer())

            mock_writer = MagicMock()

            def broken_open_file_no_fieldnames(rows: list) -> None:  # type: ignore[type-arg]
                # Provide _file and _writer but deliberately omit _fieldnames.
                sink._file = open(str(output_path), "w")  # noqa: SIM115
                sink._writer = mock_writer
                sink._fieldnames = None  # intentionally broken
                sink._hasher = __import__("hashlib").sha256()

            with (
                patch.object(sink, "_open_file", side_effect=broken_open_file_no_fieldnames),
                pytest.raises(RuntimeError, match="fieldnames"),
            ):
                sink.write([{"col": "value"}], ctx)

            # Clean up the file handle
            if sink._file is not None:
                sink._file.close()
