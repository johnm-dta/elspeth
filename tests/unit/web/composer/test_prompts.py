"""Tests for LLM message construction — build_messages and build_context_string.

Verifies:
- build_messages returns a NEW list on every call (cross-turn contamination guard)
- Message ordering: stable system → dynamic context → chat history → user message
- Dynamic context message injects pipeline state and plugin catalog
- Empty chat history handled correctly
- Context string includes validation summary
- build_context_string redacts blob storage paths
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts.freeze import deep_freeze
from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.state_machine import TerminalKind, TerminalReason, TerminalState
from elspeth.web.composer.prompts import (
    SYSTEM_PROMPT,
    build_context_string,
    build_messages,
    build_run_diagnostics_messages,
    build_system_prompt,
)
from elspeth.web.composer.state import CompositionState, PipelineMetadata, SourceSpec

EXPECTED_REDACTED_BLOB_SOURCE_PATH = "<redacted-blob-source-path>"


class StubCatalog:
    """Minimal CatalogService conforming to the protocol."""

    def list_sources(self) -> list[PluginSummary]:
        return [
            PluginSummary(
                name="csv",
                description="CSV source",
                plugin_type="source",
                config_fields=[],
            )
        ]

    def list_transforms(self) -> list[PluginSummary]:
        return [
            PluginSummary(
                name="passthrough",
                description="Uppercase transform",
                plugin_type="transform",
                config_fields=[],
            )
        ]

    def list_sinks(self) -> list[PluginSummary]:
        return [
            PluginSummary(
                name="csv",
                description="CSV sink",
                plugin_type="sink",
                config_fields=[],
            )
        ]

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        raise ValueError(f"Not implemented for stub: {plugin_type}/{name}")


def _stub_catalog() -> CatalogService:
    """Return a protocol-typed stub so mypy verifies conformance."""
    catalog: CatalogService = StubCatalog()
    return catalog


def _empty_state() -> CompositionState:
    """A minimal empty CompositionState for testing."""
    return CompositionState.from_dict(
        {
            "source": None,
            "nodes": [],
            "edges": [],
            "outputs": [],
            "metadata": {"name": "Test Pipeline", "description": ""},
            "version": 1,
        }
    )


class TestBuildMessages:
    """Message list construction and isolation."""

    def test_returns_new_list_each_call(self) -> None:
        """Critical: each call returns a distinct list object to prevent cross-turn contamination."""
        state = _empty_state()
        catalog = _stub_catalog()
        history: list[dict[str, Any]] = []

        list1 = build_messages(history, state, "Hello", catalog)
        list2 = build_messages(history, state, "Hello", catalog)
        assert list1 is not list2

    def test_mutating_returned_list_does_not_affect_next_call(self) -> None:
        """Appending to a returned list must not leak into subsequent calls."""
        state = _empty_state()
        catalog = _stub_catalog()

        list1 = build_messages([], state, "Hello", catalog)
        list1.append({"role": "assistant", "content": "I was injected"})

        list2 = build_messages([], state, "Hello", catalog)
        roles = [m["role"] for m in list2]
        assert "assistant" not in roles

    def test_message_ordering_system_context_history_user(self) -> None:
        """Messages must be: stable system, dynamic context, history, then user."""
        state = _empty_state()
        catalog = _stub_catalog()
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        messages = build_messages(history, state, "new question", catalog)

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "system"
        assert messages[1]["content"].startswith("Current pipeline state and available plugins:")
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "previous question"
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "previous answer"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "new question"

    def test_empty_history_produces_system_context_and_user_only(self) -> None:
        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages([], state, "my question", catalog)

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "system"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "my question"

    def test_system_prompt_and_dynamic_context_are_split_for_prompt_cache(self) -> None:
        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages([], state, "test", catalog)

        stable_system_content = messages[0]["content"]
        dynamic_context_content = messages[1]["content"]

        assert SYSTEM_PROMPT in stable_system_content
        assert "Current pipeline state" not in stable_system_content
        assert dynamic_context_content.startswith("Current pipeline state and available plugins:")
        assert "csv" in dynamic_context_content
        assert "passthrough" in dynamic_context_content

    def test_first_system_message_is_stable_when_state_changes(self) -> None:
        catalog = _stub_catalog()

        empty_messages = build_messages([], _empty_state(), "test", catalog)
        sourced_messages = build_messages([], _blob_source_state(), "test", catalog)

        assert empty_messages[0]["role"] == "system"
        assert sourced_messages[0]["role"] == "system"
        assert empty_messages[0]["content"] == sourced_messages[0]["content"]
        assert empty_messages[1]["role"] == "system"
        assert sourced_messages[1]["role"] == "system"
        assert empty_messages[1]["content"] != sourced_messages[1]["content"]


class TestBuildContextString:
    """Context injection into the system prompt."""

    def test_contains_state_and_plugins(self) -> None:
        state = _empty_state()
        catalog = _stub_catalog()

        context = build_context_string(state, catalog)
        parsed = json.loads(context.split("\n", 1)[1])  # Skip header line

        assert "current_state" in parsed
        assert "available_plugins" in parsed
        plugins = parsed["available_plugins"]
        assert "csv" in plugins["sources"]
        assert "passthrough" in plugins["transforms"]
        assert "csv" in plugins["sinks"]

    def test_includes_validation_summary(self) -> None:
        state = _empty_state()
        catalog = _stub_catalog()

        context = build_context_string(state, catalog)
        parsed = json.loads(context.split("\n", 1)[1])

        validation = parsed["current_state"]["validation"]
        assert "is_valid" in validation
        assert "errors" in validation

    def test_metadata_included(self) -> None:
        state = _empty_state()
        catalog = _stub_catalog()

        context = build_context_string(state, catalog)
        parsed = json.loads(context.split("\n", 1)[1])

        assert parsed["current_state"]["metadata"]["name"] == "Test Pipeline"

    def test_includes_warnings_and_suggestions(self) -> None:
        """Validation context must include warnings and suggestions, not just errors."""
        state = _empty_state()
        catalog = _stub_catalog()

        context = build_context_string(state, catalog)
        parsed = json.loads(context.split("\n", 1)[1])

        validation = parsed["current_state"]["validation"]
        assert "warnings" in validation
        assert "suggestions" in validation

    def test_context_includes_prompt_visible_state_exists_marker(self) -> None:
        """The model sees empty state as an explicit hard blocker marker."""
        catalog = _stub_catalog()

        empty_context = build_context_string(_empty_state(), catalog)
        sourced_context = build_context_string(_blob_source_state(), catalog)
        empty_parsed = json.loads(empty_context.split("\n", 1)[1])
        sourced_parsed = json.loads(sourced_context.split("\n", 1)[1])

        assert empty_parsed["composer_progress"]["state_exists"] is False
        assert sourced_parsed["composer_progress"]["state_exists"] is True


class TestBuildSystemPrompt:
    """System prompt composition with optional deployment layer."""

    def test_no_data_dir_returns_core_skill_only(self) -> None:
        """Without data_dir, returns the core skill unchanged."""
        result = build_system_prompt(None)
        assert result == SYSTEM_PROMPT

    def test_missing_deployment_skill_returns_core_only(self, tmp_path: Path) -> None:
        """data_dir with no skills/ subdir returns core skill only."""
        result = build_system_prompt(str(tmp_path))
        assert result == SYSTEM_PROMPT

    def test_deployment_skill_appended_after_separator(self, tmp_path: Path) -> None:
        """Deployment skill content is appended after a separator, in correct order."""
        deployment_content = "# Our Custom Providers\n\nUse ACME_API_KEY.\n"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "pipeline_composer.md").write_text(deployment_content)

        result = build_system_prompt(str(tmp_path))

        # Exact equality — verifies ordering, not just presence.
        assert result == SYSTEM_PROMPT + "\n\n---\n\n" + deployment_content

    def test_empty_string_data_dir_still_calls_loader(self, tmp_path: Path) -> None:
        """Empty string data_dir is not None — build_system_prompt is called."""
        # Empty string produces a relative path lookup that finds no skills/.
        # The important thing is it goes through build_system_prompt, not the
        # SYSTEM_PROMPT fast path.
        result = build_system_prompt("")
        assert result == SYSTEM_PROMPT


class TestBuildRunDiagnosticsMessages:
    """Message construction for run diagnostics explanations."""

    def test_includes_core_composer_skill_pack(self) -> None:
        messages = build_run_diagnostics_messages(
            {"run_id": "run-1", "summary": {"token_count": 1}},
            data_dir=None,
        )

        assert messages[0]["role"] == "system"
        assert SYSTEM_PROMPT in messages[0]["content"]
        assert "run diagnostics" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert '"run_id": "run-1"' in messages[1]["content"]

    def test_includes_deployment_skill_overlay(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "pipeline_composer.md").write_text("Deployment composer rules")

        messages = build_run_diagnostics_messages(
            {"run_id": "run-1", "summary": {"token_count": 1}},
            data_dir=str(tmp_path),
        )

        assert SYSTEM_PROMPT in messages[0]["content"]
        assert "Deployment composer rules" in messages[0]["content"]

    def test_requests_structured_visible_working_view(self) -> None:
        messages = build_run_diagnostics_messages(
            {"run_id": "run-1", "summary": {"token_count": 1}},
            data_dir=None,
        )

        system_content = messages[0]["content"]
        assert "strict JSON" in system_content
        assert '"headline"' in system_content
        assert '"evidence"' in system_content
        assert '"meaning"' in system_content
        assert '"next_steps"' in system_content
        assert "visible evidence" in system_content
        assert "hidden chain-of-thought" in system_content


class TestBuildMessagesWithDataDir:
    """build_messages with deployment skill overlay."""

    def test_data_dir_none_uses_core_prompt(self) -> None:
        """Default (no data_dir) uses core SYSTEM_PROMPT via fast path."""
        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages([], state, "test", catalog, data_dir=None)
        system_content = messages[0]["content"]

        # Stable system message is only the prompt prefix; dynamic context is separate.
        assert system_content == SYSTEM_PROMPT
        assert messages[1]["content"].startswith("Current pipeline state and available plugins:")

    def test_data_dir_with_deployment_skill_injects_it(self, tmp_path: Path) -> None:
        """When data_dir has a deployment skill, it appears in the system message."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "pipeline_composer.md").write_text("# Deployment: use ACME provider\n")

        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages([], state, "test", catalog, data_dir=str(tmp_path))
        system_content = messages[0]["content"]

        assert "# Deployment: use ACME provider" in system_content
        assert SYSTEM_PROMPT in system_content


def _blob_source_state(
    *,
    path: str | None = "/internal/blobs/sess123/blobid_data.csv",
    blob_ref: str | None = "blobid",
) -> CompositionState:
    """Build a CompositionState with a source whose options contain blob fields."""
    raw_options: dict[str, Any] = {"schema": {"mode": "observed"}}
    if path is not None:
        raw_options["path"] = path
    if blob_ref is not None:
        raw_options["blob_ref"] = blob_ref
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            options=deep_freeze(raw_options),
            on_success="t1",
            on_validation_failure="quarantine",
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _completed_terminal() -> TerminalState:
    """A COMPLETED TerminalState (no reason required)."""
    return TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml=None)


def _exited_terminal(reason: TerminalReason = TerminalReason.USER_PRESSED_EXIT) -> TerminalState:
    """An EXITED_TO_FREEFORM TerminalState with a reason."""
    return TerminalState(kind=TerminalKind.EXITED_TO_FREEFORM, reason=reason, pipeline_yaml=None)


class TestBuildMessagesGuidedTerminal:
    """Integration tests: build_messages with guided_terminal set.

    Verifies Codex #17: the first freeform turn after a guided exit carries the
    same deployment overlay and advisor-strip as subsequent freeform turns.
    """

    def test_guided_terminal_with_data_dir_includes_deployment_overlay(self, tmp_path: Path) -> None:
        """deployment overlay content must appear in the transition prompt system message."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        deployment_content = "# Codex17 Deployment Overlay\n"
        (skills_dir / "pipeline_composer.md").write_text(deployment_content)

        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages(
            [],
            state,
            "continue",
            catalog,
            data_dir=str(tmp_path),
            guided_terminal=_completed_terminal(),
        )

        system_content = messages[0]["content"]
        assert deployment_content.strip() in system_content, "Deployment overlay missing from guided-terminal transition prompt (Codex #17)"

    def test_guided_terminal_advisor_disabled_strips_advisor_content(self) -> None:
        """When advisor_enabled=False, advisor-specific content must be absent from the transition prompt."""
        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages(
            [],
            state,
            "continue",
            catalog,
            advisor_enabled=False,
            guided_terminal=_completed_terminal(),
        )

        system_content = messages[0]["content"]
        # The advisor subsection heading that _strip_advisor_content removes.
        assert "#### When You Are Still Stuck" not in system_content, (
            "Advisor subsection must be stripped when advisor_enabled=False (Codex #17)"
        )
        # The advisor tool name token that _strip_advisor_content removes.
        assert ", `request_advisor_hint`" not in system_content, (
            "Advisor tool token must be stripped when advisor_enabled=False (Codex #17)"
        )

    def test_guided_terminal_advisor_enabled_retains_advisor_content(self) -> None:
        """When advisor_enabled=True (default), advisor sections must remain in the transition prompt."""
        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages(
            [],
            state,
            "continue",
            catalog,
            advisor_enabled=True,
            guided_terminal=_completed_terminal(),
        )

        system_content = messages[0]["content"]
        # At least one of the two advisor-specific markers must survive.
        has_advisor_section = "#### When You Are Still Stuck" in system_content
        has_advisor_token = "request_advisor_hint" in system_content
        assert has_advisor_section or has_advisor_token, "Advisor content must be present when advisor_enabled=True"

    def test_guided_terminal_no_data_dir_matches_non_transition_core_skill(self) -> None:
        """Without data_dir the transition prompt freeform layer equals the standard system prompt."""
        state = _empty_state()
        catalog = _stub_catalog()

        transition_messages = build_messages(
            [],
            state,
            "continue",
            catalog,
            data_dir=None,
            guided_terminal=_completed_terminal(),
        )
        normal_messages = build_messages(
            [],
            state,
            "continue",
            catalog,
            data_dir=None,
        )

        # The normal freeform system content (SYSTEM_PROMPT) must be a substring
        # of the transition prompt — the transition prompt wraps it.
        normal_system = normal_messages[0]["content"]
        transition_system = transition_messages[0]["content"]
        assert normal_system in transition_system, "Transition prompt must embed the standard freeform system prompt as its final layer"

    def test_guided_terminal_exited_uses_reason_value(self) -> None:
        """EXITED_TO_FREEFORM terminal embeds the reason token in the transition prompt."""
        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages(
            [],
            state,
            "continue",
            catalog,
            guided_terminal=_exited_terminal(TerminalReason.SOLVER_EXHAUSTED),
        )

        system_content = messages[0]["content"]
        assert "solver_exhausted" in system_content

    def test_guided_terminal_exited_without_reason_raises_invariant_error_no_leak(self) -> None:
        """obs-ae69e10e00 regression: an EXITED_TO_FREEFORM TerminalState with
        ``reason=None`` violates the TerminalState invariant.  build_messages must:

        1. Raise ``InvariantError`` (server-bug sentinel routed through the
           B1-sanitized 500 handler at routes.py:3252 / 3764), NOT
           ``RuntimeError`` (which would land at FastAPI's default 500 and
           bypass the slog event + _safe_frame_strings capture).
        2. NOT embed ``pipeline_yaml`` or other TerminalState repr content in
           the exception message — same Tier-1 leak vector that B1
           (commit eb30f669) and I1 (commit ba424ad9) sanitized at
           routes.py:4634/4696.  The PR-introduced ``{guided_terminal!r}``
           formatter would have leaked source paths, plugin options, and
           secret references via the exception message into any handler that
           reads ``str(exc)`` (e.g., FastAPI default 500 surfacing).
        """
        state = _empty_state()
        catalog = _stub_catalog()
        # Construct an invalid TerminalState directly — bypass the step_advance
        # invariant that would normally prevent this combination.  Sentinel
        # strings in pipeline_yaml pin the no-leak assertion: if the {!r}
        # interpolation regresses, the assertion fires.
        sentinel_yaml = "source:\n  options:\n    secret_ref: env://LEAKED_SECRET_SENTINEL_AE69E10E00\n"
        bad_terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=None,
            pipeline_yaml=sentinel_yaml,
        )

        with pytest.raises(InvariantError) as exc_info:
            build_messages(
                [],
                state,
                "continue",
                catalog,
                guided_terminal=bad_terminal,
            )

        # Class swap pin (B1 conformance): InvariantError is the project
        # sentinel for server-side invariant violations; the route handler
        # dispatches on this exact class.
        assert type(exc_info.value) is InvariantError

        # No-leak pin (load-bearing security assertion): the corrupted
        # value's repr must not appear in the exception message.  Without
        # this assertion the {!r}-leak regression would silently re-land.
        exc_message = str(exc_info.value)
        assert "LEAKED_SECRET_SENTINEL_AE69E10E00" not in exc_message
        assert "pipeline_yaml" not in exc_message
        assert "secret_ref" not in exc_message
        assert sentinel_yaml not in exc_message
        # Invariant name is preserved for diagnostic value.
        assert "EXITED_TO_FREEFORM" in exc_message


class TestBuildContextStringRedaction:
    """Blob storage path redaction in build_context_string."""

    def test_build_context_string_redacts_blob_path(self) -> None:
        """Blob-backed source: raw path must NOT appear, blob_ref must remain."""
        state = _blob_source_state(
            path="/internal/blobs/sess123/blobid_data.csv",
            blob_ref="blobid",
        )
        catalog = _stub_catalog()

        context = build_context_string(state, catalog)

        assert "/internal/blobs/sess123/blobid_data.csv" not in context
        assert EXPECTED_REDACTED_BLOB_SOURCE_PATH in context
        assert "blobid" in context

    def test_build_context_string_non_blob_source_unaffected(self) -> None:
        """File-backed source (no blob_ref): path must be preserved."""
        state = _blob_source_state(
            path="/data/input/report.csv",
            blob_ref=None,
        )
        catalog = _stub_catalog()

        context = build_context_string(state, catalog)

        assert "/data/input/report.csv" in context

    def test_build_context_string_blob_ref_without_path_no_error(self) -> None:
        """Source with blob_ref but no path key must not raise."""
        state = _blob_source_state(path=None, blob_ref="blobid")
        catalog = _stub_catalog()

        # Should complete without error.
        context = build_context_string(state, catalog)
        assert "blobid" in context
