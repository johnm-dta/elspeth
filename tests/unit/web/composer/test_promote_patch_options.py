"""ARG_ERROR routing for patch_source_options / patch_node_options /
patch_output_options ToolArgumentError (Task 15 / Wave 4).

Rev-2 BLOCKER_A: promoted handlers MUST catch ``pydantic.ValidationError``
and re-raise as :class:`ToolArgumentError`.  The compose loop's
``ToolArgumentError`` handler at ``service.py:2480`` routes to ARG_ERROR.
A bare ``ValidationError`` escaping the handler hits ``service.py:2564``
(→ :class:`ComposerPluginCrashError`` → HTTP 500) — the wrong disposition
for Tier-3 input.

These tests pin the exception-class and ``__cause__`` chain for all three
merge-patch handlers, plus regression coverage for valid input, extra-field
rejection, and the uniformity of the ``patch`` field's Sensitive marker
against the persistence boundary.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    MANIFEST,
    PatchNodeOptionsArgumentsModel,
    PatchOutputOptionsArgumentsModel,
    PatchSourceOptionsArgumentsModel,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.tools import (
    _execute_patch_node_options,
    _execute_patch_output_options,
    _execute_patch_source_options,
)
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ctx() -> ToolContext:
    """Bare ToolContext sufficient for the argument-validation tests below.

    The Pydantic-validation tests reject the arguments before any catalog or
    data-dir consumption, so a catalog-less context is fine. ``MagicMock``
    keeps the type contract honest without spinning up a real catalog.
    """

    from unittest.mock import MagicMock

    from elspeth.web.catalog.protocol import CatalogService

    return ToolContext(catalog=MagicMock(spec=CatalogService))


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_source(options: dict[str, Any] | None = None) -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options=options or {"path": "/tmp/data.csv", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_node(node_id: str = "t1", options: dict[str, Any] | None = None) -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id=node_id,
                node_type="transform",
                plugin="uppercase",
                input="source_out",
                on_success="main",
                on_error="discard",
                options=options or {},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_llm_node(node_id: str = "llm1", options: dict[str, Any] | None = None) -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id=node_id,
                node_type="transform",
                plugin="llm",
                input="source_out",
                on_success="main",
                on_error="discard",
                options=options
                or {
                    "provider": "openrouter",
                    "model": "anthropic/claude-haiku-4.5",
                    "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    "prompt_template": "Summarise {{ row.text }}.",
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_web_scrape_and_mapper() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="fetch_pages",
                node_type="transform",
                plugin="web_scrape",
                input="rows",
                on_success="scraped_rows",
                on_error="discard",
                options={
                    "schema": {"mode": "observed"},
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "content_fingerprint",
                    "http": {
                        "abuse_contact": "noreply@dta.gov.au",
                        "scraping_reason": "DTA technical demonstration",
                        "allowed_hosts": "public_only",
                    },
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="drop_raw_html",
                node_type="transform",
                plugin="field_mapper",
                input="scraped_rows",
                on_success="clean_rows",
                on_error="discard",
                options={
                    "schema": {"mode": "observed"},
                    "mapping": {"url": "url", "content": "content"},
                    "select_only": False,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_output(sink_name: str = "out1", options: dict[str, Any] | None = None) -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(
            OutputSpec(
                name=sink_name,
                plugin="csv",
                options=options or {"path": "/tmp/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


# ---------------------------------------------------------------------------
# patch_source_options
# ---------------------------------------------------------------------------


class TestPromotePatchSourceOptionsArgErrorRouting:
    """Validates ARG_ERROR routing for bad patch_source_options arguments."""

    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """Missing required 'patch' field triggers ToolArgumentError."""
        state = _state_with_source()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_source_options({}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_patch_not_dict_raises_tool_argument_error(self) -> None:
        """Non-dict patch value rejected by Pydantic model before handler logic."""
        state = _state_with_source()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_source_options({"patch": "not-a-dict"}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_patch_none_raises_tool_argument_error(self) -> None:
        """None patch is not a valid dict."""
        state = _state_with_source()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_source_options({"patch": None}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_extra_field_raises_tool_argument_error(self) -> None:
        """extra='forbid' on the model rejects stray fields at Tier-3."""
        state = _state_with_source()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_source_options(
                {"patch": {"path": "/b"}, "node_id": "stray"},
                state,
                _ctx(),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_valid_arguments_dispatch_normally(self) -> None:
        """Valid patch dict succeeds when state has a source."""
        state = _state_with_source({"path": "/a", "schema": {"mode": "observed"}})
        result = _execute_patch_source_options({"patch": {"path": "/b"}}, state, _ctx())
        assert result.success is True
        assert result.updated_state.sources["source"].options["path"] == "/b"

    def test_manifest_entry_is_type_driven(self) -> None:
        assert "patch_source_options" in MANIFEST
        assert MANIFEST["patch_source_options"].argument_model is PatchSourceOptionsArgumentsModel

    def test_redact_collapses_patch_to_string_summary(self) -> None:
        """The Sensitive marker on 'patch' collapses the dict to a string via
        the summarizer — the raw ``dict`` value is not persisted as-is.

        When blob_ref is absent, :func:`_summarize_set_source_options` serialises
        the dict to canonical JSON (no path-redaction in that case).  The key
        contract is that the field's value TYPE changes from ``dict`` to ``str``
        at the persistence boundary — the raw LLM-supplied object never enters
        ``chat_messages.tool_calls`` as a nested dict.
        """
        raw_args = {"patch": {"api_key": "secret-ref", "path": "/data/in.csv"}}
        redacted = redact_tool_call_arguments(
            "patch_source_options",
            raw_args,
            telemetry=NoopRedactionTelemetry(),
        )
        # The patch key is present in the redacted output.
        assert "patch" in redacted
        # The summarizer collapses the dict to a string, not a nested dict.
        assert isinstance(redacted["patch"], str), "patch field should be a string summary, not a raw dict"


# ---------------------------------------------------------------------------
# patch_node_options
# ---------------------------------------------------------------------------


class TestPromotePatchNodeOptionsArgErrorRouting:
    """Validates ARG_ERROR routing for bad patch_node_options arguments."""

    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """Missing both required fields triggers ToolArgumentError."""
        state = _state_with_node()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_node_options({}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_missing_patch_raises_tool_argument_error(self) -> None:
        """node_id alone is insufficient; patch is required."""
        state = _state_with_node()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_node_options({"node_id": "t1"}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_missing_node_id_raises_tool_argument_error(self) -> None:
        """patch alone is insufficient; node_id is required."""
        state = _state_with_node()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_node_options({"patch": {"field": "x"}}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_patch_not_dict_raises_tool_argument_error(self) -> None:
        """Non-dict patch value rejected by Pydantic model."""
        state = _state_with_node()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_node_options({"node_id": "t1", "patch": 42}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_node_id_not_str_raises_tool_argument_error(self) -> None:
        """Non-string node_id rejected by Pydantic model."""
        state = _state_with_node()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_node_options({"node_id": 99, "patch": {}}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_extra_field_raises_tool_argument_error(self) -> None:
        """extra='forbid' rejects stray fields at Tier-3."""
        state = _state_with_node()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_node_options(
                {"node_id": "t1", "patch": {}, "sink_name": "stray"},
                state,
                _ctx(),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_valid_arguments_dispatch_normally(self) -> None:
        """Valid Pydantic args reach the handler body (no ToolArgumentError).

        This test class pins ARG_ERROR routing — i.e., that valid args do
        NOT raise :class:`ToolArgumentError` and instead reach the handler
        body. The handler's downstream prevalidation step rejects the
        placeholder ``"uppercase"`` plugin name (see ``_state_with_node``)
        with a structured ``rejected_mutation`` ValidationEntry, which is
        the expected behaviour for unknown plugins — but that rejection
        is a separate concern owned by the prevalidation test suite.

        Here we only assert that dispatch reached the handler: a
        :class:`ToolResult` came back rather than a propagated
        :class:`ToolArgumentError`. Mutation success on real plugins is
        covered by the source/output variants of this test which use
        real plugin names.
        """
        from elspeth.web.composer.tools import ToolResult

        state = _state_with_node("t1", {"field": "old"})
        result = _execute_patch_node_options({"node_id": "t1", "patch": {"field": "new"}}, state, _ctx())
        assert isinstance(result, ToolResult)

    def test_llm_patch_restores_prompt_template_review_requirement(self) -> None:
        """Patching LLM options must not remove the Class 3 prompt gate."""
        state = _state_with_llm_node(
            options={
                "provider": "openrouter",
                "model": "anthropic/claude-haiku-4.5",
                "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                "prompt_template": "Old {{ row.text }}.",
                "schema": {"mode": "observed"},
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "vague",
                        "kind": "vague_term",
                        "user_term": "summary",
                        "status": "pending",
                        "draft": "short summary",
                        "event_id": None,
                        "accepted_value": None,
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            }
        )

        result = _execute_patch_node_options(
            {"node_id": "llm1", "patch": {"prompt_template": "New {{ row.text }}."}},
            state,
            _ctx(),
        )

        assert result.success is True, result.data
        requirements = result.updated_state.nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY]
        # The composite LLM-review auto-stager attaches every default
        # gate; ``vague_term`` was pre-existing and copied forward, the
        # prompt template gate was added (the patched field), and the
        # model-choice gate was added because the node already has a
        # non-empty ``options.model``.
        assert [requirement["kind"] for requirement in requirements] == [
            "vague_term",
            "llm_prompt_template",
            "llm_model_choice",
        ]
        assert requirements[1]["user_term"] == "llm_prompt_template:llm1"
        assert requirements[1]["draft"] == "New {{ row.text }}."
        assert requirements[2]["user_term"] == "llm_model_choice:llm1"
        assert requirements[2]["draft"] == "anthropic/claude-haiku-4.5"

    def test_patch_rejects_unreviewed_drop_of_web_scrape_raw_fields(self) -> None:
        """A patch cannot create raw-field cleanup without the pipeline decision gate."""
        result = _execute_patch_node_options(
            {
                "node_id": "drop_raw_html",
                "patch": {
                    "mapping": {"url": "url"},
                    "select_only": True,
                },
            },
            _state_with_web_scrape_and_mapper(),
            _ctx(),
        )

        assert result.success is False
        assert "drop_raw_html_fields" in result.data["error"]
        assert "pipeline_decision" in result.data["error"]

    def test_patch_rejects_raw_cleanup_review_on_non_cleanup_node(self) -> None:
        """A raw-cleanup review belongs on the mapper that implements the drop."""
        state = CompositionState(
            source=None,
            nodes=(
                NodeSpec(
                    id="fetch_pages",
                    node_type="transform",
                    plugin="web_scrape",
                    input="rows",
                    on_success="scraped_rows",
                    on_error="discard",
                    options={
                        "schema": {"mode": "observed"},
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "content_fingerprint",
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="identify_primary_colours",
                    node_type="transform",
                    plugin="llm",
                    input="scraped_rows",
                    on_success="coloured_rows",
                    on_error="discard",
                    options={
                        "provider": "openrouter",
                        "model": "anthropic/claude-haiku-4.5",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                        "prompt_template": "Read {{ row.content }}.",
                        "schema": {"mode": "observed"},
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = _execute_patch_node_options(
            {
                "node_id": "identify_primary_colours",
                "patch": {
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "drop_raw_html_review",
                            "kind": "pipeline_decision",
                            "user_term": "drop_raw_html_fields",
                            "status": "pending",
                            "draft": "Drop the scraped raw HTML and fingerprint fields before saving the JSON output.",
                        }
                    ]
                },
            },
            state,
            _ctx(),
        )

        assert result.success is False
        assert "raw-html cleanup decision" in result.data["error"]
        assert "must be implemented by a field_mapper" in result.data["error"]

    def test_manifest_entry_is_type_driven(self) -> None:
        assert "patch_node_options" in MANIFEST
        assert MANIFEST["patch_node_options"].argument_model is PatchNodeOptionsArgumentsModel

    def test_redact_collapses_patch_to_string_summary(self) -> None:
        """The Sensitive marker on 'patch' collapses the dict to a string via
        the summarizer.  ``node_id`` is non-sensitive and passes through verbatim.
        """
        raw_args = {"node_id": "t1", "patch": {"api_key": "secret-ref", "prompt_template": "prompt-text"}}
        redacted = redact_tool_call_arguments(
            "patch_node_options",
            raw_args,
            telemetry=NoopRedactionTelemetry(),
        )
        assert "patch" in redacted
        # The summarizer collapses the dict to a string, not a nested dict.
        assert isinstance(redacted["patch"], str), "patch field should be a string summary, not a raw dict"
        # node_id is non-sensitive — passes through verbatim
        assert redacted.get("node_id") == "t1"


# ---------------------------------------------------------------------------
# patch_output_options
# ---------------------------------------------------------------------------


class TestPromotePatchOutputOptionsArgErrorRouting:
    """Validates ARG_ERROR routing for bad patch_output_options arguments."""

    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """Missing both required fields triggers ToolArgumentError."""
        state = _state_with_output()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_output_options({}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_missing_patch_raises_tool_argument_error(self) -> None:
        """sink_name alone is insufficient; patch is required."""
        state = _state_with_output()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_output_options({"sink_name": "out1"}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_missing_sink_name_raises_tool_argument_error(self) -> None:
        """patch alone is insufficient; sink_name is required."""
        state = _state_with_output()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_output_options({"patch": {"path": "/x"}}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_patch_not_dict_raises_tool_argument_error(self) -> None:
        """Non-dict patch value rejected by Pydantic model."""
        state = _state_with_output()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_output_options({"sink_name": "out1", "patch": "x"}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_sink_name_not_str_raises_tool_argument_error(self) -> None:
        """Non-string sink_name rejected by Pydantic model."""
        state = _state_with_output()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_output_options({"sink_name": 123, "patch": {}}, state, _ctx())
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_extra_field_raises_tool_argument_error(self) -> None:
        """extra='forbid' rejects stray fields at Tier-3."""
        state = _state_with_output()
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_patch_output_options(
                {"sink_name": "out1", "patch": {}, "node_id": "stray"},
                state,
                _ctx(),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_valid_arguments_dispatch_normally(self) -> None:
        """Valid sink_name + patch dict succeeds when output exists in state."""
        state = _state_with_output(
            "out1",
            {"path": "/tmp/out.csv", "schema": {"mode": "observed"}},
        )
        result = _execute_patch_output_options(
            {"sink_name": "out1", "patch": {"encoding": "utf-8"}},
            state,
            _ctx(),
        )
        assert result.success is True

    def test_manifest_entry_is_type_driven(self) -> None:
        assert "patch_output_options" in MANIFEST
        assert MANIFEST["patch_output_options"].argument_model is PatchOutputOptionsArgumentsModel

    def test_redact_collapses_patch_to_string_summary(self) -> None:
        """The Sensitive marker on 'patch' collapses the dict to a string via
        the summarizer.  ``sink_name`` is non-sensitive and passes through verbatim.
        """
        raw_args = {"sink_name": "out1", "patch": {"api_key": "secret-ref", "path": "/private/out.json"}}
        redacted = redact_tool_call_arguments(
            "patch_output_options",
            raw_args,
            telemetry=NoopRedactionTelemetry(),
        )
        assert "patch" in redacted
        # The summarizer collapses the dict to a string, not a nested dict.
        assert isinstance(redacted["patch"], str), "patch field should be a string summary, not a raw dict"
        # sink_name is non-sensitive — passes through verbatim
        assert redacted.get("sink_name") == "out1"
