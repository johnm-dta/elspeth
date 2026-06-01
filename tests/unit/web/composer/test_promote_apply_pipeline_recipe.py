"""ARG_ERROR routing + redaction for ``apply_pipeline_recipe`` (Task 14 / Wave 3).

Wave 3 sub-task 2/2 (final).  Same discipline as
``test_promote_set_pipeline.py`` (sub-task 1) and the four Task 4 / Wave 2
promotion tests.  Rev-2 BLOCKER_A applies: promoted handlers MUST catch
:class:`pydantic.ValidationError` and re-raise as
:class:`ToolArgumentError`.  A bare ``ValidationError`` escaping the
handler hits ``service.py:2564`` (→ :class:`ComposerPluginCrashError` →
HTTP 500) — wrong disposition for Tier-3 input.

Tests pin:
  * manifest shape (type-driven),
  * exception-class + ``__cause__`` chain on invalid arguments,
  * ``extra="forbid"`` rejects misrouted argument shapes,
  * empty-string ``recipe_name`` flows through to the recoverable
    ``_failure_result`` repair-hint branch (two channels for two
    failure shapes: type-error vs semantic-empty),
  * :func:`redact_tool_call_arguments` collapses the ``slots`` dict
    via the shared :func:`_summarize_set_source_options` summarizer.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    MANIFEST,
    REDACTED_BLOB_SOURCE_PATH,
    ApplyPipelineRecipeArgumentsModel,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import _execute_apply_pipeline_recipe
from elspeth.web.composer.tools._common import ToolContext


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _mock_catalog() -> MagicMock:
    return MagicMock(spec=CatalogService)


# ---------------------------------------------------------------------------
# Manifest shape pin
# ---------------------------------------------------------------------------


def test_apply_pipeline_recipe_manifest_entry_is_type_driven() -> None:
    entry = MANIFEST["apply_pipeline_recipe"]
    assert entry.argument_model is ApplyPipelineRecipeArgumentsModel
    assert entry.policy is None


# ---------------------------------------------------------------------------
# ARG_ERROR routing — bare ValidationError must NOT escape the handler
# ---------------------------------------------------------------------------


class TestPromoteApplyPipelineRecipeArgErrorRouting:
    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """A bare ``{}`` is missing both required fields (recipe_name, slots)."""
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_apply_pipeline_recipe({}, _empty_state(), ToolContext(catalog=_mock_catalog()))
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_missing_slots_raises_tool_argument_error(self) -> None:
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_apply_pipeline_recipe(
                {"recipe_name": "classify-rows-llm-jsonl"},
                _empty_state(),
                ToolContext(catalog=_mock_catalog()),
            )
        cause = exc_info.value.__cause__
        assert isinstance(cause, PydanticValidationError)
        assert any(err["loc"] == ("slots",) for err in cause.errors())

    def test_non_dict_slots_raises_tool_argument_error(self) -> None:
        """Pydantic rejects ``slots: str`` before recipe lookup."""
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_apply_pipeline_recipe(
                {"recipe_name": "classify-rows-llm-jsonl", "slots": "not a dict"},
                _empty_state(),
                ToolContext(catalog=_mock_catalog()),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_extra_field_raises_tool_argument_error(self) -> None:
        """extra='forbid' rejects fields belonging to neighbouring tools."""
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_apply_pipeline_recipe(
                {
                    "recipe_name": "classify-rows-llm-jsonl",
                    "slots": {},
                    "source": {"plugin": "csv"},  # belongs on set_pipeline
                },
                _empty_state(),
                ToolContext(catalog=_mock_catalog()),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_empty_recipe_name_flows_to_semantic_failure_branch(self) -> None:
        """Empty ``recipe_name: ""`` passes Pydantic but the handler returns
        a recoverable ``_failure_result`` with the list_recipes repair hint.

        Two channels for two failure shapes: type-error (Pydantic ARG_ERROR
        with leak-safe envelope text) vs semantic-empty (recipe lookup
        failure with a discovery-tool pointer).  The post-promotion handler
        must continue producing the repair-hinting failure result, NOT a
        bare ToolArgumentError for the empty-string case.
        """
        result = _execute_apply_pipeline_recipe(
            {"recipe_name": "", "slots": {}},
            _empty_state(),
            ToolContext(catalog=_mock_catalog()),
        )
        assert result.success is False
        assert result.data is not None
        assert "list_recipes" in result.data["error"]
        assert "non-empty 'recipe_name'" in result.data["error"]

    def test_unknown_recipe_name_returns_failure_result(self) -> None:
        """Unknown ``recipe_name`` flows through to ``apply_recipe`` and
        surfaces as a recoverable ``_failure_result``.

        This pins the post-promotion contract: argument-shape errors are
        ToolArgumentError; recipe-lookup errors are recoverable failures.
        """
        result = _execute_apply_pipeline_recipe(
            {"recipe_name": "no-such-recipe", "slots": {}},
            _empty_state(),
            ToolContext(catalog=_mock_catalog()),
        )
        assert result.success is False
        assert result.data is not None

    def test_valid_arguments_dispatch_normally(self) -> None:
        """Functional smoke: a valid call reaches the recipe-application path.

        Drives ``classify-rows-llm-jsonl`` with the minimum slot set so
        the handler descends into ``apply_recipe`` → ``_execute_set_pipeline``.
        The inner ``set_pipeline`` call exercises the Wave-3 Pydantic
        validation chain (the recipe-built args must conform to
        :class:`SetPipelineArgumentsModel`).
        """
        # The classify-rows-llm-jsonl recipe takes a blob_id + plugin
        # config + an LLM template.  The set_pipeline inner call will
        # validate the recipe output but may fail downstream on the
        # mocked catalog's empty schema map; that is acceptable for
        # this functional smoke — the assertion is that a structured
        # ToolResult comes back (not an exception).
        result = _execute_apply_pipeline_recipe(
            {
                "recipe_name": "classify-rows-llm-jsonl",
                "slots": {
                    "source_blob_id": "00000000-0000-0000-0000-000000000001",
                    "provider": "openrouter/",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "classifier_template": "Classify: {text}",
                    "label_field": "category",
                    "required_input_fields": ["text"],
                    "output_path": "outputs/labelled.jsonl",
                },
            },
            _empty_state(),
            ToolContext(catalog=_mock_catalog()),
        )
        # The recipe → set_pipeline path may succeed or produce a
        # _failure_result depending on the mock catalog's plugin
        # registrations — the post-promotion assertion is purely that
        # the call completes WITHOUT raising (no bare ValidationError
        # leaks; no ToolArgumentError on the system-built args path).
        # ``result.success`` may be either True or False; what matters
        # is the structured response.
        assert result is not None


# ---------------------------------------------------------------------------
# Redaction at the persistence boundary
# ---------------------------------------------------------------------------


_CANARY_PATH = "CANARY-APPLY-RECIPE-OUTPUT-PATH-DO-NOT-LEAK"
_CANARY_TEMPLATE = "CANARY-APPLY-RECIPE-TEMPLATE-DO-NOT-LEAK"


def test_redaction_substitutes_slots_via_summarizer() -> None:
    """``slots`` collapses to the summarizer's canonical-JSON output.

    :func:`_summarize_set_source_options` applies
    :func:`redact_source_storage_path` to the wrapped ``{"source":
    {"options": slots}}`` shape: when ``slots`` contains both ``path``
    and ``blob_ref``, the path is substituted with
    :data:`REDACTED_BLOB_SOURCE_PATH`.  Other keys pass through verbatim
    inside the JSON-string output.
    """
    tel = NoopRedactionTelemetry()
    args = {
        "recipe_name": "classify-rows-llm-jsonl",
        "slots": {
            "path": _CANARY_PATH,
            "blob_ref": "00000000-0000-0000-0000-000000000001",
        },
    }
    redacted = redact_tool_call_arguments("apply_pipeline_recipe", args, telemetry=tel)
    # recipe_name passes through verbatim (operator-controlled).
    assert redacted["recipe_name"] == "classify-rows-llm-jsonl"
    # slots collapses to the summarizer str output.
    assert isinstance(redacted["slots"], str)
    assert REDACTED_BLOB_SOURCE_PATH in redacted["slots"]
    # The canary path MUST NOT appear in the redacted output.
    serialized = json.dumps(redacted, sort_keys=True)
    assert _CANARY_PATH not in serialized
    # Telemetry recorded the manifest dispatch with the type-driven shape.
    assert tel.manifest_dispatch_calls == [{"tool_name": "apply_pipeline_recipe", "shape": "type_driven"}]


def test_redaction_preserves_non_path_slot_keys_inside_summary() -> None:
    """Non-path slot keys pass through the summarizer verbatim by design.

    The path-redacting summarizer (:func:`redact_source_storage_path`) is
    content-agnostic — it acts only when both ``path`` AND ``blob_ref``
    are present.  Other slot keys (``provider``, ``model``,
    ``classifier_template``) appear verbatim inside the summarizer's
    canonical-JSON output string.  This is acceptable for Wave 3: the
    structural Sensitive marker prevents un-redacted dict materialisation
    on the audit boundary, and Task 16 may introduce a recipe-aware
    summarizer that also redacts the template string.
    """
    tel = NoopRedactionTelemetry()
    args: dict[str, Any] = {
        "recipe_name": "classify-rows-llm-jsonl",
        "slots": {
            "classifier_template": _CANARY_TEMPLATE,
            "provider": "openrouter/",
        },
    }
    redacted = redact_tool_call_arguments("apply_pipeline_recipe", args, telemetry=tel)
    # slots is the summarizer's str output (a JSON object string).
    assert isinstance(redacted["slots"], str)
    # The template canary appears EXACTLY ONCE — inside the slots
    # summary string — and NOT at any other surface.
    serialized = json.dumps(redacted, sort_keys=True)
    assert serialized.count(_CANARY_TEMPLATE) == 1
    # No path-blob_ref pair => no REDACTED_BLOB_SOURCE_PATH substitution
    # (mirroring the Task 4 set_source pass-through pin).
    assert REDACTED_BLOB_SOURCE_PATH not in redacted["slots"]


def test_apply_recipe_crashes_on_set_pipeline_contract_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offensive-programming pin: if ``_execute_set_pipeline`` ever returns
    ``ToolResult.data`` that is neither ``None`` nor a ``dict``, the
    apply-recipe merge path crashes naturally with a ``TypeError`` from the
    dict-spread, surfacing the system-code bug rather than silently wrapping
    garbage.

    Replaces the pre-existing silent-wrap fallback flagged by the 2026-05-23
    multi-agent review: the prior code wrapped contract-violating data into
    ``{"replaced_pipeline_note": ..., "set_pipeline_data": <garbage>}`` and
    returned success, which would have surfaced confidently-wrong audit data
    to the LLM. Per CLAUDE.md "Plugin returns wrong type → CRASH"; the
    ``data`` shape is a system-code contract (``dict | None`` per the
    annotation at the success-path construction site), not Tier-3 LLM input.

    The merge previously routed the contract violation through a hand-rolled
    ``isinstance``/``AssertionError`` guard; the tier-model burndown removed
    that defensive guard (a flagged ``isinstance`` on our own authored value)
    so the contract violation now crashes the way direct access always does —
    a ``TypeError`` when the non-mapping ``data`` is spread into a dict.

    The non-empty pre-state forces the success+annotation branch — see the
    suppress-note guard at the head of the merge block. Both ``apply_recipe``
    and ``_execute_set_pipeline`` are stubbed so the test isolates the merge
    contract from the recipe catalog and pipeline-validation surfaces.
    """
    from dataclasses import replace as _replace

    from elspeth.web.composer.state import SourceSpec
    from elspeth.web.composer.tools import _common as tools_common
    from elspeth.web.composer.tools import sessions as tools_sessions

    # Pre-state has a source so ``pre_source_present`` is truthy and the
    # annotation branch is reached. Use a real SourceSpec so the dataclass
    # validation in CompositionState.__post_init__ is satisfied.
    pre_state = CompositionState(
        source=SourceSpec(plugin="csv", on_success="out", options={}, on_validation_failure="quarantine"),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="pre-existing-pipeline"),
        version=1,
    )
    post_state = _replace(pre_state, version=pre_state.version + 1)

    # Stub apply_recipe so the test does not depend on the recipe catalog.
    # Returns a dict the (stubbed) _execute_set_pipeline will ignore.
    monkeypatch.setattr(
        tools_sessions,
        "apply_recipe",
        lambda name, slots: {"source": {"plugin": "csv"}, "nodes": [], "outputs": []},
    )

    def fake_set_pipeline(args: Any, state: Any, context: Any) -> Any:
        # Return a ToolResult whose ``data`` violates the dict-or-None
        # contract — this is the "plugin returned wrong type" scenario.
        return tools_common.ToolResult(
            success=True,
            updated_state=post_state,
            validation=post_state.validate(),
            affected_nodes=("source",),
            data=42,  # int — neither dict nor None
        )

    monkeypatch.setattr(tools_sessions, "_execute_set_pipeline", fake_set_pipeline)

    with pytest.raises(TypeError) as exc_info:
        _execute_apply_pipeline_recipe(
            {"recipe_name": "anything", "slots": {}},
            pre_state,
            ToolContext(catalog=_mock_catalog()),
        )

    msg = str(exc_info.value)
    assert "mapping" in msg.lower()
