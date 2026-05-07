"""Unit tests for the JSON-Schema required-path walker.

The walker compiles tool-parameter schemas into ``_CompiledRequiredPath``
records that the pre-dispatch validator uses to reject malformed LLM tool
calls before the handler runs. Correctness here is a Tier-1 audit boundary:
a faulty walker either crashes ordinary calls (the elspeth-4e79436719 Bug A
regression) or silently accepts malformed payloads.

These tests target the private functions directly because the dispatch-level
tests in ``test_service.py`` only exercise the path through ``set_pipeline``.
The walker itself must be correct for every shape of nested schema we may
introduce in the future.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.service import (
    _ARRAY_ITEM_SEGMENT,
    _collect_required_paths,
    _CompiledRequiredPath,
    _find_missing_required_paths,
    _optional_ancestor_present,
)


class TestCollectRequiredPathsSemantics:
    def test_top_level_required_field_emits_unconditional_path(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        compiled = _collect_required_paths(schema)
        assert compiled == (_CompiledRequiredPath(path=("a",), optional_ancestor=()),)

    def test_nested_required_when_parent_required_is_unconditional(self) -> None:
        """Path is required at every level → optional_ancestor stays empty."""
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {"a1": {"type": "string"}},
                    "required": ["a1"],
                }
            },
            "required": ["a"],
        }
        compiled = _collect_required_paths(schema)
        # Two paths: ("a",) and ("a", "a1"). Both unconditional.
        paths = {p.path: p.optional_ancestor for p in compiled}
        assert paths[("a",)] == ()
        assert paths[("a", "a1")] == ()

    def test_nested_required_when_parent_optional_is_conditional(self) -> None:
        """The elspeth-4e79436719 Bug A regression case.

        Parent ``a`` is NOT in the outer ``required`` list, so its inner
        required field ``a1`` becomes a conditional-on-presence check.
        """
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {"a1": {"type": "string"}},
                    "required": ["a1"],
                }
            },
            # No outer ``required`` at all — ``a`` is optional.
        }
        compiled = _collect_required_paths(schema)
        # Only path emitted: ("a", "a1") with optional_ancestor=("a",).
        assert compiled == (_CompiledRequiredPath(path=("a", "a1"), optional_ancestor=("a",)),)

    def test_doubly_nested_optional_uses_deepest_optional_ancestor(self) -> None:
        """Two layers of optionality — ancestor walks to the deepest optional."""
        schema = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "object",
                    "properties": {
                        "y": {
                            "type": "object",
                            "properties": {"z": {"type": "string"}},
                            "required": ["z"],
                        }
                    },
                    # y is optional inside x → y becomes the deeper ancestor.
                },
            },
            "required": ["x"],
        }
        compiled = _collect_required_paths(schema)
        paths = {p.path: p.optional_ancestor for p in compiled}
        assert paths[("x",)] == ()
        # x is required at the outer level so propagation says "no optional
        # ancestor seen yet" when we step into x. Stepping into y (which is
        # NOT in x's required) records y as the optional ancestor.
        assert paths[("x", "y", "z")] == ("x", "y")

    def test_array_items_required_propagates_through_segment(self) -> None:
        """Array-item required fields stay required when the array is required."""
        schema = {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                }
            },
            "required": ["nodes"],
        }
        compiled = _collect_required_paths(schema)
        paths = {p.path: p.optional_ancestor for p in compiled}
        assert paths[("nodes",)] == ()
        assert paths[("nodes", "[]", "id")] == ()


class TestFindMissingRequiredPaths:
    def test_absent_optional_ancestor_short_circuits(self) -> None:
        """An optional object that is absent must NOT trigger inner-required errors."""
        compiled = (
            _CompiledRequiredPath(
                path=("source", "inline_blob", "filename"),
                optional_ancestor=("source", "inline_blob"),
            ),
        )
        # ``inline_blob`` is absent → no error.
        value = {"source": {"plugin": "csv"}}
        assert _find_missing_required_paths(value, compiled) == []

    def test_present_optional_ancestor_enforces_inner_required(self) -> None:
        """An optional object that IS present DOES trigger inner-required errors."""
        compiled = (
            _CompiledRequiredPath(
                path=("source", "inline_blob", "filename"),
                optional_ancestor=("source", "inline_blob"),
            ),
        )
        value = {"source": {"plugin": "csv", "inline_blob": {"mime_type": "text/csv"}}}
        assert _find_missing_required_paths(value, compiled) == ["source.inline_blob.filename"]

    def test_unconditional_path_always_enforced(self) -> None:
        """Empty optional_ancestor → unconditional check."""
        compiled = (_CompiledRequiredPath(path=("source", "plugin"), optional_ancestor=()),)
        value = {"source": {"options": {}}}
        assert _find_missing_required_paths(value, compiled) == ["source.plugin"]

    def test_array_item_paths_walk_each_index(self) -> None:
        """Array iteration produces indexed paths for each missing field."""
        compiled = (_CompiledRequiredPath(path=("nodes", "[]", "id"), optional_ancestor=()),)
        value = {"nodes": [{"id": "a"}, {}, {"id": "c"}, {}]}
        missing = _find_missing_required_paths(value, compiled)
        assert missing == ["nodes[1].id", "nodes[3].id"]


class TestSetPipelineCompiledIndex:
    """Sanity checks on the compiled index for ``set_pipeline``.

    Locks in the per-tool result so a future schema change that re-introduces
    the elspeth-4e79436719 Bug A regression fails this test directly.
    """

    @pytest.fixture
    def compiled(self) -> tuple[_CompiledRequiredPath, ...]:
        from elspeth.web.composer.service import _TOOL_REQUIRED_PATHS

        return _TOOL_REQUIRED_PATHS["set_pipeline"]

    def test_inline_blob_inner_paths_are_conditional(self, compiled: tuple[_CompiledRequiredPath, ...]) -> None:
        inline_paths = [p for p in compiled if "inline_blob" in p.path]
        assert len(inline_paths) == 3
        for p in inline_paths:
            assert p.optional_ancestor == ("source", "inline_blob"), (
                f"inline_blob inner path {p.path!r} must be conditional on the "
                "presence of source.inline_blob; otherwise the elspeth-4e79436719 "
                "Bug A regression has returned."
            )

    def test_top_level_source_fields_are_unconditional(self, compiled: tuple[_CompiledRequiredPath, ...]) -> None:
        unconditional_paths = {p.path for p in compiled if not p.optional_ancestor}
        assert ("source",) in unconditional_paths
        assert ("source", "plugin") in unconditional_paths
        assert ("source", "on_success") in unconditional_paths
        assert ("source", "options") not in unconditional_paths

    def test_set_pipeline_options_omissions_reach_handler_feedback(
        self,
        compiled: tuple[_CompiledRequiredPath, ...],
    ) -> None:
        arguments = {
            "source": {"plugin": "csv", "on_success": "source_out"},
            "nodes": [],
            "edges": [],
            "outputs": [{"sink_name": "main", "plugin": "json"}],
        }

        missing = _find_missing_required_paths(arguments, compiled)

        assert "source.options" not in missing
        assert "outputs[0].options" not in missing

    def test_array_item_required_fields_are_unconditional(self, compiled: tuple[_CompiledRequiredPath, ...]) -> None:
        unconditional_paths = {p.path for p in compiled if not p.optional_ancestor}
        assert ("nodes", "[]", "id") in unconditional_paths
        assert ("edges", "[]", "edge_type") in unconditional_paths
        assert ("outputs", "[]", "sink_name") in unconditional_paths
        assert ("outputs", "[]", "options") not in unconditional_paths


class TestOptionalAncestorPresentRefusesArraySegment:
    """Offensive guard: a future schema that introduces optional sub-objects
    inside array items would cause ``_collect_required_paths`` to emit array
    segments inside ``optional_ancestor``. ``_optional_ancestor_present`` does
    NOT yet support per-array-item ancestor evaluation; silently treating the
    array as "present" would produce wrong validation results.

    Per CLAUDE.md ("Defensive Programming: Forbidden. Offensive Programming:
    Encouraged"), the walker raises ``AssertionError`` with a diagnostic
    pointing the next maintainer at the exact extension needed, rather than
    falling through and emitting incorrect missing-required-paths results.
    """

    def test_array_segment_in_optional_ancestor_raises_with_diagnostic(self) -> None:
        with pytest.raises(AssertionError, match="optional_ancestor"):
            _optional_ancestor_present({"x": []}, ("x", _ARRAY_ITEM_SEGMENT))

    def test_diagnostic_message_names_the_extension_point(self) -> None:
        """The error message must point at ``_find_missing_required_paths``
        (the function that would need per-item ancestor evaluation), not
        at a generic 'unsupported' string — the next maintainer should be
        able to find the extension site without grepping.
        """
        with pytest.raises(AssertionError) as exc_info:
            _optional_ancestor_present({"y": [{"z": 1}]}, ("y", _ARRAY_ITEM_SEGMENT, "z"))
        message = str(exc_info.value)
        assert "_find_missing_required_paths" in message or "per-array-item" in message, (
            f"Diagnostic message must point at the extension site. Got: {message!r}"
        )

    def test_pure_object_ancestor_still_works(self) -> None:
        """Sanity: the offensive guard only fires for array segments — pure
        object ancestors must keep working.
        """
        assert _optional_ancestor_present({"a": {"b": 1}}, ("a", "b")) is True
        assert _optional_ancestor_present({"a": {}}, ("a", "b")) is False
        assert _optional_ancestor_present({}, ("a",)) is False
