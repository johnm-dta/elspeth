# tests/property/core/test_connection_name_fuzz.py
"""Fuzz tests for connection name validation boundaries.

Connection names flow into node IDs and Landscape audit records (Tier 1 data).
The validation boundary must reject dangerous inputs and ensure accepted names
produce safe, deterministic, unique node IDs.

Validation entry points:
- _validate_connection_or_sink_name() in config.py (length, chars, reserved)
- _validate_node_name_chars() in config.py (character class restriction)
- Node ID construction in dag.py (length truncation check)
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from elspeth.core.config import (
    CoalesceSettings,
    ElspethSettings,
    GateSettings,
    SinkSettings,
    SourceSettings,
    TransformSettings,
)
from elspeth.core.dag import ExecutionGraph

# =============================================================================
# Constants mirrored from config.py for property assertions
# =============================================================================

_VALID_CONNECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]*$")
_MAX_CONNECTION_NAME_LENGTH = 64
_RESERVED_EDGE_LABELS = frozenset({"continue", "fork", "on_success"})

# =============================================================================
# Strategies
# =============================================================================

# Valid connection names: start with letter/digit/underscore, then alnum/underscore/hyphen
valid_connection_names = st.from_regex(r"^[a-zA-Z][a-zA-Z0-9_-]{0,30}$", fullmatch=True).filter(
    lambda s: s not in _RESERVED_EDGE_LABELS and not s.startswith("__")
)

# Dangerous strings that should be rejected
dangerous_strings = st.sampled_from(
    [
        "",  # empty
        " ",  # whitespace only
        "  \t\n  ",  # mixed whitespace
        "\x00",  # NUL byte
        "name\x00evil",  # embedded NUL
        "continue",  # reserved
        "fork",  # reserved
        "on_success",  # reserved
        "__error_sink__",  # system-reserved prefix
        "__internal",  # system-reserved prefix
        "../../../etc/passwd",  # path traversal
        "a" * 65,  # exceeds max length
        "a" * 100,  # way over limit
        "'; DROP TABLE nodes; --",  # SQL injection
        "name<script>alert(1)</script>",  # XSS
        "name\ninjected",  # newline injection
        "name\rinjected",  # carriage return injection
        "\u200b",  # zero-width space
        "\u200d",  # zero-width joiner
        "\u202e",  # RTL override
        "\u0300",  # combining grave accent
        "café",  # non-ASCII (accented char)
        "名前",  # CJK characters
        "💀",  # emoji
        "a b",  # space in middle
        ".hidden",  # dot-prefixed
        "@mention",  # at-sign
        "name=value",  # equals sign
        "name;other",  # semicolon
        "name|other",  # pipe
    ]
)

# Unicode edge cases from Hypothesis
unicode_edge_cases = st.text(
    alphabet=st.characters(
        blacklist_categories=("L", "N"),
        blacklist_characters="_-",
    ),
    min_size=1,
    max_size=10,
)


# =============================================================================
# Rejection tests — dangerous inputs must be rejected
# =============================================================================


class TestConnectionNameRejection:
    """Verify that dangerous connection names are rejected at the settings level."""

    @given(name=dangerous_strings)
    @settings(max_examples=50)
    def test_dangerous_names_rejected_as_transform_input(self, name: str) -> None:
        """Dangerous names must be rejected when used as transform input connections."""
        with pytest.raises((ValueError, ValidationError)):
            TransformSettings(
                name="safe_transform",
                plugin="passthrough",
                input=name,
                on_success="output",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
            )

    @given(name=dangerous_strings)
    @settings(max_examples=50)
    def test_dangerous_names_rejected_as_transform_on_success(self, name: str) -> None:
        """Dangerous names must be rejected when used as transform on_success."""
        with pytest.raises((ValueError, ValidationError)):
            TransformSettings(
                name="safe_transform",
                plugin="passthrough",
                input="safe_input",
                on_success=name,
                on_error="discard",
                options={"schema": {"mode": "observed"}},
            )

    @given(name=dangerous_strings)
    @settings(max_examples=50)
    def test_dangerous_names_rejected_as_source_on_success(self, name: str) -> None:
        """Dangerous names must be rejected when used as source on_success."""
        with pytest.raises((ValueError, ValidationError)):
            SourceSettings(
                plugin="csv",
                on_success=name,
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            )

    @given(name=unicode_edge_cases)
    @settings(max_examples=100)
    def test_unicode_edge_cases_rejected(self, name: str) -> None:
        """Unicode characters outside [a-zA-Z0-9_-] must be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            TransformSettings(
                name="safe_transform",
                plugin="passthrough",
                input=name,
                on_success="output",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
            )

    @given(name=dangerous_strings)
    @settings(max_examples=50)
    def test_dangerous_names_rejected_as_route_label(self, name: str) -> None:
        """Dangerous names must be rejected when used as gate route labels."""
        with pytest.raises((ValueError, ValidationError)):
            GateSettings(
                name="safe_gate",
                input="safe_input",
                condition="True",
                routes={name: "some_sink", "false": "some_sink"},
            )

    @given(name=dangerous_strings)
    @settings(max_examples=50)
    def test_dangerous_names_rejected_as_fork_branch(self, name: str) -> None:
        """Dangerous names must be rejected when used as fork branch names."""
        with pytest.raises((ValueError, ValidationError)):
            GateSettings(
                name="safe_gate",
                input="safe_input",
                condition="True",
                routes={"true": "fork", "false": "some_sink"},
                fork_to=[name],
            )

    @given(name=dangerous_strings)
    @settings(max_examples=50)
    def test_dangerous_names_rejected_as_coalesce_branch(self, name: str) -> None:
        """Dangerous names must be rejected when used as coalesce branch names."""
        with pytest.raises((ValueError, ValidationError)):
            CoalesceSettings(
                name="safe_coalesce",
                branches=[name, "valid_branch"],
                on_success="output",
            )

    @given(name=dangerous_strings)
    @settings(max_examples=50)
    def test_dangerous_names_rejected_as_sink_name(self, name: str) -> None:
        """Dangerous names must be rejected when used as sink name keys."""
        with pytest.raises((ValueError, ValidationError)):
            ElspethSettings(
                source=SourceSettings(
                    plugin="csv",
                    on_success="safe_out",
                    options={
                        "path": "test.csv",
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                ),
                sinks={
                    name: SinkSettings(
                        plugin="json",
                        options={"path": "output.json", "schema": {"mode": "observed"}},
                    ),
                },
                transforms=[
                    TransformSettings(
                        name="t1",
                        plugin="passthrough",
                        input="safe_out",
                        on_success=name if name else "fallback",
                        on_error="discard",
                        options={"schema": {"mode": "observed"}},
                    ),
                ],
            )


# =============================================================================
# Whitespace stripping tests — validators must return stripped values
# =============================================================================


class TestWhitespaceStripping:
    """Verify that fork_to and coalesce branches strip whitespace (1wbw fix)."""

    def test_fork_to_strips_whitespace(self) -> None:
        """fork_to validator must return stripped branch names."""
        gate = GateSettings(
            name="forker",
            input="safe_input",
            condition="True",
            routes={"true": "fork", "false": "some_sink"},
            fork_to=["  branch_a  ", " branch_b"],
        )
        assert gate.fork_to == ["branch_a", "branch_b"]

    def test_coalesce_branches_strips_whitespace(self) -> None:
        """Coalesce branch validator must return stripped branch names."""
        coal = CoalesceSettings(
            name="merger",
            branches=["  branch_a  ", " branch_b"],
            on_success="output",
        )
        assert coal.branches == {"branch_a": "branch_a", "branch_b": "branch_b"}


# =============================================================================
# Acceptance tests — valid names produce safe node IDs
# =============================================================================


class TestConnectionNameAcceptance:
    """Verify that valid connection names are accepted and produce safe node IDs."""

    @given(name=valid_connection_names)
    @settings(max_examples=100)
    def test_valid_names_accepted_as_transform_input(self, name: str) -> None:
        """Valid connection names must be accepted by transform input validation."""
        # Should not raise
        t = TransformSettings(
            name="safe_transform",
            plugin="passthrough",
            input=name,
            on_success="output",
            on_error="discard",
            options={"schema": {"mode": "observed"}},
        )
        assert t.input == name

    @given(name=valid_connection_names)
    @settings(max_examples=100)
    def test_valid_names_accepted_as_source_on_success(self, name: str) -> None:
        """Valid connection names must be accepted by source on_success validation."""
        s = SourceSettings(
            plugin="csv",
            on_success=name,
            options={
                "path": "test.csv",
                "on_validation_failure": "discard",
                "schema": {"mode": "observed"},
            },
        )
        assert s.on_success == name


# =============================================================================
# Roundtrip tests — valid names through from_plugin_instances()
# =============================================================================


class TestConnectionNameRoundtrip:
    """Verify valid names survive full DAG construction roundtrip."""

    @given(name=valid_connection_names)
    @settings(max_examples=50, deadline=5000)
    def test_valid_connection_name_roundtrip(self, name: str) -> None:
        """Valid connection names produce correct edges in execution graph."""
        from elspeth.cli_helpers import instantiate_plugins_from_config

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success=name,
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            transforms=[
                TransformSettings(
                    name="t1",
                    plugin="passthrough",
                    input=name,
                    on_success="output",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        # Graph should have: source -> t1 -> output_sink
        assert graph.node_count == 3
        assert graph.edge_count == 2

        # All node IDs should be within column length
        from elspeth.core.landscape.schema import NODE_ID_COLUMN_LENGTH

        for node_info in graph.get_nodes():
            assert len(node_info.node_id) <= NODE_ID_COLUMN_LENGTH

    @given(name=valid_connection_names)
    @settings(max_examples=50, deadline=5000)
    def test_node_ids_are_database_safe(self, name: str) -> None:
        """Node IDs produced from valid connection names contain no NUL bytes."""
        from elspeth.cli_helpers import instantiate_plugins_from_config

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success=name,
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
            transforms=[
                TransformSettings(
                    name="t1",
                    plugin="passthrough",
                    input=name,
                    on_success="output",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
        )

        for node_info in graph.get_nodes():
            # No NUL bytes (would cause SQLite storage issues)
            assert "\x00" not in node_info.node_id
            # Valid UTF-8 (always true for Python str, but explicit)
            node_info.node_id.encode("utf-8")
