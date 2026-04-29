"""Skill content drift checks — verify skill files match live codebase.

These tests catch silent divergence between the static markdown skill
files and the actual plugin implementations / validation code.  They
fail in CI when:

- A plugin is added/removed/renamed without updating both skill files
- A validation warning/suggestion is added without updating the glossary
- The web skill and Claude Code skill list different plugin sets

These are contract tests, not unit tests — they verify documentation
accuracy against live code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from elspeth.plugins.infrastructure.discovery import discover_all_plugins
from elspeth.web.composer.skills import load_skill

# Paths to both skill files.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CLAUDE_CODE_SKILL = _PROJECT_ROOT / ".claude" / "skills" / "pipeline-composer" / "SKILL.md"
_WEB_SKILL_CONTENT = load_skill("pipeline_composer")


def _claude_code_skill_content() -> str:
    """Load the Claude Code skill file."""
    return _CLAUDE_CODE_SKILL.read_text(encoding="utf-8")


def _extract_backtick_names(text: str, section_header: str) -> set[str]:
    """Extract backtick-quoted plugin names from a markdown table section.

    Looks for the section starting with ``section_header`` (e.g. ``### Sources``),
    then extracts all backtick-quoted names from the first column of markdown
    tables in that section (up to the next heading of equal or higher level).
    """
    # Find the section.
    pattern = re.escape(section_header)
    match = re.search(pattern, text)
    if not match:
        return set()

    # Determine heading level (count leading #).
    heading_level = len(section_header) - len(section_header.lstrip("#"))

    # Extract until next heading of equal or higher level, or end of text.
    rest = text[match.end() :]
    # Match headings with <= heading_level number of #s (e.g., ## or ### for ###).
    next_heading = re.search(r"\n#{1," + str(heading_level) + r"} ", rest)
    section_text = rest[: next_heading.start()] if next_heading else rest

    # Extract backtick-quoted names from table rows (first column after |).
    names: set[str] = set()
    for line in section_text.split("\n"):
        # Match table rows: | `name` | ... |
        row_match = re.match(r"\|\s*`([^`]+)`\s*\|", line)
        if row_match:
            names.add(row_match.group(1))
    return names


class TestPluginNameDrift:
    """Verify skill files list all registered plugins."""

    @pytest.fixture(autouse=True)
    def _discover(self) -> None:
        """Discover all plugins once for the test class."""
        discovered = discover_all_plugins()
        self.source_names = {cls.name for cls in discovered["sources"]}  # type: ignore[attr-defined]
        self.transform_names = {cls.name for cls in discovered["transforms"]}  # type: ignore[attr-defined]
        self.sink_names = {cls.name for cls in discovered["sinks"]}  # type: ignore[attr-defined]

    def test_web_skill_lists_all_sources(self) -> None:
        """Every registered source plugin appears in the web skill."""
        skill_sources = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Sources")
        missing = self.source_names - skill_sources
        assert not missing, f"Source plugins missing from web skill: {missing}"

    def test_web_skill_lists_all_transforms(self) -> None:
        """Every registered transform plugin appears in the web skill."""
        skill_transforms = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Transforms")
        missing = self.transform_names - skill_transforms
        assert not missing, f"Transform plugins missing from web skill: {missing}"

    def test_web_skill_lists_all_sinks(self) -> None:
        """Every registered sink plugin appears in the web skill."""
        skill_sinks = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Sinks")
        missing = self.sink_names - skill_sinks
        assert not missing, f"Sink plugins missing from web skill: {missing}"

    def test_web_skill_has_no_phantom_sources(self) -> None:
        """Web skill does not list source plugins that don't exist."""
        skill_sources = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Sources")
        phantom = skill_sources - self.source_names
        assert not phantom, f"Phantom source plugins in web skill (not registered): {phantom}"

    def test_web_skill_has_no_phantom_transforms(self) -> None:
        """Web skill does not list transform plugins that don't exist."""
        skill_transforms = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Transforms")
        phantom = skill_transforms - self.transform_names
        assert not phantom, f"Phantom transform plugins in web skill (not registered): {phantom}"

    def test_web_skill_has_no_phantom_sinks(self) -> None:
        """Web skill does not list sink plugins that don't exist."""
        skill_sinks = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Sinks")
        phantom = skill_sinks - self.sink_names
        assert not phantom, f"Phantom sink plugins in web skill (not registered): {phantom}"

    @pytest.mark.skipif(not _CLAUDE_CODE_SKILL.exists(), reason="Claude Code skill not found")
    def test_claude_code_skill_lists_all_sources(self) -> None:
        """Every registered source plugin appears in the Claude Code skill."""
        cc_content = _claude_code_skill_content()
        skill_sources = _extract_backtick_names(cc_content, "### Sources")
        missing = self.source_names - skill_sources
        assert not missing, f"Source plugins missing from Claude Code skill: {missing}"

    @pytest.mark.skipif(not _CLAUDE_CODE_SKILL.exists(), reason="Claude Code skill not found")
    def test_claude_code_skill_lists_all_transforms(self) -> None:
        """Every registered transform plugin appears in the Claude Code skill."""
        cc_content = _claude_code_skill_content()
        skill_transforms = _extract_backtick_names(cc_content, "### Transforms")
        missing = self.transform_names - skill_transforms
        assert not missing, f"Transform plugins missing from Claude Code skill: {missing}"

    @pytest.mark.skipif(not _CLAUDE_CODE_SKILL.exists(), reason="Claude Code skill not found")
    def test_claude_code_skill_lists_all_sinks(self) -> None:
        """Every registered sink plugin appears in the Claude Code skill."""
        cc_content = _claude_code_skill_content()
        skill_sinks = _extract_backtick_names(cc_content, "### Sinks")
        missing = self.sink_names - skill_sinks
        assert not missing, f"Sink plugins missing from Claude Code skill: {missing}"


class TestStateValidatorPluginDrift:
    """Verify static plugin sets in composer state validators match the runtime registry.

    The composer's ``validate()`` uses static sets (e.g. ``_FILE_SINK_PLUGINS``) to
    decide which plugin names trigger which warnings. If a static set drifts from
    the runtime sink registry, the composer either (a) skips warnings for plugins
    that *do* exist (false negative) or (b) implies plugins exist that don't, so
    a pipeline passes composer pre-validation only to crash at runtime when the
    sink registry rejects the unknown plugin name.

    These tests enforce: static sets must be subsets of the runtime registry.
    """

    @pytest.fixture(autouse=True)
    def _discover(self) -> None:
        """Discover all sink plugin names once for the test class."""
        discovered = discover_all_plugins()
        self.sink_names = {cls.name for cls in discovered["sinks"]}  # type: ignore[attr-defined]

    def test_file_sinks_subset_of_registered_sinks(self) -> None:
        """``_FILE_SINK_PLUGINS`` ⊆ runtime sink registry.

        Tracks elspeth-f7c63d7346: the prior set ``{csv, json, jsonl, text,
        parquet, xml}`` advertised four phantom sinks (``jsonl``, ``text``,
        ``parquet``, ``xml``) that the runtime sink registry has never
        registered, deferring a guaranteed runtime crash past composer
        pre-validation.
        """
        from elspeth.web.composer.state import _FILE_SINK_PLUGINS

        phantom = _FILE_SINK_PLUGINS - self.sink_names
        assert not phantom, (
            f"_FILE_SINK_PLUGINS contains plugin names not in the runtime sink "
            f"registry: {sorted(phantom)}. Either register those sink plugins or "
            f"remove them from _FILE_SINK_PLUGINS in src/elspeth/web/composer/state.py."
        )


class TestTwoFileDivergence:
    """Verify the web skill and Claude Code skill list the same plugins."""

    @pytest.mark.skipif(not _CLAUDE_CODE_SKILL.exists(), reason="Claude Code skill not found")
    def test_source_plugins_match(self) -> None:
        """Both skill files list the same source plugins."""
        web = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Sources")
        cc = _extract_backtick_names(_claude_code_skill_content(), "### Sources")
        assert web == cc, f"Source divergence — web-only: {web - cc}, cc-only: {cc - web}"

    @pytest.mark.skipif(not _CLAUDE_CODE_SKILL.exists(), reason="Claude Code skill not found")
    def test_transform_plugins_match(self) -> None:
        """Both skill files list the same transform plugins."""
        web = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Transforms")
        cc = _extract_backtick_names(_claude_code_skill_content(), "### Transforms")
        assert web == cc, f"Transform divergence — web-only: {web - cc}, cc-only: {cc - web}"

    @pytest.mark.skipif(not _CLAUDE_CODE_SKILL.exists(), reason="Claude Code skill not found")
    def test_sink_plugins_match(self) -> None:
        """Both skill files list the same sink plugins."""
        web = _extract_backtick_names(_WEB_SKILL_CONTENT, "### Sinks")
        cc = _extract_backtick_names(_claude_code_skill_content(), "### Sinks")
        assert web == cc, f"Sink divergence — web-only: {web - cc}, cc-only: {cc - web}"


class TestValidationGlossaryCompleteness:
    """Verify the web skill glossary covers all validation warnings and suggestions."""

    def test_all_warnings_in_glossary(self) -> None:
        """Every warning from validate() has a recognizable entry in the glossary.

        Extracts the distinctive opening phrase from each warning and checks
        that the web skill's Validation Warning Glossary mentions it.
        """
        # Import validate's source to extract warning message patterns.
        from elspeth.web.composer.state import CompositionState

        # Build states that trigger each warning category.
        # W1: unreachable output
        state_w1 = CompositionState.from_dict(
            {
                "source": {"plugin": "csv", "on_success": "t1_in", "options": {}, "on_validation_failure": "discard"},
                "nodes": [
                    {
                        "id": "t1",
                        "node_type": "transform",
                        "plugin": "passthrough",
                        "input": "t1_in",
                        "on_success": "main_out",
                        "on_error": "discard",
                        "options": {},
                    }
                ],
                "edges": [],
                "outputs": [
                    {"name": "main_out", "plugin": "csv", "options": {}, "on_write_failure": "discard"},
                    {"name": "orphan_out", "plugin": "csv", "options": {}, "on_write_failure": "discard"},
                ],
                "metadata": {"name": "Test", "description": ""},
                "version": 1,
            }
        )
        v1 = state_w1.validate()

        # Collect all warnings from the test state.
        all_warnings = list(v1.warnings)

        # Check each warning's distinctive phrase appears in the skill.
        for warning in all_warnings:
            # Extract the first distinctive clause (before the em dash or first variable).
            # We check that the glossary contains a recognizable fragment.
            found = False
            # Try substrings of increasing specificity.
            for fragment in [
                "not referenced by any on_success",
                "does not match any node input or output",
                "has no outgoing edges",
                "filename extension suggests a different format",
                "appears incomplete:",  # W5: transform missing required options
                "has empty '",  # W5: transform has empty required option
                "has no path configured",  # W6: file sink missing path
                "has empty path",  # W6: file sink empty path
            ]:
                if fragment in warning.message:
                    assert fragment in _WEB_SKILL_CONTENT, (
                        f"Validation warning not in glossary: {warning!r}\nExpected fragment: {fragment!r}"
                    )
                    found = True
                    break
            if not found:
                # Unknown warning pattern — fail to flag it.
                pytest.fail(f"Unrecognized validation warning pattern (not in test coverage): {warning!r}")

        # Check suggestions.
        suggestion_fragments = [
            "Consider adding error routing",
            "Consider adding a second output",
            "Source has no explicit schema",
        ]
        for fragment in suggestion_fragments:
            assert fragment in _WEB_SKILL_CONTENT, f"Validation suggestion not in glossary: {fragment!r}"
