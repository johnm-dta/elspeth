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


def _section_between(text: str, start_anchor: str, end_anchor: str) -> str:
    """Extract a bounded markdown section; fail loudly if anchors drift."""
    start = text.find(start_anchor)
    end = text.find(end_anchor, start)
    if start == -1 or end == -1:
        raise AssertionError(f"Could not locate skill anchors: {start_anchor!r} ... {end_anchor!r}")
    return text[start:end]


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


class TestLLMModelGuidance:
    """Pin skill guidance that prevents model-identifier hallucination."""

    def test_web_skill_tells_composer_to_list_models_before_choosing_llm_model(self) -> None:
        """The composer skill must direct LLM configs through live model discovery.

        The runtime value-source check rejects OpenRouter model identifiers that
        are not present in the LiteLLM catalog. The skill therefore needs to
        tell the composer agent to call ``list_models`` and choose from that
        response instead of assuming a familiar model string is available.
        """
        assert 'list_models(provider="openrouter/")' in _WEB_SKILL_CONTENT
        assert "do not assume" in _WEB_SKILL_CONTENT
        assert "choose one from that response" in _WEB_SKILL_CONTENT
        assert "never invent identifiers" in _WEB_SKILL_CONTENT


class TestInterpretationReviewGuidance:
    """Pin the prompt guidance that makes Phase 5b observable in LLM runs."""

    def test_web_skill_teaches_subjective_term_interpretation_review(self) -> None:
        """The skill must do more than list ``request_interpretation_review``.

        A live local smoke run can emit a valid LLM pipeline with
        ``prompt_template`` while silently baking "cool" into the prompt if the
        skill only enumerates the tool. Keep the usage contract explicit so a
        future prompt edit cannot remove the rule while leaving the tool name in
        Step 0.
        """
        guidance = _section_between(
            _WEB_SKILL_CONTENT,
            "### Subjective Interpretation Review",
            "### TERMINATION GATE",
        )

        assert "request_interpretation_review" in guidance
        assert "subjective or underspecified" in guidance
        assert "{{interpretation:<term>}}" in guidance
        assert "After the state-staging tool succeeds" in guidance
        assert "before any final reply" in guidance
        assert "Do not ask the user to confirm subjective terms in normal assistant prose" in guidance
        assert "Do not silently bake" in guidance
        assert "interpretation_review_disabled" in guidance


class TestEngineValidatorPluginDrift:
    """Verify static plugin sets in engine-side validators match the runtime registry.

    The engine's ``validate_sink_failsink_destinations`` uses a static set
    (``_ALLOWED_FAILSINK_PLUGINS``) to decide which target plugins are
    eligible failsinks. If the set drifts from the runtime sink registry,
    a pipeline can pass engine validation only to crash at runtime when
    ``get_sink_by_name`` raises ``PluginNotFoundError`` for the phantom
    plugin name. This is the engine-layer counterpart to
    ``TestStateValidatorPluginDrift`` — same anti-pattern, same fix recipe.

    These tests enforce: static sets must be subsets of the runtime registry.
    """

    @pytest.fixture(autouse=True)
    def _discover(self) -> None:
        """Discover all sink plugin names once for the test class."""
        discovered = discover_all_plugins()
        self.sink_names = {cls.name for cls in discovered["sinks"]}  # type: ignore[attr-defined]

    def test_allowed_failsink_plugins_subset_of_registered_sinks(self) -> None:
        """``_ALLOWED_FAILSINK_PLUGINS`` ⊆ runtime sink registry.

        Tracks elspeth-3ef528e3e3: the prior set ``{csv, json, xml}``
        advertised one phantom sink (``xml``) that the runtime sink
        registry has never registered, deferring a guaranteed runtime
        crash past engine pre-validation
        (``validate_sink_failsink_destinations`` accepts the failsink, then
        ``get_sink_by_name("xml")`` raises ``PluginNotFoundError`` at run
        time). Co-drift cluster with elspeth-f7c63d7346 (state.py side).
        """
        from elspeth.engine.orchestrator.validation import _ALLOWED_FAILSINK_PLUGINS

        phantom = _ALLOWED_FAILSINK_PLUGINS - self.sink_names
        assert not phantom, (
            f"_ALLOWED_FAILSINK_PLUGINS contains plugin names not in the runtime "
            f"sink registry: {sorted(phantom)}. Either register those sink plugins "
            f"or remove them from _ALLOWED_FAILSINK_PLUGINS in "
            f"src/elspeth/engine/orchestrator/validation.py."
        )


class TestComposerToolNameDrift:
    """Verify the skill's Step-0 tool enumeration matches get_tool_definitions().

    The skill at ``src/elspeth/web/composer/skills/pipeline_composer.md``
    instructs the LLM to load schemas for every composer tool before
    calling any of them. The Step-0 list is hand-maintained as bulleted
    categories (Discovery / State-preview / Build-edit / Diagnostics /
    Blobs / Secrets). If ``get_tool_definitions()`` in
    ``src/elspeth/web/composer/tools.py`` adds, removes, or renames a
    tool without a matching skill update, the LLM either fails to load
    a tool it needs (InputValidationError on first call) or wastes a
    budget turn loading a tool that isn't registered. This drift gate
    catches both directions and is the Scope-A deliverable from
    Followup D (docs/skill-rgr/followup-D-tool-inventory-bootstrap.md).

    Scope-B (a single ``composer_bootstrap`` tool that returns the
    manifest) is tracked separately as ``elspeth-4f85bd6652`` and is
    deferred until the ``batch3_bootstrap`` harness scenario produces
    RED on production for at least one model.
    """

    @staticmethod
    def _extract_skill_step0_tool_names(skill_text: str) -> set[str]:
        """Extract tool names from the Step-0 bullet categories.

        The Step-0 section is bounded by two stable anchors:

        - start: ``**Step 0 (mandatory before any pipeline work):**``
        - end:   ``If any tool you intend to call still shows a placeholder``

        Within that range, only lines starting with ``- **`` (category
        bullets) are scanned. This deliberately excludes the section's
        explanatory text, which mentions ``get_tool_definitions()`` and
        would otherwise pollute the result. Each category bullet's
        backtick-quoted ``snake_case`` identifiers are tool names.

        Adding a new bullet category (e.g. ``- **Audit:** ...``) just
        works. Reformatting that drops the bullet prefix or renames the
        anchors will surface as an explicit AssertionError below rather
        than as a silent test passing on an empty set.
        """
        start_anchor = "**Step 0 (mandatory before any pipeline work):**"
        end_anchor = "If any tool you intend to call still shows a placeholder"
        start = skill_text.find(start_anchor)
        end = skill_text.find(end_anchor, start)
        if start == -1 or end == -1:
            raise AssertionError(
                "Could not locate Step-0 section anchors in pipeline_composer.md. "
                "If the section was renamed, update _extract_skill_step0_tool_names "
                "(start_anchor / end_anchor) so the drift gate keeps enforcing the "
                "skill ↔ get_tool_definitions() invariant."
            )
        section = skill_text[start:end]
        names: set[str] = set()
        for line in section.splitlines():
            if line.startswith("- **"):
                names.update(re.findall(r"`([a-z_][a-z0-9_]*)`", line))
        return names

    def test_skill_step0_matches_get_tool_definitions(self) -> None:
        """Skill Step-0 enumeration == ``get_tool_definitions()`` name set.

        Bidirectional check: any tool in the runtime that's missing from
        the skill is a 'silent gap' (the LLM won't know to load it); any
        tool in the skill that's not in the runtime is a 'phantom' (the
        LLM will waste a budget turn on a tool that doesn't exist).
        Both are caught by the same set-equality assertion.
        """
        from elspeth.web.composer.tools import get_tool_definitions

        runtime_names = {defn["name"] for defn in get_tool_definitions()}
        skill_names = self._extract_skill_step0_tool_names(_WEB_SKILL_CONTENT)

        # Sanity: anchors matched but the regex returned nothing — the
        # bullet format probably changed and the gate would silently
        # pass. Fail loudly instead.
        assert skill_names, (
            "Step-0 anchors matched but no tool names were extracted from the "
            "bulleted categories. The bullet format probably changed (the "
            "extractor expects lines starting with '- **'). Adjust "
            "_extract_skill_step0_tool_names or the bullet format must be "
            "restored — the drift gate would silently pass otherwise."
        )

        missing_from_skill = runtime_names - skill_names
        phantom_in_skill = skill_names - runtime_names

        assert not missing_from_skill, (
            f"Tools registered in get_tool_definitions() but not enumerated in "
            f"the skill's Step-0 list: {sorted(missing_from_skill)}. Add them to "
            f"src/elspeth/web/composer/skills/pipeline_composer.md (Step-0 "
            f"section) under the appropriate category, or the LLM will hit "
            f"InputValidationError when it tries to call the tool without "
            f"having loaded its deferred schema."
        )
        assert not phantom_in_skill, (
            f"Tools enumerated in the skill's Step-0 list but not registered in "
            f"get_tool_definitions(): {sorted(phantom_in_skill)}. Either "
            f"register them in src/elspeth/web/composer/tools.py or remove them "
            f"from the skill — the LLM cannot call a tool that isn't in "
            f"get_tool_definitions(), and the bullet sets a false expectation."
        )


class TestRunnableComposerSkillExamples:
    """Focused runnable-example checks for guidance that models often copy verbatim."""

    def test_secret_guidance_never_uses_legacy_secret_uri_literals(self) -> None:
        assert "secret://" not in _WEB_SKILL_CONTENT

    def test_llm_secret_guidance_uses_inline_secret_ref_marker(self) -> None:
        provider_section = _section_between(
            _WEB_SKILL_CONTENT,
            "**Provider fallback order**",
            "**The only situation where stopping is correct**",
        )

        assert "api_key: {secret_ref: OPENROUTER_API_KEY}" in provider_section
        assert 'api_key: "secret://OPENROUTER_API_KEY"' not in provider_section

    def test_wire_secret_ref_guidance_matches_live_tool_schema(self) -> None:
        from elspeth.web.composer.tools import get_tool_definitions

        wire_schema = next(defn for defn in get_tool_definitions() if defn["name"] == "wire_secret_ref")["parameters"]
        properties = set(wire_schema["properties"])
        required = set(wire_schema["required"])
        contract_section = _section_between(
            _WEB_SKILL_CONTENT,
            "### Secret Reference Wiring Contract",
            "### LLM Providers",
        )

        assert {"name", "target", "target_id", "option_key"}.issubset(properties)
        assert required == {"name", "target", "option_key"}
        assert 'wire_secret_ref(name="<NAME>", target="node", target_id="<id>", option_key="<credential_field>")' in contract_section
        assert 'wire_secret_ref(node="<id>", field="<credential_field>", ref="<NAME>")' not in contract_section

    def test_worked_set_pipeline_edge_targets_the_real_sink_connection(self) -> None:
        example = _section_between(
            _WEB_SKILL_CONTENT,
            '  "edges": [\n',
            "Trace each connection name through the diagram",
        )

        assert '"from_node": "split_lines", "to_node": "lines_out"' in example
        assert '"from_node": "split_lines", "to_node": "output_lines"' not in example

    def test_web_scrape_format_guidance_uses_config_literal_raw_for_html_output(self) -> None:
        web_scrape_guidance = _section_between(
            _WEB_SKILL_CONTENT,
            "**web_scrape** — Fetch and extract content from a URL in each row.",
            "**Canonical full options block**",
        )

        format_line = next(line for line in web_scrape_guidance.splitlines() if line.startswith("- `format`:"))
        assert '"raw"' in format_line
        assert '"html"' not in format_line

    def test_url_input_recipe_uses_matching_connection_and_full_web_scrape_options(self) -> None:
        recipe = _section_between(
            _WEB_SKILL_CONTENT,
            '- User says "use this URL: https://example.com"',
            "- User provides JSON data",
        )

        assert 'on_success: "url_rows"' in recipe
        assert 'input: "url_rows"' in recipe
        for required_option in ("schema", "url_field", "content_field", "fingerprint_field", "http"):
            assert required_option in recipe


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
