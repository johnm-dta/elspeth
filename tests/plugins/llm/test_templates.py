# tests/plugins/llm/test_templates.py
"""Tests for Jinja2 prompt template engine."""

import pytest

from elspeth.plugins.llm.templates import PromptTemplate, TemplateError


class TestPromptTemplate:
    """Tests for PromptTemplate wrapper."""

    def test_simple_variable_substitution(self) -> None:
        """Basic variable substitution works."""
        template = PromptTemplate("Hello, {{ name }}!")
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_template_with_loop(self) -> None:
        """Jinja2 loops work."""
        template = PromptTemplate(
            """
Analyze these items:
{% for item in items %}
- {{ item.name }}: {{ item.value }}
{% endfor %}
""".strip()
        )
        result = template.render(
            items=[
                {"name": "A", "value": 1},
                {"name": "B", "value": 2},
            ]
        )
        assert "- A: 1" in result
        assert "- B: 2" in result

    def test_template_with_default_filter(self) -> None:
        """Jinja2 default filter works."""
        template = PromptTemplate("Focus: {{ focus | default('general') }}")
        assert template.render() == "Focus: general"
        assert template.render(focus="quality") == "Focus: quality"

    def test_template_hash_is_stable(self) -> None:
        """Same template string produces same hash."""
        t1 = PromptTemplate("Hello, {{ name }}!")
        t2 = PromptTemplate("Hello, {{ name }}!")
        assert t1.template_hash == t2.template_hash

    def test_different_templates_have_different_hashes(self) -> None:
        """Different templates have different hashes."""
        t1 = PromptTemplate("Hello, {{ name }}!")
        t2 = PromptTemplate("Goodbye, {{ name }}!")
        assert t1.template_hash != t2.template_hash

    def test_render_returns_metadata(self) -> None:
        """render() returns prompt and audit metadata."""
        template = PromptTemplate("Analyze: {{ text }}")
        result = template.render_with_metadata(text="sample")

        assert result.prompt == "Analyze: sample"
        assert result.template_hash is not None
        assert result.variables_hash is not None
        assert result.rendered_hash is not None

    def test_undefined_variable_raises_error(self) -> None:
        """Missing required variable raises TemplateError."""
        template = PromptTemplate("Hello, {{ name }}!")
        with pytest.raises(TemplateError, match="name"):
            template.render()  # No 'name' provided

    def test_sandboxed_prevents_dangerous_operations(self) -> None:
        """Sandboxed environment blocks dangerous operations."""
        # Attempt to access dunder attributes (blocked by SandboxedEnvironment)
        dangerous = PromptTemplate("{{ ''.__class__.__mro__ }}")
        # SecurityError is wrapped in TemplateError with "Sandbox violation" message
        with pytest.raises(TemplateError, match="Sandbox violation"):
            dangerous.render()
