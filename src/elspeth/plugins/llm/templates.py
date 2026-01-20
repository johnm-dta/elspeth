# src/elspeth/plugins/llm/templates.py
"""Jinja2-based prompt templating with audit support."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from jinja2 import StrictUndefined, TemplateSyntaxError, UndefinedError
from jinja2.exceptions import SecurityError
from jinja2.sandbox import SandboxedEnvironment

from elspeth.core.canonical import canonical_json


class TemplateError(Exception):
    """Error in template rendering (including sandbox violations)."""


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered prompt with audit metadata."""

    prompt: str
    template_hash: str
    variables_hash: str
    rendered_hash: str
    # New fields for file-based templates
    template_source: str | None = None  # File path or None if inline
    lookup_hash: str | None = None  # Hash of lookup data or None
    lookup_source: str | None = None  # File path or None


def _sha256(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class PromptTemplate:
    """Jinja2 prompt template with audit trail support.

    Uses sandboxed environment to prevent dangerous operations.
    Tracks hashes of template, variables, and rendered output for audit.

    Example:
        template = PromptTemplate('''
            Analyze the following product:
            Name: {{ product.name }}
            Description: {{ product.description }}

            Provide a quality score from 1-10.
        ''')

        result = template.render_with_metadata(
            product={"name": "Widget", "description": "A useful widget"}
        )

        # result.prompt = rendered string
        # result.template_hash = hash of template
        # result.variables_hash = hash of input variables
        # result.rendered_hash = hash of final prompt
    """

    def __init__(self, template_string: str) -> None:
        """Initialize template.

        Args:
            template_string: Jinja2 template string

        Raises:
            TemplateError: If template syntax is invalid
        """
        self._template_string = template_string
        self._template_hash = _sha256(template_string)

        # Use sandboxed environment for security
        self._env = SandboxedEnvironment(
            undefined=StrictUndefined,  # Raise on undefined variables
            autoescape=False,  # No HTML escaping for prompts
        )

        try:
            self._template = self._env.from_string(template_string)
        except TemplateSyntaxError as e:
            raise TemplateError(f"Invalid template syntax: {e}") from e

    @property
    def template_hash(self) -> str:
        """SHA-256 hash of the template string."""
        return self._template_hash

    def render(self, **variables: Any) -> str:
        """Render template with variables.

        Args:
            **variables: Template variables

        Returns:
            Rendered prompt string

        Raises:
            TemplateError: If rendering fails (undefined variable, sandbox violation, etc.)
        """
        try:
            return self._template.render(**variables)
        except UndefinedError as e:
            raise TemplateError(f"Undefined variable: {e}") from e
        except SecurityError as e:
            raise TemplateError(f"Sandbox violation: {e}") from e
        except Exception as e:
            raise TemplateError(f"Template rendering failed: {e}") from e

    def render_with_metadata(self, **variables: Any) -> RenderedPrompt:
        """Render template and return with audit metadata.

        Args:
            **variables: Template variables

        Returns:
            RenderedPrompt with prompt string and all hashes
        """
        prompt = self.render(**variables)

        # Compute variables hash using canonical JSON
        variables_canonical = canonical_json(variables)
        variables_hash = _sha256(variables_canonical)

        # Compute rendered prompt hash
        rendered_hash = _sha256(prompt)

        return RenderedPrompt(
            prompt=prompt,
            template_hash=self._template_hash,
            variables_hash=variables_hash,
            rendered_hash=rendered_hash,
        )
