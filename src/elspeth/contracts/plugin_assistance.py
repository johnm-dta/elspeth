"""Plugin assistance: deterministic guidance for plugins.

Two retrieval modes (see ``PluginAssistance``):

* ``issue_code=None`` — discovery-time guidance, surfaced to operators
  via the catalog and to LLMs operating the composer via the
  ``get_plugin_assistance`` MCP tool *and* the discovery DTOs
  (``PluginSummary``, ``PluginSchemaInfo``).
* ``issue_code="..."`` — failure-time guidance, surfaced by validators
  when a specific issue code triggers.

L0 module. Carries no plugin runtime references. Consumers (catalog
service, MCP discovery, validators) attach assistance to issue codes
or call the discovery-time variant; they do not parse the prose.

Secret discipline: assistance fields MUST contain only safe option
names, plugin names, enum values, and human-readable advice. They MUST
NOT contain raw URLs, headers, prompts, row data, credentials, raw
provider errors, file paths, or exception strings. Enforcement is by
plugin authors and tests (see secret-leakage tests in Phase 3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class PluginAssistanceExample:
    """A before/after configuration sketch."""

    title: str
    before: Mapping[str, object] | None = None
    after: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.before is not None:
            freeze_fields(self, "before")
        if self.after is not None:
            freeze_fields(self, "after")


@dataclass(frozen=True, slots=True)
class PluginAssistance:
    """Deterministic, side-effect-free guidance for a plugin.

    Returned by ``BaseTransform.get_agent_assistance(issue_code=...)``,
    and the equivalent classmethods on ``BaseSource`` and ``BaseSink``.

    Two retrieval modes:

    * ``issue_code is None`` — *discovery-time guidance*. The plugin
      returns general operator-facing hints (``composer_hints``) and a
      one-line ``summary`` describing what the plugin does. Validators
      do not call this branch; the catalog/MCP discovery surface does,
      so an LLM operating the composer can see hints *before* picking
      a plugin.
    * ``issue_code is not None`` — *failure-time guidance*. Validators
      attach the issue code and surface remediation. The plugin returns
      ``suggested_fixes`` and ``examples`` keyed to that specific code.

    Audit-hash discipline: ``composer_hints`` are *advisory coaching*,
    not contract. They do **not** participate in
    ``composer_skill_hash`` or any other audit hash. The framework
    version pin (git SHA / package version) is the implicit
    reproducibility anchor for hint text. The contractually-audited
    surfaces remain the composer skill (``composer_skill_hash``) and
    the LLM identity tuple.

    Secret discipline: assistance fields MUST contain only safe option
    names, plugin names, enum values, and human-readable advice. They
    MUST NOT contain raw URLs, headers, prompts, row data, credentials,
    raw provider errors, file paths, or exception strings. Enforcement
    is by plugin authors and tests (see secret-leakage tests in
    Phase 3).
    """

    plugin_name: str
    issue_code: str | None
    summary: str
    suggested_fixes: tuple[str, ...] = ()
    examples: tuple[PluginAssistanceExample, ...] = ()
    composer_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # The type annotations name tuples, but Python does not enforce that —
        # callers can pass lists. freeze_fields coerces list -> tuple
        # (identity-preserving when already a tuple). Element-level dict
        # freezing is handled by PluginAssistanceExample.__post_init__.
        freeze_fields(self, "suggested_fixes", "examples", "composer_hints")
