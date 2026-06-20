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

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from elspeth.contracts.freeze import freeze_fields

_UNSAFE_TEXT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhttps?://[A-Za-z0-9][^\s`'\"]*", re.IGNORECASE), "raw URL"),
    (re.compile(r"\bwww\.\S+", re.IGNORECASE), "raw URL"),
    (re.compile(r"(?i)\b(?:authorization|proxy-authorization)\s*:\s*\S+"), "credential"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "credential"),
    (re.compile(r"(?i)\b(?:api[_ -]?key|password|token|secret|credential|connection[_ -]?string)\s*[:=]\s*['\"]?\S+"), "credential"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "credential"),
    (re.compile(r"(?m)(?:^|\n)Traceback \(most recent call last\):"), "exception string"),
    (re.compile(r'(?m)File "[^"]+", line \d+'), "exception string"),
    (re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):\s+"), "exception string"),
    (re.compile(r"(?<![`A-Za-z0-9_])(?:/home|/Users|/tmp|/var|/etc)/[^\s`'\"]+"), "file path"),
    (re.compile(r"\b[A-Za-z]:\\[^\s`'\"]+"), "file path"),
)

_RAW_HEADER_KEYS = {"headers", "authorization", "proxy-authorization", "cookie", "cookies"}
_RAW_ROW_DATA_KEYS = {"rows", "row_data", "sample_rows"}
_RAW_PROMPT_KEYS = {"prompt", "system_prompt", "user_prompt"}


def _assert_safe_assistance_text(value: str, path: str) -> None:
    for pattern, label in _UNSAFE_TEXT_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"unsafe assistance content in {path}: {label} is not allowed")


def _assert_safe_assistance_value(value: object, path: str) -> None:
    if isinstance(value, str):
        _assert_safe_assistance_text(value, path)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} key must be str, got {type(key).__name__}: {key!r}")
            normalized = key.strip().lower().replace("-", "_")
            child_path = f"{path}.{key}"
            if normalized in _RAW_HEADER_KEYS:
                raise ValueError(f"unsafe assistance content in {child_path}: raw headers are not allowed")
            if normalized in _RAW_ROW_DATA_KEYS:
                raise ValueError(f"unsafe assistance content in {child_path}: raw row data is not allowed")
            if normalized in _RAW_PROMPT_KEYS:
                raise ValueError(f"unsafe assistance content in {child_path}: raw prompt is not allowed")
            _assert_safe_assistance_value(child, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        for idx, item in enumerate(value):
            _assert_safe_assistance_value(item, f"{path}[{idx}]")


@dataclass(frozen=True, slots=True)
class PluginAssistanceExample:
    """A before/after configuration sketch."""

    title: str
    before: Mapping[str, object] | None = None
    after: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _assert_safe_assistance_value(self.title, "PluginAssistanceExample.title")
        _assert_safe_assistance_value(self.before, "PluginAssistanceExample.before")
        _assert_safe_assistance_value(self.after, "PluginAssistanceExample.after")
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
        _assert_safe_assistance_value(self.plugin_name, "PluginAssistance.plugin_name")
        _assert_safe_assistance_value(self.issue_code, "PluginAssistance.issue_code")
        _assert_safe_assistance_value(self.summary, "PluginAssistance.summary")
        _assert_safe_assistance_value(self.suggested_fixes, "PluginAssistance.suggested_fixes")
        _assert_safe_assistance_value(self.composer_hints, "PluginAssistance.composer_hints")
        # The type annotations name tuples, but Python does not enforce that —
        # callers can pass lists. freeze_fields coerces list -> tuple
        # (identity-preserving when already a tuple). Element-level dict
        # freezing is handled by PluginAssistanceExample.__post_init__.
        freeze_fields(self, "suggested_fixes", "examples", "composer_hints")
