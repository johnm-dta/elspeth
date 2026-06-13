"""Guard: composer hints must not name enum literals the config model rejects.

``composer_hints`` / ``summary`` strings (from ``get_agent_assistance(issue_code=None)``)
are prose fed to the pipeline-composing LLM. A hint that names an option value the
config model does not accept manufactures invalid pipelines: the LLM emits a config
the ``extra="forbid"`` / ``Literal`` field then rejects at validation time.

This is the class-preventing gate for the composer-hint-rot finding family (plugins
review 2026-06-10, Batch 2). It is deliberately HIGH-PRECISION rather than
high-recall: it only inspects ``<field>: <values>`` clauses where ``<field>`` is the
exact name of a config field that is *actually* ``Literal``-typed, and checks the
quoted enum tokens against that field's real allowed values. Keying on a real Literal
field name keeps false positives at zero — hints legitimately mention Jinja
variables, output field names, review vocabulary, and path tokens, none of which look
like ``<real_literal_field>: 'value'``.

Out of scope (FP-prone, tracked as advisory follow-up, not gated here): detecting
references to config *keys* that do not exist at all (e.g. a fictional ``write_mode``).
That requires parsing free prose for option keys and cannot be made zero-FP.
"""

from __future__ import annotations

import re
from typing import Literal, cast, get_args, get_origin

from pydantic import BaseModel

from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.plugins.infrastructure.discovery import discover_all_plugins

# Quoted enum token inside a clause tail, e.g. 'fail_if_exists'.
_QUOTED_LITERAL = re.compile(r"'([a-z][a-z0-9_]*)'")


def _field_literals(annotation: object) -> set[str]:
    """All string members of any ``Literal[...]`` anywhere in an annotation."""
    found: set[str] = set()
    if get_origin(annotation) is Literal:
        found.update(a for a in get_args(annotation) if isinstance(a, str))
    for arg in get_args(annotation):
        found |= _field_literals(arg)
    return found


def _submodels(annotation: object) -> list[type]:
    """Pydantic submodels reachable from an annotation (incl. ``Model | None``)."""
    return [a for a in (annotation, *get_args(annotation)) if hasattr(a, "model_fields")]


def _literal_fields(config_model: type[BaseModel], prefix: str = "", depth: int = 0) -> dict[str, set[str]]:
    """Map every Literal-typed config field (dotted for nested) to its allowed values."""
    result: dict[str, set[str]] = {}
    for name, field in config_model.model_fields.items():
        literals = _field_literals(field.annotation)
        if literals:
            result[prefix + name] = literals
        if depth < 2:
            for submodel in _submodels(field.annotation):
                result.update(_literal_fields(submodel, f"{prefix}{name}.", depth + 1))
    return result


def _iter_hint_bearing_plugins() -> list[tuple[str, type[BaseModel], PluginAssistance]]:
    """All discovered plugin classes that expose discovery-time composer hints."""
    out: list[tuple[str, type[BaseModel], PluginAssistance]] = []
    for classes in discover_all_plugins().values():
        for cls in classes:
            get_assistance = getattr(cls, "get_agent_assistance", None)
            if get_assistance is None:
                continue
            assistance = get_assistance(issue_code=None)
            if assistance is None or not assistance.composer_hints:
                continue
            config_model = cls.get_config_model() if hasattr(cls, "get_config_model") else getattr(cls, "config_model", None)
            if config_model is None:  # e.g. NullSource
                continue
            plugin_name = cast(str, cls.name)  # type: ignore[attr-defined]  # plugin base declares name
            out.append((plugin_name, config_model, assistance))
    return out


def test_composer_hints_only_name_real_enum_literals() -> None:
    """Every quoted enum literal a hint pins to a Literal config field must be valid."""
    violations: list[str] = []

    for plugin_name, config_model, assistance in _iter_hint_bearing_plugins():
        literal_fields = _literal_fields(config_model)
        if not literal_fields:
            continue

        strings = list(assistance.composer_hints)
        if assistance.summary:
            strings.append(assistance.summary)

        for text in strings:
            for field, allowed in literal_fields.items():
                bare = field.split(".")[-1]
                # Only a clause that names the EXACT Literal field, e.g. "collision_policy: ...".
                for clause in re.finditer(rf"\b{re.escape(bare)}:\s*([^.;]+)", text):
                    for token in _QUOTED_LITERAL.findall(clause.group(1)):
                        if token not in allowed:
                            violations.append(
                                f"{plugin_name}: hint pins {field}={token!r} but the config "
                                f"field only allows {sorted(allowed)} — clause: {clause.group(0).strip()!r}"
                            )

    assert not violations, "Composer hints name enum literals the config model rejects:\n" + "\n".join(violations)


def test_hint_consistency_gate_actually_inspects_plugins() -> None:
    """Sanity: the discovery + introspection path finds Literal-bearing plugins.

    Guards against the gate silently passing because discovery returned nothing or
    no plugin exposed a Literal field (which would make the gate vacuous).
    """
    literal_bearing = [name for name, cm, _ in _iter_hint_bearing_plugins() if _literal_fields(cm)]
    assert "csv" in literal_bearing  # collision_policy / mode are Literal fields
    assert len(literal_bearing) >= 5
