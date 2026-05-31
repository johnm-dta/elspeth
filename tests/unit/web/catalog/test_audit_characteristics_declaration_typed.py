"""Build-time guard for typo'd ``audit_characteristics`` declarations.

``AuditCharacteristic`` is a ``StrEnum`` — a subclass of ``str``. A
plugin author who writes ::

    audit_characteristics = frozenset({"signed"})

rather than ::

    audit_characteristics = frozenset({AuditCharacteristic.SIGNED})

can pass mypy under some inference paths (the right-hand side is
``frozenset[str]``, the base-class declaration is
``DeclaredAuditCharacteristics = frozenset[AuditCharacteristic]``; the
asymmetry can be silently accepted depending on the inference setup
and the bare string survives to runtime). At runtime the bare string
flows through ``_derive_audit_characteristics``'s sorted-tuple
serialisation untouched: the catalog API emits ``"signed"`` on the
wire, the frontend's metadata table looks it up, finds nothing, and
renders the grey "unknown" chip — degraded rendering with no test
failure and no audit signal.

This test walks every registered builtin plugin and asserts that every
member of every ``audit_characteristics`` frozenset is *exactly* an
``AuditCharacteristic`` enum instance, not a bare ``str`` that happens
to spell the same value. ``type(c) is AuditCharacteristic`` is
deliberate — ``isinstance`` would accept bare strings because
``StrEnum`` extends ``str``.

Failure message names the offending plugin and the offending value so
the diagnostic is immediate: the test output is the fix.
"""

from __future__ import annotations

from elspeth.contracts.enums import AuditCharacteristic
from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.catalog.schemas import PluginKind


def _make_manager() -> PluginManager:
    manager = PluginManager()
    manager.register_builtin_plugins()
    return manager


def _check_kind(kind: PluginKind, classes: list[type]) -> list[str]:
    """Return a list of human-readable failure messages, one per offence."""
    failures: list[str] = []
    for plugin_cls in classes:
        declared = plugin_cls.audit_characteristics
        for value in declared:
            # ``type(c) is AuditCharacteristic`` is stricter than
            # ``isinstance(c, AuditCharacteristic)`` — the latter would
            # accept bare strings because ``StrEnum`` inherits from
            # ``str``. The whole point of this guard is to reject bare
            # strings that pass mypy under some inference paths.
            if type(value) is not AuditCharacteristic:
                failures.append(
                    f"  - {kind} plugin {plugin_cls.name!r} "
                    f"({plugin_cls.__module__}.{plugin_cls.__qualname__}): "
                    f"audit_characteristics contains {value!r} of type "
                    f"{type(value).__name__}, expected AuditCharacteristic. "
                    f"Declare members as AuditCharacteristic enum "
                    f"references (e.g. AuditCharacteristic.SIGNED), not "
                    f"as bare strings."
                )
    return failures


def test_every_plugin_declares_audit_characteristics_as_enum_members() -> None:
    """Every registered builtin plugin's ``audit_characteristics``
    frozenset must contain only ``AuditCharacteristic`` enum instances.

    A bare-string typo (e.g. ``frozenset({"signed"})`` instead of
    ``frozenset({AuditCharacteristic.SIGNED})``) is rejected by this
    test rather than silently degrading the rendered catalog card.
    """
    manager = _make_manager()
    failures: list[str] = []
    failures.extend(_check_kind("source", list(manager.get_sources())))
    failures.extend(_check_kind("transform", list(manager.get_transforms())))
    failures.extend(_check_kind("sink", list(manager.get_sinks())))

    assert not failures, (
        "Plugin(s) declare audit_characteristics members that are not "
        "AuditCharacteristic enum instances. Bare strings render as the "
        "grey 'unknown' chip on the catalog card and silently degrade "
        "the audit-readiness signal. Offences:\n" + "\n".join(failures)
    )
