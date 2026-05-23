"""Adversarial-boundary tests for the multi-rule reaudit dispatch (C2).

Convergent panel finding C2 expanded reaudit's ``_SUPPORTED_RULES``
from ``frozenset({"trust_tier.tier_model"})`` to a registry covering
every ``BUILTIN_RULES`` entry minus explicit exclusions. The dispatch
inside ``_scan_findings_for_file`` now branches:

* ``trust_tier.tier_model`` → bespoke ``_scan_tier_model`` (layer
  imports cross file boundaries in a way ``Rule.analyze`` does not
  model).
* every other supported rule → generic ``_scan_via_rule_analyze``
  (standard ``Rule.analyze`` protocol).

These tests pin:

1. Every ``BUILTIN_RULES`` rule_id except those in
   ``_EXCLUDED_FROM_REAUDIT`` is accepted by ``--rule``.
2. Every accepted rule_id resolves to a non-empty sub-rule
   vocabulary through ``_valid_rule_ids_for``.
3. ``_EXCLUDED_FROM_REAUDIT`` rules are rejected with a clear
   diagnostic that names the excluded set.
4. The generic ``Rule.analyze`` dispatch returns findings for one
   non-tier_model rule end-to-end (smoke).

The cultural discipline (per M5, "positive-path-biased test design")
is to make these tests fail by changing the registry — if a future
edit removes a rule from ``_supported_rules`` or breaks the
vocabulary lookup, the regression should land here rather than as a
silent reaudit miscount.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from elspeth_lints.core.reaudit import (
    _EXCLUDED_FROM_REAUDIT,
    ReauditError,
    _scan_via_rule_analyze,
    _supported_rules,
    _valid_rule_ids_for,
    reaudit_entries,
)


def test_supported_rules_covers_all_builtin_except_explicit_exclusions() -> None:
    """The supported set is derived from BUILTIN_RULES minus the carve-outs.

    If a rule is added to BUILTIN_RULES, it becomes reaudit-targetable
    automatically (unless explicitly excluded). The test pins this
    derivation so a future "we forgot to add it to _SUPPORTED_RULES"
    bug surfaces here.
    """
    from elspeth_lints.rules import BUILTIN_RULES

    all_ids = {rule.id for rule in BUILTIN_RULES}
    expected = all_ids - _EXCLUDED_FROM_REAUDIT
    assert _supported_rules() == frozenset(expected)


def test_explicit_exclusions_are_real_rule_ids() -> None:
    """Every exclusion must name a rule that actually exists.

    Otherwise the exclusion is dead code: it silently does nothing
    and the operator believes a rule is carved out when it isn't.
    """
    from elspeth_lints.rules import BUILTIN_RULES

    all_ids = {rule.id for rule in BUILTIN_RULES}
    for excluded in _EXCLUDED_FROM_REAUDIT:
        assert excluded in all_ids, f"{excluded!r} is listed in _EXCLUDED_FROM_REAUDIT but not in BUILTIN_RULES — the exclusion is a no-op."


def test_every_supported_rule_resolves_a_non_empty_vocabulary() -> None:
    """Each supported rule must register a sub-rule vocabulary.

    A supported rule with an empty vocabulary would break
    ``load_allowlist`` for any directory that carries sibling
    ``per_file_rules``. This test pins the registry's completeness.
    """
    for rule_id in sorted(_supported_rules()):
        vocab = _valid_rule_ids_for(rule_id)
        assert isinstance(vocab, frozenset), f"{rule_id}: _valid_rule_ids_for must return frozenset, got {type(vocab).__name__}"
        assert vocab, f"{rule_id}: empty vocabulary returned"


def test_unregistered_rule_filter_raises_reaudit_error() -> None:
    """A rule_id with no vocabulary registration raises a clear error.

    Catches the "added to BUILTIN_RULES but forgot the
    _valid_rule_ids_for branch" footgun.
    """
    with pytest.raises(ReauditError) as exc_info:
        _valid_rule_ids_for("nonexistent.fake.rule")
    assert "No sub-rule vocabulary is registered" in str(exc_info.value)
    assert "_valid_rule_ids_for" in str(exc_info.value)


def test_reaudit_entries_rejects_unsupported_rule_filter(tmp_path: Path) -> None:
    """``--rule`` outside the supported set is rejected before any work."""
    allowlist_dir = tmp_path / "enforce_x"
    allowlist_dir.mkdir()
    with pytest.raises(ReauditError) as exc_info:
        reaudit_entries(
            root=tmp_path,
            allowlist_dir=allowlist_dir,
            rule_filter="nonexistent.fake.rule",
            since=None,
            limit=None,
            include_pre_judge=False,
        )
    message = str(exc_info.value)
    assert "is not supported by reaudit" in message
    assert "_EXCLUDED_FROM_REAUDIT" in message


def test_reaudit_entries_rejects_excluded_rule(tmp_path: Path) -> None:
    """Excluded rules (audit_evidence.nominal_base, meta.*) are rejected."""
    allowlist_dir = tmp_path / "enforce_x"
    allowlist_dir.mkdir()
    for excluded in sorted(_EXCLUDED_FROM_REAUDIT):
        with pytest.raises(ReauditError) as exc_info:
            reaudit_entries(
                root=tmp_path,
                allowlist_dir=allowlist_dir,
                rule_filter=excluded,
                since=None,
                limit=None,
                include_pre_judge=False,
            )
        assert "is not supported by reaudit" in str(exc_info.value)


def test_generic_dispatch_returns_no_findings_for_clean_file(tmp_path: Path) -> None:
    """A clean Python file produces zero findings under the generic Rule.analyze path.

    Smoke test for the non-tier_model dispatch branch.
    ``composer.catch_order`` is a self-contained rule (no sibling
    allowlist on disk), which makes it the cheapest probe of the
    generic ``Rule.analyze`` pipeline. A function-only module
    contains no except-handler chains for the rule to flag; zero
    findings is the correct empty-input behaviour.
    """
    target = tmp_path / "clean.py"
    target.write_text(
        textwrap.dedent('''\
        """Clean module — no composer exception-ordering violations."""


        def hello() -> str:
            return "hi"
    ''')
    )
    findings = _scan_via_rule_analyze(
        rule_filter="composer.catch_order",
        target_file=target,
        root=tmp_path,
    )
    assert findings == []


def test_generic_dispatch_handles_syntax_error_as_empty(tmp_path: Path) -> None:
    """A file that fails to parse contributes zero findings.

    Consistent with the CI run's behaviour: a file that doesn't parse
    can't be analysed; we don't pretend its prior findings are still
    valid. The reaudit downstream classifies the entry as
    ``ENTRY_OBSOLETE`` in that case.
    """
    target = tmp_path / "broken.py"
    target.write_text("def hello(:\n    return\n")
    findings = _scan_via_rule_analyze(
        rule_filter="composer.catch_order",
        target_file=target,
        root=tmp_path,
    )
    assert findings == []
