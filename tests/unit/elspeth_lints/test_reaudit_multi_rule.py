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

import json
import os
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from elspeth_lints.core.allowlist import AllowlistEntry, JudgeVerdict
from elspeth_lints.core.judge import DEFAULT_JUDGE_MODEL, JUDGE_POLICY_HASH, JudgeRequest, JudgeResponse
from elspeth_lints.core.reaudit import (
    _EXCLUDED_FROM_REAUDIT,
    _RULE_VOCABULARY_REGISTRY,
    ReauditDivergence,
    ReauditError,
    _entry_matches_rule_filter,
    _scan_via_rule_analyze,
    _supported_rules,
    _valid_rule_ids_for,
    reaudit_entries,
)

_TEST_JUDGE_METADATA_HMAC_KEY = "test-judge-metadata-hmac-key-2026-05-24"


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


def test_vocabulary_registry_covers_every_supported_rule() -> None:
    """The dispatch table is declarative and covers the supported rule set."""
    assert set(_RULE_VOCABULARY_REGISTRY) == set(_supported_rules())


@pytest.mark.parametrize(
    ("rule_filter", "emitted_rule_id"),
    (
        ("trust_boundary.tests", "TBE1"),
        ("trust_boundary.tests", "R_TB_TESTS_IRRELEVANT_INPUT"),
        ("trust_boundary.scope", "TBS2"),
        ("trust_boundary.tier", "TBT1"),
    ),
)
def test_trust_boundary_reaudit_vocabulary_includes_emitted_rule_ids(
    rule_filter: str,
    emitted_rule_id: str,
) -> None:
    """Trust-boundary allowlist entries use sub-rule ids, not package ids."""
    valid_rule_ids = _valid_rule_ids_for(rule_filter)
    entry = AllowlistEntry(
        key=f"src/elspeth/boundary.py:{emitted_rule_id}:handler:fp=abc123",
        owner="test-owner",
        reason="Fixture entry for rule-filter matching.",
        safety="No runtime suppression; vocabulary regression only.",
        expires=None,
    )

    assert emitted_rule_id in valid_rule_ids
    assert _entry_matches_rule_filter(entry, valid_rule_ids)


def test_unregistered_rule_filter_raises_reaudit_error() -> None:
    """A rule_id with no vocabulary registration raises a clear error.

    Catches the "added to BUILTIN_RULES but forgot the
    _valid_rule_ids_for branch" footgun.
    """
    with pytest.raises(ReauditError) as exc_info:
        _valid_rule_ids_for("nonexistent.fake.rule")
    assert "No sub-rule vocabulary is registered" in str(exc_info.value)
    assert "_RULE_VOCABULARY_REGISTRY" in str(exc_info.value)


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


def test_generic_dispatch_reaudits_composer_catch_order_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-tier rule re-locates a current finding, calls the judge, and classifies it.

    The earlier multi-rule tests pinned registry/vocabulary shape but
    did not exercise ``reaudit_entries`` through the generic
    ``Rule.analyze`` scanner plus canonical-key matcher. ``composer.catch_order``
    is the smallest self-contained rule that produces a deterministic
    finding without sibling allowlist state.

    The entry is **pre-judge** (``judge_verdict`` absent). That is the only
    honest shape for this rule: ``composer.catch_order`` is a generic
    ``protocols.Finding`` rule that never stamps ``ast_path`` (it defaults
    to ``""``), and a judge-gated entry requires a non-empty ast_path —
    the ``justify`` write path (``cli._finding_ast_path``) refuses to sign
    a finding with an empty ast_path, and the production match gate
    (``trust_boundary.shared._allowlist_match``) crashes on one. So a
    *signed* ``composer.catch_order`` entry cannot exist in production;
    a pre-judge entry exercises the same generic dispatch + canonical-key
    match + judge + classify path without fabricating an ast_path the
    scanner never emits. The pre-re-judge binding check correctly
    early-returns for pre-judge entries (no binding fields to verify).
    """
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)
    root = tmp_path / "src_root"
    target = root / "web" / "sessions" / "routes.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        textwrap.dedent(
            """\
            def f():
                try:
                    pass
                except ComposerServiceError as exc:
                    pass
                except ComposerPluginCrashError as crash:
                    pass
            """
        ),
        encoding="utf-8",
    )
    finding = _single_catch_order_finding(root=root, target=target)
    entry_key = finding.canonical_key()
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    (allowlist_dir / "_defaults.yaml").write_text(
        "version: 1\ndefaults:\n  fail_on_stale: false\n  fail_on_expired: false\n",
        encoding="utf-8",
    )
    _write_pre_judge_allow_hit(
        allowlist_dir / "composer.yaml",
        key=entry_key,
    )

    with _mock_judge_call(verdict="ACCEPTED", rationale="catch-order suppression still warranted") as client_class:
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="composer.catch_order",
            since=None,
            limit=None,
            include_pre_judge=True,
        )

    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.divergence is ReauditDivergence.PRE_JUDGE_FRESH_ACCEPT
    assert outcome.fresh_verdict is JudgeVerdict.ACCEPTED
    assert "ComposerServiceError" in outcome.code_snapshot
    assert client_class.return_value.chat.completions.create.call_count == 1


def test_reaudit_uses_non_tier_rule_definition_for_generic_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-tier-model reaudit judge call receives the active rule's definition."""
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)
    root = tmp_path / "src_root"
    target = root / "models.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        textwrap.dedent(
            """\
            from dataclasses import dataclass
            from types import MappingProxyType


            @dataclass(frozen=True)
            class FrozenConfig:
                values: dict[str, str]

                def __post_init__(self) -> None:
                    object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
            """
        ),
        encoding="utf-8",
    )
    findings = _scan_via_rule_analyze(
        rule_filter="immutability.freeze_guards",
        target_file=target,
        root=root,
    )
    finding = next(finding for finding in findings if finding.rule_id == "FG1")
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    (allowlist_dir / "_defaults.yaml").write_text(
        "version: 1\ndefaults:\n  fail_on_stale: false\n  fail_on_expired: false\n",
        encoding="utf-8",
    )
    _write_pre_judge_allow_hit(allowlist_dir / "freeze.yaml", key=finding.canonical_key())
    captured: dict[str, str] = {}

    def _fake_call_judge(request: JudgeRequest, **_kwargs: object) -> JudgeResponse:
        captured["rule_definition"] = request.rule_definition
        return JudgeResponse(
            verdict=JudgeVerdict.ACCEPTED,
            model_id=DEFAULT_JUDGE_MODEL,
            judge_rationale="freeze guard suppression still warranted",
            recorded_at=datetime.now(UTC),
            should_use_decorator=None,
            confidence=0.91,
            prompt_tokens_total=4000,
            prompt_tokens_cached=0,
            policy_hash=JUDGE_POLICY_HASH,
            judge_transport="openrouter",
        )

    monkeypatch.setattr("elspeth_lints.core.reaudit.call_judge", _fake_call_judge)

    report = reaudit_entries(
        root=root.resolve(),
        allowlist_dir=allowlist_dir,
        rule_filter="immutability.freeze_guards",
        since=None,
        limit=None,
        include_pre_judge=True,
    )

    assert report.outcomes[0].divergence is ReauditDivergence.PRE_JUDGE_FRESH_ACCEPT
    assert "FG1" in captured["rule_definition"]
    assert "Bare MappingProxyType wrap" in captured["rule_definition"]
    assert "recursive immutability" in captured["rule_definition"]
    assert "no definition available" not in captured["rule_definition"]


def _single_catch_order_finding(*, root: Path, target: Path):
    findings = _scan_via_rule_analyze(
        rule_filter="composer.catch_order",
        target_file=target,
        root=root,
    )
    assert len(findings) == 1
    return findings[0]


def _write_pre_judge_allow_hit(path: Path, *, key: str) -> None:
    """Write a pre-judge (judge_verdict-absent) allow-hit for ``key``.

    Pre-judge entries carry no judge-metadata cluster and no binding
    fields (file_fingerprint/scope_fingerprint/ast_path/signature) — the
    canonical key is their only binding signal. This is the honest entry
    shape for a generic ``Rule.analyze`` rule whose findings have an empty
    ast_path and therefore cannot be judge-signed (see the end-to-end
    test's docstring). Reaudit classifies it as PRE_JUDGE_FRESH_ACCEPT
    when the fresh judge verdict is ACCEPTED.
    """
    path.write_text(
        "\n".join(
            [
                "allow_hits:",
                f"- key: {key}",
                "  owner: test-owner",
                "  reason: |-",
                "    Broad-before-narrow catch order is quarantined during migration.",
                "  safety: |-",
                "    Reaudit replays the generic composer.catch_order scanner before accepting the entry.",
                "  expires: '2030-01-01'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _mock_openrouter_completion(*, verdict: str, rationale: str, served_model: str | None = DEFAULT_JUDGE_MODEL) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps({"verdict": verdict, "rationale": rationale, "confidence": 0.91, "should_use_decorator": None})
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    completion = MagicMock()
    completion.choices = [choice]
    completion.model = served_model
    completion.usage = MagicMock(
        prompt_tokens=4000,
        prompt_tokens_details=MagicMock(cached_tokens=0),
    )
    return completion


@contextmanager
def _mock_judge_call(*, verdict: str, rationale: str, served_model: str | None = DEFAULT_JUDGE_MODEL) -> Iterator[MagicMock]:
    fake_completion = _mock_openrouter_completion(verdict=verdict, rationale=rationale, served_model=served_model)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client) as client_class,
    ):
        yield client_class
