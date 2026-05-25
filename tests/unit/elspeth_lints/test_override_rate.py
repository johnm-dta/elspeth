"""Adversarial-boundary tests for the override-rate drift gate (C3).

Convergent panel finding C3: ``OVERRIDDEN_BY_OPERATOR`` is documented
as "a metric to watch" but watch-it-yourself is not a feedback loop.
``compute_override_rate`` makes the metric enforcing.

These tests probe the boundaries of:

1. Insufficient-data behaviour: small-N windows pass with a notice;
   the gate is informational until the denominator grows past
   ``--min-samples``. This is the documented behaviour, not an
   oversight — pin it explicitly so a future "let's be strict from
   day one" edit doesn't silently break the first weeks of operation.
2. Window inclusion math: ``judge_recorded_at >= now - window`` is
   the inclusion predicate. Entries exactly on the boundary are IN;
   entries one microsecond older are OUT.
3. Pre-judge entries (no ``judge_recorded_at``) contribute to
   neither numerator nor denominator. The gate measures rolling
   recent behaviour, not historical accumulation.
4. Future-dated entries are excluded (clock skew / tampering
   defence).
5. Threshold math: a rate strictly equal to ``--max-rate`` passes;
   strictly greater fails. The ``<=`` inclusive boundary is
   documented and tested.
6. Misconfiguration produces an ``OverrideRateError`` with an
   actionable message — never a silent fallback to default values.

Test discipline per M5: each "happy path" assertion is paired with
its boundary probe.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict
from elspeth_lints.core.judge import DEFAULT_JUDGE_MODEL, JUDGE_POLICY_HASH
from elspeth_lints.core.override_rate import (
    OverrideRateError,
    OverrideRateReport,
    append_judge_decision_event,
    compute_override_rate,
    default_counter_snapshot_path,
    write_override_rate_counter_snapshot,
)

_FAKE_JUDGE_METADATA_SIGNATURE = "hmac-sha256:v1:" + "0" * 64


def _make_allowlist_dir(tmp_path: Path) -> Path:
    """Create config/cicd/enforce_x/ inside tmp_path."""
    enforce_root = tmp_path / "config" / "cicd"
    enforce_dir = enforce_root / "enforce_x"
    enforce_dir.mkdir(parents=True)
    return enforce_root


def test_cli_help_documents_override_rate_denominator(capsys: pytest.CaptureFixture[str]) -> None:
    """Help text must match the denominator that ``compute_override_rate`` uses."""
    from elspeth_lints.core.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["check-override-rate", "--help"])

    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "fewer than this many judge-recorded entries fall inside the window" in help_text
    assert "OVERRIDDEN_BY_OPERATOR / (ACCEPTED + OVERRIDDEN_BY_OPERATOR) across the window" in help_text


def _write_entry(
    enforce_dir: Path,
    *,
    file_name: str,
    key: str,
    verdict: str,
    recorded_at: datetime,
    model_verdict: str | None = None,
    metadata_signature: str | None = _FAKE_JUDGE_METADATA_SIGNATURE,
) -> None:
    """Append one entry to a YAML file inside an enforce_x directory.

    Convenience helper — each test writes its own corpus shape.
    """
    yaml_path = enforce_dir / file_name
    iso = recorded_at.isoformat()
    extra = f"  judge_model_verdict: {model_verdict}\n" if model_verdict else ""
    signature_line = f"  judge_metadata_signature: '{metadata_signature}'\n" if metadata_signature is not None else ""
    # Binding fields (file_fingerprint + ast_path) are required co-presence
    # companions of judge_verdict (per C8-3 invariant 8). The override-rate
    # gate doesn't load through a source-root verifier so the values are
    # synthetic, just non-empty strings of the right shape. compute_override_rate
    # calls load_allowlist without source_root, so no live recompute happens.
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: {key}
          owner: alice
          reason: test
          safety: contained
          judge_verdict: {verdict}
          judge_recorded_at: '{iso}'
          judge_model: {DEFAULT_JUDGE_MODEL}
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: test rationale
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: 'body[0]'
        {signature_line.rstrip()}
        {extra}    """)
    )


# =========================================================================
# Insufficient-data behaviour
# =========================================================================


def test_zero_entries_passes_with_insufficient_data(tmp_path: Path) -> None:
    """Empty corpus: gate passes with insufficient-data signal.

    A fresh repository with no judged entries must not trip the gate
    — the rate is undefined, not 100%.
    """
    enforce_root = _make_allowlist_dir(tmp_path)
    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=10,
        max_rate=0.10,
        reference_time=datetime(2026, 5, 23, tzinfo=UTC),
    )
    assert detail.report.judged_in_window == 0
    assert detail.report.overrides_in_window == 0
    assert detail.report.insufficient_data
    assert detail.report.passes


def test_below_min_samples_passes_even_with_overrides(tmp_path: Path) -> None:
    """3 entries with 2 overrides (66.7%) passes under min_samples=10.

    A naive rate gate would trip mechanically on small N. Pin the
    documented behaviour so a future "let's be strict from day one"
    edit doesn't silently break early operation.
    """
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    _write_entry(
        enforce_dir,
        file_name="a.yaml",
        key="x.py:R1:a:fp=aa",
        verdict="ACCEPTED",
        recorded_at=now - timedelta(days=1),
    )
    # Two overrides in a 3-entry corpus is 67% — would trip 10%.
    # min_samples=10 should suppress.
    enforce_dir_b = enforce_root / "enforce_y"
    enforce_dir_b.mkdir()
    _write_entry(
        enforce_dir_b,
        file_name="b.yaml",
        key="x.py:R1:b:fp=bb",
        verdict="OVERRIDDEN_BY_OPERATOR",
        recorded_at=now - timedelta(days=2),
        model_verdict="BLOCKED",
    )
    _write_entry(
        enforce_dir_b,
        file_name="c.yaml",
        key="x.py:R1:c:fp=cc",
        verdict="OVERRIDDEN_BY_OPERATOR",
        recorded_at=now - timedelta(days=3),
        model_verdict="BLOCKED",
    )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=10,
        max_rate=0.10,
        reference_time=now,
    )
    assert detail.report.judged_in_window == 3
    assert detail.report.overrides_in_window == 2
    assert detail.report.rate == pytest.approx(2 / 3)
    assert detail.report.insufficient_data  # < min_samples
    assert detail.report.passes


# =========================================================================
# Window-boundary math
# =========================================================================


def test_entry_exactly_on_window_boundary_is_included(tmp_path: Path) -> None:
    """``judge_recorded_at >= reference_time - window`` is the inclusion test.

    An entry at the exact boundary moment must be IN the window.
    """
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    boundary = now - timedelta(days=30)
    _write_entry(
        enforce_dir,
        file_name="a.yaml",
        key="x.py:R1:a:fp=aa",
        verdict="ACCEPTED",
        recorded_at=boundary,
    )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=1,
        max_rate=0.10,
        reference_time=now,
    )
    assert detail.report.judged_in_window == 1


def test_entry_one_microsecond_before_boundary_is_excluded(tmp_path: Path) -> None:
    """An entry just outside the window must NOT contribute."""
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    before_boundary = now - timedelta(days=30, microseconds=1)
    _write_entry(
        enforce_dir,
        file_name="a.yaml",
        key="x.py:R1:a:fp=aa",
        verdict="ACCEPTED",
        recorded_at=before_boundary,
    )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=1,
        max_rate=0.10,
        reference_time=now,
    )
    assert detail.report.judged_in_window == 0


def test_judged_entry_without_metadata_signature_is_rejected(tmp_path: Path) -> None:
    """C3 must not count unsigned judged entries as authentic rate data."""
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    _write_entry(
        enforce_dir,
        file_name="a.yaml",
        key="x.py:R1:a:fp=aa",
        verdict="ACCEPTED",
        recorded_at=now - timedelta(days=1),
        metadata_signature=None,
    )

    with pytest.raises(OverrideRateError, match="judge_metadata_signature"):
        compute_override_rate(
            allowlist_root=enforce_root,
            window_days=30,
            min_samples=1,
            max_rate=0.10,
            reference_time=now,
        )


def test_future_dated_entries_raise_tampering_detected(tmp_path: Path) -> None:
    """Entries with judge_recorded_at after reference_time raise TAMPERING_DETECTED (C7-6).

    The prior behaviour silently excluded future-dated entries from
    both numerator and denominator. That is precisely the silent
    drop the gate exists to prevent: an operator could forward-date
    an override past ``reference_time`` to vanish from the rolling
    denominator, OR backdate one out of the rolling-recent window,
    and the gate would happily report a clean rate.

    Fail-closed (C7-6): we raise ``OverrideRateError`` with a
    ``TAMPERING_DETECTED`` prefix and the offending entry key. The
    CLI surfaces this as exit-2 (gate broken; operator must
    investigate).
    """
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    _write_entry(
        enforce_dir,
        file_name="a.yaml",
        key="x.py:R1:a:fp=aa",
        verdict="OVERRIDDEN_BY_OPERATOR",
        recorded_at=now + timedelta(days=1),
        model_verdict="BLOCKED",
    )

    with pytest.raises(OverrideRateError) as exc:
        compute_override_rate(
            allowlist_root=enforce_root,
            window_days=30,
            min_samples=1,
            max_rate=0.10,
            reference_time=now,
        )
    message = str(exc.value)
    assert "TAMPERING_DETECTED" in message
    assert "x.py:R1:a:fp=aa" in message
    assert "a.yaml" in message


def test_future_dated_entry_exit_code_is_two(tmp_path: Path) -> None:
    """CLI handler returns exit code 2 (gate broken) on TAMPERING_DETECTED (C7-6).

    Pins the CLI contract: ``OverrideRateError`` => exit 2 (operator
    investigates), distinct from exit 1 (rate exceeded budget — a
    business-decision outcome the operator triages).
    """
    from elspeth_lints.core.cli import main

    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    future = now + timedelta(days=1)
    _write_entry(
        enforce_dir,
        file_name="a.yaml",
        key="x.py:R1:a:fp=aa",
        verdict="OVERRIDDEN_BY_OPERATOR",
        recorded_at=future,
        model_verdict="BLOCKED",
    )

    exit_code = main(
        [
            "check-override-rate",
            "--allowlist-root",
            str(enforce_root),
            "--window-days",
            "30",
            "--min-samples",
            "1",
            "--max-rate",
            "0.10",
            "--reference-time",
            now.isoformat(),
        ]
    )
    assert exit_code == 2


def test_pre_judge_entries_contribute_to_neither_side(tmp_path: Path) -> None:
    """Entries with judge_recorded_at=None are excluded from both numerator and denominator."""
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    yaml_path = enforce_dir / "pre_judge.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: x.py:R1:legacy:fp=aa
          owner: alice
          reason: pre-judge era entry
          safety: contained
    """)
    )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=1,
        max_rate=0.10,
        reference_time=datetime(2026, 5, 23, tzinfo=UTC),
    )
    assert detail.report.judged_in_window == 0


# =========================================================================
# Threshold math
# =========================================================================


def test_rate_exactly_at_threshold_passes(tmp_path: Path) -> None:
    """The threshold is inclusive: rate == max_rate passes.

    Pinned so a future ``<`` vs ``<=`` swap surfaces as a regression
    rather than silently failing legitimate operation at the edge.
    """
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    # 1 override / 10 total = 0.10 = exactly max_rate.
    _write_entry(
        enforce_dir,
        file_name="override.yaml",
        key="x.py:R1:override:fp=aa",
        verdict="OVERRIDDEN_BY_OPERATOR",
        recorded_at=now - timedelta(days=1),
        model_verdict="BLOCKED",
    )
    for index in range(9):
        _write_entry(
            enforce_dir,
            file_name=f"a_{index}.yaml",
            key=f"x.py:R1:a{index}:fp={index:016x}",
            verdict="ACCEPTED",
            recorded_at=now - timedelta(days=1),
        )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=10,
        max_rate=0.10,
        reference_time=now,
    )
    assert detail.report.judged_in_window == 10
    assert detail.report.overrides_in_window == 1
    assert detail.report.rate == pytest.approx(0.10)
    assert not detail.report.insufficient_data
    assert detail.report.passes


def test_rate_above_threshold_fails(tmp_path: Path) -> None:
    """A rate above the budget fails the gate and lists overrides."""
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    # 2 overrides / 10 total = 0.20 — over 0.10 budget.
    for index in range(2):
        _write_entry(
            enforce_dir,
            file_name=f"override_{index}.yaml",
            key=f"x.py:R1:override{index}:fp={index:016x}",
            verdict="OVERRIDDEN_BY_OPERATOR",
            recorded_at=now - timedelta(days=index + 1),
            model_verdict="BLOCKED",
        )
    for index in range(8):
        _write_entry(
            enforce_dir,
            file_name=f"a_{index}.yaml",
            key=f"x.py:R1:a{index}:fp={index + 10:016x}",
            verdict="ACCEPTED",
            recorded_at=now - timedelta(days=index + 1),
        )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=10,
        max_rate=0.10,
        reference_time=now,
    )
    assert detail.report.judged_in_window == 10
    assert detail.report.overrides_in_window == 2
    assert detail.report.rate == pytest.approx(0.20)
    assert not detail.report.passes
    assert len(detail.override_entries) == 2


def test_per_rule_reports_partition_the_denominator(tmp_path: Path) -> None:
    """Aggregate rate must expose per-rule calibration signals."""
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    for index in range(2):
        _write_entry(
            enforce_dir,
            file_name=f"r1_override_{index}.yaml",
            key=f"x.py:R1:override{index}:fp={index:016x}",
            verdict="OVERRIDDEN_BY_OPERATOR",
            recorded_at=now - timedelta(days=1),
            model_verdict="BLOCKED",
        )
    for index in range(8):
        _write_entry(
            enforce_dir,
            file_name=f"r5_accept_{index}.yaml",
            key=f"x.py:R5:accept{index}:fp={index + 10:016x}",
            verdict="ACCEPTED",
            recorded_at=now - timedelta(days=1),
        )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=10,
        max_rate=0.50,
        reference_time=now,
    )

    by_rule = {report.rule_id: report for report in detail.per_rule_reports}
    assert detail.report.accepted_in_window == 8
    assert detail.report.overrides_in_window == 2
    assert detail.report.model_accepted_in_window == 8
    assert detail.report.model_blocked_in_window == 2
    assert by_rule["R1"].judged_in_window == 2
    assert by_rule["R1"].overrides_in_window == 2
    assert by_rule["R1"].rate == pytest.approx(1.0)
    assert by_rule["R5"].judged_in_window == 8
    assert by_rule["R5"].accepted_in_window == 8
    assert by_rule["R5"].rate == pytest.approx(0.0)


def test_absolute_override_count_budget_prevents_dilution(tmp_path: Path) -> None:
    """A large ACCEPTED population cannot dilute overrides past an absolute cap."""
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    for index in range(2):
        _write_entry(
            enforce_dir,
            file_name=f"override_{index}.yaml",
            key=f"x.py:R1:override{index}:fp={index:016x}",
            verdict="OVERRIDDEN_BY_OPERATOR",
            recorded_at=now - timedelta(days=1),
            model_verdict="BLOCKED",
        )
    for index in range(98):
        _write_entry(
            enforce_dir,
            file_name=f"accepted_{index}.yaml",
            key=f"x.py:R5:accepted{index}:fp={index + 100:016x}",
            verdict="ACCEPTED",
            recorded_at=now - timedelta(days=1),
        )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=10,
        max_rate=0.10,
        max_overrides=1,
        reference_time=now,
    )

    assert detail.report.rate == pytest.approx(0.02)
    assert detail.report.ratio_budget_exceeded is False
    assert detail.report.absolute_budget_exceeded is True
    assert not detail.report.passes


def test_counter_snapshot_is_consumed_without_yaml_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh counter snapshot lets C3 consume structured counters directly."""
    import elspeth_lints.core.override_rate as override_rate_module

    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    _write_entry(
        enforce_dir,
        file_name="accepted.yaml",
        key="x.py:R1:accepted:fp=aa",
        verdict="ACCEPTED",
        recorded_at=now - timedelta(days=1),
    )
    _write_entry(
        enforce_dir,
        file_name="override.yaml",
        key="x.py:R1:override:fp=bb",
        verdict="OVERRIDDEN_BY_OPERATOR",
        recorded_at=now - timedelta(days=1),
        model_verdict="BLOCKED",
    )
    snapshot = write_override_rate_counter_snapshot(enforce_root)

    def fail_yaml_rescan(_entry_dir: Path) -> list[object]:
        raise AssertionError("compute_override_rate reparsed YAML instead of consuming counters")

    monkeypatch.setattr(override_rate_module, "_iterate_allow_hits", fail_yaml_rescan)

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=1,
        max_rate=0.75,
        reference_time=now,
        counter_snapshot_path=snapshot.path,
    )

    assert detail.counter_source == "counter_snapshot"
    assert detail.report.judged_in_window == 2
    assert detail.report.accepted_in_window == 1
    assert detail.report.overrides_in_window == 1


def test_stale_counter_snapshot_falls_back_to_yaml_and_reports_source(tmp_path: Path) -> None:
    """Snapshot hash drift cannot hide a current YAML change."""
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    _write_entry(
        enforce_dir,
        file_name="accepted.yaml",
        key="x.py:R1:accepted:fp=aa",
        verdict="ACCEPTED",
        recorded_at=now - timedelta(days=1),
    )
    snapshot = write_override_rate_counter_snapshot(enforce_root)
    _write_entry(
        enforce_dir,
        file_name="override.yaml",
        key="x.py:R1:override:fp=bb",
        verdict="OVERRIDDEN_BY_OPERATOR",
        recorded_at=now - timedelta(days=1),
        model_verdict="BLOCKED",
    )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=1,
        max_rate=0.75,
        reference_time=now,
        counter_snapshot_path=snapshot.path,
    )

    assert detail.counter_source == "yaml"
    assert detail.report.judged_in_window == 2
    assert detail.report.overrides_in_window == 1


def test_cli_check_override_rate_refreshes_counter_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The deployed gate emits a reusable counter snapshot after a YAML scan."""
    from elspeth_lints.core.cli import main

    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    _write_entry(
        enforce_dir,
        file_name="accepted.yaml",
        key="x.py:R1:accepted:fp=aa",
        verdict="ACCEPTED",
        recorded_at=now - timedelta(days=1),
    )

    exit_code = main(
        [
            "check-override-rate",
            "--allowlist-root",
            str(enforce_root),
            "--window-days",
            "30",
            "--min-samples",
            "1",
            "--max-rate",
            "0.10",
            "--reference-time",
            now.isoformat(),
        ]
    )

    assert exit_code == 0
    assert default_counter_snapshot_path(enforce_root).exists()
    captured = capsys.readouterr()
    assert "counter_source=yaml" in captured.out
    assert "refreshed counter snapshot" in captured.err


def test_blocked_without_override_events_surface_under_override_counter(tmp_path: Path) -> None:
    """C3 reports the one-sided pressure to refactor instead of overriding."""
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    append_judge_decision_event(
        enforce_dir,
        source_file="blocked.yaml",
        entry_key="x.py:R1:blocked:fp=aa",
        rule_id="R1",
        effective_verdict=JudgeVerdict.BLOCKED,
        model_verdict=JudgeVerdict.BLOCKED,
        recorded_at=now - timedelta(days=1),
        write_disposition="blocked_without_override",
    )

    detail = compute_override_rate(
        allowlist_root=enforce_root,
        window_days=30,
        min_samples=1,
        max_rate=0.10,
        reference_time=now,
    )

    assert detail.report.blocked_without_override_in_window == 1


def test_cli_check_override_rate_prints_under_override_counter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The deployed PR-summary surface includes refactor-instead-of-override pressure."""
    from elspeth_lints.core.cli import main

    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    append_judge_decision_event(
        enforce_dir,
        source_file="blocked.yaml",
        entry_key="x.py:R1:blocked:fp=aa",
        rule_id="R1",
        effective_verdict=JudgeVerdict.BLOCKED,
        model_verdict=JudgeVerdict.BLOCKED,
        recorded_at=now - timedelta(days=1),
        write_disposition="blocked_without_override",
    )

    exit_code = main(
        [
            "check-override-rate",
            "--allowlist-root",
            str(enforce_root),
            "--window-days",
            "30",
            "--min-samples",
            "1",
            "--max-rate",
            "0.10",
            "--reference-time",
            now.isoformat(),
        ]
    )

    assert exit_code == 0
    assert "blocked_without_override_in_window=1" in capsys.readouterr().out


def test_cli_absolute_override_count_budget_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The deployed CLI honors ``--max-overrides`` as a failing gate."""
    from elspeth_lints.core.cli import main

    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    for index in range(2):
        _write_entry(
            enforce_dir,
            file_name=f"override_{index}.yaml",
            key=f"x.py:R1:override{index}:fp={index:016x}",
            verdict="OVERRIDDEN_BY_OPERATOR",
            recorded_at=now - timedelta(days=1),
            model_verdict="BLOCKED",
        )
    for index in range(98):
        _write_entry(
            enforce_dir,
            file_name=f"accepted_{index}.yaml",
            key=f"x.py:R5:accepted{index}:fp={index + 100:016x}",
            verdict="ACCEPTED",
            recorded_at=now - timedelta(days=1),
        )

    exit_code = main(
        [
            "check-override-rate",
            "--allowlist-root",
            str(enforce_root),
            "--window-days",
            "30",
            "--min-samples",
            "10",
            "--max-rate",
            "0.10",
            "--max-overrides",
            "1",
            "--reference-time",
            now.isoformat(),
        ]
    )

    assert exit_code == 1
    assert "override count 2 exceeds absolute budget 1" in capsys.readouterr().out


def test_cli_absolute_override_count_budget_fails_with_insufficient_ratio_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--max-overrides`` is an absolute cap, not a rate-only small-N guard."""
    from elspeth_lints.core.cli import main

    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    now = datetime(2026, 5, 23, tzinfo=UTC)
    _write_entry(
        enforce_dir,
        file_name="override.yaml",
        key="x.py:R1:override:fp=aa",
        verdict="OVERRIDDEN_BY_OPERATOR",
        recorded_at=now - timedelta(days=1),
        model_verdict="BLOCKED",
    )

    exit_code = main(
        [
            "check-override-rate",
            "--allowlist-root",
            str(enforce_root),
            "--window-days",
            "30",
            "--min-samples",
            "10",
            "--max-rate",
            "0.10",
            "--max-overrides",
            "0",
            "--reference-time",
            now.isoformat(),
        ]
    )

    assert exit_code == 1
    assert "override count 1 exceeds absolute budget 0" in capsys.readouterr().out


# =========================================================================
# Misconfiguration
# =========================================================================


def test_negative_window_days_rejected(tmp_path: Path) -> None:
    enforce_root = _make_allowlist_dir(tmp_path)
    with pytest.raises(OverrideRateError) as exc:
        compute_override_rate(
            allowlist_root=enforce_root,
            window_days=-1,
            min_samples=10,
            max_rate=0.10,
        )
    assert "window-days must be positive" in str(exc.value)


def test_negative_min_samples_rejected(tmp_path: Path) -> None:
    enforce_root = _make_allowlist_dir(tmp_path)
    with pytest.raises(OverrideRateError) as exc:
        compute_override_rate(
            allowlist_root=enforce_root,
            window_days=30,
            min_samples=-1,
            max_rate=0.10,
        )
    assert "min-samples must be non-negative" in str(exc.value)


def test_max_rate_out_of_range_rejected(tmp_path: Path) -> None:
    enforce_root = _make_allowlist_dir(tmp_path)
    for bad_rate in (-0.01, 1.01, 5.0):
        with pytest.raises(OverrideRateError) as exc:
            compute_override_rate(
                allowlist_root=enforce_root,
                window_days=30,
                min_samples=10,
                max_rate=bad_rate,
            )
        assert "max-rate must be in [0.0, 1.0]" in str(exc.value)


def test_naive_reference_time_rejected(tmp_path: Path) -> None:
    """A naive datetime would silently compare wrong against tz-aware entries."""
    enforce_root = _make_allowlist_dir(tmp_path)
    with pytest.raises(OverrideRateError) as exc:
        compute_override_rate(
            allowlist_root=enforce_root,
            window_days=30,
            min_samples=10,
            max_rate=0.10,
            reference_time=datetime(2026, 5, 23),  # noqa: DTZ001 - intentional naive value to test rejection
        )
    assert "timezone-aware" in str(exc.value)


def test_malformed_yaml_raises_override_rate_error(tmp_path: Path) -> None:
    """Malformed YAML raises ``OverrideRateError`` (not an uncaught traceback) — C7-5.

    ``yaml.safe_load`` raises ``yaml.YAMLError``, a sibling of
    ``Exception`` rather than a ``ValueError`` subclass. The prior
    handler caught only ``ValueError`` and let ``YAMLError`` escape
    as exit-1 ("gate broken") with a raw traceback — the wrong
    failure shape for the documented exit-2 ("operator-actionable
    structured message") contract.
    """
    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    (enforce_dir / "bad.yaml").write_text("key: [unclosed\n")

    with pytest.raises(OverrideRateError) as exc:
        compute_override_rate(
            allowlist_root=enforce_root,
            window_days=30,
            min_samples=1,
            max_rate=0.10,
            reference_time=datetime(2026, 5, 23, tzinfo=UTC),
        )
    assert "failed to parse as YAML mapping" in str(exc.value)
    assert "bad.yaml" in str(exc.value)


def test_malformed_yaml_exit_code_is_two(tmp_path: Path) -> None:
    """CLI returns exit 2 on malformed YAML (C7-5 CLI surface)."""
    from elspeth_lints.core.cli import main

    enforce_root = _make_allowlist_dir(tmp_path)
    enforce_dir = enforce_root / "enforce_x"
    (enforce_dir / "bad.yaml").write_text("key: [unclosed\n")

    exit_code = main(
        [
            "check-override-rate",
            "--allowlist-root",
            str(enforce_root),
            "--window-days",
            "30",
            "--min-samples",
            "1",
            "--max-rate",
            "0.10",
            "--reference-time",
            "2026-05-23T00:00:00+00:00",
        ]
    )
    assert exit_code == 2


def test_missing_allowlist_root_rejected(tmp_path: Path) -> None:
    with pytest.raises(OverrideRateError) as exc:
        compute_override_rate(
            allowlist_root=tmp_path / "nonexistent",
            window_days=30,
            min_samples=10,
            max_rate=0.10,
        )
    assert "not a directory" in str(exc.value)


# =========================================================================
# Report invariants
# =========================================================================


def test_report_rate_on_empty_window_is_zero() -> None:
    """The rate property returns 0.0 (not NaN, not ZeroDivisionError) for empty windows."""
    report = OverrideRateReport(
        window_days=30,
        reference_time=datetime(2026, 5, 23, tzinfo=UTC),
        min_samples=10,
        max_rate=0.10,
        max_overrides=None,
        judged_in_window=0,
        accepted_in_window=0,
        overrides_in_window=0,
        model_accepted_in_window=0,
        model_blocked_in_window=0,
        blocked_without_override_in_window=0,
    )
    assert report.rate == 0.0
