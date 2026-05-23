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

from elspeth_lints.core.override_rate import (
    OverrideRateError,
    OverrideRateReport,
    compute_override_rate,
)


def _make_allowlist_dir(tmp_path: Path) -> Path:
    """Create config/cicd/enforce_x/ inside tmp_path."""
    enforce_root = tmp_path / "config" / "cicd"
    enforce_dir = enforce_root / "enforce_x"
    enforce_dir.mkdir(parents=True)
    return enforce_root


def _write_entry(
    enforce_dir: Path,
    *,
    file_name: str,
    key: str,
    verdict: str,
    recorded_at: datetime,
    model_verdict: str | None = None,
) -> None:
    """Append one entry to a YAML file inside an enforce_x directory.

    Convenience helper — each test writes its own corpus shape.
    """
    yaml_path = enforce_dir / file_name
    iso = recorded_at.isoformat()
    extra = f"  judge_model_verdict: {model_verdict}\n" if model_verdict else ""
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: {key}
          owner: alice
          reason: test
          safety: contained
          judge_verdict: {verdict}
          judge_recorded_at: '{iso}'
          judge_model: anthropic/claude-opus-4
          judge_rationale: test rationale
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
        judged_in_window=0,
        overrides_in_window=0,
    )
    assert report.rate == 0.0
