"""Override-rate CI gate (convergent finding C3).

Convergent panel finding C3: ``OVERRIDDEN_BY_OPERATOR`` is documented
as "a metric to watch", but watch-it-yourself is not a feedback loop.
Without a mechanical anchor, the long-run equilibrium settles at
whatever override rate worst-week deadline pressure tolerates. Each
override sets precedent; no override reverses one. The override-rate
gate converts the metric from informational to enforcing.

The gate computes, across every ``allow_hits`` entry in every
``enforce_*`` directory under ``allowlist-root``, the rolling-window
override rate: how many entries judged within the last N days carry
``judge_verdict == OVERRIDDEN_BY_OPERATOR`` versus the population of
entries judged in that same window. Pre-judge entries (no
``judge_recorded_at``) are excluded from both numerator and
denominator — they pre-date the gate and contribute neither override
signal nor accept-baseline signal.

**Small-sample handling (load-bearing).** A naive rate gate would
trip mechanically on a 3-entry window with one override (33% > 10%).
The first weeks of operation produce exactly that shape: tens of
entries per week, single-digit denominators. ``--min-samples``
(default 10) short-circuits the gate to PASS when the denominator
is below the threshold, with a clear "insufficient data" note in
the report. This is the documented behaviour, not an oversight.

**Rolling time anchor.** The window is anchored on the current
UTC instant (``datetime.now(UTC)``) by default; ``--reference-time``
overrides for reproducibility in CI replays or local debugging.
Entries are included when ``judge_recorded_at >= reference_time -
window``. Window default is 30 days, configurable via
``--window-days``.

**Scope boundary.** Same as the C1 judge-coverage gate: reads
``allow_hits:`` YAML shape only. Private legacy shapes
(``allow_classes:`` for ``audit_evidence.nominal_base``, etc.) do
not carry judge metadata and are silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    JudgeVerdict,
    _parse_allow_hits,
)


@dataclass(frozen=True, slots=True)
class OverrideRateReport:
    """Result of one override-rate computation."""

    window_days: int
    reference_time: datetime
    min_samples: int
    max_rate: float

    judged_in_window: int
    overrides_in_window: int

    @property
    def rate(self) -> float:
        """Rolling override rate; 0.0 when the window is empty."""
        if self.judged_in_window == 0:
            return 0.0
        return self.overrides_in_window / self.judged_in_window

    @property
    def insufficient_data(self) -> bool:
        """True when the denominator is below ``min_samples``.

        Insufficient-data windows pass the gate with a notice; this
        keeps small-N noise from tripping the rate threshold in the
        first weeks of operation.
        """
        return self.judged_in_window < self.min_samples

    @property
    def passes(self) -> bool:
        """Gate passes on insufficient data OR rate within budget.

        A pass on insufficient data is not a silent pass: the
        operator-facing summary names ``insufficient_data`` so the
        reason is auditable.
        """
        if self.insufficient_data:
            return True
        return self.rate <= self.max_rate


@dataclass(frozen=True, slots=True)
class OverrideEntryRecord:
    """One judged entry contributing to the rolling-window computation.

    Carried in ``OverrideRateReport.contributing_entries`` for
    operator-facing reports — the gate's "which overrides tripped
    me?" question is easier to answer with the list at hand than
    with a re-scan.
    """

    source_file: str
    entry_key: str
    judge_verdict: JudgeVerdict
    judge_recorded_at: datetime


@dataclass(frozen=True, slots=True)
class OverrideRateDetail:
    """Aggregate + per-entry detail bundle returned to the CLI handler."""

    report: OverrideRateReport
    override_entries: tuple[OverrideEntryRecord, ...]


class OverrideRateError(RuntimeError):
    """The override-rate check cannot proceed."""


def compute_override_rate(
    *,
    allowlist_root: Path,
    window_days: int,
    min_samples: int,
    max_rate: float,
    reference_time: datetime | None = None,
) -> OverrideRateDetail:
    """Compute the rolling-window override rate across every ``allow_hits`` source.

    ``allowlist_root`` is typically ``config/cicd``. The function
    enumerates ``enforce_*`` subdirectories, parses ``allow_hits``
    blocks (skipping legacy shapes silently), and aggregates the
    judge metadata for the rolling window.

    Raises ``OverrideRateError`` for operator-actionable
    misconfiguration (root not a directory, negative window). Bad
    YAML at HEAD propagates as a structural anomaly per the same
    discipline as ``judge_coverage`` — HEAD is our data, corruption
    must crash, not silently skip.
    """
    if not allowlist_root.is_dir():
        raise OverrideRateError(f"--allowlist-root {allowlist_root} is not a directory")
    if window_days <= 0:
        raise OverrideRateError(f"--window-days must be positive, got {window_days}")
    if min_samples < 0:
        raise OverrideRateError(f"--min-samples must be non-negative, got {min_samples}")
    if not (0.0 <= max_rate <= 1.0):
        raise OverrideRateError(f"--max-rate must be in [0.0, 1.0], got {max_rate}")

    if reference_time is None:
        reference_time = datetime.now(UTC)
    elif reference_time.tzinfo is None:
        raise OverrideRateError("--reference-time must be timezone-aware; pass a UTC ISO-8601 timestamp (e.g. '2026-05-23T00:00:00Z')")

    window_start = reference_time - timedelta(days=window_days)

    judged_in_window = 0
    overrides_in_window = 0
    override_entries: list[OverrideEntryRecord] = []

    for entry_dir in sorted(allowlist_root.iterdir()):
        if not entry_dir.is_dir():
            continue
        if not entry_dir.name.startswith("enforce_"):
            continue
        for entry in _iterate_allow_hits(entry_dir):
            if entry.judge_recorded_at is None:
                continue
            if entry.judge_recorded_at < window_start:
                continue
            if entry.judge_recorded_at > reference_time:
                # Future-dated entry: data tampering or clock skew.
                # Excluded from the window (our window is "the last
                # N days up to now", not "the last N days plus
                # whatever lives in the future"). Surface via the
                # contributing-entries list for the operator-facing
                # log? — out of scope; the entry would also flunk
                # downstream sanity checks.
                continue
            judged_in_window += 1
            if entry.judge_verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR:
                overrides_in_window += 1
                override_entries.append(
                    OverrideEntryRecord(
                        source_file=entry.source_file,
                        entry_key=entry.key,
                        judge_verdict=entry.judge_verdict,
                        judge_recorded_at=entry.judge_recorded_at,
                    )
                )

    report = OverrideRateReport(
        window_days=window_days,
        reference_time=reference_time,
        min_samples=min_samples,
        max_rate=max_rate,
        judged_in_window=judged_in_window,
        overrides_in_window=overrides_in_window,
    )
    return OverrideRateDetail(
        report=report,
        override_entries=tuple(override_entries),
    )


def _iterate_allow_hits(directory: Path) -> list[AllowlistEntry]:
    """Yield every ``allow_hits`` entry in ``directory`` (HEAD content).

    Skips files whose root YAML mapping carries no ``allow_hits:``
    key — the cheapest filter for the common case of legacy-format
    directories. Bad YAML propagates as ``OverrideRateError`` so the
    operator gets an actionable signal at the boundary.
    """
    entries: list[AllowlistEntry] = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        if yaml_file.name == "_defaults.yaml":
            continue
        try:
            text = yaml_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise OverrideRateError(f"could not read {yaml_file}: {exc}") from exc
        try:
            data = _load_yaml_strict(text)
        except ValueError as exc:
            raise OverrideRateError(f"{yaml_file}: failed to parse as YAML mapping: {exc}") from exc
        if "allow_hits" not in data:
            continue
        entries.extend(_parse_allow_hits(data, source_file=yaml_file.name))
    return entries


def _load_yaml_strict(text: str) -> dict[str, Any]:
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")
    return raw
