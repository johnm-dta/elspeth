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

**Tampering detection (C7-6).** An entry with
``judge_recorded_at > reference_time`` (future-dated relative to
the rolling-window anchor) is the exact tampering signal this
gate exists to catch: silently excluding such an entry would let
an operator forward-date an override out of the denominator (or
backdate it out of the rolling-recent window). On encountering a
future-dated entry the gate raises ``OverrideRateError`` with a
``TAMPERING_DETECTED`` prefix and the offending key; the CLI
surfaces this as exit-2 ("gate broken — operator must
investigate") rather than producing a misleading rate.

**Scope boundary.** The C3 rate denominator is defined only over
judge-capable ``allow_hits:`` entries. Private legacy shapes
(``allow_classes:`` for ``audit_evidence.nominal_base``, etc.) do
not carry judge metadata and are excluded from the rate; C1 separately
reports non-empty legacy entry shapes as ``UNRECOGNIZED_ENTRY_SHAPE``
so they cannot be used to bypass coverage enforcement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    JudgeVerdict,
)
from elspeth_lints.core.allowlist_io import (
    AllowlistIOError,
    iter_allow_hits_from_directory,
)

COUNTER_SNAPSHOT_SCHEMA_VERSION = 1
COUNTER_SNAPSHOT_DIRNAME = ".judge-metrics"
COUNTER_SNAPSHOT_FILENAME = "override-rate-counters.json"
JUDGE_DECISION_EVENTS_FILENAME = "judge-decision-events.jsonl"


@dataclass(frozen=True, slots=True)
class OverrideRateReport:
    """Result of one override-rate computation."""

    window_days: int
    reference_time: datetime
    min_samples: int
    max_rate: float
    max_overrides: int | None

    judged_in_window: int
    accepted_in_window: int
    overrides_in_window: int
    model_accepted_in_window: int
    model_blocked_in_window: int
    blocked_without_override_in_window: int

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
    def ratio_budget_exceeded(self) -> bool:
        """True when the override ratio is over budget and sample size is sufficient."""
        return not self.insufficient_data and self.rate > self.max_rate

    @property
    def absolute_budget_exceeded(self) -> bool:
        """True when an explicit absolute override cap is exceeded."""
        return self.max_overrides is not None and self.overrides_in_window > self.max_overrides

    @property
    def passes(self) -> bool:
        """Gate passes when neither the ratio nor absolute budget is exceeded.

        A pass on insufficient ratio data is not a silent pass: the
        operator-facing summary names ``insufficient_data``. The
        absolute override cap, when configured, is independent of the
        denominator and therefore still fires under small-N windows.
        """
        return not self.ratio_budget_exceeded and not self.absolute_budget_exceeded


@dataclass(frozen=True, slots=True)
class PerRuleOverrideRateReport:
    """Per-rule slice of the rolling override-rate computation."""

    rule_id: str
    judged_in_window: int
    accepted_in_window: int
    overrides_in_window: int
    model_accepted_in_window: int
    model_blocked_in_window: int

    @property
    def rate(self) -> float:
        """Per-rule rolling override rate; 0.0 when the rule has no entries."""
        if self.judged_in_window == 0:
            return 0.0
        return self.overrides_in_window / self.judged_in_window


@dataclass(frozen=True, slots=True)
class JudgeVerdictCounterRecord:
    """Structured counter input consumed by the override-rate gate.

    The YAML allowlist remains the audit source of truth. These records
    are an operational counter projection over judged ``allow_hits``
    entries so C3 can reuse a hash-bound snapshot instead of reparsing
    all YAML on every invocation.
    """

    source_file: str
    entry_key: str
    rule_id: str
    judge_verdict: JudgeVerdict
    judge_model_verdict: JudgeVerdict | None
    judge_recorded_at: datetime
    judge_metadata_signature: str


@dataclass(frozen=True, slots=True)
class OverrideRateCounterSnapshot:
    """Hash-bound snapshot of judge verdict counters for an allowlist root."""

    path: Path
    allowlist_root: Path
    allowlist_hash: str
    generated_at: datetime
    records: tuple[JudgeVerdictCounterRecord, ...]


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
    per_rule_reports: tuple[PerRuleOverrideRateReport, ...]
    counter_source: str


class OverrideRateError(RuntimeError):
    """The override-rate check cannot proceed."""


def default_counter_snapshot_path(allowlist_root: Path) -> Path:
    """Return the default C3 counter snapshot path for an allowlist root."""
    return allowlist_root / COUNTER_SNAPSHOT_DIRNAME / COUNTER_SNAPSHOT_FILENAME


def judge_decision_events_path(allowlist_dir: Path) -> Path:
    """Return the append-only judge-decision event log path for one enforce dir."""
    return allowlist_dir / COUNTER_SNAPSHOT_DIRNAME / JUDGE_DECISION_EVENTS_FILENAME


def append_judge_decision_event(
    allowlist_dir: Path,
    *,
    source_file: str,
    entry_key: str,
    rule_id: str,
    effective_verdict: JudgeVerdict,
    model_verdict: JudgeVerdict | None,
    recorded_at: datetime,
    write_disposition: str,
) -> Path | None:
    """Append one judge-decision metric event for C3's one-sidedness counter.

    Returns the written path, or ``None`` when ``allowlist_dir`` is outside
    the shipped ``enforce_*`` aggregation layout.
    """
    if not allowlist_dir.name.startswith("enforce_"):
        return None
    if recorded_at.tzinfo is None:
        raise OverrideRateError("judge decision event recorded_at must be timezone-aware")
    if write_disposition not in {"written", "blocked_without_override"}:
        raise OverrideRateError(f"unknown judge decision write_disposition {write_disposition!r}")
    event_path = judge_decision_events_path(allowlist_dir)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_file": source_file,
        "entry_key": entry_key,
        "rule_id": rule_id,
        "effective_verdict": effective_verdict.value,
        "model_verdict": model_verdict.value if model_verdict is not None else None,
        "recorded_at": recorded_at.isoformat(),
        "write_disposition": write_disposition,
    }
    with event_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, sort_keys=True) + "\n")
    return event_path


def write_override_rate_counter_snapshot(
    allowlist_root: Path,
    *,
    snapshot_path: Path | None = None,
) -> OverrideRateCounterSnapshot:
    """Write a hash-bound counter snapshot for the current allowlist root.

    This is an operational telemetry projection, not the audit source of
    truth. The snapshot carries the current allowlist content hash; the
    gate consumes it only while that hash still matches live YAML.
    """
    if snapshot_path is None:
        snapshot_path = default_counter_snapshot_path(allowlist_root)
    records = tuple(_collect_counter_records_from_allowlists(allowlist_root))
    snapshot = OverrideRateCounterSnapshot(
        path=snapshot_path,
        allowlist_root=allowlist_root,
        allowlist_hash=_compute_allowlist_hash(allowlist_root),
        generated_at=datetime.now(UTC),
        records=records,
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(_snapshot_to_json(snapshot), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def load_override_rate_counter_snapshot(snapshot_path: Path) -> OverrideRateCounterSnapshot:
    """Load a counter snapshot from disk with structural validation."""
    try:
        raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OverrideRateError(f"counter snapshot {snapshot_path} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OverrideRateError(f"counter snapshot {snapshot_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise OverrideRateError(f"counter snapshot {snapshot_path} must be a JSON object")
    schema_version = _required_snapshot_field(raw, "schema_version", int, snapshot_path)
    if schema_version != COUNTER_SNAPSHOT_SCHEMA_VERSION:
        raise OverrideRateError(
            f"counter snapshot {snapshot_path} schema_version={schema_version} "
            f"is incompatible with this build (expected {COUNTER_SNAPSHOT_SCHEMA_VERSION})"
        )
    generated_at = _parse_snapshot_datetime(
        _required_snapshot_field(raw, "generated_at", str, snapshot_path), snapshot_path, "generated_at"
    )
    records_raw = _required_snapshot_field(raw, "records", list, snapshot_path)
    return OverrideRateCounterSnapshot(
        path=snapshot_path,
        allowlist_root=Path(_required_snapshot_field(raw, "allowlist_root", str, snapshot_path)),
        allowlist_hash=_required_snapshot_field(raw, "allowlist_hash", str, snapshot_path),
        generated_at=generated_at,
        records=tuple(_counter_record_from_json(record, snapshot_path) for record in records_raw),
    )


def compute_override_rate(
    *,
    allowlist_root: Path,
    window_days: int,
    min_samples: int,
    max_rate: float,
    max_overrides: int | None = None,
    reference_time: datetime | None = None,
    counter_snapshot_path: Path | None = None,
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
    if max_overrides is not None and max_overrides < 0:
        raise OverrideRateError(f"--max-overrides must be non-negative, got {max_overrides}")

    if reference_time is None:
        reference_time = datetime.now(UTC)
    elif reference_time.tzinfo is None:
        raise OverrideRateError("--reference-time must be timezone-aware; pass a UTC ISO-8601 timestamp (e.g. '2026-05-23T00:00:00Z')")

    window_start = reference_time - timedelta(days=window_days)
    counter_source = "yaml"
    counter_records: tuple[JudgeVerdictCounterRecord, ...] | None = None
    if counter_snapshot_path is not None and counter_snapshot_path.exists():
        snapshot = load_override_rate_counter_snapshot(counter_snapshot_path)
        current_hash = _compute_allowlist_hash(allowlist_root)
        if snapshot.allowlist_hash == current_hash:
            counter_records = snapshot.records
            counter_source = "counter_snapshot"
    if counter_records is None:
        counter_records = tuple(_collect_counter_records_from_allowlists(allowlist_root))

    judged_in_window = 0
    accepted_in_window = 0
    overrides_in_window = 0
    model_accepted_in_window = 0
    model_blocked_in_window = 0
    blocked_without_override_in_window = 0
    override_entries: list[OverrideEntryRecord] = []
    per_rule_counts: dict[str, _MutableRuleCounts] = {}

    for record in counter_records:
        if record.judge_recorded_at > reference_time:
            # C7-6: future-dated entry. This is exactly the
            # timestamp-tampering signal the override-rate gate exists
            # to catch. The signal is preserved whether the gate read a
            # live YAML entry or a hash-fresh counter snapshot.
            raise OverrideRateError(
                f"TAMPERING_DETECTED: judge_recorded_at "
                f"{record.judge_recorded_at.isoformat()} is after "
                f"reference_time {reference_time.isoformat()} for entry "
                f"{record.source_file}::{record.entry_key}; future-dated judge "
                "timestamps indicate clock skew or deliberate tampering. "
                "Investigate before re-running the override-rate gate."
            )
        if record.judge_recorded_at < window_start:
            continue
        rule_counts = per_rule_counts.setdefault(record.rule_id, _MutableRuleCounts())
        judged_in_window += 1
        rule_counts.judged_in_window += 1
        if record.judge_verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR:
            overrides_in_window += 1
            rule_counts.overrides_in_window += 1
            if record.judge_model_verdict is JudgeVerdict.BLOCKED:
                model_blocked_in_window += 1
                rule_counts.model_blocked_in_window += 1
            else:
                model_accepted_in_window += 1
                rule_counts.model_accepted_in_window += 1
            override_entries.append(
                OverrideEntryRecord(
                    source_file=record.source_file,
                    entry_key=record.entry_key,
                    judge_verdict=record.judge_verdict,
                    judge_recorded_at=record.judge_recorded_at,
                )
            )
        elif record.judge_verdict is JudgeVerdict.ACCEPTED:
            accepted_in_window += 1
            model_accepted_in_window += 1
            rule_counts.accepted_in_window += 1
            rule_counts.model_accepted_in_window += 1

    for event in _load_judge_decision_events(allowlist_root):
        recorded_at = event["recorded_at"]
        if recorded_at > reference_time:
            raise OverrideRateError(
                f"TAMPERING_DETECTED: judge decision event recorded_at "
                f"{recorded_at.isoformat()} is after reference_time "
                f"{reference_time.isoformat()} for event "
                f"{event['source_file']}::{event['entry_key']}."
            )
        if recorded_at < window_start:
            continue
        if event["effective_verdict"] is JudgeVerdict.BLOCKED and event["write_disposition"] == "blocked_without_override":
            blocked_without_override_in_window += 1

    report = OverrideRateReport(
        window_days=window_days,
        reference_time=reference_time,
        min_samples=min_samples,
        max_rate=max_rate,
        max_overrides=max_overrides,
        judged_in_window=judged_in_window,
        accepted_in_window=accepted_in_window,
        overrides_in_window=overrides_in_window,
        model_accepted_in_window=model_accepted_in_window,
        model_blocked_in_window=model_blocked_in_window,
        blocked_without_override_in_window=blocked_without_override_in_window,
    )
    return OverrideRateDetail(
        report=report,
        override_entries=tuple(override_entries),
        per_rule_reports=tuple(
            PerRuleOverrideRateReport(
                rule_id=rule_id,
                judged_in_window=counts.judged_in_window,
                accepted_in_window=counts.accepted_in_window,
                overrides_in_window=counts.overrides_in_window,
                model_accepted_in_window=counts.model_accepted_in_window,
                model_blocked_in_window=counts.model_blocked_in_window,
            )
            for rule_id, counts in sorted(per_rule_counts.items())
        ),
        counter_source=counter_source,
    )


def _collect_counter_records_from_allowlists(allowlist_root: Path) -> list[JudgeVerdictCounterRecord]:
    records: list[JudgeVerdictCounterRecord] = []
    for entry_dir in sorted(allowlist_root.iterdir()):
        if not entry_dir.is_dir():
            continue
        if not entry_dir.name.startswith("enforce_"):
            continue
        for entry in _iterate_allow_hits(entry_dir):
            if entry.judge_recorded_at is None:
                continue
            _require_signed_judge_metadata(entry)
            if entry.judge_verdict is None:
                raise OverrideRateError(
                    f"INVALID_JUDGE_METADATA: entry {entry.source_file}::{entry.key} has judge_recorded_at but no judge_verdict"
                )
            assert entry.judge_metadata_signature is not None
            records.append(
                JudgeVerdictCounterRecord(
                    source_file=entry.source_file,
                    entry_key=entry.key,
                    rule_id=_rule_id_from_canonical_key(entry.key),
                    judge_verdict=entry.judge_verdict,
                    judge_model_verdict=entry.judge_model_verdict,
                    judge_recorded_at=entry.judge_recorded_at,
                    judge_metadata_signature=entry.judge_metadata_signature,
                )
            )
    return records


def _compute_allowlist_hash(allowlist_root: Path) -> str:
    hasher = hashlib.sha256()
    candidates = list(allowlist_root.rglob("*.yaml")) + list(allowlist_root.rglob("*.yml"))
    for path in sorted(candidates, key=lambda p: p.relative_to(allowlist_root).as_posix()):
        if COUNTER_SNAPSHOT_DIRNAME in path.parts:
            continue
        rel = path.relative_to(allowlist_root).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise OverrideRateError(f"allowlist hash could not read {path}: {exc}") from exc
        hasher.update(content)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _load_judge_decision_events(allowlist_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entry_dir in sorted(allowlist_root.iterdir()):
        if not entry_dir.is_dir() or not entry_dir.name.startswith("enforce_"):
            continue
        event_path = judge_decision_events_path(entry_dir)
        if not event_path.exists():
            continue
        for line_no, line in enumerate(event_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OverrideRateError(f"judge decision event {event_path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise OverrideRateError(f"judge decision event {event_path}:{line_no} must be a JSON object")
            schema_version = _required_snapshot_field(raw, "schema_version", int, event_path)
            if schema_version != 1:
                raise OverrideRateError(f"judge decision event {event_path}:{line_no} has unsupported schema_version={schema_version}")
            events.append(
                {
                    "source_file": _required_snapshot_field(raw, "source_file", str, event_path),
                    "entry_key": _required_snapshot_field(raw, "entry_key", str, event_path),
                    "rule_id": _required_snapshot_field(raw, "rule_id", str, event_path),
                    "effective_verdict": _required_event_verdict(raw, "effective_verdict", event_path),
                    "model_verdict": _judge_verdict_from_snapshot(raw.get("model_verdict"), event_path, "model_verdict"),
                    "recorded_at": _parse_snapshot_datetime(
                        _required_snapshot_field(raw, "recorded_at", str, event_path),
                        event_path,
                        "recorded_at",
                    ),
                    "write_disposition": _required_snapshot_field(raw, "write_disposition", str, event_path),
                }
            )
    return events


def _snapshot_to_json(snapshot: OverrideRateCounterSnapshot) -> dict[str, Any]:
    return {
        "schema_version": COUNTER_SNAPSHOT_SCHEMA_VERSION,
        "allowlist_root": str(snapshot.allowlist_root.resolve()),
        "allowlist_hash": snapshot.allowlist_hash,
        "generated_at": snapshot.generated_at.isoformat(),
        "records": [
            {
                "source_file": record.source_file,
                "entry_key": record.entry_key,
                "rule_id": record.rule_id,
                "judge_verdict": record.judge_verdict.value,
                "judge_model_verdict": record.judge_model_verdict.value if record.judge_model_verdict is not None else None,
                "judge_recorded_at": record.judge_recorded_at.isoformat(),
                "judge_metadata_signature": record.judge_metadata_signature,
            }
            for record in snapshot.records
        ],
    }


def _counter_record_from_json(raw: Any, snapshot_path: Path) -> JudgeVerdictCounterRecord:
    if not isinstance(raw, dict):
        raise OverrideRateError(f"counter snapshot {snapshot_path} record must be a JSON object")
    verdict = _judge_verdict_from_snapshot(raw.get("judge_verdict"), snapshot_path, "judge_verdict")
    if verdict is None:
        raise OverrideRateError(f"counter snapshot {snapshot_path} record judge_verdict must not be null")
    return JudgeVerdictCounterRecord(
        source_file=_required_snapshot_field(raw, "source_file", str, snapshot_path),
        entry_key=_required_snapshot_field(raw, "entry_key", str, snapshot_path),
        rule_id=_required_snapshot_field(raw, "rule_id", str, snapshot_path),
        judge_verdict=verdict,
        judge_model_verdict=_judge_verdict_from_snapshot(raw.get("judge_model_verdict"), snapshot_path, "judge_model_verdict"),
        judge_recorded_at=_parse_snapshot_datetime(
            _required_snapshot_field(raw, "judge_recorded_at", str, snapshot_path),
            snapshot_path,
            "judge_recorded_at",
        ),
        judge_metadata_signature=_required_snapshot_field(raw, "judge_metadata_signature", str, snapshot_path),
    )


def _required_snapshot_field(raw: dict[str, Any], field: str, expected_type: type, snapshot_path: Path) -> Any:
    if field not in raw:
        raise OverrideRateError(f"counter snapshot {snapshot_path} missing required field {field!r}")
    value = raw[field]
    if expected_type is int and isinstance(value, bool):
        raise OverrideRateError(f"counter snapshot {snapshot_path} field {field!r} must be int, got bool")
    if not isinstance(value, expected_type):
        raise OverrideRateError(
            f"counter snapshot {snapshot_path} field {field!r} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _parse_snapshot_datetime(value: str, snapshot_path: Path, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise OverrideRateError(f"counter snapshot {snapshot_path} field {field!r} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise OverrideRateError(f"counter snapshot {snapshot_path} field {field!r} must be timezone-aware")
    return parsed


def _judge_verdict_from_snapshot(raw: Any, snapshot_path: Path, field: str) -> JudgeVerdict | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise OverrideRateError(f"counter snapshot {snapshot_path} field {field!r} must be str or null")
    try:
        return JudgeVerdict(raw)
    except ValueError as exc:
        raise OverrideRateError(f"counter snapshot {snapshot_path} field {field!r} has unknown verdict {raw!r}") from exc


def _required_event_verdict(raw: dict[str, Any], field: str, path: Path) -> JudgeVerdict:
    verdict = _judge_verdict_from_snapshot(raw.get(field), path, field)
    if verdict is None:
        raise OverrideRateError(f"judge decision event {path} field {field!r} must not be null")
    return verdict


def _require_signed_judge_metadata(entry: AllowlistEntry) -> None:
    """Fail closed when C3 would count unsigned judged metadata as authentic."""
    if entry.judge_metadata_signature is not None:
        return
    raise OverrideRateError(
        f"UNSIGNED_JUDGE_METADATA: entry {entry.source_file}::{entry.key} has "
        "judge_recorded_at but no judge_metadata_signature. The override-rate "
        "gate refuses to count unsigned judge metadata because field presence "
        "alone is forgeable YAML."
    )


def _iterate_allow_hits(directory: Path) -> list[AllowlistEntry]:
    """Yield every ``allow_hits`` entry in ``directory`` (HEAD content).

    Skips files whose root YAML mapping carries no ``allow_hits:``
    key — the cheapest filter for the common case of legacy-format
    directories. Bad YAML propagates as ``OverrideRateError`` so the
    operator gets an actionable signal at the boundary.
    """
    try:
        return iter_allow_hits_from_directory(directory)
    except AllowlistIOError as exc:
        raise OverrideRateError(str(exc)) from exc


@dataclass(slots=True)
class _MutableRuleCounts:
    judged_in_window: int = 0
    accepted_in_window: int = 0
    overrides_in_window: int = 0
    model_accepted_in_window: int = 0
    model_blocked_in_window: int = 0


def _rule_id_from_canonical_key(key: str) -> str:
    """Extract the rule ID from ``<path>:<rule>:<symbol>:fp=<hash>``."""
    parts = key.split(":", 2)
    if len(parts) < 3 or not parts[1]:
        raise OverrideRateError(f"cannot extract rule id from canonical allowlist key {key!r}")
    return parts[1]
