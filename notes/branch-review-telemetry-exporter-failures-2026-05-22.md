# Branch Review: `fix/test-audit-telemetry-exporter-failures`

- **Commit**: `83eb04212 Account handled telemetry exporter failures` (2026-05-20, John Morrissey)
- **Position**: 1 commit ahead of RC5.2, 41 behind. Sole commit on the branch.
- **Files (production)**: `contracts/errors.py` (L0), `telemetry/exporters/azure_monitor.py`, `telemetry/exporters/otlp.py`, `telemetry/manager.py`, `telemetry/protocols.py` (all L3 telemetry).
- **Files (test)**: `tests/unit/telemetry/exporters/test_azure_monitor.py`, `…/test_otlp.py`, `tests/unit/telemetry/test_manager.py`.
- **Collision with multi-source-token-scheduler worktree**: none (worktree touches `mcp/`, `plugins/`, `web/` only).

## 1. What the change actually does (high confidence)

The commit changes the **contract between exporters and `TelemetryManager`** so that exporters can *report* a "handled transport failure" without raising. Previously the protocol was:

- `export()` MUST NOT raise → on failure, log and silently return `None`. Manager has no way to know whether the call actually succeeded.
- Manager counted any non-raising call as a success (`breaker.record_success(); successes += 1`).

After the commit:

- `ExporterProtocol.export()` and `ExporterProtocol.flush()` now return `bool | None`.
- A `False` return value means "I caught a transport error, didn't raise, you should account for it."
- `None` (or any other non-False value) is treated as success.
- `TelemetryManager._dispatch_to_exporters` checks `if result is False:` and runs the same accounting branch as a raised `TELEMETRY_TRANSPORT_ERRORS` exception: `breaker.record_failure()`, `failures += 1`, `_exporter_failures[name] += 1`, warning log, then `continue` (skipping the success accounting).

Concretely, the OTLP/Azure exporters now have three distinct return-value sites updated:

1. Exporter not configured / not initialized → previously silent return → now `return False`.
2. `_flush_batch()` inner try/except catching `TELEMETRY_TRANSPORT_ERRORS` → previously logged and returned `None` → now logs and returns `False`. The `finally: self._buffer.clear()` is preserved, then a `return None` is appended after the try/except/finally so the success path explicitly returns None.
3. Outer `flush()` `try/except` mirroring (catches a transport error escaping `_flush_batch` itself) → now `return False`.

`TelemetryExporterError`'s docstring is updated to clarify it covers configure-time errors only (export-time is now in-band via `False`).

In plain English: previously, exporter transport failures that the exporter itself caught were invisible to the manager — they could not flip a circuit breaker, increment `exporter_failures`, or contribute to `consecutive_total_failures`. After this commit, **the Azure Monitor and OTLP exporters' caught transport failures now flow into the same accounting machinery as raised transport failures.**

## 2. CLAUDE.md compliance

### 2.1 Telemetry primacy — COMPLIANT (high confidence)

The branch operates entirely within the **telemetry** layer, which under CLAUDE.md is allowed to use `logger` for telemetry-system failures. The new `logger.warning("Telemetry exporter reported handled failure", …)` (manager.py) and the retained `logger.warning("Failed to ... telemetry event", …)` calls in the exporters are documented permitted uses — "logger is for … telemetry system failures." There is no attempt to record handled-failure events into the Landscape audit trail, which is correct: exporter transport failures are *operational* signals, not pipeline activity. (CLAUDE.md telemetry section: "Logger is NOT for pipeline activity. Don't log row-level decisions, transform outcomes, or call results.")

The change does NOT shift any audit-bearing signal from Landscape to logger. It only changes the channel by which one telemetry-subsystem error is *counted* (now in `health_metrics["exporter_failures"]`), which is operational telemetry and not Landscape's domain.

### 2.2 Tier model — COMPLIANT (high confidence)

`TelemetryExporterError` lives in `contracts/errors.py` (L0). Its docstring tagged it as `TIER-2: telemetry-subsystem configuration/initialization failure`. The amended docstring still scopes it correctly to configuration/initialization errors (a Tier-2 telemetry-subsystem class), explicitly stating that handled transport failures are NOT raised as this exception. No new exception types are introduced; the change is a docstring clarification only in `errors.py`.

No layer-import direction changes. `manager.py` (L3) imports `contracts/errors.py` (L0) — downward as required. `errors.py` does not gain any telemetry imports.

### 2.3 Offensive programming — COMPLIANT (medium-high confidence)

The change does not introduce any new defensive `getattr` / `hasattr` / `.get()` / silent except patterns. The pre-existing `try / except Exception` blocks gating on `if not isinstance(e, TELEMETRY_TRANSPORT_ERRORS): raise` are retained — programming errors still crash, only documented transport-error classes are caught. This is the project-sanctioned pattern for the telemetry trust boundary (telemetry-emission is documented as async best-effort).

`TelemetryManager._dispatch_to_exporters` uses `if result is False:` (identity check, not truthiness) — that's the right discipline: `None` (success) and `0`/empty-list don't accidentally trip the failure branch.

### 2.4 Layer compliance — COMPLIANT (high confidence)

- `contracts/errors.py` (L0): docstring-only change, no new imports.
- `telemetry/protocols.py` (L3): no new imports; signature widening only.
- `telemetry/manager.py` (L3): no new imports.
- `telemetry/exporters/{azure_monitor,otlp}.py` (L3): no new imports.

No upward imports introduced. The branch is layer-clean.

### 2.5 No-legacy-code — COMPLIANT (high confidence)

No compatibility shims, no `@deprecated`, no version flags. The protocol is widened in-place (`-> None` → `-> bool | None`). All affected production call sites are updated in the same commit.

## 3. Anti-pattern check: silent failure — **PARTIAL COMPLIANCE, with a real gap**

The branch title is "Account handled telemetry exporter failures" — i.e. the explicit *goal* is to stop counting transport failures as silent successes. The commit largely achieves this, but **the migration is incomplete** in three concrete ways:

### 3.1 `console.py` and `datadog.py` exporters not updated (high confidence, real regression vector)

`ExporterProtocol.export` is now typed `-> bool | None`, but `telemetry/exporters/console.py:119` and `telemetry/exporters/datadog.py:211` still declare `def export(self, event) -> None`. More importantly, **their failure-handling bodies still silently swallow transport errors and return `None`** (e.g. `datadog.py` lines 230–241 catch `TELEMETRY_TRANSPORT_ERRORS`, log a warning, and fall through). Under the new dispatch loop, this returns `None`, which the manager treats as **success** (`breaker.record_success(); successes += 1`).

The behavioural consequence: in a pipeline where datadog is the only exporter, a sustained datadog transport outage will be recorded in `health_metrics["events_emitted"]` as fully successful, with zero `exporter_failures`, no circuit-breaker trip, and no contribution to `consecutive_total_failures`. The "Account handled failures" goal is achieved for Azure Monitor and OTLP but explicitly silently violated for Datadog and Console.

Evidence (post-commit `datadog.py` lines 230–241):

```python
try:
    self._create_span_for_event(event)
except Exception as e:
    if not isinstance(e, TELEMETRY_TRANSPORT_ERRORS):
        raise  # Programming error — must crash
    # Export MUST NOT raise - log and continue
    logger.warning(
        "Failed to export telemetry event to Datadog",
        exporter=self._name,
        event_type=type(event).__name__,
        error=str(e),
    )
# implicit `return None` — counted as success
```

This is exactly the "swallow and call it success" anti-pattern that the commit title pledges to eliminate.

### 3.2 `manager.flush()` ignores exporter flush() return value (high confidence, real bug)

`protocols.py` widens `flush() -> bool | None` and OTLP/Azure `flush()` now returns `False` on handled transport failure. But `TelemetryManager.flush()` (manager.py ~line 519) discards the return value:

```python
for exporter in self._exporters:
    try:
        exporter.flush()        # ← return value dropped
    except Exception as e:
        ...
```

A pipeline-shutdown flush that hits a transport error will:
- if the exporter raises → counted via the exception path → records failure.
- if the exporter catches and returns `False` (the new path) → silently logged inside the exporter and **not** recorded in `exporter_failures` or circuit-breaker state.

This means the OTLP exporter, post-commit, has *two* paths for a flush-time transport error and they disagree about whether the manager learns of it. Inside `_flush_batch`, the `try/except` catches transport errors, logs them, and now returns `False` — but if that `False` propagates out through `flush()` and back to `manager.flush()`, it goes to /dev/null.

In particular, the test `test_flush_failure_reports_handled_failure` in `test_otlp.py` only checks that `exporter.flush()` returns False — it does not check that the manager would do anything with that signal.

### 3.3 Buffer-clear-on-failure is preserved, with the same "drop the events" outcome (medium confidence, not new)

When `_flush_batch` catches a transport error, the `finally: self._buffer.clear()` runs and the buffered events are *dropped on the floor*. The new return-value plumbing means the manager increments `events_dropped` by **one** for that event triggering the flush, but the rest of the buffered events (potentially `batch_size - 1` of them) are silently discarded without any drop counter increment. This pre-dates the commit but is more visible now that the manager's accounting branch fires. The accounting is "one bad batch == one dropped event" rather than "N dropped events." Worth a follow-up but not introduced here.

### Net judgement on anti-pattern

The commit is a genuine, principled step toward eliminating "silent count-as-success." It does so cleanly for OTLP and Azure-Monitor export(). It does *not* finish the job: console + datadog still silently swallow, and the flush path through the manager does not consume the new return value. **The change is correct as far as it goes, but it advertises a property it only partly delivers.**

## 4. Test adequacy

### 4.1 New positive test (test_manager.py) — adequate for what it asserts

`test_exporter_handled_failure_return_tracked_per_name`: builds an exporter whose `export()` returns `False`, sends one event, drains the queue, asserts:

- `events_emitted == 0`
- `events_dropped == 1`
- `exporter_failures["reported-failure"] == 1`

This validates the happy path of the new accounting machinery. **(High confidence: the assertions match the dispatch-loop semantics for the 1-exporter case.)**

### 4.2 Gap: no test for mixed success + handled-failure

The existing tests cover (a) all exporters succeed, (b) some raise, (c) all raise. There is **no** new test for "one exporter succeeds and one returns False" — the case where `successes > 0` but `_exporter_failures` should still increment for the failing name. Manual reading of the dispatch loop says this works (the `continue` skips success accounting for the failing exporter, the next exporter records success, top-level branch picks `successes > 0`). But it is not explicitly tested.

### 4.3 Gap: no test that `False` triggers circuit-breaker trip

After N handled-failure returns from the same exporter, the per-exporter circuit breaker should open and subsequent calls should be skipped. The exception-path version of this is tested (`test_property_based.py` uses `max_consecutive_failures=5`). The new return-value path is *not* tested for circuit-breaker integration. The dispatch code uses `breaker.record_failure()` in both arms, so it should work, but a regression here would be undetectable.

### 4.4 Gap: `manager.flush()` swallowing exporter-flush `False`

This is the real bug from §3.2. No test exists for "manager.flush() observes False from exporter.flush() and accounts for it" — because the manager does not, in fact, do so. A test for this would expose the bug.

### 4.5 Test-side protocol violation (cosmetic)

The new `ReportingFailureExporter` in `test_manager.py` declares `def flush(self) -> None: pass` while the protocol now says `-> bool | None`. `None` is a valid value for `bool | None`, so type-check-wise it's fine, but it's notionally inconsistent with the new contract and will get noisier as the migration completes.

## 5. Landing recommendation: **LAND_WITH_EDITS**

The change is architecturally correct, layer-clean, telemetry-primacy-correct, offensive-programming-compliant, and addresses a genuine silent-failure gap. But it advertises behaviour it only half-delivers. The fixable gaps are small and additive; none require structural rework.

### Required edits before landing

1. **Update `console.py` and `datadog.py` `export()`** to either (a) return `False` from the transport-error branch, or (b) explicitly document why their failure modes are inherently fatal-and-raise and not "handled failures." Today they silently log-and-succeed, which the new protocol now treats as broken behaviour. (Pick one; pick the same answer for `flush()` too.) High confidence this is necessary.

2. **Consume the `flush()` return value in `TelemetryManager.flush()`** (manager.py ~line 519). Mirror the dispatch-loop accounting: `if result is False:` → `breaker.record_failure()`, `_exporter_failures[exporter.name] += 1`, warning log. Otherwise the new `flush() -> bool | None` plumbing is unused on the only side that calls it. High confidence this is necessary.

3. **Add one test** for the mixed case (`exporter A succeeds, exporter B returns False` → `events_emitted == 1`, `exporter_failures["B"] == 1`) and one for circuit-breaker trip via repeated `False` returns. Medium confidence necessary; high confidence valuable.

### Edits I'd defer to a follow-up issue (not blocking)

- Buffer-clear-on-failure drops `batch_size - 1` events without incrementing `events_dropped`. Existing behaviour; not introduced by this commit. File as separate work.
- Cosmetic: align the test `ReportingFailureExporter.flush()` signature to `-> bool | None`.

## 6. Risks if landed as-is on RC5.2

1. **False sense of completeness (medium severity).** The commit title and protocol-docstring now claim that handled transport failures are accounted for. An operator inspecting `health_metrics["exporter_failures"]` for a datadog-only pipeline will see 0 failures during an outage, will trust the metric, and will get a wrong answer. Mitigation: fix per §5.1, or rename the commit to scope it to OTLP/Azure.

2. **Inconsistent flush behaviour (medium severity).** Same handled-failure event observed via `export()` is recorded; observed via `flush()` is not. Operationally surprising. Mitigation: fix per §5.2.

3. **Low collision risk with in-flight work (low severity).** The multi-source-token-scheduler worktree does not touch any of these files, so merge conflict on landing is near zero. The branch is 41 commits behind RC5.2 — rebase will be straightforward (these files have not been modified upstream in the gap; quick check: `git log RC5.2 -- src/elspeth/telemetry/manager.py` would confirm before rebase).

## 7. Confidence-and-evidence summary (SME protocol)

| Claim | Confidence | Evidence |
|---|---|---|
| Branch widens protocol to `bool \| None` and wires `False`-return into manager dispatch accounting | High | `git show 83eb04212` diff body (protocols.py, manager.py, exporters) |
| Telemetry primacy compliance (logger is OK here) | High | CLAUDE.md telemetry section explicitly permits logger for telemetry-system failures |
| Tier model / L0 compliance for `errors.py` change | High | Docstring-only change; no new imports |
| Layer compliance (no upward imports) | High | Verified by inspecting imports in each touched file |
| `console.py` / `datadog.py` still silently return success on transport failure | High | Source inspection: `datadog.py` lines 230–241; `console.py` line 119 returns `-> None`, no return-value migration |
| `manager.flush()` ignores `exporter.flush()` return value | High | `manager.py` line 519–529 verbatim: `exporter.flush()` with return-value discarded |
| `_flush_batch` buffer-clear drops `batch_size - 1` events on transport error without counting them | Medium | Pre-existing logic in `_flush_batch`; not changed by this commit |
| New test correctly exercises the single-exporter happy path | High | Direct reading of `test_exporter_handled_failure_return_tracked_per_name` against dispatch loop semantics |
| No test covers mixed-outcome dispatch or breaker-trip via `False` returns | High | `grep` of new test file; only one new test added |
| Recommend LAND_WITH_EDITS | High | Three concrete bounded gaps, none structural |
