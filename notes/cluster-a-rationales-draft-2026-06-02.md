# Cluster A Rationale Drafts — 2026-06-02

Scope: 11 allowlist entries (all R6) that carry the boilerplate reason
"Silent recovery with explicit error handling path". Each section provides
a paste-ready `justify --rationale` string derived from reading the live
source. All source references are to `src/elspeth/`.

Linter evidence: raw findings from a keyless run
(`--allowlist-dir /tmp/empty_allowlist_dir`) confirm live fingerprint→line
mappings. The try_acquire entry is confirmed STALE (zero live findings in
`rate_limit/limiter.py`).

---

## Entry 1

**Key:** `mcp/server.py:R6:_find_audit_databases:fp=badb2efd50be18f6`

**Code (mcp/server.py:717–720):**
Inside a `Path.rglob("*.db")` loop, the code calls `db_file.stat().st_mtime`
inside a `try` block. `except OSError: mtime = 0` falls through to the
`found.append(...)` call on line 722 — the file is still included in the
results list, just sorted last by recency. No file is silently dropped.

```python
try:
    mtime = db_file.stat().st_mtime
except OSError:
    mtime = 0
```

**Proposed rationale:**
`_find_audit_databases` (mcp/server.py:717–720) stats each discovered `.db`
file to sort candidates by recency. `OSError` on `stat()` is an expected
filesystem boundary condition (file deleted between `rglob` iteration and
stat, or a permission issue on a file we don't own). The catch is not a
silent drop: the file is still appended to the results list with `mtime=0`
so it appears last in the recency-sorted candidate list. The audit trail is
not involved — this helper only populates the interactive database-picker
menu in the MCP CLI; any sort-order error affects only UI convenience, not
pipeline data or Landscape integrity.

**Confidence / notes:** ACCEPT-ready. Linter confirms fp=badb2efd50be18f6
→ mcp/server.py:719. Disposition (mtime=0, file still listed) is correct
from the code; the advisor flagged that the prior safety note incorrectly
said "skipped" — this version corrects that.

---

## Entry 2

**Key:** `mcp/server.py:R6:_prompt_for_database:fp=250dc301eb3eda7c`

**Code (mcp/server.py:747–751):**
`_prompt_for_database` displays each database candidate using a relative
path when possible. `except ValueError:` catches `Path.relative_to()`
raising `ValueError` when the db path is on a different filesystem root
than `search_path`. The fallback is `display = db_path` — the absolute
path string.

```python
try:
    rel_path = Path(db_path).relative_to(search_path)
    display = f"./{rel_path}"
except ValueError:
    display = db_path
```

**Proposed rationale:**
`_prompt_for_database` (mcp/server.py:747–751) attempts to display each
candidate database as a relative path for readability. `Path.relative_to()`
raises `ValueError` when the candidate is on a different mount point or
drive root than the search directory. The `except ValueError` catch falls
back to the absolute path string — a display-only difference with no effect
on which path is returned to the caller or on any audit record. This is
path-display formatting in an interactive CLI menu; the exception is an
expected API signal from `pathlib`, not a masked error.

**Confidence / notes:** ACCEPT-ready. Linter confirms fp=250dc301eb3eda7c
→ mcp/server.py:750 `except ValueError:`.

---

## Entry 3

**Key:** `mcp/server.py:R6:_prompt_for_database:fp=dbbeb81e161a9a1d`

**Code (mcp/server.py:760–764):**
The function reads user input via `input()` in a `while True` loop.
`except (EOFError, KeyboardInterrupt)` catches Ctrl-D (EOF on stdin) and
Ctrl-C respectively. The handler writes `"\nCancelled.\n"` to stderr and
returns `None` — the documented cancellation contract of `_prompt_for_database`.

```python
try:
    choice = input().strip()
except (EOFError, KeyboardInterrupt):
    sys.stderr.write("\nCancelled.\n")
    return None
```

**Proposed rationale:**
`_prompt_for_database` (mcp/server.py:760–764) reads interactive user
input from stdin. `EOFError` (Ctrl-D / stdin closed) and `KeyboardInterrupt`
(Ctrl-C) are the canonical Python signals for "user wishes to abort an
interactive prompt". Both are caught narrowly; the handler writes a
cancellation message to stderr and returns `None` — the documented return
value meaning "user cancelled, do not open any database". This is not a
silent swallow: the absence of a selected database propagates to the caller
(`main()`), which exits cleanly. No audit trail is involved; the function
only selects a database for the MCP CLI session.

**Confidence / notes:** ACCEPT-ready. Linter confirms fp=dbbeb81e161a9a1d
→ mcp/server.py:762 `except (EOFError, KeyboardInterrupt):`.

---

## Entry 4

**Key:** `mcp/server.py:R6:_prompt_for_database:fp=b20406c8a83aad1b`

**Code (mcp/server.py:769–775):**
After reading user input, `int(choice) - 1` is evaluated. `except ValueError`
catches non-numeric input (e.g. the user typed "abc"). The handler writes
`"Please enter a number\n"` to stderr and loops — the while-loop retries the
prompt.

```python
try:
    idx = int(choice) - 1
    if 0 <= idx < len(databases):
        return databases[idx]
    sys.stderr.write(f"Please enter a number between 1 and {len(databases)}\n")
except ValueError:
    sys.stderr.write("Please enter a number\n")
```

**Proposed rationale:**
`_prompt_for_database` (mcp/server.py:769–775) converts user input to an
integer index. `int(choice)` raises `ValueError` on non-numeric input (e.g.
"abc", empty after stripping). The catch is a loop-continue: the user is
prompted again via the enclosing `while True`. This is interactive input
validation — `ValueError` from `int()` is the expected Python signal for
"unparseable string", not a masked error condition. The loop has no exit
other than a valid selection or user cancellation (entries 2/3 above). No
audit data is involved.

**Confidence / notes:** ACCEPT-ready. Linter confirms fp=b20406c8a83aad1b
→ mcp/server.py:774 `except ValueError:`.

---

## Entry 5

**Key:** `telemetry/manager.py:R6:TelemetryManager.handle_event:fp=0a1bb9f3fb968fdf`

**Code (telemetry/manager.py:360–363):**
In `BackpressureMode.DROP` mode, `put_nowait(event)` raises `queue.Full`
when the bounded queue is at capacity. The handler calls
`self._drop_oldest_and_enqueue_newest(event)` — the documented DROP-mode
overflow strategy that evicts the oldest queued event, accounts for the
drop via `_events_dropped`, and enqueues the newest.

```python
try:
    self._queue.put_nowait(event)
except queue.Full:
    self._drop_oldest_and_enqueue_newest(event)
```

**Proposed rationale:**
`TelemetryManager.handle_event` (telemetry/manager.py:360–363) operates
under `BackpressureMode.DROP`. `queue.Full` from `put_nowait()` is the
expected backpressure signal on a bounded queue at capacity — not an error,
but the normal control-flow trigger for the DROP overflow policy. The catch
dispatches to `_drop_oldest_and_enqueue_newest()` (line 374), which evicts
the oldest item, increments `_events_dropped` for accounting, and enqueues
the incoming event. Per ELSPETH's audit>telemetry>logger primacy order,
telemetry is explicitly best-effort; `queue.Full` is the designed control
signal for bounded-queue overflow handling at this layer. Dropped events are
counted (not silently lost) via `_events_dropped` protected by `_dropped_lock`.

**Confidence / notes:** ACCEPT-ready. This entry already has a detailed
judge rationale (judge_confidence=0.85). The proposed rationale adds
explicit line citations and the primacy-order justification that make it
stand alone for an auditor. Linter confirms fp=0a1bb9f3fb968fdf →
telemetry/manager.py:362.

---

## Entry 6

**Key:** `telemetry/manager.py:R6:TelemetryManager.handle_event:fp=64535719a1fd4215`

**Code (telemetry/manager.py:366–372):**
In `BackpressureMode.BLOCK` mode (default), `put(event, timeout=30.0)` blocks
for up to 30 seconds waiting for queue space. If that timeout expires,
`queue.Full` is raised. The handler logs a `logger.error` ("BLOCK mode put()
timed out - export thread may be stuck") and increments `_events_dropped`.
This is sanctioned use of `logger.error` to report a telemetry-subsystem
failure — not a pipeline-data log.

```python
try:
    self._queue.put(event, timeout=30.0)
except queue.Full:
    # Timeout hit - thread may be dead or stuck
    logger.error("BLOCK mode put() timed out - export thread may be stuck")
    with self._dropped_lock:
        self._events_dropped += 1
```

**Proposed rationale:**
`TelemetryManager.handle_event` (telemetry/manager.py:366–372) under
`BackpressureMode.BLOCK` calls `queue.put(event, timeout=30.0)`. A 30-second
timeout indicates the export thread is likely dead or severely stuck; raising
into pipeline code would crash the pipeline over a telemetry failure, which
the audit>telemetry>logger primacy order explicitly forbids. The `except
queue.Full` catch records the drop via `_events_dropped` (thread-safe,
under `_dropped_lock`) and emits a `logger.error` reporting the telemetry
subsystem's own failure — a sanctioned logger use per the primacy policy.
The event is counted as dropped, not silently discarded. No pipeline row
data and no audit trail writes are affected; this is the telemetry layer
protecting pipeline continuity over telemetry completeness.

**Confidence / notes:** ACCEPT-ready. No prior judge rationale; this is a
fresh draft. Linter confirms fp=64535719a1fd4215 → telemetry/manager.py:368.

---

## Entry 7

**Key:** `telemetry/manager.py:R6:TelemetryManager.close:fp=79190bde3fa222f5`

**Code (telemetry/manager.py:578–594):**
`close()` must guarantee the shutdown sentinel (Python `None`) reaches the
export thread so it can drain remaining events and exit cleanly. The outer
loop attempts `self._queue.put(None, timeout=0.1)`. When the queue is full,
`except queue.Full` drains one item via `get_nowait()` + `task_done()` and
retries. The loop runs up to `maxsize + 10` times; if `sentinel_sent` is
still `False` after that, `logger.error` is emitted (line 597).

```python
try:
    self._queue.put(None, timeout=0.1)
    sentinel_sent = True
    break
except queue.Full:
    # Queue full - drain one item and retry
    try:
        discarded = self._queue.get_nowait()
        self._queue.task_done()
        ...
    except queue.Empty:
        pass
```

**Proposed rationale:**
`TelemetryManager.close` (telemetry/manager.py:582) must guarantee sentinel
insertion to prevent the export thread from hanging. `queue.Full` from the
bounded `put(None, timeout=0.1)` is the expected backpressure signal during
shutdown — the queue may still hold unprocessed events. The catch implements
the sentinel-guarantee invariant: drain one item with `get_nowait()` +
`task_done()` to account for queue unfinished-task bookkeeping, then retry
within the bounded loop (`maxsize + 10` attempts). This is concurrency-primitive
control flow at a shutdown path, not a silent swallow: if all drain attempts
are exhausted without sending the sentinel, `logger.error` is emitted at
line 597 (telemetry-subsystem failure, sanctioned logger use) and the
thread-join timeout at line 601–603 surfaces the hang to the operator.
Raising `queue.Full` inside `close()` would defeat the critical
sentinel-guarantee invariant documented in the docstring.

**Confidence / notes:** ACCEPT-ready. This entry already has a detailed
judge rationale (judge_confidence=0.83). The proposed rationale adds
explicit loop-bounds and the `task_done()` accounting detail. Linter confirms
fp=79190bde3fa222f5 → telemetry/manager.py:582.

---

## Entry 8

**Key:** `telemetry/manager.py:R6:TelemetryManager.close:fp=c87058caa78a1776`

**Code (telemetry/manager.py:584–594):**
Inside the `except queue.Full` drain branch (entry 7), a second `try` calls
`self._queue.get_nowait()` to evict one item. If the queue became empty
between the failed `put` and this `get_nowait` (a benign race: another
consumer emptied it), `except queue.Empty: pass` allows the outer loop to
retry `put(None, ...)` on the next iteration.

```python
try:
    discarded = self._queue.get_nowait()
    self._queue.task_done()
    ...
except queue.Empty:
    # Queue became empty between put and get - try put again
    pass
```

**Proposed rationale:**
`TelemetryManager.close` (telemetry/manager.py:592) uses `get_nowait()` to
drain one item from a full queue before retrying sentinel insertion. `queue.Empty`
is a benign concurrency race: the queue was drained by the export thread
between the `put_nowait`-that-raised-Full and this compensating `get_nowait`.
The correct response is to retry `put(None, ...)` on the next outer-loop
iteration — which is exactly what `pass` achieves. Raising here would abort
the sentinel-guarantee protocol unnecessarily; the outer loop handles the
ultimate failure case (sentinel not sent after maxsize+10 attempts) via
`logger.error` at line 597. This is concurrency-primitive race handling in a
shutdown path, not a silent swallow: the race scenario is documented in the
comment ("Queue became empty between put and get"), and the `sentinel_sent`
flag tracks whether the protocol succeeded.

**Confidence / notes:** ACCEPT-ready. This entry already has a detailed
judge rationale (judge_confidence=0.78). The proposed rationale makes the
race scenario more explicit. Linter confirms fp=c87058caa78a1776 →
telemetry/manager.py:592.

---

## Entry 9 — ESCALATE

**Key:** `core/rate_limit/limiter.py:R6:RateLimiter.try_acquire:fp=3e1bc5e9e76ba71f`

**Code:** No live finding.

A keyless linter run against `src/elspeth/` with no allowlist returns
**zero findings of any kind** for `rate_limit/limiter.py`. The current
`try_acquire` (limiter.py:264–268) uses a `try/finally` block with
`raise_when_fail=False` to coerce the pyrate-limiter library into returning
`False` on a full bucket instead of raising `BucketFullException`. There is
no `except` clause in `try_acquire`. There are no other `except` blocks in
the file.

The signed judge rationale (judge_confidence=0.88) describes catching
`BucketFullException` — code that has been replaced by the `raise_when_fail`
approach. The `scope_fingerprint` `c88073d93e03f338...` points to the
enclosing scope of `try_acquire`, but R6 requires an `except` handler to
exist; there is none.

**Escalation:** This entry is stale. Writing a `BucketFullException` rationale
would be dishonest about what the code does. Writing a `try/finally` rationale
would describe a `finally` block as an R6 finding, which is a category error.
**Do not re-justify.**

Operator action: delete `core/rate_limit/limiter.py:R6:RateLimiter:try_acquire:fp=3e1bc5e9e76ba71f`
from `core.yaml` directly (edit the YAML file, remove the allow_hits entry).
Do NOT use `rotate` — the rotate subcommand refuses judge-gated entries with
a hard RuntimeError and will not help here. No HMAC key is needed for
deletion. No code change is required; the refactored `try_acquire` is correct
and produces no live R6 finding.

---

## Entry 10

**Key:** `plugins/infrastructure/batching/mixin.py:R6:BatchTransformMixin._release_loop:fp=188d682d00e17ea1`

**Code (batching/mixin.py:350–352):**
`_release_loop` runs in a dedicated thread, blocking on
`self._batch_buffer.wait_for_next_release(timeout=1.0)`. That call raises
`TimeoutError` when no result is ready within 1 second. `except TimeoutError:
continue` skips to the next iteration.

```python
except TimeoutError:
    # Normal during low load - just loop and try again
    continue
```

**Proposed rationale:**
`BatchTransformMixin._release_loop` (batching/mixin.py:350–352) polls
`wait_for_next_release(timeout=1.0)` to emit FIFO-ordered batch results.
`TimeoutError` is the documented sentinel from that call meaning "no result
ready within the poll window, try again" — a normal condition during low load
or pipeline idle periods. `continue` is the correct disposition: the loop
must keep running until either a result becomes available or `ShutdownError`
is raised (the only clean-exit path, handled separately at line 354).
Re-raising `TimeoutError` or returning an error result would incorrectly
terminate the release thread. `ShutdownError` and unhandled `Exception` are
both handled by distinct branches in the same try/except block (lines 354 and
361), confirming the `TimeoutError` catch is not a catch-all swallow.

**Confidence / notes:** ACCEPT-ready. This entry already has a detailed
judge rationale (judge_confidence=0.85). The proposed rationale explicitly
names the `timeout=1.0` contract and the three-way handler structure.
Linter confirms fp=188d682d00e17ea1 → batching/mixin.py:350.

---

## Entry 11

**Key:** `plugins/infrastructure/batching/mixin.py:R6:BatchTransformMixin._release_loop:fp=24666a1e729638d7`

**Code (batching/mixin.py:354–359):**
`ShutdownError` is raised by `batch_buffer.shutdown()` (called from
`shutdown_batch_processing()`) to wake the blocked release thread after
all worker results have been drained. `except ShutdownError: break` exits
the `while True` loop — the clean shutdown path documented in the docstring:
"Exit condition: ShutdownError from buffer (set during shutdown_batch_processing)."

```python
except ShutdownError:
    # Buffer was shut down - exit cleanly.
    # This is the ONLY exit path: shutdown_batch_processing() waits for
    # workers to finish, then calls buffer.shutdown(), so all completed
    # results have been drained by this point.
    break
```

**Proposed rationale:**
`BatchTransformMixin._release_loop` (batching/mixin.py:354–359) is a
dedicated release thread that must keep running until all completed results
have been drained. `ShutdownError` is ELSPETH's typed signal that
`batch_buffer.shutdown()` has been called — the documented and only clean
exit path, fired by `shutdown_batch_processing()` after waiting for all
workers to complete. `break` is the correct disposition: it exits the `while
True` loop cleanly. Re-raising `ShutdownError` would propagate it out of the
thread and cause the join to observe an unexpected exception instead of a
clean exit. This is not a silent swallow — `ShutdownError` is an intentional,
narrowly-typed control-flow signal (an ELSPETH-defined exception, defined in
`plugins/infrastructure/batching/row_reorder_buffer.py`) used as a wakeup
mechanism, not an error condition.
`TimeoutError` and unhandled `Exception` are handled by distinct branches in
the same try/except block (lines 350 and 361).

**Confidence / notes:** ACCEPT-ready. No prior judge rationale (entry has
only "Code reviewed - legitimate pattern"). Linter confirms fp=24666a1e729638d7
→ batching/mixin.py:354. This is the strongest case — `ShutdownError` is our
own exception type used as a clean-exit signal, making the EAFP pattern
unambiguous.

---

## Summary Table

| # | Key (file:rule:symbol:fp) | Verdict |
|---|--------------------------|---------|
| 1 | mcp/server.py:R6:_find_audit_databases:fp=badb2efd50be18f6 | ACCEPT-ready |
| 2 | mcp/server.py:R6:_prompt_for_database:fp=250dc301eb3eda7c | ACCEPT-ready |
| 3 | mcp/server.py:R6:_prompt_for_database:fp=dbbeb81e161a9a1d | ACCEPT-ready |
| 4 | mcp/server.py:R6:_prompt_for_database:fp=b20406c8a83aad1b | ACCEPT-ready |
| 5 | telemetry/manager.py:R6:TelemetryManager.handle_event:fp=0a1bb9f3fb968fdf | ACCEPT-ready |
| 6 | telemetry/manager.py:R6:TelemetryManager.handle_event:fp=64535719a1fd4215 | ACCEPT-ready |
| 7 | telemetry/manager.py:R6:TelemetryManager.close:fp=79190bde3fa222f5 | ACCEPT-ready |
| 8 | telemetry/manager.py:R6:TelemetryManager.close:fp=c87058caa78a1776 | ACCEPT-ready |
| 9 | core/rate_limit/limiter.py:R6:RateLimiter.try_acquire:fp=3e1bc5e9e76ba71f | **ESCALATE** — stale, no live finding |
| 10 | plugins/infrastructure/batching/mixin.py:R6:BatchTransformMixin._release_loop:fp=188d682d00e17ea1 | ACCEPT-ready |
| 11 | plugins/infrastructure/batching/mixin.py:R6:BatchTransformMixin._release_loop:fp=24666a1e729638d7 | ACCEPT-ready |

**ESCALATE detail (#9):** Entry `core/rate_limit/limiter.py:R6:RateLimiter.try_acquire:fp=3e1bc5e9e76ba71f`
is stale. The current `try_acquire` (limiter.py:264–268) uses `try/finally`
with `raise_when_fail=False` — there is no `except` clause. The linter
reports zero findings for `rate_limit/limiter.py`. The signed judge rationale
describes `BucketFullException` handling that no longer exists. Correct action:
delete the entry directly from `core.yaml` (do NOT use `rotate` — it refuses
judge-gated entries). No HMAC key needed for deletion. No code change required.

---

## Signing (operator-only)

The 10 ACCEPT-ready rationales are wired into `notes/sign_cluster_a.sh`, one
`justify` call each (`--judge-transport agent --judge-tools readonly`, so the
investigating judge re-checks every rationale and signs only on ACCEPT). The
HMAC signing key is operator-only and is read from your environment — it is NOT
in the script.

**Sign all 10:**

```bash
ELSPETH_JUDGE_METADATA_HMAC_KEY=<your-key> bash notes/sign_cluster_a.sh
```

A `BLOCKED`/failed entry is reported and skipped (its rationale needs another
pass); re-run after fixing just that one. Set `ELSPETH_JUSTIFY_OWNER` to
override the audit owner (defaults to `$USER`).

**Delete the stale #9** (separate — a deletion, not a `justify`; no HMAC key,
no judge call; fp-keyed so it survives line shifts):

```bash
python - <<'PY'
fp = "3e1bc5e9e76ba71f"
path = "config/cicd/enforce_tier_model/core.yaml"
lines = open(path).read().splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.lstrip().startswith("- key:") and fp in l)
end = len(lines)
for j in range(start + 1, len(lines)):
    l = lines[j]
    if l.startswith("- ") or (l.strip() and not l.startswith((" ", "\t", "-"))):
        end = j; break
removed = "".join(lines[start:end])
assert "try_acquire" in removed and fp in removed, "guard: refusing to delete the wrong block"
open(path, "w").write("".join(lines[:start] + lines[end:]))
print(f"deleted {end - start} lines (the stale try_acquire R6 entry) from {path}")
PY
```

After both: run the tier-model check to confirm the corpus is clean, then the
`max_allow_hits` ceiling in `_defaults.yaml` can drop by 1 (the deleted entry).
