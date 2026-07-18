# Executed Evidence

Evidence was executed on 2026-07-18 AEST against merge commit
`3c782ac3c7efb0550495be38f75800eddffa639a`. The only worktree overlay was
this dated documentation package; every non-document path matched the commit.
The exact argument vectors, safe environment, timestamps, collected node
indexes, coverage tuples, result counts, and artifact hashes are recorded in
[assessment.json](assessment.json).

## Results

| Evidence | Scope | Result | Establishes | Does not establish |
| --- | --- | --- | --- | --- |
| EV-001 | Scheduler fencing and pending-sink admission | 46 passed in 4.971 s | Narrow SQLite strict-authority, stale-token zero-mutation refusal, legacy-adapter isolation, and complete-bundle admission/refusal | Production orchestration, independent-process contention, expiry/reclaim, all subtypes, plugin composition, restart |
| EV-002 | Barrier completion and built-in sink-effect recovery | 38 passed in 2.626 s | Exact barrier snapshots, atomic consume/emission, refusal/rollback, built-in response-loss recovery, diversion, and virtual effects | Abrupt process death, external sinks, long-call takeover, every PB-06/PB-07 seam, aggregation continuation |
| EV-003 | TS-07 through TS-10 disposition images | 30 passed in 1.153 s | Exact row/event/branch-loss success images, subtype/owner/membership/bundle refusal, and rollback | Production plugin/follower composition, contention, restart, all database profiles, read-model truth tables |
| EV-004 | TS-02 source completion and pre-fix recovery | 13 passed in 1.173 s | Current `process_row` atomic ingress and rollback; strict duplicate/mismatch/attempt/malformed/conflict refusal; deterministic crash-after-repair public resume with exactly one transform call | Abrupt OS process death, independent processes, all source exclusion/failure/quarantine arms, non-SQLite profiles, all ten TS-02/PB-01 dimensions |
| **Total** | Four fresh vectors | **127 passed** | Narrow properties attached to exact catalog cells | No global leg or hard-gate completion |

Every vector used `.venv/bin/python -m pytest -p no:dotenv -q -n 0` and a
package-local JUnit path. Collection reran the same selectors with
`--collect-only`; the retained indexes contain exactly 46, 38, 30, and 13
node IDs respectively. Each run emitted one expected `PytestConfigWarning`
because disabling pytest-dotenv leaves the configured `env_files` option
unknown. No `.env` value was loaded or captured.

## EV-004 exact selectors

- `TestProcessRowNoTransforms::test_records_source_node_state`
- `TestProcessRowNoTransforms::test_fenced_ingest_commits_source_completion_before_return`
- `TestProcessRowNoTransforms::test_fenced_ingest_rolls_source_completion_back_with_scheduler_failure`
- the conflicting-state, duplicate-claim, mismatched-work-item, later-attempt,
  malformed-state, and four source-impossible-metadata reconciliation cases;
- `TestMidClaimCrashResume::test_ts02_source_completion_gap_reconciles_once_before_plugin_execution`.

The E2E case constructs the exact pre-fix post-TS-02 image, crashes after the
repair transaction but before scheduler recovery or a plugin call, and resumes
through the public orchestrator. It proves one source witness, one effective
transform call, one sink result, and no duplicate terminal outcome. It is
deliberately classified `partial` for the crash/restart dimension because the
kill is deterministic `BaseException` injection rather than a separate OS
process.

## Environment

| Fact | Captured value |
| --- | --- |
| Python | 3.13.1; `/home/john/elspeth/.venv/bin/python`; build `main`, 2024-12-19 |
| pytest / uv / Git | 9.0.3 / 0.10.2 / 2.43.0 |
| SQLite / SQLAlchemy | 3.47.1 / 2.0.45 |
| Kernel | Linux 6.8.0-124-generic x86_64 |
| Locale / timezone | `en_US.UTF-8` with remaining categories `C`; Australia/Canberra, UTC+10 |
| `PYTHONHASHSEED` | Unset |
| `.env` | Present, explicitly disabled; values not loaded or captured |
| `pyproject.toml` SHA-256 | `d084c07c35783c8fce3313d77ac0662df66c8db191b302770438f880f6fcf9e4` |
| `uv.lock` SHA-256 | `fac5b295e29ff4ee796a1ca9317c32d9bc7ff4ca1e6863d16f82666d655d680f` |

JUnit, stdout, and stderr are retained under `artifacts/`; exact node IDs are
under `nodes/`. `assessment.json` binds every retained file by SHA-256. Empty
stderr files are kept so the absence of process-level stderr remains auditable.
