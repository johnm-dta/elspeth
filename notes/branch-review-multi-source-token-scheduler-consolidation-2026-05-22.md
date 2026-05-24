# Consolidation — feat/multi-source-token-scheduler

Date: 2026-05-22
Worktree: `.worktrees/multi-source-token-scheduler`
Base: `origin/RC5.2` · **Targets RC6, not RC5.2**
Orientation note: [branch-review-multi-source-token-scheduler-architecture-2026-05-22.md](branch-review-multi-source-token-scheduler-architecture-2026-05-22.md)
Dedup map (execution detail): `.worktrees/multi-source-token-scheduler/notes/multi-source-audit-dedup-map.md`

## TL;DR

**74 tickets across 4 audits → 32 canonical findings → 5 RC6-aware tiers.**

- **4 P0s** (3 confirmed correctness bugs + 1 contingent on PRAGMA verification) block any RC6 development testing.
- **The structural meta-finding**: multi-source was added *beside* single-source, not *in place of* single-source. 26+ engine call sites still read `config.source.*`. The legacy single-source surface is preserved with explicit comments admitting it. Dim2's framing: *"The scheduler primitive itself is well-designed. The upgrade around the scheduler is unfinished."*
- **The diagnosis is unambiguous across four lenses.** Dim1 (engine) saw it as dual-truth, dim2 (architecture) named the file count (26+ callsites), dim3 (docs) saw every binding contract still teaching single-source, dim4 (tests) saw the one multi-source resume test bypassing the production path. All four lenses converged on the same diagnosis from different angles.

## What "RC6, not RC5.2" changes about how these findings sort

Memory: `project_multi_source_token_scheduler_rc6` — this branch targets RC6, a future release line. Implications for grading:

- **3 P0 correctness bugs stay P0.** They bite *any* run, not just shipped runs.
- **CLAUDE.md / docs that say "exactly 1 source per run" drop from P1 → P3.** They are *correct for RC5.2* (the currently shipping line). They become RC6 release-blockers, not merge-blockers.
- **ADRs, composer skill, binding contracts move into tier-3 (pre-publish).** Required before RC6 publishes; not before RC6 development continues.
- **`docs/release/guarantees.md` §7.1 "single-threaded in RC-3"** is correct today. It becomes false when RC6 publishes, not on merge.

## Tier 1 — Must fix now (blocks any RC6 development testing)

These are correctness bugs. They will bite the operator's own RC6 testing on the next run.

| Group | Canonical | Where | What |
|---|---|---|---|
| G1 | **elspeth-941f1508f5** | `core/landscape/scheduler_repository.py:552-592` + `engine/processor.py:2553` | **Lease self-steal on expiry.** Worker reaps its own in-flight lease via `recover_expired_leases`, crashing run with `AuditIntegrityError`. Triggered by any plugin call exceeding the 300s default lease (i.e. every LLM/HTTP pipeline). |
| G2 | **elspeth-01942858c3** | `engine/orchestrator/core.py:3497-3512` | **`next(iter(schema_contracts_by_source))` picks arbitrary source's contract on multi-source resume.** Silently validates rows under the wrong schema — Tier 1 evidence tampering, the exact failure mode the trust model exists to prevent. Dim4's elspeth-d5f0194fc8 contributes a 4-step test plan that should land with the fix. |
| G3 | **elspeth-5c5e88b071** | `engine/processor.py:2531-2657` | **`created_pending_sink_this_drain` flag blocks recovery.** Only ONE pending-sink draws per resume call; multi-token crash + resume can't converge in a single resume call. |
| ~~G28~~ | ~~**elspeth-8536552dcb**~~ | ~~`core/landscape/database.py` + `scheduler_repository.py`~~ | **VERIFIED 2026-05-23, dropped out of Tier 1.** Three-lens verification (embedded DB / systems / solution arch) confirmed PRAGMAs are applied via `@event.listens_for(engine, "connect")` at `database.py:320-329`. Scheduler shares `db.engine` — no bypass. Re-graded to P2, label moved to `tier-5-code-health`. Three follow-up tickets filed under `g28-followup` label captured the solution-arch's design-discipline concerns (type-enforcement, probe-and-assert, plain-SQLite test). |

**Fix sequencing**: G1 and G3 are independent of the rest and small enough to land first (one-SQL-predicate fixes). G2 is small in code but should land with its regression test (dim4's test plan). ~~G28 is a verification gate first.~~ G28 verified P2 on 2026-05-23 — three new follow-up tickets filed under `g28-followup` label.

**Tier 1 net count after G28 verification: 3 P0s (G1, G2, G3).**

## Tier 2 — RC6 structural cleanup (the actual RC6 work)

The "delete the dual-truth surface" work. These 13 groups collectively resolve the *"26+ sites still read `config.source`"* problem. Doing them as one coordinated structural move resolves ~12 per-site findings as a side effect.

### Dim2's 5 named structural moves (canonical framing)

1. **Pick "the source" or "sources" — delete the other** · G4 elspeth-af87655cdb (root) + G5 elspeth-781e042709 (build_execution_graph facade) + G9 elspeth-1ed6db3db4 (composer state mirror) + G11 elspeth-11a4ed2630 (token identity None-defaults)
2. **Make the orchestrator natively multi-source** · G12 elspeth-bc81207798 (per-source loop is sequential) + G8 elspeth-bdc43c911e (singular accessors stale)
3. **Extract `SchedulerDriver` from `RowProcessor`** · G26 elspeth-54e9c72f1b (3620-LOC processor) [actually tier-5, here for context]
4. **Extract `MultiSourceCoordinator`** · G12 + G26 sub-task
5. **Document QUEUE vs Coalesce; resolve sink-exempt-from-queue policy** · G30 elspeth-30e7ac9571

### Structural correctness (same tier, different theme)

| Group | Canonical | What |
|---|---|---|
| G6 | **elspeth-2e2f2184ab** | Dual writer for source schema contract (`runs.contract_json` + `run_sources` both written). Resume picks reader at read-time → ambiguity. |
| G7 | **elspeth-b680e81bce** | `_drain_in_memory_work_queue` kept solely for tests — violates CLAUDE.md *"never bypass production code paths."* Dies with the dual-truth surface. |
| G10 | **elspeth-5335eb63e4** | `unprocessed_rows` is `3-tuple|4-tuple` union discriminated by `len()` — needs `ResumedRow` dataclass. Blocked-by G6. |
| G27 | **elspeth-4678a5aa73** | **CAS races on multi-worker.** `claim_ready` / `claim_pending_sink` SELECT-then-UPDATE escalates loser to `AuditIntegrityError` instead of benign retry. Absorbs 4 related findings (asymmetric attempt-bump, `mark_blocked_barrier_terminal` rowcount mismatch, `mark_failed` optional lease window). |
| G29 | **elspeth-2b608abbd3** | **Scheduler state transitions emit no Landscape audit rows.** Audit-primacy gap: cannot reconstruct lease-expiry timelines from `token_work_items` final state alone. Companion to G17 ADR (the ADR has to commit to whether to add a `scheduler_events` table). |
| G30 | **elspeth-30e7ac9571** | Sink/QUEUE/Coalesce exemption lets multi-source MOVE-fan into a sink without QUEUE primitive. Contract hole. |

## Tier 3 — Pre-publish requirements (before RC6 ships, not before merge)

Two sub-classes: **correct-today-but-wrong-for-RC6** (P3) vs **missing-not-wrong** (P1).

### P3 — correct today, will be wrong when RC6 publishes

| Group | Canonical | What |
|---|---|---|
| G13 | **elspeth-2409a7c7bf** | `CLAUDE.md:119` *"exactly 1 per run."* Correct for RC5.2; update for RC6. |
| G14 | **elspeth-e4cf92586c** | Single-source doc corpus stale (omnibus): `docs/reference/configuration.md` + 5 other files. **Closure protocol: enumerate all 6 file paths as a comment on canonical before closing the 5 duplicates.** |
| G15 | **elspeth-bc91898548** | `docs/release/guarantees.md §7.1` *"single-threaded in RC-3."* |
| G21 | **elspeth-8c4ca2d89c** | `docs/release/elspeth-progress-rc1-to-rc5.md` missing this delivery. |
| G23 | **elspeth-dde60f76b4** | Redaction collapses source paths to constant; per-source provenance decision undocumented. **Security implications — wants explicit design decision.** |
| G31 | **elspeth-1869c9ba64** | Quarantine rows consume `ingest_sequence` numbers; gap semantics undocumented. |

### P1 — missing entirely, RC6 publish-blocker

| Group | Canonical | What |
|---|---|---|
| G16 | **elspeth-86de46bcd4** | **Composer skill teaches "every pipeline needs one source."** Engine ships multi-source but composer LLM cannot author it. Feature unreachable to its primary persona until this lands. |
| G17 | **elspeth-57d0031a14** | **No ADR for multi-source OR durable scheduler.** Two architecturally-significant deliveries have zero governance record. Absorbs QUEUE observed-schema design + scheduler multi-worker-boundary design. |
| G18 | **elspeth-06aecb78a0** | `docs/architecture/landscape.md` missing `run_sources` / `token_work_items` / new row columns. |
| G19 | **elspeth-559bce3459** | No `docs/runbooks/scheduler-lease-recovery.md`. First lease-expiry incident has no operational playbook. Bundles resume runbook + MCP guide gaps + diagnostic-MCP-tool gap. |
| G20 | **elspeth-c2aa936ad8** | `docs/contracts/system-operations.md` Coalesce invariants assume single-source `row_id` — no `source_node_id`/`ingest_sequence` disambiguation. |
| G22 | **elspeth-7f3ac1ac65** | "Do not fabricate source_row_index / ingest_sequence" lives only in an exception string. Needs plugin-protocol doc + lint rule for discoverability. |

## Tier 4 — Test gap remediation (parallel with development)

Eight gaps. Two are critical (P1), rest are P2/P3.

| Group | Canonical | What |
|---|---|---|
| G25a | **elspeth-71dcedcb66** | **P1** — Zero e2e crash-and-resume coverage for multi-source. The headline feature has no production-path crash test. |
| G25b | **elspeth-6116873e3b** | **P1** — Source-isolation tests absent (no test asserts one source's failure doesn't starve others, no credential leak test, etc.). |
| G25c | **elspeth-e8a1250782** | P2 — Hypothesis state machine still models OLD lifecycle (`CREATED, PROCESSING, FORKED, …`), not `READY → LEASED → …`. |
| G25d | **elspeth-0bae6d8a52** | P2 — Scheduler lease + claim edge tests under-covered (`recover_expired_leases` multi-expiry, `claim_pending_sink` CAS, `claim_ready` ordering, two-worker contention). |
| G25e | **elspeth-40886ef9f8** | P3 — `test_concurrent_resume.py` is misnamed; contains only rejection tests. Real coverage tracked by G25a. |
| G25f | **elspeth-9c7ae2d60e** | P2 — Test calls private `_reconstruct_resume_state` instead of `Orchestrator.resume`. Subsumed by G25a's e2e test. |
| G25g | **elspeth-e51eaed773** | P2 — No invariant test that `rows.source_node_id IS NOT NULL` survives resume. |
| G25h | **elspeth-7bb7124e8f** | P2 — Chaos fixtures (ChaosLLM / ChaosWeb / ChaosEngine) unwired against multi-source/scheduler. |

## Tier 5 — Code health (RC6-cycle, lowest urgency)

| Group | Canonical | What |
|---|---|---|
| G26 | **elspeth-54e9c72f1b** | `processor.py` is 3620 LOC; extract `SchedulerDriver`. Absorbs `orchestrator/core.py` 3894 LOC extraction (resume.py + MultiSourceCoordinator). |
| G32 | **elspeth-d869cc0113** | Tier-model allowlist churn: engine.yaml +262, web.yaml +289, core.yaml +46. Run periodic `cicd-allowlist-audit` skill. |

## What's about to happen mechanically

A background executor agent is dispatched with the dedup map and this consolidation as its spec. It will:

1. **Re-grade 32 canonicals** — set RC6-aware priority, add tier label, add a comment explaining the reframe.
2. **Merge evidence into canonicals** — for the 18 merge-then-close tickets, append unique evidence as a comment on the canonical before closing.
3. **Close 19 pure duplicates** — with cross-reference comment to canonical.
4. **G14 special-case** — enumerate all 6 stale-doc file paths as a comment on canonical *before* closing the 5 sibling tickets.

Net result: **32 canonical tickets open, re-graded; 37 tickets closed with cross-references**.

## Recommended fix order

If you want to start fixing today:

1. **G1, G3** (independent, small) — same morning.
2. **G2** (small code, must land with G2's test plan from dim4).
3. **G28 verification** — answer "are scheduler-bearing connections getting the full PRAGMA block?" That answer is a 5-minute check; if no, treat as a 4th P0 and fix before continuing.
4. **G17 ADRs** (architectural decision record before more structural code lands) — pre-commits the answer to "delete or keep the legacy facade?" so subsequent structural work doesn't ping-pong.
5. **Tier 2 structural** — sequence is operator's call, but G4 → G5 → G6 → G7 is a natural order (root facade → builder facade → dual-writer → dual-drain).

Tier 3 (docs/composer/ADRs) and Tier 4 (tests) can land in parallel with Tier 2 — they don't share code.
