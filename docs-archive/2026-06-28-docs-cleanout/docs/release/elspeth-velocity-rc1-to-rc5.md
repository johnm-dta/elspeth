# ELSPETH Velocity Report — Daily Output, January–May 2026

**Period covered:** 12 January 2026 → 19 May 2026 (128 calendar days; **123 active commit days**)
**Total commits:** 4,521 unique across the RC-1, RC-2, and RC-5.2 history snapshots
**Average velocity:** 35.3 commits per calendar day; **36.8 commits per active day**
**Author of record:** John Morrissey
**Audience:** Engineering team and engineering leadership
**Register:** Technical
**Purpose:** Per-day commit volume so the team can see how much was completed each day, where the peaks are, and what each peak delivered.

> **Reading the figures.** Commit volume is delivery telemetry, not a substitute for assurance, maintainability, or product value. Per-active-day rates of 36–80 commits should be read against the project's contributor count, which is tracked separately at the enterprise level. The recovery patterns visible in the data — 1–2 day pauses after major releases, and low-activity weeks immediately preceding phase changes — reflect a deliberate cadence pattern, not external scheduling and not slippage. Reading any single week's count without the phase context will mislead; the *Velocity by Phase* table below is the right reading frame.

This document is the **velocity / tempo** view. For cumulative feature output, see [elspeth-progress-rc1-to-rc5.md](elspeth-progress-rc1-to-rc5.md).

## What this report does not measure

Commit volume is a tempo signal, not a value signal. This report does **not** measure:

- **Effort.** A single 200-line commit and a single one-line commit each count as one commit. Days of architecture analysis that produced no commits do not appear here at all.
- **Value.** A bug-fix commit and a refactor-for-cleanliness commit each count as one commit; this report does not (and cannot) distinguish their value to the system.
- **Code volume.** Commits in this project range from one-line typo fixes to 4,000-line plugin landings. Per-day commit counts are **not** a proxy for per-day code volume.
- **Quality.** Every commit in the visible history passed pre-commit hooks, type checks, and the test suite that existed at the time. This report does not measure how well-tested or well-reviewed each commit was, and a higher commit count is not evidence of higher quality.
- **Reasoning load.** The hardest design decisions in the project (the trust-tier model, the routing trilogy, the declarative DAG model, the RC-5 composer architecture, the redaction MANIFEST) produced fewer commits per day than the implementation work that followed them, because the reasoning ran ahead of the code.

The right reading of this report is "where did the system's tempo land on each day", not "where was the most work done".

---

## Headline Statistics

| Measure | Value |
|---------|------:|
| Active days | 123 |
| Idle days | 5 (2026-02-23, 03-11, 03-16, 03-17, 04-06) |
| Median active-day commits | 27 |
| Mean active-day commits | 36.8 |
| Maximum daily commits | **177 (2026-01-20)** |
| Days ≥ 100 commits | 6 |
| Days ≥ 50 commits | 31 |

---

## Top 15 Peak Days

The peak days where epics land or RC cutovers happen. Attribution is taken from the dominant feature-commit cluster on that day.

| Rank | Date | Commits | What landed |
|-----:|------|--------:|-------------|
| 1 | **2026-01-20** | **177** | Pre-RC1 LLM pooled-execution + batch aggregation: AzureLLMTransform batch + pooled, OpenRouter `_process_batch`, public pooled-execution API. Heaviest single day in project history. |
| 2 | 2026-01-30 | 156 | RC-1 hardening burst: ChaosLLM weighted error selection + metrics export; `AuditIntegrityError` / `OrchestrationInvariantError` introduction; Azure Monitor + Datadog exporters; coalesce executor fork/join improvements; scan-group files for the eight high-risk modules. |
| 3 | **2026-05-12** | **142** | RC-5.2 composer redaction-MANIFEST pass: guided `step_2_chosen_plugin`; `RecipeOfferTurn` editable slots; demo-SLA E2E un-fixme'd; ARG_ERROR canonicalization via Pydantic `__cause__`; declarative MANIFEST entries for `request_advisor_hint` and secret-mutation tools. |
| 4 | 2026-01-18 | 125 | Pre-RC1 LLM infrastructure: OpenAI / Azure client base classes; pooled-execution scaffolding; reorder buffer with timing. |
| 5 | 2026-02-03 | 125 | RC-2.1 → RC-2.2: Langfuse SDK v3 migration; secret-resolution audit trail; schema-contract propagation; Tier 2 tracing on Azure / OpenRouter / batch LLM. |
| 6 | 2026-02-02 | 118 | RC-2 cutover day. `ELSPETH - Release Candidate 2` commit + post-cutover cleanup (display headers; FieldResolutionApplied event; Tier 1 corrupt-field-resolution crash). |
| 7 | 2026-05-14 | 97 | RC-5.2 release-stamp day: changelog finalize; per-step chat merge; phase3 compose-loop persistence merge. |
| 8 | 2026-05-19 | 94 | Phase 8 polish + Phase 6 completion-gestures merge + CI allowlist burn-down merge + Phase-5 chat-data-entry merge + ansible-ubuntu-deploy docs. |
| 9 | 2026-05-18 | 94 | Phase 7 catalog reshape merge + `fix/catalog-i1-i2-i3` (drawer error log, snapshot lock, NETWORK retirement) + plugin-coverage gate calibration. |
| 10 | 2026-05-09 | 93 | RC-5.1 composer-progress-persistence Phase 1B: `persist_compose_turn` happy path, `OperationalError` + audit-failure primacy, `IntegrityError` disposition, persist-payload DCs. |
| 11 | 2026-05-13 | 88 | Phase A coverage gap + per-step chat → RC-5.2 merge + post-rebase tier-model fix-ups + cross-step `chat_history` accumulation test. |
| 12 | 2026-01-21 | 87 | Final pre-RC1 hardening day: line-length-140 reformat + bug-burndown sweeps before RC-1 cutover. |
| 13 | 2026-01-29 | 85 | RC-1 hardening: telemetry property tests (DROP/BLOCK back-pressure, EventBus re-entrance, concurrent close); contract tests for keyword filter / content safety / prompt shield / multi-query. |
| 14 | 2026-05-17 | 84 | Phase 2C composer implementation: `ReadinessRowDetail` + `ExplainDialog` real impls; `InspectorPanel` mount; Validate button deletion with subscription wiring. |
| 15 | 2026-01-28 | 83 | RC-1 hardening: contract boundary tightening (`RoutingReason` discriminated union, `TransformErrorReason` TypedDict); type-soup cleanup. |

---

## Velocity by Phase

> **Skim aid — three velocity tiers across the project's life.**
>
> 1. **Hot cuts (Pre-RC1, RC-1 Hardening, RC-5.2 Composer Maturation): 71–80 commits / active day.** Empty-scaffold-to-shipping foundation work, plus high-tempo merge sprints around major releases.
> 2. **Mid-cadence sub-release stream (RC-2): 52.7 commits / active day.** Five rapid sub-releases each ~3–5 days apart, with shorter individual review cycles.
> 3. **Architectural-remediation and correctness sprints (RC-3, RC-4 + RC-5 cut, RC-5.1): 19–23 commits / active day.** Smaller, more carefully audited changes; each commit tends to be a focused fix with paired tests.
>
> The current RC-5.2 surge (May 12–19) is the **highest sustained tempo since RC-1 cutover** — 79.8 commits / active day over 8 consecutive active days. See the per-phase table below for the underlying numbers.

The seven canonical phases of the project, with per-day averages.

| Phase | Dates | Calendar days | Active days | Total commits | Commits / active day | Headline |
|-------|-------|--------------:|------------:|--------------:|---------------------:|----------|
| **Pre-RC1 Foundation** | 2026-01-12 → 01-22 | 11 | 11 | 782 | **71.1** | Empty scaffold → working SDA framework with LLM, Azure, plugins, DAG, checkpoint |
| **RC-1 Hardening** | 2026-01-23 → 02-02 | 11 | 11 | 811 | **73.7** | Telemetry built; ChaosLLM built; 100+ bugs closed |
| **RC-2 Sub-releases (0.1.0 → RC-2.5)** | 2026-02-03 → 02-12 | 10 | 10 | 527 | **52.7** | Key Vault, schema contracts, `PipelineRow`, WebScrape, SQLCipher, declarative DAG, ChaosWeb |
| **RC-3 Series (0.3.2 → 0.3.4)** | 2026-02-13 → 03-10 | 26 | 25 | 475 | **19.0** | Strict typing at audit boundaries, T10/T17/T18/T19 remediation, 191-bug triage, deep-freeze |
| **RC-4 + RC-5 cut** | 2026-03-11 → 04-03 | 24 | 21 | 478 | **22.8** | Dataverse, RAG, ChromaSink, `depends_on`, commencement gates, RC-5 web composer cut |
| **RC-5.1 Composer Correctness** | 2026-04-04 → 05-11 | 38 | 37 | 810 | **21.9** | Substrate framing, validator hardening, advisor escalation, 10 statistical batch plugins, audit-integrity test coverage |
| **RC-5.2 Composer Maturation** | 2026-05-12 → 05-19 | 8 | 8 | 638 | **79.8** | Guided mode, 4 phases of composer progress persistence, frontend recovery UX, MANIFEST redaction walker, RC-5.2 hot-fix integration |
| **Total** | 2026-01-12 → 05-19 | 128 | 123 | **4,521** | **36.8** | — |

---

## Full Per-Day Commit Table

> **Skim guide for the table below.** The full table preserves every active day so that any specific date can be traced. For the highlights view, use the *Top 15 Peak Days* table above. Days at ≥ 50 commits are RC cutovers, merge sprints, or land-day clusters; days at < 10 commits are typically narrow correctness/test follow-ups or recovery days after a major release. The "Notes" column is populated only for days where a notable event is attributable; absence of a note means a routine commit cluster with no single landmark.

```
Date          Commits   Notes
2026-01-12         70   Initial scaffold (commit 748666333); Phase 1 Foundation plan
2026-01-13         74   Canonical JSON two-phase normalization
2026-01-14         40
2026-01-15         28
2026-01-16         51
2026-01-17         54
2026-01-18        125   LLM pooled-execution + reorder buffer
2026-01-19         59
2026-01-20        177   Azure / OpenRouter batch + pooled — heaviest day in project
2026-01-21         87   Bug burndown before RC-1 cutover
2026-01-22         17   RC-1 release day (Initial RC tag)
2026-01-23         47
2026-01-24         77
2026-01-25         49
2026-01-26         65
2026-01-27         42
2026-01-28         83   Contract boundary hardening
2026-01-29         85   Telemetry property tests; contract tests
2026-01-30        156   ChaosLLM + Azure Monitor / Datadog exporters
2026-01-31         31
2026-02-01         58
2026-02-02        118   RC-2 cutover (commit f4f348de1); post-cutover cleanup
2026-02-03        125   RC-2.1: Langfuse v3, Key Vault, schema contracts
2026-02-04         36
2026-02-05         37
2026-02-06         27
2026-02-07         54   RC-2.3 land begins: PipelineRow migration + WebScrape
2026-02-08         38
2026-02-09         43
2026-02-10         35
2026-02-11         54
2026-02-12         78   RC-2.5: SQLCipher, declarative DAG, ChaosWeb
2026-02-13         67
2026-02-14         19
2026-02-15         31
2026-02-16          5   Light weekend
2026-02-17         12
2026-02-18          4
2026-02-19          4
2026-02-20         11
2026-02-21         11
2026-02-22         10   RC-3.2 tag (v0.3.0-rc3.2) cut by 4839e388a
2026-02-23          0   First idle day in the project history
2026-02-24          3
2026-02-25         29
2026-02-26         26
2026-02-27         34
2026-02-28         11
2026-03-01         27   archive/feature-inventory.md generated at RC-3.3 prep
2026-03-02         17   0.3.3 release (commit 736c96804)
2026-03-03          7   archive/rc4-planning-brief.md (planning complete)
2026-03-04         12
2026-03-05         24
2026-03-06         23
2026-03-07         21
2026-03-08         24
2026-03-09         21
2026-03-10         22   0.3.4 stabilization bump (commit d3442f458)
2026-03-11          0   Idle
2026-03-12          2
2026-03-13          1
2026-03-14          1
2026-03-15          6
2026-03-16          0   Idle
2026-03-17          0   Idle
2026-03-18          2
2026-03-19         21
2026-03-20         28
2026-03-21         16
2026-03-22         40   RC-4.0 implementation acceleration
2026-03-23         26
2026-03-24         22
2026-03-25         42
2026-03-26         17
2026-03-27         25
2026-03-28         20
2026-03-29         57   Dataverse / RAG / output-schema-contracts main land
2026-03-30         30
2026-03-31         12
2026-04-01         13
2026-04-02         37   16 P1 plugin/LLM bugs closed pre-RC-5 cut
2026-04-03         60   RC-5 cut (commit ecd68ad0f) — version bump + doc refresh
2026-04-04          3
2026-04-05          2
2026-04-06          0   Idle
2026-04-07         16
2026-04-08         30
2026-04-09         25   RC4.2-UX exception-hygiene merge
2026-04-10         28
2026-04-11          9
2026-04-12         21
2026-04-13          9
2026-04-14         27
2026-04-15         17
2026-04-16         12
2026-04-17         28   Top-level index refresh; broken-link sweep
2026-04-18         23
2026-04-19          8
2026-04-20         45
2026-04-21         20
2026-04-22          5
2026-04-23          5
2026-04-24          2
2026-04-25          2
2026-04-26          6
2026-04-27          2
2026-04-28         39
2026-04-29         32
2026-04-30         11
2026-05-01          6
2026-05-02          6
2026-05-03         27
2026-05-04         20
2026-05-05          7
2026-05-06         77   Composer scenario fan-out (Tier 1.5 Step B); anti-anchor hint; batch statistical quick wins
2026-05-07         31
2026-05-08         57
2026-05-09         93   Composer-progress-persistence Phase 1B
2026-05-10         23
2026-05-11         36   RC-5.1 release stamp (commit 0964922c4); 0.5.1 changelog finalize
2026-05-12        142   RC-5.2 redaction MANIFEST + ARG_ERROR canonicalization
2026-05-13         88   Per-step chat → RC-5.2; Phase A coverage gap
2026-05-14         97   RC-5.2 release stamp (commit 81546eee9); phase3 compose-loop persistence merge
2026-05-15          5   Light day
2026-05-16         34   Phase 1B default-mode frontend; panel-review fixes
2026-05-17         84   Phase 2C composer implementation
2026-05-18         94   Phase 7 catalog reshape merge; catalog i1-i2-i3 fixes
2026-05-19         94   Phase 8 polish; Phase 6 completion-gestures merge; CI allowlist burn-down
```

---

## Per-Week Summary

For a coarser view, here are the calendar weeks (Monday-anchored) with totals and themes.

| Week | Dates | Commits | Notable |
|------|-------|--------:|---------|
| W01 | 2026-01-12 → 01-18 | 442 | Project initiation; canonical JSON; LLM scaffolding |
| W02 | 2026-01-19 → 01-25 | 513 | Peak Pre-RC1 (Jan 20 = 177); LLM batch / pooled; RC-1 cutover Jan 22 |
| W03 | 2026-01-26 → 02-01 | 520 | RC-1 hardening: telemetry, ChaosLLM, contract boundary tightening |
| W04 | 2026-02-02 → 02-08 | 435 | RC-2 cutover; RC-2.1 (Key Vault, schema contracts); RC-2.2; RC-2.3 land begins |
| W05 | 2026-02-09 → 02-15 | 327 | RC-2.4 bug sprint; RC-2.5 (SQLCipher + ChaosWeb + declarative DAG) |
| W06 | 2026-02-16 → 02-22 | 57 | RC-3.2 prep + tag day (Feb 22) |
| W07 | 2026-02-23 → 03-01 | 130 | RC-3.3 architectural-remediation kickoff |
| W08 | 2026-03-02 → 03-08 | 128 | RC-3.3 release (Mar 2); steady remediation cadence |
| W09 | 2026-03-09 → 03-15 | 53 | RC-3.4 release (Mar 10); transition to RC-4 planning |
| W10 | 2026-03-16 → 03-22 | 107 | RC-4 implementation accelerates |
| W11 | 2026-03-23 → 03-29 | 209 | Dataverse + RAG + output-schema-contracts |
| W12 | 2026-03-30 → 04-05 | 157 | RC-4.1 (RAG ingestion) → RC-5 cut on Apr 3 |
| W13 | 2026-04-06 → 04-12 | 129 | RC-5 settling; exception-hygiene merge |
| W14 | 2026-04-13 → 04-19 | 124 | Composer hardening; doc refresh sweep |
| W15 | 2026-04-20 → 04-26 | 85 | Statistical batch plugin design |
| W16 | 2026-04-27 → 05-03 | 123 | Composer skill-pack updates; Phase plans |
| W17 | 2026-05-04 → 05-10 | 308 | RC-5.1 closing sprint (composer correctness); Phase 1B persistence |
| W18 | 2026-05-11 → 05-17 | 486 | RC-5.1 release (May 11); RC-5.2 release (May 14); progress-persistence Phases 1–4 |
| W19 (partial) | 2026-05-18 → 05-19 | 188 | Phase 6 / 7 / 8 polish |

---

## Velocity Observations

### Bimodal distribution

The histogram of active days is bimodal:

- **31 days at ≥ 50 commits** — these correspond to RC cutovers, multi-PR merge days, and bug-burndown sprints. Median commit on these days is a one-line code change or a test addition; the volume comes from many small commits in series.
- **92 days at < 50 commits** (median across active days is 27) — architectural and correctness work, where each commit tends to be a focused, reviewed change.

### Idle days

Only **5 calendar days** out of 128 had zero commits:

- 2026-02-23 (post-RC-3.2 quiet)
- 2026-03-11 (post-RC-3.4 break)
- 2026-03-16, 2026-03-17 (mid-March pause before RC-4 acceleration)
- 2026-04-06 (post-RC-5-cut break)

Idle days cluster immediately after a major release. The pattern is *sprint → release → 1–2 day pause → next-phase planning → next-phase implementation*. The longest idle stretch was the 2-day pause around 16–17 March 2026.

### Recovery from low-activity weeks

The lowest-activity calendar week was W06 (Feb 16–22, 57 commits). It was immediately followed by W07's RC-3.3 kickoff (130 commits) — a 2.3× rebound. The pattern repeats around W09 → W10 → W11 (53 → 107 → 209 commits over three weeks as RC-4 ramped up). Low-activity weeks reliably precede a phase change rather than indicate sustained slowdown.

### Tempo at the end of the period

The last seven days (May 13–19) have averaged **71 commits / day**; the prior seven (May 6–12) averaged **75 commits / day**. Together May 6–19 holds at **73 commits / day** — the highest sustained 14-day tempo since the Pre-RC1 run (Jan 16–22 averaged **81 commits / day** at the project's heaviest week). RC-5.2's Composer Phase 1–8 program is being delivered at near-Pre-RC1 cadence.

---

## How These Numbers Were Computed

```bash
# Active-day per-day commits across the three RC history snapshots
git log <rc1-history> <rc2-history> <rc5.2-history> \
    --format='%ad' --date=short \
  | sort | uniq -c

# Unique commit count across all three snapshots
git log <rc1-history> <rc2-history> <rc5.2-history> --format='%h' \
  | sort -u | wc -l
```

Output: 123 active-commit days, 4,521 unique commits.

Peak-day attribution is taken from the dominant `feat(...)` / `add(...)` / `implement(...)` commit cluster on each day, validated against `CHANGELOG.md`, archived release snapshots, and Git history for cross-reference.

---

## Sources

- `CHANGELOG.md`
- Git history for deleted RC-1 and RC-2 changelog snapshots
- Git history snapshots for RC-1, RC-2, RC-5.2, and `main`
- `docs/release/elspeth-progress-rc1-to-rc5.md` — companion cumulative-output document
