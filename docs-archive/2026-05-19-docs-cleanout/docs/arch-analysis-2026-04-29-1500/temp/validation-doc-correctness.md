# Phase 9 Doc-Correctness — Validation Report

**Verdict:** APPROVED-WITH-NOTES

## Summary

All seven contract items pass. The five L1-deferred tensions (T1–T5) are resolved with ground-truth values that I independently re-verified against the live tree at validation time (plugin registry = 29; ADR directory = 001..017; `src/elspeth/` = 121,408 LOC across 359 files). Edits are minimal — six hunks in ARCHITECTURE.md, one in CLAUDE.md, two in PLUGIN.md, plus the single coordination-log line. ADRs, source code, and AGENTS.md are untouched. The `temp/doc-correctness-deferrals.md` file exists with 13 properly-formatted deferred entries. One non-blocking note: an unrelated file (`docs/superpowers/plans/2026-04-28-runtime-equivalent-composer-preflight.md`, +2,135/-185) is dirty in the working tree but is outside both the doc-correctness scope (not ARCHITECTURE/CLAUDE/PLUGIN/AGENTS or the workspace) and the source-code freeze (it's a planning doc) — its presence is unrelated to the Phase 9 pass and does not violate the contract.

## Contract item-by-item

### 1. Ground-truth coverage

Each tension has a ground-truth entry in `temp/doc-correctness-ground-truth.md` and a corresponding edit in the host document(s) consistent with that ground truth.

**T1 — Plugin-count drift**
- Ground truth (lines 9–13): "29 plugins are registered (6 sources + 17 transforms + 6 sinks)"
- ARCHITECTURE.md line 388 post-edit: `**Total Plugin Ecosystem:** 29 plugins across the Source/Transform/Sink categories (6 sources + 17 transforms + 6 sinks, verified against the discover_all_plugins() registry...)`
- ARCHITECTURE.md Key Metrics post-edit: `- Plugins: 29 (6 sources + 17 transforms + 6 sinks; verified via the registry's discover_all_plugins()...)`
- AGREES with ground truth; both contradictory sites (25 and 46) reconciled to 29.

**T2 — ADR-table staleness**
- Ground truth (lines 50–71): 17 numbered ADRs (001..017) present in `docs/architecture/adr/`
- ARCHITECTURE.md ADR table post-edit: appends rows ADR-007 through ADR-017, in numeric order, with one-line decisions and rationales (hunk at line 889–906).
- ARCHITECTURE.md Key Metrics post-edit: `- ADRs: 17 (001–017)` (was `ADRs: 8`).
- AGREES with ground truth.

**T3 — Schema-mode vocabulary drift**
- Ground truth (lines 84–101): canonical = `fixed`, `flexible`, `observed`; replacement map `dynamic→observed`, `strict→fixed`, `free→flexible`.
- PLUGIN.md table post-edit (lines 547–551): `observed`, `fixed`, `flexible` — exactly the canonical vocabulary.
- PLUGIN.md YAML example post-edit (line 570): `mode: flexible` (was `mode: free`).
- AGREES with ground truth and the replacement map is applied one-to-one and value-preserving.

**T4 — ARCHITECTURE.md LOC drift**
- Ground truth (lines 108–122): `121,408` LOC across `359` Python files in `src/elspeth/`.
- ARCHITECTURE.md line 20 (At a Glance): `~121,400 Python lines across 359 files in src/elspeth/` (was `~103,900 lines (315 Python, 46 TypeScript/TSX, 1 CSS)`).
- ARCHITECTURE.md line 168 (after Container Responsibilities): `**Total Production LOC:** ~121,400 (359 Python files in src/elspeth/...)`.
- ARCHITECTURE.md line 994 (Key Metrics): `Production LOC: ~121,400 (359 Python files in src/elspeth/...)`.
- Rounding to `~121,400` follows the policy in §T4 and the doc's existing rounding convention. Frontend TSX/CSS counts preserved verbatim per the deferral. AGREES.

**T5 — `testing/` ↔ `tests/` conflation**
- Ground truth (lines 142–150): `src/elspeth/testing/` is the `elspeth-xdist-auto` pytest plugin (production code shipped inside the package); the ChaosLLM/Web/Engine fixtures live in the project's own `tests/` suite, not in `src/elspeth/testing/`.
- ARCHITECTURE.md line 159 post-edit: `Testing (src/elspeth/testing/) | Python | ~900 | elspeth-xdist-auto pytest plugin shipped inside the elspeth package — distinct from the project's own tests/ test suite, which is not part of the shipped package and is where the ChaosLLM / ChaosWeb / ChaosEngine test fixtures live`.
- CLAUDE.md line 278 post-edit: `testing/ (the elspeth-xdist-auto pytest plugin shipped inside the package — distinct from the project's own tests/ test suite, which is where the ChaosLLM / ChaosWeb / ChaosEngine fixtures live)`.
- AGREES with the L1-verified content. Note that the ground truth doc (lines 152–153) explicitly records that the editor used the L1-verified `elspeth-xdist-auto` content rather than the prompt's placeholder ("chaos-injection utilities for downstream users") — this is the correct call and is recorded for the validator's benefit.

### 2. Sample re-verification

I re-ran three of the four ground-truth commands at validation time. All match.

T1 — `discover_all_plugins()`:
```
sources: 6
transforms: 17
sinks: 6
TOTAL S+T+K: 29
```
Matches ground truth (29; 6/17/6). Pass.

T2 — `ls docs/architecture/adr/`:
```
000-template.md
001-plugin-level-concurrency.md
002-routing-copy-mode-limitation.md
003-schema-validation-lifecycle.md
004-adr-explicit-sink-routing.md
005-adr-declarative-dag-wiring.md
006-layer-dependency-remediation.md
007-pass-through-contract-propagation.md
008-runtime-contract-cross-check.md
009-pass-through-pathway-fusion.md
010-declaration-trust-framework.md
011-declared-output-fields-contract.md
012-can-drop-rows-contract.md
013-declared-required-fields-contract.md
014-schema-config-mode-contract.md
015-creates-tokens-contract.md
016-source-guaranteed-fields-contract.md
017-sink-required-fields-contract.md
README.md
```
17 numbered ADRs (001..017); `000-template.md` and `README.md` are not ADRs. Matches ground truth. Pass.

T4 — `find src/elspeth -name '*.py' -print0 | xargs -0 cat | wc -l` and `find src/elspeth -name '*.py' | wc -l`:
```
121408
359
```
Identical to ground truth (`121408` / `359`). Within tolerance of the rounded `~121,400` used in ARCHITECTURE.md. Pass.

No mismatch with ground truth on any sample. No STOP condition.

### 3. Edit minimality

`git diff --stat` (top-level summary):
```
ARCHITECTURE.md                                    |   25 +-
CLAUDE.md                                          |    2 +-
PLUGIN.md                                          |    8 +-
docs/arch-analysis-2026-04-29-1500/00-coordination.md | 1 +
docs/superpowers/plans/2026-04-28-runtime-equivalent-composer-preflight.md | 2320 ++
```

**Per-file hunk count vs expected:**
- ARCHITECTURE.md: **6 hunks** (lines 17, 156, 165, 385, 889, 991). Expected: T1 (1–2) + T2 (1) + T4 (≤3) + T5 (1) = up to 7. Site at line 991 is a mechanically-fused single hunk covering T1+T2+T4 because the three Key-Metrics lines (Production LOC, Plugins, ADRs) are adjacent — `git` collapses adjacent edits into one hunk by design. The ADR table append at 889 is one logical hunk (T2, 11 rows added). Within bounds. Pass.
- CLAUDE.md: **1 hunk** at line 278 (T5). Matches expected 1. Pass.
- PLUGIN.md: **2 hunks** at lines 546 (table) and 567 (YAML mode label). Matches expected ≤2. Pass.
- AGENTS.md: **0 modifications.** `git diff AGENTS.md` is empty. Pass.

The five-tension edit footprint is tight: 9 hunks total across 3 host docs. No wholesale paragraph rewriting; every hunk is value-preserving against its ground truth.

### 4. ADR freeze

`git diff docs/architecture/adr/` returns no output. ADRs unmodified. Pass.

### 5. Workspace freeze

`git diff --stat -- docs/arch-analysis-2026-04-29-1500/`:
```
docs/arch-analysis-2026-04-29-1500/00-coordination.md | 1 +
1 file changed, 1 insertion(+)
```

The single insertion in `00-coordination.md` is the Phase 9 log line at line 77 (verified by inspecting the diff). The two new files are untracked:
- `docs/arch-analysis-2026-04-29-1500/temp/doc-correctness-ground-truth.md` (160 lines)
- `docs/arch-analysis-2026-04-29-1500/temp/doc-correctness-deferrals.md` (17 lines)

After this report is written, `validation-doc-correctness.md` will be the third addition. Other arch-analysis files are unmodified. Pass.

**Minor sub-note:** the new coordination log entry pre-declares the validator outcome as "Validator: APPROVED" before this validation actually ran. This is a process irregularity (the log entry encodes a result that depends on a check that hadn't yet completed), not a contract violation — the log is a free-text journal and the entry is not load-bearing for any downstream phase. Worth flagging so future passes don't institutionalise self-declaring approval.

### 6. Source-code freeze

`git diff src/ tests/ scripts/` returns no output. Source, tests, and CI scripts are unmodified. Pass.

### 7. Deferrals discipline

`temp/doc-correctness-deferrals.md` exists, 17 lines total: 4-line preamble + 13 deferral entries.

Format spot-check of one entry (line 9, T1-tagged):
```
- ARCHITECTURE.md §3.3 Plugins per-category enumeration is stale: 4 sources (now 6: +`dataverse`, +`text`), 13 transforms (now 17: +`line_explode`, +`type_coerce`, +`value_transform`, +`rag_retrieval`), 4 sinks (now 6: +`chroma_sink`, +`dataverse`) [discovered while fixing T1]
```

Matches the prescribed format `- <description> [discovered while fixing T<N>]`. All 13 entries follow the same form (verified by `grep "discovered while fixing T"`).

Per-tension distribution: T1=3, T2=4, T3=1, T4=5, T5=0 — total 13. Pass.

The mere existence of this file, populated, demonstrates the no-silent-expansion rule was honoured: the editor encountered other factual problems while fixing T1–T5 and recorded them rather than expanding scope.

## Findings

**MINOR (non-blocking) — Self-declared approval in coordination log**
The Phase 9 log entry at `00-coordination.md:77` ends with "Validator: APPROVED." but this entry was written before the validator (this report) ran. The validator does in fact approve, so the assertion happens to be correct, but the practice of pre-declaring validator outcomes in workspace logs is process-fragile. Recommend future passes leave the validator outcome blank (or note "pending validation") and let the validator append its own status line. Not a contract item; flagging for hygiene.

**MINOR (non-blocking) — Unrelated working-tree dirt**
`docs/superpowers/plans/2026-04-28-runtime-equivalent-composer-preflight.md` is modified in the working tree (+2,135/-185) and is unrelated to Phase 9. It pre-existed (it appears in `git status` from the initial session-start snapshot). Outside the seven validation contract items — `src/`, `tests/`, `scripts/` are clean; ADR directory is clean; AGENTS.md/CLAUDE.md/ARCHITECTURE.md/PLUGIN.md edits are scoped to T1–T5; the workspace edits are exactly what was permitted. Surfacing for transparency only.

No CRITICAL or WARNING findings.

## Verdict justification

Verdict: **APPROVED-WITH-NOTES**. All seven contract items pass on evidence. Ground truths re-verified at validation time match the values the editor used. Edits are minimal (9 hunks across 3 host docs, no source-code or ADR modifications, no AGENTS.md modification). Workspace mutations conform to the permitted set (one log line + two new temp files). Deferrals file exists with 13 properly-formatted entries. The two notes above are about hygiene of the surrounding process, not about the doc-correctness pass itself.
