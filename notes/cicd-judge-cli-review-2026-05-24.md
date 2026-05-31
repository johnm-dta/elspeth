# cicd-judge-cli prototype — multi-agent review consolidation

**Date:** 2026-05-24
**Branch:** `feat/cicd-judge-cli-prototype`
**Worktree:** `/home/john/elspeth/.worktrees/cicd-judge-cli`
**Scope:** Full code review of the prototype as it would be merged (committed work + uncommitted C1/C3 working-tree additions).
**Method:** Codebase split into 8 chunks (≤2000 LOC each, whole-file aligned). Each chunk reviewed by a pruned 7–9 agent pack (LLM-safety, LLM-diagnostician, FP-analyst, rule-designer, threat-analyst, python-code-reviewer, silent-failure-hunter, QA-analyst, code-reviewer, plus pipeline-reviewer/doc-critic for Chunks 7–8). 57 unique agent invocations total. Diff-only review for modified files; full review for new files.

## How to read this document

- **§1 TL;DR + Top-10 priority list** — what to fix first.
- **§2 Cross-cutting CRITICAL themes** — patterns that recurred across chunks. These are the load-bearing concerns, not the per-finding noise.
- **§3 Per-chunk consolidated findings** — full per-chunk severity-ranked finding lists, in the form delivered during the review.
- **§4 What's strong** — validated-by-multiple-agents positive properties; don't break these.
- **§5 Higher-order recommendations** — not single-fix items; structural choices the operator should evaluate.
- **§6 Status of the uncommitted work** — what's in the worktree but not committed.
- **§7 Raw transcripts** — pointer to the 57 sub-agent JSONL files on disk.

Throughout: file paths are absolute (`/home/john/elspeth/.worktrees/cicd-judge-cli/...`). Severity tags: CRITICAL (pre-merge blocker), MAJOR (should fix), MINOR (worth fixing), OBSERVATION (worth noting).

---

# §1 TL;DR + Top-10 pre-merge priority

The prototype's two pillars (judge gate + `@trust_boundary` decorator) demonstrably advance the state of the art over the current ~700-entry allowlist pathology. But the review surfaced **three recurrent classes of structural concern**:

1. **Checks of *presence* presented as checks of *authenticity*** — judge metadata, decorator metadata, override verdict, test_ref all fall into this trap.
2. **The audit primitive lacks integrity binding to the code it suppresses** — forgeable plain-text YAML, transplantable quartets, free-text invariants, no hash chain.
3. **Documentation overstates what mechanisms actually enforce** — plan note doctrine overstatements, decorator docstrings stronger than rules, CLAUDE.md updated for `rotate` but not for `justify`/`reaudit`.

These are the prototype's load-bearing risks, not individual code defects.

## Top-10 pre-merge action priority

| # | Action | Chunks | Notes |
|---|--------|--------|-------|
| 1 | Fix `completion.model` audit bug at `judge.py:712` | 1, 3 | 2-line fix (`model_id=completion.model or model_id`). Silent on happy path, structurally broken on any OpenRouter fallback. Audit YAML records requested-not-served model. |
| 2 | Add `temperature=0` to `judge.py:680-684` | 1, 2, 3 | 1-line fix. Without it, every reaudit sweep produces phantom `WAS_ACCEPTED_NOW_BLOCKED` divergences from sampling noise. Undermines the calibration instrument. |
| 3 | Wire `--rule` flag in `_run_justify` or delete it | 2 | `cli.py:154-159` registers `--rule`; `_run_justify` never reads `args.rule`. Audit field is whatever the scanner picked. Fabricated audit field — the exact anti-pattern the project's fabrication-decision test exists to prevent. |
| 4 | Add `ast.ClassDef` to `ast_walker.py:67` short-circuit + regression tests | 5 | 4 agents empirically reproduced: `class Helper: raw = arguments["x"]` inside a decorated function silently taints outer `raw`. Same defect class as B1 inner-scope-leak, just for class bodies. |
| 5 | Catch `yaml.YAMLError` in C1/C3 gates | 7 | Both modules catch only `ValueError`; `yaml.YAMLError` subclasses `Exception`. Malformed YAML produces uncaught traceback → exit 1 instead of documented exit-2. CI cannot distinguish "gate broken" from "gate fired". |
| 6 | Reject `JudgeVerdict.BLOCKED` at allowlist load OR fix the docstring | 8 | Loader accepts `BLOCKED` despite docstring saying "in-memory only". Plus `reaudit.py:72` explicitly handles `(BLOCKED, None, BLOCKED) → STILL_AGREES`. Self-contradictory. |
| 7 | Make `_append_entry_to_yaml` (cli.py) and `apply_plan` (rotate.py) atomic + locked | 2, 4 | Both flagged CRITICAL by 4+ agents per chunk. Same pattern: read_text → mutate → write_text. SIGTERM/disk-full mid-write corrupts Tier-1 audit data; concurrent writers race. Pattern: temp+fsync+`os.replace`+`fcntl.flock`. |
| 8 | Per-entry failure isolation in `reaudit_entries` | 3 | `_reaudit_one_entry:401` has no try/except around `call_judge`. One transient failure at entry 423 of 700 discards all 422 prior outcomes. Add `JUDGE_CALL_FAILED` divergence; emit `entries_dispatched` in report. |
| 9 | Add CLI integration tests for the +257 LOC `_run_check_judge_coverage` / `_run_check_override_rate` delta | 7 | Library is well-tested; gate-as-deployed is zero-tested. Classic VER-without-VAL gap. The whole prototype's enforcement story rests on these subcommands. |
| 10 | Fix Pillar B section header in `notes/cicd-judge-cli-prototype-plan.md` + delete duplicated block in `notes/new_ci_remediation.md` | 8 | Plan says decorator both "ships" (Status, table) and is "future work, not in prototype" (Pillar B header). Remediation note duplicates M1-N1-N2 block verbatim at lines 37-105 and 108-212. |

## Status: 8 of 8 chunks reviewed

| # | Chunk | LOC | Pack | Status |
|---|-------|-----|------|--------|
| 1 | Judge core + test_justify | 1800 | 8 agents | ✅ |
| 2 | CLI surface + integration tests | 1602 | 8 agents | ✅ |
| 3 | Reaudit + tests | 1857 | 8 agents | ✅ |
| 4 | Rotate + tests | 915 | 6 agents | ✅ |
| 5 | `@trust_boundary` decorator + tier-model integration | 1821 | 7 agents | ✅ |
| 6 | Three honesty-gate rules + shared | 1785 | 7 agents | ✅ |
| 7 | C1/C3 gates + workflow (UNCOMMITTED) | 1943 | 7 agents | ✅ |
| 8 | C2 multi-rule reaudit + allowlist + plan + CI | 1537 | 8 agents | ✅ |

---

# §2 Cross-cutting CRITICAL themes

These are the patterns that recurred across chunks. Each theme has 2–4 underlying findings; addressing the theme is higher-leverage than addressing individual findings.

| Theme | Chunks | Highest-leverage fix |
|-------|--------|----------------------|
| **Judge audit-trail integrity = presence, not authenticity** | 1, 2, 7, 8 | Hash-chain or HMAC-sign judge metadata; bind quartet to fingerprint + ast_path |
| **Non-determinism (no `temperature=0`)** | 1, 2, 3 | 1-line fix in `judge.py:680-684` |
| **`completion.model` never read → audit lies on fallback** | 1, 3 | 2-line fix in `judge.py:712` |
| **`--operator-override` has zero auth** | 1, 2 | env-var token / OIDC / git-signer binding |
| **Source/secrets exfiltration via `surrounding_code`** | 1, 2, 3 | secrets-scrubber + `is_relative_to(root)` guard; AMPLIFIED by reaudit's bulk sweep |
| **`--rule` flag is dead code** | 2 | wire it in or delete it |
| **YAML writer non-atomic, unlocked** | 2, 4 | temp+rename+fsync+flock; single helper |
| **Per-entry failure aborts entire reaudit sweep** | 3 | wrap call_judge per-entry; add `JUDGE_CALL_FAILED` divergence; surface `entries_dispatched` |
| **Override-laundering via rotate** | 4 | strip judge metadata on fingerprint rotation, OR refuse to rotate override-bearing entries |
| **Auto-stub regression cleanly removed but NOT pinned by test** | 4 | one negative-assertion test |
| **`ast_walker.py:67` ClassDef + comprehensions not short-circuited** | 5 | 4 agents empirically confirmed; one-line fix + regression tests |
| **C5-2 not closed: rule checks raising-shape, not invariant-liveness** | 5, 6 | extend `trust_boundary.tests` to verify decorated symbol appears in test body |
| **No allowlist for "closed invariant" rules — first FP blocks CI** | 6 | document escape protocol OR add narrow attested exemption |
| **C1 + C3 check presence not authenticity; C3 is one-sided gauge** | 7, 8 | hash-chain; add under-override instrument |
| **`yaml.YAMLError` not caught → uncaught traceback** | 7 | catch `(yaml.YAMLError, ValueError)` in both gates |
| **`JudgeVerdict.BLOCKED` round-trips through loader despite docstring** | 8 | one validator line OR fix the docstring |
| **Workflow concurrency gap on enforce-allowlist-judge-gates.yaml** | 7, 8 | mirror ci.yaml's concurrency block |
| **Plan/doc doctrine overstatements** | 8 | rewrite Pillar B header + soften 2 doctrine claims; delete duplicate block |

---

# §3 Per-chunk consolidated findings

## §3.1 Chunk 1 — Judge core + test_justify (1800 LOC)

**Files:** `core/judge.py` (837), `tests/unit/elspeth_lints/test_justify.py` (963).
**Pack:** llm-safety, llm-diagnostician, FP-analyst, threat-analyst, python-code-reviewer, silent-failure-hunter, QA-analyst, code-reviewer (8).

### CRITICAL
- **C1-1: `completion.model` never read → audit records requested model not served model.** `judge.py:712` writes `model_id=model_id` (function parameter); `judge.py:686` discards `completion.model`. On any OpenRouter fallback, YAML records `"anthropic/claude-opus-4"` but a different model rendered the verdict. Invisible in tests because `MagicMock().model` is a MagicMock. Solo finding, grep-verified by llm-diagnostician.
- **C1-2: Direct YAML edit / `--operator-override` self-authorisation / source exfiltration trio.** (a) No hash chain on `judge_rationale`; hand-write the YAML entry to bypass. (b) `--operator-override` is unauthenticated CLI flag — any agent self-grants `OVERRIDDEN_BY_OPERATOR`. (c) `surrounding_code` (±15 lines) ships verbatim to OpenRouter; ELSPETH had no prior outbound exfil channel. 4-agent convergence (threat T-001/T-002/T-003, llm-safety, FP-analyst).
- **C1-3: VAL gap — no judge-quality measurement.** 963 LOC of tests, all VER (parser-validates-parser). No labelled corpus, no red-team test, no gated real-LLM smoke. Mock shape derived from parser's expectations — tautology. (QA-analyst, FP-analyst — 2 agents)

### MAJOR convergent
- **M1-1: Model slug ambiguity.** `judge.py:53` default `anthropic/claude-opus-4` may float-alias on OpenRouter; plan note specifies `claude-opus-4-7`. Audit-reproducibility risk. (code-reviewer #1, python-code-reviewer M1, llm-diagnostician)
- **M1-2: Prompt injection on rationale + surrounding_code.** Only `---` delimiters; concrete jailbreak example provided. (llm-safety MAJOR, threat T-010)
- **M1-3: Policy/CLAUDE.md drift.** ~400 lines of CLAUDE.md inlined as `_STATIC_POLICY_BLOCK`. No `policy_hash` in `JudgeResponse`/YAML. Verdicts un-interpretable in 6 months if CLAUDE.md changes. (4 agents)
- **M1-4: Fail-closed regression tests absent.** Zero coverage of `httpx.ConnectError`, 401, 429, `finish_reason=length` truncation. (silent-failure M2, QA-analyst)
- **M1-5: `ACCEPTED + should_use_decorator!=None` incoherence-crash untested.** (silent-failure M1, QA-analyst, python-code-reviewer)

### MAJOR unique
- U1-1: `cast(Any, messages)` at `judge.py:683` too broad (code-reviewer #2).
- U1-2: `OPENAI_BASE_URL` / `HTTPS_PROXY` env can redirect SDK; not pinned with `trust_env=False` (threat T-007).
- U1-3: No max-size cap on `surrounding_code` → token-bill DoS (threat T-009).
- U1-4: `max_tokens=1024` + no `finish_reason="length"` check → confusing "non-JSON response" error (llm-diagnostician F2).
- U1-5: `JudgeRequest` has no `rationale_duplicate_count`/`similar_entries` — judge cannot recommend consolidation, structurally impossible (FP M1).
- U1-6: ±15-line `surrounding_code` window too narrow for tier_model rule's real FP patterns; loses decorator stacks on long methods (FP M3).
- U1-7: Output schema has no `confidence` field (FP M4).
- U1-8: Prompt has strong implicit BLOCKED prior (6+ "lean toward BLOCKED" terminals, 1 ACCEPT example), undeclared and unmeasured (FP C2).
- U1-9: `JudgeResponse.verdict` docstring claim untested for `OVERRIDDEN_BY_OPERATOR` (code-reviewer #5).
- U1-10: Single `RuntimeError` for transport vs contract violations — CLI can't differentiate; suggested `JudgeContractError(RuntimeError)` subclass (code-reviewer #9).

### What's strong
- silent-failure-hunter found **no CRITICALs in transport**: no bare excepts, no `except Exception`, no `or 0` cache fabrication, no `--owner` fallback. Fail-closed contract structurally intact.
- Cache accounting `None`/`0`/`>0` per fabrication-decision test is **exemplary** (all 4 reviewers who looked at it).
- Output schema enforcement (verdict strict match, missing-field crash, cross-field invariant) is **exemplary offensive programming**.

---

## §3.2 Chunk 2 — CLI `justify` + integration tests (1602 LOC)

**Files:** `core/cli.py` (focus: `_run_justify` + helpers, ~916 LOC), `test_judge_decorator_integration.py` (427), `test_allowlist_judge_metadata_integrity.py` (259).
**Pack:** 8 agents (dropped rule-designer — no rules).

### CRITICAL
- **C2-1: `--operator-override` zero auth at `cli.py:187-195, 810-815`.** Test `test_judge_decorator_integration.py:379-427` demonstrates bypass as a feature. 4 agents converge (llm-safety, threat-analyst T-002, code-reviewer C1, FP-analyst).
- **C2-2: `--file-path` traversal at `cli.py:884-896`.** `_resolve_target_file` returns absolute paths unguarded. No `is_relative_to(root)`. Plus `_extract_surrounding_code` has no secrets scrubber. Ships `/etc/passwd`/`.ssh/id_rsa`/`.env` content to OpenRouter. 4 agents.
- **C2-3: `--rule` flag is dead code.** `cli.py:154-159` registers it; `_run_justify` never reads `args.rule`. `JudgeRequest.rule_id` comes from scanner. Operator types `--rule R7`, audit says different. llm-diagnostician CRITICAL #1, solo, grep-verified.
- **C2-4: `call_judge` sends no `temperature` → judge non-deterministic at OpenRouter default ~1.0.** Same input produces different verdicts; reaudit produces phantom decay. (llm-diagnostician CRITICAL #2)
- **C2-5: Write→load YAML round-trip untested.** Override test asserts text-substring presence; nothing asserts the loader accepts the writer's output for a Tier-1 audit primitive. (QA-analyst CRITICAL)
- **C2-6: `_yaml_inline_scalar` writes YAML 1.1 boolean/null sentinels unquoted at `cli.py:1015-1025`.** `--owner "yes"` → `owner: yes` → reloads as `True` (bool). Verified live by python-code-reviewer. Tier-1 corruption.

### MAJOR convergent
- **M2-1: `_append_entry_to_yaml` non-atomic + unlocked (cli.py:1028-1080).** 4 agents (silent-failure M2, threat C-001, code-reviewer M1, FP-analyst M2).
- **M2-2: Integration tests miss hostile inputs and edge cases.** No test for `--file-path` outside `--root`, `--owner` with embedded newlines, `--rationale` containing YAML directives, concurrent appenders, SIGKILL mid-write, ACCEPTED+override, etc. (3 agents)
- **M2-3: Unbounded `--rationale` + hardcoded `context_lines=15` duplicated as constants in both CLI and policy block.** Changing one silently lies to the model. (llm-safety + llm-diagnostician)

### MAJOR unique
- U2-1: CLI catches only `JudgeConfigurationError`; `RuntimeError` from `call_judge` propagates as raw traceback. CI can't distinguish "judge BLOCKED" from "judge crashed" by exit code alone (llm-diagnostician M3).
- U2-2: `max_tokens=1024` hardcoded; long rationale truncates mid-JSON → confusing "non-JSON response" (llm-diagnostician M4).
- U2-3: No similar-entry scan before judge call (FP C1) — commits Chunk-1 M1 rather than mitigating; the ~117-duplicate-rationale pathology will recur on day one.
- U2-4: No CLI-side falsifiability hook (FP C2).
- U2-5: No emitted metrics — BLOCKED/ACCEPTED/override counters not surfaced; override-rate gate must re-parse all YAMLs (FP M6).
- U2-6: Allowlist anomaly tests miss non-UTC timestamps, invalid enum values, empty-string `judge_model`/`judge_rationale` (QA-analyst).
- U2-7: File-system fidelity untested: no test for existing prior-entries no-clobber, empty-header-only file, mixed pre-judge + judge-era entries (QA-analyst).
- U2-8: `_append_entry_to_yaml` not idempotent — re-running justify after partial failure creates duplicate `key:` entries (code-reviewer M2).
- U2-9: `_yaml_inline_scalar` may underescape control characters (code-reviewer M3).
- U2-10: `--symbol` help misleading about uniqueness (code-reviewer M4).
- U2-11: `_parse_symbol` raises `ValueError` reachable from `_run_justify:756` without try/except → raw traceback (code-reviewer M5, python-code-reviewer M2).
- U2-12: `ACCEPTED + should_use_decorator` incoherence-crash not tested at CLI layer (code-reviewer M6, QA-analyst).
- U2-13: BLOCKED with no decorator suggestion gives operator no remediation surface; pushes them toward `--operator-override` (FP M3).

### MINOR / OBSERVATION
- `--rationale` empty/whitespace accepted (`cli.py:170-175` plain `type=str`; `--owner` uses `_non_empty_string`, `--rationale` should too).
- `_non_empty_string` returns unstripped value.
- 90-day expiry hardcoded magic number at `cli.py:994`.
- Lazy `json` import inside `_emit_justify_output` unmotivated.
- "Operator override:  yes/no" inconsistent spacing in text output.
- Hardcoded `"Suppression gated by cicd-judge..."` safety string at `cli.py:1001-1002` flagged as deferred follow-up.

### What's strong
- **Fail-closed property preserved in CLI.** silent-failure-hunter confirmed `_run_justify:791-795` catches only `JudgeConfigurationError`; no broad swallow.
- Allowlist-loader contract tests thorough on the 4 documented invariants.
- CLI integration tests use real argparse→main→_run_justify (not synthetic Namespaces).
- Layer imports clean; no logger misuse; no defensive `.get()`/`getattr()` abuse on internal typed data.

---

## §3.3 Chunk 3 — Reaudit + tests (1857 LOC)

**Files:** `core/reaudit.py` (1006 LOC including 308 uncommitted multi-rule expansion), `test_reaudit.py` (851).
**Pack:** 8 agents.

### CRITICAL
- **C3-1: Non-determinism amplifies divergence noise.** Carryover from Chunk 2 (no `temperature=0`) — reaudit is where the harm manifests. Every sweep emits phantom `WAS_ACCEPTED_NOW_BLOCKED` flips. `ReauditDivergence` enum has no NOISE bucket. (llm-diagnostician, FP C-R1/C-R2)
- **C3-2: Per-entry failure aborts entire sweep.** `_reaudit_one_entry:401` has no try/except around `call_judge`. One transient network failure at entry 423 of 700 discards all prior outcomes. No `JUDGE_CALL_FAILED` divergence value. (silent-failure C1)
- **C3-3: Sweep-level abort invisible.** `ReauditReport` lacks `entries_dispatched`; an incomplete sweep looks identical to a clean sweep. Header missing "INCOMPLETE SWEEP: N entries never reaudited" banner. (silent-failure C2)
- **C3-4: Path traversal via forged YAML key.** `_parse_entry_key` splits attacker-controllable key on `:`; `_extract_surrounding_code` reads `../../../etc/passwd` and ships to OpenRouter. AMPLIFIES Chunk 2 C-2B at bulk scale (700 entries → 700 exfiltrations in one sweep). (llm-safety M3, threat T-R002/T-R003)
- **C3-5: Divergence taxonomy is transition-shaped, not cause-shaped.** `(stored, fresh) → label` algebra; no policy-drift, code-drift, model-noise, or falsifiability axes. Reaudit's stated job is to detect drift between stored and fresh verdicts — but at temperature ~1.0, a `WAS_ACCEPTED_NOW_BLOCKED` is indistinguishable from a `STILL_AGREES` that flipped by noise. (FP C-R1/C-R3/C-R4)
- **C3-6: 17-of-17 multi-rule dispatch branches untested.** Every test passes `rule_filter="trust_tier.tier_model"`. Stale module path or renamed constant in any of the 15 non-tier_model branches silently fails until production. (python-code-reviewer C1, QA-analyst C1)

### MAJOR convergent
- **M3-1: Documented-but-unimplemented rule_filter cross-check.** `reaudit.py:240` claims silent skip of entries with rule outside `--rule`; not implemented. (python-code-reviewer + code-reviewer M1)
- **M3-2: Markdown reports render Tier-3 model rationale unescaped** → injection vehicle into PRs/wikis. `_md_escape` only escapes `\`, `|`, `\n`. Backticks, `<`, `>`, `[`, `]`, ANSI escapes pass through. (llm-safety M4, threat T-R007)
- **M3-3: `_extract_surrounding_code` is verbatim copy of cli.py version** → same path-traversal inherited (code-reviewer M3, llm-safety M3).
- **M3-4: Reaudit AMPLIFIES Chunk 2 model-id audit-lie issue.** Bulk report claims all entries judged by named model regardless of fallback. (code-reviewer o7)
- **M3-5: `_classify_divergence` raise paths abort whole sweep** (silent-failure M2).
- **M3-6: Allowlist loader exceptions surface as raw traceback** (silent-failure M3).
- **M3-7: CLI exit code 0 on any divergence — can't be gated in CI** (silent-failure M5, QA-analyst C2).
- **M3-8: `_find_matching_finding` picks first on duplicate canonical_key silently** (python-code-reviewer).

### MAJOR unique
- M3-9: Allowlist loader exceptions surface as raw traceback (silent-failure M3).
- M3-10: `_apply_filters` `since` ordering depends on entries being pre-sorted (code-reviewer M2).
- M3-11: `_lookup_module_attr` dispatch is safely closed today but brittle pattern (rule-designer observation, code-reviewer o1).
- M3-12: JSON renderer leaks `AllowlistEntry.matched` runtime field (code-reviewer m2).
- M3-13: Future-timestamp evasion of `--since` filter (threat T-R004).
- M3-14: Selective `--rule` scope leaves other rules unaudited (threat T-R005).
- M3-15: Single malformed entry doesn't abort the sweep BUT classification raises CAN abort — inconsistent (threat T-R006).
- M3-16: No rate limiting / cost telemetry on bulk Opus calls (threat T-R008).
- M3-17: No `_DIVERGENCE_ORDER` exhaustiveness check (python-code-reviewer observation).

### What's strong
- **Read-only on YAML confirmed across agents.** No `write_text` calls in reaudit.py. Module truly read-only.
- Dataclass hygiene clean (`frozen=True, slots=True`, only scalars/tuples).
- `_lookup_module_attr` correctly offensive (`getattr` without default).
- Divergence values orthogonal over `(stored × fresh)` space and exhaustive within that space.
- `_supported_rules()` correctly = BUILTIN_RULES − `_EXCLUDED_FROM_REAUDIT` (17 of 19).

---

## §3.4 Chunk 4 — Rotate + tests (915 LOC)

**Files:** `rules/trust_tier/tier_model/rotate.py` (523), `test_rotate_tier_model.py` (392).
**Pack:** 6 agents (dropped LLM-safety, LLM-diagnostician, rule-designer — no LLM calls, no rules).

### CRITICAL
- **C4-1: `apply_plan` non-atomic, partial-write corrupts Tier-1 allowlist.** `path.write_text:473` direct overwrite; mid-loop failure or SIGTERM leaves filesystem in mixed-rotated state with no rollback or inventory. ALL 5 reviewers flagged this (threat T-R001, silent-failure C2, QA C3, code-reviewer CRITICAL-1, python-code-reviewer MAJOR).
- **C4-2: E-R001/E-R002 override-laundering.** Rotate preserves judge metadata (including `OVERRIDDEN_BY_OPERATOR`) verbatim under new fingerprint via `text.replace`. Judge never sees new fingerprint as "new finding" because rotation already created it. **Rotate is by design "edit YAML without the judge gate"** — must strip judge metadata on rotation or refuse to rotate override-bearing entries. (threat-analyst solo but profound; pin before judge metadata fields propagate further)
- **C4-3: No round-trip writer→loader contract test for Tier-1 audit primitive.** `apply_plan` output never re-parsed in any test. (QA C1)
- **C4-4: No regression test pinning the "no auto-TODO-stub" doctrine.** 4 agents confirmed the regression is genuinely fixed in code, but nothing mechanically pins it. Future contributor can silently reintroduce. (QA C2)

### MAJOR convergent
- **M4-1: `remove_stale=True` is the destructive default** — silently deletes entries whose source file wasn't in the scan scope (FP §3, silent-failure §5).
- **M4-2: N:N symmetric pairing assumes uniform metadata, doesn't enforce.** Fingerprint-sorted pairing can swap `owner`/`reason`/`safety` between entries (FP §6, threat E-R003).
- **M4-3: `_finding_covered_by_per_file_rule` duplicates `_match_finding` semantics.** "Mirrors X" comment is a drift warning, not a justification. (silent-failure §4, code-reviewer MAJOR-1)
- **M4-4: `_remove_stale_entries` indent-heuristic mis-parses pathological YAML** (silent-failure §3, code-reviewer MAJOR-2, threat T-R003).
- **M4-5: No rotation audit trail** (threat R-R001) — git commit is only record; rotation in dirty worktree invisible.
- **M4-6: Per-file rule filter not applied in rotation/unchanged branches** → steady-state growth path (FP §5).
- **M4-7: TODO debt surfaced but `apply_plan` doesn't refuse to apply when stubs exist** (silent-failure §8).

### MAJOR unique
- M4-8: Substring-collision risk in `str.replace` at key only (threat T-R002).
- M4-9: `allowlist_dir` resolution branch not tested (python-code-reviewer).
- M4-10: Multi-line block-scalar fixture missing (python-code-reviewer + QA M2).
- M4-11: No rotation test on judge-era entry — THE test for E-R001 prevention if metadata stripping is fixed (python-code-reviewer).
- M4-12: Blank-line between entries consumed by stale-removal (python-code-reviewer minor).

### What's strong
- **Auto-stub regression genuinely removed** (4 agents independently verified).
- Dataclass hygiene clean (frozen+slots, only scalars/tuples).
- Layer placement correct (L3 → L1/L0).
- No defensive programming violations.
- Offensive error handling at duplicate/missing keys correct.
- Pure `scan_for_rotations` / `plan_rotations` separation enables corpus testing.

---

## §3.5 Chunk 5 — `@trust_boundary` decorator + tier-model integration (1821 LOC)

**Files:** `contracts/trust_boundary.py` (202), `trust_boundary_suppress.py` (519), `tier_model/rule.py` (+204 delta), `ast_walker.py` (62), `test_trust_boundary.py` (242), `test_tier_model_decorator_suppression.py` (592).
**Pack:** 7 agents (dropped LLM-safety, LLM-diagnostician).

### CRITICAL
- **C5-1: `ast_walker.py:67` ClassDef not short-circuited → silent over-suppression.** Same defect class as B1, opposite scope. 4 agents converge — rule-designer reproduced 4 patterns empirically; QA-analyst, silent-failure-hunter, and python-code-reviewer all independently identified it. python-code-reviewer ran code to reproduce: `class Helper: raw = arguments["x"]` inside a decorated function silently taints outer `raw`. One-line fix: add `ast.ClassDef` to short-circuit tuple, mirror B1 regression test.
- **C5-2: `test_ref` is unverified free text.** Decorator metadata says "this function raises X on Y; here's the test that proves it" — but neither the decorator nor any analyzed CI gate verifies the test exists or actually exercises the invariant. Chunk 6 reviewed whether the trust_boundary.tests/scope/tier rules close this gap (verdict: partially — see Chunk 6 C6-1).
- **C5-3: No PR-diff surfacing for new `@trust_boundary`.** YAML allowlist edits are visible because reviewers scrutinize the directory; decorators are sprinkled across `src/elspeth/**`. No CI surface flags "this PR adds N new @trust_boundary decorators" for review focus. Same stealth-bypass channel as direct YAML edit.

### MAJOR convergent
- **M5-1: Audit-trail downgrade: suppression leaves no record.** Operator cannot answer "how many R1/R5 findings did `@trust_boundary` silence and where?" (code-reviewer #2, rule-designer, FP M2 — 3 agents)
- **M5-2: Nested-function boundary inheritance INCONSISTENT with dataflow walker's own scope-stop rules.** `_current_boundary` walks outer boundary into nested helpers, but `compute_derived_names` deliberately stops at function boundaries. Result: under-suppression in the case the comment claims to address. (code-reviewer #3, silent-failure M2 — 2 agents)
- **M5-3: Pillar B success metric structurally unverifiable.** No counters for decorators-seen, findings-suppressed-per-decorator. Plan's "M ≪ N" target unmeasurable. (FP M1)

### MAJOR unique
- M5-4: Symmetric shadowing leak — inner-scope rebinds of `source_param` (e.g. `def inner(arguments): arguments.get(...)`) silently inherit suppression. 4 patterns empirically tested by rule-designer.
- M5-5: `_BoundaryRule` Literal in L0 mirrors `_ALLOWED_BOUNDARY_RULES` in L3 without cross-reference (code-reviewer #9).
- M5-6: 7 of 12 dataflow propagation branches in `compute_derived_names` have zero test coverage (QA MISSING-3).
- M5-7: 6 of 14 diagnostic paths in `extract_boundary_metadata` untested (QA MISSING-5).
- M5-8: `_handler_is_silent` `ast.walk` descends into nested function bodies — same scope-leak shape as B1 but for R6 detection (silent-failure M3).
- M5-9: `_BoundaryMetadata` is `_`-prefixed but imported across packages by `rule.py` (code-reviewer #1).
- M5-10: Unknown decorator kwargs silently accepted into parsed dict (rule-designer OBSERVATION).
- M5-11: `compute_derived_names` 32-iteration cap is arbitrary; provably bounded by `len(local_names)` (rule-designer minor).
- M5-12: Stacked `@trust_boundary` decorators allowed; analyzer reads first, runtime overwrites — asymmetry (rule-designer minor, threat MINOR-E2).

### What's strong
- **B1 fix correctly prevents the originally-caught inner-scope-leak direction** (rule-designer empirically confirmed).
- Closed-set rules `R_TB_NONLITERAL` / `R_TB_MALFORMED` work correctly and emit findings.
- Decoration-time validation correctly raises `TypeError` on tier≠3, source_param-not-in-signature.
- Lattice escape on malformed metadata correctly fail-closed (silent-failure O1).
- 32-iteration cap correctly raises `RuntimeError` rather than truncating.
- L0 layer compliance: `trust_boundary.py` imports nothing above.
- CI wiring correctly includes the closed-set rules under `--rules trust_tier.tier_model` (silent-failure O1 corrected earlier threat-analyst misread).

---

## §3.6 Chunk 6 — Three honesty-gate rules + shared (1785 LOC)

**Files:** `rules/trust_boundary/shared.py` (190), `scope/rule.py` (209), `tier/rule.py` (116), `tests/rule.py` (286) + their tests + metadata files.
**Pack:** 7 agents.

### CRITICAL
- **C6-1: C5-2 remains open.** TBE3 verifies *some* `pytest.raises` exists in the named test body but doesn't verify: (a) test calls the decorated function, (b) raised exception matches the invariant type, (c) assertion targets the malformed-input path. 6 agents converged. The `invariant` and `source` decorator fields remain entirely free-text and uncovered.
- **C6-2: Scope rule satisfied by bare-expression dead read.** `scope/rule.py:_body_reads_name` accepts any `ast.Name(Load)` reference, including `_ = source_param` or `arguments  # noqa: B018`. Empirically tested by threat-analyst. (threat TB-06, silent-failure TB-M4)
- **C6-3: `_walk_statements` in `tests/rule.py:245` uses `ast.walk` → scope-leak.** Same bug just patched in `scope/rule.py`'s `_iter_own_scope`. Means TBE3 passes when raising-assertion is in a nested helper. Direct contradiction of the rule's documented strictness. (python-code-reviewer MAJOR)
- **C6-4: Cross-rule bypass via non-literal kwarg.** All three rules silently skip when `extract_keywords` returns `None`, deferring to `trust_tier.tier_model`. If tier_model is disabled/suppressed for any file, all three honesty rules pass. (threat TB-02)
- **C6-5: No allowlist mechanism + first legitimate FP blocks CI with no escape valve.** Plan claims "closed invariants" — FP-analyst identified concrete FP patterns (closure-pattern source_param, `**kwargs`-forwarded boundary, renamed test files). When the first one fires, options are: refactor (often Procrustean), disable the rule entirely (loses all coverage), or stop. No documented decision protocol. (FP CRITICAL ARCHITECTURAL)
- **C6-6: I/O errors crash the lint run.** `parse_python_file` has no `except` around `read_text` → `UnicodeDecodeError`/`PermissionError`/`OSError` kills entire scan. (silent-failure TB-C1)

### MAJOR convergent
- M6-1: Path-traversal in `_resolve_test_ref` — no `is_relative_to(repo_root)` guard (threat TB-03, code-reviewer #1).
- M6-2: DoS via unbounded `read_text` on attacker-chosen `test_ref` target (threat TB-04).
- M6-3: `_is_raising_call` matches any `*.raises` attribute too broadly — `mock.raises`, `sqlalchemy.exc.raises`, etc. all pass (threat TB-05, code-reviewer #4).
- M6-4: `_make_finding` reimplemented in all three rules → fingerprint-drift risk (code-reviewer #5).
- M6-5: NOTFOUND conflates 3 failure modes (file missing, parse error, function missing) (silent-failure TB-M3).
- M6-6: `repository_root()` heuristic fragile to non-canonical CI invocations (rule-designer).
- M6-7: Pytest parametrize-id format `test_func[case-1]` silently NOTFOUND (silent-failure TB-M1).
- M6-8: TBS2 + TBE2/TBE3 have no fixture files; metadata `examples_violation_count=1` mismatches reality (python-code-reviewer, rule-designer).
- M6-9: Cross-rule interaction test missing (QA M1).

### MAJOR unique
- M6-10: Multiple `@trust_boundary` stack silently masks duplicates (QA C2, rule-designer).
- M6-11: 6 mechanically-unprevented decorator-attestation lies enumerated by FP-analyst:
  1. `invariant` text lies — test exists, raises something, but not what invariant string claims.
  2. `source` text lies — function isn't on any external-data path; reviewer-only signal.
  3. `test_ref` semantic lies — test asserts raising on irrelevant input.
  4. Helper-indirected raising assertions accepted by design.
  5. Wrong `source_param` that happens to be a real read parameter.
  6. Rename-induced staleness — test_ref resolves to a function no longer exercising the path.

### What's strong
- **`tier` rule is genuinely tight** — explicit bool rejection, integer-literal check, no allowlist surface (multiple agents confirm).
- All three rules correctly wired in `BUILTIN_RULES` (`rules/__init__.py:40-42`) and invoked by `.github/workflows/ci.yaml:160`.
- Scope rule's `_iter_own_scope` correctly scope-respecting at function/lambda boundaries (the one that ISN'T leaking — contrast with `_walk_statements` C6-3).
- Division of responsibility with `tier_model.R_TB_*` documented and consistent — when tier_model is enabled.
- Negative-agreement coverage adequate for all three rules.

---

## §3.7 Chunk 7 — C1/C3 gates + workflow (1943 LOC, UNCOMMITTED)

**Files:** `core/judge_coverage.py` (429), `core/override_rate.py` (242), `enforce-allowlist-judge-gates.yaml` (141), `test_judge_coverage.py` (436), `test_override_rate.py` (438), `cli.py +257 LOC delta`.
**Pack:** 7 agents (dropped LLM agents; added pipeline-reviewer).

### CRITICAL
- **C7-1: Gates check presence, not authenticity.** Both C1 and C3 read plain-text YAML that an attacker can hand-edit with fake `judge_verdict: ACCEPTED`, fake `judge_rationale`, fake `judge_recorded_at`. Closes "forgot to run judge"; doesn't close "deliberately fabricated judge record". **C1 partially closes T-001; C3 does not close T-002.** (threat-analyst, FP-analyst — 2 agents converge)
- **C7-2: C1 PR-only trigger bypass.** Direct push to `main` or `RC*` (in workflow's `on.push` list) silently skips C1 because of `if: github.event_name == 'pull_request'`. If branch protection isn't configured to require it, T-001 is wide open. (threat-analyst, pipeline-reviewer)
- **C7-3: C1 cannot detect mutation of existing entries.** Discriminator is `(file, rule, symbol, owner_norm, reason_norm)`. Modifying `judge_verdict`/`judge_rationale`/`judge_recorded_at` on a grandfathered entry without touching owner/reason silently passes. E-R001 (rotate metadata laundering) survives C1. (threat-analyst)
- **C7-4: Silent-failure trio in `judge_coverage.py`.** (a) `_directory_has_allow_hits:267-270` silently swallows OSError (whole directory disappears from gate output); (b) substring match `"\nallow_hits:"` admits CRLF/no-newline false negatives; (c) baseline parse failures silently treated as empty contributor (every grandfathered entry flips to "new", operator sees wall of violations with no diagnostic). (silent-failure 3 CRITICALs, python-code-reviewer M1+M4)
- **C7-5: `yaml.YAMLError` not caught.** Both modules catch only `ValueError`; `yaml.YAMLError` subclasses `Exception`, not `ValueError`. Result: malformed YAML produces uncaught traceback → exit 1 instead of documented exit-2. CI cannot distinguish "gate broken" from "gate fired". (code-reviewer CRITICAL #1)
- **C7-6: Future-dated entries silently dropped without operator signal.** `override_rate.py:177-185` excludes future timestamps from both numerator and denominator with no log. Silent acceptance of tampering is exactly the failure mode the gate exists to prevent. (silent-failure M-3, python-code-reviewer M2, FP-analyst, threat-analyst — 4 agents converge)
- **C7-7: No CLI integration tests for the +257 LOC delta.** Library well-tested; gate (CLI + workflow contract) untested. Classic VER-without-VAL for the calibration instrument the whole prototype rests on. (QA C1)

### MAJOR convergent
- M7-1: C3 structural one-sidedness — "operator gave up under deadline and refactored" leaves no trace (FP, threat, Chunk 1 C1 carryover — 3 agents).
- M7-2: Threshold policy laundering — `--max-rate` not pinned in workflow, lives as CLI default (FP C3-4, threat, pipeline-reviewer — 3 agents).
- M7-3: Code duplication between gates with subtle behavioural drift (code-reviewer #4, python-code-reviewer minor).
- M7-4: No `concurrency:` block in workflow → push-rebase cycles queue unbounded (pipeline-reviewer).
- M7-5: Override-rate PASS notices buried in step logs, not surfaced to `$GITHUB_STEP_SUMMARY` (pipeline-reviewer).
- M7-6: Help-text/code mismatch on the override-rate denominator (code-reviewer #2).
- M7-7: Workflow comment lies about trigger scope ("on every PR and push" but push restricted to main/RC*) (code-reviewer #3).
- M7-8: Gate scope narrowed to `allow_hits:` shape only — `allow_classes:` and other formats are exempt from coverage enforcement (FP C1-2, threat MAJOR).

### MAJOR unique
- M7-9: Anti-gaming via dilution untested (FP M1).
- M7-10: Rotation-grandfathering relies on owner/reason as spoofing anchor (semantically-empty values defeat protection).
- M7-11: Naive `judge_recorded_at` in YAML untested (QA M3).
- M7-12: No `timeout-minutes:` (pipeline).
- M7-13: `base.sha` ≠ true merge-base (pipeline).
- M7-14: Localized git stderr fragility — `git fatal:` heuristic breaks on non-English `LANG`/`LC_ALL` (silent-failure m-1).
- M7-15: `_git_show` returns None on any non-zero exit, collapses 3 failure modes (silent-failure m-2).

### What's strong
- Discriminator design on `(owner, reason)` axis is sound — tampering can't use rotation loophole without copying author's prose verbatim (python-code-reviewer caveat).
- Threshold delegated to CLI default with ADR comment (positive per pipeline-reviewer — even though pinning would be stronger).
- Boundary math (window edges, inclusive comparison) tested.
- Fail-closed semantics preserved when gate Python crashes (correct exit non-zero).
- `frozen=True, slots=True` dataclass hygiene correct.
- L1 layer placement correct.
- `override_rate.py` correctly raises on YAML parse failures (the consistency is the issue — `judge_coverage.py` doesn't).

---

## §3.8 Chunk 8 — allowlist + multi-rule + ci.yaml + plan + CLAUDE.md (~1537 LOC)

**Files:** `core/allowlist.py` (judge-metadata additions, ~222 LOC delta), `test_reaudit_multi_rule.py` (180), `ci.yaml` delta, `notes/cicd-judge-cli-prototype-plan.md` (179), `notes/new_ci_remediation.md` (300), `CLAUDE.md` delta.
**Pack:** 8 agents (added doc-critic and pipeline-reviewer; dropped LLM and rule-designer).

### CRITICAL
- **C8-1: `JudgeVerdict.BLOCKED` round-trips through loader despite "in-memory only" docstring.** `_optional_judge_verdict` accepts it; `_validate_judge_metadata_atomic` doesn't reject it. AND `reaudit.py:72` has an explicit `(BLOCKED, None, BLOCKED) → STILL_AGREES` transition path. Architecturally inconsistent. (python-code-reviewer + code-reviewer #1 — 2 agents converge)
- **C8-2: `test_reaudit_multi_rule.py` partially closes the Chunk 3 17-branch gap.** Converts "17 of 17 untested" → "17 of 17 *registered*, 1 of 17 *executed*, 0 of 17 *divergence-classified*". Vocabulary is pinned, dispatch is not. (FP-analyst, QA-analyst, python-code-reviewer — 3 agents converge)
- **C8-3: Judge quartet NOT bound to suppressed code.** `file_fingerprint`/`ast_path` are independent optional fields; loader doesn't enforce `judge_verdict != None → fingerprint != None`. Attacker can transplant an ACCEPTED quartet onto an entry whose key/fingerprint point at different code. (threat-analyst CRITICAL)
- **C8-4: Empty-string `judge_rationale` slips through atomic validator.** Only caught by string-shape validator with generic message; if `_optional_string` is loosened, the atomic validator wouldn't catch empty rationale. Tier-1 specific guard absent. (silent-failure CRITICAL)
- **C8-5: `new_ci_remediation.md` has wholesale content duplication.** M1-N1-N2 block repeats verbatim at lines 37-105 AND 108-212. Reader cannot tell if duplicate is accidental or intentionally divergent. (doc-critic CRITICAL)
- **C8-6: Plan note Pillar B contradiction.** Status (line 3) + Delivered/Enforced table (line 14) say decorator shipped; Pillar B section header (line 95) says "future work, not in prototype." Irreconcilable. (doc-critic CRITICAL)
- **C8-7: Plan doctrine overstatements.** Lines 19 and 75: "judge's rationale is the new audit primitive" and "no longer relies on social norms or self-discipline." What C1+C3 actually enforce: schema completeness and override rate. Rationale *quality* still depends on judge model's judgment, neither perfectly reliable nor independently verified. (doc-critic CRITICAL)

### MAJOR
- M8-1: ci.yaml judge-related additions clean; cross-workflow concurrency gap (already in Chunk 7).
- M8-2: Coverage matrix inversion 3.12→3.13 undocumented (pipeline-reviewer).
- M8-3: Integration job `if:` silently excludes RC branches (pipeline-reviewer).
- M8-4: `_load_yaml_file` returns `{}` for missing file silently (silent-failure).
- M8-5: `judge_model_verdict` accepts `OVERRIDDEN_BY_OPERATOR` value — semantically nonsense (silent-failure, python-code-reviewer).
- M8-6: `yaml.safe_load` unbounded → DoS surface (threat-analyst).
- M8-7: `Allowlist.match` mutates `entry.matched=True` as side effect, no concurrent-use docstring (code-reviewer #2).
- M8-8: `_optional_datetime` docstring claims UTC-only but accepts any tz offset (python-code-reviewer).
- M8-9: Per-rule calibration not addressed; C3 doesn't partition denominator by rule_id (FP-analyst).
- M8-10: Per-file rules bypass judge entirely with no signal (FP-analyst).
- M8-11: CLAUDE.md documents `rotate` but not `justify`/`reaudit` though all three are shipped — discoverability asymmetry (code-reviewer).
- M8-12: Pillar B migration ordering not documented (test-ref must exist before decorator can pass `trust_boundary.tests`) (doc-critic).
- M8-13: C3 response protocol absent (what does operator do when rate fires?) (doc-critic).
- M8-14: Allowlist integrity tests use `allow_hits` exclusively (no `allow_classes`, mixed-shape) (silent-failure, QA-analyst).

### What's strong
- Atomic-shape validator solid for the 4 documented invariants (multiple confirm).
- `_optional_judge_verdict` correctly Tier-1 (rejects unknown values) — flagged as exemplary by threat-analyst.
- Naive datetime rejection works (caveat: UTC-only claim not enforced).
- Layer placement correct; `AllowlistEntry` non-frozen by design (`matched` mutates) correctly excluded from deep_freeze contract.
- Registry-derivation pin for multi-rule catches "added to BUILTIN_RULES but forgot vocabulary" footgun (FP, QA agree).
- ci.yaml delta is responsible — no judge duplication with separate workflow.
- CLAUDE.md delta is net-positive (Trust Flow ASCII removed; Quick Reference table covers the same content more concisely).

---

# §4 What's strong — validated across chunks

These are properties the review confirmed across multiple chunks. Don't break these in the course of addressing other findings.

- **Fail-closed property structurally preserved** in judge transport AND CLI (silent-failure confirmed Chunks 1, 2).
- **Cache accounting `None`/`0`/`>0` per fabrication-decision test** (Chunk 1 — 4 agents called this exemplary).
- **Offensive output-schema parser** crashes on incoherence (Chunk 1).
- **Atomic-shape validator** for the 4 documented invariants is solid (Chunks 1, 2, 8).
- **Layer model compliance** verified across all chunks (L0 → L1 → L3).
- **No defensive `.get()` / `getattr()` / `hasattr()` abuse on internal typed data** (confirmed every chunk).
- **B1 inner-scope-leak fix correctly handles original direction** (Chunk 5, empirically confirmed).
- **Auto-stub regression in rotate genuinely removed** (Chunk 4 — 4 agents independently verified the code; no test pins it).
- **Registry-derivation pin catches "BUILTIN_RULES + missing vocabulary" footgun** (Chunk 8).
- **Allowlist-loader contract tests for the 4 documented invariants are thorough** (Chunk 2).
- **Discriminator design (rotation grandfathering via owner+reason) is sound** — tampering can't use rotation loophole without copying author's prose verbatim (Chunk 7 caveat).
- **Read-only on YAML confirmed for reaudit** (Chunk 3).
- **`tier` honesty rule is genuinely tight** (Chunk 6, multiple agents).
- **Lattice escape on malformed decorator metadata correctly fail-closed** (Chunk 5).

---

# §5 Higher-order recommendations

These are not single-fix items. They are structural decisions the operator should evaluate.

## §5.1 Hash-chain or sign the audit primitive

Judge metadata is plain-text forgeable. This undermines T-001 mitigation across Chunks 1, 7, 8. C1 and C3 check presence, not authenticity. Without integrity binding, the prototype's "audit primitive" claim is overstated (Chunk 8 doc-critic). Options:

- HMAC-sign judge metadata blocks with a CI-only key; CI re-verifies on PR.
- Append-only signed event log (git-notes, signed manifest) separate from YAML.
- Bind quartet to `(file_fingerprint, ast_path)` so transplant attacks fail.

This is the single highest-leverage architectural fix. Without it, multiple CRITICALs across Chunks 1/2/7/8 remain theoretically open.

## §5.2 Add a falsifiability hook

`audit_review` field (nullable) so the calibration loop isn't one-sided (Chunk 1 C1, Chunk 3 C-R4, Chunk 7 under-override). A separate `elspeth-lints audit-verdict` path lets an operator who later discovered a judge ACCEPTED was wrong annotate the entry. Without it, the override-rate gate is a tripwire for honest operators only.

## §5.3 Address the C5-2 doctrine gap honestly

Either:
- (a) Extend `trust_boundary.tests` to verify the decorated symbol appears in the test body (closes the gap mechanically), OR
- (b) Downgrade the docstring claim to "raising-shape check, not invariant liveness" (closes the gap discursively).

Currently the rule advertises (a) but implements (b). Six agents flagged this; pick one.

## §5.4 Single-source the YAML writer/loader

Multiple agents flagged code duplication with subtle drift across cli.py, rotate.py, judge_coverage.py, override_rate.py. Drift is silent and produces inconsistent error messages. Lift to a single `allowlist_io` module with atomic write, structured load, shared error types.

## §5.5 Add a labelled-corpus regression suite for the judge model

VAL gap acknowledged across Chunks 1, 3, 5, 7 but no instrument exists. The prototype's quality is asserted, not measured. 10–30 historical (rationale, code-excerpt, expected-verdict) tuples run against real Opus as a gated CI job would close the calibration loop. Without it, every prompt edit ships blind.

## §5.6 Documentation cleanup

- Fix Pillar B section header in plan note (decorator shipped; corpus migration deferred).
- Soften 2 doctrine overstatements (audit primitive, social norms).
- Delete duplicate block in `new_ci_remediation.md`.
- Add `justify` and `reaudit` entries to CLAUDE.md (currently only `rotate` is documented).
- Add C3 response protocol (what does operator do when rate fires?).

---

# §6 Status of the uncommitted work

Chunks 7 and 8 reviewed code that is **uncommitted in the worktree** (untracked + unstaged at time of review):

**Untracked files:**
- `elspeth-lints/src/elspeth_lints/core/judge_coverage.py` (429 LOC)
- `elspeth-lints/src/elspeth_lints/core/override_rate.py` (242 LOC)
- `.github/workflows/enforce-allowlist-judge-gates.yaml` (141 LOC)
- `tests/unit/elspeth_lints/test_judge_coverage.py` (436 LOC)
- `tests/unit/elspeth_lints/test_override_rate.py` (438 LOC)
- `tests/unit/elspeth_lints/test_reaudit_multi_rule.py` (180 LOC)
- `notes/new_ci_remediation.md` (300 LOC)

**Unstaged edits:**
- `elspeth-lints/src/elspeth_lints/core/cli.py` (+257 LOC — the `_run_check_judge_coverage` and `_run_check_override_rate` subcommands)
- `elspeth-lints/src/elspeth_lints/core/reaudit.py` (+308 LOC — multi-rule expansion)
- `notes/cicd-judge-cli-prototype-plan.md` (+26, -2)

**Mismatch:** The plan note describes this work as shipped (Status paragraph, Delivered/Enforced table). Reality is "in worktree, ready to commit." This is itself worth flagging — for an audit-trail-integrity prototype, the design document describing a state that hasn't been committed is a small irony.

The reviews above evaluated the work **as-it-would-be-merged**, treating the working-tree state as the canonical state of the prototype.

---

# §7 Raw transcripts

The 57 sub-agent reviews each produced a JSONL transcript on disk. They include the full reasoning, tool calls, file reads, and final report for each agent. They survive this session.

**Location:** `/tmp/claude-1000/-home-john-elspeth/b351e02c-5086-4c6d-8947-2bc31ea570e3/tasks/`

**Filenames:** Each agent has a unique ID; file is `<agent-id>.output`. The agent IDs are referenced in this document only implicitly (each chunk's "8 agents back" notification carried them). If you need to map a specific finding back to its source transcript, the chunk + agent role narrows it: e.g., "Chunk 5 silent-failure-hunter" identifies one specific file in that directory.

**Format:** JSONL — each line is a JSON object representing a tool call, tool result, thinking block, or final message from the sub-agent. Inspect with `jq` or any JSONL reader. The agent's final review is the last `result` text block.

**Caveat:** I have not enumerated the 57 agent-id → file mappings in this document; that would require re-walking each chunk's dispatch log. If you need a specific agent's transcript and the chunk+role isn't sufficient to narrow it, run `ls -la /tmp/claude-1000/-home-john-elspeth/b351e02c-5086-4c6d-8947-2bc31ea570e3/tasks/ | sort -k6,7` to list by timestamp; chunks were dispatched sequentially.

---

# Appendix: Convergence counts (for triage prioritisation)

Findings flagged by multiple agents are higher-confidence signals than solo findings. Quick triage table:

| Finding | Convergence | Severity |
|---------|-------------|----------|
| Judge audit-trail integrity = presence only | Chunks 1,2,7,8 (many agents) | CRITICAL theme |
| `ast_walker.py:67` ClassDef leak | 4 agents (rule-designer, QA, silent-failure, python-code-reviewer) | CRITICAL |
| Non-determinism (no `temperature=0`) | Chunks 1,2,3 (multiple agents) | CRITICAL |
| `completion.model` audit bug | Solo (llm-diagnostician), grep-verified, then carryover in Chunk 3 | CRITICAL |
| `--operator-override` no auth | Chunks 1,2 (4 agents) | CRITICAL |
| Source/secrets exfil via surrounding_code | Chunks 1,2,3 (3+ agents per chunk) | CRITICAL |
| `--rule` flag dead code | Solo (llm-diagnostician), grep-verified | CRITICAL |
| YAML writer non-atomic | Chunks 2,4 (5 agents per chunk) | CRITICAL |
| Per-entry failure aborts reaudit sweep | Chunk 3 (silent-failure 3 separate CRITICALs) | CRITICAL |
| Override-laundering via rotate | Chunk 4 (threat solo, profound) | CRITICAL |
| C5-2 not closed: raising-shape ≠ invariant-liveness | Chunks 5,6 (6 agents) | CRITICAL |
| `yaml.YAMLError` not caught | Chunk 7 (code-reviewer solo, code-cited) | CRITICAL |
| `JudgeVerdict.BLOCKED` round-trips through loader | Chunk 8 (2 agents) | CRITICAL |
| Plan note doctrine overstatements | Chunk 8 (doc-critic solo, well-documented) | CRITICAL |
| Test_reaudit_multi_rule = registry-pinned not dispatch-executed | Chunk 8 (3 agents) | CRITICAL |

When choosing what to fix first, prioritise high-convergence findings — they are the ones that survived independent review by agents with different lenses.

---

*End of consolidation document. Cross-check with raw transcripts in `/tmp/claude-1000/-home-john-elspeth/b351e02c-5086-4c6d-8947-2bc31ea570e3/tasks/` for any finding's full evidence chain.*
