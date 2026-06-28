# Codex Panel Review — Caching / Cost / Throughput Pilot

**Date:** 2026-06-28 · **Tool:** `scripts/codex_panel_review.py` (foundation, Plan 1)
**codex CLI:** 0.142.3, default model, `--sandbox read-only` · **Run isolation:** a clean detached
git worktree (untracked `.env` / `data/*.db` absent → outside the sandbox scope).

This is the empirical input Plan 2 (orchestration) is written from: it answers whether codex
prompt-caching makes the panel approach viable, what one full run costs, what concurrency the
account sustains, and how Plan 2 should order execution.

## Corpus — the state engine (most complex code)

The five largest files in `src/elspeth/engine/` (the repo's most complex subsystem), reviewed
through both Plan-1 lenses (`solution-architect`, `security-architect`):

| File | LOC |
| --- | ---: |
| `engine/processor.py` | 5,735 |
| `engine/coalesce_executor.py` | 1,804 |
| `engine/orchestrator/core.py` | 1,529 |
| `engine/orchestrator/resume.py` | 1,178 |
| `engine/executors/sink.py` | 1,098 |

**Method.** Phase A ran `processor.py` **serially** (clean cross-lens cache numbers on the
biggest possible `[source]`). Phase B ran the other four **concurrently** (4-way), which doubled
as the `--workers` ceiling probe. 10 live calls total. A prior 1-call schema-acceptance smoke
(against the panel script itself) confirmed codex accepts the strict `--output-schema`.

## Headline results

| File | lens (run order) | input tok | cached | cache % | output | findings | duration |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| processor.py | 1 · solution | 1,956,882 | 1,802,240 | 92.1% | 13,148 | 7 | 286 s |
| processor.py | 2 · security | 3,779,822 | 3,568,512 | 94.4% | 11,258 | 1 | 296 s |
| coalesce_executor.py | 1 · solution | 787,827 | 690,176 | 87.6% | 9,520 | 6 | 234 s |
| coalesce_executor.py | 2 · security | 2,593,772 | 2,484,608 | 95.8% | 13,398 | 3 | 324 s |
| core.py | 1 · solution | 1,233,866 | 1,105,920 | 89.6% | 15,176 | 4 | 462 s |
| core.py | 2 · security | 4,797,130 | 4,532,992 | 94.5% | 24,128 | 3 | 627 s |
| resume.py | 1 · solution | 219,599 | 85,376 | **38.9%** | 4,178 | 5 | 341 s |
| resume.py | 2 · security | 661,681 | 578,432 | **87.4%** | 19,739 | 3 | 409 s |
| sink.py | 1 · solution | 1,471,908 | 1,284,352 | 87.3% | 12,051 | 5 | 293 s |
| sink.py | 2 · security | 366,615 | 317,824 | 86.7% | 14,931 | 3 | 290 s |

**Totals (10 calls):** input **17,869,102** · cached **16,450,432 (92.1%)** · uncached
**1,418,670** · output **137,527** · findings **40** · failures **0** · 429s **0**.

## 1. Caching works — but not for the reason the design assumed

Aggregate **92.1% input-token cache reuse**. That is the headline: codex's prompt cache is highly
effective for this workload.

**The dominant cached component is not `[source]`.** The design assumed a static
`[context][source][persona]` prefix whose `[source]` layer is reused across a file's lenses. In
practice codex reviews are **agentic and multi-turn**: the model reads many other repo files while
investigating, and `input_tokens` is the sum over all turns. So `input_tokens` is driven by
*investigation depth*, not the static prefix — `processor.py` consumed 5.7M input tokens across its
two lenses; `resume.py` only 0.88M. What caches well is (a) the large, file-independent `[context]`
block (`CLAUDE.md` + `AGENTS.md` + 4 skill files, inlined by `load_context`) and (b) intra-call
turn-over-turn reuse. Both are present **regardless of execution ordering**.

**Cross-lens `[source]` warming is real but secondary.** Run-order is `solution` (lens 1, cold) →
`security` (lens 2, warm). In 4 of 5 files lens 2's cache % exceeds lens 1's, and `resume.py` shows
it unmistakably: **38.9% → 87.4%** once lens 1 had warmed the `[context][source]` prefix. But the
magnitude is small next to the `[context]`-level reuse that file-parallel execution gets for free,
and it is easily masked by investigation-depth differences (`sink.py`: lens 2 did *less* reading, so
its cache % is marginally lower despite the warm prefix).

> **Measurement caveat (honest):** the clean `[source]`-marginal number the plan asked for is **not
> cleanly extractable** from this run — agentic investigation confounds `input_tokens`. The
> directional conclusion below is robust; a precise marginal would need turn-level token accounting
> the stock `run_codex_once` does not expose.

## 2. Throughput — workers ceiling ≥ 4

Phase B ran four files (8 calls) **concurrently with no rate-limit**: **zero 429s, zero failures**.
Wall-clock for Phase B was ~18 min vs ~36+ min if serial. Individual calls under contention ran
slower (`core.py`: 462 s / 627 s vs `processor.py`'s 286 s / 296 s serial), i.e. concurrency trades
a little per-call latency for ~2× throughput.

**Ceiling is ≥ 4; the actual cap was not found** (a higher probe would cost more calls). Plan 2
should default `--workers` to a conservative value (e.g. 4) and treat the first sustained 429 as the
real cap.

## 3. Per-call latency vs cache TTL

Per-call duration ranged **234–627 s (median ~310 s, ~5 min)**. This sits at the low end of the
~5–10 min prompt-cache TTL — yet lens 2 cache stayed high across the board, so **cross-lens warmth
survived the serial gap** in practice. No TTL-driven cache misses were observed.

## 4. Cost

Reported in **tokens** (objective); the dollar figure depends on your codex model/plan, so it is
left as an explicit conversion rather than asserted.

- **The 5 most-complex files (10 calls):** ~1.42M **uncached** input + ~0.14M output. ~92% of all
  input was cache-served.
- **Per file (2 lenses):** ~0.21M–0.39M uncached input, ~17k–39k output. These are **upper-bound**
  files (top of the complexity distribution); a typical file costs far less.
- **Illustrative** (substitute your plan's rates): at example API rates of ~$1.25 /M input (cached
  input is typically discounted ~10×) and ~$10 /M output, the 5-file run is on the order of a few
  dollars. **Do not** extrapolate a full-tree dollar figure from these monsters.

**Full-tree projection (the plan's ~1,800-call estimate).** A naïve extrapolation using these
engine files as the average would badly overstate cost, because they are the largest files in the
repo. Before committing a full-tree budget, run a **representative-sample** pass (e.g. 15–20 files
spanning the size distribution) and multiply the *median* per-file cost — not these upper-bound
numbers. What this pilot establishes is the **shape**: input is large but ~92% cached, output is
small (~10–25k/file), and failures/429s are zero at modest concurrency.

## 5. Signal quality

40 findings across the 5 files, **`gated=0` everywhere** — every anchor-required finding carried a
real `path:line`, so the evidence gate never had to downgrade. That validates two things on real
code: the prompt's evidence instruction reliably produces located findings, and the gate's
fail-closed design isn't firing false downgrades. `solution-architect` was denser (4–7 findings/file)
than `security-architect` (1–3) on internal engine code, as expected. The lenses produced located,
non-noise findings on the hardest code in the tree.

## 6. Decisions for Plan 2

**(a) Go / no-go — GO.** Schema accepted, 11/11 calls succeeded (incl. smoke), caching delivers
~92% input reuse, per-call latency is within the cache TTL, and output volume is modest. The
approach is viable on the most complex code in the tree.

**(b) Execution ordering — parallelize across files, keep lenses serial within a file.** This is the
key correction to the plan's premise. Pure *file-major-serial* (one file at a time) leaves
throughput on the table for only a small marginal cache gain, because:
  - the dominant cache benefit (`[context]` prefix + intra-call reuse) is **order-independent** and
    is fully harvested under file-parallel execution; and
  - concurrency sustained 4-way with no 429 and ~2× throughput.

  The cross-lens `[source]` warming (`resume.py` 39%→87%) is worth keeping, so **lenses should still
  run serially *within* a worker/file**. The recommended Plan-2 default is therefore a **worker pool
  parallel across files (`--workers` ≥ 4), each worker running its file's lenses serially** — which
  captures the `[context]` caching, the cross-lens `[source]` warming, *and* the throughput.

**(c) `--workers` default = 4** (conservative; raise until the first sustained 429).

## 7. Follow-ups surfaced by the pilot (Plan 2 / hardening)

The live run (notably the smoke against the tool's own source) surfaced, and these remain open:
- a sensitive-file/`data/` guard on `--file` itself (the clean-worktree run model mitigates this for
  pilots, but not for general operator use);
- prompt-injection hardening of the inlined `[source]` beyond the current secret-fence;
- having the gate verify an anchor's `path:line` actually exists in the tree rather than only
  shape-checking it (a Plan-2 verification pass);
- turn-level token accounting if a precise `[context]` vs `[source]` marginal is ever required.

---
*Raw per-lens artifacts were written under a gitignored, throwaway worktree and are not committed;
this summary is the durable record.*
