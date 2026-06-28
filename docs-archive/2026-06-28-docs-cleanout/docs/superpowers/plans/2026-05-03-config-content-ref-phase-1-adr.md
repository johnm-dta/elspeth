# Phase 1 — ADR + VAL Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land ADR-021 ("Audited content injection — widened `blob_ref` with `mode` discriminator") with all design closures from spec §4 and the M1 numeric caps grounded in VAL data sampled from the existing audit DB and LLM-plugin prompt corpora.

**Architecture:** Markdown-only PR. Two artefacts: a VAL-data appendix (data + analysis script + raw query results) and the ADR itself, structured against the project's ADR convention (`docs/architecture/adr/000-template.md`). Numeric caps in the ADR are cited against the VAL-data appendix; no caps left as "TBD" or "e.g. 64KB" — the panel rejected unprincipled numbers, and that rejection extends to this phase's deliverable.

**Tech Stack:** Markdown. SQL queries against the existing audit DB (`audit.db` SQLite for local; staging Postgres for production-shape data). Python script for sampling and percentile computation (`scripts/analysis/blob_inline_val_data.py`).

---

## Pre-phase verification

- [ ] **Step 1: Confirm spec is current**

```bash
git log --oneline -5 docs/superpowers/specs/2026-05-03-config-content-ref-design.md
```

Expected: Latest commit is the spec write-up. If the spec has been amended after this plan was written, re-read §4 and §1.4 before drafting the ADR.

- [ ] **Step 2: Confirm ADR namespace is free**

```bash
ls docs/architecture/adr/ | grep -E '^021-'
```

Expected: empty. ADR-021 is the next-available number. ADR-019 is the two-axis terminal model (accepted 2026-05-04; commit sequence ending at `44146aec`); ADR-019 soft-reserves ADR-020 for a future counter-rename ADR (`docs/architecture/adr/019-two-axis-terminal-model.md:331,545,568,690`). The widened-`blob_ref` work takes the next free slot after that reservation.

---

## Task 1: VAL-data sampling script

**Files:**
- Create: `scripts/analysis/blob_inline_val_data.py`
- Create: `docs/architecture/adr/021-config-content-ref-val-data.md` (appendix)

- [ ] **Step 1: Write the failing test for the sampling script**

```python
# tests/unit/scripts/analysis/test_blob_inline_val_data.py
"""Pin the percentile-computation contract for the VAL-data sampler.

The sampler walks an audit DB and returns p50/p90/p95/p99 byte-length
distributions for inline LLM/SQL/template field content. The percentile
math must match what the ADR reports — divergence here is the same
class of bug the spec's M1 finding warned against (unprincipled
numbers).
"""

import pytest

from scripts.analysis.blob_inline_val_data import compute_percentiles


def test_compute_percentiles_evenly_spaced() -> None:
    samples = list(range(1, 101))  # 1..100
    result = compute_percentiles(samples)
    assert result.p50 == 50
    assert result.p90 == 90
    assert result.p95 == 95
    assert result.p99 == 99


def test_compute_percentiles_empty_raises() -> None:
    """Empty corpora are an audit-decision, not a fallback to 0.

    Reporting p99=0 for an empty corpus would let the ADR set caps
    against vacuous evidence. The sampler crashes instead.
    """
    with pytest.raises(ValueError, match="empty corpus"):
        compute_percentiles([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/analysis/test_blob_inline_val_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.analysis.blob_inline_val_data'`.

- [ ] **Step 3: Write the sampling script**

```python
# scripts/analysis/blob_inline_val_data.py
"""Sample inline LLM/SQL/template field content from an existing audit DB
and emit byte-length percentiles for ADR-021's size-cap NFRs.

Uses the existing audit-DB schema (SQLAlchemy Core; see
src/elspeth/core/landscape/schema.py for table definitions). Reads the
recorded plugin-config rows and extracts string-typed option values for
fields that historically carried long-form content: LLM `system_prompt`,
LLM `user_prompt_template`, SQL plugin `query`, web plugin
`body_template`, classifier plugin `prompt_template`.

Outputs a JSON report with per-field percentile distributions plus the
LLM-plugin prompt corpus from `tests/data/llm_prompts/` (real-world
representative prompts the team has accumulated). The ADR cites the
JSON output by path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PercentileReport:
    p50: int
    p90: int
    p95: int
    p99: int
    sample_count: int


def compute_percentiles(samples: Iterable[int]) -> PercentileReport:
    """Compute integer-rounded byte-length percentiles for a sample corpus."""
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("empty corpus — cannot compute percentiles for cap-setting")
    quantiles = statistics.quantiles(sample_list, n=100, method="inclusive")
    return PercentileReport(
        p50=int(quantiles[49]),
        p90=int(quantiles[89]),
        p95=int(quantiles[94]),
        p99=int(quantiles[98]),
        sample_count=len(sample_list),
    )


def _sample_audit_db(audit_db_path: Path) -> dict[str, list[int]]:
    """Walk the audit DB; return field_name -> [byte_lengths]."""
    # IMPLEMENTATION: SQLAlchemy Core + json.loads on plugin_options column;
    # extract string values for the closed-list of long-form-content fields.
    raise NotImplementedError("Implement against src/elspeth/core/landscape/schema.py shape")


def _sample_test_corpus(corpus_dir: Path) -> dict[str, list[int]]:
    """Walk tests/data/llm_prompts/; return field_name -> [byte_lengths]."""
    # IMPLEMENTATION: glob {system_prompt,user_prompt_template}_*.txt;
    # measure UTF-8 byte length of each.
    raise NotImplementedError("Implement against existing prompt corpus shape")


def main() -> int:
    parser = argparse.ArgumentParser(description="VAL-data sampler for ADR-021")
    parser.add_argument("--audit-db", type=Path, required=False, help="path to audit.db")
    parser.add_argument("--corpus-dir", type=Path, default=Path("tests/data/llm_prompts"))
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination")
    args = parser.parse_args()

    samples: dict[str, list[int]] = {}
    if args.audit_db is not None:
        for field, lengths in _sample_audit_db(args.audit_db).items():
            samples.setdefault(field, []).extend(lengths)
    for field, lengths in _sample_test_corpus(args.corpus_dir).items():
        samples.setdefault(field, []).extend(lengths)

    report = {field: compute_percentiles(lengths).__dict__ for field, lengths in samples.items()}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run unit test to verify percentile math passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/analysis/test_blob_inline_val_data.py -v`
Expected: PASS.

- [ ] **Step 5: Implement `_sample_audit_db` against the actual landscape schema**

```bash
# Inspect the schema
grep -n "plugin_options\|node_states" src/elspeth/core/landscape/schema.py | head -20
```

Then complete the `_sample_audit_db` body. Implementation guidance: read `node_states.plugin_options` JSON, parse, extract values where the key is in the closed list `{"system_prompt", "user_prompt_template", "query", "body_template", "prompt_template"}`. For string-only values, count `len(value.encode("utf-8"))`.

- [ ] **Step 6: Implement `_sample_test_corpus`**

```python
def _sample_test_corpus(corpus_dir: Path) -> dict[str, list[int]]:
    samples: dict[str, list[int]] = {}
    if not corpus_dir.exists():
        return samples
    for path in corpus_dir.glob("*.txt"):
        field = path.stem.split("_")[0]  # "system_prompt_v1.txt" -> "system_prompt"
        samples.setdefault(field, []).append(len(path.read_bytes()))
    return samples
```

- [ ] **Step 7: Run the sampler against the local dev audit DB**

```bash
.venv/bin/python -m scripts.analysis.blob_inline_val_data \
    --audit-db ./examples/my_pipeline/runs/audit.db \
    --corpus-dir tests/data/llm_prompts \
    --output /tmp/blob_inline_val_data.json

cat /tmp/blob_inline_val_data.json
```

Expected: a JSON report keyed by field name, each carrying `p50/p90/p95/p99/sample_count`. If `sample_count` for any field is below 30, surface that limitation in the ADR (the panel's M1 finding requires honest evidence, including evidence of weakness).

- [ ] **Step 8: Commit**

```bash
git add scripts/analysis/blob_inline_val_data.py tests/unit/scripts/analysis/test_blob_inline_val_data.py
git commit -m "feat(scripts): add VAL-data sampler for ADR-021 size-cap NFRs

Walks the existing audit DB and the test prompt corpus to produce
byte-length percentile distributions for long-form-content fields
(system_prompt, user_prompt_template, query, body_template,
prompt_template). The ADR cites the JSON output by path.

Refs: elspeth-fdebcaa79a"
```

---

## Task 2: VAL-data appendix

**Files:**
- Create: `docs/architecture/adr/021-config-content-ref-val-data.md`

- [ ] **Step 1: Write the appendix with the sampler output embedded**

```markdown
# ADR-021 Appendix — VAL Data for Size-Cap NFRs

**Source script:** `scripts/analysis/blob_inline_val_data.py`
**Sample date:** YYYY-MM-DD
**Sources:**
- Local dev audit DB: `./examples/my_pipeline/runs/audit.db` (commit <hash>)
- Test prompt corpus: `tests/data/llm_prompts/` (commit <hash>)

## Per-field byte-length distribution

| Field | p50 | p90 | p95 | p99 | sample count |
|---|---|---|---|---|---|
| `system_prompt` | <N> | <N> | <N> | <N> | <N> |
| `user_prompt_template` | <N> | <N> | <N> | <N> | <N> |
| `query` | <N> | <N> | <N> | <N> | <N> |
| `body_template` | <N> | <N> | <N> | <N> | <N> |
| `prompt_template` | <N> | <N> | <N> | <N> | <N> |

## Cap rationale

- **Per-ref upper bound:** chosen as 2× the cross-field p99 to leave headroom
  without admitting megabyte payloads. Rationale: a 2× headroom from the p99
  observed surface accommodates 1-in-100 prompt growth without re-litigation;
  2× is a conventional buffer for capacity-planning percentile caps.
- **Per-ref lower bound (warning, not reject):** 256B. Below this size, an
  inline literal in YAML is more legible than a ref + UUID lookup; the warning
  discourages "everything is a ref" anti-patterns.
- **Aggregate per-config bytes cap:** chosen so that an N=8 config at the
  per-ref upper bound fits within the 30-second `_call_async` timeout budget,
  with a 4× resolution-latency headroom against the measured per-ref decode
  cost on standard CI infra.
- **Preflight resolution latency p95 sanity bound:** 1.5s for N=8 refs at the
  per-ref cap on standard CI infra. Tight target for nightly bench: TBD per
  the ADR's NFR table (no CI-shared-infra budgeting at p95).

The numeric caps are restated authoritatively in ADR-021 §6.
```

- [ ] **Step 2: Commit the appendix skeleton (numbers populated by Task 3)**

```bash
git add docs/architecture/adr/021-config-content-ref-val-data.md
git commit -m "docs(adr): add ADR-021 VAL-data appendix skeleton

Numeric cells populated after running the sampler against the dev
audit DB; cap rationale text complete.

Refs: elspeth-fdebcaa79a"
```

---

## Task 3: Run the sampler and populate the appendix

- [ ] **Step 1: Run the sampler against the most representative DB available**

```bash
.venv/bin/python -m scripts.analysis.blob_inline_val_data \
    --audit-db ./examples/my_pipeline/runs/audit.db \
    --corpus-dir tests/data/llm_prompts \
    --output docs/architecture/adr/021-config-content-ref-val-data.json
```

Run additionally against the staging audit DB if `elspeth-mcp --database` is configured against staging — record the connection string in the appendix's `Sources:` list. If only the local dev DB is available, surface the corpus-size limitation explicitly in the appendix.

- [ ] **Step 2: Populate the appendix table**

Edit `docs/architecture/adr/021-config-content-ref-val-data.md` and replace each `<N>` cell with the corresponding value from the JSON output. Compute the recommended caps:

```python
# In a Python REPL or scratch script:
import json
data = json.loads(open("docs/architecture/adr/021-config-content-ref-val-data.json").read())
cross_p99 = max(field["p99"] for field in data.values())
per_ref_upper_cap = cross_p99 * 2  # 2× headroom rationale
aggregate_cap = per_ref_upper_cap * 8  # N=8 nominal config size
print(f"per_ref_upper_cap = {per_ref_upper_cap}")
print(f"aggregate_cap = {aggregate_cap}")
```

- [ ] **Step 3: Commit the populated appendix**

```bash
git add docs/architecture/adr/021-config-content-ref-val-data.md docs/architecture/adr/021-config-content-ref-val-data.json
git commit -m "docs(adr): populate ADR-021 VAL-data appendix with measured caps

Sampler output: per-ref upper cap = <N> bytes; aggregate per-config
cap = <N> bytes; lower-bound warning at 256B; preflight latency
sanity bound at p95 ≤ 1.5s for N=8 refs.

Refs: elspeth-fdebcaa79a"
```

---

## Task 4: Draft ADR-021

**Files:**
- Create: `docs/architecture/adr/021-config-content-ref.md`

- [ ] **Step 1: Read the ADR template**

```bash
cat docs/architecture/adr/000-template.md
```

This anchors the ADR's structure (status / context / decision / consequences / alternatives considered). Match the format used by `005-adr-declarative-dag-wiring.md` and `018-producer-site-outcome-discrimination.md` for consistency with recent ADRs.

- [ ] **Step 2: Draft the ADR body**

```markdown
# ADR-021: Audited Content Injection — Widened `blob_ref` with `mode` Discriminator

**Status:** Proposed
**Bead:** elspeth-fdebcaa79a
**Date:** 2026-05-03
**Decision Makers:** RC5 architecture review (panel synthesis 2026-05-03)
**Depends On:** ADR-005 (declarative DAG wiring — node IDs as stable identifiers for `field_path` canonical encoding)
**Supersedes:** N/A

## Context

[Translate spec §2 (Context) and §3 (The Reframe) into ADR shape. Cite the spec by path. Reference the five-reviewer panel synthesis as the source of the design.]

## Decision

[Translate spec §4 (the ADR-row table) into the ADR's main decision section. The §4 table's twelve decision rows become twelve sub-decisions in the ADR body, each with its own rationale paragraph.]

### Numeric NFRs (cited against the VAL-data appendix)

[Pull each numeric cap from `021-config-content-ref-val-data.md` Table 1. Restate authoritatively here.]

| NFR | Value | Source |
|---|---|---|
| Per-ref upper bound | <N> bytes | VAL appendix; 2× cross-field p99 |
| Per-ref lower-bound warning | 256 bytes | VAL appendix; legibility heuristic |
| Aggregate per-config bytes cap | <N> bytes | VAL appendix; N=8 nominal at per-ref upper bound |
| Preflight latency sanity bound | p95 ≤ 1.5s | VAL appendix; CI sanity bound, not nightly tight |
| Hash-pin determinism | byte-identical round-trip | structural — `BlobServiceImpl.read_blob_content` integrity guard |
| Composer→runtime hash agreement | 0 events | SLO threshold = 0; OTel counter `composer.blob_inline.hash_mismatch_total` |
| Audit-row write failure (Tier-1) | 0 events | SLO threshold = 0; OTel counter `composer.blob_inline.audit_row_tier1_violation_total` |

## Closure rule

**No new ref forms without ADR amendment.** Future late-binding needs (env, upstream-node-output, plugin-instance refs) widen the existing `blob_ref` model or are rejected. Adding a sibling form requires amending this ADR, not a new ADR. Rationale: each new ref form has historically created parallel surfaces that share only the audit table — the architectural fix is widening, not multiplying.

## Consequences

### Positive

- Operators can inline long-form content into any plugin field without inlining megabytes of YAML or hand-rolling file reads in plugins.
- Audit trail captures `(field_path, blob_id, content_hash, byte_length, mime_type, encoding)` for every resolved ref, before the bytes flow into plugin instantiation.
- Lifecycle pinning extends to non-source references: a transform-options-referenced blob cannot be GC'd while a pipeline references it.
- The composer LLM trust boundary is mechanically enforced: the LLM cannot author content it has not seen via the blob-discovery tool.

### Negative

- Every plugin-options walk now recognises one more marker shape; recognition logic is duplicated (sibling, not unified, per S2).
- The dedupe-by-blob-id strategy in §6.3 means a single blob referenced from multiple field_paths within one run produces exactly one `blob_run_links` row — operators querying lifecycle pinning by field path must join through `blob_inline_resolutions`.
- The `field_path` canonical format (identity-anchored) means that renaming a node ID is a structural change to the audit trail; node ID renames are already discouraged by ADR-005, but this ADR amplifies that.

### Neutral

- Backwards compatibility is intentionally absent. Per CLAUDE.md no-legacy-code policy, P5 updates `set_source_from_blob` to emit explicit `mode: bind_source` in the same PR; mode-less markers are rejected as malformed.

## Alternatives Considered

### Sibling `{blob_content_ref: ID}` ref form (REJECTED)

[Translate spec §3.2 panel verdict.]

### Extending the secrets system (REJECTED)

[Translate spec §3.3 panel verdict.]

### Validation parity in P3, runtime preflight in P4 (REJECTED — phase ordering)

[Spec's Phase ordering pivot: shipping composer-green / runtime-red as a deliberate intermediate state synthesises the exact divergence Shapes 1 and 8 exist to close. Invert.]

## References

- Spec: `docs/superpowers/specs/2026-05-03-config-content-ref-design.md`
- VAL data: `docs/architecture/adr/021-config-content-ref-val-data.md`
- Five-reviewer panel synthesis: epic body of elspeth-fdebcaa79a
```

- [ ] **Step 3: Cross-check the ADR against the spec's §4 decision table**

```bash
grep -E "^\| .+ \| .+ \| .+ \| .+ \|$" docs/superpowers/specs/2026-05-03-config-content-ref-design.md | head -20
```

Every row in the spec's §4 must appear as a sub-decision in the ADR body. Missing rows are spec-coverage gaps; correct them inline before commit.

- [ ] **Step 4: Update the ADR index**

Update the active ADR index only. Frozen architecture-pack indexes are
historical snapshots and should not be replayed as part of current ADR work.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/adr/021-config-content-ref.md
git commit -m "docs(adr): land ADR-021 — audited content injection via widened blob_ref

Captures the spec §4 design decisions as ADR sub-decisions with rationale.
Numeric NFRs cited against the VAL-data appendix. Closure rule \"no new
ref forms without ADR amendment\" formalised. Phase ordering pivot
documented.

Closes design portion of elspeth-fdebcaa79a (P1)
Refs: elspeth-fdebcaa79a"
```

---

## Task 5: Open the P1 PR

- [ ] **Step 1: Push branch and open PR**

```bash
git push -u origin <branch-name>
gh pr create --title "docs(adr): ADR-021 audited content injection (widened blob_ref)" --body "$(cat <<'EOF'
## Summary

- Lands ADR-021 (\"Audited content injection — widened \`blob_ref\` with \`mode\` discriminator\") with all design closures from the spec
- Adds VAL-data sampler script and appendix; numeric NFR caps cited against measured byte-length distributions
- Documents the phase-ordering pivot (runtime side ships before composer side; otherwise composer-green / runtime-red is the very Shape 9 footgun this work closes)

Markdown-only PR. P2 onward carries production code.

## Test plan

- [ ] \`pytest tests/unit/scripts/analysis/test_blob_inline_val_data.py\` — percentile math contract
- [ ] Manual: re-run \`scripts.analysis.blob_inline_val_data\` against the staging audit DB if available; confirm cap recommendations match the appendix
- [ ] Read-through: ADR-021 sub-decisions cover every row in spec §4

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Close the planning portion of the epic comment**

After PR merges, add a comment to elspeth-fdebcaa79a:

```bash
filigree add-comment elspeth-fdebcaa79a "P1 (ADR + VAL data) merged. ADR-021 ships the design closures; numeric caps grounded in VAL-data appendix. Proceeding to P2 (L0 contracts + L1 resolver)."
```

---

## Done conditions

P1 is done when:

1. `docs/architecture/adr/021-config-content-ref.md` exists and is approved.
2. `docs/architecture/adr/021-config-content-ref-val-data.md` exists with populated cells.
3. `scripts/analysis/blob_inline_val_data.py` exists and its unit test passes.
4. PR is merged to RC5-UX (or successor).
5. F-2 and F-3 follow-up issues are filed and cited in the PR description.

Move to `2026-05-03-config-content-ref-phase-2-l0-l1.md` only after P1 is merged.
