# Composer Redaction Policy Changes — Governance Guide

**Audience.** Engineers modifying the composer redaction manifest
(`MANIFEST` in `src/elspeth/web/composer/redaction.py`) or the policy
snapshot (`tests/unit/web/composer/redaction_policy_snapshot.json`).

**Status.** Current as of spec rev-5 rev-2 iteration (2026-05-12).
See `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md`
§4.4.5 for the authoritative control definition. The review findings that
shaped this guide are summarized in that spec; generated reviewer sidecars are
not required for day-to-day policy changes.

---

## When does this guide apply?

Any PR that modifies:

- `src/elspeth/web/composer/redaction.py` (manifest entries, summarizers,
  `MANIFEST` dict)
- `src/elspeth/web/composer/tools.py` (promoted handler argument models)
- `tests/unit/web/composer/redaction_policy_snapshot.json`
- `tests/unit/web/composer/test_adequacy_guard.py`
- `tests/unit/web/composer/test_walk_model_schema.py`

...will trigger the `composer-redaction-gate` CI workflow at
`.github/workflows/composer-redaction-gate.yml`.

---

## Mechanical gates

### 1. Policy-hash snapshot (primary control)

`tests/unit/web/composer/redaction_policy_snapshot.json` contains a
SHA-256 hash per manifest entry. The `test_adequacy_guard.py` test
module asserts the live manifest's hashes match the committed file.

Any change to a manifest entry — `Sensitive[T]` annotation removed,
declarative key set narrowed, summarizer replaced, structured reason
rewritten — flips the entry's hash. This change must be committed to
the snapshot file before CI goes green.

To regenerate the snapshot after editing the manifest:

```bash
.venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py
```

### 2. Direction-aware label-gate CI step

The `composer-redaction-gate` CI workflow fires when the snapshot file
changes on a PR. It performs a **direction-aware** check:

- Computes the `sensitive_path_count` per manifest entry (the number of
  `Sensitive`-annotated field paths as enumerated by `walk_model_schema`)
  in both the PR-head snapshot and the `main` snapshot.
- If the total count across changed entries **decreased** (a weakening):
  the PR MUST carry the `policy-weaken-justified` label. Using
  `policy-strengthen` on a weakening diff fails CI.
- If the total count **increased or stayed the same** with a hash change
  (a strengthening or neutral semantic shift): the PR MUST carry the
  `policy-strengthen` label. Using `policy-weaken-justified` on a
  strengthening diff fails CI with the message "snapshot diff shows no
  coverage reduction; do not use `policy-weaken-justified` for this
  change."

### 3. Adequacy guard (CI-time, five assertions)

`tests/unit/web/composer/test_adequacy_guard.py` runs in the standard
pytest gate and asserts:

1. Registry-manifest set equality (every dispatch registry tool name
   has exactly one manifest entry; no orphans).
2. Per-entry shape walk via the shared `walk_model_schema` iterator.
3. Mass-copy uniqueness of declarative `HandlesNoSensitiveDataReason`
   text (no two tools may share the same `why_arguments_safe` or
   `why_responses_safe` string).
4. Policy-hash snapshot equality (live manifest hashes == committed
   snapshot file).
5. `extra="forbid"` on all type-driven argument models (prevents
   `arguments_canonical` / walker discrepancy).

---

## Which label should I use?

| Change | Correct label |
|---|---|
| Added a `Sensitive()` annotation to a field | `policy-strengthen` |
| Added a new tool to a registry and gave it a manifest entry | `policy-strengthen` |
| Added a new `sensitive_argument_key` | `policy-strengthen` |
| Removed a `Sensitive()` annotation | `policy-weaken-justified` |
| Removed a `sensitive_argument_key` | `policy-weaken-justified` |
| Broadened `sensitive_response_keys` to include previously-redacted keys | depends on net count change |
| Replaced a summarizer with a more informative one (same coverage) | `policy-strengthen` (hash flips) |
| Editorial change to `HandlesNoSensitiveDataReason` text only | `policy-strengthen` (hash flips via text hash) |

When using `policy-weaken-justified`, the PR body MUST include a section
headed exactly:

```
## Redaction policy weakening rationale
```

The CI step grep-asserts this section header exists (the workflow's
Python check uses substring match on the body string — exact phrasing is
load-bearing).

---

## What the weakening rationale section MUST contain

A label is a routing signal; the rationale is the substantive control.
The CI gate cannot inspect the *content* of the rationale section — that
is the auditor's job. A complete rationale answers all of the following:

1. **What was weakened, and how?** Name the manifest entry or entries
   and the specific path(s) removed or replaced. Quote the
   `sensitive_path_count` before and after.

2. **Why is this weakening *safe*?** From an auditor's perspective, not
   the implementer's. The default presumption is that redaction
   coverage exists for a reason; removing it requires showing the
   reason no longer applies. Acceptable arguments include:

   - The path was redundant because an upstream component already
     redacts the same field before the data reaches this tool.
   - The path covered a field that has been removed from the tool's
     input schema entirely — there is no data flowing through it any
     more.
   - The path was a copy-paste artefact that never matched real data
     (verifiable from run history showing zero redactions on that
     path).

   Arguments that are NOT sufficient:

   - "The path was annoying in tests" — annoyance is not an audit
     argument.
   - "I want to log this value for debugging" — see
     `logging-telemetry-policy`; this needs an audit or telemetry
     channel, not a removed redaction.
   - "Nothing sensitive ever appears here" — claims about absence of
     sensitive data require evidence, not assertion.

3. **What new mitigation replaces the prior coverage?** If the
   weakened path is being replaced (e.g., redaction moved upstream),
   name the replacement and cite its tests. If no replacement exists,
   explain why none is needed.

4. **Bulk-editorial-PR saturation acknowledgement.** This is the most
   important auditor-facing assertion. A PR that touches many files
   with a single weakening tucked in among them is exactly the failure
   mode this gate is designed to detect. The rationale section MUST
   include a sentence of the form:

   > This PR includes a weakening of the redaction policy. The
   > weakening is the substantive change, not an incidental
   > side-effect of refactoring. I am not relying on reviewer fatigue
   > to land it.

   If the author cannot make this assertion honestly, the PR should
   be split: refactoring in one PR with `policy-strengthen` (or no
   redaction label if the manifest is untouched), and the weakening
   alone in a small follow-up PR with the rationale.

---

## Escalation: when a weakening rationale is insufficient

If a reviewer (or, in a single-owner repo, the future-self auditor
reading this in CI logs) believes a weakening rationale is insufficient,
the correct response is to **re-open the PR with a stricter posture**,
not to merge with a warning attached.

Options for "stricter posture":

- Decline the weakening and keep the existing path.
- Replace the weakening with an explicit, narrower mitigation (e.g.,
  remove one specific sensitive path rather than the entire manifest
  entry).
- Split the PR so the weakening lands separately with its own
  reviewable diff.

The bar to clear is "would this hold up under formal inquiry?" — see
`CLAUDE.md` § *Auditability Standard*. If the answer is "probably, but
I'd rather not have to defend it," the PR isn't ready.

---

## Single-owner governance note

The label-gate (`policy-strengthen` / `policy-weaken-justified`) is a
**mechanical CI control**, not a code-review requirement enforced by a
human gatekeeper.

On a multi-person repository, CODEOWNERS would gate label application
via review approval from a designated security-review team. **On a
single-owner or personal-account repository, the label gate is
self-bypassing** — the repository owner can apply any label without
external review, and there is no second human in the loop.

This is an acknowledged governance gap documented in spec rev-5 §4.4.5
and the rev-2 review JSON under `M_governance_single_owner`.

**Consequence for single-owner repos:** snapshot-hash flips must be
treated as warranting a manual sanity-check of the diff direction before
merging, even when CI is green. Specifically:

1. Open the snapshot diff and verify the `sensitive_path_count` deltas
   match the intent of the change.
2. If any entry's count decreased, verify the `policy-weaken-justified`
   label is accurate — not applied out of habit or error.
3. If the PR description includes a "Redaction policy weakening
   rationale" section, verify it accurately describes the change.

When the project migrates to an organisation-owned repo, CODEOWNERS team
routing becomes available and can be added as defense-in-depth on top of
the label gate.

---

## References

- Spec §4.4.5 — label-gate CI step definition and governance note
- Spec §4.4.3 — policy-hash snapshot (hash semantics; known false-negative
  class for closure-state mutations)
- Spec §4.2.3 — summarizer purity requirement
- Spec §4.2.8 — `arguments_canonical` posture (a) intentional raw
- Spec §12.3 — rev-2 reviewer finding traceability, including the label-gate
  direction and single-owner governance finding
