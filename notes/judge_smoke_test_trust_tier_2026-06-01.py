#!/usr/bin/env python
"""Judge verdict smoke-test for the 2026-06-01 trust-tier prompt update.

Proves the cicd-judge *applies* the corrected "persisted external-origin config
re-read is Tier-3" doctrine — not merely that the prompt text contains it. (The
test suite only checks prompt presence/structure; it never exercises a verdict.)

FOUR cases across two code sites:

Cases 1 & 2 — persisted-config tier (SAME R6 quarantine guard on a persisted
`source.options` re-read in compute_proof_diagnostics):
  Case 1 — CORRECT (persisted external-origin → Tier-3, guard is honest):
           expect ACCEPTED
  Case 2 — THE BUG'S REASONING ("first-party Tier-1, guard is redundant"):
           expect BLOCKED

Cases 3 & 4 — the 2026-06-01 role clarification, as a MINIMAL PAIR. Both guard a
function's return with `isinstance(...)` → raise, both use the same "offensive /
informative crash / Plugin Ownership" rationale language; they differ ONLY in
whether the return type is guaranteed by code we control:
  Case 3 — return comes through a structural typing.Protocol (sink.write()), so
           the type is NOT runtime-enforced → the guard is the sole enforcement
           point → PRESCRIBED FORM → expect ACCEPTED
  Case 4 — return comes from our own internal helper declared -> dict[str, int],
           so the type IS guaranteed by code we control → the guard is REDUNDANT
           → expect BLOCKED ("it raises" must not launder a redundant guard)

If Case 2 is ACCEPTED the persisted-config misclassification isn't fixed; if
Case 3 is BLOCKED the role clarification didn't land (or sink.py is a genuine
prompt gap); if Case 4 is ACCEPTED — i.e. the judge accepts BOTH halves of the
pair — the role text loosened the judge into pattern-matching "isinstance+raise
= accept", which is a PROMPT failure, not a test flaw.

Run (operator env; OpenRouter transport pins temperature=0):

  env OPENROUTER_API_KEY=sk-or-... PYTHONPATH=elspeth-lints/src \
      .venv/bin/python notes/judge_smoke_test_trust_tier_2026-06-01.py

Exit 0 only if ALL FOUR verdicts match expectation. No HMAC key needed
(read-only; writes nothing).
"""

from __future__ import annotations

import sys

from elspeth_lints.core.judge import (
    JUDGE_POLICY_HASH,
    TRANSPORT_OPENROUTER,
    JudgeRequest,
    call_judge,
)

# A faithful excerpt of the fixed site: a Tier-3 re-read of persisted, operator/
# composer-authored source config, caught into a per-blob diagnostic (R6: caught
# without re-raise).
EXCERPT = '''\
    if facts.source_kind in {"csv", "json", "jsonl"}:
        # source.options is re-read here from persisted composer session state.
        try:
            schema_config = get_raw_schema_config(source.options, owner=f"source:{source.plugin}")
        except ValueError as exc:
            diagnostics.append(
                _csv_source_schema_config_error_diagnostic(blob_id=blob_id, facts=facts, exc=exc)
            )
            schema_config = None  # diagnostic recorded; skip dependent analysis for this blob
'''

R6_DEFINITION = (
    "R6 (silent-except): an exception is caught and handled without re-raising. "
    "Flags swallowing an error to a default/continue instead of letting it surface. "
    "Legitimate only at a genuine Tier-3 boundary where the catch records the failure "
    "(quarantine/diagnostic) rather than silently degrading."
)

CORRECT_RATIONALE = (
    "source.options is composer/operator-authored pipeline configuration re-read from "
    "persisted session state. We authored the SourceSpec container and the schema, but "
    "the values are external-origin. Validation at config-load time was in-flight only; "
    "re-reading it here is a fresh Tier-3 boundary (the store is mutable and the validating "
    "model can drift between write and read). The ValueError is caught and turned into a "
    "per-blob blocking diagnostic (quarantine: record what we found, skip dependent analysis) "
    "rather than crashing the preview_pipeline tool. This is the honest Tier-3 quarantine "
    "pattern, not silent recovery."
)

BUG_RATIONALE = (
    "source.options is a first-party (Tier-1) Mapping that we authored and validated once "
    "at the config-loading boundary. It lives in our own SourceSpec dataclass read back from "
    "our own database, so it is our data. A ValueError here is a genuine Tier-1 invariant "
    "break and should be allowed to propagate per the offensive-programming policy; catching "
    "it is redundant defensive handling on trusted first-party data."
)


# ── Cases 3 & 4 — the three-way role clarification (2026-06-01) ──────────────
# A MINIMAL PAIR isolating the one variable the whole exercise turned on:
# is the guarded type GUARANTEED by code we control, or not? Both cases guard a
# function's return with isinstance->raise and both dress the rationale in the
# same "offensive / informative crash / Plugin Ownership" language. They differ
# ONLY in the callee:
#   Case 3 — sink.write() returns through a structural typing.Protocol; the
#            return type is NOT runtime-enforced → the guard is the SOLE
#            enforcement point → PRESCRIBED FORM → ACCEPT.
#   Case 4 — our own internal helper declared -> dict[str, int]; the return type
#            IS guaranteed by code we control → the guard is REDUNDANT →
#            disposition 2 (GENUINE VIOLATION) → BLOCK. "It raises" must NOT
#            launder a redundant guard.
# If the judge ACCEPTs BOTH, it is pattern-matching "isinstance+raise = accept"
# and the role text loosened it — that is a PROMPT failure, not a test flaw.
SINK_EXCERPT = '''\
    write_result = sink.write(rows, ctx)
    # First-party plugin-contract guard: SinkWriteResult is this system-owned
    # sink's typed return contract (Plugin Ownership), so a wrong type is a
    # plugin bug we crash on.
    if not isinstance(write_result, SinkWriteResult):
        raise PluginContractViolation(
            f"Sink '{sink.name}' returned {type(write_result).__name__}, "
            f"expected SinkWriteResult. This is a sink plugin bug."
        )
'''

# Case 4: the guarded value comes from OUR OWN internal helper with a concrete,
# controlled return annotation — no Protocol, no plugin boundary, no external
# origin. The type is already guaranteed by code we control.
CONTROLLED_RETURN_EXCERPT = '''\
    # _collect_counts is our own module-internal helper, declared
    # -> dict[str, int]; its return type is guaranteed by code we control.
    result = self._collect_counts(rows)
    if not isinstance(result, dict):
        raise TypeError(
            f"_collect_counts must return dict, got {type(result).__name__}"
        )
    return result
'''

R5_DEFINITION = (
    "R5 (isinstance shape-guard): an isinstance() check on first-party/typed "
    "data. Flags re-checking the shape of a value whose type our own code "
    "already guarantees, instead of accessing it directly and letting a "
    "contract violation crash naturally. Legitimate only where the type is NOT "
    "actually guaranteed at runtime (a structural Protocol, an unenforced "
    "annotation, an external-origin value) so the check is real enforcement."
)

SINK_STATED_RATIONALE = (
    "SinkProtocol is a structural typing.Protocol, not a runtime-enforced base "
    "class — Python does not check that sink.write() actually returns a "
    "SinkWriteResult at runtime, and plugins are system-owned code whose wrong "
    "return type is a bug we must CRASH on (Plugin Ownership), made maximally "
    "informative. This isinstance->raise is therefore the SOLE runtime "
    "enforcement point of that contract, not a redundant re-check of a type the "
    "code already guarantees: nothing upstream guarantees it. It coerces "
    "nothing and suppresses nothing — it raises. PRESCRIBED FORM (offensive "
    "crash), not the defensive isinstance-to-suppress R5 targets."
)

# Same offensive language as Case 3, but the type IS controlled-and-guaranteed,
# so a correct judge must BLOCK: the raise does not legitimize a redundant guard.
CONTROLLED_RETURN_RATIONALE = (
    "This is an offensive isinstance->raise, not coercion or suppression: a "
    "wrong return type is a bug we crash on with a maximally informative error "
    "per Plugin Ownership and the offensive-programming policy. We surface the "
    "actual type received rather than silently degrading."
)


def _verdict_str(verdict: object) -> str:
    return str(getattr(verdict, "value", getattr(verdict, "name", verdict))).upper()


def run_case(
    label: str,
    rationale: str,
    expected: str,
    *,
    file_path: str = "web/composer/tools/generation.py",
    rule_id: str = "R6",
    symbol: str = "compute_proof_diagnostics",
    surrounding_code: str = EXCERPT,
    rule_definition: str = R6_DEFINITION,
) -> bool:
    request = JudgeRequest(
        file_path=file_path,
        rule_id=rule_id,
        symbol=symbol,
        fingerprint="fp=smoketest-2026-06-01",
        rationale=rationale,
        surrounding_code=surrounding_code,
        rule_definition=rule_definition,
    )
    resp = call_judge(request, transport=TRANSPORT_OPENROUTER)
    got = _verdict_str(resp.verdict)
    ok = got == expected
    print(f"\n=== {label} ===")
    print(f"  expected: {expected}")
    print(f"  verdict:  {got}   {'PASS' if ok else 'FAIL'}")
    print(f"  confidence: {resp.confidence}")
    print(f"  judge rationale: {resp.judge_rationale}")
    return ok


def main() -> int:
    print(f"Policy hash under test: {JUDGE_POLICY_HASH}")
    results = [
        run_case("Case 1 — correct (persisted external-origin → Tier-3)", CORRECT_RATIONALE, "ACCEPTED"),
        run_case("Case 2 — the bug's reasoning (first-party Tier-1)", BUG_RATIONALE, "BLOCKED"),
        # Cases 3 & 4 — minimal pair for the 2026-06-01 role clarification.
        # Same construct (isinstance->raise on a function's return), same
        # offensive rationale language; differ ONLY in guaranteed-vs-not.
        run_case(
            "Case 3 — isinstance->raise on a STRUCTURAL-Protocol return (unguaranteed)",
            SINK_STATED_RATIONALE, "ACCEPTED",
            file_path="engine/executors/sink.py",
            rule_id="R5",
            symbol="SinkExecutor.write",
            surrounding_code=SINK_EXCERPT,
            rule_definition=R5_DEFINITION,
        ),
        run_case(
            "Case 4 — isinstance->raise on OUR controlled -> dict return (redundant) → BLOCK",
            CONTROLLED_RETURN_RATIONALE, "BLOCKED",
            file_path="web/composer/tools/sessions.py",
            rule_id="R5",
            symbol="_summarize_recipe_counts",
            surrounding_code=CONTROLLED_RETURN_EXCERPT,
            rule_definition=R5_DEFINITION,
        ),
    ]
    passed = all(results)
    print(f"\n{'ALL PASS' if passed else 'FAILED'}: "
          f"{sum(results)}/{len(results)} verdicts matched expectation.")
    if not passed:
        print("The prompt does NOT correctly adjudicate — do not re-sign the "
              "corpus until this passes. Cases 1/2 = persisted-config tier; "
              "Cases 3/4 = minimal pair: ACCEPT an isinstance->raise on an "
              "UNGUARANTEED (structural-Protocol) return, BLOCK the same form on "
              "a return our own code GUARANTEES. Accepting BOTH = prompt loosened.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
