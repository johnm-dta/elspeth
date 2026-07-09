# PDR-0002 — Turnkey tier-model signing: adopt the stage→sign-bundle seam, retire the runbooks

Date: 2026-06-28   Status: proposed   Author: claude (product-checkpoint)   Owner sign-off: pending (tier-model/HMAC signing is an escalate-before-acting domain)
Supersedes: none   Related: vision.md (authority grant + red-gate guardrail), metrics.md (red-gate guardrail), docs/judge-signature-handoff.md, tracker elspeth-281582acc9

## Context

The `trust_tier.tier_model` allowlist seals each judge-gated suppression with an
operator-held HMAC signature. Acquiring, repairing, and rotating those signatures
was a manual, per-release ceremony (the `notes/060-*` runbooks +
`scripts/cicd/sign_accept_backlog.py`): tedious, error-prone, and structurally
risky — nothing but discipline kept the symmetric key out of an agent's reach, so
the [O1] custody rule was a *policy*, not a *control*. The owner directed building
a turnkey replacement this session.

## Options considered

1. **Keep the manual runbooks.** Pro: proven, no new code. Con: tedious and
   per-release; key-isolation stays a policy an agent could violate; the forgery
   surface (agent hand-writes `ACCEPTED`, signs, passes every gate) remains open.
2. **[chosen] Agent-stages / operator-signs seam.** The agent stages a key-free
   worklist via the `elspeth-judge` MCP server (5 fail-closed tools); the operator
   fires `sign-bundle` / `rekey` with the key; firing re-derives every binding from
   the live tree and aborts before any write on staleness. Pro: structurally
   enforces operator-only custody ([O1]) — the MCP surface fails closed if the key
   is present; signing never runs in CI (standing meta-test); the runbook ceremony
   collapses to one command. Con: the rotation / judge-gated interaction is subtle
   (a scan-time crash, caught and closed in adversarial review); live e2e not run.
3. **Asymmetric signatures (sign-private / verify-public).** Pro: agents could
   verify but not sign, structurally. Con: does not close the *new-finding* forgery
   surface; larger change. Deferred (design spec §9).

## The call

Built + shipped Option 2 to `release/0.7.0` (feature `6e0d66f9`; design spec
`b87829bc3`; implementation plan `d59fbccdd`; spec/plan refinements `af30dfc6e` /
`786a609ab`). All four review lenses converged (0 blocking / 0 major); 70 new tests
green; all five security invariants non-vacuously pinned. The obsoleted runbooks and
`sign_accept_backlog.py` were removed. Status is **proposed, not accepted**: the build
was authorized delivery, but tier-model/HMAC signing is an escalate-before-acting
domain, so three items gate to the owner — (a) adopt this as the *sanctioned* signing
workflow; (b) confirm the runbook removal stands; (c) authorize the push of
`release/0.7.0`. The live-judge e2e (real key + real LLM) has not run.

## Rationale

This converts the metrics.md red-gate guardrail ("green only on signed state; held by
operator") from a policy into a structural control, directly serving the vision's "no
gate blessed without provenance." The owner drove the work end-to-end (brainstorm →
spec approval → plan approval → directed delivery), so the escalation's *intent* —
owner awareness and consent in the signing domain — was satisfied in-loop; this PDR
records the decision and the residual sign-off rather than asserting autonomous action.

## Reversal trigger

Supersede this PDR and restore the runbooks if **any** of: the live-judge e2e fails;
the first real signing round cannot complete via `sign-bundle` without falling back to
manual editing; or the operator finds the new flow weakens custody. Recovery:
`sign_accept_backlog.py` is git-recoverable; the `notes/060-*` runbooks were gitignored
scratch (gone from disk). Also revisit at the monthly grant review (2026-07-14).
