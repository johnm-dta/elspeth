# Tier-Model Allowlist BURNDOWN — agent instructions

Your job is to **EDIT CODE** so a tier-model finding no longer exists. Your
output is a **diff**, never a rationale. If you are not changing code in
`src/elspeth/`, you are doing the WRONG THING and must stop.

Every finding you own must end in exactly one of two states:

1. **REMOVED** — the flagged construct (`isinstance` / `except` / `.get` /
   `getattr` / `hasattr`) is GONE: you deleted it, accessed the field directly,
   normalized the type upstream, or restructured so the pattern doesn't exist.
2. **CLARIFIED** — at a GENUINE external boundary you replaced the silent /
   defensive form with the SANCTIONED explicit form (record a diagnostic /
   return an explicit error result / quarantine the row), so the code is
   self-evidently a recorded boundary, not a silent swallow.

## FORBIDDEN (this is the entire point)

- **Do NOT write, refresh, or propose an allowlist `justify` / rationale to keep
  a suppression alive.** Re-justifying is working around the problem. If you
  catch yourself drafting a rationale to preserve a suppression, STOP.
- **Do NOT add a new allowlist entry. Do NOT weaken the rule or the judge.**
- **Do NOT run the judge or a reaudit sweep.** The only command you run is the
  static check (below), to confirm your edit cleared the finding.

## Decision per finding (tier doctrine — load-bearing, do NOT violate)

- **Guard on a type OUR OWN CONTROLLED code already guarantees** (our function's
  concrete return, a locally-built value, a dataclass we constructed)? →
  **DELETE the guard.** Access directly; let a contract violation crash
  naturally.
- **Catching an exception from OUR OWN code / a first-party value** (audit write,
  `canonical_json` on our dispatch output, a typed error from our own secret
  store)? → **REMOVE the catch. Let it crash** (offensive programming).
- **`.get(k, default)` / membership-then-raise on data WE control?** → direct
  subscript `d[k]`; let `KeyError` fire (or convert to a typed error with
  `from exc`).
- **GENUINE external boundary** (source row, HTTP/LLM response, parsed
  subprocess output, persisted external-authored config re-read)? The boundary
  is REAL — do NOT delete it. **CLARIFY**: replace the silent swallow / bare
  default with an EXPLICIT recorded outcome (quarantine, diagnostic, error
  result the caller routes). Removing the construct must NOT reintroduce a crash
  on external data.
- **Best-effort EPHEMERAL telemetry / progress** (WS broadcast, progress
  events)? Per the telemetry primacy order these are NOT Landscape facts.
  Restructure so the drop is EXPLICIT and intentional (a single named
  best-effort path), not a bare `except: pass/continue/return None`. If a
  pipeline-relevant fact is being lost, route it to audit/telemetry instead.
- **Control-flow SENTINEL** (`except TimeoutError: continue` polling,
  `suppress(CancelledError)` after `asyncio.shield`)? Make it explicit that it's
  a signal, not an error. If the rule still fires and the construct is genuinely
  correct AND irreducible, that is the ONE case you ESCALATE (see below) — never
  allowlist it yourself.

## Tier rules you must NOT break while burning down

- **Tier 1** (our authored audit/checkpoint values): crash on anomaly — never
  add a swallow. Removing a guard is fine ONLY if natural access still crashes
  on the anomaly — watch NON-crashing anomalies (an empty-string `""` used as a
  payload-store key is a valid `str` that won't crash; that guard stays/clarifies).
- **Tier 2** (post-validated pipeline data): no coercion; wrong type = upstream
  bug → crash.
- **Tier 3** (external): validate/coerce/quarantine at the boundary, record
  absence as `None`, never fabricate.

## Workflow

1. Read the `tier-model-deep-dive` skill before touching trust-boundary code.
2. For each finding: edit `src/elspeth/...`, then run:
   ```
   env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
     check --rules trust_tier.tier_model --root src/elspeth \
     --allowlist config/cicd/enforce_tier_model
   ```
   The finding for your file MUST be gone (or now allowlist-clean).
3. Run the file's tests (`pytest` on the relevant module) — your edit must not
   break them; if a test pinned the old buggy behavior, fix the test.
4. **Stage ONLY the files you edited.** Do NOT `git restore`/`git checkout`
   other sessions' files. Do NOT touch the allowlist dir or
   `fingerprint_baseline.json` — the integrator reconciles those centrally.
5. Report: per finding, the diff summary + confirmation the check cleared, and a
   single ESCALATION list (file:line + one sentence) for any genuinely
   irreducible exception the operator must rule on.
