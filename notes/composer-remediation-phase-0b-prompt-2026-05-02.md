# Phase 0.b prompt — Composer gate primitive root-cause investigation and fix

**Use this as a self-contained prompt for a fresh session or subagent.** It has no
dependencies on prior conversation context.

---

## Prompt begins

You are picking up Phase 0.b of the composer remediation program. Read these
two files before doing anything else — they explain why this work exists:

- `notes/composer-remediation-program-2026-05-01.md` — the program doc
- `notes/composer-llm-eval-2026-05-01.md` — the source audit (search for
  "gate primitive" and "S3 v1, v2, v4")

Tracking issue: **`elspeth-209b7e3a2b`** *(P1 bug, currently `triage`,
blocked_by `elspeth-2c3d63037c`)* — "Composer gate primitive crashes — root
cause fix (with new diagnostics from elspeth-2c3d63037c)".

### Phase 0.a status (do this check first)

Phase 0.a (`elspeth-2c3d63037c`, the diagnostic surface) was due to close in
the session immediately preceding this one via the bug-verification-protocol +
closure-summary path. **Verify before proceeding:**

```bash
filigree show elspeth-2c3d63037c
filigree show elspeth-209b7e3a2b
```

Expected state:
- `elspeth-2c3d63037c` — `closed`. Closure summary cites the four
  regression tests in `tests/unit/web/sessions/test_routes.py` (around lines
  4034, 4078, 4136, 4180) and explicitly notes the (a) reading of the Phase 0
  acceptance gate language: "regression test reproducing the original 500
  path" = "structured-errors fix has regression coverage" (NOT "the gate
  primitive 500 itself no longer occurs" — that's THIS issue's territory).
- `elspeth-209b7e3a2b` — `triage`, `is_ready: true` after 0.a closure
  removes the dependency edge.

If 0.a is not closed, stop and ask the operator. The agent prompt that closes
0.a is at the end of `notes/composer-remediation-phase-3-prompt-2026-05-02.md`'s
review chain (or just refer to it as "the bug-verification protocol on
`test_state_data_carries_structured_errors_before_save_for_atomicity` and
closure with the four-test crosswalk").

### Critical context: staging is on this machine

`elspeth.foundryside.dev` is a **source-checkout systemd/Caddy deploy on this
local machine** (NOT a remote system; NOT `scripts/deploy-vm.sh`). This means:

- The reproducer can be run with local `curl` against `https://elspeth.foundryside.dev/api/...`
- Tracebacks land in `journalctl -u elspeth-web.service` on this machine
- Restart with `systemctl restart elspeth-web.service` if you need to redeploy
- The deploy reflects whatever's in the source checkout at HEAD on the
  current branch — verify with `git log -1` before running the reproducer

Do not treat staging as a remote system. The whole investigation is local.

### What 0.b is and is not

**0.b is**: identify the actual server-side defect that causes
`POST /api/sessions/{sid}/messages` to return HTTP 500 with
`error_type: composer_plugin_error` whenever the LLM attempts to add a
`gate` node, and land a fix.

**0.b is not**:
- A diagnostic-surface fix (that's 0.a, already closed).
- A frontend rendering issue (no UI work in scope here).
- A composer/runtime agreement test (those go in the agreement suite per
  Phase 3 conventions; the regression test for the gate-primitive shape
  belongs in this work, not as an extension of Phase 3).

### Step 1 — Capture the staging reproducer data

The issue's "Acquiring the data" section lists the reproducer steps. Execute
them and capture the structured `validation_errors` from the response, plus
the journald output. The full reproducer script:

```bash
# Setup
HOST="https://elspeth.foundryside.dev"
USER="dta_user"
PASS="<get from operator if needed — likely in a local secrets file>"

# 1. Login
TOKEN=$(curl -sk -X POST "$HOST/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$USER\",\"password\":\"$PASS\"}" \
  | jq -r .access_token)

# 2. New session
SID=$(curl -sk -X POST "$HOST/api/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' | jq -r .id)

# 3. Inline blob — small CSV with the customer_tier column
curl -sk -X POST "$HOST/api/sessions/$SID/blobs/inline" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"tickets.csv","content":"ticket_id,customer_tier,subject\n1,enterprise,login broken\n2,free,how do I reset password\n3,pro,billing question\n"}'

# 4. Capture journal cursor BEFORE sending the gate-creation message
JOURNAL_CURSOR=$(journalctl -u elspeth-web.service -n 0 --show-cursor 2>&1 | tail -1)

# 5. Send the gate-creation message
RESPONSE=$(curl -sk -X POST "$HOST/api/sessions/$SID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Build a workflow that splits rows by customer_tier — enterprise to outputs/high.jsonl, others to outputs/low.jsonl. Use a gate."}')

echo "$RESPONSE" | jq .

# 6. Read the journal entries that landed during the message handling
journalctl -u elspeth-web.service --after-cursor="$JOURNAL_CURSOR" --no-pager

# 7. Pull the persisted state to confirm the half-written-node corruption
curl -sk -X GET "$HOST/api/sessions/$SID/state" \
  -H "Authorization: Bearer $TOKEN" | jq .
```

**Save the captured outputs to `notes/composer-phase-0b-staging-capture-<DATE>.md`** — this is the
durable artefact for the issue's triage. Include:

- The 500 response body (specifically the `validation_errors` list)
- The journald output (specifically `exception_class=`, `exception_message=`,
  and `frame=` lines from the structured log event 0.a's fix added)
- The persisted state showing the half-written gate node
- The session ID for reproducibility

### Step 2 — Update the issue with the captured data

```bash
filigree add-comment elspeth-209b7e3a2b "$(cat <<'EOF'
## Staging reproducer captured

[paste captured `validation_errors` JSON]

## Journald exception trail

```
[paste exception_class= / exception_message= / frame= lines]
```

## Persisted state corruption

[paste relevant slice of /state response showing plugin=null / partial node]

## Triage outcome

Based on the captured `exception_class`, the applicable fix branch is:

- [ ] AttributeError-in-validation — the composer's gate-node validation walks an attribute the model doesn't expose
- [ ] YAML-emission — `composer/yaml_generator.py` fails to render the gate node into a runnable shape
- [ ] instantiate_runtime_plugins — the engine-side plugin instantiation chokes on the composer's gate representation
- [ ] Other — explain

Capture artefact: notes/composer-phase-0b-staging-capture-<DATE>.md
EOF
)"
```

Then transition the issue: `triage` → `confirmed` (assuming the reproducer
fired) or `open` if you can't reproduce.

### Step 3 — Land the fix

Once the captured data identifies the fix branch, land the structural fix.
Engineering rules from `CLAUDE.md` apply:

- **Plugins are system code, not user code.** A plugin bug is a CRASH
  candidate during development, not a candidate for defensive `getattr`
  workarounds in the engine.
- **No legacy-code shims.** If the fix means renaming a method or changing a
  signature, change every call site in the same commit.
- **Trust boundaries unchanged.** The composer's input is internal trusted
  state (Tier 1) — its bugs crash; user-supplied options inside the gate
  node are Tier 3 — those quarantine.
- **Layer discipline.** The fix is most likely in `src/elspeth/web/composer/`
  (L3) or `src/elspeth/engine/` (L2). If you find yourself adding a
  TYPE_CHECKING import upward across layers, restructure instead — the
  CLAUDE.md "When a New Cross-Layer Need Arises" section lists the
  resolution priority order.

The fix MUST also restore the **atomic state mutation** invariant the
program doc names at line 49: failed compose mutations leave state
untouched (no version bump, no half-written nodes). 0.a may or may not
have addressed this — verify in the production code before declaring the
fix complete. If the version counter still bumps on a guaranteed-invalid
gate-creation request after your fix, the fix is incomplete.

### Step 4 — Write the regression test (AFTER the fix lands)

**Do not write this test before the fix lands.** Per the Phase 3 prompt's
anti-pattern guidance (`notes/composer-remediation-phase-3-prompt-2026-05-02.md`
lines 254-258): a speculative test against an unfixed bug locks in the
wrong behaviour and becomes harder to remove than to write fresh.

After the fix lands, add a test in
`tests/integration/pipeline/test_composer_runtime_agreement.py` as **Shape 8**
in the closed-list registry. The test should:

1. Drive the same gate-node creation path the staging reproducer hit, but
   in-process via `validate_pipeline()` and `Orchestrator.run()` (not via
   HTTP — agreement-suite tests are in-process, not E2E).
2. Assert the gate-node creation succeeds without raising.
3. Assert the persisted state's version counter is correctly bumped (one
   bump per successful mutation; zero bumps on a guaranteed-invalid mutation).
4. Apply the **bug-verification protocol** documented in the suite's
   module docstring (lines 62-74): manually revert the fix, confirm the
   test fails with the captured `exception_class`, restore. Document the
   protocol verbatim in the test's docstring.

Update the closed-list shape registry in the module docstring (lines 14-56)
to add Shape 8 with the originating eval session/run IDs from the
2026-05-01 audit (S3 = session 98573481-e8bc-4a03-8467-d3a86effcd56), the
closing issue (`elspeth-209b7e3a2b`), the test class, and the commit hash.

### Step 5 — Verify and close

```bash
# Full test suite
.venv/bin/python -m pytest tests/integration/
.venv/bin/python -m pytest tests/unit/

# Type and lint
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/ tests/

# Tier-model enforcement
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model

# 5x flake check on the new test
for i in {1..5}; do
  .venv/bin/python -m pytest tests/integration/pipeline/test_composer_runtime_agreement.py -k Shape8 -v || break
done

# Re-run the staging reproducer to confirm the original 500 is gone
# (script from Step 1 — should now return either success or a structured 4xx)
```

Close the issue with a summary that:
- Names the root-cause fix branch identified
- Cites the commit hash with the fix
- Cites the test file + class + line number for Shape 8
- Includes the bug-verification protocol result (which line was reverted,
  which exception class fired)
- Notes that the staging reproducer now returns success/4xx (paste the
  new response body)

### Acceptance gate

1. `elspeth-209b7e3a2b` is `closed` with the closure summary above
2. The staging reproducer (Step 1 script) returns success or a structured
   4xx — NOT a 500
3. The composer's persisted state version counter does NOT bump on a
   guaranteed-invalid gate-creation request (atomicity invariant restored
   if 0.a didn't already cover it)
4. Shape 8 lives in the agreement-suite registry and passes 5x sequentially
5. The full integration + unit test suite passes (no flakes, no
   regressions in adjacent composer tests)
6. `notes/composer-phase-0b-staging-capture-<DATE>.md` exists as the
   durable triage artefact

### What this work does NOT cover

- **Phase 4 (eval re-run, `elspeth-599ecf69fa`).** Once 0.b is closed,
  Phase 4 can include the S3 gate-routing scenario again. That's a
  separate handover; do not extend this work into running the LLM eval.
- **rows_routed structural fix (`elspeth-obs-abc8baa1cd`).** Phase 2.2
  closure rationale notes this as a known structural gap in the counter
  model. Out of scope here.
- **Frontend rendering of new RunStatus values (`elspeth-81fc655835`).**
  Separate task. Out of scope.

### Anti-patterns to avoid

1. **Don't write the regression test before the fix lands.** Speculative
   tests against unfixed bugs lock in wrong behaviour. The bug-verification
   protocol is unable to discriminate "test passes because fix is correct"
   from "test passes because test was written to match observed buggy
   behaviour."
2. **Don't add defensive `getattr`/`isinstance` to silence the crash.**
   The composer is system code, not user code; CLAUDE.md's
   "Defensive Programming: Forbidden" rules apply with full force here.
   The fix is a structural correction, not a try/except wrapper.
3. **Don't swap the Phase 3 anti-pattern by adding `xfail` markers.**
   If the fix is partial, the test should not exist yet. Land the fix in
   one commit; land the test in the next.
4. **Don't broaden the issue scope.** This issue is the gate-primitive
   crash specifically. If the investigation surfaces adjacent composer
   bugs (e.g. coalesce primitive crashes in the same code path), file
   them as new bugs and close 209b7e3a2b on the gate-primitive scope
   only. Scope creep is the enemy of the closure summary.
5. **Don't skip the durable capture artefact.** The
   `notes/composer-phase-0b-staging-capture-<DATE>.md` file is the
   evidence trail an auditor or future maintainer will read. CLAUDE.md's
   auditability standard applies to investigation artefacts as much as
   to runtime audit data: "if it's not recorded, it didn't happen."

### Reference materials

- **Issue:** `elspeth-209b7e3a2b` (read its description in full —
  it lists the three candidate fix branches)
- **Sibling closed issue:** `elspeth-2c3d63037c` (0.a, the diagnostic
  surface)
- **Program doc:** `notes/composer-remediation-program-2026-05-01.md`
  (Phase 0 section, lines 28-59)
- **Source audit:** `notes/composer-llm-eval-2026-05-01.md` (search for
  "gate primitive")
- **Project guidance:** `/home/john/elspeth/CLAUDE.md` — auditability
  standard, three-tier trust model, "no defensive programming" rules,
  layer dependency rules
- **Agreement suite (where Shape 8 lands):**
  `tests/integration/pipeline/test_composer_runtime_agreement.py`
- **Diagnostic-surface tests (existing pattern for what runtime-preflight
  failure events look like):**
  `tests/unit/web/sessions/test_routes.py` lines 4034, 4078, 4136, 4180

When closing `elspeth-209b7e3a2b`, the closure is the durable contract.
Future agents reading the program doc should be able to grep for the
issue ID and find the root-cause fix branch named, the commit hash cited,
and the regression test pinned. Take time on the closure summary.

## Prompt ends
