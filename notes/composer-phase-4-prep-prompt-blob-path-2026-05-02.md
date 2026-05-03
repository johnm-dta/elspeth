# Phase 4-prep prompt — Composer blob source path shape mismatch

**Use this as a self-contained prompt for a fresh session or subagent.** It has no
dependencies on prior conversation context.

---

## Prompt begins

You are picking up the Phase 4-prep work for the composer remediation
program. Read these three files before doing anything else:

- `notes/composer-remediation-program-2026-05-01.md` — the program doc
- `notes/composer-phase-0b-staging-capture-2026-05-02.md` — the 0.b
  investigation lineage that surfaced this bug
- `notes/composer-llm-eval-2026-05-01.md` — source audit, search for "S3"
  and "blob"

Tracking issue: **`elspeth-07089fbaa3`** *(P1 bug, currently `triage`,
parent `elspeth-528bde62bb`)* — "Composer-stored blob source path uses
wrong shape (`data/blobs/<bid>/<filename>` vs canonical
`data/blobs/<sid>/<bid>_<filename>`)".

### Why this work exists

Phase 0.b (`elspeth-209b7e3a2b`, commit `86e39385`) closed the composer
gate-primitive crash. With that fix, `/validate` succeeds for gate-routing
pipelines built from a CSV blob source. **But the runtime source plugin
still 500s** with `FileNotFoundError` because the path stored in
`composition_states.source` is the wrong shape:

- **Canonical** (per `src/elspeth/web/blobs/service.py:210-211`):
  `<data_dir>/blobs/<session_id>/<blob_id>_<filename>`
- **What's actually stored in `composition_states.source`**:
  `data/blobs/<blob_id>/<filename>` (relative path, BID-as-dir, no
  `<bid>_` prefix on filename)

The blobs DB row carries the correct canonical `storage_path`. The
composer tools layer at `src/elspeth/web/composer/tools.py:1919` stores
`"path": blob["storage_path"]` which would be correct. So the bug is
either upstream (the LLM bypasses this code path) or downstream (the path
is rewritten between tool execution and DB persistence).

**Phase 4 implications:** This is now the binding blocker on Phase 4
(`elspeth-599ecf69fa`) S3 baseline restoration. The eval can run other
scenarios (S1A/S1B/S2 v1/S2 v2) without this fix; S3 (gate routing
producing per-tier output files) cannot. Phase 4 either skips S3 and
explicitly carves it out in the eval-prep notes, or this lands first and
S3 returns to the eval scenario list.

### Critical context (do not lose these)

- `elspeth.foundryside.dev` is a **source-checkout systemd/Caddy deploy
  on this local machine** — not a remote system. Run reproducers with
  local `curl` or in-process Python; tracebacks land in `journalctl -u
  elspeth-web.service`. Restart with `systemctl restart
  elspeth-web.service` after fix.
- The previous Phase 0.b investigation has a captured staging session id:
  `588b94c8-919c-43ab-ae2c-8a3033de8109`. The composition state for that
  session shows the wrong-shape path; you can inspect it without
  re-running staging by querying the local sessions DB. The Phase 0.b
  capture artefact will tell you the exact DB query path.
- Per CLAUDE.md trust tiers, the path field in `composition_states.source`
  is **Tier 1 audit data** — what we wrote to our own DB. If it's wrong,
  it's a bug in our composer/persistence code, not a coercion issue.
  Crash on read if you find structurally invalid stored paths during
  investigation; do not silently rewrite them.

### Step 1 — Reproduce locally

The bug is reproducible without staging — the composer tools layer can be
exercised via the MCP composer interface or directly via the FastAPI
client in tests. Start in-process:

```bash
# 1. Look at the captured wrong-shape state from the prior staging session
sqlite3 ./data/web.db <<'EOF'
.headers on
.mode column
SELECT id, session_id, json_extract(state_data, '$.source.options.path') AS source_path,
       json_extract(state_data, '$.source.options.blob_ref') AS blob_ref
FROM composition_states
WHERE session_id = '588b94c8-919c-43ab-ae2c-8a3033de8109'
ORDER BY version DESC LIMIT 5;
EOF

# 2. Cross-check with the corresponding blob row's canonical storage_path
sqlite3 ./data/web.db <<'EOF'
.headers on
.mode column
SELECT id, session_id, filename, storage_path
FROM blobs
WHERE session_id = '588b94c8-919c-43ab-ae2c-8a3033de8109';
EOF
```

You should see `composition_states.source.options.path` carrying the wrong
shape and `blobs.storage_path` carrying the canonical shape, for the same
blob. The diff between those two strings is the exact divergence to fix.

If the staging DB has been wiped, repeat the staging reproducer from
`notes/composer-phase-0b-staging-capture-2026-05-02.md` step 1 (the
`POST /api/auth/login` → session → inline blob → message chain) and
inspect the resulting `composition_states.source` row.

### Step 2 — Triage the divergence

The agent's prior triage named two candidate branches:

- **(a) LLM tool path bypass.** The LLM may have called `set_source`
  with manual `options` instead of `set_source_from_blob`. The former
  doesn't go through `_execute_set_source_from_blob`'s
  `path: blob["storage_path"]` line and would let the LLM supply any
  path string it wants.
- **(b) Path rewrite after tool execution.** Something between
  `_execute_set_source_from_blob` and `composition_states.source` write
  is mutating the path. Two recent commits to examine before assuming a
  new defect:
    - `5c17d380 fix(web): normalize blob source paths` — what
      "normalize" means here is suspicious; the canonical shape is
      session-scoped + bid-prefixed and "normalize" could plausibly
      strip either of those structural pieces
    - `83e6228d fix(composer): preserve redacted blob source path in
      state` — preservation logic that might flatten the shape during
      redaction

Decision tree:

1. Inspect the captured staging row to see exactly what `source.options`
   keys are present. If `blob_ref` is set and equals the blob's id, the
   LLM went through `_execute_set_source_from_blob` — branch (b). If
   `blob_ref` is absent or empty, branch (a).
2. If branch (b), `git show 5c17d380 -- src/elspeth/web/composer/` and
   `git show 83e6228d -- src/elspeth/web/composer/` to see what those
   commits changed. The bug almost certainly lives in code those commits
   touched.
3. If branch (a), check the LLM tool dispatch in
   `src/elspeth/web/composer/tools.py` to confirm which tool was actually
   called. The MCP tool registry will tell you which tools are exposed
   and which the LLM has access to.

### Step 3 — Land the fix

The fix shape depends on which branch:

- **Branch (a) fix:** make `set_source` reject manual `path` for
  blob-backed sources, or auto-rewrite to canonical when `blob_ref` is
  present. Per CLAUDE.md "no defensive programming" rules, prefer
  rejection (force the LLM to use `set_source_from_blob`) over silent
  rewrite.
- **Branch (b) fix:** revert or amend whichever commit introduced the
  flattening. The canonical shape is load-bearing for runtime path
  resolution and must round-trip through composer state intact.

The fix MUST also include a **Tier 1 read guard** at the runtime
source-plugin path that crashes informatively if it sees a non-canonical
shape — per the auditability standard, the runtime should not silently
fail with `FileNotFoundError` on a structurally wrong path; it should
crash with a message that says "composition_states.source.options.path
has shape X but expected canonical shape `<data_dir>/blobs/<sid>/<bid>_<filename>`
— this indicates a bug in composer persistence, see issue
elspeth-07089fbaa3 for context." That guard prevents the bug from
silently re-occurring under a different cause.

Engineering rules from `CLAUDE.md`:
- Plugins are system code, not user code. Plugin or composer bugs CRASH;
  defensive workarounds are forbidden.
- No legacy-code shims. If the fix changes the canonical shape, change
  every consumer in the same commit; don't add backwards-compatibility
  for the old shape.
- Layer discipline. The fix is most likely in
  `src/elspeth/web/composer/` (L3) or `src/elspeth/web/blobs/` (L3).
  The runtime guard might touch `src/elspeth/plugins/sources/` (L3).
  No upward imports needed.

### Step 4 — Write the regression test (AFTER the fix lands)

**Do not write this test before the fix lands.** Speculative tests
against unfixed bugs lock in wrong behaviour; the bug-verification
protocol cannot discriminate "test passes because fix is correct" from
"test passes because test was written to match observed buggy
behaviour." Phase 3's anti-pattern guidance applies (Phase 3 prompt
lines 254-258).

After the fix lands, add the test. **This is not an agreement-suite
shape** — agreement suite is for validator/runtime divergence on
pipeline shape. This is a composer-persistence-correctness defect; the
right home is `tests/integration/web/composer/` or
`tests/unit/web/composer/`, depending on whether you can drive it from
the FastAPI client or only via direct tools-layer calls.

The test should:

1. Create a blob via the blobs service.
2. Drive the composer tools path that the LLM uses (whichever branch
   the triage identified) to set the source from that blob.
3. Read back `composition_states.source.options.path` and assert it
   equals the blob's canonical `storage_path`.
4. Apply the **bug-verification protocol** from
   `tests/integration/pipeline/test_composer_runtime_agreement.py`
   module docstring (lines 62-74): manually revert the fix, confirm the
   test fails with the expected wrong-shape assertion, restore.
   Document the protocol verbatim in the test docstring.

If the fix added a runtime read guard (Step 3 recommendation), add a
second test asserting that a manually-corrupted
`composition_states.source.options.path` row raises the structured
informative error rather than `FileNotFoundError`. That pins the
auditability invariant.

### Step 5 — End-to-end S3 verification

This is the durable proof Phase 4 needs. After fix + tests land:

```bash
# Re-run the eval's S3 scenario reproducer against the local staging
# deploy. Use the same auth/session/inline-blob/message chain from the
# Phase 0.b capture artefact's Step 1 script. The success criterion:

# Composer:
#   - POST /api/sessions/{sid}/messages succeeds (no 500, no 422)
#   - GET /api/sessions/{sid}/state shows source.options.path in
#     canonical shape <data_dir>/blobs/<sid>/<bid>_<filename>
#   - POST /api/sessions/{sid}/validate returns is_valid: true

# Runtime:
#   - POST /api/sessions/{sid}/execute returns 202
#   - GET /api/runs/{rid} eventually reports status: completed
#     (NOT completed_with_failures, NOT failed)
#   - The two output files exist on disk:
#       data/outputs/high.jsonl with enterprise-tier rows
#       data/outputs/low.jsonl with non-enterprise rows
#     and the row counts match what the source CSV provided
```

Capture the `/state` snapshot (proving canonical path), the run id, and
the output-file row counts as the durable closure evidence.

### Step 6 — Verify and close

```bash
# Full test suites
.venv/bin/python -m pytest tests/integration/
.venv/bin/python -m pytest tests/unit/

# Type and lint
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/ tests/

# Tier-model enforcement
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model

# 5x flake check on the new tests
for i in {1..5}; do
  .venv/bin/python -m pytest <new-test-paths> -v || break
done
```

Close the issue with a summary that:
- Names which branch ((a) or (b)) the bug was in, and the specific code
  location of the defect
- Cites the commit hash with the fix
- Cites the test file + class + line number for the regression test(s)
- Includes the bug-verification protocol result (which line was reverted,
  which assertion fired)
- Pastes the S3 end-to-end evidence: pre-fix and post-fix
  `composition_states.source.options.path` shapes, the post-fix run id,
  and the post-fix output-file row counts
- Notes Phase 4 readiness: with this closed, the only remaining open
  follow-ups for Phase 4 are the two frontend tasks
  (`elspeth-81fc655835`, `elspeth-5434e2924a`) which only block if
  the eval is scored against the dashboard UI, not the API surface

### Acceptance gate

1. `elspeth-07089fbaa3` is `closed` with the closure summary above
2. `composition_states.source.options.path` round-trips canonical for
   blob-backed sources via the composer tool path the LLM uses
3. The S3 scenario from `notes/composer-llm-eval-2026-04-28.md`
   produces real per-tier output files end-to-end on the local staging
   deploy
4. Regression tests pin the canonical-shape invariant + (if added) the
   runtime read-guard invariant; both pass 5x sequentially
5. The full integration + unit test suite passes (no flakes, no
   regressions in adjacent composer or blobs tests)
6. Phase 4 (`elspeth-599ecf69fa`) is unblocked from this issue's
   dependency edge (add `blocked_by` if not already present, then
   remove on closure)

### What this work does NOT cover

- **Phase 4 (eval re-run, `elspeth-599ecf69fa`).** This is Phase 4-prep,
  not Phase 4 itself. Once this closes, Phase 4 can run cleanly with S3
  in scope. Do not expand into the LLM eval here.
- **`elspeth-1ff5c76432`** (file-sink preflight contract violation,
  P2) — separate architectural cleanup; out of scope. The Phase 0.b
  catch-list extension keeps the system correct in the meantime.
- **Frontend rendering tasks** (`elspeth-81fc655835`,
  `elspeth-5434e2924a`) — separate UI work; out of scope. Block only
  Phase 4 scoring against UI, not API.

### Anti-patterns to avoid

1. **Don't silently rewrite the wrong-shape paths in production data.**
   If the captured staging row has the wrong shape, that's evidence; do
   not run a migration to "fix" old rows during this investigation. The
   bug is in the write path, not the read path. Migrations belong in a
   separate change after the write-path fix is verified.
2. **Don't add a defensive `getattr`/`isinstance` guard around the
   runtime source plugin's path resolution.** Per CLAUDE.md
   "Defensive Programming: Forbidden", the runtime should crash
   informatively on structurally invalid stored paths, not coerce them
   into a working shape.
3. **Don't expand scope into adjacent blob lifecycle bugs.** If
   investigation surfaces other blob-related defects (lifetime
   management, reference counting, deletion sequencing), file them as
   new bugs and close this one on the path-shape scope only.
4. **Don't write the regression test before the fix lands.** Phase 3's
   anti-pattern guidance applies in full: speculative tests are not
   discriminative.
5. **Don't use `git stash`** during investigation. Per project memory
   (`feedback_no_git_stash.md`), the stash/pop cycle has caused data
   loss historically. If you need to inspect pre-change state, use
   `git show HEAD:<path>` or a throwaway `git worktree add`.

### Reference materials

- **Issue:** `elspeth-07089fbaa3` (read its description in full)
- **Investigation lineage:**
  `notes/composer-phase-0b-staging-capture-2026-05-02.md` (Phase 0.b's
  staging captures surfaced this bug as out-of-scope finding #1)
- **Canonical storage path definition:**
  `src/elspeth/web/blobs/service.py:210-211`
  (`<data_dir>/blobs/<session_id>/<blob_id>_<filename>`)
- **Composer tools entry point:**
  `src/elspeth/web/composer/tools.py:1919`
  (where `path: blob["storage_path"]` should be assigned)
- **Suspect commits:**
  - `5c17d380` (`fix(web): normalize blob source paths`)
  - `83e6228d` (`fix(composer): preserve redacted blob source path in state`)
- **Project guidance:** `/home/john/elspeth/CLAUDE.md` — auditability
  standard, three-tier trust model (this is Tier 1 audit data),
  defensive-programming-forbidden rules, layer dependency rules
- **Memory:** `feedback_eval_attribution_can_mislead.md` — the prior
  Phase 0.b experience showed that issue titles can inherit framing from
  the originating eval. This issue's framing is well-bounded (the
  evidence is concrete: two strings that differ), but the same
  discipline applies — verify the actual fault location before
  committing to a fix branch.

When closing `elspeth-07089fbaa3`, the closure summary is the durable
contract. Future agents reading the program doc should be able to grep
for the issue ID and find the fix branch named, the commit hash cited,
and the S3 end-to-end evidence pasted. Take time on the closure summary.

## Prompt ends
