# Reify campaign — session findings & handoff (2026-05-30)

Session executed `notes/reify-genuine-debt-prompt.md` (reify ~136 genuine tier_model
debt entries). **One entry completed + verified; campaign re-scoped with the operator
after discovering a structural blocker the handoff prompt did not account for.**

## Final repo state (verified)
- **Full tier_model gate: GREEN** (`check --rules trust_tier.tier_model --root
  src/elspeth` → exit 0, 0 findings, 0 tracebacks, shape-only mode).
- Working tree = **1 completed reification** + the operator's **uncommitted 27-entry
  signing pass** (core/plugins/telemetry/web.yaml — untouched by this session).

### Completed (1/136) — fully verified
**`contracts/schema_contract_factory.py:R1:create_contract_from_config`**
- Fix: replaced `normalized_to_original.get(fd.name, fd.name)` (R1 implicit-default
  idiom) with an explicit `if fd.name in normalized_to_original: ... else: fd.name`
  branch. Behaviour-preserving (partial-resolution = absence→identity is a *legitimate*
  contract, pinned by `test_partial_resolution`; doctrine
  `feedback_tier1_explicit_vs_implicit_fabrication` blesses `if !x: x=default`).
- Verified: git-diff oracle (fix on disk, FieldContract construction intact),
  `GATE_EXIT=0 R1_HITS=0`, `20 passed`, ruff + mypy clean. Allowlist entry removed
  from contracts.yaml. **Keyless-safe** (no signed neighbours in that file).

## THE STRUCTURAL BLOCKER (operator action required)
Editing any source file invalidates the whole-file `file_fingerprint` stored on
**every signed (judge-ACCEPTED) entry in that file**; re-signing needs the
operator-only HMAC key. Discovered empirically — an edit to `tier_registry.py` (which
has 3 signed entries) broke the gate load with a `file_fingerprint` mismatch.

Measured across the 49 reify-target files: **17 keyless-safe** (0 signed neighbours),
**32 blocked** (need an operator re-sign after each edit). The operator's just-completed
248-entry signing pass created this coupling; the reify prompt predates it.

### Operator decisions (2026-05-30)
1. **Cleanup** = revert the 3 half-applied/broken files to HEAD. DONE — gate restored.
2. **Campaign path** = **keyless-safe + clean-yaml subset only**. Agent reifies only
   files that are BOTH keyless-safe AND have their allowlist entries in *clean* yamls;
   operator commits the signing pass to unblock the rest later.

## The keyless-safe subset (17 files) and what's unblocked NOW
Keyless-safe (0 signed neighbours): contracts/schema_contract_factory.py ✅DONE ·
core/dag/graph.py · core/events.py · engine/orchestrator/core.py · mcp/server.py ·
plugins/infrastructure/manager.py · plugins/sinks/csv_sink.py · plugins/sinks/json_sink.py ·
plugins/transforms/rag/transform.py · tui/widgets/lineage_tree.py ·
web/composer/guided/recipe_match.py · web/composer/tools/transforms.py ·
web/composer/yaml_generator.py · web/execution/failure_samples.py ·
web/execution/preflight.py · web/execution/validation.py · web/middleware/rate_limit.py

**Second axis — clean vs dirty yaml.** core/* and web/* keyless-safe files have their
allowlist entries in the *dirty* yamls carrying the operator's uncommitted signing pass.
Editing/committing there risks a main-checkout pre-commit stash/pop destroying that
unstaged work. So files unblocked on BOTH axes right now:
**`engine/orchestrator/core.py`, `mcp/server.py` (×4 R6), `tui/widgets/lineage_tree.py`** (+ the done schema_contract_factory).
The rest of the keyless-safe set unblocks once the operator commits the signing pass.

## NEXT SESSION — resume plan
1. Operator: commit the 27-entry signing pass (authoritative keyed gate first) →
   unblocks the clean-yaml constraint for core/* and web/* keyless-safe files.
2. Agent: work the keyless-safe subset using the per-entry protocol below.
3. For the 32 coupled files: per-file handshake — agent edits code + removes BLOCK
   entry, operator runs keyed `justify`/rotate to refresh remaining signed entries'
   `file_fingerprint`.

## Per-entry protocol (validated)
- Judge flags the forbidden *idiom* correctly; verify its *remediation* against
  code+tests before acting (crash vs explicit-branch vs sign vs migrate). The R1/R5
  detectors are broader than `describe_rule()` prose — confirm with the gate.
- RED→GREEN via scoped gate: `... check --rules trust_tier.tier_model --root
  src/elspeth --files <file>` (shape-only mode, keyless).
- `/tmp/allowtool.py {show,remove} <yaml> <key>` = textual, format-preserving
  allowlist-entry surgery by key (no YAML round-trip).
- **Harness reliability caveat (this session):** Bash stdout / wide Reads / temp-file
  Reads were intermittently corrupted (hallucinated file contents, fake "success"/
  "passed"/"exit 0", line-1-repeat on small files). Trust `git diff` as the oracle;
  capture verification as minimal single tokens and read with `offset` to dodge the
  repeat corruption. `Edit` is content-addressed so it fails safe on hallucinated text.

## Do NOT
- Hold/use `ELSPETH_JUDGE_METADATA_HMAC_KEY` (operator-only).
- Edit a file with signed neighbours without an operator re-sign lined up.
- `git commit` in the main checkout while the signing pass is unstaged (stash/pop risk).
