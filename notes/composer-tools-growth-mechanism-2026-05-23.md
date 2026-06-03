# Composer Tools: Growth Mechanism Diagnosis

**Date:** 2026-05-23
**Scope:** `src/elspeth/web/composer/tools/` post-refactor (RC5.2 branch `composer-tools-rearchitect`)
**Files read:** `_dispatch.py` (1451 lines), `__init__.py` (456 lines), `redaction.py` (2854 lines), `service.py` (4653 lines)

---

## 1. The Feedback Loop

**Loop R1 — Signature Explosion drives Registry Fragmentation drives Fragmentation Tax drives next Explosion:**

```
new tool with novel context-kwarg shape
  → existing registries won't hold it (wrong signature contract)
  → new registry created OR hardcoded if-branch added in execute_tool()
  → total files that must change per tool increases
  → each subsequent addition feels individually justified ("just one more entry")
  → next novel shape finds the same pattern and replicates it
```

**Concrete evidence — the 7 current registries and their divergence reason:**

| Registry | Location | Why separate from _MUTATION_TOOLS |
|---|---|---|
| `_DISCOVERY_TOOLS` | `_dispatch.py:1136` | `ToolHandler` signature, no prior-validation overhead |
| `_MUTATION_TOOLS` | `_dispatch.py:1159` | `ToolHandler` signature, carries prior-validation wrapping |
| `_BLOB_DISCOVERY_TOOLS` | `_dispatch.py:1175` | `BlobToolHandler` — needs `session_engine`, `session_id` |
| `_BLOB_MUTATION_TOOLS` | `_dispatch.py:1183` | Same extra kwargs + quota accounting |
| `_SECRET_DISCOVERY_TOOLS` | `_dispatch.py:1191` | `SecretToolHandler` — needs `secret_service`, `user_id` |
| `_SECRET_MUTATION_TOOLS` | `_dispatch.py:1197` | Same |
| `_SESSION_AWARE_TOOL_HANDLERS` | `sessions.py`, imported via `__init__.py:62` | Async handlers; sync registries can't hold them |

Three tools don't fit even this 7-registry taxonomy and get hardcoded `if tool_name ==` branches at `_dispatch.py:1301, 1313, 1326` (`preview_pipeline`, `diff_pipeline`, `set_pipeline`). Their signatures carry kwargs (`runtime_preflight`, `baseline`, `user_message_id`) that post-date the registry design.

**No balancing loop exists.** The `test_skill_drift` gate at `tests/unit/web/composer/test_skill_drift.py:355` is the canary that announces the redundancy — it exists precisely because the same tool-name information lives in at least 6 places — but it does not prevent accumulation; it merely catches the cheapest omission (forgetting the skill markdown).

**Files a developer must touch to add one tool today:**

1. Handler function — in a plane module
2. `_dispatch.py` — registry dict entry (or new `if tool_name ==` branch if signature is novel)
3. `_dispatch.py:106–1133` — inline JSON schema block (hand-maintained, 45 `"name":` entries)
4. `__init__.py:251–455` — `__all__` list (currently 200+ entries)
5. `redaction.py:2443` — `MANIFEST` entry (manually annotated with sensitivity reasoning)
6. `pipeline_composer.md` skill — Step-0 tool enumeration
7. `test_skill_drift.py` — if the skill Step-0 doesn't auto-update, CI fails

If the tool has a new context-kwarg shape: add an 8th touch (a new registry declaration or new constant like `_BLOB_PROVENANCE_MUTATION_TOOLS`).

That is 7–8 files minimum. The **O(N)** growth in `_dispatch.py` and `redaction.py` mirrors the original `tools.py` growth. The 12-module split relocated bytes without addressing this.

---

## 2. Primary Archetype: Shifting the Burden

**Quick fix:** Every time a tool's signature doesn't fit the existing registry, add a new registry, a new handler-type alias, or a hardcoded branch. This works — the tool ships.

**Fundamental solution:** Every tool's complete contract (handler, JSON schema, discovery classification, redaction shape) lives in a single self-describing declaration. Registries are derived programmatically. No manual synchronisation, no MANIFEST, no `__all__` maintenance.

**Why the fundamental solution has been deferred:** Each registry addition is locally cheap and doesn't feel structural. The MANIFEST in `redaction.py` must be maintained by a separate team concern (security/audit). The `__all__` is invisible until mypy complains. The inline JSON schema in `get_tool_definitions()` is where the LLM description lives so editors "naturally" put prose there.

**Where the audit already names this (one level down):** Finding C1 applies Shifting the Burden to the `sources.py`/`sessions.py` cluster specifically. This diagnosis applies the same archetype one abstraction floor up, to the full registry design.

**Secondary archetype: Fixes That Fail**

The 12-module split is the symptomatic fix. It removes the single-file size signal. But `_dispatch.py` is already 1451 lines, `service.py` is 4653 lines, and `redaction.py` is 2854 lines — the same accumulation pattern has reproduced in adjacent files. The split produced a new symptom: finding C2 (`_all_tools` vs `_all_tools_v2` at `_dispatch.py:1201, 1401`), two co-existing "all tools" definitions that disagree about which tools exist. That is the side-effect the archetype predicts.

---

## 3. Which Audit Findings are Loop-Driven vs Incidental

**Loop-driven (consequences of registry fragmentation architecture):**

- **C2** — `_all_tools` / `_all_tools_v2` disagreement (`_dispatch.py:1201, 1401`): direct artifact of the split that produced `_SESSION_AWARE_TOOL_HANDLERS` after `_all_tools` was defined
- **C1** — cross-plane import cluster in `sources.py`/`sessions.py`: the "Not here (yet)" suppression comment (`blobs.py:18–20`) is Shifting the Burden; the boundary is only awkward because tool identity is fragmented across planes
- **B2** — 5 deferred-import sites in `generation.py`: CLAUDE.md names these the Shifting the Burden anti-pattern explicitly; they are cycle workarounds caused by the plane modules importing from each other rather than from a shared declaration
- **C3** — `sinks.py` at 31 lines (1 symbol): premature boundary; a per-tool declaration model would make this trivially co-locate with the `list_sinks` handler
- **C4, C5, C6** — cross-module private-symbol coupling: each of these is a seam created by the plane-module split that didn't change where information is *owned*; under a per-tool declaration model, the ownership question resolves at declaration time
- **A3, B5** — double-validation artifact: the `_handle_*` / `_execute_*` bifurcation mirrors the registry split; the outer handle function exists to manage the prior-validation wrapping that the registry owns, making the inner validate redundant

**Incidental hygiene (would exist under any architecture):**

- A1, A2 — `dataclasses.replace()` drift: a code-quality issue independent of registry design
- B1 — mutable `_VALIDATION_ERROR_PATTERNS`: typing hygiene
- B3 — `secret_service: Any | None`: protocol gap, independent of tool fragmentation
- B4 — defensive `if output is None` after success path: CLAUDE.md offensive-programming violation, not structural
- C7 — `_common.py` cohesion borderline: a concern regardless of how registries are structured
- D1, D2 — blob lock leak and quota race: operational correctness bugs predating the refactor

---

## 4. Structural Intervention

**Meadows Level 5 — Rules** (what counts as a valid tool definition).

The current rule, implicitly: "A tool is valid when its handler exists in a plane module, its schema appears in `get_tool_definitions()`, its name appears in one of 7 registries, its redaction shape appears in `MANIFEST`, and its name appears in `__init__.__all__` and the skill markdown." That rule distributes authority across 6 files with no single source of truth.

**The intervention:** Change the rule to: "A tool is valid when and only when a `ToolDeclaration` object exists, co-located with its handler, containing the handler reference, the JSON schema, the discovery/mutation/session-aware classification, and the redaction shape. The registries, MANIFEST, `__all__`, and `get_tool_definitions()` return value are all derived from the set of `ToolDeclaration` objects at import time."

This is *not* a recommendation of decorators specifically — a `ToolDeclaration` dataclass at module scope achieves the same invariant without magic. The mechanism is implementation; the structural change is that declaration and handler are co-located and the registry becomes derived, not maintained.

**What this does not require:** Changing the handler signatures or the plane module split. The plane modules keep their handlers. The declarations can live next to the handlers. `_dispatch.py` becomes a thin aggregator.

---

## 5. Empirical Prediction (Testability)

**Under current architecture — adding tool #40:**
- Minimum 7 files: plane handler + `_dispatch.py` (registry + schema) + `__init__.py` + `redaction.py` + skill markdown + `test_skill_drift` data
- If novel kwargs: 8th file (new constant / registry)
- `test_skill_drift` will fire for at least 1 of the next 3 tool additions (per base rate of the existing drift findings)

**Under the intervention:**
- 1 file: the `ToolDeclaration` plus the handler function, co-located in the plane module
- `_dispatch.py`, `__init__.py __all__`, `redaction.py MANIFEST`, and `get_tool_definitions()` require zero manual edits
- `test_skill_drift` becomes obsolete (the skill markdown can be generated from the declarations)

**Falsification:** If the intervention is correct, a git diff for "add tool #40" touches ≤2 files. If it still touches ≥5, the declaration model failed to centralise ownership and the structural fix is incomplete.

**Service.py growth prediction:** `service.py` currently uses `is_discovery_tool()` at 10+ call sites (`service.py:549, 566, 596, 2270, 2314, 2380, 2480, 2636, 3104, 3309`). Under the intervention these become a single lookup on the declaration's classification field; `service.py` growth due to tool-count increase drops to zero.

---

## Confidence Assessment

| Claim | Confidence | Basis |
|---|---|---|
| 7-registry taxonomy with specific line citations | High | Verified `_dispatch.py:1136–1197` |
| 3 hardcoded if-branches | High | Verified `_dispatch.py:1301, 1313, 1326` |
| 7–8 files per new tool today | High | Verified all sites: `_dispatch.py`, `__init__.py:251`, `redaction.py:2443`, `test_skill_drift.py:355` |
| service.py 10+ `is_discovery_tool` call sites | High | Grep-verified |
| Shifting the Burden primary archetype | High | Two-path structure (registry quick-fix vs declaration fundamental) confirmed in code |
| Fixes That Fail secondary archetype | High | `_all_tools` / `_all_tools_v2` split is a confirmed side-effect |
| Intervention mechanism (ToolDeclaration) will work | Moderate | Structural reasoning; not yet validated by implementation |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Diagnosis misidentifies primary driver | Low | Low | 7-registry evidence is deterministic; loop is structurally visible |
| Intervention scope underestimated (redaction.py coupling may require protocol changes) | Medium | Medium | Prototype the ToolDeclaration against 2–3 existing tools before committing the full migration |
| Split-first, declaration-second sequence loses momentum | Medium | High | Close elspeth-5aa2e8c2a1 only when a ToolDeclaration prototype exists for at least the blob tool cluster |

## Information Gaps

- [ ] **Git history of `get_tool_definitions()`** — knowing when the inline JSON schema block was separated from the registry dicts would confirm whether the split was deliberate or accidental
- [ ] **Whether `redaction.py` MANIFEST can accept derived entries** — the security annotation (handles_no_sensitive_data, argument_summarizers) may require human review that can't be mechanically derived, which would constrain how fully the intervention can consolidate that file

## Caveats

The "6 files minimum" count assumes the developer knows all 6 are required. In practice the `test_skill_drift` gate catches the omission of the skill markdown on CI, not locally — the loop's actual cost includes the false-pass-then-CI-fail cycle. The empirical prediction should be measured against the *next* 3 tool additions on this codebase, not in a synthetic test.
