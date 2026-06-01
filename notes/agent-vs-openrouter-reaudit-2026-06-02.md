# Agent (claude_code SDK) vs OpenRouter reaudit — verdict comparison

- **Agent run:** `c32057f7` (`--judge-transport agent`, `notes/judge_reaudit_new_policy_agent_2026-06-01.sh`)
- **OpenRouter run:** `a1f5839537` (full 240-entry corpus, temp=0)
- Snapshot taken at agent ~134 entries scored (run still in flight).

## Two confounds that dominate the raw numbers

1. **Agent SDK "default" resolved to `claude-opus-4-7`, NOT 4.8.** Even though 4.8
   is out, the claude_code preset's pinned default in the installed Agent SDK is
   4.7 (`fresh_model_id == "claude-opus-4-7[1m]"`). "Use the default, don't pin"
   gives 4.7. So this is an opus-4.7 datapoint, not 4.8.
2. **84 of 134 commonly-scored entries are DRIFT artifacts, not judgments.** The
   agent run is on the **post-burndown working tree**; the openrouter run predates
   the burndown edits. 71 `ENTRY_OBSOLETE` + 13 `BINDING_DRIFT` (all with
   `fresh_model_id: None` — no LLM call) line up with the 61 burndown-edited files.
   **The agent made only 52 real LLM verdicts.**

## On the 52 real verdicts: agreement ≈ 84%

- 42 same divergence-class as openrouter.
- **8 genuine verdict differences**, split **4 stricter / 4 looser** — symmetric,
  i.e. NOT a systematic accuracy/strictness lift.

## The 8 genuine splits — and what they reveal

The split direction is a **stylistic difference, not a competence one**:

### Agent STRICTER (4× ACCEPTED→BLOCKED) — blocks on rationale *prose* thinness
- `telemetry/manager.py:handle_event` — agent **explicitly says "the code likely IS
  the prescribed form"** but BLOCK-PENDINGs because the rationale is a bare 7-word
  assertion ("Silent recovery with explicit error handling path") — a future auditor
  can't interpret it. → BLOCK-PENDING working exactly as designed.
- `engine/_best_effort.py:best_effort` — agent BLOCK-PENDING: the freestanding
  contextmanager excerpt can't *show* the "primary audit event already recorded"
  invariant; wants it structurally visible. (OR credited the docstring.)
- `contracts/composer_interpretation.py:_validate_enum_member` — agent BLOCK-PENDING:
  "contract/parsing boundary" is conclusory, doesn't establish the data flow.
  (OR credited the code: param typed `object` ⇒ shape not guaranteed ⇒ offensive
  form legit. OR reasoned from visible code; agent demanded the rationale spell it
  out. **Arguably agent over-strict here.**)
- `contracts/url.py:from_raw_url` — agent GENUINE VIOLATION: catching first-party
  `get_fingerprint_key()` ValueError to set `have_key` is exception-as-control-flow
  on **our own** code; honest fix is a non-exception API. **Agent's call is arguably
  SHARPER than OR's "capability probe" accept — worth acting on regardless of
  transport (the burndown should consider fixing this).**

### Agent LOOSER (4× BLOCKED→ACCEPTED) — credits code-visible facts over rationale slips
- `contracts/plugin_context.py:record_validation_error` — agent ACCEPT: the function
  IS the Tier-3 quarantine path (returns ValidationErrorToken), repr_hash fallback is
  meaning-preserving. OR saw the *same* fact but BLOCK-PENDINGed on rationale thinness.
- `plugins/sources/json_source.py:_load_json_array` — agent ACCEPT despite the
  rationale's nominal slip (names UnicodeDecodeError; actual catch is
  JSONDecodeError/ValueError) — "slip doesn't undermine the substance." OR
  BLOCK-PENDINGed *precisely because* the rationale doesn't describe THIS site.
  **Real philosophical split: must the recorded rationale be literally accurate (OR)
  or just point at correct code (agent)? For an audit trail, OR's literalism is the
  safer stance.**
- `plugins/transforms/rag/config.py:_get_providers` — agent ACCEPT (standard optional-
  dependency `ModuleNotFoundError→pass`; downstream KeyError crashes loudly). OR
  BLOCK-PENDING ("no inference — if it's not recorded it didn't happen"; wants a
  breadcrumb). Both defensible.

### The one that needs YOU to rule (substantive doctrinal contradiction)
- `plugins/transforms/rag/query.py:_build_field_only` (R5) — **flat contradiction:**
  - AGENT: PRESCRIBED FORM — `extracted: Any` param, requires str, wrong type is a
    Plugin-Ownership violation that must CRASH; offensive isinstance→raise is the
    sanctioned form.
  - OPENROUTER: GENUINE VIOLATION — it's Tier-2 data; re-checking shape with isinstance
    is the *forbidden* defensive pattern; honest fix is `extracted: str` + let it crash
    naturally.
  - This is the R5 offensive-vs-defensive boundary on a Tier-2 `Any` param — genuinely
    contestable. The transports splitting here is a signal the **entry needs human
    adjudication**, not that either transport is better. (Matches the earlier
    VERIFY-THEN-DECIDE "design decision" note.)

## Verdict on the operator hypothesis ("coding agent > general endpoint")

**Not supported at this n.** No accuracy lift (84% agree; 4 stricter / 4 looser).
The agent is *differently calibrated*, not *better*: it holds the recorded rationale
**prose** to a higher "an auditor must interpret this from the rationale alone" bar,
while **forgiving rationale slips when the code is visibly correct**. OpenRouter is
more literal — the rationale must accurately and completely describe THIS site.
Both philosophies are defensible; for an audit trail OR's literalism is the safer
default, and OR is temp=0 reproducible while the agent (unpinned temp, 4.7) is not —
so some of the 8 splits could be variance.

To actually answer the hypothesis: run BOTH transports on the **same** (post-burndown)
tree, and add a third tie-breaker pass on just the 8 splits. Until then, keep
openrouter as the canonical deterministic re-check.

## THIRD confound (operator, 2026-06-02): the prompt changed between the runs

- OpenRouter `a1f` started 2026-06-01 **11:43**; agent `c32057f7` started **22:14** —
  ~10.5h apart, and the role-clarification/disposition-naming policy edits landed in
  that window. Current `JUDGE_POLICY_HASH = sha256:08052cb8…` is **uncommitted**
  (working-tree only), and the run-state sidecar does **not** stamp the *fresh* policy
  hash on outcomes — so the two runs were almost certainly judged under **different
  prompts**. The numeric comparison therefore conflates THREE axes: transport ×
  model-version (4.7) × prompt-version. Treat the 8 splits as un-attributable noise,
  not a transport signal.

## THE REAL EXPERIMENT (free, tool-using harness investigated the 3 blind blocks)

Per advisor: don't build a `--judge-tools` flag to test "does agency resolve the blind
BLOCK-PENDINGs" — just read the callers now. Result:

- **CASE 1 `_validate_enum_member` → RESOLVES to ACCEPT.** `value` is
  `self.choice/interpretation_source/kind`, validated at `InterpretationEventRecord`
  construction (composer_interpretation.py:244-247); param typed `object`; records are
  composer/LLM-authored ⇒ external origin. The isinstance→raise is offensive enum-contract
  enforcement at a construction boundary. The agent's blind BLOCK-PENDING was **pure
  blindness** — a looking-agent corrects its own over-block.
- **CASE 2 `best_effort` → RESOLVES to ACCEPT.** Call site processor.py:736-757: the code
  `raise AuditIntegrityError(...) from record_failure` if the FAILED outcome can't be
  recorded, and **only then** enters `with best_effort(...)` to emit TokenCompleted
  **telemetry**. The "audit-recorded-first" invariant the agent said it couldn't see is
  proven one call site up. Looking-agent flips to ACCEPT.
- **CASE 3 `_build_field_only` → IRREDUCIBLE (needs operator).** `extracted =
  row_data[self._query_field]` (query.py:102) ⇒ Tier-2 row data, `Any`-typed seam. Whether
  isinstance→raise is forbidden-defensive (OR) or sanctioned-offensive (agent) depends on
  whether the schema contract guarantees `query_field` is `str`. That's a design decision,
  not resolvable by more reading — even a looking-agent lands here.

**Verdict: 2/3 blind blocks dissolve under investigation (both false-blocks from blindness);
1/3 is a genuine human call a looking-agent correctly leaves alone.** This is exactly the
behaviour you'd want from a tool-enabled judge, and it VALIDATES the operator's intuition.
The earlier "wash" was an artifact of blinding the agent + three stacked confounds.

## Immediate actionables (independent of the comparison)
1. **`url.py:from_raw_url`** — first-party exception-as-control-flow; the burndown
   should fix to a non-exception capability API (both transports' reasoning supports
   "this is suspect"; agent flags it as a violation).
2. **`rag/query.py:_build_field_only`** — operator must rule R5 offensive-vs-defensive.
