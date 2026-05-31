# Investigation prompt — composer big-brother escalation gap

**Date:** 2026-05-09
**Audience:** Engineer or agent investigating whether the big-brother (`request_advisor_hint`) feature should fire on the failure modes observed today, and if so, why it doesn't.
**Context:** Three skill-text gaps were closed today on RC5.1 (commit `906c6332`, see `docs/composer/evidence/composer-skill-gaps-2026-05-09.md`). The skill edits eliminated silent shape downgrade and improved source-binding correctness, but two failure modes remain that look exactly like what the big-brother feature was added for.

## What we're seeing

The cheap composer model (`openrouter/openai/gpt-5.4-mini`) hits walls on multi-component pipelines and either:

1. **Builds the simpler shape unilaterally with a disclosure paragraph** (P2v5, session `9ef043c5-2515-47f7-8c9f-582b3d07a607`). The skill says "do not build the simpler shape unilaterally — refuse and stop" but the LLM builds anyway and tells the user about the gap after the fact. The skill rule is being read but not followed under cognitive load.

2. **Asserts false facts about composer capabilities** (P2-ambig, session `7fd7c082-751a-4752-8cb2-c42c94411a32`). On a deliberately-ambiguous fork+coalesce prompt, the assistant text said: *"the composer here does not provide a `coalesce` node/plugin, so I can't build the requested 'two parallel paths combined into one output' workflow as specified."* This is **factually wrong** — coalesce exists, was used successfully in earlier trials today (sessions `7fc50727`, `9b5c6156`), and is documented in the skill. The LLM apparently got rejected building it (sink options issue), then concluded the plugin doesn't exist, and reported that as a capability gap to the user. **A confidently-wrong capability claim in the audit trail is a Tier-1 trust failure.**

3. **Cascade convergence collapses** (P3v2, session `6eff7e30-4cab-455c-b7c9-c9d155022427`). On the deep_routing prompt (8-node-deep DAG, 5 gates, 7 sinks), the LLM jumped to atomic `set_pipeline` without calling `get_plugin_schema` for unfamiliar transforms, hit `keyword_filter` plugin-config rejections, and exhausted budget. This is the case the new "Discover first, build atomically second" rule (Convergence Guardrail #0) was meant to catch — but the rule didn't fire under load.

The unifying observation: when the cheap model is uncertain, it **does not** escalate to a smarter model. It either guesses (case 2), partial-builds-and-discloses (case 1), or burns budget iteratively without converging (case 3). Project memory (`project_demo_big_brother_mcp_tool.md`) names this exact gap as the third leg of the upcoming tech demo: "cheap composer model can escalate to expensive model for advice when stuck".

## What the big-brother feature is

Per the skill, `request_advisor_hint` is a tool that "forwards your problem statement and context to a frontier model and returns guidance text." Per `pipeline_composer.md` lines 539-607 and the operator-facing config:

- It is **opt-in** — disabled by default (`composer_advisor_enabled` default in `src/elspeth/web/config.py`).
- When disabled, the tool simply does not appear in the LLM's tool list.
- When enabled, it is supposed to be called for: (a) reactive validation loops after at least 2 unchanged-failure retries, (b) proactive security/safety wiring, (c) red-listed plugins (`llm`, `database`, `dataverse`, `azure_content_safety`, `azure_prompt_shield`, `rag_retrieval`, `chroma_sink`).
- Budget is per compose request, capped by `composer_advisor_max_calls_per_compose`.

## The investigation question

**When the cheap composer model hits a multi-round failure that smells like "stuck on plugin-config", "stuck on cognitive overload", or "asserting capability gaps that don't exist", should the big-brother tool be available, and if so, why didn't it fire on today's three failure cases?**

Sub-questions:

1. **Is `request_advisor_hint` enabled on staging at all?** Check `deploy/elspeth-web.env` for `COMPOSER_ADVISOR_ENABLED` (or whatever the env-var is per `config.py`). If disabled, that fully explains why the LLM never tried — the tool wasn't in its list. If enabled, dig deeper.

2. **If enabled, why didn't the LLM call it on the three failures?** Look at sessions `9ef043c5`, `7fd7c082`, `6eff7e30` in `data/sessions.db` and check the `tool_calls` JSON column for any `request_advisor_hint` invocations. If none, the LLM had access but didn't recognize the trigger. The skill's trigger criteria require "at least two retry attempts on the same validator error and the failure mode has not changed" — but the failures here mutate (each retry produces a different rejection). The cheap model may interpret "same validator error" too literally and never qualify the trigger.

3. **Are the documented triggers right for the demo's actual stuck-cases?** The current triggers are validator loops, security wiring, and red-listed plugins. **None of them match today's failures**:
   - P2v5 / P2-ambig: structural-shape inability — not a validator loop, not a red-listed plugin.
   - P3v2: cascade plugin-config burnout — validator loop in spirit but each rejection is a *different* plugin-config error, so the LLM may not see it as "same validator error".

   The demo big-brother memory says it was added for "when stuck" — but the skill's "stuck" definition is narrower. **Either widen the skill's trigger criteria or add a new trigger family** ("structural shape uncertainty", "cascade composition with 3+ unfamiliar plugins") that matches today's cases.

4. **Should the LLM auto-escalate before asserting capability gaps to the user?** Case 2 (false coalesce claim) is the most damaging failure mode — the user is told a feature doesn't exist when it does. The skill currently has no rule "before claiming a composer capability does not exist, call `request_advisor_hint` to verify". Such a rule would convert "cheap-model confabulation" into "cheap-model asks expensive model, expensive model says yes coalesce exists, cheap model retries with the right shape." This is a high-yield trigger and might be the single most important addition.

5. **Is there a server-side telemetry path that detects "stuck" without the LLM having to recognize it?** Right now the trigger is LLM-driven (the LLM decides when to call the tool). A server-side circuit-breaker that counts unsuccessful set_pipeline calls in a compose request and force-injects an advisor result on the 3rd failure would close the gap without depending on LLM judgement. Worth thinking about whether this exists or should.

## Hypotheses to test

| Hypothesis | Test |
|---|---|
| Big-brother is disabled on staging entirely | `grep COMPOSER_ADVISOR /home/john/elspeth/deploy/elspeth-web.env` and check `src/elspeth/web/config.py:59` for default |
| Big-brother is enabled but the LLM doesn't recognize the trigger conditions | Query `data/sessions.db` for any `request_advisor_hint` tool calls in the three sessions named above; if zero, the LLM had access but the criteria didn't match |
| Big-brother triggers are too narrow for the demo's actual stuck-cases | Compare the documented triggers (skill lines 539-607) to today's failure modes; mismatched coverage is the root cause |
| The LLM cannot tell "stuck" from "iterating" | Observe the rejection texts across the three sessions; if each rejection is a distinct error class, the "two unchanged retries" trigger never qualifies even though the LLM is genuinely stuck |
| Server-side stuck-detection would catch what LLM-driven detection misses | Check `src/elspeth/web/composer/service.py` for any compose-request-level circuit-breaker on `set_pipeline` failure count |

## Files most likely relevant

| Path | Relevance |
|---|---|
| `src/elspeth/web/config.py` | `composer_advisor_enabled` default and env-var binding |
| `deploy/elspeth-web.env` | Production-equivalent settings on staging |
| `src/elspeth/web/composer/skills/pipeline_composer.md` (lines 539-607) | Trigger criteria the LLM is supposed to follow |
| `src/elspeth/web/composer/tools.py` | Tool registration; presence in tool list when enabled |
| `src/elspeth/web/composer/service.py` | Compose-request orchestration, where a server-side circuit-breaker would live |
| `data/sessions.db` chat_messages table | Tool-call audits for the three named failure sessions |

## Reproduction sessions (already persisted)

| Session ID | Failure mode | Prompt |
|---|---|---|
| `9ef043c5-2515-47f7-8c9f-582b3d07a607` | Built simpler shape unilaterally with disclosure | "trim descriptions… two side-by-side copies under path_a/path_b… save merged as JSONL" |
| `7fd7c082-751a-4752-8cb2-c42c94411a32` | Asserted coalesce doesn't exist (false) | "trim descriptions… process each row two ways in parallel and combine… save as JSONL" |
| `6eff7e30-4cab-455c-b7c9-c9d155022427` | 8-node-deep cascade exhausted budget on plugin-config | Loan triage workflow with 5 chained gates (full prompt in commit message) |

## Output the investigation should produce

1. A clear answer to "is the feature enabled, and if not, what's the smallest configuration change to enable it on staging".
2. A diff (or proposed diff) to `pipeline_composer.md` widening the trigger criteria to cover today's three failure modes — specifically (a) structural-shape uncertainty, (b) capability-gap claims before verification, (c) cascade composition with multiple unfamiliar plugins.
3. A recommendation (yes / no / needs design) on adding a server-side compose-level circuit-breaker that auto-injects advisor results when LLM-driven escalation doesn't qualify.
4. If the feature is wired but not firing, a re-run of one of the three named prompts after the trigger-criteria diff is applied, with the resulting `request_advisor_hint` tool-call audit captured to confirm the trigger now qualifies.

## Constraints

- **Lite by default.** Operator wants to minimise frontier-model spend. Big-brother should fire only on genuine stuck-cases, not as a safety blanket. The bar for adding a trigger is "this case would have wasted ≥3 cheap-model rounds anyway, so one frontier call is net-cheaper."
- **Trust over cost.** Case 2 (false capability claim) is more important than case 1 or case 3 because it puts wrong facts into the audit trail. If we have to pick one trigger to add, it's "verify capability claims before asserting them to the user."
- **Demo timing.** This is queued behind the §7.6 hardening track (per memory `project_demo_big_brother_mcp_tool.md`). The investigation can land first; the actual deployment of widened triggers can be co-ordinated with §7.6.

## Don't

- Don't push the engine-fix or skill commits — they're committed-not-pushed by convention on RC5.1 (operator pushes when ready).
- Don't run the investigation against a fresh DB; the three named sessions are the canonical evidence.
- Don't assume the big-brother feature works — verify enablement first. If the env-var is unset, the entire investigation may be answered in one line.
