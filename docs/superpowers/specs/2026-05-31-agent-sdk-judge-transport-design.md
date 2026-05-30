# Design: Claude-Agent-SDK Judge Transport

**Date:** 2026-05-31
**Status:** Draft — awaiting operator review
**Topic:** Add a selectable second transport for the cicd-judge that routes the
verdict call through the **Claude Agent SDK** (Claude Code, programmatic)
instead of OpenRouter, as a cheaper way to achieve the same judging effect —
while keeping the judge's behavior, response contract, and audit record intact.

---

## 1. Problem / motivation

The cicd-judge (`elspeth-lints/src/elspeth_lints/core/judge.py`) gates new
tier-model allowlist entries. Today its only transport is the OpenAI-compatible
SDK pointed at OpenRouter (`call_judge`, `judge.py:786`), billed per-token via
`OPENROUTER_API_KEY`.

The operator wants a second transport using the **Claude Agent SDK**
(`claude_agent_sdk.query`), for two reasons, in priority order:

1. **Cost — the primary intent.** A cheaper way to achieve the *same* effect:
   route judging through a Claude subscription / Agent-SDK credit pool rather
   than OpenRouter per-token billing.
2. **Quality — a hoped-for bonus.** Claude Code is steered toward coding
   behavior more strongly than a raw model call, so verdicts on *code*
   suppressions may improve.

This is a **drop-in transport swap**, not a re-architecture of the judge: same
read-excerpt-and-return-verdict behavior, same `JudgeResponse` contract, same
audit primitive. Explicitly **not** a tool-enabled agentic judge (that was
considered and rejected — it would break the reaudit reproducibility contract;
see §10).

## 2. Goals / Non-goals

**Goals**
- A selectable `--judge-transport {openrouter,agent}` on `justify` and
  `reaudit`, defaulting to `openrouter` (no behavior change unless opted in).
- The agent transport produces a `JudgeResponse` that passes the **same**
  response-contract validation as the OpenRouter path — one parser, one set of
  validators, no divergent coercion.
- The audit record honestly captures **which transport** produced a verdict
  (signed `judge_transport`) and, via the already-signed call timestamp, the
  era of the `claude_code` preset that shaped it — to a standard that passes an
  IRAP-grade review of our own CI tooling.
- The auth/billing path is explicit, so the "cheaper" assumption is verifiable
  rather than silently false.

**Non-goals (out of scope for this change)**
- A tool-enabled judge (Read/Grep beyond the excerpt). Rejected: non-reproducible
  multi-turn verdicts fight the reaudit "a re-run is a signal, not noise"
  contract. See §10.
- Changing the judge's policy, prompt content, verdict schema, or the
  `should_use_decorator` nudge logic. The `_STATIC_POLICY_BLOCK` is reused
  verbatim.
- Migrating the *default* transport away from OpenRouter. OpenRouter stays the
  default; agent is opt-in.
- Product-grade (Landscape) auditability for the cicd-judge. See §4 — the
  standards are deliberately different.

## 3. Architecture

### 3.1 The transport boundary

`call_judge` is the single chokepoint; everything downstream consumes a
`JudgeResponse`. Introduce a transport seam there. **Shared and unchanged**
across both transports:

- `JudgeRequest` / `JudgeResponse` dataclasses (`judge.py:644`, `:676`)
- `_STATIC_POLICY_BLOCK` / `JUDGE_POLICY_HASH` (`:568`, `:615`)
- `_build_user_message_blocks` (`:740`) — the untrusted-data JSON payload
- The **entire response-contract layer**: `_parse_judge_payload` (`:1012`),
  `_verdict_from_string` (`:1037`), `_required_str_field` / `_optional_str_field`
  / `_required_confidence_field` (`:1053`+), and the cross-field
  `should_use_decorator`↔`BLOCKED` check (`:916`).

Only two things differ per transport: **(a)** how the system+user prompt is sent
and **(b)** how the assistant text and usage accounting are extracted from the
response object. Both transports funnel their extracted assistant text through
the *same* `_parse_judge_payload`, so a verdict is validated identically
regardless of origin.

### 3.2 The agent transport

```python
# Conceptual shape — exact API confirmed at plan time against claude_agent_sdk.
options = ClaudeAgentOptions(
    system_prompt={"type": "preset", "preset": "claude_code", "append": _STATIC_POLICY_BLOCK},
    allowed_tools=[],
    disallowed_tools=["Bash", "Read", "Write", "Edit", "Grep", "Glob", "WebSearch", "WebFetch"],
    setting_sources=[],            # block project CLAUDE.md / .claude skills
    permission_mode="bypassPermissions",
    model=<requested model>,
)
# query() is async; call_judge is sync → bridge with asyncio.run.
# Drain the AsyncIterator to the final ResultMessage; pull assistant text
# (the model emits the verdict JSON per the policy block's output schema),
# then feed it to the shared _parse_judge_payload.
```

- **No tools.** `allowed_tools=[]` plus an explicit `disallowed_tools` denylist
  plus `permission_mode="bypassPermissions"` so the call never blocks on a
  permission prompt and never reads the filesystem. The judge sees only the
  excerpt in the prompt, exactly as today.
- **`setting_sources=[]`** blocks the *project's* `CLAUDE.md` and `.claude/`
  skills from leaking into the system prompt. The `claude_code` preset is the
  one intentional Anthropic-side addition (operator decision; see §4.2).
- **Response shape:** prefer reusing the text→`_parse_judge_payload` path over
  the SDK's `output_format=json_schema` / `structured_output`, so both
  transports share the identical strict contract (the `extra_fields` rejection,
  the confidence-range check, the verdict-enum check). The SDK's
  `structured_output` is noted as a fallback only if the preset proves
  unreliable at emitting bare JSON.

## 4. Auditability — and the standard that applies

### 4.1 Two different standards, deliberately

ELSPETH's **product** audit trail (the Landscape) is the legal record for
regulated SDA pipelines: "every decision traceable to source data,
configuration, and code version… withstands formal inquiry." That bar is
non-negotiable **for the product**.

The **cicd-judge is our own CI tooling**, not the product's audit trail. Its
auditability standard is *"could this pass an IRAP"* — and signed verdicts plus
transport/version provenance clear that comfortably. We **strive** toward the
product ideal but do not sacrifice good engineering (here: the coding-steered
preset's better verdicts) to meet a bar the tooling does not need. This spec
applies the IRAP-grade standard, not the Landscape standard.

### 4.2 What the audit record captures

- **`judge_transport` — signed.** A new field on `JudgeResponse` and the
  persisted allowlist entry, values `"openrouter"` / `"claude_agent_sdk"`,
  carried **inside the HMAC-signed payload** (operator decision). It is
  tamper-evident and bound to the verdict: "how the verdict was produced" is
  verdict metadata.
- **Preset-drift provenance = the existing signed timestamp.** The `claude_code`
  preset is an Anthropic-controlled system prompt that drifts over time and is
  **not** captured by `JUDGE_POLICY_HASH`. Rather than capture an exact preset
  version (a probe mechanism we deliberately skip), we rely on
  `judge_recorded_at` — the verdict's call time, **already** in the
  HMAC-signed payload (`judge.py:581`). The timestamp **bounds the preset era**:
  an auditor can reason about which `claude_code` prompt Anthropic shipped on
  that date. This is intentionally IRAP-grade, not exact-version-grade — good
  enough for our own tooling, and it adds no new field. So the audit answer to
  "what shaped this verdict" is `policy_hash` (our appended block) **+**
  `judge_transport` **+** the signed `judge_recorded_at` timestamp. No redundant
  second timestamp is recorded — reusing the existing one avoids fabricating a
  duplicate datum.
- **`policy_hash` semantics unchanged.** It still hashes only
  `_STATIC_POLICY_BLOCK` (our appended instruction). Documented explicitly:
  under the agent transport the preset is an additional, separately-recorded
  influence — `policy_hash` is "our policy", not "the entire prompt." This is an
  honest, IRAP-passing representation; we record the preset's identity and
  version rather than pretending it isn't there.
- **`model_id`** = the served model (from the SDK `ResultMessage.model_usage`
  key), mirroring the OpenRouter path's "record what was actually served, not
  what was requested" rule (`judge.py:937`).
- **Cache accounting:** map the SDK `usage` dict
  (`input_tokens`, `cache_read_input_tokens`) onto
  `prompt_tokens_total` / `prompt_tokens_cached`, preserving the
  `None`-vs-`0` distinction the dataclass docstring is explicit about
  (`judge.py:696`). `total_cost_usd` is available from the SDK and may be
  recorded as an additional optional field (decision deferred to plan; not
  required for the contract).

## 5. The v2 signed-payload coupling (cross-feature dependency)

`judge_transport` enters the **HMAC-signed payload** (the only new signed key —
the timestamp provenance reuses the already-signed `judge_recorded_at`).
Adding any key to that payload changes the signature shape of **every** entry —
which is exactly the migration event the **scope-fingerprint v2** work
(`docs/superpowers/specs/2026-05-31-judge-scope-fingerprint-design.md` /
`docs/superpowers/plans/2026-05-31-judge-scope-fingerprint.md`) already performs.

**These two features share the v2 payload revision.** Resolution:

- `judge_transport` becomes part of the **v2 payload** defined by the
  scope-fingerprint plan (Task 4 of that plan), not an independent payload
  change.
- The scope-fingerprint plan's v1→v2 batch-migrate (Task 10/12) **backfills
  `judge_transport="openrouter"`** for all 221 existing entries — truthful,
  since every existing verdict was OpenRouter-produced.
- **Sequencing:** this transport feature lands **with or after** the
  scope-fingerprint v2 work, never independently, to avoid two uncoordinated
  changes to the signed payload. The scope-fingerprint plan gets a one-line
  forward-reference noting that v2's payload includes `judge_transport`.

If the scope-fingerprint work is deferred, this feature defines its own v2
payload bump including `judge_transport`; the two must then be reconciled to a
single payload schema before either ships. The plan will make the chosen
ordering explicit.

## 6. Selection & auth path

- **`--judge-transport {openrouter,agent}`** on `justify` and `reaudit`,
  default `openrouter`. Config precedence (CLI flag > env) per project standard;
  an `ELSPETH_JUDGE_TRANSPORT` env var may back the flag for CI convenience
  (decision deferred to plan).
- **`JudgeConfigurationError`** (the existing class, `judge.py:618`) gains the
  agent path: SDK not installed (`uv pip install` guidance), or no working
  Claude Code authentication.
- **Auth path made explicit.** The SDK authenticates via an installed +
  logged-in Claude Code CLI (subscription / Agent-SDK monthly credit pool,
  separate from interactive limits as of 2026-06-15) **or** `ANTHROPIC_API_KEY`
  (per-token) **or** Bedrock/Vertex/Azure. The configuration-error and `--help`
  text must state which path is active, so the "cheaper" assumption is
  verifiable — falling back to `ANTHROPIC_API_KEY` is per-token Anthropic
  billing and may not be cheaper than OpenRouter.

## 7. Determinism caveat

The Agent SDK does not expose `temperature`. The OpenRouter path pins
`temperature=0` as load-bearing for verdict reproducibility (`judge.py:882`,
closes C2-4: phantom `WAS_ACCEPTED_NOW_BLOCKED` reaudit divergences). Agent-mode
verdicts are therefore inherently less reproducible.

Mitigation, not elimination: the signed `judge_transport` field lets reaudit
**attribute** a divergence on an agent-written entry to transport noise rather
than real source drift. Documented at the seam. Operators reauditing
agent-written entries should expect, and can now distinguish, transport-induced
divergence. (Reaudit may run under either transport independently of how an
entry was originally written; running reaudit under `openrouter` keeps the
re-check deterministic regardless of an entry's origin transport.)

## 8. Affected code (design-level; the plan enumerates exactly)

- `core/judge.py`: factor a transport seam in `call_judge`; add the agent
  transport implementation (async-bridged `query` call + `ResultMessage`
  extraction); add the `judge_transport` field to `JudgeResponse` (the call-time
  timestamp already exists as `recorded_at`); reuse all shared validators.
- `core/allowlist.py`: `judge_transport` added to the v2 signed
  payload in `compute_judge_metadata_signature`; the `AllowlistEntry` schema +
  loader + atomic validator carry the new fields. **Coordinated with the
  scope-fingerprint v2 tasks** (§5).
- `core/cli.py`: `--judge-transport` on `justify` / `reaudit`; thread the
  selected transport into `call_judge`; `_build_yaml_entry_text` emits the new
  fields; the v1→v2 migrate command backfills `judge_transport="openrouter"`.
- `pyproject.toml` (`elspeth-lints`): a `[judge-agent]` optional-dependency
  extra for `claude-agent-sdk` (mirrors the existing `[judge]` extra for the
  OpenRouter path).
- Consumers that round-trip an `AllowlistEntry` (`reaudit_sidecar.py`,
  `judge_coverage.py`): carry the new fields (same retarget pattern as the
  scope-fingerprint plan's Task 8).

## 9. Testing

- A **fake in-process transport** injected at the seam — unit tests make **no**
  real SDK calls (same posture as the OpenRouter path today; real network calls
  stay out of CI).
- **Contract-parity test:** a fake agent transport and a fake OpenRouter
  transport returning the same model JSON must produce `JudgeResponse` objects
  that survive identical validation and differ only in `judge_transport` /
  `model_id`.
- **Provenance test:** an agent-transport `JudgeResponse` written via `justify`
  persists `judge_transport: claude_agent_sdk` and a signed `judge_recorded_at`
  timestamp, and the signature verifies; tampering `judge_transport` fails
  signature verification.
- **Config-error test:** agent transport with the SDK absent raises
  `JudgeConfigurationError` with install guidance; with the SDK present but no
  auth, the error names the auth path.
- All via the real loader/justify/verify paths — no bypass of production code.

## 10. Out of scope / separate tracks (recorded so they are not lost)

1. **Tool-enabled judge** (Read/Grep beyond the ±30-line excerpt). Considered
   and rejected: non-reproducible multi-turn verdicts break the reaudit
   contract. Could be revisited as a `justify`-only "deep review" mode if a
   future need justifies the determinism cost — but not here.
2. **Hashing the resolved preset prompt** (turning the preset into a fully
   hashed influence). Rejected as gold-plating for our own tooling: only
   feasible if the SDK surfaces the resolved system prompt, and the
   `judge_transport` + signed-timestamp provenance already meets the IRAP-grade
   bar.
3. **Making agent the default transport.** Not now; OpenRouter remains default
   until the agent path has a track record.
