# AWS Bedrock Prompt And Content Shielding Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:using-git-worktrees, superpowers:test-driven-development,
> using-security-architect, wardline-gate, and superpowers:executing-plans.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit AWS-native prompt-attack shielding and harmful-content
screening that can be placed before or after any LLM call, is fully audited,
uses ECS task-role credentials, fails closed, and is live-proved independently
of the Bedrock model provider.

**Problem:** ELSPETH currently exposes `azure_prompt_shield` and
`azure_content_safety`. They are deliberately separate because prompt
injection/jailbreak/leakage is a control-plane integrity problem, while harmful
content is a safety/moderation problem. Plan 09 adds a LiteLLM-backed Bedrock
model provider and maps a provider `content_filter` finish reason to a typed
error, but it does not create, attach, invoke, audit, or verify a Bedrock
Guardrail. A model refusal is not proof that either required control ran.

**Architecture:** Use Amazon Bedrock Guardrails through the model-independent
`bedrock-runtime.ApplyGuardrail` operation. Add two explicit transform plugins
sharing one audited client and response parser:

- `aws_bedrock_prompt_shield` evaluates untrusted user/document fields against
  the guardrail's `PROMPT_ATTACK` filter. Its product vocabulary remains prompt
  shield, and its outputs distinguish user-prompt and document/input analysis.
- `aws_bedrock_content_safety` evaluates input or output fields against the
  guardrail's harmful-content filters: `HATE`, `INSULTS`, `SEXUAL`, `VIOLENCE`,
  and `MISCONDUCT`.

Each plugin calls `ApplyGuardrail` with a pinned guardrail identifier, numeric
version, `source=INPUT|OUTPUT`, and `outputScope=FULL`. It validates the full
assessment structure against the expected policy family and fails closed on a
missing, duplicate, unknown, malformed, or contradictory result. The guardrail
resource owns filter strengths and block/detect actions; pipeline YAML cannot
weaken those AWS-side policies. A blocked/intervened result becomes explicit
transform output suitable for a gate or quarantine route—it is never converted
into a silent allow.

This is independent of model invocation: the transforms can protect scraped
documents before OpenRouter, Azure OpenAI, Bedrock, or a future provider, and
can screen model output after any provider. Plan 09 may optionally add
provider-level Bedrock Guardrail enforcement later as defense in depth, but
that cannot replace these field-level, auditable, routable transforms.

AWS's content-filter vocabulary is not Azure parity. In particular, the
Bedrock harmful-content categories above do not include Azure Content Safety's
`self_harm` category. Plan 15 must document that residual and require an
explicit operator policy decision or an additional approved control before a
deployment claims equivalent self-harm coverage.

**No automatic insertion:** Composer assistance may recommend an available
authorized prompt shield when untrusted internet/user content will flow into an
LLM, and may recommend content safety when policy calls for moderation. It must
not silently mutate topology. Only an explicit user instruction or an
operator-enforced authoring policy may require placement.

**Target-LLM discovery and selection:** The composer receives an explicit
request-scoped inventory of usable prompt/content controls, derived from plugin
registration, operator enable flags, validated deployment configuration, and
the current user's/server's secret-reference availability. It must select only
an advertised implementation, using the operator's preference when more than
one compatible shield is enabled. Installed-but-disabled or unconfigured
plugins are not advertised. The model sees approved reference placeholders,
never resolved secret/config values or availability failure details.

**Depends on:** Plan 06 (boto3/botocore/default credential-chain support), Plan
08A (the complete web authoring security gate must precede new web-reachable AWS
plugin registration), and shared signed-tier/trust-boundary/Wardline baseline
`elspeth-8166b310e7`. It does not depend on Plan 09 because `ApplyGuardrail` is
model-independent. If Plan 09 merges first, rebase Plan 15 before its composer/
catalog work and rerun the complete focused suite. Plans 03 integration, 10,
and 12 depend on Plan 15.

**Primary references:** Amazon Bedrock, “Detect and filter harmful content by
using Amazon Bedrock Guardrails”; “Detect prompt attacks with Amazon Bedrock
Guardrails”; “Use the ApplyGuardrail API in your application”; Bedrock Runtime
`ApplyGuardrail` API/SDK model; AWS IAM service authorization for Amazon
Bedrock.

**Tech Stack:** boto3/botocore Bedrock Runtime, Pydantic v2, ELSPETH transform
contracts and audited external-call client pattern, pytest/Stubber/Hypothesis,
CloudWatch operator telemetry, Wardline.

---

### Task 0: Atomic ownership, service-model pin, and policy preconditions

- [ ] Require `release/0.7.1`, preserve unrelated changes, and create a clean
  isolated implementation worktree at the live release tip. Do not stash,
  reset, or stage another worker's files.
- [ ] Require Plan 06, Plan 08A, and `elspeth-8166b310e7` closed. Atomically
  start the exact Plan 15 Filigree step; do not use claim-plus-status.
- [ ] Record `BASE_SHA`, dirty paths, installed boto3/botocore versions, and
  the live botocore service model for `ApplyGuardrail`. Tests and implementation
  must use that real model and `botocore.stub.Stubber`, not a guessed response.
- [ ] Re-read the Azure shield/content transforms, shared audited-client
  patterns, composer shield availability, plugin discovery/catalog contracts,
  Plans 03/09/10/12, and the logging/telemetry policy before editing.
- [ ] Record the approved dedicated Guardrail contract for acceptance: one
  prompt-attack guardrail containing only the expected `PROMPT_ATTACK` family,
  and one content-safety guardrail containing the five listed harmful-content
  categories. Both use immutable numeric versions; `DRAFT` is forbidden in
  production and live acceptance.

---

### Task 1: Shared audited Bedrock Guardrails client and strict parser

**Files:**

- Create: `src/elspeth/plugins/transforms/aws/guardrails_client.py`
- Create or modify: `src/elspeth/plugins/transforms/aws/common.py`
- Create: `tests/unit/plugins/transforms/aws/test_guardrails_client.py`
- Create: `tests/property/plugins/transforms/aws/test_guardrails_response_properties.py`
- Modify: `tests/unit/telemetry/test_plugin_wiring.py`

**Interfaces:**

```python
class BedrockGuardrailRequest:
    guardrail_identifier: str
    guardrail_version: str       # decimal, immutable version only
    region_name: str | None
    source: Literal["INPUT", "OUTPUT"]
    control: Literal["prompt_attack", "content_safety"]
    text: str

class BedrockGuardrailDecision:
    intervened: bool
    action: Literal["NONE", "GUARDRAIL_INTERVENED"]
    assessments: tuple[GuardrailAssessment, ...]
    usage_units: Mapping[str, int]
```

- [ ] RED config/parser tests: require nonblank bounded identifier and decimal
  version, optional valid region, closed source/control values, bounded nonblank
  text, and no unknown keys. Reject `DRAFT`, floats, signs, whitespace/control
  characters, ARNs with query-like suffixes, bool-as-int, empty content, and
  oversized content before an AWS call.
- [ ] Parameterize rejection of access key, secret key, session token, profile,
  role, external ID, endpoint URL, proxy, TLS override, signing, account, and
  credential-provider fields. The client uses `boto3.client("bedrock-runtime",
  region_name=...)` and the default chain only.
- [ ] Use the installed botocore model to build real Stubber responses for
  safe/detected/intervened prompt attacks and every harmful-content category.
  Pin exact AWS request keys: identifier, version, source, one bounded text
  content block, and `outputScope="FULL"`.
- [ ] Fail closed on absent/blank/unknown action, missing assessments, missing
  expected filters, duplicate categories, wrong control family, bad confidence/
  action shapes, inconsistent top-level versus assessment action, booleans or
  non-finite numeric values, negative/overflow usage, extra content-bearing
  output, and response collections beyond configured bounds.
- [ ] For prompt shield require exactly the prompt-attack assessment family.
  For content safety require exactly the five listed categories. If AWS adds a
  category or changes the response model, stop with a static schema error until
  ELSPETH explicitly adjudicates it; do not silently allow unknown policy data.
- [ ] Implement bounded retry classification for throttling, timeout,
  connection, and transient 5xx errors using existing AWS/common retry policy.
  Auth/permission, validation, resource/version not found, malformed response,
  and intervention are non-retryable. Public errors contain only control and
  stable error class, never provider message/body, identifier/ARN, region,
  request ID, URL, account, or content.
- [ ] Audit first. The Landscape external-call record stores the control,
  identifier/version fingerprint, source, input payload fingerprint/length,
  action, category/action/confidence facts, usage, duration, attempt count, and
  typed failure. It must not duplicate raw text, provider blocked messaging,
  sensitive-information matches, response body, AWS ARN/account/request ID, or
  credentials; the existing row/payload audit remains the content authority.
- [ ] Only after the audit call succeeds, emit payload-free operational
  telemetry with control, closed category/action, bounded confidence/usage,
  duration, attempt, and blocked outcome. If audit persistence fails, emit no
  telemetry. If telemetry fails, retain the audit and decision.
- [ ] Property-test arbitrary nested/malformed provider data and malicious
  mapping/property objects. The parser never dynamically invokes provider
  properties, never leaks a sentinel in `str`/`repr`/logs/telemetry, and either
  returns a fully validated decision or one static typed error.
- [ ] Run focused client/property/audit-order/telemetry tests. Stage only Task 1
  files and commit `feat(aws): add audited Bedrock Guardrails client`.

---

### Task 2: Explicit prompt-shield and content-safety transforms

**Files:**

- Create: `src/elspeth/plugins/transforms/aws/bedrock_prompt_shield.py`
- Create: `src/elspeth/plugins/transforms/aws/bedrock_content_safety.py`
- Modify: `src/elspeth/plugins/transforms/aws/__init__.py`
- Create: `tests/unit/plugins/transforms/aws/test_bedrock_prompt_shield.py`
- Create: `tests/unit/plugins/transforms/aws/test_bedrock_content_safety.py`
- Create: `tests/unit/contracts/transform_contracts/test_bedrock_prompt_shield_contract.py`
- Create: `tests/unit/contracts/transform_contracts/test_bedrock_content_safety_contract.py`
- Create: `tests/integration/plugins/transforms/aws/test_bedrock_guardrails_live.py`
- Modify: plugin discovery/catalog/hash/golden tests and generated artifacts
- Modify: `docs/reference/configuration.md`

**Plugin contracts:**

```yaml
- name: aws_bedrock_prompt_shield
  options:
    fields: [document_text]
    analysis_type: document       # user_prompt | document | both
    guardrail_identifier: ${BEDROCK_PROMPT_GUARDRAIL_ID}
    guardrail_version: ${BEDROCK_PROMPT_GUARDRAIL_VERSION}
    region_name: ap-southeast-2

- name: aws_bedrock_content_safety
  options:
    fields: [model_output]
    source: OUTPUT
    guardrail_identifier: ${BEDROCK_CONTENT_GUARDRAIL_ID}
    guardrail_version: ${BEDROCK_CONTENT_GUARDRAIL_VERSION}
    region_name: ap-southeast-2
```

- [ ] Mirror house transform contracts: field existence/type validation,
  deterministic per-row output, no mutation of input, batch/row identity,
  lifecycle/probe behavior, declared output schema, bounded concurrency/rate
  limiting, and clean close. Do not copy Azure HTTP-key/endpoint assumptions.
- [ ] Prompt-shield RED tests cover user-prompt, document, and both modes;
  safe/detected/intervened outcomes; multiple fields; duplicate field rejection;
  missing/non-string/oversized content; and strict propagation of the shared
  client's fail-closed decisions. Output fields clearly name prompt attack and
  analysis target; they never call it content safety.
- [ ] Content-safety RED tests cover INPUT and OUTPUT, all five categories,
  multiple simultaneous detections, safe/detected/intervened outcomes, and the
  same field/batch/failure contracts. Output fields clearly name harmful-content
  categories; they never imply prompt-injection protection.
- [ ] A provider intervention routes as a transform result, not an exception,
  so a pipeline can gate/quarantine it. Transport/auth/schema failures remain
  typed failures and fail the row/run according to existing engine policy.
  Static fallback text from AWS is never substituted into business data.
- [ ] Add forward invariant probes using a fake client only; they prove output
  schema and no input mutation without making an external call. Live AWS
  behavior belongs only to the explicitly selected integration test and Plans
  10/12 acceptance.
- [ ] Register both plugins and update discovery, catalog schema, field
  ordering, source hashes, goldens, and plugin contract tests. A tree with only
  one registered plugin is incomplete and must fail the plan guard test.
- [ ] Add configuration/reference docs that distinguish controls, show explicit
  placement before/after an LLM, require default credentials/task role, forbid
  `DRAFT`, and state the Bedrock-vs-Azure category gap including `self_harm`.
- [ ] Live developer smoke is gated by explicit
  `ELSPETH_RUN_BEDROCK_GUARDRAILS_LIVE=1`. Once selected, missing region,
  identifier/version, credentials, permission, or expected assessment is a
  failure, not a skip. Use safe AWS-published sentinel examples; persist no raw
  input/output.
- [ ] Run focused transform/contract/discovery/catalog/hash/golden suites.
  Stage only Task 2 files and commit
  `feat(transforms): add Bedrock prompt and content shields`.

---

### Task 3: Provider-neutral composer capability and recommendation semantics

**Files:**

- Modify: `src/elspeth/web/interpretation_state.py`
- Modify: `src/elspeth/web/config.py`
- Create: `src/elspeth/web/shield_capabilities.py`
- Modify: `src/elspeth/web/composer/prompts.py`
- Modify: `src/elspeth/web/composer/tools/_availability.py`
- Modify: `src/elspeth/web/composer/tools/_shield_availability.py`
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`
- Modify: `src/elspeth/plugins/transforms/web_scrape.py`
- Modify: `src/elspeth/plugins/transforms/llm/transform.py`
- Modify: `tests/unit/web/test_config.py`
- Create: `tests/unit/web/test_shield_capabilities.py`
- Modify: `tests/unit/web/composer/test_prompts.py`
- Modify: `tests/unit/web/composer/test_tools.py`
- Modify: `tests/unit/web/composer/tools/test_shield_availability.py`
- Modify: related catalog/assistance tests and LLM golden/hash

**Availability interface exposed to the target LLM:**

```json
{
  "available_security_controls": {
    "prompt_shields": [
      {
        "plugin": "aws_bedrock_prompt_shield",
        "preferred": true,
        "analysis_targets": ["user_prompt", "document"],
        "config_refs": {
          "guardrail_identifier": "${ELSPETH_WEB__AWS_BEDROCK_PROMPT_GUARDRAIL_IDENTIFIER}",
          "guardrail_version": "${ELSPETH_WEB__AWS_BEDROCK_PROMPT_GUARDRAIL_VERSION}",
          "region_name": "${ELSPETH_WEB__AWS_BEDROCK_GUARDRAILS_REGION_NAME}"
        }
      }
    ],
    "content_safety": [
      {
        "plugin": "aws_bedrock_content_safety",
        "preferred": true,
        "stages": ["INPUT", "OUTPUT"]
      }
    ]
  }
}
```

Only usable controls appear. The prompt never receives disabled plugin rows,
availability failure reasons, resolved secret/config values, Guardrail
identifiers/versions, region values, account/role/permission data, or raw
catalog configuration. It may receive only the closed, approved environment or
secret **reference placeholders** required to author the selected plugin.

**Server-side settings:**

```python
WebSettings.aws_bedrock_prompt_shield_enabled: bool = False
WebSettings.aws_bedrock_content_safety_enabled: bool = False
WebSettings.aws_bedrock_prompt_guardrail_identifier: str | None = None
WebSettings.aws_bedrock_prompt_guardrail_version: str | None = None
WebSettings.aws_bedrock_content_guardrail_identifier: str | None = None
WebSettings.aws_bedrock_content_guardrail_version: str | None = None
WebSettings.aws_bedrock_guardrails_region_name: str | None = None
WebSettings.prompt_shield_preference: tuple[
    Literal["aws_bedrock_prompt_shield", "azure_prompt_shield"], ...
] = ("aws_bedrock_prompt_shield", "azure_prompt_shield")
```

These values are ingested through the existing `ELSPETH_WEB__...` environment
surface and Secrets Manager task-definition injection where the operator treats
an identifier as sensitive. The inventory exposes only the plugin and bounded
capabilities plus the canonical reference placeholders above. After choosing
an advertised plugin, the target LLM reads its ordinary catalog schema/secret
requirements and uses only those approved placeholders; it never receives the
resolved secret/config values.

- [ ] Replace the Azure-name-specific authorized/available check with one
  provider-neutral `prompt_shield` capability whose implementations are a
  closed set: `azure_prompt_shield` and `aws_bedrock_prompt_shield`. Availability
  requires the selected plugin to be registered and its required secret/config
  references satisfiable; checking one implementation must not require the
  other provider's secret.
- [ ] Add explicit deployment enablement and preference settings for the AWS
  implementations. `aws_bedrock_prompt_shield` is available only when the
  plugin is registered, its enable flag is true, region plus immutable
  identifier/version configuration is valid, and startup/doctor capability
  probing has not failed. `aws_bedrock_content_safety` uses its own independent
  flag and config. Installed-but-disabled, partially configured, `DRAFT`, or
  failed-probe controls are unavailable. Azure availability continues to use
  the current per-user/server `AZURE_CONTENT_SAFETY_KEY` secret-reference
  check. Default all new AWS enable flags to false.
- [ ] Produce one provider-neutral, request-scoped capability inventory from
  the catalog plus server settings, secret-reference availability, and the
  last validated deployment capability state. Pass that same immutable
  inventory to prompt construction, guided B/C warning selection, tool
  discovery, and validation; do not reimplement four subtly different checks.
  Bind probe state to the effective config fingerprint and process startup;
  missing, stale, differently fingerprinted, or failed state is absent.
  Unknown/exceptional availability fails closed to absent. A later AWS outage
  still fails the transform call closed even if the startup capability was
  healthy.
- [ ] Serialize only available implementations into
  `available_security_controls`. If exactly one compatible prompt shield is
  available, the target LLM must name that plugin. If multiple are available,
  it must choose the operator-preferred compatible implementation; preference
  is a closed server-side ordering, never inferred from secret values or model
  taste. If none is available, it must emit the State-C reconsider advisory and
  must not invent Azure, AWS, or a generic shield node.
- [ ] RED matrix tests cover Azure-only, AWS-only, both with Azure preferred,
  both with AWS preferred, neither, registered-but-disabled, secret missing,
  partial AWS config, failed AWS probe, unavailable catalog entry, and an
  availability checker exception. Assert the target prompt and transform-list
  tool expose exactly the usable/preferred rows plus only the canonical
  reference placeholders; no resolved secret/config value, unapproved env name,
  Guardrail value, ARN, region value, permission, or failure-detail sentinel.
- [ ] Preserve the semantic distinction: content safety is never presented as
  prompt shielding. Add a separate provider-neutral content-safety capability
  for policy-driven moderation recommendations.
- [ ] RED assistance tests: untrusted web/user/document content flowing into an
  LLM triggers a strong prompt-shield recommendation when an authorized
  implementation is available and names the available/preferred plugin rather
  than hard-coding Azure. It does not auto-insert, choose an unavailable
  provider, expose configuration material, or claim the pipeline is safe. When policy
  explicitly requires a shield, validation blocks an unshielded path rather
  than silently rewriting it.
- [ ] Add graph-order tests: the prompt shield must dominate every path from the
  named untrusted source/field to the LLM input; content safety must be placed on
  the configured input/output side. A shield on an unrelated branch or after
  the vulnerable LLM does not satisfy the requirement.
- [ ] Keep generic Azure/OpenRouter behavior and existing recommendation text
  stable except for provider-neutral naming. Rebase on Plan 09 if its LLM
  transform edits are already merged, regenerate the hash/golden once after the
  final source edit, and rerun all three-provider LLM tests.
- [ ] Stage only Task 3 files and commit
  `feat(composer): generalize shield capabilities for Bedrock`.

---

### Task 4: AWS IAM, live candidate acceptance, and closeout gates

**Files:**

- Modify: `docs/runbooks/aws-ecs-deployment.md`
- Modify: `tests/unit/web/test_aws_ecs_runbook_contract.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py`
- Modify: `docs/release/guarantees.md`

- [ ] Add resource-scoped task-role permission for
  `bedrock:ApplyGuardrail` on the two approved immutable guardrail versions.
  If startup/live contract inspection uses `GetGuardrail`, grant only
  `bedrock:GetGuardrail` on the same resources. Keep model `InvokeModel`
  permissions separate. No wildcard resource unless AWS makes exact scoping
  mechanically impossible and the operator records that limitation.
- [ ] Run the production acceptance helper inside the candidate ECS task using
  the task role. It evaluates safe prompt/content fixtures, one AWS-published
  prompt-attack sentinel, and one approved non-graphic content-policy sentinel.
  Require the exact safe versus intervened decisions, expected category family,
  immutable version, audit-first Landscape records, and payload-free telemetry.
- [ ] Add negative tests for wrong version, missing permission, malformed
  response, throttling exhaustion, and a guardrail missing the required policy
  family. Every selected negative lane fails closed with static evidence.
- [ ] Evidence stores only run/control correlation, content hash/length,
  guardrail version fingerprint, category/action/confidence/usage, timestamps,
  and typed result. It forbids raw fixture, AWS returned text/body, PII match,
  identifier/ARN/account/request ID, task credentials, and exception text.
- [ ] Prove no prompt/content/credential sentinel appears in container logs,
  CloudWatch telemetry, evidence files, public API errors, or `repr` output.
  Landscape retains only the governed content reference/fingerprint contract.
- [ ] Run all focused AWS guardrail/client/transform/composer/runbook/acceptance
  tests, Ruff format/check, mypy, plugin contracts/hashes/goldens, signed-tier
  and trust-boundary gates, then `wardline scan . --fail-on ERROR`.
- [ ] Run pre-commit on exact staged paths, prove changed-path equals staged-
  path, commit final corrections, and require a clean worktree.
- [ ] Comment on and close only the Plan 15 Filigree step with commits and exact
  gates. Do not start Plan 10 or claim Plan 12.

**Acceptance handoff:** Plan 03's integration proof must recognize both plugins
and the provider-neutral capability. Plan 10 must package the code, IAM, runbook,
and in-task helper. Plan 12 must run the live task-role lane with zero skips and
prove audit-first records, correct prompt/content decisions, redaction, and
telemetry. Missing Guardrails functionality, a provider refusal standing in for
it, `DRAFT`, wildcard credentials/endpoints, malformed-response allow, automatic
topology insertion, or a claim of self-harm parity is NO-GO.
