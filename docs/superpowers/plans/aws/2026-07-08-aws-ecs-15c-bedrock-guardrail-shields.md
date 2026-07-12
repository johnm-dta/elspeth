# Plan 15C Bedrock Guardrail Shields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Also use
> superpowers:test-driven-development, using-security-architect,
> logging-telemetry-policy, and wardline-gate. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Add audited AWS Bedrock prompt-attack and harmful-content transforms
that consume Plan 15B's universal web policy/profile/capability seams, fail
closed on every positive detection, and use ECS task-role credentials.

**Architecture:** Use model-independent Bedrock Runtime `ApplyGuardrail` with
two explicit transforms and one shared strict client/parser. CLI/batch accepts
explicit Guardrail identity/version/region; web authoring sees only an opaque
operator profile plus safe row options, resolved by Plan 15B before runtime.
The transforms declare provider-neutral typed capabilities so the existing
15B inventory, preference, recommend/required, coverage, validation, and audit
machinery works without an AWS-specific availability subsystem.

**Tech Stack:** boto3/botocore Bedrock Runtime, `botocore.stub.Stubber`,
Pydantic v2, ELSPETH external-call audit contracts, pytest, Hypothesis,
CloudWatch telemetry, Wardline.

**Depends on:** Plan 06 and Plan 15B (`elspeth-0674a06468`). Plan 15B already
depends on Plan 09 and Plan 08A, so those ordering constraints are transitive.

**Blocks:** Plan 03B integration proof, Plan 10 packaging/deployment, and Plan
12 closeout.

**Ownership boundary:** This plan owns reusable plugin code and a sanitized
live checker. Plan 03 owns static doctor inventory/config proof. Plan 10 owns
ECS IAM/task-definition/runbook/acceptance-command integration. Plan 12 owns
executing the final live candidate proof and GO/NO-GO.

---

### Task 0: Claim the slice and pin the installed Bedrock service model

**Files:**

- Read: `docs/superpowers/specs/2026-07-12-universal-web-plugin-policy-design.md`
- Read: `src/elspeth/web/plugin_policy/profiles.py`
- Read: `src/elspeth/contracts/plugin_capabilities.py`
- Read: `src/elspeth/plugins/transforms/azure/prompt_shield.py`
- Read: `src/elspeth/plugins/transforms/azure/content_safety.py`
- Read: `src/elspeth/plugins/infrastructure/clients/base.py`
- Read: `src/elspeth/plugins/infrastructure/clients/http.py`

- [ ] **Step 1: Create an isolated worktree from the integrated Plan 15B tip**

```bash
git status --short
git check-ignore -q .worktrees
BASE_SHA="$(git rev-parse release/0.7.1)"
git worktree add .worktrees/aws-ecs-15c-bedrock-guardrails -b feat/aws-ecs-15c-bedrock-guardrails "$BASE_SHA"
cd .worktrees/aws-ecs-15c-bedrock-guardrails
git rev-parse HEAD
```

Expected: Plan 06 and Plan 15B close commits are ancestors of `HEAD`. Stop if
they are not; do not cherry-pick partial generic-policy work.

- [ ] **Step 2: Atomically start the existing shield step**

```bash
filigree start-work elspeth-7d1f35e3d8 --assignee codex --actor codex
```

Expected: the renamed Plan 15C tracker step enters `in_progress`.

- [ ] **Step 3: Record SDK versions and inspect the real operation model**

```bash
uv run --extra aws python - <<'PY'
import boto3
import botocore

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test",
)
operation = client.meta.service_model.operation_model("ApplyGuardrail")
print("boto3", boto3.__version__)
print("botocore", botocore.__version__)

def walk(path, shape, ancestry=frozenset()):
    shape_id = id(shape)
    print(
        path,
        "type=", shape.type_name,
        "required=", sorted(getattr(shape, "required_members", ())),
        "enum=", tuple(getattr(shape, "enum", ()) or ()),
    )
    if shape_id in ancestry:
        print(path, "recursive")
        return
    nested = ancestry | {shape_id}
    if shape.type_name == "structure":
        for name, member in sorted(shape.members.items()):
            walk(f"{path}.{name}", member, nested)
    elif shape.type_name == "list":
        walk(f"{path}[]", shape.member, nested)
    elif shape.type_name == "map":
        walk(f"{path}.<key>", shape.key, nested)
        walk(f"{path}.<value>", shape.value, nested)

walk("input", operation.input_shape)
walk("output", operation.output_shape)
PY
```

Expected: the installed model includes `guardrailIdentifier`,
`guardrailVersion`, `source`, `content`, `outputScope`, `action`, `outputs`,
`assessments`, and `usage`. Recursively record nested assessment/filter/usage
member names and service enums, including whether this pinned model exposes
the positive-detection boolean. Store the exact versions and model summary in
a Filigree comment; tests use this installed model through `Stubber`. If the
installed model cannot express an unambiguous positive detection while
top-level action remains `NONE`, detect-only profiles are unsupported and
startup fails closed rather than inferring safety from `action=NONE`.

---

### Task 1: Add Bedrock operator profiles and typed plugin declarations

**Files:**

- Create: `src/elspeth/plugins/transforms/aws/__init__.py`
- Create: `src/elspeth/plugins/transforms/aws/guardrail_profiles.py`
- Modify: `src/elspeth/plugins/infrastructure/discovery.py`
- Modify: `src/elspeth/web/config.py`
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/plugin_policy/profiles.py`
- Create: `tests/unit/plugins/transforms/aws/test_guardrail_profiles.py`
- Modify: `tests/unit/plugins/test_discovery.py`
- Modify: `tests/unit/web/test_config.py`
- Modify: `tests/unit/web/plugin_policy/test_profiles.py`

- [ ] **Step 1: Write failing profile validation and redaction tests**

```python
def test_bedrock_profile_requires_numeric_version_and_closed_control() -> None:
    profile = BedrockGuardrailProfileSettings.model_validate(
        {
            "alias": "prompt-default",
            "plugin": "aws_bedrock_prompt_shield",
            "guardrail_identifier": "gr-operator-private",
            "guardrail_version": "7",
            "region": "us-east-1",
        }
    )
    assert profile.guardrail_version == "7"
    assert "gr-operator-private" not in repr(profile)


@pytest.mark.parametrize("version", ["DRAFT", "0", "1.2", "latest", ""])
def test_profile_rejects_non_numeric_immutable_version(version: str) -> None:
    with pytest.raises(ValidationError):
        _profile(guardrail_version=version)
```

Also reject duplicate aliases, unknown plugin/control combinations, invalid
region, empty identifier, raw binding echo in errors, and multiple profiles
for one plugin without an explicit default alias.

Pin offline closed validators for Guardrail identifier/ARN and region. Accept
only the installed service model's documented identifier grammar and the SDK's
static AWS region-name vocabulary/regex; validation makes no network call.
Pydantic errors use Plan 15B's structured `include_input=False` sanitizer and
marker tests prove identifier/region/profile payloads never appear.

- [ ] **Step 2: Implement frozen Bedrock profile settings**

`BedrockGuardrailProfileSettings` stores alias, exact plugin ID, masked/private
identifier, numeric version, and region. It contains no access keys or custom
endpoint. AWS credentials always use the default SDK chain/task role.

If more than one profile is configured for a plugin, require one
`bedrock_guardrail_default_profiles` mapping entry; never select a private
profile lexically. `plugin_preferences` selects the implementation, while the
default-profile mapping selects that implementation's private binding.

- [ ] **Step 3: Register the generic Plan 15B profile resolver**

The Bedrock resolver publishes only:

```yaml
profile: prompt-default
fields: [prompt]
schema: {mode: observed}
```

plus content safety's safe `source: INPUT|OUTPUT`. It rejects raw
`guardrail_identifier`, `guardrail_version`, `region`, endpoint, credential,
environment-marker, and profile-plus-raw combinations in web state. Lowering
copies the state and inserts private bindings only for validation/execution.

- [ ] **Step 4: Extend non-recursive discovery for `transforms/aws`**

Add the directory to `PLUGIN_SCAN_CONFIG`. AWS modules keep boto3/botocore
imports lazy so base installations can discover metadata without importing an
optional SDK at module load. Missing `aws` extra for an explicitly allowlisted
AWS plugin fails startup; unrequested AWS plugins may remain absent. Implement
Plan 15B's generic `check_local_requirements` hook for each Bedrock profile: it
imports and version-checks `boto3`/`botocore` without network only when the
plugin is authorized/profiled. Isolated-install tests prove a base install
boots when AWS is unallowlisted, an allowlisted Bedrock plugin fails startup
with sanitized `plugin_unavailable`, and `uv run --extra aws` succeeds.

- [ ] **Step 5: Run and commit profile infrastructure**

```bash
uv run --extra aws pytest -q \
  tests/unit/plugins/transforms/aws/test_guardrail_profiles.py \
  tests/unit/plugins/test_discovery.py \
  tests/unit/web/test_config.py \
  tests/unit/web/plugin_policy/test_profiles.py
git add src/elspeth/plugins/transforms/aws src/elspeth/plugins/infrastructure/discovery.py \
  src/elspeth/web/config.py src/elspeth/web/app.py src/elspeth/web/plugin_policy/profiles.py \
  tests/unit/plugins tests/unit/web
git commit -m "feat(aws): add Bedrock Guardrail profiles"
```

---

### Task 2: Implement one audited ApplyGuardrail client and strict parser

**Files:**

- Create: `src/elspeth/plugins/transforms/aws/guardrails_client.py`
- Create: `tests/unit/plugins/transforms/aws/test_guardrails_client.py`
- Create: `tests/property/plugins/transforms/aws/test_guardrails_response_properties.py`
- Modify: `tests/unit/telemetry/test_manager.py`
- Modify: `tests/unit/telemetry/test_filtering.py`

- [ ] **Step 1: Write Stubber tests for the exact request contract**

Pin the request parameters:

```python
expected = {
    "guardrailIdentifier": "gr-test",
    "guardrailVersion": "7",
    "source": "INPUT",
    "outputScope": "FULL",
    "content": [
        {
            "text": {
                "text": "untrusted text",
                "qualifiers": ["guard_content"],
            }
        }
    ],
}
```

Both transforms must use `guard_content`; prompt-attack filters are not
credited without it. Tests use
`Stubber.add_response("apply_guardrail", safe_response_fixture(),
expected_params=expected)` against the installed service model.

- [ ] **Step 2: Define three valid decision states before parsing code**

Write fixtures/tests for:

1. safe: top action `NONE`, every required filter `detected=false`, no outputs;
2. detect-only positive: top action `NONE`, at least one required filter
   `detected=true` with assessment action `NONE`, no outputs; and
3. intervened: top action `GUARDRAIL_INTERVENED`, at least one required filter
   `detected=true`/`BLOCKED`, with a bounded 1..N list of documented canned
   outputs.

Detect-only positive and intervened both return a non-retryable blocked
decision. Only state 1 is safe.

- [ ] **Step 3: Implement typed decisions and bounded usage**

```python
@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    detected: bool
    intervened: bool
    matched_filters: tuple[str, ...]
    usage: GuardrailUsage
    request_id: str | None


@dataclass(frozen=True, slots=True)
class GuardrailUsage:
    units: tuple[tuple[str, int], ...]
```

At implementation time derive and pin the complete allowed usage-key set from
the installed `ApplyGuardrail` output model (including topic, content, word,
sensitive-information, contextual-grounding, automated-reasoning, image, free,
and any other members actually present). Require non-negative bounded integer
values, sort them canonically, reject unknown keys/type drift/oversized values,
and add a fixture containing every installed member. Do not silently reduce
usage to four counters and do not return provider output text.

- [ ] **Step 4: Implement strict response validation**

With one content block require exactly one assessment. Bind the accepted nested
shape to Task 0's installed model and the dedicated Guardrail policy created by
Plans 10/12: prompt profiles must expose the installed `PROMPT_ATTACK` filter;
content profiles must expose exactly the approved harmful-content categories
for that configured policy. A FULL-output live fixture pins the legitimate
non-detected entries rather than guessing a provider-global list.

When the installed model exposes `detected`, require a real boolean on every
credited filter and use it to distinguish detect-only positives from safety.
If the installed model lacks that positive fact, reject detect-mode profiles at
startup or require an intervention-only policy; never treat top-level
`action=NONE` alone as proof of safety. Reject missing/duplicate/unknown
configured filters, malformed/oversized collections, contradictory top and
assessment actions, output under safe/detect-only, and zero or over-limit
outputs under intervention. Accept a bounded 1..N valid output list, validate
every object/text length, then discard every text value immediately without
stringify/log/audit/business output. Property tests include multiple legitimate
intervention outputs.

- [ ] **Step 5: Configure exactly one retry owner**

Use `botocore.config.Config` with standard mode,
`total_max_attempts=3`, bounded connect/read timeouts, and no application retry
loop. Retryable SDK categories are throttling, service unavailable/internal
server, and connection/connect/read timeout errors. Access denied, validation,
resource/version not found, service quota, malformed response, and positive
detection are non-retryable. Sanitize terminal errors and record SDK attempt
count from response metadata when available.

- [ ] **Step 6: Record audit before bounded telemetry**

Implement the SDK wrapper as an `AuditedClientBase` subclass. Allocate one call
index, execute/parse, then call `_record_call(...)` with bounded `CallPayload`
fields (operation enum, salted target fingerprint, status/attempt/latency,
matched filter enums, usage counters, request-ID-present boolean). Only after
that repository call commits may `_telemetry_emit(ExternalCallCompleted(...))`
run. Audit failure suppresses telemetry and propagates. Never record text,
Guardrail identity, profile binding, canned output, AWS error body, account, or
role; tests spy on the exact call order for safe, detected, service-error, and
audit-failure paths.

- [ ] **Step 7: Add malformed-response property tests**

Generate duplicate/unknown/missing filters, contradictory actions,
non-booleans, oversized outputs/assessments, and arbitrary canned text. Assert
every malformed shape fails closed and no exception/result contains generated
text.

- [ ] **Step 8: Run and commit the client/parser**

```bash
uv run --extra aws pytest -q \
  tests/unit/plugins/transforms/aws/test_guardrails_client.py \
  tests/property/plugins/transforms/aws/test_guardrails_response_properties.py \
  tests/unit/telemetry/test_manager.py \
  tests/unit/telemetry/test_filtering.py
git add src/elspeth/plugins/transforms/aws/guardrails_client.py \
  tests/unit/plugins/transforms/aws/test_guardrails_client.py \
  tests/property/plugins/transforms/aws/test_guardrails_response_properties.py \
  tests/unit/telemetry/test_manager.py \
  tests/unit/telemetry/test_filtering.py
git commit -m "feat(aws): add strict Bedrock Guardrail client"
```

---

### Task 3: Add prompt-shield and content-safety transforms

**Files:**

- Create: `src/elspeth/plugins/transforms/aws/bedrock_prompt_shield.py`
- Create: `src/elspeth/plugins/transforms/aws/bedrock_content_safety.py`
- Create: `tests/unit/plugins/transforms/aws/test_bedrock_prompt_shield.py`
- Create: `tests/unit/plugins/transforms/aws/test_bedrock_content_safety.py`
- Modify: `tests/unit/plugins/test_discovery.py`
- Modify: `tests/unit/plugins/test_builtin_plugin_metadata.py`
- Modify: `src/elspeth/web/audit_readiness/boundary_expectations.py`
- Modify: `tests/unit/web/audit_readiness/test_boundary_predicate_parity.py`
- Create: `tests/golden/web/catalog/knob_schema/transform__aws_bedrock_prompt_shield.json`
- Create: `tests/golden/web/catalog/knob_schema/transform__aws_bedrock_content_safety.json`
- Create: `tests/golden/web/catalog/policy_view/transform__aws_bedrock_prompt_shield.json`
- Create: `tests/golden/web/catalog/policy_view/transform__aws_bedrock_content_safety.json`
- Modify: `docs/reference/configuration.md`

- [ ] **Step 1: Write failing transform behavior tests**

For both transforms pin safe pass-through, positive detect-only error,
intervention error, transport/service error, malformed-response error,
multi-field handling, missing/non-string field behavior, and on-error routing.
Any `detected=true` returns `TransformResult.error`; no provider canned text is
added to row data or error text.

- [ ] **Step 2: Implement explicit CLI config and typed capabilities**

CLI config requires Guardrail ID, numeric version, region, fields, schema, and
safe source options. It has no access-key or endpoint fields. Web public config
is supplied by the profile resolver from Task 1.

Declare:

```python
policy_capabilities = frozenset(
    {
        CapabilityDeclaration(
            PluginCapability.PROMPT_SHIELD,
            ControlRole.INPUT,
            blocks_positive_detection=True,
        )
    }
)
```

for prompt shield and the equivalent `CONTENT_SAFETY` declaration for content
safety with `ControlRole.OUTPUT`. `determinism = Determinism.EXTERNAL_CALL`;
input passes through only on an explicitly safe decision. Add both exact plugin
IDs to `EXPECTED_TRANSFORM_DETERMINISMS` and update derived boundary inventory
counts/parity tests. Implement non-empty `get_agent_assistance` for each plugin
without exposing private fields.

- [ ] **Step 3: Implement field evaluation and fail-closed aggregation**

Evaluate each configured field independently with `guard_content`. If any
field detects or intervenes, return one sanitized non-retryable error and route
through normal `on_error`. A field/transport/parser error also fails the row.
Only every field returning explicit safe may pass the original row unchanged.

- [ ] **Step 4: Document the AWS/Azure coverage difference**

Document that AWS harmful-content categories in this plugin do not include
Azure Content Safety's `self_harm` category. A deployment must not claim parity
without an additional approved control. Document explicit CLI configuration,
opaque web profiles, task-role credentials, detect-only blocking, and
INPUT/OUTPUT content-safety use.

- [ ] **Step 5: Finalize source hashes and generate goldens**

```bash
uv run python - <<'PY'
from pathlib import Path
from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash

for path, class_name in (
    (Path("src/elspeth/plugins/transforms/aws/bedrock_prompt_shield.py"), "AWSBedrockPromptShield"),
    (Path("src/elspeth/plugins/transforms/aws/bedrock_content_safety.py"), "AWSBedrockContentSafety"),
):
    fix_source_file_hash(path, class_name, compute_source_file_hash(path))
PY
uv run --extra aws pytest \
  tests/unit/web/catalog/test_knob_schema_golden.py \
  tests/unit/web/catalog/test_policy_view_golden.py -q
```

Generate ordinary knob-schema goldens from full `CatalogServiceImpl`; they are
the explicit trained-operator CLI schemas and therefore include Guardrail
ID/version/region. Separately generate policy-view goldens from a deterministic
`PolicyCatalogView` snapshot/profile registry; only these web goldens show
`profile` and safe row options. Private ID/version/region fields are absent from
the policy-view files.

- [ ] **Step 6: Run and commit the transforms**

```bash
uv run --extra aws pytest -q \
  tests/unit/plugins/transforms/aws/test_bedrock_prompt_shield.py \
  tests/unit/plugins/transforms/aws/test_bedrock_content_safety.py \
  tests/unit/plugins/test_discovery.py \
  tests/unit/plugins/test_builtin_plugin_metadata.py \
  tests/unit/contracts/test_plugin_assistance_coverage.py \
  tests/integration/web/test_catalog_discovery.py \
  tests/unit/web/audit_readiness/test_boundary_predicate_parity.py \
  tests/unit/web/catalog/test_knob_schema_golden.py \
  tests/unit/web/catalog/test_policy_view_golden.py
git add src/elspeth/plugins/transforms/aws \
  src/elspeth/web/audit_readiness/boundary_expectations.py \
  tests/unit/plugins tests/unit/web/audit_readiness/test_boundary_predicate_parity.py \
  tests/integration/web/test_catalog_discovery.py tests/golden/web/catalog \
  docs/reference/configuration.md
git commit -m "feat(aws): add Bedrock prompt and content shields"
```

---

### Task 4: Prove Plan 15B integration without duplicating policy machinery

**Files:**

- Create: `tests/integration/web/test_bedrock_guardrail_policy.py`
- Modify: `tests/integration/web/test_plugin_policy_end_to_end.py`
- Modify: `tests/unit/web/plugin_policy/test_availability.py`
- Modify: `tests/unit/web/plugin_policy/test_coverage.py`
- Modify: `tests/unit/web/composer/test_prompts.py`
- Modify: `tests/unit/web/execution/test_validation.py`

- [ ] **Step 1: Add the complete web profile/availability matrix**

Cover not allowlisted, allowlisted-without-profile, one profile, multiple
profiles with/without default, raw private-field injection, profile/raw mixed
form, Azure-only, AWS-only, both with preference, neither, credential/profile
rotation, recommend, and required modes.

Split the time models explicitly: operator profile changes require process
restart and produce a new process policy/binding-generation fingerprint;
request-scoped credential deletion/rotation is detected by the fresh execution
snapshot/resolution without restart. Secret references may be cached as opaque
references, but resolved credential values never enter state, snapshots, or
background job arguments.

- [ ] **Step 2: Prove target-LLM inventory is safe and deterministic**

The prompt includes only plugin ID, opaque alias, safe capabilities,
preference, and mode. Assert absence of identifier, version, region, secret/env
names, account/role, endpoint, and local failure detail. Selection comes from
the existing 15B snapshot; no AWS availability helper is added.

- [ ] **Step 3: Prove required graph coverage and runtime lowering**

Prompt shield must dominate every untrusted path into LLM; content safety must
post-dominate every LLM output path to a sink. Validation lowers profiles
privately and execution uses task-role SDK credentials. Landscape policy
evidence records only IDs/alias-safe selection hashes. Plant marker private
identifier/version/region values and prove zero occurrences across authored and
run config JSON, every node record, complete Landscape export, HTTP errors,
logs, and telemetry as well as policy-evidence rows/prompts.

- [ ] **Step 4: Add offensive no-duplication guards**

Tests fail if code introduces `aws_bedrock_*_enabled`, provider-specific
preference settings, a second snapshot/inventory class, startup-probe-based
authorization, raw profile bindings in web schemas, or imports AWS policy code
into generic prompt/coverage modules.

- [ ] **Step 5: Run and commit integration**

```bash
uv run --extra aws pytest -q \
  tests/integration/web/test_bedrock_guardrail_policy.py \
  tests/integration/web/test_plugin_policy_end_to_end.py \
  tests/unit/web/plugin_policy/test_availability.py \
  tests/unit/web/plugin_policy/test_coverage.py \
  tests/unit/web/composer/test_prompts.py \
  tests/unit/web/execution/test_validation.py
git add tests/integration/web tests/unit/web
git commit -m "test(aws): integrate Guardrails with web plugin policy"
```

---

### Task 5: Add reusable live proof and hand off ECS ownership

**Files:**

- Create: `src/elspeth/plugins/transforms/aws/guardrails_live_check.py`
- Create: `tests/unit/plugins/transforms/aws/test_guardrails_live_check.py`
- Create: `tests/integration/plugins/transforms/aws/test_bedrock_guardrails_live.py`
- Modify: `docs/superpowers/plans/aws/2026-07-08-aws-ecs-03-doctor-cli.md`
- Modify: `docs/superpowers/plans/aws/2026-07-08-aws-ecs-10-packaging-docker.md`
- Modify: `docs/superpowers/plans/aws/2026-07-08-aws-ecs-12-integration-closeout.md`
- Modify: `docs/superpowers/plans/aws/2026-07-08-aws-ecs-00-overview.md`

- [ ] **Step 1: Write failing live-check receipt/redaction tests**

The callable accepts resolved operator profile objects and operator-approved
benign/blocked fixture text, invokes the same client/parser, and returns a
bounded receipt:

```python
@dataclass(frozen=True, slots=True)
class GuardrailLiveReceipt:
    plugin_id: str
    profile_alias: str
    safe_case_passed: bool
    attack_case_blocked: bool
    request_ids_present: bool
```

No receipt, exception, log, or JSON contains Guardrail bindings, fixture text,
credential detail, provider output, account, or role.

- [ ] **Step 2: Implement the reusable checker and opt-in live test**

Register the exact pytest marker `live_aws` and gate ordinary development with
`ELSPETH_RUN_LIVE_BEDROCK_GUARDRAILS=1`. The live test uses the default AWS
credential chain, explicit profile aliases, numeric versions, and region. It
skips only when that env gate is absent in ordinary development. Plans 10/12
run this exact zero-skip command and fail if collection reports any skip:

```bash
ELSPETH_RUN_LIVE_BEDROCK_GUARDRAILS=1 \
  uv run --extra aws pytest \
  tests/integration/plugins/transforms/aws/test_bedrock_guardrails_live.py \
  -m live_aws -q -rs
```

The test contains an explicit guard that fails rather than skips when the env
gate is set but required approved profile/fixture inputs are absent. Fixture
outcomes are operator-owned and bound to the exact Guardrail policy/version in
the acceptance record; the reusable checker makes no claim that one universal
attack string must trigger every policy. It persists receipts only, never text.

- [ ] **Step 3: Update cross-plan ownership contracts**

Plan 03 verifies static plugin/profile configuration shape. Plan 10 adapts the
checker into its acceptance command, task role/IAM, image, task definition, and
runbook. Plan 12 executes safe/attack cases against the candidate deployment,
correlates sanitized Landscape/telemetry evidence, and owns cleanup/GO-NO-GO.
This plan does not edit a Plan-10-created `aws_ecs_acceptance.py` file.

- [ ] **Step 4: Run focused and repository gates**

```bash
uv run --extra aws pytest -q \
  tests/unit/plugins/transforms/aws \
  tests/property/plugins/transforms/aws \
  tests/integration/web/test_bedrock_guardrail_policy.py \
  tests/integration/plugins/transforms/aws/test_bedrock_guardrails_live.py
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src/elspeth/plugins/transforms/aws src/elspeth/web/plugin_policy
PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check \
  --rules plugin_contract.component_type,plugin_contract.plugin_hashes \
  --root src/elspeth
git diff --check
wardline scan . --fail-on ERROR
```

Expected: local suites/gates pass; the opt-in live test is the only locally
conditional lane and its zero-skip ownership is explicit in Plans 10/12.

- [ ] **Step 5: Commit and close Plan 15C**

```bash
git add src/elspeth/plugins/transforms/aws \
  tests/unit/plugins/transforms/aws \
  tests/integration/plugins/transforms/aws/test_bedrock_guardrails_live.py \
  docs/superpowers/plans
git commit -m "feat(aws): add reusable Guardrail live proof"
```

Expected: the worker reports its implementation commit and evidence without
closing from the feature worktree. The integration coordinator rebases/merges,
reruns the full Plan 15C handoff, and closes `elspeth-7d1f35e3d8` with the
integrated `release/0.7.1@<sha>` anchor. This still does not claim live ECS
acceptance; Plans 10 and 12 remain blocked until their owned proof runs succeed.

## No-go conditions

Do not close Plan 15C if any of these remain:

- `guard_content` is absent from either request;
- top-level `NONE` is treated as safe when any required filter detects;
- legitimate intervention `outputs` are rejected or their text is retained;
- response parsing is not pinned to the installed botocore model or omits any
  installed usage counter;
- SDK and application retry loops multiply the attempt budget;
- web YAML exposes identifier/version/region/endpoint/credential/env markers;
- user secrets can shadow operator-profile bindings;
- private Guardrail bindings appear anywhere in Landscape/export/log/error or
  telemetry surfaces;
- an allowlisted AWS plugin boots successfully without its locally required
  `aws` dependencies;
- per-shield enable/preference flags or a second availability inventory exist;
- probe health is used as authorization;
- generic 15B policy logic is reimplemented in AWS modules; or
- this slice claims final ECS live acceptance owned by Plans 10/12.
