# Runbook: Deploy ELSPETH on ECS with Bedrock Opus and Sonnet

Use this runbook to configure an existing ELSPETH AWS ECS/Fargate deployment
with operator-approved Amazon Bedrock profiles for Claude Opus 4.6 and Claude
Sonnet 4.6. It covers model access, task-role IAM, ELSPETH profile configuration,
deployment, verification, and rollback.

For a new environment, first provision the ALB, ECS service, Aurora PostgreSQL,
EFS, Secrets Manager values, and telemetry described in
[AWS ECS Fargate deployment](aws-ecs-deployment.md). This runbook starts from an
existing service and changes only its Bedrock access and task definition.

## Current application boundary

ELSPETH has two distinct uses of an LLM:

1. **Pipeline LLM profiles** back `transform:llm` nodes and the tutorial
   pipeline. Bedrock is supported here through the ECS task role.
2. **The composer LLM** interprets chat and builds pipelines. Its boot-time
   provider contract currently supports Anthropic, Azure, OpenAI, and
   OpenRouter credentials, but not the Bedrock credential chain.

This runbook therefore makes both Bedrock models available to pipelines,
selects Sonnet for the tutorial, and leaves the composer on its existing
supported provider. Do not set `ELSPETH_WEB__COMPOSER_MODEL` or
`ELSPETH_WEB__COMPOSER_ADVISOR_MODEL` to `bedrock/...`: the service will report
the composer unavailable at startup. Making the composer itself use Bedrock is
an application change, not an ECS configuration change.

## Target configuration

| ELSPETH alias | Bedrock inference profile | Use |
|---|---|---|
| `bedrock-opus` | `global.anthropic.claude-opus-4-6-v1` | Higher-capability pipeline LLM |
| `bedrock-sonnet` | `global.anthropic.claude-sonnet-4-6` | General pipeline LLM and tutorial default |

The corresponding ELSPETH model strings are:

```text
bedrock/global.anthropic.claude-opus-4-6-v1
bedrock/global.anthropic.claude-sonnet-4-6
```

These are global cross-Region inference profiles. Bedrock may route requests
outside the source Region. If policy requires a bounded geography or a single
Region, stop here and select a supported inference profile from the live AWS
catalog; then replace the profile IDs, model strings, and IAM resources below.
An SCP that blocks any destination used by a cross-Region profile causes the
whole inference request to fail.

The web UI does not discover arbitrary account models. The operator defines
the complete selectable list in `ELSPETH_WEB__LLM_PROFILES`; web users see only
the opaque aliases. Adding the two entries above is what makes Opus and Sonnet
selectable. `transform:llm` is a required core web plugin, so do not add it to
`ELSPETH_WEB__PLUGIN_ALLOWLIST`.

## Prerequisites

- AWS CLI 2.27.42 or newer, `jq`, and a principal allowed to inspect ECS,
  Bedrock, and IAM.
- An approved immutable ELSPETH image digest. The image must include the
  `webui`, `llm`, `aws`, and `postgres` extras. No rebuild is needed for a
  configuration-only change to an image that already has them.
- An ECS/Fargate service with separate task execution and runtime task roles.
- Account authorization to accept the Anthropic terms and submit Anthropic's
  first-time-use details.
- A valid AWS Marketplace payment method. Model access and invocation are
  separate: enabling a model does not invoke it, but live verification does.
- Existing composer configuration on a supported provider. Preserve its model,
  advisor model, and credential secret unchanged.

Run the commands below in one shell. Replace the deployment-specific values.

```bash
set -Eeuo pipefail
umask 077

export AWS_PROFILE="<deployment-profile>"
export AWS_REGION="ap-southeast-1"
export ECS_CLUSTER="<cluster-name-or-arn>"
export ECS_SERVICE="<service-name-or-arn>"
export WEB_CONTAINER_NAME="elspeth-web"
export BASE_URL="https://<elspeth-hostname>"

export OPUS_BASE_MODEL="anthropic.claude-opus-4-6-v1"
export SONNET_BASE_MODEL="anthropic.claude-sonnet-4-6"
export OPUS_INFERENCE_PROFILE="global.anthropic.claude-opus-4-6-v1"
export SONNET_INFERENCE_PROFILE="global.anthropic.claude-sonnet-4-6"

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

ACCOUNT_ID=$(aws sts get-caller-identity \
  --profile "$AWS_PROFILE" \
  --query Account \
  --output text)
test -n "$ACCOUNT_ID" && test "$ACCOUNT_ID" != None
```

## 1. Confirm the exact models and inference profiles

Do not copy model identifiers from an old deployment. Confirm that the base
models are active and the global profiles exist in the deployment Region.

```bash
for model in "$OPUS_BASE_MODEL" "$SONNET_BASE_MODEL"; do
  aws bedrock get-foundation-model \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --model-identifier "$model" \
    --query 'modelDetails.{id:modelId,status:modelLifecycle.status}' \
    --output json \
    | jq -e --arg model "$model" '.id == $model and .status == "ACTIVE"'
done

for inference_profile in "$OPUS_INFERENCE_PROFILE" "$SONNET_INFERENCE_PROFILE"; do
  aws bedrock get-inference-profile \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --inference-profile-identifier "$inference_profile" \
    --output json \
    | jq -e --arg profile "$inference_profile" \
        '.inferenceProfileId == $profile and .status == "ACTIVE" and (.models | length > 0)'
done
```

If either lookup fails, use live discovery rather than guessing a replacement:

```bash
aws bedrock list-foundation-models \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --by-provider Anthropic \
  --query 'modelSummaries[?modelLifecycle.status==`ACTIVE`].[modelName,modelId]' \
  --output table

aws bedrock list-inference-profiles \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --type-equals SYSTEM_DEFINED \
  --query 'inferenceProfileSummaries[?contains(inferenceProfileName, `Claude`)].[inferenceProfileName,inferenceProfileId,status]' \
  --output table
```

Treat a model-version change as a deployment change: review it, update the IAM
resources and profile JSON together, register a new task-definition revision,
and repeat the live checks.

## 2. Enable Anthropic model access once per account

The identity that enables model access is not the ECS task role. Use an
authorized human or provisioning role with the required AWS Marketplace and
Bedrock model-access permissions. Review the model terms before continuing.

The simplest route is the Bedrock console:

1. Open **Amazon Bedrock → Model catalog** in `ap-southeast-1`.
2. Open Claude Opus 4.6 and Claude Sonnet 4.6.
3. Submit the Anthropic first-time-use details when prompted.
4. Request or enable access to both models and accept the applicable terms.
5. Wait for the account subscription to finish.

For an organization-managed CLI workflow, follow AWS's model-access procedure:
submit the Anthropic use-case form once, list each model's agreement offers,
and create the approved agreement. Do not improvise the form contents or accept
terms without the organization's authorization.

Access is ready only when all four fields pass for both base models:

```bash
for model in "$OPUS_BASE_MODEL" "$SONNET_BASE_MODEL"; do
  aws bedrock get-foundation-model-availability \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --model-id "$model" \
    --output json \
    | tee "$WORK_DIR/${model}.availability.json" \
    | jq -e '
        .agreementAvailability.status == "AVAILABLE"
        and .authorizationStatus == "AUTHORIZED"
        and .entitlementAvailability == "AVAILABLE"
        and .regionAvailability == "AVAILABLE"
      '
done
```

`PENDING`, `NOT_AVAILABLE`, `NOT_AUTHORIZED`, or `ERROR` is a stop condition.
Do not work around it by putting Marketplace permissions on the task role.
After access is enabled, runtime callers need Bedrock invocation permission,
not AWS Marketplace subscription permission.

## 3. Grant the runtime task role access to only these models

First capture the current task definition and identify its runtime role. The
`taskRoleArn` is the application identity; the `executionRoleArn` is for ECS
image pulls, logs, and secret injection. Bedrock permission belongs on the
former.

```bash
PREVIOUS_TASK_DEFINITION=$(aws ecs describe-services \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query 'services[0].taskDefinition' \
  --output text)
test -n "$PREVIOUS_TASK_DEFINITION" && test "$PREVIOUS_TASK_DEFINITION" != None

aws ecs describe-task-definition \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --task-definition "$PREVIOUS_TASK_DEFINITION" \
  --query taskDefinition \
  --output json >"$WORK_DIR/current-task-definition.json"

TASK_ROLE_ARN=$(jq -er '.taskRoleArn' "$WORK_DIR/current-task-definition.json")
EXECUTION_ROLE_ARN=$(jq -er '.executionRoleArn' "$WORK_DIR/current-task-definition.json")
test "$TASK_ROLE_ARN" != "$EXECUTION_ROLE_ARN"
TASK_ROLE_NAME=${TASK_ROLE_ARN##*/}
```

Create a resource-scoped policy. The model ARNs below are the resources
returned by the two global inference profiles in `ap-southeast-1`; the policy
includes both the account inference-profile ARN and every routed foundation
model ARN.

```bash
jq -n \
  --arg region "$AWS_REGION" \
  --arg account "$ACCOUNT_ID" \
  --arg opus "$OPUS_BASE_MODEL" \
  --arg sonnet "$SONNET_BASE_MODEL" \
  --arg opus_profile "$OPUS_INFERENCE_PROFILE" \
  --arg sonnet_profile "$SONNET_INFERENCE_PROFILE" \
  '{
    Version: "2012-10-17",
    Statement: [{
      Sid: "InvokeApprovedBedrockModels",
      Effect: "Allow",
      Action: ["bedrock:InvokeModel"],
      Resource: [
        "arn:aws:bedrock:\($region):\($account):inference-profile/\($opus_profile)",
        "arn:aws:bedrock:\($region):\($account):inference-profile/\($sonnet_profile)",
        "arn:aws:bedrock:::foundation-model/\($opus)",
        "arn:aws:bedrock:::foundation-model/\($sonnet)",
        "arn:aws:bedrock:\($region)::foundation-model/\($opus)",
        "arn:aws:bedrock:\($region)::foundation-model/\($sonnet)"
      ]
    }]
  }' >"$WORK_DIR/elspeth-bedrock-runtime-policy.json"

jq -e '.Statement[0].Resource | length == 6' \
  "$WORK_DIR/elspeth-bedrock-runtime-policy.json" >/dev/null
```

Apply that policy through the deployment's Terraform or CloudFormation. For a
manually managed disposable environment, the equivalent AWS CLI operation is:

```bash
aws iam put-role-policy \
  --profile "$AWS_PROFILE" \
  --role-name "$TASK_ROLE_NAME" \
  --policy-name elspeth-bedrock-opus-sonnet \
  --policy-document "file://$WORK_DIR/elspeth-bedrock-runtime-policy.json"
```

Do not add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_PROFILE`, credential-file variables, role overrides, or Bedrock endpoint
overrides to the task definition. ELSPETH deliberately uses the ECS task-role
credential chain.

## 4. Add the two operator profiles to the task definition

The complete value is a JSON object keyed by opaque alias. A Bedrock entry may
set `provider`, `model`, `region_name`, and `max_tokens`. Do not set
`credential_scope`, `credential_ref`, provider endpoints, or
`timeout_seconds`; those fields are invalid for a Bedrock web profile.

The commands below merge the two aliases into the current profile map. They do
not delete existing profiles or touch the existing plugin allowlist,
preferences, control modes, or Bedrock Guardrail profiles.

```bash
CURRENT_LLM_PROFILES=$(jq -er --arg container "$WEB_CONTAINER_NAME" '
  .containerDefinitions[]
  | select(.name == $container)
  | ([.environment[]? | select(.name == "ELSPETH_WEB__LLM_PROFILES") | .value][0] // "{}")
' "$WORK_DIR/current-task-definition.json")

LLM_PROFILES=$(jq -cn \
  --argjson existing "$CURRENT_LLM_PROFILES" \
  --arg region "$AWS_REGION" \
  --arg opus "bedrock/$OPUS_INFERENCE_PROFILE" \
  --arg sonnet "bedrock/$SONNET_INFERENCE_PROFILE" \
  '$existing + {
    "bedrock-opus": {
      "provider": "bedrock",
      "model": $opus,
      "region_name": $region,
      "max_tokens": 8192
    },
    "bedrock-sonnet": {
      "provider": "bedrock",
      "model": $sonnet,
      "region_name": $region,
      "max_tokens": 8192
    }
  }')

jq -e '
  .["bedrock-opus"].provider == "bedrock"
  and .["bedrock-sonnet"].provider == "bedrock"
' <<<"$LLM_PROFILES" >/dev/null
```

The `8192` output-token cap is a deliberate operational default, not a model
limit. Raise or lower it only after reviewing latency, output needs, and cost.

Reject a task definition that supplies any of the values being changed through
ECS secret bindings as well as ordinary environment entries; duplicate names
make the effective configuration ambiguous.

```bash
jq -e --arg container "$WEB_CONTAINER_NAME" '
  .containerDefinitions[]
  | select(.name == $container)
  | ((.secrets // []) | map(.name) | all(
      . != "ELSPETH_WEB__LLM_PROFILES"
      and . != "ELSPETH_WEB__TUTORIAL_LLM_PROFILE"
      and . != "ELSPETH_BEDROCK_LIVE_TEST_MODEL"
      and . != "AWS_REGION"
    ))
' "$WORK_DIR/current-task-definition.json" >/dev/null
```

Create a registrable task-definition document. Set only `AWS_REGION` for AWS
SDK Region selection; do not add `AWS_DEFAULT_REGION`, which the AWS acceptance
contract rejects. Preserve the composer settings and credentials exactly as
they are.

```bash
jq \
  --arg container "$WEB_CONTAINER_NAME" \
  --arg profiles "$LLM_PROFILES" \
  --arg tutorial_profile "bedrock-sonnet" \
  --arg live_model "bedrock/$SONNET_INFERENCE_PROFILE" \
  --arg region "$AWS_REGION" '
  def setenv($name; $value):
    .environment = (
      (.environment // [] | map(select(.name != $name)))
      + [{"name": $name, "value": $value}]
    );

  del(
    .taskDefinitionArn,
    .revision,
    .status,
    .requiresAttributes,
    .compatibilities,
    .registeredAt,
    .registeredBy,
    .deregisteredAt
  )
  | .containerDefinitions |= map(
      if .name == $container then
        setenv("ELSPETH_WEB__LLM_PROFILES"; $profiles)
        | setenv("ELSPETH_WEB__TUTORIAL_LLM_PROFILE"; $tutorial_profile)
        | setenv("ELSPETH_BEDROCK_LIVE_TEST_MODEL"; $live_model)
        | setenv("AWS_REGION"; $region)
      else . end
    )
' "$WORK_DIR/current-task-definition.json" \
  >"$WORK_DIR/candidate-task-definition.json"

jq -e --arg container "$WEB_CONTAINER_NAME" '
  [.containerDefinitions[] | select(.name == $container)] | length == 1
' "$WORK_DIR/candidate-task-definition.json" >/dev/null
```

Before registration, prove that the only task-definition differences for this
configuration-only rollout are the four environment values above. Do not print
the full task definition into chat, tickets, or logs; it contains private
deployment configuration and secret ARNs.

```bash
normalize_bedrock_candidate() {
  jq --arg container "$WEB_CONTAINER_NAME" '
    del(
      .taskDefinitionArn,
      .revision,
      .status,
      .requiresAttributes,
      .compatibilities,
      .registeredAt,
      .registeredBy,
      .deregisteredAt
    )
    | .containerDefinitions |= map(
        if .name == $container then
          .environment = ((.environment // []) | map(
            select(.name != "ELSPETH_WEB__LLM_PROFILES"
              and .name != "ELSPETH_WEB__TUTORIAL_LLM_PROFILE"
              and .name != "ELSPETH_BEDROCK_LIVE_TEST_MODEL"
              and .name != "AWS_REGION")
          ))
        else . end
      )
  ' "$1"
}

normalize_bedrock_candidate "$WORK_DIR/current-task-definition.json" \
  >"$WORK_DIR/current-normalized.json"
normalize_bedrock_candidate "$WORK_DIR/candidate-task-definition.json" \
  >"$WORK_DIR/candidate-normalized.json"
cmp --silent \
  "$WORK_DIR/current-normalized.json" \
  "$WORK_DIR/candidate-normalized.json"

jq -e \
  --arg container "$WEB_CONTAINER_NAME" \
  --arg profiles "$LLM_PROFILES" \
  --arg live_model "bedrock/$SONNET_INFERENCE_PROFILE" \
  --arg region "$AWS_REGION" '
  .containerDefinitions[]
  | select(.name == $container)
  | reduce (.environment // [])[] as $item ({}; .[$item.name] = $item.value)
  | .ELSPETH_WEB__LLM_PROFILES == $profiles
    and .ELSPETH_WEB__TUTORIAL_LLM_PROFILE == "bedrock-sonnet"
    and .ELSPETH_BEDROCK_LIVE_TEST_MODEL == $live_model
    and .AWS_REGION == $region
' "$WORK_DIR/candidate-task-definition.json" >/dev/null
```

Register the revision:

```bash
CANDIDATE_TASK_DEFINITION=$(aws ecs register-task-definition \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cli-input-json "file://$WORK_DIR/candidate-task-definition.json" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)
test -n "$CANDIDATE_TASK_DEFINITION" && test "$CANDIDATE_TASK_DEFINITION" != None
```

## 5. Run the deployment doctor before changing the service

Run the candidate as a one-shot task on the same network as the service. This
checks the deployment contract and both PostgreSQL schemas without admitting
traffic.

```bash
NETWORK_CONFIGURATION=$(aws ecs describe-services \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query 'services[0].networkConfiguration' \
  --output json)

DOCTOR_OVERRIDES=$(jq -cn --arg container "$WEB_CONTAINER_NAME" '{
  containerOverrides: [{
    name: $container,
    command: ["doctor", "aws-ecs", "--json"]
  }]
}')

DOCTOR_TASK_ARN=$(aws ecs run-task \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --task-definition "$CANDIDATE_TASK_DEFINITION" \
  --launch-type FARGATE \
  --network-configuration "$NETWORK_CONFIGURATION" \
  --overrides "$DOCTOR_OVERRIDES" \
  --query 'tasks[0].taskArn' \
  --output text)
test -n "$DOCTOR_TASK_ARN" && test "$DOCTOR_TASK_ARN" != None

aws ecs wait tasks-stopped \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --tasks "$DOCTOR_TASK_ARN"

DOCTOR_EXIT_CODE=$(aws ecs describe-tasks \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --tasks "$DOCTOR_TASK_ARN" \
  --query "tasks[0].containers[?name=='$WEB_CONTAINER_NAME'].exitCode | [0]" \
  --output text)
test "$DOCTOR_EXIT_CODE" = 0
```

If the service uses a capacity-provider strategy, pass that same strategy to
`run-task` and omit `--launch-type FARGATE`. A failed doctor is a stop
condition; inspect the one-shot task's stopped reason and CloudWatch log stream
before changing the service.

## 6. Deploy the candidate revision

The current ELSPETH ECS posture is one web task and permits planned downtime,
so use a zero-overlap replacement. Do not briefly run old and new application
revisions together against the same deployment state.

```bash
aws ecs update-service \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --service "$ECS_SERVICE" \
  --task-definition "$CANDIDATE_TASK_DEFINITION" \
  --deployment-configuration 'maximumPercent=100,minimumHealthyPercent=0' \
  --force-new-deployment >/dev/null

aws ecs wait services-stable \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE"

ACTIVE_TASK_DEFINITION=$(aws ecs describe-services \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query 'services[0].taskDefinition' \
  --output text)
test "$ACTIVE_TASK_DEFINITION" = "$CANDIDATE_TASK_DEFINITION"
```

## 7. Verify the running application

### Health and readiness

The liveness endpoint is `/api/health`; the ALB must route on `/api/ready`.
`/api/system/status` is the sanitized configuration/readiness view.

```bash
curl --fail --silent --show-error "$BASE_URL/api/health" | jq -e .
curl --fail --silent --show-error "$BASE_URL/api/ready" | jq -e .
curl --fail --silent --show-error "$BASE_URL/api/system/status" \
  | jq -e '
      .tutorial_ready == true
      and .composer_available == true
      and (.composer_provider | IN("anthropic", "azure", "azure_ai", "openai", "openrouter"))
    '
```

If authentication protects the status endpoint in the target environment,
supply an operator bearer token through the organization's normal secret-safe
HTTP workflow. A `401` from `/api/auth/me` before login is not a readiness
failure.

### Bedrock call through the deployed task role

ECS Exec must already be enabled for the service and authorized for the
operator. Run ELSPETH's bounded Bedrock smoke for the configured tutorial
model:

```bash
ACTIVE_TASK_ARN=$(aws ecs list-tasks \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --service-name "$ECS_SERVICE" \
  --desired-status RUNNING \
  --query 'taskArns[0]' \
  --output text)
test -n "$ACTIVE_TASK_ARN" && test "$ACTIVE_TASK_ARN" != None

aws ecs execute-command \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --task "$ACTIVE_TASK_ARN" \
  --container "$WEB_CONTAINER_NAME" \
  --interactive \
  --command 'python -m elspeth.web.aws_ecs_acceptance verify-bedrock'
```

This proves Sonnet invocation, response metadata handling, Region selection,
and task-role credentials. It does not prove Opus, pipeline lowering, or the
end-user workflow.

### Meaningful application proof

Complete these two short runs through the deployed web application:

1. Run the tutorial to completion. Inspect the committed pipeline and confirm
   its LLM node stores `profile: bedrock-sonnet`; confirm the run reaches a
   terminal successful status and contains a real LLM call record.
2. Create or copy a minimal text-source → LLM → text-sink pipeline. Select
   `bedrock-opus`, use a bounded prompt such as `Summarise {{ row.text }} in one
   sentence`, run one row, and confirm the terminal run and LLM call record.

The first run proves the configured tutorial selection. The second proves that
the non-default Opus alias is exposed, lowered to its private operator binding,
invoked through the ECS task role, and recorded by ELSPETH. Merely seeing two
aliases in the UI is not sufficient.

## Rollback

This change does not alter either database schema. If startup, readiness, or a
live model call fails, roll the service back to the captured task definition:

```bash
aws ecs update-service \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --service "$ECS_SERVICE" \
  --task-definition "$PREVIOUS_TASK_DEFINITION" \
  --deployment-configuration 'maximumPercent=100,minimumHealthyPercent=0' \
  --force-new-deployment >/dev/null

aws ecs wait services-stable \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE"
```

After rollback, decide whether to retain the model-access agreement and inline
task-role policy. Removing either is a separate IAM/model-access change; it is
not required to restore the previous application task definition.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Composer reports unavailable after rollout | Composer model was changed to `bedrock/...` | Restore the previous supported composer model and credential; leave Bedrock only in `LLM_PROFILES` |
| `transform:llm` says an operator profile is unavailable | Alias missing from the process-frozen profile map, invalid JSON, or old task still serving | Inspect sanitized status, confirm the active task-definition ARN, correct the profile map, and register/redeploy a new revision |
| Bedrock returns `AccessDeniedException` | Agreement not active, FTU missing, task-role policy incomplete, or an SCP denies a routed Region/model | Recheck all four availability fields, profile model ARNs, task role, and SCPs |
| Sonnet works but Opus does not | Opus agreement or one Opus ARN is missing from IAM | Compare the Opus availability and `get-inference-profile` output with the policy resources |
| `ValidationException` identifies the model/profile | Stale or malformed model identifier | Repeat live model and inference-profile discovery; keep the `bedrock/` prefix only in ELSPETH config |
| Tutorial uses the wrong model | `ELSPETH_WEB__TUTORIAL_LLM_PROFILE` names the wrong alias or an old task is serving | Set it to `bedrock-sonnet`, register a new revision, and replace the task |
| Bedrock profile validation fails on timeout | `timeout_seconds` was supplied | Remove it; current Bedrock web profiles do not accept an operator timeout field |
| Model works in the engineer's shell but not ECS | Shell credentials have broader access than the task role | Diagnose from inside the running task and fix `taskRoleArn`; never copy shell credentials into ECS |
| Cross-Region invocation fails despite correct IAM | SCP or data-residency controls block a destination Region | Allow every destination used by the profile, or select a permitted geographic/in-Region profile |

## Handoff checklist

- [ ] Exact Opus and Sonnet base models are active in the deployment Region.
- [ ] Both global inference profiles are active and their routed model ARNs
      were reviewed.
- [ ] Anthropic FTU and both model agreements report fully available.
- [ ] Marketplace permissions were used only by the onboarding identity, not
      the ECS task role.
- [ ] Runtime `taskRoleArn` has resource-scoped `bedrock:InvokeModel` for both
      inference profiles and all routed foundation-model ARNs.
- [ ] No static AWS credentials, credential-profile variables, role overrides,
      or endpoint overrides are present in the task definition.
- [ ] `ELSPETH_WEB__LLM_PROFILES` contains both opaque aliases.
- [ ] `ELSPETH_WEB__TUTORIAL_LLM_PROFILE=bedrock-sonnet`.
- [ ] Existing composer provider/model/credential settings are unchanged.
- [ ] Candidate doctor passed before service update.
- [ ] Service is stable on the exact candidate task-definition ARN.
- [ ] Health, readiness, sanitized status, Sonnet tutorial, and Opus pipeline
      checks passed.
- [ ] Previous task-definition ARN is recorded and rollback was not needed.

## References

- [ELSPETH AWS ECS deployment](aws-ecs-deployment.md)
- [ELSPETH AWS ECS health and readiness](../operator/aws-ecs-health-and-readiness.md)
- [ELSPETH web plugin policy configuration](../reference/configuration.md#web-plugin-policy)
- [AWS: Request access to models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
- [AWS: Claude Opus 4.6 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-6.html)
- [AWS: Claude Sonnet 4.6 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html)
- [AWS: Global cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html)
- [AWS: Geographic cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/geographic-cross-region-inference.html)
- [AWS: Supported models and inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)
- [AWS: Discover foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-get-info.html)
