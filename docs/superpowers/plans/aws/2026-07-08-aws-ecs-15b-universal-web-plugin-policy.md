# Plan 15B Universal Web Plugin Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Also use
> superpowers:test-driven-development, using-security-architect,
> config-contracts-guide, logging-telemetry-policy, and wardline-gate. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give operators one core-only, kind-qualified, fail-closed policy that
controls every web-visible and web-executable plugin while leaving CLI/batch
access to the complete installed registry.

**Architecture:** Keep `PluginManager` and `CatalogServiceImpl` as the full
process registry. Compile frozen `WebPluginPolicy` and request-scoped
`PluginAvailabilitySnapshot` values, wrap the catalog with the snapshot, and
enforce the same decision at discovery, authoring, import, validation,
execution, delayed export, readiness, and audit boundaries. Web LLM and future
Guardrail configurations use opaque operator profiles; explicit expert config
remains available to CLI/batch.

**Tech Stack:** Python 3.12+, Pydantic v2, FastAPI, SQLAlchemy, React/TypeScript,
ELSPETH plugin/catalog/composer/runtime contracts, pytest, Hypothesis, Vitest,
Playwright, Ruff, mypy, Wardline.

**Depends on:** Plan 15A (`elspeth-130dc48252`), Plan 05
(`elspeth-1a1c31bcce`), Plan 08A (`elspeth-c0103e6c88`), Plan 09
(`elspeth-e8dc754360`), Plan 11 (`elspeth-25286192ee`), and Plan 14
(`elspeth-f5d5dddddf`), and transitively the shared
signed-tier/trust-boundary/Wardline baseline. Plan 15B is the integration owner
for the final readiness, guarded Landscape opener, and telemetry paths on these
shared web surfaces.

**Blocks:** Plan 15C Bedrock Guardrail shields, Plan 03B integration proof,
Plan 10 packaging/deployment, and Plan 12 closeout.

**Deployment rule:** No commit or intermediate branch containing only catalog
filtering is deployable. The Filigree step closes only after the runtime and
audit backstops pass.

---

### Task 0: Claim the exact slice and freeze the code-reality map

**Files:**

- Read: `docs/superpowers/specs/2026-07-12-universal-web-plugin-policy-design.md`
- Read: `src/elspeth/plugins/infrastructure/manager.py`
- Read: `src/elspeth/web/catalog/service.py`
- Read: `src/elspeth/web/composer/tools/_common.py`
- Read: `src/elspeth/web/execution/validation.py`
- Read: `src/elspeth/web/execution/service.py`
- Read: `src/elspeth/web/composer/recipes.py`
- Read: `src/elspeth/web/interpretation_state.py`

- [ ] **Step 1: Create an isolated implementation worktree at the integrated prerequisites**

```bash
git status --short
git check-ignore -q .worktrees
BASE_SHA="$(git rev-parse release/0.7.1)"
git worktree add .worktrees/aws-ecs-15b-plugin-policy -b feat/aws-ecs-15b-plugin-policy "$BASE_SHA"
cd .worktrees/aws-ecs-15b-plugin-policy
git rev-parse HEAD
```

Expected: the release tip contains the close commits recorded on Plans 15A,
05, 08A, 09, 11, and 14. If any recorded close commit is not an ancestor of
`HEAD`, stop; do not merge plan branches independently or text-merge `uv.lock`.

- [ ] **Step 2: Atomically start the Filigree step**

```bash
filigree start-work elspeth-0674a06468 --assignee codex --actor codex
```

Expected: the step enters `in_progress` only when every dependency is closed.

- [ ] **Step 3: Record the baseline and current bypass inventory**

```bash
git rev-parse HEAD
git status --short
uv run pytest -q \
  tests/unit/web/catalog \
  tests/unit/web/composer/test_tools.py \
  tests/unit/web/execution/test_validation.py \
  tests/integration/web/test_catalog_discovery.py
```

Expected: the selected baseline passes. Record the SHA and test result in a
Filigree comment before editing.

---

### Task 1: Define canonical plugin identities, capabilities, profiles, and policy compilation

**Files:**

- Create: `src/elspeth/contracts/plugin_capabilities.py`
- Create: `src/elspeth/web/plugin_policy/__init__.py`
- Create: `src/elspeth/web/plugin_policy/models.py`
- Create: `src/elspeth/web/plugin_policy/compiler.py`
- Create: `src/elspeth/web/plugin_policy/profiles.py`
- Modify: `src/elspeth/web/config.py`
- Modify: `src/elspeth/web/app.py`
- Create: `tests/unit/web/plugin_policy/test_models.py`
- Create: `tests/unit/web/plugin_policy/test_compiler.py`
- Create: `tests/unit/web/plugin_policy/test_profiles.py`
- Modify: `tests/unit/web/test_config.py`
- Modify: `tests/unit/web/test_app.py`

- [ ] **Step 1: Write failing identity, core, preference, hash, and config tests**

Pin the required set and canonical parser:

```python
REQUIRED_WEB_PLUGIN_IDS = frozenset(
    {
        PluginId("source", "csv"),
        PluginId("source", "json"),
        PluginId("source", "text"),
        PluginId("sink", "csv"),
        PluginId("sink", "json"),
        PluginId("sink", "text"),
        PluginId("transform", "field_mapper"),
        PluginId("transform", "llm"),
        PluginId("transform", "web_scrape"),
    }
)


def test_policy_hash_sorts_sets_but_preserves_preference_order() -> None:
    settings = _settings(
        allowlist=("transform:azure_prompt_shield", "sink:database"),
        prompt_order=("transform:azure_prompt_shield", "transform:aws_bedrock_prompt_shield"),
    )
    first = compile_policy(_registry(), settings)
    reordered_set = compile_policy(_registry(), replace(settings, allowlist=tuple(reversed(settings.allowlist))))
    reversed_preference = compile_policy(
        _registry(),
        replace(settings, prompt_order=tuple(reversed(settings.prompt_order))),
    )

    assert first.policy_hash == reordered_set.policy_hash
    assert first.policy_hash != reversed_preference.policy_hash
```

Add startup-failure tests for missing core, malformed/wrong-case/wrong-kind
IDs, duplicate allowlist entries, explicitly allowed but uninstalled plugins,
incomplete preference orders, capability mismatches, and `required` control
mode with no authorized implementation. Add a generic local-requirement test
plugin whose class is discoverable but optional package is absent: unallowlisted
startup succeeds, explicit authorization fails with sanitized
`plugin_unavailable`, and a satisfied local requirement succeeds without any
network call.

Also reject every authorized plugin whose `plugin_version` is missing,
`0.0.0`, or not canonical SemVer, or whose `source_file_hash` is `None` or not
the exact `sha256:` plus 16-lowercase-hex repository contract. Base-class
defaults are not valid code identities. Marker tests prove invalid metadata is
reported by kind-qualified plugin ID and closed reason only, never by raw
attribute value.

- [ ] **Step 2: Run the new domain tests and verify they fail**

```bash
uv run pytest -q tests/unit/web/plugin_policy/test_models.py tests/unit/web/plugin_policy/test_compiler.py
```

Expected: collection fails because the policy package does not exist.

- [ ] **Step 3: Implement the closed capability and configuration-authority contracts**

`contracts/plugin_capabilities.py` is stdlib-only:

```python
class PluginCapability(StrEnum):
    LLM = "llm"
    PROMPT_SHIELD = "prompt_shield"
    CONTENT_SAFETY = "content_safety"


class ControlMode(StrEnum):
    RECOMMEND = "recommend"
    REQUIRED = "required"


class WebConfigAuthority(StrEnum):
    USER_CONFIGURABLE = "user_configurable"
    USER_CONFIGURABLE_WITH_POLICY = "user_configurable_with_policy"
    OPERATOR_PROFILED = "operator_profiled"


class ControlRole(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class ContentTrust(StrEnum):
    TRUSTED_INTERNAL = "trusted_internal"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True, slots=True)
class CapabilityDeclaration:
    capability: PluginCapability
    control_role: ControlRole | None = None
    blocks_positive_detection: bool = False
```

Keep open `capability_tags` separate. A plugin cannot claim a closed capability
using a bare string. `PROMPT_SHIELD` requires role `INPUT`, `CONTENT_SAFETY`
requires role `OUTPUT`, and `LLM` has no control role. Add a closed producer
trust declaration: all web user inputs, source rows, and external-call outputs
are `UNTRUSTED`; only explicitly code-declared internal constants may be
`TRUSTED_INTERNAL`. Unknown producers are untrusted.

- [ ] **Step 4: Implement immutable policy and snapshot models**

```python
@dataclass(frozen=True, slots=True, order=True)
class PluginId:
    kind: Literal["source", "transform", "sink"]
    name: str

    @classmethod
    def parse(cls, raw: str) -> "PluginId":
        match = re.fullmatch(r"(source|transform|sink):([a-z][a-z0-9_]*)", raw)
        if match is None:
            raise ValueError("invalid kind-qualified plugin id")
        return cls(cast(Literal["source", "transform", "sink"], match.group(1)), match.group(2))

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True, slots=True)
class WebPluginPolicy:
    schema_version: int
    required: frozenset[PluginId]
    configured_optional: frozenset[PluginId]
    authorized: frozenset[PluginId]
    preferences: tuple[tuple[PluginCapability, tuple[PluginId, ...]], ...]
    control_modes: tuple[tuple[PluginCapability, ControlMode], ...]
    plugin_code_identities: tuple[tuple[PluginId, str, str], ...]
    policy_hash: str


@dataclass(frozen=True, slots=True)
class PluginAvailabilitySnapshot:
    policy_hash: str
    principal_scope: str
    available: frozenset[PluginId]
    unavailable: tuple[PluginAvailability, ...]
    selected: tuple[tuple[PluginCapability, PluginId | None], ...]
    usable_profile_aliases: tuple[tuple[PluginId, tuple[str, ...]], ...]
    selected_profile_aliases: tuple[tuple[PluginId, str | None], ...]
    binding_generation_fingerprint: str
    snapshot_hash: str
```

Parsing errors are wrapped by the settings layer with only safe setting name
and collection index; the raw token is never interpolated. Canonical hashing
sorts sets and map keys but preserves each preference tuple.
The policy hash includes every authorized plugin's exact canonical
`(PluginId, plugin_version, source_file_hash)` identity, so a code change
changes authorization evidence. The snapshot hash includes usable aliases,
safe selected/default aliases, and a server-keyed generation fingerprint; it
never hashes raw or low-entropy private bindings. Add multi-process/restart
determinism and code/profile-generation drift tests. Reason codes are closed
and sanitized; they contain no secret names or values.

- [ ] **Step 5: Add WebSettings and explicit runtime conversion**

Add universal settings plus typed LLM profiles:

```python
plugin_allowlist: tuple[str, ...] = ()
plugin_preferences: Mapping[PluginCapability, tuple[str, ...]] = Field(default_factory=dict)
plugin_control_modes: Mapping[PluginCapability, ControlMode] = Field(
    default_factory=lambda: {
        PluginCapability.PROMPT_SHIELD: ControlMode.RECOMMEND,
        PluginCapability.CONTENT_SAFETY: ControlMode.RECOMMEND,
    }
)
llm_profiles: Mapping[str, WebLLMProfileSettings] = Field(default_factory=dict)
tutorial_llm_profile: str | None = None
```

`WebLLMProfileSettings` has a validated opaque alias, a closed provider
variant from Plan 09's registry, approved model ID, explicit
`credential_scope: server|user`, credential-ref binding, and provider-specific
bounded settings. Server scope resolves only through the operator server store;
user scope resolves only through the current principal's store, so ordinary
user-first lookup cannot shadow an operator binding. It has masked/sanitized
`repr` and forbids arbitrary base URLs. Convert these fields immediately into
immutable tuples in `RuntimeWebPluginConfig.from_settings`; add a
field-consumption alignment test so no setting can be accepted and ignored.
The JSON/YAML shape is an alias-keyed mapping, not an order-bearing tuple;
reject alias/key disagreements and duplicates after canonicalization.

- [ ] **Step 6: Parse JSON settings without echoing raw environment values**

Extend `_settings_from_env` with explicit tuple and object collections. Route
all Pydantic errors through one structured sanitizer built from
`ValidationError.errors(include_input=False)` and safe field paths. Error
messages name `ELSPETH_WEB__PLUGIN_PREFERENCES` or the relevant setting but do
not include `value!r`, `input_value`, raw JSON, nested model reprs, private
bindings, or secret references. Marker-value tests cover malformed JSON, wrong
JSON type, unknown fields, duplicate profile aliases, startup compiler errors,
HTTP errors/logs, and masked profile `repr`.

- [ ] **Step 7: Compile and store the process policy**

After constructing the complete catalog and before composer services, call:

```python
app.state.web_plugin_policy = compile_web_plugin_policy(
    registry=get_shared_plugin_manager(),
    settings=RuntimeWebPluginConfig.from_settings(settings),
)
```

The compiler performs local validation only; it makes no credential, network,
filesystem, or paid provider call.

- [ ] **Step 8: Run and commit the policy foundation**

```bash
uv run pytest -q \
  tests/unit/web/plugin_policy/test_models.py \
  tests/unit/web/plugin_policy/test_compiler.py \
  tests/unit/web/plugin_policy/test_profiles.py \
  tests/unit/web/test_config.py \
  tests/unit/web/test_app.py
git add \
  src/elspeth/contracts/plugin_capabilities.py \
  src/elspeth/web/plugin_policy \
  src/elspeth/web/config.py \
  src/elspeth/web/app.py \
  tests/unit/web/plugin_policy \
  tests/unit/web/test_config.py \
  tests/unit/web/test_app.py
git commit -m "feat(web): compile universal plugin policy"
```

Expected: tests pass and only policy/config files enter the commit.

---

### Task 2: Publish typed capabilities and operator-profiled web schemas

**Files:**

- Modify: `src/elspeth/contracts/plugin_protocols.py`
- Modify: `src/elspeth/plugins/infrastructure/base.py`
- Modify: `src/elspeth/plugins/transforms/llm/transform.py`
- Modify: `src/elspeth/plugins/transforms/azure/prompt_shield.py`
- Modify: `src/elspeth/plugins/transforms/azure/content_safety.py`
- Modify: `src/elspeth/web/catalog/schemas.py`
- Modify: `src/elspeth/web/catalog/service.py`
- Modify: `src/elspeth/web/plugin_policy/profiles.py`
- Modify: `tests/unit/plugins/infrastructure/test_base_metadata.py`
- Modify: `tests/unit/web/catalog/test_service.py`
- Modify: `tests/unit/web/catalog/test_schemas.py`
- Create: `tests/unit/web/catalog/test_policy_view_golden.py`
- Modify: `tests/unit/web/test_interpretation_state.py`
- Modify: `tests/golden/web/catalog/knob_schema/transform__llm.json`
- Create: `tests/golden/web/catalog/policy_view/transform__llm.json`

- [ ] **Step 1: Write failing metadata and public-schema tests**

```python
def test_security_controls_use_typed_capabilities() -> None:
    assert AzurePromptShield.policy_capabilities == frozenset(
        {CapabilityDeclaration(PluginCapability.PROMPT_SHIELD, ControlRole.INPUT, blocks_positive_detection=True)}
    )
    assert AzureContentSafety.policy_capabilities == frozenset(
        {CapabilityDeclaration(PluginCapability.CONTENT_SAFETY, ControlRole.OUTPUT, blocks_positive_detection=True)}
    )


def test_web_llm_schema_exposes_profile_not_private_provider_binding(policy_view: PolicyCatalogView) -> None:
    schema = policy_view.get_schema("transform", "llm").json_schema
    rendered = json.dumps(schema)
    assert '"profile"' in rendered
    assert '"api_key"' not in rendered
    assert '"base_url"' not in rendered
```

Also test that bare strings fail metadata contract checks and the complete
internal catalog still emits the CLI provider-discriminated LLM schema.

- [ ] **Step 2: Add metadata defaults to all plugin protocols/bases**

```python
web_config_authority: WebConfigAuthority = WebConfigAuthority.USER_CONFIGURABLE
policy_capabilities: frozenset[CapabilityDeclaration] = frozenset()
```

Declare LLM as `OPERATOR_PROFILED` with `LLM`; Azure controls as
`USER_CONFIGURABLE_WITH_POLICY` plus their respective security capability.
Update catalog DTOs to serialize these closed values.

- [ ] **Step 3: Implement generic operator-profile schema and lowering seams**

Define an `OperatorProfileResolver` protocol keyed by `PluginId` with the exact
methods `public_schema(full_schema, available_aliases) -> PluginSchemaInfo`,
`lower_options(alias, safe_options, credential_context) -> LoweredPluginConfig`,
`profile_availability(principal, server_store, user_store) -> tuple[ProfileAvailability, ...]`,
and `check_local_requirements(alias) -> LocalRequirementResult`.

`ProfileAvailability` is frozen and contains alias, `credential_scope`, usable
boolean, and a closed sanitized reason code. Server-scoped aliases consult only
the server store; user-scoped aliases consult only the current principal's
store. A user value with the same name can never shadow a server binding.
Availability is rebuilt per request/execution, so secret deletion/rotation
narrows the next snapshot. Public schemas enumerate only usable aliases for the
current principal.

An `OperatorProfileRegistry` owns the resolver map and exposes
`public_schema(plugin_id, full_schema, snapshot)`, `lower_options(plugin_id,
alias, safe_options, credential_context)`, `profile_availability(...)`, and
`check_local_requirements(plugin_id, alias)`; it is constructed with the frozen
policy so call signatures stay stable. The LLM resolver accepts
`profile` plus safe prompt/row/schema options, rejects
raw `provider`, `model`, `api_key`, endpoint, or profile-plus-raw mixtures, and
lowers to a copied runtime config. `LoweredPluginConfig` carries separate
`executable_options` and `audit_safe_options`: executable options may contain
private provider/model/credential references in memory, while the audit-safe
copy retains alias plus non-reversible fingerprints only. Persisted composition
state, `runs.config_json`, node `config_json`, exports, errors, logs, and
telemetry must receive only authored/audit-safe values. Add marker tests proving
private provider/model/endpoint/credential/profile-binding values have zero
occurrences in every persisted/exported/observable surface.

- [ ] **Step 4: Replace name-based prompt-shield credit with typed metadata**

Remove `_AUTHORIZED_PROMPT_SHIELD_PLUGINS` as an authorization source. Existing
advisory graph code receives typed declarations/effective configuration. No
plugin receives control credit from its name.

- [ ] **Step 5: Recompute hashes/goldens and run contract tests**

```bash
uv run python - <<'PY'
from pathlib import Path
from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash

for path, class_name in (
    (Path("src/elspeth/plugins/transforms/llm/transform.py"), "LLMTransform"),
    (Path("src/elspeth/plugins/transforms/azure/prompt_shield.py"), "AzurePromptShield"),
    (Path("src/elspeth/plugins/transforms/azure/content_safety.py"), "AzureContentSafety"),
):
    fix_source_file_hash(path, class_name, compute_source_file_hash(path))
PY
uv run pytest -q \
  tests/unit/plugins/infrastructure/test_base_metadata.py \
  tests/unit/web/catalog/test_service.py \
  tests/unit/web/catalog/test_schemas.py \
  tests/unit/web/test_interpretation_state.py \
  tests/unit/web/catalog/test_knob_schema_golden.py \
  tests/unit/web/catalog/test_policy_view_golden.py
```

Generate the ordinary `tests/golden/web/catalog/knob_schema/transform__llm.json`
with the existing `CatalogServiceImpl` helper and keep it as the full explicit
CLI schema. Separately generate
`tests/golden/web/catalog/policy_view/transform__llm.json` from a
`PolicyCatalogView` built with a deterministic principal snapshot and usable
LLM profile. The policy-view golden contains `profile` and safe row/prompt
options only. Expected: both suites pass; private provider fields are absent
only from the policy-view golden.

- [ ] **Step 6: Commit typed capabilities and profiles**

```bash
git add src/elspeth/contracts src/elspeth/plugins/infrastructure/base.py \
  src/elspeth/plugins/transforms/llm/transform.py \
  src/elspeth/plugins/transforms/azure \
  src/elspeth/web/catalog src/elspeth/web/plugin_policy/profiles.py \
  src/elspeth/web/interpretation_state.py tests/unit tests/golden/web/catalog
git commit -m "feat(web): publish typed plugin capabilities"
```

---

### Task 3: Build one request snapshot and one policy catalog view

**Files:**

- Create: `src/elspeth/web/plugin_policy/availability.py`
- Create: `src/elspeth/web/catalog/policy_view.py`
- Modify: `src/elspeth/web/catalog/routes.py`
- Modify: `src/elspeth/web/dependencies.py`
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/composer/tools/_availability.py`
- Delete: `src/elspeth/web/composer/tools/_shield_availability.py`
- Create: `tests/unit/web/plugin_policy/test_availability.py`
- Create: `tests/unit/web/catalog/test_policy_view.py`
- Modify: `tests/unit/web/catalog/test_routes.py`
- Modify: `tests/integration/web/test_catalog_discovery.py`

- [ ] **Step 1: Write failing availability and enumeration tests**

Cover core-only, optional allowed, installed-but-disabled, missing request
credential, missing operator profile, Azure-only, multiple implementations,
and neither implementation:

```python
def test_user_secret_can_narrow_but_never_expand_policy() -> None:
    snapshot = build_snapshot(
        policy=_core_only_policy(),
        catalog=_catalog_with_azure_prompt_shield(),
        principal=_principal_with("AZURE_CONTENT_SAFETY_KEY"),
    )
    assert PluginId("transform", "azure_prompt_shield") not in snapshot.available


def test_direct_schema_url_cannot_enumerate_disabled_plugin(client: TestClient) -> None:
    response = client.get("/api/catalog/transforms/azure_prompt_shield/schema")
    assert response.status_code == 404
    assert "azure_prompt_shield" not in response.json()["detail"]
```

No resolver/principal context is a closed failure for credentialed plugins.

- [ ] **Step 2: Implement the pure snapshot builder**

For each authorized ID, combine registration, generic local requirement checks,
per-alias profile availability, explicit credential scope, and catalog-declared
secret requirements. Select by complete preference order. An operator-profiled
plugin is available only when at least one alias is usable for this principal;
its public schema contains only those aliases. Revalidate the selected alias and
scope immediately before execution. Compute a fingerprint from sanitized
identities/reason codes and the server-keyed binding generation only. Do not
call remote providers and do not cache across principals. Tests cover
server-only, user-only, mixed aliases, shadow attempts, secret deletion,
credential rotation, profile restart generation, and a selected alias becoming
unusable between authoring and execution.

- [ ] **Step 3: Implement `PolicyCatalogView`**

```python
class PolicyCatalogView(CatalogService):
    def __init__(
        self,
        full: CatalogService,
        snapshot: PluginAvailabilitySnapshot,
        profiles: OperatorProfileRegistry,
    ) -> None:
        self._full = full
        self._snapshot = snapshot
        self._profiles = profiles

    def list_transforms(self) -> list[PluginSummary]:
        return [item for item in self._full.list_transforms() if PluginId("transform", item.name) in self._snapshot.available]

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        plugin_id = PluginId(plugin_type, name)
        self._require_available(plugin_id)
        return self._profiles.public_schema(plugin_id, self._full.get_schema(plugin_type, name), self._snapshot)
```

All error lists derive from the same visible set. `post_call_hints` applies the
same gate. Add a two-principal test whose usable alias sets differ and prove
each schema enumerates only that snapshot's aliases.

- [ ] **Step 4: Make catalog routes principal-aware and fingerprinted**

Each handler obtains the authenticated `UserIdentity`, builds one snapshot,
uses one view, and returns/headers the snapshot fingerprint. Add a dedicated
`GET /api/catalog/policy` response containing fingerprint, safe capability
groups, selections, and control modes so the frontend never infers policy from
three unrelated list requests. Every principal-dependent catalog list, schema,
assistance, and policy response sets `Cache-Control: private, no-store` and
`Vary: Authorization, Cookie` (plus any existing representation headers).
Route tests exercise two principals through a cache-simulating client and prove
no response can be reused across them.

- [ ] **Step 5: Remove the fail-open secret-only/Azure-only availability path**

Refactor callers to consume the snapshot. Delete `_shield_availability.py`;
there must be exactly one availability implementation.

- [ ] **Step 6: Run and commit snapshot/catalog tests**

```bash
uv run pytest -q \
  tests/unit/web/plugin_policy/test_availability.py \
  tests/unit/web/catalog/test_policy_view.py \
  tests/unit/web/catalog/test_routes.py \
  tests/integration/web/test_catalog_discovery.py
git add src/elspeth/web/plugin_policy src/elspeth/web/catalog \
  src/elspeth/web/dependencies.py src/elspeth/web/app.py \
  src/elspeth/web/composer/tools tests/unit/web tests/integration/web/test_catalog_discovery.py
git commit -m "feat(web): expose request-scoped plugin catalog"
```

---

### Task 4: Thread the snapshot through freeform composition and close direct-tool bypasses

**Files:**

- Modify: `src/elspeth/web/composer/tools/_common.py`
- Modify: `src/elspeth/web/composer/tools/_dispatch.py`
- Modify: `src/elspeth/web/composer/tool_batch.py`
- Modify: `src/elspeth/web/composer/tools/sources.py`
- Modify: `src/elspeth/web/composer/tools/transforms.py`
- Modify: `src/elspeth/web/composer/tools/outputs.py`
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/composer/tools/generation.py`
- Modify: `src/elspeth/web/composer/prompts.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/composer_mcp/server.py`
- Modify: `tests/unit/web/composer/test_tools.py`
- Modify: `tests/unit/web/composer/test_prompts.py`
- Modify: `tests/unit/web/composer/test_failure_schema_augmentation.py`
- Modify: `tests/property/web/composer/test_compose_loop_invariants.py`

- [ ] **Step 1: Write direct-call and single-snapshot regressions**

```python
def test_direct_upsert_cannot_name_hidden_registered_plugin() -> None:
    result = execute_tool(
        "upsert_node",
        {"id": "shield", "plugin": "azure_prompt_shield", "options": _azure_shield_options()},
        _state(),
        _core_only_policy_view(),
        plugin_snapshot=_core_only_snapshot(),
    )
    assert result.success is False
    assert result.data["error_code"] == "plugin_not_enabled"


def test_one_compose_turn_threads_one_snapshot_object() -> None:
    service = _service(snapshot_factory=_counting_factory())
    service.compose(session_id="session-1", user_id="user-1", message="Build a CSV pipeline")
    assert snapshot_factory.calls == 1
    assert prompt_snapshot is tool_snapshot is validation_snapshot
```

Add equivalent tests for source, transform, aggregation, sink, `set_pipeline`,
schema, assistance, post-call hints, and failure-schema augmentation.

- [ ] **Step 2: Make `ToolContext` require policy context on web dispatch**

Add `plugin_snapshot` and policy catalog view fields. `_validate_plugin_name`
first distinguishes not installed from not enabled/unavailable, then validates
the public schema. No optional default may mean allow-all.

- [ ] **Step 3: Gate all canonical mutations and bulk replacement**

Use the shared helper in source, transform/aggregation, output, and
`_execute_set_pipeline`. New selections require availability. A mutation that
removes/replaces an already-disabled historical component is allowed; a bulk
candidate must contain only available identities.

- [ ] **Step 4: Emit the safe target-LLM capability inventory**

`build_context_string` emits available IDs, safe capability groups, selected
implementation, and control mode from the snapshot. It emits no secret ref
names, LLM provider binding, profile private values, environment names, or
failure details. Intersect session schema-loaded caches with available IDs
before rendering.

- [ ] **Step 5: Make the local Composer MCP explicitly trained-operator**

The MCP path is not the web service. Construct an explicit
`PluginAvailabilitySnapshot.for_trained_operator(full_catalog)` and matching
full policy view in `_dispatch_tool`. Do not rely on `None` to mean allow-all;
tests pin that only this named constructor can produce the unrestricted view.

- [ ] **Step 6: Run and commit the freeform boundary**

```bash
uv run pytest -q \
  tests/unit/web/composer/test_tools.py \
  tests/unit/web/composer/test_prompts.py \
  tests/unit/web/composer/test_failure_schema_augmentation.py \
  tests/unit/web/composer/test_service.py \
  tests/property/web/composer/test_compose_loop_invariants.py \
  tests/unit/composer_mcp
git add src/elspeth/web/composer src/elspeth/composer_mcp tests/unit/web/composer tests/property/web/composer tests/unit/composer_mcp
git commit -m "feat(web): enforce plugin policy in freeform authoring"
```

---

### Task 5: Close guided, recipe, YAML-import, and saved-state bypasses

**Files:**

- Modify: `src/elspeth/web/composer/guided/_discovery.py`
- Modify: `src/elspeth/web/composer/guided/steps.py`
- Modify: `src/elspeth/web/composer/guided/emitters.py`
- Modify: `src/elspeth/web/composer/guided/chat_solver.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/composer/recipes.py`
- Modify: `src/elspeth/web/composer/tools/recipes.py`
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/web/sessions/routes/composer/state.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/routes/_helpers.py`
- Modify: `tests/unit/web/composer/test_recipes.py`
- Modify: `tests/unit/web/composer/test_yaml_importer.py`
- Modify: `tests/integration/web/composer/guided/test_step_handlers.py`
- Modify: `tests/integration/web/composer/guided/test_step_3_e2e.py`
- Modify: `tests/integration/web/composer/guided/test_chain_discovery_loop.py`
- Modify: `tests/integration/web/composer/guided/test_sink_discovery_loop.py`

- [ ] **Step 1: Write bypass and atomicity tests before wiring**

Pin direct guided submission, hard-coded guided prompt names, deterministic
recipe fast path, recipe application, and YAML import:

```python
def test_import_with_disabled_plugin_is_atomic(client: TestClient, session: SessionRecord) -> None:
    before = client.get(f"/api/sessions/{session.id}/state").json()
    response = client.post(f"/api/sessions/{session.id}/import", files={"file": _yaml_with_database_sink()})
    after = client.get(f"/api/sessions/{session.id}/state").json()

    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "plugin_not_enabled"
    assert after == before
```

Also prove saved disabled states remain readable, exportable in their authored
form, and can delete/replace the disabled component without fetching its
private schema. Export must not lower or reveal a private profile binding.

- [ ] **Step 2: Thread the same request snapshot through guided code**

Remove bare `ToolContext` construction from guided steps. Emitters, discovery
loops, form submissions, final step-3 accept, and hard-coded prompt lists all
consume the request policy view. Guided completion runs required-control
coverage; intermediate edits do not.

- [ ] **Step 3: Declare/filter recipe dependencies before side effects**

Add required IDs to `RecipeSpec`, including alternatives for dynamic source
slots. Filter recipe listing and validate the built candidate before blob or
state persistence. The freeform deterministic fast path skips an unavailable
recipe before `create_blob`.

- [ ] **Step 4: Enforce web import before persistence**

Keep `composition_state_from_runtime_yaml` CLI-neutral. Immediately after
structural parse, scan all sources, transform/aggregation nodes, and sinks with
the request snapshot. Return the typed policy error before blob binding or
`save_state`.

- [ ] **Step 5: Return sanitized historical-policy findings**

Extend state responses with component ID, kind-qualified plugin ID, closed
reason code, and snapshot fingerprint. Never return private schema or binding
details. Frontend work in Task 8 consumes this field.

- [ ] **Step 6: Run and commit the guided/import boundary**

```bash
uv run pytest -q \
  tests/unit/web/composer/test_recipes.py \
  tests/unit/web/composer/test_yaml_importer.py \
  tests/integration/web/composer/guided
git add src/elspeth/web/composer/guided src/elspeth/web/composer/recipes.py \
  src/elspeth/web/composer/tools/recipes.py src/elspeth/web/composer/tools/sessions.py \
  src/elspeth/web/composer/service.py src/elspeth/web/sessions tests/unit/web/composer \
  tests/integration/web/composer/guided
git commit -m "feat(web): enforce plugin policy in guided and import paths"
```

---

### Task 6: Enforce plugin availability and required-control coverage in validation

**Files:**

- Create: `src/elspeth/web/plugin_policy/coverage.py`
- Create: `src/elspeth/web/plugin_policy/validation.py`
- Modify: `src/elspeth/web/interpretation_state.py`
- Modify: `src/elspeth/web/execution/schemas.py`
- Modify: `src/elspeth/web/execution/validation.py`
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `src/elspeth/web/schema_probe.py`
- Modify: Plan-04-owned AWS ECS doctor initialization/compatibility module at its integrated path
- Create: `tests/unit/web/plugin_policy/test_coverage.py`
- Modify: `tests/unit/web/execution/test_validation.py`
- Modify: `tests/unit/web/execution/test_validation_value_source.py`
- Modify: `tests/unit/web/test_interpretation_state.py`

- [ ] **Step 1: Write failing ordered-validation and coverage tests**

```python
def test_disabled_plugin_fails_before_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = monkeypatch.spy(DatabaseSink, "__init__")
    result = validate_pipeline(_state_with_database_sink(), _settings(), yaml_generator, plugin_snapshot=_core_snapshot())

    assert result.is_valid is False
    assert result.checks[0].name == "plugin_enablement"
    assert result.errors[0].error_code == "plugin_not_enabled"
    constructor.assert_not_called()
```

Required prompt-shield tests cover every untrusted/user/external-content path
into each LLM. Required content-safety tests cover every LLM output path to a
sink. Include fan-in, fan-out, queues, cycles, alternate routes, shield after
LLM, content safety before LLM, detect-only metadata, and multiple controls.
Pin the closed trust-source rules: user text, source rows, and every
external-call result are untrusted; only code-declared internal constants are
trusted; missing/unknown declarations are untrusted. Pin role mismatch so an
OUTPUT control can never satisfy INPUT coverage or vice versa.

- [ ] **Step 2: Extract queue-aware typed graph coverage**

Move reusable stream graph/dominance logic from `interpretation_state.py` to
`plugin_policy/coverage.py`. Credit a node only when its typed declaration and
effective config prove blocking semantics and the declaration's role matches
the edge being protected. Prompt shield must dominate every
relevant LLM input path; content safety must post-dominate every LLM-to-sink
output path. Unknown/fan-in/cycle conditions fail safe.

- [ ] **Step 3: Add stable validation checks before YAML/plugin construction**

`validate_pipeline(..., plugin_snapshot: PluginAvailabilitySnapshot,
profile_registry: OperatorProfileRegistry)` runs:

1. state existence;
2. `plugin_enablement`;
3. operator-profile public-option validation/lowering;
4. `required_control_availability` and graph coverage;
5. existing path/secret/YAML/runtime checks.

Return component-attributed `plugin_not_enabled`, `plugin_unavailable`,
`required_control_unavailable`, or `required_control_coverage` blockers.

- [ ] **Step 4: Thread one snapshot through validate endpoints and composer preflight**

`ExecutionService.validate_state` and `ComposerServiceImpl._runtime_preflight`
receive/build one current-principal snapshot and pass the same object to the
validator. No web call site may omit the argument.

- [ ] **Step 5: Run and commit validation enforcement**

```bash
uv run pytest -q \
  tests/unit/web/plugin_policy/test_coverage.py \
  tests/unit/web/execution/test_validation.py \
  tests/unit/web/execution/test_validation_value_source.py \
  tests/unit/web/test_interpretation_state.py
git add src/elspeth/web/plugin_policy src/elspeth/web/interpretation_state.py \
  src/elspeth/web/execution tests/unit/web/plugin_policy tests/unit/web/execution \
  tests/unit/web/test_interpretation_state.py
git commit -m "feat(web): validate plugin policy and control coverage"
```

---

### Task 7: Add execution, runtime-construction, and delayed-export backstops

**Files:**

- Modify: `src/elspeth/web/execution/preflight.py`
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `src/elspeth/web/execution/protocol.py`
- Modify: `tests/unit/web/execution/test_preflight_side_effects.py`
- Modify: `tests/unit/web/execution/test_service.py`
- Modify: `tests/unit/web/test_schema_probe.py`
- Modify: `tests/testcontainer/web/test_schema_probe_postgres.py`
- Modify: `docs/reference/configuration.md`
- Modify: `tests/integration/web/test_execute_pipeline.py`
- Modify: `tests/unit/test_cli_helpers_sink_factory.py`
- Modify: `tests/unit/test_cli_orchestrator_teardown.py`

- [ ] **Step 1: Write execution-bypass and CLI-separation tests**

Prove execution of an old now-disabled state fails before web run creation and
before any plugin constructor. Prove delayed export cannot instantiate a sink
outside the frozen run snapshot. Prove CLI `instantiate_plugins_from_config`
and `make_sink_factory` still accept every installed plugin without
`WebSettings` or a web snapshot.

- [ ] **Step 2: Require a snapshot at the web preflight boundary**

```python
def instantiate_runtime_plugins(
    settings: ElspethSettings,
    *,
    plugin_snapshot: PluginAvailabilitySnapshot,
) -> PluginBundle:
    require_settings_plugins_available(settings, plugin_snapshot)
    return instantiate_plugins_from_config(settings, preflight_mode=True)
```

Apply the same guard to `build_validated_runtime_graph`. Keep
`plugins/infrastructure/runtime_factory.py` web-policy-free.

- [ ] **Step 3: Freeze a fresh run snapshot before run creation**

After loading the final persisted state and current principal, build one fresh
snapshot, re-resolve the selected profile aliases/scopes, validate/lower the
state, and only then call session `create_run` or submit background execution.
Pass the frozen snapshot, in-memory executable settings, and distinct
audit-safe authored settings to `_run_pipeline`. `RunLifecycleRepository.begin_run`
and node registration receive only the audit-safe form; plugin constructors
receive only the executable form. Background arguments must never serialize
resolved secret values. Marker tests inspect Landscape `runs.config_json`, all
node configs, exporter output, logs, errors, and telemetry for zero private
binding occurrences.

- [ ] **Step 4: Bind delayed sink creation to the same approval**

Wrap `make_sink_factory(settings)` in a web closure that calls
`require_settings_sink_available` against the frozen run snapshot before each
construction. It may not recompute policy or select a different sink.

- [ ] **Step 5: Run and commit runtime enforcement**

```bash
uv run pytest -q \
  tests/unit/web/execution/test_preflight_side_effects.py \
  tests/unit/web/execution/test_service.py \
  tests/integration/web/test_execute_pipeline.py \
  tests/unit/test_cli_helpers_sink_factory.py \
  tests/unit/test_cli_orchestrator_teardown.py
git add src/elspeth/web/execution tests/unit/web/execution tests/integration/web/test_execute_pipeline.py \
  tests/unit/test_cli_helpers_sink_factory.py tests/unit/test_cli_orchestrator_teardown.py
git commit -m "feat(web): enforce plugin policy at runtime"
```

---

### Task 8: Isolate frontend catalog caches and render repairable disabled state

**Files:**

- Create: `src/elspeth/web/frontend/src/stores/pluginCatalogStore.ts`
- Create: `src/elspeth/web/frontend/src/stores/pluginCatalogStore.test.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/types/api.ts`
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx`
- Modify: `src/elspeth/web/frontend/src/hooks/useNarrativeMode.ts`
- Modify: `src/elspeth/web/frontend/src/stores/authStore.ts`
- Modify: `src/elspeth/web/frontend/src/components/sidebar/ImportYamlModal.tsx`
- Modify: corresponding `*.test.ts` and `*.test.tsx` files

- [ ] **Step 1: Write failing principal/fingerprint cache tests**

```typescript
it("never reuses a catalog across principal or snapshot fingerprints", async () => {
  const store = createPluginCatalogStore();
  await store.load({ principal: "local:alice", fingerprint: "a" });
  await store.load({ principal: "oidc:alice", fingerprint: "b" });

  expect(api.fetchPluginPolicy).toHaveBeenCalledTimes(2);
  expect(store.getState().key).toBe("oidc:alice:b");
});
```

Test logout, 401, secret mutation, fingerprint change, schema cache, disabled
saved component rendering, and keyboard/screen-reader repair labels.

- [ ] **Step 2: Replace module-global and component-lifetime caches**

The store key is principal namespace plus snapshot fingerprint. It owns list
and schema caches together. Auth change/logout/401 clears the store. A changed
fingerprint discards schemas before rendering. `useNarrativeMode` and
`CatalogDrawer` consume only this store. A successful user-secret
create/update/delete operation emits an explicit catalog-invalidation event;
the store clears, refetches `/api/catalog/policy`, and does not render stale
aliases while the refetch is pending.

- [ ] **Step 3: Render disabled historical components without private schema**

Use sanitized state policy findings to show identity, reason category, and
remove/replace actions. Do not call the hidden plugin schema endpoint.

- [ ] **Step 4: Run and commit frontend isolation**

```bash
cd src/elspeth/web/frontend
npm test -- --run \
  src/stores/pluginCatalogStore.test.ts \
  src/components/catalog/CatalogDrawer.test.tsx \
  src/hooks/useNarrativeMode.test.ts \
  src/stores/authStore.test.ts \
  src/components/sidebar/ImportYamlModal.test.tsx
cd ../../../..
git add src/elspeth/web/frontend
git commit -m "feat(web): isolate plugin catalog by policy snapshot"
```

---

### Task 9: Persist web plugin-policy evidence atomically in Landscape

**Files:**

- Create: `src/elspeth/contracts/plugin_policy_audit.py`
- Modify: `src/elspeth/core/landscape/schema.py`
- Modify: `src/elspeth/core/landscape/database.py`
- Modify: `src/elspeth/core/landscape/run_lifecycle_repository.py`
- Modify: `src/elspeth/core/landscape/exporter.py`
- Modify: `src/elspeth/engine/orchestrator/core.py`
- Modify: `src/elspeth/engine/orchestrator/run_lifecycle.py`
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `tests/unit/core/landscape/test_schema.py`
- Modify: `tests/unit/core/landscape/test_schema_epoch_and_required_columns.py`
- Modify: `tests/unit/core/landscape/test_database_compatibility_guards.py`
- Modify: `tests/unit/core/landscape/test_run_lifecycle_repository.py`
- Modify: `tests/unit/core/landscape/test_exporter.py`
- Modify: `tests/unit/engine/orchestrator/test_landscape_registration.py`
- Modify: `tests/unit/web/execution/test_service.py`

- [ ] **Step 1: Write failing schema/atomicity/audit-order tests**

Create tests for one optional evidence row per web run, no row for CLI runs,
FK/canonical JSON/hash validation, insert rollback with run/attribution/leader,
export inclusion, and audit persistence before telemetry.

- [ ] **Step 2: Define the frozen L0 evidence DTO**

```python
@dataclass(frozen=True, slots=True)
class WebPluginPolicyEvidence:
    schema_version: int
    policy_hash: str
    snapshot_hash: str
    authorized_plugin_ids: tuple[str, ...]
    available_plugin_ids: tuple[str, ...]
    control_modes: tuple[tuple[str, str], ...]
    selected_implementations: tuple[tuple[str, str | None], ...]
    selected_profile_aliases: tuple[tuple[str, str | None], ...]
    plugin_code_identities: tuple[tuple[str, str, str], ...]
    binding_generation_fingerprint: str
    decision_codes: tuple[str, ...]
```

Validate lowercase SHA-256 fields and canonical sorted sets. Selected aliases
are safe opaque names; private profile bindings, raw/low-entropy binding hashes,
principal secret names, and remote payloads are forbidden.

- [ ] **Step 3: Add `run_web_plugin_policy` as an optional one-to-one table**

Use `run_id` as PK/FK plus schema version, hashes, and canonical JSON fields.
Update structural schema verification. Bump `SQLITE_SCHEMA_EPOCH` from 22 to
23 and update its exact-value test. A populated epoch-22 database is stale:
validate-only startup/doctor must reject it with the existing pre-1.0
drop/recreate instruction rather than adding this table in place. Fresh schema
creation includes the table and stamps epoch 23. Do not add nullable/placeholder
policy columns to `runs`; CLI runs legitimately have no policy-evidence row.

Change `_sync_sqlite_schema_epoch` ordering deliberately: inspect whether the
database is populated and reject any populated epoch below 23 before
`metadata.create_all()` or an automatic stamp can relabel it. Only a genuinely
fresh empty database may be created and stamped at 23. Tests prove an epoch-22
database missing the new table remains epoch 22 and unchanged after refusal.

Apply the same one-way posture to PostgreSQL. The structural probe classifies
an existing non-empty Landscape missing `run_web_plugin_policy` as STALE, not a
repairable PARTIAL, and the schema-owner doctor refuses `--init-schema` until
the database operator completes the approved export/archive (where retention
applies), drop/recreate, and fresh owner initialization. A fresh empty
PostgreSQL database is initialized with the complete epoch-23 shape. The
DML-only runtime role never receives DDL. Add testcontainer cases for current,
fresh initialize, old populated refusal with zero catalog mutation, and runtime
DDL denial. There is no automated in-place migration or rollback.

- [ ] **Step 4: Insert evidence in the existing begin-run transaction**

Thread optional evidence through orchestrator initialization and
`RunLifecycleRepository.begin_run`. Insert it in the same transaction as run,
attribution, and leader coordination rows. Web execution must supply it; CLI
supplies `None`. Emit `RunStarted` telemetry only after the transaction commits.
Policy denial/readiness telemetry is likewise emitted only after its Landscape
audit event/row commits, uses bounded closed reason codes and counts, and omits
principal, alias, profile, secret, and private-binding detail. Tests inject an
audit failure and prove no success/denial telemetry escapes first.

- [ ] **Step 5: Export the evidence and run audit tests**

```bash
uv run pytest -q \
  tests/unit/core/landscape/test_schema.py \
  tests/unit/core/landscape/test_schema_epoch_and_required_columns.py \
  tests/unit/core/landscape/test_database_compatibility_guards.py \
  tests/unit/core/landscape/test_run_lifecycle_repository.py \
  tests/unit/core/landscape/test_exporter.py \
  tests/unit/engine/orchestrator/test_landscape_registration.py \
  tests/unit/web/execution/test_service.py
```

Expected: all tests pass, including CLI absence and web atomicity.

Run the PostgreSQL compatibility proof explicitly:

```bash
docker info
uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py -m testcontainer -q
```

Docker absence or any skip is BLOCKED for this schema-changing slice.

- [ ] **Step 6: Commit Landscape evidence**

```bash
git add src/elspeth/contracts/plugin_policy_audit.py src/elspeth/core/landscape \
  src/elspeth/engine/orchestrator src/elspeth/web/execution/service.py src/elspeth/web/schema_probe.py \
  tests/unit/core/landscape tests/unit/engine/orchestrator tests/unit/web/execution/test_service.py \
  tests/unit/web/test_schema_probe.py tests/testcontainer/web/test_schema_probe_postgres.py \
  docs/reference/configuration.md
git commit -m "feat(audit): record web plugin policy evidence"
```

Update the release/schema compatibility record and operator docs to name
Landscape epoch 23, the new table, destructive reset requirement, archive
approval, and the fact that rolling code back to epoch 22 after recreation is
unsafe. Plans 10 and 12 must bind candidate and rollback decisions to this
epoch-23 record.

---

### Task 10: Add policy/readiness rows and make tutorial LLM profiles executable

**Files:**

- Modify: `src/elspeth/web/audit_readiness/models.py`
- Modify: `src/elspeth/web/audit_readiness/service.py`
- Modify: `src/elspeth/web/audit_readiness/explain.py`
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/composer/recipes.py`
- Modify: `src/elspeth/web/composer/tutorial_service.py`
- Modify: `src/elspeth/web/composer/tutorial_run_routes.py`
- Modify: `src/elspeth/web/frontend/src/App.tsx`
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx`
- Modify: `tests/unit/web/audit_readiness/test_service.py`
- Modify: `tests/integration/web/test_tutorial_routes.py`
- Modify: `tests/unit/web/composer/test_tutorial_service.py`
- Modify: frontend tutorial/system-status tests

- [ ] **Step 1: Write failing readiness and tutorial profile tests**

Pin separate rows for policy compilation, required core, local capability
configuration, live health, tutorial profile, and tutorial required-control
coverage. A missing tutorial profile
makes tutorial readiness unhealthy but does not make the server fail startup or
hide CSV/JSON/text authoring.

- [ ] **Step 2: Lower the tutorial recipe through the configured LLM profile**

Replace raw tutorial recipe slots `provider`, `model`, and `api_key_secret`
with `profile: settings.tutorial_llm_profile`. The ordinary operator-profile
resolver lowers it at validation/execution. The target/composer sees only the
alias and safe role metadata.

- [ ] **Step 3: Check the full tutorial plugin set before live run**

Before `_run_live_tutorial`, verify the persisted state uses only currently
available kind-qualified core plugins (`source:csv|source:json`,
`transform:web_scrape`, `transform:llm`, `transform:field_mapper`, and
`sink:json`) and the selected LLM profile is credential-ready for the principal.
Run the same `required_control_coverage` validator against the fully lowered
tutorial candidate during readiness and again immediately before launch. If a
required prompt/content control is absent, readiness is unhealthy and launch
returns the typed 409 blocker. Do not auto-insert a control. An optional future
operator-authored recipe variant may name the chosen control explicitly and
pass through normal audited state mutation.

- [ ] **Step 4: Separate local availability from remote health**

Enabled/configured external plugins are available for authoring from local
facts. Owned doctor/readiness probes report health separately and may make a
readiness row unhealthy; they do not rewrite policy or snapshot authorization.

- [ ] **Step 5: Run and commit readiness/tutorial integration**

```bash
uv run pytest -q \
  tests/unit/web/audit_readiness/test_service.py \
  tests/integration/web/test_tutorial_routes.py \
  tests/unit/web/composer/test_tutorial_service.py
cd src/elspeth/web/frontend && npm test -- --run && cd ../../../..
git add src/elspeth/web/audit_readiness src/elspeth/web/app.py \
  src/elspeth/web/composer src/elspeth/web/frontend tests/unit/web tests/integration/web/test_tutorial_routes.py
git commit -m "feat(web): expose plugin policy and tutorial readiness"
```

---

### Task 11: Seal parity, bypass, isolation, and repository-wide acceptance

**Files:**

- Create: `tests/integration/web/test_plugin_policy_end_to_end.py`
- Create: `tests/property/web/test_plugin_policy_properties.py`
- Modify: `docs/reference/configuration.md`
- Modify: `docs/superpowers/plans/aws/2026-07-08-aws-ecs-10-packaging-docker.md`
- Modify: `docs/superpowers/plans/aws/2026-07-08-aws-ecs-00-overview.md`

- [ ] **Step 1: Add one end-to-end surface parity matrix**

For each core-only, Azure-only, AWS-ready-fixture-only, both-with-preference,
and neither configuration, assert identical identities/selections across:

```text
catalog API == UI catalog == guided discovery == freeform prompt
== schema/assistance tools == recipe list == validation == execution evidence
```

The AWS fixture uses the generic 15B profile seam without registering the real
Plan 15C plugin. Include direct-tool, guided-submit, recipe-fast-path,
YAML-import, old-state, validation, runtime, and delayed-export attempts.

- [ ] **Step 2: Add property tests for canonical policy stability**

Generate installed/allowed sets and complete preference orders. Prove input
set order does not change policy hash, preference order does, unavailable
principals never gain authority, and every snapshot-selected ID belongs to
both authorized and available sets.

- [ ] **Step 3: Add offensive architecture guards**

AST/import tests fail if:

- web execution calls the full manager without the policy preflight;
- a web `ToolContext` is created without snapshot/view;
- `_shield_availability.py` or a provider-specific enable flag returns;
- core/CLI modules import `elspeth.web.plugin_policy`;
- target prompt code reads `os.environ` or profile private fields; or
- web policy absence defaults to allow-all.

- [ ] **Step 4: Document exact operator configuration and restart behavior**

Document core membership, kind-qualified optional allowlist, preference
completeness, recommend/required coverage, LLM profiles, tutorial readiness,
JSON environment syntax, startup failure modes, saved-state remediation,
restart requirement, CLI separation, and sanitized audit/readiness surfaces.

- [ ] **Step 5: Run focused backend and frontend acceptance**

```bash
uv run pytest -q \
  tests/unit/web/plugin_policy \
  tests/unit/web/catalog \
  tests/unit/web/composer \
  tests/unit/web/execution \
  tests/unit/web/audit_readiness \
  tests/integration/web/test_catalog_discovery.py \
  tests/integration/web/test_execute_pipeline.py \
  tests/integration/web/test_plugin_policy_end_to_end.py \
  tests/integration/web/composer/guided \
  tests/property/web/test_plugin_policy_properties.py
cd src/elspeth/web/frontend
npm test -- --run
npx playwright test tests/e2e/tutorial.spec.ts
cd ../../../..
```

Expected: all selected tests pass; no policy test is skipped.

- [ ] **Step 6: Run repository gates**

```bash
uv lock --check
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run mypy src/ elspeth-lints/src/
uv run python scripts/check_contracts.py
PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check \
  --rules plugin_contract.component_type,plugin_contract.plugin_hashes \
  --root src/elspeth
uv run pytest -q
git diff --check
wardline scan . --fail-on ERROR
```

Expected: every command exits zero. Wardline findings are fixed at the policy
boundary; no baseline waiver is added for new external-input paths.

- [ ] **Step 7: Commit documentation and close the tracker slice**

```bash
git add tests/integration/web/test_plugin_policy_end_to_end.py \
  tests/property/web/test_plugin_policy_properties.py \
  docs/reference/configuration.md docs/superpowers/plans/aws/2026-07-08-aws-ecs-10-packaging-docker.md \
  docs/superpowers/plans/aws/2026-07-08-aws-ecs-00-overview.md
git commit -m "test(web): seal universal plugin policy"
```

Expected: the worker reports its commits and evidence without closing from the
feature worktree. The integration coordinator rebases/merges into
`release/0.7.1`, reruns the complete runtime/audit policy handoff, and closes
`elspeth-0674a06468` with the integrated `release/0.7.1@<sha>` anchor. Only then
may Plan 15C register Bedrock controls through the generic seams.
