# Universal Web Plugin Policy Design

Date: 2026-07-12
Status: approved design
Branch context: `release/0.7.1`

## Purpose

Give operators one universal, fail-closed way to decide which installed
ELSPETH plugins the web application may expose and execute. The policy applies
to every web authoring path, including the catalog UI, guided composition,
freeform composition, recipes, YAML import, validation, and execution. It does
not restrict the CLI or batch YAML surface, which remains the trained-operator
interface to every installed plugin.

This design replaces provider-specific visibility flags. Azure and AWS prompt
or content controls, cloud storage, databases, RAG, web scraping, and future
plugins all use the same authorization mechanism.

## Decisions

1. Plugin installation, web authorization, and request-time usability are
   different facts.
2. All installed plugins remain registered in the process-wide
   `PluginManager`; CLI and batch behavior is unchanged.
3. The web application defaults to a small code-owned core. Installing a new
   plugin never silently expands the web attack, data-egress, or cost surface.
4. Operators add optional web plugins through a kind-qualified allowlist.
5. One immutable request snapshot drives every web surface. Hiding a catalog
   row is not treated as an authorization control.
6. Security-control selection uses closed, typed capability metadata rather
   than plugin names or open-ended catalog tags.
7. Prompt and content controls are recommended by default, never silently
   inserted. An explicit operator policy may require them.
8. Policy changes require a server restart in v1.

## Scope

This design owns:

- the required web plugin core;
- a new line-oriented text sink;
- operator allowlist, preference, and control-mode settings;
- startup policy compilation and validation;
- request-scoped availability snapshots;
- policy-filtered web catalog and composer inventory;
- authoring, import, validation, and execution enforcement;
- target-LLM awareness of usable prompt and content controls;
- LLM tutorial availability and its web risk boundary;
- readiness, audit, error, and cache behavior; and
- regression and acceptance coverage.

The AWS Bedrock shield plan consumes this design. It must not add separate
`aws_bedrock_*_enabled` settings or expose Guardrail identifiers through web
pipeline YAML.

## Non-goals

- Restricting `elspeth run`, batch YAML, or ordinary CLI plugin discovery.
- Unregistering or conditionally importing plugins to implement web policy.
- XML source or sink plugins. XML is deliberately parked.
- Runtime policy hot reload.
- Claiming that a remote provider is healthy merely because a plugin is
  available for authoring.
- Silently repairing or replacing disabled plugins in saved pipelines.
- Automatically inserting recommended security controls without an explicit
  user instruction or required operator policy.

## Current Reality

Built-in plugins are discovered and registered in one shared
`PluginManager`. `CatalogServiceImpl` caches the complete registry. Web catalog
routes expose that complete catalog, while composer discovery applies only a
request-secret filter. Direct mutation tools, guided submissions, recipes,
YAML import, validation, and execution can still name any registered plugin.

`validate_pipeline` and the runtime factory validate registration and plugin
configuration, not deployment authorization. Therefore presentation-only
filtering is bypassable and cannot be extended safely with per-shield flags.

The current built-in inventory includes `csv`, `json`, and `text` sources, but
only `csv` and `json` sinks. There is no text sink and no XML plugin. The LLM
transform is needed by the tutorial and must be part of the required web core.

## Terminology

### Installed

A plugin class was discovered and registered in the full process registry.
Installation is a code/package fact and grants no web authority.

### Authorized

The plugin belongs to the code-owned required core or appears in the
operator's optional web allowlist.

### Available for authoring

The plugin is authorized, has locally valid server-owned configuration, and
has any request-scoped credential references it declares. This is a local
configuration fact, not a remote-health claim.

### Operationally healthy

An explicit readiness or doctor probe succeeded within its freshness window.
Health is reported separately and does not silently rewrite authorization.
The real plugin call remains authoritative and fails closed.

### Plugin ID

A canonical kind-qualified identifier:

```text
source:csv
transform:llm
sink:text
```

The grammar is `source|transform|sink`, followed by `:`, followed by the
registered lowercase plugin name. Kind qualification is mandatory because
source and sink names intentionally overlap.

## Required Web Core

The required set is a code-owned constant, not plugin self-declared metadata:

```text
source:csv
source:json
source:text
sink:csv
sink:json
sink:text
transform:field_mapper
transform:llm
transform:web_scrape
```

Core membership cannot be removed by configuration. Startup fails if any core
plugin is missing from the installed registry. A future change to core
membership is a reviewed product/code change with contract and migration
tests, not an operator override.

The canonical live tutorial is `csv|json source -> web_scrape -> llm ->
field_mapper -> json sink`, so all three transforms belong to the required
core. `field_mapper` is local. `web_scrape` remains constrained by its existing
SSRF, public-host, and wire-visible operator-identity policy. `transform:llm`
is available for authoring only when at least one operator-owned web LLM
profile and its credential requirements are present.

## Operator Configuration

`WebSettings` gains three universal policy settings plus the plugin-specific
LLM profile settings required to risk-manage the tutorial. Environment values
use JSON for collections and mappings and continue to reject unknown
`ELSPETH_WEB__*` names.

```yaml
plugin_allowlist:
  - transform:azure_prompt_shield
  - transform:aws_bedrock_content_safety

plugin_preferences:
  prompt_shield:
    - transform:azure_prompt_shield
  content_safety:
    - transform:aws_bedrock_content_safety

plugin_control_modes:
  prompt_shield: recommend
  content_safety: recommend

llm_profiles:
  tutorial-default:
    provider: openrouter
    model: openai/gpt-5-mini
    credential_scope: server
    credential_ref: OPENROUTER_API_KEY

tutorial_llm_profile: tutorial-default
```

### `plugin_allowlist`

- Type: ordered input collection, canonicalized to a set in the runtime
  policy.
- Default: empty.
- Meaning: optional additions only; the effective authorized set is required
  core union allowlist.
- Duplicate, malformed, unknown, wrong-kind, wrong-case, or uninstalled IDs
  fail startup.
- An optional plugin whose package dependency is absent may remain absent when
  not requested. Explicitly allowing it fails startup instead of silently
  degrading.

### `plugin_preferences`

- Type: mapping from typed capability to an ordered tuple of authorized
  plugin IDs.
- Every listed plugin must declare the matching typed capability.
- Entries must be unique and authorized.
- When more than one authorized plugin implements a capability, its preference
  list must include every authorized implementation exactly once. This makes
  selection deterministic and prevents an accidental lexical fallback.
- A preference never authorizes a plugin.

### `plugin_control_modes`

- Type: mapping from security-control capability to `recommend|required`.
- Applies only to closed security controls such as `prompt_shield` and
  `content_safety`, not arbitrary discovery tags.
- Default for both initial controls: `recommend`.
- `recommend` makes the composer explain and propose the preferred usable
  control without mutating topology automatically.
- `required` makes final validation and execution reject relevant unprotected
  paths.
- Setting a capability to `required` without authorizing at least one
  implementation is a contradictory startup configuration and fails boot.
  An authorized implementation whose request credential is absent remains a
  request-time `required_control_unavailable` blocker rather than a startup
  failure, because credential availability may legitimately differ by user.

The settings layer converts explicitly into frozen runtime policy structures.
Alignment tests cover every field and collection parser so an accepted
setting cannot be silently ignored.

### `llm_profiles` and `tutorial_llm_profile`

- LLM profiles are operator-owned, typed provider/model bindings. Web-authored
  YAML selects an opaque alias and safe row/prompt options; the server lowers
  the alias to the explicit provider config after authoring and before
  validation/execution.
- Profile models validate provider-specific model, endpoint, credential-source,
  retry, and cost restrictions. Raw provider credentials and arbitrary
  endpoints remain forbidden in web-authored YAML.
- Credential scope is explicit: `server` resolves only from the operator
  server store and `user` resolves only from the current principal's store.
  User-first fallback is never used for an operator-profile binding, so a user
  secret cannot shadow a server credential.
- `tutorial_llm_profile` must name exactly one configured LLM profile when the
  tutorial is expected to be ready. When absent, the server still starts and
  ordinary core pipelines work, but tutorial readiness is unhealthy.
- The target LLM may see the alias and safe capability metadata, never the
  resolved provider binding or credential reference name.

## Startup Policy Compilation

Startup performs this deterministic sequence:

1. Discover and register the full plugin registry.
2. Parse every installed class into a canonical `(kind, name)` identity.
3. Verify every required-core identity exists exactly once.
4. Parse and validate the optional allowlist against the installed registry.
5. Validate typed capability declarations and preference completeness.
6. Validate all operator-owned plugin profiles and plugin-specific web
   restrictions without making user-data or paid provider calls.
7. Build a frozen `WebPluginPolicy` containing schema version, required IDs,
   configured optional IDs, effective authorized IDs, preferences, control
   modes, and plugin code identities.
8. Compute a stable, order-independent policy hash.
9. Store the policy on application state and inject it into all web services.

The full registry remains the authority for CLI diagnostics. The web policy
is a separate authorization view and must never mutate the shared manager.

## Configuration Authority Classes

Authorization to use a plugin does not imply authority to set every plugin
option. Each web-reachable plugin follows one of three configuration modes:

- `user_configurable`: ordinary data plugins such as CSV, JSON, and text,
  subject to existing path and schema policies.
- `user_configurable_with_policy`: plugins whose user-authored options remain
  visible but are narrowed by a plugin-specific web policy.
- `operator_profiled`: privileged plugins such as Bedrock Guardrails. Web YAML
  may select only an opaque profile alias and safe row-level options. The
  server resolves private identifiers, versions, regions, and other bindings
  after authoring and before validation/execution.

`transform:llm` is operator-profiled on the web so its provider/model/cost
authority is as explicit as Guardrail authority. CLI/batch retains the
provider-discriminated explicit configuration.

Raw operator bindings are never accepted from web-authored YAML. User secrets
cannot shadow operator-profile values. CLI/batch YAML may continue using the
plugin's explicit expert configuration contract.

Lowering produces two distinct values: an in-memory executable configuration
and an audit-safe authored configuration. Only the executable copy may contain
private provider/model/region/credential references. Landscape run/node
configuration, exports, saved state, logs, errors, telemetry, and background
job arguments receive only aliases, safe authored options, and non-reversible
server-keyed generation fingerprints. Resolved secret values are never cached
in either policy or snapshot.

## Typed Capability Contract

The current `capability_tags` field remains an open discovery vocabulary. It
must not drive security decisions.

Add a closed typed capability declaration for policy-sensitive behavior. The
initial vocabulary includes:

```text
llm
prompt_shield
content_safety
```

Capability metadata also declares the enforcement semantics needed to credit
a control, including whether it blocks or only detects and which input/output
role it covers. Validation credits a security control from typed metadata and
effective configuration, never from a plugin-name match.

Azure prompt shield and content safety gain the corresponding declarations.
The AWS Guardrail plugins must declare the same provider-neutral capabilities
while retaining documented differences in policy coverage, including the AWS
self-harm residual.

## Request-scoped Availability Snapshot

For each authenticated catalog request, guided request, composer turn, or
execution preflight, the server builds exactly one frozen
`PluginAvailabilitySnapshot` from:

- the process-lifetime `WebPluginPolicy`;
- the installed registry identities;
- locally valid per-alias operator-profile configuration;
- each alias's explicit server/user credential scope plus the current
  principal's scoped secret-reference inventory; and
- non-secret deployment settings required by the plugin.

The snapshot records authorized, available-for-authoring, unavailable, and
usable profile-alias identities with sanitized reason codes. An
operator-profiled plugin is available only when at least one alias is usable,
and its public schema exposes only usable aliases. Missing policy or principal
context is fail-closed for any plugin that requires it. Per-user secrets may
narrow the authorized set but never expand it; user-store values never satisfy
or shadow a server-scoped alias.

Live remote probes do not participate in `available_for_authoring`. Doctor and
readiness expose operational health separately. This prevents a stale probe
cache from becoming an authorization oracle while avoiding the false claim
that configured means healthy.

The snapshot has a stable fingerprint over code identities, safe alias state,
selection, and a server-keyed binding-generation fingerprint, never raw or
low-entropy private bindings. All surfaces in one request or composer
turn receive the same object/fingerprint; no surface independently recomputes
availability midway through a turn.

## Catalog And Target-LLM Inventory

`CatalogServiceImpl` remains the complete internal catalog. A request-scoped
policy catalog view wraps it:

- `list_sources`, `list_transforms`, and `list_sinks` return only available
  summaries;
- `get_schema` and assistance/hint calls reject unavailable identities;
- direct schema URLs cannot enumerate disabled plugins; and
- errors list only identities visible through the same view.

The UI catalog, guided pickers, guided prompts, freeform context, schema tools,
recipe discovery, and repair suggestions all consume this view. Hard-coded
plugin lists and hard-coded repair recommendations are removed or checked
against the snapshot.

Every principal-dependent catalog list, schema, assistance, and policy HTTP
response is `Cache-Control: private, no-store` and varies on the authentication
surfaces in use. Frontend caches are keyed by principal namespace plus snapshot
fingerprint and are invalidated after user-secret mutation; browser/proxy cache
behavior is not trusted as the only isolation control.

The target LLM receives only:

- available plugin identities and safe catalog metadata;
- available implementations grouped by typed capability;
- the preferred available implementation for each capability; and
- the applicable `recommend|required` mode.

It never receives resolved secrets, environment-variable names, Guardrail
identifiers/versions, AWS account/role/region bindings, or credential failure
details.

Selection is deterministic:

1. No usable implementation: report the capability unavailable and do not
   invent configuration.
2. One usable implementation: select it.
3. More than one: select the first usable entry in the operator's complete
   preference order.

Selection does not itself mutate the pipeline. In recommend mode the composer
proposes the control. In required mode the authoring/validation policy requires
coverage, while topology changes still pass through ordinary audited mutation
surfaces.

## Enforcement Boundaries

Visibility is a usability feature. Authorization is enforced independently at
every state-changing and execution boundary.

### Discovery

Catalog HTTP, UI caches, composer discovery tools, guided emitters, schema
tools, assistance, recipes, and preview repairs use the policy catalog view.
A recipe is visible only if every plugin it requires is available.

### Incremental authoring

`ToolContext` carries the immutable snapshot. Canonical source, transform,
aggregation, and sink mutation validators reject a newly selected unavailable
plugin before changing state. Direct tool invocation cannot bypass discovery.

Historical states containing unavailable plugins remain editable for
remediation. Deleting or replacing an unavailable component is allowed;
adding a new unavailable component is not. A bulk replacement is accepted
only when every referenced plugin is available. Required-control coverage is
not enforced on each incremental edit because that would prevent construction;
it is enforced at explicit validation, guided completion, and execution.

### Guided authoring

Guided discovery, forms, HTTP submissions, deterministic fast paths, and final
commits use the same snapshot. Guided code must not create bare `ToolContext`
instances or embed complete-registry plugin names in prompts.

### Recipes

Each recipe has a declared or mechanically derived required-plugin set. Recipe
listing filters on that set, and application checks the fully built candidate
state before persistence.

### YAML import

Web YAML import scans every source, transform/aggregation, and sink identity
after structural parse and before persistence. A disabled or unavailable
identity rejects the import atomically with a typed error; the prior state is
unchanged. CLI loading is unaffected.

### Validation

`validate_pipeline` receives the snapshot and runs a stable ordered
`plugin_enablement` check before plugin construction or external work. It
returns component-attributed blockers for unauthorized or unavailable
plugins. Required-control path coverage is evaluated from typed capability
metadata and graph dominance, not names.

In the initial closed policy, required prompt shielding must dominate every
untrusted/user/external-content path into an LLM node. Required content safety
must post-dominate every LLM output path before it reaches a sink. Queue,
fan-in, fan-out, alternate-route, and cycle ambiguity fails safe. Ordinary
incremental edits may temporarily lack coverage so users can construct or
repair a graph; explicit validation, guided completion, and execution enforce
the requirement.

### Execution

Execution loads the final persisted state, builds a fresh snapshot for the
current principal, freezes it for the run, and rechecks the full state before
creating a run record or constructing plugins. This closes the gap between an
earlier authoring snapshot and rotated/deleted configuration.

The web preflight/runtime-build boundary requires the frozen run snapshot and
rechecks settings before delegating to the CLI-neutral runtime factory.
Delayed sink creation and export factories are bound to the same approved
settings and snapshot. No web path resolves a class directly from the full
manager without a preceding policy check.

If a credential rotates after the run snapshot, resolution or the external
call fails closed. The snapshot is evidence of the configuration decision, not
a cache of secret values.

## Saved-state And Policy-change Behavior

Policy changes require restart and apply to new requests and runs. A saved
pipeline may therefore reference a plugin no longer authorized.

Such a pipeline:

- remains readable and exportable;
- displays the unavailable component identity with a repair status;
- does not receive the disabled plugin's private schema or bindings;
- may be edited to remove or replace the component; and
- cannot validate as execution-ready or run until repaired or reauthorized.

The system never silently deletes, replaces, or grandfather-authorizes the
component.

## LLM Tutorial Risk Boundary

`transform:llm` is required for the tutorial and belongs to the core. Web
authoring selects an operator-owned LLM profile alias and safe prompt/row
options. The server resolves the profile to its approved provider, model,
endpoint shape, credential source, and retry/cost budgets. Arbitrary base
URLs, ambient-role provider selection, and unbounded retry/capacity settings
remain rejected at the web boundary even though equivalent expert CLI
configuration may be accepted.

Tutorial readiness requires one preferred usable LLM provider/model
configuration. If none exists, the tutorial reports a precise readiness
blocker instead of inventing credentials or weakening provider policy.
CSV/JSON/text authoring remains usable when tutorial LLM readiness is absent.

## Text Sink

Add a built-in `sink:text` that is the line-oriented inverse of
`source:text` for a configured string field.

The sink contract includes:

- required `path` and `field` options;
- UTF-8 default with a closed ASCII-compatible encoding set (`utf-8`, `ascii`,
  `latin-1`, `cp1252`) so canonical LF, byte hashing, and rollback offsets are
  one contract;
- `write|append` modes and the shared collision/resume policy;
- exactly one configured string value written per row;
- canonical `\n` record separators;
- rejection of embedded CR/LF and non-string values rather than lossy
  coercion or escaping;
- preservation of empty strings as blank records;
- content hashing and normal artifact/audit lifecycle behavior;
- preflight mode with no filesystem mutation; and
- plugin assistance, schema, catalog, hash, golden, unit, property, resume,
  and round-trip tests.

`source:text` round-trip tests set `strip_whitespace=False` and
`skip_blank_lines=False` where exact blank/whitespace preservation is the
asserted behavior.

## Error Contract

Startup configuration failures are fatal and sanitized. They name the setting
and canonical plugin/capability identity but never echo raw environment values
or private profile bindings.

Web boundary errors use stable codes:

- `plugin_not_installed`: impossible startup/config identity or stale import;
- `plugin_not_enabled`: installed but outside the web authorization policy;
- `plugin_unavailable`: authorized but missing locally valid configuration or
  request credential requirements;
- `plugin_preference_invalid`: incomplete or contradictory capability order;
- `required_control_unavailable`: required capability has no usable
  implementation; and
- `required_control_coverage`: a relevant graph path lacks the required
  control.

Unknown plugin errors must not list full-registry names. Expected input/config
failures become typed validation results; system registry contradictions
remain loud Tier-1 failures.

## Readiness And Doctor

Readiness reports separate rows for:

- policy compilation;
- required-core presence;
- tutorial LLM configuration;
- each enabled external capability's local configuration; and
- any explicitly owned live probe.

An enabled plugin with invalid operator configuration fails startup. A
configured external provider whose live probe is failed or stale makes the
corresponding readiness row unhealthy, but does not rewrite the static policy.
Probe output is sanitized and bounded.

AWS doctor and deployment acceptance verify the effective web policy includes
the AWS plugins the deployment claims, plus the applicable profiles and
control preferences. They do not treat package installation alone as
availability.

## Audit And Telemetry

Landscape remains the authoritative audit store. A separate optional
one-row-per-web-run `run_web_plugin_policy` record is inserted atomically with
the run. CLI runs do not fabricate web policy evidence. Web run evidence
records:

- policy schema version and policy hash;
- run availability-snapshot hash;
- effective authorized and available plugin identities;
- control modes;
- selected capability implementations; and
- sanitized policy decision codes.

It never records secret values, secret-reference values, private profile
bindings, raw remote failure payloads, or user-shadowable environment names.
The effective pipeline configuration/hash remains the execution truth.

Operator telemetry may emit bounded counts and health states after the
corresponding audit decision is durable. Telemetry never substitutes for the
Landscape record.

## Cache And Isolation Rules

- The static policy is immutable for the process lifetime.
- One request/turn snapshot is shared by all backend consumers.
- Frontend catalog/schema caches are keyed by principal and policy/snapshot
  fingerprint or cleared on authentication/policy changes.
- Composer schema-loaded caches are intersected with the current snapshot.
- No request availability result is cached across principals.
- Local and OIDC principals with the same textual user ID remain isolated by
  the existing scoped-secret identity contract.
- No raw user secret may shadow an operator-profile binding.

## Verification And Acceptance

### Policy compilation

- Empty allowlist produces exactly the required core.
- Required core cannot be removed or replaced.
- Missing core, unknown ID, duplicate ID, wrong kind/case, uninstalled
  explicitly requested plugin, invalid capability metadata, and incomplete
  preference orders fail startup.
- Policy hashes are stable and input-order-independent.

### CLI separation

- CLI plugin listing, config loading, validation, and execution still accept
  every installed plugin independently of `WebSettings`.
- Web policy modules do not leak into core configuration or the CLI-neutral
  runtime factory.

### Web parity and bypasses

- Catalog list/schema routes, frontend catalog, guided pickers/prompts,
  freeform prompt inventory, discovery/schema/assistance tools, recipes, and
  repair suggestions agree on one snapshot.
- Direct individual mutation, bulk replacement, guided commit, deterministic
  fast path, YAML import, validation, preflight, execution, and delayed export
  reject unavailable plugins.
- Rejection happens before persistence, run creation, plugin construction, or
  external calls as appropriate.
- A saved disabled-plugin pipeline remains readable and can be repaired.

### Multi-user and time-of-check behavior

- Different users see only their own credential-dependent availability.
- Logout/login and policy fingerprints invalidate frontend caches.
- Secret deletion or configuration change after authoring blocks execution.
- Secret rotation after snapshot never reuses a cached resolved value.

### Capabilities and shielding

- Azure-only, AWS-only, both, and neither matrices produce matching catalog,
  target-LLM, validation, and execution decisions.
- Multiple implementations follow the complete operator preference order.
- Missing required capability and incomplete graph coverage fail closed.
- Recommend mode never auto-inserts a node.
- Prompt shield and content safety remain distinct typed capabilities.
- Bedrock Guardrail private bindings never appear in authored YAML, prompts,
  errors, audit payloads, or telemetry.

### Tutorial and text

- Tutorial readiness is healthy only with an approved usable LLM provider.
- LLM provider/model/endpoint/retry restrictions hold even though the plugin
  is core.
- Text sink covers collision, append/resume, encoding, blank records,
  embedded-newline rejection, strict type handling, content hash, preflight,
  and source/sink round trips.

### Repository gates

Implementation runs targeted unit/integration/property/frontend suites,
configuration-contract checks, mypy/ruff/formatting, plugin catalog/hash/golden
checks, full pytest, and the repository's signed-tier and Wardline gates.

## Plan Integration

Implementation planning should split the existing shield slice rather than
hide this prerequisite inside the AWS plugin work:

1. A universal web plugin-policy foundation lands first, including the text
   sink, required core, policy compilation, catalog view, request snapshots,
   enforcement, audit/readiness surfaces, and CLI separation.
2. The Bedrock Guardrail shield slice then adds AWS implementations,
   operator-profile bindings, typed control capabilities, and live AWS proof
   on top of that foundation.
3. Packaging/runbook and final integration plans consume both slices and prove
   the effective ECS policy, tutorial LLM readiness, target-LLM selection, and
   Landscape evidence.

The tracker dependency graph must encode this ordering. Prose sequencing is
not sufficient.

## Rejected Alternatives

### Per-plugin enable flags

Rejected because every plugin would add settings, validation, prompt wiring,
tests, and bypass risk. This is the piecemeal model the design replaces.

### Unregister disabled plugins

Rejected because the shared registry serves CLI and web, historical state
needs stable identities, and package installation is not web authorization.

### Catalog filtering only

Rejected because direct tools, guided submissions, imports, saved states, and
runtime construction bypass presentation.

### Named deployment profiles as the sole policy

Rejected because profiles obscure the exact effective permission set and
drift as plugins are added. Named profiles may later expand to an explicit
allowlist, but the canonical stored/runtime contract remains kind-qualified
IDs.

### Live provider health as authorization

Rejected because transient or stale probe state is not an authority source.
Authorization, local authoring availability, and operational health remain
separate.
