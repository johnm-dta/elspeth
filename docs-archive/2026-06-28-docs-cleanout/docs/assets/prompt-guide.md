# Deep Review and Redesign of the ELSPETH Pipeline Composer Prompt

## Provenance, Date, And Scope

- Source prompt reviewed: `src/elspeth/web/composer/skills/pipeline_composer.md`
- Review artifact remediated: 2026-05-20
- Scope: prompt-architecture review and redesign guidance for ELSPETH's Pipeline Composer skill. This document does not certify runtime behaviour beyond the repository source prompt and the live tool/schema contracts referenced by the tests in `tests/unit/web/composer/test_skill_drift.py`.
- Citation policy: inline references to the source prompt point to the repository path above. External prompt-engineering claims are backed by the bibliography at the end of this document. The previous model-transcript citation markers were removed because they are not resolvable outside the review session.

## Executive summary

I treated the uploaded **“Pipeline Composer Skill”** document as the original prompt under review. It defines an ELSPETH pipeline-composition assistant with strong audit, validation, and tool-use constraints. If that is not the intended prompt, the analysis below will not be reliable; the minimum materials needed for a definitive review are the exact raw prompt text, the target model/API, the live tool schemas or function definitions, representative successful outputs, and representative failure cases. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography]

The core weakness is architectural rather than local. The prompt is doing too many jobs at once: policy manual, workflow engine, schema reference, troubleshooting guide, recipe catalogue, and runtime notes. That makes instruction selection harder, especially when the model is under long-context pressure. OpenAI, Anthropic, and Google all recommend clear sectioning, explicit hierarchy, tight scope, and iterative evaluation; OpenAI also warns that long-context prompts can suffer “lost in the middle” failures and should be paired with evaluations at different context sizes. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography]

The most important concrete reliability risks are instruction collisions, stale or inconsistent tool references, excessive embedding of implementation details, and missing priority rules for exception cases. One especially serious inconsistency is secret handling: the prompt’s tool inventory names `list_secret_refs`, later guidance tells the model to use `list_user_secrets`, and the prompt alternates between `secret://...` syntax and `{secret_ref: ...}` syntax. Another recurrent risk is clarification behaviour: the prompt sometimes forbids asking technical questions, but elsewhere requires operator questions for wire-visible `web_scrape` headers or multi-path output shape. Those are both defensible rules, but they should be governed by one explicit clarification policy rather than scattered exceptions. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`]

My recommended redesign is to turn the monolith into a **hierarchical prompt** with a short non-negotiable core, an explicit “when to ask vs when to proceed” policy, one canonical secret/model/tool flow, one canonical build/repair loop, and one canonical final response format. The revised prompt below preserves the original prompt’s load-bearing behaviour—anti-fabrication, no silent shape downgrades, audit boundary discipline, preview-before-complete, and strict validation—but removes most stale implementation references and collapses duplicated guidance into a single decision policy. That redesign is highly likely to reduce ambiguity, and moderately likely to improve reliability before formal evals; exact gains still need to be measured with task-specific tests. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography]

**Confidence assessment:** it is **highly likely** that the diagnosis below captures the prompt’s main structural failure modes, because the evidence is explicit in the uploaded prompt and aligns with current primary-source guidance from OpenAI, Anthropic, and Google. It is **moderately likely** that the revised prompt will materially improve outcome quality without further changes, because vendor docs consistently stress that prompt quality must be validated empirically and that some failures are better solved by model choice, structured outputs, retrieval, or eval-driven iteration than by prompt text alone. [Bibliography]

## Source material and assumptions

The review assumes the uploaded file is the live prompt and that its intended task is: **compose or edit ELSPETH pipelines with available tools, keep every decision auditable, validate the pipeline, and only then present a user-facing summary**. That intent is explicit in the opening section, in the audit primacy rules, in the tool-category inventory, and in the requirement to finish only after `preview_pipeline` succeeds. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`]

This is not a review of ELSPETH itself as a product, nor a verification of every runtime claim in the prompt text. Some portions of the original prompt reference internal files, bug IDs, validator paths, and optional tools or deployment-specific behaviours. Those references may be useful in internal documentation, but they are brittle prompt payload because they can drift independently from the model-facing tool schema. Prompting guidance from the major model providers consistently favours clear instructions, explicit structure, and empirically-tested prompts over mixing operational docs and prompt logic into one giant instruction block. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography]

If the uploaded file is not the exact source prompt, the missing items to provide are straightforward: the exact prompt text; its placement as system, developer, or user content; the intended model family; any control parameters such as temperature or reasoning settings; the available tools/function schemas; and a small eval set of good and bad examples. Anthropic explicitly recommends defining success criteria, building ways to test them, and starting from a real first-draft prompt before optimization; OpenAI’s eval guidance likewise recommends defining the task, running tests, and iterating against measured results. [Bibliography]

## Diagnosis of the original prompt

The strongest failure mode is **scope overload**. The prompt merges at least six different classes of instruction: audit policy, user-response policy, runtime tool policy, pipeline-building workflow, schema/wiring semantics, and cookbook-style patterns. A model reading that prompt must continually classify which guidance is authoritative for the present turn. That burden gets worse in long contexts and increases the chance that a locally salient but globally secondary instruction will win attention. OpenAI’s guidance to separate identity, instructions, examples, and context, plus Google’s guidance to prioritise critical instructions and structure long contexts carefully, both point in the opposite direction from the original design. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography]

A second major failure mode is **exception handling without a clean priority policy**. The prompt forbids ordinary technical implementation questions, but also requires operator questions for multi-path output shape and for `web_scrape.http.abuse_contact` and `scraping_reason`. It also forbids prose confirmation for subjective terms while requiring a dedicated review tool when available. None of these rules is inherently wrong. The problem is that they are distributed across distant sections, so the model has to discover and reconcile them at runtime. This pattern invites both false positives, where the model asks too often, and false negatives, where it fails to ask when required. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`]

A third failure mode is **schema and tool drift embedded in the prompt itself**. The prompt’s canonical tool list names `list_secret_refs`, `validate_secret_ref`, and `wire_secret_ref`, but the later credential discipline refers to `list_user_secrets`. The prompt then instructs `api_key: "secret://OPENROUTER_API_KEY"` in one section and later states that the only auditable wired form is `{secret_ref: "OPENROUTER_API_KEY"}`. These are not stylistic differences; they are mutually incompatible operational instructions. A model presented with both is likely to behave inconsistently across turns depending on local attention. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`]

A fourth failure mode is **overfitting the prompt to internal implementation details**. The prompt references file names, validator internals, service functions, and historical bug IDs such as `web/composer/yaml_generator.py`, `web/composer/service.py`, `composer/validate`, and `elspeth-7197f92457`. Those may be useful to human maintainers, but they are poor prompt material unless the model can query them directly. If the underlying implementation changes, the prompt becomes stale. Even worse, the presence of low-level internal names can encourage the model to over-assert behaviour it cannot verify from live tools. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`]

A fifth failure mode is **repetition without consolidation**. The original prompt repeats structurally identical ideas in multiple places: do not fabricate, do not silently downgrade shape, preview before completion, preserve operator inputs, use recipes when possible, disclose failure/discard paths, and wire secrets correctly. Repetition is not always bad, but in prompts it should normally reinforce one canonical rule, not create multiple partial restatements with local variants. Current provider guidance tends to favour consistent delimiters, stable ordering, and aligned examples rather than duplicative rule clusters. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography]

A sixth failure mode is **prompt size as a reliability hazard**. OpenAI explicitly notes that long-context prompts with complex instructions can lose information in the middle and should be tested at different context lengths; the `Lost in the Middle` literature makes the same point. The original prompt is exceptionally dense and includes many far-separated dependencies, such as tool inventory at the top, secret wiring near the end, and operational exceptions in the middle. That is a classic layout for middle-context neglect. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography]

### Failure modes and missing constraints table

| Failure mode | Evidence in the original prompt | Why it fails in practice | Recommended fix |
|---|---|---|---|
| Mixed concerns in one prompt | Audit policy, subjective-term review, termination gate, schema vocabulary, workflow, recipes, common patterns, and secret wiring all live in the same instruction body. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | The model must retrieve the right rule family before acting, increasing attention and conflict costs. | Split into a short operational core plus external reference docs/tests. |
| Secret-handling inconsistency | Tool list says `list_secret_refs`; later text says `list_user_secrets`; one section uses `secret://...`, another says only `{secret_ref: ...}` is wired. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | Directly contradictory instructions cause unstable behaviour and validator drift. | Keep one canonical secret flow only. |
| Clarification policy is implicit, not centralised | “Do not ask technical implementation questions” sits far from mandatory questions for subjective terms, multi-path outputs, and wire-visible scrape headers. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | The model can under-ask or over-ask because exception priority is not front-loaded. | Add one explicit “when you may ask a question” section. |
| Internal code paths in prompt text | References to internal files, validators, service functions, and bug IDs. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | These details go stale, are not live-queryable, and encourage unsupported assertions. | Move implementation notes to docs; keep only user-observable rules in prompt. |
| Repeated non-negotiables with local variants | Preview gate, no-fabrication, no shape downgrade, preserve operator strings, and disclosure rules recur in several places. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | Repetition without a master rule produces interpretation drift, not reinforcement. | Consolidate into one core rules block and one final response block. |
| Long-context retrieval risk | Critical rules are far apart; OpenAI warns long contexts can lose information in the middle. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography] | A model may attend to locally recent text rather than globally correct policy. | Shorten the prompt and pair with evals at multiple context lengths. |
| Missing precedence between live schemas and static docs | The prompt says tool definitions are authoritative, but still embeds stale schema advice later. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | The model is not told how to resolve conflicts cleanly. | Explicit priority order: live schema > validator errors > prompt defaults. |
| Prompt-only formatting where structured outputs would be stronger | The original prompt relies heavily on prose instructions for conformity. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | Text-only constraints are weaker than schema-backed outputs for predictable formatting. | Use JSON Schema / function calling / structured outputs where available. [Bibliography] |

## Revised prompt suite

The revised suite below applies the same evidence-backed principles across all variants: keep the authority chain explicit, separate invariant rules from workflow, reduce duplicated instructions, use delimiters and labelled sections, prefer structured output mechanisms over prompt-only formatting when the runtime supports them, and keep model-specific details in implementation settings rather than burying them inside the prompt. [Bibliography]

### Revised prompt optimized for clarity, reliability, and robustness

```text
You are ELSPETH Pipeline Composer.

Your job is to compose or edit ELSPETH pipelines using the provided composer tools.
You must understand the user’s requested workflow, build the pipeline, validate it,
and only then explain what you built.

## Priority order

When guidance conflicts, follow this order:
1. Live tool schemas and current tool responses
2. The non-negotiable rules below
3. The user’s stated product requirements
4. Conservative defaults

If a static prompt instruction conflicts with a live tool schema or validator error,
trust the live schema/error and adapt. Do not invent undocumented behaviour.

## Non-negotiable rules

- Never fabricate facts about plugins, models, runtime behaviour, audit backend,
  prior state, or undocumented ELSPETH internals.
- Never silently simplify or downgrade the requested workflow shape.
- Never expose or invent audit-backend details.
- Never call generate_yaml. The final readiness gate is preview_pipeline with
  is_valid = true.
- Never end a build turn while the pipeline is invalid, unless:
  - a genuinely required input is missing, or
  - the exact requested shape cannot be built and you must return a named-gap refusal.
- If you use create_blob, bind that blob to the pipeline in the same turn.
  create_blob alone is not completion.
- Use only current tool names and schemas. Do not rely on remembered signatures.
- Preserve user-supplied strings verbatim unless the user explicitly authorises a
  recorded normalisation step.

## When you may ask the user a question

Ask only when one of these is true:

1. A required input is missing.
   Examples: no source file/URL, no destination, no classification categories.

2. A multi-path workflow is requested but the output shape is ambiguous.
   Ask whether the result should be:
   - one merged output,
   - separate outputs per branch,
   - or both.

3. A web_scrape step requires wire-visible values that the user did not provide.
   You must not invent:
   - http.abuse_contact
   - http.scraping_reason

4. A user term is subjective or underspecified and will change runtime behaviour.
   If request_interpretation_review is available, use it instead of a normal
   confirmation question.

5. No supported credential/model combination is available for an LLM step.

If none of the above is true, do not ask permission and do not ask technical
implementation questions. Choose conservative defaults and continue.

## Canonical tool-use rules

- Use get_tool_definitions if you need to confirm available composer tools.
- Use list_models before choosing any LLM model identifier.
- For secrets, use exactly this flow:
  1. list_secret_refs
  2. validate_secret_ref
  3. For new nodes, inline {secret_ref: NAME}
  4. For existing nodes, use wire_secret_ref
- Never use secret://... or ${VAR} as a substitute for a wired secret reference.
- Prefer apply_pipeline_recipe when a registered recipe fits the request.
- Prefer set_pipeline when rebuilding incomplete or placeholder-heavy state.
- Prefer patch/upsert tools for small edits to an already valid pipeline.
- Every csv/json/text source must include options.schema.
  Default to schema.mode = observed unless fixed/flexible is specifically needed.
- All node.input values and route targets are connection names, not node IDs.

## Exact-shape rule

If the user requested a specific structure, build that exact structure.
Do not commit a degraded pipeline and then describe it as done.

If you cannot build the exact requested shape, return a named-gap refusal:
- state the specific gap,
- state the closest safe alternative,
- stop without committing the downgraded build.

## Audit rule

If the user asks about audit logging, audit database, Landscape, or where audit
records go, call get_audit_info and paraphrase only the returned summary.
Do not invent backend type, location, path, or encryption details.
Do not create an “audit sink” to satisfy that request.

## Subjective-term rule

If a subjective user term will be operationalised in an LLM prompt, surface the
interpretation before finalising the pipeline.
If request_interpretation_review is available:
- stage a placeholder such as {{interpretation:term}}
- call request_interpretation_review for that term
- do not silently bake in your private definition

## Build algorithm

1. Identify the requested outcome, exact workflow shape, hard constraints, and any
   missing required input.

2. Decide whether to:
   - patch an existing valid pipeline,
   - rebuild incomplete state,
   - or use a recipe.

3. Discover only the minimum schemas/models/plugins needed to proceed.

4. Build or patch the pipeline.

5. Call preview_pipeline.

6. If preview_pipeline is invalid:
   - read the validator output,
   - call explain_validation_error if needed,
   - repair the pipeline,
   - preview again,
   - repeat until green or until a required-input / named-gap stop applies.

7. Before any final user-facing completion message, confirm that:
   - preview_pipeline succeeded,
   - is_valid = true,
   - no blocking warning remains unresolved.

## Final response format after a green preview

State, in plain English:
- what the pipeline does
- source, transforms, and outputs
- every on_success / on_failure / on_validation_failure / discard path
- any discard or data-loss behaviour
- any external sink failure file or fallback path
- every important decision you made on the user’s behalf
- any assumptions encoded in the pipeline

Keep the summary concrete, auditable, and specific to the built state.
```

This version keeps the original prompt’s essential safety and audit behaviour, but it collapses scattered exception logic into a single hierarchy and one build loop. That aligns much better with current guidance on explicit instruction order, clear message roles, structured delimiters, and task-specific prompting. [Bibliography]

### Concise alternate version

```text
You are ELSPETH Pipeline Composer. Build or edit ELSPETH pipelines with the
provided tools, validate them, and only then explain the result.

Rules:
- Never fabricate plugin/runtime/audit facts.
- Never silently simplify the requested workflow shape.
- Never call generate_yaml.
- Never finish a build turn unless preview_pipeline returns is_valid = true,
  unless a required input is missing or the exact requested shape cannot be built.
- Trust live tool schemas and validator errors over remembered prompt text.

Ask the user a question only if:
- a required input is missing,
- a multi-path output shape is ambiguous,
- web_scrape needs wire-visible headers the user did not provide,
- a subjective term must be operationalised,
- or no usable credential/model exists.

Tool rules:
- list_models before choosing an LLM model
- list_secret_refs -> validate_secret_ref -> {secret_ref: NAME} / wire_secret_ref
- never use secret://... or ${VAR}
- every csv/json/text source needs options.schema
- node.input and route targets are connection names, not node IDs
- prefer recipes when they fit
- prefer set_pipeline for rebuilds, patch/upsert for small edits

Audit:
- If asked about audit, call get_audit_info and paraphrase only its summary.
- Do not invent audit backend details.
- Do not create an audit sink.

Workflow:
1. Identify requested outcome and exact shape.
2. Discover only what is needed.
3. Build or patch.
4. preview_pipeline.
5. Repair and re-preview until green.

Final response after a green preview:
- what the pipeline does
- source / transforms / outputs
- all routing and discard behaviour
- failure paths
- decisions made on the user’s behalf
- remaining assumptions
```

This concise version is best when the runtime already provides rich tool schemas and the main goal is to reduce prompt length, cost, and middle-context loss. That is consistent with OpenAI’s advice to keep prompts simple and direct for reasoning models, and with Anthropic’s note that simpler prompts are often better for latency- and cost-sensitive applications. [Bibliography]

### Verbose alternate version

```text
You are ELSPETH Pipeline Composer, an assistant that composes or edits ELSPETH
pipelines through tool calls. Your job is to turn a user’s requested workflow
into a valid pipeline state and then explain the built result in a way that is
truthful, auditable, and concrete.

## Mission

For each request, determine:
- what the user wants the pipeline to do,
- what exact structural shape they asked for,
- what information is missing,
- what tool actions are required to build or edit the state,
- and whether the resulting pipeline is valid.

You are not a general explainer of ELSPETH internals. You are a pipeline composer.

## Authority and conflict resolution

Follow this order:
1. Live tool schemas and current validator responses
2. Rules in this prompt
3. The user’s explicit product requirements
4. Conservative defaults

If you encounter a conflict, do not improvise.
Prefer the live schema or validator response, then adapt.

## High-integrity behaviour

Never fabricate:
- plugin existence
- runtime behaviour
- audit backend details
- tool availability
- undocumented system internals
- prior state
- meaning that the user did not provide

If a capability is not confirmed by live tools or clearly documented state,
say you cannot confirm it rather than asserting it.

## No silent downgrades

If the user asked for a specific shape—such as fork/coalesce, parallel paths,
two outputs, or branch-specific routing—you must either:
- build that exact shape, or
- return a named-gap refusal without committing a degraded build

Do not build a simpler version and describe it as complete.

## Clarification policy

Do not ask permission to use tools.
Do not ask ordinary technical implementation questions when conservative defaults
are sufficient.

Ask the user only when one of these is true:
- a required input is missing
- a multi-path output shape is ambiguous
- web_scrape requires wire-visible values not supplied by the user
- a subjective term would change runtime behaviour
- no supported secret/model combination exists

If request_interpretation_review exists, use it for subjective terms instead of
normal prose confirmation.

## Audit boundary

If the user asks about audit logging, Landscape, audit DB, where audit records
go, or similar topics:
- call get_audit_info
- paraphrase only the returned summary
- do not invent backend type, location, DB path, or encryption details
- do not satisfy the request by creating an “audit sink”

## Canonical composer rules

- Use get_tool_definitions if needed to confirm current tool availability.
- Use list_models before selecting any LLM model identifier.
- For secrets, use:
  list_secret_refs -> validate_secret_ref -> {secret_ref: NAME} or wire_secret_ref
- Never use secret://... or ${VAR} as a substitute for a wired secret.
- Prefer apply_pipeline_recipe when the request matches a recipe.
- Prefer set_pipeline for rebuilds of incomplete or placeholder-heavy state.
- Prefer patch/upsert tools for narrow edits to an already valid pipeline.
- Every csv/json/text source must include options.schema.
- Default to schema.mode = observed unless there is a clear reason to use fixed
  or flexible.
- node.input and route targets are connection names, not node IDs.
- Preserve user-supplied strings verbatim unless the user explicitly authorises
  a recorded normalisation step.

## web_scrape special rules

Never invent or silently alter wire-visible values.
You must not invent:
- http.abuse_contact
- http.scraping_reason

If they are missing, ask for them.
Do not wire them as secrets.

## Build procedure

1. Read the request and identify:
   - desired outcome
   - exact pipeline shape
   - hard constraints
   - missing required inputs

2. Decide whether to:
   - patch existing valid state,
   - rebuild state atomically,
   - or use a recipe.

3. Discover only the minimum schemas, plugins, models, and secrets needed.

4. Build or patch the state.

5. Call preview_pipeline.

6. If invalid:
   - inspect the error,
   - call explain_validation_error if needed,
   - repair,
   - preview again,
   - repeat until valid or until a valid stop condition applies.

7. Do not write a completion message until preview_pipeline is green.

## Valid stop conditions

You may stop without a green preview only if:
- a required user input is genuinely missing, or
- the exact requested shape cannot be built and you return a named-gap refusal

In both cases, be explicit and concrete.
Do not say “I tried X and it failed” and stop there.

## Final response after success

After preview_pipeline returns is_valid = true, explain:
- what the pipeline does
- source, transforms, outputs
- every route and discard path
- any validation-failure behaviour
- any failure sink or external-write fallback
- decisions made on the user’s behalf
- assumptions still encoded in the pipeline

Your final response must describe the built state, not a plan.
```

This version is better for weaker reasoning models or self-hosted agents because it repeats the most important concepts only once, makes the stop conditions explicit, and removes stale implementation references while keeping core behaviour intact. That matches the provider guidance that clear instructions, labelled structure, aligned examples, and explicit output criteria improve reliability. [Bibliography]

### Step-by-step guardrails version for safety and reproducibility

```text
You are ELSPETH Pipeline Composer.

Follow this procedure exactly.

## Step A: establish the task
Identify:
- requested business outcome
- exact requested pipeline shape
- required inputs
- constraints that must not be changed

If a required input is missing, ask one narrow question and stop.

## Step B: resolve ambiguity
Ask a question only if:
- a required input is missing
- a multi-path output shape is ambiguous
- web_scrape needs wire-visible values not supplied by the user
- a subjective term would affect runtime behaviour
- no supported secret/model combination exists

Otherwise proceed without asking permission.

## Step C: trust live runtime over memory
Use live tool schemas and current validation results as the source of truth.
If prompt text and live schema disagree, follow the live schema.

## Step D: select tools and dependencies
- Use get_tool_definitions if needed
- Use list_models before choosing an LLM model
- Use list_secret_refs and validate_secret_ref before wiring credentials
- For new nodes, inline {secret_ref: NAME}
- For existing nodes, use wire_secret_ref
- Never use secret://... or ${VAR}

## Step E: choose build strategy
- If a recipe fits, prefer the recipe
- If existing state is incomplete or scaffold-like, rebuild with set_pipeline
- If existing state is valid and only small edits are needed, patch/upsert

## Step F: enforce composition invariants
- Every csv/json/text source must include options.schema
- Default to schema.mode = observed if no stronger requirement exists
- node.input and route targets are connection names, not node IDs
- Preserve user-supplied strings verbatim unless the user explicitly approves a
  recorded normalisation step
- Never invent web_scrape.http.abuse_contact or scraping_reason

## Step G: protect audit integrity
- Never fabricate runtime or audit facts
- Never create an audit sink to answer audit questions
- If asked about audit, call get_audit_info and paraphrase only its summary
- Never silently downgrade the requested workflow shape

## Step H: build
Construct or patch the pipeline.

If you used create_blob, bind it to the pipeline in the same turn.

## Step I: validate
Call preview_pipeline.

If preview_pipeline is invalid:
1. read the error
2. call explain_validation_error if needed
3. repair
4. preview again

Repeat until:
- preview_pipeline is valid, or
- you hit a valid stop condition

## Step J: valid stop conditions
The only acceptable non-success stops are:
- genuinely missing required user input
- named-gap refusal because the exact requested shape cannot be built

Do not stop with an error summary alone.

## Step K: final response after success
Only after preview_pipeline is_valid = true, report:
- what the pipeline does
- source / transforms / outputs
- all route behaviour
- discard or validation-failure behaviour
- external sink fallback behaviour
- decisions made on the user’s behalf
- assumptions encoded in the final state

Your response must describe the actual built pipeline, not a plan.
```

This version is the safest for reproducibility because it makes the build loop procedural, uses explicit stop conditions, and keeps the prompt compact enough to reduce attention drift. Reproducibility is further strengthened when paired with fixed model settings, snapshot pinning where supported, and a stable eval set. [Bibliography]

## Abridged side-by-side comparison of the original and revised prompts

Because the original prompt is a large skill document rather than a short single-purpose prompt, the most useful side-by-side comparison is by **operative section**, not by reproducing the full raw text line-for-line. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`]

| Prompt area | Original prompt | Revised prompt | Expected effect |
|---|---|---|---|
| Overall architecture | Monolithic instruction set mixing policy, cookbook, schema reference, validator lore, and runtime notes. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | Short operational core with explicit hierarchy, workflow, and final-output contract. | Lower retrieval burden; fewer attention collisions. |
| Clarification policy | Exceptions are scattered across subjective terms, multi-path shapes, and `web_scrape` headers. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | One central “ask only when …” policy. | More consistent questioning behaviour. |
| Secret wiring | Contradictory use of `list_user_secrets` vs `list_secret_refs`, and `secret://...` vs `{secret_ref: ...}`. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | One canonical flow only: `list_secret_refs -> validate_secret_ref -> {secret_ref}` / `wire_secret_ref`. | Removes schema drift and validator ambiguity. |
| Internal implementation notes | Includes internal file paths, validator internals, and bug IDs. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | Removes most non-queryable implementation details from the prompt. | Less brittleness when backend code changes. |
| Validation loop | Strong but verbose termination gate with repeated edge-case instructions. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | One build/preview/repair loop with explicit success and stop conditions. | Keeps the same safety property with less prompt mass. |
| Audit behaviour | Strong anti-fabrication and “do not invent audit backend” rules. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | Preserved almost unchanged, but consolidated into a single audit boundary rule. | Maintains safety while improving readability. |
| Final user summary | Disclosure rules appear later in the document and are fairly detailed. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | Final response format is near the end of the operative prompt and explicitly required after green preview. | Better alignment between build success and user-facing explanation. |
| Long-context risk | Critical rules are far apart across the prompt body. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] | Key rules are front-loaded and grouped by decision point. | Reduced “lost in the middle” exposure. [Bibliography] |

## Evaluation tests, metrics, and validation plan

The evaluation framework should be **task-specific, automated where possible, calibrated with human review, and run continuously as the prompt evolves**. OpenAI recommends eval-driven development, logging everything, and combining metrics with human judgment; Anthropic similarly recommends specific, measurable success criteria, task-specific evals, and code-based grading where possible. [Bibliography]

The most useful prompt metrics for this skill are not generic BLEU-style text metrics. They are workflow metrics: **green-preview completion rate**, **silent-downgrade rate**, **required-question recall**, **unnecessary-question rate**, **schema/secret syntax correctness**, **repair-loop convergence**, and **final-summary completeness**. For production, I would also track **token cost**, **latency**, and **regressions across model/prompt version changes**. That approach aligns with current guidance to define concrete objectives, structure tests so they can be scored, and maintain continuous evaluation over time. [Bibliography]

### Test cases with expected outputs

| Test case | Input or setup | Failure targeted | Expected output or behaviour |
|---|---|---|---|
| Basic valid build | “Classify support tickets into Hardware, Software, or Other and save as JSONL.” | Over-asking / under-building | Builds without unnecessary questions, selects a valid model only after `list_models`, previews green, and summarises routes. |
| Missing source | “Classify these tickets” with no file, no pasted data, and no blob | Missing required input | Asks one narrow source question and stops; does not pretend a build exists. |
| Subjective term | “Rate how cool these sites are.” | Silent interpretation | Uses subjective-term flow; if review tool exists, stages placeholder and calls it rather than silently defining “cool.” |
| Multi-path ambiguity | “Process each row two ways and save the results.” | Hidden output-shape assumption | Asks whether output should be merged, separate, or both before building. |
| Audit request | “Add SQLite audit logging to the pipeline.” | Fabricated audit sink | Does not add a sink; calls `get_audit_info` and explains that audit is operator-managed. |
| Secret wiring drift | Runtime exposes `list_secret_refs`; model sees no `list_user_secrets` tool | Stale tool reference | Uses only `list_secret_refs` flow; does not emit `list_user_secrets` or `secret://...`. |
| Exact-shape refusal | User requests a fork/coalesce shape that the runtime cannot support | Silent downgrade | Returns a named-gap refusal and does not commit a simplified pipeline. |
| `web_scrape` wire-visible headers absent | Request includes web scraping but no `abuse_contact` or `scraping_reason` | Invalid defaulting | Asks for the missing wire-visible values; does not invent or secret-wire them. |
| Invalid preview repair | First build previews red with a missing schema field | Early surrender | Repairs and re-previews; does not stop after merely reporting the error. |
| Blob-only stall | Prompt path creates a blob but does not attach it | Non-terminal blob misuse | Immediately binds the blob in the same turn; never ends on `create_blob` alone. |
| Connection-name confusion | Upstream node ID differs from connection name | Wiring semantics error | Uses connection names, not node IDs, and converges to a green preview. |
| Adversarial unsupported plugin | User asks for a plugin name not confirmed by tools | Hallucinated capability | Says it cannot confirm that plugin rather than inventing support. |
| Long prompt stress | Prompt plus context is artificially padded with unrelated low-priority detail | Lost-in-the-middle regression | Still obeys the highest-priority rules; if not, the prompt fails the eval and should be shortened further. |
| Summary completeness | Build includes discard routes and an external sink failsink | Disclosure omissions | Final summary explicitly states discard/data-loss behaviour and fallback file path. |

A practical validation plan is to run these in four layers. Start with **unit-style prompt tests** that score single rules such as secret syntax, audit handling, and clarification behaviour. Then run **scenario tests** that represent whole user jobs. Add **adversarial tests** that deliberately try to trigger fabrication, over-asking, or silent simplification. Finally, run **regression tests** on every prompt edit and every model upgrade. OpenAI’s eval docs recommend exactly this kind of iteration-from-baseline approach, and Anthropic’s guidance is similarly explicit that success criteria and edge cases should be tested, not inferred from “vibes.” [Bibliography]

For scoring, use a mix of deterministic and rubric-based graders. Deterministic checks should cover whether the prompt selected the right tool family, used the right secret syntax, asked a question only when allowed, and reached `preview_pipeline is_valid = true` when the inputs were sufficient. Rubric-based grading is useful for the final summary: whether it mentions all routes, all discard/failure behaviour, and all decisions made on the user’s behalf. Anthropic recommends code-based grading first where possible and careful rubrics for LLM-based grading; OpenAI similarly recommends automation wherever possible and maintaining agreement with human review. [Bibliography]

## Prompt-engineering patterns and implementation notes for common LLMs

Three implementation patterns matter most here. First, use the highest-authority prompt channel available—developer or system instructions for role and invariant rules—rather than burying invariants inside ordinary user content. Second, use clear labelled sections or XML/Markdown delimiters so the model can distinguish identity, instructions, examples, and context. Third, when the runtime supports schema-backed outputs or function calling, use those instead of relying on text-only formatting instructions. Those recommendations are consistent across OpenAI, Anthropic, and Google documentation. [Bibliography]

OpenAI’s current guidance also matters for model choice. For GPT-style reasoning models, keep prompts **simple and direct**, avoid explicit “think step by step” instructions unless you have evidence they help, and use delimiters to mark distinct sections. For reproducibility, pin model snapshots, fix the `seed` where supported, keep all other parameters constant, and evaluate when changing prompt or model versions. OpenAI’s docs further recommend changing `temperature` or `top_p`, but not both at once, and note that lower temperature is more deterministic. [Bibliography]

Anthropic’s guidance is similar in spirit but slightly different in emphasis. Claude’s prompt docs recommend XML tags for structuring complex prompts, explicit tool directions for action-taking, detailed output formatting, examples for consistency, and prompt chaining for complex tasks. They also recommend using structured outputs rather than prompt-only techniques when strict JSON conformance matters, and the latest Claude docs emphasise the `effort` setting rather than over-prescribing every reasoning step in the prompt. [Bibliography]

Google’s Gemini guidance is the clearest on long-context arrangement: place critical instructions in the system instruction or very beginning, but when supplying large context blocks, provide the context first and put the specific question at the end. Gemini’s docs also recommend few-shot examples, while warning that too many examples can overfit, and structured outputs via JSON Schema when the final response format must be machine-readable. [Bibliography]

For self-hosted open-source models, keep the prompt itself simpler and move reliability into **decoding controls and application-side validation**. Hugging Face’s `GenerationConfig` documents `max_new_tokens` as the preferred way to cap output length, while vLLM documents `temperature`, `top_p`, `seed`, `stop`, and `max_tokens`. For this kind of deterministic tool-using task, a low-randomness setup is usually best: low temperature, fixed seed, bounded output tokens, and application-side validation of the final tool actions and summary structure. [Bibliography]

### Recommended implementation settings by model family

| Model family | Role framing | Decoding and control | Output control | Notes |
|---|---|---|---|---|
| GPT-style OpenAI models | Put the revised prompt in a developer message or `instructions`; keep user requests separate. [Bibliography] | Use low temperature for deterministic workflow tasks; change `temperature` or `top_p`, not both; pin snapshots and use `seed` where available. [Bibliography] | Prefer Structured Outputs or function calling for machine-readable actions. [Bibliography] | For reasoning models, keep prompts direct and avoid unnecessary explicit CoT requests. [Bibliography] |
| Claude | Put invariants in the system prompt; use XML tags and clear sections. [Bibliography] | Prefer the model’s effort/thinking controls over giant hand-written reasoning scripts. [Bibliography] | Use Structured Outputs when strict JSON is required. [Bibliography] | Explicit tool-use instructions materially improve action-taking. [Bibliography] |
| Gemini | Put critical constraints in system instructions or at the start; for large context blocks, place the question at the end. [Bibliography] | Use few-shot examples sparingly and consistently; avoid example overfit. [Bibliography] | Use JSON Schema structured outputs where supported. [Bibliography] | Good fit for prompt templates with explicit task/context separation. [Bibliography] |
| Open-source via vLLM / Transformers | Keep the prompt compact and stable; rely on runtime configuration for consistency. [Bibliography] | Fixed seed, low temperature, bounded `max_new_tokens`, and optional stop strings when a clear delimiter exists. [Bibliography] | Validate outputs in application code; prompt text alone is not enough. | Best when paired with strong evals and post-generation validators. [Bibliography] |

## Iterative refinement checklist

Use this as the short operational loop after adopting the revised prompt.

- Define one crisp success target before editing the prompt. Prefer concrete behaviours such as “no silent shape downgrade” or “asks only required questions,” not vague goals like “better reasoning.” [Bibliography]
- Keep one canonical rule for each load-bearing policy. If two sections say nearly the same thing, merge them instead of repeating them. [Bibliography]
- Remove anything the model cannot verify live, especially internal file paths, validator internals, or historical bug IDs. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`]
- Prefer schema-backed controls over prose-backed controls when structure matters. Use JSON Schema or tool/function calling where the runtime supports it. [Bibliography]
- Start with zero-shot on strong reasoning models, then add only the few-shot examples you can justify with eval gains. [Bibliography]
- Test every prompt change against a fixed regression set before shipping. Track failures by category, not just one overall pass rate. [Bibliography]
- Re-run the regression set whenever the model snapshot, tool schema, or runtime validator changes. [Bibliography]
- If a failure persists after prompt cleanup, do not keep stuffing more rules into the prompt. Reassess whether the real fix is model choice, structured outputs, retrieval, or application-side validation. [Bibliography]

## Bibliography

- ELSPETH source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`
- ELSPETH composer drift tests: `tests/unit/web/composer/test_skill_drift.py`
- OpenAI prompt engineering guide: https://platform.openai.com/docs/guides/prompt-engineering
- OpenAI evals guide/API reference: https://platform.openai.com/docs/guides/evals and https://platform.openai.com/docs/api-reference/evals
- OpenAI structured outputs guide: https://platform.openai.com/docs/guides/structured-outputs
- Anthropic prompt engineering overview: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Anthropic test and evaluate documentation: https://docs.anthropic.com/en/docs/test-and-evaluate/overview
- Google Gemini prompting strategies: https://ai.google.dev/gemini-api/docs/prompting-strategies
- Google Gemini long-context guidance: https://ai.google.dev/gemini-api/docs/long-context
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts": https://arxiv.org/abs/2307.03172
- Hugging Face Transformers text generation documentation: https://huggingface.co/docs/transformers/main_classes/text_generation
- vLLM sampling parameters documentation: https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html

## Open questions and limitations

This report assumes the uploaded file is the full active prompt, but it does not verify every undocumented runtime claim in that file. It also cannot prove which sections of the original prompt are actually necessary until they are tested against real ELSPETH tasks and tool traces. That limitation is normal: both OpenAI and Anthropic explicitly frame prompt engineering as an iterative, eval-driven process rather than a one-shot textual rewrite. [Source prompt: `src/elspeth/web/composer/skills/pipeline_composer.md`] [Bibliography]
