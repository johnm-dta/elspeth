# Transport + Primitive Composition Spec

**Date:** 2026-03-30
**Status:** Draft design spec
**Context:** Follow-on to the primitive plugin pack discussion and Azure Blob plugin review

---

## Summary

ELSPETH's current Azure Blob plugins are format-specific monoliths:

- `azure_blob` source = Azure auth + blob download + `csv|json|jsonl` parsing
- `azure_blob` sink = Azure auth + blob upload + `csv|json|jsonl` serialization

This works, but it hard-codes storage transport and data primitive into one plugin. The desired architecture is:

- **primitive** = the row/format behavior
- **transport** = where bytes come from or go to

In short:

- source side: `transport fetch -> primitive decode`
- sink side: `primitive encode -> transport persist`

This spec proposes a transport-composition model that lets Azure Blob wrap byte-oriented primitives (`csv`, `json`, `jsonl`, `text`) without pretending that Azure itself is the format.

---

## Problem Statement

The current Azure Blob plugins have three problems:

1. **Format duplication**
   - CSV/JSON/JSONL parsing and serialization logic exists in both local and Azure-oriented plugins.

2. **False abstraction**
   - `azure_blob` looks like a first-class primitive, but it is really "Azure transport + one of a few file-like formats."

3. **Poor extensibility**
   - Adding `text` support, or later `s3`/`gcs`, invites more copy-paste unless the transport/format boundary is made explicit.

The goal is not to create a maximal abstraction. The goal is to make the real seam honest and reusable.

---

## Proposed Model

### Core idea

Split current plugin responsibilities into two conceptual layers:

1. **Transport**
   - fetch bytes from somewhere
   - persist bytes somewhere
   - handle auth, connection, upload/download, overwrite semantics, retries, transport audit calls

2. **Primitive codec**
   - decode bytes into rows plus schema/field-resolution behavior
   - encode rows into bytes plus header/display behavior where relevant

### Source direction

```text
external storage -> transport fetch -> bytes -> primitive decoder -> SourceRow stream
```

### Sink direction

```text
rows -> primitive encoder -> bytes -> transport persist -> external storage
```

---

## Naming

Avoid `preload` / `postsave` in code. They describe timing, not responsibility.

Use:

- `transport`
- `codec`

Recommended mental model:

- **transport**: Azure Blob, local file, future S3/GCS
- **codec**: csv, json, jsonl, text

Recommended user-facing language:

- "transport-backed primitive"
- "blob transport with text codec"

---

## Scope Boundary

### Good fits for transport composition

These are byte-oriented primitives and should compose well with Azure Blob:

- `csv`
- `json`
- `jsonl`
- `text`

### Poor fits

These should remain standalone plugins:

- `null`
  - no payload exists; transport adds nothing
- `console`
  - process-local emission target, not storage transport
- `sqlite`
  - not a codec in the same sense; it is a structured persistence engine, not "rows <-> bytes blob" in the normal plugin lifecycle

### Consequence

The composition model should explicitly target **byte-stream/file-like primitives**, not all primitives.

---

## Architecture Options

### Option A: Shared codec helpers only

Extract shared internal helpers:

- decode CSV bytes to source rows
- decode JSON bytes to source rows
- encode rows to CSV bytes
- encode rows to JSON/JSONL bytes

Then:

- local file plugins call the same helpers after reading/writing local files
- Azure Blob plugins call the same helpers after download/before upload

**Pros**
- lowest implementation risk
- minimal config churn
- preserves current plugin model

**Cons**
- transport and primitive are still not first-class concepts in config
- composition is internal, not explicit

### Option B: Transport-aware wrapper plugins

Create a transport plugin that accepts a primitive/codec config:

```yaml
source:
  plugin: blob_transport
  options:
    transport:
      type: azure_blob
      container: input
      blob_path: urls.txt
    codec:
      type: text
      column: url
```

**Pros**
- abstraction is honest
- future transports become straightforward
- primitive logic can be reused directly

**Cons**
- requires config/schema redesign
- requires new orchestration/instantiation rules or nested plugin validation

### Option C: Two-layer runtime, incremental surface

Internally build transport + codec composition first, but keep a compatibility plugin surface:

- `azure_blob` remains as user-facing plugin for now
- internally it delegates to:
  - Azure transport adapter
  - codec implementation selected by `format`

Then later, if desired, expose nested config explicitly.

**Pros**
- best migration path
- least disruptive
- unlocks reuse immediately

**Cons**
- transitional duplication in names/concepts

### Recommendation

Adopt **Option C now**, with code structured so it can evolve toward **Option B** later.

That means:

- factor transport and codec seams internally first
- do not force a big config migration immediately
- keep current user-facing Azure plugin names until the internal model proves itself

---

## Proposed Internal Interfaces

These do not need to be public plugin types immediately, but they should be explicit in code.

### Source codec protocol

```python
class SourceCodec(Protocol):
    def decode(self, payload: bytes, ctx: SourceContext) -> Iterator[SourceRow]: ...
    def get_schema_contract(self) -> SchemaContract | None: ...
    def get_field_resolution(self) -> tuple[Mapping[str, str], str | None] | None: ...
```

### Sink codec protocol

```python
class SinkCodec(Protocol):
    def encode(self, rows: list[dict[str, Any]], ctx: SinkContext) -> bytes: ...
    def validate_output_target(self) -> OutputValidationResult: ...
```

### Read transport protocol

```python
class ByteReadTransport(Protocol):
    def fetch(self, ctx: SourceContext) -> bytes: ...
```

### Write transport protocol

```python
class ByteWriteTransport(Protocol):
    def persist(self, payload: bytes, ctx: SinkContext) -> ArtifactDescriptor: ...
```

These protocols are conceptual anchors. The actual implementation may begin as private helpers and typed dataclasses rather than formal public protocols.

---

## Mapping to Current Code

### Transport logic that already exists

Current Azure transport logic lives mostly in:

- `src/elspeth/plugins/infrastructure/azure_auth.py`
- `src/elspeth/plugins/sources/azure_blob_source.py`
- `src/elspeth/plugins/sinks/azure_blob_sink.py`

Reusable transport responsibilities already present:

- auth validation via `AzureAuthConfig`
- Blob client/container client creation
- blob download/upload
- audit `record_call(...)` around transport operations
- overwrite and rendered blob path behavior on sink side

### Primitive/codec logic currently welded in

Source-side welded logic:

- CSV parsing
- JSON array parsing
- JSONL parsing
- schema contract creation/locking
- field normalization and field-resolution metadata

Sink-side welded logic:

- CSV serialization
- JSON/JSONL serialization
- display-header application
- content hashing tied to encoded payload

These are the pieces to peel out.

---

## Trust Boundary and Audit Rules

### Source side

- Azure transport response bytes are **Tier 3 external data**
- codec decode/parse is the trust boundary
- parse failures are quarantine/audit events, not crashes, unless the failure is in our own code

This mirrors existing source rules and must not change.

### Sink side

- incoming rows are **Tier 2 pipeline data**
- codec encoding is our code operating on already-validated data
- transport upload is an external call boundary and must remain wrapped/audited

This also matches the current sink trust model.

### Audit primacy

Transport composition must not move or dilute current audit behavior:

- transport calls still record first-class audit events
- payload/artifact hashing still reflects the encoded bytes actually sent or read
- no row-level logging should be added as a substitute for audit state

---

## Config Evolution

### Phase 1: Internal composition, current config preserved

Keep current Azure config shape:

```yaml
source:
  plugin: azure_blob
  options:
    container: input
    blob_path: urls.txt
    format: text
```

and

```yaml
sink:
  plugin: azure_blob
  options:
    container: output
    blob_path: out.jsonl
    format: jsonl
```

Internally, `format` selects a codec implementation.

Note:
- current Azure Blob source/sink do not support `text`
- this phase would add `text` as another codec, not as an Azure-specific special case

### Phase 2: Explicit nested config if desired

Later, expose transport/codec explicitly:

```yaml
source:
  plugin: transport_source
  options:
    transport:
      type: azure_blob
      container: input
      blob_path: urls.txt
    codec:
      type: text
      column: url
```

This should be deferred until internal composition has proven stable.

---

## Non-Goals

This spec does not propose:

- generic transport composition for `null`
- generic transport composition for `console`
- replacing `sqlite` with a blob-shippable database-file pattern
- changing DAG semantics or executor architecture
- changing the audit model

---

## Migration Strategy

### Step 1

Extract internal codec helpers for:

- source decode:
  - csv
  - json
  - jsonl
  - text
- sink encode:
  - csv
  - json
  - jsonl
  - text

### Step 2

Refactor Azure Blob plugins to delegate to codec helpers.

### Step 3

Refactor local file plugins to reuse the same helpers where it improves duplication without harming clarity.

### Step 4

Decide whether the internal model is strong enough to justify explicit nested transport/codec config.

---

## Feasibility Assessment

### Overall

**Feasibility: high**

The repo already has:

- an isolated Azure auth layer
- clearly separable transport calls
- primitive plugins that already define the desired format behavior

### Main engineering cost

The cost is not "can this be done?" The cost is:

- extracting parse/serialize logic without regressing schema contract behavior
- preserving source field normalization and sink display-header semantics
- keeping tests on the real production path

### Main technical risk

The highest-risk area is **source contract behavior**, especially:

- first-row inference
- locked contract validation
- field-resolution/original-header preservation

Those behaviors are subtle and should remain codec-owned, not transport-owned.

---

## Recommendation

Proceed with internal transport/codec composition for **byte-oriented primitives only**.

Concretely:

- keep `null`, `console`, and `sqlite` standalone
- teach Azure Blob source/sink to delegate to codec implementations
- add `text` codec support there as part of the refactor
- defer any public nested `transport + codec` config until after the internal seam is proven

This gives ELSPETH the cleaner architecture you described without forcing a premature public API redesign.

---

## Suggested Follow-On Work

If this direction is approved, the next doc should be an implementation plan covering:

1. internal codec extraction
2. Azure Blob source refactor to codec composition
3. Azure Blob sink refactor to codec composition
4. `text` codec support for Azure Blob
5. compatibility and regression test coverage
