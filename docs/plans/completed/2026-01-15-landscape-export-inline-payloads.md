# Landscape Export: Inline Payloads + Chunked JSON

Status: Draft
Owner: TBD
Date: 2026-01-15

## Problem Statement

The current audit export contains hashes/refs but not the actual row data. The
database is an operational artifact, not the auditor-facing deliverable. For
audits, we need an export artifact that includes the full data needed to
reconstruct how the pipeline produced each outcome.

## Goals

- Export includes full-fidelity payloads:
  - Source row data (as ingested)
  - Pre-transform (node_state input) data
  - Post-transform (node_state output) data
  - Routing decision reason data
  - Sink input data
  - Call request/response payloads when available
- Export remains a single logical artifact, but can be chunked into multiple
  files for size (default 10MB chunks).
- Payload inclusion is configurable but defaults to **on**.
- Export is self-contained and reconstructable for a subset of entries without
  relying on the operational DB.

## Non-goals

- Redaction policies (out of scope for this pass).
- Compression or encryption (future work).
- New storage backends beyond filesystem payload store (future work).
- DAG replays or deterministic re-execution engines (existing design).

## Proposed Design

### 1) Payload Capture + Reference Strategy

We already have a payload store abstraction, but it is not wired into the
engine. We will store JSON-serialized payloads and save a reference hash in the
audit DB.

Payloads to store:
- **Source rows**: stored when creating a `Row` record.
- **Node states**: store input and output payloads for every node state.
- **Routing reasons**: store routing reason payloads when present.
- **External calls**: store request and response payloads.

Storage method:
- Serialize payload to canonical JSON bytes.
- Store in payload store by content hash.
- Save reference in DB (ref hash).

### 2) Schema Changes (Audit Tables)

Add to `node_states`:
- `input_data_ref` (nullable)
- `output_data_ref` (nullable)

Existing fields used:
- `rows.source_data_ref`
- `routing_events.reason_ref`
- `calls.request_ref`, `calls.response_ref`

Note: DB migration or reset required for existing runs. This design assumes a
fresh schema for new runs.

### 3) Export Format Changes

We will update the exporter to include payloads in the JSON export when enabled.

New export fields:
- `row.source_data` (inlined payload)
- `node_state.input_data`, `node_state.output_data`
- `routing_event.reason`
- `call.request`, `call.response`

Export remains a flat list of record dicts, with payloads inlined directly in
the relevant record. This supports reconstructing the pipeline for subsets of
rows without DB access.

### 4) Chunked JSON Export (Default)

Export writes JSON in multiple files when size exceeds a threshold:
- Default chunk size: `10_000_000` bytes (10MB)
- Chunk filenames: `audit_trail.000.json`, `audit_trail.001.json`, ...
- The configured sink path is used as the base prefix.

Manifest (recommended):
- Write an `audit_trail.manifest.json` file with:
  - `run_id`
  - `chunk_count`
  - `chunk_size_bytes`
  - `total_record_count`
  - `sha256` for each chunk file

Rationale: chunking keeps files manageable without losing the “single logical
artifact” property. The manifest makes validation and reconstruction explicit.

### 5) Configuration Changes

Add to `LandscapeExportSettings`:
- `include_payloads: bool = True`
- `chunk_size_bytes: int | None = 10_000_000`
  - `None` disables chunking (single file).

Optional future settings:
- `manifest: bool = True` (if we want to toggle it separately).

### 6) Engine Integration

Wire payload store into runtime:
- Build `FilesystemPayloadStore` from `payload_store` settings.
- Pass into `LandscapeRecorder` during run.
- Add to `PluginContext` for potential plugin access.

Update recorders:
- `create_row(..., payload_ref=...)`
- `begin_node_state(..., input_data_ref=...)`
- `complete_node_state(..., output_data_ref=...)`
- `record_routing_event(..., reason_ref=...)`
- `record_call(..., request_ref=..., response_ref=...)` (once call recording exists)

### 7) Export Logic Integration

Exporter changes:
- Add `include_payloads` flag to `export_run(...)`.
- If enabled, fetch payload bytes from payload store and attach decoded JSON to
  exported records.

Orchestrator export changes:
- When chunking is enabled, bypass the JSON sink and write files directly to
  disk using a chunking writer (since current JSONSink writes a single file).
- CSV export remains unchanged (payloads are not suitable for CSV).

## Risks / Open Questions

- Large payloads can make export huge; chunking mitigates size but increases
  file count. We should bound memory usage during export by streaming records.
- We need a clear policy for binary payloads (e.g., encode bytes via
  canonical_json, which already encodes bytes to base64).
- Signing and chunking: signatures are per-record today. If we add a manifest,
  it should include per-chunk hashes to validate chunk integrity.

## Test Plan

- Unit tests for payload store integration:
  - Row data stored and retrieved by `source_data_ref`.
  - Node state input/output payload refs stored and retrieved.
  - Routing reason payload stored and retrieved.
- Export integration:
  - JSON export includes `source_data`, `input_data`, `output_data`, `reason`.
  - Chunked export writes multiple files with expected sizes.
  - Manifest file matches chunk hashes and record counts.

## Implementation Phases

1) Schema + recorder changes (payload refs on node_states).
2) Wire payload store into orchestration.
3) Exporter payload inclusion.
4) Chunked JSON export writer + manifest.
5) Update examples/docs/tests.
