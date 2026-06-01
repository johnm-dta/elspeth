#!/usr/bin/env bash
# Operator paste-and-run: sign 19 ACCEPTED plugins.yaml entries.
# Run from repo root in the certificated (HMAC-keyed) shell.
# All rationales were pre-screened via --dry-run (judge: ACCEPTED).
set -euo pipefail

: "${ELSPETH_JUDGE_METADATA_HMAC_KEY:?HMAC key required}"

J() {
  PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --judge-transport openrouter --owner "$USER" "$@"
}

# ── csv_source.py ────────────────────────────────────────────────────────────

J --file-path plugins/sources/csv_source.py --rule R6 \
  --symbol 'CSVSource._load_from_file' --fingerprint fp=054e76166ade23bc \
  --rationale 'CSV source schema validation catches pydantic ValidationError on Tier-3 rows that violate the configured schema. Records the validation failure via ctx.record_validation_error and yields a quarantined SourceRow per the on_validation_failure policy. Never coerces the error.'

J --file-path plugins/sources/csv_source.py --rule R6 \
  --symbol 'CSVSource._load_from_file' --fingerprint fp=0fb2b154fb016195 \
  --rationale 'StopIteration at the top of the main data-row loop is the normal iterator-exhaustion signal from next_nonblank_record() -- it signals end-of-file on csv.reader. The bare break is the conventional Python iterator-EOF idiom, not error swallowing. No data is lost and no audit record is required for normal file exhaustion.'

J --file-path plugins/sources/csv_source.py --rule R6 \
  --symbol 'CSVSource._load_from_file' --fingerprint fp=bb231ad98805d136 \
  --rationale 'CSV source schema validation catches pydantic ValidationError on Tier-3 rows that violate the configured schema. Records the validation failure via ctx.record_validation_error and yields a quarantined SourceRow per the on_validation_failure policy. Never coerces the error.'

J --file-path plugins/sources/csv_source.py --rule R6 \
  --symbol 'CSVSource._load_from_file' --fingerprint fp=7b8583638b096231 \
  --rationale 'CSV source reading Tier-3 external bytes; csv.Error during main data row iteration means the parser is in a corrupted state (bad quoting, unmatched quotes). Records the error via ctx.record_validation_error and stops processing to avoid silent data loss from corrupted parser state. Never coerces the error into a continued parse.'

# ── dataverse.py ─────────────────────────────────────────────────────────────

J --file-path plugins/sources/dataverse.py --rule R1 \
  --symbol 'DataverseSource._normalize_row_fields' --fingerprint fp=b9db16c1b599becb \
  --rationale 'Dataverse API rows are Tier-3 and may be sparse; the field_resolution mapping is built from the first row but Dataverse may return fields absent from that first row in later rows. mapping.get(k) returns None specifically to detect unmapped fields, which are then normalized individually. None is the correct sentinel for field-not-in-mapping and is never used as a fabricated default value.'

# ── infrastructure/manager.py ────────────────────────────────────────────────

J --file-path plugins/infrastructure/manager.py --rule R4 \
  --symbol 'PluginManager.register' --fingerprint fp=0e88b71c015f8aa0 \
  --rationale 'Rollback-preserves-primary-error pattern: a broad Exception catch in the registration path triggers unregister and cache-refresh rollback, then re-raises the original failure. The inner try/except on the rollback itself logs the rollback error but re-raises the outer exception unchanged so the secondary failure never buries the original diagnostic.'

# ── infrastructure/batching/mixin.py ─────────────────────────────────────────

J --file-path plugins/infrastructure/batching/mixin.py --rule R9 \
  --symbol 'BatchTransformMixin._release_loop' --fingerprint fp=2f93bd40222e5b89 \
  --rationale 'Release loop submission tracking cleanup: .pop(..., None) on the _batch_submissions dict after a result has been released. The entry may already be absent if evict_submission() ran concurrently (batch timeout race). Returning None when not found is the correct documented behavior for this race-aware cleanup. This is process-internal state, not defensive masking of a bug.'

J --file-path plugins/infrastructure/batching/mixin.py --rule R6 \
  --symbol 'BatchTransformMixin._release_loop' --fingerprint fp=24666a1e729638d7 \
  --rationale 'ShutdownError is the designed exit signal from _batch_buffer.wait_for_next_release(): it fires when shutdown_batch_processing() calls buffer.shutdown() after all workers have finished processing. The bare break is the only exit path from the release loop -- it is not error swallowing but the documented clean-shutdown protocol. By contract, shutdown_batch_processing() drains workers before calling buffer.shutdown(), so all completed results have already been emitted.'

# ── sinks/chroma_sink.py ─────────────────────────────────────────────────────

J --file-path plugins/sinks/chroma_sink.py --rule R6 \
  --symbol 'ChromaSink.write' --fingerprint fp=8501da8c0e5bf4c0 \
  --rationale 'Required field extraction: the row dict is indexed by the configured id_field and document_field; KeyError means the required field is absent from the row. Per-row divert via _divert_row records the absence with the specific missing field name as the reason. This is an explicit quarantine result handled per on_write_failure policy, not silent swallowing.'

J --file-path plugins/sinks/chroma_sink.py --rule R6 \
  --symbol 'ChromaSink.write' --fingerprint fp=c52a6a2457cc3cdc \
  --rationale 'Optional metadata fields are absent from some rows; except KeyError continue is the correct expression of per-field optionality. Not error swallowing -- metadata absence is valid and the row proceeds normally with whatever metadata was present.'

J --file-path plugins/sinks/chroma_sink.py --rule R5 \
  --symbol 'ChromaSink.write' --fingerprint fp=459fb5a3b1e6a043 \
  --rationale 'Tier-2 operation-unsafe guard at the ChromaDB external boundary: isinstance(value, (str, int, float, bool)) detects metadata values whose types ChromaDB does not accept. A wrong-typed metadata value (e.g. dict or datetime) is value-unsafe at this Tier-2 to external boundary -- the type is valid Python but causes an operation failure at the ChromaDB API. Per-row divert prevents an unattributable batch rejection. This is the Tier-2 value-unsafe wrap pattern prescribed by the data manifesto.'

J --file-path plugins/sinks/chroma_sink.py --rule R5 \
  --symbol 'ChromaSink.write' --fingerprint fp=801f187823d20f74 \
  --rationale 'Tier-2 operation-unsafe guard at the ChromaDB external boundary: isinstance(value, float) narrows to float before math.isfinite() to detect non-finite floats (NaN/Infinity). The type is correct (float) but the value causes a ChromaDB operation failure. This is the Tier-2 value-unsafe wrap pattern -- type-valid but operation-unsafe values must be wrapped before crossing into the external system.'

# ── sinks/azure_blob_sink.py ─────────────────────────────────────────────────

J --file-path plugins/sinks/azure_blob_sink.py --rule R6 \
  --symbol 'AzureBlobSink.write' --fingerprint fp=304f148f13df7df3 \
  --rationale 'Tier-2 pipeline rows for JSON/JSONL Azure Blob upload may contain non-serializable values (NaN, Infinity, or non-serializable objects); json.dumps(allow_nan=False) raises ValueError or TypeError. Per-row divert rather than batch abort: one bad value does not abort the entire blob upload. The exception is narrow (ValueError, TypeError), each failing row is diverted via _divert_row with an explicit reason, and the remaining serializable rows are uploaded.'

# ── sinks/database_sink.py ───────────────────────────────────────────────────

J --file-path plugins/sinks/database_sink.py --rule R6 \
  --symbol 'DatabaseSink._insert_with_per_row_diversion' --fingerprint fp=9e9d1f67fecbd6f2 \
  --rationale 'Batch IntegrityError triggers the per-row diversion strategy: the batch savepoint is rolled back and rows are re-inserted individually to isolate the constraint-violating rows. This is not swallowing: the batch failure is expected when constraints are violated and the row-by-row pass identifies and diverts only the offending rows. Connection and programming errors are not caught here and propagate unhandled.'

J --file-path plugins/sinks/database_sink.py --rule R6 \
  --symbol 'DatabaseSink._insert_with_per_row_diversion' --fingerprint fp=d64a7ca89b0cfe0e \
  --rationale 'Per-row IntegrityError in the diversion pass: each row is executed in its own savepoint and a constraint violation rolls back that savepoint and diverts the row via _divert_row with the specific constraint violation detail. Rows that succeed are committed into the outer transaction. Not error swallowing: each diverted row carries the original exception reason.'

# ── sinks/csv_sink.py ────────────────────────────────────────────────────────

J --file-path plugins/sinks/csv_sink.py --rule R6 \
  --symbol 'CSVSink._stage_rows_per_row' --fingerprint fp=73ebd4f4b271eb4f \
  --rationale 'Tier-2 pipeline rows may trigger DictWriter serialization failures (ValueError from extrasaction=raise on fields outside the locked column set, or csv.Error from the writer); per-row trial-encode in a throwaway buffer diverts only the failing row so the batch continues. The exceptions are narrow (ValueError, csv.Error). A value whose str() itself raises is a broken object and propagates unhandled as a plugin bug.'

# ── sinks/json_sink.py ───────────────────────────────────────────────────────

J --file-path plugins/sinks/json_sink.py --rule R6 \
  --symbol 'JSONSink.write' --fingerprint fp=70e172c6d4480200 \
  --rationale 'Tier-2 pipeline rows for JSON array write may contain non-JSON-serializable values (NaN, Infinity, or non-serializable objects); json.dumps(allow_nan=False) raises ValueError or TypeError. Per-row divert via _divert_row with explicit reason rather than aborting the entire JSON array write. One bad value does not invalidate the batch.'

J --file-path plugins/sinks/json_sink.py --rule R6 \
  --symbol 'JSONSink._write_jsonl_content' --fingerprint fp=7f2a8740a6f881fa \
  --rationale 'Tier-2 pipeline rows may contain non-JSON-serializable values (NaN, Infinity, or non-serializable objects); json.dumps(allow_nan=False) raises ValueError or TypeError on such values. Per-row divert rather than batch abort: one bad value does not stop the remaining rows from being written to the JSONL file. The exception is narrow (ValueError, TypeError), the failing row is diverted via _divert_row with explicit reason.'

# ── sources/azure_blob_source.py ─────────────────────────────────────────────

J --file-path plugins/sources/azure_blob_source.py --rule R6 \
  --symbol 'AzureBlobSource._load_csv' --fingerprint fp=e59ce88529f38df2 \
  --rationale 'CSV blob data from Azure Blob Storage is Tier-3 external content; UnicodeDecodeError means the blob bytes cannot be decoded with the configured encoding. Records the failure via ctx.record_validation_error and yields a quarantined SourceRow if not in discard mode, rather than propagating an unhandled decode error. The exception is narrow (UnicodeDecodeError only).'

echo ""
echo "All 19 entries signed."
