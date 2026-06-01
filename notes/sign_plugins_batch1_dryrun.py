"""
Dry-run justify script for 28 unsigned plugins.yaml entries.
Run as: set -a && source .env && set +a && .venv/bin/python notes/sign_plugins_batch1_dryrun.py
Writes results to notes/sign_plugins_batch1_results.txt
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

ENTRIES = [
    # (file_path_rel_to_root, rule, symbol, fingerprint, rationale)
    # --- csv_source.py ---
    (
        "plugins/sources/csv_source.py", "R6", "CSVSource._load_from_file", "fp=054e76166ade23bc",
        "CSV source parsing reads Tier-3 external bytes; csv.Error during skip_rows means the parser "
        "consumed an unknown amount of data (e.g. unmatched quote swallowed subsequent lines). The handler "
        "records the error via ctx.record_validation_error and stops processing to avoid silent data loss "
        "from corrupted parser state. Never coerces the error into a continued parse.",
    ),
    (
        "plugins/sources/csv_source.py", "R6", "CSVSource._load_from_file", "fp=0fb2b154fb016195",
        "CSV source reads Tier-3 external bytes; StopIteration after skip_rows means skip_rows consumed "
        "all content before a header row was found. Records the absence via ctx.record_validation_error "
        "so the audit trail shows skip_rows exhausted the file. Returns cleanly with no fabricated rows.",
    ),
    (
        "plugins/sources/csv_source.py", "R6", "CSVSource._load_from_file", "fp=bb231ad98805d136",
        "CSV source schema validation catches pydantic ValidationError on Tier-3 rows that violate the "
        "configured schema. Records the validation failure via ctx.record_validation_error and yields a "
        "quarantined SourceRow per the on_validation_failure policy. Never coerces the error.",
    ),
    (
        "plugins/sources/csv_source.py", "R6", "CSVSource._load_from_file", "fp=7b8583638b096231",
        "CSV source reading Tier-3 external bytes; csv.Error during main data row iteration means the "
        "parser is in a corrupted state (bad quoting, unmatched quotes). Records the error via "
        "ctx.record_validation_error and stops processing to avoid silent data loss from corrupted "
        "parser state. Never coerces the error into a continued parse.",
    ),
    # --- dataverse.py ---
    (
        "plugins/sources/dataverse.py", "R1", "DataverseSource._normalize_row_fields", "fp=b9db16c1b599becb",
        "Dataverse API rows are Tier-3 and may be sparse; the field_resolution mapping is built from the "
        "first row but Dataverse may return fields absent from that first row in later rows. mapping.get(k) "
        "returns None specifically to detect unmapped fields, which are then normalized individually. "
        "None is the correct sentinel for field-not-in-mapping and is never used as a fabricated default value.",
    ),
    # --- manager.py ---
    (
        "plugins/infrastructure/manager.py", "R4", "PluginManager.register", "fp=0e88b71c015f8aa0",
        "Rollback-preserves-primary-error pattern: a broad Exception catch in the registration path triggers "
        "unregister and cache-refresh rollback, then re-raises the original failure. The inner try/except on "
        "the rollback itself logs the rollback error but re-raises the outer exception unchanged so the "
        "secondary failure never buries the original diagnostic.",
    ),
    # --- batching/mixin.py ---
    (
        "plugins/infrastructure/batching/mixin.py", "R9", "BatchTransformMixin._release_loop", "fp=2f93bd40222e5b89",
        "Release loop submission tracking cleanup: .pop(..., None) on the _batch_submissions dict after a "
        "result has been released. The entry may already be absent if evict_submission() ran concurrently "
        "(batch timeout race). Returning None when not found is the correct documented behavior for this "
        "race-aware cleanup. This is process-internal state, not defensive masking of a bug.",
    ),
    (
        "plugins/infrastructure/batching/mixin.py", "R6", "BatchTransformMixin._release_loop", "fp=24666a1e729638d7",
        "Release loop output-port failure handler: when the emit fails after the token is known, the "
        "exception is wrapped into an ExceptionResult so the waiting caller gets an error response rather "
        "than hanging indefinitely. When token is None (pre-unpack), the exception re-raises immediately "
        "as a first-party invariant violation. This is the thread-boundary exception transport pattern: "
        "waiters cannot receive bare exceptions so they are channeled through ExceptionResult.",
    ),
    # --- http.py ---
    (
        "plugins/infrastructure/clients/http.py", "R1", "AuditedHTTPClient._parse_response_body", "fp=c10e1a372d3013e8",
        "HTTP response content-type header is a Tier-3 external boundary; content_type = "
        "response.headers.get('content-type', '') uses empty string default when the header is absent. "
        "An absent content-type maps to binary encoding which is the correct safe choice for unknown "
        "content types. Not a fabricated absence but a legitimate dispatch decision.",
    ),
    # --- llm.py R1 ---
    (
        "plugins/infrastructure/clients/llm.py", "R1", "_extract_usage_from_provider_response", "fp=153b600f7b824428",
        "LLM provider usage mappings are Tier-3 external data; providers may omit prompt_tokens, "
        "completion_tokens, or total_tokens. dict.get with None default preserves honest absence: "
        "missing fields become None in the downstream TokenUsage.from_dict normalization, not fabricated "
        "zeros. Never coerces a missing count.",
    ),
    (
        "plugins/infrastructure/clients/llm.py", "R1", "_extract_usage_from_provider_response", "fp=54f676db05cacf08",
        "LLM provider usage mappings are Tier-3 external data; providers may omit prompt_tokens, "
        "completion_tokens, or total_tokens. dict.get with None default preserves honest absence: "
        "missing fields become None in the downstream TokenUsage.from_dict normalization, not fabricated "
        "zeros. Never coerces a missing count.",
    ),
    (
        "plugins/infrastructure/clients/llm.py", "R1", "_extract_usage_from_provider_response", "fp=3ce6385129d45fcc",
        "LLM provider usage mappings are Tier-3 external data; providers may omit prompt_tokens, "
        "completion_tokens, or total_tokens. dict.get with None default preserves honest absence: "
        "missing fields become None in the downstream TokenUsage.from_dict normalization, not fabricated "
        "zeros. Never coerces a missing count.",
    ),
    # --- llm.py R2 ---
    (
        "plugins/infrastructure/clients/llm.py", "R2", "_extract_usage_from_provider_response", "fp=92c580e9594a61bd",
        "LLM provider SDK usage objects vary by version and may omit token count attributes; "
        "getattr with None default preserves honest absence: missing attributes become None in the "
        "downstream TokenUsage.from_dict normalization, not fabricated zeros. The parameter is typed "
        "as Any because providers use both dict-shaped and object-shaped usage payloads. Never coerces.",
    ),
    (
        "plugins/infrastructure/clients/llm.py", "R2", "_extract_usage_from_provider_response", "fp=d741dbbe7cf1857f",
        "LLM provider SDK usage objects vary by version and may omit token count attributes; "
        "getattr with None default preserves honest absence: missing attributes become None in the "
        "downstream TokenUsage.from_dict normalization, not fabricated zeros. The parameter is typed "
        "as Any because providers use both dict-shaped and object-shaped usage payloads. Never coerces.",
    ),
    # --- rag/transform.py ---
    (
        "plugins/transforms/rag/transform.py", "R2", "RAGRetrievalTransform._configured_collection_name", "fp=757eb1e99aea1a3e",
        "Provider config is a discriminated union of typed provider-specific models; different providers "
        "use either collection or index attribute for the readiness-audit name. getattr with None default "
        "is the only non-hasattr way to probe which naming convention the provider uses. If neither "
        "attribute is present, FrameworkBugError is raised immediately as offensive detection of a "
        "misconfigured provider config, not silent coercion.",
    ),
    (
        "plugins/transforms/rag/transform.py", "R5", "RAGRetrievalTransform._configured_collection_name", "fp=b22badfe2569d3ea",
        "Provider config field values come from getattr with None default which returns None when the "
        "attribute is absent. isinstance(value, str) guards against a provider config that has the "
        "attribute but with a non-string value, which is a first-party contract violation. The guard "
        "rejects unusable values rather than passing them to the readiness audit call. If neither "
        "attribute yields a non-empty string, FrameworkBugError is raised.",
    ),
    # --- chroma_sink.py ---
    (
        "plugins/sinks/chroma_sink.py", "R6", "ChromaSink.write", "fp=8501da8c0e5bf4c0",
        "ChromaDB metadata fields are optional per-field: a row may lack any of the configured metadata "
        "fields and a missing metadata field is not an error. The except KeyError continue skips the "
        "field rather than diverting the row. The row is valid without the metadata field and continues "
        "processing normally. No audit evidence is lost.",
    ),
    (
        "plugins/sinks/chroma_sink.py", "R5", "ChromaSink.write", "fp=12aaa412bd91a810",
        "ChromaDB rejects metadata values that are not str, int, float, bool, or None and also rejects "
        "non-finite floats. isinstance type discrimination is required to identify and divert rows with "
        "incompatible metadata values before they hit the ChromaDB API and cause an unattributable batch "
        "failure. The check prevents a runtime API error from swallowing the row without trace. "
        "Diverted rows carry an explicit reason and are handled per on_write_failure policy.",
    ),
    (
        "plugins/sinks/chroma_sink.py", "R5", "ChromaSink.write", "fp=87156d7ec508a40b",
        "ChromaDB rejects non-finite float metadata values (NaN, Infinity). isinstance(value, float) "
        "narrows to float before the math.isfinite check, which is the only way to test for non-finite "
        "floats without raising TypeError on non-numeric types. The check identifies and diverts rows "
        "with non-finite metadata values before they hit the ChromaDB API. Diverted rows carry an "
        "explicit reason and are handled per on_write_failure policy.",
    ),
    (
        "plugins/sinks/chroma_sink.py", "R6", "ChromaSink.write", "fp=fe6fa1b761d7dd9c",
        "ChromaDB metadata fields are optional per-field: a row may lack any of the configured metadata "
        "fields and a missing metadata field is not an error. The except KeyError continue skips the "
        "field rather than diverting the row. The row is valid without the metadata field and continues "
        "processing normally. No audit evidence is lost.",
    ),
    # --- web_scrape.py ---
    (
        "plugins/transforms/web_scrape.py", "R5", "WebScrapeTransform.get_post_call_hints", "fp=f919002b3beb8525",
        "config_snapshot contains LLM-supplied option values where text_separator type is not statically "
        "guaranteed (the parameter is Mapping[str, object]). isinstance(sep, str) is required before "
        "the substring membership check since passing a non-string to the in operator raises TypeError. "
        "If sep is not a string (malformed LLM output), no hint is generated rather than crashing the "
        "hints function. This is a type-narrowing guard at a Tier-3 untyped input seam.",
    ),
    # --- azure_blob_sink.py ---
    (
        "plugins/sinks/azure_blob_sink.py", "R6", "AzureBlobSink.write", "fp=304f148f13df7df3",
        "Tier-2 pipeline rows for JSON/JSONL Azure Blob upload may contain non-serializable values "
        "(NaN, Infinity, or non-serializable objects); json.dumps(allow_nan=False) raises ValueError "
        "or TypeError. Per-row divert rather than batch abort: one bad value does not abort the entire "
        "blob upload. The exception is narrow (ValueError, TypeError), each failing row is diverted via "
        "_divert_row with an explicit reason, and the remaining serializable rows are uploaded.",
    ),
    # --- database_sink.py ---
    (
        "plugins/sinks/database_sink.py", "R6", "DatabaseSink._insert_with_per_row_diversion", "fp=9e9d1f67fecbd6f2",
        "Batch IntegrityError triggers the per-row diversion strategy: the batch savepoint is rolled "
        "back and rows are re-inserted individually to isolate the constraint-violating rows. This is "
        "not swallowing: the batch failure is expected when constraints are violated and the row-by-row "
        "pass identifies and diverts only the offending rows. Connection and programming errors are not "
        "caught here and propagate unhandled.",
    ),
    (
        "plugins/sinks/database_sink.py", "R6", "DatabaseSink._insert_with_per_row_diversion", "fp=d64a7ca89b0cfe0e",
        "Per-row IntegrityError in the diversion pass: each row is executed in its own savepoint and a "
        "constraint violation rolls back that savepoint and diverts the row via _divert_row with the "
        "specific constraint violation detail. Rows that succeed are committed into the outer transaction. "
        "Not error swallowing: each diverted row carries the original exception reason.",
    ),
    # --- csv_sink.py ---
    (
        "plugins/sinks/csv_sink.py", "R6", "CSVSink._stage_rows_per_row", "fp=73ebd4f4b271eb4f",
        "Tier-2 pipeline rows may trigger DictWriter serialization failures (ValueError from "
        "extrasaction=raise on fields outside the locked column set, or csv.Error from the writer); "
        "per-row trial-encode in a throwaway buffer diverts only the failing row so the batch continues. "
        "The exceptions are narrow (ValueError, csv.Error). A value whose str() itself raises is a "
        "broken object and propagates unhandled as a plugin bug.",
    ),
    # --- json_sink.py ---
    (
        "plugins/sinks/json_sink.py", "R6", "JSONSink.write", "fp=70e172c6d4480200",
        "JSONL write rollback guard: any failure in the write path (write_jsonl_content, flush, hash, "
        "stat) must roll back the file to pre_write_pos to prevent partial writes from corrupting the "
        "JSONL file. The broad except is always followed by raise so no error is swallowed. The inner "
        "OSError handler on the rollback preserves the original error via raise from rollback_err.",
    ),
    (
        "plugins/sinks/json_sink.py", "R6", "JSONSink._write_jsonl_content", "fp=7f2a8740a6f881fa",
        "Tier-2 pipeline rows may contain non-JSON-serializable values (NaN, Infinity, or non-serializable "
        "objects); json.dumps(allow_nan=False) raises ValueError or TypeError on such values. Per-row "
        "divert rather than batch abort: one bad value does not stop the remaining rows from being "
        "written to the JSONL file. The exception is narrow (ValueError, TypeError), the failing row is "
        "diverted via _divert_row with explicit reason.",
    ),
    # --- azure_blob_source.py ---
    (
        "plugins/sources/azure_blob_source.py", "R6", "AzureBlobSource._load_csv", "fp=e59ce88529f38df2",
        "CSV blob data from Azure Blob Storage is Tier-3 external content; UnicodeDecodeError means the "
        "blob bytes cannot be decoded with the configured encoding. Records the failure via "
        "ctx.record_validation_error and yields a quarantined SourceRow if not in discard mode, rather "
        "than propagating an unhandled decode error. The exception is narrow (UnicodeDecodeError only).",
    ),
]


def run_dry_run(file_path, rule, symbol, fingerprint, rationale):
    cmd = [
        PYTHON, "-m", "elspeth_lints.core.cli", "justify",
        "--root", "src/elspeth",
        "--allowlist-dir", "config/cicd/enforce_tier_model",
        "--file-path", file_path,
        "--rule", rule,
        "--symbol", symbol,
        "--fingerprint", fingerprint,
        "--rationale", rationale,
        "--owner", "claude-sonnet-4-6",
        "--dry-run",
    ]
    env = {"PYTHONPATH": "elspeth-lints/src"}
    import os
    env.update(os.environ)

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env,
        cwd=str(REPO_ROOT),
    )
    return result.stdout + result.stderr


def main():
    out_path = REPO_ROOT / "notes" / "sign_plugins_batch1_results.txt"
    with out_path.open("w") as f:
        for i, (file_path, rule, symbol, fingerprint, rationale) in enumerate(ENTRIES, 1):
            key = f"{file_path}:{rule}:{symbol}:{fingerprint}"
            print(f"[{i:02d}/{len(ENTRIES)}] {key}", flush=True)
            f.write(f"\n{'='*80}\n")
            f.write(f"ENTRY {i:02d}: {key}\n")
            f.write(f"RATIONALE: {rationale}\n")
            f.write(f"{'='*80}\n")
            output = run_dry_run(file_path, rule, symbol, fingerprint, rationale)
            f.write(output)
            f.flush()
            # Extract verdict from output
            verdict = "UNKNOWN"
            for line in output.splitlines():
                if "ACCEPTED" in line:
                    verdict = "ACCEPTED"
                    break
                if "BLOCKED" in line or "REJECTED" in line:
                    verdict = "BLOCKED"
                    break
            print(f"  → {verdict}")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
