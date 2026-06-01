"""
Second-round dry-run: corrected rationales for 5 wrong-rationale BLOCKED entries.
Run as: set -a && source .env && set +a && .venv/bin/python notes/sign_plugins_batch2_dryrun.py
"""

import subprocess, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

ENTRIES = [
    # Entry 01 corrected: judge said fp=054e76166ade23bc is ValidationError at line 549
    # (same quarantine pattern that bb231ad98805d136 used — that was ACCEPTED)
    (
        "plugins/sources/csv_source.py", "R6", "CSVSource._load_from_file", "fp=054e76166ade23bc",
        "CSV source schema validation catches pydantic ValidationError on Tier-3 rows that violate "
        "the configured schema. Records the validation failure via ctx.record_validation_error and "
        "yields a quarantined SourceRow per the on_validation_failure policy. Never coerces the error.",
    ),
    # Entry 02 corrected: judge said fp=0fb2b154fb016195 is StopIteration bare break (normal EOF)
    (
        "plugins/sources/csv_source.py", "R6", "CSVSource._load_from_file", "fp=0fb2b154fb016195",
        "StopIteration at the top of the main data-row loop is the normal iterator-exhaustion signal "
        "from next_nonblank_record() — it signals end-of-file on csv.reader. The bare break is the "
        "conventional Python iterator-EOF idiom, not error swallowing. No data is lost and no "
        "audit record is required for normal file exhaustion.",
    ),
    # Entry 08 corrected: judge said fp=24666a1e729638d7 is ShutdownError:break (designed exit)
    (
        "plugins/infrastructure/batching/mixin.py", "R6", "BatchTransformMixin._release_loop", "fp=24666a1e729638d7",
        "ShutdownError is the designed exit signal from _batch_buffer.wait_for_next_release(): it "
        "fires when shutdown_batch_processing() calls buffer.shutdown() after all workers have "
        "finished processing. The bare break is the only exit path from the release loop — it is "
        "not error swallowing but the documented clean-shutdown protocol. By contract, "
        "shutdown_batch_processing() drains workers before calling buffer.shutdown(), so all "
        "completed results have already been emitted.",
    ),
    # Entry 17 corrected: judge said fp=8501da8c0e5bf4c0 is required-field KeyError divert
    (
        "plugins/sinks/chroma_sink.py", "R6", "ChromaSink.write", "fp=8501da8c0e5bf4c0",
        "Required field extraction: the row dict is indexed by the configured id_field and "
        "document_field; KeyError means the required field is absent from the row. Per-row "
        "divert via _divert_row records the absence with the specific missing field name as "
        "the reason. This is an explicit quarantine result handled per on_write_failure policy, "
        "not silent swallowing.",
    ),
    # Entry 26 corrected: judge said fp=70e172c6d4480200 is ValueError/TypeError divert in write()
    (
        "plugins/sinks/json_sink.py", "R6", "JSONSink.write", "fp=70e172c6d4480200",
        "Tier-2 pipeline rows for JSON array write may contain non-JSON-serializable values "
        "(NaN, Infinity, or non-serializable objects); json.dumps(allow_nan=False) raises "
        "ValueError or TypeError. Per-row divert via _divert_row with explicit reason rather "
        "than aborting the entire JSON array write. One bad value does not invalidate the batch.",
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
    env.update(os.environ)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env, cwd=str(REPO_ROOT))
    return result.stdout + result.stderr


def main():
    out_path = REPO_ROOT / "notes" / "sign_plugins_batch2_results.txt"
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
            verdict = "UNKNOWN"
            for line in output.splitlines():
                if "ACCEPTED" in line: verdict = "ACCEPTED"; break
                if "BLOCKED" in line or "REJECTED" in line: verdict = "BLOCKED"; break
            print(f"  → {verdict}")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
