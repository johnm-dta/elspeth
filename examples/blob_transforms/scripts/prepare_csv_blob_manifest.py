"""Prepare local CSV blobs for the blob_csv_expand example."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from elspeth.core.payload_store import FilesystemPayloadStore

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = ROOT / "examples" / "blob_transforms"
PAYLOAD_DIR = EXAMPLE_DIR / "payloads"
MANIFEST_PATH = EXAMPLE_DIR / "input" / "csv_blob_manifest.csv"

INPUTS = (
    ("feed_a", ROOT / "examples" / "multi_worker_showcase" / "input" / "feed_a.csv"),
    ("feed_b", ROOT / "examples" / "multi_worker_showcase" / "input" / "feed_b.csv"),
)


def main() -> None:
    store = FilesystemPayloadStore(PAYLOAD_DIR)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("manifest_index", "source_name", "source_url", "blob_ref"),
        )
        writer.writeheader()
        for manifest_index, (source_name, path) in enumerate(INPUTS):
            content = path.read_bytes()
            blob_ref = store.store(content)
            writer.writerow(
                {
                    "manifest_index": manifest_index,
                    "source_name": source_name,
                    "source_url": path.as_posix(),
                    "blob_ref": blob_ref,
                }
            )

    sys.stdout.write(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}\n")
    sys.stdout.write(f"Stored {len(INPUTS)} blobs in {PAYLOAD_DIR.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
