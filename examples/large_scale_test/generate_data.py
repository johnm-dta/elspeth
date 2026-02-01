#!/usr/bin/env python3
"""
Generate procedurally-generated CSV data for large-scale testing.

Creates a CSV with configurable row count (default 50k) containing:
- id: Sequential row identifier
- value: Random float between 0-10000
- category: One of 5 categories (A, B, C, D, E)
- priority: Random integer 1-5
- timestamp: ISO 8601 timestamp incrementing by seconds

Usage:
    python generate_data.py              # 50,000 rows
    python generate_data.py 100000       # 100,000 rows
    python generate_data.py 10000        # 10,000 rows
"""

import csv
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def generate_data(num_rows: int = 50_000, output_path: Path | None = None) -> None:
    """Generate procedural CSV data."""
    if output_path is None:
        output_path = Path(__file__).parent / "input.csv"

    categories = ["A", "B", "C", "D", "E"]
    base_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

    print(f"Generating {num_rows:,} rows to {output_path}...")  # noqa: T201

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value", "category", "priority", "timestamp"])

        for i in range(1, num_rows + 1):
            row = [
                i,  # id
                round(random.uniform(0, 10000), 2),  # value
                random.choice(categories),  # category
                random.randint(1, 5),  # priority
                (base_timestamp + timedelta(seconds=i)).isoformat(),  # timestamp
            ]
            writer.writerow(row)

            # Progress indicator every 10k rows
            if i % 10_000 == 0:
                print(f"  {i:,} rows written...")  # noqa: T201

    print(f"✓ Generated {num_rows:,} rows successfully")  # noqa: T201


if __name__ == "__main__":
    num_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000

    if num_rows < 1 or num_rows > 1_000_000:
        print("Error: Row count must be between 1 and 1,000,000", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    generate_data(num_rows)
