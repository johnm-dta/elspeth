#!/usr/bin/env python3
"""Ratchet the public method count of PluginAuditWriterAdapter.

Prevents the adapter from growing back into a facade. This is a ratchet,
not a budget: the ceiling tracks the current method count exactly. Growth
fails; slack also fails, with instructions to lower the ratchet so every
reduction is locked in.
Run as: python scripts/cicd/enforce_adapter_budget.py
"""

from __future__ import annotations

import inspect
import sys

from elspeth.core.landscape.plugin_audit_writer import PluginAuditWriterAdapter

RATCHET = 13


def main() -> int:
    public_methods = [
        name for name, method in inspect.getmembers(PluginAuditWriterAdapter, predicate=inspect.isfunction) if not name.startswith("_")
    ]

    count = len(public_methods)
    if count > RATCHET:
        print(f"FAIL: PluginAuditWriterAdapter has {count} public methods (ratchet: {RATCHET})")
        print(f"Methods: {', '.join(sorted(public_methods))}")
        print("\nIf a new method is genuinely needed, consider whether the caller")
        print("should inject the specific repository directly instead.")
        return 1

    if count < RATCHET:
        print(f"FAIL: ratchet has slack — PluginAuditWriterAdapter has {count} public methods but the ratchet allows {RATCHET}.")
        print(f"Lower RATCHET to {count} in scripts/cicd/enforce_adapter_budget.py to lock the reduction in.")
        return 1

    print(f"OK: PluginAuditWriterAdapter has exactly {count} public methods (ratchet tight)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
