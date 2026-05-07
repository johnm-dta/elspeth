#!/usr/bin/env python3
"""Enforce method-count budget on _PluginAuditWriterAdapter.

Prevents the adapter from growing back into a facade. The budget is 20 methods.
Run as: python scripts/cicd/enforce_adapter_budget.py
"""

from __future__ import annotations

import inspect
import sys

from elspeth.core.landscape.factory import _PluginAuditWriterAdapter

BUDGET = 20


def main() -> int:
    public_methods = [
        name for name, method in inspect.getmembers(_PluginAuditWriterAdapter, predicate=inspect.isfunction) if not name.startswith("_")
    ]

    count = len(public_methods)
    if count > BUDGET:
        print(f"FAIL: _PluginAuditWriterAdapter has {count} public methods (budget: {BUDGET})")
        print(f"Methods: {', '.join(sorted(public_methods))}")
        print("\nIf a new method is genuinely needed, consider whether the caller")
        print("should inject the specific repository directly instead.")
        return 1

    print(f"OK: _PluginAuditWriterAdapter has {count}/{BUDGET} public methods")
    return 0


if __name__ == "__main__":
    sys.exit(main())
