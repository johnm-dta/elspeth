# tests/contracts/test_leaf_boundary.py
"""Tests that contracts remains a leaf module with no core dependencies.

The contracts package should be importable without loading elspeth.core,
which pulls in heavy dependencies like pandas, numpy, sqlalchemy, networkx.
This enables lightweight CLI commands and test utilities.

BUG FIX: P2-2026-01-20-contracts-config-reexport-breaks-leaf-boundary
"""

import subprocess
import sys


class TestContractsLeafBoundary:
    """Verify contracts doesn't import core modules."""

    def test_contracts_enums_does_not_import_core(self) -> None:
        """Importing elspeth.contracts.enums should not load elspeth.core.

        This is a regression test for P2-2026-01-20. Before the fix,
        importing contracts.enums loaded 1,200+ modules including all of core.
        """
        # Run in subprocess to get clean import state
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
before = set(sys.modules.keys())
import elspeth.contracts.enums
after = set(sys.modules.keys())
new_modules = after - before

# Check that core modules are NOT loaded
core_modules = [m for m in new_modules if m.startswith('elspeth.core')]
if core_modules:
    print(f"FAIL: Core modules loaded: {core_modules}")
    sys.exit(1)
print(f"OK: {len(new_modules)} modules loaded, no core modules")
""",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Core modules were loaded:\n{result.stdout}\n{result.stderr}"

    def test_contracts_types_does_not_import_core(self) -> None:
        """Importing elspeth.contracts.types should not load elspeth.core."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
before = set(sys.modules.keys())
import elspeth.contracts.types
after = set(sys.modules.keys())
new_modules = after - before

core_modules = [m for m in new_modules if m.startswith('elspeth.core')]
if core_modules:
    print(f"FAIL: Core modules loaded: {core_modules}")
    sys.exit(1)
print(f"OK: {len(new_modules)} modules loaded, no core modules")
""",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Core modules were loaded:\n{result.stdout}\n{result.stderr}"

    def test_contracts_results_does_not_import_core(self) -> None:
        """Importing elspeth.contracts.results should not load elspeth.core."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
before = set(sys.modules.keys())
import elspeth.contracts.results
after = set(sys.modules.keys())
new_modules = after - before

core_modules = [m for m in new_modules if m.startswith('elspeth.core')]
if core_modules:
    print(f"FAIL: Core modules loaded: {core_modules}")
    sys.exit(1)
print(f"OK: {len(new_modules)} modules loaded, no core modules")
""",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Core modules were loaded:\n{result.stdout}\n{result.stderr}"

    def test_contracts_package_does_not_import_core(self) -> None:
        """Importing elspeth.contracts should not load elspeth.core.

        This is the main regression test - after the fix, the contracts
        package init should not pull in core.config or other core modules.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
before = set(sys.modules.keys())
import elspeth.contracts
after = set(sys.modules.keys())
new_modules = after - before

core_modules = [m for m in new_modules if m.startswith('elspeth.core')]
if core_modules:
    print(f"FAIL: Core modules loaded: {core_modules}")
    sys.exit(1)
print(f"OK: {len(new_modules)} modules loaded, no core modules")
""",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Core modules were loaded:\n{result.stdout}\n{result.stderr}"
