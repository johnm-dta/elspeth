"""Drift guard for the authoritative declaration-contract bootstrap module.

Every production executor module that calls ``register_declaration_contract(...)``
at module scope must appear as a direct import in
``engine/executors/declaration_contract_bootstrap.py``. This catches the
"contract exists, manifest updated, but bootstrap import forgotten" failure mode
before ``prepare_for_run()`` fails in production.
"""

from __future__ import annotations

import ast
from pathlib import Path

_EXECUTORS_DIR = Path(__file__).resolve().parents[3] / "src" / "elspeth" / "engine" / "executors"
_BOOTSTRAP_PATH = _EXECUTORS_DIR / "declaration_contract_bootstrap.py"


def _modules_with_registration_calls() -> set[str]:
    modules: set[str] = set()
    for path in _EXECUTORS_DIR.glob("*.py"):
        if path.name == "__init__.py" or path.name == _BOOTSTRAP_PATH.name:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "register_declaration_contract":
                modules.add(path.stem)
        # One module-level registration call is enough to require bootstrap import.
    return modules


def _bootstrap_imported_modules() -> set[str]:
    tree = ast.parse(_BOOTSTRAP_PATH.read_text(), filename=str(_BOOTSTRAP_PATH))
    modules: set[str] = set()
    prefix = "elspeth.engine.executors."
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    modules.add(alias.name.removeprefix(prefix))
    return modules


def test_every_registration_module_has_matching_bootstrap_import() -> None:
    registered_modules = _modules_with_registration_calls()
    imported_modules = _bootstrap_imported_modules()
    missing = registered_modules - imported_modules
    extra = imported_modules - registered_modules

    assert not missing, (
        "declaration_contract_bootstrap.py is missing imports for executor "
        f"module(s) with register_declaration_contract(...) call sites: {sorted(missing)!r}"
    )
    assert not extra, (
        f"declaration_contract_bootstrap.py imports executor module(s) that no longer register declaration contracts: {sorted(extra)!r}"
    )
