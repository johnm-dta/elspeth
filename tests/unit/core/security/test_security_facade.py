"""Regression tests for the elspeth.core.security package facade."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import Any, cast


def _run_isolated_import_probe(code: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def test_fingerprint_facade_import_does_not_load_web_or_secret_loader_modules() -> None:
    result = _run_isolated_import_probe(
        """
        import json
        import sys

        from elspeth.core.security import secret_fingerprint

        print(json.dumps({
            "fingerprint_len": len(secret_fingerprint("value", key=b"k")),
            "web_loaded": "elspeth.core.security.web" in sys.modules,
            "secret_loader_loaded": "elspeth.core.security.secret_loader" in sys.modules,
            "config_secrets_loaded": "elspeth.core.security.config_secrets" in sys.modules,
        }))
        """
    )

    assert result == {
        "fingerprint_len": 64,
        "web_loaded": False,
        "secret_loader_loaded": False,
        "config_secrets_loaded": False,
    }


def test_legacy_web_facade_symbol_imports_lazily() -> None:
    result = _run_isolated_import_probe(
        """
        import json
        import sys

        from elspeth.core.security import SSRFBlockedError

        print(json.dumps({
            "name": SSRFBlockedError.__name__,
            "web_loaded": "elspeth.core.security.web" in sys.modules,
        }))
        """
    )

    assert result == {
        "name": "SSRFBlockedError",
        "web_loaded": True,
    }
