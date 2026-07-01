"""Shared single-file tier_model scan helper.

Promoted out of ``core/cli.py`` so every consumer -- ``cli.py`` (``justify`` and
``migrate-judge-scope``), ``bundle_verify.py``, and the ``elspeth-judge`` MCP
server -- runs the *one* scan call site and cannot drift on the scanner set.

This is a pure scan utility with no signing/HMAC state, which is why it is safe
to share; the security-critical signing helpers stay co-located in ``cli.py``.
The canonical scanner pair is bound here (imported from the tier_model
``rule`` module), so no call site supplies its own -- the same
``scan_file``/``scan_layer_imports_file`` family ``scan_for_rotations`` uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def scan_single_file_findings(*, target_file: Path, root: Path) -> list[Any]:
    """Re-run both tier_model scanners against a single file.

    Merges the R1-R7 findings from ``scan_file`` with the layer-import
    violations + TC warnings from ``scan_layer_imports_file``. Mirrors the way
    ``scan_for_rotations`` combines them, so a downstream symbol-match pass sees
    the same finding set the CI run would see.
    """
    from elspeth_lints.rules.trust_tier.tier_model.rule import (
        scan_file,
        scan_layer_imports_file,
    )

    findings: list[Any] = list(scan_file(target_file, root))
    layer_violations, layer_tc = scan_layer_imports_file(target_file, root)
    findings.extend(layer_violations)
    findings.extend(layer_tc)
    return findings
