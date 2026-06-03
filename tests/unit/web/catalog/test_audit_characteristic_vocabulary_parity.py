"""Python↔TypeScript vocabulary parity for the AuditCharacteristic enum.

Closes the parity loop for `audit_characteristics`, which the catalog API
emits as `string[]` (see `src/elspeth/web/frontend/src/types/index.ts`).
The TypeScript metadata table at
`src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.ts`
must enumerate one entry per `AuditCharacteristic` enum member; otherwise
a new Python member silently renders as a grey "unknown" chip in the UI.

Python is the source of truth. When you add a new member to
``AuditCharacteristic``, add a matching entry to ``AUDIT_CHARACTERISTICS``
in the TS file (and a renderer for any new tone) in the same commit.
"""

from __future__ import annotations

import re
from pathlib import Path

import elspeth
from elspeth.contracts.enums import AuditCharacteristic

_PACKAGE_ROOT = Path(elspeth.__file__).parent
_TS_METADATA_PATH = _PACKAGE_ROOT / "web" / "frontend" / "src" / "components" / "catalog" / "auditCharacteristics.ts"

# The TS metadata array uses the canonical record form `flag: "<value>",`
# (Prettier-stable). This regex matches that exact shape and ignores any
# other quoted string in the file (tooltips, labels, comments).
_FLAG_RECORD_RE = re.compile(r'^\s*flag:\s*"([^"]+)"\s*,\s*$', re.MULTILINE)


def _ts_flag_vocabulary() -> set[str]:
    """Parse the TS metadata file and return the set of declared flag values."""
    text = _TS_METADATA_PATH.read_text(encoding="utf-8")
    return set(_FLAG_RECORD_RE.findall(text))


def test_audit_characteristic_python_matches_ts_metadata() -> None:
    py_flags = {member.value for member in AuditCharacteristic}
    ts_flags = _ts_flag_vocabulary()

    missing_in_ts = py_flags - ts_flags
    missing_in_py = ts_flags - py_flags

    assert not missing_in_ts, (
        "AuditCharacteristic members present in Python but missing from "
        f"{_TS_METADATA_PATH.name}: {sorted(missing_in_ts)}. "
        "Add a matching entry to AUDIT_CHARACTERISTICS in that file."
    )
    assert not missing_in_py, (
        f"Flags present in {_TS_METADATA_PATH.name} but absent from "
        f"AuditCharacteristic: {sorted(missing_in_py)}. "
        "Add a matching member to AuditCharacteristic in "
        "src/elspeth/contracts/enums.py."
    )


def test_ts_metadata_file_is_readable() -> None:
    """Smoke-test: anchor path resolves to a real file with at least one record.

    Guards against the parity test silently passing because the regex matched
    nothing (e.g. file moved, Prettier rewrote the record shape).
    """
    assert _TS_METADATA_PATH.is_file(), f"Expected TS metadata at {_TS_METADATA_PATH} — anchor path is wrong."
    assert _ts_flag_vocabulary(), f'No `flag: "..."` records matched in {_TS_METADATA_PATH}. The regex or the file format has drifted.'
