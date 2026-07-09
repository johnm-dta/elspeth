"""Input-free rendering of Pydantic validation failures (elspeth-a300402c58).

``str(ValidationError)`` echoes the offending INPUT VALUE by default. Source
plugins sit on the Tier-3 boundary and their quarantine error text lands
verbatim in ``node_states.error_json``, the DIVERT routing reason, and audit
exports — surfaces the input-data hashing discipline deliberately keeps raw
payloads out of. Render loc/msg/type only; the full raw row still travels on
``SourceRow.row`` to the designated quarantine sink by design.
"""

from __future__ import annotations

from pydantic import ValidationError


def safe_validation_error_text(exc: ValidationError) -> str:
    """Render a ``ValidationError`` without echoing input values.

    Keeps the triage-relevant parts of each error — field location, human
    message, and error type code — and drops the ``input`` echo that
    ``str(exc)`` would include.
    """
    details = exc.errors(include_input=False, include_context=False, include_url=False)
    parts = []
    for detail in details:
        loc = ".".join(str(item) for item in detail["loc"]) or "<root>"
        parts.append(f"{loc}: {detail['msg']} [{detail['type']}]")
    noun = "error" if len(details) == 1 else "errors"
    return f"{len(details)} validation {noun}: " + "; ".join(parts)
