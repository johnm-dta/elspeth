"""Shared, empty-safe computation of the 16-char audit ``error_hash``.

``error_hash`` is a content fingerprint of a row's originating error, used for
per-row audit attribution. Hashing an empty message collapses every
empty-message error into the constant ``sha256("")`` prefix
(``e3b0c44298fc1c14``), so distinct empty-message failures become
indistinguishable in the audit trail — defeating the attributability guarantee
(elspeth-501c14847b). An empty message is a legitimate input here
(``str(ValueError()) == ""``), so the fix substitutes a type-qualified marker
for empty input rather than rejecting it on the failure-recording path.

Crucially this changes ONLY the empty case: for any non-empty message the result
is byte-identical to the previous inline ``sha256(msg.encode()).hexdigest()[:16]``,
so no existing audit hash changes and no fingerprint-baseline reconciliation is
needed.
"""

from __future__ import annotations

import hashlib


def compute_error_hash(message: str, *, exception_type: str | None = None) -> str:
    """Return the 16-char sha256 prefix of an error message, empty-safe.

    For a non-empty ``message`` the result equals
    ``sha256(message.encode()).hexdigest()[:16]`` exactly. For an empty
    ``message`` a marker is hashed instead — qualified by ``exception_type`` when
    available — so empty-message errors remain distinguishable by type instead of
    all colliding into the constant ``sha256("")`` prefix.
    """
    if not message:
        message = f"<no-message:{exception_type}>" if exception_type else "<no-message>"
    return hashlib.sha256(message.encode()).hexdigest()[:16]
