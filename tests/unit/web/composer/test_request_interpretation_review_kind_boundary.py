"""Trust-boundary test for ``_request_interpretation_review_kind_from_arguments``.

Binds the ``@trust_boundary(tier=3, source_param="arguments", suppresses=("R5",))``
decorator on the helper in ``web/composer/service.py``. The function builds the
``InterpretationKind`` discriminator for an audit row from the LLM tool-call
payload (Tier 3); a non-string ``kind`` must be REJECTED with a typed
``AuditIntegrityError`` rather than coerced or written into a fabricated row.
This test exercises that rejection through the ``arguments`` source_param so
the trust-boundary suppression of the R5 ``isinstance`` guard is test-backed
(TBE1).
"""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.service import _request_interpretation_review_kind_from_arguments


def test_non_str_kind_raises_audit_integrity_error() -> None:
    """A non-string ``kind`` in the Tier-3 tool-call arguments is rejected."""
    with pytest.raises(AuditIntegrityError, match=r"invalid kind"):
        _request_interpretation_review_kind_from_arguments({"kind": 123})


def test_unknown_kind_member_raises_audit_integrity_error() -> None:
    """A string ``kind`` that is not a valid InterpretationKind member is rejected."""
    with pytest.raises(AuditIntegrityError, match=r"invalid kind"):
        _request_interpretation_review_kind_from_arguments({"kind": "not_a_real_kind"})
