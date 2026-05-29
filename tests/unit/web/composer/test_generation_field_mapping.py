"""Trust-boundary honesty test for ``_csv_source_field_mapping``.

Binds the malformed-input invariant claimed by the ``@trust_boundary(tier=3,
source_param='options', suppresses=('R5',))`` decorator on
``elspeth.web.composer.tools.generation._csv_source_field_mapping``: the CSV
source ``field_mapping`` arrives inside external / LLM-authored composer source
options (Tier-3, zero-trust). A non-``Mapping`` ``field_mapping`` must be
rejected with ``ValueError`` (offensive validation at the boundary) and never
silently coerced. The companion ``enforce_trust_boundary_honesty`` gate requires
this raising-shape test to exist and to invoke the decorated function through
``source_param``.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.tools.generation import _csv_source_field_mapping


def test_csv_source_field_mapping_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _csv_source_field_mapping(options={"field_mapping": ["not", "a", "mapping"]})
