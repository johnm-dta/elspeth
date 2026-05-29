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

from types import MappingProxyType

import pytest

from elspeth.web.composer.tools.generation import _csv_source_field_mapping


def test_csv_source_field_mapping_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _csv_source_field_mapping(options={"field_mapping": ["not", "a", "mapping"]})


def test_csv_source_field_mapping_accepts_frozen_mappingproxy() -> None:
    # Regression guard for the lint-dodge bug: CompositionState options are
    # deep-frozen, so field_mapping arrives as a MappingProxyType. A
    # `type(raw) is dict` check wrongly REJECTS this (raises); the correct
    # isinstance(raw, Mapping) guard must ACCEPT it. A list-only test would not
    # catch this — both the buggy and fixed checks reject a list.
    result = _csv_source_field_mapping(options={"field_mapping": MappingProxyType({"external_id": "customer_id"})})
    assert result == {"external_id": "customer_id"}
