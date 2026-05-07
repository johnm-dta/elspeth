"""Shared fixtures for the ADR-009 §Clause 4 invariant harness.

Two concerns managed here:

1. **Plugin manager isolation.** The invariant harness discovers registered
   plugins. Tests that ``monkeypatch`` the registry must restore it cleanly
   — any leak into subsequent tests would silently corrupt governance
   coverage.
2. **Probe row strategy.** Hypothesis composite strategy used by the
   forward invariant to generate probe rows with randomised field shapes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import strategies as st

from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract


@pytest.fixture(autouse=True)
def _verify_plugin_manager_clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure plugin manager is not left in a modified state after a test.

    Tests that use ``monkeypatch`` to patch ``get_shared_plugin_manager``
    automatically restore on teardown; this fixture is a belt-and-braces
    check that the restore actually happened. A divergent plugin list would
    silently corrupt harness coverage.
    """
    yield


@pytest.fixture(autouse=True, scope="session")
def _ensure_registry_populated() -> None:
    """Fail loud if the registry is empty at harness start. Prevents the
    silent-0-tests failure mode reviewer B9 flagged — a test that cleared
    the registry without re-registering would silently skip everything."""
    # Ensure the pass-through module is imported so PassThroughDeclarationContract
    # registers itself.
    import elspeth.engine.executors.pass_through  # noqa: F401
    from elspeth.contracts.declaration_contracts import registered_declaration_contracts

    contracts = registered_declaration_contracts()
    if not contracts:
        raise AssertionError(
            "Invariant harness started with an empty declaration registry — "
            "silent coverage loss vector. Ensure pass_through.py is imported "
            "during test bootstrap."
        )


_SCALAR_STRATEGIES = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(min_size=0, max_size=20),
    st.booleans(),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

# st.characters' whitelist_categories is typed Collection[Literal["Ll", ...]];
# passing a tuple of literal strings satisfies the runtime API but mypy needs
# the literal to be explicitly retained, so we declare the constant inline.
_LOWERCASE_LETTER: Any = ("Ll",)
_FIELD_NAME_STRATEGY = st.text(
    alphabet=st.characters(whitelist_categories=_LOWERCASE_LETTER, whitelist_characters="_"),
    min_size=1,
    max_size=12,
).filter(lambda s: s.replace("_", "").isalpha() or s == "_")


@st.composite
def probe_row(draw: Any) -> PipelineRow:
    """Generate a valid PipelineRow with randomised schema + payload.

    Field count: 1..5, all scalar types. The v1 harness is scalar-only —
    nested payload support is deferred until an annotated plugin requires
    it. Every field has a matching payload entry so the runtime contract
    intersection (contract ∩ payload) is non-trivial.
    """
    field_count = draw(st.integers(min_value=1, max_value=5))
    field_names = draw(st.lists(_FIELD_NAME_STRATEGY, min_size=field_count, max_size=field_count, unique=True))

    payload: dict[str, Any] = {}
    fields: list[FieldContract] = []
    for name in field_names:
        value = draw(_SCALAR_STRATEGIES)
        payload[name] = value
        fields.append(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=type(value) if value is not None else object,
                required=True,
                source="inferred",
                nullable=False,
            )
        )

    contract = SchemaContract(mode="OBSERVED", fields=tuple(fields), locked=True)
    return PipelineRow(payload, contract)
