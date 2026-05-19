"""Tests for the Hypothesis strategy drift guard in ``conftest.py``.

The drift guard (``_assert_default_factory_dict_models_have_overrides``)
runs at conftest import time and crashes loudly when a MANIFEST model with
``Field(default_factory=dict)`` lacks an explicit
``st.register_type_strategy(...)`` override.

These tests exercise the guard's contract directly:

* **Negative path** (no drift): the real ``MANIFEST`` + the real
  ``_OVERRIDE_REGISTERED_MODELS`` must NOT raise.  This pins the
  invariant that the current state is consistent.
* **Positive path** (drift introduced): a synthetic manifest containing a
  fake Pydantic model with ``Field(default_factory=dict)`` and an empty
  ``registered`` tuple MUST raise ``RuntimeError`` whose message names the
  fake model.  This pins the guard's offensive behaviour.
* **Field-discovery path**: ``_models_needing_override`` reports the
  default-factory-dict fields of the synthetic model exactly.

F6 follow-up: see ``docs/composer/evidence/composer-phase-2-followup-prompt-F1-F6.md``.
Spec section: §4.2.6.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from elspeth.web.composer.redaction import MANIFEST, ToolRedaction

from .conftest import (
    _OVERRIDE_REGISTERED_MODELS,
    _assert_default_factory_dict_models_have_overrides,
    _models_needing_override,
)


class _FakeManifestModel(BaseModel):
    """Synthetic argument model used to provoke the drift guard.

    Mirrors the shape of a real composer-redaction argument model:
    string-keyed dict field with ``Field(default_factory=dict)``.  The model
    is intentionally module-level so the guard can address it by
    qualified-name in its error message.
    """

    name: str
    options: dict[str, object] = Field(default_factory=dict)


def test_drift_guard_passes_on_current_manifest() -> None:
    """Negative path: real MANIFEST + real registrations must NOT raise.

    Pins the invariant that the conftest is internally consistent.  If
    this fails, conftest import would also fail and the entire composer
    test slice would refuse to collect — surfacing the regression
    visibly rather than silently.
    """
    _assert_default_factory_dict_models_have_overrides(MANIFEST, _OVERRIDE_REGISTERED_MODELS)


def test_drift_guard_raises_when_manifest_model_lacks_override() -> None:
    """Positive path: a fake model with default_factory=dict + empty registered must raise."""
    synthetic_manifest = {"_fake_tool": ToolRedaction(argument_model=_FakeManifestModel)}
    with pytest.raises(RuntimeError) as exc_info:
        _assert_default_factory_dict_models_have_overrides(synthetic_manifest, ())
    message = str(exc_info.value)
    assert "_FakeManifestModel" in message, f"Drift-guard error message should name the offending model. Got: {message!r}"
    assert "options" in message, f"Drift-guard error message should name the offending field. Got: {message!r}"
    assert "tests/unit/web/composer/conftest.py" in message, (
        f"Drift-guard error message should point at the conftest for remediation. Got: {message!r}"
    )
    assert "F6" in message, f"Drift-guard error message should reference F6 follow-up. Got: {message!r}"


def test_drift_guard_silent_when_fake_model_is_registered() -> None:
    """Boundary: if the synthetic model IS in the registered tuple, the guard does not raise."""
    synthetic_manifest = {"_fake_tool": ToolRedaction(argument_model=_FakeManifestModel)}
    # No exception expected.
    _assert_default_factory_dict_models_have_overrides(synthetic_manifest, (_FakeManifestModel,))


def test_models_needing_override_identifies_default_factory_dict_fields() -> None:
    """Field-discovery: the helper reports exactly the default_factory=dict fields."""
    synthetic_manifest = {"_fake_tool": ToolRedaction(argument_model=_FakeManifestModel)}
    result = _models_needing_override(synthetic_manifest)
    assert _FakeManifestModel in result, f"Expected _FakeManifestModel in needs-override map. Got: {list(result)}"
    assert result[_FakeManifestModel] == ["options"], f"Expected ['options'] field list. Got: {result[_FakeManifestModel]!r}"


def test_models_needing_override_ignores_list_default_factory() -> None:
    """Scope: only ``default_factory is dict`` is flagged, not ``default_factory=list``.

    A model with ``Field(default_factory=list)`` does not trigger the
    Hypothesis sentinel-arm problem that motivates the override pattern,
    so the guard must not flag it.
    """

    class _ListFactoryModel(BaseModel):
        name: str
        items: list[str] = Field(default_factory=list)

    synthetic_manifest = {"_fake_tool": ToolRedaction(argument_model=_ListFactoryModel)}
    result = _models_needing_override(synthetic_manifest)
    assert _ListFactoryModel not in result, f"default_factory=list should NOT be flagged. Got: {list(result)}"


def test_models_needing_override_walks_nested_submodels() -> None:
    """Transitive: a nested Pydantic submodel with default_factory=dict is detected."""

    class _NestedDictModel(BaseModel):
        options: dict[str, object] = Field(default_factory=dict)

    class _ParentModel(BaseModel):
        name: str
        nested: _NestedDictModel | None = None

    synthetic_manifest = {"_fake_tool": ToolRedaction(argument_model=_ParentModel)}
    result = _models_needing_override(synthetic_manifest)
    assert _NestedDictModel in result, f"Drift guard must walk into nested submodels. Got: {list(result)}"
    # _ParentModel itself has no default_factory=dict field, so it must
    # NOT appear in the result — only the offending nested model.
    assert _ParentModel not in result, f"_ParentModel has no default_factory=dict; it must not be flagged. Got: {list(result)}"
