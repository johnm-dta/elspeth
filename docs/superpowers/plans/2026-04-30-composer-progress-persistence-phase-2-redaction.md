# Composer Progress Persistence — Phase 2: Redaction Framework

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `Sensitive[T]` type-driven redaction as the primary primitive (Meadows Level-4 leverage), retain `ToolRedactionPolicy` only as a legacy escape valve gated behind an explicit ClassVar opt-out, and add a recursive CI-time adequacy guard plus a policy-hash snapshot to detect weakening.

**Architecture:** Pure redaction-layer work. Phase 1 schema is in place; this phase does not modify the database, the compose loop, or the frontend. The redaction layer is L3, alongside the composer tools.

**Tech Stack:** Python 3.13, Pydantic 2.x (`Annotated`, `model_fields`), pytest, hashlib for snapshot.

**Spec sections:** §3 ADR row "Redaction primitive (R4)", §4.2 (ToolRedactionPolicy + Sensitive[T] primitives), §4.3 (sentinel rules), §4.4 (recursive adequacy guard), §4.7 (initial policy declarations), §11 Phase 2 scope.

---

## File Structure

### Files to create

- `src/elspeth/web/composer/redaction.py` — extend with `Sensitive[T]` marker, `_SensitiveMarker` class, `ToolRedactionPolicy` (rev-4 shape), `HandlesNoSensitiveDataReason`, `apply_redaction_policy`, `apply_response_redaction`, `redact_tool_call`, `lookup_tool_class` integration.
- `src/elspeth/web/composer/_tool_base.py` — shared `ComposerTool` base class with the `EXEMPT_FROM_*` ClassVars. (If a base class already exists, modify it.)
- `tests/unit/web/composer/test_sensitive_marker.py` — primitive tests.
- `tests/unit/web/composer/test_redaction_policy.py` — replace existing rev-3 file; structured-reason validators, recursive adequacy, snapshot test.
- `tests/unit/web/composer/redaction_policy_snapshot.json` — committed snapshot.
- `.github/CODEOWNERS` — add or extend with the redaction routes.

### Files to modify

- `src/elspeth/web/composer/tools.py` — for every existing tool, either annotate argument/response model fields with `Sensitive[T]` OR set `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True` and supply a legacy `ToolRedactionPolicy` plus `HandlesNoSensitiveDataReason`.
- `src/elspeth/web/composer/redaction.py` — extend if it already exists; otherwise create. The existing `redact_source_storage_path` helper stays in this file unchanged.

### Files NOT touched in Phase 2

- `src/elspeth/web/composer/service.py` — Phase 3 wires the loop.
- Anything under `src/elspeth/web/sessions/` — Phase 1 owns the schema.
- Anything under `src/elspeth/web/frontend/` — Phase 4.

---

## Task 1: `_SensitiveMarker` and `Sensitive[T]` constructor

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py`
- Create: `tests/unit/web/composer/test_sensitive_marker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/composer/test_sensitive_marker.py`:

```python
"""Tests for the Sensitive[T] type-driven redaction primitive (spec §4.2.1)."""
from __future__ import annotations

from typing import Annotated, get_args, get_origin

from pydantic import BaseModel

from elspeth.web.composer.redaction import Sensitive, _SensitiveMarker


def test_sensitive_returns_marker_instance():
    marker = Sensitive()
    assert isinstance(marker, _SensitiveMarker)
    assert marker.summarizer is None


def test_sensitive_with_summarizer():
    summarizer = lambda v: f"<sum:{len(v)}>"
    marker = Sensitive(summarizer=summarizer)
    assert marker.summarizer is summarizer


def test_sensitive_marker_visible_in_pydantic_model_fields():
    """The marker must round-trip through Pydantic's metadata system so the
    persistence layer can read it via model_fields[name].metadata."""

    class M(BaseModel):
        ok: str
        secret: Annotated[str, Sensitive()]

    secret_field = M.model_fields["secret"]
    markers = [m for m in secret_field.metadata if isinstance(m, _SensitiveMarker)]
    assert len(markers) == 1
    ok_field = M.model_fields["ok"]
    assert not any(isinstance(m, _SensitiveMarker) for m in ok_field.metadata)


def test_sensitive_marker_summarizer_identity_preserved():
    summarizer = lambda v: f"<sum:{len(v)}>"

    class M(BaseModel):
        secret: Annotated[bytes, Sensitive(summarizer=summarizer)]

    secret_field = M.model_fields["secret"]
    marker = next(m for m in secret_field.metadata if isinstance(m, _SensitiveMarker))
    assert marker.summarizer is summarizer
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_sensitive_marker.py -v
```
Expected: FAIL — `Sensitive` and `_SensitiveMarker` are not defined.

- [ ] **Step 3: Add `Sensitive` and `_SensitiveMarker` to redaction.py**

In `src/elspeth/web/composer/redaction.py`:

```python
from collections.abc import Callable
from typing import Any


class _SensitiveMarker:
    """Annotated metadata marker indicating a Pydantic field MUST be
    redacted at the persistence boundary. Spec §4.2.1.

    summarizer: optional per-field replacement function. If None, the
        sentinel '<redacted>' is substituted at persistence time. If
        present, the function receives the original value and returns
        the replacement string.
    """

    __slots__ = ("summarizer",)

    def __init__(self, summarizer: Callable[[Any], str] | None = None) -> None:
        self.summarizer = summarizer


def Sensitive(  # noqa: N802 — capitalised to read as a type alias at use sites
    *, summarizer: Callable[[Any], str] | None = None
) -> _SensitiveMarker:
    """Field-level redaction marker. Use as Pydantic field metadata via
    Annotated:

        class SetSourceArguments(BaseModel):
            path: Annotated[str, Sensitive(summarizer=redact_source_storage_path)]
            options: Annotated[dict, Sensitive()]
            label: str   # not sensitive, persisted verbatim
    """
    return _SensitiveMarker(summarizer=summarizer)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_sensitive_marker.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_sensitive_marker.py
git commit -m "feat(composer): add Sensitive[T] type-driven redaction marker (composer-progress-persistence phase 2)"
```

---

## Task 2: Recursive redaction layer

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py`
- Test: `tests/unit/web/composer/test_sensitive_marker.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/web/composer/test_sensitive_marker.py`:

```python
def test_apply_sensitive_redaction_replaces_marked_field():
    from elspeth.web.composer.redaction import apply_sensitive_redaction

    class M(BaseModel):
        public_id: str
        secret: Annotated[str, Sensitive()]

    raw = M(public_id="abc", secret="hunter2")
    redacted = apply_sensitive_redaction(raw)
    assert redacted == {"public_id": "abc", "secret": "<redacted>"}


def test_apply_sensitive_redaction_uses_summarizer():
    from elspeth.web.composer.redaction import apply_sensitive_redaction

    class M(BaseModel):
        blob: Annotated[bytes, Sensitive(summarizer=lambda b: f"<inline-blob:{len(b)}-bytes>")]

    raw = M(blob=b"A" * 1024)
    redacted = apply_sensitive_redaction(raw)
    assert redacted == {"blob": "<inline-blob:1024-bytes>"}


def test_apply_sensitive_redaction_recurses_into_nested_model():
    from elspeth.web.composer.redaction import apply_sensitive_redaction

    class Inner(BaseModel):
        api_key: Annotated[str, Sensitive()]
        public: str

    class Outer(BaseModel):
        inner: Inner
        label: str

    raw = Outer(inner=Inner(api_key="topsecret", public="ok"), label="hello")
    redacted = apply_sensitive_redaction(raw)
    assert redacted == {
        "inner": {"api_key": "<redacted>", "public": "ok"},
        "label": "hello",
    }


def test_apply_sensitive_redaction_summarizer_exception_falls_back_to_class_sentinel():
    """Spec §8.1 / RSK-03: a summarizer that raises must NOT propagate; the
    persistence boundary substitutes `<redacted-summarizer-error:{exc_type}>`
    using the exception class only (no message echo per security I-3)."""

    def boom(_):
        raise ValueError("AKIA-LEAK-IN-MESSAGE")

    class M(BaseModel):
        secret: Annotated[str, Sensitive(summarizer=boom)]

    raw = M(secret="real-secret-value")
    redacted = apply_sensitive_redaction(raw)
    assert redacted == {"secret": "<redacted-summarizer-error:ValueError>"}
    assert "AKIA-LEAK-IN-MESSAGE" not in repr(redacted)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_sensitive_marker.py -v
```
Expected: FAIL — `apply_sensitive_redaction` not implemented.

- [ ] **Step 3: Implement the recursive redaction layer**

In `src/elspeth/web/composer/redaction.py`:

```python
from typing import Any
from pydantic import BaseModel


def apply_sensitive_redaction(model: BaseModel, *, telemetry=None) -> dict[str, Any]:
    """Walk a Pydantic model and produce a dict with marked fields redacted.

    For each field:
      - If field metadata contains _SensitiveMarker:
          - With summarizer: substitute summarizer(value); on exception,
            substitute `<redacted-summarizer-error:{exc_class}>` and
            increment telemetry.summarizer_errors_total if available.
          - Without summarizer: substitute `<redacted>`.
      - If field type is a nested BaseModel: recurse.
      - Otherwise: pass through.

    Spec §4.2.1, §8.1 fallback contract.
    """
    result: dict[str, Any] = {}
    for name, field_info in type(model).model_fields.items():
        value = getattr(model, name)
        marker = next(
            (m for m in field_info.metadata if isinstance(m, _SensitiveMarker)),
            None,
        )
        if marker is not None:
            if marker.summarizer is None:
                result[name] = "<redacted>"
            else:
                try:
                    result[name] = marker.summarizer(value)
                except Exception as exc:  # noqa: BLE001 — boundary: spec §8.1
                    if telemetry is not None:
                        telemetry.summarizer_errors_total.add(1)
                    result[name] = f"<redacted-summarizer-error:{type(exc).__name__}>"
        elif isinstance(value, BaseModel):
            result[name] = apply_sensitive_redaction(value, telemetry=telemetry)
        else:
            result[name] = value
    return result
```

(Direct attribute access via `getattr(model, name)` here is acceptable: Pydantic's `model_fields` is the source of truth for which attributes exist on the model, so this is structural lookup against an enumerated set, not defensive `getattr` on an unknown surface. CLAUDE.md's `getattr` ban targets defensive attribute access, not structured iteration over a known schema.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_sensitive_marker.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_sensitive_marker.py
git commit -m "feat(composer): add recursive apply_sensitive_redaction (composer-progress-persistence phase 2)"
```

---

## Task 3: `HandlesNoSensitiveDataReason` structured dataclass

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py`
- Create: `tests/unit/web/composer/test_redaction_policy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/composer/test_redaction_policy.py`:

```python
"""Tests for the rev-4 ToolRedactionPolicy and HandlesNoSensitiveDataReason."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from elspeth.web.composer.redaction import (
    HandlesNoSensitiveDataReason,
    ToolRedactionPolicy,
)


def test_reason_requires_locations():
    with pytest.raises(ValueError, match="sensitive_data_locations is empty"):
        HandlesNoSensitiveDataReason(
            sensitive_data_locations=(),
            why_arguments_safe="x" * 40,
            why_responses_safe="y" * 40,
            last_reviewed_iso=date.today(),
        )


def test_reason_requires_minimum_length():
    with pytest.raises(ValueError, match="why_arguments_safe is shorter than 32"):
        HandlesNoSensitiveDataReason(
            sensitive_data_locations=("server-side resolver",),
            why_arguments_safe="too short",
            why_responses_safe="y" * 40,
            last_reviewed_iso=date.today(),
        )


def test_reason_accepts_valid():
    reason = HandlesNoSensitiveDataReason(
        sensitive_data_locations=("server-side resolver",),
        why_arguments_safe="Arguments contain only secret names, never values; resolver is server-side.",
        why_responses_safe="Responses confirm the secret was wired without echoing material.",
        last_reviewed_iso=date.today(),
    )
    assert reason.last_reviewed_iso == date.today()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v
```
Expected: FAIL — `HandlesNoSensitiveDataReason` not defined.

- [ ] **Step 3: Implement the dataclass**

In `src/elspeth/web/composer/redaction.py`:

```python
from dataclasses import dataclass
from datetime import date

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class HandlesNoSensitiveDataReason:
    """Structured justification required for handles_no_sensitive_data=True
    in a legacy ToolRedactionPolicy. Spec §4.2.2.

    sensitive_data_locations: where sensitive material actually lives if
        not in this tool's arguments or responses (e.g., 'server-side
        secret resolver', 'request headers stripped before tool dispatch').
        Must be non-empty.

    why_arguments_safe: prose explanation of why every argument is safe to
        persist verbatim. Adequacy guard rejects values shorter than 32 chars.

    why_responses_safe: same rule for responses.

    last_reviewed_iso: ISO-8601 date when this justification was last
        reviewed. Adequacy-guard fails if older than 365 days at test time.
    """

    sensitive_data_locations: tuple[str, ...]
    why_arguments_safe: str
    why_responses_safe: str
    last_reviewed_iso: date

    def __post_init__(self) -> None:
        if not self.sensitive_data_locations:
            raise ValueError(
                "sensitive_data_locations is empty; declare at least one location "
                "where sensitive material related to this tool exists, OR migrate "
                "the tool's arguments/responses to use Sensitive[T] annotations."
            )
        for label, value in (
            ("why_arguments_safe", self.why_arguments_safe),
            ("why_responses_safe", self.why_responses_safe),
        ):
            if len(value.strip()) < 32:
                raise ValueError(
                    f"{label} is shorter than 32 characters; the structured "
                    f"justification requires concrete reasoning, not a placeholder."
                )
        freeze_fields(self, "sensitive_data_locations")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_redaction_policy.py
git commit -m "feat(composer): add HandlesNoSensitiveDataReason structured justification (composer-progress-persistence phase 2)"
```

---

## Task 4: `ToolRedactionPolicy` rev-4 shape with `known_response_keys`

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py`
- Test: `tests/unit/web/composer/test_redaction_policy.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/web/composer/test_redaction_policy.py`:

```python
def test_policy_orphan_summarizer_rejected():
    with pytest.raises(ValueError, match="orphan summarizers"):
        ToolRedactionPolicy(
            sensitive_argument_keys=("path",),
            argument_summarizers={"not_in_keys": lambda v: "<x>"},
            known_response_keys=("ok",),
        )


def test_policy_handles_true_requires_reason_struct():
    with pytest.raises(ValueError, match="non-None handles_no_sensitive_data_reason_struct"):
        ToolRedactionPolicy(
            handles_no_sensitive_data=True,
            handles_no_sensitive_data_reason_struct=None,
        )


def test_policy_handles_false_with_reason_rejected():
    reason = HandlesNoSensitiveDataReason(
        sensitive_data_locations=("x",),
        why_arguments_safe="x" * 40,
        why_responses_safe="x" * 40,
        last_reviewed_iso=date.today(),
    )
    with pytest.raises(ValueError, match="only meaningful when handles_no_sensitive_data=True"):
        ToolRedactionPolicy(
            handles_no_sensitive_data=False,
            handles_no_sensitive_data_reason_struct=reason,
            known_response_keys=("ok",),
        )


def test_policy_handles_false_requires_known_response_keys():
    with pytest.raises(ValueError, match="known_response_keys must be declared"):
        ToolRedactionPolicy(handles_no_sensitive_data=False)


def test_policy_handles_true_does_not_require_known_response_keys():
    reason = HandlesNoSensitiveDataReason(
        sensitive_data_locations=("server-side resolver",),
        why_arguments_safe="Arguments are inventory metadata only; values resolve server-side.",
        why_responses_safe="Responses confirm wire-up; never echo material.",
        last_reviewed_iso=date.today(),
    )
    policy = ToolRedactionPolicy(
        handles_no_sensitive_data=True,
        handles_no_sensitive_data_reason_struct=reason,
    )
    assert policy.known_response_keys == ()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v
```
Expected: FAIL — `ToolRedactionPolicy` not yet defined.

- [ ] **Step 3: Implement the dataclass**

In `src/elspeth/web/composer/redaction.py`:

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolRedactionPolicy:
    """Legacy declarative redaction policy. Use Sensitive[T] annotations
    instead when possible. Consulted only when EXEMPT_FROM_TYPE_DRIVEN_REDACTION
    is True on the tool class. Spec §4.2.2.
    """

    sensitive_argument_keys: tuple[str, ...] = ()
    sensitive_response_keys: tuple[str, ...] = ()
    known_response_keys: tuple[str, ...] = ()
    argument_summarizers: Mapping[str, Callable[[Any], str]] = field(default_factory=dict)
    handles_no_sensitive_data: bool = False
    handles_no_sensitive_data_reason_struct: HandlesNoSensitiveDataReason | None = None

    def __post_init__(self) -> None:
        orphan_summarizers = set(self.argument_summarizers) - set(self.sensitive_argument_keys)
        if orphan_summarizers:
            raise ValueError(
                f"argument_summarizers keys {sorted(orphan_summarizers)} are not declared in "
                f"sensitive_argument_keys; orphan summarizers indicate a policy bug."
            )

        if self.handles_no_sensitive_data and self.handles_no_sensitive_data_reason_struct is None:
            raise ValueError(
                "handles_no_sensitive_data=True requires a non-None "
                "handles_no_sensitive_data_reason_struct."
            )
        if not self.handles_no_sensitive_data and self.handles_no_sensitive_data_reason_struct is not None:
            raise ValueError(
                "handles_no_sensitive_data_reason_struct is only meaningful "
                "when handles_no_sensitive_data=True."
            )

        if not self.handles_no_sensitive_data and not self.known_response_keys:
            raise ValueError(
                "known_response_keys must be declared (non-empty) when "
                "handles_no_sensitive_data=False."
            )

        freeze_fields(
            self,
            "sensitive_argument_keys",
            "sensitive_response_keys",
            "known_response_keys",
            "argument_summarizers",
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_redaction_policy.py
git commit -m "feat(composer): rev-4 ToolRedactionPolicy with known_response_keys (composer-progress-persistence phase 2)"
```

---

## Task 5: `apply_response_redaction` with unknown-key fail-closed

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py`
- Test: `tests/unit/web/composer/test_redaction_policy.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_apply_response_redaction_known_keys_passthrough():
    from elspeth.web.composer.redaction import apply_response_redaction
    policy = ToolRedactionPolicy(known_response_keys=("ok", "value"))
    redacted = apply_response_redaction({"ok": True, "value": 42}, policy=policy, telemetry=None)
    assert redacted == {"ok": True, "value": 42}


def test_apply_response_redaction_unknown_key_fail_closed():
    from elspeth.web.composer.redaction import apply_response_redaction

    class FakeTelem:
        def __init__(self):
            self.unknown_response_key_total = type("C", (), {"add": lambda self_, n: setattr(self_, "v", getattr(self_, "v", 0) + n)})()
            self.unknown_response_key_total.v = 0

    telem = FakeTelem()
    policy = ToolRedactionPolicy(known_response_keys=("ok",))
    redacted = apply_response_redaction({"ok": True, "leaked": "AKIA-X"}, policy=policy, telemetry=telem)
    assert redacted["ok"] is True
    assert redacted["leaked"].startswith("<redacted-unknown-key:")
    assert "AKIA-X" not in redacted["leaked"]
    assert telem.unknown_response_key_total.v == 1


def test_apply_response_redaction_sensitive_key_substituted():
    from elspeth.web.composer.redaction import apply_response_redaction
    policy = ToolRedactionPolicy(
        sensitive_response_keys=("api_key",),
        known_response_keys=("api_key", "ok"),
    )
    redacted = apply_response_redaction(
        {"api_key": "secret", "ok": True},
        policy=policy, telemetry=None,
    )
    assert redacted == {"api_key": "<redacted>", "ok": True}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement `apply_response_redaction`**

```python
def apply_response_redaction(
    response: Mapping[str, Any],
    *,
    policy: ToolRedactionPolicy,
    telemetry,
) -> dict[str, Any]:
    """Apply a legacy ToolRedactionPolicy to a tool response dict.

    Behaviour:
      - Keys listed in policy.sensitive_response_keys are replaced by '<redacted>'.
      - Keys NOT in policy.known_response_keys (if known_response_keys is non-empty)
        are fail-closed redacted with `<redacted-unknown-key:{n}-bytes>`.
      - All other keys pass through.

    Telemetry:
      - Increments telemetry.unknown_response_key_total per unknown key encountered.
    """
    result: dict[str, Any] = {}
    sensitive = set(policy.sensitive_response_keys)
    known = set(policy.known_response_keys)

    for key, value in response.items():
        if key in sensitive:
            result[key] = "<redacted>"
        elif known and key not in known:
            length = len(repr(value))
            if telemetry is not None:
                telemetry.unknown_response_key_total.add(1)
            result[key] = f"<redacted-unknown-key:{length}-bytes>"
        else:
            result[key] = value
    return result
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_redaction_policy.py
git commit -m "feat(composer): apply_response_redaction with unknown-key fail-closed (composer-progress-persistence phase 2)"
```

---

## Task 6: `ComposerTool` base class with `EXEMPT_FROM_*` ClassVars

**Files:**
- Create: `src/elspeth/web/composer/_tool_base.py`  (or modify the existing tool base if one exists; locate via `grep -n "class ComposerTool" src/elspeth/web/composer/`)
- Test: `tests/unit/web/composer/test_redaction_policy.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_composer_tool_base_class_has_exempt_classvars():
    from elspeth.web.composer._tool_base import ComposerTool
    assert ComposerTool.EXEMPT_FROM_ADEQUACY_CHECK is False
    assert ComposerTool.EXEMPT_FROM_TYPE_DRIVEN_REDACTION is False


def test_subclass_can_override():
    from elspeth.web.composer._tool_base import ComposerTool

    class FakeTool(ComposerTool):
        EXEMPT_FROM_TYPE_DRIVEN_REDACTION = True

    assert FakeTool.EXEMPT_FROM_TYPE_DRIVEN_REDACTION is True
    assert FakeTool.EXEMPT_FROM_ADEQUACY_CHECK is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k composer_tool_base
```
Expected: FAIL.

- [ ] **Step 3: Locate or create the base class**

```bash
grep -rn "class.*ComposerTool\b\|class.*Tool\b.*:$" src/elspeth/web/composer/ | head -20
```

If a base class already exists for composer tools, modify it to add the ClassVars. Otherwise create `src/elspeth/web/composer/_tool_base.py`:

```python
"""Shared base class for composer tools. Spec §4.4.2."""
from __future__ import annotations

from typing import ClassVar


class ComposerTool:
    """Base class. Subclasses register via the existing tools registry.

    EXEMPT_FROM_ADEQUACY_CHECK: skip the adequacy guard (§4.4) entirely
        for this tool. Set True only with security CODEOWNERS approval
        and a docstring justification.

    EXEMPT_FROM_TYPE_DRIVEN_REDACTION: opt out of Sensitive[T] type-driven
        redaction and use a legacy ToolRedactionPolicy instead. Set True
        only with security CODEOWNERS approval. Tools with this flag MUST
        declare a non-default ToolRedactionPolicy as a class attribute or
        registry entry.
    """

    EXEMPT_FROM_ADEQUACY_CHECK: ClassVar[bool] = False
    EXEMPT_FROM_TYPE_DRIVEN_REDACTION: ClassVar[bool] = False
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k composer_tool_base
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/_tool_base.py tests/unit/web/composer/test_redaction_policy.py
git commit -m "feat(composer): ComposerTool base with EXEMPT_FROM_* ClassVars (composer-progress-persistence phase 2)"
```

---

## Task 7: Recursive adequacy guard

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py` — add `walk_adequacy(tool_class)` helper
- Test: `tests/unit/web/composer/test_redaction_policy.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_adequacy_guard_passes_for_sensitive_annotated_model():
    from typing import Annotated
    from pydantic import BaseModel
    from elspeth.web.composer._tool_base import ComposerTool
    from elspeth.web.composer.redaction import walk_adequacy, Sensitive

    class Args(BaseModel):
        path: Annotated[str, Sensitive()]

    class FakeTool(ComposerTool):
        ARGUMENT_MODEL = Args
        RESPONSE_MODEL = Args

    walk_adequacy(FakeTool)  # no exception


def test_adequacy_guard_fails_on_unannotated_string():
    from pydantic import BaseModel
    from elspeth.web.composer._tool_base import ComposerTool
    from elspeth.web.composer.redaction import walk_adequacy

    class Args(BaseModel):
        secret_no_marker: str

    class FakeTool(ComposerTool):
        ARGUMENT_MODEL = Args
        RESPONSE_MODEL = Args

    with pytest.raises(AssertionError, match="secret_no_marker"):
        walk_adequacy(FakeTool)


def test_adequacy_guard_fails_on_any_typed_field():
    from typing import Any
    from pydantic import BaseModel
    from elspeth.web.composer._tool_base import ComposerTool
    from elspeth.web.composer.redaction import walk_adequacy

    class Args(BaseModel):
        wildcard: Any

    class FakeTool(ComposerTool):
        ARGUMENT_MODEL = Args
        RESPONSE_MODEL = Args

    with pytest.raises(AssertionError, match="Any-typed"):
        walk_adequacy(FakeTool)


def test_adequacy_guard_recurses_into_nested_model():
    from typing import Annotated
    from pydantic import BaseModel
    from elspeth.web.composer._tool_base import ComposerTool
    from elspeth.web.composer.redaction import walk_adequacy, Sensitive

    class Inner(BaseModel):
        api_key: str   # NOT annotated; should fail

    class Args(BaseModel):
        inner: Inner

    class FakeTool(ComposerTool):
        ARGUMENT_MODEL = Args
        RESPONSE_MODEL = Args

    with pytest.raises(AssertionError, match="api_key"):
        walk_adequacy(FakeTool)


def test_adequacy_guard_recent_review_passes():
    from datetime import date
    from elspeth.web.composer._tool_base import ComposerTool
    from elspeth.web.composer.redaction import (
        walk_adequacy, ToolRedactionPolicy, HandlesNoSensitiveDataReason,
    )

    class FakeTool(ComposerTool):
        EXEMPT_FROM_TYPE_DRIVEN_REDACTION = True
        ARGUMENT_MODEL = None     # raw-dict tool
        RESPONSE_MODEL = None
        REDACTION_POLICY = ToolRedactionPolicy(
            handles_no_sensitive_data=True,
            handles_no_sensitive_data_reason_struct=HandlesNoSensitiveDataReason(
                sensitive_data_locations=("server-side resolver",),
                why_arguments_safe="Arguments are inventory metadata only; values resolve server-side.",
                why_responses_safe="Responses confirm wire-up; never echo material.",
                last_reviewed_iso=date.today(),
            ),
        )

    walk_adequacy(FakeTool)


def test_adequacy_guard_stale_review_fails():
    from datetime import date, timedelta
    from elspeth.web.composer._tool_base import ComposerTool
    from elspeth.web.composer.redaction import (
        walk_adequacy, ToolRedactionPolicy, HandlesNoSensitiveDataReason,
    )

    class FakeTool(ComposerTool):
        EXEMPT_FROM_TYPE_DRIVEN_REDACTION = True
        ARGUMENT_MODEL = None
        RESPONSE_MODEL = None
        REDACTION_POLICY = ToolRedactionPolicy(
            handles_no_sensitive_data=True,
            handles_no_sensitive_data_reason_struct=HandlesNoSensitiveDataReason(
                sensitive_data_locations=("server-side resolver",),
                why_arguments_safe="Arguments are inventory metadata only; values resolve server-side.",
                why_responses_safe="Responses confirm wire-up; never echo material.",
                last_reviewed_iso=date.today() - timedelta(days=400),
            ),
        )

    with pytest.raises(AssertionError, match="last_reviewed_iso"):
        walk_adequacy(FakeTool)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k adequacy
```
Expected: FAIL — `walk_adequacy` not implemented.

- [ ] **Step 3: Implement the adequacy walker**

In `src/elspeth/web/composer/redaction.py`:

```python
from datetime import date, timedelta
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel


def walk_adequacy(tool_class: type) -> None:
    """Recursive adequacy guard (spec §4.4.1).

    Asserts that every string/bytes/Any field on the tool's argument and
    response models is either annotated with Sensitive[T] OR covered by a
    legacy ToolRedactionPolicy. Raises AssertionError on any violation.
    """
    if tool_class.EXEMPT_FROM_ADEQUACY_CHECK:
        return

    if tool_class.EXEMPT_FROM_TYPE_DRIVEN_REDACTION:
        _check_legacy_policy(tool_class)
        return

    arg_model = tool_class.ARGUMENT_MODEL
    rsp_model = tool_class.RESPONSE_MODEL

    if arg_model is None or rsp_model is None:
        raise AssertionError(
            f"{tool_class.__name__}: argument or response model is None but "
            f"EXEMPT_FROM_TYPE_DRIVEN_REDACTION is False. Either annotate "
            f"the model fields with Sensitive[T] or set the exempt flag."
        )

    _walk_model(tool_class.__name__ + ".ARGUMENT_MODEL", arg_model)
    _walk_model(tool_class.__name__ + ".RESPONSE_MODEL", rsp_model)


def _walk_model(prefix: str, model: type[BaseModel]) -> None:
    """Walk model fields recursively per spec §4.4.1 table."""
    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        path = f"{prefix}.{name}"

        # Sensitive marker present? Pass.
        if any(isinstance(m, _SensitiveMarker) for m in field_info.metadata):
            continue

        # Pydantic submodel? Recurse.
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            _walk_model(path, annotation)
            continue

        # Any-typed? Fail.
        if annotation is Any or annotation is object:
            raise AssertionError(
                f"{path} is Any-typed; narrow the type or annotate with Sensitive[T]."
            )

        # str/bytes without Sensitive marker? Fail.
        if annotation in (str, bytes):
            raise AssertionError(
                f"{path} is {annotation.__name__}-typed but not annotated with "
                f"Sensitive[T] and not covered by a legacy policy. Add the "
                f"annotation or migrate the tool to legacy policy with "
                f"EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True."
            )

        # dict[str, T] / list[T] with non-scalar T? Fail-closed.
        origin = get_origin(annotation)
        if origin in (dict, list):
            args = get_args(annotation)
            value_type = args[-1] if args else None
            if value_type is None or value_type in (Any, object):
                raise AssertionError(
                    f"{path} is an opaque {origin.__name__} container; declare "
                    f"a Sensitive[T]-annotated value type or migrate to legacy policy."
                )
            if isinstance(value_type, type) and issubclass(value_type, BaseModel):
                _walk_model(path + "[*]", value_type)
                continue
            if value_type in (str, bytes):
                raise AssertionError(
                    f"{path} is a container of unannotated {value_type.__name__}; "
                    f"add Sensitive[T] or migrate to legacy policy."
                )

        # Discriminated union / Union: walk each arm.
        if origin is Union:
            for arm in get_args(annotation):
                if isinstance(arm, type) and issubclass(arm, BaseModel):
                    _walk_model(path + f"<{arm.__name__}>", arm)


def _check_legacy_policy(tool_class: type) -> None:
    """For tools with EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True, validate the
    associated ToolRedactionPolicy and its review freshness."""
    policy = tool_class.REDACTION_POLICY
    if not isinstance(policy, ToolRedactionPolicy):
        raise AssertionError(
            f"{tool_class.__name__}: EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True but "
            f"REDACTION_POLICY is not a ToolRedactionPolicy instance."
        )
    if policy.handles_no_sensitive_data:
        reason = policy.handles_no_sensitive_data_reason_struct
        assert reason is not None  # construction-time invariant
        age = date.today() - reason.last_reviewed_iso
        if age > timedelta(days=365):
            raise AssertionError(
                f"{tool_class.__name__}: last_reviewed_iso is "
                f"{age.days} days old (> 365). Schedule and complete a "
                f"redaction-policy review, then update the date."
            )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k adequacy
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_redaction_policy.py
git commit -m "feat(composer): recursive adequacy guard with stale-review detection (composer-progress-persistence phase 2)"
```

---

## Task 8: Policy-hash snapshot test

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py` — add `policy_hash` helper
- Create: `tests/unit/web/composer/redaction_policy_snapshot.json`
- Test: `tests/unit/web/composer/test_redaction_policy.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_policy_hash_is_stable_for_same_policy():
    from elspeth.web.composer.redaction import policy_hash
    p1 = ToolRedactionPolicy(known_response_keys=("ok",))
    p2 = ToolRedactionPolicy(known_response_keys=("ok",))
    assert policy_hash(p1) == policy_hash(p2)


def test_policy_hash_changes_when_keys_added():
    from elspeth.web.composer.redaction import policy_hash
    p1 = ToolRedactionPolicy(known_response_keys=("ok",))
    p2 = ToolRedactionPolicy(known_response_keys=("ok", "value"))
    assert policy_hash(p1) != policy_hash(p2)


def test_policy_hash_independent_of_summarizer_identity():
    """The hash uses the SET of summarizer-keys, not callable identity, so
    that refactoring a summarizer function does not require a snapshot
    update unless it changes which keys are summarized."""
    from elspeth.web.composer.redaction import policy_hash
    p1 = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        argument_summarizers={"path": lambda v: f"<{v}>"},
        known_response_keys=("ok",),
    )
    p2 = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        argument_summarizers={"path": lambda v: f"<{v}>"},
        known_response_keys=("ok",),
    )
    assert policy_hash(p1) == policy_hash(p2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k policy_hash
```
Expected: FAIL.

- [ ] **Step 3: Implement `policy_hash`**

In `src/elspeth/web/composer/redaction.py`:

```python
import hashlib
import json


def policy_hash(policy: ToolRedactionPolicy) -> str:
    """Deterministic SHA-256 of the policy's shape (spec §4.4.3). Used by
    the snapshot test to detect weakening across commits."""
    canon = json.dumps(
        {
            "sensitive_argument_keys": sorted(policy.sensitive_argument_keys),
            "sensitive_response_keys": sorted(policy.sensitive_response_keys),
            "known_response_keys": sorted(policy.known_response_keys),
            "summarizer_keys": sorted(policy.argument_summarizers.keys()),
            "handles_no_sensitive_data": policy.handles_no_sensitive_data,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canon.encode()).hexdigest()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k policy_hash
```
Expected: PASS.

- [ ] **Step 5: Add the snapshot test that walks the registry**

Add to `tests/unit/web/composer/test_redaction_policy.py`:

```python
import json
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).parent / "redaction_policy_snapshot.json"


def test_legacy_policy_snapshot_unchanged():
    """Spec §4.4.3: the snapshot file pins every legacy ToolRedactionPolicy.
    Changes require an explicit commit to the snapshot file routed via
    CODEOWNERS to security."""
    from elspeth.web.composer.redaction import policy_hash
    from elspeth.web.composer._tool_base import ComposerTool

    # Walk the actual tool registry. Replace the import below if the project
    # uses a different registry helper.
    from elspeth.web.composer.tools import REGISTERED_TOOLS

    actual = {
        tool_class.__name__: policy_hash(tool_class.REDACTION_POLICY)
        for tool_class in REGISTERED_TOOLS
        if getattr(tool_class, "EXEMPT_FROM_TYPE_DRIVEN_REDACTION", False)
    }
    expected = json.loads(SNAPSHOT_PATH.read_text()) if SNAPSHOT_PATH.exists() else {}
    assert actual == expected, (
        f"Legacy redaction policy hashes drifted. Update {SNAPSHOT_PATH} "
        f"with the new hashes AND label the PR `policy-strengthen` or "
        f"`policy-weaken-justified` per security CODEOWNERS review.\n"
        f"actual={actual}\nexpected={expected}"
    )
```

- [ ] **Step 6: Initialise the snapshot file**

Create `tests/unit/web/composer/redaction_policy_snapshot.json`:

```json
{}
```

The snapshot starts empty. Tools that are migrated to `Sensitive[T]` (the preferred path) will not appear here. Tools that remain on the legacy escape valve will be added to this file as Task 9 progresses.

- [ ] **Step 7: Run the snapshot test**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py::test_legacy_policy_snapshot_unchanged -v
```
Expected: PASS (empty snapshot matches no-legacy-tools state).

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_redaction_policy.py tests/unit/web/composer/redaction_policy_snapshot.json
git commit -m "feat(composer): policy-hash snapshot test for legacy redaction policies (composer-progress-persistence phase 2)"
```

---

## Task 9: Migrate every existing tool — annotate models or declare legacy policy

**Files:**
- Modify: `src/elspeth/web/composer/tools.py` — for each existing tool, either annotate its argument/response model fields with `Sensitive[T]` OR set `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True` plus a legacy `ToolRedactionPolicy` and `HandlesNoSensitiveDataReason`.
- Modify: `tests/unit/web/composer/redaction_policy_snapshot.json` — add hash for any tool that adopted the legacy escape valve.

- [ ] **Step 1: Enumerate the existing tools**

```bash
grep -n "REGISTERED_TOOLS\|register_tool\|class.*Tool.*ComposerTool" src/elspeth/web/composer/tools.py
```

Expected: a list of tools. For each tool, identify its argument and response Pydantic models.

- [ ] **Step 2: For each tool, decide migration approach**

For each tool, follow this decision table:

- Tool has well-defined Pydantic argument and response models, no third-party constraints → **annotate with `Sensitive[T]`** (preferred). Set the exempt flag to False (the default) and add `Sensitive(summarizer=...)` to every field that should be redacted.
- Tool accepts `dict` or another non-Pydantic argument type → **legacy escape valve**. Set `EXEMPT_FROM_TYPE_DRIVEN_REDACTION = True`. Declare `REDACTION_POLICY` with appropriate `sensitive_argument_keys`, `sensitive_response_keys`, `known_response_keys`, and a `HandlesNoSensitiveDataReason` if `handles_no_sensitive_data=True`.

The starting reference is the spec §4.7 table. Use it as the seed; refine per actual tool implementation.

- [ ] **Step 3: For each tool, write a unit test confirming adequacy**

For each tool you migrate, add a test that calls `walk_adequacy(ToolClass)` and confirms it does not raise. Place these tests in `tests/unit/web/composer/test_redaction_policy.py`:

```python
def test_set_source_tool_adequacy():
    from elspeth.web.composer.tools import SetSourceTool
    from elspeth.web.composer.redaction import walk_adequacy
    walk_adequacy(SetSourceTool)


# ... one test per tool ...
```

- [ ] **Step 4: Update the snapshot for any legacy-policy tools**

For each tool that ended up on the legacy escape valve, compute its hash:

```bash
.venv/bin/python -c "
from elspeth.web.composer.tools import REGISTERED_TOOLS
from elspeth.web.composer.redaction import policy_hash
import json
out = {
    cls.__name__: policy_hash(cls.REDACTION_POLICY)
    for cls in REGISTERED_TOOLS
    if getattr(cls, 'EXEMPT_FROM_TYPE_DRIVEN_REDACTION', False)
}
print(json.dumps(out, indent=2, sort_keys=True))
" > tests/unit/web/composer/redaction_policy_snapshot.json
```

- [ ] **Step 5: Run the full Phase-2 test suite**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py tests/unit/web/composer/test_sensitive_marker.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_redaction_policy.py tests/unit/web/composer/redaction_policy_snapshot.json
git commit -m "feat(composer): migrate existing tools to Sensitive[T] / legacy escape valve (composer-progress-persistence phase 2)"
```

---

## Task 10: `redact_tool_call` integration helper

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py`
- Test: `tests/unit/web/composer/test_redaction_policy.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_redact_tool_call_uses_sensitive_for_pydantic_args():
    from typing import Annotated
    from pydantic import BaseModel
    from elspeth.web.composer._tool_base import ComposerTool
    from elspeth.web.composer.redaction import redact_tool_call, Sensitive

    class Args(BaseModel):
        path: Annotated[str, Sensitive()]
        label: str

    class FakeTool(ComposerTool):
        ARGUMENT_MODEL = Args
        RESPONSE_MODEL = Args

    # Synthetic ToolCall-shaped object.
    tool_call = type("TC", (), {
        "id": "tc_1",
        "function": type("F", (), {"name": "fake", "arguments": '{"path": "/tmp/x", "label": "ok"}'})(),
    })()

    redacted = redact_tool_call(tool_call, FakeTool)
    assert redacted == {
        "id": "tc_1",
        "function": {
            "name": "fake",
            "arguments": {"path": "<redacted>", "label": "ok"},
        },
    }


def test_redact_tool_call_uses_legacy_policy_when_exempt():
    from elspeth.web.composer._tool_base import ComposerTool
    from elspeth.web.composer.redaction import redact_tool_call, ToolRedactionPolicy

    class FakeTool(ComposerTool):
        EXEMPT_FROM_TYPE_DRIVEN_REDACTION = True
        ARGUMENT_MODEL = None
        RESPONSE_MODEL = None
        REDACTION_POLICY = ToolRedactionPolicy(
            sensitive_argument_keys=("path",),
            known_response_keys=("ok",),
        )

    tool_call = type("TC", (), {
        "id": "tc_1",
        "function": type("F", (), {"name": "fake", "arguments": '{"path": "/tmp/x", "label": "ok"}'})(),
    })()

    redacted = redact_tool_call(tool_call, FakeTool)
    assert redacted["function"]["arguments"]["path"] == "<redacted>"
    assert redacted["function"]["arguments"]["label"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k redact_tool_call
```
Expected: FAIL.

- [ ] **Step 3: Implement `redact_tool_call`**

In `src/elspeth/web/composer/redaction.py`:

```python
def redact_tool_call(tool_call: Any, tool_class: type) -> dict[str, Any]:
    """Apply the appropriate redaction primitive to a ToolCall and return
    a dict ready for chat_messages.tool_calls JSON column.

    Spec §5.2.1 (called from the compose loop's redaction step).
    """
    arguments_dict = json.loads(tool_call.function.arguments)

    if tool_class.EXEMPT_FROM_TYPE_DRIVEN_REDACTION:
        policy = tool_class.REDACTION_POLICY
        redacted_args = _apply_legacy_argument_redaction(arguments_dict, policy)
    else:
        # Type-driven path: parse into the argument model, then walk it.
        arg_model = tool_class.ARGUMENT_MODEL.model_validate(arguments_dict)
        redacted_args = apply_sensitive_redaction(arg_model)

    return {
        "id": tool_call.id,
        "function": {
            "name": tool_call.function.name,
            "arguments": redacted_args,
        },
    }


def _apply_legacy_argument_redaction(
    arguments: dict[str, Any], policy: ToolRedactionPolicy
) -> dict[str, Any]:
    """Apply a legacy ToolRedactionPolicy to argument dict."""
    result: dict[str, Any] = {}
    sensitive = set(policy.sensitive_argument_keys)
    summarizers = policy.argument_summarizers
    for key, value in arguments.items():
        if key in sensitive:
            if key in summarizers:
                try:
                    result[key] = summarizers[key](value)
                except Exception as exc:  # noqa: BLE001
                    result[key] = f"<redacted-summarizer-error:{type(exc).__name__}>"
            else:
                result[key] = "<redacted>"
        else:
            result[key] = value
    return result
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k redact_tool_call
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_redaction_policy.py
git commit -m "feat(composer): redact_tool_call dispatcher (composer-progress-persistence phase 2)"
```

---

## Task 11: `lookup_tool_class` + `MissingToolError`

**Files:**
- Modify: `src/elspeth/web/composer/redaction.py` (or wherever the registry lookup currently lives)
- Test: `tests/unit/web/composer/test_redaction_policy.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_lookup_tool_class_returns_registered_class():
    from elspeth.web.composer.redaction import lookup_tool_class
    cls = lookup_tool_class("set_source")
    assert cls is not None


def test_lookup_tool_class_raises_for_unknown_name():
    from elspeth.web.composer.redaction import lookup_tool_class, MissingToolError
    with pytest.raises(MissingToolError):
        lookup_tool_class("does_not_exist_anywhere")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k lookup_tool_class
```
Expected: FAIL.

- [ ] **Step 3: Implement the lookup**

In `src/elspeth/web/composer/redaction.py`:

```python
class MissingToolError(LookupError):
    """Raised when a tool name is not in the registry. The adequacy guard
    ensures this is impossible by construction; if it ever fires, the
    registry is corrupt and the audit trail must not proceed."""


def lookup_tool_class(tool_name: str) -> type:
    """Return the registered tool class for the given name."""
    from elspeth.web.composer.tools import REGISTERED_TOOLS  # local to avoid import cycle

    for cls in REGISTERED_TOOLS:
        if getattr(cls, "TOOL_NAME", cls.__name__) == tool_name:
            return cls
    raise MissingToolError(
        f"No composer tool registered with name {tool_name!r}; the redaction "
        f"layer cannot persist an audit row for an unknown tool. Re-build "
        f"the registry or correct the tool name at the call site."
    )
```

(The `getattr` above is structural over a small known set, not defensive against the unknown — see CLAUDE.md note. If the existing tools code uses a different lookup convention, follow that convention.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_redaction_policy.py -v -k lookup_tool_class
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/redaction.py tests/unit/web/composer/test_redaction_policy.py
git commit -m "feat(composer): lookup_tool_class + MissingToolError (composer-progress-persistence phase 2)"
```

---

## Task 12: CODEOWNERS rules

**Files:**
- Modify or create: `.github/CODEOWNERS`

- [ ] **Step 1: Check whether CODEOWNERS exists**

```bash
ls -la .github/CODEOWNERS 2>&1
```

- [ ] **Step 2: Add the redaction-routing rules**

Append to (or create) `.github/CODEOWNERS`:

```
# Composer redaction primitives (spec §4.4.5)
src/elspeth/web/composer/redaction.py            @elspeth/security
src/elspeth/web/composer/_tool_base.py           @elspeth/security
src/elspeth/web/composer/tools.py                @elspeth/security
tests/unit/web/composer/redaction_policy_snapshot.json   @elspeth/security
tests/unit/web/composer/test_redaction_policy.py @elspeth/security
```

(Adjust the team handle `@elspeth/security` to match the actual GitHub team. If the team does not yet exist, file a follow-up to create it; in the interim, route to the project's senior reviewer.)

- [ ] **Step 3: Commit**

```bash
git add .github/CODEOWNERS
git commit -m "docs(security): CODEOWNERS routing for redaction primitives (composer-progress-persistence phase 2)"
```

---

## Task 13: Final Phase 2 CI run

- [ ] **Step 1: Run the redaction test suite**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_sensitive_marker.py tests/unit/web/composer/test_redaction_policy.py -v
```
Expected: PASS.

- [ ] **Step 2: Run mypy and ruff**

```bash
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/ tests/
```
Expected: clean.

- [ ] **Step 3: Run the tier-model and freeze-guard CI scripts**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check
```
Expected: both green.

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "feat(composer): progress persistence phase 2 — redaction framework" --body "$(cat <<'EOF'
## Summary

Phase 2 of composer-progress-persistence (spec §11):
- Adds `Sensitive[T]` type-driven redaction primitive
- Adds `ToolRedactionPolicy` rev-4 shape with `known_response_keys` allowlist + structured `HandlesNoSensitiveDataReason`
- Adds recursive adequacy guard (closes spec security T-4)
- Adds policy-hash snapshot test (closes spec security T-3)
- Migrates existing tools to either `Sensitive[T]` or the legacy escape valve
- Adds CODEOWNERS routing for redaction surface

## Spec

`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` revision 4. Reviewer-finding traceability table at §12.1.

## Depends on

Phase 1 PR (data layer + sync primitive). Reviewers may merge in order without re-reviewing Phase 1.

## Out of scope (later phases)

- Compose-loop integration (Phase 3)
- Frontend recovery panel (Phase 4)

## Test plan

- [x] `Sensitive[T]` marker propagates through Pydantic metadata
- [x] Recursive `apply_sensitive_redaction` with summarizer fallback (RSK-03)
- [x] `ToolRedactionPolicy` construction-time validators
- [x] `apply_response_redaction` unknown-key fail-closed (security I-1)
- [x] Recursive adequacy guard catches Any-typed, opaque dict/list, unannotated str/bytes (security T-4)
- [x] Stale-review detection (`last_reviewed_iso` > 365 days fails)
- [x] Policy-hash snapshot pinned in `redaction_policy_snapshot.json`
- [x] Every existing tool has a unit test calling `walk_adequacy(ToolClass)`
- [x] tier-model and freeze-guard CI green
- [x] mypy and ruff clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 2 Done When

All 13 tasks above are complete. Specifically:

1. ✅ `Sensitive[T]` is the primary redaction primitive.
2. ✅ Every existing composer tool either uses `Sensitive[T]` annotations OR is on the legacy escape valve with `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True` plus a structured `HandlesNoSensitiveDataReason`.
3. ✅ The recursive adequacy guard rejects Any-typed fields, opaque containers, and stale reviews.
4. ✅ The policy-hash snapshot is committed and protected by CODEOWNERS.
5. ✅ The `redact_tool_call` and `apply_response_redaction` helpers are ready to be called from the compose loop in Phase 3.
6. ✅ tier-model, freeze-guard, mypy, ruff CI green.

Phase 3 begins after this PR merges. Phase 3 will use the redaction layer this phase delivered.
