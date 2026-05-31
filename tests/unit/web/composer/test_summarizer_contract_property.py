"""Property test for every committed summarizer (spec §9 RSK-03).

For every summarizer callable referenced by a MANIFEST entry, generate
representative inputs (Hypothesis strategies match the field's declared
type) and assert:

  (i)  the summarizer returns ``str`` for every input;
  (ii) the summarizer does not raise.

Failure here is a system bug, not a test issue; this test is a regression
net for the spec §9 RSK-03 contract ("summarizer MUST NOT raise on any
reachable input value; MUST return ``str``").

Two summarizer surfaces are enumerated:

  * **Type-driven entries** — every ``_SensitiveMarker.summarizer`` exposed
    by ``walk_model_schema(M)`` for ``M = entry.argument_model`` and (when
    present) ``entry.response_model``.  The strategy for each call site is
    derived from the schema-walked ``TraversalNode.field_type``.
  * **Declarative entries** — every callable in
    ``entry.policy.argument_summarizers``.  The strategy for each call site
    is derived from the summarizer's own first-positional-parameter type
    annotation via ``typing.get_type_hints``.

50 examples per summarizer is the spec's representative count for this
class of test (settings(max_examples=50, deadline=None)).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.web.composer.redaction import (
    MANIFEST,
    _SensitiveMarker,
    walk_model_schema,
)


def _collect_summarizer_call_sites() -> list[tuple[str, str, Callable[[Any], str], type | object]]:
    """Enumerate every committed summarizer + its expected input type.

    Returns a list of ``(label, surface, summarizer, input_type)`` tuples
    where:

      * ``label`` is the parametrize id (e.g., ``"set_source:options"``).
      * ``surface`` is ``"argument_model"``, ``"response_model"``, or
        ``"argument_summarizers"``.
      * ``summarizer`` is the callable under test.
      * ``input_type`` is the type to drive the Hypothesis strategy.

    For declarative entries the input type is read from the summarizer's
    first-positional-parameter annotation (the summarizer's signature is the
    canonical type contract for declarative dict keys).
    """
    sites: list[tuple[str, str, Callable[[Any], str], type | object]] = []
    for tool_name, entry in MANIFEST.items():
        if entry.argument_model is not None:
            # Walk the argument_model for nodes carrying a summarizer.
            for node in walk_model_schema(entry.argument_model):
                for marker in node.metadata:
                    if isinstance(marker, _SensitiveMarker) and marker.summarizer is not None:
                        sites.append(
                            (
                                f"{tool_name}:argument_model:{node.path}",
                                "argument_model",
                                marker.summarizer,
                                node.field_type,
                            )
                        )
            # Walk the response_model (if any) the same way.
            if entry.response_model is not None:
                for node in walk_model_schema(entry.response_model):
                    for marker in node.metadata:
                        if isinstance(marker, _SensitiveMarker) and marker.summarizer is not None:
                            sites.append(
                                (
                                    f"{tool_name}:response_model:{node.path}",
                                    "response_model",
                                    marker.summarizer,
                                    node.field_type,
                                )
                            )
        else:
            # Declarative branch: enumerate argument_summarizers keys.
            assert entry.policy is not None  # ToolRedaction invariant
            for key, fn in entry.policy.argument_summarizers.items():
                # The summarizer's own first-positional-parameter annotation
                # is the canonical type contract for declarative keys (the
                # declarative manifest does not declare a Pydantic model for
                # the key, so there is no schema-derived type to consult).
                hints = get_type_hints(fn)
                params = list(inspect.signature(fn).parameters.values())
                assert params, f"Summarizer {fn.__qualname__!r} for {tool_name}:{key!r} has no parameters."
                first_param_name = params[0].name
                assert first_param_name in hints, (
                    f"Summarizer {fn.__qualname__!r} for {tool_name}:{key!r} has no type annotation on its first "
                    f"parameter {first_param_name!r}; the property test cannot derive a Hypothesis strategy without "
                    f"the annotation. Add a precise type annotation matching the declarative key's expected shape."
                )
                input_type = hints[first_param_name]
                sites.append(
                    (
                        f"{tool_name}:argument_summarizers:{key}",
                        "argument_summarizers",
                        fn,
                        input_type,
                    )
                )
    return sites


_SUMMARIZER_CALL_SITES = _collect_summarizer_call_sites()


@pytest.mark.parametrize(
    ("label", "surface", "summarizer", "input_type"),
    _SUMMARIZER_CALL_SITES,
    ids=[site[0] for site in _SUMMARIZER_CALL_SITES],
)
def test_summarizer_returns_str_and_does_not_raise(
    label: str,
    surface: str,
    summarizer: Callable[[Any], str],
    input_type: type | object,
) -> None:
    """Spec §9 RSK-03: summarizer must return str and must not raise.

    Hypothesis generates ~50 representative inputs typed by the declared
    parameter type. A summarizer that raises or returns non-str fails the
    test with a falsifying example diagnostic identifying the offending
    summarizer, surface, and label.
    """

    @given(value=st.from_type(input_type))  # type: ignore[arg-type]
    @settings(max_examples=50, deadline=None)
    def check(value: Any) -> None:
        result = summarizer(value)
        assert isinstance(result, str), (
            f"Summarizer {summarizer.__qualname__!r} (call site {label!r}, surface {surface!r}) returned "
            f"{type(result).__name__}, expected str. Spec §9 RSK-03 requires every summarizer to return str on "
            f"every reachable input; a non-str return at this boundary triggers AuditIntegrityError at runtime."
        )

    check()
