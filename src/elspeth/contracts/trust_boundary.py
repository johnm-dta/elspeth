"""Tier-3 trust-boundary metadata decorator.

The :func:`trust_boundary` decorator marks a function as a Tier-3 external-data
boundary and records the suppressions, source description, and behavioural
invariant for static analysis. The decorator itself is **metadata-only**: it
performs no runtime validation, coercion, or exception wrapping. The wrapper
is a strict passthrough — it calls the wrapped function with the same args
and kwargs and returns its result unchanged.

The metadata is attached to the wrapper as the ``__trust_boundary__``
attribute, a frozen :class:`TrustBoundaryMetadata` dataclass. Static analysis
tooling (the ``trust_tier.tier_model`` elspeth-lints rule, and the companion
``enforce_trust_boundary_*`` CI gates) reads this attribute to:

* limit suppression of Tier-3 defensive-pattern rules (e.g. R1 silent
  ``.get()``, R5 ``isinstance`` shape guard) to the function body, and only on
  names derived from ``source_param``;
* verify that the function under the decorator actually reads from
  ``source_param`` (a separate gate prevents the decorator from becoming a
  whole-function exemption cloak);
* verify that ``test_ref`` (when present) points to a pytest node whose own
  body contains a raising assertion for malformed-input rejection, invokes
  the decorated function through ``source_param``, matches the exception type
  declared by ``invariant`` when one is present, and matches the recorded
  ``test_fingerprint``. The ``source`` prose remains reviewer-facing
  documentation; static analysis does not prove a whole-repository external
  data call graph for that field.

The decorator itself enforces only what is checkable at decoration time:

1. ``tier`` must be ``3``. Tier-1 and Tier-2 invariants must crash on
   anomaly per the project's data manifesto; a suppression decorator at
   those tiers is a category error. Passing ``tier=1`` or ``tier=2`` raises
   :class:`TypeError` at module import.
2. ``source_param`` must name an actual parameter of the wrapped function.
   The signature is inspected via :func:`inspect.signature` at decoration
   time, so a typo or post-refactor drift fails on import rather than at
   the first call site.

Scope and rationale: see
``notes/cicd-judge-cli-prototype-plan.md`` (Pillar B). The decorator is the
in-code counterpart to YAML allowlist entries; its existence reduces the
volume of per-line suppressions in
``config/cicd/enforce_tier_model/`` for the common case of a
function-scoped external-data boundary.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, ParamSpec, TypeVar

__all__ = [
    "BoundaryRule",
    "TrustBoundaryMetadata",
    "trust_boundary",
]


# Rules the decorator is allowed to suppress inside a boundary function.
#
# Scoped intentionally narrow for the prototype: R1 (silent ``.get()`` on
# external data) and R5 (``isinstance`` shape guard at the boundary) are the
# two highest-volume Tier-3 defensive-pattern violations in the current
# allowlist. New rule IDs are added here only after the operator confirms
# that a static-analysis story for that rule's suppression is in place
# (the rule must be derivable from ``source_param`` via the tier_model
# dataflow walk).
type BoundaryRule = Literal["R1", "R5"]


# Type-preservation machinery for the decorator. ``P`` captures the wrapped
# function's full parameter signature (positional + keyword), and ``R``
# captures its return type. Together with ``Concatenate`` at call sites
# (handled by the type system, not by us), this lets a method like
# ``def f(self, x: int) -> str`` retain its precise static type after
# decoration.
P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class TrustBoundaryMetadata:
    """Frozen metadata record attached to a ``@trust_boundary``-decorated function.

    All fields are immutable by construction:

    * ``tier`` is :data:`Literal[3]` (int scalar).
    * ``source``, ``source_param``, ``invariant``, ``qualname`` are ``str``.
    * ``test_ref`` and ``test_fingerprint`` are ``str | None``.
    * ``non_raising`` is a ``bool``. When ``True``, the boundary's contract is
      that it RETURNS a sentinel (``None``/empty/result object) on malformed
      ``source_param`` input and never raises on it — so a raising honesty
      test is structurally impossible. The ``trust_boundary.tests`` gate then
      replaces the raising-test requirement with a mechanical check that no
      ``raise`` in the function body is control-dependent on a
      ``source_param``-derived guard. ``non_raising=True`` is mutually
      exclusive with ``test_ref``/``test_fingerprint``.
    * ``suppresses`` is a ``tuple`` of :data:`BoundaryRule` Literals.
    * ``func`` is the wrapped :class:`Callable` (not a container; not
      deep-freezable, but immutable in the sense that ``frozen=True``
      blocks reassignment of the attribute on the metadata record).

    No ``__post_init__`` deep-freeze guard is required because no field is
    a mutable container type.
    """

    tier: Literal[3]
    source: str
    source_param: str
    suppresses: tuple[BoundaryRule, ...]
    invariant: str
    qualname: str
    func: Callable[..., Any]
    test_ref: str | None = None
    test_fingerprint: str | None = None
    non_raising: bool = False


def trust_boundary(
    *,
    tier: Literal[3],
    source: str,
    source_param: str,
    suppresses: tuple[BoundaryRule, ...],
    invariant: str,
    test_ref: str | None = None,
    test_fingerprint: str | None = None,
    non_raising: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as a Tier-3 external-data trust boundary.

    The decorator attaches a :class:`TrustBoundaryMetadata` record to the
    wrapped function as the ``__trust_boundary__`` attribute and otherwise
    returns a strict passthrough wrapper. No runtime validation is performed
    by the decorator itself; the metadata is consumed by static analysis.

    Args:
        tier: Trust tier. Must be ``3``. Other tiers raise :class:`TypeError`
            at decoration time — Tier-1 and Tier-2 must crash on anomaly, not
            suppress.
        source: Human-readable description of the external data source
            (e.g. ``"LLM tool-call arguments emitted by composer model"``).
        source_param: Name of the function parameter that carries the
            external data. Validated against the wrapped function's
            signature at decoration time; raises :class:`TypeError` if it
            is not a parameter name on the function.
        suppresses: Tuple of :data:`BoundaryRule` IDs that the static
            analyzer may suppress inside the function body, restricted
            to names derived from ``source_param``.
        invariant: Operator-facing description of what the function
            guarantees on malformed input (e.g. ``"raises
            ToolArgumentError on shape mismatch; never coerces silently"``).
        test_ref: Optional pytest nodeid of the test that should cover the
            malformed-input invariant. A separate CI gate enforces that this
            is present, points to a real test, and that the named test's own
            body contains a raising assertion that directly invokes the
            decorated function through ``source_param``.
        test_fingerprint: Optional canonical AST fingerprint of the referenced
            test body. The companion CI gate requires this whenever
            ``test_ref`` is present and reports drift if the nodeid still
            resolves but the test function was renamed, repurposed, or edited
            after review.
        non_raising: Declare that this boundary returns a sentinel
            (``None``/empty/result object) on malformed ``source_param`` input
            and never raises on it. Set this for optional-extraction,
            advisory, and convert-to-result boundaries whose contract is
            return-not-raise — a raising honesty test cannot exist for them.
            When ``True``, the ``trust_boundary.tests`` gate skips the
            raising-test requirement and instead mechanically verifies that no
            ``raise`` in the function body is control-dependent on a
            ``source_param``-derived guard (i.e. the boundary genuinely cannot
            raise on bad input). Mutually exclusive with ``test_ref`` /
            ``test_fingerprint`` — a boundary either raises-and-is-tested or is
            non-raising, never both. Passing both raises :class:`TypeError` at
            decoration time.

    Returns:
        A decorator that, when applied to a function, returns a wrapper
        with the same signature, return type, and :func:`functools.wraps`
        attributes (``__name__``, ``__qualname__``, ``__doc__``,
        ``__module__``, ``__wrapped__``, ``__dict__``), plus an additional
        ``__trust_boundary__`` attribute holding a frozen
        :class:`TrustBoundaryMetadata` record.

    Raises:
        TypeError: If ``tier`` is not ``3``, or if ``source_param`` does
            not name a parameter of the wrapped function, or if the function
            is already decorated with ``@trust_boundary``, or if
            ``non_raising=True`` is combined with ``test_ref`` /
            ``test_fingerprint``. These checks fire at decoration time
            (module import) rather than at call time.
    """
    if tier != 3:
        # Offensive: catch misuse at the decoration site. Tier-1 and Tier-2
        # have data-manifesto-mandated crash semantics; a suppression
        # decorator at those tiers is structurally wrong, not configurably
        # wrong, so we reject it at import rather than offering a flag.
        raise TypeError("@trust_boundary only applies to tier=3 trust boundaries; Tier-1 and Tier-2 must crash, not suppress.")

    if non_raising and (test_ref is not None or test_fingerprint is not None):
        # A non-raising boundary returns a sentinel on malformed input, so a
        # raising honesty test cannot exist for it; carrying a test_ref would
        # be a contradiction. Reject at the decoration site rather than letting
        # the static honesty gate be the only place the lie is caught.
        raise TypeError(
            "@trust_boundary(non_raising=True) is mutually exclusive with test_ref/test_fingerprint; "
            "a non-raising boundary returns a sentinel on malformed input and cannot have a raising test."
        )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        existing_metadata = func.__dict__["__trust_boundary__"] if "__trust_boundary__" in func.__dict__ else None
        if existing_metadata is not None:
            raise TypeError(
                f"@trust_boundary cannot be stacked on {func.__qualname__}; "
                "a function may carry exactly one trust-boundary metadata record."
            )

        signature = inspect.signature(func)
        if source_param not in signature.parameters:
            raise TypeError(
                f"@trust_boundary(source_param={source_param!r}) does not "
                f"name a parameter of {func.__qualname__}; "
                f"signature parameters are "
                f"{tuple(signature.parameters)!r}."
            )

        metadata = TrustBoundaryMetadata(
            tier=tier,
            source=source,
            source_param=source_param,
            suppresses=suppresses,
            invariant=invariant,
            qualname=func.__qualname__,
            func=func,
            test_ref=test_ref,
            test_fingerprint=test_fingerprint,
            non_raising=non_raising,
        )

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        # Attach the metadata as a well-known dunder so the static analyzer
        # can locate it from an AST FunctionDef without having to import
        # and call the wrapper. ``functools.wraps`` copies ``__dict__``
        # contents from func; we set this on the wrapper after the copy,
        # so the attribute belongs to the wrapper and not (accidentally)
        # to the original function object.
        wrapper.__trust_boundary__ = metadata  # type: ignore[attr-defined]
        return wrapper

    return decorator
