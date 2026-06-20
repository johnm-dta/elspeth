from collections.abc import Mapping

from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="LLM tool-call arguments",
    source_param="arguments",
    suppresses=("R5",),
    invariant="returns None on malformed input",
    non_raising=True,
)
def extract(arguments):  # type: ignore[no-untyped-def]
    # LIE: declares non_raising, but the isinstance guard on ``source``
    # (derived from arguments) RAISES on the malformed branch — the boundary
    # demonstrably raises on bad input, so a raising test is possible and the
    # non_raising claim is false.
    source = arguments["source"] if "source" in arguments else None
    if not isinstance(source, Mapping):
        raise ValueError("source must be a mapping")
    return source["content"] if "content" in source else None
