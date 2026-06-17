from collections.abc import Mapping

from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="LLM tool-call arguments",
    source_param="arguments",
    suppresses=("R5",),
    invariant="returns None on any malformed/absent branch; never raises on arguments",
    non_raising=True,
)
def extract(arguments):  # type: ignore[no-untyped-def]
    # Non-raising boundary: every malformed branch RETURNS None. The isinstance
    # guard on ``source`` (derived from arguments) leads to a return, not a
    # raise, so the honesty gate's mechanical check passes with no test_ref.
    source = arguments["source"] if "source" in arguments else None
    if not isinstance(source, Mapping):
        return None
    return source["content"] if "content" in source else None
