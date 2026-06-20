from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="x",
    source_param="data",
    suppresses=("R1",),
    invariant="returns None on absence",
    non_raising=True,
    test_ref="tests/test_boundary.py::test_rejects_malformed",
    test_fingerprint="deadbeef",
)
def foo(data):  # type: ignore[no-untyped-def]
    # CONTRADICTION: non_raising=True cannot coexist with a raising test_ref.
    return data["x"] if "x" in data else None
