from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="x",
    source_param="data",
    suppresses=("R1",),
    invariant="y",
    test_ref="tests/test_no_sep.py",
)
def foo(data):  # type: ignore[no-untyped-def]
    return data["x"]
