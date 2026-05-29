from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="x",
    source_param="payload",
    suppresses=("R1",),
    invariant="y",
    test_ref="tests/test_x.py::test_y",
)
def foo(data):
    return data["x"]
