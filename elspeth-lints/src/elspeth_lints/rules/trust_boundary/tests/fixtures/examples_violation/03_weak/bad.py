from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="x",
    source_param="data",
    suppresses=("R1",),
    invariant="y",
    test_ref="tests/test_boundary.py::test_no_raise",
    test_fingerprint="c3ebec8c7e077028f1e2f0293db7089878dbab89cebc10dd6bc4ac029a86264a",
)
def foo(data):  # type: ignore[no-untyped-def]
    return data["x"]
