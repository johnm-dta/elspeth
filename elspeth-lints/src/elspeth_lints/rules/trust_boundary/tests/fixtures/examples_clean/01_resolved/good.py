from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="LLM tool-call arguments",
    source_param="arguments",
    suppresses=("R1",),
    invariant="raises ValueError on shape mismatch",
    test_ref="tests/test_boundary.py::test_rejects_malformed",
    test_fingerprint="c2762a53f533585788a0a6a6552b40a18672a6ca057e140aa3318cca31884b04",
)
def execute(arguments):  # type: ignore[no-untyped-def]
    return arguments["nodes"]
