from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="LLM tool-call arguments",
    source_param="arguments",
    suppresses=("R1",),
    invariant="raises ToolArgumentError on shape mismatch",
    test_ref="tests/test_boundary.py::test_rejects_malformed",
)
def execute(arguments):
    return arguments["nodes"]
