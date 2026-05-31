from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="LLM tool-call arguments",
    source_param="arguments",
    suppresses=("R1",),
    invariant="raises ToolArgumentError on shape mismatch",
    test_ref="tests/test_x.py::test_y",
)
def execute(arguments):
    return arguments["nodes"]
