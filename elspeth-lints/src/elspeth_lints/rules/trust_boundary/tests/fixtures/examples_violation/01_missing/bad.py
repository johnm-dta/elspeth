from elspeth.contracts.trust_boundary import trust_boundary


@trust_boundary(
    tier=3,
    source="x",
    source_param="data",
    suppresses=("R1",),
    invariant="y",
)
def foo(data):
    return data["x"]
