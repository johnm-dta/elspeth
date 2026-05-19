from elspeth.core.dag.models import GraphValidationError


def validate():
    raise GraphValidationError("bad", component_id="node_1")
