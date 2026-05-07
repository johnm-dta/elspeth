"""Authoritative import surface for production declaration-contract registration.

Every production module that calls ``register_declaration_contract(...)`` at
module scope MUST be imported here. ``tests/unit/engine/
test_declaration_contract_bootstrap_drift.py`` AST-scans
``src/elspeth/engine/executors/`` and fails CI if a registered contract module
lands without its matching bootstrap import.

CLOSED SET: update this file in the same commit as any
``EXPECTED_CONTRACT_SITES`` change.
"""

import elspeth.engine.executors.can_drop_rows as _can_drop_rows
import elspeth.engine.executors.declared_output_fields as _declared_output_fields
import elspeth.engine.executors.declared_required_fields as _declared_required_fields
import elspeth.engine.executors.pass_through as _pass_through
import elspeth.engine.executors.schema_config_mode as _schema_config_mode
import elspeth.engine.executors.sink_required_fields as _sink_required_fields
import elspeth.engine.executors.source_guaranteed_fields as _source_guaranteed_fields

REGISTERED_CONTRACT_MODULES = (
    _can_drop_rows,
    _declared_output_fields,
    _declared_required_fields,
    _pass_through,
    _schema_config_mode,
    _sink_required_fields,
    _source_guaranteed_fields,
)
