# Vulture whitelist - known false positives
# These are used by frameworks or Python protocols, not dead code

# Pydantic model_config (used by Pydantic, not called directly)
model_config  # type: ignore

# Context manager protocol (__exit__ parameters are required by signature)
exc_type  # type: ignore
exc_val  # type: ignore
exc_tb  # type: ignore

# SQLAlchemy event listener callback signature
connection_record  # type: ignore

# TYPE_CHECKING imports (used for type annotations only)

# Typer CLI commands (decorated, called by framework)
main  # type: ignore
plugins_list  # type: ignore
purge  # type: ignore
resume  # type: ignore
explain  # type: ignore
run  # type: ignore
validate  # type: ignore

# Textual framework attributes (used by framework, not called directly)
TITLE  # type: ignore
CSS  # type: ignore
BINDINGS  # type: ignore
compose  # type: ignore
action_refresh  # type: ignore
action_help  # type: ignore
render  # type: ignore
on_tree_select  # type: ignore

# Enum members (accessed by value, not called)
PENDING  # type: ignore
EXECUTING  # type: ignore
GATE  # type: ignore
AGGREGATION  # type: ignore
COALESCE  # type: ignore
SEEDED  # type: ignore
QUARANTINED  # type: ignore
COALESCED  # type: ignore
LLM  # type: ignore
HTTP  # type: ignore
SQL  # type: ignore
FILESYSTEM  # type: ignore
SUCCESS  # type: ignore
ERROR  # type: ignore
REQUIRE_ALL  # type: ignore
QUORUM  # type: ignore
BEST_EFFORT  # type: ignore
FIRST  # type: ignore

# Pydantic validators (called by Pydantic during validation)
validate_condition_expression  # type: ignore
validate_routes  # type: ignore
validate_fork_consistency  # type: ignore
validate_policy_requirements  # type: ignore
validate_merge_requirements  # type: ignore
validate_interval  # type: ignore
validate_on_validation_failure  # type: ignore
validate_on_error  # type: ignore
validate_path_not_empty  # type: ignore

# pluggy hook implementations (called by plugin system)
register_sources  # type: ignore
register_transforms  # type: ignore
register_sinks  # type: ignore
builtin_sources  # type: ignore
builtin_transforms  # type: ignore
builtin_sinks  # type: ignore

# Protocol/ABC method stubs (parameters unused in abstract body)
branch_outputs  # type: ignore

# Dataclass fields (accessed as attributes, not called)
traceback  # type: ignore
rule  # type: ignore
matched_value  # type: ignore
threshold  # type: ignore
comparison  # type: ignore
fields_modified  # type: ignore
validation_errors  # type: ignore
final_data  # type: ignore
output_type  # type: ignore
output_id  # type: ignore
state_type  # type: ignore
backend  # type: ignore
pool_size  # type: ignore
initial_delay_seconds  # type: ignore
max_delay_seconds  # type: ignore
exponential_base  # type: ignore
aggregation_boundaries  # type: ignore
idempotent  # type: ignore
quorum_threshold  # type: ignore
expected_branches  # type: ignore
input_schema_hash  # type: ignore
output_schema_hash  # type: ignore

# Protocol lifecycle methods (may be overridden by implementations)
# NOTE: on_register removed from all base classes and protocols (WP-11)
# NOTE: should_trigger and reset removed from BaseAggregation (WP-06)

# ast.NodeVisitor methods (called dynamically by .visit())
visit_NamedExpr  # type: ignore
visit_JoinedStr  # type: ignore
visit_FormattedValue  # type: ignore

# TYPE_CHECKING imports in spans.py (same pattern as context.py)
# Already covered by the opentelemetry import above

# SQLAlchemy event listeners (called by SQLAlchemy on connect/etc)
set_sqlite_pragma  # type: ignore
