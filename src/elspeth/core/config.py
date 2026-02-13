# src/elspeth/core/config.py
"""
Configuration schema and loading for Elspeth pipelines.

Uses Pydantic for validation and Dynaconf for multi-source loading.
Settings are frozen (immutable) after construction.
"""

import ast
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts.enums import OutputMode, RunMode

# Reserved edge labels that cannot be used as user-defined routing names.
# "continue" is used for sequential edges, "fork" is a gate-only routing action,
# and "on_success" is used for terminal routing edges in the DAG builder.
# Using these as user-defined labels would cause edge_map collisions in the orchestrator,
# leading to routing events recorded against wrong edges (audit corruption).
_RESERVED_EDGE_LABELS = frozenset({"continue", "fork", "on_success"})

# Names used in node_id generation must stay short enough to fit
# landscape.schema nodes_table.c.node_id (String(64)).
# Worst-case generated format overhead is ~25 chars:
#   "{prefix}_{name}_{hash12}"
# so keep {name} <= 38.
_MAX_NODE_NAME_LENGTH = 38

# Connection labels and route labels are engine-owned identifiers.
# Keep them bounded to avoid unbounded memory/key growth.
_MAX_CONNECTION_NAME_LENGTH = 64
_MAX_ROUTE_LABEL_LENGTH = 64


def _validate_max_length(value: str, *, field_label: str, max_length: int) -> str:
    """Enforce bounded identifier length for routing/node names."""
    if len(value) > max_length:
        raise ValueError(f"{field_label} exceeds max length {max_length} (got {len(value)})")
    return value


# Node names become identifiers in the DAG and must start with a letter.
_VALID_NODE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")
# Connection/sink names are routing labels — they can start with a digit
# or single underscore (e.g., "123_sink", "_private_sink") but not double
# underscore (checked separately).
_VALID_CONNECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]*$")


def _validate_node_name_chars(value: str, *, field_label: str) -> None:
    """Enforce character-class restriction on processing node names.

    Node names must start with a letter and contain only letters, digits,
    underscores, and hyphens. This prevents NUL injection, whitespace
    smuggling, unicode confusables, and SQL/shell metacharacters.
    """
    if not _VALID_NODE_NAME_RE.match(value):
        raise ValueError(
            f"{field_label} '{value}' contains invalid characters. "
            "Node names must start with a letter and contain only letters, digits, underscores, and hyphens."
        )


def _validate_connection_name_chars(value: str, *, field_label: str) -> None:
    """Enforce character-class restriction on connection/sink names.

    Connection names can start with a letter or digit and contain only
    letters, digits, underscores, and hyphens.
    """
    if not _VALID_CONNECTION_NAME_RE.match(value):
        raise ValueError(
            f"{field_label} '{value}' contains invalid characters. "
            "Names must start with a letter or digit and contain only letters, digits, underscores, and hyphens."
        )


def _validate_connection_or_sink_name(value: str, *, field_label: str) -> str:
    """Validate user-supplied connection/sink identifiers used for routing."""
    _validate_max_length(value, field_label=field_label, max_length=_MAX_CONNECTION_NAME_LENGTH)
    _validate_connection_name_chars(value, field_label=field_label)
    if value in _RESERVED_EDGE_LABELS:
        raise ValueError(f"{field_label} '{value}' is reserved. Reserved: {sorted(_RESERVED_EDGE_LABELS)}")
    if value.startswith("__"):
        raise ValueError(f"{field_label} '{value}' starts with '__', which is reserved for system edges")
    return value


class SecretsConfig(BaseModel):
    """Configuration for secret loading.

    Secrets can come from environment variables (default) or Azure Key Vault.
    Key Vault authentication uses Azure DefaultAzureCredential (Managed Identity,
    Azure CLI login, or service principal environment variables).

    When using Key Vault, an explicit mapping from env var names to Key Vault
    secret names is required.

    IMPORTANT: vault_url must be a literal HTTPS URL. Environment variable
    references like ${AZURE_KEYVAULT_URL} are NOT supported because secrets
    must be loaded before environment variable resolution occurs.

    Example (env - default):
        secrets:
          source: env  # Uses .env file and environment variables

    Example (keyvault):
        secrets:
          source: keyvault
          vault_url: https://my-vault.vault.azure.net
          mapping:
            AZURE_OPENAI_KEY: azure-openai-key
            ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key
    """

    model_config = {"frozen": True, "extra": "forbid"}

    source: Literal["env", "keyvault"] = Field(
        default="env",
        description="Secret source: 'env' (environment variables) or 'keyvault' (Azure Key Vault)",
    )
    vault_url: str | None = Field(
        default=None,
        description="Azure Key Vault URL - must be literal HTTPS URL, no ${VAR} allowed",
    )
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from env var names to Key Vault secret names",
    )

    @field_validator("vault_url", mode="before")
    @classmethod
    def validate_vault_url_format(cls, v: Any) -> Any:
        """Validate vault_url is a valid HTTPS URL without env var references.

        Note: mode="before" means this receives raw input before Pydantic type coercion.
        The return type is Any because we pass through non-strings for Pydantic to reject.
        """
        if v is None:
            return v

        # Let Pydantic handle type coercion for non-strings
        # This runs mode="before", so YAML like `vault_url: 123` arrives as int
        if not isinstance(v, str):
            return v  # Pydantic will reject with clean "str type expected" error

        # P0-3: Reject ${VAR} references (chicken-egg problem)
        if "${" in v:
            raise ValueError(
                "vault_url cannot contain ${VAR} references. "
                "Use a literal URL like 'https://my-vault.vault.azure.net'. "
                "Environment variables are resolved AFTER secrets are loaded."
            )

        # Validate URL format
        try:
            parsed = urlparse(v)
        except Exception as e:
            raise ValueError(f"Invalid URL: {v}") from e

        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {v}")

        # P0-3: Require HTTPS
        if parsed.scheme.lower() != "https":
            raise ValueError(f"vault_url must use HTTPS protocol, got: {parsed.scheme}://")

        # Normalize: strip trailing slash
        return v.rstrip("/")

    @model_validator(mode="after")
    def validate_keyvault_requirements(self) -> "SecretsConfig":
        """Validate that keyvault source has required fields."""
        if self.source == "keyvault":
            if not self.vault_url:
                raise ValueError("vault_url is required when source is 'keyvault'")
            if not self.mapping:
                raise ValueError("mapping is required when source is 'keyvault' (cannot be empty)")
        return self


class SecretFingerprintError(Exception):
    """Raised when secret fingerprinting fails.

    This occurs when:
    - Secret-like field names are found in config but ELSPETH_FINGERPRINT_KEY
      is not set and ELSPETH_ALLOW_RAW_SECRETS is not 'true'
    - A config dict contains both a secret field (e.g., 'api_key') and the
      corresponding fingerprint field ('api_key_fingerprint'), which would
      allow the pre-existing value to overwrite the computed HMAC fingerprint
    """

    pass


class TriggerConfig(BaseModel):
    """Trigger configuration for aggregation batches.

    Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
    The engine evaluates all configured triggers after each accept and fires when
    ANY condition is met.

    Trigger types:
    - count: Fire after N rows accumulated
    - timeout: Fire after N seconds since first accept
    - condition: Fire when expression evaluates to true (batch-level metrics only)

    Note: end_of_source is IMPLICIT - always checked at source exhaustion.
    It is not configured here because it always applies.

    Condition Context:
        Trigger conditions evaluate at BATCH level, not row level. Available variables:
        - batch_count: Number of rows accumulated in current batch (int)
        - batch_age_seconds: Time elapsed since first row accepted (float)

        Individual row data is NOT accessible in trigger conditions. For row-level
        routing decisions, use Gates instead of triggers.

    Example YAML (combined triggers):
        trigger:
          count: 1000           # Fire after 1000 rows
          timeout_seconds: 3600         # Or after 1 hour
          condition: "row['batch_count'] >= 100 and row['batch_age_seconds'] < 30"  # Or batch metrics
    """

    model_config = {"frozen": True, "extra": "forbid"}

    count: int | None = Field(
        default=None,
        gt=0,
        description="Fire after N rows accumulated",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Fire after N seconds since first accept",
    )
    condition: str | None = Field(
        default=None,
        description="Fire when expression evaluates to true (batch-level: batch_count, batch_age_seconds)",
    )

    @field_validator("condition")
    @classmethod
    def validate_condition_expression(cls, v: str | None) -> str | None:
        """Validate condition is a valid boolean expression at config time.

        Per CLAUDE.md Three-Tier Trust Model: trigger config is "our data" (Tier 1).
        Non-boolean expressions must be rejected at config time, not silently coerced.
        """
        if v is None:
            return v

        from elspeth.engine.expression_parser import (
            ExpressionParser,
            ExpressionSecurityError,
            ExpressionSyntaxError,
        )

        try:
            parser = ExpressionParser(v)
        except ExpressionSyntaxError as e:
            raise ValueError(f"Invalid condition syntax: {e}") from e
        except ExpressionSecurityError as e:
            raise ValueError(f"Forbidden construct in condition: {e}") from e

        # Trigger conditions are batch-level only: row may only expose these keys.
        allowed_row_keys = frozenset({"batch_count", "batch_age_seconds"})

        def _extract_string_key(node: ast.AST) -> str | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            return None

        parsed = ast.parse(v, mode="eval")
        invalid_keys: set[str] = set()

        for node in ast.walk(parsed):
            key: str | None = None

            # row['key']
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "row":
                key = _extract_string_key(node.slice)
                if key is None:
                    raise ValueError(
                        "Trigger condition row keys must be string literals. Allowed keys: row['batch_count'], row['batch_age_seconds']."
                    )

            # row.get('key') / row.get('key', default)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "row"
                and node.func.attr == "get"
                and node.args
            ):
                key = _extract_string_key(node.args[0])
                if key is None:
                    raise ValueError(
                        "Trigger condition row.get() keys must be string literals. "
                        "Allowed keys: row['batch_count'], row['batch_age_seconds']."
                    )

            if key is None:
                continue
            if key not in allowed_row_keys:
                invalid_keys.add(key)

        if invalid_keys:
            invalid_display = ", ".join(sorted(repr(key) for key in invalid_keys))
            raise ValueError(
                "Trigger condition references unsupported row keys: "
                f"{invalid_display}. Allowed keys: row['batch_count'], row['batch_age_seconds']."
            )

        # P2-2026-01-31: Reject non-boolean expressions
        # Per CLAUDE.md: "if bool(result)" coercion is forbidden for our data
        if not parser.is_boolean_expression():
            raise ValueError(
                f"Trigger condition must be a boolean expression that returns True/False. "
                f"Got: {v!r} which returns a non-boolean value. "
                f"Use comparisons (>=, ==, etc.) or boolean operators (and, or, not) "
                f"to create conditions like: row['batch_count'] >= 100"
            )
        return v

    @model_validator(mode="after")
    def validate_at_least_one_trigger(self) -> "TriggerConfig":
        """At least one trigger must be configured."""
        if self.count is None and self.timeout_seconds is None and self.condition is None:
            raise ValueError("at least one trigger must be configured (count, timeout_seconds, or condition)")
        return self

    @property
    def has_count(self) -> bool:
        """Whether count trigger is configured."""
        return self.count is not None

    @property
    def has_timeout(self) -> bool:
        """Whether timeout trigger is configured."""
        return self.timeout_seconds is not None

    @property
    def has_condition(self) -> bool:
        """Whether condition trigger is configured."""
        return self.condition is not None


class AggregationSettings(BaseModel):
    """Aggregation configuration for batching rows.

    Aggregations collect rows until a trigger fires, then process the batch.
    The engine evaluates trigger conditions - plugins only accept/reject rows.

    Output modes:
    - passthrough: Batch releases all accepted rows unchanged
    - transform: Batch applies a transform function to produce results

    Example YAML:
        aggregations:
          - name: batch_stats
            plugin: stats_aggregation
            trigger:
              count: 100
            output_mode: transform
            expected_output_count: 1  # Optional: validate N->1 aggregation
            options:
              fields: ["value"]
              compute_mean: true
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = Field(description="Aggregation identifier (unique within pipeline)")
    plugin: str = Field(description="Plugin name to instantiate")
    input: str = Field(description="Named input connection (must match an upstream on_success value)")
    on_success: str | None = Field(
        default=None,
        description="Connection name or sink name for aggregation output",
    )
    trigger: TriggerConfig = Field(description="When to flush the batch")
    output_mode: OutputMode = Field(
        default=OutputMode.TRANSFORM,
        description="How batch produces output rows",
    )
    expected_output_count: int | None = Field(
        default=None,
        description="Optional: validate aggregation produces exactly this many output rows.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate aggregation name is not empty or reserved."""
        if not v or not v.strip():
            raise ValueError("Aggregation name must not be empty")
        v = v.strip()
        _validate_max_length(v, field_label="Aggregation name", max_length=_MAX_NODE_NAME_LENGTH)
        _validate_node_name_chars(v, field_label="Aggregation name")
        if v in _RESERVED_EDGE_LABELS:
            raise ValueError(f"Aggregation name '{v}' is reserved. Reserved: {sorted(_RESERVED_EDGE_LABELS)}")
        if v.startswith("__"):
            raise ValueError(f"Aggregation name '{v}' starts with '__', which is reserved for system edges")
        return v

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        """Validate input connection name is not empty."""
        if not v or not v.strip():
            raise ValueError("Aggregation input connection must not be empty")
        value = v.strip()
        return _validate_connection_or_sink_name(value, field_label="Aggregation input connection name")

    @field_validator("on_success")
    @classmethod
    def validate_on_success(cls, v: str | None) -> str | None:
        """Ensure on_success is not empty string."""
        if v is not None and not v.strip():
            raise ValueError("on_success must be a connection name, sink name, or omitted entirely")
        if v is None:
            return None
        value = v.strip()
        return _validate_connection_or_sink_name(value, field_label="Aggregation on_success connection name")

    @field_validator("output_mode", mode="before")
    @classmethod
    def reject_single_mode(cls, v: Any) -> Any:
        """Reject deprecated 'single' mode with helpful migration message."""
        if v == "single":
            raise ValueError(
                "output_mode='single' has been removed (bug elspeth-rapid-nd3). "
                "Use output_mode='transform' instead. For N->1 aggregations, add "
                "expected_output_count=1 to validate cardinality."
            )
        return v


class GateSettings(BaseModel):
    """Gate configuration for config-driven routing.

    Gates are defined in YAML and evaluated by the engine using ExpressionParser.
    The condition expression determines routing; route labels map to destinations.

    Example YAML:
        gates:
          - name: quality_check
            condition: "row['confidence'] >= 0.85"
            routes:
              high: quality_ok
              low: review_sink
          - name: parallel_analysis
            condition: "True"
            routes:
              all: fork
            fork_to:
              - path_a
              - path_b
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = Field(description="Gate identifier (unique within pipeline)")
    input: str = Field(description="Named input connection (must match an upstream on_success value)")
    condition: str = Field(description="Expression to evaluate (validated by ExpressionParser)")
    routes: dict[str, str] = Field(
        max_length=32,
        description="Maps route labels to destinations (connection name, sink name, or 'fork')",
    )
    fork_to: list[str] | None = Field(
        default=None,
        max_length=32,
        description="List of paths for fork operations",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate gate name is not empty or reserved."""
        if not v or not v.strip():
            raise ValueError("Gate name must not be empty")
        v = v.strip()
        _validate_max_length(v, field_label="Gate name", max_length=_MAX_NODE_NAME_LENGTH)
        _validate_node_name_chars(v, field_label="Gate name")
        if v in _RESERVED_EDGE_LABELS:
            raise ValueError(f"Gate name '{v}' is reserved. Reserved: {sorted(_RESERVED_EDGE_LABELS)}")
        if v.startswith("__"):
            raise ValueError(f"Gate name '{v}' starts with '__', which is reserved for system edges")
        return v

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        """Validate input connection name is not empty."""
        if not v or not v.strip():
            raise ValueError("Gate input connection must not be empty")
        value = v.strip()
        return _validate_connection_or_sink_name(value, field_label="Gate input connection name")

    @field_validator("condition")
    @classmethod
    def validate_condition_expression(cls, v: str) -> str:
        """Validate that condition is a valid expression at config time."""
        from elspeth.engine.expression_parser import (
            ExpressionParser,
            ExpressionSecurityError,
            ExpressionSyntaxError,
        )

        try:
            ExpressionParser(v)
        except ExpressionSyntaxError as e:
            raise ValueError(f"Invalid condition syntax: {e}") from e
        except ExpressionSecurityError as e:
            raise ValueError(f"Forbidden construct in condition: {e}") from e
        return v

    @field_validator("routes")
    @classmethod
    def validate_routes(cls, v: dict[str, str]) -> dict[str, str]:
        """Routes must have at least one entry with valid destinations.

        Also validates that route labels don't use reserved edge labels,
        which would cause edge_map collisions in the orchestrator.

        Note: Sink destinations are NOT restricted to identifier-like names.
        Sink names can be any valid dict key (including hyphenated names like "my-sink").
        The DAG builder validates that destinations are actual sink keys at graph
        compilation time (ExecutionGraph.from_plugin_instances), which provides better error
        messages referencing available sinks.
        """
        if not v:
            raise ValueError("routes must have at least one entry")

        for label, destination in v.items():
            if not label:
                raise ValueError("Route labels must not be empty")
            _validate_connection_or_sink_name(label, field_label="Route label")

            # "fork" is a special routing action consumed by fork_to branch wiring.
            if destination == "fork":
                continue
            if destination == "continue":
                raise ValueError("Route destination 'continue' has been removed. Use an explicit connection name or sink name.")
            # Sink/connection-name validation is structural. Resolution to an
            # actual sink or producer happens during DAG compilation.
            _validate_connection_or_sink_name(
                destination,
                field_label=f"Route destination for label '{label}'",
            )
        return v

    @field_validator("fork_to")
    @classmethod
    def validate_fork_to_labels(cls, v: list[str] | None) -> list[str] | None:
        """Validate fork branch names don't use reserved edge labels.

        Fork branches become edge labels in the DAG, so they must not collide
        with reserved labels like 'continue'.
        """
        if v is None:
            return v

        stripped = []
        for branch in v:
            if not branch or not branch.strip():
                raise ValueError("Fork branch names must not be empty")
            value = branch.strip()
            _validate_connection_or_sink_name(value, field_label="Fork branch name")
            stripped.append(value)
        return stripped

    @model_validator(mode="after")
    def validate_fork_consistency(self) -> "GateSettings":
        """Ensure fork_to is provided when routes use 'fork' destination."""
        has_fork_route = any(dest == "fork" for dest in self.routes.values())
        if has_fork_route and not self.fork_to:
            raise ValueError("fork_to is required when any route destination is 'fork'")
        if self.fork_to and not has_fork_route:
            raise ValueError("fork_to is only valid when a route destination is 'fork'")
        return self

    @model_validator(mode="after")
    def validate_boolean_routes(self) -> "GateSettings":
        """Validate route labels match the condition's return type.

        Boolean expressions (comparisons, and/or, not) must use "true"/"false"
        as route labels. Using labels like "above"/"below" for a condition like
        `row['amount'] > 1000` is a config error - the expression evaluates to
        True/False, not "above"/"below".
        """
        from elspeth.engine.expression_parser import ExpressionParser

        parser = ExpressionParser(self.condition)
        if parser.is_boolean_expression():
            route_labels = set(self.routes.keys())
            expected_labels = {"true", "false"}

            # Check for common mistakes
            if route_labels != expected_labels:
                missing = expected_labels - route_labels
                extra = route_labels - expected_labels

                # Build helpful error message
                msg_parts = [f"Gate '{self.name}' has a boolean condition ({self.condition!r}) but route labels don't match."]

                if extra:
                    msg_parts.append(f"Found labels {sorted(extra)!r} but boolean expressions evaluate to True/False, not these values.")
                if missing:
                    msg_parts.append(f"Missing required labels: {sorted(missing)!r}.")
                msg_parts.append('Use routes: {"true": <destination>, "false": <destination>}')

                raise ValueError(" ".join(msg_parts))

        return self

    # NOTE: routes dict and fork_to list are mutable containers on a frozen model.
    # Pydantic frozen=True prevents attribute reassignment but not container mutation.
    # Runtime immutability is enforced at the DAG builder level (dag.py line ~1320-1323)
    # where NodeInfo.config is wrapped in MappingProxyType. Freezing here would break
    # copy.deepcopy() used in the config loading pipeline (MappingProxyType is not picklable).


class CoalesceSettings(BaseModel):
    """Configuration for coalesce (token merging) operations.

    Coalesce merges tokens from parallel fork paths back into a single token.
    Tokens are correlated by row_id (same source row that was forked).

    Memory Implications:
        When using 'require_all' or 'quorum' policies, tokens arriving early
        are held in memory waiting for their siblings. If one branch consistently
        fails before reaching coalesce, the other branches' tokens will be held
        until flush_pending() is called at end-of-source.

        For high-volume pipelines with unreliable branches:
        - Consider 'best_effort' with timeout for faster memory release
        - Consider 'first' to avoid holding tokens entirely
        - Monitor rows_coalesce_failed counter for excessive failures

    Example YAML:
        coalesce:
          - name: merge_analysis
            branches:
              - sentiment_path
              - entity_path
            policy: require_all
            merge: union

          - name: quorum_merge
            branches:
              - fast_model
              - slow_model
              - fallback_model
            policy: quorum
            quorum_count: 2
            merge: nested
            timeout_seconds: 30
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = Field(description="Unique identifier for this coalesce point")
    branches: dict[str, str] = Field(
        min_length=2,
        description="Branch identity → input connection mapping. List format normalized to identity dict.",
    )

    @field_validator("branches", mode="before")
    @classmethod
    def normalize_branches(cls, v: Any) -> dict[str, str]:
        """Normalize list format to identity dict.

        branches: [a, b] becomes branches: {a: a, b: b}
        This is config ergonomics — list is cleaner when no transforms are needed.

        Rejects duplicate branch names in list form — dict comprehension
        would silently discard them, hiding a config error.
        """
        if isinstance(v, list):
            if len(v) != len(set(v)):
                dupes = sorted({b for b in v if v.count(b) > 1})
                raise ValueError(f"Duplicate branch names in list: {dupes}")
            return {b: b for b in v}
        return v  # type: ignore[no-any-return]  # Pydantic validates dict[str, str]

    policy: Literal["require_all", "quorum", "best_effort", "first"] = Field(
        default="require_all",
        description="How to handle partial arrivals",
    )
    merge: Literal["union", "nested", "select"] = Field(
        default="union",
        description="How to combine row data from branches",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Max wait time (required for best_effort, optional for quorum)",
    )
    quorum_count: int | None = Field(
        default=None,
        gt=0,
        description="Minimum branches required (required for quorum policy)",
    )
    select_branch: str | None = Field(
        default=None,
        description="Which branch to take for 'select' merge strategy",
    )
    on_success: str | None = Field(
        default=None,
        description="Sink name for coalesce output. Required when coalesce is terminal (no downstream transforms).",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate coalesce name is not empty or reserved."""
        if not v or not v.strip():
            raise ValueError("Coalesce name must not be empty")
        value = v.strip()
        _validate_max_length(value, field_label="Coalesce name", max_length=_MAX_NODE_NAME_LENGTH)
        _validate_node_name_chars(value, field_label="Coalesce name")
        if value in _RESERVED_EDGE_LABELS:
            raise ValueError(f"Coalesce name '{value}' is reserved. Reserved: {sorted(_RESERVED_EDGE_LABELS)}")
        if value.startswith("__"):
            raise ValueError(f"Coalesce name '{value}' starts with '__', which is reserved for system edges")
        return value

    @field_validator("branches")
    @classmethod
    def validate_branch_names(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure coalesce branch names (keys) and input connections (values) are valid."""
        validated: dict[str, str] = {}
        for branch_name, input_connection in v.items():
            if not branch_name or not branch_name.strip():
                raise ValueError("Coalesce branch names must not be empty")
            if not input_connection or not input_connection.strip():
                raise ValueError(f"Coalesce branch '{branch_name}' input connection must not be empty")
            key = branch_name.strip()
            val = input_connection.strip()
            _validate_connection_or_sink_name(key, field_label="Coalesce branch name")
            _validate_connection_or_sink_name(val, field_label=f"Coalesce branch '{key}' input connection")
            validated[key] = val
        return validated

    @model_validator(mode="after")
    def validate_policy_requirements(self) -> "CoalesceSettings":
        """Validate policy-specific requirements."""
        if self.policy == "quorum" and self.quorum_count is None:
            raise ValueError(f"Coalesce '{self.name}': quorum policy requires quorum_count")
        if self.policy == "quorum" and self.quorum_count is not None and self.quorum_count > len(self.branches):
            raise ValueError(
                f"Coalesce '{self.name}': quorum_count ({self.quorum_count}) cannot exceed number of branches ({len(self.branches)})"
            )
        if self.policy == "best_effort" and self.timeout_seconds is None:
            raise ValueError(f"Coalesce '{self.name}': best_effort policy requires timeout_seconds")
        return self

    @model_validator(mode="after")
    def validate_merge_requirements(self) -> "CoalesceSettings":
        """Validate merge strategy requirements."""
        if self.merge == "select" and self.select_branch is None:
            raise ValueError(f"Coalesce '{self.name}': select merge strategy requires select_branch")
        if self.select_branch is not None and self.select_branch not in self.branches:
            raise ValueError(
                f"Coalesce '{self.name}': select_branch '{self.select_branch}' must be one of the expected branches: {sorted(self.branches.keys())}"
            )
        return self

    @field_validator("on_success")
    @classmethod
    def validate_on_success(cls, v: str | None) -> str | None:
        """Ensure on_success sink name is not empty or system-reserved."""
        if v is not None and not v.strip():
            raise ValueError("on_success must be a sink name or omitted entirely")
        if v is None:
            return None
        value = v.strip()
        return _validate_connection_or_sink_name(value, field_label="Coalesce on_success sink name")


class SourceSettings(BaseModel):
    """Source plugin configuration per architecture.

    Phase 3 addition: on_success lifted from SourceDataConfig (options layer)
    to settings level (3a3f-A). Source must declare where valid rows go.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    plugin: str = Field(description="Plugin name (csv_local, json, http_poll, etc.)")
    on_success: str = Field(
        ...,  # Required - no default
        description="Connection name or sink name for rows that pass source validation",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )

    @field_validator("on_success")
    @classmethod
    def validate_on_success(cls, v: str) -> str:
        """Ensure on_success is not empty."""
        if not v or not v.strip():
            raise ValueError("Source on_success must be a connection name or sink name")
        value = v.strip()
        return _validate_connection_or_sink_name(value, field_label="Source on_success connection name")


class TransformSettings(BaseModel):
    """Transform plugin configuration per architecture.

    Note: Gate routing is now config-driven only (see GateSettings).
    Plugin-based gates were removed - use the gates: section instead.

    Phase 3 additions (declarative DAG wiring):
        name: User-facing wiring label. Drives node IDs in the DAG and
            appears in Landscape audit records. Must be unique across all
            processing nodes (transforms, gates, aggregations, coalesce).
        input: Named input connection. Declares which upstream node's
            on_success output feeds this transform. Matched by the DAG
            builder to create explicit edges (no positional inference).
        on_success: Lifted from options layer (3a3f-A). Sink name or
            connection name for successfully processed rows.
        on_error: Lifted from options layer alongside on_success.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = Field(description="Unique identifier for this transform (drives node IDs and audit records)")
    plugin: str = Field(description="Plugin name")
    input: str = Field(description="Named input connection (must match an upstream on_success value)")
    on_success: str = Field(
        description="Connection name or sink name for successfully processed rows",
    )
    on_error: str = Field(
        description="Sink name for rows that cannot be processed, or 'discard'",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate transform name is not empty or reserved."""
        if not v or not v.strip():
            raise ValueError("Transform name must not be empty")
        v = v.strip()
        _validate_max_length(v, field_label="Transform name", max_length=_MAX_NODE_NAME_LENGTH)
        _validate_node_name_chars(v, field_label="Transform name")
        if v in _RESERVED_EDGE_LABELS:
            raise ValueError(f"Transform name '{v}' is reserved. Reserved: {sorted(_RESERVED_EDGE_LABELS)}")
        if v.startswith("__"):
            raise ValueError(f"Transform name '{v}' starts with '__', which is reserved for system edges")
        return v

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        """Validate input connection name is not empty."""
        if not v or not v.strip():
            raise ValueError("Transform input connection must not be empty")
        value = v.strip()
        return _validate_connection_or_sink_name(value, field_label="Transform input connection name")

    @field_validator("on_success")
    @classmethod
    def validate_on_success(cls, v: str) -> str:
        """Ensure on_success is a valid connection or sink name."""
        if not v.strip():
            raise ValueError("on_success must be a connection name or sink name")
        value = v.strip()
        return _validate_connection_or_sink_name(value, field_label="Transform on_success connection name")

    @field_validator("on_error")
    @classmethod
    def validate_on_error(cls, v: str) -> str:
        """Ensure on_error is a valid sink name or 'discard'."""
        if not v.strip():
            raise ValueError("on_error must be a sink name or 'discard'")
        value = v.strip()
        if value == "discard":
            return value
        return _validate_connection_or_sink_name(value, field_label="Transform on_error sink name")


class SinkSettings(BaseModel):
    """Sink plugin configuration per architecture."""

    model_config = {"frozen": True, "extra": "forbid"}

    plugin: str = Field(description="Plugin name (csv, json, database, webhook, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )


class LandscapeExportSettings(BaseModel):
    """Landscape export configuration for audit compliance.

    Exports audit trail to a configured sink after run completes.
    Optional cryptographic signing for legal-grade integrity.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    enabled: bool = Field(
        default=False,
        description="Enable audit trail export after run completes",
    )
    sink: str | None = Field(
        default=None,
        description="Sink name to export to (must be defined in sinks)",
    )
    format: Literal["csv", "json"] = Field(
        default="csv",
        description="Export format: csv (human-readable) or json (machine)",
    )
    sign: bool = Field(
        default=False,
        description="HMAC sign each record for integrity verification",
    )


class LandscapeSettings(BaseModel):
    """Landscape audit system configuration per architecture."""

    model_config = {"frozen": True, "extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable audit trail recording")
    backend: Literal["sqlite", "sqlcipher", "postgresql"] = Field(
        default="sqlite",
        description="Database backend type (sqlcipher requires the 'security' extra)",
    )
    # NOTE: Using str instead of Path - Path mangles PostgreSQL DSNs like
    # "postgresql://user:pass@host/db" (pathlib interprets // as UNC path)
    url: str = Field(
        default="sqlite:///./state/audit.db",
        description="Full SQLAlchemy database URL",
    )
    encryption_key_env: str = Field(
        default="ELSPETH_AUDIT_KEY",
        description="Environment variable holding the SQLCipher passphrase (backend=sqlcipher only)",
    )
    export: LandscapeExportSettings = Field(
        default_factory=LandscapeExportSettings,
        description="Post-run audit export configuration",
    )
    dump_to_jsonl: bool = Field(
        default=False,
        description="Write an append-only JSONL change journal for emergency backup",
    )
    dump_to_jsonl_path: str | None = Field(
        default=None,
        description="Optional path for JSONL change journal (default: derived from landscape.url)",
    )
    dump_to_jsonl_fail_on_error: bool = Field(
        default=False,
        description="Fail the run if the JSONL journal cannot be written",
    )
    dump_to_jsonl_include_payloads: bool = Field(
        default=False,
        description="Inline payload store contents in the JSONL journal (request/response bodies)",
    )
    dump_to_jsonl_payload_base_path: str | None = Field(
        default=None,
        description="Optional payload store base path for inlining payloads (default: payload_store.base_path)",
    )

    @field_validator("url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format at config time.

        Catches malformed URLs early (fail-fast) rather than at first DB access.
        Uses SQLAlchemy's own URL parser for accurate validation.
        """
        from sqlalchemy.engine.url import make_url
        from sqlalchemy.exc import ArgumentError

        try:
            parsed = make_url(v)
            # Verify we got a valid driver/scheme
            if not parsed.drivername:
                raise ValueError("Database URL missing driver (e.g., 'sqlite', 'postgresql')")
        except ArgumentError as e:
            raise ValueError(f"Invalid database URL format: {e}") from e
        return v

    @model_validator(mode="after")
    def validate_sqlcipher_backend(self) -> "LandscapeSettings":
        """Validate that sqlcipher backend uses a SQLite-compatible URL."""
        if self.backend == "sqlcipher" and not self.url.startswith("sqlite"):
            raise ValueError("backend='sqlcipher' requires a SQLite URL (sqlcipher is wire-compatible with SQLite)")
        return self


class ConcurrencySettings(BaseModel):
    """Parallel processing configuration per architecture."""

    model_config = {"frozen": True, "extra": "forbid"}

    max_workers: int = Field(
        default=4,
        gt=0,
        description="Maximum parallel workers (default 4, production typically 16)",
    )


class DatabaseSettings(BaseModel):
    """Database connection configuration."""

    model_config = {"frozen": True, "extra": "forbid"}

    url: str = Field(description="SQLAlchemy database URL")
    pool_size: int = Field(default=5, gt=0, description="Connection pool size")
    echo: bool = Field(default=False, description="Echo SQL statements")


class ServiceRateLimit(BaseModel):
    """Rate limit configuration for a specific service."""

    model_config = {"frozen": True, "extra": "forbid"}

    requests_per_minute: int = Field(default=60, gt=0, description="Maximum requests per minute")


class RateLimitSettings(BaseModel):
    """Configuration for rate limiting external calls.

    Example YAML:
        rate_limit:
          enabled: true
          default_requests_per_minute: 60
          persistence_path: ./rate_limits.db
          services:
            openai:
              requests_per_minute: 100
            weather_api:
              requests_per_minute: 120
    """

    model_config = {"frozen": True, "extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable rate limiting for external calls")
    default_requests_per_minute: int = Field(default=60, gt=0, description="Default per-minute rate limit for unconfigured services")
    persistence_path: str | None = Field(default=None, description="SQLite path for cross-process limits")
    services: dict[str, ServiceRateLimit] = Field(default_factory=dict, description="Per-service rate limit configurations")

    def get_service_config(self, service_name: str) -> ServiceRateLimit:
        """Get rate limit config for a service, with fallback to defaults."""
        if service_name in self.services:
            return self.services[service_name]
        return ServiceRateLimit(
            requests_per_minute=self.default_requests_per_minute,
        )


class CheckpointSettings(BaseModel):
    """Configuration for crash recovery checkpointing.

    Checkpoint frequency trade-offs:
    - every_row: Safest, can resume from any row. Higher I/O overhead.
    - every_n: Balance safety and performance. Lose up to N-1 rows on crash.
    - aggregation_only: Fastest, checkpoint only at aggregation flushes.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    enabled: bool = True
    frequency: Literal["every_row", "every_n", "aggregation_only"] = "every_row"
    checkpoint_interval: int | None = Field(default=None, gt=0)  # Required if frequency == "every_n"
    aggregation_boundaries: bool = True  # Always checkpoint at aggregation flush

    @model_validator(mode="after")
    def validate_interval(self) -> "CheckpointSettings":
        if self.frequency == "every_n" and self.checkpoint_interval is None:
            raise ValueError("checkpoint_interval required when frequency='every_n'")
        return self


class RetrySettings(BaseModel):
    """Retry behavior configuration."""

    model_config = {"frozen": True, "extra": "forbid"}

    max_attempts: int = Field(default=3, gt=0, description="Maximum retry attempts")
    initial_delay_seconds: float = Field(default=1.0, gt=0, description="Initial backoff delay")
    max_delay_seconds: float = Field(default=60.0, gt=0, description="Maximum backoff delay")
    exponential_base: float = Field(default=2.0, gt=1.0, description="Exponential backoff base")


class PayloadStoreSettings(BaseModel):
    """Payload store configuration."""

    model_config = {"frozen": True, "extra": "forbid"}

    backend: str = Field(default="filesystem", description="Storage backend type")
    base_path: Path = Field(
        default=Path(".elspeth/payloads"),
        description="Base path for filesystem backend",
    )
    retention_days: int = Field(default=90, gt=0, description="Payload retention in days")


class ExporterSettings(BaseModel):
    """Configuration for a single telemetry exporter.

    Example YAML:
        telemetry:
          exporters:
            - name: console
              options:
                pretty: true
            - name: otlp
              options:
                endpoint: https://otel.example.com
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = Field(description="Exporter name (console, otlp, azure_monitor, datadog)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Exporter-specific configuration options",
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Exporter name cannot be empty."""
        if not v.strip():
            raise ValueError("exporter name cannot be empty")
        return v


class TelemetrySettings(BaseModel):
    """Configuration for pipeline telemetry emission.

    Telemetry emits structured events about pipeline execution for monitoring,
    observability, and debugging. Events flow to configured exporters.

    Example YAML:
        telemetry:
          enabled: true
          granularity: rows
          backpressure_mode: drop
          fail_on_total_exporter_failure: false
          exporters:
            - name: console
              options:
                pretty: true
            - name: otlp
              options:
                endpoint: https://otel.example.com

    Granularity levels (from least to most verbose):
        - lifecycle: Only run start/complete/failed events
        - rows: Lifecycle + row-level events
        - full: Rows + external call events (LLM, HTTP, etc.)

    Backpressure modes:
        - block: Block pipeline when exporters can't keep up (safest)
        - drop: Drop events when buffer is full (lossy, no pipeline impact)
        - slow: Adaptive rate limiting (not yet implemented)
    """

    model_config = {"frozen": True, "extra": "forbid"}

    enabled: bool = Field(
        default=False,
        description="Enable telemetry emission",
    )
    granularity: Literal["lifecycle", "rows", "full"] = Field(
        default="lifecycle",
        description="Event granularity: lifecycle (minimal), rows, or full (verbose)",
    )
    backpressure_mode: Literal["block", "drop", "slow"] = Field(
        default="block",
        description="How to handle backpressure when exporters can't keep up",
    )
    fail_on_total_exporter_failure: bool = Field(
        default=True,
        description="Fail the run if all exporters fail (when enabled)",
    )
    max_consecutive_failures: int = Field(
        default=10,
        gt=0,
        description="Number of consecutive total exporter failures before disabling telemetry or raising an error",
    )
    exporters: list[ExporterSettings] = Field(
        default_factory=list,
        description="List of telemetry exporters to send events to",
    )

    @model_validator(mode="after")
    def validate_exporters_when_enabled(self) -> "TelemetrySettings":
        """Warn if telemetry is enabled but no exporters are configured."""
        # Note: This is a warning case, not an error. Telemetry with no exporters
        # just means events are produced but not exported anywhere - useful for
        # testing or when exporters are added dynamically.
        return self


class ElspethSettings(BaseModel):
    """Top-level Elspeth configuration matching architecture specification.

    This is the single source of truth for pipeline configuration.
    All settings are validated and frozen after construction.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # Required - core pipeline definition
    source: SourceSettings = Field(
        description="Source plugin configuration (exactly one per run)",
    )
    sinks: dict[str, SinkSettings] = Field(
        max_length=50,
        description="Named sink configurations (one or more required)",
    )

    # Run mode configuration
    run_mode: RunMode = Field(
        default=RunMode.LIVE,
        description="Execution mode: live (real calls), replay (use recorded), verify (compare)",
    )
    replay_from: str | None = Field(
        default=None,
        description="Run ID to replay/verify against (required for replay/verify modes)",
    )

    # Optional - transform chain
    transforms: list[TransformSettings] = Field(
        default_factory=list,
        max_length=500,
        description="Ordered list of transforms/gates to apply",
    )

    # Optional - engine-level gates (config-driven routing)
    gates: list[GateSettings] = Field(
        default_factory=list,
        max_length=100,
        description="Engine-level gates for config-driven routing (evaluated by ExpressionParser)",
    )

    # Optional - coalesce configuration (for merging fork paths)
    coalesce: list[CoalesceSettings] = Field(
        default_factory=list,
        max_length=100,
        description="Coalesce configurations for merging forked paths",
    )

    # Optional - aggregations (config-driven batching)
    aggregations: list[AggregationSettings] = Field(
        default_factory=list,
        max_length=100,
        description="Aggregation configurations for batching rows",
    )

    # Optional - subsystem configuration with defaults
    landscape: LandscapeSettings = Field(
        default_factory=LandscapeSettings,
        description="Audit trail configuration",
    )
    concurrency: ConcurrencySettings = Field(
        default_factory=ConcurrencySettings,
        description="Parallel processing configuration",
    )
    retry: RetrySettings = Field(
        default_factory=RetrySettings,
        description="Retry behavior configuration",
    )
    payload_store: PayloadStoreSettings = Field(
        default_factory=PayloadStoreSettings,
        description="Large payload storage configuration",
    )
    checkpoint: CheckpointSettings = Field(
        default_factory=CheckpointSettings,
        description="Crash recovery checkpoint configuration",
    )
    rate_limit: RateLimitSettings = Field(
        default_factory=RateLimitSettings,
        description="Rate limiting configuration",
    )
    telemetry: TelemetrySettings = Field(
        default_factory=TelemetrySettings,
        description="Telemetry and observability configuration",
    )

    @model_validator(mode="after")
    def validate_export_sink_exists(self) -> "ElspethSettings":
        """Ensure export.sink references a defined sink when enabled."""
        if self.landscape.export.enabled:
            if self.landscape.export.sink is None:
                raise ValueError("landscape.export.sink is required when export is enabled")
            if self.landscape.export.sink not in self.sinks:
                raise ValueError(
                    f"landscape.export.sink '{self.landscape.export.sink}' not found in sinks. Available sinks: {list(self.sinks.keys())}"
                )
        return self

    @model_validator(mode="after")
    def validate_globally_unique_node_names(self) -> "ElspethSettings":
        """Ensure all processing node names are unique across types.

        Node names become node IDs in the DAG and appear in audit records.
        A name collision between a transform and a gate (for example) would
        create ambiguous audit entries and routing errors.
        """
        all_names: list[tuple[str, str]] = []
        for t in self.transforms:
            all_names.append((t.name, "transform"))
        for g in self.gates:
            all_names.append((g.name, "gate"))
        for a in self.aggregations:
            all_names.append((a.name, "aggregation"))
        for c in self.coalesce:
            all_names.append((c.name, "coalesce"))
        for sink_name in self.sinks:
            all_names.append((sink_name, "sink"))

        seen: dict[str, str] = {}
        for name, node_type in all_names:
            if name in seen:
                raise ValueError(
                    f"Node name '{name}' is used by both {seen[name]} and {node_type}. "
                    f"All node names must be unique across transforms, gates, "
                    f"aggregations, coalesce nodes, and sinks."
                )
            seen[name] = node_type
        return self

    @model_validator(mode="after")
    def validate_replay_from(self) -> "ElspethSettings":
        """Ensure replay_from is set when mode requires it.

        Replay and verify modes need a source run ID to replay/compare against.
        Live mode does not require (and ignores) replay_from.
        """
        if self.run_mode in (RunMode.REPLAY, RunMode.VERIFY) and not self.replay_from:
            raise ValueError(f"replay_from is required when run_mode is '{self.run_mode.value}'")
        return self

    @field_validator("sinks")
    @classmethod
    def validate_sinks_not_empty(cls, v: dict[str, SinkSettings]) -> dict[str, SinkSettings]:
        """At least one sink is required."""
        if not v:
            raise ValueError("At least one sink is required")
        return v

    @field_validator("sinks")
    @classmethod
    def validate_sink_names_lowercase(cls, v: dict[str, SinkSettings]) -> dict[str, SinkSettings]:
        """Sink names must be lowercase identifiers.

        This is enforced explicitly rather than silently normalized to:
        1. Fail fast with clear error messages
        2. Avoid case mismatches between keys and references
        3. Ensure consistency with environment variable overrides (which are uppercased by Dynaconf)
        """
        non_lowercase = [name for name in v if name != name.lower()]
        if non_lowercase:
            # Provide helpful suggestions
            suggestions = [f"'{name}' -> '{name.lower()}'" for name in non_lowercase]
            raise ValueError(f"Sink names must be lowercase. Found: {non_lowercase}. Suggested fixes: {', '.join(suggestions)}")

        for sink_name in v:
            _validate_max_length(sink_name, field_label="Sink name", max_length=_MAX_NODE_NAME_LENGTH)
            _validate_connection_name_chars(sink_name, field_label="Sink name")
            if sink_name in _RESERVED_EDGE_LABELS:
                raise ValueError(f"Sink name '{sink_name}' is reserved. Reserved sink/edge labels: {sorted(_RESERVED_EDGE_LABELS)}")
            if sink_name.startswith("__"):
                raise ValueError(f"Sink name '{sink_name}' starts with '__', which is reserved for system edges")
        return v


# Regex pattern for ${VAR} or ${VAR:-default} syntax
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively expand ${VAR} and ${VAR:-default} patterns in config values.

    Args:
        config: Configuration dict (may contain nested structures)

    Returns:
        New dict with environment variables expanded
    """
    import os

    def _expand_string(value: str) -> str:
        """Expand ${VAR} patterns in a string."""

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)  # None if no default specified
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            # No env var and no default - fail fast with clear error
            raise ValueError(
                f"Required environment variable '{var_name}' is not set. "
                f"Either set the variable or use ${{{{var_name}}:-default}} syntax for optional values."
            )

        return _ENV_VAR_PATTERN.sub(replacer, value)

    def _expand_value(value: Any) -> Any:
        """Expand env vars in a single value."""
        if isinstance(value, str):
            return _expand_string(value)
        elif isinstance(value, dict):
            return {k: _expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_expand_value(item) for item in value]
        else:
            return value

    return {k: _expand_value(v) for k, v in config.items()}


# Secret field names that should be fingerprinted (exact matches, case-insensitive)
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "api-key",
        "authorization",
        "connection_string",
        "credential",
        "password",
        "secret",
        "token",
        "x-api-key",
    }
)

# Secret field suffixes that should be fingerprinted (case-insensitive)
_SECRET_FIELD_SUFFIXES = ("_secret", "_key", "_token", "_password", "_credential", "_connection_string")


def _is_secret_field(field_name: str) -> bool:
    """Check if a field name represents a secret that should be fingerprinted."""
    normalized = field_name.lower()
    return normalized in _SECRET_FIELD_NAMES or normalized.endswith(_SECRET_FIELD_SUFFIXES)


def _fingerprint_secrets(
    options: dict[str, Any],
    *,
    fail_if_no_key: bool = True,
) -> dict[str, Any]:
    """Recursively replace secret fields with their fingerprints.

    Walks nested dicts and lists to find and fingerprint all secret fields,
    not just top-level ones.

    Args:
        options: Plugin options dict (may contain nested structures)
        fail_if_no_key: If True, raise if ELSPETH_FINGERPRINT_KEY not set
                        and secrets are found. If False, redact secrets
                        without fingerprinting (for dev mode).

    Returns:
        New dict with secrets replaced by fingerprints (or redacted)

    Raises:
        SecretFingerprintError: If secrets found but no fingerprint key available
                                and fail_if_no_key is True
    """
    from elspeth.core.security import get_fingerprint_key, secret_fingerprint

    # Check if we have a fingerprint key available
    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    def _process_value(key: str, value: Any) -> tuple[str, Any, bool]:
        """Process a single value, returning (new_key, new_value, was_secret)."""
        if isinstance(value, dict):
            return key, _recurse(value), False
        elif isinstance(value, list):
            return key, [_process_value("", item)[1] for item in value], False
        elif isinstance(value, str) and _is_secret_field(key):
            # This is a secret field
            if have_key:
                fp = secret_fingerprint(value)
                return f"{key}_fingerprint", fp, True
            elif fail_if_no_key:
                raise SecretFingerprintError(
                    f"Secret field '{key}' found but ELSPETH_FINGERPRINT_KEY "
                    "is not set. Either set the environment variable or use "
                    "ELSPETH_ALLOW_RAW_SECRETS=true for development "
                    "(not recommended for production)."
                )
            else:
                # Dev mode: keep original value (user explicitly opted in)
                return key, value, False
        else:
            return key, value, False

    def _recurse(d: dict[str, Any]) -> dict[str, Any]:
        # Detect collision: a secret field and its _fingerprint counterpart both present.
        # Without this check, the pre-existing _fingerprint value would silently overwrite
        # the computed HMAC, allowing an attacker to inject a fake fingerprint.
        for key in d:
            if isinstance(d[key], str) and _is_secret_field(key):
                fp_key = f"{key}_fingerprint"
                if fp_key in d:
                    raise SecretFingerprintError(
                        f"Config contains both '{key}' and '{fp_key}'. "
                        f"The '{fp_key}' field is auto-generated from '{key}' during "
                        f"fingerprinting — remove '{fp_key}' from your configuration."
                    )
        result = {}
        for key, value in d.items():
            new_key, new_value, _was_secret = _process_value(key, value)
            result[new_key] = new_value
        return result

    return _recurse(options)


def _sanitize_dsn(
    url: str,
    *,
    fail_if_no_key: bool = True,
) -> tuple[str, str | None, bool]:
    """Sanitize a database connection URL by removing/fingerprinting the password.

    Args:
        url: Database connection URL (SQLAlchemy format)
        fail_if_no_key: If True, raise if password found but no fingerprint key.
                        If False (dev mode), just remove password without fingerprint.

    Returns:
        Tuple of (sanitized_url, password_fingerprint or None, had_password)
        The third element indicates whether the original URL had a password.

    Raises:
        SecretFingerprintError: If password found, no key available,
                                and fail_if_no_key=True

    Example:
        >>> _sanitize_dsn("postgresql://user:secret@host/db")
        ("postgresql://user@host/db", "abc123...", True)
    """
    from sqlalchemy.engine import URL
    from sqlalchemy.engine.url import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        parsed = make_url(url)
    except ArgumentError:
        # Not a valid SQLAlchemy URL - return as-is (might be a path or other format)
        return url, None, False

    if parsed.password is None:
        # No password in URL
        return url, None, False

    # Check if we have a fingerprint key
    from elspeth.core.security import get_fingerprint_key

    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    # Compute fingerprint if we have a key
    password_fingerprint = None
    if have_key:
        from elspeth.core.security import secret_fingerprint

        password_fingerprint = secret_fingerprint(parsed.password)
    elif fail_if_no_key:
        raise SecretFingerprintError(
            "Database URL contains a password but ELSPETH_FINGERPRINT_KEY "
            "is not set. Either set the environment variable or use "
            "ELSPETH_ALLOW_RAW_SECRETS=true for development "
            "(not recommended for production)."
        )
    # else: dev mode - just remove password without fingerprint

    # Reconstruct URL without password using URL.create()
    # NOTE: Do NOT use parsed.set(password=None) - it replaces with '***' not removal
    sanitized = URL.create(
        drivername=parsed.drivername,
        username=parsed.username,
        password=None,
        host=parsed.host,
        port=parsed.port,
        database=parsed.database,
        query=parsed.query,
    )

    return str(sanitized), password_fingerprint, True


def _expand_config_templates(
    raw_config: dict[str, Any],
    settings_path: Path | None = None,
) -> dict[str, Any]:
    """Expand template_file and lookup_file references in config.

    This function is called at load time to expand file references into
    their contents. Secrets are NOT fingerprinted here - that happens
    in resolve_config() when creating the audit copy.

    Args:
        raw_config: Raw config dict from Dynaconf
        settings_path: Path to settings file for resolving relative template paths

    Returns:
        Config with template files expanded (secrets still present)

    Raises:
        TemplateFileError: If template/lookup files not found or invalid
    """
    if settings_path is None:
        return raw_config

    config = dict(raw_config)

    # === Transform plugin options - expand template files ===
    if "transforms" in config and isinstance(config["transforms"], list):
        plugins = []
        for plugin_config in config["transforms"]:
            if isinstance(plugin_config, dict):
                plugin = dict(plugin_config)
                if "options" in plugin and isinstance(plugin["options"], dict):
                    plugin["options"] = _expand_template_files(plugin["options"], settings_path)
                plugins.append(plugin)
            else:
                plugins.append(plugin_config)
        config["transforms"] = plugins

    # === Aggregation options - expand template files ===
    if "aggregations" in config and isinstance(config["aggregations"], list):
        aggregations = []
        for agg_config in config["aggregations"]:
            if isinstance(agg_config, dict):
                agg = dict(agg_config)
                if "options" in agg and isinstance(agg["options"], dict):
                    agg["options"] = _expand_template_files(agg["options"], settings_path)
                aggregations.append(agg)
            else:
                aggregations.append(agg_config)
        config["aggregations"] = aggregations

    return config


def _fingerprint_config_for_audit(
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint secrets in config for audit storage.

    Called by resolve_config() to create a copy safe for audit storage.
    The original config (with secrets) is untouched.

    Processes:
    - source.options
    - sinks.*.options
    - transforms[*].options
    - aggregations[*].options
    - landscape.url (DSN password)

    Args:
        config_dict: Config dict to fingerprint (will be copied)

    Returns:
        Deep copy with secrets fingerprinted

    Raises:
        SecretFingerprintError: If secrets found but no fingerprint key
                                and ELSPETH_ALLOW_RAW_SECRETS is not set
    """
    import copy
    import os

    # Check dev mode override
    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"
    fail_if_no_key = not allow_raw

    # Deep copy to avoid mutating the original
    config = copy.deepcopy(config_dict)

    # === Landscape URL (DSN password) ===
    if "landscape" in config and isinstance(config["landscape"], dict):
        landscape = config["landscape"]
        if "url" in landscape and isinstance(landscape["url"], str):
            # _sanitize_dsn returns (sanitized_url, fingerprint, had_password)
            sanitized_url, password_fp, had_password = _sanitize_dsn(
                landscape["url"],
                fail_if_no_key=fail_if_no_key,
            )
            landscape["url"] = sanitized_url
            if password_fp:
                landscape["url_password_fingerprint"] = password_fp
            elif had_password and not fail_if_no_key:
                # Dev mode: password was removed but not fingerprinted
                landscape["url_password_redacted"] = True

    # === Source options ===
    if "source" in config and isinstance(config["source"], dict):
        ds = config["source"]
        if "options" in ds and isinstance(ds["options"], dict):
            ds["options"] = _fingerprint_secrets(ds["options"], fail_if_no_key=fail_if_no_key)

    # === Sink options ===
    if "sinks" in config and isinstance(config["sinks"], dict):
        for sink in config["sinks"].values():
            if isinstance(sink, dict) and "options" in sink and isinstance(sink["options"], dict):
                sink["options"] = _fingerprint_secrets(sink["options"], fail_if_no_key=fail_if_no_key)

    # === Transform plugin options ===
    if "transforms" in config and isinstance(config["transforms"], list):
        for plugin in config["transforms"]:
            if isinstance(plugin, dict) and "options" in plugin and isinstance(plugin["options"], dict):
                plugin["options"] = _fingerprint_secrets(plugin["options"], fail_if_no_key=fail_if_no_key)

    # === Aggregation options ===
    if "aggregations" in config and isinstance(config["aggregations"], list):
        for agg in config["aggregations"]:
            if isinstance(agg, dict) and "options" in agg and isinstance(agg["options"], dict):
                agg["options"] = _fingerprint_secrets(agg["options"], fail_if_no_key=fail_if_no_key)

    # === Telemetry exporter options ===
    if "telemetry" in config and isinstance(config["telemetry"], dict):
        telemetry = config["telemetry"]
        exporters = telemetry.get("exporters")
        if isinstance(exporters, list):
            for exporter in exporters:
                if isinstance(exporter, dict) and "options" in exporter and isinstance(exporter["options"], dict):
                    exporter["options"] = _fingerprint_secrets(exporter["options"], fail_if_no_key=fail_if_no_key)

    return config


class TemplateFileError(Exception):
    """Error loading template or lookup file."""


def _resolve_template_path(file_ref: str, settings_path: Path, label: str) -> Path:
    """Resolve a template/lookup/prompt file path with containment check.

    Relative paths are resolved against the settings file's parent directory.
    Absolute paths are used as-is. In both cases, the resolved path must remain
    within the settings directory tree to prevent path traversal attacks.

    Args:
        file_ref: File reference from config (relative or absolute)
        settings_path: Path to the settings file
        label: Human-readable label for error messages (e.g. "Template file")

    Returns:
        Resolved, validated Path

    Raises:
        TemplateFileError: If path escapes settings directory or file not found
    """
    config_root = settings_path.parent.resolve()
    file_path = Path(file_ref)
    if not file_path.is_absolute():
        file_path = (config_root / file_path).resolve()
    else:
        file_path = file_path.resolve()

    # Containment check: resolved path must be under the config directory
    try:
        file_path.relative_to(config_root)
    except ValueError:
        raise TemplateFileError(
            f"{label} path traversal blocked: {file_ref!r} resolves to {file_path} which is outside config directory {config_root}"
        ) from None

    if not file_path.exists():
        raise TemplateFileError(f"{label} not found: {file_path}")

    return file_path


def _expand_template_files(
    options: dict[str, Any],
    settings_path: Path,
) -> dict[str, Any]:
    """Expand template_file, lookup_file, and system_prompt_file to loaded content.

    Args:
        options: Plugin options dict
        settings_path: Path to settings file for resolving relative paths

    Returns:
        New dict with files loaded and paths recorded:
        - template_file → template (content) + template_source (path)
        - lookup_file → lookup (content) + lookup_source (path)
        - system_prompt_file → system_prompt (content) + system_prompt_source (path)

    Raises:
        TemplateFileError: If files not found, invalid, or path traversal detected
    """
    result = dict(options)

    # Handle template_file
    if "template_file" in result:
        if "template" in result:
            raise TemplateFileError("Cannot specify both 'template' and 'template_file'")
        template_file = result.pop("template_file")
        template_path = _resolve_template_path(template_file, settings_path, "Template file")

        result["template"] = template_path.read_text(encoding="utf-8")
        result["template_source"] = template_file

    # Handle lookup_file
    if "lookup_file" in result:
        if "lookup" in result:
            raise TemplateFileError("Cannot specify both 'lookup' and 'lookup_file'")
        lookup_file = result.pop("lookup_file")
        lookup_path = _resolve_template_path(lookup_file, settings_path, "Lookup file")

        try:
            loaded = yaml.safe_load(lookup_path.read_text(encoding="utf-8"))
            # Coerce None (empty file) to {} so it gets a distinct hash from "no lookup"
            # This ensures empty lookup files are auditable as "intentionally empty"
            result["lookup"] = loaded if loaded is not None else {}
        except yaml.YAMLError as e:
            raise TemplateFileError(f"Invalid YAML in lookup file: {e}") from e

        result["lookup_source"] = lookup_file

    # Handle system_prompt_file
    if "system_prompt_file" in result:
        if "system_prompt" in result:
            raise TemplateFileError("Cannot specify both 'system_prompt' and 'system_prompt_file'")
        system_prompt_file = result.pop("system_prompt_file")
        system_prompt_path = _resolve_template_path(system_prompt_file, settings_path, "System prompt file")

        result["system_prompt"] = system_prompt_path.read_text(encoding="utf-8")
        result["system_prompt_source"] = system_prompt_file

    return result


def _lowercase_schema_keys(obj: Any, *, _preserve_nested: bool = False, _in_sinks: bool = False) -> Any:
    """Lowercase dictionary keys for Pydantic schema matching, preserving user data.

    Dynaconf returns keys in UPPERCASE when they come from environment variables,
    but Pydantic expects lowercase field names. User data inside 'options' and
    'routes' dicts must be preserved exactly as written - these contain
    case-sensitive keys that must match runtime values:
    - options: {"Score": "score"} where "Score" must match the LLM's JSON field name
    - routes: {"High": "quality_ok"} where "High" must match the gate condition result

    Sink name handling:
    - FULLY UPPERCASE names (e.g., 'OUTPUT') are lowercased - these come from
      env vars like ELSPETH_SINKS__OUTPUT__PLUGIN where Dynaconf uppercases
    - Mixed-case names (e.g., 'MySink') are PRESERVED so the validator can
      catch them and give a helpful "use lowercase" error

    Args:
        obj: Any value - dicts are processed recursively, lists have their
             elements processed, other types pass through unchanged.
        _preserve_nested: Internal flag - when True, stop lowercasing keys
             (we're inside an 'options' or 'routes' dict).
        _in_sinks: Internal flag - when True, we're processing sink name keys.

    Returns:
        The input with schema-level dict keys lowercased, but user data preserved.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Determine the new key
            if _preserve_nested:
                # Inside options: preserve all keys exactly
                new_key = k
            elif _in_sinks:
                # Sink names: lowercase only if FULLY UPPERCASE (env var origin)
                # Preserve mixed-case so validator can catch and give helpful error
                new_key = k.lower() if k.isupper() else k
            else:
                # Normal schema keys: always lowercase
                new_key = k.lower()

            # Determine how to process children
            if _preserve_nested:
                # Already inside options/routes: stay in preserve mode, ignore special keys
                child = _lowercase_schema_keys(v, _preserve_nested=True, _in_sinks=False)
            elif new_key == "options":
                # Options: preserve everything inside (user data)
                child = _lowercase_schema_keys(v, _preserve_nested=True, _in_sinks=False)
            elif new_key == "routes":
                # Routes: preserve everything inside (user-defined route labels)
                child = _lowercase_schema_keys(v, _preserve_nested=True, _in_sinks=False)
            elif new_key == "sinks":
                # Entering sinks dict: next level has sink name keys
                child = _lowercase_schema_keys(v, _preserve_nested=False, _in_sinks=True)
            elif _in_sinks:
                # At sink name level: value is SinkSettings, resume normal lowercasing
                child = _lowercase_schema_keys(v, _preserve_nested=False, _in_sinks=False)
            else:
                # Normal recursion
                child = _lowercase_schema_keys(v, _preserve_nested=_preserve_nested, _in_sinks=False)

            result[new_key] = child
        return result
    if isinstance(obj, list):
        return [_lowercase_schema_keys(item, _preserve_nested=_preserve_nested, _in_sinks=_in_sinks) for item in obj]
    return obj


def load_settings(config_path: Path) -> ElspethSettings:
    """Load settings from YAML file with environment variable overrides.

    Uses Dynaconf for multi-source loading with precedence:
    1. Environment variables (ELSPETH_*) - highest priority
    2. Config file (settings.yaml)
    3. Defaults from Pydantic schema - lowest priority

    Environment variable format: ELSPETH_DATABASE__URL for nested keys.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated ElspethSettings instance

    Raises:
        ValidationError: If configuration fails Pydantic validation
        FileNotFoundError: If config file doesn't exist
    """
    from dynaconf import Dynaconf  # type: ignore[attr-defined]  # dynaconf has no type stubs

    # Explicit check for file existence (Dynaconf silently accepts missing files)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load from file + environment
    dynaconf_settings = Dynaconf(
        envvar_prefix="ELSPETH",
        settings_files=[str(config_path)],
        environments=False,  # No [default]/[production] sections
        load_dotenv=False,  # Don't auto-load .env
        merge_enabled=True,  # Deep merge nested dicts
    )

    # Dynaconf returns uppercase keys; convert to lowercase for Pydantic
    raw_dict = dynaconf_settings.as_dict()
    raw_config = _lowercase_schema_keys(raw_dict)

    # Explicitly reject removed default_sink in YAML before allowlist filtering.
    # This MUST happen before the allowlist (which would silently strip it).
    if "default_sink" in raw_config:
        raise ValueError(
            "'default_sink' has been removed. Use explicit 'on_success' routing instead.\n"
            "Migration: Set top-level 'source.on_success: <sink_or_connection_name>' and "
            "top-level 'transforms[].on_success: <sink_or_connection_name>' (not inside "
            "plugin options). Ensure each terminal path routes to a sink.\n"
            "Then remove the 'default_sink' line from your pipeline YAML."
        )

    # Positive allowlist: only pass keys that ElspethSettings knows about.
    # Dynaconf injects internal settings (LOAD_DOTENV, ENVIRONMENTS, SETTINGS_FILES,
    # MERGE_ENABLED, ALLOW_RAW_SECRETS, etc.) which must be excluded. A positive
    # allowlist is robust against Dynaconf version changes — no whack-a-mole.
    known_fields = set(ElspethSettings.model_fields.keys())
    raw_config = {k: v for k, v in raw_config.items() if k in known_fields}

    # Expand ${VAR} and ${VAR:-default} patterns in config values
    raw_config = _expand_env_vars(raw_config)

    # Expand template files in plugin options before validation
    # NOTE: Secrets are NOT fingerprinted here - they stay available for runtime.
    # Fingerprinting happens in resolve_config() when creating the audit copy.
    raw_config = _expand_config_templates(raw_config, settings_path=config_path)

    return ElspethSettings(**raw_config)


def resolve_config(settings: ElspethSettings) -> dict[str, Any]:
    """Convert validated settings to a dict for audit storage.

    This is the resolved configuration that gets stored in Landscape
    for reproducibility. It includes all settings (explicit + defaults).

    IMPORTANT: This function fingerprints secrets before returning.
    The returned dict is safe for audit storage but should NOT be used
    for runtime operations that need actual secret values.

    Args:
        settings: Validated ElspethSettings instance

    Returns:
        Dict representation suitable for JSON serialization (secrets fingerprinted)
    """
    config_dict = settings.model_dump(mode="json")
    # Fingerprint secrets for audit storage
    return _fingerprint_config_for_audit(config_dict)
