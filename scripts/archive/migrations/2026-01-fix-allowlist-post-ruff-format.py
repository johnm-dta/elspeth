#!/usr/bin/env python3
"""Update no_bug_hiding allowlist with new line numbers."""

from pathlib import Path

import yaml

# Stale entries to remove (old line numbers that no longer exist)
STALE_KEYS = [
    "engine/orchestrator.py:R4:Orchestrator:_cleanup_transforms:line=209",
    "engine/orchestrator.py:R1:Orchestrator:_validate_route_destinations:line=246",
    "engine/orchestrator.py:R1:Orchestrator:_validate_route_destinations:line=253",
    "engine/orchestrator.py:R1:Orchestrator:_validate_route_destinations:line=269",
    "engine/orchestrator.py:R2:Orchestrator:_validate_source_quarantine_destination:line=342",
    "engine/orchestrator.py:R1:Orchestrator:_execute_run:line=672",
    "engine/orchestrator.py:R2:Orchestrator:resume:line=1334",
    "cli.py:R1:health:line=1456",
    "cli.py:R4:health:line=1468",
    "cli.py:R1:health:line=1484",
    "cli.py:R4:health:line=1496",
    "cli.py:R4:health:line=1560",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=350",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=356",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=357",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=363",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=369",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=371",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=377",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=382",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=388",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=392",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=393",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=398",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=399",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=417",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=418",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=423",
    "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=424",
    "core/dag.py:R1:ExecutionGraph:_get_missing_required_fields:line=821",
    "core/dag.py:R1:ExecutionGraph:_get_missing_required_fields:line=822",
]

# New entries to add (updated line numbers AFTER ruff format)
NEW_ENTRIES = [
    {
        "key": "cli.py:R1:health:line=1517",
        "owner": "architecture",
        "reason": "Trust boundary: environment variable GIT_COMMIT_SHA is optional external input",
        "safety": "Empty string default is safe - commit SHA is informational only",
        "expires": None,
    },
    {
        "key": "cli.py:R4:health:line=1529",
        "owner": "architecture",
        "reason": "Trust boundary: git subprocess may fail for many reasons (not installed, not a repo, etc.)",
        "safety": "Graceful fallback to 'unknown' - health check reports git as unavailable",
        "expires": None,
    },
    {
        "key": "cli.py:R1:health:line=1545",
        "owner": "architecture",
        "reason": "Trust boundary: environment variable DATABASE_URL is optional external input",
        "safety": "Empty string default triggers skip logic - database check only runs if configured",
        "expires": None,
    },
    {
        "key": "cli.py:R4:health:line=1557",
        "owner": "architecture",
        "reason": "Trust boundary: database connection may fail for many reasons (network, auth, etc.)",
        "safety": "Error recorded in checks dict with status='error' - health check reports DB as unhealthy",
        "expires": None,
    },
    {
        "key": "cli.py:R4:health:line=1621",
        "owner": "architecture",
        "reason": "Trust boundary: plugin manager initialization may fail due to misconfiguration",
        "safety": "Error recorded in checks dict with status='error' - health check reports plugins as unhealthy",
        "expires": None,
    },
    {
        "key": "engine/orchestrator.py:R4:Orchestrator:_cleanup_transforms:line=214",
        "owner": "architecture",
        "reason": "Best-effort cleanup: one plugin failure shouldn't prevent others from cleanup",
        "safety": "Logged with warning, other plugins still get close() called",
        "expires": None,
    },
    {
        "key": "engine/orchestrator.py:R1:Orchestrator:_validate_route_destinations:line=251",
        "owner": "architecture",
        "reason": "Building lookup for gate names - only gates (not transforms) are in map",
        "safety": "None means 'not a gate at this sequence' - correctly skipped",
        "expires": None,
    },
    {
        "key": "engine/orchestrator.py:R1:Orchestrator:_validate_route_destinations:line=258",
        "owner": "architecture",
        "reason": "Building lookup for gate names - only gates (not transforms) are in map",
        "safety": "None means 'not a gate at this sequence' - correctly skipped",
        "expires": None,
    },
    {
        "key": "engine/orchestrator.py:R1:Orchestrator:_validate_route_destinations:line=274",
        "owner": "architecture",
        "reason": "Fallback for error message - use node_id if gate name not found",
        "safety": "Error message still produced with node_id as fallback",
        "expires": None,
    },
    {
        "key": "engine/orchestrator.py:R2:Orchestrator:_validate_source_quarantine_destination:line=347",
        "owner": "architecture",
        "reason": "Optional protocol extension - _on_validation_failure only on sources with SourceDataConfig",
        "safety": "None default means source doesn't support quarantine routing - validation skipped",
        "expires": None,
    },
    {
        "key": "engine/orchestrator.py:R1:Orchestrator:_execute_run:line=680",
        "owner": "architecture",
        "reason": "Optional schema config with explicit default - not all nodes define schemas",
        "safety": "Default value is well-defined: dynamic fields",
        "expires": None,
    },
    {
        "key": "engine/orchestrator.py:R2:Orchestrator:resume:line=1345",
        "owner": "architecture",
        "reason": "Optional protocol attribute: _schema_class only exists on sources with Pydantic schemas",
        "safety": "None default means source uses dynamic schema - type restoration skipped, backward compatible",
        "expires": None,
    },
    # Post-ruff-format line numbers for dag.py
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=365",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have config attribute (legacy plugins)",
        "safety": "Empty dict default is safe - used for deterministic node ID generation",
        "expires": None,
    },
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=372",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have output_schema attribute (dynamic schemas)",
        "safety": "None default is safe - missing schema means dynamic schema",
        "expires": None,
    },
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=378",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have config attribute (legacy plugins)",
        "safety": "Empty dict default is safe - used for deterministic node ID generation",
        "expires": None,
    },
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=386",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have input_schema attribute (dynamic schemas)",
        "safety": "None default is safe - missing schema means dynamic schema",
        "expires": None,
    },
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=397",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have config attribute (legacy plugins)",
        "safety": "Empty dict default is safe - used for deterministic node ID generation",
        "expires": None,
    },
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=407",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have input_schema attribute (dynamic schemas)",
        "safety": "None default is safe - missing schema means dynamic schema",
        "expires": None,
    },
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=408",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have output_schema attribute (dynamic schemas)",
        "safety": "None default is safe - missing schema means dynamic schema",
        "expires": None,
    },
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=432",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have input_schema attribute (dynamic schemas)",
        "safety": "None default is safe - missing schema means dynamic schema",
        "expires": None,
    },
    {
        "key": "core/dag.py:R2:ExecutionGraph:from_plugin_instances:line=433",
        "owner": "architecture",
        "reason": "Trust boundary: plugin instances may not have output_schema attribute (dynamic schemas)",
        "safety": "None default is safe - missing schema means dynamic schema",
        "expires": None,
    },
]


def main() -> None:
    allowlist_path = Path("config/cicd/no_bug_hiding.yaml")

    # Load current allowlist
    with open(allowlist_path) as f:
        data = yaml.safe_load(f)

    # Remove stale entries
    original_count = len(data["allow_hits"])
    data["allow_hits"] = [entry for entry in data["allow_hits"] if entry["key"] not in STALE_KEYS]
    removed_count = original_count - len(data["allow_hits"])
    print(f"Removed {removed_count} stale entries")

    # Add new entries
    for new_entry in NEW_ENTRIES:
        # Check if entry already exists
        if not any(e["key"] == new_entry["key"] for e in data["allow_hits"]):
            data["allow_hits"].append(new_entry)
            print(f"Added: {new_entry['key']}")

    # Write back
    with open(allowlist_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("\nAllowlist updated successfully!")
    print(f"Total entries: {len(data['allow_hits'])}")


if __name__ == "__main__":
    main()
