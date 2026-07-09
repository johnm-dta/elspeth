"""Ownership checks for composer helper modules.

These tests keep sibling modules from reaching back into ``service.py`` for
private implementation helpers after those helpers have been given cohesive
owner modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_SERVICE_MODULE = "elspeth.web.composer.service"


def _imports_from_service(relative_path: str) -> set[str]:
    tree = ast.parse((_ROOT / relative_path).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _SERVICE_MODULE:
            imported.update(alias.name for alias in node.names)
    return imported


def test_availability_owns_provider_contracts() -> None:
    service_imports = _imports_from_service("src/elspeth/web/composer/availability.py")

    assert (
        not {
            "_PROVIDER_REQUIRED_ENV_KEYS",
            "_infer_provider_from_model_name",
            "_infer_provider_from_unprefixed_model_name",
        }
        & service_imports
    )


def test_tool_batch_owns_discovery_cache_helpers() -> None:
    service_imports = _imports_from_service("src/elspeth/web/composer/tool_batch.py")

    assert (
        not {
            "_CachedDiscoveryPayload",
            "_RuntimePreflightCache",
            "_MAX_PENDING_PROPOSALS_PER_TURN",
            "_arg_error_payload",
            "_cached_discovery_payload",
            "_make_cache_key",
            "_result_from_cached_discovery_payload",
            "_serialize_tool_result",
            "_tool_result_mutated_composition_state",
        }
        & service_imports
    )


def test_turn_audit_uses_tool_error_payload_owner() -> None:
    service_imports = _imports_from_service("src/elspeth/web/composer/turn_audit.py")

    assert "_INVALID_TOOL_ARGUMENTS_REDACTION_STATUS" not in service_imports


def test_guided_discovery_uses_result_serializer_owner() -> None:
    service_imports = _imports_from_service("src/elspeth/web/composer/guided/_discovery.py")

    assert "_serialize_tool_result" not in service_imports


def test_no_tool_finalize_uses_no_tool_policy_owner() -> None:
    service_imports = _imports_from_service("src/elspeth/web/composer/no_tool_finalize.py")

    assert (
        not {
            "_blocking_result_from_tool_invocations",
            "_compose_empty_state_message",
            "_compose_preflight_failure_message",
            "_enforce_augmentation_prefix_invariant",
            "_is_pending_interpretation_handoff",
            "_last_mutation_was_pending_proposal",
            "_no_mutation_empty_state_validation",
            "_RuntimePreflightCache",
            "_state_is_structurally_empty",
            "_user_request_expects_pipeline_mutation",
        }
        & service_imports
    )
