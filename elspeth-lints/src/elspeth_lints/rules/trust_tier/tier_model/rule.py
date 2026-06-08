#!/usr/bin/env python3
"""
Tier Model Enforcement Tool

AST-based static analysis that detects defensive programming patterns that
violate the three-tier trust model and fails CI unless explicitly allowlisted.

ADR-023 documents why ELSPETH-specific invariants like this remain custom
Python analyzer rules instead of CodeQL/Semgrep/ast-grep queries:
docs/architecture/adr/023-custom-python-ci-analyzer.md

Enforces ELSPETH's data manifesto (see CLAUDE.md):
- Tier 1 (Audit Database): Full trust - crash on any anomaly
- Tier 2 (Pipeline Data): Elevated trust - expect types, wrap operations on values
- Tier 3 (External Data): Zero trust - validate at boundary, coerce where possible

Usage:
    python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
    python -m elspeth_lints.core.cli dump-edges --root src/elspeth --format json --output edges.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from calendar import monthrange
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, ClassVar

from elspeth_lints.core.allowlist import Allowlist as Allowlist
from elspeth_lints.core.allowlist import (
    AllowlistBudgetViolation as AllowlistBudgetViolation,
)
from elspeth_lints.core.allowlist import AllowlistEntry as AllowlistEntry
from elspeth_lints.core.allowlist import (
    FindingKey,
    find_scope_fallback_entry,
    load_allowlist,
    verify_entry_binding_against_finding,
)
from elspeth_lints.core.allowlist import PerFileRule as PerFileRule
from elspeth_lints.core.ast_walker import iter_own_scope
from elspeth_lints.core.protocols import (
    Finding as LintFinding,
)
from elspeth_lints.core.protocols import RuleContext, RuleMetadata, RuleScope, Severity
from elspeth_lints.rules.trust_tier.tier_model.metadata import RULE_ID, RULE_METADATA
from elspeth_lints.rules.trust_tier.tier_model.scope_fingerprint import (
    compute_scope_fingerprint,
    enclosing_scope_node,
)
from elspeth_lints.rules.trust_tier.tier_model.trust_boundary_suppress import (
    BoundaryFinding,
    BoundaryMetadata,
    DerivedNameState,
    assignment_target_names,
    extract_boundary_metadata,
    subject_is_rooted,
)

# =============================================================================
# Data Structures
# =============================================================================


def _add_months(today: date, months: int) -> date:
    """Return a date months after today, clamped to month length."""
    month_index = today.month - 1 + months
    year = today.year + month_index // 12
    month = month_index % 12 + 1
    day = min(today.day, monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class Finding:
    """A detected violation of the no-bug-hiding policy.

    ``ast_path`` is the AST field/index path from the module root to the
    finding's subject node, joined with ``"/"`` (e.g.
    ``"body[3]/body[1]/value/args[0]"``). It is the AST-level address the
    fingerprint already encodes; surfacing it explicitly is what closes
    the C8-3 quartet-transplant attack — the loader/matcher pair (in
    ``elspeth_lints.core.allowlist``) verifies that an allowlist entry
    carrying judge metadata still points at the same AST node it was
    judged against. For layer-import findings (which have no AST subject)
    we synthesise a stable address of the form ``import:<module>`` so
    every Finding instance carries a non-empty ast_path.

    ``scope_fingerprint`` is the v2 binding primitive — the 64-char hex
    fingerprint of the innermost enclosing scope (FunctionDef /
    AsyncFunctionDef / ClassDef, or the whole module at module level) of
    the finding's subject node. It replaces the whole-file
    ``file_fingerprint`` for judge-gated entries, so editing an unrelated
    scope in the same file no longer invalidates an entry's signature. It
    is computed *forward* by the visitor from the live ``node_stack``
    (never reverse-resolved from ``ast_path``) and verified at match time.

    ``scope_depth`` is K, the count of ``ast_path`` components strictly above
    the enclosing scope (the index of the enclosing scope node in
    ``node_stack``). The scope-relative suffix ``ast_path.split("/")[K:]`` is
    invariant under module-body shifts (adding/removing a module-level statement
    moves only the leading index, which lives in the first K components). It is
    the within-scope discriminator the allowlist key-match fallback uses to tell
    two same-rule findings in one scope apart. Module-level findings have no
    def/class scope, so K=0 and the fallback always returns None for them: any
    module-body edit (e.g. adding an import) changes the module's
    scope_fingerprint, so the scope-fingerprint-equality check fails.
    """

    rule_id: str
    file_path: str
    line: int
    col: int
    symbol_context: tuple[str, ...]  # e.g., ("ClassName", "method_name")
    fingerprint: str
    code_snippet: str
    message: str
    ast_path: str = ""
    scope_fingerprint: str = ""
    scope_depth: int = 0

    @property
    def canonical_key(self) -> str:
        """Generate the canonical key for allowlist matching."""
        symbol_part = ":".join(self.symbol_context) if self.symbol_context else "_module_"
        return f"{self.file_path}:{self.rule_id}:{symbol_part}:fp={self.fingerprint}"

    def suggested_allowlist_entry(self) -> dict[str, Any]:
        """Generate a suggested allowlist entry for this finding."""
        today = datetime.now(UTC).date()
        return {
            "key": self.canonical_key,
            "owner": "<your-name>",
            "reason": "<explain why this is at a trust boundary>",
            "safety": "<explain how failures are handled>",
            "expires": _add_months(today, 3).isoformat(),
        }


@dataclass(frozen=True)
class CheckResult:
    """Structured result for check-mode analysis."""

    allowlist_path: Path
    violations: list[Finding]
    suppressed_boundary_findings: list[Finding]
    stale_entries: list[AllowlistEntry]
    expired_entries: list[AllowlistEntry]
    expired_file_rules: list[PerFileRule]
    unused_file_rules: list[PerFileRule]
    layer_warnings: list[Finding]
    exceeded_file_rules: list[PerFileRule]
    budget_violations: list[AllowlistBudgetViolation]

    @property
    def has_errors(self) -> bool:
        """Return whether check-mode should fail."""
        return bool(
            self.violations
            or self.stale_entries
            or self.expired_entries
            or self.expired_file_rules
            or self.unused_file_rules
            or self.exceeded_file_rules
            or self.budget_violations
        )


# =============================================================================
# Rule Definitions
# =============================================================================

RULES: dict[str, dict[str, Any]] = {
    "R1": {
        "name": "dict.get",
        "description": "dict.get() usage can hide missing key bugs",
        "remediation": "Access dict keys directly (dict[key]) and fix the schema/contract if KeyError occurs",
    },
    "R2": {
        "name": "getattr",
        "description": "getattr() with default can hide missing attribute bugs",
        "remediation": "Access attributes directly (obj.attr) and fix the type/contract if AttributeError occurs",
    },
    "R3": {
        "name": "hasattr",
        "description": "hasattr() is banned — use isinstance, protocols, or try/except AttributeError",
        "remediation": "Replace with isinstance() for type checks, try/except AttributeError for attribute probing, or protocols for structural typing",
        "banned": True,
    },
    "R4": {
        "name": "broad-except",
        "description": "Broad exception handling can suppress bugs",
        "remediation": "Catch specific exceptions, or re-raise after logging/quarantining",
    },
    "TC": {
        "name": "type-checking-layer",
        "description": "TYPE_CHECKING import crosses layer boundary (annotation-only, no runtime coupling)",
        "remediation": "Allowlist if the dependency is accepted, or move the type to a lower layer",
    },
    "R5": {
        "name": "isinstance",
        "description": "isinstance() checks can mask contract violations outside explicit trust boundaries",
        "remediation": "Validate at Tier-3 boundaries or rely on contracts; do not use isinstance to hide bugs",
    },
    "R6": {
        "name": "silent-except",
        "description": "Exception handling that swallows errors without re-raise or explicit error result",
        "remediation": "Raise the exception or return an explicit error/quarantine result",
    },
    "R7": {
        "name": "contextlib.suppress",
        "description": "contextlib.suppress() silently ignores exceptions",
        "remediation": "Handle exceptions explicitly or allow them to raise",
    },
    "R8": {
        "name": "dict.setdefault",
        "description": "dict.setdefault() hides missing-key bugs by mutating defaults",
        "remediation": "Access keys directly and fix the schema/contract if KeyError occurs",
    },
    "R9": {
        "name": "dict.pop-default",
        "description": "dict.pop(key, default) hides missing-key bugs with implicit defaults",
        "remediation": "Access keys directly and fix the schema/contract if KeyError occurs",
    },
    "L1": {
        "name": "upward-import",
        "description": "Import from a higher layer violates the dependency hierarchy (contracts→core→engine→plugins)",
        "remediation": "Move code down, extract primitives, or restructure caller (see CLAUDE.md Layer Dependency Rules)",
    },
    # =========================================================================
    # @trust_boundary decorator hygiene
    # =========================================================================
    # These findings fire on the decorator itself, not on suppressed
    # patterns. They guarantee that a malformed ``@trust_boundary`` cannot
    # silently disable suppression analysis: if the analyzer cannot read the
    # decorator's metadata, the author hears about it AND the decorator is
    # treated as inert (so any rule violations inside the function remain
    # visible). Severity matches the rest of tier_model — ERROR — because a
    # malformed decorator defeats the suppression argument and must not be
    # merged.
    "R_TB_NONLITERAL": {
        "name": "trust-boundary-non-literal-arg",
        "description": (
            "@trust_boundary kwargs must be static literals; references to "
            "names, calls, or comprehensions defeat the static suppression analysis."
        ),
        "remediation": (
            "Inline the literal value(s) in the decorator call. Constants "
            "imported from another module are still literals at the call site "
            "because the decorator is keyword-only; replace any computed values."
        ),
    },
    "R_TB_MALFORMED": {
        "name": "trust-boundary-malformed-metadata",
        "description": ("@trust_boundary metadata is the wrong shape (e.g. 'suppresses' is not a tuple, 'source_param' is not a string)."),
        "remediation": (
            "Compare the call to the documented signature in "
            "src/elspeth/contracts/trust_boundary.py. 'suppresses' must be "
            "tuple[str, ...]; 'source_param' must be a non-empty string."
        ),
    },
    "R_TB_STACKED": {
        "name": "trust-boundary-stacked-decorator",
        "description": ("Multiple @trust_boundary decorators on one function make the boundary metadata ambiguous."),
        "remediation": (
            "Keep exactly one @trust_boundary decorator per function. Merge the "
            "intended source_param, suppresses, invariant, and test_ref into a "
            "single decorator or split the boundary into separate functions."
        ),
    },
    "R_TB_UNKNOWN_KWARG": {
        "name": "trust-boundary-unknown-kwarg",
        "description": ("@trust_boundary contains a kwarg outside the runtime decorator signature."),
        "remediation": (
            "Remove the unknown kwarg or correct its spelling. The analyzer "
            "treats the decorator as inert until its metadata matches "
            "src/elspeth/contracts/trust_boundary.py."
        ),
    },
}


def describe_rule(rule_id: str) -> str:
    """Render a one-line definition of ``rule_id`` for the cicd-judge prompt.

    The judge is rule-agnostic: it never hardcodes rule semantics. Callers
    pass the rule's own definition (sourced here from the authoritative
    ``RULES`` table) into the per-finding ``JudgeRequest`` so the judge can
    evaluate whether a rationale addresses the *specific* defensive pattern
    the analyzer flagged — without baking rule knowledge into the static,
    hashed policy block. Unknown ids degrade to a neutral label rather than
    raising: the judge still has the rule_id, it just lacks an expansion.
    """
    rule = RULES.get(rule_id)
    if rule is None:
        return f"{rule_id}: (no definition available; evaluate on the doctrine and code alone)"
    return f"{rule_id} ({rule['name']}): {rule['description']} Remediation: {rule['remediation']}"


# =============================================================================
# Layer Hierarchy (import direction enforcement)
# =============================================================================

# Layer numbers: lower = deeper (fewer allowed dependencies)
LAYER_HIERARCHY: dict[str, int] = {
    "contracts": 0,  # L0 — leaf, imports nothing above
    "core": 1,  # L1 — can import contracts only
    "engine": 2,  # L2 — can import core, contracts
}
# Everything else (plugins, mcp, tui, telemetry, testing, cli*) is implicitly L3.

LAYER_NAMES: dict[int, str] = {
    0: "L0/contracts",
    1: "L1/core",
    2: "L2/engine",
    3: "L3/application",
}


def _get_file_layer(relative_path: str) -> int:
    """Determine the layer from a path relative to the scan root.

    Supports both ``--root=src/elspeth`` (paths like ``contracts/...``) and
    ``--root=src`` (paths like ``elspeth/contracts/...``).
    """
    parts = relative_path.split("/")
    idx = 1 if parts[0] == "elspeth" else 0
    top = parts[idx] if idx < len(parts) else ""
    return LAYER_HIERARCHY.get(top, 3)


def _get_import_target_layer(module_name: str) -> int | None:
    """Determine the target layer from a fully qualified import.

    Returns None for non-elspeth imports.
    """
    if not module_name.startswith("elspeth."):
        return None
    parts = module_name.split(".")
    if len(parts) < 2:
        return None
    return LAYER_HIERARCHY.get(parts[1], 3)


def _find_type_checking_lines(tree: ast.Module) -> set[int]:
    """Collect line numbers of import statements inside ``if TYPE_CHECKING:`` blocks.

    Recurses into the block body so imports nested inside ``try``/``if``/``with``
    under ``if TYPE_CHECKING:`` are still recognised as annotation-only (they would
    otherwise be misclassified as runtime L1 violations). Only ``node.body`` is
    walked, NOT ``node.orelse`` — the ``else:`` branch of ``if TYPE_CHECKING:`` is
    the runtime fallback and its imports are genuinely runtime.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for stmt in node.body:
                for child in ast.walk(stmt):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        lines.add(child.lineno)
    return lines


# =============================================================================
# AST Visitor
# =============================================================================


class TierModelVisitor(ast.NodeVisitor):
    """AST visitor that detects bug-hiding patterns."""

    _FASTAPI_ROUTE_METHODS: ClassVar[frozenset[str]] = frozenset({"get", "post", "put", "patch", "delete", "head", "options", "websocket"})
    _R1_HTTP_GET_MODULES: ClassVar[frozenset[str]] = frozenset({"httpx"})
    _R1_HTTP_CLIENT_CONSTRUCTORS: ClassVar[frozenset[str]] = frozenset(
        {
            "aiohttp.ClientSession",
            "httpx.AsyncClient",
            "httpx.Client",
        }
    )
    _R1_QUEUE_CONSTRUCTORS: ClassVar[frozenset[str]] = frozenset({"asyncio.Queue"})
    # CLOSED LIST: audited R5b boundary-normalization helpers from
    # docs/audit/2026-05-19-cicd-allowlist-audit.md and findings/fp-analyst.md.
    # Do not replace this with a broad ``web/**`` or ``plugins/**`` glob; new
    # contexts need their own audit evidence and regression tests.
    _R5_NAMED_BOUNDARY_CONTEXTS: ClassVar[dict[str, frozenset[str]]] = {
        "engine/dependency_resolver.py": frozenset({"_load_depends_on"}),
        "plugins/infrastructure/clients/retrieval/azure_search.py": frozenset({"_parse_response"}),
        "plugins/infrastructure/clients/retrieval/chroma.py": frozenset({"_parse_and_build_chunks"}),
        "plugins/transforms/azure/prompt_shield.py": frozenset({"_analyze_prompt"}),
        "web/app.py": frozenset({"_settings_from_env"}),
        "web/auth/local.py": frozenset({"_required_visible_string_claim"}),
        "web/auth/oidc.py": frozenset(
            {
                "_get_jwk_algorithm",
                "_get_token_algorithm",
                "_validate_discovery_document",
                "_validate_jwks_document",
                "get_user_info",
                "optional_profile_claim",
            }
        ),
        "web/composer/_semantic_validator.py": frozenset({"_is_config_probe_exception"}),
        "web/composer/audit.py": frozenset(
            {
                "_normalize_audit_payload",
                "_result_to_audit_payload",
                "begin_dispatch",
                "begin_dispatch_or_arg_error",
                "build_canonicalization_sentinel",
                "canonicalize_pydantic_cause",
            }
        ),
        "web/composer/guided/protocol.py": frozenset({"validate_payload"}),
        # The Tier-3 LiteLLM/provider response parsers were extracted from
        # service.py into a dedicated module (llm_response_parsing.py) and
        # the cross-module public names had their underscore prefix removed
        # on 2026-05-23 (elspeth-da023db7e7 rename refactor). Same semantics,
        # same boundary, new file + new names: this is a 1:1 successor
        # inclusion, not a list extension. Internal helpers retain their
        # underscore prefix because they are not cross-module-imported.
        "web/composer/llm_response_parsing.py": frozenset(
            {
                "_first_response_message",
                "_json_safe_provider_artifact",
                "_provider_cost_from_response",
                "_provider_details_payload",
                "_reasoning_metadata_from_response",
                "_response_field",
                "_safe_provider_request_id",
                "apply_anthropic_cache_markers",
                "attach_llm_calls",
                "build_llm_call_record",
                "safe_response_model",
                "supports_anthropic_prompt_cache_markers",
                "token_usage_from_response",
            }
        ),
        "web/composer/recipes.py": frozenset({"_coerce_slot"}),
        "web/composer/redaction.py": frozenset(
            {
                "_apply",
                "_count_sensitive",
                "_has_sensitive",
                "_is_descendable",
                "_redact_via_policy",
                "_redact_via_schema",
                "_walk_type",
                "provider",
                "walk_model_schema",
            }
        ),
        "web/composer/service.py": frozenset(
            {
                "_cached_runtime_preflight",
                "_compose_loop",
                # _dispatch_tool_batch was extracted from _compose_loop on
                # 2026-05-23 (compose-loop-decomp refactor). The Tier-3
                # boundary that validates the LLM's tool_call.function.arguments
                # against the dict-shape contract — `if not isinstance(
                # decoded_arguments, dict):` — moved with the code. Same
                # semantics, same boundary, new method name: this is a
                # 1:1 successor inclusion, not a list extension.
                "_dispatch_tool_batch",
                "_litellm_completion_supports_param",
                "_matching_interpretation_placeholder_count",
                "_optional_ancestor_present",
                "_try_apply_freeform_recipe_intent",
                "_validate_advisor_arguments",
            }
        ),
        "web/composer/source_inspection.py": frozenset({"_facts_from_objects", "_inspect_json", "_inspect_jsonl"}),
        "web/composer/state.py": frozenset(
            {
                "_coalesce_branch_connections",
                "_coalesce_branch_names",
                "_declared_input_fields_option",
                "_is_config_probe_exception",
                "_is_static_contract_probe_exception",
                "_serialize_branches",
                "_validate_web_scrape_abuse_contact_not_reserved",
                "from_dict",
            }
        ),
        "web/execution/fanout_guard.py": frozenset(
            {
                "_count_csv_source_rows",
                "_count_json_source_rows",
                "_credential_ref",
                "_provider_calls_per_row",
                "_remote_source_limit",
                "_source_path",
                "_string_option",
            }
        ),
        "web/execution/preflight.py": frozenset({"resolve_runtime_yaml_paths"}),
        "web/execution/routes.py": frozenset({"_run_integrity_http"}),
        "web/execution/schemas.py": frozenset({"_enforce_data_type"}),
        "web/execution/service.py": frozenset({"_on_pipeline_done", "_run_pipeline", "_sanitize_error_for_client"}),
        "web/execution/validation.py": frozenset(
            {
                "_collect_secret_refs",
                "_find_identity_node_advisories",
                "_infer_component_type_from_plugin_error",
                "_mask_pending_interpretation_placeholders_for_authoring_preflight",
                "validate_pipeline",
            }
        ),
        "web/sessions/_auto_title.py": frozenset({"_auto_title_exception_class", "maybe_auto_title_session"}),
        "web/sessions/routes.py": frozenset(
            {
                "_composer_persisted_validation",
                "_dispatch_guided_respond",
                "_extract_runtime_model_snapshot",
                "_state_data_from_composer_state",
            }
        ),
        "web/sessions/service.py": frozenset({"_patch_llm_transform_prompt", "_unwrap_envelope"}),
        "web/sessions/telemetry.py": frozenset({"observed_value"}),
    }

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        self.suppressed_findings: list[Finding] = []
        self.symbol_stack: list[str] = []
        self.class_stack: list[ast.ClassDef] = []
        self.function_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.path_stack: list[str] = []
        self.node_stack: list[ast.AST] = []
        self._decorator_lines: set[int] = set()  # Track lines that are decorators
        self._import_aliases: dict[str, str] = {}
        # Stack of (metadata, derived-name state) pairs — one entry per nested
        # function. ``None`` means the function has no ``@trust_boundary``
        # decorator. We push on entry to every function so popping is symmetric
        # and we can look at "the innermost enclosing decorated function" via
        # a reverse walk of the stack.
        self._boundary_stack: list[tuple[BoundaryMetadata, DerivedNameState] | None] = []

    def visit(self, node: ast.AST) -> Any:
        """Visit a node while retaining ancestor context for receiver-shape checks."""
        self.node_stack.append(node)
        try:
            return super().visit(node)
        finally:
            self.node_stack.pop()

    def _ancestor_node(self, depth: int) -> ast.AST | None:
        """Return an ancestor where depth=1 is parent of the current node."""
        if depth < 1 or len(self.node_stack) <= depth:
            return None
        return self.node_stack[-(depth + 1)]

    def _scope_fingerprint_for_current_node(self) -> str:
        """Return the enclosing-scope fingerprint for the node being visited.

        ``self.node_stack`` holds every ancestor of the current node,
        outermost-first (the module is index 0). Reversed it is innermost-
        first, the shape ``enclosing_scope_node`` expects. When there is no
        enclosing def/class the module root (``node_stack[0]``) is the
        fallback target.
        """
        ancestors = list(reversed(self.node_stack))
        scope = enclosing_scope_node(ancestors)
        if scope is not None:
            return compute_scope_fingerprint(scope)
        module = self.node_stack[0]
        assert isinstance(module, ast.Module)  # node_stack[0] is always the module root
        return compute_scope_fingerprint(None, module=module)

    def _scope_depth_for_current_node(self) -> int:
        """Return K = the count of ast_path components above the enclosing scope.

        ``path_stack`` and ``node_stack`` are index-aligned: ``path_stack[i]`` is
        the edge ``node_stack[i] -> node_stack[i+1]`` (the descent helpers push the
        edge label, then ``visit`` pushes the child). So for enclosing scope
        ``S = node_stack[K]`` the components strictly above ``S`` are
        ``path_stack[0:K]`` (these shift when a module-level statement is
        added/removed) and the within-scope suffix is ``path_stack[K:]`` (stable).

        Returns 0 when the current node has no enclosing def/class (module level):
        the whole path is "scope-relative" and the fallback always returns None
        for them: any module-body edit (e.g. adding an import) changes the
        module's scope_fingerprint, so the scope-fingerprint-equality check fails.
        """
        scope = enclosing_scope_node(list(reversed(self.node_stack)))
        if scope is None:
            return 0
        for index, node in enumerate(self.node_stack):
            if node is scope:
                return index
        # enclosing_scope_node only ever returns a node drawn from node_stack.
        raise AssertionError("enclosing scope not found in node_stack")

    def _get_code_snippet(self, lineno: int) -> str:
        """Get the source line for a given line number."""
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return "<source unavailable>"

    def _fingerprint_node(self, rule_id: str, node: ast.AST) -> str:
        """Generate a stable fingerprint for a finding.

        The fingerprint is based on:
        - rule_id (ensures distinctness across rules)
        - AST path (field/index path from root, stable across formatting)
        - AST dump without line/column attributes
        """
        ast_path = "/".join(self.path_stack)
        node_dump = ast.dump(node, include_attributes=False, annotate_fields=True)
        payload = f"{rule_id}|{ast_path}|{node_dump}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _current_boundary(self) -> tuple[BoundaryMetadata, DerivedNameState] | None:
        """Return the active ``@trust_boundary`` context, if any.

        Suppression is intentionally limited to the decorated function's own
        lexical body. Nested functions, lambdas, and class bodies do not
        inherit an outer function's boundary metadata; if they need boundary
        suppression, they must carry their own ``@trust_boundary``.

        Returns ``None`` if no enclosing function carries a ``@trust_boundary``.
        """
        if not self._boundary_stack:
            return None
        return self._boundary_stack[-1]

    def _current_derived_state(self) -> DerivedNameState | None:
        boundary = self._current_boundary()
        if boundary is None:
            return None
        return boundary[1]

    def _finding_subject_node(self, rule_id: str, node: ast.AST) -> ast.AST | None:
        """Return the AST node whose rootedness determines suppression.

        Only R1 and R5 are listed as suppressible by the documented
        ``BoundaryRule`` Literal in ``elspeth.contracts.trust_boundary``;
        the helper still returns sensible subject nodes for the other
        dict/attr defensive-pattern rules so that adding them to
        ``BoundaryRule`` later is a one-line decorator-side change.

        Rules that cannot meaningfully be "rooted at a value" (exception
        handlers, contextlib.suppress, broad except) return ``None``:
        suppression of those rules via the decorator is never honoured
        because the trust-boundary doctrine sanctions defensive *value
        access* on external data, not blanket exception swallowing.
        """
        if rule_id in {"R1", "R8", "R9"} and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # ``x.get(...)`` / ``x.setdefault(...)`` / ``x.pop(...)`` —
            # rooted at the receiver ``func.value``.
            return node.func.value
        if rule_id == "R5" and isinstance(node, ast.Call) and node.args:
            # ``isinstance(target, ...)`` — rooted at the first arg.
            return node.args[0]
        if rule_id == "R2" and isinstance(node, ast.Call) and node.args:
            # ``getattr(obj, name, default)`` — rooted at the first arg (obj).
            return node.args[0]
        return None

    def _is_suppressed_by_boundary(self, rule_id: str, node: ast.AST) -> bool:
        """Return True if a ``@trust_boundary`` covers this finding.

        See module docstring of :mod:`trust_boundary_suppress` for the
        auditability rationale: there is no telemetry surface for the
        suppression event; the decorator's existence (visible in code
        review) is the audit signal.
        """
        return self._boundary_suppression_for(rule_id, node) is not None

    def _boundary_suppression_for(self, rule_id: str, node: ast.AST) -> BoundaryMetadata | None:
        """Return active boundary metadata if it suppresses this finding."""
        boundary = self._current_boundary()
        if boundary is None:
            return None
        metadata, derived_state = boundary
        if rule_id not in metadata.suppresses:
            return None
        subject = self._finding_subject_node(rule_id, node)
        if subject is None:
            return None
        if not subject_is_rooted(subject, derived_state.snapshot()):
            return None
        return metadata

    def _add_finding(self, rule_id: str, node: ast.expr | ast.stmt | ast.ExceptHandler, message: str) -> None:
        """Record a finding, unless an enclosing ``@trust_boundary`` covers it."""
        suppressed_by = self._boundary_suppression_for(rule_id, node)
        if suppressed_by is not None:
            self._add_suppressed_boundary_observation(rule_id, node, suppressed_by)
            return
        self.findings.append(
            Finding(
                rule_id=rule_id,
                file_path=self.file_path,
                line=node.lineno,
                col=node.col_offset,
                symbol_context=tuple(self.symbol_stack),
                fingerprint=self._fingerprint_node(rule_id, node),
                code_snippet=self._get_code_snippet(node.lineno),
                message=message,
                ast_path="/".join(self.path_stack) or "<module-root>",
                scope_fingerprint=self._scope_fingerprint_for_current_node(),
                scope_depth=self._scope_depth_for_current_node(),
            )
        )

    def _add_suppressed_boundary_observation(
        self,
        original_rule_id: str,
        node: ast.expr | ast.stmt | ast.ExceptHandler,
        metadata: BoundaryMetadata,
    ) -> None:
        """Record a non-failing observation for a trust-boundary suppression."""
        suppresses = tuple(sorted(metadata.suppresses))
        self.suppressed_findings.append(
            Finding(
                rule_id="R_TB_SUPPRESSED",
                file_path=self.file_path,
                line=node.lineno,
                col=node.col_offset,
                symbol_context=tuple(self.symbol_stack),
                fingerprint=self._fingerprint_node(f"R_TB_SUPPRESSED:{original_rule_id}", node),
                code_snippet=self._get_code_snippet(node.lineno),
                message=(
                    f"@trust_boundary suppressed {original_rule_id} under "
                    f"source_param={metadata.source_param!r}; source={metadata.source!r}; "
                    f"test_ref={metadata.test_ref!r}; suppresses={suppresses!r}"
                ),
                ast_path="/".join(self.path_stack) or "<module-root>",
                scope_fingerprint=self._scope_fingerprint_for_current_node(),
                scope_depth=self._scope_depth_for_current_node(),
            )
        )

    def _add_boundary_diagnostic(self, diagnostic: BoundaryFinding) -> None:
        """Surface a malformed-decorator diagnostic as an ordinary finding.

        These diagnostics target the decorator call site itself (not the
        function body), so they bypass the suppression check — a malformed
        decorator cannot suppress its own error.
        """
        node = diagnostic.node
        self.findings.append(
            Finding(
                rule_id=diagnostic.rule_id,
                file_path=self.file_path,
                line=node.lineno,
                col=node.col_offset,
                symbol_context=tuple(self.symbol_stack),
                fingerprint=self._fingerprint_node(diagnostic.rule_id, node),
                code_snippet=self._get_code_snippet(node.lineno),
                message=diagnostic.message,
                ast_path="/".join(self.path_stack) or "<module-root>",
                # When this fires the FunctionDef is already on node_stack, so
                # the stamped scope is the function's OWN scope, not an
                # enclosing one. R_TB_MALFORMED is a fix-don't-suppress
                # diagnostic and is never judge-gated, so the exact value is
                # don't-care — only the non-empty invariant is load-bearing.
                scope_fingerprint=self._scope_fingerprint_for_current_node(),
                scope_depth=self._scope_depth_for_current_node(),
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        """Track import aliases used by receiver-type heuristics."""
        for alias in node.names:
            root_name = alias.name.split(".", 1)[0]
            self._import_aliases[alias.asname or root_name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-import aliases used by receiver-type heuristics."""
        if node.module is None:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            self._import_aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context."""
        outer_state = self._current_derived_state()
        if outer_state is not None:
            outer_state.assign_target_names((node.name,), is_derived=False)
        self.symbol_stack.append(node.name)
        self.class_stack.append(node)
        self._boundary_stack.append(None)
        try:
            self.generic_visit(node)
        finally:
            self._boundary_stack.pop()
            self.class_stack.pop()
            self.symbol_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Visit lambda bodies without inheriting enclosing boundary suppression."""
        self._boundary_stack.append(None)
        try:
            self.generic_visit(node)
        finally:
            self._boundary_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function context."""
        # Collect decorator lines — .get() calls here are not dict access
        for decorator in node.decorator_list:
            self._decorator_lines.add(decorator.lineno)
        outer_state = self._current_derived_state()
        if outer_state is not None:
            outer_state.assign_target_names((node.name,), is_derived=False)
        self.symbol_stack.append(node.name)
        self.function_stack.append(node)
        self._enter_boundary_context(node)
        try:
            self.generic_visit(node)
        finally:
            self._boundary_stack.pop()
            self.function_stack.pop()
            self.symbol_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track async function context."""
        # Collect decorator lines — .get() calls here are not dict access
        for decorator in node.decorator_list:
            self._decorator_lines.add(decorator.lineno)
        outer_state = self._current_derived_state()
        if outer_state is not None:
            outer_state.assign_target_names((node.name,), is_derived=False)
        self.symbol_stack.append(node.name)
        self.function_stack.append(node)
        self._enter_boundary_context(node)
        try:
            self.generic_visit(node)
        finally:
            self._boundary_stack.pop()
            self.function_stack.pop()
            self.symbol_stack.pop()

    def _enter_boundary_context(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Parse ``@trust_boundary`` (if present) and push the suppression context.

        Always pushes onto ``_boundary_stack`` so the pop in the visit
        wrappers is unconditional. A ``None`` entry means "no decorator on
        this function" (the common case). Malformed decorators push
        ``None`` AND emit a finding so the suppression is rejected loudly.

        ``source_param`` is validated against the function signature here
        (rather than relying on the runtime decorator's check) so the rule
        can flag the decorator at lint time even if the function is never
        imported. A mismatch is reported as R_TB_MALFORMED.
        """
        metadata, diagnostics = extract_boundary_metadata(node, import_aliases=self._import_aliases)
        for diagnostic in diagnostics:
            self._add_boundary_diagnostic(diagnostic)

        if metadata is None:
            self._boundary_stack.append(None)
            return

        # Confirm source_param actually names a parameter of the function.
        # ``args.args`` covers positional + keyword-or-positional;
        # ``kwonlyargs`` covers keyword-only; ``posonlyargs`` covers
        # positional-only. ``vararg`` / ``kwarg`` cover ``*args`` / ``**kwargs``.
        # If a decorator nominates ``**kwargs`` as the source_param, that is
        # legitimate — the function body subscripts ``kwargs`` like a dict —
        # so we include vararg/kwarg in the parameter set.
        param_names: set[str] = set()
        param_names.update(arg.arg for arg in node.args.posonlyargs)
        param_names.update(arg.arg for arg in node.args.args)
        param_names.update(arg.arg for arg in node.args.kwonlyargs)
        if node.args.vararg is not None:
            param_names.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            param_names.add(node.args.kwarg.arg)

        if metadata.source_param not in param_names:
            self._add_boundary_diagnostic(
                BoundaryFinding(
                    rule_id="R_TB_MALFORMED",
                    message=(
                        f"@trust_boundary(source_param={metadata.source_param!r}) does not "
                        f"name a parameter of {node.name!r}; "
                        f"signature parameters are {sorted(param_names)!r}."
                    ),
                    node=metadata.decorator_node,
                )
            )
            self._boundary_stack.append(None)
            return

        self._boundary_stack.append((metadata, DerivedNameState.from_source_param(metadata.source_param)))

    def _is_default_return_value(self, value: ast.expr | None) -> bool:
        """True if return value is a silent default (None, empty container, empty string, zero)."""
        if value is None:
            return True
        if isinstance(value, ast.Constant):
            return value.value in (None, "", 0, 0.0, False)
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return len(value.elts) == 0
        if isinstance(value, ast.Dict):
            return len(value.keys) == 0
        return False

    def _is_transform_result_error_call(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "error"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "TransformResult"
        )

    def _assigned_transform_error_names(self, nodes: list[ast.AST]) -> set[str]:
        names: set[str] = set()
        for child in nodes:
            if isinstance(child, ast.Assign) and self._is_transform_result_error_call(child.value):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif (
                isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
                and child.value is not None
                and self._is_transform_result_error_call(child.value)
            ):
                names.add(child.target.id)
        return names

    def _routes_transform_error_to_completion(self, nodes: list[ast.AST]) -> bool:
        """True when a handler delivers an explicit TransformResult.error."""
        error_names = self._assigned_transform_error_names(nodes)
        for child in nodes:
            if not (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "_complete_ticket"):
                continue
            for arg in child.args:
                if self._is_transform_result_error_call(arg):
                    return True
                if isinstance(arg, ast.Name) and arg.id in error_names:
                    return True
        return False

    def _handler_is_silent(self, node: ast.ExceptHandler) -> bool:
        """Return True if the except handler swallows errors without re-raise or explicit return."""
        own_scope_nodes = [child for statement in node.body for child in iter_own_scope(statement)]
        has_raise = any(isinstance(child, ast.Raise) for child in own_scope_nodes)
        if has_raise:
            return False

        if self._routes_transform_error_to_completion(own_scope_nodes):
            return False

        returns: list[ast.Return] = [child for child in own_scope_nodes if isinstance(child, ast.Return)]
        if returns:
            # If all returns are silent defaults, treat as swallow.
            return all(self._is_default_return_value(ret.value) for ret in returns)

        # No raise, no return: likely swallow (even if logging).
        return True

    def _is_likely_non_dict_get(self, node: ast.Call) -> bool:
        """Return True if this .get() call is likely NOT a dict.get().

        Heuristics (conservative — only skip when confident):
        1. Decorator context: @router.get("/path") is a route decorator
        2. URL-like first arg: client.get("https://...") is an HTTP method
        3. ChromaDB keywords: collection.get(ids=[...]) is SDK retrieval
        4. Receiver type: httpx module/client and asyncio.Queue receivers are
           transport/queue APIs, not dicts

        Note: f-string URLs on unknown receivers are NOT filtered because we
        cannot statically determine their runtime value. They must be tied to a
        known transport receiver before this rule suppresses them.
        """
        # Heuristic 1: Decorator context
        if node.lineno in self._decorator_lines:
            return True

        # Heuristic 2: Known module/client/queue receiver type.
        if self._is_known_non_dict_get_receiver(node):
            return True

        # Heuristic 3: URL-like first argument
        if node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                val = first_arg.value
                if val.startswith(("/", "http://", "https://")):
                    return True

        # Heuristic 4: ChromaDB-specific keywords
        # IMPORTANT: Only include keywords that are unambiguous to ChromaDB/vector DBs.
        # Generic pagination keywords (limit, offset) are NOT included because they
        # collide with SQLAlchemy, Django ORM, and other common patterns.
        chromadb_keywords = {"ids", "include", "where"}
        call_keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        return bool(call_keywords & chromadb_keywords)

    def _qualified_import_name(self, expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            return self._import_aliases.get(expr.id)
        if isinstance(expr, ast.Attribute):
            base = self._qualified_import_name(expr.value)
            if base is None:
                return None
            return f"{base}.{expr.attr}"
        return None

    def _call_constructs_known_non_dict_get_receiver(self, call: ast.Call) -> bool:
        constructor = self._qualified_import_name(call.func)
        if constructor is None:
            return False
        return constructor in self._R1_HTTP_CLIENT_CONSTRUCTORS or constructor in self._R1_QUEUE_CONSTRUCTORS

    def _target_matches_receiver(self, target: ast.expr, receiver: ast.expr) -> bool:
        if isinstance(target, ast.Name) and isinstance(receiver, ast.Name):
            return target.id == receiver.id
        if isinstance(target, ast.Attribute) and isinstance(receiver, ast.Attribute):
            return (
                target.attr == receiver.attr
                and isinstance(target.value, ast.Name)
                and isinstance(receiver.value, ast.Name)
                and target.value.id == receiver.value.id
            )
        return False

    def _walk_scope_nodes(self, node: ast.AST) -> Iterator[ast.AST]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            yield child
            yield from self._walk_scope_nodes(child)

    def _receiver_assigned_known_type_before(self, scope: ast.AST, receiver: ast.expr, lineno: int) -> bool:
        known: bool | None = None
        for child in self._walk_scope_nodes(scope):
            if getattr(child, "lineno", lineno) >= lineno:
                continue
            if isinstance(child, ast.Assign):
                value_is_known = isinstance(child.value, ast.Call) and self._call_constructs_known_non_dict_get_receiver(child.value)
                for target in child.targets:
                    if self._target_matches_receiver(target, receiver):
                        known = value_is_known
            elif isinstance(child, ast.AnnAssign):
                value_is_known = isinstance(child.value, ast.Call) and self._call_constructs_known_non_dict_get_receiver(child.value)
                if self._target_matches_receiver(child.target, receiver):
                    known = value_is_known
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                for item in child.items:
                    if item.optional_vars is None:
                        continue
                    value_is_known = isinstance(item.context_expr, ast.Call) and self._call_constructs_known_non_dict_get_receiver(
                        item.context_expr
                    )
                    if self._target_matches_receiver(item.optional_vars, receiver):
                        known = value_is_known
        return known is True

    def _current_class_init_assigns_known_type(self, receiver: ast.expr) -> bool:
        current_class = self._current_class()
        if current_class is None or not isinstance(receiver, ast.Attribute):
            return False
        if not isinstance(receiver.value, ast.Name) or receiver.value.id != "self":
            return False
        for stmt in current_class.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
                return self._receiver_assigned_known_type_before(stmt, receiver, lineno=10**9)
        return False

    def _is_known_non_dict_get_receiver(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute):
            return False
        receiver = node.func.value
        receiver_module = self._qualified_import_name(receiver)
        if receiver_module in self._R1_HTTP_GET_MODULES:
            return True
        current_function = self._current_function()
        if current_function is not None and self._receiver_assigned_known_type_before(current_function, receiver, node.lineno):
            return True
        return self._current_class_init_assigns_known_type(receiver)

    def _decorator_leaf_name(self, decorator: ast.expr) -> str | None:
        """Return the called/decorated symbol leaf name for decorator classification."""
        expr = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return expr.attr
        return None

    def _constant_keyword_value(self, node: ast.Call, keyword_name: str) -> object:
        for keyword in node.keywords:
            if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
        return None

    def _current_function(self) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        return self.function_stack[-1] if self.function_stack else None

    def _current_class(self) -> ast.ClassDef | None:
        return self.class_stack[-1] if self.class_stack else None

    def _current_class_is_frozen_dataclass(self) -> bool:
        current_class = self._current_class()
        if current_class is None:
            return False
        for decorator in current_class.decorator_list:
            if self._decorator_leaf_name(decorator) != "dataclass":
                continue
            if isinstance(decorator, ast.Call) and self._constant_keyword_value(decorator, "frozen") is True:
                return True
        return False

    def _post_init_self_field_aliases_before(self, lineno: int) -> set[str]:
        current_function = self._current_function()
        if current_function is None or current_function.name != "__post_init__":
            return set()
        aliases: set[str] = set()
        for stmt in current_function.body:
            if stmt.lineno >= lineno:
                continue
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target = stmt.target
                value = stmt.value
            if not (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"
            ):
                continue
            aliases.add(target.id)
        return aliases

    def _is_self_field_isinstance(self, node: ast.Call) -> bool:
        if not node.args:
            return False
        target = node.args[0]
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            return True
        return isinstance(target, ast.Name) and target.id in self._post_init_self_field_aliases_before(node.lineno)

    def _is_tier1_frozen_dataclass_post_init_guard(self, node: ast.Call) -> bool:
        current_function = self._current_function()
        if current_function is None or current_function.name != "__post_init__":
            return False
        return self._current_class_is_frozen_dataclass() and self._is_self_field_isinstance(node)

    def _is_pydantic_before_validator(self) -> bool:
        current_function = self._current_function()
        if current_function is None:
            return False
        for decorator in current_function.decorator_list:
            name = self._decorator_leaf_name(decorator)
            if not isinstance(decorator, ast.Call):
                continue
            if name in {"field_validator", "model_validator"}:
                return self._constant_keyword_value(decorator, "mode") == "before"
            if name == "validator":
                return self._constant_keyword_value(decorator, "pre") is True
        return False

    def _is_fastapi_route_handler(self) -> bool:
        if not self.file_path.startswith("web/"):
            return False
        current_function = self._current_function()
        if current_function is None:
            return False
        return any(self._decorator_leaf_name(decorator) in self._FASTAPI_ROUTE_METHODS for decorator in current_function.decorator_list)

    def _is_named_tier3_boundary_context(self) -> bool:
        current_function = self._current_function()
        if current_function is None:
            return False
        return current_function.name in self._R5_NAMED_BOUNDARY_CONTEXTS.get(self.file_path, frozenset())

    def _is_allowed_r5_context(self, node: ast.Call) -> bool:
        """Return True for R5a/R5b contexts where isinstance is the desired guard."""
        return (
            self._is_tier1_frozen_dataclass_post_init_guard(node)
            or self._is_pydantic_before_validator()
            or self._is_fastapi_route_handler()
            or self._is_named_tier3_boundary_context()
        )

    def _is_immediate_setdefault_grouping_call(self, node: ast.Call) -> bool:
        """Return True for setdefault(...).append/extend(...) grouping idioms."""
        parent = self._ancestor_node(1)
        grandparent = self._ancestor_node(2)
        return (
            isinstance(parent, ast.Attribute)
            and parent.value is node
            and parent.attr in {"append", "extend"}
            and isinstance(grandparent, ast.Call)
            and grandparent.func is parent
        )

    def _visit_ast_child(self, field_name: str, node: ast.AST | None) -> None:
        if node is None:
            return
        self.path_stack.append(field_name)
        try:
            self.visit(node)
        finally:
            self.path_stack.pop()

    def _visit_ast_list_item(self, field_name: str, index: int, node: ast.AST) -> None:
        self.path_stack.append(f"{field_name}[{index}]")
        try:
            self.visit(node)
        finally:
            self.path_stack.pop()

    def _value_depends_on_boundary(self, value: ast.AST, snapshot: frozenset[str]) -> bool:
        state = self._current_derived_state()
        if state is None:
            return False
        return state.expression_depends_on_current_names(value, snapshot=snapshot)

    def _assign_targets_from_value(self, targets: Iterable[ast.expr], value: ast.AST, snapshot: frozenset[str]) -> None:
        state = self._current_derived_state()
        if state is None:
            return
        state.assign_targets(targets, is_derived=self._value_depends_on_boundary(value, snapshot))

    def _set_current_derived_names(self, names: frozenset[str]) -> None:
        state = self._current_derived_state()
        if state is None:
            return
        state.names.clear()
        state.names.update(names)

    def _visit_statement_sequence_from_snapshot(
        self,
        field_name: str,
        statements: Sequence[ast.stmt],
        snapshot: frozenset[str],
    ) -> frozenset[str]:
        self._set_current_derived_names(snapshot)
        for index, statement in enumerate(statements):
            self._visit_ast_list_item(field_name, index, statement)
        state = self._current_derived_state()
        return frozenset() if state is None else state.snapshot()

    @staticmethod
    def _intersect_snapshots(snapshots: Sequence[frozenset[str]]) -> frozenset[str]:
        if not snapshots:
            return frozenset()
        joined = set(snapshots[0])
        for snapshot in snapshots[1:]:
            joined.intersection_update(snapshot)
        return frozenset(joined)

    def visit_Assign(self, node: ast.Assign) -> None:
        state = self._current_derived_state()
        snapshot = frozenset() if state is None else state.snapshot()

        self._visit_ast_child("value", node.value)
        for index, target in enumerate(node.targets):
            self._visit_ast_list_item("targets", index, target)

        self._assign_targets_from_value(node.targets, node.value, snapshot)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        state = self._current_derived_state()
        snapshot = frozenset() if state is None else state.snapshot()

        if node.value is not None:
            self._visit_ast_child("value", node.value)
        self._visit_ast_child("target", node.target)
        self._visit_ast_child("annotation", node.annotation)

        if node.value is not None:
            self._assign_targets_from_value((node.target,), node.value, snapshot)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        state = self._current_derived_state()
        snapshot = frozenset() if state is None else state.snapshot()
        is_derived = subject_is_rooted(node.target, snapshot) or self._value_depends_on_boundary(node.value, snapshot)

        self._visit_ast_child("target", node.target)
        self._visit_ast_child("value", node.value)

        if state is not None:
            state.assign_target(node.target, is_derived=is_derived)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        state = self._current_derived_state()
        snapshot = frozenset() if state is None else state.snapshot()
        is_derived = self._value_depends_on_boundary(node.value, snapshot)

        self._visit_ast_child("value", node.value)
        self._visit_ast_child("target", node.target)

        if state is not None:
            state.assign_target(node.target, is_derived=is_derived)

    def visit_If(self, node: ast.If) -> None:
        state = self._current_derived_state()
        if state is None:
            self.generic_visit(node)
            return

        self._visit_ast_child("test", node.test)
        branch_start = state.snapshot()
        body_end = self._visit_statement_sequence_from_snapshot("body", node.body, branch_start)
        orelse_end = self._visit_statement_sequence_from_snapshot("orelse", node.orelse, branch_start) if node.orelse else branch_start
        self._set_current_derived_names(self._intersect_snapshots((body_end, orelse_end)))

    def _visit_try_like(self, node: ast.Try | ast.TryStar) -> None:
        state = self._current_derived_state()
        if state is None or not node.handlers:
            self.generic_visit(node)
            return

        branch_start = state.snapshot()
        body_end = self._visit_statement_sequence_from_snapshot("body", node.body, branch_start)
        if node.orelse:
            body_end = self._visit_statement_sequence_from_snapshot("orelse", node.orelse, body_end)
        branch_ends = [body_end]

        for index, handler in enumerate(node.handlers):
            self.path_stack.append(f"handlers[{index}]")
            try:
                if handler.type is not None:
                    self._visit_ast_child("type", handler.type)
                branch_ends.append(self._visit_statement_sequence_from_snapshot("body", handler.body, branch_start))
            finally:
                self.path_stack.pop()

        self._set_current_derived_names(self._intersect_snapshots(tuple(branch_ends)))
        for index, statement in enumerate(node.finalbody):
            self._visit_ast_list_item("finalbody", index, statement)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_try_like(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_try_like(node)

    def visit_While(self, node: ast.While) -> None:
        state = self._current_derived_state()
        if state is None:
            self.generic_visit(node)
            return

        self._visit_ast_child("test", node.test)
        loop_entry = state.snapshot()
        body_end = self._visit_statement_sequence_from_snapshot("body", node.body, loop_entry)
        joined = self._intersect_snapshots((loop_entry, body_end))
        if node.orelse:
            orelse_end = self._visit_statement_sequence_from_snapshot("orelse", node.orelse, joined)
            joined = self._intersect_snapshots((joined, orelse_end))
        self._set_current_derived_names(joined)

    def _visit_for_like(self, node: ast.For | ast.AsyncFor) -> None:
        state = self._current_derived_state()
        snapshot = frozenset() if state is None else state.snapshot()
        target_is_derived = subject_is_rooted(node.iter, snapshot)

        self._visit_ast_child("iter", node.iter)
        if state is not None:
            state.assign_target(node.target, is_derived=target_is_derived)
        self._visit_ast_child("target", node.target)
        for index, statement in enumerate(node.body):
            self._visit_ast_list_item("body", index, statement)
        for index, statement in enumerate(node.orelse):
            self._visit_ast_list_item("orelse", index, statement)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for_like(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for_like(node)

    def _visit_with_like(self, node: ast.With | ast.AsyncWith) -> None:
        state = self._current_derived_state()
        for index, item in enumerate(node.items):
            self.path_stack.append(f"items[{index}]")
            try:
                snapshot = frozenset() if state is None else state.snapshot()
                optional_vars_is_derived = subject_is_rooted(item.context_expr, snapshot)
                self._visit_ast_child("context_expr", item.context_expr)
                if item.optional_vars is not None:
                    if state is not None:
                        state.assign_target(item.optional_vars, is_derived=optional_vars_is_derived)
                    self._visit_ast_child("optional_vars", item.optional_vars)
            finally:
                self.path_stack.pop()
        for index, statement in enumerate(node.body):
            self._visit_ast_list_item("body", index, statement)

    def _restore_comprehension_targets(self, original_names: set[str], target_names: set[str]) -> None:
        state = self._current_derived_state()
        if state is None:
            return
        for name in target_names:
            if name in original_names:
                state.names.add(name)
            else:
                state.names.discard(name)

    def _visit_comprehension_generators(self, generators: list[ast.comprehension]) -> tuple[set[str], set[str]]:
        state = self._current_derived_state()
        original_names = set() if state is None else set(state.names)
        target_names: set[str] = set()
        for index, generator in enumerate(generators):
            self.path_stack.append(f"generators[{index}]")
            try:
                snapshot = frozenset() if state is None else state.snapshot()
                target_is_derived = subject_is_rooted(generator.iter, snapshot)
                self._visit_ast_child("iter", generator.iter)
                if state is not None:
                    state.assign_target(generator.target, is_derived=target_is_derived)
                target_names.update(assignment_target_names(generator.target))
                self._visit_ast_child("target", generator.target)
                for if_index, if_node in enumerate(generator.ifs):
                    self._visit_ast_list_item("ifs", if_index, if_node)
            finally:
                self.path_stack.pop()
        return original_names, target_names

    def visit_ListComp(self, node: ast.ListComp) -> None:
        original_names, target_names = self._visit_comprehension_generators(node.generators)
        try:
            self._visit_ast_child("elt", node.elt)
        finally:
            self._restore_comprehension_targets(original_names, target_names)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        original_names, target_names = self._visit_comprehension_generators(node.generators)
        try:
            self._visit_ast_child("elt", node.elt)
        finally:
            self._restore_comprehension_targets(original_names, target_names)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        original_names, target_names = self._visit_comprehension_generators(node.generators)
        try:
            self._visit_ast_child("key", node.key)
            self._visit_ast_child("value", node.value)
        finally:
            self._restore_comprehension_targets(original_names, target_names)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        original_names, target_names = self._visit_comprehension_generators(node.generators)
        try:
            self._visit_ast_child("elt", node.elt)
        finally:
            self._restore_comprehension_targets(original_names, target_names)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect R1 (dict.get), R2 (getattr), R3 (hasattr), R5 (isinstance), R8/R9 defaults."""
        # R1: dict.get() - Call(func=Attribute(attr="get"))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and not self._is_likely_non_dict_get(node):
            self._add_finding(
                "R1",
                node,
                f"Potential dict.get() usage: {self._get_code_snippet(node.lineno)}",
            )

        # R8: dict.setdefault() - mutating default on missing key
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and not self._is_immediate_setdefault_grouping_call(node)
        ):
            self._add_finding(
                "R8",
                node,
                f"dict.setdefault() hides missing keys: {self._get_code_snippet(node.lineno)}",
            )

        # R9: dict.pop(key, default) - implicit default on missing key
        if isinstance(node.func, ast.Attribute) and node.func.attr == "pop" and (len(node.args) >= 2 or node.keywords):
            self._add_finding(
                "R9",
                node,
                f"dict.pop() with default hides missing keys: {self._get_code_snippet(node.lineno)}",
            )

        # R2: getattr() - Call(func=Name("getattr"))
        # Only flag if there's a default argument (3 args)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and (len(node.args) >= 3 or node.keywords):
            self._add_finding(
                "R2",
                node,
                f"getattr() with default hides AttributeError: {self._get_code_snippet(node.lineno)}",
            )

        # R3: hasattr() - Call(func=Name("hasattr"))
        if isinstance(node.func, ast.Name) and node.func.id == "hasattr":
            self._add_finding(
                "R3",
                node,
                f"hasattr() branches around missing attributes: {self._get_code_snippet(node.lineno)}",
            )

        # R5: isinstance() - runtime type checks can mask contract violations
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance" and not self._is_allowed_r5_context(node):
            self._add_finding(
                "R5",
                node,
                f"isinstance() used: {self._get_code_snippet(node.lineno)}",
            )

        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Detect R7: contextlib.suppress usage."""
        for item in node.items:
            ctx_expr = item.context_expr
            if isinstance(ctx_expr, ast.Call):
                func = ctx_expr.func
                if (isinstance(func, ast.Name) and func.id == "suppress") or (isinstance(func, ast.Attribute) and func.attr == "suppress"):
                    self._add_finding(
                        "R7",
                        node,
                        f"contextlib.suppress() used: {self._get_code_snippet(node.lineno)}",
                    )
        self._visit_with_like(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Detect R7: contextlib.suppress usage in async context managers."""
        for item in node.items:
            ctx_expr = item.context_expr
            if isinstance(ctx_expr, ast.Call):
                func = ctx_expr.func
                if (isinstance(func, ast.Name) and func.id == "suppress") or (isinstance(func, ast.Attribute) and func.attr == "suppress"):
                    self._add_finding(
                        "R7",
                        node,
                        f"contextlib.suppress() used: {self._get_code_snippet(node.lineno)}",
                    )
        self._visit_with_like(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Detect R4: broad exception suppression."""
        # Check for bare except or except Exception
        is_broad = False
        if node.type is None:
            # bare except:
            is_broad = True
        elif isinstance(node.type, ast.Name) and node.type.id in (
            "Exception",
            "BaseException",
        ):
            is_broad = True
        elif isinstance(node.type, ast.Tuple):
            # except (Exception, ...):
            for elt in node.type.elts:
                if isinstance(elt, ast.Name) and elt.id in (
                    "Exception",
                    "BaseException",
                ):
                    is_broad = True
                    break

        if is_broad:
            # Check if the handler re-raises
            has_reraise = False
            for statement in node.body:
                for child in iter_own_scope(statement):
                    if isinstance(child, ast.Raise):
                        has_reraise = True
                        break
                if has_reraise:
                    break

            if not has_reraise:
                self._add_finding(
                    "R4",
                    node,
                    f"Broad exception caught without re-raise: {self._get_code_snippet(node.lineno)}",
                )

        # R6: specific exception swallowed without re-raise or explicit return
        if not is_broad and self._handler_is_silent(node):
            self._add_finding(
                "R6",
                node,
                f"Exception swallowed without re-raise or explicit error: {self._get_code_snippet(node.lineno)}",
            )

        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Visit a node, tracking AST path for stable fingerprints."""
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        self.path_stack.append(f"{field_name}[{index}]")
                        self.visit(item)
                        self.path_stack.pop()
            elif isinstance(value, ast.AST):
                self.path_stack.append(field_name)
                self.visit(value)
                self.path_stack.pop()


# =============================================================================
# File Scanning
# =============================================================================


def scan_file(file_path: Path, root: Path) -> list[Finding]:
    """Scan a single Python file for bug-hiding patterns."""
    findings, _suppressed_findings = scan_file_with_observations(file_path, root)
    return findings


def scan_file_with_observations(file_path: Path, root: Path) -> tuple[list[Finding], list[Finding]]:
    """Scan a single Python file and return violations plus suppression observations."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return [], []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return [], []

    source_lines = source.splitlines()
    relative_path = str(file_path.relative_to(root))

    visitor = TierModelVisitor(relative_path, source_lines)
    visitor.visit(tree)
    return visitor.findings, visitor.suppressed_findings


def scan_directory(
    root: Path,
    exclude_patterns: list[str] | None = None,
) -> list[Finding]:
    """Scan all Python files in a directory tree."""
    findings, _suppressed_findings = scan_directory_with_observations(root, exclude_patterns)
    return findings


def scan_directory_with_observations(
    root: Path,
    exclude_patterns: list[str] | None = None,
) -> tuple[list[Finding], list[Finding]]:
    """Scan all Python files and return violations plus suppression observations."""
    exclude_patterns = exclude_patterns or []
    findings: list[Finding] = []
    suppressed_findings: list[Finding] = []

    for py_file in root.rglob("*.py"):
        relative = py_file.relative_to(root)
        # Skip vendored/third-party directories
        if any(part in _ALWAYS_EXCLUDED_DIRS for part in relative.parts):
            continue
        # Check user-specified exclusions
        skip = False
        for pattern in exclude_patterns:
            if relative.match(pattern) or str(relative).startswith(pattern.rstrip("*/")):
                skip = True
                break
        if skip:
            continue

        file_findings, file_suppressed_findings = scan_file_with_observations(py_file, root)
        findings.extend(file_findings)
        suppressed_findings.extend(file_suppressed_findings)

    return findings, suppressed_findings


# =============================================================================
# Layer Import Scanning
# =============================================================================


def scan_layer_imports_file(
    file_path: Path,
    root: Path,
) -> tuple[list[Finding], list[Finding]]:
    """Scan a single file for upward layer imports.

    Returns:
        violations: Findings for runtime upward imports (fail CI unless allowlisted)
        tc_findings: Findings for TYPE_CHECKING upward imports (warnings, allowlistable)
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [], []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return [], []

    relative_path = str(file_path.relative_to(root))
    source_lines = source.splitlines()
    file_layer = _get_file_layer(relative_path)

    # L3 files (plugins, mcp, tui, etc.) can import anything — skip
    if file_layer >= 3:
        return [], []

    tc_lines = _find_type_checking_lines(tree)
    violations: list[Finding] = []
    tc_findings: list[Finding] = []

    # Layer-import findings have no AST subject (the imported module IS the
    # address) and are never judge-gated — layer-import suppressions use
    # per-file rules / TC warnings, not signed entries. The whole module is
    # the natural scope, so stamp the module fingerprint rather than leave
    # the field empty; this keeps the non-empty invariant uniform and avoids
    # a future reader mistaking these sites for a missed AST-subject one.
    module_scope_fingerprint = compute_scope_fingerprint(None, module=tree)

    # Resolve `from elspeth import X` against the real package dir so a subpackage
    # (X=plugins) is treated as a layer import while a plain attribute
    # (X=__version__) is not. Root-robust: --root may be src/elspeth (paths like
    # core/...) or src (paths like elspeth/core/...).
    elspeth_pkg_root = root / "elspeth" if relative_path.split("/")[0] == "elspeth" else root

    for node in ast.walk(tree):
        # Collect (module_name, line, col) targets from import nodes
        targets: list[tuple[str, int, int]] = []
        if isinstance(node, ast.ImportFrom):
            # Resolve relative imports (level>0) and absolute ones to the target
            # module. The layer is keyed on the resolved module's top-level
            # package, so only the bare-``elspeth`` package-root form needs
            # per-alias resolution.
            resolved = _resolve_relative_module(relative_path, node.level, node.module)
            if resolved is None:
                pass
            elif resolved == "elspeth":
                for alias in node.names:
                    candidate = f"elspeth.{alias.name}"
                    if _module_name_to_path(candidate, elspeth_pkg_root) is not None:
                        targets.append((candidate, node.lineno, node.col_offset))
            else:
                targets.append((resolved, node.lineno, node.col_offset))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                targets.append((alias.name, node.lineno, node.col_offset))

        for module_name, line, col in targets:
            target_layer = _get_import_target_layer(module_name)
            if target_layer is None or target_layer <= file_layer:
                continue

            snippet = source_lines[line - 1].strip() if line <= len(source_lines) else "<source unavailable>"
            from_name = LAYER_NAMES[file_layer]
            to_name = LAYER_NAMES[target_layer]

            # Layer-import findings have no AST subject; the import target
            # module IS the address. Synthesise a stable ast_path of the
            # form ``import:<module>`` so binding verification works the
            # same way as AST-rooted findings.
            import_ast_path = f"import:{module_name}"
            if line in tc_lines:
                tc_payload = f"TC|{relative_path}|{module_name}"
                tc_fp = hashlib.sha256(tc_payload.encode()).hexdigest()[:16]
                tc_findings.append(
                    Finding(
                        rule_id="TC",
                        file_path=relative_path,
                        line=line,
                        col=col,
                        symbol_context=(),
                        fingerprint=tc_fp,
                        code_snippet=snippet,
                        message=f"TYPE_CHECKING import: {from_name} annotates with {to_name} ({module_name})",
                        ast_path=import_ast_path,
                        scope_fingerprint=module_scope_fingerprint,
                    )
                )
            else:
                # Fingerprint: keyed on file + imported module (stable across reformatting)
                payload = f"L1|{relative_path}|{module_name}"
                fp = hashlib.sha256(payload.encode()).hexdigest()[:16]

                violations.append(
                    Finding(
                        rule_id="L1",
                        file_path=relative_path,
                        line=line,
                        col=col,
                        symbol_context=(),
                        fingerprint=fp,
                        code_snippet=snippet,
                        message=f"Upward import: {from_name} imports from {to_name} ({module_name})",
                        ast_path=import_ast_path,
                        scope_fingerprint=module_scope_fingerprint,
                    )
                )

    return violations, tc_findings


def scan_layer_imports_directory(
    root: Path,
    exclude_patterns: list[str] | None = None,
) -> tuple[list[Finding], list[Finding]]:
    """Scan all Python files for upward layer imports."""
    exclude_patterns = exclude_patterns or []
    all_violations: list[Finding] = []
    all_tc_findings: list[Finding] = []

    for py_file in root.rglob("*.py"):
        relative = py_file.relative_to(root)
        # Skip vendored/third-party directories
        if any(part in _ALWAYS_EXCLUDED_DIRS for part in relative.parts):
            continue
        # Check user-specified exclusions
        skip = False
        for pattern in exclude_patterns:
            if relative.match(pattern) or str(relative).startswith(pattern.rstrip("*/")):
                skip = True
                break
        if skip:
            continue

        violations, tc_findings = scan_layer_imports_file(py_file, root)
        all_violations.extend(violations)
        all_tc_findings.extend(tc_findings)

    return all_violations, all_tc_findings


# =============================================================================
# Edge-Dump Mode (Phase 0 — L3↔L3 import-graph oracle, Δ2 dump-edges)
#
# Additive subcommand. Shares the path→layer table and AST-based import walking
# of `check`, but emits the full intra-layer edge graph rather than a violations
# list. Always exits 0 unless the tool itself errors. NEVER fails the build.
# =============================================================================


_LAYER_NAME_TO_INT: dict[str, int] = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


@dataclass(frozen=True)
class _ImportSite:
    """A single import statement that produces one edge contribution."""

    src_file: str  # relative path under --root
    line: int
    target_module: str  # fully qualified target (e.g. "elspeth.plugins.transforms.llm")
    type_checking: bool
    conditional: bool
    reexport: bool


def _module_name_to_path(module_name: str, root: Path) -> Path | None:
    """Resolve `elspeth.X.Y.Z` to a file path under root.

    Tries submodule (`X/Y/Z.py`) first, then package (`X/Y/Z/__init__.py`).
    Returns None for non-elspeth modules or modules that don't resolve.
    """
    if not module_name.startswith("elspeth"):
        return None
    parts = module_name.split(".")
    if parts[0] != "elspeth":
        return None
    rel_parts = parts[1:]
    if not rel_parts:
        return None
    candidate = root.joinpath(*rel_parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    pkg_init = root.joinpath(*rel_parts, "__init__.py")
    if pkg_init.is_file():
        return pkg_init
    return None


def _resolve_import_target(
    module_name: str,
    imported_name: str | None,
    root: Path,
) -> Path | None:
    """Resolve an import statement to a target file.

    For ``from M import N``, try ``M.N`` (submodule) first, then ``M`` (package).
    For ``import M``, resolve M directly.
    """
    if imported_name is not None:
        sub = _module_name_to_path(f"{module_name}.{imported_name}", root)
        if sub is not None:
            return sub
    return _module_name_to_path(module_name, root)


def _resolve_relative_module(
    relative_path: str,
    level: int,
    module_name: str | None,
) -> str | None:
    """Resolve a ``from .x import y`` (level≥1) to its absolute ``elspeth.<...>`` form.

    ``relative_path`` is the importing file's path relative to --root.
    Returns the absolute module name, or None if the relative reference is invalid.
    """
    if level == 0:
        return module_name
    pkg_parts = list(Path(relative_path).parent.parts)
    # `from . import X` from a file inside `pkg/` means "X within pkg" — level=1, drop 0 parts.
    # `from .. import X` means "X within parent of pkg" — level=2, drop 1 part. Etc.
    drop = level - 1
    if drop > len(pkg_parts):
        return None
    base_parts = pkg_parts[: len(pkg_parts) - drop] if drop > 0 else pkg_parts
    suffix_parts: list[str] = []
    if module_name:
        suffix_parts.extend(module_name.split("."))
    full_parts = ["elspeth", *base_parts, *suffix_parts]
    return ".".join(full_parts)


def _find_conditional_import_lines(tree: ast.Module) -> set[int]:
    """Collect line numbers of imports inside non-TYPE_CHECKING ``if`` or ``try`` blocks.

    Imports inside ``if TYPE_CHECKING:`` are NOT included here (they are tagged
    via the dedicated TYPE_CHECKING line set).
    """
    cond_lines: set[int] = set()
    for node in ast.walk(tree):
        is_conditional = False
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
            is_conditional = True
        elif isinstance(node, ast.Try):
            is_conditional = True
        if is_conditional:
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    cond_lines.add(child.lineno)
    return cond_lines


def _file_subsystem(relative_path: str) -> str:
    """Collapse a file path to its parent-directory (Python-package) granularity.

    ``plugins/transforms/llm/azure_batch.py`` → ``plugins/transforms/llm``.
    ``plugins/__init__.py`` → ``plugins``.
    Top-level files (``cli.py``) → ``.`` (root marker).
    """
    parent = str(Path(relative_path).parent)
    return parent if parent != "." else "."


def _file_loc(path: Path) -> int:
    """Count lines in a Python file. Returns 0 on read error."""
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return 0


def _gather_import_sites(
    py_file: Path,
    relative_path: str,
    tree: ast.Module,
    tc_lines: set[int],
    cond_lines: set[int],
) -> list[tuple[str | None, ast.Import | ast.ImportFrom, str, str | None]]:
    """Walk a tree and emit (resolved_module_name, node, target_alias_name, raw_relative_module).

    Returned list is consumed by the edge-builder.  ``resolved_module_name`` is
    the fully-qualified target (after relative-import resolution). The node is
    always either ``ast.Import`` or ``ast.ImportFrom`` (both carry ``lineno``).
    """
    out: list[tuple[str | None, ast.Import | ast.ImportFrom, str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative_module(relative_path, node.level, node.module)
            for alias in node.names:
                out.append((resolved, node, alias.name, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node, "", None))
    return out


def scan_dump_edges(
    root: Path,
    include_layers: frozenset[int],
    collapse_to_subsystem: bool,
    exclude_patterns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[str]]]:
    """Build the edge graph for ``dump-edges``.

    Returns (nodes, edges, sccs).  All collections are sorted deterministically.
    """
    exclude_patterns = exclude_patterns or []

    file_count: dict[str, int] = {}
    file_loc: dict[str, int] = {}
    node_layer: dict[str, int] = {}
    raw_edges: list[tuple[str, str, _ImportSite]] = []

    for py_file in sorted(root.rglob("*.py")):
        relative = py_file.relative_to(root)
        if any(part in _ALWAYS_EXCLUDED_DIRS for part in relative.parts):
            continue
        skip = False
        for pattern in exclude_patterns:
            if relative.match(pattern) or str(relative).startswith(pattern.rstrip("*/")):
                skip = True
                break
        if skip:
            continue

        rel_str = str(relative)
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            print(f"Warning: could not parse {py_file}: {exc}", file=sys.stderr)
            continue

        src_layer = _get_file_layer(rel_str)
        src_node_id = _file_subsystem(rel_str) if collapse_to_subsystem else rel_str

        # Always update node stats so unknown-layer nodes can still be excluded
        # in the final filter (consistent stats regardless of edge inclusion).
        file_count[src_node_id] = file_count.get(src_node_id, 0) + 1
        file_loc[src_node_id] = file_loc.get(src_node_id, 0) + len(source.splitlines())
        node_layer.setdefault(src_node_id, src_layer)

        if src_layer not in include_layers:
            continue

        tc_lines = _find_type_checking_lines(tree)
        cond_lines = _find_conditional_import_lines(tree)
        is_init = py_file.name == "__init__.py"

        for resolved_module, node, alias_name, _raw_module in _gather_import_sites(py_file, rel_str, tree, tc_lines, cond_lines):
            if resolved_module is None:
                continue
            if not resolved_module.startswith("elspeth"):
                continue

            target_path = _resolve_import_target(resolved_module, alias_name if alias_name else None, root)
            if target_path is None:
                continue

            try:
                target_rel = target_path.relative_to(root)
            except ValueError:
                continue
            target_rel_str = str(target_rel)
            target_layer = _get_file_layer(target_rel_str)
            if target_layer not in include_layers:
                continue

            tgt_node_id = _file_subsystem(target_rel_str) if collapse_to_subsystem else target_rel_str
            if collapse_to_subsystem and src_node_id == tgt_node_id:
                continue  # Δ3 rule 6: drop intra-subsystem self-edges with collapse ON.

            line = node.lineno
            is_tc = line in tc_lines
            is_cond = line in cond_lines
            # Δ3 rule 9: re-export when source is __init__.py AND it's a relative import.
            # Conservative: only relative imports inside __init__.py count; absolute imports
            # in __init__.py are also re-exports if target is a sibling submodule, but the
            # signal would be too noisy without __all__ tracking. Sticking to the relative-
            # import heuristic that catches the common pattern from ADR-006 era.
            is_reexport = bool(is_init and isinstance(node, ast.ImportFrom) and node.level > 0)

            target_module_qualified = f"{resolved_module}.{alias_name}" if alias_name else resolved_module
            raw_edges.append(
                (
                    src_node_id,
                    tgt_node_id,
                    _ImportSite(
                        src_file=rel_str,
                        line=line,
                        target_module=target_module_qualified,
                        type_checking=is_tc,
                        conditional=is_cond,
                        reexport=is_reexport,
                    ),
                )
            )

    # Aggregate by (src, tgt). Edge attributes use AND-aggregation: an aggregated
    # edge is type_checking_only iff EVERY underlying site is TC (any non-TC site
    # means the edge has runtime coupling). Same for conditional and reexport.
    edge_buckets: dict[tuple[str, str], list[_ImportSite]] = {}
    for src, tgt, site in raw_edges:
        edge_buckets.setdefault((src, tgt), []).append(site)

    edges_out: list[dict[str, Any]] = []
    for (src, tgt), sites in sorted(edge_buckets.items()):
        sites_sorted = sorted(sites, key=lambda s: (s.src_file, s.line, s.target_module))
        edges_out.append(
            {
                "from": src,
                "to": tgt,
                "weight": len(sites),
                "type_checking_only": all(s.type_checking for s in sites),
                "conditional": all(s.conditional for s in sites),
                "reexport": all(s.reexport for s in sites),
                "sample_sites": [{"file": s.src_file, "line": s.line} for s in sites_sorted[:3]],
            }
        )

    nodes_out: list[dict[str, Any]] = []
    edge_endpoint_ids: set[str] = set()
    for e in edges_out:
        edge_endpoint_ids.add(e["from"])
        edge_endpoint_ids.add(e["to"])

    for nid in sorted(file_count.keys()):
        layer = node_layer[nid]
        # Include the node only if its layer is in the include set.
        # Endpoint reachability ensures targets pulled in by edges are also included.
        if layer not in include_layers and nid not in edge_endpoint_ids:
            continue
        nodes_out.append(
            {
                "id": nid,
                "layer": LAYER_NAMES[layer],
                "file_count": file_count[nid],
                "loc": file_loc[nid],
            }
        )

    # SCC detection — Δ5 mandate. Edge endpoints not previously in nodes_out
    # (edges may cross into other layers if include_layers spans more than one)
    # are included by the local Tarjan implementation so the graph is closed.
    # Non-trivial SCCs (size ≥ 2) are reported.
    sccs_raw = _nontrivial_strongly_connected_components(
        node_ids=[n["id"] for n in nodes_out],
        edge_pairs=[(e["from"], e["to"]) for e in edges_out],
    )

    return nodes_out, edges_out, sccs_raw


def _nontrivial_strongly_connected_components(
    *,
    node_ids: list[str],
    edge_pairs: list[tuple[str, str]],
) -> list[list[str]]:
    """Return deterministic non-trivial SCCs using Tarjan's algorithm."""
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for src, tgt in edge_pairs:
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set())

    index_counter = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def strongconnect(node_id: str) -> None:
        nonlocal index_counter
        indices[node_id] = index_counter
        lowlinks[node_id] = index_counter
        index_counter += 1
        stack.append(node_id)
        on_stack.add(node_id)

        for successor in sorted(adjacency[node_id]):
            if successor not in indices:
                strongconnect(successor)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indices[successor])

        if lowlinks[node_id] != indices[node_id]:
            return

        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node_id:
                break
        if len(component) >= 2:
            components.append(sorted(component))

    for node_id in sorted(adjacency):
        if node_id not in indices:
            strongconnect(node_id)

    components.sort(key=lambda items: (len(items), items[0] if items else ""))
    return components


# =============================================================================
# Edge-Dump Output Formatters (JSON / Mermaid / DOT)
# =============================================================================


_STABLE_PLACEHOLDER = "<stable>"


def _tool_version_for_dump(use_stable_placeholder: bool) -> str:
    """Return a tool-version identifier.

    Uses a content hash of the enforcer file (cheaper and more deterministic than
    a git rev-parse subprocess; effectively the same identifying property).
    """
    if use_stable_placeholder:
        return _STABLE_PLACEHOLDER
    try:
        own_path = Path(__file__).resolve()
        digest = hashlib.sha256(own_path.read_bytes()).hexdigest()[:12]
        return f"sha256:{digest}"
    except (OSError, NameError):
        return "unknown"


def _generated_at(use_stable_placeholder: bool) -> str:
    if use_stable_placeholder:
        return _STABLE_PLACEHOLDER
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_dump_edges_json(
    *,
    root: Path,
    include_layers: frozenset[int],
    collapse_to_subsystem: bool,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    sccs: list[list[str]],
    use_stable_placeholder: bool,
) -> str:
    """Render the edge graph as deterministic JSON (Δ4 schema)."""
    layer_names_sorted = sorted(LAYER_NAMES[layer] for layer in include_layers)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": _generated_at(use_stable_placeholder),
        "tool_version": _tool_version_for_dump(use_stable_placeholder),
        "scope": {
            "root": str(root).replace("\\", "/"),
            "layers_included": layer_names_sorted,
            "collapsed_to_subsystem": collapse_to_subsystem,
        },
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "type_checking_edges": sum(1 for e in edges if e["type_checking_only"]),
            "conditional_edges": sum(1 for e in edges if e["conditional"]),
            "reexport_edges": sum(1 for e in edges if e["reexport"]),
            "scc_count": len(sccs),
            "largest_scc_size": max((len(s) for s in sccs), default=0),
        },
        "strongly_connected_components": sccs,
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def _mermaid_safe(node_id: str) -> str:
    """Sanitize a node id for Mermaid (no slashes, dots, etc.)."""
    return node_id.replace("/", "_").replace(".", "_").replace("-", "_") or "_root_"


def render_dump_edges_mermaid(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    """Render the edge graph as a Mermaid flowchart with subsystem subgraphs."""
    lines: list[str] = ["flowchart LR"]
    # Group nodes by their first path segment (top-level subsystem) for subgraphs.
    groups: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        top = n["id"].split("/", 1)[0] if "/" in n["id"] else n["id"]
        groups.setdefault(top, []).append(n)

    for top in sorted(groups):
        safe_top = _mermaid_safe(top)
        lines.append(f"    subgraph {safe_top}[{top}]")
        for n in sorted(groups[top], key=lambda x: x["id"]):
            safe_id = _mermaid_safe(n["id"])
            label = n["id"]
            lines.append(f'        {safe_id}["{label}<br/><sub>{n["loc"]} LOC</sub>"]')
        lines.append("    end")

    for e in edges:
        src = _mermaid_safe(e["from"])
        tgt = _mermaid_safe(e["to"])
        if e["type_checking_only"]:
            arrow = "-.->|TC|"
        elif e["conditional"]:
            arrow = "-.->|cond|"
        elif e["weight"] >= 10:
            arrow = f"==>|{e['weight']}|"
        else:
            arrow = f"-->|{e['weight']}|" if e["weight"] > 1 else "-->"
        lines.append(f"    {src} {arrow} {tgt}")

    return "\n".join(lines) + "\n"


def _dot_safe(node_id: str) -> str:
    return '"' + node_id.replace('"', '\\"') + '"'


def render_dump_edges_dot(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    """Render the edge graph as a Graphviz digraph."""
    lines: list[str] = ["digraph l3_imports {", "    rankdir=LR;", "    node [shape=box, style=rounded];"]

    # Cluster by top-level subsystem.
    groups: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        top = n["id"].split("/", 1)[0] if "/" in n["id"] else n["id"]
        groups.setdefault(top, []).append(n)

    for cluster_idx, top in enumerate(sorted(groups)):
        lines.append(f'    subgraph "cluster_{cluster_idx}" {{')
        lines.append(f"        label={_dot_safe(top)};")
        lines.append("        style=dashed;")
        for n in sorted(groups[top], key=lambda x: x["id"]):
            label = f"{n['id']}\\n{n['loc']} LOC"
            lines.append(f"        {_dot_safe(n['id'])} [label={_dot_safe(label)}];")
        lines.append("    }")

    for e in edges:
        attrs: list[str] = []
        if e["type_checking_only"]:
            attrs.append('style="dashed"')
            attrs.append('label="TC"')
        elif e["conditional"]:
            attrs.append('style="dotted"')
            attrs.append('label="cond"')
        else:
            attrs.append(f'label="{e["weight"]}"')
            if e["weight"] >= 10:
                attrs.append("penwidth=3")
        attr_str = ", ".join(attrs)
        lines.append(f"    {_dot_safe(e['from'])} -> {_dot_safe(e['to'])} [{attr_str}];")

    lines.append("}")
    return "\n".join(lines) + "\n"


# =============================================================================
# Allowlist Handling
# =============================================================================


_BANNED_RULES = frozenset(rule_id for rule_id, rule_def in RULES.items() if rule_def.get("banned"))
_ALL_RULE_IDS = frozenset(RULES.keys())
_ALLOWLIST_PATTERN_TAGS = frozenset(
    {
        "audit-record-fallback",
        "display-fallback",
        "external-boundary-validator",
        "external-dependency-optional",
        "ordered-fallback",
        "plugin-contract-offensive-guard",
        "post-init-offensive-guard",
        "post-point-of-no-return-recovery",
        "rollback-preserves-primary-error",
        "retry-orchestration",
        "tier3-boundary-validator",
        "tier3-narrow-catch",
        "union-dispatch",
    }
)

# Directories that are always excluded from scanning — vendored/third-party code
# that happens to contain .py files but is not part of the ELSPETH codebase.
_ALWAYS_EXCLUDED_DIRS = ("node_modules",)


def _validate_allowlist_governance(allowlist: Allowlist) -> None:
    """Apply tier_model's domain-specific governance to a loaded ``Allowlist``.

    The core loader (``elspeth_lints.core.allowlist.load_allowlist``) is generic.
    tier_model adds three checks the generic loader does not:

    1. **Banned-rule discipline**: ``RULES`` entries marked ``banned: True``
       cannot be allowlisted via ``allow_hits[].key`` (the ``rule_id`` is the
       second ``:``-separated segment) or via ``per_file_rules[].rules``.
    2. **Registry coherence on allow_hits keys**: core only validates
       ``per_file_rules[].rules`` against ``valid_rule_ids``; the rule-id
       embedded in an ``allow_hits[].key`` is opaque to core and is checked
       here.
    3. **Pattern-tag governance**: an ``allow_hits[].pattern`` value must be
       drawn from the closed ``_ALLOWLIST_PATTERN_TAGS`` vocabulary, and
       ``owner: bugfix`` entries require either an expiry or a pattern tag
       (a "permanent bugfix" needs an architectural category, not silent debt).

    Raises ``ValueError`` on any violation. The CLI converts that to exit-2.
    Replaces the previous ``sys.exit(1)`` / ``print(..., file=sys.stderr)``
    pattern, which violated the CLAUDE.md audit-primacy order (audit/exception
    before stderr logging).
    """
    for entry in allowlist.entries:
        source_ctx = f" in {entry.source_file}" if entry.source_file else ""
        parts = entry.key.split(":")
        if len(parts) < 2:
            raise ValueError(f"allow_hits entry has malformed key (expected 'file:rule_id:...' format){source_ctx}: {entry.key!r}")
        rule_id = parts[1]
        if rule_id in _BANNED_RULES:
            raise ValueError(f"allow_hits entry uses banned rule {rule_id} (cannot be allowlisted){source_ctx}: {entry.key}")
        if rule_id not in _ALL_RULE_IDS:
            raise ValueError(f"allow_hits entry has unknown rule ID '{rule_id}'{source_ctx}: {entry.key}")
        if entry.pattern is not None and entry.pattern not in _ALLOWLIST_PATTERN_TAGS:
            raise ValueError(f"allow_hits entry has unknown pattern tag{source_ctx}: {entry.pattern!r}")
        if entry.owner == "bugfix" and entry.expires is None and entry.pattern is None:
            raise ValueError(f"owner=bugfix allow_hits entry must define expires or pattern{source_ctx}: {entry.key}")

    for rule in allowlist.per_file_rules:
        source_ctx = f" in {rule.source_file}" if rule.source_file else ""
        rule_ids = set(rule.rules)
        banned = rule_ids & _BANNED_RULES
        if banned:
            raise ValueError(
                f"per_file_rules entry for '{rule.pattern}'{source_ctx} uses banned rule(s) {sorted(banned)} (cannot be allowlisted)"
            )
        # Note: unknown-rule-id check is enforced by the core loader when
        # ``valid_rule_ids=_ALL_RULE_IDS`` is passed to ``load_allowlist``.


def _finding_key_for(finding: Finding) -> FindingKey:
    """Convert a tier_model ``Finding`` to the shared allowlist key shape."""
    return FindingKey(
        file_path=finding.file_path,
        rule_id=finding.rule_id,
        symbol_context=finding.symbol_context,
        fingerprint=finding.fingerprint,
    )


def _match_per_file_rule(rules: list[PerFileRule], finding_key: FindingKey) -> PerFileRule | None:
    """Return the first per-file rule matching ``finding_key``."""
    for rule in rules:
        if rule.matches(finding_key):
            return rule
    return None


def _match_finding(allowlist: Allowlist, finding: Finding) -> AllowlistEntry | PerFileRule | None:
    """Match a tier_model ``Finding`` against ``allowlist``.

    Preserves tier_model's historical match order: per-file rules are checked
    **before** specific entries. Core's ``Allowlist.match`` checks entries
    first; that order would silently shift ``matched`` / ``matched_count``
    accounting (and thus ``get_unused_entries`` / ``get_unused_rules``
    diagnostics) for findings covered by both an exact entry and a per-file
    rule. We keep the historical order to preserve the production
    ``contracts.yaml`` semantics across this consolidation.
    """
    finding_key = _finding_key_for(finding)
    matched_rule = _match_per_file_rule(allowlist.per_file_rules, finding_key)
    if matched_rule is not None:
        matched_rule.matched_count += 1
        return matched_rule
    for entry in allowlist.entries:
        if entry.matches(finding_key):
            # C8-3 in-file transplant defence: a quartet copied onto a
            # different AST node within the same file still matches the
            # canonical key (the key carries fingerprint, which is
            # path-derived, so the keys differ — but a hand-edited
            # transplant could rebind the key and copy the quartet). The
            # persisted ast_path is the AST-level address the judge
            # actually inspected; it must equal the live finding's
            # ast_path. Mismatch ⇒ tampering or unannounced refactor.
            verify_entry_binding_against_finding(
                entry,
                file_path=finding.file_path,
                ast_path=finding.ast_path,
                scope_fingerprint=finding.scope_fingerprint,
            )
            entry.matched = True
            return entry
    # Exact key missed. A judge-gated v2 entry whose module-rooted ast_path
    # drifted (an unrelated module-level statement shifted the leading index)
    # but whose enclosing scope + within-scope position are unchanged is the
    # same suppression, relocated. The fallback's predicate (scope_fingerprint
    # equality + within-scope ast_path suffix equality) IS its binding check and
    # is at least as strong as the exact-match transplant defence, so we do NOT
    # also call verify_entry_binding_against_finding (which asserts ast_path
    # equality and would crash by construction on the fallback path).
    fallback = find_scope_fallback_entry(
        allowlist.entries,
        canonical_key=finding.canonical_key,
        scope_fingerprint=finding.scope_fingerprint,
        ast_path=finding.ast_path,
        scope_depth=finding.scope_depth,
    )
    if fallback is not None:
        fallback.matched = True
        return fallback
    return None


def _load_tier_model_allowlist(path: Path, *, source_root: Path | None = None) -> Allowlist:
    """Load the tier-model allowlist via core's loader, then apply local governance.

    Missing allowlist files are configuration errors, not empty allowlists:
    the allowlist YAML is Tier-1 audit data and a ghost path must fail closed.
    Governance (banned rules, pattern-tag vocabulary, bugfix-owner discipline,
    registry coherence on ``allow_hits[].key``) is applied after the generic load.

    ``source_root`` is passed through to the core loader so judge-gated
    entries can have their ``file_fingerprint`` verified against the
    current bytes of the source file at the path encoded in their key.
    Cross-file quartet transplants fail at load time; in-file transplants
    are caught at match time in :func:`_match_finding`.
    """
    allowlist = load_allowlist(path, valid_rule_ids=_ALL_RULE_IDS, source_root=source_root)
    _validate_allowlist_governance(allowlist)
    return allowlist


# =============================================================================
# Reporting
# =============================================================================


def _suggest_module_file(finding: Finding, allowlist_path: Path) -> str:
    """Suggest the appropriate module YAML file for a finding.

    Maps the finding's file path to the per-module YAML file name.
    Only meaningful when allowlist_path is a directory.
    """
    if not allowlist_path.is_dir():
        return str(allowlist_path)

    file_path = finding.file_path
    # Bare filenames (no /) like cli.py, cli_helpers.py → cli.yaml
    if "/" not in file_path:
        stem = file_path.removesuffix(".py")
        if stem.startswith("cli"):
            return str(allowlist_path / "cli.yaml")
        return str(allowlist_path / f"{stem}.yaml")

    # First path segment determines module file
    module = file_path.split("/", 1)[0]
    return str(allowlist_path / f"{module}.yaml")


def format_finding_text(finding: Finding) -> str:
    """Format a finding for text output."""
    rule = RULES.get(finding.rule_id, {})
    lines = [
        f"\n{finding.file_path}:{finding.line}:{finding.col}",
        f"  Rule: {finding.rule_id} - {rule.get('name', 'unknown')}",
        f"  Code: {finding.code_snippet}",
        f"  Context: {'.'.join(finding.symbol_context) if finding.symbol_context else '<module>'}",
        f"  Issue: {rule.get('description', finding.message)}",
        f"  Fix: {rule.get('remediation', 'Review and fix the underlying issue')}",
        f"  Allowlist key: {finding.canonical_key}",
    ]
    return "\n".join(lines)


def format_stale_entry_text(entry: AllowlistEntry) -> str:
    """Format a stale allowlist entry for text output."""
    base = f"\n  Key: {entry.key}\n  Owner: {entry.owner}\n  Reason: {entry.reason}"
    if entry.source_file:
        base += f"\n  Source: {entry.source_file}"
    return base


def format_expired_entry_text(entry: AllowlistEntry) -> str:
    """Format an expired allowlist entry for text output."""
    return f"\n  Key: {entry.key}\n  Owner: {entry.owner}\n  Expired: {entry.expires}"


def report_json(
    violations: list[Finding],
    stale_entries: list[AllowlistEntry],
    expired_entries: list[AllowlistEntry],
    suppressed_boundary_findings: list[Finding] | None = None,
    expired_file_rules: list[PerFileRule] | None = None,
    unused_file_rules: list[PerFileRule] | None = None,
    layer_warnings: list[Finding] | None = None,
    exceeded_file_rules: list[PerFileRule] | None = None,
    budget_violations: list[AllowlistBudgetViolation] | None = None,
) -> str:
    """Generate JSON report."""
    result: dict[str, Any] = {
        "violations": [
            {
                "rule_id": f.rule_id,
                "file": f.file_path,
                "line": f.line,
                "col": f.col,
                "context": list(f.symbol_context),
                "fingerprint": f.fingerprint,
                "code": f.code_snippet,
                "message": f.message,
                "key": f.canonical_key,
            }
            for f in violations
        ],
        "stale_allowlist_entries": [{"key": e.key, "owner": e.owner, "reason": e.reason} for e in stale_entries],
        "expired_allowlist_entries": [{"key": e.key, "owner": e.owner, "expires": str(e.expires)} for e in expired_entries],
    }
    if expired_file_rules:
        result["expired_file_rules"] = [
            {"pattern": r.pattern, "rules": r.rules, "reason": r.reason, "expires": str(r.expires)} for r in expired_file_rules
        ]
    if suppressed_boundary_findings:
        result["suppressed_trust_boundary_findings"] = [
            {
                "rule_id": f.rule_id,
                "file": f.file_path,
                "line": f.line,
                "col": f.col,
                "context": list(f.symbol_context),
                "fingerprint": f.fingerprint,
                "code": f.code_snippet,
                "message": f.message,
                "key": f.canonical_key,
            }
            for f in suppressed_boundary_findings
        ]
    if unused_file_rules:
        result["unused_file_rules"] = [{"pattern": r.pattern, "rules": r.rules, "reason": r.reason} for r in unused_file_rules]
    if exceeded_file_rules:
        result["exceeded_file_rules"] = [
            {"pattern": r.pattern, "rules": r.rules, "matched": r.matched_count, "max_hits": r.max_hits, "reason": r.reason}
            for r in exceeded_file_rules
        ]
    if budget_violations:
        result["allowlist_budget_violations"] = [
            {"category": v.category, "current": v.current, "max_allowed": v.max_allowed} for v in budget_violations
        ]
    if layer_warnings:
        result["layer_warnings"] = [
            {
                "rule_id": f.rule_id,
                "file": f.file_path,
                "line": f.line,
                "message": f.message,
                "key": f.canonical_key,
            }
            for f in layer_warnings
        ]
    return json.dumps(result, indent=2)


def collect_check_result(
    root: Path,
    *,
    allowlist_path: Path | None = None,
    exclude_patterns: list[str] | None = None,
    files: list[Path] | None = None,
) -> CheckResult:
    """Run check-mode analysis and return the structured legacy result."""
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"{resolved_root} is not a directory")

    resolved_allowlist_path = allowlist_path if allowlist_path is not None else _default_allowlist_path(resolved_root)
    allowlist = _load_tier_model_allowlist(resolved_allowlist_path, source_root=resolved_root)
    exclude_patterns = exclude_patterns or []
    files = files or []

    all_tc_findings: list[Finding] = []
    suppressed_boundary_findings: list[Finding] = []
    if files:
        all_findings: list[Finding] = []
        for file_path in files:
            resolved = file_path.resolve()
            try:
                resolved.relative_to(resolved_root)
            except ValueError:
                continue
            file_findings, file_suppressed = scan_file_with_observations(resolved, resolved_root)
            all_findings.extend(file_findings)
            suppressed_boundary_findings.extend(file_suppressed)
            layer_v, layer_tc = scan_layer_imports_file(resolved, resolved_root)
            all_findings.extend(layer_v)
            all_tc_findings.extend(layer_tc)
    else:
        all_findings, suppressed_boundary_findings = scan_directory_with_observations(resolved_root, exclude_patterns)
        layer_v, layer_tc = scan_layer_imports_directory(resolved_root, exclude_patterns)
        all_findings.extend(layer_v)
        all_tc_findings.extend(layer_tc)

    violations: list[Finding] = []
    for finding in all_findings:
        if finding.rule_id in _BANNED_RULES or _match_finding(allowlist, finding) is None:
            violations.append(finding)

    layer_warnings: list[Finding] = []
    for tc_finding in all_tc_findings:
        if _match_finding(allowlist, tc_finding) is None:
            layer_warnings.append(tc_finding)

    if files:
        stale_entries: list[AllowlistEntry] = []
        expired_entries: list[AllowlistEntry] = []
        expired_file_rules: list[PerFileRule] = []
        unused_file_rules: list[PerFileRule] = []
        exceeded_file_rules: list[PerFileRule] = []
    else:
        stale_entries = allowlist.get_unused_entries() if allowlist.fail_on_stale else []
        expired_entries = allowlist.get_expired_entries() if allowlist.fail_on_expired else []
        expired_file_rules = allowlist.get_expired_rules() if allowlist.fail_on_expired else []
        unused_file_rules = allowlist.get_unused_rules() if allowlist.fail_on_stale else []
        exceeded_file_rules = allowlist.get_exceeded_rules()

    return CheckResult(
        allowlist_path=resolved_allowlist_path,
        violations=violations,
        suppressed_boundary_findings=suppressed_boundary_findings,
        stale_entries=stale_entries,
        expired_entries=expired_entries,
        expired_file_rules=expired_file_rules,
        unused_file_rules=unused_file_rules,
        layer_warnings=layer_warnings,
        exceeded_file_rules=exceeded_file_rules,
        budget_violations=allowlist.get_budget_violations(),
    )


def _default_allowlist_path(root: Path) -> Path:
    """Return the tier-model allowlist path for a repo or scan-root path."""
    relative_dir = Path("config") / "cicd" / "enforce_tier_model"
    relative_file = Path("config") / "cicd" / "enforce_tier_model.yaml"
    local_dir = root / relative_dir
    if local_dir.is_dir():
        return local_dir
    local_file = root / relative_file
    if local_file.exists():
        return local_file

    if not _looks_like_elspeth_source_root(root):
        return local_dir

    candidates = root.parents
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        dir_path = resolved / relative_dir
        if dir_path.is_dir():
            return dir_path
        file_path = resolved / relative_file
        if file_path.exists():
            return file_path
    return root / relative_dir


def _looks_like_elspeth_source_root(root: Path) -> bool:
    """Return whether upward allowlist discovery is safe for a source-tree root."""
    return (root.name == "elspeth" and root.parent.name == "src") or (root.name == "src" and (root / "elspeth").is_dir())


def _legacy_finding_to_lint(finding: Finding) -> LintFinding:
    rule = RULES.get(finding.rule_id, {})
    return LintFinding(
        rule_id=finding.rule_id,
        file_path=finding.file_path,
        line=finding.line,
        column=finding.col,
        message=finding.message,
        fingerprint=finding.fingerprint,
        severity=Severity.NOTE if finding.rule_id == "R_TB_SUPPRESSED" else RULE_METADATA.severity,
        suggestion=rule.get("remediation"),
    )


def _allowlist_diagnostics_to_lints(result: CheckResult) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for entry in result.stale_entries:
        findings.append(
            _diagnostic_finding(
                result.allowlist_path,
                entry.source_file,
                message=f"Stale tier-model allowlist entry: {entry.key}",
                fingerprint_payload=f"stale:{entry.key}",
            )
        )
    for entry in result.expired_entries:
        findings.append(
            _diagnostic_finding(
                result.allowlist_path,
                entry.source_file,
                message=f"Expired tier-model allowlist entry: {entry.key}",
                fingerprint_payload=f"expired:{entry.key}:{entry.expires}",
            )
        )
    for rule in result.expired_file_rules:
        findings.append(
            _diagnostic_finding(
                result.allowlist_path,
                rule.source_file,
                message=f"Expired tier-model per-file rule: {rule.pattern}",
                fingerprint_payload=f"expired-file-rule:{rule.pattern}:{rule.expires}",
            )
        )
    for rule in result.unused_file_rules:
        findings.append(
            _diagnostic_finding(
                result.allowlist_path,
                rule.source_file,
                message=f"Unused tier-model per-file rule: {rule.pattern}",
                fingerprint_payload=f"unused-file-rule:{rule.pattern}",
            )
        )
    for rule in result.exceeded_file_rules:
        findings.append(
            _diagnostic_finding(
                result.allowlist_path,
                rule.source_file,
                message=f"Tier-model per-file rule exceeded max_hits: {rule.pattern} matched {rule.matched_count}/{rule.max_hits}",
                fingerprint_payload=f"exceeded-file-rule:{rule.pattern}:{rule.max_hits}",
            )
        )
    for violation in result.budget_violations:
        findings.append(
            _diagnostic_finding(
                result.allowlist_path,
                "_defaults.yaml",
                message=(f"Tier-model allowlist budget exceeded: {violation.category} is {violation.current}/{violation.max_allowed}"),
                fingerprint_payload=f"budget:{violation.category}:{violation.max_allowed}",
            )
        )
    return findings


def _diagnostic_finding(allowlist_path: Path, source_file: str, *, message: str, fingerprint_payload: str) -> LintFinding:
    path = allowlist_path / source_file if source_file and allowlist_path.is_dir() else allowlist_path
    return LintFinding(
        rule_id=RULE_ID,
        file_path=_repo_display_path(path),
        line=1,
        column=0,
        message=message,
        fingerprint=hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16],
        severity=Severity.ERROR,
    )


def _repo_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _display_rule_file_path(file_path: Path, root: Path) -> str:
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


def _source_lines_for_rule(file_path: Path) -> list[str]:
    try:
        return file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


@dataclass(frozen=True, slots=True)
class TierModelRule:
    """Run the legacy trust-tier analyzer through the elspeth-lints protocol."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[LintFinding]:
        """Analyze one syntax tree for focused tests, or the whole repository root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            visitor = TierModelVisitor(_display_rule_file_path(file_path, context.root), _source_lines_for_rule(file_path))
            visitor.visit(tree)
            return [_legacy_finding_to_lint(finding) for finding in [*visitor.findings, *visitor.suppressed_findings]]

        result = collect_check_result(context.root, allowlist_path=context.allowlist_dir_override)
        return [
            _legacy_finding_to_lint(finding) for finding in [*result.violations, *result.suppressed_boundary_findings]
        ] + _allowlist_diagnostics_to_lints(result)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="No Bug-Hiding Enforcement Tool - detect defensive patterns that mask bugs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check subcommand
    check_parser = subparsers.add_parser("check", help="Check for bug-hiding patterns")
    check_parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to scan",
    )
    check_parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Path to allowlist YAML file or directory of YAML files",
    )
    check_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob patterns to exclude (can be specified multiple times)",
    )
    check_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    check_parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific files to check (from pre-commit). If empty, scans --root directory.",
    )

    # dump-edges subcommand (Phase 0 — L3↔L3 import-graph oracle)
    dump_parser = subparsers.add_parser(
        "dump-edges",
        help="Emit a deterministic import-graph for analysis (does NOT fail on graph content)",
    )
    dump_parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to scan",
    )
    dump_parser.add_argument(
        "--format",
        choices=["json", "mermaid", "dot"],
        default="json",
        help="Output format (default: json)",
    )
    dump_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (required for json/dot; mermaid may write to stdout when omitted)",
    )
    dump_parser.add_argument(
        "--include-layer",
        action="append",
        default=None,
        choices=["L0", "L1", "L2", "L3"],
        help="Layer(s) to include in the graph; repeatable. Default: L3 only.",
    )
    dump_parser.add_argument(
        "--collapse-to-subsystem",
        dest="collapse_to_subsystem",
        action="store_true",
        default=True,
        help="Aggregate edges to package (parent-directory) granularity (default: ON)",
    )
    dump_parser.add_argument(
        "--no-collapse",
        dest="collapse_to_subsystem",
        action="store_false",
        help="Disable subsystem collapse; emit file-level edges",
    )
    dump_parser.add_argument(
        "--no-timestamp",
        action="store_true",
        default=False,
        help="Replace generated_at and tool_version with stable placeholders for diff-friendly output",
    )
    dump_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob patterns to exclude (repeatable)",
    )

    args = parser.parse_args()

    if args.command == "check":
        return run_check(args)
    if args.command == "dump-edges":
        return run_dump_edges(args)

    return 0


def run_dump_edges(args: argparse.Namespace) -> int:
    """Run the dump-edges command.

    Always exits 0 unless the tool itself errors. Cycle detection is observational —
    a stderr WARNING is printed when non-trivial SCCs are found, but the exit code
    is unaffected (Δ5: not enforcement).
    """
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    # Resolve include-layer set (default: L3 only).
    layer_strs = args.include_layer or ["L3"]
    include_layers = frozenset(_LAYER_NAME_TO_INT[s] for s in layer_strs)

    # JSON and DOT require --output; Mermaid may write to stdout.
    if args.format in ("json", "dot") and args.output is None:
        print(f"Error: --output is required for --format {args.format}", file=sys.stderr)
        return 1

    nodes, edges, sccs = scan_dump_edges(
        root=root,
        include_layers=include_layers,
        collapse_to_subsystem=args.collapse_to_subsystem,
        exclude_patterns=args.exclude,
    )

    if args.format == "json":
        rendered = render_dump_edges_json(
            root=args.root,  # the un-resolved value, so output is portable
            include_layers=include_layers,
            collapse_to_subsystem=args.collapse_to_subsystem,
            nodes=nodes,
            edges=edges,
            sccs=sccs,
            use_stable_placeholder=args.no_timestamp,
        )
    elif args.format == "mermaid":
        rendered = render_dump_edges_mermaid(nodes, edges)
    elif args.format == "dot":
        rendered = render_dump_edges_dot(nodes, edges)
    else:
        # argparse's `choices` should make this unreachable.
        print(f"Error: unknown format {args.format!r}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        # Mermaid → stdout when --output omitted
        print(rendered, end="")

    if sccs:
        print(
            f"WARNING: {len(sccs)} non-trivial strongly-connected component(s) detected at "
            f"{','.join(sorted(LAYER_NAMES[layer] for layer in include_layers))}.",
            file=sys.stderr,
        )
        if args.output is not None:
            print(
                f"         See {args.output} stats.scc_count for details.",
                file=sys.stderr,
            )

    return 0


def run_check(args: argparse.Namespace) -> int:
    """Run the check command."""
    try:
        result = collect_check_result(
            args.root,
            allowlist_path=args.allowlist,
            exclude_patterns=args.exclude,
            files=args.files,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(
            report_json(
                result.violations,
                result.stale_entries,
                result.expired_entries,
                result.suppressed_boundary_findings,
                result.expired_file_rules,
                result.unused_file_rules,
                result.layer_warnings,
                result.exceeded_file_rules,
                result.budget_violations,
            )
        )
    else:
        # Text format
        if result.violations:
            print(f"\n{'=' * 60}")
            print(f"VIOLATIONS FOUND: {len(result.violations)}")
            print("=" * 60)
            for v in result.violations:
                print(format_finding_text(v))

        if result.layer_warnings:
            print(f"\n{'=' * 60}")
            print(f"LAYER WARNINGS (TYPE_CHECKING imports): {len(result.layer_warnings)}")
            print("(Allowlist with rule TC to suppress — not a failure)")
            print("=" * 60)
            for w in result.layer_warnings:
                print(f"  {w.file_path}:{w.line} — {w.message}")
                print(f"    Code: {w.code_snippet}")
                print(f"    Allowlist key: {w.canonical_key}")

        if result.suppressed_boundary_findings:
            print(f"\n{'=' * 60}")
            print(f"TRUST BOUNDARY SUPPRESSIONS (observations): {len(result.suppressed_boundary_findings)}")
            print("(Non-failing audit records for R1/R5 findings suppressed by @trust_boundary)")
            print("=" * 60)
            for finding in result.suppressed_boundary_findings:
                print(f"  {finding.file_path}:{finding.line} — {finding.message}")
                print(f"    Code: {finding.code_snippet}")

        if result.stale_entries:
            print(f"\n{'=' * 60}")
            print(f"STALE ALLOWLIST ENTRIES: {len(result.stale_entries)}")
            print("(These entries don't match any code - remove them)")
            print("=" * 60)
            for e in result.stale_entries:
                print(format_stale_entry_text(e))

        if result.expired_entries:
            print(f"\n{'=' * 60}")
            print(f"EXPIRED ALLOWLIST ENTRIES: {len(result.expired_entries)}")
            print("(These entries have passed their expiration date)")
            print("=" * 60)
            for e in result.expired_entries:
                print(format_expired_entry_text(e))

        if result.expired_file_rules:
            print(f"\n{'=' * 60}")
            print(f"EXPIRED PER-FILE RULES: {len(result.expired_file_rules)}")
            print("(These rules have passed their expiration date)")
            print("=" * 60)
            for r in result.expired_file_rules:
                print(f"\n  Pattern: {r.pattern}")
                print(f"  Rules: {r.rules}")
                print(f"  Reason: {r.reason}")
                print(f"  Expired: {r.expires}")

        if result.unused_file_rules:
            print(f"\n{'=' * 60}")
            print(f"UNUSED PER-FILE RULES: {len(result.unused_file_rules)}")
            print("(These rules didn't match any code - consider removing)")
            print("=" * 60)
            for r in result.unused_file_rules:
                print(f"\n  Pattern: {r.pattern}")
                print(f"  Rules: {r.rules}")
                print(f"  Reason: {r.reason}")

        if result.exceeded_file_rules:
            print(f"\n{'=' * 60}")
            print(f"EXCEEDED PER-FILE RULES: {len(result.exceeded_file_rules)}")
            print("(These rules matched more findings than max_hits allows - review new additions)")
            print("=" * 60)
            for r in result.exceeded_file_rules:
                print(f"\n  Pattern: {r.pattern}")
                print(f"  Rules: {r.rules}")
                print(f"  Matched: {r.matched_count} (max_hits: {r.max_hits})")
                print(f"  Reason: {r.reason}")

        if result.budget_violations:
            print(f"\n{'=' * 60}")
            print(f"ALLOWLIST BUDGET EXCEEDED: {len(result.budget_violations)}")
            print("(Allowlist counts exceeded configured ratchet ceilings - delete entries or update the budget deliberately)")
            print("=" * 60)
            for budget_violation in result.budget_violations:
                print(f"\n  Category: {budget_violation.category}")
                print(f"  Current: {budget_violation.current}")
                print(f"  Max allowed: {budget_violation.max_allowed}")

        if result.has_errors:
            print(f"\n{'=' * 60}")
            print("CHECK FAILED")
            print("=" * 60)
            if result.violations:
                target = _suggest_module_file(result.violations[0], result.allowlist_path)
                print(f"\nTo allowlist a violation, add an entry to {target}")
                print("Example entry:")
                import pprint

                pprint.pprint(result.violations[0].suggested_allowlist_entry())
        else:
            print("\nNo bug-hiding patterns detected. Check passed.")

    return 1 if result.has_errors else 0


RULE = TierModelRule()


if __name__ == "__main__":
    sys.exit(main())
