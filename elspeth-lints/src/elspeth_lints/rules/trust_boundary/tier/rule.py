"""Trust-boundary tier rule implementation.

Honesty gate: the ``tier`` kwarg of every ``@trust_boundary(...)`` must be the
literal integer ``3``. Anything else — a wrong integer, a string ``"3"``, a
boolean, a name reference — is rejected. The runtime decorator already raises
:class:`TypeError` for the wrong-integer case at import time; this static
rule catches the wider failure mode where the value isn't even comparable
to an int at decoration time (e.g. an :class:`ast.Name` that happens to
evaluate to 3 at runtime — the static analyzer cannot prove that, and the
honesty contract is that the decorator's metadata is literal and
human-readable).

Non-literal kwargs on the decorator now produce a TBT2 finding rather
than silently deferring to ``trust_tier.tier_model`` — see the
``trust_boundary.shared.KeywordExtraction`` docstring and ticket
elspeth-1f4634235a (C6-4).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import PythonFileReadError, PythonSyntaxError, walk_python_files
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.trust_boundary.shared import (
    display_path,
    extract_keywords,
    filter_allowlisted_findings,
    iter_trust_boundary_decorators,
    load_honesty_gate_allowlist,
    make_decorator_finding,
)
from elspeth_lints.rules.trust_boundary.tier.metadata import (
    RULE_ID,
    RULE_INVALID,
    RULE_METADATA,
    RULE_NONLITERAL,
    SUGGESTION_INVALID,
    SUGGESTION_NONLITERAL,
)

_ALLOWLIST_RULE_IDS = frozenset({RULE_INVALID, RULE_NONLITERAL})


@dataclass(frozen=True, slots=True)
class TrustBoundaryTierRule:
    """Detect ``@trust_boundary`` decorators whose ``tier`` is not literally ``3``."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one tree directly (for focused tests) or walk the scan root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return analyze_tree(tree, display_path(file_path, context.root))
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


def analyze_tree(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return ``trust_boundary.tier`` findings for one parsed syntax tree."""
    findings: list[Finding] = []
    for func_node, call in iter_trust_boundary_decorators(tree):
        extraction = extract_keywords(call)
        if extraction.kwargs is None:
            # Non-literal kwarg (or **-unpacking / positional args). Per the
            # C6-4 honesty-gate hardening (epic elspeth-2ed3bb0f7d, ticket
            # elspeth-1f4634235a) we self-enforce instead of deferring to
            # ``trust_tier.tier_model`` — suppressing tier_model on a file
            # must NOT grant honesty-gate immunity here. The redundant
            # finding when tier_model is also active is deliberate.
            assert extraction.nonliteral_message is not None  # tagged union invariant
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    file_path=file_path,
                    call=call,
                    rule_id=RULE_NONLITERAL,
                    message=extraction.nonliteral_message,
                    suggestion=SUGGESTION_NONLITERAL,
                    symbol_context=(func_node.name,),
                )
            )
            continue
        kwargs = extraction.kwargs
        if "tier" not in kwargs:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    file_path=file_path,
                    call=call,
                    rule_id=RULE_INVALID,
                    message=("@trust_boundary is missing the 'tier' kwarg; tier=3 is mandatory."),
                    suggestion=SUGGESTION_INVALID,
                    symbol_context=(func_node.name,),
                )
            )
            continue
        tier_value = kwargs["tier"]
        # ``bool`` is a subclass of ``int`` in Python; reject booleans explicitly
        # so ``tier=True`` (which would equal ``1``) cannot pass.
        if isinstance(tier_value, bool) or not isinstance(tier_value, int) or tier_value != 3:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    file_path=file_path,
                    call=call,
                    rule_id=RULE_INVALID,
                    message=(f"@trust_boundary tier must be the literal integer 3, got {tier_value!r} ({type(tier_value).__name__})."),
                    suggestion=SUGGESTION_INVALID,
                    symbol_context=(func_node.name,),
                )
            )
    return findings


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Walk every Python file under ``root`` and aggregate findings."""
    allowlist = load_honesty_gate_allowlist(
        root,
        allowlist_dir_override=allowlist_dir_override,
        valid_rule_ids=_ALLOWLIST_RULE_IDS,
    )
    findings: list[Finding] = []
    for item in walk_python_files(root):
        # Skip non-analysable per-file results. The CLI driver surfaces
        # both syntax errors and read errors as per-file diagnostic
        # findings; per-rule loops have nothing rule-specific to add.
        if isinstance(item, (PythonSyntaxError, PythonFileReadError)):
            continue
        findings.extend(analyze_tree(item.tree, display_path(item.path, root)))
    return filter_allowlisted_findings(findings, allowlist)


RULE = TrustBoundaryTierRule()
