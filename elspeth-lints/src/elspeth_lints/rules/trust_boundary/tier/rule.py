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
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import PythonSyntaxError, walk_python_files
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.trust_boundary.shared import display_path, extract_keywords, iter_trust_boundary_decorators
from elspeth_lints.rules.trust_boundary.tier.metadata import (
    RULE_ID,
    RULE_INVALID,
    RULE_METADATA,
    SUGGESTION_INVALID,
)


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
        return scan_root(context.root)


def analyze_tree(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return ``trust_boundary.tier`` findings for one parsed syntax tree."""
    findings: list[Finding] = []
    for _func_node, call in iter_trust_boundary_decorators(tree):
        kwargs = extract_keywords(call)
        if kwargs is None:
            # Non-literal kwargs are reported by ``trust_tier.tier_model``'s
            # ``R_TB_NONLITERAL`` / ``R_TB_MALFORMED``; this rule degrades
            # silently to avoid double-reporting.
            continue
        if "tier" not in kwargs:
            findings.append(
                _make_finding(
                    file_path=file_path,
                    call=call,
                    message=(
                        "@trust_boundary is missing the 'tier' kwarg; "
                        "tier=3 is mandatory."
                    ),
                )
            )
            continue
        tier_value = kwargs["tier"]
        # ``bool`` is a subclass of ``int`` in Python; reject booleans explicitly
        # so ``tier=True`` (which would equal ``1``) cannot pass.
        if isinstance(tier_value, bool) or not isinstance(tier_value, int) or tier_value != 3:
            findings.append(
                _make_finding(
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary tier must be the literal integer 3, got {tier_value!r} "
                        f"({type(tier_value).__name__})."
                    ),
                )
            )
    return findings


def scan_root(root: Path) -> list[Finding]:
    """Walk every Python file under ``root`` and aggregate findings.

    No allowlist: tier=3 is a closed invariant. An exemption would be a hole
    in the data manifesto; the rule has no legitimate suppression surface.
    """
    findings: list[Finding] = []
    for item in walk_python_files(root):
        if isinstance(item, PythonSyntaxError):
            continue
        findings.extend(analyze_tree(item.tree, display_path(item.path, root)))
    return findings


def _make_finding(*, file_path: str, call: ast.Call, message: str) -> Finding:
    fingerprint = hashlib.sha256(
        f"{RULE_INVALID}|{file_path}|{call.lineno}|{call.col_offset}".encode()
    ).hexdigest()[:16]
    return Finding(
        rule_id=RULE_INVALID,
        file_path=file_path,
        line=call.lineno,
        column=call.col_offset,
        message=message,
        fingerprint=fingerprint,
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION_INVALID,
    )


RULE = TrustBoundaryTierRule()
