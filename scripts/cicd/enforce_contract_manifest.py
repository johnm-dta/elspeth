#!/usr/bin/env python3
"""
Declaration Contract Manifest Enforcement Tool

AST-based static analysis that prevents drift between the declaration-trust
registry manifest (``EXPECTED_CONTRACTS`` in
``src/elspeth/contracts/declaration_contracts.py``) and the set of
``register_declaration_contract(...)`` call sites under ``src/``.

This is the CI backstop for ADR-010 §Decision 3 and the fix direction of
issue elspeth-b03c6112c0 (C2). The runtime check in
``Orchestrator.prepare_for_run()`` is the last line of defence; this scanner
catches the drift at PR time so the failure never reaches production.

Two failure modes detected:

- ``MC1`` (manifest_drift_extra_registration): a module calls
  ``register_declaration_contract(SomeContract())`` where
  ``SomeContract.name`` is NOT listed in the manifest. Without CI
  enforcement this is exactly the "silent new-contract-with-no-manifest-entry"
  vector reviewer B9 flagged — an attacker (or a hurried engineer) could
  add a contract module but forget to update the manifest; bootstrap would
  then raise, but only at first run, and the extra contract would still be
  invisible to every consumer that introspects the manifest.

- ``MC2`` (manifest_drift_missing_registration): the manifest lists a
  contract name that has no corresponding
  ``register_declaration_contract(...)`` call site anywhere under ``src/``.
  This is the symmetric failure mode: a contract was removed (or never
  landed) but the manifest still claims it. Bootstrap would refuse to start
  the orchestrator, which is the desired *runtime* outcome, but CI must
  block the merge earlier.

``tests/`` is intentionally NOT scanned. Tests register transient contracts
(e.g. ``_UnexpectedContract`` in ``test_orchestrator_registry_bootstrap``)
that MUST NOT appear in the manifest. Scanning tests would produce false
positives.

Usage:
    python scripts/cicd/enforce_contract_manifest.py check
    python scripts/cicd/enforce_contract_manifest.py check \\
        --source-root src/elspeth \\
        --manifest-file src/elspeth/contracts/declaration_contracts.py \\
        --allowlist config/cicd/enforce_contract_manifest
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NoReturn

import yaml

# =============================================================================
# Constants
# =============================================================================

RULE_ID_EXTRA = "MC1"
RULE_ID_MISSING = "MC2"
RULE_ID_MARKER_WITHOUT_MANIFEST = "MC3a"
RULE_ID_MANIFEST_WITHOUT_MARKER = "MC3b"
RULE_ID_TRIVIAL_BODY = "MC3c"

_REGISTER_FUNC_NAME = "register_declaration_contract"
_MANIFEST_SYMBOL = "EXPECTED_CONTRACT_SITES"
_DECORATOR_NAME = "implements_dispatch_site"
_VALID_DISPATCH_SITES: frozenset[str] = frozenset(
    {
        "pre_emission_check",
        "post_emission_check",
        "batch_flush_check",
        "boundary_check",
    }
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class Finding:
    """A manifest-drift finding — either MC1 (extra) or MC2 (missing)."""

    rule_id: str
    file_path: str  # For MC2, this is the manifest file path.
    line: int  # For MC2, this is the manifest line.
    contract_name: str
    detail: str

    @property
    def canonical_key(self) -> str:
        """Allowlist key: file:rule_id:contract_name."""
        return f"{self.file_path}:{self.rule_id}:{self.contract_name}"


@dataclass(frozen=True)
class RegistrationCall:
    """A ``register_declaration_contract(X())`` call site discovered by the scanner."""

    file_path: str
    line: int
    contract_name: str | None  # None if the call could not be statically resolved.
    detail: str  # Human-readable description for error messages / indeterminate cases.
    # Per-site markers discovered by AST inspection of the contract's class
    # body (H2 extension). None when contract_name is also None.
    marker_sites: frozenset[str] | None = None
    # Methods with trivial bodies keyed by site name — populated for MC3c.
    trivial_body_sites: frozenset[str] = frozenset()


@dataclass
class AllowlistEntry:
    """A single allowlist entry permitting a specific finding."""

    key: str
    owner: str
    reason: str
    task: str
    expires: date | None
    matched: bool = field(default=False, compare=False)
    source_file: str = field(default="", compare=False)


@dataclass
class Allowlist:
    """Parsed allowlist configuration."""

    entries: list[AllowlistEntry]
    fail_on_stale: bool = True
    fail_on_expired: bool = True

    def match(self, finding: Finding) -> AllowlistEntry | None:
        for entry in self.entries:
            if entry.key == finding.canonical_key:
                entry.matched = True
                return entry
        return None

    def get_stale_entries(self) -> list[AllowlistEntry]:
        return [e for e in self.entries if not e.matched]

    def get_expired_entries(self) -> list[AllowlistEntry]:
        today = datetime.now(UTC).date()
        return [e for e in self.entries if e.expires and e.expires < today]


# =============================================================================
# Manifest Extraction
# =============================================================================


def extract_manifest(manifest_file: Path) -> tuple[dict[str, frozenset[str]], dict[str, int], int]:
    """Parse ``EXPECTED_CONTRACT_SITES = MappingProxyType({...})`` from the
    manifest file.

    Returns ``(name_to_sites, name_to_line, assign_line)`` where:
      * ``name_to_sites`` — dict mapping contract name → frozenset of
        dispatch-site names.
      * ``name_to_line`` — dict mapping contract name → source line of its
        key-string literal (for MC2 finding anchors).
      * ``assign_line`` — the ``EXPECTED_CONTRACT_SITES =`` assignment line.

    Crashes loudly if the symbol is missing, not a MappingProxyType(dict)
    literal, or contains non-string members — the scanner's correctness
    depends on the manifest being statically analysable.
    """
    try:
        source = manifest_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Fatal: Could not read manifest {manifest_file}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        tree = ast.parse(source, filename=str(manifest_file))
    except SyntaxError as exc:
        print(f"Fatal: SyntaxError in manifest {manifest_file}: {exc}", file=sys.stderr)
        sys.exit(1)

    for node in tree.body:
        target_names: list[str] = []
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value = node.value
        elif isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value

        if _MANIFEST_SYMBOL not in target_names or value is None:
            continue

        return _parse_manifest_value(value, manifest_file, node.lineno)

    print(
        f"Fatal: {_MANIFEST_SYMBOL} not found in {manifest_file}. The manifest symbol must exist at module scope for the scanner to run.",
        file=sys.stderr,
    )
    sys.exit(1)


def _parse_manifest_value(
    value: ast.expr,
    manifest_file: Path,
    assign_line: int,
) -> tuple[dict[str, frozenset[str]], dict[str, int], int]:
    """Extract per-contract site claims from
    ``EXPECTED_CONTRACT_SITES = MappingProxyType({name: frozenset({sites}),...})``.
    """
    if not isinstance(value, ast.Call):
        _fatal_manifest_shape(manifest_file, "RHS must be a MappingProxyType({...}) call")
    proxy_call: ast.Call = value

    func = proxy_call.func
    func_name = ""
    if isinstance(func, ast.Name):
        func_name = func.id
    elif isinstance(func, ast.Attribute):
        func_name = func.attr
    if func_name != "MappingProxyType":
        _fatal_manifest_shape(
            manifest_file,
            f"RHS call must be MappingProxyType(...), got {func_name}(...)",
        )
    if len(proxy_call.args) != 1:
        _fatal_manifest_shape(
            manifest_file,
            "MappingProxyType() must have exactly one positional argument",
        )

    inner = proxy_call.args[0]
    if not isinstance(inner, ast.Dict):
        _fatal_manifest_shape(
            manifest_file,
            "MappingProxyType() argument must be a dict literal",
        )

    name_to_sites: dict[str, frozenset[str]] = {}
    name_to_line: dict[str, int] = {}
    for key_node, value_node in zip(inner.keys, inner.values, strict=True):
        if key_node is None:
            _fatal_manifest_shape(
                manifest_file,
                f"dict unpacking (``**x``) is not supported (line {value_node.lineno})",
            )
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            _fatal_manifest_shape(
                manifest_file,
                f"manifest key must be a string literal (got {ast.dump(key_node)} at line {key_node.lineno})",
            )
        contract_name: str = key_node.value
        if contract_name in name_to_sites:
            print(
                f"Fatal: duplicate manifest contract {contract_name!r} at {manifest_file}:{key_node.lineno}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Parse the value — expected to be frozenset({"site1", "site2"}).
        if not isinstance(value_node, ast.Call):
            _fatal_manifest_shape(
                manifest_file,
                f"value for {contract_name!r} must be a frozenset(...) call (line {value_node.lineno})",
            )
        value_call: ast.Call = value_node
        vc_func = value_call.func
        vc_func_name = ""
        if isinstance(vc_func, ast.Name):
            vc_func_name = vc_func.id
        elif isinstance(vc_func, ast.Attribute):
            vc_func_name = vc_func.attr
        if vc_func_name != "frozenset":
            _fatal_manifest_shape(
                manifest_file,
                f"value call for {contract_name!r} must be frozenset(...), got {vc_func_name}(...)",
            )
        if len(value_call.args) != 1:
            _fatal_manifest_shape(
                manifest_file,
                f"frozenset() for {contract_name!r} must have exactly one positional argument",
            )
        sites_node = value_call.args[0]
        if not isinstance(sites_node, (ast.Set, ast.Tuple, ast.List)):
            _fatal_manifest_shape(
                manifest_file,
                f"frozenset() argument for {contract_name!r} must be a set/tuple/list literal of string constants",
            )

        sites: set[str] = set()
        for elt in sites_node.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                _fatal_manifest_shape(
                    manifest_file,
                    f"dispatch-site member must be a string literal for {contract_name!r} (got {ast.dump(elt)} at line {elt.lineno})",
                )
            if elt.value not in _VALID_DISPATCH_SITES:
                _fatal_manifest_shape(
                    manifest_file,
                    f"unknown dispatch site {elt.value!r} for contract {contract_name!r}; valid sites: {sorted(_VALID_DISPATCH_SITES)!r}",
                )
            sites.add(elt.value)
        name_to_sites[contract_name] = frozenset(sites)
        name_to_line[contract_name] = key_node.lineno

    return name_to_sites, name_to_line, assign_line


def _fatal_manifest_shape(manifest_file: Path, reason: str) -> NoReturn:
    print(
        f"Fatal: {_MANIFEST_SYMBOL} in {manifest_file} has unsupported shape: {reason}. "
        f"The scanner requires a plain ``frozenset({{...}})`` of string literals.",
        file=sys.stderr,
    )
    sys.exit(1)


# =============================================================================
# Registration Call Site Scanning
# =============================================================================


def scan_source_tree(source_root: Path, repo_root: Path, manifest_file: Path) -> list[RegistrationCall]:
    """Walk ``source_root`` and return every ``register_declaration_contract(...)``
    call site with its statically-resolved contract name (or ``None`` if
    resolution failed).
    """
    calls: list[RegistrationCall] = []
    for py_file in sorted(source_root.rglob("*.py")):
        # Skip the manifest file itself — its definitions aren't registrations.
        if py_file.resolve() == manifest_file.resolve():
            continue
        calls.extend(_scan_file(py_file, repo_root))
    return calls


def _scan_file(file_path: Path, repo_root: Path) -> list[RegistrationCall]:
    """Scan a single file for register_declaration_contract(...) calls."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Fatal: Could not read {file_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        print(f"Fatal: SyntaxError in {file_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        relative_path = str(file_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        relative_path = file_path.name

    # Build a map of top-level class definitions in this file, keyed by name.
    # Used to resolve ``register_declaration_contract(LocalClass())`` → the
    # class's ``name = "..."`` attribute.
    classes_by_name: dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes_by_name[node.name] = node

    calls: list[RegistrationCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_register_call(node):
            continue
        calls.append(_resolve_call(node, classes_by_name, relative_path))
    return calls


def _is_register_call(node: ast.Call) -> bool:
    """Return True if ``node`` calls ``register_declaration_contract``.

    Accepts both bare and qualified forms:
    - ``register_declaration_contract(...)`` — Name
    - ``dc.register_declaration_contract(...)`` — Attribute
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == _REGISTER_FUNC_NAME
    if isinstance(func, ast.Attribute):
        return func.attr == _REGISTER_FUNC_NAME
    return False


def _resolve_call(
    call_node: ast.Call,
    classes_by_name: dict[str, ast.ClassDef],
    relative_path: str,
) -> RegistrationCall:
    """Extract the registered contract's ``name`` attribute from the call arg.

    The canonical shape is ``register_declaration_contract(SomeClass())`` —
    a zero-arg constructor call on a class defined in the same file. We
    resolve ``SomeClass`` via ``classes_by_name`` and extract its
    ``name = "..."`` class-level assignment.

    Returns a ``RegistrationCall`` with ``contract_name=None`` if the call
    cannot be resolved statically. Scanner callers treat unresolved calls as
    MC1 findings with detail explaining why resolution failed.
    """
    if len(call_node.args) != 1:
        return RegistrationCall(
            file_path=relative_path,
            line=call_node.lineno,
            contract_name=None,
            detail=(f"register_declaration_contract expects exactly 1 argument; got {len(call_node.args)}."),
        )

    arg = call_node.args[0]

    # Case 1: register_declaration_contract(SomeClass())
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
        class_name = arg.func.id
        class_node = classes_by_name.get(class_name)
        if class_node is None:
            return RegistrationCall(
                file_path=relative_path,
                line=call_node.lineno,
                contract_name=None,
                detail=(
                    f"could not resolve class {class_name!r} in same file. "
                    f"Move the contract class into the same module as the registration "
                    f"call, or update the scanner to handle cross-module resolution."
                ),
            )
        contract_name = _extract_class_name_attribute(class_node)
        if contract_name is None:
            return RegistrationCall(
                file_path=relative_path,
                line=call_node.lineno,
                contract_name=None,
                detail=(f'class {class_name!r} has no statically-resolvable ``name = "..."`` class-level string attribute.'),
            )
        marker_sites, trivial_sites = _extract_marker_sites_and_trivial_bodies(class_node)
        return RegistrationCall(
            file_path=relative_path,
            line=call_node.lineno,
            contract_name=contract_name,
            detail=f"class {class_name}",
            marker_sites=marker_sites,
            trivial_body_sites=trivial_sites,
        )

    # Case 2: register_declaration_contract(some_instance) — indirect reference.
    if isinstance(arg, ast.Name):
        return RegistrationCall(
            file_path=relative_path,
            line=call_node.lineno,
            contract_name=None,
            detail=(
                f"argument is a Name ({arg.id!r}); static resolution requires an "
                f"inline ``register_declaration_contract(SomeClass())`` call in the "
                f"module that defines SomeClass."
            ),
        )

    # Case 3: any other expression — dict, call chain, etc.
    return RegistrationCall(
        file_path=relative_path,
        line=call_node.lineno,
        contract_name=None,
        detail=("argument shape is not statically resolvable. Use the canonical ``register_declaration_contract(SomeClass())`` form."),
    )


def _extract_class_name_attribute(class_node: ast.ClassDef) -> str | None:
    """Return the ``name = "..."`` class-level string, or None if not present.

    Walks the class body looking for a top-level ``Assign`` or ``AnnAssign``
    to the identifier ``name`` with a string-constant RHS. Only direct
    class-body assignments count; nested methods are ignored.
    """
    for stmt in class_node.body:
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "name"
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            return stmt.value.value
        if (
            isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "name" for t in stmt.targets)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            return stmt.value.value
    return None


def _extract_marker_sites_and_trivial_bodies(
    class_node: ast.ClassDef,
) -> tuple[frozenset[str], frozenset[str]]:
    """Walk the class body collecting ``@implements_dispatch_site`` markers.

    Returns ``(marker_sites, trivial_body_sites)`` where:
      * ``marker_sites`` — sites the contract claims via the decorator.
      * ``trivial_body_sites`` — a subset of ``marker_sites`` whose method
        body is structurally trivial (MC3c).

    Per the D1 correction (comment #418 on H2), this scanner only inspects
    direct class body; mixin-inherited overrides are not resolved. Contracts
    using mixin inheritance MUST carry the marker on the concrete class.
    """
    marker_sites: set[str] = set()
    trivial_sites: set[str] = set()
    for stmt in class_node.body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        site_name: str | None = None
        for decorator in stmt.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name != _DECORATOR_NAME:
                continue
            if not decorator.args:
                continue
            site_arg = decorator.args[0]
            if not (isinstance(site_arg, ast.Constant) and isinstance(site_arg.value, str)):
                continue
            site_name = site_arg.value
            break
        if site_name is None:
            continue
        marker_sites.add(site_name)
        if _is_body_structurally_trivial(stmt.body):
            trivial_sites.add(site_name)
    return frozenset(marker_sites), frozenset(trivial_sites)


def _is_body_structurally_trivial(body: list[ast.stmt]) -> bool:
    """Return True iff the method body consists only of trivial statements.

    MC3c trivial body definition (plan-review W5 + Security S2-003): body
    consisting ONLY of any combination of:
      * ``pass`` statements.
      * ``return None`` / bare ``return``.
      * ``...`` (Ellipsis literal — ``Expr(value=Constant(value=Ellipsis))``).
      * Docstring — a LEADING ``Expr(value=Constant(value=<str>))`` (first
        statement only); subsequent bare literal expressions are trivial.
      * Bare literal expression statements (Constant values other than a
        leading docstring).

    A body is NON-trivial if it contains at least ONE:
      * ``Raise``, ``Return`` with non-None-Constant value.
      * ``Call`` (any function or method call).
      * ``Attribute`` read on a non-self name.
      * ``Assign`` / ``AugAssign`` / ``AnnAssign``.
      * Control-flow (``If``, ``For``, ``While``, ``With``, ``Try``, ...).

    Implemented by classifying each statement. The first statement may be a
    docstring (permitted once); subsequent string-constants count as trivial
    literal expressions, not docstrings.
    """
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Return):
            # ``return`` (no value) and ``return None`` are trivial.
            if stmt.value is None:
                continue
            if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                continue
            return False
        if isinstance(stmt, ast.Expr):
            # Bare expression statement. Trivial iff it is a Constant literal.
            if isinstance(stmt.value, ast.Constant):
                continue
            return False
        return False
    return True


# =============================================================================
# Finding Computation
# =============================================================================


def compute_findings(
    manifest_sites: dict[str, frozenset[str]],
    manifest_name_to_line: dict[str, int],
    registrations: list[RegistrationCall],
    manifest_file_rel: str,
    manifest_assign_line: int,
) -> list[Finding]:
    """Produce MC1 / MC2 / MC3a / MC3b / MC3c findings from the scan results.

    - MC1 (extra): registration with no manifest entry.
    - MC2 (missing): manifest entry with no registration.
    - MC3a (marker without manifest): contract's @implements_dispatch_site
      marker names a site NOT listed in the manifest.
    - MC3b (manifest without marker): manifest names a site NOT marked on
      the contract class.
    - MC3c (trivial body): marked site's method body is structurally
      trivial (pass / ... / bare literal / bare return).
    """
    findings: list[Finding] = []

    registered_names_found: set[str] = set()
    for call in registrations:
        if call.contract_name is None:
            findings.append(
                Finding(
                    rule_id=RULE_ID_EXTRA,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"<unresolved:{call.file_path}:{call.line}>",
                    detail=(f"register_declaration_contract call is not statically resolvable: {call.detail}"),
                )
            )
            continue

        registered_names_found.add(call.contract_name)
        if call.contract_name not in manifest_sites:
            findings.append(
                Finding(
                    rule_id=RULE_ID_EXTRA,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=call.contract_name,
                    detail=(f"contract {call.contract_name!r} is registered but not listed in EXPECTED_CONTRACT_SITES manifest."),
                )
            )
            continue

        # Per-site comparison (MC3a/b/c).
        expected_sites = manifest_sites[call.contract_name]
        marker_sites = call.marker_sites or frozenset()

        # MC3a: marker without manifest entry.
        for extra_site in sorted(marker_sites - expected_sites):
            findings.append(
                Finding(
                    rule_id=RULE_ID_MARKER_WITHOUT_MANIFEST,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"{call.contract_name}::{extra_site}",
                    detail=(
                        f"contract {call.contract_name!r} claims dispatch site "
                        f"{extra_site!r} via @implements_dispatch_site, but the "
                        f"site is NOT listed in EXPECTED_CONTRACT_SITES[{call.contract_name!r}]. "
                        f"Either add the site to the manifest or remove the marker."
                    ),
                )
            )

        # MC3b: manifest entry without marker.
        for missing_site in sorted(expected_sites - marker_sites):
            findings.append(
                Finding(
                    rule_id=RULE_ID_MANIFEST_WITHOUT_MARKER,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"{call.contract_name}::{missing_site}",
                    detail=(
                        f"contract {call.contract_name!r} manifest names "
                        f"dispatch site {missing_site!r}, but the contract "
                        f"class has no @implements_dispatch_site({missing_site!r}) "
                        f"marker on any method. Under multi-level inheritance the "
                        f"marker MUST be on the concrete class "
                        f"(per D1 correction, comment #418 on H2)."
                    ),
                )
            )

        # MC3c: trivial body on a marked site.
        for trivial_site in sorted(call.trivial_body_sites & expected_sites):
            findings.append(
                Finding(
                    rule_id=RULE_ID_TRIVIAL_BODY,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"{call.contract_name}::{trivial_site}",
                    detail=(
                        f"contract {call.contract_name!r} method implementing "
                        f"{trivial_site!r} has a structurally trivial body "
                        f"(pass / ... / bare return / literal-only). An opt-in "
                        f"without an implementation is an empty-body bypass "
                        f"surface (Security S2-003) — the contract appears to "
                        f"fire, audits as fired, and semantically did nothing."
                    ),
                )
            )

    # MC2: manifest entry with no registration call.
    for name, line in manifest_name_to_line.items():
        if name not in registered_names_found:
            findings.append(
                Finding(
                    rule_id=RULE_ID_MISSING,
                    file_path=manifest_file_rel,
                    line=line if line else manifest_assign_line,
                    contract_name=name,
                    detail=(f"manifest lists contract {name!r} but no register_declaration_contract(...) call site resolves to it."),
                )
            )

    return findings


# =============================================================================
# Allowlist Loading
# =============================================================================


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(f"Error: Invalid YAML in allowlist {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _parse_entries(data: dict[str, Any], source_file: str = "") -> list[AllowlistEntry]:
    entries: list[AllowlistEntry] = []
    source_ctx = f" in {source_file}" if source_file else ""

    for item in data.get("allow_contracts", []):
        key: str = item.get("key", "")
        if not key:
            print(
                f"Error: allow_contracts entry is missing 'key' field{source_ctx}",
                file=sys.stderr,
            )
            sys.exit(1)

        expires_str: str | None = item.get("expires")
        expires_date: date | None = None
        if expires_str:
            try:
                expires_date = datetime.strptime(expires_str, "%Y-%m-%d").replace(tzinfo=UTC).date()
            except ValueError:
                print(
                    f"Warning: Invalid date format for expires: {expires_str}{source_ctx}",
                    file=sys.stderr,
                )

        entries.append(
            AllowlistEntry(
                key=key,
                owner=item.get("owner", "unknown"),
                reason=item.get("reason", ""),
                task=item.get("task", ""),
                expires=expires_date,
                source_file=source_file,
            )
        )

    return entries


def load_allowlist_from_directory(directory: Path) -> Allowlist:
    defaults_path = directory / "_defaults.yaml"
    defaults: dict[str, Any] = {}
    if defaults_path.exists():
        defaults_data = _load_yaml_file(defaults_path)
        defaults = defaults_data.get("defaults", {})

    yaml_files = sorted(f for f in directory.glob("*.yaml") if f.name != "_defaults.yaml")

    all_entries: list[AllowlistEntry] = []
    for yaml_file in yaml_files:
        data = _load_yaml_file(yaml_file)
        all_entries.extend(_parse_entries(data, source_file=yaml_file.name))

    return Allowlist(
        entries=all_entries,
        fail_on_stale=defaults.get("fail_on_stale", True),
        fail_on_expired=defaults.get("fail_on_expired", True),
    )


def load_allowlist(path: Path) -> Allowlist:
    if path.is_dir():
        return load_allowlist_from_directory(path)

    if not path.exists():
        return Allowlist(entries=[])

    data = _load_yaml_file(path)
    defaults = data.get("defaults", {})
    return Allowlist(
        entries=_parse_entries(data),
        fail_on_stale=defaults.get("fail_on_stale", True),
        fail_on_expired=defaults.get("fail_on_expired", True),
    )


# =============================================================================
# Reporting
# =============================================================================


def format_finding(finding: Finding) -> str:
    return f"{finding.file_path}:{finding.line}: [{finding.rule_id}] {finding.contract_name} — {finding.detail}"


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Declaration Contract Manifest Enforcement — keeps EXPECTED_CONTRACTS "
            "in sync with register_declaration_contract(...) call sites "
            "(ADR-010 §Decision 3, issue elspeth-b03c6112c0 / C2)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check for MC1/MC2 violations")
    check_parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Directory to scan for register_declaration_contract calls (default: src/elspeth).",
    )
    check_parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help=("Python file containing the EXPECTED_CONTRACTS frozenset (default: src/elspeth/contracts/declaration_contracts.py)."),
    )
    check_parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Path to allowlist YAML file or directory of YAML files",
    )

    args = parser.parse_args()

    if args.command == "check":
        return run_check(args)

    return 0


def run_check(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).parent.parent.parent

    source_root: Path = (args.source_root or (repo_root / "src" / "elspeth")).resolve()
    manifest_file: Path = (args.manifest_file or (repo_root / "src" / "elspeth" / "contracts" / "declaration_contracts.py")).resolve()

    if not source_root.is_dir():
        print(f"Error: source root {source_root} is not a directory", file=sys.stderr)
        return 1
    if not manifest_file.is_file():
        print(f"Error: manifest file {manifest_file} is not a file", file=sys.stderr)
        return 1

    allowlist_path: Path | None = args.allowlist
    if allowlist_path is None:
        dir_path = repo_root / "config" / "cicd" / "enforce_contract_manifest"
        file_path = repo_root / "config" / "cicd" / "enforce_contract_manifest.yaml"
        allowlist_path = dir_path if dir_path.is_dir() else file_path

    allowlist = load_allowlist(allowlist_path)

    manifest_sites, manifest_name_to_line, assign_line = extract_manifest(manifest_file)
    try:
        manifest_file_rel = str(manifest_file.relative_to(repo_root))
    except ValueError:
        manifest_file_rel = manifest_file.name

    registrations = scan_source_tree(source_root, repo_root, manifest_file)
    all_findings = compute_findings(
        manifest_sites,
        manifest_name_to_line,
        registrations,
        manifest_file_rel,
        assign_line,
    )

    violations: list[Finding] = []
    for finding in all_findings:
        if allowlist.match(finding) is None:
            violations.append(finding)

    stale_entries = allowlist.get_stale_entries() if allowlist.fail_on_stale else []
    expired_entries = allowlist.get_expired_entries() if allowlist.fail_on_expired else []

    has_errors = bool(violations or stale_entries or expired_entries)

    if violations:
        print(f"\n{'=' * 60}")
        print(f"CONTRACT MANIFEST DRIFT: {len(violations)} finding(s)")
        print(
            "MC1 = registration without manifest entry; MC2 = manifest entry without registration;\n"
            "MC3a = @implements_dispatch_site marker claims a site not in manifest;\n"
            "MC3b = manifest names a site with no @implements_dispatch_site marker;\n"
            "MC3c = marked dispatch site's method body is structurally trivial."
        )
        print("=" * 60)
        for v in violations:
            print(format_finding(v))
        print()
        print(f"Manifest: {manifest_file_rel} ({_MANIFEST_SYMBOL})")
        print("Fix direction:")
        print("  MC1 — either add the contract name to EXPECTED_CONTRACT_SITES, or remove")
        print("        the register_declaration_contract(...) call site.")
        print("  MC2 — either restore the register_declaration_contract(...) call site, or")
        print("        remove the name from EXPECTED_CONTRACT_SITES.")
        print("  MC3a — either add the site to EXPECTED_CONTRACT_SITES or remove the marker.")
        print("  MC3b — add @implements_dispatch_site(<site>) on the concrete class")
        print("         (under multi-level inheritance the marker MUST be on the")
        print("         concrete class; mixin-inherited overrides are not detected).")
        print("  MC3c — replace the trivial body (pass / ... / bare return / literal) with a")
        print("         non-trivial implementation, or drop the marker entirely.")
        print()
        print("To allowlist a transitional finding, add an entry to the allowlist directory:")
        print(f"  {allowlist_path}")
        print()
        print("Example allowlist entry (YAML):")
        print("  allow_contracts:")
        print(f"  - key: {violations[0].canonical_key}")
        print("    owner: <your-name>")
        print("    reason: <explain the transitional state>")
        print("    task: <filigree issue that will resolve this>")
        print("    expires: <YYYY-MM-DD>")

    if stale_entries:
        print(f"\n{'=' * 60}")
        print(f"STALE ALLOWLIST ENTRIES: {len(stale_entries)}")
        print("(These entries do not match any finding — remove them)")
        print("=" * 60)
        for e in stale_entries:
            source = f" (from {e.source_file})" if e.source_file else ""
            print(f"  Key: {e.key}{source}")
            print(f"  Owner: {e.owner}")
            print(f"  Reason: {e.reason}")

    if expired_entries:
        print(f"\n{'=' * 60}")
        print(f"EXPIRED ALLOWLIST ENTRIES: {len(expired_entries)}")
        print("=" * 60)
        for e in expired_entries:
            print(f"  Key: {e.key}")
            print(f"  Owner: {e.owner}")
            print(f"  Expired: {e.expires}")

    if has_errors:
        print(f"\n{'=' * 60}")
        print("CHECK FAILED")
        print("=" * 60)
    else:
        total_site_claims = sum(len(sites) for sites in manifest_sites.values())
        print(
            f"\nNo contract-manifest drift. Manifest has {len(manifest_sites)} contract(s) "
            f"with {total_site_claims} total dispatch-site claim(s); all registered and "
            f"all markers match. Check passed."
        )

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
