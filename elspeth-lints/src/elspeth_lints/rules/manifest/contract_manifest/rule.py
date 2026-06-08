"""Declaration-contract manifest rule implementation."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.manifest.contract_manifest.metadata import (
    RULE_ID,
    RULE_MC1,
    RULE_MC2,
    RULE_MC3A,
    RULE_MC3B,
    RULE_MC3C,
    RULE_METADATA,
    SUGGESTION,
)

_REGISTER_FUNC_NAME = "register_declaration_contract"
_MANIFEST_SYMBOL = "EXPECTED_CONTRACT_SITES"
_DECORATOR_NAME = "implements_dispatch_site"
_CANONICAL_CONTRACTS_MODULE = "elspeth.contracts.declaration_contracts"
_MARKER_SITE_KEYWORD = "site_name"
_VALID_DISPATCH_SITES: frozenset[str] = frozenset(
    {
        "pre_emission_check",
        "post_emission_check",
        "batch_flush_check",
        "boundary_check",
    }
)


@dataclass(frozen=True, slots=True)
class CanonicalBindings:
    """Per-file local names that PROVABLY resolve to the canonical contract API.

    The scanner must not trust the textual final name of a call or decorator:
    a module can locally shadow ``register_declaration_contract`` or
    ``implements_dispatch_site`` with a no-op of the same name and slip a fake
    registration past CI (runtime would never register it). A symbol counts only
    when it is imported from ``elspeth.contracts.declaration_contracts``.
    """

    register_names: frozenset[str]
    marker_names: frozenset[str]
    module_aliases: frozenset[str]


def _canonical_bindings(tree: ast.AST) -> CanonicalBindings:
    """Collect the local names bound to the canonical contract API in one file."""
    register_names: set[str] = set()
    marker_names: set[str] = set()
    module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module != _CANONICAL_CONTRACTS_MODULE:
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == _REGISTER_FUNC_NAME:
                    register_names.add(local)
                elif alias.name == _DECORATOR_NAME:
                    marker_names.add(local)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _CANONICAL_CONTRACTS_MODULE and alias.asname:
                    module_aliases.add(alias.asname)
    return CanonicalBindings(
        register_names=frozenset(register_names),
        marker_names=frozenset(marker_names),
        module_aliases=frozenset(module_aliases),
    )


@dataclass(frozen=True, slots=True)
class ContractFinding:
    """A legacy contract-manifest finding."""

    rule_id: str
    file_path: str
    line: int
    contract_name: str
    detail: str

    @property
    def canonical_key(self) -> str:
        """Return the legacy allowlist and parity fingerprint key."""
        return f"{self.file_path}:{self.rule_id}:{self.contract_name}"


@dataclass(frozen=True, slots=True)
class RegistrationCall:
    """A register_declaration_contract(...) call discovered by AST scanning."""

    file_path: str
    line: int
    contract_name: str | None
    detail: str
    marker_sites: frozenset[str] | None = None
    trivial_body_sites: frozenset[str] = frozenset()


@dataclass(slots=True)
class ContractAllowlistEntry:
    """A single legacy allow_contracts entry."""

    key: str
    owner: str
    reason: str
    task: str
    expires: date | None
    matched: bool = field(default=False, compare=False)


@dataclass(slots=True)
class ContractAllowlist:
    """Legacy allow_contracts allowlist."""

    entries: list[ContractAllowlistEntry]

    def match(self, finding: ContractFinding) -> ContractAllowlistEntry | None:
        """Return the matching entry for a finding, if any."""
        for entry in self.entries:
            if entry.key == finding.canonical_key:
                entry.matched = True
                return entry
        return None


@dataclass(frozen=True, slots=True)
class ContractManifestRule:
    """Detect declaration contract manifest drift."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run the repository-scoped contract-manifest scan."""
        del tree, file_path
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Scan a repository or src/elspeth root and apply the legacy allowlist."""
    source_root, manifest_file = resolve_scan_roots(root)
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(source_root, "enforce_contract_manifest")
    )
    allowlist = load_contract_allowlist(allowlist_dir)
    manifest_sites, manifest_name_to_line, assign_line = extract_manifest(manifest_file)
    manifest_file_rel = repo_relative_display_path(manifest_file, source_root)
    registrations = scan_source_tree(source_root, manifest_file)
    contract_findings = compute_findings(
        manifest_sites,
        manifest_name_to_line,
        registrations,
        manifest_file_rel,
        assign_line,
    )
    active = [finding for finding in contract_findings if allowlist.match(finding) is None]
    return [to_lint_finding(finding) for finding in active]


def resolve_scan_roots(root: Path) -> tuple[Path, Path]:
    """Return ``(source_root, manifest_file)`` for repo or src/elspeth roots."""
    root = root.resolve()
    direct_manifest = root / "contracts" / "declaration_contracts.py"
    if direct_manifest.is_file():
        return root, direct_manifest

    repo_manifest = root / "src" / "elspeth" / "contracts" / "declaration_contracts.py"
    if repo_manifest.is_file():
        return root / "src" / "elspeth", repo_manifest

    return root, direct_manifest


def extract_manifest(manifest_file: Path) -> tuple[dict[str, frozenset[str]], dict[str, int], int]:
    """Parse EXPECTED_CONTRACT_SITES from a declaration-contract manifest."""
    tree = ast.parse(manifest_file.read_text(encoding="utf-8"), filename=str(manifest_file))
    for node in tree.body:
        target_names: list[str] = []
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value = node.value
        elif isinstance(node, ast.Assign):
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value = node.value
        if _MANIFEST_SYMBOL in target_names and value is not None:
            return _parse_manifest_value(value, manifest_file, node.lineno)
    raise ValueError(f"{_MANIFEST_SYMBOL} not found in {manifest_file}")


def _parse_manifest_value(
    value: ast.expr,
    manifest_file: Path,
    assign_line: int,
) -> tuple[dict[str, frozenset[str]], dict[str, int], int]:
    if not isinstance(value, ast.Call):
        raise ValueError(f"{_MANIFEST_SYMBOL} in {manifest_file} must be MappingProxyType(...)")
    func_name = _call_name(value)
    if func_name != "MappingProxyType" or len(value.args) != 1:
        raise ValueError(f"{_MANIFEST_SYMBOL} in {manifest_file} must be MappingProxyType({{...}})")
    inner = value.args[0]
    if not isinstance(inner, ast.Dict):
        raise ValueError(f"{_MANIFEST_SYMBOL} in {manifest_file} must wrap a dict literal")

    name_to_sites: dict[str, frozenset[str]] = {}
    name_to_line: dict[str, int] = {}
    for key_node, value_node in zip(inner.keys, inner.values, strict=True):
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            raise ValueError(f"{manifest_file}:{getattr(key_node, 'lineno', assign_line)} manifest key must be a string literal")
        contract_name = key_node.value
        if contract_name in name_to_sites:
            raise ValueError(f"{manifest_file}:{key_node.lineno} duplicate manifest contract {contract_name!r}")
        sites = _parse_sites(value_node, manifest_file, contract_name)
        name_to_sites[contract_name] = frozenset(sites)
        name_to_line[contract_name] = key_node.lineno
    return name_to_sites, name_to_line, assign_line


def _parse_sites(value_node: ast.expr, manifest_file: Path, contract_name: str) -> set[str]:
    if not isinstance(value_node, ast.Call) or _call_name(value_node) != "frozenset" or len(value_node.args) != 1:
        raise ValueError(f"{manifest_file}:{value_node.lineno} {contract_name!r} sites must be frozenset(...)")
    sites_node = value_node.args[0]
    if not isinstance(sites_node, (ast.Set, ast.Tuple, ast.List)):
        raise ValueError(f"{manifest_file}:{sites_node.lineno} {contract_name!r} sites must be a literal collection")
    sites: set[str] = set()
    for elt in sites_node.elts:
        if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
            raise ValueError(f"{manifest_file}:{elt.lineno} dispatch-site member must be a string literal")
        if elt.value not in _VALID_DISPATCH_SITES:
            raise ValueError(f"{manifest_file}:{elt.lineno} unknown dispatch site {elt.value!r}")
        sites.add(elt.value)
    return sites


def scan_source_tree(source_root: Path, manifest_file: Path) -> list[RegistrationCall]:
    """Return register_declaration_contract(...) calls under source_root."""
    calls: list[RegistrationCall] = []
    for py_file in sorted(source_root.rglob("*.py")):
        if py_file.resolve() == manifest_file.resolve():
            continue
        calls.extend(_scan_file(py_file, source_root))
    return calls


def _scan_file(file_path: Path, source_root: Path) -> list[RegistrationCall]:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    relative_path = repo_relative_display_path(file_path, source_root)
    bindings = _canonical_bindings(tree)
    classes_by_name = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    calls: list[RegistrationCall] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_register_call(node, bindings):
            calls.append(_resolve_call(node, classes_by_name, relative_path, bindings))
    return calls


def _is_register_call(node: ast.Call, bindings: CanonicalBindings) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in bindings.register_names
    if isinstance(func, ast.Attribute) and func.attr == _REGISTER_FUNC_NAME:
        return isinstance(func.value, ast.Name) and func.value.id in bindings.module_aliases
    return False


def _resolve_call(
    call_node: ast.Call,
    classes_by_name: dict[str, ast.ClassDef],
    relative_path: str,
    bindings: CanonicalBindings,
) -> RegistrationCall:
    if len(call_node.args) != 1:
        return RegistrationCall(
            file_path=relative_path,
            line=call_node.lineno,
            contract_name=None,
            detail=f"register_declaration_contract expects exactly 1 argument; got {len(call_node.args)}.",
        )

    arg = call_node.args[0]
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
        class_name = arg.func.id
        class_node = classes_by_name.get(class_name)
        if class_node is None:
            return RegistrationCall(
                file_path=relative_path,
                line=call_node.lineno,
                contract_name=None,
                detail=(
                    f"could not resolve class {class_name!r} in same file. Move the contract class into the same module as the registration "
                    "call, or update the scanner to handle cross-module resolution."
                ),
            )
        contract_name = _extract_class_name_attribute(class_node)
        if contract_name is None:
            return RegistrationCall(
                file_path=relative_path,
                line=call_node.lineno,
                contract_name=None,
                detail=f'class {class_name!r} has no statically-resolvable ``name = "..."`` class-level string attribute.',
            )
        marker_sites, trivial_sites = _extract_marker_sites_and_trivial_bodies(class_node, bindings)
        return RegistrationCall(
            file_path=relative_path,
            line=call_node.lineno,
            contract_name=contract_name,
            detail=f"class {class_name}",
            marker_sites=marker_sites,
            trivial_body_sites=trivial_sites,
        )

    if isinstance(arg, ast.Name):
        return RegistrationCall(
            file_path=relative_path,
            line=call_node.lineno,
            contract_name=None,
            detail=(
                f"argument is a Name ({arg.id!r}); static resolution requires an inline "
                "``register_declaration_contract(SomeClass())`` call in the module that defines SomeClass."
            ),
        )

    return RegistrationCall(
        file_path=relative_path,
        line=call_node.lineno,
        contract_name=None,
        detail="argument shape is not statically resolvable. Use the canonical ``register_declaration_contract(SomeClass())`` form.",
    )


def _extract_class_name_attribute(class_node: ast.ClassDef) -> str | None:
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
            and any(isinstance(target, ast.Name) and target.id == "name" for target in stmt.targets)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            return stmt.value.value
    return None


def _extract_marker_sites_and_trivial_bodies(
    class_node: ast.ClassDef, bindings: CanonicalBindings
) -> tuple[frozenset[str], frozenset[str]]:
    marker_sites: set[str] = set()
    trivial_sites: set[str] = set()
    for stmt in class_node.body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        site_name = _dispatch_site_marker(stmt, bindings)
        if site_name is None:
            continue
        marker_sites.add(site_name)
        if _is_body_structurally_trivial(stmt.body):
            trivial_sites.add(site_name)
    return frozenset(marker_sites), frozenset(trivial_sites)


def _dispatch_site_marker(function_node: ast.FunctionDef, bindings: CanonicalBindings) -> str | None:
    for decorator in function_node.decorator_list:
        if not isinstance(decorator, ast.Call) or not _is_canonical_marker(decorator, bindings):
            continue
        site = _marker_site_value(decorator)
        if site is not None:
            return site
    return None


def _is_canonical_marker(decorator: ast.Call, bindings: CanonicalBindings) -> bool:
    func = decorator.func
    if isinstance(func, ast.Name):
        return func.id in bindings.marker_names
    if isinstance(func, ast.Attribute) and func.attr == _DECORATOR_NAME:
        return isinstance(func.value, ast.Name) and func.value.id in bindings.module_aliases
    return False


def _marker_site_value(decorator: ast.Call) -> str | None:
    """Read the dispatch-site name from positional OR ``site_name=`` keyword form.

    The runtime decorator accepts ``site_name`` as a normal keyword parameter, so
    ``@implements_dispatch_site(site_name="post_emission_check")`` is a valid
    marker the scanner must recognise (else it raises a spurious MC3b).
    """
    if decorator.args:
        first = decorator.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    for keyword in decorator.keywords:
        if keyword.arg == _MARKER_SITE_KEYWORD and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return None


def _is_body_structurally_trivial(body: list[ast.stmt]) -> bool:
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Return):
            if stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                continue
            return False
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        return False
    return True


def compute_findings(
    manifest_sites: dict[str, frozenset[str]],
    manifest_name_to_line: dict[str, int],
    registrations: list[RegistrationCall],
    manifest_file_rel: str,
    manifest_assign_line: int,
) -> list[ContractFinding]:
    """Produce legacy MC1/MC2/MC3a/MC3b/MC3c findings."""
    findings: list[ContractFinding] = []
    registered_names_found: set[str] = set()

    for call in registrations:
        if call.contract_name is None:
            findings.append(
                ContractFinding(
                    rule_id=RULE_MC1,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"<unresolved:{call.file_path}:{call.line}>",
                    detail=f"register_declaration_contract call is not statically resolvable: {call.detail}",
                )
            )
            continue

        if call.contract_name in registered_names_found:
            # Runtime register_declaration_contract raises ValueError on a
            # duplicate name; CI must fail closed on a tree bootstrap would
            # reject rather than silently dedup it into a set.
            findings.append(
                ContractFinding(
                    rule_id=RULE_MC1,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"{call.contract_name}::duplicate@L{call.line}",
                    detail=(
                        f"contract name {call.contract_name!r} is registered more than once; runtime "
                        "register_declaration_contract rejects duplicate names with ValueError. Remove the "
                        "redundant register_declaration_contract(...) call."
                    ),
                )
            )
            continue

        registered_names_found.add(call.contract_name)
        if call.contract_name not in manifest_sites:
            findings.append(
                ContractFinding(
                    rule_id=RULE_MC1,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=call.contract_name,
                    detail=f"contract {call.contract_name!r} is registered but not listed in EXPECTED_CONTRACT_SITES manifest.",
                )
            )
            continue

        expected_sites = manifest_sites[call.contract_name]
        marker_sites = call.marker_sites or frozenset()
        for extra_site in sorted(marker_sites - expected_sites):
            findings.append(
                ContractFinding(
                    rule_id=RULE_MC3A,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"{call.contract_name}::{extra_site}",
                    detail=(
                        f"contract {call.contract_name!r} claims dispatch site {extra_site!r} via @implements_dispatch_site, but the "
                        f"site is NOT listed in EXPECTED_CONTRACT_SITES[{call.contract_name!r}]. Either add the site to the manifest or "
                        "remove the marker."
                    ),
                )
            )
        for missing_site in sorted(expected_sites - marker_sites):
            findings.append(
                ContractFinding(
                    rule_id=RULE_MC3B,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"{call.contract_name}::{missing_site}",
                    detail=(
                        f"contract {call.contract_name!r} manifest names dispatch site {missing_site!r}, but the contract class has no "
                        f"@implements_dispatch_site({missing_site!r}) marker on any method. Under multi-level inheritance the marker "
                        "MUST be on the concrete class (per D1 correction, comment #418 on H2)."
                    ),
                )
            )
        for trivial_site in sorted(call.trivial_body_sites & expected_sites):
            findings.append(
                ContractFinding(
                    rule_id=RULE_MC3C,
                    file_path=call.file_path,
                    line=call.line,
                    contract_name=f"{call.contract_name}::{trivial_site}",
                    detail=(
                        f"contract {call.contract_name!r} method implementing {trivial_site!r} has a structurally trivial body "
                        "(pass / ... / bare return / literal-only). An opt-in without an implementation is an empty-body bypass "
                        "surface (Security S2-003) — the contract appears to fire, audits as fired, and semantically did nothing."
                    ),
                )
            )

    for name, line in manifest_name_to_line.items():
        if name not in registered_names_found:
            findings.append(
                ContractFinding(
                    rule_id=RULE_MC2,
                    file_path=manifest_file_rel,
                    line=line if line else manifest_assign_line,
                    contract_name=name,
                    detail=f"manifest lists contract {name!r} but no register_declaration_contract(...) call site resolves to it.",
                )
            )
    return findings


def to_lint_finding(finding: ContractFinding) -> Finding:
    """Convert a legacy finding into the elspeth-lints protocol."""
    return Finding(
        rule_id=finding.rule_id,
        file_path=finding.file_path,
        line=finding.line,
        column=0,
        message=finding.detail,
        fingerprint=finding.canonical_key,
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION,
    )


def load_contract_allowlist(path: Path) -> ContractAllowlist:
    """Load legacy allow_contracts YAML from a file or directory."""
    if path.is_dir():
        entries: list[ContractAllowlistEntry] = []
        for yaml_file in sorted(file for file in path.glob("*.yaml") if file.name != "_defaults.yaml"):
            entries.extend(_parse_allowlist_entries(_load_yaml(yaml_file)))
        return ContractAllowlist(entries=entries)
    if not path.exists():
        return ContractAllowlist(entries=[])
    return ContractAllowlist(entries=_parse_allowlist_entries(_load_yaml(path)))


def _parse_allowlist_entries(data: dict[str, Any]) -> list[ContractAllowlistEntry]:
    entries: list[ContractAllowlistEntry] = []
    for item in data.get("allow_contracts", []):
        key = item.get("key", "")
        if not key:
            continue
        entries.append(
            ContractAllowlistEntry(
                key=str(key),
                owner=str(item.get("owner", "unknown")),
                reason=str(item.get("reason", "")),
                task=str(item.get("task", "")),
                expires=_parse_date(item.get("expires")),
            )
        )
    return entries


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _parse_date(raw: object) -> date | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, date):
        return raw
    # Fail closed: a malformed ``expires`` must raise, not silently become
    # ``None``. Swallowing it leaves ``fail_on_expired`` unable to enforce a
    # typoed expiry, so a one-character diff disables the time bound.
    if not isinstance(raw, str):
        raise ValueError(f"allow_contracts entry expires must be YYYY-MM-DD, null, or absent; got {raw!r}")
    try:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC).date()
    except ValueError as exc:
        raise ValueError(f"allow_contracts entry expires must be YYYY-MM-DD; got {raw!r}") from exc


def repo_relative_display_path(file_path: Path, source_root: Path) -> str:
    """Return the legacy repo-relative path for src/elspeth scan roots."""
    display_root = source_root.parent.parent if source_root.name == "elspeth" and source_root.parent.name == "src" else source_root
    try:
        return file_path.resolve().relative_to(display_root.resolve()).as_posix()
    except ValueError:
        return file_path.name


def allowlist_path_for_root(root: Path, directory_name: str) -> Path:
    """Find a CI allowlist directory from a scan root or repository cwd."""
    relative = Path("config") / "cicd" / directory_name
    candidates = [root, Path.cwd(), *root.parents]
    for candidate in candidates:
        path = candidate / relative
        if path.exists():
            return path
    return root / relative


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


RULE = ContractManifestRule()
