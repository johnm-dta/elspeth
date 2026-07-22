"""Composer capability-parity live-acceptance oracle (Plan 05 Task 5).

This module is the *oracle*: given sanitized evidence exported from a single
deployed revision, it decides whether the two-LLM colour hybrid pipeline
actually ran end to end against a **real** provider and produced a graph and
business output that satisfy the design contract
(``docs/superpowers/specs/2026-07-13-two-llm-colour-hybrid-pipeline-design.md``,
sections 6.x / 8.5).

It verifies, for ONE deployed revision:

* two DISTINCT LLM nodes, each covering all ten source rows;
* a real require-all UNION coalesce over both named branches;
* error routes that reach the failure output;
* the exact eight-field cleanup
  (``color_name``, ``hex``, ``blue_amount``, ``blue_confidence``,
  ``blue_reason``, ``red_amount``, ``red_confidence``, ``red_reason``);
* ten successes and zero failures;
* integer / range / non-empty-reason contracts on every business row;
* unique input identities;
* the ABSENCE of raw responses, token usage, model metadata, and branch
  bookkeeping from the business output.

**The live-provider proof is intrinsic, never self-declared.** The oracle
cannot call the provider, so its only leverage is properties a real provider
response yields and a scripted/mock/replay/cache double cannot forge without
literally fabricating provider request ids and usage: a non-sentinel
``model_returned``, a non-empty ``provider_request_id``, and positive
``prompt_tokens`` / ``completion_tokens``. A fake run that *lies* in its
manifest (``provider.mode == "live"``) is still rejected on those intrinsic
per-call properties. This is the acceptance bar for the whole oracle: get the
discriminator wrong and Task 6's live proof validates nothing.

Evidence-directory hygiene is enforced on load: the resolved revision directory
and every file under it must have restrictive permissions (no group/other
write), contain no symlinks, and contain no path that resolves outside the
revision directory. Any credential, cookie, authorization header, resolved
secret, or raw provider response found anywhere in the evidence is rejected —
the exporter must have stripped them (``redact_evidence`` does so before write),
and their presence means retention discipline failed.

The module is import-safe offline: no live-run / litellm / httpx dependency is
imported at module scope. The ``run`` CLI path (which drives a real surface
against ``$ELSPETH_EVAL_BASE_URL`` with environment-only credentials) lazily
imports its dependencies and is never exercised by this workflow's tests — it
is the entry point Task 6 invokes against staging.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #

#: Evidence-schema versions this oracle understands.
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})

#: The ten-row colour corpus size.
EXPECTED_ROW_COUNT = 10

#: Two branches over ten rows.
EXPECTED_ASSESSMENTS = 2 * EXPECTED_ROW_COUNT

#: One completed coalesce per source row.
EXPECTED_COALESCES = EXPECTED_ROW_COUNT

#: The exact hybrid output contract after cleanup (design §"Hybrid Output
#: Contract"). Order is the design's declared field order.
APPROVED_OUTPUT_FIELDS: tuple[str, ...] = (
    "color_name",
    "hex",
    "blue_amount",
    "blue_confidence",
    "blue_reason",
    "red_amount",
    "red_confidence",
    "red_reason",
)
APPROVED_OUTPUT_FIELD_SET: frozenset[str] = frozenset(APPROVED_OUTPUT_FIELDS)

#: Fields that identify a source row uniquely (design "Input").
IDENTITY_FIELDS: tuple[str, ...] = ("color_name", "hex")

#: Branch-exclusive typed contracts.
AMOUNT_FIELDS: tuple[str, ...] = ("blue_amount", "red_amount")
CONFIDENCE_FIELDS: tuple[str, ...] = ("blue_confidence", "red_confidence")
REASON_FIELDS: tuple[str, ...] = ("blue_reason", "red_reason")

BLUE_BRANCH_FIELDS: frozenset[str] = frozenset({"blue_amount", "blue_confidence", "blue_reason"})
RED_BRANCH_FIELDS: frozenset[str] = frozenset({"red_amount", "red_confidence", "red_reason"})

ALLOWED_SURFACES: frozenset[str] = frozenset({"freeform", "guided_full", "guided_staged"})

#: Reserved LLM-transform output suffixes (see
#: ``src/elspeth/plugins/transforms/llm/__init__.py``): raw response text lives
#: under the bare query name; ``_usage`` / ``_model`` carry token usage and the
#: responding model; ``_response`` / ``_error`` and the audit-hash suffixes are
#: intermediate bookkeeping. None may survive the cleanup into a business row.
_LEAKAGE_SUFFIXES: tuple[str, ...] = (
    "_usage",
    "_model",
    "_response",
    "_error",
    "_template_hash",
    "_variables_hash",
    "_lookup_hash",
    "_lookup_source",
    "_system_prompt_source",
    "_template_source",
)

#: Branch-bookkeeping keys that must never appear in a business row.
_BRANCH_BOOKKEEPING_KEYS: frozenset[str] = frozenset(
    {
        "copy_index",
        "branch",
        "branch_id",
        "branch_name",
        "coalesce_id",
        "merge_id",
        "token_id",
        "row_id",
        "lineage",
        "lineage_hash",
        "expansion_group_id",
    }
)

#: Substrings that mark a model identifier as a scripted / mock / replay double
#: rather than a real provider model. Matched case-insensitively as whole
#: ``[a-z0-9]+`` segments so a real ``anthropic/claude-...`` is never flagged.
_FAKE_MODEL_MARKERS: frozenset[str] = frozenset(
    {
        "mock",
        "fake",
        "test",
        "stub",
        "dummy",
        "canned",
        "scripted",
        "replay",
        "cache",
        "cached",
        "sentinel",
        "offline",
        "deterministic",
        "placeholder",
        "example",
    }
)

#: Provider-request-id sentinels a double emits when it forges a plausible id.
_FAKE_REQUEST_ID_MARKERS: frozenset[str] = frozenset(
    {"mock", "fake", "test", "stub", "replay", "cache", "scripted", "sentinel", "deterministic"}
)

#: Key names (lowercased) whose presence anywhere in the evidence means a
#: credential / secret / raw provider response was retained. Deliberately
#: precise: ``prompt_tokens`` / ``completion_tokens`` / ``pending_tokens`` are
#: token *counts*, not auth tokens, so a bare ``token`` substring is NOT used.
_SENSITIVE_KEY_RE = re.compile(
    r"""(
        authorization
      | (^|[_-])cookie($|[_-])
      | set[_-]?cookie
      | api[_-]?key
      | apikey
      | x[_-]api[_-]?key
      | secret
      | password
      | passwd
      | credential
      | access[_-]?token
      | refresh[_-]?token
      | session[_-]?token
      | id[_-]?token
      | auth[_-]?token
      | bearer
      | private[_-]?key
      | client[_-]?secret
      | raw[_-]?response
      | response[_-]?body
      | raw[_-]?prompt
      | prompt[_-]?text
      | system[_-]?prompt
      | resolved[_-]?secret
    )""",
    re.VERBOSE,
)

#: Obvious secret markers that can appear in string *values* under a benign key.
_SENSITIVE_VALUE_RE = re.compile(r"(bearer\s+\S|-----BEGIN [A-Z ]*PRIVATE KEY-----|sk-[A-Za-z0-9]{16,})", re.IGNORECASE)

# Evidence file names under ``<evidence-dir>/<revision>/``.
MANIFEST_FILE = "manifest.json"
GRAPH_FILE = "graph.json"
RUN_LLM_CALLS_FILE = "run_llm_calls.json"
RUN_ACCOUNTING_FILE = "run_accounting.json"
BUSINESS_OUTPUT_FILE = "business_output.json"
INPUT_IDENTITIES_FILE = "input_identities.json"

EVIDENCE_FILES: tuple[str, ...] = (
    MANIFEST_FILE,
    GRAPH_FILE,
    RUN_LLM_CALLS_FILE,
    RUN_ACCOUNTING_FILE,
    BUSINESS_OUTPUT_FILE,
    INPUT_IDENTITIES_FILE,
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class AcceptanceError(Exception):
    """A live-acceptance check failed; ``code`` is a stable machine label."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class EvidenceHygieneError(AcceptanceError):
    """Evidence-directory hygiene / redaction discipline failed."""


# --------------------------------------------------------------------------- #
# Parsed evidence + report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Evidence:
    """Parsed, hygiene-checked evidence for one deployed revision."""

    revision: str
    surface: str
    source_dir: Path
    manifest: Mapping[str, Any]
    graph: Mapping[str, Any]
    run_llm_calls: Sequence[Mapping[str, Any]]
    run_accounting: Mapping[str, Any]
    business_output: Mapping[str, Any]
    input_identities: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class AcceptanceReport:
    """The set of checks a piece of evidence passed."""

    revision: str
    surface: str
    checks: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Small typed accessors (fail closed with a stable code)
# --------------------------------------------------------------------------- #


def _require_mapping(value: Any, code: str, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(code, f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def _require_list(value: Any, code: str, what: str) -> list[Any]:
    # A JSON array parses to ``list``; a JSONL / object root does not.
    if not isinstance(value, list):
        raise AcceptanceError(code, f"{what} must be a JSON array, got {type(value).__name__}")
    return value


def _require_str(value: Any, code: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceError(code, f"{what} must be a non-empty string")
    return value


def _require_int(value: Any, code: str, what: str) -> int:
    # ``bool`` is a subclass of ``int`` — reject it explicitly (batch_stats does
    # the same for the same reason).
    if type(value) is not int:
        raise AcceptanceError(code, f"{what} must be an integer, got {type(value).__name__}")
    return value


def _require_exact_int(value: Any, expected: int, code: str, what: str) -> None:
    if _require_int(value, code, what) != expected:
        raise AcceptanceError(code, f"{what} must be {expected}, got {value!r}")


# --------------------------------------------------------------------------- #
# Recursive sensitive-content scan
# --------------------------------------------------------------------------- #


def _walk(value: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def assert_no_sensitive(value: Any, *, where: str) -> None:
    """Reject evidence that retained a credential / secret / raw response.

    Scans every key name and every string value. Fail-closed: presence means
    the exporter's redaction did not run, so the evidence is not admissible.
    """
    for path, node in _walk(value):
        if isinstance(node, Mapping):
            for key in node:
                if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key.lower()):
                    raise EvidenceHygieneError(
                        "sensitive_content_retained",
                        f"{where}: sensitive key {key!r} at {path} must never be retained",
                    )
        if isinstance(node, str) and _SENSITIVE_VALUE_RE.search(node):
            raise EvidenceHygieneError(
                "sensitive_content_retained",
                f"{where}: value at {path} looks like a secret / bearer token and must never be retained",
            )


def redact_evidence(value: Any) -> Any:
    """Return a deep copy with sensitive keys dropped and secret values masked.

    The exporter calls this before writing evidence to disk so that
    :func:`assert_no_sensitive` passes on load. It is the inverse discipline of
    the load-time fail-closed scan: strip on the way out, reject on the way in.
    """
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key.lower()):
                continue
            redacted[key] = redact_evidence(child)
        return redacted
    if isinstance(value, list):
        return [redact_evidence(child) for child in value]
    if isinstance(value, str) and _SENSITIVE_VALUE_RE.search(value):
        return "[redacted]"
    return value


# --------------------------------------------------------------------------- #
# Directory hygiene
# --------------------------------------------------------------------------- #


def _reject_lax_permissions(target: Path, *, what: str) -> None:
    mode = stat.S_IMODE(target.lstat().st_mode)
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise EvidenceHygieneError(
            "lax_permissions",
            f"{what} {target} is group/other-writable (mode {mode:#o}); evidence must be restrictively permissioned",
        )


def resolve_revision_dir(evidence_dir: str | os.PathLike[str], revision: str) -> Path:
    """Resolve ``<evidence-dir>/<revision>``, enforcing containment + hygiene.

    Rejects a revision token containing path separators, a revision directory
    that is a symlink, any symlink under it, any file resolving outside it, and
    any group/other-writable permission.
    """
    if not revision or "/" in revision or "\\" in revision or revision in {".", ".."}:
        raise EvidenceHygieneError("bad_revision_token", f"revision {revision!r} is not a plain directory name")

    base = Path(evidence_dir).resolve(strict=True)
    revision_dir = base / revision

    if revision_dir.is_symlink():
        raise EvidenceHygieneError("symlink_rejected", f"revision directory {revision_dir} is a symlink")
    if not revision_dir.is_dir():
        raise EvidenceHygieneError("missing_revision_dir", f"revision directory {revision_dir} does not exist")

    resolved_root = revision_dir.resolve(strict=True)
    if resolved_root.parent != base:
        raise EvidenceHygieneError(
            "revision_dir_escapes",
            f"revision directory {revision_dir} resolves to {resolved_root}, outside evidence dir {base}",
        )

    _reject_lax_permissions(revision_dir, what="revision directory")

    for child in _iter_tree(revision_dir):
        if child.is_symlink():
            raise EvidenceHygieneError("symlink_rejected", f"symlink not allowed in evidence tree: {child}")
        resolved_child = child.resolve(strict=True)
        if resolved_root not in resolved_child.parents and resolved_child != resolved_root:
            raise EvidenceHygieneError(
                "path_escapes_revision_dir",
                f"path {child} resolves to {resolved_child}, outside {resolved_root}",
            )
        _reject_lax_permissions(child, what="evidence path")

    return revision_dir


def _iter_tree(root: Path) -> Iterator[Path]:
    for child in sorted(root.iterdir()):
        yield child
        if child.is_dir() and not child.is_symlink():
            yield from _iter_tree(child)


def _load_json_file(revision_dir: Path, name: str) -> Any:
    path = revision_dir / name
    if not path.is_file():
        raise EvidenceHygieneError("missing_evidence_file", f"required evidence file {name} is absent from {revision_dir}")
    try:
        # allow_nan default True: a NaN token round-trips so the finite-value
        # contract check (not the parser) is what rejects it.
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AcceptanceError("malformed_evidence_json", f"{name} is not valid JSON: {exc}") from exc


def load_evidence(evidence_dir: str | os.PathLike[str], revision: str) -> Evidence:
    """Load + hygiene-check evidence for ``revision`` from ``evidence_dir``."""
    revision_dir = resolve_revision_dir(evidence_dir, revision)

    manifest = _require_mapping(_load_json_file(revision_dir, MANIFEST_FILE), "manifest_shape", "manifest")
    graph = _require_mapping(_load_json_file(revision_dir, GRAPH_FILE), "graph_shape", "graph")
    run_llm_calls = _require_list(_load_json_file(revision_dir, RUN_LLM_CALLS_FILE), "run_llm_calls_shape", "run_llm_calls")
    run_accounting = _require_mapping(_load_json_file(revision_dir, RUN_ACCOUNTING_FILE), "run_accounting_shape", "run_accounting")
    business_output = _require_mapping(_load_json_file(revision_dir, BUSINESS_OUTPUT_FILE), "business_output_shape", "business_output")
    input_identities = _require_list(_load_json_file(revision_dir, INPUT_IDENTITIES_FILE), "input_identities_shape", "input_identities")

    documents: dict[str, Any] = {
        MANIFEST_FILE: manifest,
        GRAPH_FILE: graph,
        RUN_LLM_CALLS_FILE: run_llm_calls,
        RUN_ACCOUNTING_FILE: run_accounting,
        BUSINESS_OUTPUT_FILE: business_output,
        INPUT_IDENTITIES_FILE: input_identities,
    }
    for name, document in documents.items():
        assert_no_sensitive(document, where=name)

    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise AcceptanceError(
            "unsupported_schema_version",
            f"manifest schema_version {schema_version!r} not in {sorted(SUPPORTED_SCHEMA_VERSIONS)}",
        )

    surface = _require_str(manifest.get("surface"), "surface_shape", "manifest.surface")
    if surface not in ALLOWED_SURFACES:
        raise AcceptanceError("unknown_surface", f"manifest.surface {surface!r} not in {sorted(ALLOWED_SURFACES)}")

    return Evidence(
        revision=revision,
        surface=surface,
        source_dir=revision_dir,
        manifest=manifest,
        graph=graph,
        run_llm_calls=[_require_mapping(call, "run_llm_call_shape", f"run_llm_calls[{i}]") for i, call in enumerate(run_llm_calls)],
        run_accounting=run_accounting,
        business_output=business_output,
        input_identities=[
            _require_mapping(row, "input_identity_shape", f"input_identities[{i}]") for i, row in enumerate(input_identities)
        ],
    )


# --------------------------------------------------------------------------- #
# Verification checks
# --------------------------------------------------------------------------- #


def _check_single_revision(evidence: Evidence) -> None:
    """Every document must agree on exactly the accepted revision (one run)."""
    revision = evidence.revision
    for name, document in (
        (MANIFEST_FILE, evidence.manifest),
        (GRAPH_FILE, evidence.graph),
        (RUN_ACCOUNTING_FILE, evidence.run_accounting),
        (BUSINESS_OUTPUT_FILE, evidence.business_output),
    ):
        found = document.get("revision")
        if found != revision:
            raise AcceptanceError(
                "mixed_revision",
                f"{name} declares revision {found!r} but evidence is for {revision!r}; evidence must cover exactly one revision",
            )

    commit = _require_str(evidence.manifest.get("commit_id"), "commit_shape", "manifest.commit_id")
    graph_commit = evidence.graph.get("commit_id")
    if graph_commit is not None and graph_commit != commit:
        raise AcceptanceError("mixed_revision", f"graph commit_id {graph_commit!r} != manifest commit_id {commit!r}")

    # A run distinct from the committed proposal is a mixed-provenance record.
    _require_str(evidence.manifest.get("proposal_id"), "proposal_shape", "manifest.proposal_id")
    _require_str(evidence.manifest.get("run_id"), "run_shape", "manifest.run_id")


def _live_provider_calls(evidence: Evidence) -> list[Mapping[str, Any]]:
    """Return the run LLM calls after proving each is a real provider success.

    Intrinsic discriminator (never the self-declared ``provider.mode``): every
    assessment call must be a SUCCESS carrying a non-sentinel ``model_returned``,
    a non-empty non-sentinel ``provider_request_id``, and positive
    ``prompt_tokens`` / ``completion_tokens`` usage. A scripted/mock/replay/cache
    double cannot produce these without forging provider ids and usage.
    """
    calls = evidence.run_llm_calls
    if not calls:
        raise AcceptanceError("no_run_llm_calls", "evidence records no runtime LLM assessment calls")

    provider = _require_mapping(evidence.manifest.get("provider"), "provider_shape", "manifest.provider")
    declared_mode = provider.get("mode")
    if declared_mode != "live":
        # An honest fake is rejected here; a lying fake is caught intrinsically below.
        raise AcceptanceError("provider_not_live", f"manifest.provider.mode must be 'live', got {declared_mode!r}")

    for index, call in enumerate(calls):
        what = f"run_llm_calls[{index}]"
        status = call.get("status")
        if status != "success":
            raise AcceptanceError("fake_provider", f"{what} status {status!r} is not a real provider success")

        model = _require_str(call.get("model_returned"), "fake_provider", f"{what}.model_returned")
        if _has_fake_marker(model, _FAKE_MODEL_MARKERS):
            raise AcceptanceError("fake_provider", f"{what}.model_returned {model!r} is a scripted/mock sentinel, not a real model")

        request_id = _require_str(call.get("provider_request_id"), "fake_provider", f"{what}.provider_request_id")
        if _has_fake_marker(request_id, _FAKE_REQUEST_ID_MARKERS):
            raise AcceptanceError("fake_provider", f"{what}.provider_request_id {request_id!r} is a forged sentinel")

        usage = _require_mapping(call.get("usage"), "fake_provider", f"{what}.usage")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise AcceptanceError("fake_provider", f"{what}.usage.prompt_tokens {prompt_tokens!r} is not a positive real usage count")
        if type(completion_tokens) is not int or completion_tokens <= 0:
            raise AcceptanceError(
                "fake_provider", f"{what}.usage.completion_tokens {completion_tokens!r} is not a positive real usage count"
            )

    return list(calls)


def _has_fake_marker(value: str, markers: Iterable[str] = _FAKE_MODEL_MARKERS) -> bool:
    segments = set(re.split(r"[^a-z0-9]+", value.lower()))
    return bool(segments & set(markers))


def _llm_nodes(evidence: Evidence) -> list[Mapping[str, Any]]:
    nodes = _require_list(evidence.graph.get("nodes"), "graph_nodes_shape", "graph.nodes")
    return [_require_mapping(node, "graph_node_shape", "graph.nodes[]") for node in nodes if node.get("node_type") == "llm"]


def _check_two_distinct_llm_branches(evidence: Evidence, calls: Sequence[Mapping[str, Any]]) -> None:
    """Two distinct LLM nodes, blue + red, each covering all ten source rows."""
    llm_nodes = _llm_nodes(evidence)
    node_ids = [_require_str(node.get("id"), "llm_node_id_shape", "llm node id") for node in llm_nodes]
    if len(set(node_ids)) != 2:
        raise AcceptanceError(
            "two_distinct_llm_nodes",
            f"expected exactly two distinct LLM nodes, found {sorted(set(node_ids))}",
        )

    # The two branches must be independently configured: disjoint typed outputs
    # covering exactly the six branch-exclusive fields (blue_* and red_*).
    output_field_sets: list[frozenset[str]] = []
    for node in llm_nodes:
        fields = frozenset(_require_list(node.get("output_fields"), "llm_output_fields_shape", f"{node.get('id')}.output_fields"))
        output_field_sets.append(fields)
    if output_field_sets[0] & output_field_sets[1]:
        raise AcceptanceError("two_distinct_llm_nodes", "the two LLM branches share output fields; they must be independent")
    union = output_field_sets[0] | output_field_sets[1]
    if union != (BLUE_BRANCH_FIELDS | RED_BRANCH_FIELDS):
        raise AcceptanceError(
            "two_distinct_llm_nodes",
            f"LLM branch output fields {sorted(union)} do not match the required blue/red typed fields",
        )
    if {BLUE_BRANCH_FIELDS, RED_BRANCH_FIELDS} != set(output_field_sets):
        raise AcceptanceError("two_distinct_llm_nodes", "LLM branches must be exactly one blue and one red assessment")

    # Every call must belong to one of the two graph LLM nodes.
    call_node_ids = {_require_str(call.get("branch_node_id"), "call_branch_shape", "run_llm_calls[].branch_node_id") for call in calls}
    if call_node_ids != set(node_ids):
        raise AcceptanceError(
            "two_distinct_llm_nodes",
            f"runtime LLM calls reference nodes {sorted(call_node_ids)}, graph declares {sorted(set(node_ids))}",
        )

    identities = _identity_set(evidence)
    for node_id in node_ids:
        covered = {
            _identity_key(_require_mapping(call.get("input_identity"), "call_identity_shape", "run_llm_calls[].input_identity"))
            for call in calls
            if call.get("branch_node_id") == node_id
        }
        if covered != identities:
            missing = identities - covered
            raise AcceptanceError(
                "branch_missing_rows",
                f"LLM branch {node_id!r} did not assess all ten rows; missing {sorted(missing)}",
            )


def _check_require_all_union_merge(evidence: Evidence) -> None:
    merge = _require_mapping(evidence.graph.get("merge"), "merge_shape", "graph.merge")
    mode = merge.get("mode")
    if mode != "require_all":
        raise AcceptanceError("wrong_merge", f"coalesce mode {mode!r} is not 'require_all'")
    if merge.get("semantics") != "union":
        raise AcceptanceError("wrong_merge", f"coalesce semantics {merge.get('semantics')!r} is not 'union'")

    required = set(_require_list(merge.get("required_branches"), "merge_branches_shape", "graph.merge.required_branches"))
    llm_ids = {_require_str(node.get("id"), "llm_node_id_shape", "llm node id") for node in _llm_nodes(evidence)}
    if required != llm_ids:
        raise AcceptanceError(
            "wrong_merge",
            f"require-all coalesce must require both LLM branches {sorted(llm_ids)}, requires {sorted(required)}",
        )


def _outputs_by_role(evidence: Evidence) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    outputs = _require_list(evidence.graph.get("outputs"), "outputs_shape", "graph.outputs")
    success = [o for o in outputs if o.get("role") == "success"]
    failure = [o for o in outputs if o.get("role") == "failure"]
    if len(success) != 1 or len(failure) != 1:
        raise AcceptanceError(
            "output_roles",
            f"graph must declare exactly one success and one failure output; got {len(success)} success / {len(failure)} failure",
        )
    return success[0], failure[0]


def _check_error_routes_reach_failure(evidence: Evidence) -> None:
    _success_output, failure_output = _outputs_by_role(evidence)
    failure_sink = _require_str(failure_output.get("sink_name"), "failure_sink_shape", "failure output sink_name")

    edges = _require_list(evidence.graph.get("edges"), "edges_shape", "graph.edges")
    error_targets = {edge.get("to") for edge in edges if edge.get("role") in {"error", "on_error"}}
    if failure_sink not in error_targets:
        raise AcceptanceError(
            "unrouted_failure",
            f"no error-role edge reaches the failure output {failure_sink!r}; branch failures would be lost",
        )


def _check_cleanup_field_contract(evidence: Evidence) -> None:
    contract = _require_mapping(evidence.graph.get("field_contract"), "field_contract_shape", "graph.field_contract")
    success_fields = tuple(_require_list(contract.get("success_fields"), "success_fields_shape", "graph.field_contract.success_fields"))
    if success_fields != APPROVED_OUTPUT_FIELDS:
        raise AcceptanceError(
            "cleanup_field_contract",
            f"cleanup retains {list(success_fields)}, must retain exactly {list(APPROVED_OUTPUT_FIELDS)}",
        )


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in IDENTITY_FIELDS)


def _identity_set(evidence: Evidence) -> set[tuple[Any, ...]]:
    for row in evidence.input_identities:
        for field in IDENTITY_FIELDS:
            _require_str(row.get(field), "identity_field_shape", f"input identity field {field}")
    keys = [_identity_key(row) for row in evidence.input_identities]
    if len(keys) != EXPECTED_ROW_COUNT:
        raise AcceptanceError("input_identity_count", f"expected {EXPECTED_ROW_COUNT} input identities, got {len(keys)}")
    if len(set(keys)) != len(keys):
        raise AcceptanceError("duplicate_identity", "input identities are not unique")
    return set(keys)


def _check_business_output(evidence: Evidence) -> None:
    body = evidence.business_output
    # Root must be a JSON array of objects — a JSONL / object root is rejected.
    success_rows = _require_list(body.get("success"), "jsonl_root", "business_output.success")
    failure_rows = _require_list(body.get("failure"), "failure_output_shape", "business_output.failure")

    if failure_rows:
        raise AcceptanceError("unexpected_failures", f"business output has {len(failure_rows)} failure rows; expected zero")
    if len(success_rows) != EXPECTED_ROW_COUNT:
        raise AcceptanceError("success_row_count", f"expected {EXPECTED_ROW_COUNT} successful rows, got {len(success_rows)}")

    identities = _identity_set(evidence)
    seen_identities: set[tuple[Any, ...]] = set()

    for index, raw_row in enumerate(success_rows):
        row = _require_mapping(raw_row, "business_row_shape", f"business_output.success[{index}]")
        keys = set(row.keys())

        # No raw response / usage / model metadata / branch bookkeeping leaked.
        # Checked before the exact-field gate so a leaked provider-metadata field
        # gets the specific ``leaked_metadata`` code rather than the generic
        # extra-field one.
        _assert_no_leaked_fields(row, index)

        # Exact eight-field cleanup.
        if keys != APPROVED_OUTPUT_FIELD_SET:
            extra = keys - APPROVED_OUTPUT_FIELD_SET
            missing = APPROVED_OUTPUT_FIELD_SET - keys
            raise AcceptanceError(
                "field_contract_violation",
                f"business row {index} fields != approved eight (extra={sorted(extra)}, missing={sorted(missing)})",
            )

        # Typed integer amounts in inclusive 0..100.
        for field in AMOUNT_FIELDS:
            amount = row[field]
            if type(amount) is not int:  # rejects bool (subclass of int)
                raise AcceptanceError("amount_not_integer", f"business row {index}: {field}={amount!r} is not an integer")
            if not (0 <= amount <= 100):
                raise AcceptanceError("amount_out_of_range", f"business row {index}: {field}={amount} outside 0..100")

        # Numeric confidences in inclusive 0..1, finite (rejects NaN/Infinity).
        for field in CONFIDENCE_FIELDS:
            confidence = row[field]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise AcceptanceError("confidence_not_number", f"business row {index}: {field}={confidence!r} is not a number")
            if not math.isfinite(float(confidence)):
                raise AcceptanceError("confidence_not_finite", f"business row {index}: {field}={confidence!r} is not finite")
            if not (0.0 <= float(confidence) <= 1.0):
                raise AcceptanceError("confidence_out_of_range", f"business row {index}: {field}={confidence} outside 0..1")

        # Non-empty reasons.
        for field in REASON_FIELDS:
            reason = row[field]
            if not isinstance(reason, str) or not reason.strip():
                raise AcceptanceError("reason_empty", f"business row {index}: {field} must be a non-empty sentence")

        identity = _identity_key(row)
        if identity not in identities:
            raise AcceptanceError("unknown_business_identity", f"business row {index} identity {identity} is not a known input row")
        if identity in seen_identities:
            raise AcceptanceError("duplicate_identity", f"business row {index} repeats identity {identity}")
        seen_identities.add(identity)

    if seen_identities != identities:
        raise AcceptanceError("business_identity_coverage", "business rows do not cover exactly the ten input identities")


def _assert_no_leaked_fields(row: Mapping[str, Any], index: int) -> None:
    for key in row:
        lowered = key.lower()
        if any(lowered.endswith(suffix) for suffix in _LEAKAGE_SUFFIXES):
            raise AcceptanceError("leaked_metadata", f"business row {index}: field {key!r} leaks raw response / usage / model metadata")
        if lowered in _BRANCH_BOOKKEEPING_KEYS:
            raise AcceptanceError("leaked_metadata", f"business row {index}: field {key!r} leaks branch bookkeeping")


def _check_run_accounting(evidence: Evidence, calls: Sequence[Mapping[str, Any]]) -> None:
    accounting = evidence.run_accounting

    _require_exact_int(accounting.get("assessments"), EXPECTED_ASSESSMENTS, "assessment_count", "run_accounting.assessments")
    if len(calls) != EXPECTED_ASSESSMENTS:
        raise AcceptanceError(
            "assessment_count",
            f"expected {EXPECTED_ASSESSMENTS} runtime LLM assessment records, got {len(calls)}",
        )
    _require_exact_int(accounting.get("coalesces_completed"), EXPECTED_COALESCES, "coalesce_count", "run_accounting.coalesces_completed")
    _require_exact_int(accounting.get("successes"), EXPECTED_ROW_COUNT, "success_count", "run_accounting.successes")
    _require_exact_int(accounting.get("failures"), 0, "failure_count", "run_accounting.failures")

    pending = accounting.get("pending_tokens")
    if type(pending) is not int or pending != 0:
        raise AcceptanceError("nonterminal_accounting", f"run_accounting.pending_tokens must be 0, got {pending!r}")
    if accounting.get("closed") is not True:
        raise AcceptanceError("nonterminal_accounting", "run_accounting.closed must be true; open accounting is not a terminal run")


def _check_authoring_discipline(evidence: Evidence) -> None:
    """At most one automatic structured repair and no operator correction.

    Task 6 runs each surface from a clean session and allows at most one
    automatic structured repair and NO operator topology/configuration
    correction; the oracle enforces both from the manifest so a run that was
    hand-steered to success cannot be accepted as an autonomous derivation.
    """
    provider = _require_mapping(evidence.manifest.get("provider"), "provider_shape", "manifest.provider")

    repair_count = provider.get("repair_count", 0)
    if type(repair_count) is not int or repair_count < 0:
        raise AcceptanceError("repair_count_shape", f"manifest.provider.repair_count must be a non-negative int, got {repair_count!r}")
    if repair_count > 1:
        raise AcceptanceError("excess_repair", f"{repair_count} structured repairs exceeds the single-repair budget")

    corrections = provider.get("operator_corrections", [])
    corrections_list = _require_list(corrections, "corrections_shape", "manifest.provider.operator_corrections")
    for correction in corrections_list:
        correction_map = _require_mapping(correction, "correction_shape", "operator correction")
        kind = correction_map.get("kind")
        if kind in {"topology", "configuration", "config"}:
            raise AcceptanceError(
                "operator_topology_correction",
                f"operator applied a {kind!r} correction; the graph was not derived autonomously",
            )


def verify(evidence: Evidence) -> AcceptanceReport:
    """Run every ORACLE-CONTRACT check; raise :class:`AcceptanceError` on failure."""
    _check_single_revision(evidence)
    calls = _live_provider_calls(evidence)
    _check_two_distinct_llm_branches(evidence, calls)
    _check_require_all_union_merge(evidence)
    _check_error_routes_reach_failure(evidence)
    _check_cleanup_field_contract(evidence)
    _check_business_output(evidence)
    _check_run_accounting(evidence, calls)
    _check_authoring_discipline(evidence)

    return AcceptanceReport(
        revision=evidence.revision,
        surface=evidence.surface,
        checks=(
            "single_revision",
            "live_provider",
            "two_distinct_llm_branches",
            "require_all_union_merge",
            "error_routes_reach_failure",
            "cleanup_field_contract",
            "business_output",
            "run_accounting",
            "authoring_discipline",
        ),
    )


def verify_evidence_dir(evidence_dir: str | os.PathLike[str], revision: str) -> AcceptanceReport:
    """Load, hygiene-check, and verify evidence for ``revision``."""
    return verify(load_evidence(evidence_dir, revision))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class LiveCollectionUnavailable(RuntimeError):
    """No staging evidence collector is wired for the requested surface.

    The oracle refuses to fabricate provider evidence. Task 6 supplies the
    concrete staging collector (or exports evidence via the Playwright journey
    and calls ``verify``); this default declines rather than invent a run.
    """


class LiveEvidenceCollector:
    """Callable that produces a sanitized evidence document from a real run.

    A collector drives one deployed composer surface against ``base_url`` with
    environment credentials, executes the derived pipeline against the live
    provider, and reads back the committed graph, runtime LLM audit, run
    accounting, and business outputs as the six-document evidence mapping. It
    must NOT return credentials or raw provider responses — the export path runs
    :func:`redact_evidence` regardless, but a collector should sanitize at the
    source.
    """

    def __call__(
        self,
        *,
        base_url: str,
        api_key: str,
        surface: str,
        fixture: Path,
        intent: Path,
        revision: str,
    ) -> Mapping[str, Any]:
        raise LiveCollectionUnavailable(
            f"no staging evidence collector is wired for surface {surface!r}; "
            f"Task 6 drives {base_url} and exports evidence for `live_acceptance verify`. "
            "This oracle never fabricates provider evidence."
        )


def run_live(
    *,
    base_url: str,
    api_key: str,
    surface: str,
    fixture: Path,
    intent: Path,
    revision: str,
    evidence_dir: Path,
    collector: LiveEvidenceCollector | None = None,
) -> AcceptanceReport:
    """Collect, redact, persist, and verify evidence for one live run.

    This is the offline-testable machinery of the ``run`` command: it never
    reads the environment and never imports a provider SDK. A live run injects
    no collector and gets :class:`LiveEvidenceCollector` (which declines); Task
    6 / tests inject a concrete collector. The collected document is redacted
    and written with restrictive permissions, then re-loaded and verified
    through the same path an operator's ``verify`` invocation takes.
    """
    active = collector if collector is not None else LiveEvidenceCollector()
    document = active(
        base_url=base_url,
        api_key=api_key,
        surface=surface,
        fixture=fixture,
        intent=intent,
        revision=revision,
    )
    revision_dir = _write_evidence_dir(evidence_dir, revision, document)
    return verify_evidence_dir(revision_dir.parent, revision)


def _run_live(args: argparse.Namespace) -> int:
    """CLI glue for ``run``: gate on env credentials, then delegate to :func:`run_live`.

    Key-gated and environment-only for credentials. The deterministic test suite
    exercises :func:`run_live` with an injected collector; the CLI path performs
    a REAL provider run and is never invoked by the tests.
    """
    base_url = args.base_url or os.environ.get("ELSPETH_EVAL_BASE_URL")
    if not base_url:
        raise SystemExit("run: --base-url or $ELSPETH_EVAL_BASE_URL is required")
    api_key = os.environ.get("ELSPETH_EVAL_API_KEY")
    if not api_key:
        raise SystemExit("run: provider credentials must be supplied via $ELSPETH_EVAL_API_KEY (never as an argument)")

    try:
        report = run_live(
            base_url=base_url,
            api_key=api_key,
            surface=args.surface,
            fixture=Path(args.fixture),
            intent=Path(args.intent),
            revision=args.revision,
            evidence_dir=Path(args.evidence_dir),
        )
    except AcceptanceError as exc:
        print(f"REJECTED [{exc.code}] {exc.message}", file=sys.stderr)
        return 1
    print(f"ACCEPTED surface={report.surface} revision={report.revision} checks={len(report.checks)}")
    return 0


def _write_evidence_dir(evidence_dir: Path, revision: str, document: Mapping[str, Any]) -> Path:
    """Write redacted evidence with restrictive permissions and return the dir.

    Shared by the live ``run`` path and available to acceptance tooling. Every
    document is passed through :func:`redact_evidence` before write, and the
    directory + files are chmodded so :func:`assert_no_sensitive` and the
    permission gate pass on reload.
    """
    if "/" in revision or "\\" in revision:
        raise ValueError(f"revision {revision!r} must be a plain directory name")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    revision_dir = evidence_dir / revision
    revision_dir.mkdir(mode=0o700, exist_ok=True)
    revision_dir.chmod(0o700)
    for name in EVIDENCE_FILES:
        payload = redact_evidence(document[name])
        path = revision_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        path.chmod(0o600)
    return revision_dir


def _run_verify(args: argparse.Namespace) -> int:
    try:
        report = verify_evidence_dir(args.evidence_dir, args.revision)
    except AcceptanceError as exc:
        print(f"REJECTED [{exc.code}] {exc.message}", file=sys.stderr)
        return 1
    if args.surface is not None and report.surface != args.surface:
        print(f"REJECTED [surface_mismatch] evidence surface {report.surface!r} != requested {args.surface!r}", file=sys.stderr)
        return 1
    print(f"ACCEPTED surface={report.surface} revision={report.revision} checks={len(report.checks)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="live_acceptance", description="Composer capability-parity live-acceptance oracle")
    sub = parser.add_subparsers(dest="command", required=True)

    verify_parser = sub.add_parser("verify", help="verify already-exported evidence for one revision")
    verify_parser.add_argument("--evidence-dir", required=True)
    verify_parser.add_argument("--revision", required=True)
    verify_parser.add_argument("--surface", choices=sorted(ALLOWED_SURFACES), default=None)
    verify_parser.set_defaults(func=_run_verify)

    run_parser = sub.add_parser("run", help="drive one live surface, export evidence, then verify (credentials env-only)")
    run_parser.add_argument("--surface", required=True, choices=sorted(ALLOWED_SURFACES))
    run_parser.add_argument("--base-url", default=None, help="deployed base URL (defaults to $ELSPETH_EVAL_BASE_URL)")
    run_parser.add_argument("--fixture", required=True, help="path to the ten-row colour CSV")
    run_parser.add_argument("--intent", required=True, help="path to the outcome-only request text")
    run_parser.add_argument("--revision", required=True)
    run_parser.add_argument("--evidence-dir", required=True)
    run_parser.set_defaults(func=_run_live)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func: Any = args.func
    result = func(args)
    return int(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
