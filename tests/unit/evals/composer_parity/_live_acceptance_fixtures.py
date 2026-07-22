"""Shared, sanitized evidence builders for the live-acceptance oracle tests.

Not a test module. The oracle lives in
``evals/composer-parity/live_acceptance.py``; the hyphenated directory is not an
importable package, so it is loaded from its file path and registered in
``sys.modules`` (dataclass field-type resolution reads ``sys.modules``). Both
the unit suite and the server-side integration suite import ``la`` and
``build_valid_evidence`` from here so the canonical valid evidence has one
source of truth.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

# _live_acceptance_fixtures -> composer_parity -> evals -> unit -> tests -> repo
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODULE_PATH = _REPO_ROOT / "evals" / "composer-parity" / "live_acceptance.py"
_MODULE_NAME = "composer_parity_live_acceptance"


def _load_oracle() -> Any:
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass field-type resolution reads sys.modules.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


la = _load_oracle()

REVISION = "rev0123456789"
COMMIT = "c" * 40

# The ten canonical colour rows (design "Input"; matches two_llm_colour.csv).
COLOURS: tuple[tuple[str, str], ...] = (
    ("Pure Red", "#FF0000"),
    ("Pure Blue", "#0000FF"),
    ("Purple", "#800080"),
    ("Magenta", "#FF00FF"),
    ("Cyan", "#00FFFF"),
    ("Navy", "#000080"),
    ("Orange", "#FF7F00"),
    ("Teal", "#008080"),
    ("Grey", "#808080"),
    ("White", "#FFFFFF"),
)


def _business_row(index: int, name: str, hex_value: str) -> dict[str, Any]:
    return {
        "color_name": name,
        "hex": hex_value,
        "blue_amount": (index * 7) % 101,
        "blue_confidence": round(0.5 + index * 0.03, 3),
        "blue_reason": f"{name} carries a measurable blue component.",
        "red_amount": (index * 11) % 101,
        "red_confidence": round(0.4 + index * 0.02, 3),
        "red_reason": f"{name} carries a measurable red component.",
    }


def _llm_call(node_id: str, index: int, name: str, hex_value: str) -> dict[str, Any]:
    return {
        "branch_node_id": node_id,
        "input_identity": {"color_name": name, "hex": hex_value},
        "status": "success",
        "model_returned": "anthropic/claude-sonnet-4.6",
        "provider_request_id": f"gen-{node_id}-{index:02d}",
        "usage": {"prompt_tokens": 120 + index, "completion_tokens": 40 + index, "total_tokens": 160 + 2 * index},
    }


def build_valid_evidence(*, revision: str = REVISION, surface: str = "guided_full") -> dict[str, Any]:
    """A complete, sanitized evidence document that satisfies every check."""
    identities = [{"color_name": name, "hex": hex_value} for name, hex_value in COLOURS]
    success_rows = [_business_row(i, name, hex_value) for i, (name, hex_value) in enumerate(COLOURS)]
    calls: list[dict[str, Any]] = []
    for node_id in ("blue_llm", "red_llm"):
        for i, (name, hex_value) in enumerate(COLOURS):
            calls.append(_llm_call(node_id, i, name, hex_value))

    graph = {
        "revision": revision,
        "commit_id": COMMIT,
        "source": {"id": "colours", "plugin": "csv", "row_count": 10},
        "nodes": [
            {"id": "fork", "node_type": "fork"},
            {"id": "blue_llm", "node_type": "llm", "output_fields": ["blue_amount", "blue_confidence", "blue_reason"]},
            {"id": "red_llm", "node_type": "llm", "output_fields": ["red_amount", "red_confidence", "red_reason"]},
            {"id": "merge", "node_type": "coalesce"},
            {"id": "cleanup", "node_type": "transform"},
        ],
        "edges": [
            {"from": "source", "to": "fork", "role": "success"},
            {"from": "fork", "to": "blue_llm", "role": "success"},
            {"from": "fork", "to": "red_llm", "role": "success"},
            {"from": "blue_llm", "to": "merge", "role": "success"},
            {"from": "red_llm", "to": "merge", "role": "success"},
            {"from": "merge", "to": "cleanup", "role": "success"},
            {"from": "cleanup", "to": "success_output", "role": "success"},
            {"from": "blue_llm", "to": "failure_output", "role": "error"},
            {"from": "red_llm", "to": "failure_output", "role": "error"},
            {"from": "merge", "to": "failure_output", "role": "error"},
        ],
        "merge": {"node_id": "merge", "mode": "require_all", "semantics": "union", "required_branches": ["blue_llm", "red_llm"]},
        "outputs": [
            {"sink_name": "success_output", "plugin": "json", "role": "success"},
            {"sink_name": "failure_output", "plugin": "json", "role": "failure"},
        ],
        "field_contract": {"success_fields": list(la.APPROVED_OUTPUT_FIELDS)},
    }

    return {
        la.MANIFEST_FILE: {
            "schema_version": 1,
            "revision": revision,
            "surface": surface,
            "proposal_id": "prop-abc",
            "commit_id": COMMIT,
            "run_id": "run-xyz",
            "base_url_host": "elspeth.foundryside.dev",
            "provider": {"mode": "live", "repair_count": 0, "operator_corrections": []},
        },
        la.GRAPH_FILE: graph,
        la.RUN_LLM_CALLS_FILE: calls,
        la.RUN_ACCOUNTING_FILE: {
            "revision": revision,
            "assessments": 20,
            "coalesces_completed": 10,
            "pending_tokens": 0,
            "closed": True,
            "successes": 10,
            "failures": 0,
        },
        la.BUSINESS_OUTPUT_FILE: {"revision": revision, "success": success_rows, "failure": []},
        la.INPUT_IDENTITIES_FILE: identities,
    }


def write_evidence(evidence_dir: Path, revision: str, document: dict[str, Any]) -> Path:
    """Write evidence WITHOUT redaction so load-time hygiene is exercised.

    Directory + files get restrictive perms so the permission gate passes; a
    permission negative overrides that explicitly.
    """
    revision_dir = evidence_dir / revision
    revision_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    revision_dir.chmod(0o700)
    for name, payload in document.items():
        path = revision_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o600)
    return revision_dir
