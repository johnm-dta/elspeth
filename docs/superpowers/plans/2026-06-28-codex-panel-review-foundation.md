# Codex Panel Review — Foundation (Plan 1 of 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the single-file foundation of `codex_panel_review.py` — layered/cached prompt, panel finding schema, category-aware structured gate, and a serial per-lens runner over **one** file using the *stock* `run_codex_once` — then run the caching/cost/throughput pilot that supplies the empirical numbers Plan 2 is written from.

**Architecture:** A new, self-contained `scripts/codex_panel_review.py` that reuses the small, pure helpers from `codex_audit_common.py` (`run_codex_once`, rate limiter, usage parsing, `load_context`, log helpers, `has_file_line_evidence`, `structured_output_path_for_report`, `_structured_findings`) but adds its own prompt builder, finding schema, and a new structured-first gate `apply_panel_evidence_gate`. Plan 1 deliberately runs lenses **serially over a single file** — no streaming runner, no worker pool, no merge/synthesis (those are Plan 2). Serial-on-one-file is exactly what the pilot needs to measure cross-lens prompt-cache reuse.

**Tech Stack:** Python 3.12 (`.venv` = 3.12.3), `asyncio`, the `codex` CLI (`codex exec --json --output-schema`), `pytest`. No new third-party deps.

## Global Constraints

[Every task's requirements implicitly include this section. Values copied from the spec `docs/superpowers/specs/2026-06-28-codex-panel-review-design.md`.]

- **Python interpreter:** the repo's main `.venv` is **Python 3.12** (`.venv/bin/python` reports 3.12.3; verify live before assuming) — run all tests with `.venv/bin/python -m pytest`.
- **Do not modify** `run_codex_once`, `run_codex_with_retry_and_logging`, or `apply_evidence_gate` in `codex_audit_common.py`, nor any of the four sibling scripts (`codex_test_defect_hunt.py`, `codex_integration_seam_hunt.py`, `codex_exemption_validator.py`, `codex_tier_model_rejudge.py`). New code is additive only.
- **Finding schema keeps `priority`** (values exactly `P0|P1|P2|P3`) — it is the field every shared writer reads (`_priority_from_structured_finding`). Never put a P-level in a field named `severity`.
- **`--output-schema` JSON must include a non-empty `markdown_report` string field** — `run_codex_once` calls `_extract_structured_markdown(..., "markdown_report")` whenever `output_schema` is set and raises if that field is absent (`codex_audit_common.py:365`).
- **The gate must rewrite the sidecar unconditionally (last)** — `_structured_findings` ignores a sidecar older than its `.md` (`codex_audit_common.py:820`); `run_codex_once` writes the `.md` *after* the sidecar, so a freshly-produced sidecar is "stale" until the gate rewrites it.
- **No unit tests assert persona/prompt *text* content** (project doctrine: skill/LLM prompts are not code). Personas are validated by *running* them in the pilot. Tests may assert prompt *structure/ordering*, not wording.
- **Module import:** the script is run directly (`python scripts/codex_panel_review.py`, where `scripts/` is `sys.path[0]`) **and** imported by tests as `scripts.codex_panel_review` (repo root on path). Use the verified dual-import shim (Task 1).

## Plan 2 boundary notes (do NOT implement here — recorded so module seams are drawn to fit)

- **Streaming runner** (`codex exec --json` line-by-line event surfacing) is Plan 2. Plan 1 uses stock `run_codex_once` (buffered) — fine for the pilot, which only needs `cached_input_tokens` + call durations, both already available.
- **Worker pool + completion-sentinel resume + commit pinning** are Plan 2.
- **Cross-lens merge + synthesis pass + counted/raw output layout** are Plan 2. Plan 1 writes per-lens sidecars only. The file-level "write the counted sidecar last" rule is a Plan 2 concern; Plan 1's per-lens analogue (gate rewrites the sidecar last) is implemented here.
- **Own retry/rate-limit/log sequencing:** the panel cannot reuse `run_codex_with_retry_and_logging` (it hardcodes `run_codex_once` + `apply_evidence_gate`). Plan 1's runner does rate-limit + log + failure-capture inline; full retry lands with the Plan 2 streaming runner.

## File Structure

- `scripts/codex_panel_review.py` — **Create.** CLI + foundation functions: dual-import shim, module constants, `build_layered_prompt`, `load_persona`, `route_lenses`, `apply_panel_evidence_gate`, `run_file_lenses` (async, serial), `main` (single-file mode).
- `scripts/codex_lenses/panel_finding.schema.json` — **Create.** The `--output-schema` document (`markdown_report` + `findings[]`).
- `scripts/codex_lenses/security-architect.md` — **Create.** One persona prompt.
- `scripts/codex_lenses/solution-architect.md` — **Create.** A second persona (≥2 lenses are required to measure cross-lens cache reuse in the pilot).
- `tests/unit/scripts/test_codex_panel_review.py` — **Create.** Unit tests for the pure-Python pieces (shim, constants, prompt ordering, persona loader, routing, gate, runner with `run_codex_once` monkeypatched).
- `docs/quality-audit/PANEL_PILOT.md` — **Create (during Task 9).** The pilot writeup: measured cache hit-rate, per-call durations, `--workers` ceiling, and the one-time cost projection.

---

### Task 1: Module scaffold, dual-import shim, and constants

**Files:**
- Create: `scripts/codex_panel_review.py`
- Test: `tests/unit/scripts/test_codex_panel_review.py`

**Interfaces:**
- Produces: module `scripts.codex_panel_review` exposing constants
  `PANEL_FINDING_SCHEMA: Path`, `LENSES_DIR: Path`, `DEFAULT_OUTPUT_DIR: str`,
  `STRICT_CATEGORIES: frozenset[str]`, `RELAXED_CATEGORIES: frozenset[str]`,
  `ANCHOR_REQUIRED_CATEGORIES: frozenset[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/scripts/test_codex_panel_review.py
"""Unit tests for the codex_panel_review foundation (no codex calls)."""
from __future__ import annotations

from pathlib import Path

from scripts import codex_panel_review as cpr


def test_module_constants_present_and_typed():
    assert isinstance(cpr.PANEL_FINDING_SCHEMA, Path)
    assert cpr.PANEL_FINDING_SCHEMA.name == "panel_finding.schema.json"
    assert isinstance(cpr.LENSES_DIR, Path)
    # priority-bearing categories that must carry a file:line anchor
    assert cpr.STRICT_CATEGORIES == frozenset({"bug", "correctness", "security", "smell"})
    assert cpr.RELAXED_CATEGORIES == frozenset({"improvement", "efficiency"})
    assert cpr.ANCHOR_REQUIRED_CATEGORIES == cpr.STRICT_CATEGORIES | frozenset({"easy-win"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py::test_module_constants_present_and_typed -v`
Expected: FAIL — import/collection error: `ImportError: cannot import name 'codex_panel_review' from 'scripts'` (`scripts/__init__.py` exists, so a missing submodule raises a plain `ImportError`, not `ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/codex_panel_review.py
#!/usr/bin/env python3
"""Pre-1.0 sandblasting SME review fleet (foundation: single-file, serial lenses).

See docs/superpowers/specs/2026-06-28-codex-panel-review-design.md.
"""
from __future__ import annotations

from pathlib import Path

# Dual-import shim: bare import works when run as a script (scripts/ on sys.path);
# the package form works when pytest imports this as scripts.codex_panel_review.
try:  # pragma: no cover - exercised by both run modes
    from codex_audit_common import (  # type: ignore[import-not-found]
        append_log,
        ensure_log_file,
        has_file_line_evidence,
        load_context,
        make_codex_rate_limiter,
        run_codex_once,
        structured_output_path_for_report,
        utc_now,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.codex_audit_common import (
        append_log,
        ensure_log_file,
        has_file_line_evidence,
        load_context,
        make_codex_rate_limiter,
        run_codex_once,
        structured_output_path_for_report,
        utc_now,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
LENSES_DIR = Path(__file__).resolve().parent / "codex_lenses"
PANEL_FINDING_SCHEMA = LENSES_DIR / "panel_finding.schema.json"
# Per-lens output goes in the *-raw sibling, NOT the counted `findings-panel/`
# tree. iter_report_files (codex_audit_common.py:792) only skips `by-priority/`
# + the metadata filenames — it does NOT skip an arbitrary subdir — so per-lens
# `.md` files written under the counted tree would be double-counted by Plan 2's
# generate_summary. Keeping raw detail in its own sibling from day one matches
# the spec's output layout and avoids seeding that double-count.
DEFAULT_OUTPUT_DIR = "docs/quality-audit/findings-panel-raw"

STRICT_CATEGORIES = frozenset({"bug", "correctness", "security", "smell"})
RELAXED_CATEGORIES = frozenset({"improvement", "efficiency"})
ANCHOR_REQUIRED_CATEGORIES = STRICT_CATEGORIES | frozenset({"easy-win"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py::test_module_constants_present_and_typed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/codex_panel_review.py tests/unit/scripts/test_codex_panel_review.py
git commit --no-verify -m "feat(panel): scaffold codex_panel_review module + constants"
```

---

### Task 2: Panel finding `--output-schema` document

**Files:**
- Create: `scripts/codex_lenses/panel_finding.schema.json`
- Test: `tests/unit/scripts/test_codex_panel_review.py`

**Interfaces:**
- Produces: a JSON Schema object with required top-level `markdown_report` (string)
  and `findings` (array); each finding requires `priority`, `lens`, `category`,
  `summary`, `evidence`.

- [ ] **Step 1: Write the failing test**

```python
def test_panel_schema_shape():
    import json
    schema = json.loads(cpr.PANEL_FINDING_SCHEMA.read_text(encoding="utf-8"))
    assert schema["type"] == "object"
    # markdown_report is mandatory: run_codex_once extracts it (common.py:365)
    assert "markdown_report" in schema["required"]
    assert schema["properties"]["markdown_report"]["type"] == "string"
    finding = schema["properties"]["findings"]["items"]
    assert finding["properties"]["priority"]["enum"] == ["P0", "P1", "P2", "P3"]
    # priority, NOT severity, carries the P-level
    assert "severity" not in finding["properties"]
    for req in ("priority", "lens", "category", "summary", "evidence"):
        assert req in finding["required"]
    # strict structured-outputs contract: under additionalProperties:false the API
    # rejects any optional property, so EVERY declared property must be required
    # (genuinely-optional fields are nullable union types, never absent from required).
    assert set(finding["required"]) == set(finding["properties"])
    evidence_item = finding["properties"]["evidence"]["items"]
    assert set(evidence_item["required"]) == set(evidence_item["properties"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py::test_panel_schema_shape -v`
Expected: FAIL — `FileNotFoundError` (schema file does not exist yet)

- [ ] **Step 3: Write minimal implementation**

> **OpenAI strict structured-outputs constraint (verified against `codex` 0.142.3):** with `additionalProperties: false`, the API requires **every** key in `properties` to also appear in `required` — a schema with optional properties is rejected with `HTTP 400 invalid_json_schema`, which `run_codex_once` turns into a `RuntimeError` (`codex_audit_common.py:658`) on the *first* pilot call (Tasks 1–8 never catch it because the Step-1 test is `json.loads` + key-membership only). Genuinely-optional fields are therefore declared as **nullable union types**, and the Task 3 prompt instructs the model to emit `null` (not omit). `markdown_report`, `priority`, `lens`, `category`, `summary`, `evidence`, `path`, and `claim` stay non-nullable; everything else is nullable. `gate_note` is declared here because the Task 6 gate writes it back into this same sidecar (and would otherwise re-violate `additionalProperties: false`).

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["markdown_report", "findings"],
  "properties": {
    "markdown_report": {
      "type": "string",
      "description": "Human-readable narration of what this lens reviewed and concluded."
    },
    "findings": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["priority", "lens", "category", "confidence", "effort", "impact", "summary", "evidence", "suggested_fix", "target_file", "gate_note"],
        "properties": {
          "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
          "lens": {"type": "string"},
          "category": {
            "type": "string",
            "enum": ["bug", "correctness", "smell", "efficiency", "improvement", "easy-win", "security", "design"]
          },
          "confidence": {"type": ["string", "null"], "enum": ["high", "medium", "low", null]},
          "effort": {"type": ["string", "null"], "enum": ["trivial", "small", "medium", "large", null]},
          "impact": {"type": ["string", "null"]},
          "summary": {"type": "string"},
          "evidence": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": ["path", "line", "claim"],
              "properties": {
                "path": {"type": "string"},
                "line": {"type": ["integer", "null"]},
                "claim": {"type": "string"}
              }
            }
          },
          "suggested_fix": {"type": ["string", "null"]},
          "target_file": {"type": ["string", "null"]},
          "gate_note": {"type": ["string", "null"]}
        }
      }
    }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py::test_panel_schema_shape -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/codex_lenses/panel_finding.schema.json tests/unit/scripts/test_codex_panel_review.py
git commit --no-verify -m "feat(panel): add panel finding output-schema (markdown_report + findings)"
```

---

### Task 3: `build_layered_prompt` — stable-prefix-first ordering (the caching prerequisite)

**Files:**
- Modify: `scripts/codex_panel_review.py`
- Test: `tests/unit/scripts/test_codex_panel_review.py`

**Interfaces:**
- Produces: `build_layered_prompt(*, context: str, file_source: str, file_path: str, persona: str, lens: str) -> str`.
  Ordering invariant: `context` → `file_source` → `persona/instructions` → per-call tail (`file_path`, `lens`). `file_path` must appear only in the tail (never in the cacheable head).

- [ ] **Step 1: Write the failing test**

```python
def test_layered_prompt_orders_stable_content_first():
    prompt = cpr.build_layered_prompt(
        context="CONTEXT_MARKER project rules",
        file_source="SOURCE_MARKER def f(): ...",
        file_path="src/elspeth/web/foo.py",
        persona="PERSONA_MARKER act as a security architect",
        lens="security-architect",
    )
    i_ctx = prompt.index("CONTEXT_MARKER")
    i_src = prompt.index("SOURCE_MARKER")
    i_persona = prompt.index("PERSONA_MARKER")
    i_path = prompt.index("src/elspeth/web/foo.py")
    # stable (cacheable) content first, variable per-call content last
    assert i_ctx < i_src < i_persona < i_path
    # the focus path must NOT appear in the cacheable head (before persona)
    assert prompt[:i_persona].count("src/elspeth/web/foo.py") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py::test_layered_prompt_orders_stable_content_first -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_layered_prompt'`

- [ ] **Step 3: Write minimal implementation**

```python
def build_layered_prompt(
    *,
    context: str,
    file_source: str,
    file_path: str,
    persona: str,
    lens: str,
) -> str:
    """Layer the prompt most-shared -> least-shared to ENABLE the codex prompt
    cache to key on [context][source] across every lens of a file (necessary, not
    sufficient — actual cross-process reuse is verified empirically in Task 9). The
    per-call tail (file_path, lens) is the only variable content and comes last."""
    return (
        f"{context}\n\n"
        "=== TARGET FILE SOURCE (inlined; you MAY read any other repo file via the "
        "read-only sandbox for investigation) ===\n"
        f"```\n{file_source}\n```\n\n"
        "=== REVIEW LENS ===\n"
        f"{persona}\n\n"
        "Output: a JSON object matching the provided schema — a non-empty "
        "`markdown_report` narration string and a `findings` array. Every finding "
        f'MUST set `lens` to "{lens}" and `priority` to one of P0|P1|P2|P3. For '
        "categories bug/correctness/security/smell/easy-win, `evidence` MUST cite a "
        "real path and integer line. The schema is STRICT: every finding MUST "
        "include ALL fields — for any field you have no value for (confidence, "
        "effort, impact, suggested_fix, target_file, gate_note, or an evidence "
        "`line`), emit JSON `null`; do NOT omit the key. If you find nothing, "
        "return an empty findings array and say so in markdown_report.\n"
        f"--- review target: {file_path} · lens: {lens} ---\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py::test_layered_prompt_orders_stable_content_first -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/codex_panel_review.py tests/unit/scripts/test_codex_panel_review.py
git commit --no-verify -m "feat(panel): layered stable-prefix-first prompt builder"
```

---

### Task 4: `load_persona` and `route_lenses`

**Files:**
- Modify: `scripts/codex_panel_review.py`
- Test: `tests/unit/scripts/test_codex_panel_review.py`

**Interfaces:**
- Consumes: `LENSES_DIR` (Task 1).
- Produces:
  - `load_persona(lens: str, *, lenses_dir: Path = LENSES_DIR) -> str` — reads `<lenses_dir>/<lens>.md`; raises `FileNotFoundError` if absent.
  - `route_lenses(file_path: Path, *, override: list[str] | None = None) -> list[str]` — returns the applicable lens names for a file; `override` (from `--lenses`) wins verbatim. Plan 1 roster: `solution-architect` + `security-architect` for every file; `python-engineer` only if defined and the file is `*.py` (kept minimal here; full roster is Plan 2).

- [ ] **Step 1: Write the failing test**

```python
def test_load_persona_reads_and_errors(tmp_path):
    (tmp_path / "demo.md").write_text("PERSONA BODY", encoding="utf-8")
    assert cpr.load_persona("demo", lenses_dir=tmp_path) == "PERSONA BODY"
    import pytest
    with pytest.raises(FileNotFoundError):
        cpr.load_persona("missing", lenses_dir=tmp_path)


def test_route_lenses_default_and_override():
    py = Path("src/elspeth/web/foo.py")
    routed = cpr.route_lenses(py)
    assert "solution-architect" in routed and "security-architect" in routed
    # explicit override is returned verbatim
    assert cpr.route_lenses(py, override=["security-architect"]) == ["security-architect"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py -k "load_persona or route_lenses" -v`
Expected: FAIL — attributes do not exist yet

- [ ] **Step 3: Write minimal implementation**

```python
# Plan 1 roster (file-predicate pairs). Full routing table is Plan 2.
_EVERY_FILE = ("solution-architect", "security-architect")


def load_persona(lens: str, *, lenses_dir: Path = LENSES_DIR) -> str:
    path = lenses_dir / f"{lens}.md"
    if not path.exists():
        raise FileNotFoundError(f"persona prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def route_lenses(file_path: Path, *, override: list[str] | None = None) -> list[str]:
    if override:
        return list(override)
    lenses = list(_EVERY_FILE)
    if file_path.suffix == ".py" and (LENSES_DIR / "python-engineer.md").exists():
        lenses.append("python-engineer")
    return lenses
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py -k "load_persona or route_lenses" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/codex_panel_review.py tests/unit/scripts/test_codex_panel_review.py
git commit --no-verify -m "feat(panel): persona loader + minimal lens routing"
```

---

### Task 5: Persona prompt files (validated by running, not by tests)

**Files:**
- Create: `scripts/codex_lenses/security-architect.md`
- Create: `scripts/codex_lenses/solution-architect.md`

**Interfaces:** none (these are LLM prompts, not code). Per project doctrine there is **no unit test asserting their text**; they are validated in the Task 9 pilot by reading output for signal/noise.

- [ ] **Step 1: Write `security-architect.md`**

```markdown
You are a principal security architect reviewing a single source file as part of
a pre-1.0 sandblasting pass. You have read-only access to the whole repository.

Focus:
- Trust-boundary handling of external input (web request bodies, query params,
  headers, file contents, LLM/tool output, DB rows). Find the boundary, not the sink.
- Injection (SQL/command/template/path), SSRF, unsafe deserialization, secret
  handling, authz/authn gaps, unsafe defaults, and data egress of sensitive fields.
- Prefer concrete, located findings. For every bug/security finding, cite the
  exact path and line in `evidence`.

For each issue, emit one finding with: category (usually `security`, or `bug`/
`correctness` when appropriate), priority P0–P3, confidence, effort, a one-line
`impact` (why it matters), a `summary`, `evidence[]` with path+line, and a
`suggested_fix`. If the file is clean from this lens, return no findings and say
so in `markdown_report`.
```

- [ ] **Step 2: Write `solution-architect.md`**

```markdown
You are a principal solution architect reviewing a single source file as part of
a pre-1.0 sandblasting pass. You have read-only access to the whole repository.

Focus:
- Responsibility/cohesion: does this file do one thing? Coupling to neighbours,
  leaky abstractions, misplaced logic, and boundary violations.
- Code smell, duplication, and inefficiency that raise the cost of change.
- Improvement opportunities and **easy wins** (high impact, low effort) we are
  leaving on the table — but an easy win you cannot point at is not easy, so cite
  a path+line for any `easy-win`.

For each issue, emit one finding with: category (`design`/`smell`/`efficiency`/
`improvement`/`easy-win`), priority P0–P3, confidence, an honest `effort`, a
one-line `impact`, a `summary`, `evidence[]`, and a `suggested_fix`. Sort your
own narration by impact-per-effort. If nothing is worth raising, return no
findings and say so in `markdown_report`.
```

- [ ] **Step 3: Verify the files load via the loader**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import codex_panel_review as c; print(len(c.load_persona('security-architect')), len(c.load_persona('solution-architect')))"`
Expected: two non-zero lengths printed (e.g. `742 808`)

- [ ] **Step 4: Commit**

```bash
git add scripts/codex_lenses/security-architect.md scripts/codex_lenses/solution-architect.md
git commit --no-verify -m "feat(panel): security-architect + solution-architect personas"
```

---

### Task 6: `apply_panel_evidence_gate` — category-aware, structured-first, rewrites sidecar last

**Files:**
- Modify: `scripts/codex_panel_review.py`
- Test: `tests/unit/scripts/test_codex_panel_review.py`

**Interfaces:**
- Consumes: `RELAXED_CATEGORIES`, `ANCHOR_REQUIRED_CATEGORIES` (the gate body dispatches on these two; `STRICT_CATEGORIES` is consumed only transitively, at `ANCHOR_REQUIRED_CATEGORIES`'s definition in Task 1).
- Produces: `apply_panel_evidence_gate(sidecar_path: Path, *, lens: str) -> int` — returns the number of findings downgraded. Stamps each finding's `lens` from the (known-correct) caller value rather than trusting the model's self-report; is **fail-closed** (any priority-bearing category not covered by the anchor/relaxed arms — e.g. `design` or a future enum value — is downgraded unless it carries a non-empty `impact` rationale); mutates the sidecar JSON in place; and **always rewrites it** (so its mtime exceeds the per-lens `.md`).

- [ ] **Step 1: Write the failing test**

```python
def _write_sidecar(tmp_path, findings):
    import json
    md = tmp_path / "src__x.md"
    md.write_text("narration\n", encoding="utf-8")
    sidecar = tmp_path / "src__x.md.structured.json"
    sidecar.write_text(json.dumps({"markdown_report": "n", "findings": findings}), encoding="utf-8")
    return md, sidecar


def test_gate_strict_downgrades_without_line(tmp_path):
    import json
    _, sidecar = _write_sidecar(tmp_path, [
        {"priority": "P1", "lens": "security-architect", "category": "security",
         "summary": "s", "evidence": [{"path": "src/x.py", "claim": "no line"}]},
    ])
    n = cpr.apply_panel_evidence_gate(sidecar, lens="security-architect")
    assert n == 1
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P3"


def test_gate_relaxed_keeps_lineless_improvement(tmp_path):
    import json
    _, sidecar = _write_sidecar(tmp_path, [
        {"priority": "P2", "lens": "solution-architect", "category": "improvement",
         "impact": "removes a whole retry class", "summary": "add retry abstraction",
         "evidence": []},
    ])
    n = cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect")
    assert n == 0
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P2"


def test_gate_easy_win_requires_anchor(tmp_path):
    import json
    _, sidecar = _write_sidecar(tmp_path, [
        {"priority": "P2", "lens": "solution-architect", "category": "easy-win",
         "impact": "tiny", "summary": "rename for clarity", "evidence": []},
    ])
    assert cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect") == 1
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P3"


def test_gate_design_without_impact_downgrades(tmp_path):
    # fail-closed: `design` is in no category set, so an unsubstantiated design
    # finding (no anchor, no impact) is downgraded rather than riding through.
    import json
    _, sidecar = _write_sidecar(tmp_path, [
        {"priority": "P0", "lens": "solution-architect", "category": "design",
         "summary": "module splits two responsibilities", "evidence": []},
    ])
    assert cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect") == 1
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P3"


def test_gate_design_with_impact_kept(tmp_path):
    import json
    _, sidecar = _write_sidecar(tmp_path, [
        {"priority": "P1", "lens": "solution-architect", "category": "design",
         "impact": "two responsibilities inflate every change's blast radius",
         "summary": "split module", "evidence": []},
    ])
    assert cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect") == 0
    assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P1"


def test_gate_stamps_lens_over_model_value(tmp_path):
    # the gate trusts the caller's lens, never the model-supplied field.
    import json
    _, sidecar = _write_sidecar(tmp_path, [
        {"priority": "P2", "lens": "WRONG-model-value", "category": "improvement",
         "impact": "real", "summary": "s", "evidence": []},
    ])
    cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect")
    assert json.loads(sidecar.read_text())["findings"][0]["lens"] == "solution-architect"


def test_gate_no_category_rides_through_ungated(tmp_path):
    # drift guard: with neither a file:line anchor nor an impact rationale, NO
    # category in the schema enum survives at its original priority — a future
    # enum value with no gate handling would fail here.
    import json
    schema = json.loads(cpr.PANEL_FINDING_SCHEMA.read_text(encoding="utf-8"))
    enum = schema["properties"]["findings"]["items"]["properties"]["category"]["enum"]
    for category in [c for c in enum if c is not None]:
        _, sidecar = _write_sidecar(tmp_path, [
            {"priority": "P0", "lens": "solution-architect", "category": category,
             "summary": "s", "evidence": []},
        ])
        assert cpr.apply_panel_evidence_gate(sidecar, lens="solution-architect") == 1, category
        assert json.loads(sidecar.read_text())["findings"][0]["priority"] == "P3"


def test_gate_rewrites_sidecar_last_so_not_stale(tmp_path):
    # Reproduces the mtime trap: run_codex_once writes the .md AFTER the sidecar,
    # so the fresh sidecar looks stale until the gate rewrites it.
    import json
    import time
    try:
        from codex_audit_common import _structured_findings, structured_output_path_for_report
    except ModuleNotFoundError:
        from scripts.codex_audit_common import _structured_findings, structured_output_path_for_report

    md = tmp_path / "src__y.md"
    sidecar = structured_output_path_for_report(md)
    sidecar.write_text(json.dumps({"markdown_report": "n", "findings": [
        {"priority": "P1", "lens": "security-architect", "category": "security",
         "summary": "s", "evidence": [{"path": "src/y.py", "line": 5, "claim": "c"}]}]}), encoding="utf-8")
    time.sleep(0.02)
    md.write_text("narration\n", encoding="utf-8")  # .md now newer -> sidecar looks stale
    assert _structured_findings(md) is None  # trap confirmed
    cpr.apply_panel_evidence_gate(sidecar, lens="security-architect")  # rewrites sidecar last
    findings = _structured_findings(md)
    assert findings is not None and findings[0]["priority"] == "P1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py -k gate -v`
Expected: FAIL — `apply_panel_evidence_gate` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
import json


def _has_line_anchor(finding: dict) -> bool:
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return False
    return any(
        isinstance(e, dict) and isinstance(e.get("line"), int) and bool(e.get("path"))
        for e in evidence
    )


def _has_impact(finding: dict) -> bool:
    # `impact` is nullable in the strict schema, so finding.get("impact") may be
    # None. Coerce with `or ""` BEFORE strip() — str(None) == "None" reads truthy,
    # which would silently defeat the relaxed/fail-closed downgrade.
    return bool(str(finding.get("impact") or "").strip())


def apply_panel_evidence_gate(sidecar_path: Path, *, lens: str) -> int:
    """Category-aware, fail-CLOSED structured gate. Downgrades anchor-required
    findings lacking a path+line, and any other priority-bearing finding lacking
    an `impact` rationale (relaxed AND uncovered categories like `design` or a
    typo'd/future enum value — nothing rides through ungated). Stamps each
    finding's `lens` from the known-correct caller value (never trusts the model's
    self-reported lens). Always rewrites the sidecar so its mtime exceeds the
    per-lens .md (defeats the staleness guard at codex_audit_common.py:820).
    Returns the downgrade count."""
    raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"panel sidecar must be a JSON object: {sidecar_path}")
    findings = raw.get("findings")
    if not isinstance(findings, list):
        raise RuntimeError(f"panel sidecar missing findings array: {sidecar_path}")

    downgraded = 0
    for finding in findings:
        if not isinstance(finding, dict):
            raise RuntimeError(f"panel finding must be an object: {sidecar_path}")
        finding["lens"] = lens  # deterministic stamp; do not trust the model's value
        category = finding.get("category")
        if category in ANCHOR_REQUIRED_CATEGORIES and not _has_line_anchor(finding):
            finding["priority"] = "P3"
            finding["confidence"] = "low"
            finding["gate_note"] = "needs verification: no file:line anchor"
            downgraded += 1
        elif category in RELAXED_CATEGORIES and not _has_impact(finding):
            finding["priority"] = "P3"
            finding["gate_note"] = "needs rationale: empty impact"
            downgraded += 1
        elif (
            category not in ANCHOR_REQUIRED_CATEGORIES
            and category not in RELAXED_CATEGORIES
            and not _has_impact(finding)
        ):
            # Fail-closed: `design` and any unknown/typo/future category need at
            # least an `impact` rationale, or they are downgraded — no silent
            # pass-through of an unsubstantiated high-priority claim.
            finding["priority"] = "P3"
            finding["gate_note"] = f"uncovered category {category!r}: needs impact rationale"
            downgraded += 1

    # Unconditional rewrite (last) — see docstring.
    sidecar_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return downgraded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py -k gate -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/codex_panel_review.py tests/unit/scripts/test_codex_panel_review.py
git commit --no-verify -m "feat(panel): category-aware structured evidence gate + mtime-trap guard"
```

---

### Task 7: `run_file_lenses` — serial per-lens runner over one file (stock `run_codex_once`)

**Files:**
- Modify: `scripts/codex_panel_review.py`
- Test: `tests/unit/scripts/test_codex_panel_review.py`

**Interfaces:**
- Consumes: `run_codex_once` (stock), `build_layered_prompt`, `load_persona`, `route_lenses`, `apply_panel_evidence_gate`, `PANEL_FINDING_SCHEMA`, rate limiter, `append_log`.
- Produces: `async run_file_lenses(*, file_path: Path, lenses: list[str], output_dir: Path, repo_root: Path, context: str, model: str | None, reasoning_effort: str | None, rate_limiter, log_path: Path, log_lock) -> dict[str, int]`. Returns aggregate stats: summed `input_tokens`, `cached_input_tokens`, `output_tokens`, plus `gated` and `failed` counts. Per-lens report path: `output_dir/<relpath>.<lens>.md`; sidecar via `structured_output_path_for_report`.

- [ ] **Step 1: Write the failing test**

```python
import asyncio


def test_run_file_lenses_loops_gates_and_aggregates(tmp_path, monkeypatch):
    import json
    try:
        from codex_audit_common import structured_output_path_for_report
    except ModuleNotFoundError:
        from scripts.codex_audit_common import structured_output_path_for_report

    # run_file_lenses reads the target's source, so z.py must exist on disk.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "z.py").write_text("def z(): ...", encoding="utf-8")

    calls = []

    async def fake_run_codex_once(**kwargs):
        # Simulate codex: write the structured sidecar then the .md (engine order).
        calls.append(kwargs["output_path"])
        out = kwargs["output_path"]
        sidecar = structured_output_path_for_report(out)
        sidecar.write_text(json.dumps({"markdown_report": "n", "findings": [
            {"priority": "P1", "lens": "x", "category": "security", "summary": "s",
             "evidence": [{"path": "src/z.py", "line": 1, "claim": "c"}]}]}), encoding="utf-8")
        out.write_text("narration\n", encoding="utf-8")
        return {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 10, "total_tokens": 110}

    monkeypatch.setattr(cpr, "run_codex_once", fake_run_codex_once)

    stats = asyncio.run(cpr.run_file_lenses(
        file_path=tmp_path / "src" / "z.py",
        lenses=["security-architect", "solution-architect"],
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        context="CTX",
        model=None,
        reasoning_effort=None,
        rate_limiter=None,
        log_path=tmp_path / "log.md",
        log_lock=asyncio.Lock(),
    ))
    assert len(calls) == 2                       # one call per lens, serial
    assert stats["cached_input_tokens"] == 120   # 60 * 2 aggregated
    assert stats["failed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py -k run_file_lenses -v`
Expected: FAIL — `run_file_lenses` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
import time


async def run_file_lenses(
    *,
    file_path: Path,
    lenses: list[str],
    output_dir: Path,
    repo_root: Path,
    context: str,
    model: str | None,
    reasoning_effort: str | None,
    rate_limiter,
    log_path: Path,
    log_lock,
) -> dict[str, int]:
    """Run a file's lenses SERIALLY (file-major), so lens 2..N reuse lens 1's
    warm [context][source] prompt-cache prefix. Stock run_codex_once; the panel
    gate runs after each lens."""
    file_source = file_path.read_text(encoding="utf-8")
    relative = file_path.relative_to(repo_root)
    agg = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "gated": 0, "failed": 0}

    for lens in lenses:
        output_path = (output_dir / relative).with_suffix((output_dir / relative).suffix + f".{lens}.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        persona = load_persona(lens)
        prompt = build_layered_prompt(
            context=context, file_source=file_source,
            file_path=str(relative.as_posix()), persona=persona, lens=lens,
        )
        start = time.monotonic()
        status, note = "ok", ""
        try:
            if rate_limiter is not None:
                # public method on AsyncRequestRateLimiter (codex_audit_common.py:274);
                # avoids importing the private _await_rate_limiter, which has no
                # dual-import shim and would ModuleNotFoundError under pytest.
                await rate_limiter.acquire()
            usage = await run_codex_once(
                file_path=file_path,
                output_path=output_path,
                model=model,
                prompt=prompt,
                repo_root=repo_root,
                file_display=str(relative.as_posix()),
                output_display=str(output_path.relative_to(repo_root).as_posix()) if output_path.is_relative_to(repo_root) else str(output_path),
                output_schema=PANEL_FINDING_SCHEMA,
                structured_markdown_field="markdown_report",
                reasoning_effort=reasoning_effort,
            )
            gated = apply_panel_evidence_gate(structured_output_path_for_report(output_path), lens=lens)
            agg["gated"] += gated
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
                agg[key] += int(usage.get(key, 0))
            note = f"gated={gated}; cached={usage.get('cached_input_tokens', 0)}"
        except Exception as exc:  # capture-and-continue so the pilot completes
            status, note = "failed", str(exc)[:200]
            agg["failed"] += 1
        finally:
            await append_log(
                log_path=log_path, log_lock=log_lock, timestamp=utc_now(),
                status=status, file_display=str(relative.as_posix()),
                output_display=f"lens={lens}", model=model or "default",
                duration_s=time.monotonic() - start, note=note,
            )
    return agg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py -k run_file_lenses -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/codex_panel_review.py tests/unit/scripts/test_codex_panel_review.py
git commit --no-verify -m "feat(panel): serial per-lens file runner (stock run_codex_once + gate)"
```

---

### Task 8: `main` — single-file CLI + `--dry-run`

**Files:**
- Modify: `scripts/codex_panel_review.py`
- Test: `tests/unit/scripts/test_codex_panel_review.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `build_arg_parser() -> argparse.ArgumentParser` and `main(argv: list[str] | None = None) -> int`. Plan 1 flags: `--file` (required), `--lenses` (comma-sep override), `--model`, `--reasoning-effort`, `--output-dir` (default `DEFAULT_OUTPUT_DIR`), `--context-files` (nargs +), `--rate-limit` (int), `--dry-run`.

- [ ] **Step 1: Write the failing test**

```python
def test_dry_run_prints_lens_plan(tmp_path, capsys, monkeypatch):
    target = tmp_path / "src" / "elspeth" / "web" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("def foo(): ...", encoding="utf-8")
    monkeypatch.setattr(cpr, "REPO_ROOT", tmp_path)
    rc = cpr.main(["--file", str(target), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "security-architect" in out and "solution-architect" in out
    assert "foo.py" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py -k dry_run -v`
Expected: FAIL — `main` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
import argparse
import asyncio
import shutil
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Codex panel review (foundation: single file).")
    p.add_argument("--file", required=True, help="Source file to review.")
    p.add_argument("--lenses", default=None, help="Comma-separated lens override.")
    p.add_argument("--model", default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--context-files", nargs="+", default=None)
    p.add_argument("--rate-limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"file not found: {file_path}", file=sys.stderr)
        return 1
    override = [s.strip() for s in args.lenses.split(",")] if args.lenses else None
    lenses = route_lenses(file_path, override=override)

    if args.dry_run:
        print(f"Would review {file_path.name} through {len(lenses)} lenses:")
        for lens in lenses:
            print(f"  - {lens}")
        return 0

    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    log_path = output_dir / "PANEL_LOG.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_log_file(log_path, header_title="Codex Panel Review Log")
    context = load_context(REPO_ROOT, extra_files=args.context_files, include_skills=True)
    rate_limiter = make_codex_rate_limiter(args.rate_limit)

    stats = asyncio.run(run_file_lenses(
        file_path=file_path, lenses=lenses, output_dir=output_dir, repo_root=REPO_ROOT,
        context=context, model=args.model, reasoning_effort=args.reasoning_effort,
        rate_limiter=rate_limiter, log_path=log_path, log_lock=asyncio.Lock(),
    ))
    cached = stats["cached_input_tokens"]
    total_in = stats["input_tokens"]
    rate = (cached / total_in * 100) if total_in else 0.0
    print(f"lenses={len(lenses)} gated={stats['gated']} failed={stats['failed']} "
          f"cache_hit={rate:.1f}% (cached_input={cached}/{total_in})")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py -k dry_run -v`
Expected: PASS

- [ ] **Step 5: Run the full test file + the existing-scripts-unaffected smoke**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_codex_panel_review.py tests/unit/scripts/test_codex_audit_common.py -v`
Expected: PASS (new file green; `test_codex_audit_common.py` unchanged and still green — proves the additive work didn't disturb shared behaviour)

- [ ] **Step 6: Commit**

```bash
git add scripts/codex_panel_review.py tests/unit/scripts/test_codex_panel_review.py
git commit --no-verify -m "feat(panel): single-file CLI with --dry-run + cache-hit reporting"
```

---

### Task 9: Run the caching / cost / throughput pilot (real codex calls)

**Files:**
- Create: `docs/quality-audit/PANEL_PILOT.md`

**Interfaces:** none. This task **calls the real `codex` CLI** and records numbers; it is a measurement task, not TDD. It needs the codex CLI on PATH and a working account.

- [ ] **Step 1: Schema-acceptance smoke (one cheap call — fail fast on strict-mode rejection)**

Before any measurement, confirm `codex` actually *accepts* `panel_finding.schema.json`. Strict structured-outputs (`additionalProperties:false`) rejects any schema with optional properties; a rejection is an `HTTP 400 invalid_json_schema` that `run_codex_once` turns into a `RuntimeError` (`codex_audit_common.py:658`), which `run_file_lenses` then swallows via capture-and-continue — so a non-compliant schema fails *every* lens silently.
Run on a small file, one lens: `.venv/bin/python scripts/codex_panel_review.py --file scripts/codex_panel_review.py --lenses security-architect`
Expected: `failed=0`, with a non-empty per-lens `.md` + `.structured.json` sidecar under `findings-panel-raw/`. If you see `failed=1` with `codex exec failed ... code 1` / `invalid_json_schema` in `PANEL_LOG.md`, the schema is not strict-compliant — fix Task 2 before going further.

- [ ] **Step 2: One-file pilot (warm-cache check)**

Pick one representative file from the driving subsystem (the web UX rewrite):
Run: `.venv/bin/python scripts/codex_panel_review.py --file src/elspeth/web/composer/recipes.py`
Observe the printed `cache_hit=…%`. Read `docs/quality-audit/findings-panel-raw/PANEL_LOG.md` for per-lens durations and `cached=` notes, and the per-lens `…<lens>.md.usage.json` sidecars for each lens's exact `input_tokens` + `cached_input_tokens` (the aggregate stdout blends the cold lens 1 with the warm lens 2..N, so the clean per-lens hit-rate lives only in those sidecars).

- [ ] **Step 3: Record the numbers in `docs/quality-audit/PANEL_PILOT.md`**

Capture, with the actual observed values:
- per-lens call durations (min/median/max) vs the ~5–10 min cache TTL;
- `cached_input_tokens` / `input_tokens` hit-rate for lens 2..N of the file (lens 1 is the cold warm-up);
- whether the second lens actually hit the `[context][source]` prefix.

- [ ] **Step 4: Isolate the marginal `[source]` cache layer (what uniquely justifies file-major-serial)**

`load_context` inlines the same `AGENTS.md`+`CLAUDE.md`+skills for every file, so the `[context]` prefix is file-*independent* — it would be cached even under file-parallel execution. Only `[source]` is reused *because* a file's lenses run serially. Measure its marginal contribution: run one extra single-lens call on the same file as a **separate** invocation and record its `cached_input_tokens` as the `[context]`-only baseline; then `[source]` marginal ≈ (lens-2 combined cached − context-only baseline). Record both — they decide whether file-major-serial earns its keep (Step 6).

- [ ] **Step 5: ~5-file pilot + workers ceiling**

Run the single-file command across ~5 files (a quick shell loop is fine) and, separately, probe the concurrency ceiling by launching a few concurrent single-file runs; the first sustained 429s mark the `--workers` cap. Record both.

- [ ] **Step 6: Write the decision branches into `PANEL_PILOT.md`**

Per the spec's decision branch:
1. **TTL / cost:** if median per-call duration exceeds the ~5–10 min cache TTL, compute the no-cache one-time cost projection for a full tree (~1,800 calls) and a subsection (e.g. `src/elspeth/web`), and state go/no-go against budget.
2. **`[source]` marginal (from Step 4):** if `[source]` marginal savings are small relative to total cache savings, record that Plan 2's file-major-serial ordering is **not** justified by caching and that file-parallel should be the Plan 2 default (still harvesting `[context]`-level caching); if `[source]` is a large share, file-major-serial is confirmed.

These numbers become Plan 2's `--workers` default, execution-ordering choice, and cost prose.

- [ ] **Step 7: Commit**

```bash
git add docs/quality-audit/PANEL_PILOT.md
git commit --no-verify -m "docs(panel): caching/cost/throughput pilot results + workers ceiling"
```

---

## Self-Review

**1. Spec coverage (foundation slice only — streaming/worker-pool/merge/synthesis/outputs/dashboard are Plan 2):**
- Layered stable-prefix-first prompt + inlined source → Task 3. ✅
- Finding schema as a true superset (keeps `priority`; `markdown_report` for the engine; **OpenAI strict-mode compliant** — every prop in `required`, optionals as nullable unions) → Task 2. ✅
- Category-aware, **fail-closed** structured gate (anchor/relaxed/uncovered + deterministic `lens` stamp) reusing only the evidence primitive; markdown gate untouched → Task 6. ✅
- Per-lens isolation (own sidecar; no by-index coupling) → Tasks 6–7. ✅
- Serial file-major execution for cache reuse → Task 7. ✅
- `--lenses` override + minimal routing → Tasks 4, 8. ✅
- Caching verified empirically (schema-acceptance smoke + `[source]`-layer isolation + decision branches + one-time cost) → Task 9. ✅
- Persona files validated by running, not tests → Task 5. ✅
- Existing-scripts-unaffected check → Task 8 Step 5. ✅
- **Deferred to Plan 2 (intentional):** streaming runner, worker pool, completion-sentinel resume, commit pinning, `--path`/`--since`/`--files` scope selection, cross-lens merge, synthesis, counted/raw output layout, `write_panel_findings_index`, global rollup, dashboard, full retry. Listed in "Plan 2 boundary notes."

**2. Placeholder scan:** every code step contains runnable code; every command has expected output. Task 9 is explicitly a measurement task (no code), which is legitimate (it records empirical numbers the spec mandates before a full run).

**3. Type consistency:** `apply_panel_evidence_gate(sidecar_path: Path, *, lens: str) -> int` defined in Task 6, called by the runner in Task 7 with `lens=lens`. `run_file_lenses(...) -> dict[str,int]` returns the keys Task 8 reads (`cached_input_tokens`, `input_tokens`, `gated`, `failed`). `route_lenses`/`load_persona` signatures match across Tasks 4, 7, 8. `build_layered_prompt` keyword args match between Tasks 3 and 7. Schema field names (`priority`, `markdown_report`, `evidence[].line`) consistent across Tasks 2, 6, 7.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-codex-panel-review-foundation.md`. After this plan and the pilot, write Plan 2 (orchestration) using the pilot's measured numbers.
