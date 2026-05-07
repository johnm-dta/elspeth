from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import codex_audit_common  # type: ignore[import-not-found]  # noqa: E402
import codex_bug_hunt  # type: ignore[import-not-found]  # noqa: E402


def _minimal_report() -> str:
    return """## Summary

No concrete bug found in the target file

## Evidence

src/elspeth/example.py:1 shows no defect.

## Severity

- Severity: trivial
- Priority: P3

## Root Cause Hypothesis

No bug identified.
"""


def _bug_report(*, summary: str = "Real bug", priority: str = "P1") -> str:
    return f"""## Summary

{summary}

## Evidence

src/elspeth/example.py:12 shows the issue.

## Severity

- Severity: major
- Priority: {priority}

## Root Cause Hypothesis

The target file violates the contract.
"""


def test_build_prompt_keeps_target_data_after_static_cacheable_prefix(tmp_path: Path) -> None:
    first_file = tmp_path / "src" / "elspeth" / "core" / "first.py"
    second_file = tmp_path / "src" / "elspeth" / "core" / "second.py"

    first_prompt = codex_bug_hunt._build_prompt(
        first_file,
        template="BUG TEMPLATE",
        context="SHARED REPOSITORY CONTEXT",
        extra_message="per-run migration note",
        target_context="static prepass for first",
        allowlist_context="per-file allowlist note",
    )
    second_prompt = codex_bug_hunt._build_prompt(
        second_file,
        template="BUG TEMPLATE",
        context="SHARED REPOSITORY CONTEXT",
        extra_message="per-run migration note",
        target_context="static prepass for second",
        allowlist_context="per-file allowlist note",
    )

    first_prefix = first_prompt.split("Target-specific context:", 1)[0]
    second_prefix = second_prompt.split("Target-specific context:", 1)[0]

    assert first_prefix == second_prefix
    assert "Target file:" not in first_prefix
    assert "per-run migration note" in first_prefix
    assert "static prepass for first" not in first_prefix
    assert "per-file allowlist note" not in first_prefix
    assert first_prompt.index("Repository context (read-only):") < first_prompt.index("Target-specific context:")
    assert "static prepass for first" in first_prompt.split("Target-specific context:", 1)[1]


def test_load_tier_allowlist_reads_every_module_yaml(tmp_path: Path) -> None:
    allowlist_dir = tmp_path / "config" / "cicd" / "enforce_tier_model"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "_defaults.yaml").write_text("defaults:\n  fail_on_stale: true\n", encoding="utf-8")
    (allowlist_dir / "contracts.yaml").write_text(
        """
per_file_rules:
- pattern: contracts/schema.py
  rules: [R1]
  reason: contract rule
""",
        encoding="utf-8",
    )
    (allowlist_dir / "core.yaml").write_text(
        """
per_file_rules:
- pattern: core/config.py
  rules: [R5]
  reason: core rule
""",
        encoding="utf-8",
    )

    rules = codex_bug_hunt._load_tier_allowlist(tmp_path)

    assert {rule["pattern"] for rule in rules} == {"contracts/schema.py", "core/config.py"}


def test_allowlist_entries_match_globs_relative_to_package_root(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "elspeth"
    target_file = package_root / "core" / "security" / "web.py"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("", encoding="utf-8")

    context = codex_bug_hunt._allowlist_entries_for_file(
        target_file,
        package_root,
        [
            {
                "pattern": "core/security/*",
                "rules": ["R1"],
                "reason": "security boundary rule",
                "max_hits": 2,
            }
        ],
    )

    assert context is not None
    assert "Rules R1" in context
    assert "security boundary rule" in context
    assert "max_hits=2" in context


@pytest.mark.asyncio
async def test_run_codex_once_passes_output_schema_and_extracts_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "report.md"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    captured_cmd: list[str] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            del input
            raw_path = Path(captured_cmd[captured_cmd.index("--output-last-message") + 1])
            raw_path.write_text('{"markdown_report": "## Summary\\n\\nNo concrete bug found"}', encoding="utf-8")
            return b"", b""

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> FakeProcess:
        del kwargs
        captured_cmd.extend(cmd)
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    await codex_audit_common.run_codex_once(
        file_path=tmp_path / "target.py",
        output_path=output_path,
        model="gpt-5.5",
        prompt="review this file",
        repo_root=tmp_path,
        file_display="target.py",
        output_display="report.md",
        output_schema=schema_path,
        structured_markdown_field="markdown_report",
    )

    assert "--output-schema" in captured_cmd
    assert captured_cmd[captured_cmd.index("--output-schema") + 1] == str(schema_path)
    assert output_path.read_text(encoding="utf-8") == "## Summary\n\nNo concrete bug found\n"


@pytest.mark.asyncio
async def test_run_codex_once_uses_stdin_jsonl_and_writes_usage_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "report.md"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    captured_cmd: list[str] = []
    captured_input: bytes | None = None

    class FakeProcess:
        returncode = 0

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            nonlocal captured_input
            captured_input = input
            raw_path = Path(captured_cmd[captured_cmd.index("--output-last-message") + 1])
            raw_path.write_text(json.dumps({"markdown_report": _minimal_report()}), encoding="utf-8")
            stdout = "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {
                                "input_tokens": 2000,
                                "cached_input_tokens": 1500,
                                "output_tokens": 250,
                            },
                        }
                    ),
                ]
            ).encode()
            return stdout, b""

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> FakeProcess:
        assert kwargs["stdin"] == asyncio.subprocess.PIPE
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        captured_cmd.extend(cmd)
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    usage = await codex_audit_common.run_codex_once(
        file_path=tmp_path / "target.py",
        output_path=output_path,
        model=None,
        prompt="review this file",
        repo_root=tmp_path,
        file_display="target.py",
        output_display="report.md",
        output_schema=schema_path,
        structured_markdown_field="markdown_report",
    )

    assert "--json" in captured_cmd
    assert captured_cmd[-1] == "-"
    assert captured_input == b"review this file"
    assert usage["input_tokens"] == 2000
    assert usage["cached_input_tokens"] == 1500
    assert usage["output_tokens"] == 250

    usage_path = output_path.with_suffix(output_path.suffix + ".usage.json")
    usage_json = json.loads(usage_path.read_text(encoding="utf-8"))
    assert usage_json["cached_input_tokens"] == 1500


@pytest.mark.asyncio
async def test_run_codex_once_exposes_runtime_controls_and_workspace_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "report.md"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    captured_cmd: list[str] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            del input
            raw_path = Path(captured_cmd[captured_cmd.index("--output-last-message") + 1])
            raw_path.write_text(json.dumps({"markdown_report": _minimal_report()}), encoding="utf-8")
            return b"", b""

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> FakeProcess:
        del kwargs
        captured_cmd.extend(cmd)
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    await codex_audit_common.run_codex_once(
        file_path=tmp_path / "target.py",
        output_path=output_path,
        model="gpt-5.5",
        prompt="review this file",
        repo_root=tmp_path,
        file_display="target.py",
        output_display="report.md",
        output_schema=schema_path,
        structured_markdown_field="markdown_report",
        profile="bug-hunt",
        reasoning_effort="xhigh",
        service_tier="fast",
        ephemeral=True,
        timeout_s=5.0,
    )

    assert "--cd" in captured_cmd
    assert captured_cmd[captured_cmd.index("--cd") + 1] == str(tmp_path)
    assert "--profile" in captured_cmd
    assert captured_cmd[captured_cmd.index("--profile") + 1] == "bug-hunt"
    assert "--ephemeral" in captured_cmd
    assert 'model_reasoning_effort="xhigh"' in captured_cmd
    assert 'service_tier="fast"' in captured_cmd


@pytest.mark.asyncio
async def test_run_codex_once_times_out_and_cleans_up_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "report.md"
    terminated = False
    waited = False

    class FakeProcess:
        returncode: int | None = None

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            del input
            await asyncio.sleep(60)
            return b"", b""

        def terminate(self) -> None:
            nonlocal terminated
            terminated = True
            self.returncode = -15

        async def wait(self) -> None:
            nonlocal waited
            waited = True

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> FakeProcess:
        del cmd, kwargs
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(asyncio.TimeoutError):
        await codex_audit_common.run_codex_once(
            file_path=tmp_path / "target.py",
            output_path=output_path,
            model=None,
            prompt="review this file",
            repo_root=tmp_path,
            file_display="target.py",
            output_display="report.md",
            timeout_s=0.01,
        )

    assert terminated
    assert waited


@pytest.mark.asyncio
async def test_codex_rate_limiter_backpressures_instead_of_failing_at_quota() -> None:
    now = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return now

    async def sleep(delay_s: float) -> None:
        nonlocal now
        sleeps.append(delay_s)
        now += delay_s

    limiter = codex_audit_common.AsyncRequestRateLimiter(
        max_calls=1,
        period_s=0.25,
        clock=clock,
        sleep=sleep,
    )

    await limiter.acquire()
    await limiter.acquire()

    assert sleeps == [0.25]


@pytest.mark.asyncio
async def test_run_codex_with_retry_awaits_rate_limiter_and_returns_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "report.md"
    log_path = tmp_path / "CODEX_LOG.md"
    codex_audit_common.ensure_log_file(log_path, header_title="Test Log")
    acquired: list[str] = []

    class FakeLimiter:
        async def try_acquire_async(self, name: str) -> bool:
            acquired.append(name)
            return True

    async def fake_run_codex_once(**kwargs: Any) -> dict[str, int]:
        Path(kwargs["output_path"]).write_text(_minimal_report(), encoding="utf-8")
        return {
            "input_tokens": 1000,
            "cached_input_tokens": 750,
            "output_tokens": 125,
            "total_tokens": 1125,
        }

    monkeypatch.setattr(codex_audit_common, "run_codex_once", fake_run_codex_once)

    result = await codex_audit_common.run_codex_with_retry_and_logging(
        file_path=tmp_path / "target.py",
        output_path=output_path,
        model=None,
        prompt="review",
        repo_root=tmp_path,
        log_path=log_path,
        log_lock=asyncio.Lock(),
        file_display="target.py",
        output_display="report.md",
        rate_limiter=FakeLimiter(),
    )

    assert acquired == ["codex_api"]
    assert result["input_tokens"] == 1000
    assert result["cached_input_tokens"] == 750
    assert "cached_input_tokens=750" in log_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_run_codex_with_retry_rate_limits_each_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "report.md"
    log_path = tmp_path / "CODEX_LOG.md"
    codex_audit_common.ensure_log_file(log_path, header_title="Test Log")
    acquired: list[str] = []
    attempts = 0

    class FakeLimiter:
        async def try_acquire_async(self, name: str) -> bool:
            acquired.append(name)
            return True

    async def fake_run_codex_once(**kwargs: Any) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        Path(kwargs["output_path"]).write_text(_minimal_report(), encoding="utf-8")
        return {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(codex_audit_common, "run_codex_once", fake_run_codex_once)
    monkeypatch.setattr(codex_audit_common, "RETRY_MULTIPLIER", 0)
    monkeypatch.setattr(codex_audit_common, "RETRY_MIN_WAIT_S", 0)
    monkeypatch.setattr(codex_audit_common, "RETRY_MAX_WAIT_S", 0)

    result = await codex_audit_common.run_codex_with_retry_and_logging(
        file_path=tmp_path / "target.py",
        output_path=output_path,
        model=None,
        prompt="review",
        repo_root=tmp_path,
        log_path=log_path,
        log_lock=asyncio.Lock(),
        file_display="target.py",
        output_display="report.md",
        rate_limiter=FakeLimiter(),
    )

    assert attempts == 2
    assert acquired == ["codex_api", "codex_api"]
    assert result["input_tokens"] == 1


def test_load_context_prefers_agents_skills_over_legacy_claude_skills(tmp_path: Path) -> None:
    skill_name = "tier-model-deep-dive"
    agents_skill = tmp_path / ".agents" / "skills" / skill_name / "SKILL.md"
    claude_skill = tmp_path / ".claude" / "skills" / skill_name / "SKILL.md"
    agents_skill.parent.mkdir(parents=True)
    claude_skill.parent.mkdir(parents=True)
    agents_skill.write_text("agents skill body", encoding="utf-8")
    claude_skill.write_text("claude skill body", encoding="utf-8")

    context = codex_audit_common.load_context(tmp_path, include_skills=True)

    assert "agents skill body" in context
    assert "claude skill body" not in context


def test_load_context_includes_agents_before_legacy_claude(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("codex project instructions", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("legacy project instructions", encoding="utf-8")

    context = codex_audit_common.load_context(tmp_path)

    assert "--- AGENTS.md ---" in context
    assert "--- CLAUDE.md ---" in context
    assert context.index("--- AGENTS.md ---") < context.index("--- CLAUDE.md ---")


def test_bug_hunt_defaults_to_structured_output_schema(tmp_path: Path) -> None:
    repo_root = tmp_path
    default_schema = repo_root / "scripts" / "schemas" / "codex_bug_hunt_report.schema.json"
    default_schema.parent.mkdir(parents=True)
    default_schema.write_text("{}", encoding="utf-8")

    assert (
        codex_bug_hunt._resolve_structured_output_schema(
            repo_root,
            structured_output=True,
            structured_output_schema=None,
            no_structured_output=False,
        )
        == default_schema
    )
    assert (
        codex_bug_hunt._resolve_structured_output_schema(
            repo_root,
            structured_output=False,
            structured_output_schema=None,
            no_structured_output=True,
        )
        is None
    )


def test_bug_hunt_schema_exposes_machine_readable_findings() -> None:
    schema_path = Path(__file__).resolve().parents[3] / "scripts" / "schemas" / "codex_bug_hunt_report.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert "findings" in schema["required"]
    finding_schema = schema["properties"]["findings"]["items"]
    assert {"target_file", "primary_fix_file", "evidence", "confidence"} <= set(finding_schema["required"])


def test_static_prepass_context_extracts_target_specific_leads(tmp_path: Path) -> None:
    repo_root = tmp_path
    package_root = repo_root / "src" / "elspeth"
    target = package_root / "core" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        """
import logging
from elspeth.core.config import Settings

logger = logging.getLogger(__name__)

def run(row):
    try:
        return row.get("amount")
    except Exception:
        return None
""",
        encoding="utf-8",
    )

    context = codex_bug_hunt._build_static_prepass_context(target, repo_root=repo_root, root_dir=package_root)

    assert "Static Pre-pass Context" in context
    assert "from elspeth.core.config import Settings" in context
    assert "row.get" in context
    assert "except Exception" in context
    assert "logging.getLogger" in context


def test_subsystem_context_maps_public_api_imports_and_adjacent_tests(tmp_path: Path) -> None:
    repo_root = tmp_path
    root_dir = repo_root / "src" / "elspeth" / "core"
    root_dir.mkdir(parents=True)
    (root_dir / "alpha.py").write_text(
        """
from elspeth.core.beta import Beta

class Alpha:
    pass

def build_alpha():
    return Beta()
""",
        encoding="utf-8",
    )
    (root_dir / "beta.py").write_text("class Beta:\n    pass\n", encoding="utf-8")
    test_path = repo_root / "tests" / "unit" / "core" / "test_alpha.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("from elspeth.core.alpha import Alpha\n", encoding="utf-8")

    context = codex_bug_hunt._build_subsystem_context(root_dir, repo_root=repo_root)

    assert "Subsystem Static Map" in context
    assert "src/elspeth/core/alpha.py" in context
    assert "class Alpha" in context
    assert "def build_alpha" in context
    assert "from elspeth.core.beta import Beta" in context
    assert "tests/unit/core/test_alpha.py" in context


@pytest.mark.asyncio
async def test_run_batches_warms_cache_with_first_file_before_parallel_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    root_dir = repo_root / "src" / "elspeth"
    files = [root_dir / f"target_{index}.py" for index in range(3)]
    for file_path in files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("value = 1\n", encoding="utf-8")

    active = 0
    warmed = False
    started: list[str] = []

    async def fake_run_codex_with_retry_and_logging(**kwargs: Any) -> dict[str, int]:
        nonlocal active, warmed
        file_path = Path(kwargs["file_path"])
        started.append(file_path.name)
        if file_path == files[0]:
            assert active == 0
        else:
            assert warmed
        active += 1
        await asyncio.sleep(0)
        Path(kwargs["output_path"]).write_text(_minimal_report(), encoding="utf-8")
        active -= 1
        if file_path == files[0]:
            warmed = True
        return {"gated": 0, "input_tokens": 10, "cached_input_tokens": 5, "output_tokens": 1, "total_tokens": 11}

    monkeypatch.setattr(codex_bug_hunt, "run_codex_with_retry_and_logging", fake_run_codex_with_retry_and_logging)

    stats = await codex_bug_hunt._run_batches(
        files=files,
        output_dir=repo_root / "docs" / "bugs" / "generated",
        model=None,
        prompt_template="BUG TEMPLATE",
        repo_root=repo_root,
        skip_existing=False,
        batch_size=3,
        root_dir=root_dir,
        log_path=repo_root / "docs" / "bugs" / "process" / "CODEX_LOG.md",
        context="CONTEXT",
        rate_limit=None,
        organize_by_priority=False,
        bugs_open_dir=None,
        deduplicate=False,
        tier_allowlist=[],
        allowlist_root=root_dir,
        structured_output_schema=None,
        warm_up_cache=True,
    )

    assert started[0] == "target_0.py"
    assert stats["input_tokens"] == 30
    assert stats["cached_input_tokens"] == 15


def test_generate_summary_and_index_ignore_generated_metadata_and_priority_copies(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    (output_dir / "core.md").write_text(_bug_report(summary="Primary report", priority="P1"), encoding="utf-8")
    (output_dir / "clean.md").write_text(_minimal_report(), encoding="utf-8")
    (output_dir / "RUN_METADATA.md").write_text("# Metadata\n", encoding="utf-8")
    (output_dir / "SUMMARY.md").write_text("# Summary\n", encoding="utf-8")
    copied = output_dir / "by-priority" / "P1" / "core.md"
    copied.parent.mkdir(parents=True)
    copied.write_text(_bug_report(summary="Copied report", priority="P1"), encoding="utf-8")

    stats = codex_audit_common.generate_summary(output_dir, no_defect_marker="No concrete bug found")
    codex_audit_common.write_findings_index(
        output_dir=output_dir,
        repo_root=tmp_path,
        title="Index",
        file_column_label="Source File",
        no_defect_marker="No concrete bug found",
        clean_section_title="Clean Files",
    )

    assert stats["P1"] == 1
    assert stats["no_defect"] == 1
    assert "unknown" not in stats
    index = (output_dir / "FINDINGS_INDEX.md").read_text(encoding="utf-8")
    assert "Primary report" in index
    assert "Copied report" not in index
    assert "RUN_METADATA" not in index


def test_organize_by_priority_rebuilds_priority_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    stale = output_dir / "by-priority" / "P1" / "stale.md"
    stale.parent.mkdir(parents=True)
    stale.write_text(_bug_report(summary="Stale report", priority="P1"), encoding="utf-8")
    (output_dir / "new.md").write_text(_bug_report(summary="New report", priority="P2"), encoding="utf-8")

    codex_bug_hunt._organize_by_priority(output_dir)

    assert not stale.exists()
    assert (output_dir / "by-priority" / "P2" / "new.md").exists()


def test_structured_findings_drive_summary_and_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    report = output_dir / "core.md"
    report.write_text(_bug_report(summary="Markdown fallback priority", priority="P3"), encoding="utf-8")
    report.with_suffix(report.suffix + ".structured.json").write_text(
        json.dumps(
            {
                "markdown_report": report.read_text(encoding="utf-8"),
                "findings": [
                    {
                        "target_file": "src/elspeth/core.py",
                        "primary_fix_file": "src/elspeth/core.py",
                        "summary": "Structured finding wins",
                        "severity": "major",
                        "priority": "P1",
                        "confidence": "high",
                        "evidence": [{"path": "src/elspeth/core.py", "line": 12, "claim": "contract drift"}],
                        "no_defect_reason": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stats = codex_audit_common.generate_summary(output_dir, no_defect_marker="No concrete bug found")
    codex_audit_common.write_findings_index(
        output_dir=output_dir,
        repo_root=tmp_path,
        title="Index",
        file_column_label="Source File",
        no_defect_marker="No concrete bug found",
        clean_section_title="Clean Files",
    )

    assert stats["P1"] == 1
    assert "P3" not in stats
    index = (output_dir / "FINDINGS_INDEX.md").read_text(encoding="utf-8")
    assert "Structured finding wins" in index
    assert "high" in index
    assert "src/elspeth/core.py:12" in index


def test_evidence_gate_downgrades_structured_findings_with_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    report = output_dir / "core.md"
    report.write_text(
        """## Summary

Insufficiently supported claim

## Evidence

This is suspicious but has no file or line citation.

## Severity

- Severity: critical
- Priority: P0

## Root Cause Hypothesis

Unknown.
""",
        encoding="utf-8",
    )
    report.with_suffix(report.suffix + ".structured.json").write_text(
        json.dumps(
            {
                "markdown_report": report.read_text(encoding="utf-8"),
                "findings": [
                    {
                        "target_file": "src/elspeth/core.py",
                        "primary_fix_file": "src/elspeth/core.py",
                        "summary": "Insufficiently supported claim",
                        "severity": "critical",
                        "priority": "P0",
                        "confidence": "low",
                        "evidence": [{"path": "src/elspeth/core.py", "line": None, "claim": "No concrete citation"}],
                        "no_defect_reason": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    gated = codex_audit_common.apply_evidence_gate(report)

    assert gated == 1
    structured = json.loads(report.with_suffix(report.suffix + ".structured.json").read_text(encoding="utf-8"))
    assert structured["findings"][0]["priority"] == "P3"
    assert structured["findings"][0]["severity"] == "trivial"
    assert codex_audit_common.generate_summary(output_dir, no_defect_marker="No concrete bug found")["P3"] == 1


def test_write_run_metadata_includes_extra_parameters(tmp_path: Path) -> None:
    codex_audit_common.write_run_metadata(
        output_dir=tmp_path,
        repo_root=tmp_path,
        start_time="2026-04-29T00:00:00+00:00",
        end_time="2026-04-29T00:01:00+00:00",
        duration_s=60.0,
        files_scanned=3,
        model=None,
        batch_size=2,
        rate_limit=15,
        git_commit="abc123",
        title="Bug Hunt Run Metadata",
        script_name="scripts/codex_bug_hunt.py",
        extra_parameters={
            "Codex CLI": "codex-cli 0.125.0",
            "Output Mode": "structured-json-to-markdown",
        },
    )

    metadata = (tmp_path / "RUN_METADATA.md").read_text(encoding="utf-8")

    assert "- **Codex CLI:** codex-cli 0.125.0" in metadata
    assert "- **Output Mode:** structured-json-to-markdown" in metadata
