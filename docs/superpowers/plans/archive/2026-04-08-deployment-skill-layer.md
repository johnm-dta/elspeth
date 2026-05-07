# Deployment-Specific Skill Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread the existing `data_dir` through the composer service to `build_messages`, so deployment-specific skill overlays in `data/skills/pipeline_composer.md` are loaded and appended to the system prompt.

**Architecture:** The infrastructure is already built: `load_deployment_skill()` in `skills/__init__.py` and `build_system_prompt(data_dir)` in `prompts.py`. What remains is wiring `data_dir` from the service into `_build_messages`, creating the `data/skills/` directory with a template file, writing tests, and adding a `.gitignore` so deployment skills stay out of version control.

**Tech Stack:** Python, pytest, markdown

**Filigree issue:** `elspeth-d6ffab874a`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/elspeth/web/composer/service.py` | Modify (line 347) | Pass `self._data_dir` to `build_messages` |
| `tests/unit/web/composer/test_prompts.py` | Modify | Add tests for `build_system_prompt`, `load_deployment_skill`, and `data_dir` parameter on `build_messages` |
| `tests/unit/web/composer/test_skills_loader.py` | Create | Focused tests for `load_skill` and `load_deployment_skill` |
| `data/skills/.gitignore` | Create | Keep deployment skills out of version control (except the template) |
| `data/skills/pipeline_composer.md.example` | Create | Template showing the deployment skill format |

---

### Task 1: Thread `data_dir` through `_build_messages` in the service

**Files:**
- Modify: `src/elspeth/web/composer/service.py:335-352`

- [ ] **Step 1: Update `_build_messages` to pass `data_dir`**

In `src/elspeth/web/composer/service.py`, update the `_build_messages` method to pass `self._data_dir`:

```python
    def _build_messages(
        self,
        chat_history: list[dict[str, Any]],
        state: CompositionState,
        user_message: str,
    ) -> list[dict[str, Any]]:
        """Build the message list. Returns a NEW list on every call.

        This is critical: the tool-use loop appends to this list during
        iteration. Returning a cached reference would cause cross-turn
        contamination.
        """
        return build_messages(
            chat_history=chat_history,
            state=state,
            user_message=user_message,
            catalog=self._catalog,
            data_dir=self._data_dir,
        )
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py tests/unit/web/composer/test_prompts.py -x -q`
Expected: All 28 tests pass (no functional change yet — `data_dir` points to `/data` which has no `skills/` subdir, so `load_deployment_skill` returns `""` and behaviour is identical).

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/composer/service.py
git commit -m "feat(composer): thread data_dir through _build_messages for deployment skill overlay"
```

---

### Task 2: Write tests for the skill loader functions

**Files:**
- Create: `tests/unit/web/composer/test_skills_loader.py`

- [ ] **Step 1: Write tests for `load_skill` and `load_deployment_skill`**

```python
"""Tests for skill pack loading — core and deployment layers.

Verifies:
- load_skill loads core skills from the package directory
- load_skill raises FileNotFoundError for missing skills
- load_deployment_skill returns "" when data_dir is None
- load_deployment_skill returns "" when skill file does not exist
- load_deployment_skill returns content when skill file exists
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.web.composer.skills import load_deployment_skill, load_skill


class TestLoadSkill:
    """Core skill loading from package directory."""

    def test_loads_existing_skill(self) -> None:
        """The pipeline_composer skill exists and loads as non-empty string."""
        content = load_skill("pipeline_composer")
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Pipeline Composer" in content

    def test_missing_skill_raises_file_not_found(self) -> None:
        """Requesting a non-existent skill must crash — not return empty."""
        with pytest.raises(FileNotFoundError):
            load_skill("nonexistent_skill_that_does_not_exist")


class TestLoadDeploymentSkill:
    """Deployment skill overlay loading."""

    def test_returns_empty_when_data_dir_is_none(self) -> None:
        """None data_dir means no deployment layer — return empty string."""
        assert load_deployment_skill("pipeline_composer", None) == ""

    def test_returns_empty_when_data_dir_has_no_skills_dir(self, tmp_path: Path) -> None:
        """data_dir exists but has no skills/ subdirectory."""
        assert load_deployment_skill("pipeline_composer", tmp_path) == ""

    def test_returns_empty_when_skill_file_missing(self, tmp_path: Path) -> None:
        """data_dir has skills/ but the specific skill file is absent."""
        (tmp_path / "skills").mkdir()
        assert load_deployment_skill("pipeline_composer", tmp_path) == ""

    def test_returns_content_when_skill_file_exists(self, tmp_path: Path) -> None:
        """Deployment skill file exists — return its content."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_file = skills_dir / "pipeline_composer.md"
        skill_file.write_text("# Deployment Skill\n\nCustom provider info here.\n")

        result = load_deployment_skill("pipeline_composer", tmp_path)
        assert result == "# Deployment Skill\n\nCustom provider info here.\n"

    def test_accepts_string_data_dir(self, tmp_path: Path) -> None:
        """data_dir can be a string path, not just a Path object."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "pipeline_composer.md").write_text("content")

        result = load_deployment_skill("pipeline_composer", str(tmp_path))
        assert result == "content"
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_skills_loader.py -v`
Expected: 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_skills_loader.py
git commit -m "test(composer): add tests for load_skill and load_deployment_skill"
```

---

### Task 3: Write tests for `build_system_prompt` and `build_messages` with `data_dir`

**Files:**
- Modify: `tests/unit/web/composer/test_prompts.py`

- [ ] **Step 1: Add test class for `build_system_prompt`**

Add the following to the end of `tests/unit/web/composer/test_prompts.py`:

```python
from elspeth.web.composer.prompts import build_system_prompt


class TestBuildSystemPrompt:
    """System prompt composition with optional deployment layer."""

    def test_no_data_dir_returns_core_skill_only(self) -> None:
        """Without data_dir, returns the core skill unchanged."""
        result = build_system_prompt(None)
        assert result == SYSTEM_PROMPT

    def test_missing_deployment_skill_returns_core_only(self, tmp_path: Path) -> None:
        """data_dir with no skills/ subdir returns core skill only."""
        result = build_system_prompt(str(tmp_path))
        assert result == SYSTEM_PROMPT

    def test_deployment_skill_appended_after_separator(self, tmp_path: Path) -> None:
        """Deployment skill content is appended after a separator."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "pipeline_composer.md").write_text("# Our Custom Providers\n\nUse ACME_API_KEY.\n")

        result = build_system_prompt(str(tmp_path))

        assert result.startswith(SYSTEM_PROMPT)
        assert "\n\n---\n\n" in result
        assert "# Our Custom Providers" in result
        assert "ACME_API_KEY" in result


class TestBuildMessagesWithDataDir:
    """build_messages with deployment skill overlay."""

    def test_data_dir_none_uses_core_prompt(self) -> None:
        """Default (no data_dir) uses core SYSTEM_PROMPT."""
        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages([], state, "test", catalog, data_dir=None)
        system_content = messages[0]["content"]

        assert SYSTEM_PROMPT in system_content

    def test_data_dir_with_deployment_skill_injects_it(self, tmp_path: Path) -> None:
        """When data_dir has a deployment skill, it appears in the system message."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "pipeline_composer.md").write_text("# Deployment: use ACME provider\n")

        state = _empty_state()
        catalog = _stub_catalog()

        messages = build_messages([], state, "test", catalog, data_dir=str(tmp_path))
        system_content = messages[0]["content"]

        assert "# Deployment: use ACME provider" in system_content
        assert SYSTEM_PROMPT in system_content
```

Also add `from pathlib import Path` to the imports at the top of the file if not already present.

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_prompts.py -v`
Expected: All tests pass (original 8 + new 5 = 13).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_prompts.py
git commit -m "test(composer): add tests for build_system_prompt and build_messages with data_dir"
```

---

### Task 4: Create `data/skills/` directory with template and `.gitignore`

**Files:**
- Create: `data/skills/.gitignore`
- Create: `data/skills/pipeline_composer.md.example`

- [ ] **Step 1: Create the `.gitignore`**

Create `data/skills/.gitignore`:

```
# Deployment-specific skill overlays are NOT version-controlled.
# They contain company/environment-specific knowledge (providers,
# secret naming, domain patterns) that varies per deployment.
#
# To create a deployment skill:
#   cp pipeline_composer.md.example pipeline_composer.md
#   Edit pipeline_composer.md with your deployment-specific content
#
# The .example file IS tracked — it's the template.

*
!.gitignore
!*.example
```

- [ ] **Step 2: Create the example template**

Create `data/skills/pipeline_composer.md.example`:

```markdown
# Deployment-Specific Pipeline Composer Knowledge

This file is loaded after the core pipeline composer skill and appended
to the LLM agent's system prompt. Use it to inject knowledge specific
to your deployment — providers, models, secrets, patterns, vocabulary.

Delete sections you don't need. The core skill covers all system-level
knowledge; this file is for your organisation's specifics.

## Available Providers and Models

<!-- List the LLM providers and models configured in this deployment. -->

| Provider | Secret Name | Available Models | Default |
|----------|-------------|------------------|---------|
| openrouter | `OPENROUTER_API_KEY` | `anthropic/claude-3.5-sonnet`, `openai/gpt-4o` | `anthropic/claude-3.5-sonnet` |

## Secret Naming Conventions

<!-- How are secrets named in this deployment? -->

All secrets follow the pattern `SERVICE_API_KEY` (e.g., `OPENROUTER_API_KEY`,
`AZURE_STORAGE_KEY`). Check `list_secret_refs` to see what's configured.

## Common Patterns for This Organisation

<!-- Add organisation-specific pipeline patterns here. -->
<!-- Follow the format from the core skill's Common Pipeline Patterns section. -->

## Domain Vocabulary

<!-- If your users use domain-specific terms, map them here. -->

| User says | Means |
|-----------|-------|
| "case file" | JSON input with case metadata |
| "evidence report" | JSON sink output |

## Default Preferences

<!-- Organisation-wide defaults the agent should apply. -->

- Default output format: JSON with indent 2
- Default LLM temperature: 0.0
- Default error handling: route to quarantine sink
```

- [ ] **Step 3: Commit**

```bash
git add data/skills/.gitignore data/skills/pipeline_composer.md.example
git commit -m "feat(composer): add deployment skill directory with template and gitignore"
```

---

### Task 5: Verify full integration end-to-end

- [ ] **Step 1: Run all composer tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ -v`
Expected: All tests pass.

- [ ] **Step 2: Run type checker on modified files**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/prompts.py src/elspeth/web/composer/service.py src/elspeth/web/composer/skills/__init__.py --no-error-summary`
Expected: No type errors.

- [ ] **Step 3: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: No new violations (changes are all within L3 application layer).

- [ ] **Step 4: Verify deployment skill template loads correctly**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path
from elspeth.web.composer.skills import load_deployment_skill

# Template file should NOT load (it's .md.example, not .md)
result = load_deployment_skill('pipeline_composer', 'data')
assert result == '', f'Expected empty, got {len(result)} chars'
print('PASS: .example file correctly ignored')

# Simulate deployment by copying example to .md
import shutil, tempfile
with tempfile.TemporaryDirectory() as tmp:
    skills = Path(tmp) / 'skills'
    skills.mkdir()
    shutil.copy('data/skills/pipeline_composer.md.example', skills / 'pipeline_composer.md')
    result = load_deployment_skill('pipeline_composer', tmp)
    assert len(result) > 0
    assert 'Deployment-Specific' in result
    print(f'PASS: deployment skill loaded ({len(result)} chars)')
"
```
Expected: Both assertions pass.

- [ ] **Step 5: Final commit (if any fixups needed)**

If any fixups were needed, commit them:
```bash
git add -A
git commit -m "fix(composer): deployment skill layer fixups from integration verification"
```
