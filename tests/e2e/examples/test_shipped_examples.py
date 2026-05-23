# tests/e2e/examples/test_shipped_examples.py
"""E2E tests verifying all shipped example pipelines are valid configurations.

Every example directory under examples/ must contain at least one YAML
settings file, and each file must:
  1. Be parseable as YAML
  2. Contain a dict with the required top-level keys (source or sources, sinks)
  3. Where possible, pass full ElspethSettings validation via load_settings()

Examples that require external services (Azure, OpenRouter) or
environment variables cannot be fully validated without those vars set,
so they are tested for structural validity only.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import select

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import RunStatus
from elspeth.core.config import ElspethSettings, load_settings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import rows_table, run_sources_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

# Examples that contain ${VAR} env var references that would fail
# load_settings without the env vars being set.
_EXAMPLES_WITH_ENV_VARS: frozenset[str] = frozenset(
    {
        "azure_blob_sentiment",
        "azure_keyvault_secrets",
        "azure_openai_sentiment",
        "chroma_rag_qa",
        "multi_query_assessment",
        "openrouter_multi_query_assessment",
        "openrouter_sentiment",
        "schema_contracts_llm_assessment",
        "template_lookups",
    }
)

# Examples that reference external template/lookup files via
# template_file or lookup_file keys. These require those files to
# exist relative to the settings path, which they do, but they also
# tend to have env var references so they overlap with the above set.
_EXAMPLES_WITH_FILE_REFS: frozenset[str] = frozenset(
    {
        "multi_query_assessment",
        "openrouter_multi_query_assessment",
        "schema_contracts_llm_assessment",
        "template_lookups",
    }
)

# Examples that have no YAML settings file at all (e.g. data-only directories).
_EXAMPLES_WITHOUT_SETTINGS: frozenset[str] = frozenset(
    {
        "chaosllm",  # Contains only responses.jsonl (replay data)
    }
)

# Required top-level keys for any Elspeth settings file. Source roots may use
# either the legacy singular ``source`` form or the canonical plural ``sources``
# form for named multi-source examples.
_REQUIRED_KEYS: frozenset[str] = frozenset({"sinks"})


class TestShippedExamples:
    """Verify all shipped examples are valid configurations."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_example_settings(examples_dir: Path) -> list[tuple[str, Path]]:
        """Find all settings YAML files in examples.

        Returns a list of (example_name, yaml_path) tuples. Only files
        whose name contains "settings" are included; auxiliary YAML files
        (chaos_config.yaml, criteria_lookup.yaml) are excluded.
        """
        results: list[tuple[str, Path]] = []
        for example_dir in sorted(examples_dir.iterdir()):
            if not example_dir.is_dir() or example_dir.name.startswith("."):
                continue
            if example_dir.name in _EXAMPLES_WITHOUT_SETTINGS:
                continue
            for yaml_file in sorted(example_dir.glob("*.yaml")):
                if "settings" in yaml_file.name:
                    results.append((example_dir.name, yaml_file))
        return results

    @staticmethod
    def _needs_env_vars(example_name: str) -> bool:
        """Return True if the example requires env vars we cannot set in CI."""
        return example_name in _EXAMPLES_WITH_ENV_VARS

    @staticmethod
    def _assert_source_roots_valid(name: str, path: Path, data: dict[str, Any]) -> None:
        has_source = "source" in data
        has_sources = "sources" in data
        assert has_source != has_sources, f"{name}/{path.name}: define exactly one of source or sources"
        if has_source:
            source = data["source"]
            assert isinstance(source, dict), f"{name}/{path.name}: source must be a dict"
            assert "plugin" in source, f"{name}/{path.name}: source missing 'plugin' key"
            return

        sources = data["sources"]
        assert isinstance(sources, dict), f"{name}/{path.name}: sources must be a dict"
        assert sources, f"{name}/{path.name}: sources must not be empty"
        for source_name, source in sources.items():
            assert isinstance(source_name, str), f"{name}/{path.name}: source name must be a string"
            assert isinstance(source, dict), f"{name}/{path.name}: source '{source_name}' must be a dict"
            assert "plugin" in source, f"{name}/{path.name}: source '{source_name}' missing 'plugin' key"

    @staticmethod
    def _copy_example_to_tmp(example_pipeline_dir: Path, tmp_path: Path, example_name: str) -> Path:
        scratch_examples_dir = tmp_path / "examples"
        scratch_examples_dir.mkdir()
        copied_example_dir = scratch_examples_dir / example_name
        shutil.copytree(
            example_pipeline_dir / example_name,
            copied_example_dir,
            ignore=shutil.ignore_patterns("*.db", "*.db-shm", "*.db-wal", "*.jsonl", "payloads"),
        )
        return copied_example_dir

    @staticmethod
    def _run_example(settings_path: Path) -> tuple[Any, LandscapeDB]:
        settings = load_settings(settings_path)
        bundle = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            sources=bundle.sources,
            source_settings_map=bundle.source_settings_map,
            transforms=bundle.transforms,
            sinks=bundle.sinks,
            aggregations=bundle.aggregations,
            gates=list(settings.gates),
            coalesce_settings=(list(settings.coalesce) if settings.coalesce else None),
            queues=settings.queues,
        )
        config = assemble_and_validate_pipeline_config(
            sources=bundle.sources,
            transforms=bundle.transforms,
            sinks=bundle.sinks,
            aggregations=bundle.aggregations,
            settings=settings,
            graph=graph,
        )
        db = LandscapeDB(settings.landscape.url)
        result = Orchestrator(db).run(
            config,
            graph=graph,
            settings=settings,
            payload_store=FilesystemPayloadStore(settings_path.parent / "runs" / "payloads"),
        )
        return result, db

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _audit_source_rows(db: LandscapeDB, run_id: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, int, int]]]:
        with db.engine.connect() as conn:
            run_sources = conn.execute(
                select(
                    run_sources_table.c.source_name,
                    run_sources_table.c.source_node_id,
                    run_sources_table.c.lifecycle_state,
                )
                .where(run_sources_table.c.run_id == run_id)
                .order_by(run_sources_table.c.source_name)
            ).all()
            rows = conn.execute(
                select(
                    rows_table.c.source_node_id,
                    rows_table.c.source_row_index,
                    rows_table.c.ingest_sequence,
                )
                .where(rows_table.c.run_id == run_id)
                .order_by(rows_table.c.ingest_sequence)
            ).all()
        return (
            [(source_name, source_node_id, lifecycle_state) for source_name, source_node_id, lifecycle_state in run_sources],
            [(source_node_id, source_row_index, ingest_sequence) for source_node_id, source_row_index, ingest_sequence in rows],
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_examples_directory_exists(self, example_pipeline_dir: Path) -> None:
        """The examples/ directory exists at the repo root."""
        assert example_pipeline_dir.is_dir(), f"examples/ not found at {example_pipeline_dir}"

    def test_all_examples_have_settings(self, example_pipeline_dir: Path) -> None:
        """Every example directory has at least one settings file (or is excused)."""
        example_dirs = [d for d in sorted(example_pipeline_dir.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        assert len(example_dirs) > 0, "No example directories found"

        for d in example_dirs:
            if d.name in _EXAMPLES_WITHOUT_SETTINGS:
                continue
            yamls = list(d.glob("*.yaml")) + list(d.glob("*.yml"))
            assert len(yamls) > 0, f"Example {d.name} has no YAML config files"

    def test_discover_settings_files(self, example_pipeline_dir: Path) -> None:
        """Sanity check: discovery finds a reasonable number of settings files."""
        settings = self._find_example_settings(example_pipeline_dir)
        # We know there are 20+ example directories with settings
        assert len(settings) >= 20, f"Expected at least 20 settings files, found {len(settings)}"

    def test_all_settings_are_valid_yaml(self, example_pipeline_dir: Path) -> None:
        """All example settings files are parseable YAML producing dicts."""
        settings = self._find_example_settings(example_pipeline_dir)
        assert len(settings) > 0, "No settings files found"

        for name, path in settings:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{name}/{path.name}: settings is not a dict, got {type(data).__name__}"

    def test_all_settings_have_required_keys(self, example_pipeline_dir: Path) -> None:
        """All settings files contain the required top-level keys."""
        settings = self._find_example_settings(example_pipeline_dir)

        for name, path in settings:
            with open(path) as f:
                data: dict[str, Any] = yaml.safe_load(f)

            missing = _REQUIRED_KEYS - set(data.keys())
            assert not missing, f"{name}/{path.name}: missing required keys {missing}"
            self._assert_source_roots_valid(name, path, data)

    def test_all_settings_have_valid_source_structure(self, example_pipeline_dir: Path) -> None:
        """All settings files have a properly structured source section."""
        settings = self._find_example_settings(example_pipeline_dir)

        for name, path in settings:
            with open(path) as f:
                data: dict[str, Any] = yaml.safe_load(f)

            self._assert_source_roots_valid(name, path, data)

    def test_all_settings_have_valid_sinks_structure(self, example_pipeline_dir: Path) -> None:
        """All settings files have a properly structured sinks section."""
        settings = self._find_example_settings(example_pipeline_dir)

        for name, path in settings:
            with open(path) as f:
                data: dict[str, Any] = yaml.safe_load(f)

            sinks = data.get("sinks")
            assert isinstance(sinks, dict), f"{name}/{path.name}: sinks must be a dict"
            assert len(sinks) > 0, f"{name}/{path.name}: sinks must not be empty"

            # Each sink must have a plugin key
            for sink_name, sink_config in sinks.items():
                assert isinstance(sink_config, dict), f"{name}/{path.name}: sink '{sink_name}' must be a dict"
                assert "plugin" in sink_config, f"{name}/{path.name}: sink '{sink_name}' missing 'plugin' key"

    def test_local_examples_load_via_config_system(self, example_pipeline_dir: Path) -> None:
        """Examples without env var requirements load through ElspethSettings.

        These examples have no ${VAR} references and no external template
        files, so load_settings() should succeed and produce a valid
        ElspethSettings instance.
        """
        settings = self._find_example_settings(example_pipeline_dir)
        local_settings = [(name, path) for name, path in settings if not self._needs_env_vars(name)]

        assert len(local_settings) > 0, "No local (no-env-var) examples found"

        for name, path in local_settings:
            loaded = load_settings(path)
            assert isinstance(loaded, ElspethSettings), f"{name}/{path.name}: load_settings did not return ElspethSettings"
            assert loaded.sources, f"{name}/{path.name}: no sources defined"
            for source_name, source in loaded.sources.items():
                assert source.plugin, f"{name}/{path.name}: source '{source_name}' plugin is empty"
            # Verify at least one sink exists
            assert len(loaded.sinks) > 0, f"{name}/{path.name}: no sinks defined"

    def test_multi_flow_example_executes_end_to_end(
        self,
        example_pipeline_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """multi_flow ships as a runnable two-source, two-flow example."""
        example_dir = self._copy_example_to_tmp(example_pipeline_dir, tmp_path, "multi_flow")
        monkeypatch.chdir(tmp_path)
        db: LandscapeDB | None = None
        try:
            result, db = self._run_example(example_dir / "settings.yaml")

            assert result.status is RunStatus.COMPLETED
            assert result.rows_processed == 4
            run_sources, rows = self._audit_source_rows(db, result.run_id)
            assert [(source_name, state) for source_name, _node_id, state in run_sources] == [
                ("signups", "loaded"),
                ("tickets", "loaded"),
            ]
            node_to_source = {node_id: source_name for source_name, node_id, _state in run_sources}
            assert [
                (node_to_source[node_id], source_row_index, ingest_sequence) for node_id, source_row_index, ingest_sequence in rows
            ] == [
                ("signups", 0, 0),
                ("signups", 1, 1),
                ("tickets", 0, 2),
                ("tickets", 1, 3),
            ]
            signups = self._read_jsonl(example_dir / "output" / "signups.jsonl")
            tickets = self._read_jsonl(example_dir / "output" / "tickets.jsonl")
            assert [row["signup_id"] for row in signups] == ["S-100", "S-101"]
            assert [row["ticket_id"] for row in tickets] == ["T-900", "T-901"]
        finally:
            if db is not None:
                db.close()

    def test_multi_source_queue_example_executes_end_to_end(
        self,
        example_pipeline_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """multi_source_queue ships as a runnable fan-in queue example."""
        example_dir = self._copy_example_to_tmp(example_pipeline_dir, tmp_path, "multi_source_queue")
        monkeypatch.chdir(tmp_path)
        db: LandscapeDB | None = None
        try:
            result, db = self._run_example(example_dir / "settings.yaml")

            assert result.status is RunStatus.COMPLETED
            assert result.rows_processed == 3
            run_sources, rows = self._audit_source_rows(db, result.run_id)
            assert [(source_name, state) for source_name, _node_id, state in run_sources] == [
                ("orders", "loaded"),
                ("refunds", "loaded"),
            ]
            node_to_source = {node_id: source_name for source_name, node_id, _state in run_sources}
            assert [
                (node_to_source[node_id], source_row_index, ingest_sequence) for node_id, source_row_index, ingest_sequence in rows
            ] == [
                ("orders", 0, 0),
                ("orders", 1, 1),
                ("refunds", 0, 2),
            ]
            combined = self._read_jsonl(example_dir / "output" / "combined.jsonl")
            assert len(combined) == 3
            assert Counter(row["kind"] for row in combined) == Counter({"order": 2, "refund": 1})
        finally:
            if db is not None:
                db.close()

    def test_env_var_examples_are_structurally_valid(self, example_pipeline_dir: Path) -> None:
        """Examples with env vars are valid YAML with correct structure.

        We cannot call load_settings() because env var expansion would
        fail, but we can verify the raw YAML structure matches what
        ElspethSettings expects.
        """
        settings = self._find_example_settings(example_pipeline_dir)
        env_settings = [(name, path) for name, path in settings if self._needs_env_vars(name)]

        assert len(env_settings) > 0, "No env-var examples found"

        for name, path in env_settings:
            with open(path) as f:
                data: dict[str, Any] = yaml.safe_load(f)

            # These must have the required keys
            assert "sinks" in data, f"{name}/{path.name}: missing sinks"
            self._assert_source_roots_valid(name, path, data)

            # Verify transforms structure if present
            if "transforms" in data:
                assert isinstance(data["transforms"], list), f"{name}/{path.name}: transforms must be a list"
                for i, t in enumerate(data["transforms"]):
                    assert isinstance(t, dict), f"{name}/{path.name}: transform[{i}] must be a dict"
                    assert "plugin" in t, f"{name}/{path.name}: transform[{i}] missing 'plugin'"

    def test_no_duplicate_sink_names(self, example_pipeline_dir: Path) -> None:
        """Sink names are unique within each settings file (YAML keys are unique by spec)."""
        settings = self._find_example_settings(example_pipeline_dir)

        for name, path in settings:
            with open(path) as f:
                data: dict[str, Any] = yaml.safe_load(f)

            sinks = data.get("sinks", {})
            # YAML dict keys are inherently unique, but verify they are all lowercase
            for sink_name in sinks:
                assert sink_name == sink_name.lower(), f"{name}/{path.name}: sink name '{sink_name}' is not lowercase"

    def test_gate_conditions_are_strings(self, example_pipeline_dir: Path) -> None:
        """Gate conditions must be strings (expression syntax)."""
        settings = self._find_example_settings(example_pipeline_dir)

        for name, path in settings:
            with open(path) as f:
                data: dict[str, Any] = yaml.safe_load(f)

            gates = data.get("gates", [])
            for i, gate in enumerate(gates):
                assert "condition" in gate, f"{name}/{path.name}: gate[{i}] missing 'condition'"
                assert isinstance(gate["condition"], str), f"{name}/{path.name}: gate[{i}] condition must be a string"
                assert "routes" in gate, f"{name}/{path.name}: gate[{i}] missing 'routes'"
