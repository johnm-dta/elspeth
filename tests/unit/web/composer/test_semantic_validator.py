"""Tests for validate_semantic_contracts algorithm."""

from __future__ import annotations

from elspeth.contracts.plugin_semantics import (
    ContentKind,
    SemanticOutcome,
    TextFraming,
)
from elspeth.web.composer._semantic_validator import validate_semantic_contracts
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)


def _wardline_state(*, text_separator: str = " ", scrape_format: str = "text"):
    """Build the canonical Wardline-shape composition: scrape -> explode -> sink.

    Required to satisfy real config validation:
    - csv source has schema (DataPluginConfig.schema_config is required)
    - web_scrape transform has schema, on_success, on_error
    - line_explode transform has schema, on_success, on_error
    Without these, plugin construction fails as a "draft config" error,
    the validator's tolerant probe path silently skips, and the test
    becomes vacuous (no error raised, but no contract emitted either).
    """
    return CompositionState(
        metadata=PipelineMetadata(name="wardline"),
        version=1,
        edges=(),
        source=SourceSpec(
            plugin="csv",
            on_success="scrape_in",
            options={
                "path": "data/url.csv",
                "schema": {"mode": "fixed", "fields": ["url: str"]},
            },
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="scrape",
                node_type="transform",
                plugin="web_scrape",
                input="scrape_in",
                on_success="explode_in",
                on_error="errors",
                options={
                    "schema": {"mode": "flexible", "fields": ["url: str"]},
                    "required_input_fields": ["url"],
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "fingerprint",
                    "format": scrape_format,
                    "text_separator": text_separator,
                    "http": {
                        "abuse_contact": "x@example.com",
                        "scraping_reason": "t",
                        "timeout": 5,
                        "allowed_hosts": "public_only",
                    },
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="explode",
                node_type="transform",
                plugin="line_explode",
                input="explode_in",
                on_success="sink",
                on_error="errors",
                options={
                    "schema": {"mode": "flexible", "fields": ["content: str"]},
                    "source_field": "content",
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        outputs=(
            OutputSpec(
                name="sink",
                plugin="json",
                options={"path": "out.json"},
                on_write_failure="discard",
            ),
            OutputSpec(
                name="errors",
                plugin="json",
                options={"path": "err.json"},
                on_write_failure="discard",
            ),
        ),
    )


class TestValidateSemanticContracts:
    def test_compact_text_produces_conflict(self):
        state = _wardline_state(text_separator=" ", scrape_format="text")
        errors, contracts = validate_semantic_contracts(state)

        assert len(errors) == 1
        error = errors[0]
        assert error.severity == "high"
        assert "line_explode" in error.message
        assert "web_scrape" in error.message
        assert "content" in error.message
        # Diagnostic must include the requirement_code so a UI / agent
        # can look up plugin-owned assistance.
        assert "line_explode.source_field.line_framed_text" in error.message
        # Generic diagnostic must NOT contain fix-prose tokens — those
        # belong in PluginAssistance, not the validator.
        assert "text_separator" not in error.message
        assert "use markdown" not in error.message.lower()
        assert "set " not in error.message.lower()  # imperative fix-language

        assert len(contracts) == 1
        contract = contracts[0]
        assert contract.from_id == "scrape"
        assert contract.to_id == "explode"
        assert contract.producer_field == "content"
        assert contract.consumer_field == "content"
        assert contract.outcome is SemanticOutcome.CONFLICT
        assert contract.requirement.requirement_code == "line_explode.source_field.line_framed_text"
        assert contract.producer_facts is not None
        assert contract.producer_facts.text_framing is TextFraming.COMPACT

    def test_newline_text_passes(self):
        state = _wardline_state(text_separator="\n", scrape_format="text")
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()
        assert len(contracts) == 1
        assert contracts[0].outcome is SemanticOutcome.SATISFIED
        assert contracts[0].producer_facts.text_framing is TextFraming.NEWLINE_FRAMED

    def test_markdown_passes(self):
        state = _wardline_state(scrape_format="markdown")
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()
        assert contracts[0].outcome is SemanticOutcome.SATISFIED
        assert contracts[0].producer_facts.content_kind is ContentKind.MARKDOWN

    def test_source_fed_consumer_emits_no_semantic_contract(self):
        # Phase 1 design: source -> transform edges are out of scope.
        # The validator skips them entirely (no contract, no error).
        # If this test fails, Phase 1 has accidentally re-enabled
        # source-fed semantic checking.
        state = CompositionState(
            metadata=PipelineMetadata(name="t"),
            version=1,
            edges=(),
            source=SourceSpec(
                plugin="csv",
                on_success="explode_in",
                options={
                    "path": "x.csv",
                    "schema": {"mode": "fixed", "fields": ["content: str"]},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="explode",
                    node_type="transform",
                    plugin="line_explode",
                    input="explode_in",
                    on_success="sink",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["content: str"]},
                        "source_field": "content",
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            outputs=(
                OutputSpec(
                    name="sink",
                    plugin="json",
                    options={"path": "out.json"},
                    on_write_failure="discard",
                ),
                OutputSpec(
                    name="errors",
                    plugin="json",
                    options={"path": "err.json"},
                    on_write_failure="discard",
                ),
            ),
        )
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()
        assert contracts == ()

    def test_undeclared_transform_producer_with_fail_policy_emits_error(self):
        # Real registered transform that does NOT declare output_semantics:
        # `passthrough` (src/elspeth/plugins/transforms/passthrough.py:39).
        # passthrough → line_explode is exactly the "pass-through degrades
        # to UNKNOWN" case the design decision documents.
        state = CompositionState(
            metadata=PipelineMetadata(name="t"),
            version=1,
            edges=(),
            source=SourceSpec(
                plugin="csv",
                on_success="pt_in",
                options={
                    "path": "x.csv",
                    "schema": {"mode": "fixed", "fields": ["content: str"]},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="pt",
                    node_type="transform",
                    plugin="passthrough",
                    input="pt_in",
                    on_success="explode_in",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["content: str"]},
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="explode",
                    node_type="transform",
                    plugin="line_explode",
                    input="explode_in",
                    on_success="sink",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["content: str"]},
                        "source_field": "content",
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            outputs=(
                OutputSpec(
                    name="sink",
                    plugin="json",
                    options={"path": "out.json"},
                    on_write_failure="discard",
                ),
                OutputSpec(
                    name="errors",
                    plugin="json",
                    options={"path": "err.json"},
                    on_write_failure="discard",
                ),
            ),
        )
        errors, contracts = validate_semantic_contracts(state)

        assert len(contracts) == 1
        assert contracts[0].outcome is SemanticOutcome.UNKNOWN
        assert contracts[0].consumer_plugin == "line_explode"
        assert contracts[0].producer_plugin == "passthrough"

        # FAIL policy → UNKNOWN producer fails.
        assert len(errors) == 1
        assert "no semantic facts" in errors[0].message.lower() or "undeclared" in errors[0].message.lower()
        assert errors[0].component == "node:explode"

    def test_gate_between_producer_and_consumer_is_traversed(self):
        # Gates are STRUCTURAL — plugin=None and condition/routes carry
        # the routing logic. Verified via tests/unit/web/composer/
        # test_yaml_generator.py:72 which uses the same shape.
        state = CompositionState(
            metadata=PipelineMetadata(name="t"),
            version=1,
            edges=(),
            source=SourceSpec(
                plugin="csv",
                on_success="src_in",
                options={
                    "path": "x.csv",
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="scrape",
                    node_type="transform",
                    plugin="web_scrape",
                    input="src_in",
                    on_success="gate_in",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["url: str"]},
                        "required_input_fields": ["url"],
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "fingerprint",
                        "format": "markdown",
                        "http": {
                            "abuse_contact": "x@example.com",
                            "scraping_reason": "t",
                            "timeout": 5,
                            "allowed_hosts": "public_only",
                        },
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="g",
                    node_type="gate",
                    plugin=None,
                    input="gate_in",
                    on_success=None,
                    on_error=None,
                    options={},
                    condition="row['content']",
                    routes={"yes": "explode_in", "no": "errors"},
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="explode",
                    node_type="transform",
                    plugin="line_explode",
                    input="explode_in",
                    on_success="sink",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["content: str"]},
                        "source_field": "content",
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            outputs=(
                OutputSpec(
                    name="sink",
                    plugin="json",
                    options={"path": "out.json"},
                    on_write_failure="discard",
                ),
                OutputSpec(
                    name="errors",
                    plugin="json",
                    options={"path": "err.json"},
                    on_write_failure="discard",
                ),
            ),
        )
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()  # markdown is line-compatible
        assert len(contracts) == 1
        assert contracts[0].outcome is SemanticOutcome.SATISFIED
        assert contracts[0].from_id == "scrape"  # walked through gate
        assert contracts[0].consumer_plugin == "line_explode"
        assert contracts[0].producer_plugin == "web_scrape"


class TestWardlineRegressionPin:
    """Exact options shape from the original Wardline regression YAML.

    Sourced from data/wardline_line_export_pipeline.yaml at the commit
    immediately before text_separator: '\\n' was added. If the new
    semantic-validator surface fails to block this shape, the original
    regression has recurred.

    The brief requires this test exercises ``state.validate()``, not just
    ``validate_semantic_contracts`` directly — a future refactor that
    detaches the validator from the wired surface must fail this test, not
    silently pass.
    """

    def _wardline_broken_yaml_state(self) -> CompositionState:
        # Options copied verbatim from the broken YAML revision.
        return CompositionState(
            metadata=PipelineMetadata(name="wardline-line-export"),
            version=1,
            edges=(),
            source=SourceSpec(
                plugin="csv",
                on_success="scrape_in",
                options={
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                    "path": "data/blobs/<uuid>/<uuid>_wardline_url.csv",
                    "on_validation_failure": "quarantine",
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="scrape_page",
                    node_type="transform",
                    plugin="web_scrape",
                    input="scrape_in",
                    on_success="explode_in",
                    on_error="errors",
                    options={
                        "schema": {
                            "mode": "flexible",
                            "fields": ["url: str"],
                        },
                        "required_input_fields": ["url"],
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "content_fingerprint",
                        "format": "text",
                        # text_separator OMITTED -> defaults to single space.
                        "fingerprint_mode": "content",
                        "strip_elements": ["script", "style"],
                        "http": {
                            "abuse_contact": "pipeline@example.com",
                            "scraping_reason": ("User requested Wardline contents exported as line-oriented JSON"),
                            "timeout": 30,
                            "allowed_hosts": "public_only",
                        },
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="split_lines",
                    node_type="transform",
                    plugin="line_explode",
                    input="explode_in",
                    on_success="sink",
                    on_error="errors",
                    options={
                        "schema": {
                            "mode": "flexible",
                            "fields": ["content: str"],
                        },
                        "source_field": "content",
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            outputs=(
                OutputSpec(
                    name="sink",
                    plugin="json",
                    options={"path": "out.json"},
                    on_write_failure="discard",
                ),
                OutputSpec(
                    name="errors",
                    plugin="json",
                    options={"path": "err.json"},
                    on_write_failure="discard",
                ),
            ),
        )

    def _wardline_fixed_yaml_state(self) -> CompositionState:
        """Same shape as the broken state but with the Wardline fix applied:
        text_separator: '\\n'. Used to assert the validator accepts the
        post-fix YAML so the test pins both directions of the regression.
        """
        broken = self._wardline_broken_yaml_state()
        # Mutate via with_node so we don't have to repeat the full options dict.
        scrape_options = dict(broken.nodes[0].options)
        scrape_options["text_separator"] = "\n"
        fixed_scrape = NodeSpec(
            id=broken.nodes[0].id,
            node_type=broken.nodes[0].node_type,
            plugin=broken.nodes[0].plugin,
            input=broken.nodes[0].input,
            on_success=broken.nodes[0].on_success,
            on_error=broken.nodes[0].on_error,
            options=scrape_options,
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        return broken.with_node(fixed_scrape)

    def test_wardline_broken_yaml_blocked_by_semantic_validator(self) -> None:
        state = self._wardline_broken_yaml_state()
        errors, contracts = validate_semantic_contracts(state)

        assert len(errors) == 1
        assert errors[0].component == "node:split_lines"

        assert len(contracts) == 1
        contract = contracts[0]
        assert contract.from_id == "scrape_page"
        assert contract.to_id == "split_lines"
        assert contract.outcome is SemanticOutcome.CONFLICT
        assert contract.requirement.requirement_code == "line_explode.source_field.line_framed_text"

    def test_wardline_broken_yaml_blocked_by_full_validate(self) -> None:
        state = self._wardline_broken_yaml_state()
        result = state.validate()
        assert result.is_valid is False
        # The wired surface must surface the structured violation, addressed by
        # plugin/requirement_code — not the legacy text_separator prose.
        assert any(
            entry.component == "node:split_lines"
            and "line_explode" in entry.message
            and "line_explode.source_field.line_framed_text" in entry.message
            for entry in result.errors
        )

    def test_wardline_fixed_yaml_passes_semantic_validator(self) -> None:
        """The post-fix YAML (text_separator='\\n') must NOT be flagged by the
        semantic validator. Pinning both directions of the regression means a
        future change that makes the validator too strict also fails this test
        — not just one that makes it too loose.
        """
        state = self._wardline_fixed_yaml_state()
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()
        assert len(contracts) == 1
        assert contracts[0].outcome is SemanticOutcome.SATISFIED

    def test_wardline_fixed_yaml_passes_full_validate(self) -> None:
        state = self._wardline_fixed_yaml_state()
        result = state.validate()
        # No semantic-contract entry should appear on split_lines.
        assert not any(
            entry.component == "node:split_lines" and "line_explode.source_field.line_framed_text" in entry.message
            for entry in result.errors
        )


class TestSemanticValidatorSecretLeakage:
    SENTINEL = "PASSWORD_SENTINEL_x9q7r3"

    def test_sentinel_does_not_appear_in_validator_output(self):
        # Build a Wardline state with sentinel in non-field-name options.
        state = CompositionState(
            metadata=PipelineMetadata(name="t"),
            version=1,
            edges=(),
            source=SourceSpec(
                plugin="csv",
                on_success="scrape_in",
                options={
                    "path": f"data/{self.SENTINEL}/url.csv",
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="scrape",
                    node_type="transform",
                    plugin="web_scrape",
                    input="scrape_in",
                    on_success="explode_in",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["url: str"]},
                        "required_input_fields": ["url"],
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "fingerprint",
                        "format": "text",
                        "text_separator": " ",
                        "http": {
                            "abuse_contact": f"x+{self.SENTINEL}@example.com",
                            "scraping_reason": f"reason-{self.SENTINEL}",
                            "timeout": 5,
                            "allowed_hosts": "public_only",
                        },
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="explode",
                    node_type="transform",
                    plugin="line_explode",
                    input="explode_in",
                    on_success="sink",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["content: str"]},
                        "source_field": "content",
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            outputs=(
                OutputSpec(
                    name="sink",
                    plugin="json",
                    options={"path": f"out-{self.SENTINEL}.json"},
                    on_write_failure="discard",
                ),
                OutputSpec(
                    name="errors",
                    plugin="json",
                    options={"path": "err.json"},
                    on_write_failure="discard",
                ),
            ),
        )

        errors, contracts = validate_semantic_contracts(state)
        for entry in errors:
            assert self.SENTINEL not in entry.message, f"Sentinel leaked in error message: {entry.message!r}"
            assert self.SENTINEL not in entry.component
        for contract in contracts:
            assert self.SENTINEL not in repr(contract)
