def test_profile_unavailable_finding_enumerates_available_aliases() -> None:
    """When operator profiles EXIST and the options simply failed to select
    one, the finding must say so and name the aliases — telling the model
    'an operator must enable a profile' when 'sonnet' is sitting right there
    caused a false honest decline and blind repairs (live sessions 6b9da203 +
    e47fd8df)."""
    from elspeth.web.plugin_policy.validation import _profile_unavailable_finding

    class _Component:
        component_id = "node:llm_1"
        component_type = "transform"

    from elspeth.web.plugin_policy.models import PluginId

    finding = _profile_unavailable_finding(_Component(), PluginId("transform", "llm"), available_aliases=("sonnet",))
    assert "sonnet" in finding.message
    assert "operator must enable" not in finding.message

    unconfigured = _profile_unavailable_finding(_Component(), PluginId("transform", "llm"))
    assert "sonnet" not in unconfigured.message
