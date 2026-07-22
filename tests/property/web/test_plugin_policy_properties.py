"""Canonical stability and monotonic-authority properties for web policy."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from elspeth.contracts.plugin_capabilities import ControlMode, PluginCapability
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId, WebPluginPolicy

_REQUIRED = frozenset({PluginId("source", "csv"), PluginId("sink", "json")})
_OPTIONAL = (
    PluginId("sink", "database"),
    PluginId("transform", "azure_prompt_shield"),
    PluginId("transform", "azure_content_safety"),
)


def _policy(
    optional: frozenset[PluginId],
    preference: tuple[PluginId, ...] = (),
) -> WebPluginPolicy:
    return WebPluginPolicy.create(
        required=_REQUIRED,
        configured_optional=optional,
        preferences=((PluginCapability.PROMPT_SHIELD, preference),) if preference else (),
        control_modes=((PluginCapability.PROMPT_SHIELD, ControlMode.RECOMMEND),),
        plugin_code_identities=tuple((plugin_id, "1.0.0", "sha256:0123456789abcdef") for plugin_id in sorted(_REQUIRED | optional)),
    )


@given(st.lists(st.sampled_from(_OPTIONAL), unique=True))
def test_policy_hash_is_stable_under_set_input_order(items: list[PluginId]) -> None:
    assert _policy(frozenset(items)).policy_hash == _policy(frozenset(reversed(items))).policy_hash


@given(st.permutations(_OPTIONAL))
def test_preference_order_changes_policy_hash(order: list[PluginId] | tuple[PluginId, ...]) -> None:
    ordered = tuple(order)
    assert (
        _policy(frozenset(_OPTIONAL), ordered).policy_hash
        != _policy(
            frozenset(_OPTIONAL),
            tuple(reversed(ordered)),
        ).policy_hash
    )


@given(st.sets(st.sampled_from(_OPTIONAL)))
def test_principal_snapshot_can_narrow_but_never_expand_authority(usable: set[PluginId]) -> None:
    policy = _policy(frozenset(_OPTIONAL))
    available = _REQUIRED | frozenset(usable)
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash=policy.policy_hash,
        principal_scope="local:property",
        available=available,
        unavailable=(),
        selected=(),
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="a" * 64,
    )
    assert snapshot.available <= policy.authorized


@given(st.sets(st.sampled_from(_OPTIONAL)))
def test_every_selected_implementation_is_authorized_and_available(usable: set[PluginId]) -> None:
    policy = _policy(frozenset(_OPTIONAL))
    available = _REQUIRED | frozenset(usable)
    selected = min(usable) if usable else None
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash=policy.policy_hash,
        principal_scope="local:property",
        available=available,
        unavailable=(),
        selected=((PluginCapability.PROMPT_SHIELD, selected),),
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="b" * 64,
    )
    assert all(
        plugin_id is None or (plugin_id in snapshot.available and plugin_id in policy.authorized)
        for _capability, plugin_id in snapshot.selected
    )
