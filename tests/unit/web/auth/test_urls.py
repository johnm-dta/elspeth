"""Closed-origin validation for browser-facing OIDC endpoints."""

from __future__ import annotations

import pytest

from elspeth.web.auth.urls import (
    validate_oidc_browser_endpoints,
    validate_oidc_browser_origins,
)

ISSUER = "https://cognito-idp.ap-southeast-2.amazonaws.com/pool-id"
COGNITO_ORIGIN = "https://example.auth.ap-southeast-2.amazoncognito.com"
AUTHORIZATION_ENDPOINT = f"{COGNITO_ORIGIN}/oauth2/authorize"
TOKEN_ENDPOINT = f"{COGNITO_ORIGIN}/oauth2/token"


def test_same_issuer_origin_pair_is_accepted_without_allowlist() -> None:
    pair = validate_oidc_browser_endpoints(
        "https://issuer.example.com/oauth2/authorize",
        "https://issuer.example.com/oauth2/token",
        issuer="https://issuer.example.com/pool",
    )
    assert pair == (
        "https://issuer.example.com/oauth2/authorize",
        "https://issuer.example.com/oauth2/token",
    )


def test_cross_origin_pair_requires_exact_allowlist_member() -> None:
    with pytest.raises(ValueError, match="browser endpoint origin is not allowed"):
        validate_oidc_browser_endpoints(AUTHORIZATION_ENDPOINT, TOKEN_ENDPOINT, issuer=ISSUER)

    assert validate_oidc_browser_endpoints(
        AUTHORIZATION_ENDPOINT,
        TOKEN_ENDPOINT,
        issuer=ISSUER,
        allowed_origins=(COGNITO_ORIGIN,),
    ) == (AUTHORIZATION_ENDPOINT, TOKEN_ENDPOINT)


@pytest.mark.parametrize(
    ("authorization_endpoint", "token_endpoint"),
    [
        ("http://issuer.example.com/authorize", "https://issuer.example.com/token"),
        ("https://issuer.example.com/authorize", "http://issuer.example.com/token"),
        ("", "https://issuer.example.com/token"),
        ("https://issuer.example.com/authorize\n", "https://issuer.example.com/token"),
        (r"https:\\issuer.example.com\authorize", "https://issuer.example.com/token"),
        ("https://issuer.example.com/%zz", "https://issuer.example.com/token"),
        ("https://issuer.example.com:bad/authorize", "https://issuer.example.com/token"),
        ("https://issuer.example.com:0/authorize", "https://issuer.example.com:0/token"),
        ("https://user:password@issuer.example.com/authorize", "https://issuer.example.com/token"),
        ("https://issuer.example.com/authorize?code=secret", "https://issuer.example.com/token"),
        ("https://issuer.example.com/authorize#secret", "https://issuer.example.com/token"),
        ("https://issuer.example.com/", "https://issuer.example.com/token"),
        ("https://issuer.example.com/authorize", "https://issuer.example.com"),
        ("https://127.0.0.1/authorize", "https://127.0.0.1/token"),
        ("https://169.254.169.254/authorize", "https://169.254.169.254/token"),
        ("https://10.0.0.1/authorize", "https://10.0.0.1/token"),
        ("https://*.example.com/authorize", "https://*.example.com/token"),
        ("https://issuer.example.com./authorize", "https://issuer.example.com./token"),
        ("https://bücher.example/authorize", "https://bücher.example/token"),
        ("https://bad_host.example/authorize", "https://bad_host.example/token"),
        ("https://-bad.example/authorize", "https://-bad.example/token"),
        ("https://bad..example/authorize", "https://bad..example/token"),
        ("https://127.1/authorize", "https://127.1/token"),
        ("https://0177.0.0.1/authorize", "https://0177.0.0.1/token"),
        ("https://0x7f000001/authorize", "https://0x7f000001/token"),
        ("https://2130706433/authorize", "https://2130706433/token"),
        ("https://[fe80::1%25eth0]/authorize", "https://[fe80::1%25eth0]/token"),
    ],
)
def test_adversarial_endpoint_values_fail_closed(
    authorization_endpoint: str,
    token_endpoint: str,
) -> None:
    with pytest.raises(ValueError) as raised:
        validate_oidc_browser_endpoints(
            authorization_endpoint,
            token_endpoint,
            issuer="https://issuer.example.com/pool",
            allowed_origins=("https://issuer.example.com",),
        )
    rendered = str(raised.value)
    if authorization_endpoint:
        assert authorization_endpoint not in rendered
    if token_endpoint:
        assert token_endpoint not in rendered
    assert "password" not in rendered
    assert "secret" not in rendered


@pytest.mark.parametrize(
    "origin",
    [
        "http://example.com",
        "https://example.com/path",
        "https://example.com/;params",
        "https://example.com/?query=secret",
        "https://example.com/#fragment",
        "https://user@example.com",
        "https://user:password@example.com",
        "https://*.example.com",
        "https://example.com.",
        "https://bücher.example",
    ],
)
def test_allowlist_entries_are_bare_safe_origins(origin: str) -> None:
    with pytest.raises(ValueError, match="allowed origin") as raised:
        validate_oidc_browser_origins((origin,))
    assert origin not in str(raised.value)


def test_allowlist_normalizes_and_rejects_duplicate_origins() -> None:
    assert validate_oidc_browser_origins((" https://EXAMPLE.com:8443/ ",)) == ("https://example.com:8443",)
    with pytest.raises(ValueError, match="duplicate"):
        validate_oidc_browser_origins(("https://example.com", "https://example.com:443/"))


@pytest.mark.parametrize(
    "allowed_origin",
    [
        "https://sibling.auth.ap-southeast-2.amazoncognito.com",
        "https://auth.ap-southeast-2.amazoncognito.com",
        "https://evil-example.auth.ap-southeast-2.amazoncognito.com",
        f"{COGNITO_ORIGIN}:444",
        "https://xn--bcher-kva.example",
    ],
)
def test_origin_equality_does_not_use_suffix_or_similarity(allowed_origin: str) -> None:
    with pytest.raises(ValueError, match="browser endpoint origin is not allowed"):
        validate_oidc_browser_endpoints(
            AUTHORIZATION_ENDPOINT,
            TOKEN_ENDPOINT,
            issuer=ISSUER,
            allowed_origins=(allowed_origin,),
        )


def test_default_port_and_mixed_host_case_compare_by_normalized_origin() -> None:
    assert validate_oidc_browser_endpoints(
        "https://EXAMPLE.AUTH.ap-southeast-2.amazoncognito.com:443/oauth2/authorize",
        "https://example.auth.ap-southeast-2.amazoncognito.com/oauth2/token",
        issuer=ISSUER,
        allowed_origins=(COGNITO_ORIGIN,),
    )[0].startswith("https://EXAMPLE.AUTH")


def test_nondefault_port_must_match_allowlist_and_both_endpoints() -> None:
    with pytest.raises(ValueError, match="same origin"):
        validate_oidc_browser_endpoints(
            f"{COGNITO_ORIGIN}:8443/oauth2/authorize",
            f"{COGNITO_ORIGIN}/oauth2/token",
            issuer=ISSUER,
            allowed_origins=(f"{COGNITO_ORIGIN}:8443",),
        )
    assert validate_oidc_browser_endpoints(
        f"{COGNITO_ORIGIN}:8443/oauth2/authorize",
        f"{COGNITO_ORIGIN}:8443/oauth2/token",
        issuer=ISSUER,
        allowed_origins=(f"{COGNITO_ORIGIN}:8443",),
    )


def test_public_ipv6_literal_compares_using_canonical_address() -> None:
    assert validate_oidc_browser_endpoints(
        "https://[2606:4700:4700::1111]/authorize",
        "https://[2606:4700:4700:0:0:0:0:1111]:443/token",
        issuer="https://issuer.example.com/pool",
        allowed_origins=("https://[2606:4700:4700::1111]",),
    )


def test_authorization_and_token_endpoint_origins_must_match() -> None:
    with pytest.raises(ValueError, match="same origin"):
        validate_oidc_browser_endpoints(
            AUTHORIZATION_ENDPOINT,
            "https://other.auth.ap-southeast-2.amazoncognito.com/oauth2/token",
            issuer=ISSUER,
            allowed_origins=(COGNITO_ORIGIN, "https://other.auth.ap-southeast-2.amazoncognito.com"),
        )


@pytest.mark.parametrize(
    "smuggled",
    [
        f"https://evil.example/path/{COGNITO_ORIGIN}/oauth2/authorize",
        f"https://evil.example/authorize?next={COGNITO_ORIGIN}",
        f"https://evil.example/{COGNITO_ORIGIN.replace('/', '%2f')}",
        "https://evil.example/authorize%3fnext%3dhttps%3a%2f%2fexample.com",
    ],
)
def test_embedding_allowed_url_does_not_authorize_initial_destination(smuggled: str) -> None:
    with pytest.raises(ValueError):
        validate_oidc_browser_endpoints(
            smuggled,
            TOKEN_ENDPOINT,
            issuer=ISSUER,
            allowed_origins=(COGNITO_ORIGIN,),
        )
