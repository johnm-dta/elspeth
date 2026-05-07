"""Regression coverage for SSRF DNS failure branches in web security checks."""

from __future__ import annotations

import ipaddress
import time

import pytest

from elspeth.core.security.web import (
    ALWAYS_BLOCKED_RANGES,
    NetworkError,
    SSRFBlockedError,
    _validate_ip_address,
    validate_url_for_ssrf,
)


class TestSSRFDnsFailureBranches:
    """Fail-closed behavior for resolver failures in validate_url_for_ssrf()."""

    def test_dns_timeout_is_translated_to_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolver timeout must surface as explicit NetworkError."""

        def _slow_resolve(hostname: str) -> list[str]:
            time.sleep(5)  # Much longer than the 0.01s timeout
            return ["127.0.0.1"]

        monkeypatch.setattr("elspeth.core.security.web._resolve_hostname", _slow_resolve)

        with pytest.raises(NetworkError, match=r"DNS resolution timeout \(0\.01s\): example\.com"):
            validate_url_for_ssrf("https://example.com/path", timeout=0.01)

    def test_unexpected_resolver_exception_is_wrapped_as_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unexpected resolver exceptions must be wrapped, not leaked raw."""

        def _exploding_resolve(hostname: str) -> list[str]:
            raise RuntimeError("resolver thread exploded")

        monkeypatch.setattr("elspeth.core.security.web._resolve_hostname", _exploding_resolve)

        with pytest.raises(NetworkError, match=r"DNS resolution failed: example\.com: resolver thread exploded"):
            validate_url_for_ssrf("https://example.com")

    def test_empty_dns_resolution_results_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolver returning no IPs must fail closed with NetworkError."""
        monkeypatch.setattr("elspeth.core.security.web._resolve_hostname", lambda hostname: [])

        with pytest.raises(NetworkError, match=r"DNS resolution returned no addresses: example\.com"):
            validate_url_for_ssrf("https://example.com")


# ===========================================================================
# Bug 7.5: DNS timeout effectiveness (daemon thread cleanup)
# ===========================================================================


class TestDnsTimeoutEffectiveness:
    """Bug 7.5: DNS resolution uses a bounded thread pool to avoid resource leaks."""

    def test_timeout_does_not_block_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After timeout, the caller must return promptly without blocking."""

        def _very_slow_resolve(hostname: str) -> list[str]:
            time.sleep(30)
            return ["127.0.0.1"]

        monkeypatch.setattr("elspeth.core.security.web._resolve_hostname", _very_slow_resolve)

        start = time.monotonic()
        with pytest.raises(NetworkError, match="DNS resolution timeout"):
            validate_url_for_ssrf("https://example.com", timeout=0.05)
        elapsed = time.monotonic() - start

        # Should return in well under 1 second, not 30
        assert elapsed < 1.0, f"Timeout took {elapsed:.1f}s — caller blocked on DNS thread"

    def test_ssrf_blocked_error_propagated_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SSRFBlockedError from resolver must propagate unwrapped."""

        def _ssrf_resolve(hostname: str) -> list[str]:
            raise SSRFBlockedError("blocked by test")

        monkeypatch.setattr("elspeth.core.security.web._resolve_hostname", _ssrf_resolve)

        with pytest.raises(SSRFBlockedError, match="blocked by test"):
            validate_url_for_ssrf("https://example.com")

    def test_dns_pool_is_bounded(self) -> None:
        """Thread pool must have a fixed upper bound on worker count."""
        from elspeth.core.security.web import _DNS_POOL_SIZE, _dns_pool

        assert _dns_pool._max_workers == _DNS_POOL_SIZE
        assert _DNS_POOL_SIZE <= 16, "Pool size should be modest to prevent resource exhaustion"


# ===========================================================================
# Bug 7.6: Port parsing
# ===========================================================================


class TestPortParsing:
    """Bug 7.6: Port 0 and invalid ports must be rejected."""

    def test_port_zero_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Port 0 is falsy but must be explicitly blocked."""
        with pytest.raises(SSRFBlockedError, match="Port 0"):
            validate_url_for_ssrf("https://example.com:0/path")

    def test_explicit_port_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit port (non-zero) should be used in the request."""
        monkeypatch.setattr(
            "elspeth.core.security.web._resolve_hostname",
            lambda hostname: ["93.184.216.34"],
        )
        result = validate_url_for_ssrf("https://example.com:8443/path")
        assert result.port == 8443

    def test_default_https_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No explicit port on HTTPS should default to 443."""
        monkeypatch.setattr(
            "elspeth.core.security.web._resolve_hostname",
            lambda hostname: ["93.184.216.34"],
        )
        result = validate_url_for_ssrf("https://example.com/path")
        assert result.port == 443

    def test_default_http_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No explicit port on HTTP should default to 80."""
        monkeypatch.setattr(
            "elspeth.core.security.web._resolve_hostname",
            lambda hostname: ["93.184.216.34"],
        )
        result = validate_url_for_ssrf("http://example.com/path")
        assert result.port == 80


# ===========================================================================
# ALWAYS_BLOCKED_RANGES: unconditional blocking
# ===========================================================================


class TestAlwaysBlockedRanges:
    """ALWAYS_BLOCKED_RANGES cannot be bypassed by allowed_ranges."""

    def test_cloud_metadata_blocked_even_with_allow_private(self) -> None:
        """169.254.169.254 blocked even when allowed_ranges covers everything."""
        allow_private = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address("169.254.169.254", allowed_ranges=allow_private)

    def test_ipv6_link_local_always_blocked(self) -> None:
        """fe80:: addresses are always blocked (IPv6 link-local)."""
        allow_private = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address("fe80::1", allowed_ranges=allow_private)

    def test_ipv4_broadcast_always_blocked(self) -> None:
        """255.255.255.255 is always blocked."""
        allow_private = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address("255.255.255.255", allowed_ranges=allow_private)

    def test_ipv4_multicast_always_blocked(self) -> None:
        """224.x.x.x is always blocked."""
        allow_private = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address("224.0.0.1", allowed_ranges=allow_private)

    def test_ipv6_multicast_always_blocked(self) -> None:
        """ff02::1 is always blocked."""
        allow_private = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address("ff02::1", allowed_ranges=allow_private)

    def test_constant_contains_expected_ranges(self) -> None:
        """Verify ALWAYS_BLOCKED_RANGES has all documented entries."""
        range_strs = {str(r) for r in ALWAYS_BLOCKED_RANGES}
        assert "169.254.0.0/16" in range_strs
        assert "fe80::/10" in range_strs
        assert "255.255.255.255/32" in range_strs
        assert "224.0.0.0/4" in range_strs
        assert "ff00::/8" in range_strs


# ===========================================================================
# allowed_ranges: selective blocklist bypass
# ===========================================================================


class TestAllowedRanges:
    """allowed_ranges parameter enables selective blocklist bypass."""

    def test_loopback_allowed_when_in_allowed_ranges(self) -> None:
        """127.0.0.1 passes when 127.0.0.0/8 is in allowed_ranges."""
        allowed = (ipaddress.ip_network("127.0.0.0/8"),)
        _validate_ip_address("127.0.0.1", allowed_ranges=allowed)  # Should not raise

    def test_loopback_blocked_without_allowed_ranges(self) -> None:
        """127.0.0.1 is still blocked when allowed_ranges is empty (default)."""
        with pytest.raises(SSRFBlockedError, match="Blocked IP range"):
            _validate_ip_address("127.0.0.1")

    def test_precise_allowlist_only_allows_matching_ip(self) -> None:
        """Allowing 127.0.0.1/32 does NOT allow 127.0.0.2."""
        allowed = (ipaddress.ip_network("127.0.0.1/32"),)
        _validate_ip_address("127.0.0.1", allowed_ranges=allowed)  # OK
        with pytest.raises(SSRFBlockedError):
            _validate_ip_address("127.0.0.2", allowed_ranges=allowed)

    def test_private_range_allowed_selectively(self) -> None:
        """10.0.0.0/8 allowed does not allow 192.168.1.1."""
        allowed = (ipaddress.ip_network("10.0.0.0/8"),)
        _validate_ip_address("10.1.2.3", allowed_ranges=allowed)  # OK
        with pytest.raises(SSRFBlockedError):
            _validate_ip_address("192.168.1.1", allowed_ranges=allowed)

    def test_public_ip_still_allowed_without_allowlist(self) -> None:
        """Public IPs pass even when allowed_ranges is empty."""
        _validate_ip_address("8.8.8.8")  # Should not raise

    def test_cross_family_no_match(self) -> None:
        """IPv4 allowlist does not match IPv6 addresses (cross-family)."""
        allowed = (ipaddress.ip_network("127.0.0.0/8"),)
        # IPv4-mapped IPv6 for 127.0.0.1 — should NOT match IPv4 allowlist
        with pytest.raises(SSRFBlockedError):
            _validate_ip_address("::ffff:127.0.0.1", allowed_ranges=allowed)

    def test_ipv6_allowlist_works(self) -> None:
        """IPv6 allowlist entry matches IPv6 addresses."""
        allowed = (ipaddress.ip_network("::1/128"),)
        _validate_ip_address("::1", allowed_ranges=allowed)  # OK


# ===========================================================================
# allowed_ranges through validate_url_for_ssrf
# ===========================================================================


class TestAllowedRangesFullPath:
    """allowed_ranges works end-to-end through validate_url_for_ssrf."""

    def test_loopback_allowed_via_full_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """127.0.0.1 passes full validation when allowed."""
        monkeypatch.setattr("elspeth.core.security.web._resolve_hostname", lambda h: ["127.0.0.1"])
        allowed = (ipaddress.ip_network("127.0.0.0/8"),)
        result = validate_url_for_ssrf("http://localhost/page", allowed_ranges=allowed)
        assert result.resolved_ip == "127.0.0.1"

    def test_cloud_metadata_blocked_via_full_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """169.254.169.254 blocked via full path even with allow_private."""
        monkeypatch.setattr("elspeth.core.security.web._resolve_hostname", lambda h: ["169.254.169.254"])
        allow_private = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            validate_url_for_ssrf("http://metadata/", allowed_ranges=allow_private)


class TestSSRFIPv4MappedIPv6MetadataBypass:
    """Regression: elspeth-05f9f00ffa — ::ffff:169.254.169.254 must be blocked
    unconditionally, even when broad IPv6 allowed_ranges bypass ::ffff:0:0/96."""

    def test_ipv4_mapped_metadata_blocked_by_validate_ip(self) -> None:
        """::ffff:169.254.169.254 must be always-blocked, not just standard-blocked."""
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address("::ffff:169.254.169.254")

    def test_ipv4_mapped_metadata_blocked_even_with_broad_ipv6_allowlist(self) -> None:
        """Broad IPv6 allowed_ranges must NOT bypass metadata endpoint blocking."""
        broad_ipv6 = (ipaddress.ip_network("::/0"),)
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address("::ffff:169.254.169.254", allowed_ranges=broad_ipv6)

    def test_ipv4_mapped_metadata_blocked_via_full_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: DNS returning ::ffff:169.254.169.254 must be blocked."""
        monkeypatch.setattr(
            "elspeth.core.security.web._resolve_hostname",
            lambda h: ["::ffff:169.254.169.254"],
        )
        allow_private = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            validate_url_for_ssrf("http://metadata.internal/", allowed_ranges=allow_private)

    def test_non_metadata_ipv4_mapped_still_allowed(self) -> None:
        """::ffff:10.0.0.1 should still be allowable (not metadata)."""
        allowed = (ipaddress.ip_network("::/0"),)
        _validate_ip_address("::ffff:10.0.0.1", allowed_ranges=allowed)  # must not raise


class TestResolveHostnameOrdering:
    """Regression: _resolve_hostname must preserve DNS resolver ordering."""

    def test_preserves_ip_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """IPs should come back in the order the resolver returned them."""
        import socket

        from elspeth.core.security.web import _resolve_hostname

        # Simulate getaddrinfo returning ordered IPs
        fake_results = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.3", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.2", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),  # duplicate
        ]
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: fake_results)

        result = _resolve_hostname("example.com")
        assert result == ["10.0.0.3", "10.0.0.1", "10.0.0.2"]
