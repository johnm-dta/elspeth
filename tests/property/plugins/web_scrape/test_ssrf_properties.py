# tests/property/plugins/web_scrape/test_ssrf_properties.py
"""Property-based tests for SSRF prevention invariants.

Security-critical properties of core/security/web.py:
- All blocked IP ranges are properly rejected
- IPv4-mapped IPv6 bypass is prevented (::ffff:x.x.x.x)
- Multi-homed hosts are fail-closed (ANY blocked IP blocks the request)
- Connection URL format is correct (IPv6 brackets, port/path preservation)
- Zone-scoped IPv6 is fail-closed (unparseable = blocked)
- Scheme validation is case-insensitive

These properties protect against SSRF attacks including the DNS rebinding
TOCTOU vulnerability documented in the repair manifest.
"""

from __future__ import annotations

import ipaddress
import socket
from unittest.mock import patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.core.security.web import (
    BLOCKED_IP_RANGES,
    SSRFBlockedError,
    SSRFSafeRequest,
    _validate_ip_address,
    validate_url_for_ssrf,
    validate_url_scheme,
)

# =============================================================================
# IP Address Generation Strategies
# =============================================================================


@st.composite
def blocked_ipv4_from_range(draw: st.DrawFn) -> tuple[str, ipaddress.IPv4Network]:
    """Generate an IPv4 address from a randomly chosen blocked range.

    Returns (ip_string, network) so test assertions can reference the range.
    """
    ipv4_ranges = [r for r in BLOCKED_IP_RANGES if r.version == 4]
    network = draw(st.sampled_from(ipv4_ranges))
    host_int = draw(
        st.integers(
            min_value=int(network.network_address),
            max_value=int(network.broadcast_address),
        )
    )
    return str(ipaddress.IPv4Address(host_int)), network


@st.composite
def blocked_ipv6_from_range(draw: st.DrawFn) -> tuple[str, ipaddress.IPv6Network]:
    """Generate an IPv6 address from a randomly chosen blocked range.

    Excludes ::ffff:0:0/96 (IPv4-mapped) since those are tested separately.
    """
    ipv6_ranges = [r for r in BLOCKED_IP_RANGES if r.version == 6 and r != ipaddress.ip_network("::ffff:0:0/96")]
    network = draw(st.sampled_from(ipv6_ranges))
    host_int = draw(
        st.integers(
            min_value=int(network.network_address),
            max_value=int(network.broadcast_address),
        )
    )
    return str(ipaddress.IPv6Address(host_int)), network


@st.composite
def ipv4_mapped_ipv6(draw: st.DrawFn) -> str:
    """Generate IPv4-mapped IPv6 addresses (::ffff:x.x.x.x) from blocked IPv4 ranges.

    This is the critical bypass vector: attackers encode blocked IPv4 addresses
    as IPv6 to evade IPv4-only blocklists.
    """
    ipv4_str, _ = draw(blocked_ipv4_from_range())
    ipv4_addr = ipaddress.IPv4Address(ipv4_str)
    # Construct the mapped address via the standard library
    mapped = ipaddress.IPv6Address(f"::ffff:{ipv4_addr}")
    return str(mapped)


@st.composite
def safe_public_ipv4(draw: st.DrawFn) -> str:
    """Generate public IPv4 addresses known to be outside all blocked ranges."""
    # Use well-known public DNS/CDN IPs — safe by construction
    return draw(
        st.sampled_from(
            [
                "8.8.8.8",
                "8.8.4.4",
                "1.1.1.1",
                "1.0.0.1",
                "208.67.222.222",
                "208.67.220.220",
                "142.251.32.46",
                "151.101.1.140",
            ]
        )
    )


def _mock_dns_to(ips: list[str]):
    """Create a mock getaddrinfo that resolves to the given IP list."""

    def _mock_getaddrinfo(*args, **kwargs):
        results = []
        for ip in ips:
            if ":" in ip:
                results.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0)))
            else:
                results.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)))
        return results

    return _mock_getaddrinfo


# =============================================================================
# Property Tests: Blocked IP Range Coverage
# =============================================================================


class TestBlockedIPv4Ranges:
    """Every IPv4 address in a blocked range must be rejected."""

    @given(data=blocked_ipv4_from_range())
    @settings(max_examples=500)
    def test_all_blocked_ipv4_rejected(self, data: tuple[str, ipaddress.IPv4Network]) -> None:
        """Property: Any IPv4 in BLOCKED_IP_RANGES raises SSRFBlockedError."""
        ip_str, _network = data
        with pytest.raises(SSRFBlockedError, match=r"[Bb]locked IP range"):
            _validate_ip_address(ip_str)

    @given(data=blocked_ipv4_from_range())
    @settings(max_examples=200)
    def test_blocked_ipv4_via_full_validation(self, data: tuple[str, ipaddress.IPv4Network]) -> None:
        """Property: Blocked IPv4 rejected through the full validate_url_for_ssrf path."""
        ip_str, _ = data
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([ip_str])), pytest.raises(SSRFBlockedError):
            validate_url_for_ssrf("http://test.example.com/path")


class TestBlockedIPv6Ranges:
    """Every IPv6 address in a blocked range must be rejected."""

    @given(data=blocked_ipv6_from_range())
    @settings(max_examples=300)
    def test_all_blocked_ipv6_rejected(self, data: tuple[str, ipaddress.IPv6Network]) -> None:
        """Property: Any IPv6 in blocked ranges raises SSRFBlockedError."""
        ip_str, _network = data
        with pytest.raises(SSRFBlockedError, match=r"[Bb]locked IP range"):
            _validate_ip_address(ip_str)


class TestIPv4MappedIPv6Bypass:
    """IPv4-mapped IPv6 (::ffff:x.x.x.x) must NOT bypass the blocklist.

    This is the CRITICAL bypass vector. The ::ffff:0:0/96 range in
    BLOCKED_IP_RANGES exists specifically to catch this.
    """

    @given(mapped_ip=ipv4_mapped_ipv6())
    @settings(max_examples=300)
    def test_ipv4_mapped_ipv6_blocked(self, mapped_ip: str) -> None:
        """Property: ::ffff:x.x.x.x is blocked when x.x.x.x is blocked."""
        with pytest.raises(SSRFBlockedError):
            _validate_ip_address(mapped_ip)

    @given(mapped_ip=ipv4_mapped_ipv6())
    @settings(max_examples=100)
    def test_ipv4_mapped_via_full_validation(self, mapped_ip: str) -> None:
        """Property: Mapped addresses blocked through full validation path."""
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([mapped_ip])), pytest.raises(SSRFBlockedError):
            validate_url_for_ssrf("https://test.example.com/")


# =============================================================================
# Property Tests: Zone-Scoped IPv6 Fail-Closed
# =============================================================================


class TestZoneScopedIPv6:
    """Zone-scoped IPv6 (fe80::1%eth0) must be fail-closed.

    Zone-scoped IPs are blocked either because:
    1. Python's ipaddress can't parse the zone ID → "Unparseable IP" error
    2. The underlying address is in a blocked range (e.g. fe80::/10) → "Blocked IP range" error
    Either way, the request must be rejected.
    """

    @given(
        zone=st.text(
            min_size=1,
            max_size=10,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        )
    )
    @settings(max_examples=100)
    def test_zone_scoped_ipv6_rejected(self, zone: str) -> None:
        """Property: fe80::1%<zone> is always blocked (fail-closed).

        Some zone IDs (e.g. numeric "0") are stripped by Python 3.9+,
        causing the IP to parse and match fe80::/10. Other zones cause
        a ValueError. Both paths must result in SSRFBlockedError.
        """
        ip_str = f"fe80::1%{zone}"
        with pytest.raises(SSRFBlockedError):
            _validate_ip_address(ip_str)


# =============================================================================
# Property Tests: Multi-Homed Host Fail-Closed
# =============================================================================


class TestMultiHomedHost:
    """If a hostname resolves to multiple IPs and ANY is blocked, reject all."""

    @given(
        blocked=blocked_ipv4_from_range(),
        safe=safe_public_ipv4(),
    )
    @settings(max_examples=200)
    def test_mixed_ips_blocked_if_any_unsafe(self, blocked: tuple[str, ipaddress.IPv4Network], safe: str) -> None:
        """Property: Mixed safe+blocked IPs are rejected (fail-closed)."""
        blocked_ip, _ = blocked
        # DNS returns both safe and blocked IPs
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([safe, blocked_ip])), pytest.raises(SSRFBlockedError):
            validate_url_for_ssrf("http://multihomed.example.com/")

    @given(
        blocked=blocked_ipv4_from_range(),
        safe=safe_public_ipv4(),
    )
    @settings(max_examples=100)
    def test_order_doesnt_matter(self, blocked: tuple[str, ipaddress.IPv4Network], safe: str) -> None:
        """Property: Blocked IP is caught regardless of order in DNS results."""
        blocked_ip, _ = blocked
        # Blocked IP first
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([blocked_ip, safe])), pytest.raises(SSRFBlockedError):
            validate_url_for_ssrf("http://multihomed.example.com/")


# =============================================================================
# Property Tests: Connection URL Format
# =============================================================================


class TestConnectionURLFormat:
    """SSRFSafeRequest.connection_url must produce valid URLs."""

    @given(
        port=st.integers(min_value=1, max_value=65535),
        path=st.text(
            min_size=0,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/-_"),
        ),
        scheme=st.sampled_from(["http", "https"]),
    )
    @settings(max_examples=200)
    def test_ipv4_connection_url_no_brackets(self, port: int, path: str, scheme: str) -> None:
        """Property: IPv4 addresses in connection_url have no brackets."""
        if not path.startswith("/"):
            path = "/" + path
        req = SSRFSafeRequest(
            original_url=f"{scheme}://example.com{path}",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=port,
            path=path,
            scheme=scheme,
            bare_hostname="example.com",
        )
        url = req.connection_url
        assert "[" not in url, f"IPv4 URL should not have brackets: {url}"
        assert f"93.184.216.34:{port}" in url
        assert url.startswith(f"{scheme}://")
        assert url.endswith(path)

    @given(
        port=st.integers(min_value=1, max_value=65535),
        scheme=st.sampled_from(["http", "https"]),
    )
    @settings(max_examples=200)
    def test_ipv6_connection_url_has_brackets(self, port: int, scheme: str) -> None:
        """Property: IPv6 addresses in connection_url have brackets per RFC 2732."""
        req = SSRFSafeRequest(
            original_url=f"{scheme}://example.com/",
            resolved_ip="2606:4700:4700::1111",
            host_header="example.com",
            port=port,
            path="/",
            scheme=scheme,
            bare_hostname="example.com",
        )
        url = req.connection_url
        assert "[2606:4700:4700::1111]" in url, f"IPv6 URL must have brackets: {url}"
        assert f"[2606:4700:4700::1111]:{port}" in url

    @given(
        query=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="=&"),
        ),
        fragment=st.text(
            min_size=0,
            max_size=10,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100)
    def test_path_query_fragment_preserved(self, query: str, fragment: str) -> None:
        """Property: Path, query string, and fragment survive in connection_url."""
        path = f"/api/v1?{query}"
        if fragment:
            path = f"{path}#{fragment}"
        req = SSRFSafeRequest(
            original_url=f"https://example.com{path}",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=443,
            path=path,
            scheme="https",
            bare_hostname="example.com",
        )
        assert req.connection_url.endswith(path)


# =============================================================================
# Property Tests: Scheme Validation
# =============================================================================


class TestSchemeValidation:
    """Scheme validation must be case-insensitive and strict."""

    @given(
        scheme=st.sampled_from(["http", "https", "HTTP", "HTTPS", "Http", "hTtPs"]),
    )
    @settings(max_examples=50)
    def test_allowed_schemes_case_insensitive(self, scheme: str) -> None:
        """Property: http/https accepted in any case."""
        validate_url_scheme(f"{scheme}://example.com/")

    @given(
        scheme=st.sampled_from(
            [
                "file",
                "ftp",
                "gopher",
                "data",
                "javascript",
                "FILE",
                "FTP",
                "GOPHER",
                "DATA",
                "JAVASCRIPT",
            ]
        ),
    )
    @settings(max_examples=50)
    def test_forbidden_schemes_rejected(self, scheme: str) -> None:
        """Property: Non-http(s) schemes always rejected."""
        with pytest.raises(SSRFBlockedError, match="Forbidden scheme"):
            validate_url_scheme(f"{scheme}://payload")


# =============================================================================
# Property Tests: Full Path Validation
# =============================================================================


class TestFullPathValidation:
    """End-to-end validation through validate_url_for_ssrf."""

    @given(safe_ip=safe_public_ipv4())
    @settings(max_examples=100)
    def test_safe_ips_produce_valid_request(self, safe_ip: str) -> None:
        """Property: Safe IPs produce a well-formed SSRFSafeRequest."""
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([safe_ip])):
            result = validate_url_for_ssrf("https://example.com/api?key=val#sec")

        assert result.resolved_ip == safe_ip
        assert result.host_header == "example.com"
        assert result.scheme == "https"
        assert result.port == 443
        assert "/api" in result.path
        assert "key=val" in result.path
        assert "#sec" in result.path
        assert result.sni_hostname == "example.com"

    def test_no_hostname_rejected(self) -> None:
        """Edge case: URL with no hostname is rejected."""
        with pytest.raises(SSRFBlockedError, match="no hostname"):
            validate_url_for_ssrf("http:///path")

    @given(safe_ip=safe_public_ipv4())
    @settings(max_examples=50)
    def test_default_port_http_80(self, safe_ip: str) -> None:
        """Property: HTTP without explicit port defaults to 80."""
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([safe_ip])):
            result = validate_url_for_ssrf("http://example.com/")
        assert result.port == 80

    @given(safe_ip=safe_public_ipv4())
    @settings(max_examples=50)
    def test_default_port_https_443(self, safe_ip: str) -> None:
        """Property: HTTPS without explicit port defaults to 443."""
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([safe_ip])):
            result = validate_url_for_ssrf("https://example.com/")
        assert result.port == 443

    @given(
        safe_ip=safe_public_ipv4(),
        port=st.integers(min_value=1, max_value=65535),
    )
    @settings(max_examples=100)
    def test_explicit_port_preserved(self, safe_ip: str, port: int) -> None:
        """Property: Explicit port in URL is preserved."""
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([safe_ip])):
            result = validate_url_for_ssrf(f"https://example.com:{port}/")
        assert result.port == port

    @given(safe_ip=safe_public_ipv4())
    @settings(max_examples=50)
    def test_ipv4_preferred_over_ipv6(self, safe_ip: str) -> None:
        """Property: When both IPv4 and IPv6 resolve, IPv4 is selected."""
        safe_ipv6 = "2606:4700:4700::1111"
        with patch("socket.getaddrinfo", side_effect=_mock_dns_to([safe_ipv6, safe_ip])):
            result = validate_url_for_ssrf("https://example.com/")
        # IPv4 preferred for compatibility
        assert result.resolved_ip == safe_ip


# =============================================================================
# Property Tests: Allowlist-Blocklist Interaction Invariants
# =============================================================================


class TestAllowlistBlocklistInvariants:
    """Properties governing the interaction between allowed_ranges and blocklists."""

    @given(data=blocked_ipv4_from_range())
    @settings(max_examples=200)
    def test_allowed_ip_bypasses_blocklist(self, data: tuple[str, ipaddress.IPv4Network]) -> None:
        """Property: An IP in allowed_ranges is NOT blocked by BLOCKED_IP_RANGES."""
        ip_str, network = data
        from elspeth.core.security.web import ALWAYS_BLOCKED_RANGES

        ip = ipaddress.ip_address(ip_str)
        assume(not any(ip in abr for abr in ALWAYS_BLOCKED_RANGES))
        allowed = (network,)
        _validate_ip_address(ip_str, allowed_ranges=allowed)  # Should not raise

    @given(data=blocked_ipv4_from_range())
    @settings(max_examples=200)
    def test_without_allowlist_still_blocked(self, data: tuple[str, ipaddress.IPv4Network]) -> None:
        """Property: Same IPs are blocked when allowed_ranges is empty."""
        ip_str, _ = data
        with pytest.raises(SSRFBlockedError):
            _validate_ip_address(ip_str)

    @given(safe_ip=safe_public_ipv4())
    @settings(max_examples=100)
    def test_public_ip_unaffected_by_allowlist(self, safe_ip: str) -> None:
        """Property: Public IPs pass regardless of allowed_ranges content."""
        # With no allowlist
        _validate_ip_address(safe_ip)
        # With an allowlist for private ranges
        allowed = (ipaddress.ip_network("10.0.0.0/8"),)
        _validate_ip_address(safe_ip, allowed_ranges=allowed)


class TestAlwaysBlockedInvariants:
    """Properties: ALWAYS_BLOCKED_RANGES cannot be overridden."""

    @given(
        ip_int=st.integers(
            min_value=int(ipaddress.ip_address("169.254.0.0")),
            max_value=int(ipaddress.ip_address("169.254.255.255")),
        )
    )
    @settings(max_examples=200)
    def test_link_local_never_allowed(self, ip_int: int) -> None:
        """Property: Any 169.254.x.x IP is always blocked, even with allowlist."""
        ip_str = str(ipaddress.IPv4Address(ip_int))
        allow_private = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address(ip_str, allowed_ranges=allow_private)

    @given(
        ip_int=st.integers(
            min_value=int(ipaddress.ip_address("224.0.0.0")),
            max_value=int(ipaddress.ip_address("239.255.255.255")),
        )
    )
    @settings(max_examples=100)
    def test_multicast_never_allowed(self, ip_int: int) -> None:
        """Property: Any 224.x.x.x-239.x.x.x IP is always blocked."""
        ip_str = str(ipaddress.IPv4Address(ip_int))
        allow_private = (ipaddress.ip_network("0.0.0.0/0"),)
        with pytest.raises(SSRFBlockedError, match="Always-blocked"):
            _validate_ip_address(ip_str, allowed_ranges=allow_private)
