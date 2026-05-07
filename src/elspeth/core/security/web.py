"""Web security infrastructure for SSRF prevention and URL validation.

This module provides defense-in-depth against Server-Side Request Forgery (SSRF):

1. **Scheme validation**: Only HTTP/HTTPS allowed (blocks file://, ftp://, etc.)
2. **IP blocking**: Comprehensive blocklist covering private, loopback, metadata endpoints
3. **DNS rebinding prevention**: IP pinning via SSRFSafeRequest (resolve once, use that IP)

The key innovation is SSRFSafeRequest + validate_url_for_ssrf() which eliminates
the TOCTOU (Time-of-Check to Time-of-Use) vulnerability in traditional SSRF defenses:

    # VULNERABLE (old pattern - resolved twice, attacker can return different IPs):
    # validate then httpx.get(url) → second DNS lookup → 169.254.169.254

    # SECURE (resolved once, IP pinned for connection):
    safe_request = validate_url_for_ssrf(url)  # Resolves and validates
    httpx.get(safe_request.connection_url, headers={"Host": safe_request.host_header})
"""

from __future__ import annotations

import ipaddress
import queue
import socket
import urllib.parse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv6Network


class SSRFBlockedError(Exception):
    """URL validation failed due to security policy (SSRF prevention)."""

    pass


class NetworkError(Exception):
    """Network operation failed (DNS, connection, etc.)."""

    pass


ALLOWED_SCHEMES = {"http", "https"}

# Comprehensive blocklist for SSRF prevention
# Each range blocks a specific attack vector - do not remove without security review
BLOCKED_IP_RANGES = [
    # IPv4 ranges
    ipaddress.ip_network("0.0.0.0/8"),  # Current network (RFC 1122) - can route to localhost
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("10.0.0.0/8"),  # Private Class A (RFC 1918)
    ipaddress.ip_network("172.16.0.0/12"),  # Private Class B (RFC 1918)
    ipaddress.ip_network("192.168.0.0/16"),  # Private Class C (RFC 1918)
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local (AWS/Azure/GCP metadata endpoints)
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT (RFC 6598) - shared ISP space, often internal
    # IPv6 ranges
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique local (RFC 4193) - private
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local (RFC 4291) - can reach metadata
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped IPv6 - CRITICAL: bypass vector!
]

# Unconditionally blocked — no allowlist can bypass these.
# Cloud metadata endpoints are the #1 SSRF target (IAM credential exfiltration).
# Broadcast/multicast are never valid HTTP targets.
#
# NOTE: ::ffff:0:0/96 (IPv4-mapped IPv6) is intentionally NOT here — it is in
# BLOCKED_IP_RANGES where it can be bypassed by allowed_ranges. This is correct:
# in "allow_private" mode, ::ffff:10.x.x.x should be allowed (the operator asked
# for private access). However, ::ffff:169.254.0.0/112 IS here to unconditionally
# block the IPv4-mapped form of the metadata endpoint. Without this entry, a broad
# IPv6 allowed_range covering ::ffff:0:0/96 would bypass the standard blocklist
# before the IPv4 169.254.0.0/16 check could catch it (IPv6 addresses are checked
# against IPv6 networks, not IPv4 networks).
ALWAYS_BLOCKED_RANGES = (
    ipaddress.ip_network("169.254.0.0/16"),  # IPv4 link-local (AWS/Azure/GCP metadata)
    ipaddress.ip_network("::ffff:169.254.0.0/112"),  # IPv4-mapped metadata endpoint
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local (same attack surface)
    ipaddress.ip_network("255.255.255.255/32"),  # IPv4 broadcast
    ipaddress.ip_network("224.0.0.0/4"),  # IPv4 multicast
    ipaddress.ip_network("ff00::/8"),  # IPv6 multicast
)

# Bounded thread pool for DNS resolution.  Caps concurrent getaddrinfo threads
# so repeated timeouts (e.g. blackholed resolver) cannot exhaust OS threads.
_DNS_POOL_SIZE = 8
_dns_pool = ThreadPoolExecutor(max_workers=_DNS_POOL_SIZE, thread_name_prefix="dns_resolve")


def validate_url_scheme(url: str) -> None:
    """Validate URL scheme is in allowlist (http/https only).

    Args:
        url: URL to validate

    Raises:
        SSRFBlockedError: If scheme is not in allowlist
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SSRFBlockedError(f"Forbidden scheme: {parsed.scheme}")


def _resolve_hostname(hostname: str) -> list[str]:
    """Resolve hostname to all IP addresses (IPv4 + IPv6).

    Uses getaddrinfo() instead of gethostbyname() to support IPv6.

    Args:
        hostname: Hostname to resolve

    Returns:
        List of unique IP addresses (IPv4 and/or IPv6)

    Raises:
        NetworkError: If DNS resolution fails
    """
    try:
        # AF_UNSPEC = return both IPv4 and IPv6
        # SOCK_STREAM = TCP (filters out UDP-only results)
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        # results[i] = (family, type, proto, canonname, sockaddr)
        # sockaddr = (ip, port) for IPv4 or (ip, port, flow, scope) for IPv6
        # sockaddr[0] is always the IP address string for both AF_INET and AF_INET6
        seen: set[str] = set()
        ips: list[str] = []
        for r in results:
            ip = str(r[4][0])
            if ip not in seen:
                seen.add(ip)
                ips.append(ip)
        return ips
    except socket.gaierror as e:
        raise NetworkError(f"DNS resolution failed: {hostname}: {e}") from e


def _validate_ip_address(
    ip_str: str,
    *,
    allowed_ranges: Sequence[IPv4Network | IPv6Network] = (),
) -> None:
    """Validate that an IP address is not in any blocked range.

    Three-tier check order:
    1. ALWAYS_BLOCKED_RANGES — unconditional, no allowlist bypass
    2. allowed_ranges — if IP matches, skip blocklist (early return)
    3. BLOCKED_IP_RANGES — standard blocklist

    Args:
        ip_str: IP address string (IPv4 or IPv6)
        allowed_ranges: IP networks that bypass the standard blocklist.
            Does NOT bypass ALWAYS_BLOCKED_RANGES.

    Raises:
        SSRFBlockedError: If IP is blocked or unparseable (fail-closed)
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError as e:
        # Fail CLOSED: if we can't parse the IP (e.g. zone-scoped IPv6 like
        # "fe80::1%eth0"), block the request rather than allowing it through.
        raise SSRFBlockedError(f"Unparseable IP address: {ip_str!r}: {e}") from e

    # 1. Always-blocked — unconditional, no bypass
    for never_allow in ALWAYS_BLOCKED_RANGES:
        if ip in never_allow:
            raise SSRFBlockedError(f"Always-blocked IP range: {ip_str} in {never_allow}")

    # 2. Allowlist — if IP matches an allowed range, skip blocklist
    # Note: cross-family checks (e.g. IPv6 in IPv4 network) return False
    # in CPython 3.7+ — no TypeError to catch.
    for allowed in allowed_ranges:
        if ip in allowed:
            return

    # 3. Standard blocklist
    for blocked in BLOCKED_IP_RANGES:
        if ip in blocked:
            raise SSRFBlockedError(f"Blocked IP range: {ip_str} in {blocked}")


@dataclass(frozen=True, slots=True)
class SSRFSafeRequest:
    """Request with pre-validated IP to prevent DNS rebinding attacks.

    This dataclass captures everything needed to make a request to a specific
    IP address while preserving the original hostname for Host header and TLS SNI.

    The key insight is that we resolve DNS once, validate the IP, then force the
    HTTP client to connect to that exact IP by rewriting the URL. The Host header
    ensures virtual hosting works, and SNI ensures TLS certificate verification.

    Attributes:
        original_url: The URL as provided by the user
        resolved_ip: The validated IP address we will connect to
        host_header: HTTP Host header value — includes port for non-default ports
            (e.g. "example.com:8443" for https on port 8443)
        port: Port number (explicit or default based on scheme)
        path: URL path including query string
        scheme: "http" or "https"
        bare_hostname: Bare hostname without port, for TLS SNI

    Example:
        safe_request = validate_url_for_ssrf("https://example.com:8443/path?q=1")
        # safe_request.connection_url = "https://93.184.216.34:8443/path?q=1"
        # safe_request.host_header = "example.com:8443"
        # safe_request.sni_hostname = "example.com"
    """

    original_url: str
    resolved_ip: str
    host_header: str
    port: int
    path: str
    scheme: str
    bare_hostname: str

    @property
    def connection_url(self) -> str:
        """URL with hostname replaced by resolved IP for direct connection.

        For IPv6 addresses, brackets are added per RFC 2732.
        """
        # IPv6 addresses need brackets in URLs
        ip_for_url = f"[{self.resolved_ip}]" if ":" in self.resolved_ip else self.resolved_ip
        return f"{self.scheme}://{ip_for_url}:{self.port}{self.path}"

    @property
    def sni_hostname(self) -> str:
        """Hostname for TLS SNI (Server Name Indication).

        Always the bare hostname without port — TLS SNI must not include port.
        """
        return self.bare_hostname


def validate_url_for_ssrf(
    url: str,
    timeout: float = 5.0,
    *,
    allowed_ranges: Sequence[IPv4Network | IPv6Network] = (),
) -> SSRFSafeRequest:
    """Validate URL and return request with pinned IP for SSRF-safe connection.

    This is the primary entry point for SSRF-safe HTTP requests. It:
    1. Validates the URL scheme (http/https only)
    2. Resolves DNS with timeout
    3. Validates ALL resolved IPs against blocklist
    4. Returns SSRFSafeRequest with pinned IP

    The returned SSRFSafeRequest should be used with get_ssrf_safe() or similar
    methods that connect directly to the resolved IP.

    Args:
        url: URL to validate and prepare for fetching
        timeout: DNS resolution timeout in seconds
        allowed_ranges: Additional IP ranges to permit (e.g. RFC1918 for
            internal services). These bypass the default blocklist check.

    Returns:
        SSRFSafeRequest ready for secure fetching

    Raises:
        SSRFBlockedError: If scheme is forbidden or IP is blocked
        NetworkError: If DNS resolution fails or times out
    """
    # Step 1: Validate scheme
    validate_url_scheme(url)

    # Step 2: Parse URL components
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise SSRFBlockedError("URL has no hostname")

    # Determine port (explicit or default)
    # parsed.port can raise ValueError for out-of-range ports (e.g. 99999).
    # Port 0 is falsy but must be explicitly rejected (not a valid HTTP port).
    try:
        explicit_port = parsed.port
    except ValueError as e:
        raise SSRFBlockedError(f"Invalid port in URL: {e}") from e

    if explicit_port is not None:
        if explicit_port == 0:
            raise SSRFBlockedError("Port 0 is not allowed")
        port = explicit_port
    else:
        port = 443 if parsed.scheme.lower() == "https" else 80

    # Build path (including query string and fragment)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    if parsed.fragment:
        path = f"{path}#{parsed.fragment}"

    # Step 3: Resolve DNS with timeout via bounded thread pool.
    # Using a shared ThreadPoolExecutor (max _DNS_POOL_SIZE workers) instead
    # of spawning one daemon thread per call.  Under repeated timeouts (e.g.
    # blackholed DNS resolver), at most _DNS_POOL_SIZE threads are blocked in
    # getaddrinfo; additional requests queue and time out without creating new
    # threads.
    result_queue: queue.Queue[tuple[str, list[str] | BaseException]] = queue.Queue()

    def _resolve_worker() -> None:
        try:
            result_queue.put(("ok", _resolve_hostname(hostname)))
        except BaseException as exc:
            result_queue.put(("error", exc))

    _dns_pool.submit(_resolve_worker)

    try:
        status, value = result_queue.get(timeout=timeout)
    except queue.Empty as exc:
        raise NetworkError(f"DNS resolution timeout ({timeout}s): {hostname}") from exc

    if status == "error":
        err = value
        assert isinstance(err, BaseException)
        if isinstance(err, (SSRFBlockedError, NetworkError)):
            raise err
        raise NetworkError(f"DNS resolution failed: {hostname}: {err}") from err

    ip_list: list[str] = value  # type: ignore[assignment]

    if not ip_list:
        raise NetworkError(f"DNS resolution returned no addresses: {hostname}")

    # Step 4: Validate ALL resolved IPs
    # Attacker could return a mix of safe and unsafe IPs - block if ANY is unsafe
    for ip_str in ip_list:
        _validate_ip_address(ip_str, allowed_ranges=allowed_ranges)

    # Select IP to use (prefer IPv4 for compatibility)
    ipv4_addrs = [ip for ip in ip_list if ":" not in ip]
    selected_ip = ipv4_addrs[0] if ipv4_addrs else ip_list[0]

    # Host header must include port for non-default ports (RFC 7230 §5.4).
    # SNI hostname is always bare (no port) — TLS SNI is hostname-only.
    scheme_lower = parsed.scheme.lower()
    default_port = 443 if scheme_lower == "https" else 80
    host_header = f"{hostname}:{port}" if port != default_port else hostname

    return SSRFSafeRequest(
        original_url=url,
        resolved_ip=selected_ip,
        host_header=host_header,
        port=port,
        path=path,
        scheme=scheme_lower,
        bare_hostname=hostname,
    )
