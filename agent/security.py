"""Login check and SSRF-safe URL fetching.

A public app that fetches user-supplied URLs must never be tricked into calling
internal addresses (localhost, cloud metadata, private networks). Every URL and
every redirect hop is validated before a request is made.
"""
from __future__ import annotations

import hmac
import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests

from .config import Settings

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,31}$")
ALLOWED_PORTS = {None, 80, 443}
ALLOWED_CONTENT = ("text/html", "application/xhtml+xml", "text/plain", "application/pdf")
USER_AGENT = "Mozilla/5.0 (compatible; BrandContentAgent/1.0; +research-demo)"
MAX_REDIRECTS = 4


class UrlBlocked(ValueError):
    """The URL points somewhere we refuse to fetch."""


class FetchError(RuntimeError):
    """The fetch failed for a non-security reason (HTTP error, timeout, size, type)."""


# ----------------------------------------------------------------------------
# login
# ----------------------------------------------------------------------------
def normalize_username(raw: str | None) -> str | None:
    if not raw:
        return None
    name = re.sub(r"\s+", "-", raw.strip().lower())
    return name if USERNAME_RE.match(name) else None


def check_login(username: str, password: str, settings: Settings) -> bool:
    """Gradio auth callback: any valid display name + the shared access code."""
    if normalize_username(username) is None:
        return False
    name = normalize_username(username)
    if settings.users:  # individual accounts: each person has their own password
        expected = settings.users.get(name)
        ok = expected is not None and hmac.compare_digest((password or "").encode(), expected.encode())
        if not ok:
            time.sleep(0.5)
        return ok
    if not settings.access_code:
        return settings.allow_no_auth
    ok = hmac.compare_digest(password.encode(), settings.access_code.encode())
    if not ok:
        time.sleep(0.5)  # blunt brute-force slow-down
    return ok


# ----------------------------------------------------------------------------
# SSRF guard
# ----------------------------------------------------------------------------
def _is_public(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_url(url: str) -> str:
    """Return the cleaned URL or raise UrlBlocked."""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UrlBlocked("Only http(s) URLs are allowed.")
    host = parsed.hostname
    if not host:
        raise UrlBlocked("URL has no host.")
    if parsed.username or parsed.password:
        raise UrlBlocked("URLs with credentials are not allowed.")
    try:
        port = parsed.port
    except ValueError:
        raise UrlBlocked("Invalid port.")
    if port not in ALLOWED_PORTS:
        raise UrlBlocked("Only ports 80 and 443 are allowed.")
    try:
        infos = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise UrlBlocked(f"Could not resolve host '{host}'.")
    if not infos:
        raise UrlBlocked(f"Could not resolve host '{host}'.")
    for info in infos:
        if not _is_public(info[4][0]):
            raise UrlBlocked("That address is not a public internet host.")
    return url


@dataclass
class FetchResult:
    url: str
    content_type: str
    data: bytes


def safe_get(url: str, settings: Settings, session: requests.Session | None = None) -> FetchResult:
    """GET a public URL with redirect re-validation, size cap and type allow-list."""
    sess = session or requests.Session()
    current = validate_url(url)
    for _ in range(MAX_REDIRECTS + 1):
        try:
            resp = sess.get(
                current,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain,application/pdf;q=0.9,*/*;q=0.1"},
                timeout=(5, settings.fetch_timeout),
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise FetchError(f"Could not fetch: {exc.__class__.__name__}")
        try:
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location")
                if not loc:
                    raise FetchError("Redirect without Location header.")
                current = validate_url(urljoin(current, loc))
                continue
            if resp.status_code >= 400:
                raise FetchError(f"HTTP {resp.status_code}")
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype and not any(ctype.startswith(a) for a in ALLOWED_CONTENT):
                raise FetchError(f"Unsupported content type: {ctype}")
            chunks, total = [], 0
            for block in resp.iter_content(65536):
                total += len(block)
                if total > settings.max_fetch_bytes:
                    raise FetchError("Page is too large.")
                chunks.append(block)
            return FetchResult(url=current, content_type=ctype or "text/html", data=b"".join(chunks))
        finally:
            resp.close()
    raise FetchError("Too many redirects.")
