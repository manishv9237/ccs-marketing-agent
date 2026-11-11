"""Find brand sources on the web: search engines + Wikipedia + well-known site pages.

The user reviews the candidate list and ticks what to ingest, so noisy search
results never reach the knowledge base automatically.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from .config import Settings

log = logging.getLogger("agent.discover")

SITE_PATHS = ["/", "/about", "/about-us", "/company", "/products", "/sustainability", "/newsroom"]
BLOCKED_HOSTS = ("facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com", "youtube.com", "pinterest.com", "tiktok.com", "reddit.com", "quora.com")


@dataclass
class Candidate:
    title: str
    url: str
    snippet: str
    origin: str  # site | wikipedia | search

    @property
    def host(self) -> str:
        return (urlparse(self.url).hostname or "").removeprefix("www.")

    @property
    def label(self) -> str:
        t = re.sub(r"\s+", " ", self.title or self.url).strip()[:80]
        return f"[{self.origin}] {t} — {self.host}"


def clean_domain(raw: str) -> str:
    raw = (raw or "").strip().lower()
    raw = re.sub(r"^https?://", "", raw).split("/")[0].removeprefix("www.")
    return raw if re.match(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$", raw) else ""


def _search_ddg(query: str, n: int) -> list[Candidate]:
    from ddgs import DDGS  # lazy import; optional dependency

    out = []
    for r in DDGS().text(query, max_results=n) or []:
        out.append(Candidate(r.get("title", ""), r.get("href", ""), r.get("body", ""), "search"))
    return out


def _search_tavily(query: str, n: int, key: str) -> list[Candidate]:
    r = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "max_results": n}, timeout=20)
    r.raise_for_status()
    return [Candidate(x.get("title", ""), x.get("url", ""), x.get("content", "")[:200], "search") for x in r.json().get("results", [])]


def _search_brave(query: str, n: int, key: str) -> list[Candidate]:
    r = requests.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n},
                     headers={"X-Subscription-Token": key, "Accept": "application/json"}, timeout=20)
    r.raise_for_status()
    return [Candidate(x.get("title", ""), x.get("url", ""), x.get("description", "")[:200], "search") for x in r.json().get("web", {}).get("results", [])]


def _wikipedia(name: str) -> list[Candidate]:
    r = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "opensearch", "search": name, "limit": 2, "format": "json"},
                     headers={"User-Agent": "BrandContentAgent/1.0 (research-demo)"}, timeout=15)
    r.raise_for_status()
    _, titles, descs, urls = r.json()
    return [Candidate(t, u, d, "wikipedia") for t, d, u in zip(titles, descs, urls)]


def pick_provider(settings: Settings) -> str:
    if settings.search_provider != "auto":
        return settings.search_provider
    if settings.tavily_api_key:
        return "tavily"
    if settings.brave_api_key:
        return "brave"
    return "ddg"


def discover_sources(brand_name: str, industry: str, domain: str, settings: Settings, max_results: int = 15,
                     search_fn=None, wiki_fn=None) -> tuple[list[Candidate], list[str]]:
    """Return (candidates, warnings). search_fn/wiki_fn are injectable for tests."""
    warnings: list[str] = []
    cands: list[Candidate] = []
    dom = clean_domain(domain)

    if dom:
        cands += [Candidate(f"{brand_name} — {p if p != '/' else 'home'}", f"https://{dom}{p}", "", "site") for p in SITE_PATHS]

    wiki = wiki_fn or _wikipedia
    try:
        cands += wiki(brand_name)
    except Exception as exc:
        warnings.append(f"Wikipedia lookup failed ({exc.__class__.__name__}).")

    provider = pick_provider(settings)
    if provider != "none":
        if search_fn is None:
            search_fn = {
                "ddg": lambda q, n: _search_ddg(q, n),
                "tavily": lambda q, n: _search_tavily(q, n, settings.tavily_api_key),
                "brave": lambda q, n: _search_brave(q, n, settings.brave_api_key),
            }.get(provider)
        queries = [f"{brand_name} official site about us", f"{brand_name} brand mission values", f"{brand_name} products services {industry}".strip(), f"{brand_name} press release news"]
        got_any = False
        for q in queries:
            try:
                cands += search_fn(q, 6) if search_fn else []
                got_any = True
            except Exception as exc:
                warnings.append(f"Web search unavailable via '{provider}' ({exc.__class__.__name__}). Add a domain or paste URLs instead.")
                break
        if not got_any and not any("Web search" in w for w in warnings):
            warnings.append("Web search returned nothing.")

    seen: set[str] = set()
    out: list[Candidate] = []
    for c in cands:
        url = (c.url or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        host = (urlparse(url).hostname or "").removeprefix("www.")
        if any(host == b or host.endswith("." + b) for b in BLOCKED_HOSTS):
            continue
        key = url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out[:max_results], warnings
