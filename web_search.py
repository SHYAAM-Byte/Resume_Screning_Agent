"""
web_search.py

Enterprise AI Research & Decision Intelligence Agent
------------------------------------------------------
Reusable web research module.

Responsibilities (and ONLY these):
    - Accept a research query and search multiple web sources for
      relevant, structured information.
    - Normalize every provider's raw output into a common
      (title, url, source, snippet) shape.
    - Remove duplicate results across providers.
    - Rank results, giving a modest preference to known authoritative
      domains, while otherwise preserving each provider's own
      relevance ordering.
    - Fail gracefully: a provider outage degrades the result set
      instead of raising out of `search()`.

This module does NOT implement:
    - LLM reasoning / synthesis        (see ai_engine.py)
    - Document parsing / RAG           (see rag.py)
    - Workflow orchestration           (see research_agent.py)
    - Any UI                           (see app.py)

Isolation
----------
This module is intentionally self-contained: it imports only the
standard library and third-party search/HTTP libraries, never any
sibling module in this project (research_agent.py, ai_engine.py,
prompts.py, utils.py, app.py). It can be developed, tested, or swapped
out independently of the rest of the application.

Public interface (what research_agent.py calls)
--------------------------------------------------
    class WebSearchClient:
        def search(self, query: str, max_results: int = 6) -> list[dict]

    Each returned dict has the keys:
        "title"   : str  -- page/article title
        "url"     : str  -- source URL
        "source"  : str  -- human-readable publisher/domain label
        "snippet" : str  -- relevant excerpt / summary text

A module-level `search(query, max_results=6)` convenience function is
also provided for callers that don't need a configured client.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote, urlparse

import requests
from ddgs import DDGS
from ddgs.exceptions import (
    RatelimitException,
    TimeoutException,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SearchResult",
    "SearchProvider",
    "DuckDuckGoProvider",
    "WikipediaProvider",
    "WebSearchClient",
    "search",
]

DEFAULT_MAX_RESULTS = 6
DEFAULT_PER_PROVIDER_RESULTS = 8
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 10

# Domains treated as more authoritative than an average web page. This
# is a deliberately simple, transparent heuristic -- not a claim of
# factual correctness -- used only to break ties in ranking.
_AUTHORITATIVE_DOMAIN_MARKERS: Tuple[str, ...] = (
    ".gov",
    ".edu",
    "wikipedia.org",
    "who.int",
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "un.org",
    "nist.gov",
    "sec.gov",
    "reuters.com",
    "bloomberg.com",
    "nature.com",
    "sciencedirect.com",
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SearchResult:
    """A single, normalized web search result."""

    title: str
    url: str
    source: str
    snippet: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "snippet": self.snippet,
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _domain_label(url: str) -> str:
    """Human-readable source label derived from a URL's domain."""
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return "Web"
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or "Web"


def _normalize_url(url: str) -> str:
    """Collapse scheme/case/trailing-slash/query differences so the
    same page isn't counted twice across providers."""
    try:
        parsed = urlparse(url.strip().lower())
    except ValueError:
        return url.strip().lower()
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc}{path}"


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text or "").strip()


def _authority_score(url: str) -> int:
    """1 if the URL's domain matches a known-authoritative marker, else 0."""
    netloc = urlparse(url).netloc.lower()
    return 1 if any(marker in netloc for marker in _AUTHORITATIVE_DOMAIN_MARKERS) else 0


# ----------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------
class SearchProvider(ABC):
    """A single web source. Implementations must never raise out of
    `search()` for ordinary failure modes (network errors, rate
    limits, empty results) -- they should log and return an empty
    list instead, so one provider's outage never breaks the others.
    """

    name: str = "generic"

    @abstractmethod
    def search(self, query: str, max_results: int) -> List[SearchResult]:
        raise NotImplementedError


class DuckDuckGoProvider(SearchProvider):
    """General web search via DuckDuckGo (no API key required)."""

    name = "DuckDuckGo"

    def __init__(self, timeout: int = DEFAULT_PROVIDER_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = ddgs.text(query, max_results=max_results) or []
        except (RatelimitException, TimeoutException) as exc:
            logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
            return []
        except Exception as exc:  # defense in depth against library changes
            logger.warning("Unexpected DuckDuckGo error for %r: %s", query, exc)
            return []

        results: List[SearchResult] = []
        for item in raw_results:
            url = (item.get("href") or "").strip()
            title = (item.get("title") or "").strip()
            if not url or not title:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    source=_domain_label(url),
                    snippet=(item.get("body") or "").strip(),
                )
            )
        return results


class WikipediaProvider(SearchProvider):
    """Encyclopedic, generally authoritative background via the public
    Wikipedia search API (no API key required).
    """

    name = "Wikipedia"
    ENDPOINT = "https://en.wikipedia.org/w/api.php"

    def __init__(self, timeout: int = DEFAULT_PROVIDER_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }
        try:
            response = requests.get(self.ENDPOINT, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Wikipedia search failed for %r: %s", query, exc)
            return []
        except ValueError as exc:
            logger.warning("Wikipedia returned invalid JSON for %r: %s", query, exc)
            return []

        hits = ((data or {}).get("query") or {}).get("search") or []
        results: List[SearchResult] = []
        for hit in hits:
            title = (hit.get("title") or "").strip()
            if not title:
                continue
            page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            results.append(
                SearchResult(
                    title=title,
                    url=page_url,
                    source="Wikipedia",
                    snippet=_strip_html(hit.get("snippet", "")),
                )
            )
        return results


# ----------------------------------------------------------------------
# Public client
# ----------------------------------------------------------------------
class WebSearchClient:
    """Reusable web research interface.

    Runs each configured provider, merges the results, removes
    duplicates, and returns a ranked, structured result set. This is
    the only entry point other modules (e.g. research_agent.py) should
    depend on -- individual providers are an implementation detail and
    may be added, removed, or reordered without changing this
    interface.
    """

    def __init__(
        self,
        providers: Optional[List[SearchProvider]] = None,
        per_provider_results: int = DEFAULT_PER_PROVIDER_RESULTS,
    ) -> None:
        self.providers: List[SearchProvider] = (
            providers if providers is not None else [DuckDuckGoProvider(), WikipediaProvider()]
        )
        self.per_provider_results = per_provider_results

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, str]]:
        """Search all configured providers for `query` and return up to
        `max_results` deduplicated, ranked results as plain dicts with
        keys: title, url, source, snippet.

        Raises:
            ValueError: if `query` is empty. Provider-level failures
                (network errors, rate limits, etc.) are handled
                internally and simply reduce the result set rather
                than raising.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("A non-empty search query is required.")
        if max_results <= 0:
            return []

        ranked_hits: List[Tuple[int, SearchResult]] = []
        next_rank = 0
        for provider in self.providers:
            try:
                provider_results = provider.search(query, self.per_provider_results)
            except Exception as exc:
                # Providers are expected to handle their own failures;
                # this is a last-resort guard so one bad provider can
                # never take down the whole search.
                logger.warning("%s provider raised unexpectedly: %s", provider.name, exc)
                continue

            for result in provider_results:
                ranked_hits.append((next_rank, result))
                next_rank += 1

        deduped = self._deduplicate(ranked_hits)
        deduped.sort(key=lambda entry: (-_authority_score(entry[1].url), entry[0]))

        return [result.to_dict() for _, result in deduped[:max_results]]

    @staticmethod
    def _deduplicate(
        ranked_hits: List[Tuple[int, SearchResult]]
    ) -> List[Tuple[int, SearchResult]]:
        seen: Set[str] = set()
        unique: List[Tuple[int, SearchResult]] = []
        for rank, result in ranked_hits:
            key = _normalize_url(result.url)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append((rank, result))
        return unique


# ----------------------------------------------------------------------
# Module-level convenience function
# ----------------------------------------------------------------------
_default_client: Optional[WebSearchClient] = None


def search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, str]]:
    """Convenience wrapper around a lazily-created, module-level
    WebSearchClient, for callers that don't need a custom provider
    configuration.
    """
    global _default_client
    if _default_client is None:
        _default_client = WebSearchClient()
    return _default_client.search(query, max_results=max_results)