from __future__ import annotations
 
import traceback
import time
import urllib.request
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
 
 
# ---------------------------------------------------------------------------
# Failure record — one per provider attempt
# ---------------------------------------------------------------------------
 
@dataclass
class ProviderFailure:
    provider_name: str
    error_type: str        # e.g. "HTTPError", "TimeoutError", "JSONDecodeError"
    error_msg: str         # short human-readable message
    traceback_str: str     # full traceback for debugging
 
 
# ---------------------------------------------------------------------------
# DiagnosticResult — returned when every provider fails
# ---------------------------------------------------------------------------
 
@dataclass
class DiagnosticResult:
    """
    Returned instead of raising an exception when all providers fail.
    Pass this to your LLM prompt so Buddy can explain the situation.
    """
    failures: List[ProviderFailure]
    timestamp: float = field(default_factory=time.time)
 
    def as_buddy_explanation(self) -> str:
        """
        Plain-English explanation Buddy can speak aloud.
        Keeps it short and natural — no stack traces in the TTS output.
        """
        if not self.failures:
            return "I couldn't fetch the latest news, though I'm not sure why."
 
        parts = []
        for f in self.failures:
            name = f.provider_name
            err = f.error_type
            if "Timeout" in err or "timeout" in f.error_msg.lower():
                parts.append(f"{name} timed out")
            elif "401" in f.error_msg or "403" in f.error_msg or "API key" in f.error_msg:
                parts.append(f"{name} rejected the request — the API key may have expired")
            elif "404" in f.error_msg:
                parts.append(f"{name} returned a not-found error")
            elif "JSONDecodeError" in err or "ParseError" in err:
                parts.append(f"{name} sent back a malformed response")
            elif "ConnectionRefused" in err or "Network" in err:
                parts.append(f"{name} couldn't be reached — possible network issue")
            else:
                parts.append(f"{name} failed with {err}")
 
        joined = "; ".join(parts)
        return (
            f"I tried {len(self.failures)} news source{'s' if len(self.failures) != 1 else ''} "
            f"but none of them worked right now: {joined}. "
            f"I'll try again next time."
        )
 
    def as_debug_summary(self) -> str:
        """Full diagnostic dump for your terminal logs."""
        lines = [f"[NEWS DIAGNOSTIC] All {len(self.failures)} provider(s) failed:"]
        for i, f in enumerate(self.failures, 1):
            lines.append(f"  [{i}] {f.provider_name}: {f.error_type} — {f.error_msg}")
            lines.append(f"      Traceback:\n{f.traceback_str}")
        return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# Individual providers
# ---------------------------------------------------------------------------
 
class GNewsProvider:
    """
    Uses the GNews API (free tier: 100 req/day).
    Set GNEWS_API_KEY in your environment or pass api_key directly.
    https://gnews.io
    """
    name = "GNews"
 
    def __init__(self, api_key: str, timeout: int = 5):
        self.api_key = api_key
        self.timeout = timeout
 
    def fetch_headlines(self, *, locale: str = "de", limit: int = 3) -> List[str]:
        lang = locale.split("-")[0]  # "de-DE" -> "de"
        url = (
            f"https://gnews.io/api/v4/top-headlines"
            f"?lang={lang}&max={limit}&apikey={self.api_key}"
        )
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return [a["title"] for a in data.get("articles", [])[:limit]]
 
 
class NewsAPIProvider:
    """
    Uses newsapi.org (free tier: 100 req/day, headlines only).
    https://newsapi.org
    """
    name = "NewsAPI"
 
    def __init__(self, api_key: str, timeout: int = 5):
        self.api_key = api_key
        self.timeout = timeout
 
    def fetch_headlines(self, *, locale: str = "de", limit: int = 3) -> List[str]:
        country = locale.split("-")[-1].lower()  # "de-DE" -> "de"
        url = (
            f"https://newsapi.org/v2/top-headlines"
            f"?country={country}&pageSize={limit}&apiKey={self.api_key}"
        )
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return [a["title"] for a in data.get("articles", [])[:limit]]
 
 
class RSSProvider:
    """
    Parses a public RSS feed — no API key required.
    Defaults to Deutsche Welle (German) and BBC (English) depending on locale.
    """
    name = "RSS"
 
    FEEDS: Dict[str, str] = {
        "de": "https://rss.dw.com/rdf/rss-de-all",
        "en": "https://feeds.bbci.co.uk/news/rss.xml",
        "default": "https://feeds.bbci.co.uk/news/rss.xml",
    }
 
    def __init__(self, feed_url: Optional[str] = None, timeout: int = 6):
        self.feed_url = feed_url  # override if you want a specific feed
        self.timeout = timeout
 
    def fetch_headlines(self, *, locale: str = "de-DE", limit: int = 3) -> List[str]:
        lang = locale.split("-")[0].lower()
        url = self.feed_url or self.FEEDS.get(lang, self.FEEDS["default"])
 
        req = urllib.request.Request(url, headers={"User-Agent": "BuddyBot/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            xml_bytes = resp.read()
 
        root = ET.fromstring(xml_bytes)
        titles: List[str] = []
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(title_el.text.strip())
            if len(titles) >= limit:
                break
        return titles
 
 
class HackerNewsProvider:
    """
    Last-resort: pulls top HN stories (tech-heavy, no locale, no key needed).
    """
    name = "HackerNews"
 
    def __init__(self, timeout: int = 6):
        self.timeout = timeout
 
    def fetch_headlines(self, *, locale: str = "en", limit: int = 3) -> List[str]:
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            ids = json.loads(resp.read().decode())[:limit]
 
        titles = []
        for story_id in ids:
            item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            with urllib.request.urlopen(item_url, timeout=self.timeout) as resp:
                item = json.loads(resp.read().decode())
            titles.append(item.get("title", "Untitled"))
        return titles
 
 
# ---------------------------------------------------------------------------
# FallbackNewsProvider — the main class you plug into ExploreInternet
# ---------------------------------------------------------------------------
 
class FallbackNewsProvider:
    """
    Tries each provider in order.
 
    On success: returns the headlines list as normal.
    On total failure: returns a single-item list containing the
    DiagnosticResult.as_buddy_explanation() string so Buddy can
    speak it naturally — no exception propagates to the assistant loop.
 
    The DiagnosticResult is also stored as self.last_diagnostic so
    you can log or inspect it after the fact.
    """
 
    def __init__(self, providers: List[Any]):
        self.providers = providers
        self.last_diagnostic: Optional[DiagnosticResult] = None
 
    def fetch_headlines(self, *, locale: str = "de-DE", limit: int = 3) -> List[str]:
        failures: List[ProviderFailure] = []
 
        for provider in self.providers:
            name = getattr(provider, "name", type(provider).__name__)
            try:
                results = provider.fetch_headlines(locale=locale, limit=limit)
                if results:
                    self.last_diagnostic = None
                    return results
                # provider returned empty — treat as soft failure
                failures.append(ProviderFailure(
                    provider_name=name,
                    error_type="EmptyResponse",
                    error_msg="Provider returned an empty list",
                    traceback_str="(no exception)",
                ))
            except Exception as exc:
                tb = traceback.format_exc()
                failures.append(ProviderFailure(
                    provider_name=name,
                    error_type=type(exc).__name__,
                    error_msg=str(exc),
                    traceback_str=tb,
                ))
 
        # All providers failed
        diagnostic = DiagnosticResult(failures=failures)
        self.last_diagnostic = diagnostic
 
        print(diagnostic.as_debug_summary())  # goes to your terminal log
 
        # Return the human explanation as a "headline" so the existing
        # format_news_brief() path still works without modification.
        return [f"[News unavailable] {diagnostic.as_buddy_explanation()}"]
 
 
# ---------------------------------------------------------------------------
# Factory — call this in explore_internet.py
# ---------------------------------------------------------------------------
 
def build_news_provider(
    gnews_api_key: Optional[str] = None,
    newsapi_key: Optional[str] = None,
    rss_feed_url: Optional[str] = None,
) -> FallbackNewsProvider:
    """
    Builds the provider chain from whatever keys are available.
    Providers with missing keys are skipped gracefully.
 
    Example (in explore_internet.py or config.py):
        import os
        news_provider = build_news_provider(
            gnews_api_key=os.getenv("GNEWS_API_KEY"),
            newsapi_key=os.getenv("NEWSAPI_KEY"),
        )
        explorer = ExploreInternet(config, news_provider=news_provider)
    """
    providers: List[Any] = []
 
    if gnews_api_key:
        providers.append(GNewsProvider(api_key=gnews_api_key))
 
    if newsapi_key:
        providers.append(NewsAPIProvider(api_key=newsapi_key))
 
    # RSS needs no key — always add it
    providers.append(RSSProvider(feed_url=rss_feed_url))
 
    # HackerNews as absolute last resort
    providers.append(HackerNewsProvider())
 
    return FallbackNewsProvider(providers=providers)