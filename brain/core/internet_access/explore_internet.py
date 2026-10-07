# explore_internet.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Tuple
import time
import os
from brain.core.internet_access.fallback_news_provider import build_news_provider



# NEWS FETCH API
GNEWS_API_KEY = "GNEWS_API_KEY"      # name of the env var that holds the key
NEW_API_KEY = "NEWSAPI_KEY"        # name of the env var that holds the key


class WeatherProvider(Protocol):
    def fetch_weather(self, *, location: str) -> Dict[str, Any]:
        """
        Return a dict with at least:
          condition: str
          temp_c: float|int
          today_summary: str
          high_c: float|int
          low_c: float|int
        """



class NewsProvider(Protocol):
    def fetch_headlines(self, *, locale: str, limit: int = 3) -> List[str]:
        """Return a list of headline strings (already clean)."""



class SearchProvider(Protocol):
    def search(self, *, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Return list of results:
          [{"title": "...", "snippet": "...", "url": "..."}, ...]
        """




class DummyWeatherProvider:
    def fetch_weather(self, *, location: str) -> Dict[str, Any]:
        return {
            "condition": "Clear",
            "temp_c": 12,
            "today_summary": "Mostly clear",
            "high_c": 14,
            "low_c": 6,
            "location": location,
            "source": "dummy",
        }


class DummyNewsProvider:
    def fetch_headlines(self, *, locale: str, limit: int = 3) -> List[str]:
        samples = [
            "Tech: New open-source model improves local inference performance.",
            "World: Markets react to fresh economic data ahead of central bank remarks.",
            "Science: Researchers publish new findings on battery longevity.",
        ]
        return samples[:limit]


class DummySearchProvider:
    def search(self, *, query: str, limit: int = 5) -> List[Dict[str, str]]:
        # Helps you develop the UI/logic without internet access.
        results = []
        for i in range(limit):
            results.append({
                "title": f"Dummy result {i+1} for '{query}'",
                "snippet": "This is a placeholder snippet. Replace DummySearchProvider later.",
                "url": f"https://example.com/search?q={query}&i={i+1}",
            })
        return results




@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self):
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        e = self._store.get(key)
        if not e:
            return None
        if time.time() >= e.expires_at:
            self._store.pop(key, None)
            return None
        return e.value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._store[key] = _CacheEntry(value=value, expires_at=time.time() + ttl_seconds)



@dataclass
class ExploreInternetConfig:
    enabled: bool = False                 # internet allowed?
    default_location: str = "Bocholt, DE"
    default_locale: str = "de-DE"
    weather_cache_ttl_s: int = 10 * 60    # 10 min
    news_cache_ttl_s: int = 15 * 60       # 15 min
    search_cache_ttl_s: int = 30 * 60     # 30 min


class ExploreInternet:


    def __init__(
        self,
        config: Optional[ExploreInternetConfig] = None,
        weather_provider: Optional[WeatherProvider] = None,
        news_provider: Optional[NewsProvider] = None,
        search_provider: Optional[SearchProvider] = None,
    ):
        self.config = config or ExploreInternetConfig()
        self.cache = TTLCache()

        # If not provided, fall back to dummy providers (safe for offline dev)
        self.weather_provider = weather_provider or DummyWeatherProvider()
        # self.news_provider = news_provider or DummyNewsProvider()
        self.search_provider = search_provider or DummySearchProvider()

        # fallback
        self.news_provider = news_provider or build_news_provider(
            gnews_api_key=os.getenv(GNEWS_API_KEY),
            newsapi_key=os.getenv(NEW_API_KEY)
        )

    # -------- formatting helpers (static) --------

    @staticmethod
    def format_weather_brief(weather: Dict[str, Any]) -> str:
        try:
            return (
                f"Weather: {weather['condition']} {weather['temp_c']}°C now. "
                f"Today: {weather['today_summary']} (high {weather['high_c']}°C / low {weather['low_c']}°C)."
            )
        except Exception:
            # never crash your assistant because of a formatting/key issue
            return "Weather: unavailable."

    @staticmethod
    def format_news_brief(headlines: List[str]) -> str:
        if not headlines:
            return "News: No headlines found."
        bullets = "\n".join([f"- {h}" for h in headlines[:3]])
        return f"Top headlines:\n{bullets}"

    @staticmethod
    def format_search_results(results: List[Dict[str, str]]) -> str:
        if not results:
            return "No results."
        lines = []
        for r in results:
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            lines.append(f"- {title}\n  {snippet}\n  {url}")
        return "\n".join(lines)

    # -------- public API --------

    def get_weather_brief(self, *, location: Optional[str] = None) -> str:
        location = location or self.config.default_location
        cache_key = f"weather:{location}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.config.enabled:
            # offline or permission denied -> use dummy provider output anyway
            weather = self.weather_provider.fetch_weather(location=location)
            brief = self.format_weather_brief(weather)
            self.cache.set(cache_key, brief, self.config.weather_cache_ttl_s)
            return brief

        # Enabled: still call provider (dummy or real)
        try:
            weather = self.weather_provider.fetch_weather(location=location)
            brief = self.format_weather_brief(weather)
        except Exception:
            brief = "Weather: unavailable."

        self.cache.set(cache_key, brief, self.config.weather_cache_ttl_s)
        return brief

    def get_news_brief(self, *, locale: Optional[str] = None, limit: int = 3) -> str:
        locale = locale or self.config.default_locale
        cache_key = f"news:{locale}:{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.config.enabled:
            headlines = self.news_provider.fetch_headlines(locale=locale, limit=limit)
            brief = self.format_news_brief(headlines)
            self.cache.set(cache_key, brief, self.config.news_cache_ttl_s)
            return brief

        try:
            headlines = self.news_provider.fetch_headlines(locale=locale, limit=limit)
            brief = self.format_news_brief(headlines)
        except Exception:
            brief = "News: unavailable."

        self.cache.set(cache_key, brief, self.config.news_cache_ttl_s)
        return brief

    def search_web(self, *, query: str, limit: int = 5) -> List[Dict[str, str]]:
        query = (query or "").strip()
        if not query:
            return []

        cache_key = f"search:{query}:{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return 

        if not self.config.enabled:
            results = self.search_provider.search(query=query, limit=limit)
            self.cache.set(cache_key, results, self.config.search_cache_ttl_s)
            return results  # ← returns dummy results when disabled

        try:
            results = self.search_provider.search(query=query, limit=limit)
        except Exception:
            results = []

        self.cache.set(cache_key, results, self.config.search_cache_ttl_s)
        return results
    
    def get_startup_briefing(
        self,
        *,
        location: Optional[str] = None,
        locale: Optional[str] = None,
        include_weather: bool = True,
        include_news: bool = True,
    ) -> str:
        parts: List[str] = []
        if include_weather:
            parts.append(self.get_weather_brief(location=location))
        if include_news:
            parts.append(self.get_news_brief(locale=locale))
        return "\n".join([p for p in parts if p])
    
    def get_startup_data(self):
        weather_raw = self.weather_provider.fetch_weather(location=self.config.default_location)
        headlines = self.news_provider.fetch_headlines(locale=self.config.default_locale, limit=3)

        return {
            "weather": weather_raw,
            "headlines": headlines
        }


if __name__ == "__main__":
    explorer = ExploreInternet(ExploreInternetConfig(enabled=False))
    print(explorer.get_startup_briefing())
    print("\nSearch demo:")
    results = explorer.search_web(query="best local speech to text python", limit=3)
    print(ExploreInternet.format_search_results(results))