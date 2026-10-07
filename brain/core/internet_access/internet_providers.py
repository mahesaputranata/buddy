# internet_providers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

import requests

try:
    import feedparser
except Exception:
    feedparser = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# ----------------------------
# Helpers
# ----------------------------

@dataclass
class ProviderConfig:
    timeout_s: int = 8
    user_agent: str = "Buddy/1.0 (+local assistant)"
    # For Open-Meteo geocoding: choose results in this language (optional)
    geocode_language: str = "en"


def _session(cfg: ProviderConfig) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": cfg.user_agent})
    return s


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


# ----------------------------
# Open-Meteo Weather Provider (no API key)
# ----------------------------

class OpenMeteoWeatherProvider:
    """
    Uses:
      - Geocoding: https://geocoding-api.open-meteo.com/v1/search
      - Forecast:  https://api.open-meteo.com/v1/forecast

    Returns dict keys expected by ExploreInternet.format_weather_brief().
    """

    def __init__(self, cfg: Optional[ProviderConfig] = None):
        self.cfg = cfg or ProviderConfig()
        self.sess = _session(self.cfg)

    def _geocode(self, location: str) -> Optional[Dict[str, Any]]:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": location,
            "count": 1,
            "language": self.cfg.geocode_language,
            "format": "json",
        }
        r = self.sess.get(url, params=params, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()
        print("[GEOCODE DEBUG]", location, "->", data.get("results"))
        results = data.get("results") or []
        return results[0] if results else None

    @staticmethod
    def _wx_code_to_text(code: Optional[int]) -> str:
        # Open-Meteo WMO weather interpretation codes.
        # This mapping is intentionally compact; extend if you want more nuance.
        if code is None:
            return "Unknown"
        mapping = {
            0: "Clear",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Rime fog",
            51: "Light drizzle",
            53: "Drizzle",
            55: "Dense drizzle",
            56: "Freezing drizzle",
            57: "Freezing drizzle",
            61: "Light rain",
            63: "Rain",
            65: "Heavy rain",
            66: "Freezing rain",
            67: "Freezing rain",
            71: "Light snow",
            73: "Snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Rain showers",
            81: "Rain showers",
            82: "Violent rain showers",
            85: "Snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm w/ hail",
            99: "Thunderstorm w/ hail",
        }
        return mapping.get(int(code), f"Weather code {code}")

    def fetch_weather(self, *, location: str) -> Dict[str, Any]:
        # Put this at class-level later if you want; fine here for now
        KNOWN_LOCATIONS = {
            "essen": (51.4556, 7.0116, "Essen, North Rhine-Westphalia, Germany"),
            "essen, nrw": (51.4556, 7.0116, "Essen, North Rhine-Westphalia, Germany"),
            "essen, germany": (51.4556, 7.0116, "Essen, North Rhine-Westphalia, Germany"),
            "bocholt": (51.8386, 6.6153, "Bocholt, North Rhine-Westphalia, Germany"),
            "bocholt, de": (51.8386, 6.6153, "Bocholt, North Rhine-Westphalia, Germany"),
            "bocholt, germany": (51.8386, 6.6153, "Bocholt, North Rhine-Westphalia, Germany"),
        }

        # 1) Try geocoding as-is
        geo = None
        try:
            geo = self._geocode(location)
        except Exception:
            geo = None

        if geo:
            lat = geo["latitude"]
            lon = geo["longitude"]
            resolved_name = ", ".join(
                [p for p in [geo.get("name"), geo.get("admin1"), geo.get("country")] if p]
            )
        else:
            # 2) Fallback: known location map
            key = location.strip().lower()
            if key in KNOWN_LOCATIONS:
                lat, lon, resolved_name = KNOWN_LOCATIONS[key]
            else:
                # 3) Try a more explicit geocode once
                geo2 = None
                try:
                    geo2 = self._geocode(location + ", Germany")
                except Exception:
                    geo2 = None

                if geo2:
                    lat = geo2["latitude"]
                    lon = geo2["longitude"]
                    resolved_name = ", ".join(
                        [p for p in [geo2.get("name"), geo2.get("admin1"), geo2.get("country")] if p]
                    )
                else:
                    # 4) Hard failure: return safe dict (NO bare return!)
                    return {
                        "condition": "Unknown",
                        "temp_c": "?",
                        "today_summary": "Location not found",
                        "high_c": "?",
                        "low_c": "?",
                        "location": location,
                        "source": "open-meteo",
                    }

        # Forecast request
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "auto",
        }
        r = self.sess.get(url, params=params, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        cur = data.get("current") or {}
        daily = data.get("daily") or {}

        temp = cur.get("temperature_2m")
        cur_code = cur.get("weather_code")
        condition = self._wx_code_to_text(cur_code)

        d_codes = daily.get("weather_code") or []
        d_max = daily.get("temperature_2m_max") or []
        d_min = daily.get("temperature_2m_min") or []
        today_code = d_codes[0] if len(d_codes) > 0 else None
        high = d_max[0] if len(d_max) > 0 else None
        low = d_min[0] if len(d_min) > 0 else None

        today_summary = self._wx_code_to_text(today_code)

        return {
            "condition": condition,
            "temp_c": temp if temp is not None else "?",
            "today_summary": today_summary,
            "high_c": high if high is not None else "?",
            "low_c": low if low is not None else "?",
            "location": resolved_name or location,
            "source": "open-meteo",
        }


# ----------------------------
# RSS News Provider (no API key)
# ----------------------------

class RSSNewsProvider:
    """
    Fetches top headlines from RSS feeds.
    Uses feedparser (recommended). If feedparser isn't available, raises.
    """

    def __init__(self, feeds: Optional[List[str]] = None, cfg: Optional[ProviderConfig] = None):
        self.cfg = cfg or ProviderConfig()
        self.sess = _session(self.cfg)

        # A reasonable default mix (you can edit these)
        # self.feeds = feeds or [
        #     # Germany / DE
        #     "https://www.tagesschau.de/xml/rss2",
        #     "https://www.dw.com/en/top-stories/s-9097?maca=en-rss-en-top-1022-rdf",
        #     # Tech
        #     "https://www.theverge.com/rss/index.xml",
        #     # World
        #     "https://feeds.bbci.co.uk/news/world/rss.xml",
        # ]
        self.feeds = feeds or [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://rss.cnn.com/rss/edition_world.rss",
            "https://www.aljazeera.com/xml/rss/all.xml",
            "https://www.reuters.com/rssFeed/worldNews",  
        ]

        if feedparser is None:
            raise RuntimeError("feedparser not installed. Run: pip install feedparser")

    def fetch_headlines(self, *, locale: str, limit: int = 3) -> List[str]:
        limit = _clamp(limit, 1, 10)


        items: List[Dict[str, Any]] = []

        for url in self.feeds:
            # feedparser can fetch itself, but we fetch with requests to control timeouts/UA
            try:
                r = self.sess.get(url, timeout=self.cfg.timeout_s)
                r.raise_for_status()
                parsed = feedparser.parse(r.content)
            except Exception:
                continue

            for e in parsed.entries[:10]:
                title = (getattr(e, "title", "") or "").strip()
                if not title:
                    continue

                # published_parsed is a time.struct_time in many feeds
                published_ts = None
                pp = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
                if pp:
                    try:
                        published_ts = int(time.mktime(pp))
                    except Exception:
                        published_ts = None

                items.append({
                    "title": title,
                    "published_ts": published_ts or 0,
                })

        # If no timestamps, this becomes “whatever order we collected”
        items.sort(key=lambda x: x["published_ts"], reverse=True)

        # De-dupe titles
        seen = set()
        headlines: List[str] = []
        for it in items:
            t = it["title"]
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            headlines.append(t)
            if len(headlines) >= limit:
                break

        return headlines


# ----------------------------
# DuckDuckGo HTML Search Provider (no API key)
# ----------------------------

# class DuckDuckGoHTMLSearchProvider:
#     """
#     Scrapes DuckDuckGo's HTML endpoint.
#     Not as stable as an official API, but simple and keyless.
#     """

#     def __init__(self, cfg: Optional[ProviderConfig] = None):
#         self.cfg = cfg or ProviderConfig()
#         self.sess = _session(self.cfg)
#         self.sess.headers.update({
#             "User-Agent": self.cfg.user_agent,
#             "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
#             "Referer": "https://duckduckgo.com/",
#         })

#     def search(self, *, query: str, limit: int = 5) -> List[Dict[str, str]]:
#         limit = _clamp(limit, 1, 10)
#         q = (query or "").strip()
#         if not q:
#             return []

#         url = "https://lite.duckduckgo.com/lite/"
#         r = self.sess.post(url, data={"q": q}, timeout=self.cfg.timeout_s)
#         r.raise_for_status()

#         if BeautifulSoup is None:
#             return [{"title": f"Results for: {q}", "snippet": "Install beautifulsoup4 for rich parsing.", "url": ""}]

#         soup = BeautifulSoup(r.text, "html.parser")
#         results: List[Dict[str, str]] = []

#         # Results are in <a rel="nofollow" class="result-link" ...> sometimes,
#         # but structure can vary. This approach is robust:
#         for a in soup.select("a"):
#             href = a.get("href") or ""
#             title = a.get_text(" ", strip=True)
#             if not title or not href:
#                 continue

#             # DDG lite includes many links; filter for external ones
#             if "duckduckgo.com" in href:
#                 continue
#             if not (href.startswith("http://") or href.startswith("https://")):
#                 continue

#             results.append({"title": title, "snippet": "", "url": href})
#             if len(results) >= limit:
#                 break

#         return results

class DuckDuckGoHTMLSearchProvider:
    def __init__(self, cfg: Optional[ProviderConfig] = None):
        self.cfg = cfg or ProviderConfig()
        self.sess = _session(self.cfg)
        self.sess.headers.update({
            "User-Agent": self.cfg.user_agent,
            "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
            "Referer": "https://duckduckgo.com/",
        })

    def search(self, *, query: str, limit: int = 5) -> List[Dict[str, str]]:
        limit = _clamp(limit, 1, 10)
        q = (query or "").strip()
        if not q:
            return []

        url = "https://lite.duckduckgo.com/lite/"
        r = self.sess.post(url, data={"q": q}, timeout=self.cfg.timeout_s)
        r.raise_for_status()

        if BeautifulSoup is None:
            return [{"title": f"Results for: {q}", "snippet": "Install beautifulsoup4.", "url": ""}]

        soup = BeautifulSoup(r.text, "html.parser")

        results: List[Dict[str, str]] = []

        # DDG lite often uses rows: link row then snippet row
        rows = soup.select("table tr")
        i = 0
        while i < len(rows) and len(results) < limit:
            row = rows[i]
            a = row.find("a", href=True)
            if not a:
                i += 1
                continue

            href = a["href"].strip()
            title = a.get_text(" ", strip=True)

            # skip internal / junk
            if not title or not href:
                i += 1
                continue
            if "duckduckgo.com" in href:
                i += 1
                continue
            if not (href.startswith("http://") or href.startswith("https://")):
                i += 1
                continue

            snippet = ""
            # snippet is often in the next row
            if i + 1 < len(rows):
                snip_txt = rows[i + 1].get_text(" ", strip=True)
                # avoid capturing navigation garbage
                if snip_txt and len(snip_txt) > 20 and "http" not in snip_txt.lower():
                    snippet = snip_txt

            results.append({"title": title, "snippet": snippet, "url": href})
            i += 2

        return results