import re
import requests
from bs4 import BeautifulSoup


class WebPageFetcher:
    def __init__(self, timeout_s: int = 10, max_chars: int = 12000):
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self.cache = {}  

    def fetch_page_text(self, url: str) -> str:
        url = (url or "").strip()
        if not url:
            return ""

        if url in self.cache:
            return self.cache[url]

        try:
            resp = requests.get(
                url,
                timeout=self.timeout_s,
                headers={
                    "User-Agent": "Buddy/1.0 (+local assistant)"
                }
            )
            resp.raise_for_status()
        except Exception:
            return ""

        text = self._extract_text_from_html(resp.text)
        text = self._clean_text(text)

        if self.max_chars and len(text) > self.max_chars:
            text = text[:self.max_chars].rsplit(" ", 1)[0].strip()

        self.cache[url] = text
        return text

    def _extract_text_from_html(self, html: str) -> str:
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        # try article-first extraction
        article = soup.find("article")
        if article:
            text = article.get_text(separator=" ", strip=True)
            if text:
                return text

        body = soup.body
        if body:
            return body.get_text(separator=" ", strip=True)

        return soup.get_text(separator=" ", strip=True)

    def _clean_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b(cookie|privacy policy|terms of use)\b", "", text, flags=re.I)
        return text.strip()