import re
from urllib.parse import urlparse


class WebFollowupHelper:
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def llm_summarize_search_results(self, query: str, results: list[dict]) -> str:
        compact = []
        for i, r in enumerate(results[:5], start=1):
            compact.append(
                f"Result {i}:\n"
                f"Title: {r.get('title', '(untitled)')}\n"
                f"Snippet: {r.get('snippet', '')}\n"
                f"Source: {self.extract_source_name(r.get('url', ''))}"
            )

        joined = "\n\n".join(compact)

        prompt = f"""
    You are helping present the first response after a web search.

    User search query:
    {query}

    Top search results:
    {joined}

    Write one short paragraph that does all of the following:
    1. Briefly summarize what the results suggest overall.
    2. Identify which result seems most useful for directly answering the user's question.
    3. Give a quick recap of 2 or 3 top results by mentioning their source names and why they are useful.

    Rules:
    - Focus on the user's intent.
    - If the user is asking for a current price, current number, latest value, or live information, prefer direct price or live-tracker sources over background news.
    - Be concise and natural.
    - Do not invent facts.
    - Do not print full URLs.
    - Do not use bullet points.
    - Do not ask a generic follow-up question like "Would you like me to..."
    - Return only the final answer text.
    """
        fallback = (
            f"I found several useful results for {query}. "
            f"One of the top results looks like the best direct source, while others provide background and related context."
        )
        return self._run_llm(prompt, "Web Search Intro", fallback, max_sentences=4)

    def extract_source_name(self, url: str) -> str:
        if not url:
            return "Unknown source"

        try:
            netloc = urlparse(url).netloc.lower().replace("www.", "")
            domain = netloc.split(":")[0]

            known = {
                "bbc.com": "BBC",
                "bbc.co.uk": "BBC",
                "reuters.com": "Reuters",
                "cnn.com": "CNN",
                "nytimes.com": "The New York Times",
                "theguardian.com": "The Guardian",
                "wikipedia.org": "Wikipedia",
                "github.com": "GitHub",
            }

            for k, v in known.items():
                if domain == k or domain.endswith("." + k):
                    return v

            parts = [p for p in domain.split(".") if p]
            if len(parts) >= 2:
                main = parts[-2]
            elif parts:
                main = parts[0]
            else:
                return "Unknown source"

            return main.replace("-", " ").title()

        except Exception:
            return "Unknown source"

    def build_web_result_payload(self, result: dict, n: int, user_input: str, query: str = "") -> dict:
        title = (result.get("title") or "(untitled)").strip()
        snippet = (result.get("snippet") or "").strip()
        url = (result.get("url") or "").strip()
        source = self.extract_source_name(url)

        return {
            "result_number": n,
            "user_request": user_input,
            "query": query,
            "title": title,
            "snippet": snippet,
            "source": source,
            "url": url,
        }

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        text = text.replace("**", "").replace("__", "")
        text = re.sub(r"\n{3,}", "\n\n", text)

        junk_prefixes = [
            "here is a concise summary:",
            "here’s a concise summary:",
            "summary:",
            "here is a short comparison:",
            "here’s a short comparison:",
            "comparison:",
            "based on the information provided,",
            "based only on the information provided,",
        ]

        lower = text.lower()
        for prefix in junk_prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip(" \n:-")
                break

        # remove generic assistant-y endings
        ending_markers = [
            "would you like me to",
            "let me know if you want",
            "i can also help",
            "i can help you",
        ]

        lower = text.lower()
        cut_positions = [lower.find(marker) for marker in ending_markers if lower.find(marker) != -1]
        if cut_positions:
            text = text[:min(cut_positions)].rstrip(" \n")

        return text.strip()

    def _trim_to_sentence_limit(self, text: str, max_sentences: int = 3) -> str:
        text = (text or "").strip()
        if not text:
            return text

        parts = re.split(r'(?<=[.!?])\s+', text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= max_sentences:
            return " ".join(parts)

        return " ".join(parts[:max_sentences]).strip()

    def _run_llm(self, prompt: str, mode: str, fallback: str, max_sentences: int = 3) -> str:
        try:
            out = self.llm_fn(prompt, mode)
        except Exception:
            return fallback

        if isinstance(out, dict):
            out = out.get("response") or out.get("text") or str(out)

        out = self._normalize_text(str(out))
        out = self._trim_to_sentence_limit(out, max_sentences=max_sentences)

        return out or fallback

    def llm_summarize_web_result(self, payload: dict, page_text: str = "") -> str:
        extra = ""
        if page_text:
            extra = f"""

    Page content:
    {page_text}
    """

        prompt = f"""
    You are helping summarize a saved web search result for the user.

    User request:
    {payload['user_request']}

    Search query:
    {payload['query']}

    Result number:
    {payload['result_number']}

    Source:
    {payload['source']}

    Title:
    {payload['title']}

    Snippet:
    {payload['snippet']}
    {extra}

    Write a concise, natural summary based on the information above.
    Prefer the page content if it is available.
    Do not invent facts.
    Do not print the full URL.
    If the information is limited, say so.
    Keep the answer to 2 or 3 sentences.
    Return only the final answer text.
    """
        fallback = f"It looks like result {payload['result_number']} from {payload['source']} is about {payload['title']}."
        return self._run_llm(prompt, "Web Followup", fallback, max_sentences=3)

    def llm_compare_web_results(self, payload_a: dict, payload_b: dict, user_input: str) -> str:
        prompt = f"""
You are comparing two saved web search results for the user.

User request:
{user_input}

Result 1:
- Number: {payload_a['result_number']}
- Source: {payload_a['source']}
- Title: {payload_a['title']}
- Snippet: {payload_a['snippet']}

Result 2:
- Number: {payload_b['result_number']}
- Source: {payload_b['source']}
- Title: {payload_b['title']}
- Snippet: {payload_b['snippet']}

Give a short, grounded comparison of:
1. what each result seems to cover
2. the likely difference in angle or focus
3. which one the user may want to open next

Do not invent facts.
Do not print the full URL.
Keep the answer brief and natural.
"""
        fallback = (
            f"Result {payload_a['result_number']} seems to focus on {payload_a['title']}, "
            f"while result {payload_b['result_number']} seems to focus on {payload_b['title']}."
        )
        return self._run_llm(prompt, "Web Compare", fallback, max_sentences=4)