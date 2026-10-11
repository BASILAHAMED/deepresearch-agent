"""Searcher — executes sub-queries across multiple SerpApi engines."""

from __future__ import annotations

from urllib.parse import urlparse

import serpapi
from deepresearch.config import Config
from deepresearch.state import AgentState


# ── Engine-specific search parameters ───────────────────────────────────────

ENGINE_PARAMS = {
    "google": {
        "engine": "google",
        "num": 10,
        "gl": "in",
        "hl": "en",
    },
    "google_scholar": {
        "engine": "google_scholar",
        "num": 10,
    },
    "google_news": {
        "engine": "google_news",
        "gl": "in",
        "hl": "en",
    },
    "google_patents": {
        "engine": "google_patents",
    },
    "google_trends": {
        "engine": "google_trends",
        "data_type": "TIMESERIES",
        "geo": "IN",
    },
}

# Cap results per engine query (google_news returns 100+ uncapped)
MAX_RESULTS_PER_QUERY = 10


def _extract_sources(results: dict, engine: str, query: str) -> list[dict]:
    """Extract normalized sources from SerpApi response based on engine type."""
    sources = []

    if engine == "google_trends":
        interest = results.get("interest_over_time", {})
        timeline = interest.get("timeline_data", [])
        if timeline:
            values = []
            for point in timeline[-6:]:
                for v in point.get("values", []):
                    values.append(f"{point.get('date', '?')}: {v.get('value', '?')}")
            sources.append({
                "title": f"Google Trends: {query}",
                "url": f"https://trends.google.com/trends/explore?q={query}&geo=IN",
                "snippet": f"Search interest trend (last data points): {'; '.join(values[:6])}",
                "engine": "google_trends",
                "query": query,
                "relevance_score": 0.0,
                "credibility": "high",
            })
        return sources

    # Standard organic results (+ news_results for google_news engine)
    organic = results.get("organic_results", [])
    if engine == "google_news":
        organic = results.get("news_results", organic)

    # Cap results to avoid flooding
    organic = organic[:MAX_RESULTS_PER_QUERY]

    for r in organic:
        title = r.get("title", "")
        snippet = r.get("snippet", r.get("publication_info", {}).get("summary", ""))
        url = r.get("link", r.get("result_id", ""))

        # Skip sources with no useful content
        if not title or not url:
            continue

        sources.append({
            "title": title,
            "url": url,
            "snippet": snippet or title,
            "engine": engine,
            "query": query,
            "relevance_score": 0.0,
            "credibility": "unknown",
        })

    # Knowledge graph if present (Google)
    kg = results.get("knowledge_graph", {})
    if kg and kg.get("title"):
        desc = kg.get("description", "")
        sources.append({
            "title": f"[Knowledge Graph] {kg['title']}",
            "url": kg.get("source", {}).get("link", ""),
            "snippet": desc[:500] if desc else kg.get("title", ""),
            "engine": engine,
            "query": query,
            "relevance_score": 0.0,
            "credibility": "high",
        })

    # AI Overview / featured snippet
    ai_overview = results.get("ai_overview", {})
    if ai_overview:
        text_blocks = ai_overview.get("text_blocks", [])
        ai_text = " ".join(
            b.get("snippet", b.get("text", ""))
            for b in text_blocks
            if b.get("type") in ("paragraph", "text")
        )
        if ai_text:
            refs = ai_overview.get("references", [])
            ref_urls = [ref.get("link", "") for ref in refs[:3]]
            sources.append({
                "title": "[AI Overview] " + ai_text[:80],
                "url": ref_urls[0] if ref_urls else "",
                "snippet": ai_text[:500],
                "engine": f"{engine}_ai_overview",
                "query": query,
                "relevance_score": 0.0,
                "credibility": "medium",
            })

    # Scholar-specific: cited_by info
    if engine == "google_scholar":
        for r in organic:
            cited = r.get("inline_links", {}).get("cited_by", {})
            if cited:
                for s in sources:
                    if s["url"] == r.get("link", ""):
                        cite_count = cited.get("total", 0)
                        s["snippet"] += f" [Cited by {cite_count}]"
                        if cite_count > 50:
                            s["credibility"] = "high"
                        elif cite_count > 10:
                            s["credibility"] = "medium"

    # Patent-specific
    if engine == "google_patents":
        for r in organic:
            patent_id = r.get("patent_id", "")
            if patent_id:
                for s in sources:
                    if patent_id in s.get("url", ""):
                        s["snippet"] += f" [Patent: {patent_id}]"

    return sources


def _dedupe_and_filter(sources: list[dict]) -> list[dict]:
    """Remove duplicates, empty content, and cap per-domain noise."""
    seen_urls = set()
    domain_counts: dict[str, int] = {}
    filtered = []

    for s in sources:
        url = s.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        # Skip very short snippets
        if len(s.get("snippet", "")) < 20:
            continue

        # Cap 3 results per domain to reduce noise
        try:
            domain = urlparse(url).netloc
        except Exception:
            domain = url
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if domain_counts[domain] > 3:
            continue

        filtered.append(s)

    return filtered


def search_node(state: AgentState) -> dict:
    """Execute all pending sub-queries via SerpApi."""
    config = Config()
    client = serpapi.Client(api_key=config.serpapi_key)

    new_sources = []
    search_log = []
    updated_sub_queries = []

    for sq in state.get("sub_queries", []):
        if sq["status"] != "pending":
            updated_sub_queries.append(sq)
            continue

        for engine in sq["engines"]:
            base_params = ENGINE_PARAMS.get(engine, {"engine": "google"})
            params = {**base_params, "q": sq["question"]}

            try:
                results = client.search(params)
                extracted = _extract_sources(dict(results), engine, sq["question"])
                new_sources.extend(extracted)
                search_log.append(
                    f"✅ [{engine}] \"{sq['question']}\" → {len(extracted)} results"
                )
            except Exception as e:
                search_log.append(
                    f"⚠️ [{engine}] \"{sq['question']}\" → Error: {str(e)[:100]}"
                )

        updated_sq = {**sq, "status": "searched"}
        updated_sub_queries.append(updated_sq)

    # Dedupe and filter before evaluation
    raw_count = len(new_sources)
    filtered = _dedupe_and_filter(new_sources)

    return {
        "sources": filtered,
        "sub_queries": updated_sub_queries,
        "search_log": search_log,
        "status": "evaluating",
        "progress_log": [
            f"🔍 Search complete: {raw_count} raw → {len(filtered)} quality sources across "
            f"{len(search_log)} engine queries"
        ],
    }
