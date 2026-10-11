"""Evaluator â assesses source quality, identifies coverage gaps, decides whether to continue."""

from __future__ import annotations

import json
from urllib.parse import urlparse
from deepresearch.config import Config
from deepresearch.llm import get_llm_client, parse_json_response
from deepresearch.state import AgentState


EVAL_SYSTEM = """You are a research quality evaluator. Given a research question and collected sources,
do TWO things:

1. Assess OVERALL coverage: how well do these sources together answer the research question?
2. Identify specific GAPS: what aspects of the question are NOT yet covered?

Consider:
- Do the sources address the core question directly?
- Are there multiple perspectives / viewpoints represented?
- Is there data from authoritative sources (academic, government, major news)?
- Are recent developments covered?
- What specific sub-topics still need investigation?

Respond with ONLY valid JSON (no markdown fences):
{
  "quality_score": 0.7,
  "coverage_assessment": "2-3 sentence summary of what IS well covered",
  "coverage_gaps": ["specific gap 1", "specific gap 2"],
  "context_summary": "Brief synthesis of key findings so far",
  "should_continue": true
}

QUALITY SCORE GUIDE:
- 0.3-0.4 = Sources exist but barely address the question
- 0.5-0.6 = Core question partially answered, significant gaps remain
- 0.7-0.8 = Most aspects covered with credible sources, minor gaps
- 0.85+ = Comprehensive coverage with strong, diverse evidence"""


def _compute_source_scores(sources, question):
    q_words = set(question.lower().split())
    scored = []
    for s in sources:
        score = 0.3
        title = s.get("title", "").lower()
        snippet = s.get("snippet", "").lower()
        engine = s.get("engine", "")
        combined = title + " " + snippet
        overlap = sum(1 for w in q_words if len(w) > 3 and w in combined)
        overlap_ratio = overlap / max(len([w for w in q_words if len(w) > 3]), 1)
        score += overlap_ratio * 0.4
        if engine == "google_scholar":
            score += 0.15
            if s.get("credibility") == "unknown": s["credibility"] = "medium"
        elif engine == "google_news":
            score += 0.05
            if s.get("credibility") == "unknown": s["credibility"] = "medium"
        elif engine == "google_patents":
            score += 0.1
            if s.get("credibility") == "unknown": s["credibility"] = "medium"
        elif engine == "google_trends":
            score += 0.1
            s["credibility"] = "high"
        if "[Knowledge Graph]" in s.get("title", ""):
            score += 0.2
            s["credibility"] = "high"
        if "[AI Overview]" in s.get("title", ""):
            score += 0.1
        if "Cited by" in s.get("snippet", ""):
            try:
                c = int(s["snippet"].split("Cited by ")[1].split("]")[0])
                if c > 100: score += 0.2; s["credibility"] = "high"
                elif c > 20: score += 0.1; s["credibility"] = "high"
            except: pass
        try:
            domain = urlparse(s.get("url", "")).netloc.lower()
            if any(d in domain for d in [".gov",".edu",".ac.","who.int","worldbank","nature.com","sciencedirect","springer","ieee"]):
                score += 0.15; s["credibility"] = "high"
            elif any(d in domain for d in ["reuters","bloomberg","bbc","nytimes","thehindu","livemint","economictimes"]):
                score += 0.1; s["credibility"] = "high"
        except: pass
        s["relevance_score"] = min(round(score, 2), 1.0)
        scored.append(s)
    return scored


def evaluate_node(state):
    config = Config()
    invoke, _ = get_llm_client(config)
    sources = state.get("sources", [])
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", config.max_iterations)
    if not sources:
        return {"coverage_gaps": ["No sources found"], "quality_score": 0.0,
                "should_continue": iteration < max_iter,
                "status": "planning" if iteration < max_iter else "synthesizing",
                "progress_log": ["No sources to evaluate"]}
    scored_sources = _compute_source_scores(sources, state["question"])
    sorted_sources = sorted(scored_sources, key=lambda s: s["relevance_score"], reverse=True)
    top_sources = sorted_sources[:25]
    source_summaries = [{"title": s["title"], "snippet": s["snippet"][:250],
                         "engine": s["engine"], "credibility": s.get("credibility","unknown"),
                         "relevance": s["relevance_score"]} for s in top_sources]
    engines_used = list(set(s["engine"] for s in sources))
    nl = chr(10)
    user_msg = (
        f"Research question: {state['question']}{nl}{nl}"
        f"Iteration: {iteration}/{max_iter}{nl}"
        f"Total sources: {len(sources)} | Engines: {', '.join(engines_used)}{nl}{nl}"
        f"Top {len(top_sources)} sources (ranked by relevance):{nl}"
        f"{json.dumps(source_summaries, indent=2)}"
    )
    result = invoke(EVAL_SYSTEM, user_msg, temperature=0.2)
    parsed = parse_json_response(result)
    if isinstance(parsed, list):
        parsed = {"coverage_gaps": parsed, "quality_score": 0.6, "should_continue": True}
    quality_score = parsed.get("quality_score", 0.5)
    coverage_gaps = parsed.get("coverage_gaps", [])
    context_summary = parsed.get("context_summary", parsed.get("coverage_assessment", ""))
    engine_bonus = min(len(engines_used) * 0.03, 0.15)
    high_rel_count = sum(1 for s in scored_sources if s["relevance_score"] >= 0.6)
    high_rel_bonus = min(high_rel_count * 0.01, 0.1)
    quality_score = min(round(quality_score + engine_bonus + high_rel_bonus, 2), 1.0)
    should_continue_llm = parsed.get("should_continue", False)
    should_continue = (should_continue_llm and iteration < max_iter
                       and quality_score < 0.85 and len(coverage_gaps) > 0)
    plan = state.get("plan", {})
    plan["context_summary"] = context_summary
    next_status = "planning" if should_continue else "synthesizing"
    high_quality = sum(1 for s in scored_sources if s.get("relevance_score", 0) >= 0.6)
    high_cred = sum(1 for s in scored_sources if s.get("credibility") == "high")
    return {"sources": scored_sources, "coverage_gaps": coverage_gaps,
            "quality_score": quality_score, "should_continue": should_continue,
            "plan": plan, "status": next_status,
            "progress_log": [
                f"Evaluation: quality={quality_score:.0%}, "
                f"{high_quality} relevant + {high_cred} high-credibility sources, "
                f"{len(coverage_gaps)} gaps"
                + (" -> continuing" if should_continue else " -> synthesizing")]}
