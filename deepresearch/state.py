"""Agent state definition for the LangGraph research workflow."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict


# ── Source: a single piece of evidence found during search ──────────────────

@dataclass
class Source:
    """A single search result / evidence item."""
    title: str
    url: str
    snippet: str
    engine: str  # which SerpApi engine produced this
    query: str  # the query that found it
    relevance_score: float = 0.0  # 0-1, set by evaluator
    credibility: str = "unknown"  # high / medium / low / unknown

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "engine": self.engine,
            "query": self.query,
            "relevance_score": self.relevance_score,
            "credibility": self.credibility,
        }


# ── Sub-query: a decomposed part of the original research question ──────────

@dataclass
class SubQuery:
    """A single research sub-question with its target search engines."""
    question: str
    engines: list[str]  # e.g. ["google_scholar", "google_news"]
    rationale: str = ""  # why this sub-query matters
    status: str = "pending"  # pending / searched / evaluated


# ── Research plan ───────────────────────────────────────────────────────────

@dataclass
class ResearchPlan:
    """The decomposed research plan."""
    original_question: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    context_summary: str = ""  # what we know so far
    iteration: int = 0


# ── LangGraph state ─────────────────────────────────────────────────────────

def _merge_sources(existing: list[dict], new: list[dict]) -> list[dict]:
    """Deduplicate sources by URL."""
    seen = {s["url"] for s in existing}
    merged = list(existing)
    for s in new:
        if s["url"] not in seen:
            seen.add(s["url"])
            merged.append(s)
    return merged


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph research agent."""
    # Input
    question: str

    # Planning
    plan: dict  # serialized ResearchPlan
    sub_queries: list[dict]  # list of SubQuery dicts

    # Search results
    sources: Annotated[list[dict], _merge_sources]  # deduplicated sources
    search_log: Annotated[list[str], operator.add]  # log of searches performed

    # Evaluation
    coverage_gaps: list[str]  # topics not yet covered
    quality_score: float  # 0-1 overall quality

    # Iteration control
    iteration: int
    max_iterations: int
    should_continue: bool

    # Final output
    report: str
    report_metadata: dict  # citation count, engines used, etc.

    # Status / progress
    status: str  # planning / searching / evaluating / synthesizing / done
    progress_log: Annotated[list[str], operator.add]
