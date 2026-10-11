"""Planner — decomposes a research question into sub-queries with engine targets."""

from __future__ import annotations

from deepresearch.config import Config
from deepresearch.llm import get_llm_client, parse_json_response
from deepresearch.state import AgentState


PLAN_SYSTEM = """\
You are a research planning agent. Given a complex research question, decompose it into
3-7 focused sub-queries that together would comprehensively answer the original question.

For each sub-query, select the most appropriate SerpApi search engines:
- "google": General web search — good for overviews, recent info, definitions
- "google_scholar": Academic papers, citations — for scientific claims, statistics, studies
- "google_news": Recent news articles — for current events, trends, announcements
- "google_patents": Patent search — for technology innovations, IP landscape
- "google_trends": Search interest over time — for popularity trends, seasonal patterns

Think about:
1. What factual claims need verification?
2. What data/statistics would strengthen the answer?
3. What recent developments are relevant?
4. Are there academic perspectives needed?
5. What counter-arguments or alternative views exist?

Respond with ONLY valid JSON (no markdown fences):
{
  "sub_queries": [
    {
      "question": "specific search query",
      "engines": ["google_scholar", "google"],
      "rationale": "why this sub-query matters for answering the original question"
    }
  ],
  "research_strategy": "brief description of the overall approach"
}"""

REPLAN_SYSTEM = """\
You are a research planning agent refining a research plan after initial results.

You've already searched and found some information, but there are coverage gaps.
Generate 2-4 NEW sub-queries to fill the identified gaps. Do NOT repeat previous queries.

Available search engines:
- "google": General web search
- "google_scholar": Academic papers, citations
- "google_news": Recent news articles
- "google_patents": Patent search
- "google_trends": Search interest trends

Respond with ONLY valid JSON (no markdown fences):
{
  "sub_queries": [
    {
      "question": "specific search query",
      "engines": ["google_scholar"],
      "rationale": "what gap this fills"
    }
  ]
}"""


def plan_node(state: AgentState) -> dict:
    """Decompose the question into sub-queries (initial or re-plan)."""
    config = Config()
    invoke, _ = get_llm_client(config)
    iteration = state.get("iteration", 0)

    if iteration == 0:
        # Initial planning
        user_msg = f"Research question: {state['question']}"
        result = invoke(PLAN_SYSTEM, user_msg, temperature=0.4)
        parsed = parse_json_response(result)

        sub_queries = []
        for sq in parsed["sub_queries"]:
            sub_queries.append({
                "question": sq["question"],
                "engines": sq["engines"],
                "rationale": sq.get("rationale", ""),
                "status": "pending",
            })

        return {
            "sub_queries": sub_queries,
            "plan": {
                "original_question": state["question"],
                "strategy": parsed.get("research_strategy", ""),
                "iteration": 0,
            },
            "iteration": 1,
            "status": "searching",
            "progress_log": [
                f"📋 Plan created: {len(sub_queries)} sub-queries across "
                f"{len(set(e for sq in sub_queries for e in sq['engines']))} search engines"
            ],
        }
    else:
        # Re-planning based on gaps
        gaps = state.get("coverage_gaps", [])
        previous_queries = [sq["question"] for sq in state.get("sub_queries", [])]

        user_msg = (
            f"Original question: {state['question']}\n\n"
            f"Previous queries (do NOT repeat): {previous_queries}\n\n"
            f"Coverage gaps to fill: {gaps}\n\n"
            f"Current knowledge summary: {state.get('plan', {}).get('context_summary', 'N/A')}"
        )
        result = invoke(REPLAN_SYSTEM, user_msg, temperature=0.4)
        parsed = parse_json_response(result)

        new_sub_queries = state.get("sub_queries", [])[:]
        for sq in parsed["sub_queries"]:
            new_sub_queries.append({
                "question": sq["question"],
                "engines": sq["engines"],
                "rationale": sq.get("rationale", ""),
                "status": "pending",
            })

        return {
            "sub_queries": new_sub_queries,
            "iteration": iteration + 1,
            "status": "searching",
            "progress_log": [
                f"🔄 Re-plan iteration {iteration + 1}: {len(parsed['sub_queries'])} new sub-queries to fill gaps"
            ],
        }
