"""Evaluator — assesses source quality, identifies coverage gaps, decides whether to continue."""

from __future__ import annotations

import json
from deepresearch.config import Config
from deepresearch.llm import get_llm_client, parse_json_response
from deepresearch.state import AgentState


EVAL_SYSTEM = """\
You are a research quality evaluator. Assess the collected sources against the original
research question and identify what's still missing.

For each source, assign:
- relevance_score (0.0-1.0): How directly relevant is this to the research question?
- credibility: "high" (academic, government, established org), "medium" (reputable media, expert blogs),
  "low" (forums, unknown sources, thin content)

Then identify coverage gaps — aspects of the research question that haven't been
adequately addressed by current sources.

Finally, give an overall quality_score (0.0-1.0) representing how well the current
evidence base answers the original question.

Respond with ONLY valid JSON (no markdown fences):
{
  "source_assessments": [
    {
      "url": "source url",
      "relevance_score": 0.8,
      "credibility": "high"
    }
  ],
  "coverage_gaps": ["gap 1 description", "gap 2 description"],
  "quality_score": 0.6,
  "context_summary": "Brief summary of what we know so far from the sources",
  "should_continue": true
}"""


def evaluate_node(state: AgentState) -> dict:
    """Evaluate source quality and identify coverage gaps."""
    config = Config()
    invoke, _ = get_llm_client(config)

    sources = state.get("sources", [])
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", config.max_iterations)

    if not sources:
        return {
            "coverage_gaps": ["No sources found — all searches failed"],
            "quality_score": 0.0,
            "should_continue": iteration < max_iter,
            "status": "planning" if iteration < max_iter else "synthesizing",
            "progress_log": ["⚠️ No sources to evaluate"],
        }

    # Prepare source summary for LLM (limit to avoid token overflow)
    source_summaries = []
    for s in sources[:40]:  # cap at 40 sources for context window
        source_summaries.append({
            "url": s["url"],
            "title": s["title"],
            "snippet": s["snippet"][:200],
            "engine": s["engine"],
        })

    user_msg = (
        f"Original research question: {state['question']}\n\n"
        f"Iteration: {iteration}/{max_iter}\n\n"
        f"Collected sources ({len(sources)} total, showing up to 40):\n"
        f"{json.dumps(source_summaries, indent=2)}"
    )

    result = invoke(EVAL_SYSTEM, user_msg, temperature=0.2)
    parsed = parse_json_response(result)

    # Apply assessments back to sources
    assessment_map = {a["url"]: a for a in parsed.get("source_assessments", [])}
    updated_sources = []
    for s in sources:
        assessment = assessment_map.get(s["url"], {})
        updated_s = {**s}
        if "relevance_score" in assessment:
            updated_s["relevance_score"] = assessment["relevance_score"]
        if "credibility" in assessment:
            updated_s["credibility"] = assessment["credibility"]
        updated_sources.append(updated_s)

    quality_score = parsed.get("quality_score", 0.5)
    coverage_gaps = parsed.get("coverage_gaps", [])
    context_summary = parsed.get("context_summary", "")

    # Decide whether to continue searching
    should_continue_llm = parsed.get("should_continue", False)
    should_continue = (
        should_continue_llm
        and iteration < max_iter
        and quality_score < 0.85
        and len(coverage_gaps) > 0
    )

    # Update plan with context summary
    plan = state.get("plan", {})
    plan["context_summary"] = context_summary

    next_status = "planning" if should_continue else "synthesizing"

    high_quality = sum(1 for s in updated_sources if s.get("relevance_score", 0) >= 0.7)

    return {
        "sources": updated_sources,
        "coverage_gaps": coverage_gaps,
        "quality_score": quality_score,
        "should_continue": should_continue,
        "plan": plan,
        "status": next_status,
        "progress_log": [
            f"📊 Evaluation: quality={quality_score:.0%}, "
            f"{high_quality}/{len(updated_sources)} high-relevance sources, "
            f"{len(coverage_gaps)} gaps identified"
            + (" → continuing research" if should_continue else " → synthesizing report")
        ],
    }
