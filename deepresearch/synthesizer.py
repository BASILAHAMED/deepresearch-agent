"""Synthesizer — generates the final research report with citations and confidence scores."""

from __future__ import annotations

import json
from deepresearch.config import Config
from deepresearch.llm import get_llm_client
from deepresearch.state import AgentState


SYNTH_SYSTEM = """\
You are a research synthesis expert. Generate a comprehensive, well-structured research
report based on the collected and evaluated sources.

REPORT STRUCTURE:
1. **Executive Summary** — 2-3 sentence answer to the research question
2. **Key Findings** — Main discoveries organized by theme, each citing specific sources
3. **Evidence Analysis** — Deeper analysis of the most important evidence, noting
   agreements and contradictions between sources
4. **Data & Statistics** — Any quantitative data found, with sources
5. **Gaps & Limitations** — What couldn't be fully answered, and why
6. **Conclusion & Confidence Assessment** — Final synthesis with an overall confidence
   level (High/Medium/Low) and reasoning

CITATION FORMAT:
- Use inline citations like [1], [2], etc.
- Include a numbered reference list at the end
- Only cite sources that are actually referenced in the text

QUALITY STANDARDS:
- Distinguish between well-supported claims and preliminary findings
- Note when sources contradict each other
- Be specific about what the evidence shows vs. what it suggests
- Avoid speculation beyond what sources support
- Use clear, professional language

Generate the report in MARKDOWN format."""


def synthesize_node(state: AgentState) -> dict:
    """Generate the final research report."""
    config = Config()
    invoke, _ = get_llm_client(config)

    sources = state.get("sources", [])
    quality_score = state.get("quality_score", 0.0)
    coverage_gaps = state.get("coverage_gaps", [])

    # Sort sources by relevance score, then filter to top sources
    sorted_sources = sorted(sources, key=lambda s: s.get("relevance_score", 0), reverse=True)
    top_sources = sorted_sources[:30]  # use top 30 most relevant

    # Prepare source data for LLM
    source_data = []
    for i, s in enumerate(top_sources, 1):
        source_data.append({
            "ref_num": i,
            "title": s["title"],
            "url": s["url"],
            "snippet": s["snippet"][:300],
            "engine": s["engine"],
            "relevance": s.get("relevance_score", 0),
            "credibility": s.get("credibility", "unknown"),
        })

    # Count engines used
    engines_used = list(set(s["engine"] for s in sources))
    queries_made = list(set(s["query"] for s in sources))

    user_msg = (
        f"# Research Question\n{state['question']}\n\n"
        f"# Research Process\n"
        f"- Iterations: {state.get('iteration', 1)}\n"
        f"- Total sources found: {len(sources)}\n"
        f"- Search engines used: {', '.join(engines_used)}\n"
        f"- Unique queries: {len(queries_made)}\n"
        f"- Quality score: {quality_score:.0%}\n\n"
        f"# Coverage Gaps\n{json.dumps(coverage_gaps)}\n\n"
        f"# Sources (ranked by relevance)\n{json.dumps(source_data, indent=2)}"
    )

    report = invoke(SYNTH_SYSTEM, user_msg, temperature=0.3)

    # Compile metadata
    metadata = {
        "total_sources": len(sources),
        "cited_sources": len(top_sources),
        "engines_used": engines_used,
        "queries_made": queries_made,
        "iterations": state.get("iteration", 1),
        "quality_score": quality_score,
        "coverage_gaps": coverage_gaps,
        "high_credibility_sources": sum(
            1 for s in sources if s.get("credibility") == "high"
        ),
    }

    return {
        "report": report,
        "report_metadata": metadata,
        "status": "done",
        "progress_log": [
            f"✅ Report generated: {len(top_sources)} sources cited, "
            f"{len(engines_used)} engines used across {state.get('iteration', 1)} iterations"
        ],
    }
