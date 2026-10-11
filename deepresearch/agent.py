"""LangGraph agent — orchestrates the plan→search→evaluate→synthesize loop."""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from deepresearch.state import AgentState
from deepresearch.planner import plan_node
from deepresearch.searcher import search_node
from deepresearch.evaluator import evaluate_node
from deepresearch.synthesizer import synthesize_node


def _route_after_evaluate(state: AgentState) -> str:
    """Decide whether to re-plan (more searching) or synthesize."""
    if state.get("should_continue", False):
        return "planner"
    return "synthesizer"


def build_graph() -> StateGraph:
    """Construct and compile the research agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", plan_node)
    graph.add_node("searcher", search_node)
    graph.add_node("evaluator", evaluate_node)
    graph.add_node("synthesizer", synthesize_node)

    # Define edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "evaluator")

    # Conditional: evaluate → re-plan or synthesize
    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluate,
        {
            "planner": "planner",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_research(
    question: str,
    max_iterations: int = 3,
    callback=None,
) -> dict:
    """Run the full research pipeline.

    Args:
        question: The research question to investigate.
        max_iterations: Maximum plan→search→evaluate loops.
        callback: Optional function called with (node_name, state) after each step.

    Returns:
        Final agent state with report and metadata.
    """
    graph = build_graph()

    initial_state = {
        "question": question,
        "plan": {},
        "sub_queries": [],
        "sources": [],
        "search_log": [],
        "coverage_gaps": [],
        "quality_score": 0.0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_continue": True,
        "report": "",
        "report_metadata": {},
        "status": "planning",
        "progress_log": [f"🚀 Starting research: \"{question}\""],
    }

    if callback:
        callback("init", initial_state)

    # Stream through the graph to get per-node updates
    final_state = None
    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if callback and node_output:
                # Merge into running state for callback
                if final_state is None:
                    final_state = {**initial_state, **node_output}
                else:
                    final_state.update(node_output)
                callback(node_name, final_state)

    # Get final state
    if final_state is None:
        final_state = graph.invoke(initial_state)

    return final_state
