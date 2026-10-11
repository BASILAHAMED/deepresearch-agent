"""
DeepResearch Agent — Streamlit UI
Autonomous multi-step research agent powered by SerpApi + LLM reasoning.
"""

import streamlit as st
import time
import json
import os
from dotenv import load_dotenv
from deepresearch.config import Config
from deepresearch.agent import run_research

# Load .env file automatically
load_dotenv()


# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepResearch Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .source-card {
        padding: 12px; border-radius: 8px; margin: 4px 0;
        border-left: 4px solid #4CAF50;
        background: rgba(76, 175, 80, 0.05);
    }
    .source-card.medium { border-left-color: #FF9800; }
    .source-card.low { border-left-color: #f44336; }
    .metric-card {
        text-align: center; padding: 16px;
        border-radius: 8px; background: rgba(100, 100, 255, 0.05);
        border: 1px solid rgba(100, 100, 255, 0.2);
    }
    .engine-badge {
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 0.75em; margin: 2px; background: rgba(100, 100, 255, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    serpapi_key = st.text_input(
        "SerpApi Key", type="password",
        value=st.session_state.get("serpapi_key", os.getenv("SERPAPI_KEY", "")),
        help="Get your free key at serpapi.com",
    )
    if serpapi_key:
        st.session_state["serpapi_key"] = serpapi_key

    _default_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_provider = st.selectbox(
        "LLM Provider", ["openai", "anthropic"],
        index=1 if _default_provider == "anthropic" else 0,
    )

    if llm_provider == "openai":
        llm_key = st.text_input(
            "OpenAI API Key", type="password",
            value=st.session_state.get("openai_key", os.getenv("OPENAI_API_KEY", "")),
        )
        if llm_key:
            st.session_state["openai_key"] = llm_key
        llm_model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
        )
    else:
        llm_key = st.text_input(
            "Anthropic API Key", type="password",
            value=st.session_state.get("anthropic_key", os.getenv("ANTHROPIC_API_KEY", "")),
        )
        if llm_key:
            st.session_state["anthropic_key"] = llm_key
        llm_model = st.selectbox(
            "Model",
            ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-4-6"],
        )

    max_iterations = st.slider(
        "Max Research Iterations", 1, 5, 3,
        help="More iterations = deeper research but more API calls",
    )

    st.divider()
    st.caption("🔬 DeepResearch Agent v1.0.0")
    st.caption("Built for SerpApi India Hackathon 2026")

    st.divider()
    st.markdown("### 📊 How it works")
    st.markdown("""
    1. **Plan** — Decomposes your question into sub-queries
    2. **Search** — Queries 5 SerpApi engines (Google, Scholar, News, Patents, Trends)
    3. **Evaluate** — Assesses source quality & identifies gaps
    4. **Re-plan** — Generates new queries to fill gaps (iterates)
    5. **Synthesize** — Produces a cited research report
    """)


# ── Main UI ─────────────────────────────────────────────────────────────────

st.title("🔬 DeepResearch Agent")
st.markdown(
    "Autonomous multi-step research powered by **SerpApi** live search data + LLM reasoning. "
    "Ask a complex question and get a comprehensive, cited research report."
)

# Example questions
with st.expander("💡 Example research questions", expanded=False):
    examples = [
        "What is the current state of sodium-ion battery technology for electric vehicles in India?",
        "How are Indian startups using AI in agriculture, and what results have they achieved?",
        "What are the environmental and economic impacts of India's semiconductor fab plans?",
        "How does India's UPI digital payments system compare to other countries' digital payment systems?",
        "What is the current evidence on microplastics in Indian drinking water sources?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}"):
            st.session_state["question_input"] = ex

question = st.text_area(
    "🔍 Research Question",
    value=st.session_state.get("question_input", ""),
    height=100,
    placeholder="Enter a complex research question...",
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("🚀 Research", type="primary", use_container_width=True)
with col2:
    if st.session_state.get("last_report"):
        if st.button("📥 Download Report"):
            st.download_button(
                "Download as Markdown",
                st.session_state["last_report"],
                file_name="deepresearch_report.md",
                mime="text/markdown",
            )

# ── Run Research ────────────────────────────────────────────────────────────

if run_button and question:
    import os
    # Use sidebar values (which default from .env via load_dotenv)
    _serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY", "")
    _llm_key = llm_key or (os.getenv("ANTHROPIC_API_KEY", "") if llm_provider == "anthropic" else os.getenv("OPENAI_API_KEY", ""))

    os.environ["SERPAPI_KEY"] = _serpapi_key
    os.environ["LLM_PROVIDER"] = llm_provider
    if llm_provider == "openai":
        os.environ["OPENAI_API_KEY"] = _llm_key
    else:
        os.environ["ANTHROPIC_API_KEY"] = _llm_key

    config = Config()
    config.serpapi_key = _serpapi_key
    config.llm_model = llm_model
    config.llm_provider = llm_provider
    if llm_provider == "openai":
        config.openai_api_key = _llm_key
    else:
        config.anthropic_api_key = _llm_key

    errors = config.validate()
    if errors:
        for e in errors:
            st.error(f"❌ {e}")
    else:
        # Progress area
        progress_container = st.container()
        progress_bar = progress_container.progress(0, text="Starting research...")
        status_area = progress_container.empty()
        log_area = progress_container.expander("📋 Research Log", expanded=True)

        step_count = [0]
        total_steps = 3 + (max_iterations - 1) * 2  # plan+search+eval + re-plan+search per iter
        logs = []

        def ui_callback(node: str, state: dict):
            step_count[0] += 1
            progress = min(step_count[0] / total_steps, 0.95)
            step_labels = {
                "init": "🚀 Initializing...",
                "planner": "📋 Planning research strategy...",
                "searcher": "🔍 Searching across SerpApi engines...",
                "evaluator": "📊 Evaluating source quality...",
                "synthesizer": "✍️ Synthesizing research report...",
            }
            progress_bar.progress(progress, text=step_labels.get(node, f"Processing: {node}"))

            new_logs = state.get("progress_log", [])
            for log in new_logs:
                if log not in logs:
                    logs.append(log)

            with log_area:
                for log in logs:
                    st.text(log)

        try:
            result = run_research(
                question=question,
                max_iterations=max_iterations,
                callback=ui_callback,
            )

            progress_bar.progress(1.0, text="✅ Research complete!")
            time.sleep(0.5)

            # ── Display Results ─────────────────────────────────
            st.divider()

            # Metadata metrics
            meta = result.get("report_metadata", {})
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            with mcol1:
                st.metric("Total Sources", meta.get("total_sources", 0))
            with mcol2:
                st.metric("Sources Cited", meta.get("cited_sources", 0))
            with mcol3:
                st.metric("Iterations", meta.get("iterations", 0))
            with mcol4:
                st.metric("Engines Used", len(meta.get("engines_used", [])))
            with mcol5:
                qs = meta.get("quality_score", 0)
                st.metric("Quality", f"{qs:.0%}")

            # Engine badges
            engines = meta.get("engines_used", [])
            if engines:
                engine_labels = {
                    "google": "🌐 Google",
                    "google_scholar": "🎓 Scholar",
                    "google_news": "📰 News",
                    "google_patents": "📜 Patents",
                    "google_trends": "📈 Trends",
                }
                badges = " ".join(
                    f"`{engine_labels.get(e, e)}`" for e in engines
                )
                st.markdown(f"**Search engines used:** {badges}")

            st.divider()

            # Report
            st.markdown("## 📄 Research Report")
            report = result.get("report", "No report generated.")
            st.markdown(report)
            st.session_state["last_report"] = report

            # Sources explorer
            with st.expander(f"📚 All Sources ({meta.get('total_sources', 0)})", expanded=False):
                sources = result.get("sources", [])
                sorted_sources = sorted(
                    sources,
                    key=lambda s: s.get("relevance_score", 0),
                    reverse=True,
                )
                for i, s in enumerate(sorted_sources, 1):
                    cred = s.get("credibility", "unknown")
                    rel = s.get("relevance_score", 0)
                    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(cred, "⚪")
                    st.markdown(
                        f"**{i}. {s.get('title', 'Untitled')}** {emoji} "
                        f"(relevance: {rel:.0%}, credibility: {cred})\n\n"
                        f"  *Engine: {s.get('engine', '?')} | Query: \"{s.get('query', '')}\"*\n\n"
                        f"  {s.get('snippet', '')[:200]}...\n\n"
                        f"  🔗 [{s.get('url', '')}]({s.get('url', '')})"
                    )
                    st.divider()

            # Search log
            with st.expander("🔍 Search Log", expanded=False):
                for log in result.get("search_log", []):
                    st.text(log)

            # Coverage gaps
            gaps = meta.get("coverage_gaps", [])
            if gaps:
                with st.expander("⚠️ Coverage Gaps", expanded=False):
                    for gap in gaps:
                        st.warning(gap)

        except Exception as e:
            st.error(f"❌ Research failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif run_button and not question:
    st.warning("Please enter a research question.")
