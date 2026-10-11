# 🔬 DeepResearch Agent

**Autonomous multi-step research agent powered by SerpApi live search data + LLM reasoning.**

> Built for the [SerpApi India Hackathon 2026](https://serpapi.github.io/serpapi-india-hackathon-2026/) — AI Agents Track

DeepResearch Agent takes a complex research question and autonomously:
1. **Plans** — Decomposes it into focused sub-queries, selecting the optimal SerpApi engine for each
2. **Searches** — Executes queries across 5 SerpApi engines (Google, Scholar, News, Patents, Trends)
3. **Evaluates** — Assesses source quality, relevance, and credibility; identifies coverage gaps
4. **Iterates** — Generates new queries to fill gaps (up to N iterations)
5. **Synthesizes** — Produces a comprehensive, cited research report with confidence assessments

Unlike simple search-and-summarize tools, DeepResearch Agent **plans its own search strategy**, **evaluates what it doesn't know yet**, and **iterates** until coverage is sufficient — mimicking how a skilled human researcher works.

![DeepResearch Agent](https://img.shields.io/badge/Track-AI_Agents-blue) ![Python](https://img.shields.io/badge/Python-3.11+-green) ![SerpApi](https://img.shields.io/badge/SerpApi-5_Engines-orange)

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Engine Search** | Queries Google Search, Scholar, News, Patents, and Trends via SerpApi — choosing the right engine for each sub-question |
| **Adaptive Planning** | LLM decomposes the question, then re-plans based on what's missing after each search round |
| **Source Evaluation** | Each source is scored for relevance (0-1) and credibility (high/medium/low) |
| **Iterative Deepening** | Up to 3 research iterations, stopping early when quality threshold is met |
| **Cited Reports** | Final report includes inline citations, a reference list, and a confidence assessment |
| **AI Overview Extraction** | Captures Google's AI Overview snippets and knowledge graph data when available |
| **Gap Analysis** | Explicitly reports what couldn't be fully answered and why |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Question                         │
└──────────────────────┬──────────────────────────────────┘
                       ▼
              ┌────────────────┐
              │    PLANNER     │ ← LLM decomposes into sub-queries
              │  (LangGraph)   │   + selects target engines
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │   SEARCHER     │ ← Executes via SerpApi
              │  Google Search │   • google (web)
              │  Google Scholar│   • google_scholar
              │  Google News   │   • google_news
              │  Google Patents│   • google_patents
              │  Google Trends │   • google_trends
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │   EVALUATOR    │ ← LLM assesses quality
              │  Scores sources│   + identifies gaps
              │  Finds gaps    │   + decides: continue or stop?
              └───────┬────────┘
                      │
                ┌─────┴─────┐
                ▼           ▼
         [Gaps found]  [Sufficient]
           ▼                ▼
    Back to PLANNER   ┌────────────────┐
    (new queries)     │  SYNTHESIZER   │ ← LLM generates cited report
                      │  Cross-refs    │   with confidence scores
                      │  Cites sources │
                      └────────────────┘
                              ▼
                     📄 Research Report
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A free [SerpApi account](https://serpapi.com/users/sign_up?plan=free) (250 credits/month)
- An [OpenAI](https://platform.openai.com/) or [Anthropic](https://console.anthropic.com/) API key

### Installation

```bash
# Clone the repository
git clone https://github.com/BASILAHAMED/deepresearch-agent.git
cd deepresearch-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run the Streamlit UI

```bash
streamlit run app.py
```

### Run from CLI

```bash
# Basic usage
python -m deepresearch.cli "What is the current state of sodium-ion batteries for EVs in India?"

# With options
python -m deepresearch.cli "How are Indian startups using AI in agriculture?" \
    --max-iterations 3 \
    --output report.md

# Full JSON output
python -m deepresearch.cli "Impact of UPI on financial inclusion" --json
```

## 🔍 SerpApi Integration — Deep & Meaningful

DeepResearch Agent doesn't just call SerpApi once — it makes **10-25+ meaningful API calls per research question** across **5 different engines**, with each call driven by the agent's own planning:

| Engine | What It Provides | When The Agent Uses It |
|---|---|---|
| `google` | Web results, Knowledge Graph, AI Overview | General facts, overviews, definitions |
| `google_scholar` | Academic papers with citation counts | Scientific claims, statistics, studies |
| `google_news` | Recent news articles | Current events, announcements, trends |
| `google_patents` | Patent filings and claims | Technology landscape, innovation |
| `google_trends` | Search interest over time | Popularity, seasonal patterns |

The agent also extracts:
- **Google Knowledge Graph** data when available
- **AI Overview** snippets with source references
- **Citation counts** from Google Scholar for credibility scoring
- **Patent IDs** from Google Patents

## 📊 Example Output

**Question:** *"What is the current evidence on microplastics in Indian drinking water sources?"*

**Agent behavior:**
```
🚀 Starting research
📋 Plan: 5 sub-queries across 4 engines
🔍 Search: [google_scholar] "microplastics drinking water India study" → 10 results
🔍 Search: [google_news] "microplastics water contamination India 2024 2025" → 8 results
🔍 Search: [google] "microplastics health effects drinking water" → 10 results
🔍 Search: [google_scholar] "microplastics removal water treatment India" → 10 results
🔍 Search: [google_trends] "microplastics India" → 1 trend result
📊 Evaluation: quality=62%, 28/39 relevant, 3 gaps
🔄 Re-plan: 3 new queries to fill gaps
🔍 Search: [google_scholar] "microplastics concentration levels Indian rivers" → 10 results
🔍 Search: [google_news] "India microplastics regulation policy" → 7 results
🔍 Search: [google] "microplastics testing kits India consumer" → 9 results
📊 Evaluation: quality=84%, 45/64 relevant, 1 gap
✅ Report: 30 sources cited, 4 engines, 2 iterations
```

## ⚙️ Configuration

| Environment Variable | Required | Default | Description |
|---|---|---|---|
| `SERPAPI_KEY` | ✅ | — | Your SerpApi API key |
| `LLM_PROVIDER` | ❌ | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | If using OpenAI | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | — | Anthropic API key |

## 🧪 How It Differs From Existing Tools

| Feature | Simple RAG / Search Bot | DeepResearch Agent |
|---|---|---|
| Search strategy | Fixed queries | LLM-planned, adaptive |
| Engines | 1 (usually Google) | 5 (Google, Scholar, News, Patents, Trends) |
| Iteration | None | Up to 3 rounds with gap analysis |
| Source evaluation | None | Per-source relevance + credibility scoring |
| Output | Raw summary | Structured report with citations & confidence |
| Coverage awareness | No | Yes — identifies what it couldn't find |

## 🛠️ Tech Stack

- **Python 3.11+** — Core language
- **LangGraph** — Agent orchestration with stateful graph execution
- **SerpApi** — Live search data from 5 Google engines
- **OpenAI / Anthropic** — LLM reasoning (planning, evaluation, synthesis)
- **Streamlit** — Interactive web UI

## 📝 AI Tools Disclosure

This project was developed with assistance from AI coding tools (Hermes Agent). The architecture, prompts, and logic were designed by the team, with AI assisting in code generation and refinement.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👥 Team

Built by **GenOS Team** for the SerpApi India Hackathon 2026.
