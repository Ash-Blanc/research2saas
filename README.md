# Research-to-SaaS Discovery Platform

> Transform research papers into validated SaaS ideas using AI agents, citation graph analysis, and market validation

## 🎯 Overview

This platform bridges the gap between academic research and commercial applications by:

1. **Finding relevant papers** using Semantic Scholar API with async, rate-limited requests
2. **Building citation graphs** to discover research clusters and evolution paths
3. **Generating application ideas** from theoretical research
4. **Clustering ideas** into coherent SaaS products
5. **Validating against market** reality (competitors, funding, patents)

## 🏗️ Package Structure

```
src/research2saas/
├── __init__.py          # Public API exports
├── config.py            # Centralized settings (pydantic-settings)
├── models/              # Shared Pydantic models
│   ├── paper.py         # Paper, PaperCluster
│   └── validation.py    # MarketValidation, CompetitorAnalysis
├── tools/               # Agno toolkits
│   └── semantic_scholar.py  # SemanticScholarTools (async, rate-limited)
├── analysis/            # Analysis engines
│   ├── citation_graph.py    # CitationGraphAnalyzer
│   └── market_validator.py  # MarketValidator
├── agents/              # Agno agent definitions
│   ├── discovery.py     # paper_discovery_agent
│   ├── ideation.py      # application_brainstormer
│   └── validation.py    # market_validation_agent
└── workflows/           # End-to-end pipelines
    ├── idea_to_saas.py       # Paper → SaaS Concepts
    └── saas_to_improvement.py # SaaS → Research Improvements
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Ash-Blanc/research2saas
cd research2saas

# Install dependencies with uv (recommended)
uv sync

# Or install in editable mode
uv pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (optional - free tier works fine)
```

**Adding new dependencies:**
```bash
# Use uv add (updates pyproject.toml and lockfile automatically)
uv add package-name

# For dev dependencies
uv add --dev package-name
```

### Usage

```python
from research2saas import (
    Paper,
    SemanticScholarTools,
    CitationGraphAnalyzer,
    IdeaToSaaSWorkflow,
    get_settings,
)

# Use Semantic Scholar tools directly
tools = SemanticScholarTools()
paper = await tools.get_paper("arXiv:1706.03762")  # Attention Is All You Need
lineage = await tools.build_research_lineage(paper["id"])

# Run the full workflow
workflow = IdeaToSaaSWorkflow()
result = await workflow.run(seed_paper_id="arXiv:1706.03762")
print(f"SaaS Concepts: {len(result.saas_concepts)}")
```

### Running the UI and Backend

The platform provides an AgentOS backend that you can connect to the Agno UI:

#### Option 1: Use Online Agno UI (Recommended)

```bash
# 1. Start the backend server (runs on port 7777)
uv run python server.py

# 2. In your browser, go to: https://os.agno.com
# 3. Sign in and click "Add new OS"
# 4. Select "Local" and enter: http://localhost:7777
# 5. Name it "Research-to-SaaS" and click "Connect"
```

#### Option 2: Run Local Agent UI

```bash
# 1. Start the backend
uv run python server.py

# 2. In a new terminal, install and run Agent UI
npx create-agent-ui@latest
cd agent-ui
npm run dev

# 3. Open browser to: http://localhost:3000
# 4. Connect to backend: http://localhost:7777
```

**Available Agents in UI:**
- 🔍 **Paper Discovery Agent** - Find research papers via ArXiv and Semantic Scholar
- 🔬 **Application Research Agent** - Discover practical applications of theory
- 💡 **Application Brainstormer** - Generate SaaS product ideas
- 🎯 **SaaS Clustering Agent** - Organize ideas into coherent products
- ✅ **Market Validation Agent** - Validate against real market data


## 🔧 Key Components

### SemanticScholarTools

Async-first toolkit for paper discovery:
- **Rate limiting**: Token bucket with automatic retry (free tier: 100 req/5min)
- **Caching**: LRU cache with 1-hour TTL
- **Batch operations**: Fetch up to 500 papers in one call
- **ML recommendations**: Native Semantic Scholar recommendations

### CitationGraphAnalyzer

NetworkX-based graph analysis:
- Community detection (Louvain algorithm)
- PageRank & betweenness centrality
- Application pathway finding
- Temporal trend analysis

### MarketValidator

Market validation for SaaS ideas:
- Competitor discovery
- Patent risk assessment
- Funding signal detection
- Market size estimation

## 📊 Configuration

Environment variables (all optional):

```bash
# Semantic Scholar API (optional - generous free tier available)
S2_API_KEY=your_key_here

# LLM Provider
MISTRAL_API_KEY=your_key_here

# Cache settings
S2_CACHE_TTL=3600
S2_CACHE_SIZE=1000
```

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Built with [Agno](https://agno.com) - Multi-agent framework
- [Semantic Scholar](https://www.semanticscholar.org) - Paper discovery API
- [arXiv](https://arxiv.org) - Open access research papers