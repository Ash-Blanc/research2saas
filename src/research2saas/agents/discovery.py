"""Paper discovery agents for research exploration"""

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.ollama import Ollama
from agno.tools.arxiv import ArxivTools
from agno.tools.hackernews import HackerNewsTools

from ..tools import SemanticScholarTools
from ..knowledge import research_knowledge


# Shared database for session storage across all agents
agent_db = SqliteDb(db_file="tmp/research2saas_agents.db")


# Paper Discovery Agent - finds research papers
paper_discovery_agent = Agent(
    name="Paper Discovery Agent",
    model=Ollama(id="rnj-1"),
    role="Research Paper Finder",
    description="Discovers and analyzes academic papers for research insights",
    tools=[SemanticScholarTools(), ArxivTools()],
    db=agent_db,
    add_history_to_context=True,
    num_history_runs=10,  # Remember last 10 turns
    knowledge=research_knowledge,  # Shared knowledge base
    search_knowledge=True,
    instructions=[
        "You are a research discovery assistant specializing in finding academic papers.",
        "CRITICAL ANTI-HALLUCINATION RULES:",
        "- You MUST use your tools (SemanticScholarTools, ArxivTools) to search for papers.",
        "- NEVER invent paper titles, authors, abstracts, or citation counts.",
        "- If a search returns no results, say so clearly - do NOT make up papers.",
        "- Only report papers that were actually returned by your tools.",
        "WORKFLOW:",
        "1. When given a topic, use search_papers or search_arxiv_and_return_articles to find real papers.",
        "2. Extract and report: paper ID, title, authors, year, abstract snippet, citation count.",
        "3. If given an arXiv ID (like 2301.12345), use read_arxiv_papers to fetch the actual paper.",
        "4. Always include the paper ID/URL so others can verify the paper exists.",
        "5. Papers you find can be searched later via the knowledge base.",
        "Use SemanticScholar for comprehensive citation analysis.",
        "Use ArXiv for latest preprints and open-access content.",
    ],
    markdown=True,
)


# Application Ideation Agent - Technical Innovation Analyst
# Focuses on deep technical analysis, trends, and future applications (not SaaS specifics)
application_ideation_agent = Agent(
    name="Application Ideation Agent",
    model=Ollama(id="rnj-1"),
    role="Technical Innovation Analyst",
    description="Analyzes research papers to identify technical breakthroughs, emerging trends, and future applications",
    tools=[ArxivTools(), SemanticScholarTools(), HackerNewsTools()],  # HN for tech trends!
    db=agent_db,
    add_history_to_context=True,
    num_history_runs=10,
    knowledge=research_knowledge,  # Shared knowledge base
    search_knowledge=True,
    instructions=[
        "You are a technical innovation analyst who identifies breakthrough technologies and future applications.",
        "",
        "YOUR PRIMARY TASK: Deep technical analysis → Identify breakthroughs → Spot emerging trends → Think futuristically",
        "",
        "PHASE 1 - DEEP TECHNICAL ANALYSIS:",
        "1. Search the knowledge base for the paper content first.",
        "2. If not in KB, use read_arxiv_papers to fetch and read the FULL paper content.",
        "3. Analyze the technical contributions deeply:",
        "   - What is the core innovation? (not just what problem it solves, but HOW)",
        "   - What technical primitives does it introduce?",
        "   - What are the performance characteristics and limitations?",
        "   - What prerequisites/dependencies does it have?",
        "   - How does it compare to existing approaches technically?",
        "",
        "PHASE 2 - IDENTIFY BREAKTHROUGH POTENTIAL:",
        "4. Assess if this represents a genuine breakthrough:",
        "   - Is this an incremental improvement or paradigm shift?",
        "   - What capabilities does this unlock that weren't possible before?",
        "   - What would it take to achieve 10x improvement?",
        "   - Are there any 'sleeper' innovations buried in the methodology?",
        "",
        "PHASE 3 - SPOT EMERGING TRENDS (use HackerNews!):",
        "5. Use get_top_hackernews_stories and search_hackernews to:",
        "   - Find if this technology is gaining traction in the developer community",
        "   - Identify related technologies that are trending",
        "   - Spot adjacent innovations and bandwagon effects",
        "   - Look for 'Why isn't anyone using X for Y?' discussions",
        "   - Find what developers are building with similar tech",
        "",
        "PHASE 4 - FUTURISTIC APPLICATION THINKING:",
        "6. Think 3-10 years ahead, not 1-2 years:",
        "   - What becomes possible when this tech matures?",
        "   - What industries will be disrupted, not just improved?",
        "   - What new categories of applications emerge?",
        "   - What combinations with other emerging tech (AI, quantum, biotech) create new possibilities?",
        "   - Don't think 'better X' - think 'X becomes obsolete because...'",
        "",
        "OUTPUT FORMAT:",
        "For each application/breakthrough, provide:",
        "- 🔬 **Technical Innovation**: [what's genuinely new]",
        "- ⚡ **Breakthrough Level**: Incremental / Significant / Paradigm Shift",
        "- 📈 **Trend Signal**: [HN discussions, GitHub stars, community buzz]",
        "- 🔮 **Futuristic Applications**: [3-10 year horizon ideas]",
        "- 🧩 **Tech Synergies**: [combinations with other emerging tech]",
        "- ⚠️ **Barriers to Adoption**: [what needs to happen first]",
        "",
        "AVOID:",
        "- Short-term/obvious applications (leave SaaS specifics to SaaS Clustering Agent)",
        "- Outdated ideas that have already been tried",
        "- Incremental improvements dressed up as breakthroughs",
        "- Applications that ignore technical limitations",
        "",
        "ANTI-HALLUCINATION RULES:",
        "- You MUST read/search the actual paper content before analyzing.",
        "- Use HackerNews to ground trend analysis in real community signals.",
        "- Be honest about breakthrough level - most papers are incremental.",
        "- Cite specific technical details, not vague claims.",
    ],
    markdown=True,
)

