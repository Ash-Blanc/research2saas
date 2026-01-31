"""
Complete Research-to-SaaS Discovery Platform using Agno
Integrates: ArXiv search, ConnectedPapers, Citation Graph Analysis, Market Validation
"""

from agno.agent import Agent
from agno.team import Team
from agno.workflow import Workflow, Step, Parallel
from agno.tools.arxiv import ArxivTools
from agno.tools.reasoning import ReasoningTools
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType
from agno.models.mistral import Mistral

from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio

# Import our custom modules
from citation_graph_clustering import CitationGraphAnalyzer, Paper, PaperCluster
from market_validator import MarketValidator, MarketValidation, ValidationStatus
from semantic_scholar_tools import SemanticScholarTools  # Ultra-fast S2 API client


# ============================================================================
# KNOWLEDGE BASES - Separate tables for different entity types
# ============================================================================

papers_kb = Knowledge(
    vector_db=PgVector(
        table_name="arxiv_papers",
        search_type=SearchType.hybrid,  # Semantic + keyword search
    ),
    description="ArXiv papers with abstracts and metadata"
)

applications_kb = Knowledge(
    vector_db=PgVector(
        table_name="application_ideas",
        search_type=SearchType.hybrid,
    ),
    description="Application ideas generated from papers"
)

saas_kb = Knowledge(
    vector_db=PgVector(
        table_name="saas_products",
        search_type=SearchType.hybrid,
    ),
    description="SaaS product ideas with market validation"
)

graph_connections_kb = Knowledge(
    vector_db=PgVector(
        table_name="graph_connections",
        search_type=SearchType.hybrid,
    ),
    description="Citation graph connections and clusters"
)


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

# 1. Paper Discovery Agent - finds relevant papers
paper_discovery_agent = Agent(
    name="Paper Discovery Agent",
    model=Mistral(id="mistral-large-latest"),
    tools=[
        ArxivTools(max_results=10),
        SemanticScholarTools()  # Replaces ConnectedPapers with faster S2 API
    ],
    knowledge=[papers_kb],
    instructions=[
        "You are an expert at finding relevant research papers.",
        "When given an idea or topic:",
        "1. Use ArxivTools to find 5-7 most relevant papers via semantic search",
        "2. For each promising paper, use ConnectedPapers to explore citation network",
        "3. Identify both foundational papers and recent developments",
        "4. Store papers in knowledge base for future reference",
        "5. Return a curated list with relevance scores and brief summaries"
    ],
    markdown=True,
    show_tool_calls=True
)


# 2. Citation Graph Agent - builds and analyzes citation networks
citation_graph_agent = Agent(
    name="Citation Graph Analyst",
    model=Mistral(id="mistral-large-latest"),
    tools=[SemanticScholarTools()],  # Async S2 client with batch operations
    knowledge=[papers_kb, graph_connections_kb],
    instructions=[
        "You analyze citation networks to find research patterns.",
        "Your tasks:",
        "1. Build citation graph from seed papers using ConnectedPapers",
        "2. Identify clusters of related research",
        "3. Find 'application pathway' papers (theory → implementation)",
        "4. Detect cross-domain bridge papers",
        "5. Track research evolution over time",
        "Store all connections in the graph_connections knowledge base."
    ],
    markdown=True
)


# 3. Application Ideation Agent - generates application ideas
application_brainstormer = Agent(
    name="Application Ideation Agent",
    model=Mistral(id="mistral-large-latest"),
    tools=[ReasoningTools()],
    knowledge=[papers_kb, applications_kb],
    instructions=[
        "You are creative at finding practical applications for research.",
        "Given research papers, generate application ideas across:",
        "- B2B SaaS products",
        "- Developer tools and APIs", 
        "- Enterprise solutions",
        "- Consumer applications",
        "- Industry-specific tools",
        "For each idea, specify:",
        "- Target user/market",
        "- Core value proposition",
        "- Key technical requirements",
        "- Potential challenges",
        "Use structured reasoning to evaluate feasibility.",
        "Store generated applications in applications_kb."
    ],
    markdown=True,
    show_tool_calls=True
)


# 4. SaaS Clustering Agent - groups and ranks SaaS ideas
saas_clustering_agent = Agent(
    name="SaaS Clustering Agent",
    model=Mistral(id="mistral-large-latest"),
    tools=[ReasoningTools()],
    knowledge=[applications_kb, saas_kb],
    instructions=[
        "You cluster and rank SaaS ideas for viability.",
        "Process:",
        "1. Group similar application ideas into coherent themes",
        "2. Eliminate redundancies and merge overlapping concepts",
        "3. Score each cluster on:",
        "   - Technical feasibility (1-10)",
        "   - Market potential (1-10)",
        "   - Time to MVP (estimate in months)",
        "   - Competitive moat strength",
        "4. Identify top 3-5 most promising clusters",
        "5. For each cluster, define:",
        "   - Core product concept",
        "   - Target market segment",
        "   - Key differentiators",
        "   - Go-to-market strategy outline",
        "Store final SaaS concepts in saas_kb."
    ],
    markdown=True,
    show_tool_calls=True,
    structured_outputs=True
)


# 5. Market Validation Agent - validates ideas against real market
market_validation_agent = Agent(
    name="Market Validation Agent",
    model=Mistral(id="mistral-large-latest"),
    tools=[],  # Will use web_search from coordinator
    knowledge=[saas_kb],
    instructions=[
        "You validate SaaS ideas against market reality.",
        "For each idea, research:",
        "1. Existing competitors (find 5-10 direct/indirect)",
        "2. Recent funding activity in this space",
        "3. Market size estimates and growth rates",
        "4. Relevant patents (check patent databases)",
        "5. Industry trends and adoption signals",
        "Produce a validation report with:",
        "- Competition analysis",
        "- Market opportunity assessment",
        "- Risk factors",
        "- Go/No-go recommendation",
        "Store validation results in saas_kb alongside ideas."
    ],
    markdown=True
)


# 6. Research Frontier Scout - finds cutting-edge work
research_frontier_agent = Agent(
    name="Research Frontier Scout",
    model=Mistral(id="mistral-large-latest"),
    tools=[ArxivTools(), SemanticScholarTools()],  # S2 find_research_frontier method
    knowledge=[papers_kb],
    instructions=[
        "You identify cutting-edge research directions.",
        "Focus on papers from the last 6-12 months.",
        "Look for:",
        "- Novel architectures and methods",
        "- Papers from top-tier conferences (NeurIPS, ICML, ICLR, CVPR)",
        "- Rapidly accumulating citations",
        "- Industry lab publications (OpenAI, DeepMind, Meta AI)",
        "Identify emerging trends before they become mainstream."
    ],
    markdown=True
)


# 7. Technology Bridge Agent - connects disparate research areas
tech_bridge_agent = Agent(
    name="Technology Bridge Agent",
    model=Mistral(id="mistral-large-latest"),
    tools=[SemanticScholarTools()],  # S2 find_cross_domain_papers method
    knowledge=[papers_kb, graph_connections_kb],
    instructions=[
        "You find cross-domain research connections.",
        "Given papers from different areas, find 'bridge papers' that:",
        "- Cite work from multiple domains",
        "- Represent novel combinations",
        "- Enable new applications",
        "Example: Vision + NLP = Vision-Language Models",
        "These bridges often represent breakthrough opportunities."
    ],
    markdown=True
)


# ============================================================================
# WORKFLOWS - Main pipelines
# ============================================================================

class IdeaToSaaSWorkflow(Workflow):
    """
    Pipeline: Raw Idea → Papers → Applications → SaaS Clusters → Market Validation
    """
    
    def __init__(self):
        super().__init__(
            name="Idea to SaaS Discovery",
            description="Transform a raw idea into validated SaaS concepts"
        )
        
        self.graph_analyzer = CitationGraphAnalyzer()
        self.market_validator = MarketValidator()
    
    async def run(self, idea: str) -> Dict:
        """Execute the full pipeline"""
        results = {
            "original_idea": idea,
            "papers": [],
            "clusters": [],
            "applications": [],
            "saas_concepts": [],
            "validations": []
        }
        
        # STEP 1: Discover relevant papers
        print("🔍 Step 1: Discovering relevant papers...")
        paper_response = await paper_discovery_agent.run(
            f"Find the 5 most relevant ArXiv papers for this idea: {idea}\n"
            f"Include both foundational work and recent developments."
        )
        results["papers"] = self._extract_papers(paper_response)
        
        # STEP 2: Build citation graph and find clusters
        print("🕸️  Step 2: Building citation graph...")
        if results["papers"]:
            seed_paper = results["papers"][0]
            # Build graph using ConnectedPapers
            # (In production, this would integrate with actual API)
            graph_response = await citation_graph_agent.run(
                f"Build citation graph starting from paper: {seed_paper['title']}\n"
                f"Identify: research clusters, application pathways, derivative works"
            )
            results["clusters"] = self._extract_clusters(graph_response)
        
        # STEP 3: Generate application ideas (parallel for each cluster)
        print("💡 Step 3: Generating application ideas...")
        app_tasks = []
        for cluster in results["clusters"][:3]:  # Top 3 clusters
            task = application_brainstormer.run(
                f"Given this research cluster theme: {cluster['theme']}\n"
                f"Key papers: {cluster.get('papers', [])}\n"
                f"Generate 5 practical application ideas spanning B2B, developer tools, and enterprise."
            )
            app_tasks.append(task)
        
        if app_tasks:
            app_responses = await asyncio.gather(*app_tasks)
            for response in app_responses:
                results["applications"].extend(self._extract_applications(response))
        
        # STEP 4: Cluster applications into SaaS concepts
        print("🎯 Step 4: Clustering into SaaS concepts...")
        if results["applications"]:
            saas_response = await saas_clustering_agent.run(
                f"Cluster these {len(results['applications'])} application ideas:\n"
                f"{results['applications']}\n\n"
                f"Group into 3-5 distinct SaaS products. Eliminate redundancy. "
                f"Rank by: feasibility, market potential, and competitive moat."
            )
            results["saas_concepts"] = self._extract_saas_concepts(saas_response)
        
        # STEP 5: Market validation (parallel for each SaaS concept)
        print("✅ Step 5: Validating against market...")
        validation_tasks = []
        for saas_concept in results["saas_concepts"][:3]:  # Top 3 concepts
            task = market_validation_agent.run(
                f"Validate this SaaS idea: {saas_concept['description']}\n"
                f"Research: competitors, funding activity, market size, patents.\n"
                f"Provide: competition analysis, risks, go/no-go recommendation."
            )
            validation_tasks.append(task)
        
        if validation_tasks:
            validation_responses = await asyncio.gather(*validation_tasks)
            results["validations"] = [
                self._extract_validation(r) for r in validation_responses
            ]
        
        # STEP 6: Generate final report
        print("📊 Generating final report...")
        results["summary"] = self._generate_summary(results)
        
        return results
    
    def _extract_papers(self, response) -> List[Dict]:
        """Extract papers from agent response"""
        # Parse response to extract paper information
        # In production, use structured outputs
        return []
    
    def _extract_clusters(self, response) -> List[Dict]:
        """Extract clusters from graph analysis"""
        return []
    
    def _extract_applications(self, response) -> List[Dict]:
        """Extract application ideas"""
        return []
    
    def _extract_saas_concepts(self, response) -> List[Dict]:
        """Extract SaaS concepts"""
        return []
    
    def _extract_validation(self, response) -> Dict:
        """Extract market validation results"""
        return {}
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate executive summary"""
        return f"""
        Research-to-SaaS Discovery Report
        
        Original Idea: {results['original_idea']}
        Papers Analyzed: {len(results['papers'])}
        Research Clusters: {len(results['clusters'])}
        Applications Generated: {len(results['applications'])}
        SaaS Concepts: {len(results['saas_concepts'])}
        Market Validations: {len(results['validations'])}
        
        Top Opportunities:
        {self._format_top_opportunities(results)}
        """
    
    def _format_top_opportunities(self, results: Dict) -> str:
        """Format top opportunities"""
        # Combine SaaS concepts with validations
        return "See detailed report above"


class SaaSToImprovementWorkflow(Workflow):
    """
    Reverse Pipeline: Existing SaaS → Papers → Improvement Ideas
    """
    
    def __init__(self):
        super().__init__(
            name="SaaS to Research-Driven Improvements",
            description="Find research that can 10x an existing SaaS product"
        )
    
    async def run(self, saas_description: str) -> Dict:
        """Execute reverse pipeline"""
        results = {
            "saas_product": saas_description,
            "technologies": [],
            "papers": [],
            "improvements": []
        }
        
        # STEP 1: Identify core technologies
        print("🔍 Step 1: Identifying core technologies...")
        tech_agent = Agent(
            name="Tech Identifier",
            model=Mistral(id="mistral-large-latest"),
            instructions=[
                "Analyze this SaaS product and identify:",
                "- Core technologies used",
                "- Key algorithms/models",
                "- Technical bottlenecks",
                "- Areas for improvement"
            ]
        )
        
        tech_response = await tech_agent.run(
            f"Analyze this SaaS: {saas_description}\n"
            f"What are the core technologies and potential bottlenecks?"
        )
        results["technologies"] = self._extract_technologies(tech_response)
        
        # STEP 2: Find relevant research for each technology
        print("📚 Step 2: Finding relevant research...")
        paper_tasks = []
        for tech in results["technologies"][:3]:  # Top 3 technologies
            task = paper_discovery_agent.run(
                f"Find recent papers (2023-2024) about: {tech}\n"
                f"Focus on: performance improvements, novel architectures, scaling"
            )
            paper_tasks.append(task)
        
        if paper_tasks:
            paper_responses = await asyncio.gather(*paper_tasks)
            for response in paper_responses:
                results["papers"].extend(self._extract_papers(response))
        
        # STEP 3: Generate specific improvement ideas
        print("💡 Step 3: Generating improvement ideas...")
        if results["papers"]:
            improvement_agent = Agent(
                name="Improvement Generator",
                model=Mistral(id="mistral-large-latest"),
                tools=[ReasoningTools()],
                instructions=[
                    "Given research papers and a SaaS product:",
                    "Generate specific, actionable improvement ideas:",
                    "- Performance optimizations (10x throughput)",
                    "- Quality improvements (better accuracy/results)",
                    "- Cost reductions (cheaper infrastructure)",
                    "- New features (enabled by research)",
                    "For each idea: expected impact, implementation effort, risks"
                ]
            )
            
            improvement_response = await improvement_agent.run(
                f"SaaS Product: {saas_description}\n"
                f"Research Papers: {results['papers']}\n\n"
                f"Generate 5-7 specific improvement ideas with impact estimates."
            )
            results["improvements"] = self._extract_improvements(improvement_response)
        
        return results
    
    def _extract_technologies(self, response) -> List[str]:
        return []
    
    def _extract_papers(self, response) -> List[Dict]:
        return []
    
    def _extract_improvements(self, response) -> List[Dict]:
        return []


# ============================================================================
# ROUTER TEAM - Intelligent pipeline selection
# ============================================================================

router_agent = Agent(
    name="Pipeline Router",
    model=Mistral(id="mistral-large-latest"),
    instructions=[
        "You route user requests to the appropriate workflow.",
        "Routes:",
        "- 'idea_to_saas': Raw idea with no technical specifics",
        "- 'saas_to_improvement': Existing SaaS/product description",
        "- 'paper_to_applications': ArXiv paper link/ID/title",
        "Return JSON: {'pipeline': 'X', 'confidence': 0.9}"
    ],
    markdown=False,
    structured_outputs=True
)


coordinator = Team(
    name="Research-to-SaaS Coordinator",
    mode="route",
    router=router_agent,
    members=[
        IdeaToSaaSWorkflow(),
        SaaSToImprovementWorkflow()
    ],
    knowledge=[papers_kb, applications_kb, saas_kb, graph_connections_kb],
    description="Intelligently routes to the right research-to-SaaS pipeline"
)


# ============================================================================
# DEPLOYMENT
# ============================================================================

def deploy_platform():
    """Deploy as production API"""
    from agno.runtime import AgentOS
    from agno.interfaces import AGUI, AISdk
    
    agent_os = AgentOS(
        name="Research-to-SaaS Platform",
        description="Transform research into validated SaaS ideas",
        workflows=[
            IdeaToSaaSWorkflow(),
            SaaSToImprovementWorkflow()
        ],
        teams=[coordinator],
        knowledge_bases=[papers_kb, applications_kb, saas_kb, graph_connections_kb],
        interfaces=[
            AGUI(theme="professional"),  # Web UI
            AISdk()  # REST API
        ],
        monitoring=True,  # Track on agno.com
        cache_enabled=True  # Cache paper lookups
    )
    
    # Production server
    agent_os.serve(
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_idea_to_saas():
    """Example: Transform idea into SaaS"""
    workflow = IdeaToSaaSWorkflow()
    
    idea = "AI system that helps developers understand and fix security vulnerabilities in code"
    
    results = await workflow.run(idea)
    
    print(f"\n{'='*80}")
    print(f"Idea: {idea}")
    print(f"{'='*80}")
    print(f"\nPapers Found: {len(results['papers'])}")
    print(f"Applications Generated: {len(results['applications'])}")
    print(f"SaaS Concepts: {len(results['saas_concepts'])}")
    print(f"\n{results['summary']}")


async def example_saas_to_improvement():
    """Example: Find research to improve existing SaaS"""
    workflow = SaaSToImprovementWorkflow()
    
    saas = """
    We're a code completion tool that uses GPT-4 to suggest code. 
    Currently we have latency issues (500ms average) and accuracy could be better.
    """
    
    results = await workflow.run(saas)
    
    print(f"\n{'='*80}")
    print(f"SaaS: {saas}")
    print(f"{'='*80}")
    print(f"\nTechnologies Identified: {len(results['technologies'])}")
    print(f"Relevant Papers: {len(results['papers'])}")
    print(f"Improvement Ideas: {len(results['improvements'])}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_idea_to_saas())
    # asyncio.run(example_saas_to_improvement())
    
    # Or deploy to production
    # deploy_platform()