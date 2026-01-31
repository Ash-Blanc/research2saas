"""Paper discovery agents for research exploration"""

from agno.agent import Agent
from agno.tools.arxiv import ArxivTools

from ..tools import SemanticScholarTools


# Paper Discovery Agent - finds research papers
paper_discovery_agent = Agent(
    name="Paper Discovery Agent",
    role="Research Paper Finder",
    description="Discovers and analyzes academic papers for research insights",
    tools=[SemanticScholarTools(), ArxivTools()],
    instructions=[
        "You are a research discovery assistant specializing in finding academic papers.",
        "Use SemanticScholar for comprehensive paper search and citation analysis.",
        "Use Arxiv for latest preprints and open-access papers.",
        "Always provide paper titles, authors, years, and citation counts.",
        "Identify key papers that form the foundation of a research area.",
        "Highlight highly influential citations and emerging research trends."
    ]
)


# Application-focused research agent
application_research_agent = Agent(
    name="Application Research Agent",
    role="Research-to-Application Bridge",
    description="Finds papers that represent practical applications of research",
    tools=[SemanticScholarTools()],
    instructions=[
        "You specialize in finding practical applications of theoretical research.",
        "Look for papers with keywords like: implementation, system, framework, tool.",
        "Use find_application_papers to identify practical implementations.",
        "Identify the evolution from theory to practice in citation chains.",
        "Highlight papers that bridge academia and industry."
    ]
)
