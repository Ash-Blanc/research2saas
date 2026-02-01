"""
Research-to-SaaS AgentOS Server
Run this to start the backend API for Agno UI
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agno.os import AgentOS
from agno.team import Team
from agno.db.sqlite import SqliteDb
from agno.models.ollama import Ollama

# Import agents from the package
from research2saas.agents import (
    paper_discovery_agent,
    application_ideation_agent,  # Merged from Application Research + Brainstormer
    saas_clustering_agent,
    market_validation_agent,
)

# Import shared knowledge base
from research2saas.knowledge import research_knowledge

# Shared database for session storage
team_db = SqliteDb(db_file="tmp/research2saas_team.db")

# Create a team that coordinates all agents with proper instructions
research_to_saas_team = Team(
    name="Research-to-SaaS Team",
    description="A team that discovers research papers and transforms them into validated SaaS product ideas",
    model=Ollama(id="rnj-1"),  # Team leader model
    db=team_db,
    add_history_to_context=True,
    num_history_runs=10,  # Remember last 10 turns
    # Shared knowledge base for all agents to access paper content
    knowledge=research_knowledge,
    search_knowledge=True,
    members=[
        paper_discovery_agent,
        application_ideation_agent,
        saas_clustering_agent,
        market_validation_agent,
    ],
    # Team leader instructions for coordination
    instructions=[
        "You are the team leader coordinating research-to-SaaS discovery.",
        "CRITICAL: You must delegate tasks to the appropriate team members - do NOT answer questions yourself.",
        "KNOWLEDGE BASE: Papers discovered by the team are stored in a shared knowledge base. Search it for paper content.",
        "WORKFLOW:",
        "1. When given a research topic or paper ID, FIRST delegate to Paper Discovery Agent to find real papers.",
        "2. WAIT for Paper Discovery Agent to return actual paper data before proceeding.",
        "3. Then delegate to Application Ideation Agent to derive applications AND generate SaaS product ideas.",
        "4. For multiple papers, use SaaS Clustering Agent to identify patterns across research clusters.",
        "5. Finally delegate to Market Validation Agent to validate the most promising ideas.",
        "ANTI-HALLUCINATION RULES:",
        "- NEVER invent or imagine paper titles, authors, or content.",
        "- NEVER proceed to ideation without real paper data from the discovery agent.",
        "- If Paper Discovery Agent finds no papers, tell the user to try a different query.",
        "- All ideas MUST be grounded in actual research paper content.",
        "Share member responses so each agent can build on previous work.",
    ],
    share_member_interactions=True,  # Members see each other's work
    show_members_responses=True,  # Show what each member contributes
    markdown=True,
)

# Create AgentOS with both individual agents and the team
agent_os = AgentOS(
    agents=[
        paper_discovery_agent,
        application_ideation_agent,
        saas_clustering_agent,
        market_validation_agent,
    ],
    teams=[research_to_saas_team],
    name="Research-to-SaaS Platform",
    description="Transform research papers into validated SaaS product ideas",
)

# FastAPI app for uvicorn
app = agent_os.get_app()

if __name__ == "__main__":
    # Start the server (module path must match this filename)
    agent_os.serve("server:app")

