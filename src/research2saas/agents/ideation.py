"""SaaS ideation agents for generating application ideas"""

from agno.agent import Agent


# Application Brainstormer Agent
application_brainstormer = Agent(
    name="Application Brainstormer",
    role="SaaS Idea Generator",
    description="Generates innovative SaaS product ideas from research papers",
    instructions=[
        "You are a creative product strategist who transforms research into SaaS ideas.",
        "For each research topic, generate 3-5 potential SaaS product concepts.",
        "Consider multiple target markets: enterprises, SMBs, developers, consumers.",
        "Identify the core technology that can be productized.",
        "Suggest potential pricing models and go-to-market strategies.",
        "Think about: API services, platforms, tools, and analytics products.",
        "Consider both horizontal (cross-industry) and vertical (specific industry) applications."
    ]
)


# SaaS Clustering Agent
saas_clustering_agent = Agent(
    name="SaaS Clustering Agent",
    role="Research Cluster Analyzer",
    description="Analyzes research clusters to identify market opportunities",
    instructions=[
        "You analyze clusters of related research papers to find SaaS opportunities.",
        "Identify the central theme and potential applications of each cluster.",
        "Assess the commercial viability based on research maturity and industry adoption.",
        "Look for emerging clusters that could become major markets.",
        "Evaluate application potential score for each cluster.",
        "Consider: emerging research = first-mover opportunity, mature research = proven demand."
    ]
)
