"""
Research-to-SaaS Discovery Platform
Transform academic research into validated SaaS product opportunities

Features:
- Semantic Scholar integration for paper discovery
- Citation graph analysis for research clustering
- Market validation for SaaS ideas
- End-to-end workflows from paper to product
"""

__version__ = "0.1.0"

# Configuration
from .config import get_settings, Settings

# Models
from .models import (
    Paper,
    PaperCluster,
    ValidationStatus,
    CompetitorAnalysis,
    PatentInfo,
    FundingSignal,
    MarketValidation,
)

# Tools
from .tools import SemanticScholarTools, SemanticScholarToolsSync

# Analysis
from .analysis import CitationGraphAnalyzer, MarketValidator

# Workflows
from .workflows import IdeaToSaaSWorkflow, SaaSToImprovementWorkflow

__all__ = [
    # Version
    "__version__",
    # Config
    "get_settings",
    "Settings",
    # Models
    "Paper",
    "PaperCluster",
    "ValidationStatus",
    "CompetitorAnalysis",
    "PatentInfo",
    "FundingSignal",
    "MarketValidation",
    # Tools
    "SemanticScholarTools",
    "SemanticScholarToolsSync",
    # Analysis
    "CitationGraphAnalyzer",
    "MarketValidator",
    # Workflows
    "IdeaToSaaSWorkflow",
    "SaaSToImprovementWorkflow",
]
