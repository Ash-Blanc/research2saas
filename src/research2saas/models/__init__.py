"""Shared data models for research2saas platform"""

from .paper import Paper, PaperCluster
from .validation import (
    ValidationStatus,
    CompetitorAnalysis,
    PatentInfo,
    FundingSignal,
    MarketValidation,
)

__all__ = [
    "Paper",
    "PaperCluster",
    "ValidationStatus",
    "CompetitorAnalysis",
    "PatentInfo",
    "FundingSignal",
    "MarketValidation",
]
