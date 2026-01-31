"""Market validation models"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ValidationStatus(str, Enum):
    """Market validation status"""
    CLEAR_OPPORTUNITY = "clear_opportunity"
    CROWDED_MARKET = "crowded_market"
    EMERGING_SPACE = "emerging_space"
    NEEDS_RESEARCH = "needs_research"
    HIGH_RISK = "high_risk"


class CompetitorAnalysis(BaseModel):
    """Analysis of existing competitor"""
    name: str
    url: str
    description: str = ""
    funding: Optional[str] = None
    founded_year: Optional[int] = None
    market_position: str = "challenger"  # "leader", "challenger", "niche"
    similarity_score: float = 0.0


class PatentInfo(BaseModel):
    """Patent information"""
    patent_id: str
    title: str
    assignee: str = "Unknown"
    filing_date: str = "Unknown"
    status: str = "Unknown"
    relevance_score: float = 0.0


class FundingSignal(BaseModel):
    """Funding activity signal"""
    company: str
    amount: str = "Undisclosed"
    date: str = "Unknown"
    investors: List[str] = Field(default_factory=list)
    round_type: str = "unknown"  # "seed", "series_a", etc.


class MarketValidation(BaseModel):
    """Complete market validation report"""
    idea: str
    status: ValidationStatus
    confidence_score: float = 0.5
    
    # Market analysis
    competitors: List[CompetitorAnalysis] = Field(default_factory=list)
    market_size_estimate: Optional[str] = None
    growth_rate: Optional[str] = None
    
    # IP landscape
    relevant_patents: List[PatentInfo] = Field(default_factory=list)
    patent_risk_level: str = "low"  # "low", "medium", "high"
    
    # Funding signals
    recent_funding: List[FundingSignal] = Field(default_factory=list)
    funding_trend: str = "unknown"  # "increasing", "stable", "declining"
    
    # Key insights
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Supporting data
    search_results: List[Dict] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=datetime.now)
    
    def competitor_count(self) -> int:
        return len(self.competitors)
    
    def top_competitors(self, limit: int = 5) -> List[CompetitorAnalysis]:
        return self.competitors[:limit]
