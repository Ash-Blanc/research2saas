"""Paper and research cluster models"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np


class Paper(BaseModel):
    """Research paper with metadata"""
    
    id: str
    title: str
    abstract: str = ""
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    venue: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    fields_of_study: List[str] = Field(default_factory=list)
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    is_open_access: bool = False
    publication_date: Optional[str] = None
    publication_types: List[str] = Field(default_factory=list)
    
    # Citation context (when retrieved as citation/reference)
    is_influential: bool = False
    contexts: List[str] = Field(default_factory=list)
    intents: List[str] = Field(default_factory=list)
    
    # Computed metrics
    similarity_score: float = 0.0
    citation_velocity: float = 0.0
    application_score: float = 0.0
    matched_keywords: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class PaperCluster(BaseModel):
    """Cluster of related research papers"""
    
    id: str
    papers: List[Paper] = Field(default_factory=list)
    central_papers: List[Paper] = Field(default_factory=list)
    theme: str = ""
    application_potential: float = 0.0
    temporal_trend: str = "unknown"  # "emerging", "mature", "declining"
    
    def avg_citation_count(self) -> float:
        """Average citation count across cluster"""
        if not self.papers:
            return 0.0
        return float(np.mean([p.citation_count for p in self.papers]))
    
    def year_range(self) -> Tuple[int, int]:
        """Year range of papers in cluster"""
        if not self.papers:
            return (0, 0)
        years = [p.year for p in self.papers if p.year]
        if not years:
            return (0, 0)
        return (min(years), max(years))
    
    def paper_count(self) -> int:
        """Number of papers in cluster"""
        return len(self.papers)
