"""
Citation Graph Clustering for Research-to-SaaS Platform
Implements advanced graph analysis for discovering research clusters and application pathways
"""

import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from datetime import datetime
import json


@dataclass
class Paper:
    """Represents a research paper with metadata"""
    id: str
    title: str
    abstract: str
    year: int
    authors: List[str]
    citation_count: int
    venue: Optional[str] = None
    keywords: List[str] = None
    arxiv_id: Optional[str] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": self.authors,
            "citation_count": self.citation_count,
            "venue": self.venue,
            "keywords": self.keywords or [],
            "arxiv_id": self.arxiv_id
        }


@dataclass
class PaperCluster:
    """Represents a cluster of related papers"""
    id: str
    papers: List[Paper]
    central_papers: List[Paper]  # Most influential in cluster
    theme: str  # Auto-generated theme description
    application_potential: float  # 0-1 score
    temporal_trend: str  # "emerging", "mature", "declining"
    
    def avg_citation_count(self) -> float:
        return np.mean([p.citation_count for p in self.papers])
    
    def year_range(self) -> Tuple[int, int]:
        years = [p.year for p in self.papers]
        return (min(years), max(years))
    
    def to_dict(self):
        return {
            "id": self.id,
            "theme": self.theme,
            "paper_count": len(self.papers),
            "central_papers": [p.to_dict() for p in self.central_papers],
            "application_potential": self.application_potential,
            "temporal_trend": self.temporal_trend,
            "year_range": self.year_range(),
            "avg_citations": self.avg_citation_count()
        }


class CitationGraphAnalyzer:
    """Advanced citation graph analysis for research discovery"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.papers: Dict[str, Paper] = {}
        self.clusters: List[PaperCluster] = []
        
        # Application-indicating keywords
        self.application_keywords = {
            "implementation", "system", "framework", "tool", "platform",
            "application", "production", "deployment", "real-world",
            "case study", "empirical", "practical", "industry"
        }
        
        # Theory-indicating keywords
        self.theory_keywords = {
            "theoretical", "analysis", "proof", "bound", "complexity",
            "algorithm", "model", "approach", "method", "novel"
        }
    
    def add_paper(self, paper: Paper):
        """Add paper to graph"""
        self.papers[paper.id] = paper
        self.graph.add_node(
            paper.id,
            **paper.to_dict()
        )
    
    def add_citation(self, citing_paper_id: str, cited_paper_id: str):
        """Add citation edge (citing -> cited)"""
        self.graph.add_edge(citing_paper_id, cited_paper_id, relation="cites")
    
    def build_from_connected_papers(self, seed_paper_id: str, 
                                   cp_client, max_papers: int = 100):
        """Build graph from ConnectedPapers API"""
        from connectedpapers import ConnectedPapersClient
        
        graph_data = cp_client.get_graph(seed_paper_id)
        
        # Add target paper
        target = graph_data.target_paper
        target_paper = Paper(
            id=target.id,
            title=target.title,
            abstract=getattr(target, 'abstract', ''),
            year=target.year,
            authors=target.authors,
            citation_count=getattr(target, 'citation_count', 0)
        )
        self.add_paper(target_paper)
        
        # Add connected papers
        for node in graph_data.nodes[:max_papers]:
            paper = Paper(
                id=node.id,
                title=node.title,
                abstract=getattr(node, 'abstract', ''),
                year=node.year,
                authors=node.authors,
                citation_count=getattr(node, 'citation_count', 0)
            )
            self.add_paper(paper)
            
            # Infer citation relationship based on year
            if node.year < target.year:
                # Likely cited by target
                self.add_citation(target.id, node.id)
            elif node.year > target.year:
                # Likely cites target
                self.add_citation(node.id, target.id)
    
    def detect_communities(self, min_cluster_size: int = 3) -> List[PaperCluster]:
        """Detect research communities using Louvain algorithm"""
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Use Louvain method for community detection
        communities = nx.community.louvain_communities(undirected, seed=42)
        
        clusters = []
        for idx, community in enumerate(communities):
            if len(community) < min_cluster_size:
                continue
            
            # Get papers in this community
            community_papers = [self.papers[paper_id] for paper_id in community]
            
            # Find central papers (highest PageRank within community)
            subgraph = self.graph.subgraph(community)
            pagerank = nx.pagerank(subgraph)
            central_ids = sorted(pagerank.keys(), 
                               key=lambda x: pagerank[x], 
                               reverse=True)[:5]
            central_papers = [self.papers[pid] for pid in central_ids]
            
            # Generate theme from central paper titles
            theme = self._generate_cluster_theme(central_papers)
            
            # Calculate application potential
            app_potential = self._calculate_application_potential(community_papers)
            
            # Determine temporal trend
            temporal_trend = self._analyze_temporal_trend(community_papers)
            
            cluster = PaperCluster(
                id=f"cluster_{idx}",
                papers=community_papers,
                central_papers=central_papers,
                theme=theme,
                application_potential=app_potential,
                temporal_trend=temporal_trend
            )
            clusters.append(cluster)
        
        self.clusters = sorted(clusters, 
                             key=lambda c: c.application_potential, 
                             reverse=True)
        return self.clusters
    
    def find_application_pathway(self, 
                                theory_paper_id: str,
                                max_hops: int = 5) -> List[Dict]:
        """Find citation path from theory to application papers"""
        # Find all application-oriented papers
        app_papers = [
            pid for pid, data in self.graph.nodes(data=True)
            if self._is_application_paper(data)
        ]
        
        # Find shortest paths
        pathways = []
        for app_paper_id in app_papers:
            try:
                # Try both directions
                if nx.has_path(self.graph, theory_paper_id, app_paper_id):
                    path = nx.shortest_path(self.graph, 
                                          theory_paper_id, 
                                          app_paper_id)
                    if len(path) <= max_hops + 1:
                        pathways.append({
                            "target": app_paper_id,
                            "path": path,
                            "length": len(path) - 1,
                            "papers": [self.papers[pid] for pid in path]
                        })
            except nx.NetworkXNoPath:
                continue
        
        # Sort by shortest path first
        return sorted(pathways, key=lambda x: x["length"])
    
    def find_cross_domain_bridges(self, 
                                 domain1_keywords: List[str],
                                 domain2_keywords: List[str]) -> List[Paper]:
        """Find papers that bridge two different research domains"""
        domain1_papers = set()
        domain2_papers = set()
        
        # Identify papers in each domain
        for pid, data in self.graph.nodes(data=True):
            title_abstract = (data['title'] + ' ' + data['abstract']).lower()
            
            if any(kw.lower() in title_abstract for kw in domain1_keywords):
                domain1_papers.add(pid)
            if any(kw.lower() in title_abstract for kw in domain2_keywords):
                domain2_papers.add(pid)
        
        # Find papers that cite both domains
        bridge_papers = []
        for pid in self.graph.nodes():
            if pid in domain1_papers or pid in domain2_papers:
                continue
            
            cited_papers = set(self.graph.successors(pid))
            
            cites_domain1 = bool(cited_papers & domain1_papers)
            cites_domain2 = bool(cited_papers & domain2_papers)
            
            if cites_domain1 and cites_domain2:
                bridge_papers.append(self.papers[pid])
        
        # Sort by citation count (quality proxy)
        return sorted(bridge_papers, 
                     key=lambda p: p.citation_count, 
                     reverse=True)
    
    def track_research_evolution(self, 
                                seed_paper_id: str,
                                years_forward: int = 5) -> Dict:
        """Track how research evolved over time from a seed paper"""
        seed_paper = self.papers[seed_paper_id]
        seed_year = seed_paper.year
        
        # Group papers by year
        evolution = defaultdict(list)
        
        # BFS from seed paper
        visited = set()
        queue = [(seed_paper_id, 0)]  # (paper_id, generation)
        
        while queue:
            current_id, generation = queue.pop(0)
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            current_paper = self.papers[current_id]
            
            # Only track papers within time window
            if current_paper.year <= seed_year + years_forward:
                year_diff = current_paper.year - seed_year
                evolution[year_diff].append({
                    "paper": current_paper,
                    "generation": generation
                })
                
                # Add papers that cite this one
                for citing_id in self.graph.predecessors(current_id):
                    if citing_id not in visited:
                        queue.append((citing_id, generation + 1))
        
        return dict(evolution)
    
    def calculate_impact_score(self, paper_id: str) -> Dict[str, float]:
        """Calculate multi-dimensional impact score for a paper"""
        paper = self.papers[paper_id]
        
        # Direct citations
        direct_citations = len(list(self.graph.predecessors(paper_id)))
        
        # PageRank (overall importance in graph)
        pagerank = nx.pagerank(self.graph)[paper_id]
        
        # Betweenness centrality (bridge between communities)
        betweenness = nx.betweenness_centrality(self.graph)[paper_id]
        
        # Temporal impact (how many recent papers cite it)
        recent_citers = [
            self.papers[pid] for pid in self.graph.predecessors(paper_id)
            if self.papers[pid].year >= 2020
        ]
        recent_impact = len(recent_citers) / max(direct_citations, 1)
        
        return {
            "direct_citations": direct_citations,
            "pagerank": pagerank,
            "betweenness": betweenness,
            "recent_impact": recent_impact,
            "composite_score": (
                0.3 * min(direct_citations / 100, 1.0) +
                0.3 * pagerank * 100 +
                0.2 * betweenness * 10 +
                0.2 * recent_impact
            )
        }
    
    def _generate_cluster_theme(self, central_papers: List[Paper]) -> str:
        """Generate theme description from central papers"""
        # Extract common words from titles (simplified - in prod use LLM)
        from collections import Counter
        
        words = []
        stopwords = {'a', 'an', 'the', 'for', 'in', 'on', 'with', 'using', 'based'}
        
        for paper in central_papers[:3]:
            title_words = paper.title.lower().split()
            words.extend([w for w in title_words if w not in stopwords])
        
        common_words = Counter(words).most_common(3)
        theme_words = [word for word, _ in common_words]
        
        return " ".join(theme_words).title()
    
    def _calculate_application_potential(self, papers: List[Paper]) -> float:
        """Score cluster's potential for practical applications"""
        score = 0.0
        
        # Check for application keywords
        app_keyword_count = 0
        for paper in papers:
            text = (paper.title + ' ' + paper.abstract).lower()
            app_keyword_count += sum(
                1 for kw in self.application_keywords if kw in text
            )
        
        # Normalize
        app_keyword_ratio = app_keyword_count / len(papers)
        score += min(app_keyword_ratio * 0.3, 0.3)
        
        # Recent papers indicate active development
        recent_papers = [p for p in papers if p.year >= 2022]
        recency_score = len(recent_papers) / len(papers)
        score += recency_score * 0.3
        
        # Citation diversity (cited by different research areas)
        avg_citations = np.mean([p.citation_count for p in papers])
        citation_score = min(avg_citations / 100, 1.0)
        score += citation_score * 0.4
        
        return min(score, 1.0)
    
    def _analyze_temporal_trend(self, papers: List[Paper]) -> str:
        """Determine if cluster is emerging, mature, or declining"""
        years = [p.year for p in papers]
        year_counts = Counter(years)
        
        # Get recent vs old paper counts
        recent_years = [y for y in years if y >= 2022]
        old_years = [y for y in years if y < 2020]
        
        recent_ratio = len(recent_years) / len(papers)
        
        if recent_ratio > 0.5:
            return "emerging"
        elif len(old_years) > len(recent_years):
            return "declining"
        else:
            return "mature"
    
    def _is_application_paper(self, paper_data: Dict) -> bool:
        """Check if paper is application-oriented"""
        text = (paper_data['title'] + ' ' + paper_data['abstract']).lower()
        return any(kw in text for kw in self.application_keywords)
    
    def export_graph(self, filepath: str):
        """Export graph to JSON for visualization"""
        data = {
            "nodes": [
                {
                    "id": pid,
                    **self.papers[pid].to_dict()
                }
                for pid in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relation": self.graph[u][v].get("relation", "cites")
                }
                for u, v in self.graph.edges()
            ],
            "clusters": [cluster.to_dict() for cluster in self.clusters]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_cluster_summary(self) -> List[Dict]:
        """Get summary of all clusters for quick review"""
        return [
            {
                "id": cluster.id,
                "theme": cluster.theme,
                "size": len(cluster.papers),
                "application_potential": cluster.application_potential,
                "trend": cluster.temporal_trend,
                "top_paper": cluster.central_papers[0].title,
                "year_range": cluster.year_range()
            }
            for cluster in self.clusters
        ]


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CitationGraphAnalyzer()
    
    # Example: Add some papers manually
    papers = [
        Paper("p1", "Attention Is All You Need", "Transformer architecture...", 
              2017, ["Vaswani"], 50000, "NeurIPS"),
        Paper("p2", "BERT: Pre-training of Deep Bidirectional Transformers", 
              "BERT model...", 2018, ["Devlin"], 40000, "NAACL"),
        Paper("p3", "GPT-3: Language Models are Few-Shot Learners", 
              "GPT-3...", 2020, ["Brown"], 30000, "NeurIPS"),
        Paper("p4", "ChatGPT: Optimizing Language Models for Dialogue", 
              "ChatGPT application...", 2022, ["OpenAI"], 10000),
    ]
    
    for paper in papers:
        analyzer.add_paper(paper)
    
    # Add citations
    analyzer.add_citation("p2", "p1")  # BERT cites Transformer
    analyzer.add_citation("p3", "p1")  # GPT-3 cites Transformer
    analyzer.add_citation("p3", "p2")  # GPT-3 cites BERT
    analyzer.add_citation("p4", "p3")  # ChatGPT cites GPT-3
    
    # Analyze
    clusters = analyzer.detect_communities(min_cluster_size=2)
    print(f"Found {len(clusters)} clusters")
    
    pathways = analyzer.find_application_pathway("p1", max_hops=3)
    print(f"Found {len(pathways)} application pathways")
    
    impact = analyzer.calculate_impact_score("p1")
    print(f"Impact score for Transformer: {impact}")