"""
ConnectedPapers Tool for Agno
Wraps connectedpapers-py library as an Agno tool
"""

from typing import List, Dict, Optional
from agno.tools import Toolkit
from connectedpapers import ConnectedPapersClient
import logging


logger = logging.getLogger(__name__)


class ConnectedPapersTools(Toolkit):
    """
    Agno toolkit for ConnectedPapers API
    
    Provides tools to:
    - Find similar papers via citation graph
    - Get prior works (papers cited by target)
    - Get derivative works (papers citing target)
    - Find application-oriented papers
    - Build complete research lineage
    """
    
    def __init__(self):
        super().__init__(name="connected_papers")
        self.client = ConnectedPapersClient()
        
        self.application_keywords = [
            "application", "implementation", "system", "framework",
            "tool", "platform", "deployment", "production", "case study",
            "empirical", "real-world", "practical", "industry"
        ]
    
    def get_similar_papers(self, 
                          paper_id: str, 
                          limit: int = 10) -> List[Dict]:
        """
        Find papers similar to the given paper using citation graph analysis.
        
        Args:
            paper_id: ArXiv ID, DOI, or other paper identifier
            limit: Maximum number of similar papers to return
            
        Returns:
            List of similar papers with metadata
        """
        try:
            graph = self.client.get_graph(paper_id)
            
            similar = []
            for node in graph.nodes[:limit]:
                similar.append({
                    "id": node.id,
                    "title": node.title,
                    "year": node.year,
                    "authors": node.authors,
                    "citation_count": getattr(node, 'citation_count', 0),
                    "similarity_score": getattr(node, 'similarity', 0.0),
                    "url": f"https://www.connectedpapers.com/main/{node.id}"
                })
            
            # Sort by similarity
            similar.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Found {len(similar)} similar papers for {paper_id}")
            return similar
            
        except Exception as e:
            logger.error(f"Error finding similar papers: {e}")
            return []
    
    def get_prior_works(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """
        Get foundational papers that the target paper builds upon.
        These are papers published before the target and cited by it.
        
        Args:
            paper_id: Target paper identifier
            limit: Maximum number of prior works to return
            
        Returns:
            List of foundational papers, sorted by citation count
        """
        try:
            graph = self.client.get_graph(paper_id)
            target_year = graph.target_paper.year
            
            # Papers published before target paper
            prior = [
                node for node in graph.nodes 
                if node.year < target_year
            ]
            
            # Sort by citation count (impact)
            prior.sort(key=lambda x: getattr(x, 'citation_count', 0), reverse=True)
            
            result = []
            for node in prior[:limit]:
                result.append({
                    "id": node.id,
                    "title": node.title,
                    "year": node.year,
                    "authors": node.authors,
                    "citation_count": getattr(node, 'citation_count', 0),
                    "url": f"https://www.connectedpapers.com/main/{node.id}"
                })
            
            logger.info(f"Found {len(result)} prior works for {paper_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error finding prior works: {e}")
            return []
    
    def get_derivative_works(self, 
                            paper_id: str, 
                            limit: int = 10,
                            recent_only: bool = False) -> List[Dict]:
        """
        Get papers that built upon or cite the target paper.
        These are papers published after the target.
        
        Args:
            paper_id: Target paper identifier
            limit: Maximum number of derivative works to return
            recent_only: If True, only return papers from last 2 years
            
        Returns:
            List of derivative papers, sorted by year (newest first)
        """
        try:
            graph = self.client.get_graph(paper_id)
            target_year = graph.target_paper.year
            
            # Papers published after target
            derivative = [
                node for node in graph.nodes
                if node.year > target_year
            ]
            
            # Filter by recency if requested
            if recent_only:
                current_year = 2024  # Update as needed
                derivative = [
                    node for node in derivative
                    if node.year >= current_year - 2
                ]
            
            # Sort by year (newest first)
            derivative.sort(key=lambda x: x.year, reverse=True)
            
            result = []
            for node in derivative[:limit]:
                result.append({
                    "id": node.id,
                    "title": node.title,
                    "year": node.year,
                    "authors": node.authors,
                    "citation_count": getattr(node, 'citation_count', 0),
                    "url": f"https://www.connectedpapers.com/main/{node.id}"
                })
            
            logger.info(f"Found {len(result)} derivative works for {paper_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error finding derivative works: {e}")
            return []
    
    def find_application_papers(self, 
                               paper_id: str,
                               limit: int = 10) -> List[Dict]:
        """
        Find papers that likely represent practical applications or implementations
        of the theory/methods in the target paper.
        
        Identifies papers with keywords like: implementation, system, application, etc.
        
        Args:
            paper_id: Target paper identifier
            limit: Maximum number of application papers to return
            
        Returns:
            List of application-oriented papers
        """
        try:
            graph = self.client.get_graph(paper_id)
            
            applications = []
            for node in graph.nodes:
                title_lower = node.title.lower()
                
                # Check for application keywords
                if any(kw in title_lower for kw in self.application_keywords):
                    applications.append({
                        "id": node.id,
                        "title": node.title,
                        "year": node.year,
                        "authors": node.authors,
                        "citation_count": getattr(node, 'citation_count', 0),
                        "similarity_score": getattr(node, 'similarity', 0.0),
                        "matched_keywords": [
                            kw for kw in self.application_keywords 
                            if kw in title_lower
                        ],
                        "url": f"https://www.connectedpapers.com/main/{node.id}"
                    })
            
            # Sort by relevance (similarity score)
            applications.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Found {len(applications)} application papers for {paper_id}")
            return applications[:limit]
            
        except Exception as e:
            logger.error(f"Error finding application papers: {e}")
            return []
    
    def build_research_lineage(self, paper_id: str) -> Dict:
        """
        Build complete research lineage for a paper:
        - Foundational papers (prior works)
        - The target paper itself
        - Derivative works (papers building on it)
        - Similar contemporary papers
        - Application-oriented papers
        
        This gives a comprehensive view of the paper's position in research landscape.
        
        Args:
            paper_id: Target paper identifier
            
        Returns:
            Dictionary with complete lineage
        """
        try:
            graph = self.client.get_graph(paper_id)
            
            # Get target paper info
            target = graph.target_paper
            target_info = {
                "id": target.id,
                "title": target.title,
                "year": target.year,
                "authors": target.authors,
                "citation_count": getattr(target, 'citation_count', 0)
            }
            
            lineage = {
                "target_paper": target_info,
                "foundations": self.get_prior_works(paper_id, limit=10),
                "derivatives": self.get_derivative_works(paper_id, limit=10),
                "similar": self.get_similar_papers(paper_id, limit=10),
                "applications": self.find_application_papers(paper_id, limit=10),
                "metadata": {
                    "total_connected_papers": len(graph.nodes),
                    "analysis_date": str(graph.target_paper.year)
                }
            }
            
            logger.info(f"Built research lineage for {paper_id}")
            return lineage
            
        except Exception as e:
            logger.error(f"Error building research lineage: {e}")
            return {
                "error": str(e),
                "target_paper": {},
                "foundations": [],
                "derivatives": [],
                "similar": [],
                "applications": []
            }
    
    def find_research_frontier(self, 
                              paper_id: str,
                              years_back: int = 2) -> List[Dict]:
        """
        Find cutting-edge papers at the research frontier.
        Returns recent derivative works that represent current research directions.
        
        Args:
            paper_id: Target paper identifier
            years_back: How many years back to look for recent papers
            
        Returns:
            List of recent papers representing research frontier
        """
        try:
            graph = self.client.get_graph(paper_id)
            current_year = 2024  # Update as needed
            cutoff_year = current_year - years_back
            
            # Get recent papers
            frontier = [
                node for node in graph.nodes
                if node.year >= cutoff_year
            ]
            
            # Sort by citation velocity (citations per year)
            frontier_with_velocity = []
            for node in frontier:
                citation_count = getattr(node, 'citation_count', 0)
                years_since_pub = max(current_year - node.year, 1)
                citation_velocity = citation_count / years_since_pub
                
                frontier_with_velocity.append({
                    "id": node.id,
                    "title": node.title,
                    "year": node.year,
                    "authors": node.authors,
                    "citation_count": citation_count,
                    "citation_velocity": citation_velocity,
                    "similarity_score": getattr(node, 'similarity', 0.0),
                    "url": f"https://www.connectedpapers.com/main/{node.id}"
                })
            
            # Sort by citation velocity (hot papers)
            frontier_with_velocity.sort(
                key=lambda x: x['citation_velocity'], 
                reverse=True
            )
            
            logger.info(f"Found {len(frontier_with_velocity)} frontier papers")
            return frontier_with_velocity[:10]
            
        except Exception as e:
            logger.error(f"Error finding research frontier: {e}")
            return []
    
    def find_cross_domain_papers(self,
                                paper_id1: str,
                                paper_id2: str) -> List[Dict]:
        """
        Find papers that bridge two different research areas.
        These are papers that cite both input papers or appear in both citation graphs.
        
        Args:
            paper_id1: First paper identifier
            paper_id2: Second paper identifier
            
        Returns:
            List of bridge papers connecting the two domains
        """
        try:
            # Get both citation graphs
            graph1 = self.client.get_graph(paper_id1)
            graph2 = self.client.get_graph(paper_id2)
            
            # Get paper IDs from both graphs
            ids1 = {node.id for node in graph1.nodes}
            ids2 = {node.id for node in graph2.nodes}
            
            # Find papers that appear in both
            bridge_ids = ids1 & ids2
            
            # Get full info for bridge papers
            bridges = []
            for node in graph1.nodes:
                if node.id in bridge_ids:
                    bridges.append({
                        "id": node.id,
                        "title": node.title,
                        "year": node.year,
                        "authors": node.authors,
                        "citation_count": getattr(node, 'citation_count', 0),
                        "url": f"https://www.connectedpapers.com/main/{node.id}"
                    })
            
            # Sort by citation count
            bridges.sort(key=lambda x: x['citation_count'], reverse=True)
            
            logger.info(f"Found {len(bridges)} bridge papers")
            return bridges
            
        except Exception as e:
            logger.error(f"Error finding cross-domain papers: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize tool
    cp_tools = ConnectedPapersTools()
    
    # Example: Analyze the Transformer paper
    paper_id = "1706.03762"  # "Attention Is All You Need"
    
    print("Getting research lineage...")
    lineage = cp_tools.build_research_lineage(paper_id)
    
    print(f"\nTarget: {lineage['target_paper']['title']}")
    print(f"Foundations: {len(lineage['foundations'])} papers")
    print(f"Derivatives: {len(lineage['derivatives'])} papers")
    print(f"Applications: {len(lineage['applications'])} papers")
    
    print("\nTop Application Papers:")
    for app in lineage['applications'][:3]:
        print(f"  - {app['title']} ({app['year']})")
    
    print("\nResearch Frontier:")
    frontier = cp_tools.find_research_frontier(paper_id, years_back=2)
    for paper in frontier[:3]:
        print(f"  - {paper['title']} ({paper['year']}) - {paper['citation_velocity']:.1f} cites/year")