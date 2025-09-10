"""
Network analysis module for TLM.

Provides comprehensive graph/network statistics and algorithms
for analyzing relationships in data - essential for modern ML applications
like document networks, knowledge graphs, and recommendation systems.
"""

from .graph import Graph, WeightedGraph
from .centrality import (
    degree_centrality, betweenness_centrality, closeness_centrality, 
    eigenvector_centrality, pagerank
)
from .clustering import (
    clustering_coefficient, global_clustering_coefficient,
    local_clustering_coefficient, transitivity
)
from .topology import (
    density, diameter, average_path_length, degree_distribution,
    assortativity_coefficient, modularity
)
from .paths import (
    shortest_path, shortest_path_length, all_shortest_paths,
    is_connected, connected_components, strongly_connected_components
)
from .community import (
    louvain_communities, girvan_newman_communities,
    modularity_communities, greedy_modularity_communities
)

__all__ = [
    # Core data structures
    'Graph', 'WeightedGraph',
    
    # Centrality measures
    'degree_centrality', 'betweenness_centrality', 'closeness_centrality',
    'eigenvector_centrality', 'pagerank',
    
    # Clustering
    'clustering_coefficient', 'global_clustering_coefficient',
    'local_clustering_coefficient', 'transitivity',
    
    # Network topology
    'density', 'diameter', 'average_path_length', 'degree_distribution',
    'assortativity_coefficient', 'modularity',
    
    # Paths and connectivity
    'shortest_path', 'shortest_path_length', 'all_shortest_paths',
    'is_connected', 'connected_components', 'strongly_connected_components',
    
    # Community detection
    'louvain_communities', 'girvan_newman_communities',
    'modularity_communities', 'greedy_modularity_communities'
]