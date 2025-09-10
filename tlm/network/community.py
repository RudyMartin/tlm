"""
Community detection algorithms for network analysis in TLM.

Provides algorithms to find groups of densely connected nodes
in networks - essential for understanding network structure.
"""

from typing import List, Dict, Set, Tuple
import random
from collections import defaultdict

from .graph import Graph, Node
from .topology import modularity

def greedy_modularity_communities(graph: Graph, seed: int = None) -> List[List[Node]]:
    """
    Find communities using greedy modularity optimization.
    
    Starts with each node in its own community and iteratively merges
    communities to maximize modularity.
    
    Args:
        graph: Input graph
        seed: Random seed for reproducibility
    
    Returns:
        List of communities (each community is a list of nodes)
    """
    if seed is not None:
        random.seed(seed)
    
    # Start with each node in its own community
    communities = [[node] for node in graph.nodes]
    
    if not communities:
        return []
    
    best_modularity = modularity(graph, communities)
    improved = True
    
    while improved:
        improved = False
        best_merge = None
        best_mod = best_modularity
        
        # Try all possible merges
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                # Try merging community i with community j
                test_communities = communities.copy()
                merged_community = test_communities[i] + test_communities[j]
                test_communities = [test_communities[k] for k in range(len(test_communities)) 
                                  if k != i and k != j] + [merged_community]
                
                mod = modularity(graph, test_communities)
                if mod > best_mod:
                    best_mod = mod
                    best_merge = (i, j)
                    improved = True
        
        # Apply best merge if found
        if best_merge:
            i, j = best_merge
            merged_community = communities[i] + communities[j]
            communities = [communities[k] for k in range(len(communities)) 
                          if k != i and k != j] + [merged_community]
            best_modularity = best_mod
    
    return communities


def label_propagation_communities(graph: Graph, max_iter: int = 100, seed: int = None) -> List[List[Node]]:
    """
    Find communities using label propagation algorithm.
    
    Each node adopts the label that is most common among its neighbors.
    
    Args:
        graph: Input graph
        max_iter: Maximum number of iterations
        seed: Random seed for reproducibility
    
    Returns:
        List of communities
    """
    if seed is not None:
        random.seed(seed)
    
    # Initialize each node with unique label
    labels = {node: i for i, node in enumerate(graph.nodes)}
    
    for iteration in range(max_iter):
        # Random order of nodes
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        
        changed = False
        
        for node in nodes:
            neighbors = graph.neighbors(node)
            
            if not neighbors:
                continue
            
            # Count neighbor labels
            label_counts = defaultdict(int)
            for neighbor in neighbors:
                label_counts[labels[neighbor]] += 1
            
            # Find most common label
            if label_counts:
                most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
                
                if labels[node] != most_common_label:
                    labels[node] = most_common_label
                    changed = True
        
        if not changed:
            break
    
    # Group nodes by label
    communities_dict = defaultdict(list)
    for node, label in labels.items():
        communities_dict[label].append(node)
    
    return list(communities_dict.values())


def girvan_newman_communities(graph: Graph, num_communities: int = None) -> List[List[Node]]:
    """
    Find communities using Girvan-Newman algorithm.
    
    Iteratively removes edges with highest betweenness centrality
    until desired number of communities is reached.
    
    Args:
        graph: Input graph
        num_communities: Target number of communities (default: stop when modularity stops improving)
    
    Returns:
        List of communities
    """
    from .centrality import betweenness_centrality
    from .paths import connected_components
    
    temp_graph = graph.copy()
    
    if num_communities is None:
        # Use modularity to decide when to stop
        best_communities = [list(temp_graph.nodes)]
        best_modularity = modularity(temp_graph, best_communities)
        
        while temp_graph.num_edges > 0:
            # Calculate edge betweenness
            edge_betweenness = _edge_betweenness_centrality(temp_graph)
            
            if not edge_betweenness:
                break
            
            # Remove edge with highest betweenness
            max_edge = max(edge_betweenness.items(), key=lambda x: x[1])[0]
            temp_graph.remove_edge(max_edge[0], max_edge[1])
            
            # Check current communities
            current_communities = connected_components(temp_graph)
            current_modularity = modularity(graph, current_communities)  # Use original graph
            
            if current_modularity > best_modularity:
                best_communities = current_communities
                best_modularity = current_modularity
        
        return best_communities
    
    else:
        # Stop when we have desired number of communities
        while temp_graph.num_edges > 0:
            components = connected_components(temp_graph)
            
            if len(components) >= num_communities:
                return components[:num_communities]
            
            # Calculate edge betweenness
            edge_betweenness = _edge_betweenness_centrality(temp_graph)
            
            if not edge_betweenness:
                break
            
            # Remove edge with highest betweenness
            max_edge = max(edge_betweenness.items(), key=lambda x: x[1])[0]
            temp_graph.remove_edge(max_edge[0], max_edge[1])
        
        return connected_components(temp_graph)


def louvain_communities(graph: Graph, resolution: float = 1.0, seed: int = None) -> List[List[Node]]:
    """
    Find communities using Louvain algorithm (simplified version).
    
    Two-phase algorithm: local moving and aggregation.
    
    Args:
        graph: Input graph
        resolution: Resolution parameter for modularity
        seed: Random seed for reproducibility
    
    Returns:
        List of communities
    """
    if seed is not None:
        random.seed(seed)
    
    # Start with each node in its own community
    node_to_community = {node: i for i, node in enumerate(graph.nodes)}
    
    improved = True
    iteration = 0
    max_iterations = 100
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Random order of nodes
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        
        for node in nodes:
            current_community = node_to_community[node]
            best_community = current_community
            best_gain = 0.0
            
            # Calculate current modularity contribution
            current_mod = _modularity_gain(graph, node, current_community, 
                                         node_to_community, resolution)
            
            # Try moving to neighbor communities
            neighbor_communities = set()
            for neighbor in graph.neighbors(node):
                neighbor_communities.add(node_to_community[neighbor])
            
            for community in neighbor_communities:
                if community != current_community:
                    gain = _modularity_gain(graph, node, community, 
                                          node_to_community, resolution) - current_mod
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community
            
            # Move node if beneficial
            if best_community != current_community and best_gain > 0:
                node_to_community[node] = best_community
                improved = True
    
    # Group nodes by community
    communities_dict = defaultdict(list)
    for node, community in node_to_community.items():
        communities_dict[community].append(node)
    
    return [community for community in communities_dict.values() if community]


def modularity_communities(graph: Graph) -> List[List[Node]]:
    """
    Alias for greedy_modularity_communities for compatibility.
    """
    return greedy_modularity_communities(graph)


def conductance(graph: Graph, community: List[Node]) -> float:
    """
    Calculate conductance of a community.
    
    Conductance measures the fraction of edges leaving the community
    relative to the total degree of the community.
    
    Args:
        graph: Input graph
        community: List of nodes in the community
    
    Returns:
        Conductance value between 0 and 1
    """
    community_set = set(community)
    
    internal_edges = 0
    external_edges = 0
    total_degree = 0
    
    for node in community:
        degree = graph.degree(node)
        total_degree += degree
        
        for neighbor in graph.neighbors(node):
            if neighbor in community_set:
                internal_edges += 1
            else:
                external_edges += 1
    
    # Each internal edge counted twice
    internal_edges //= 2
    
    if total_degree == 0:
        return 0.0
    
    # Conductance = external edges / min(volume(S), volume(V-S))
    complement_volume = 2 * graph.num_edges - total_degree
    min_volume = min(total_degree, complement_volume)
    
    return external_edges / min_volume if min_volume > 0 else 0.0


# Helper functions

def _edge_betweenness_centrality(graph: Graph) -> Dict[Tuple[Node, Node], float]:
    """
    Calculate betweenness centrality for all edges.
    
    Returns:
        Dictionary mapping edges to betweenness values
    """
    from .centrality import _single_source_shortest_path_lengths
    from collections import defaultdict
    
    edge_betweenness = defaultdict(float)
    
    # For each node pair, find shortest paths and count edge usage
    for source in graph.nodes:
        # Single source shortest paths
        distances = _single_source_shortest_path_lengths(graph, source)
        
        for target in graph.nodes:
            if source != target and target in distances:
                # Find all shortest paths from source to target
                paths = _all_shortest_paths_edges(graph, source, target, distances)
                
                # Each path contributes equally to edge betweenness
                weight = 1.0 / len(paths) if paths else 0.0
                
                for path_edges in paths:
                    for edge in path_edges:
                        # Normalize edge direction
                        normalized_edge = tuple(sorted(edge))
                        edge_betweenness[normalized_edge] += weight
    
    return dict(edge_betweenness)


def _all_shortest_paths_edges(graph: Graph, source: Node, target: Node, 
                             distances: Dict[Node, int]) -> List[List[Tuple[Node, Node]]]:
    """
    Find all shortest paths between source and target as lists of edges.
    """
    if target not in distances:
        return []
    
    target_distance = distances[target]
    
    def _build_paths(current: Node, current_distance: int) -> List[List[Tuple[Node, Node]]]:
        if current == target:
            return [[]]  # Empty path (no more edges needed)
        
        paths = []
        for neighbor in graph.neighbors(current):
            if neighbor in distances and distances[neighbor] == current_distance + 1:
                # This neighbor is on a shortest path
                for sub_path in _build_paths(neighbor, current_distance + 1):
                    paths.append([(current, neighbor)] + sub_path)
        
        return paths
    
    return _build_paths(source, distances[source])


def _modularity_gain(graph: Graph, node: Node, community: int, 
                    node_to_community: Dict[Node, int], resolution: float) -> float:
    """
    Calculate modularity gain from moving node to community.
    """
    # Simplified modularity calculation for single node
    m = graph.num_edges
    if m == 0:
        return 0.0
    
    node_degree = graph.degree(node)
    community_nodes = [n for n, c in node_to_community.items() if c == community]
    
    # Edges from node to community
    edges_to_community = 0
    for neighbor in graph.neighbors(node):
        if node_to_community[neighbor] == community:
            edges_to_community += 1
    
    # Community total degree
    community_degree = sum(graph.degree(n) for n in community_nodes)
    
    # Modularity contribution
    return (edges_to_community / m - resolution * (node_degree * community_degree) / (2 * m * m))