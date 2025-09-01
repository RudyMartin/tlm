"""
Clustering and transitivity measures for network analysis in TLM.

Provides functions to measure local and global clustering properties
of networks - essential for understanding network structure.
"""

from typing import Dict, List, Optional
from .graph import Graph, Node

def clustering_coefficient(graph: Graph, node: Node) -> float:
    """
    Calculate local clustering coefficient for a single node.
    
    Local clustering coefficient measures how close a node's neighbors
    are to being a complete graph (clique).
    
    Args:
        graph: Input graph
        node: Node to calculate clustering coefficient for
    
    Returns:
        Clustering coefficient between 0 and 1
        0 = neighbors form no triangles
        1 = neighbors form complete graph
    """
    neighbors = graph.neighbors(node)
    k = len(neighbors)
    
    # Need at least 2 neighbors to form triangles
    if k < 2:
        return 0.0
    
    # Count triangles (edges between neighbors)
    triangles = 0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if graph.has_edge(neighbors[i], neighbors[j]):
                triangles += 1
    
    # Possible edges between k neighbors is k*(k-1)/2
    possible_edges = k * (k - 1) // 2
    
    return triangles / possible_edges if possible_edges > 0 else 0.0


def local_clustering_coefficient(graph: Graph) -> Dict[Node, float]:
    """
    Calculate local clustering coefficient for all nodes.
    
    Args:
        graph: Input graph
    
    Returns:
        Dictionary mapping nodes to their clustering coefficients
    """
    return {node: clustering_coefficient(graph, node) for node in graph.nodes}


def global_clustering_coefficient(graph: Graph) -> float:
    """
    Calculate global clustering coefficient (average of local coefficients).
    
    This is the average clustering coefficient across all nodes.
    
    Args:
        graph: Input graph
    
    Returns:
        Global clustering coefficient
    """
    local_coeffs = local_clustering_coefficient(graph)
    
    if not local_coeffs:
        return 0.0
    
    return sum(local_coeffs.values()) / len(local_coeffs)


def transitivity(graph: Graph) -> float:
    """
    Calculate transitivity (global clustering coefficient based on triangles).
    
    Transitivity is the ratio of triangles to connected triples in the graph.
    More robust than average local clustering for sparse graphs.
    
    Args:
        graph: Input graph
    
    Returns:
        Transitivity between 0 and 1
    """
    triangles = 0
    connected_triples = 0
    
    # Count triangles and connected triples
    for node in graph.nodes:
        neighbors = graph.neighbors(node)
        k = len(neighbors)
        
        if k < 2:
            continue
        
        # Connected triples centered at this node
        connected_triples += k * (k - 1) // 2
        
        # Triangles involving this node
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if graph.has_edge(neighbors[i], neighbors[j]):
                    triangles += 1
    
    # Each triangle is counted 3 times (once for each vertex)
    triangles //= 3
    
    return (3 * triangles) / connected_triples if connected_triples > 0 else 0.0


def average_clustering(graph: Graph, nodes: Optional[List[Node]] = None) -> float:
    """
    Calculate average clustering coefficient for specified nodes.
    
    Args:
        graph: Input graph
        nodes: Nodes to include in average (default: all nodes)
    
    Returns:
        Average clustering coefficient
    """
    if nodes is None:
        nodes = graph.nodes
    
    if not nodes:
        return 0.0
    
    total = sum(clustering_coefficient(graph, node) for node in nodes)
    return total / len(nodes)


def square_clustering(graph: Graph, node: Node) -> float:
    """
    Calculate square clustering coefficient for a node.
    
    Square clustering measures clustering based on 4-cycles (squares)
    rather than triangles. Useful for bipartite-like networks.
    
    Args:
        graph: Input graph
        node: Node to calculate square clustering for
    
    Returns:
        Square clustering coefficient
    """
    neighbors = graph.neighbors(node)
    k = len(neighbors)
    
    if k < 2:
        return 0.0
    
    squares = 0
    
    # Look for 4-cycles: node -> neighbor1 -> other -> neighbor2 -> node
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            neighbor1 = neighbors[i]
            neighbor2 = neighbors[j]
            
            # Find common neighbors of neighbor1 and neighbor2 (excluding node)
            common = set(graph.neighbors(neighbor1)) & set(graph.neighbors(neighbor2))
            common.discard(node)
            
            squares += len(common)
    
    # Maximum possible squares
    max_squares = k * (k - 1) // 2
    
    return squares / max_squares if max_squares > 0 else 0.0


def rich_club_coefficient(graph: Graph, k: int, normalized: bool = False) -> float:
    """
    Calculate rich club coefficient for degree k.
    
    Rich club coefficient measures the tendency of high-degree nodes
    to be connected to other high-degree nodes.
    
    Args:
        graph: Input graph
        k: Degree threshold
        normalized: If True, normalize by random graph expectation
    
    Returns:
        Rich club coefficient
    """
    # Find nodes with degree > k
    rich_nodes = [node for node in graph.nodes if graph.degree(node) > k]
    
    if len(rich_nodes) < 2:
        return 0.0
    
    # Count edges between rich nodes
    edges_between_rich = 0
    for i, node1 in enumerate(rich_nodes):
        for j in range(i + 1, len(rich_nodes)):
            node2 = rich_nodes[j]
            if graph.has_edge(node1, node2):
                edges_between_rich += 1
    
    # Maximum possible edges between rich nodes
    max_edges = len(rich_nodes) * (len(rich_nodes) - 1) // 2
    
    coefficient = edges_between_rich / max_edges if max_edges > 0 else 0.0
    
    if normalized:
        # Normalize by expectation for random graph with same degree sequence
        total_edges = graph.num_edges
        total_possible = graph.num_nodes * (graph.num_nodes - 1) // 2
        expected = total_edges / total_possible if total_possible > 0 else 0
        coefficient = coefficient / expected if expected > 0 else 0
    
    return coefficient


def k_core_decomposition(graph: Graph) -> Dict[Node, int]:
    """
    Perform k-core decomposition of the graph.
    
    The k-core is the maximal subgraph where each node has degree >= k.
    The core number of a node is the highest k for which it belongs to a k-core.
    
    Args:
        graph: Input graph
    
    Returns:
        Dictionary mapping nodes to their core numbers
    """
    # Initialize core numbers with degrees
    core_numbers = {node: graph.degree(node) for node in graph.nodes}
    
    # Create list of nodes sorted by degree
    nodes_by_degree = sorted(graph.nodes, key=lambda x: core_numbers[x])
    
    for node in nodes_by_degree:
        current_core = core_numbers[node]
        
        # Update neighbors' core numbers
        for neighbor in graph.neighbors(node):
            if core_numbers[neighbor] > current_core:
                core_numbers[neighbor] = min(core_numbers[neighbor], 
                                           max(current_core, 
                                               core_numbers[neighbor] - 1))
    
    return core_numbers


def core_number(graph: Graph, node: Node) -> int:
    """
    Get the core number for a specific node.
    
    Args:
        graph: Input graph
        node: Node to get core number for
    
    Returns:
        Core number of the node
    """
    core_numbers = k_core_decomposition(graph)
    return core_numbers.get(node, 0)


def local_efficiency(graph: Graph, node: Node) -> float:
    """
    Calculate local efficiency for a node.
    
    Local efficiency measures how efficiently information flows
    between neighbors when the node is removed.
    
    Args:
        graph: Input graph
        node: Node to calculate local efficiency for
    
    Returns:
        Local efficiency between 0 and 1
    """
    neighbors = graph.neighbors(node)
    
    if len(neighbors) < 2:
        return 0.0
    
    # Create subgraph of neighbors
    subgraph = graph.subgraph(neighbors)
    
    # Calculate average shortest path length in subgraph
    total_efficiency = 0.0
    num_pairs = 0
    
    for i, node1 in enumerate(neighbors):
        for j in range(i + 1, len(neighbors)):
            node2 = neighbors[j]
            
            # Find shortest path length between neighbors in subgraph
            path_length = _shortest_path_length(subgraph, node1, node2)
            
            if path_length > 0:
                total_efficiency += 1.0 / path_length
            
            num_pairs += 1
    
    return total_efficiency / num_pairs if num_pairs > 0 else 0.0


def global_efficiency(graph: Graph) -> float:
    """
    Calculate global efficiency of the graph.
    
    Global efficiency measures how efficiently information flows
    across the entire network.
    
    Args:
        graph: Input graph
    
    Returns:
        Global efficiency between 0 and 1
    """
    nodes = graph.nodes
    n = len(nodes)
    
    if n < 2:
        return 0.0
    
    total_efficiency = 0.0
    num_pairs = 0
    
    for i, node1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            
            path_length = _shortest_path_length(graph, node1, node2)
            
            if path_length > 0:
                total_efficiency += 1.0 / path_length
            
            num_pairs += 1
    
    return total_efficiency / num_pairs if num_pairs > 0 else 0.0


# Helper functions

def _shortest_path_length(graph: Graph, source: Node, target: Node) -> int:
    """
    Find shortest path length between two nodes using BFS.
    
    Returns:
        Shortest path length, or -1 if no path exists
    """
    if source == target:
        return 0
    
    from collections import deque
    
    visited = {source}
    queue = deque([(source, 0)])
    
    while queue:
        current, distance = queue.popleft()
        
        for neighbor in graph.neighbors(current):
            if neighbor == target:
                return distance + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1  # No path found