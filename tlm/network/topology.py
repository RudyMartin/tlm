"""
Network topology metrics for TLM.

Provides functions to analyze overall network structure and properties
like density, diameter, degree distribution, etc.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import math

from .graph import Graph, Node

def density(graph: Graph) -> float:
    """
    Calculate network density.
    
    Density is the fraction of possible edges that actually exist.
    
    Args:
        graph: Input graph
    
    Returns:
        Density between 0 and 1
        0 = no edges, 1 = complete graph
    """
    n = graph.num_nodes
    m = graph.num_edges
    
    if n < 2:
        return 0.0
    
    max_edges = n * (n - 1) // 2
    return (2 * m) / (n * (n - 1)) if max_edges > 0 else 0.0


def diameter(graph: Graph) -> int:
    """
    Calculate network diameter (longest shortest path).
    
    Diameter measures the maximum distance between any two connected nodes.
    
    Args:
        graph: Input graph
    
    Returns:
        Diameter of the graph, or -1 if graph is disconnected
    """
    max_distance = 0
    nodes = graph.nodes
    
    for i, source in enumerate(nodes):
        distances = _single_source_shortest_paths(graph, source)
        
        # Check if all nodes are reachable
        if len(distances) != len(nodes):
            return -1  # Graph is disconnected
        
        # Find maximum distance from this source
        max_dist_from_source = max(distances.values())
        max_distance = max(max_distance, max_dist_from_source)
    
    return max_distance


def radius(graph: Graph) -> int:
    """
    Calculate network radius (minimum eccentricity).
    
    Radius is the minimum of all node eccentricities.
    
    Args:
        graph: Input graph
    
    Returns:
        Radius of the graph, or -1 if disconnected
    """
    eccentricities = []
    nodes = graph.nodes
    
    for source in nodes:
        distances = _single_source_shortest_paths(graph, source)
        
        # Check if all nodes are reachable
        if len(distances) != len(nodes):
            return -1  # Graph is disconnected
        
        # Eccentricity is maximum distance from this node
        eccentricity = max(distances.values()) if distances else 0
        eccentricities.append(eccentricity)
    
    return min(eccentricities) if eccentricities else 0


def average_path_length(graph: Graph) -> float:
    """
    Calculate average shortest path length.
    
    Average over all pairs of connected nodes.
    
    Args:
        graph: Input graph
    
    Returns:
        Average path length, or -1 if graph is disconnected
    """
    nodes = graph.nodes
    n = len(nodes)
    
    if n < 2:
        return 0.0
    
    total_distance = 0
    total_pairs = 0
    
    for i, source in enumerate(nodes):
        distances = _single_source_shortest_paths(graph, source)
        
        for j in range(i + 1, len(nodes)):
            target = nodes[j]
            if target in distances:
                total_distance += distances[target]
                total_pairs += 1
            else:
                # Graph is disconnected
                return -1.0
    
    return total_distance / total_pairs if total_pairs > 0 else 0.0


def degree_distribution(graph: Graph) -> Dict[int, int]:
    """
    Calculate degree distribution.
    
    Args:
        graph: Input graph
    
    Returns:
        Dictionary mapping degree values to their frequencies
    """
    degrees = [graph.degree(node) for node in graph.nodes]
    return dict(Counter(degrees))


def degree_histogram(graph: Graph) -> Tuple[List[int], List[int]]:
    """
    Get degree histogram as (degrees, counts) lists.
    
    Args:
        graph: Input graph
    
    Returns:
        Tuple of (degree_values, counts)
    """
    dist = degree_distribution(graph)
    degrees = sorted(dist.keys())
    counts = [dist[d] for d in degrees]
    return degrees, counts


def assortativity_coefficient(graph: Graph) -> float:
    """
    Calculate degree assortativity coefficient.
    
    Measures the tendency of nodes to connect to others with similar degrees.
    > 0: assortative (high degree nodes connect to high degree nodes)
    < 0: disassortative (high degree nodes connect to low degree nodes)
    
    Args:
        graph: Input graph
    
    Returns:
        Assortativity coefficient between -1 and 1
    """
    edges = graph.edges
    
    if not edges:
        return 0.0
    
    # Get degrees for all edge endpoints
    degrees_u = []
    degrees_v = []
    
    for u, v in edges:
        deg_u = graph.degree(u)
        deg_v = graph.degree(v)
        degrees_u.append(deg_u)
        degrees_v.append(deg_v)
    
    # Calculate correlation coefficient
    n = len(edges)
    
    if n == 0:
        return 0.0
    
    # Means
    mean_u = sum(degrees_u) / n
    mean_v = sum(degrees_v) / n
    
    # Covariance and variances
    cov = sum((du - mean_u) * (dv - mean_v) for du, dv in zip(degrees_u, degrees_v))
    var_u = sum((du - mean_u) ** 2 for du in degrees_u)
    var_v = sum((dv - mean_v) ** 2 for dv in degrees_v)
    
    # Correlation coefficient
    denominator = math.sqrt(var_u * var_v)
    return cov / denominator if denominator > 0 else 0.0


def modularity(graph: Graph, communities: List[List[Node]]) -> float:
    """
    Calculate modularity of a community partition.
    
    Modularity measures the quality of a division of nodes into communities.
    Higher values indicate better community structure.
    
    Args:
        graph: Input graph
        communities: List of communities (each community is list of nodes)
    
    Returns:
        Modularity value (typically between -1 and 1)
    """
    m = graph.num_edges
    
    if m == 0:
        return 0.0
    
    # Create community membership mapping
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    
    modularity_sum = 0.0
    
    for u in graph.nodes:
        for v in graph.nodes:
            # Kronecker delta: 1 if same community, 0 otherwise
            delta = 1 if community_map.get(u) == community_map.get(v) else 0
            
            # Actual edge: 1 if edge exists, 0 otherwise
            A_uv = 1 if graph.has_edge(u, v) else 0
            
            # Expected edges under null model
            k_u = graph.degree(u)
            k_v = graph.degree(v)
            expected = (k_u * k_v) / (2 * m)
            
            modularity_sum += (A_uv - expected) * delta
    
    return modularity_sum / (2 * m)


def small_world_coefficient(graph: Graph, niter: int = 5, seed: Optional[int] = None) -> float:
    """
    Calculate small-world coefficient (sigma).
    
    Sigma = (C/C_rand) / (L/L_rand)
    where C is clustering, L is average path length, rand = random graph
    
    Args:
        graph: Input graph
        niter: Number of random graphs to average over
        seed: Random seed for reproducibility
    
    Returns:
        Small-world coefficient
        > 1 indicates small-world properties
    """
    from .clustering import global_clustering_coefficient
    import random
    
    if seed is not None:
        random.seed(seed)
    
    # Original graph properties
    C = global_clustering_coefficient(graph)
    L = average_path_length(graph)
    
    if L <= 0:  # Disconnected graph
        return 0.0
    
    # Generate random graphs and calculate averages
    C_rand_sum = 0.0
    L_rand_sum = 0.0
    
    for _ in range(niter):
        # Create random graph with same degree sequence
        random_graph = _random_graph_same_degree_sequence(graph)
        
        C_rand_sum += global_clustering_coefficient(random_graph)
        L_rand = average_path_length(random_graph)
        if L_rand > 0:
            L_rand_sum += L_rand
    
    C_rand = C_rand_sum / niter
    L_rand = L_rand_sum / niter
    
    if C_rand == 0 or L_rand == 0:
        return 0.0
    
    return (C / C_rand) / (L / L_rand)


def rich_club_coefficient_all(graph: Graph, normalized: bool = False) -> Dict[int, float]:
    """
    Calculate rich club coefficient for all degree values.
    
    Args:
        graph: Input graph
        normalized: Whether to normalize by random expectation
    
    Returns:
        Dictionary mapping degree values to rich club coefficients
    """
    from .clustering import rich_club_coefficient
    
    degrees = set(graph.degree(node) for node in graph.nodes)
    coefficients = {}
    
    for k in sorted(degrees):
        coefficients[k] = rich_club_coefficient(graph, k, normalized)
    
    return coefficients


def efficiency(graph: Graph) -> float:
    """
    Calculate global efficiency of the network.
    
    Efficiency is the average of inverse distances between all pairs.
    More robust than average path length for disconnected graphs.
    
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
    total_pairs = 0
    
    for i, source in enumerate(nodes):
        distances = _single_source_shortest_paths(graph, source)
        
        for j in range(i + 1, len(nodes)):
            target = nodes[j]
            if target in distances and distances[target] > 0:
                total_efficiency += 1.0 / distances[target]
            total_pairs += 1
    
    return total_efficiency / total_pairs if total_pairs > 0 else 0.0


def wiener_index(graph: Graph) -> int:
    """
    Calculate Wiener index (sum of all shortest path lengths).
    
    Args:
        graph: Input graph
    
    Returns:
        Wiener index, or -1 if graph is disconnected
    """
    nodes = graph.nodes
    total_distance = 0
    
    for i, source in enumerate(nodes):
        distances = _single_source_shortest_paths(graph, source)
        
        # Check if all nodes are reachable
        if len(distances) != len(nodes):
            return -1  # Graph is disconnected
        
        # Add distances to nodes with higher indices (avoid double counting)
        for j in range(i + 1, len(nodes)):
            target = nodes[j]
            total_distance += distances[target]
    
    return total_distance


def randic_index(graph: Graph) -> float:
    """
    Calculate Randić index (sum of 1/sqrt(deg(u)*deg(v)) over all edges).
    
    Used in chemical graph theory and network analysis.
    
    Args:
        graph: Input graph
    
    Returns:
        Randić index
    """
    randic_sum = 0.0
    
    for u, v in graph.edges:
        deg_u = graph.degree(u)
        deg_v = graph.degree(v)
        
        if deg_u > 0 and deg_v > 0:
            randic_sum += 1.0 / math.sqrt(deg_u * deg_v)
    
    return randic_sum


# Helper functions

def _single_source_shortest_paths(graph: Graph, source: Node) -> Dict[Node, int]:
    """BFS to find shortest paths from source to all reachable nodes."""
    from collections import deque
    
    distances = {source: 0}
    queue = deque([source])
    
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        
        for neighbor in graph.neighbors(current):
            if neighbor not in distances:
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
    
    return distances


def _random_graph_same_degree_sequence(graph: Graph) -> Graph:
    """
    Create random graph with same degree sequence.
    
    Uses configuration model approach.
    """
    import random
    
    # Create degree sequence
    degree_sequence = [graph.degree(node) for node in graph.nodes]
    nodes = list(graph.nodes)
    
    # Create stubs (half-edges)
    stubs = []
    for i, degree in enumerate(degree_sequence):
        stubs.extend([nodes[i]] * degree)
    
    # Randomly pair stubs to create edges
    random.shuffle(stubs)
    edges = []
    
    for i in range(0, len(stubs), 2):
        if i + 1 < len(stubs):
            u, v = stubs[i], stubs[i + 1]
            if u != v:  # Avoid self-loops
                edges.append((u, v))
    
    return Graph(edges)