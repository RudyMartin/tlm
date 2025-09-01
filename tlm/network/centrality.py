"""
Centrality measures for network analysis in TLM.

Provides all major centrality algorithms to identify important nodes
in networks - critical for document ranking, influence analysis, etc.
"""

from typing import Dict, List, Optional
from collections import defaultdict, deque
import math

from .graph import Graph, WeightedGraph, DirectedGraph, Node

def degree_centrality(graph: Graph, normalized: bool = True) -> Dict[Node, float]:
    """
    Calculate degree centrality for all nodes.
    
    Degree centrality measures the fraction of nodes a given node is connected to.
    
    Args:
        graph: Input graph
        normalized: If True, normalize by (n-1) where n is number of nodes
    
    Returns:
        Dictionary mapping nodes to degree centrality values
    """
    centrality = {}
    n = graph.num_nodes
    
    for node in graph.nodes:
        degree = graph.degree(node)
        if normalized and n > 1:
            centrality[node] = degree / (n - 1)
        else:
            centrality[node] = float(degree)
    
    return centrality


def betweenness_centrality(graph: Graph, normalized: bool = True) -> Dict[Node, float]:
    """
    Calculate betweenness centrality for all nodes.
    
    Betweenness centrality measures how often a node lies on shortest paths
    between other nodes - identifies bridges and bottlenecks.
    
    Args:
        graph: Input graph  
        normalized: If True, normalize by ((n-1)(n-2)/2)
    
    Returns:
        Dictionary mapping nodes to betweenness centrality values
    """
    centrality = {node: 0.0 for node in graph.nodes}
    n = graph.num_nodes
    
    # Use Brandes algorithm
    for source in graph.nodes:
        # BFS to find shortest paths
        stack = []
        predecessors = {node: [] for node in graph.nodes}
        sigma = {node: 0.0 for node in graph.nodes}
        sigma[source] = 1.0
        distance = {node: -1 for node in graph.nodes}
        distance[source] = 0
        
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            stack.append(current)
            
            for neighbor in graph.neighbors(current):
                # First time we reach this neighbor?
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[current] + 1
                
                # Shortest path to neighbor via current?
                if distance[neighbor] == distance[current] + 1:
                    sigma[neighbor] += sigma[current]
                    predecessors[neighbor].append(current)
        
        # Accumulation phase
        delta = {node: 0.0 for node in graph.nodes}
        
        while stack:
            node = stack.pop()
            for pred in predecessors[node]:
                delta[pred] += (sigma[pred] / sigma[node]) * (1 + delta[node])
            
            if node != source:
                centrality[node] += delta[node]
    
    # Normalization
    if normalized and n > 2:
        norm = 2.0 / ((n - 1) * (n - 2))
        centrality = {node: c * norm for node, c in centrality.items()}
    
    return centrality


def closeness_centrality(graph: Graph, normalized: bool = True) -> Dict[Node, float]:
    """
    Calculate closeness centrality for all nodes.
    
    Closeness centrality measures how close a node is to all other nodes.
    High closeness = can reach other nodes quickly.
    
    Args:
        graph: Input graph
        normalized: If True, normalize by (n-1)
    
    Returns:
        Dictionary mapping nodes to closeness centrality values
    """
    centrality = {}
    
    for node in graph.nodes:
        distances = _single_source_shortest_path_lengths(graph, node)
        
        # Calculate average distance to reachable nodes
        reachable_distances = [d for d in distances.values() if d > 0]
        
        if not reachable_distances:
            centrality[node] = 0.0
        else:
            avg_distance = sum(reachable_distances) / len(reachable_distances)
            centrality[node] = 1.0 / avg_distance
            
            if normalized:
                # Multiply by fraction of reachable nodes
                n = graph.num_nodes
                if n > 1:
                    centrality[node] *= len(reachable_distances) / (n - 1)
    
    return centrality


def eigenvector_centrality(graph: Graph, max_iter: int = 100, tol: float = 1e-6) -> Dict[Node, float]:
    """
    Calculate eigenvector centrality for all nodes.
    
    Eigenvector centrality measures influence based on the influence of neighbors.
    A node has high eigenvector centrality if it's connected to other high-centrality nodes.
    
    Args:
        graph: Input graph
        max_iter: Maximum number of iterations for power method
        tol: Convergence tolerance
    
    Returns:
        Dictionary mapping nodes to eigenvector centrality values
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    
    if n == 0:
        return {}
    
    # Initialize centrality vector
    centrality = {node: 1.0 / math.sqrt(n) for node in nodes}
    
    # Power iteration to find dominant eigenvector
    for iteration in range(max_iter):
        old_centrality = centrality.copy()
        
        # Matrix-vector multiplication: A * x
        for node in nodes:
            centrality[node] = sum(old_centrality[neighbor] 
                                 for neighbor in graph.neighbors(node))
        
        # Normalize
        norm = math.sqrt(sum(c * c for c in centrality.values()))
        if norm == 0:
            break
            
        centrality = {node: c / norm for node, c in centrality.items()}
        
        # Check convergence
        diff = sum(abs(centrality[node] - old_centrality[node]) 
                  for node in nodes)
        if diff < tol:
            break
    
    return centrality


def pagerank(graph: Graph, alpha: float = 0.85, max_iter: int = 100, 
             tol: float = 1e-6) -> Dict[Node, float]:
    """
    Calculate PageRank centrality for all nodes.
    
    PageRank measures the probability that a random walker ends up at each node.
    Originally developed for ranking web pages.
    
    Args:
        graph: Input graph
        alpha: Damping parameter (probability of following an edge vs random jump)
        max_iter: Maximum number of iterations  
        tol: Convergence tolerance
    
    Returns:
        Dictionary mapping nodes to PageRank values
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    
    if n == 0:
        return {}
    
    # Initialize PageRank values
    pagerank_values = {node: 1.0 / n for node in nodes}
    
    for iteration in range(max_iter):
        old_values = pagerank_values.copy()
        
        for node in nodes:
            # PageRank contribution from neighbors
            rank_sum = sum(old_values[neighbor] / graph.degree(neighbor)
                          for neighbor in graph.neighbors(node))
            
            # PageRank formula with damping
            pagerank_values[node] = (1 - alpha) / n + alpha * rank_sum
        
        # Check convergence
        diff = sum(abs(pagerank_values[node] - old_values[node]) 
                  for node in nodes)
        if diff < tol:
            break
    
    return pagerank_values


def katz_centrality(graph: Graph, alpha: float = 0.1, beta: float = 1.0,
                   max_iter: int = 100, tol: float = 1e-6) -> Dict[Node, float]:
    """
    Calculate Katz centrality for all nodes.
    
    Katz centrality measures influence by counting walks of all lengths,
    with longer walks weighted exponentially less.
    
    Args:
        graph: Input graph
        alpha: Attenuation factor (should be < 1/λ_max)
        beta: Weight given to immediate neighbors
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Dictionary mapping nodes to Katz centrality values
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    
    if n == 0:
        return {}
    
    # Initialize centrality
    centrality = {node: beta for node in nodes}
    
    for iteration in range(max_iter):
        old_centrality = centrality.copy()
        
        for node in nodes:
            # Katz centrality formula
            neighbor_sum = sum(old_centrality[neighbor] 
                             for neighbor in graph.neighbors(node))
            centrality[node] = beta + alpha * neighbor_sum
        
        # Check convergence
        diff = sum(abs(centrality[node] - old_centrality[node]) 
                  for node in nodes)
        if diff < tol:
            break
    
    return centrality


def load_centrality(graph: Graph, normalized: bool = True) -> Dict[Node, float]:
    """
    Calculate load centrality (stress centrality) for all nodes.
    
    Load centrality measures the fraction of shortest paths that pass through each node.
    Similar to betweenness but counts paths rather than pairs.
    
    Args:
        graph: Input graph
        normalized: If True, normalize values
    
    Returns:
        Dictionary mapping nodes to load centrality values
    """
    load = {node: 0.0 for node in graph.nodes}
    n = graph.num_nodes
    
    # For each pair of nodes, find all shortest paths
    for source in graph.nodes:
        # Single source shortest paths with path counting
        distances, path_counts, predecessors = _shortest_paths_with_counts(graph, source)
        
        for target in graph.nodes:
            if source != target and target in distances:
                # Add load for all nodes on shortest paths from source to target
                _add_load_for_paths(graph, source, target, predecessors, 
                                   path_counts, load)
    
    # Normalization
    if normalized and n > 2:
        max_possible = (n - 1) * (n - 2)
        load = {node: l / max_possible for node, l in load.items()}
    
    return load


def harmonic_centrality(graph: Graph, normalized: bool = True) -> Dict[Node, float]:
    """
    Calculate harmonic centrality for all nodes.
    
    Harmonic centrality is the sum of reciprocal distances to all other nodes.
    Better than closeness for disconnected graphs.
    
    Args:
        graph: Input graph
        normalized: If True, normalize by (n-1)
    
    Returns:
        Dictionary mapping nodes to harmonic centrality values
    """
    centrality = {}
    n = graph.num_nodes
    
    for node in graph.nodes:
        distances = _single_source_shortest_path_lengths(graph, node)
        
        # Sum of reciprocal distances (excluding self)
        harmonic_sum = sum(1.0 / d for d in distances.values() if d > 0)
        
        if normalized and n > 1:
            centrality[node] = harmonic_sum / (n - 1)
        else:
            centrality[node] = harmonic_sum
    
    return centrality


# Helper functions

def _single_source_shortest_path_lengths(graph: Graph, source: Node) -> Dict[Node, int]:
    """BFS to find shortest path lengths from source to all other nodes."""
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


def _shortest_paths_with_counts(graph: Graph, source: Node):
    """
    BFS variant that also counts number of shortest paths.
    
    Returns:
        distances: shortest distances from source
        path_counts: number of shortest paths to each node
        predecessors: predecessors on shortest paths
    """
    distances = {source: 0}
    path_counts = {source: 1}
    predecessors = defaultdict(list)
    queue = deque([source])
    
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        
        for neighbor in graph.neighbors(current):
            if neighbor not in distances:
                # First time reaching neighbor
                distances[neighbor] = current_distance + 1
                path_counts[neighbor] = path_counts[current]
                predecessors[neighbor] = [current]
                queue.append(neighbor)
            elif distances[neighbor] == current_distance + 1:
                # Another shortest path to neighbor
                path_counts[neighbor] += path_counts[current]
                predecessors[neighbor].append(current)
    
    return distances, path_counts, predecessors


def _add_load_for_paths(graph: Graph, source: Node, target: Node,
                       predecessors: Dict, path_counts: Dict,
                       load: Dict[Node, float]) -> None:
    """Add load contribution for shortest paths from source to target."""
    if target not in path_counts:
        return
    
    visited = set()
    stack = [target]
    
    # Traverse backwards from target to source
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        
        if node != source and node != target:
            # This node is on a shortest path
            load[node] += 1.0 / path_counts[target]
        
        # Add predecessors to stack
        for pred in predecessors[node]:
            if pred not in visited:
                stack.append(pred)