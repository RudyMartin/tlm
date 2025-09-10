"""
Core graph data structures for network analysis in TLM.

Provides pure Python implementations of Graph and WeightedGraph classes
with all essential operations for network statistics.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque

# Type definitions
Node = Union[int, str]
Edge = Tuple[Node, Node]
WeightedEdge = Tuple[Node, Node, float]

class Graph:
    """
    Undirected graph implementation using adjacency lists.
    
    Pure Python implementation with zero dependencies.
    Optimized for network statistics and analysis.
    """
    
    def __init__(self, edges: Optional[List[Edge]] = None):
        """
        Initialize graph from list of edges.
        
        Args:
            edges: List of (node1, node2) tuples
        """
        self._adjacency: Dict[Node, Set[Node]] = defaultdict(set)
        self._nodes: Set[Node] = set()
        self._edge_count = 0
        
        if edges:
            for edge in edges:
                self.add_edge(edge[0], edge[1])
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        if node not in self._nodes:
            self._nodes.add(node)
            self._adjacency[node] = set()
    
    def add_edge(self, u: Node, v: Node) -> None:
        """Add an edge between nodes u and v."""
        self.add_node(u)
        self.add_node(v)
        
        if v not in self._adjacency[u]:
            self._adjacency[u].add(v)
            self._adjacency[v].add(u)
            self._edge_count += 1
    
    def remove_edge(self, u: Node, v: Node) -> None:
        """Remove edge between nodes u and v."""
        if u in self._adjacency and v in self._adjacency[u]:
            self._adjacency[u].remove(v)
            self._adjacency[v].remove(u)
            self._edge_count -= 1
    
    def has_edge(self, u: Node, v: Node) -> bool:
        """Check if edge exists between nodes u and v."""
        return u in self._adjacency and v in self._adjacency[u]
    
    def neighbors(self, node: Node) -> List[Node]:
        """Get list of neighbors for a node."""
        return list(self._adjacency.get(node, set()))
    
    def degree(self, node: Node) -> int:
        """Get degree of a node."""
        return len(self._adjacency.get(node, set()))
    
    @property
    def nodes(self) -> List[Node]:
        """Get list of all nodes."""
        return list(self._nodes)
    
    @property
    def edges(self) -> List[Edge]:
        """Get list of all edges."""
        edges = []
        visited = set()
        
        for u in self._nodes:
            for v in self._adjacency[u]:
                edge = tuple(sorted([u, v]))
                if edge not in visited:
                    edges.append((u, v))
                    visited.add(edge)
        
        return edges
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in graph."""
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
        """Number of edges in graph."""
        return self._edge_count
    
    def adjacency_matrix(self) -> List[List[int]]:
        """
        Get adjacency matrix representation.
        
        Returns:
            n x n matrix where matrix[i][j] = 1 if edge exists, 0 otherwise
        """
        nodes = sorted(self._nodes)
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        for u in nodes:
            for v in self._adjacency[u]:
                i, j = node_to_idx[u], node_to_idx[v]
                matrix[i][j] = 1
        
        return matrix
    
    def subgraph(self, nodes: List[Node]) -> 'Graph':
        """Create subgraph containing only specified nodes."""
        node_set = set(nodes)
        subgraph_edges = []
        
        for u, v in self.edges:
            if u in node_set and v in node_set:
                subgraph_edges.append((u, v))
        
        return Graph(subgraph_edges)
    
    def copy(self) -> 'Graph':
        """Create a copy of the graph."""
        return Graph(self.edges)
    
    def __len__(self) -> int:
        """Number of nodes in graph."""
        return self.num_nodes
    
    def __contains__(self, node: Node) -> bool:
        """Check if node is in graph."""
        return node in self._nodes
    
    def __iter__(self):
        """Iterator over nodes."""
        return iter(self._nodes)


class WeightedGraph(Graph):
    """
    Weighted undirected graph extending base Graph class.
    
    Adds edge weights while maintaining all base functionality.
    """
    
    def __init__(self, edges: Optional[List[WeightedEdge]] = None):
        """
        Initialize weighted graph from list of weighted edges.
        
        Args:
            edges: List of (node1, node2, weight) tuples
        """
        super().__init__()
        self._weights: Dict[Tuple[Node, Node], float] = {}
        
        if edges:
            for edge in edges:
                if len(edge) == 3:
                    self.add_edge(edge[0], edge[1], edge[2])
                else:
                    self.add_edge(edge[0], edge[1], 1.0)
    
    def add_edge(self, u: Node, v: Node, weight: float = 1.0) -> None:
        """Add weighted edge between nodes u and v."""
        super().add_edge(u, v)
        
        # Store weight for both directions (undirected graph)
        self._weights[(u, v)] = weight
        self._weights[(v, u)] = weight
    
    def remove_edge(self, u: Node, v: Node) -> None:
        """Remove weighted edge between nodes u and v."""
        super().remove_edge(u, v)
        
        # Remove weights
        self._weights.pop((u, v), None)
        self._weights.pop((v, u), None)
    
    def get_edge_weight(self, u: Node, v: Node) -> Optional[float]:
        """Get weight of edge between nodes u and v."""
        return self._weights.get((u, v))
    
    def set_edge_weight(self, u: Node, v: Node, weight: float) -> None:
        """Set weight of existing edge."""
        if self.has_edge(u, v):
            self._weights[(u, v)] = weight
            self._weights[(v, u)] = weight
    
    @property
    def weighted_edges(self) -> List[WeightedEdge]:
        """Get list of all weighted edges."""
        edges = []
        visited = set()
        
        for u in self._nodes:
            for v in self._adjacency[u]:
                edge = tuple(sorted([u, v]))
                if edge not in visited:
                    weight = self._weights[(u, v)]
                    edges.append((u, v, weight))
                    visited.add(edge)
        
        return edges
    
    def adjacency_matrix(self) -> List[List[float]]:
        """
        Get weighted adjacency matrix.
        
        Returns:
            n x n matrix where matrix[i][j] = weight if edge exists, 0 otherwise
        """
        nodes = sorted(self._nodes)
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for u in nodes:
            for v in self._adjacency[u]:
                i, j = node_to_idx[u], node_to_idx[v]
                matrix[i][j] = self._weights[(u, v)]
        
        return matrix
    
    def total_weight(self) -> float:
        """Get total weight of all edges."""
        return sum(weight for (u, v), weight in self._weights.items() if u <= v)
    
    def copy(self) -> 'WeightedGraph':
        """Create a copy of the weighted graph."""
        return WeightedGraph(self.weighted_edges)


class DirectedGraph:
    """
    Directed graph implementation using adjacency lists.
    
    Supports directed network analysis including strongly connected components.
    """
    
    def __init__(self, edges: Optional[List[Edge]] = None):
        """
        Initialize directed graph from list of edges.
        
        Args:
            edges: List of (source, target) tuples
        """
        self._adjacency: Dict[Node, Set[Node]] = defaultdict(set)
        self._reverse_adjacency: Dict[Node, Set[Node]] = defaultdict(set)
        self._nodes: Set[Node] = set()
        self._edge_count = 0
        
        if edges:
            for edge in edges:
                self.add_edge(edge[0], edge[1])
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        if node not in self._nodes:
            self._nodes.add(node)
            self._adjacency[node] = set()
            self._reverse_adjacency[node] = set()
    
    def add_edge(self, u: Node, v: Node) -> None:
        """Add directed edge from u to v."""
        self.add_node(u)
        self.add_node(v)
        
        if v not in self._adjacency[u]:
            self._adjacency[u].add(v)
            self._reverse_adjacency[v].add(u)
            self._edge_count += 1
    
    def has_edge(self, u: Node, v: Node) -> bool:
        """Check if directed edge exists from u to v."""
        return u in self._adjacency and v in self._adjacency[u]
    
    def successors(self, node: Node) -> List[Node]:
        """Get list of successors (outgoing neighbors) for a node."""
        return list(self._adjacency.get(node, set()))
    
    def predecessors(self, node: Node) -> List[Node]:
        """Get list of predecessors (incoming neighbors) for a node."""
        return list(self._reverse_adjacency.get(node, set()))
    
    def out_degree(self, node: Node) -> int:
        """Get out-degree of a node."""
        return len(self._adjacency.get(node, set()))
    
    def in_degree(self, node: Node) -> int:
        """Get in-degree of a node."""
        return len(self._reverse_adjacency.get(node, set()))
    
    @property
    def nodes(self) -> List[Node]:
        """Get list of all nodes."""
        return list(self._nodes)
    
    @property
    def edges(self) -> List[Edge]:
        """Get list of all directed edges."""
        edges = []
        for u in self._nodes:
            for v in self._adjacency[u]:
                edges.append((u, v))
        return edges
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in graph."""
        return len(self._nodes)
    
    @property 
    def num_edges(self) -> int:
        """Number of edges in graph."""
        return self._edge_count
    
    def reverse(self) -> 'DirectedGraph':
        """Get reverse graph (all edges reversed)."""
        reversed_edges = [(v, u) for u, v in self.edges]
        return DirectedGraph(reversed_edges)