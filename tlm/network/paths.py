"""
Path analysis and connectivity functions for network analysis in TLM.

Provides functions for shortest paths, connectivity analysis,
and component detection in networks.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import deque, defaultdict

from .graph import Graph, DirectedGraph, Node

def shortest_path(graph: Graph, source: Node, target: Node) -> List[Node]:
    """
    Find shortest path between two nodes using BFS.
    
    Args:
        graph: Input graph
        source: Source node
        target: Target node
    
    Returns:
        List of nodes representing shortest path, empty if no path exists
    """
    if source == target:
        return [source]
    
    # BFS with path reconstruction
    queue = deque([source])
    visited = {source}
    parent = {source: None}
    
    while queue:
        current = queue.popleft()
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
                
                if neighbor == target:
                    # Reconstruct path
                    path = []
                    node = target
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    return path[::-1]
    
    return []  # No path found


def shortest_path_length(graph: Graph, source: Node, target: Node) -> int:
    """
    Find length of shortest path between two nodes.
    
    Args:
        graph: Input graph
        source: Source node
        target: Target node
    
    Returns:
        Length of shortest path, -1 if no path exists
    """
    if source == target:
        return 0
    
    queue = deque([(source, 0)])
    visited = {source}
    
    while queue:
        current, distance = queue.popleft()
        
        for neighbor in graph.neighbors(current):
            if neighbor == target:
                return distance + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1  # No path found


def all_shortest_paths(graph: Graph, source: Node, target: Node) -> List[List[Node]]:
    """
    Find all shortest paths between two nodes.
    
    Args:
        graph: Input graph
        source: Source node
        target: Target node
    
    Returns:
        List of paths (each path is a list of nodes)
    """
    if source == target:
        return [[source]]
    
    # BFS to find distance and track all predecessors
    queue = deque([source])
    visited = {source: 0}
    predecessors = defaultdict(list)
    
    while queue:
        current = queue.popleft()
        current_dist = visited[current]
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                # First time visiting neighbor
                visited[neighbor] = current_dist + 1
                predecessors[neighbor].append(current)
                queue.append(neighbor)
            elif visited[neighbor] == current_dist + 1:
                # Another shortest path to neighbor
                predecessors[neighbor].append(current)
    
    # Reconstruct all paths
    if target not in visited:
        return []  # No path exists
    
    def _build_paths(node: Node, path: List[Node]) -> List[List[Node]]:
        if node == source:
            return [path[::-1]]
        
        paths = []
        for pred in predecessors[node]:
            paths.extend(_build_paths(pred, path + [pred]))
        return paths
    
    return _build_paths(target, [target])


def single_source_shortest_paths(graph: Graph, source: Node, 
                                cutoff: Optional[int] = None) -> Dict[Node, List[Node]]:
    """
    Find shortest paths from source to all reachable nodes.
    
    Args:
        graph: Input graph
        source: Source node
        cutoff: Maximum path length to consider
    
    Returns:
        Dictionary mapping target nodes to shortest paths
    """
    if source not in graph:
        return {}
    
    paths = {source: [source]}
    queue = deque([source])
    visited = {source: 0}
    
    while queue:
        current = queue.popleft()
        current_dist = visited[current]
        
        if cutoff is not None and current_dist >= cutoff:
            continue
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited[neighbor] = current_dist + 1
                paths[neighbor] = paths[current] + [neighbor]
                queue.append(neighbor)
    
    return paths


def is_connected(graph: Graph) -> bool:
    """
    Check if graph is connected.
    
    Args:
        graph: Input graph
    
    Returns:
        True if graph is connected, False otherwise
    """
    if graph.num_nodes == 0:
        return True
    
    # Start BFS from any node
    start_node = next(iter(graph.nodes))
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == graph.num_nodes


def connected_components(graph: Graph) -> List[List[Node]]:
    """
    Find all connected components in the graph.
    
    Args:
        graph: Input graph
    
    Returns:
        List of components (each component is a list of nodes)
    """
    visited = set()
    components = []
    
    for node in graph.nodes:
        if node not in visited:
            # Start new component
            component = []
            queue = deque([node])
            visited.add(node)
            
            while queue:
                current = queue.popleft()
                component.append(current)
                
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    return components


def strongly_connected_components(graph: DirectedGraph) -> List[List[Node]]:
    """
    Find strongly connected components in directed graph using Tarjan's algorithm.
    
    Args:
        graph: Input directed graph
    
    Returns:
        List of strongly connected components
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = set()
    components = []
    
    def _strongconnect(node: Node):
        # Set depth index for node
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        
        # Consider successors
        for successor in graph.successors(node):
            if successor not in index:
                # Successor not yet visited; recurse
                _strongconnect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                # Successor is in stack and hence in current SCC
                lowlinks[node] = min(lowlinks[node], index[successor])
        
        # If node is root node, pop stack and generate SCC
        if lowlinks[node] == index[node]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == node:
                    break
            components.append(component)
    
    # Start algorithm for each unvisited node
    for node in graph.nodes:
        if node not in index:
            _strongconnect(node)
    
    return components


def weakly_connected_components(graph: DirectedGraph) -> List[List[Node]]:
    """
    Find weakly connected components in directed graph.
    
    Two nodes are weakly connected if there's an undirected path between them.
    
    Args:
        graph: Input directed graph
    
    Returns:
        List of weakly connected components
    """
    visited = set()
    components = []
    
    for node in graph.nodes:
        if node not in visited:
            # Start new component
            component = []
            queue = deque([node])
            visited.add(node)
            
            while queue:
                current = queue.popleft()
                component.append(current)
                
                # Check both successors and predecessors
                neighbors = set(graph.successors(current)) | set(graph.predecessors(current))
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    return components


def is_strongly_connected(graph: DirectedGraph) -> bool:
    """
    Check if directed graph is strongly connected.
    
    Args:
        graph: Input directed graph
    
    Returns:
        True if strongly connected, False otherwise
    """
    components = strongly_connected_components(graph)
    return len(components) == 1


def articulation_points(graph: Graph) -> List[Node]:
    """
    Find articulation points (cut vertices) in the graph.
    
    Articulation points are nodes whose removal increases number of components.
    
    Args:
        graph: Input graph
    
    Returns:
        List of articulation points
    """
    visited = set()
    parent = {}
    low = {}
    disc = {}
    time = [0]
    articulation = set()
    
    def _bridge_util(u: Node):
        # Mark current node as visited
        visited.add(u)
        disc[u] = low[u] = time[0]
        time[0] += 1
        children = 0
        
        # Recurse for all adjacent vertices
        for v in graph.neighbors(u):
            if v not in visited:
                children += 1
                parent[v] = u
                _bridge_util(v)
                
                # Update low value
                low[u] = min(low[u], low[v])
                
                # u is articulation point in following cases:
                # 1) u is root and has more than one child
                if parent.get(u) is None and children > 1:
                    articulation.add(u)
                
                # 2) u is not root and low[v] >= disc[u]
                if parent.get(u) is not None and low[v] >= disc[u]:
                    articulation.add(u)
                    
            elif v != parent.get(u):
                # Update low value for back edge
                low[u] = min(low[u], disc[v])
    
    # Find articulation points in all components
    for node in graph.nodes:
        if node not in visited:
            _bridge_util(node)
    
    return list(articulation)


def bridges(graph: Graph) -> List[Tuple[Node, Node]]:
    """
    Find bridges (cut edges) in the graph.
    
    Bridges are edges whose removal increases number of components.
    
    Args:
        graph: Input graph
    
    Returns:
        List of bridge edges
    """
    visited = set()
    parent = {}
    low = {}
    disc = {}
    time = [0]
    bridges_list = []
    
    def _bridge_util(u: Node):
        # Mark current node as visited
        visited.add(u)
        disc[u] = low[u] = time[0]
        time[0] += 1
        
        # Recurse for all adjacent vertices
        for v in graph.neighbors(u):
            if v not in visited:
                parent[v] = u
                _bridge_util(v)
                
                # Update low value
                low[u] = min(low[u], low[v])
                
                # If low[v] > disc[u], then edge u-v is a bridge
                if low[v] > disc[u]:
                    bridges_list.append((u, v))
                    
            elif v != parent.get(u):
                # Update low value for back edge
                low[u] = min(low[u], disc[v])
    
    # Find bridges in all components
    for node in graph.nodes:
        if node not in visited:
            _bridge_util(node)
    
    return bridges_list


def node_connectivity(graph: Graph, source: Node, target: Node) -> int:
    """
    Calculate node connectivity between two nodes.
    
    Node connectivity is the minimum number of nodes that must be removed
    to disconnect source from target.
    
    Args:
        graph: Input graph
        source: Source node
        target: Target node
    
    Returns:
        Node connectivity (0 if already disconnected)
    """
    if source == target:
        return float('inf')
    
    if not shortest_path(graph, source, target):
        return 0
    
    # Use max-flow min-cut theorem
    # Create auxiliary graph where each node (except source/target) 
    # is split into in-node and out-node with capacity 1
    
    # For simplicity, use iterative node removal approach
    connectivity = 0
    temp_graph = graph.copy()
    
    # Remove nodes one by one until source and target are disconnected
    for node in graph.nodes:
        if node != source and node != target:
            # Try removing this node
            neighbors = temp_graph.neighbors(node)
            temp_graph._nodes.remove(node)
            temp_graph._adjacency.pop(node, None)
            
            # Remove edges to this node
            for neighbor in neighbors:
                if neighbor in temp_graph._adjacency:
                    temp_graph._adjacency[neighbor].discard(node)
            
            connectivity += 1
            
            # Check if source and target are still connected
            if not shortest_path(temp_graph, source, target):
                return connectivity
    
    return connectivity


def edge_connectivity(graph: Graph, source: Node, target: Node) -> int:
    """
    Calculate edge connectivity between two nodes.
    
    Edge connectivity is the minimum number of edges that must be removed
    to disconnect source from target.
    
    Args:
        graph: Input graph
        source: Source node
        target: Target node
    
    Returns:
        Edge connectivity (0 if already disconnected)
    """
    if source == target:
        return float('inf')
    
    paths = all_shortest_paths(graph, source, target)
    if not paths:
        return 0
    
    # Use Ford-Fulkerson algorithm concept
    # For simplicity, use edge-disjoint paths counting
    max_flow = 0
    temp_graph = graph.copy()
    
    while True:
        path = shortest_path(temp_graph, source, target)
        if not path:
            break
        
        # Remove edges along this path
        for i in range(len(path) - 1):
            temp_graph.remove_edge(path[i], path[i + 1])
        
        max_flow += 1
    
    return max_flow


def eccentricity(graph: Graph, node: Node) -> int:
    """
    Calculate eccentricity of a node.
    
    Eccentricity is the maximum distance from the node to any other node.
    
    Args:
        graph: Input graph
        node: Node to calculate eccentricity for
    
    Returns:
        Eccentricity, or -1 if graph is disconnected
    """
    distances = _single_source_shortest_paths(graph, node)
    
    if len(distances) != graph.num_nodes:
        return -1  # Graph is disconnected
    
    return max(distances.values()) if distances else 0


def center(graph: Graph) -> List[Node]:
    """
    Find center nodes (nodes with minimum eccentricity).
    
    Args:
        graph: Input graph
    
    Returns:
        List of center nodes
    """
    eccentricities = {node: eccentricity(graph, node) for node in graph.nodes}
    
    # Filter out disconnected cases
    valid_eccentricities = {node: ecc for node, ecc in eccentricities.items() if ecc >= 0}
    
    if not valid_eccentricities:
        return []
    
    min_eccentricity = min(valid_eccentricities.values())
    return [node for node, ecc in valid_eccentricities.items() if ecc == min_eccentricity]


def periphery(graph: Graph) -> List[Node]:
    """
    Find periphery nodes (nodes with maximum eccentricity).
    
    Args:
        graph: Input graph
    
    Returns:
        List of periphery nodes
    """
    eccentricities = {node: eccentricity(graph, node) for node in graph.nodes}
    
    # Filter out disconnected cases
    valid_eccentricities = {node: ecc for node, ecc in eccentricities.items() if ecc >= 0}
    
    if not valid_eccentricities:
        return []
    
    max_eccentricity = max(valid_eccentricities.values())
    return [node for node, ecc in valid_eccentricities.items() if ecc == max_eccentricity]


# Helper functions

def _single_source_shortest_paths(graph: Graph, source: Node) -> Dict[Node, int]:
    """BFS to find shortest path lengths from source to all reachable nodes."""
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