import networkx as nx
import heapq

def build_shortest_path_tree(graph, source):
    """
    Build a shortest path tree using NetworkX's built-in single_source_dijkstra function.
    """
    distances, paths = nx.single_source_dijkstra(graph, source, weight='weight')
    
    # Create a predecessors dictionary
    predecessors = {}
    for target, path in paths.items():
        if len(path) > 1:
            predecessors[target] = path[-2]
    
    return distances, predecessors

def reconstruct_path(predecessors, source, target):
    """
    Reconstruct the shortest path from source to target using predecessors.
    """
    path = [target]
    current = target
    
    while current != source:
        if current not in predecessors:
            return None  # No path exists
        current = predecessors[current]
        path.append(current)
    
    return list(reversed(path))

def hoffman_k_shortest_paths(graph, source, target, k):
    components = [graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)]
    
    k_paths = []
    for component in components:
        if source in component.nodes and target in component.nodes:
            k_paths.extend(find_paths_in_component(component, source, target, k))
    
    return k_paths

def find_paths_in_component(graph, source, target, k):
    # Build shortest path tree first
    distances, predecessors = build_shortest_path_tree(graph, source)
    
    k_paths = []
    candidates = []
    visited_paths = set()
    
    # Initial shortest path
    initial_path = reconstruct_path(predecessors, source, target)
    initial_distance = nx.path_weight(graph, initial_path, weight='weight')
    heapq.heappush(candidates, (initial_distance, initial_path))
    
    while candidates and len(k_paths) < k:
        current_distance, current_path = heapq.heappop(candidates)
        current_path_tuple = tuple(current_path)
        
        if current_path_tuple in visited_paths:
            continue
        
        visited_paths.add(current_path_tuple)
        k_paths.append(current_path)
        
        for i in range(len(current_path) - 1):
            spur_node = current_path[i]
            root_path = current_path[:i+1]
            
            temp_graph = graph.copy()
            for path in k_paths:
                if path[:i+1] == root_path:
                    try:
                        temp_graph.remove_edge(path[i], path[i+1])
                    except Exception:
                        pass
            
            # Skip if no path found or spur node is the target
            try:
                # Use modified shortest path method with tree information
                spur_distances, spur_predecessors = build_shortest_path_tree(temp_graph, spur_node)
                spur_path = reconstruct_path(spur_predecessors, spur_node, target)
                
                # Only proceed if a valid spur path is found
                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    total_distance = nx.path_weight(graph, total_path, weight='weight')
                    
                    if tuple(total_path) not in visited_paths:
                        heapq.heappush(candidates, (total_distance, total_path))
            except Exception:
                continue
    
    return k_paths

# Example usage remains the same
G = nx.DiGraph()
G.add_weighted_edges_from([
    ('A', 'B', 4),
    ('A', 'C', 2),
    ('B', 'C', 1),
    ('B', 'D', 5),
    ('C', 'D', 8),
    ('C', 'E', 10),
    ('D', 'E', 2),
    ('D', 'F', 6),
    ('E', 'F', 3),
    # Isolated component
    ('X', 'Y', 1),
    ('Y', 'Z', 2)
])

import networkx as nx
import random
import time

def generate_large_weighted_graph(num_nodes, num_edges, max_weight=100):
    G = nx.DiGraph()
    
    # Ensure graph is connected
    for i in range(1, num_nodes):
        # Connect each node to a previous node
        prev_node = random.randint(0, i-1)
        weight = random.randint(1, max_weight)
        G.add_weighted_edges_from([(prev_node, i, weight)])
    
    # Add additional random edges
    while G.number_of_edges() < num_edges:
        u = random.randint(0, num_nodes-1)
        v = random.randint(0, num_nodes-1)
        if u != v and not G.has_edge(u, v):
            weight = random.randint(1, max_weight)
            G.add_weighted_edges_from([(u, v, weight)])
    
    return G

# Generate a large graph
random.seed(42)  # For reproducibility
large_graph = generate_large_weighted_graph(num_nodes=10000, num_edges=50000)

# Measure time for k-shortest paths
start_time = time.time()
k_shortest = hoffman_k_shortest_paths(large_graph, 0, 999, 50)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
print("Number of k-shortest paths found:", len(k_shortest))

# Print details of found paths
for i, path in enumerate(k_shortest, 1):
    print(f"Path {i}:", path, 
          f"Weight: {nx.path_weight(large_graph, path, weight='weight')}")