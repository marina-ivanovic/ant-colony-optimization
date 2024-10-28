import random
import networkx as nx
import numpy as np
import pandas as pd

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to initialize pheromones on graph edges
def initialize_pheromones(graph):
    pheromones = {(min(node1, node2), max(node1, node2)): 1.0 for node1, node2 in graph.edges()}
    return pheromones

# Function to load data from a file and create a graph
def load_data(file_name):
    data = pd.read_csv(file_name, sep=':', header=None, names=['node', 'neighbors'])
    graph = nx.Graph()

    for index, row in data.iterrows():
        # Extract node information
        node_id = int(row['node'].split('(')[0])
        coordinates = [float(value) for value in row['node'].split('(')[1][:-1].split(',')]
        graph.add_node(node_id, coordinates=coordinates)

        # Extract neighbor information
        if isinstance(row['neighbors'], str):
            neighbors = [int(neighbor) for neighbor in row['neighbors'].split(',')]
            graph.add_edges_from((node_id, neighbor) for neighbor in neighbors)

    return graph

# Function to choose the next node based on pheromone levels and heuristic information
def choose_next_node(graph, current_node, available_nodes, pheromones, alpha, beta):
    # Calculate total probability
    total = sum((pheromones[(min(current_node, node), max(current_node, node))] ** alpha) * 
                (1.0 / max(1e-10, euclidean_distance(graph.nodes[current_node]["coordinates"], graph.nodes[node]["coordinates"]) ** beta))
                for node in available_nodes)
    
    # Calculate probabilities for each available node
    probabilities = [(pheromones[(min(current_node, node), max(current_node, node))] ** alpha) * 
                     (1.0 / max(1e-10, euclidean_distance(graph.nodes[current_node]["coordinates"], graph.nodes[node]["coordinates"]) ** beta))
                     / total for node in available_nodes]

    # Additional check to ensure the current_node is in available_nodes before accessing coordinates
    if current_node not in graph.nodes:
        raise ValueError(f"Node {current_node} not found in the graph.")
    
    # Use weighted random selection to choose the next node
    chosen_node = random.choices(available_nodes, weights=probabilities)[0]
    return chosen_node

# Function to update pheromones on edges based on the chosen path
def update_pheromones(path, pheromones, q, path_length, decay_factor):
    for i in range(len(path) - 1):
        edge = (min(path[i], path[i+1]), max(path[i], path[i+1]))
        pheromones[edge] += q / path_length

    # Apply pheromone decay to all edges
    for edge in pheromones:
        pheromones[edge] *= decay_factor

# Ant Colony Optimization algorithm
def aco_algorithm(graph, start_node, end_node, num_ants, num_iterations, alpha, beta, q, decay_factor):
    pheromones = initialize_pheromones(graph)
    best_path = None
    best_length = float('inf')

    # Main ACO loop
    for _ in range(num_iterations):
        for _ in range(num_ants):
            current_node = start_node
            ant_path = [current_node]

            # Move through the graph until reaching the end_node
            while current_node != end_node:
                available_nodes = list(graph.neighbors(current_node))
                chosen_node = choose_next_node(graph, current_node, available_nodes, pheromones, alpha, beta)
                pheromones[(min(current_node, chosen_node), max(current_node, chosen_node))] *= decay_factor
                ant_path.append(chosen_node)
                current_node = chosen_node

            # Calculate the length of the path
            path_length = sum(euclidean_distance(graph.nodes[ant_path[i]]["coordinates"], graph.nodes[ant_path[i + 1]]["coordinates"]) for i in range(len(ant_path) - 1))

            # Update best path if a shorter one is found
            if path_length < best_length:
                best_length = path_length
                best_path = ant_path

            # Update pheromones based on the ant's path
            update_pheromones(ant_path, pheromones, q, path_length, decay_factor)

    return best_path, best_length

if __name__ == "__main__":
    # Load graph from data file
    graph = load_data("data_path_nodes.txt")
    
    start_node, end_node = 3653296222, 3653134376
    
    # Define ACO parameters
    # Explanation of ACO parameters:
    # - num_ants: Number of ants deployed in the ACO algorithm. More ants explore a larger part of the solution space.
    # - num_iterations: Number of iterations the ACO algorithm runs. Each iteration allows ants to construct paths and update pheromone levels.
    # - alpha: Pheromone influence parameter. Higher alpha emphasizes the impact of pheromones in ant decision-making.
    # - beta: Heuristic influence parameter. Higher beta gives more weight to heuristic information (e.g., distance between nodes) in decision-making.
    # - q: Pheromone deposit parameter. It represents the amount of pheromone deposited by an ant on its path. Higher q results in more pheromone deposition.
    # - decay_factor: Pheromone decay factor. It controls the rate at which pheromones evaporate over time. A higher value means slower decay.
    num_ants = 10
    num_iterations = 5
    alpha = 3.0
    beta = 1.0
    q = 2.0
    decay_factor = 0.95

    best_path, best_length = aco_algorithm(graph, start_node, end_node, num_ants, num_iterations, alpha, beta, q, decay_factor)

    print("Best path:", best_path)
    print("Best path length:", best_length)