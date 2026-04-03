import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def plot_graph(adj_matrix):
    G = nx.from_scipy_sparse_array(adj_matrix)
    pos = nx.spring_layout(G)
    fig = plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_color='lightblue', edge_color='gray', node_size=10, with_labels=False)
    plt.show()

def grid(N, periodic=False):
    G = nx.grid_2d_graph(N, N, periodic=periodic)
    adj = nx.adjacency_matrix(G)
    # Nodes are (row, col) tuples — use directly as positions
    pos = np.array([[col, -row] for row, col in G.nodes()], dtype=np.float64)
    return adj, pos

def load_graphml(filename):
    G = nx.read_graphml(filename)
    adj = nx.adjacency_matrix(G)
    # Check if nodes have x/y position attributes
    pos = None
    sample_node = list(G.nodes(data=True))[0][1]
    if 'x' in sample_node and 'y' in sample_node:
        pos = np.array([[float(data['x']), float(data['y'])] for _, data in G.nodes(data=True)], dtype=np.float64)
    return adj, pos
