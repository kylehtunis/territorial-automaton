import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
from tqdm import tqdm

_STRONG_A = np.int8(0)
_WEAK_A = np.int8(1)
_FACTIONLESS = np.int8(2)
_WEAK_B = np.int8(3)
_STRONG_B = np.int8(4)
N_STATES = 5

class TA_Params:
    def __init__(self, adj_matrix, initial_state=None, T=1.0, h=0.0, w=0.5, g=0.5, seed=None):
        self.adj_matrix = adj_matrix
        self.adj_indptr = adj_matrix.indptr
        self.adj_indices = adj_matrix.indices
        self.adj_data = adj_matrix.data
        self.N = adj_matrix.shape[0]
        self.T = T
        self.h = h
        self.w = w
        self.g = g
        self.seed = seed
        if initial_state is None:
            self.initial_state = np.full(self.N, _FACTIONLESS, dtype=np.int8)
        else:
            self.initial_state = initial_state

    def __str__(self):
        return f"TA_Params(N={self.N}, T={self.T}, h={self.h}, w={self.w}, g={self.g})"
    
class TA_Metrics:
    def __init__(self, state, energy, scalars, snapshot=None):
        self.faction_sizes = np.bincount(state, minlength=N_STATES)
        self.energy = energy
        self.order = np.mean(scalars[state])  # Map states to [1, w, 0, -w, -1] and take mean
        self.snapshot = snapshot
    def __str__(self):
        return f"TA_Metrics(faction_sizes={self.faction_sizes}, energy={self.energy:.4f}, order={self.order:.4f}, snapshot={'Yes' if self.snapshot is not None else 'No'})"
    
class TerritorialAutomaton:
    def __init__(self, params):
        self.params = params
        self.state = np.copy(params.initial_state)
        self.nodes = np.arange(self.params.N)
        self.interaction_matrix = self.compute_interaction_matrix()
        self.state_scalars = np.array([1, self.params.w, 0, -self.params.w, -1], dtype=np.float32)
        self.h_weights = -1 * self.params.h * self.state_scalars
        if self.params.seed is not None:
            self.rng = np.random.default_rng(self.params.seed)
        else:
            self.rng = np.random.default_rng()

    def compute_interaction_matrix(self):
        interaction_matrix = np.array([[-1, -self.params.w, self.params.g, self.params.w, 1],
                                       [-self.params.w, -(self.params.w**2), self.params.g*self.params.w, self.params.w**2, self.params.w],
                                       [self.params.g, self.params.g*self.params.w, 0, self.params.w*self.params.g, self.params.g],
                                       [self.params.w, self.params.w**2, self.params.g*self.params.w, -self.params.w**2, -self.params.w],
                                        [1, self.params.w, self.params.g, -self.params.w, -1]])
        return interaction_matrix

    def step(self):
        node_order = self.rng.permutation(self.nodes)
        # Generate random numbers for stochastic choice
        rng_rolls = self.rng.random(size=self.params.N)
        _step(
            node_order,
            self.state,
            self.params.adj_indptr,
            self.params.adj_indices,
            self.params.adj_data,
            self.interaction_matrix,
            self.h_weights,
            self.params.T,
            rng_rolls
        )

    def run(self, n_warmup, n_experiment, snapshots=False, progbar=False):
        self.state = np.copy(self.params.initial_state) # Ensure we start from the initial state
        metrics = []
        for _ in tqdm(range(n_warmup), desc="Warming up", disable=not progbar):
            self.step()
        for _ in tqdm(range(n_experiment), desc="Running experiment", disable=not progbar):
            self.step()
            energy = _compute_total_energy(
                self.nodes,
                self.state,
                self.params.adj_indptr,
                self.params.adj_indices,
                self.params.adj_data,
                self.interaction_matrix,
                self.h_weights
            )
            snapshot = np.copy(self.state) if snapshots else None
            metrics.append(TA_Metrics(self.state, energy, self.state_scalars, snapshot))
        return metrics

@njit
def _step(node_order, states, adj_indptr, adj_indices, adj_data, interaction_matrix, h_weights, T, rng_rolls):
    N = len(states)
    for node in node_order:
        states[node] = _update_node(
            node,
            states,
            adj_indptr,
            adj_indices,
            adj_data,
            interaction_matrix,
            h_weights,
            T,
            rng_rolls
        )

@njit
def _update_node(node_idx, states, adj_indptr, adj_indices, adj_data, interaction_matrix, h_weights, T, rng_rolls):
    """
    Update the state of a single node based on Heat Bath dynamics.

    Parameters:
    - node_idx: Index of the node to update.
    - states: Current states of all nodes (1D array length N).
    - adj_indptr: Row pointers of the CSR adjacency matrix.
    - adj_indices: Column indices of the CSR adjacency matrix.
    - adj_data: Data elements of the CSR adjacency matrix.
    - interaction_matrix: Matrix defining interactions between states (N_STATES x N_STATES).
    - h_weights: External field weights for each state (1D array length N_STATES).
    - T: Temperature parameter controlling randomness.
    - rng_rolls: Pre-generated random numbers for stochastic choice (1D array length N_STATES).
    """
    local_energy = np.zeros(N_STATES, dtype=np.float64)
    neighbors_start = adj_indptr[node_idx]
    neighbors_end = adj_indptr[node_idx + 1]
    for s in range(N_STATES):
        local_energy[s] = h_weights[s]
        for i in range(neighbors_start, neighbors_end):
            weight = adj_data[i]
            local_energy[s] += weight * interaction_matrix[s, states[adj_indices[i]]]
    adjusted_energy = (local_energy/T) - np.min(local_energy/T)
    probabilities = np.exp(-adjusted_energy) / np.sum(np.exp(-adjusted_energy))
    new_state = min(np.searchsorted(np.cumsum(probabilities), rng_rolls[node_idx]), N_STATES - 1)
    return new_state

@njit
def _compute_total_energy(nodes, states, adj_indptr, adj_indices, adj_data, interaction_matrix, h_weights):
    total_energy = 0.0
    for node_idx in nodes:
        s = states[node_idx]
        local_energy = h_weights[s]
        neighbors_start = adj_indptr[node_idx]
        neighbors_end = adj_indptr[node_idx + 1]
        for i in range(neighbors_start, neighbors_end):
            weight = adj_data[i]
            local_energy += 0.5 * (weight * interaction_matrix[s, states[adj_indices[i]]]) # 0.5 to avoid double on undirected graphs
        total_energy += local_energy
    return total_energy
