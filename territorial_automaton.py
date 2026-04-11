import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
from tqdm import tqdm

STRONG_A = np.int8(0)
WEAK_A = np.int8(1)
FACTIONLESS = np.int8(2)
WEAK_B = np.int8(3)
STRONG_B = np.int8(4)
N_STATES = 5

class TA_Params:
    def __init__(self, adj_matrix, initial_state=None, T=1.0, h=0.0, theta=0.5, kappa=0.5, seed=None):
        self.adj_matrix = adj_matrix
        self.adj_indptr = adj_matrix.indptr
        self.adj_indices = adj_matrix.indices
        self.adj_data = adj_matrix.data
        self.N = adj_matrix.shape[0]
        self.T = T
        self.h = h
        self.theta = theta
        self.kappa = kappa
        self.seed = seed
        if initial_state is None:
            self.initial_state = np.full(self.N, FACTIONLESS, dtype=np.int8)
        else:
            self.initial_state = initial_state

    def __str__(self):
        return f"TA_Params(N={self.N}, T={self.T}, h={self.h}, theta={self.theta}, kappa={self.kappa})"
    
class TA_Metrics:
    def __init__(self, state, energy, scalars, snapshot=None):
        self.faction_sizes = np.bincount(state, minlength=N_STATES)
        self.faction_a = self.faction_sizes[STRONG_A] + self.faction_sizes[WEAK_A]
        self.faction_b = self.faction_sizes[STRONG_B] + self.faction_sizes[WEAK_B]
        self.energy = energy
        self.order = np.mean(scalars[state])  # Map states to [1, theta, 0, -theta, -1] and take mean
        self.snapshot = snapshot
    def __str__(self):
        return f"TA_Metrics(faction_sizes={self.faction_sizes}, energy={self.energy:.4f}, order={self.order:.4f}, snapshot={'Yes' if self.snapshot is not None else 'No'})"
    
class TA_Result:
    def __init__(self, metrics: list[TA_Metrics]):
        self.metrics = metrics
        self.orders = np.array([m.order for m in metrics])
        self.energies = np.array([m.energy for m in metrics])
        self.faction_sizes = np.zeros((N_STATES, len(metrics)), dtype=np.int32)
        for i in range(N_STATES):
            self.faction_sizes[i] = np.array([m.faction_sizes[i] for m in metrics])
    
class TerritorialAutomaton:
    def __init__(self, params):
        self.params = params
        self.state = np.copy(params.initial_state)
        self.nodes = np.arange(self.params.N)
        self.interaction_matrix = self.compute_interaction_matrix()
        self.state_scalars = np.array([1, self.params.theta, 0, -self.params.theta, -1], dtype=np.float32)
        self.h_weights = -1 * self.params.h * self.state_scalars
        if self.params.seed is not None:
            self.rng = np.random.default_rng(self.params.seed)
        else:
            self.rng = np.random.default_rng()

    def compute_interaction_matrix(self):
        interaction_matrix = np.array([[-1, -self.params.theta, self.params.kappa, self.params.theta, 1],
                                       [-self.params.theta, -(self.params.theta**2), self.params.kappa*self.params.theta, self.params.theta**2, self.params.theta],
                                       [self.params.kappa, self.params.kappa*self.params.theta, 0, self.params.theta*self.params.kappa, self.params.kappa],
                                       [self.params.theta, self.params.theta**2, self.params.kappa*self.params.theta, -self.params.theta**2, -self.params.theta],
                                        [1, self.params.theta, self.params.kappa, -self.params.theta, -1]])
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
            self.T,
            rng_rolls
        )

    def run(self, n_warmup, n_experiment, initial_state=None, dynamic_T=None, snapshots=False, progbar=False):
        if dynamic_T is not None:
            dynamic_T = np.asarray(dynamic_T, dtype=np.float64)
            if dynamic_T.shape != (n_experiment,):
                raise ValueError(f"dynamic_T must have length n_experiment ({n_experiment}), got {dynamic_T.shape}")
        self.state = np.copy(self.params.initial_state) # Ensure we start from the initial state
        if initial_state is not None:
            if len(initial_state) != self.params.N:
                raise ValueError(f"initial_state must have length N ({self.params.N}), got {len(initial_state)}")
            self.state = np.copy(initial_state)
        metrics = []
        self.T = self.params.T
        for _ in tqdm(range(n_warmup), desc="Warming up", disable=not progbar):
            self.step()
        for i in tqdm(range(n_experiment), desc="Running experiment", disable=not progbar):
            if dynamic_T is not None:
                self.T = dynamic_T[i]
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
        return TA_Result(metrics)

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
    alpha = 0.0
    local_energy = np.zeros(N_STATES, dtype=np.float64)
    neighbors_start = adj_indptr[node_idx]
    neighbors_end = adj_indptr[node_idx + 1]
    sum_weights = max(np.sum(adj_data[neighbors_start:neighbors_end]), 1e-10)  # Avoid division by zero
    for s in range(N_STATES):
        for i in range(neighbors_start, neighbors_end):
            weight = adj_data[i]
            local_energy[s] += weight * interaction_matrix[s, states[adj_indices[i]]]
    # degree normalization
    local_energy = local_energy*(1 / sum_weights)**alpha + h_weights
    if T == 0:
        min_energy = np.min(local_energy)
        # Find all states tied at the minimum and pick randomly among them
        tied = np.where(local_energy == min_energy)[0]
        if len(tied) == 1:
            return tied[0]
        return tied[min(np.searchsorted(np.linspace(0, 1, len(tied) + 1)[1:], rng_rolls[node_idx]), len(tied) - 1)]
    # normalize energies for numerical accuracy
    adjusted_energy = (local_energy/T) - np.min(local_energy/T)
    boltz = np.exp(-adjusted_energy)
    probabilities = boltz / np.sum(boltz)
    new_state = min(np.searchsorted(np.cumsum(probabilities), rng_rolls[node_idx]), N_STATES - 1)
    return new_state

@njit
def _compute_total_energy(nodes, states, adj_indptr, adj_indices, adj_data, interaction_matrix, h_weights):
    alpha = 0.0
    total_energy = 0.0
    for node_idx in nodes:
        s = states[node_idx]
        local_energy = 0.0
        neighbors_start = adj_indptr[node_idx]
        neighbors_end = adj_indptr[node_idx + 1]
        sum_weights = max(np.sum(adj_data[neighbors_start:neighbors_end]), 1e-10)  # Avoid division by zero
        for i in range(neighbors_start, neighbors_end):
            weight = adj_data[i]
            local_energy += 0.5 * (weight * interaction_matrix[s, states[adj_indices[i]]]) # 0.5 to avoid double on undirected graphs
        local_energy = local_energy*(1 / sum_weights)**alpha + h_weights[s]
        total_energy += local_energy
    return total_energy
