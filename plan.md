# Karis General Model: Implementation & Analysis Plan

## Model Summary

A five-state cellular automaton on complex networks with heat bath dynamics derived from an explicit Hamiltonian. States: Strong A, Weak A, Factionless, Weak B, Strong B. Designed to study adversarial territorial competition and phase transitions across network topologies.

---

## Architecture

Three-layer structure separating concerns cleanly.

### Layer 1: Topology

A module of factory functions that each return a scipy sparse CSR adjacency matrix and an optional metadata dict (positions, labels — for visualization only, never used in dynamics).

Functions:
- `make_ba_network(n, m)` — Barabási-Albert
- `make_er_network(n, p)` — Erdős-Rényi
- `make_grid_network(rows, cols)` — regular lattice
- `make_sw_network(n, k, p)` — small-world (Watts-Strogatz)
- `load_sw_galaxy(path)` — Star Wars galaxy network

NetworkX is used for construction, then `nx.to_scipy_sparse_array(G)` converts to CSR format. NetworkX objects are never passed to the dynamics layer.

### Layer 2: Dynamics

A lean `KarisModel` class.

**Inputs:**
- Sparse CSR adjacency matrix
- Parameter dataclass: `(w, g, h, T)`
- Initial state array (`np.int8`, length `n_nodes`)
- Random seed

**At init:**
- Store adjacency matrix and parameters
- Initialize RNG via `np.random.default_rng(seed)`
- Precompute 5×5 interaction energy matrix δ from `(w, g)`

**State representation:**
- Single `np.int8` array, states encoded as integers 0–4
- Scalar values on the factional spectrum: S_A = -1, W_A = -w, F = 0, W_B = +w, S_B = +1
- No degree normalization: the topology's structural features (hubs, degree distribution) influence dynamics directly

**`run(n_equilib, n_measure)` method:**
- Calls `step()` for `n_equilib` steps, discarding statistics
- Calls `step()` for `n_measure` steps, collecting observables
- Returns collected observables

**`step()` method:**
- Generate shuffled node order for this sweep
- Loop over all nodes in shuffled order, calling Numba-jitted update function
- After full sweep: compute and store step statistics

**Numba-jitted update function (standalone, no class dependency):**
- Input: node index, current state array, adjacency data (indices/indptr from CSR), interaction matrix, parameters (h, T), scalar values array, RNG state
- Get neighbor indices from sparse matrix
- Read neighbors' current states (already updated if earlier in this sweep — this is what makes it asynchronous)
- Compute local energy E(s) for each of the five possible states at node i, using the interaction matrix and edge weights
- Add external field contribution: `-h × s` for each candidate state
- Compute heat bath probabilities: `p(s) = exp(-E(s) / T) / Z_i` where `Z_i = Σ_s exp(-E(s) / T)`
- Sample new state from this categorical distribution
- Update state array in-place

**Why heat bath over single-spin-flip Glauber:** With five states, single-spin-flip dynamics waste many proposals on rejected transitions, especially near criticality. Heat bath computes the full local Boltzmann distribution over all five states and samples directly — no accept/reject step. This automatically satisfies detailed balance for multi-state systems and is the standard approach for Potts-type models.

**Step statistics (computed after each sweep):**
- Count of nodes in each state
- Primary order parameter: `m = mean(s_i)` (magnetization analog — captures both faction balance and commitment level)
- Secondary order parameter: `f = (n_A - n_B) / N` where n_A counts all A-faction nodes (Strong + Weak) and n_B likewise (faction dominance irrespective of commitment)
- Optional: total energy, cluster sizes at specified intervals

### Layer 3: Experiments

**Parameter container:** dataclass with fields `(w, g, h, T)` and a method to produce modified copies for sweeps.

**Initial condition utilities:**
- `uniform_random(n, rng)` — each node assigned a uniformly random state
- `seeded(n, faction_a_indices, faction_b_indices)` — specified seeds, rest Factionless
- `specified_fractions(n, fractions, rng)` — random assignment with controlled faction ratios
- `all_factionless(n)` — blank slate. Note: this is an absorbing state at low T (zero interaction energy, zero field contribution). Useful for testing whether thermal fluctuations alone can nucleate factions, and for probing the temperature threshold at which field advantage overcomes structural inertia.

**Sweep function:** `sweep_parameter(topology, base_params, param_name, param_values, n_runs, n_steps, initial_condition_fn)`
- For each parameter value × run: create KarisModel with unique seed, run, collect results
- Parallelized across runs via `joblib.Parallel`
- Returns structured results (dict of arrays, or xarray dataset)

**Data storage:** HDF5 via `h5py` or `zarr` for large sweeps. Structured as `topology/param_name/param_value/run_id → observables`.

### Star Wars Overlay (separate from core)

A thin script that:
- Loads the galaxy topology from Layer 1
- Uses Layer 2 dynamics
- Adds canon events as scheduled interventions: at step t, call `model.state[node_indices] = new_state` and `model.params.h += delta`
- Canon events live entirely outside the dynamics layer

---

## The Hamiltonian

### Interaction Energy Matrix (δ)

5×5 symmetric matrix indexed by (S_A, W_A, F, W_B, S_B):

```
         S_A    W_A     F     W_B    S_B
S_A  [   -1     -w      g      w      1   ]
W_A  [   -w    -w²     gw      w²     w   ]
F    [    g     gw      0      gw     g    ]
W_B  [    w     w²     gw     -w²    -w   ]
S_B  [    1      w      g     -w     -1    ]
```

This matrix is fully determined by two symmetry constraints and two free parameters:
- **A↔B antisymmetry:** swapping all A and B labels negates off-diagonal signs, reflecting the equivalence of the two factions
- **Factionless neutrality:** the Factionless row/column is symmetric with respect to A and B
- **w** ∈ [0, 1]: controls weak state contribution. At w=0, weak states are energetically identical to Factionless. At w=1, they are identical to strong states. Interpolates smoothly between a three-state model (Strong A, Factionless, Strong B) and one where commitment level is irrelevant.
- **g** ∈ [0, 1]: controls aggression/mobilization — the energetic cost of faction-Factionless borders relative to faction-faction borders. At g=0, factions are indifferent to Factionless neighbors. At g=1, bordering Factionless territory is as costly as bordering enemy territory.

### Full Hamiltonian

```
H = ½ Σ_i Σ_{j ∈ neighbors(i)} w_ij × δ(s_i, s_j)  -  h × Σ_i s_i
```

Where:
- First term: pairwise interaction energy (summed over all neighbor pairs with factor ½ to avoid double-counting edges)
- Second term: external field (on-site, biases system toward one faction)
- `s_i`: scalar value of node i's state on the factional spectrum
- `w_ij`: edge weight between nodes i and j

No degree normalization is applied. The topology's structural features — hub dominance, degree heterogeneity, community structure — influence dynamics directly. This is an intentional design choice: the central question is how topology affects the phase transition, so modulating topological influence via a normalization parameter would obscure the answer. Degree normalization is a natural extension if needed.

### Heat Bath Dynamics

At each node update, the new state is sampled from the local equilibrium distribution:

```
E_i(s) = Σ_{j ∈ neighbors(i)} w_ij × δ(s, s_j)  -  h × s
p(s) = exp(-E_i(s) / T) / Z_i
Z_i = Σ_s exp(-E_i(s) / T)
```

Where the sum in Z_i runs over all five possible states. The new state for node i is drawn from this categorical distribution.

- T > 0: temperature/volatility parameter
- Low T: distribution concentrates on the lowest-energy state
- High T: distribution approaches uniform (random state assignment)

---

## Parameters

| Parameter | Symbol | Role | Range |
|-----------|--------|------|-------|
| Weak state weight | w | Scales weak state contribution relative to strong | [0, 1] |
| Aggression | g | Faction interaction with Factionless territory | [0, 1] |
| External field | h | Bias toward one faction | (-∞, +∞) |
| Temperature | T | Volatility / stochastic noise level | (0, +∞) |

---

## Implementation Priorities

### Performance
- State as `np.int8` array, no enums or dicts in hot path
- Adjacency as scipy sparse CSR, precomputed at init
- Interaction matrix precomputed at init
- All random numbers pre-generated per sweep as arrays
- Transition logic in standalone `@njit` function
- Parallelization across runs via joblib, never within a single run

### Reproducibility
- `np.random.default_rng(seed)` for all randomness
- Each run gets a unique seed derived from a master seed + run index
- Full parameter snapshots saved with results

### Libraries
- `numpy` — core numerics
- `scipy.sparse` — CSR adjacency matrices
- `numba` — JIT compilation of transition logic
- `joblib` — parallel execution across runs
- `networkx` — graph construction only (never in simulation loop)
- `h5py` or `zarr` — structured data storage for sweep results
- `xarray` — optional, for organizing multi-dimensional sweep results

---

## Experimental Plan

### Phase 0: Sanity Checks

Single BA topology, fixed reasonable parameters.

- Verify equilibration: does the order parameter settle?
- Verify symmetry: h=0 with symmetric initial conditions → ~50/50 outcomes?
- Verify field response: extreme h → one-faction dominance?
- Verify temperature response: very high T → disorder, very low T → frozen domains?
- Check that results are reproducible with the same seed
- Measure autocorrelation times for the order parameter at several (T, h) points to establish baseline sweep counts for later phases

### Phase 1: Single-Parameter Sweeps

Single BA topology. Sweep each parameter individually, others held fixed. ~20–50 runs per parameter value.

Priority order:
1. **h sweep** (most natural control parameter, most likely to reveal phase transition). Plot order parameter vs h — expect sigmoid. This is the direct comparison to the Star Wars hero diagram.
2. **T sweep** at h=0. Plot order parameter and its variance vs T — look for critical temperature.
3. **g sweep** — how does aggression/mobilization affect the transition?
4. **w sweep** — how does weak-state weight affect behavior?

Measure autocorrelation times near any identified transitions and increase n_equilib / n_measure accordingly. Near criticality, autocorrelation times diverge (critical slowing down), so sweep counts cannot be held constant across parameter values.

Deliverable: qualitative understanding of what each parameter does, rough locations of interesting behavior.

### Phase 2: Two-Parameter Phase Diagrams

Still single BA topology. Increase runs per point (~50–100).

Priority heatmaps:
1. **(T, h) plane** — the primary phase diagram. Order parameter as color. Identify phase boundary.
2. **(g, h) plane** — how aggression shifts the transition.
3. **(w, h) plane** — how weak-state weight affects critical behavior.

Deliverable: phase boundaries in the most important parameter planes, identification of the parameter regime where the model shows the richest behavior.

---

## Future Directions (post-Phase 2)

- **Topology variation:** Run h and T sweeps on BA, ER, grid, small-world, and Star Wars topologies in the interesting parameter regime from Phase 2. Compare whether each shows a transition, how sharp it is, and where the critical point falls.
- **Finite-size scaling:** For topologies with clear transitions, repeat sweeps at multiple network sizes (N = 500–10,000). Test whether transitions sharpen with N, measure variance scaling exponents, and extract finite-size corrections to the critical point. Pick one or two topologies for this — do not scale everything.
- **Scaling collapse and universality:** Full finite-size scaling collapse to determine universality class. Critical exponent measurements (β, γ, ν). Comparison against known universality classes (Ising, voter model, directed percolation).
- **Cluster analysis:** Cluster size distributions at criticality — test for power-law behavior.
- **Star Wars paper:** Canon events on galaxy topology, reproducing the Galactic Civil War timeline.
- **Extensions:** Multi-faction (3+ factions), per-faction parameters (asymmetric w, g), continuous state (scalar field on networks).