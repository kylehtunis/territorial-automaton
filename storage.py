import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix


class Store:
    """HDF5-backed storage for a single exploration (set of related simulation runs).

    One file per exploration. All runs share the same topology.
    Supports 1D and 2D parameter sweeps via per-run param_values dicts.
    """

    def __init__(self, f):
        self._f = f

    @classmethod
    def create(cls, path, description, adj_matrix, topology_info,
               base_params, sweep_params=None, n_warmup=0, pos=None):
        """Create a new exploration file.

        Parameters
        ----------
        path : str or Path
            File path for the .h5 file.
        description : str
            Human-readable description of this exploration.
        adj_matrix : scipy.sparse.csr_matrix
            The adjacency matrix shared by all runs.
        topology_info : dict
            Metadata about the topology, e.g. {"type": "grid", "rows": 50, "cols": 50}.
        base_params : dict
            Default parameter values, e.g. {"w": 0.5, "g": 0.5, "h": 0.0, "T": 1.0}.
        sweep_params : list of str, optional
            Names of parameters being swept, e.g. ["T"] or ["T", "h"].
        n_warmup : int
            Number of warmup steps used per run.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} already exists. Use ExplorationStore.open() to append.")

        f = h5py.File(path, "w")
        f.attrs["description"] = description
        f.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        f.attrs["N"] = adj_matrix.shape[0]
        f.attrs["topology_info"] = json.dumps(topology_info)
        f.attrs["base_params"] = json.dumps(base_params)
        f.attrs["sweep_params"] = json.dumps(sweep_params or [])
        f.attrs["n_warmup"] = n_warmup

        topo = f.create_group("topology")
        topo.create_dataset("indptr", data=adj_matrix.indptr)
        topo.create_dataset("indices", data=adj_matrix.indices)
        topo.create_dataset("data", data=adj_matrix.data)
        if pos is not None:
            topo.create_dataset("pos", data=np.asarray(pos, dtype=np.float64))

        f.create_group("runs")
        f.flush()
        return cls(f)

    @classmethod
    def open(cls, path):
        """Open an existing exploration file for reading and appending."""
        f = h5py.File(path, "a")
        return cls(f)

    # -- Write --

    def save_run(self, result, seed, param_values, dynamic_T=None, save_snapshots=False):
        """Save results from one simulation run.

        Parameters
        ----------
        result : TA_Result
            Output from TerritorialAutomaton.run().
        seed : int or None
            RNG seed used for this run.
        param_values : dict
            The parameter values for this run, e.g. {"T": 2.269}.
        dynamic_T : np.ndarray, optional
            Dynamic temperature schedule if used.
        save_snapshots : bool
            If True and result contains snapshots, save them (compressed).

        Returns
        -------
        str
            The run ID (e.g. "run_0000").
        """
        run_id = self._next_run_id()
        grp = self._f["runs"].create_group(run_id)

        grp.attrs["seed"] = seed if seed is not None else -1
        grp.attrs["param_values"] = json.dumps(param_values)
        n_experiment = len(result.orders)
        grp.attrs["n_experiment"] = n_experiment

        grp.create_dataset("energy", data=result.energies.astype(np.float64))
        grp.create_dataset("order", data=result.orders.astype(np.float32))
        grp.create_dataset("faction_sizes", data=result.faction_sizes.T.astype(np.int32))

        if dynamic_T is not None:
            grp.create_dataset("dynamic_T", data=np.asarray(dynamic_T, dtype=np.float64))

        if save_snapshots and result.metrics[0].snapshot is not None:
            snapshots = np.array([m.snapshot for m in result.metrics], dtype=np.int8)
            grp.create_dataset("snapshots", data=snapshots,
                               chunks=True, compression="gzip", compression_opts=4)

        self._f.flush()
        return run_id

    # -- Read --

    def load_run(self, run_id):
        """Load a single run's data.

        Returns
        -------
        dict with keys: energy, order, faction_sizes, param_values, seed, n_experiment,
        and optionally dynamic_T, snapshots.
        """
        if run_id not in self._f["runs"]:
            raise KeyError(f"Run '{run_id}' not found.")
        grp = self._f["runs"][run_id]

        result = {
            "run_id": run_id,
            "seed": int(grp.attrs["seed"]),
            "param_values": json.loads(grp.attrs["param_values"]),
            "n_experiment": int(grp.attrs["n_experiment"]),
            "energy": grp["energy"][:],
            "order": grp["order"][:],
            "faction_sizes": grp["faction_sizes"][:],
        }
        if result["seed"] == -1:
            result["seed"] = None
        if "dynamic_T" in grp:
            result["dynamic_T"] = grp["dynamic_T"][:]
        if "snapshots" in grp:
            result["snapshots"] = grp["snapshots"][:]
        return result

    def load_all_runs(self):
        """Load all runs as a dict keyed by run ID."""
        return {run_id: self.load_run(run_id) for run_id in self.run_ids}

    def load_topology(self):
        """Reconstruct the CSR adjacency matrix and positions from stored topology data.

        Returns (adj_matrix, pos) where pos is None if not stored.
        """
        topo = self._f["topology"]
        adj = csr_matrix((topo["data"][:], topo["indices"][:], topo["indptr"][:]))
        pos = topo["pos"][:] if "pos" in topo else None
        return adj, pos

    # -- Info --

    @property
    def run_ids(self):
        """Sorted list of run IDs in this file."""
        return sorted(self._f["runs"].keys())

    @property
    def n_runs(self):
        return len(self._f["runs"])

    @property
    def metadata(self):
        """File-level metadata as a dict."""
        return {
            "description": self._f.attrs["description"],
            "created_at": self._f.attrs["created_at"],
            "N": int(self._f.attrs["N"]),
            "topology_info": json.loads(self._f.attrs["topology_info"]),
            "base_params": json.loads(self._f.attrs["base_params"]),
            "sweep_params": json.loads(self._f.attrs["sweep_params"]),
            "n_warmup": int(self._f.attrs["n_warmup"]),
        }

    def summary(self):
        """Return a human-readable summary of this exploration."""
        meta = self.metadata
        lines = [
            f"Exploration: {meta['description']}",
            f"  Created: {meta['created_at']}",
            f"  Topology: {meta['topology_info']}  (N={meta['N']})",
            f"  Base params: {meta['base_params']}",
            f"  Sweep params: {meta['sweep_params']}",
            f"  Warmup steps: {meta['n_warmup']}",
            f"  Runs: {self.n_runs}",
        ]
        if self.n_runs > 0:
            param_vals = [json.loads(self._f["runs"][rid].attrs["param_values"])
                          for rid in self.run_ids]
            for p in meta["sweep_params"]:
                vals = sorted(set(pv[p] for pv in param_vals if p in pv))
                lines.append(f"    {p}: {len(vals)} unique values, range [{vals[0]}, {vals[-1]}]")
        return "\n".join(lines)

    # -- Lifecycle --

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        if self._f.id.valid:
            return f"ExplorationStore({self._f.filename!r}, {self.n_runs} runs)"
        return "ExplorationStore(closed)"

    # -- Internal --

    def _next_run_id(self):
        existing = self._f["runs"].keys()
        if not existing:
            return "run_0000"
        max_idx = max(int(k.split("_")[1]) for k in existing)
        return f"run_{max_idx + 1:04d}"
