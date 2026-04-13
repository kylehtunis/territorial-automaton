import territorial_automaton as ta

import multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm


class TA_SummaryResult:
    """Aggregated statistics across n_runs for a single model configuration."""
    def __init__(self, mean_abs_order, mean_factionless_fraction, order_variance):
        self.mean_abs_order = mean_abs_order
        self.mean_factionless_fraction = mean_factionless_fraction
        self.order_variance = order_variance  # run-to-run variance of time-averaged |m|


def run_models(models, n_warmup, n_experiment, n_runs, seed=42, store=None):
    """Run each model n_runs times in parallel. Returns one TA_SummaryResult per model.

    If store is provided, already-completed runs (matched by seed) are loaded
    instead of re-run. Fresh results are saved to the store from the main process.
    """
    n_models = len(models)
    seed_index = store.get_seed_index() if store else {}

    # Build tasks only for runs not already in the store
    tasks = []
    task_seeds = []
    for run_idx in range(n_runs):
        for model_idx, model in enumerate(models):
            task_seed = seed + run_idx * n_models + model_idx
            if task_seed not in seed_index:
                tasks.append((model, n_warmup, n_experiment, task_seed))
                task_seeds.append(task_seed)

    n_total = n_runs * n_models
    n_cached = n_total - len(tasks)

    # Run missing tasks, saving each result to the store as it arrives
    fresh_results = {}
    if tasks:
        if n_cached > 0:
            print(f"Resuming: {n_cached}/{n_total} runs loaded from store, running {len(tasks)} remaining")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for i, result in enumerate(tqdm(
                pool.imap(_run_with_random_ic_star, tasks),
                total=len(tasks), smoothing=0,
            )):
                task_seed = task_seeds[i]
                fresh_results[task_seed] = result
                if store is not None:
                    model = tasks[i][0]
                    store.save_run(result, seed=task_seed,
                                   param_values={'T': model.params.T, 'theta': model.params.theta, 'kappa': model.params.kappa})

    # Aggregate per model: use fresh results or load stored ones
    summaries = []
    for model_idx, model in enumerate(models):
        model_runs = []
        for run_idx in range(n_runs):
            task_seed = seed + run_idx * n_models + model_idx
            if task_seed in fresh_results:
                model_runs.append(fresh_results[task_seed])
            else:
                model_runs.append(store.load_result(seed_index[task_seed]))
        summaries.append(_aggregate_runs(model_runs, model.params.N))
    return summaries


def run_models_2d(models, n_warmup, n_experiment, n_runs, seed=42, store=None):
    """Run a 2D grid of models. Returns a 2D list of TA_SummaryResult."""
    flat_models = [model for row in models for model in row]
    flat_summaries = run_models(
        flat_models, n_warmup, n_experiment, n_runs, seed, store=store,
    )
    results_2d, idx = [], 0
    for row in models:
        results_2d.append(flat_summaries[idx:idx + len(row)])
        idx += len(row)
    return results_2d


def _aggregate_runs(results, N):
    per_run_mean_abs_order = np.array([np.mean(np.abs(r.orders)) for r in results])
    per_run_mean_factionless = np.array([
        np.mean(r.faction_sizes[ta.FACTIONLESS]) / N for r in results
    ])
    return TA_SummaryResult(
        mean_abs_order=np.mean(per_run_mean_abs_order),
        mean_factionless_fraction=np.mean(per_run_mean_factionless),
        order_variance=np.var(per_run_mean_abs_order, ddof=1),
    )


def _run_with_random_ic_star(args):
    return _run_with_random_ic(*args)


def _run_with_random_ic(model, n_warmup, n_experiment, seed=None):
    if seed is not None:
        model.rng = np.random.default_rng(seed)
    initial_state = model.rng.choice(ta.STATES, size=model.params.N)
    return model.run(n_warmup, n_experiment, initial_state=initial_state)
