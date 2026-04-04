import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from tqdm import tqdm
from territorial_automaton import N_STATES

# State colors: Strong A (blue), Weak A (lightblue), Factionless (gray), Weak B (salmon), Strong B (red)
STATE_COLORS = ['#1f77b4','#aec7e8','#444444', '#ff9896', '#d62728']
STATE_LABELS = ['Strong A', 'Weak A', 'Factionless', 'Weak B', 'Strong B']

def animate_simulation(metrics, adj_matrix, pos=None, interval=200, node_size=50, save_path=None):
    """
    Create an animation of network states over time from simulation metrics.

    Parameters:
    - metrics: List of TA_Metrics objects with .snapshot arrays (non-None).
    - adj_matrix: Scipy sparse CSR adjacency matrix.
    - pos: Optional node positions as an (N, 2) array or dict {node: (x, y)}.
            Falls back to spring layout if not provided.
    - interval: Milliseconds between frames.
    - node_size: Size of each node marker.
    - save_path: Optional filename to save the animation (e.g. 'sim.mp4', 'sim.gif').

    Returns:
    - matplotlib FuncAnimation object.
    """
    snapshots = [m.snapshot for m in metrics]
    if snapshots[0] is None:
        raise ValueError("Metrics do not contain state snapshots. Re-run with store_snapshots=True.")

    G = nx.from_scipy_sparse_array(adj_matrix)

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    if isinstance(pos, dict):
        pos_array = np.array([pos[i] for i in range(len(G))])
    else:
        pos_array = pos

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges once (they don't change)
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        edge_x.extend([pos_array[u, 0], pos_array[v, 0], None])
        edge_y.extend([pos_array[u, 1], pos_array[v, 1], None])
    ax.plot(edge_x, edge_y, color='#cccccc', linewidth=0.3, zorder=0)

    # Initial node scatter
    colors = [STATE_COLORS[s] for s in snapshots[0]]
    scatter = ax.scatter(pos_array[:, 0], pos_array[:, 1], c=colors, s=node_size, zorder=1, edgecolors='none')

    title = ax.set_title(f'Step 0 | Order: {metrics[0].order:.3f} | Energy: {metrics[0].energy:.1f}')
    ax.set_axis_off()

    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)
                      for c, l in zip(STATE_COLORS, STATE_LABELS)]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.8)

    def update(frame):
        colors = [STATE_COLORS[s] for s in snapshots[frame]]
        scatter.set_facecolors(colors)
        title.set_text(f'Step {frame} | Order: {metrics[frame].order:.3f} | Energy: {metrics[frame].energy:.1f}')
        return scatter, title

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=interval, blit=False)

    if save_path is not None:
        writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
        with tqdm(total=len(snapshots), desc='Saving animation') as pbar:
            anim.save(save_path, writer=writer, progress_callback=lambda *_: pbar.update(1))

    return anim
