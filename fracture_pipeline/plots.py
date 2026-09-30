"""Figure helpers shared by fracture_analysis.py, fracture_batch.py and
inspect_pipeline.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tile_images(paths, titles, out_path, cols, suptitle=None, tile_size=(3.5, 3.2)):
    """Lay saved PNGs out on a grid (row-major) and save one figure."""
    n = len(paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(tile_size[0] * cols, tile_size[1] * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p, t in zip(axes.ravel(), paths, titles):
        ax.imshow(plt.imread(str(p)))
        ax.set_title(t, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def snapshot_strip(sim_dir, n_steps, title, out_path, cols=None):
    """Tile the fracture_step*.png snapshots run_fracture saved in `sim_dir`,
    each labelled with its load fraction t. None if there are no snapshots."""
    snaps = sorted(Path(sim_dir).glob("fracture_step*.png"))
    if not snaps:
        return None
    titles = [f"t={int(p.stem.replace('fracture_step', '')) / n_steps:.2f}" for p in snaps]
    return tile_images(snaps, titles, out_path, cols=cols or len(snaps), suptitle=title)
