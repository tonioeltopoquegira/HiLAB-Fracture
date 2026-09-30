"""
Step-by-step visual check of the latent -> fracture-solver pipeline
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fracture_pipeline.conversion import (
    decode_to_mask, load_design_file, mirror_half_beam, repair_connectivity,
    largest_connected_component, to_display_rgb, DISPLAY_CMAP, BRIDGE_COLOR,
)
from fracture_pipeline.plots import snapshot_strip, tile_images
from train_hilab import load_vitvae
from fracture_analysis import run_fracture
from bo_fracture_opt import CONFIG as BO_CONFIG

FP = BO_CONFIG["fracture_params"]


def plot_step(arr, title, path, is_image=False):
    """Save one pipeline-stage array with geometry drawn black.
    is_image=True: pixel intensities (black = geometry), shown as-is.
    is_image=False: solid mask (1 = solid), shown via DISPLAY_CMAP (1 -> black)."""
    cmap = "gray" if is_image else DISPLAY_CMAP
    solid_frac = float((arr < 0.5).mean() if is_image else (arr >= 0.5).mean())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"shape={arr.shape}  solid_frac={solid_frac:.3f}", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_overlay(mask, highlight, color, title, path):
    """Solid mask drawn black with `highlight` pixels drawn in `color`."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.imshow(to_display_rgb(mask, extra_masks=[(highlight, color)]), origin="upper")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    # Defaults mirror bo_fracture_opt.CONFIG so this inspects exactly what the BO solves.
    p.add_argument("--decoder-checkpoint", default=BO_CONFIG["decoder_checkpoint"])
    p.add_argument("--latent-dim", type=int, default=BO_CONFIG["latent_dim"])
    p.add_argument("--latent-index", type=int, default=None,
                    help="If given, perturb initial_base with seeded noise for variety.")
    p.add_argument("--design-file", default=None,
                    help="Skip the decoder entirely and inspect an existing PNG/NPY design file instead.")
    p.add_argument("--nelx-half", type=int, default=BO_CONFIG["nelx_half"])
    p.add_argument("--nely", type=int, default=BO_CONFIG["nely"])
    p.add_argument("--threshold", type=float, default=BO_CONFIG["binarize_threshold"])
    p.add_argument("--reconnect-mode", default=FP["reconnect_mode"], choices=["closing", "bridge", "none"])
    p.add_argument("--steps", type=int, default=FP["n_steps"])
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument("--max-load", type=float, default=FP["max_load"])
    p.add_argument("--center-frac", type=float, default=FP["center_frac"],
                    help="loaded top strip width as a fraction of the full beam width")
    p.add_argument("--refine", type=int, default=FP["refine"],
                    help="split each pixel refine x refine for the solve (h/eps = 1/(ell_factor*refine))")
    p.add_argument("--out-dir", default="fracture_pipeline_check")
    args = p.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    if args.design_file is not None:
        # load_design_file returns FEM orientation (row 0 = bottom); flip back
        # to image orientation so every stage is plotted top-row-first.
        half_bin = np.flipud(load_design_file(args.design_file))
        saved.append(plot_step(half_bin, "0. loaded design file (already binary) — black=solid",
                                os.path.join(out_dir, "00_loaded_design.png")))
    else:
        decoder = load_vitvae(args.decoder_checkpoint, args.latent_dim)
        z = np.asarray(BO_CONFIG["initial_base"], dtype=np.float32)
        if args.latent_index is not None:
            rng = np.random.RandomState(args.latent_index)
            z = z + rng.normal(scale=0.5, size=z.shape).astype(np.float32)

        rgb, gray, continuous, half_bin = decode_to_mask(
            decoder, z, nelx=args.nelx_half, nely=args.nely, threshold=args.threshold)

        # raw decoder output (pixel image, black = geometry)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].imshow(rgb); axes[0].set_title("Raw decoder RGB", fontsize=9); axes[0].axis("off")
        axes[1].imshow(gray, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Raw decoder grayscale", fontsize=9); axes[1].axis("off")
        plt.tight_layout()
        p1 = os.path.join(out_dir, "01_raw_decoder_output.png")
        fig.savefig(p1, dpi=130); plt.close(fig)
        saved.append(p1)

        saved.append(plot_step(continuous, "2. Resize to half-beam grid",
                                os.path.join(out_dir, "02_rotated_resized_continuous.png"), is_image=True))
        saved.append(plot_step(half_bin, f"3. Solid mask pixel < {args.threshold}",
                                os.path.join(out_dir, "03_binarized_half_beam.png")))

    # mirror half -> full beam
    full = mirror_half_beam(half_bin)
    saved.append(plot_step(full, "Mirrored to full beam",
                            os.path.join(out_dir, "04_mirrored_full_beam.png")))

    # repair connectivity (gap closing / bridging)
    repaired, added = repair_connectivity(full, mode=args.reconnect_mode)
    n_added = int(added.sum())
    frac_added = n_added / max(1, int((full >= 0.5).sum()))
    saved.append(plot_overlay(
        full, added, BRIDGE_COLOR,
        f"5. Repair connectivity (mode={args.reconnect_mode}) "
        f"blue=+{n_added}px added (+{frac_added*100:.1f}%)",
        os.path.join(out_dir, "05_repaired_connectivity.png")))

    # keep only the largest connected component
    kept, dropped = largest_connected_component(repaired)
    n_dropped = int(dropped.sum())
    saved.append(plot_overlay(
        kept, dropped >= 0.5, (1, 0, 0),
        f"6. Largest connected component (meshed for FEM) "
        f"red=dropped orphan islands ({n_dropped}px)",
        os.path.join(out_dir, "06_largest_component.png")))

    if n_dropped > 0:
        print(f"WARNING: {n_dropped}px of solid material ({n_dropped/max(1,int((repaired>=0.5).sum()))*100:.1f}%) "
              f"is disconnected from the main load path and will be IGNORED by the FEM solve.")

    #run the actual solver on this exact design
    design_npy = os.path.join(out_dir, "final_design_fed_to_solver.npy")
    np.save(design_npy, half_bin.astype(np.float32))
    solve_out_dir = os.path.join(out_dir, "07_solve")
    _, energy, elapsed = run_fracture(
        design_path=design_npy, out_dir=solve_out_dir,
        **{**FP, "n_steps": args.steps, "plot_every": args.plot_every,
           "max_load": args.max_load, "reconnect_mode": args.reconnect_mode,
           "center_frac": args.center_frac, "refine": args.refine},
    )
    print(f"Absorbed energy = {energy:.6e}  (time {elapsed:.2f}s)")
    saved.append(os.path.join(solve_out_dir, "00_design.png"))

    # crack evolution + force–displacement
    strip = snapshot_strip(
        solve_out_dir, args.steps,
        f"Crack evolution | absorbed energy = {energy:.4e}",
        os.path.join(out_dir, "08_crack_evolution.png"), cols=3,
    )
    if strip:
        saved.append(strip)
    saved.append(os.path.join(solve_out_dir, "force_displacement.png"))

    saved = [s for s in saved if os.path.exists(s)]
    mosaic_path = tile_images(saved, [os.path.basename(s) for s in saved],
                              os.path.join(out_dir, "pipeline_overview.png"),
                              cols=1, tile_size=(9, 3.2))

    print(f"Saved {len(saved)} stage images + mosaic to {out_dir}/")
    print(f"Mosaic: {mosaic_path}")


if __name__ == "__main__":
    main()
