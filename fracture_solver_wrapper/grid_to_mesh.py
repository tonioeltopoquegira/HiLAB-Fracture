#!/usr/bin/env python3
"""
grid_to_mesh.py — convert a 2D topopt density grid into a triangular mesh (DOLFINx).

Every SOLID pixel/cell is split into 2 triangles, sharing deduplicated nodes.

Reuses the existing, debugged geometry pipeline (1) load (2) mirror (3) reconnect 
(4) keep only the largest connected component 
from fracture_analysis so the meshed region is the exact same load-bearing structure the fast solver uses.

"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fracture_pipeline.conversion import (
    load_design_file, mirror_half_beam, repair_connectivity, largest_connected_component,
)


# ─────────────────────────────────────────────────────────────────────────────
def load_grid(design_path, mirror=True, reconnect_mode="none", bridge_width=2,
              keep_largest=False):
    """
    Density file into binary material grid (Ny, Nx), 1=material (black pixel
    for PNGs). FEM j=0=bottom. Delegates to fracture_pipeline.conversion so
    it uses exactly the same loader as fracture_analysis.run_fracture.
    """
    rho = load_design_file(design_path)

    if mirror:
        rho = mirror_half_beam(rho)

    solid, bridge = repair_connectivity(rho, mode=reconnect_mode, bridge_width=bridge_width)

    if not keep_largest:
        return solid, bridge
    kept, _dropped = largest_connected_component(solid)
    return kept, bridge


def triangulate(solid, hx=None, mirror=True):
    """
    2 triangles per solid cell, shared nodes.
    """
    Ny, Nx = solid.shape
    if hx is None:
        hx = 1.0/(Nx//2 if mirror else Nx)

    gid = lambda j, i: j*(Nx+1) + i          
    js, is_ = np.where(solid)                 

    
    n00 = gid(js,   is_  ); n10 = gid(js,   is_+1)
    n01 = gid(js+1, is_  ); n11 = gid(js+1, is_+1)
    
    tris_global = np.vstack([np.column_stack([n00, n10, n11]),
                             np.column_stack([n00, n11, n01])])

   
    used = np.unique(tris_global)
    remap = -np.ones((Nx+1)*(Ny+1), np.int64)
    remap[used] = np.arange(len(used))
    triangles = remap[tris_global]

    uj, ui = np.divmod(used, Nx+1)
    points = np.column_stack([ui*hx, uj*hx]).astype(float)   # (M,2)
    return points, triangles, used, (Nx, Ny, hx)


def boundary_facets(points, triangles, used, grid, mirror=True):
    """
    Find boundary edges (in exactly one triangle) and tag the loading setup:
        1 = support 
        3 = load   
        4 = free 

    Returns list of (n0, n1, phys_tag).
    """
    Nx, Ny, hx = grid
    e = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]                       

    y = points[:, 1]
    ymin, ymax = y.min(), y.max()
    tol = 0.5*hx                               

    facets = []
    for n0, n1 in bnd:
        y0, y1 = y[n0], y[n1]
        if max(y0, y1) <= ymin + tol:   phys = 1     
        elif min(y0, y1) >= ymax - tol: phys = 3     
        else:                           phys = 4     
        facets.append((int(n0), int(n1), phys))
    return facets


def write_msh22(path, points, triangles, facets):
    """Pure-Python Gmsh 2.2 ASCII writer (no meshio/gmsh dependency)."""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(points)}\n")
        for i, (x, y) in enumerate(points, 1):
            f.write(f"{i} {x:.10g} {y:.10g} 0\n")
        f.write("$EndNodes\n")
        nel = len(facets) + len(triangles)
        f.write(f"$Elements\n{nel}\n")
        eid = 1
        # boundary lines first (elm-type 1 = 2-node line); tags: phys, geom
        for n0, n1, phys in facets:
            f.write(f"{eid} 1 2 {phys} {phys} {n0+1} {n1+1}\n"); eid += 1
        # surface triangles (elm-type 2 = 3-node triangle); phys group 10
        for a, b, c in triangles:
            f.write(f"{eid} 2 2 10 10 {a+1} {b+1} {c+1}\n"); eid += 1
        f.write("$EndElements\n")


def _draw_mesh(ax, points, triangles, facets):
    from matplotlib.collections import PolyCollection
    ax.set_facecolor("white")                       # empty space = white
    ax.add_collection(PolyCollection(points[triangles], facecolors="0.7",
                                     edgecolors="0.25", linewidths=0.15))
    for n0, n1, phys in facets:
        c = {1: "green", 3: "red"}.get(phys)
        if c:
            ax.plot([points[n0, 0], points[n1, 0]],
                    [points[n0, 1], points[n1, 1]], color=c, lw=3)
    ax.autoscale_view(); ax.set_aspect("equal")


def plot_mesh(path, points, triangles, facets):
    fig, ax = plt.subplots(figsize=(11, 4))
    _draw_mesh(ax, points, triangles, facets)
    ax.set_axis_off()                                 # no frame, no ticks, no title bg
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    pad = 0.06 * (ymax - ymin)
    xc = 0.5 * (xmin + xmax)
    ax.text(xc, ymax + pad, "Load", color="red", ha="center", va="bottom",
            fontsize=14, fontweight="bold")
    ax.text(xc, ymin - pad, "Support", color="green", ha="center", va="top",
            fontsize=14, fontweight="bold")
    ax.set_ylim(ymin - 2.2 * pad, ymax + 2.2 * pad)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    fig.savefig(str(path).rsplit(".", 1)[0] + ".eps", format="eps")
    plt.close(fig)


def plot_design_vs_mesh(path, design_path, points, triangles, facets, mirror):
    """Original design PNG (top) beside the resulting mesh (bottom)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7))
    dp = str(design_path)
    if dp.lower().endswith((".png", ".jpg", ".jpeg")):
        from PIL import Image
        raw = np.array(Image.open(dp).convert("L"))
        a1.imshow(raw, cmap="gray", origin="upper", vmin=0, vmax=255)
        a1.set_title(f"Original design PNG ({raw.shape[1]}×{raw.shape[0]})  "
                     f"— material = black", fontsize=10)
    else:
        arr = np.load(dp); arr = arr[0] if arr.ndim == 3 else arr
        a1.imshow(1-arr, cmap="gray", origin="upper", vmin=0, vmax=1)
        a1.set_title("Original design (.npy) — material = black", fontsize=10)
    a1.set_xlabel("col"); a1.set_ylabel("row")
    _draw_mesh(a2, points, triangles, facets)
    a2.set_title(f"Triangular mesh of the material{' (mirrored full beam)' if mirror else ''} "
                 f"— green=support, red=load", fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    fig.savefig(str(path).rsplit(".", 1)[0] + ".eps", format="eps")
    plt.close(fig)


def convert(design_path, out_dir="mesh_output", mirror=True,
            reconnect_mode="none", bridge_width=2, keep_largest=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    solid, _ = load_grid(design_path, mirror, reconnect_mode, bridge_width, keep_largest)
    points, triangles, used, grid = triangulate(solid, mirror=mirror)
    facets = boundary_facets(points, triangles, used, grid, mirror)

    write_msh22(out_dir/"mesh.msh", points, triangles, facets)
    np.savez(out_dir/"mesh.npz", points=points, triangles=triangles,
             facets=np.array([(a, b, t) for a, b, t in facets], np.int64))
    plot_mesh(out_dir/"mesh.png", points, triangles, facets)
    plot_design_vs_mesh(out_dir/"design_vs_mesh.png", design_path,
                        points, triangles, facets, mirror)

    nb = {t: sum(1 for *_, p in facets if p == t) for t in (1, 3, 4)}
    print(f"{design_path}")
    print(f"  material cells meshed : {int(solid.sum())}")
    print(f"  nodes / triangles     : {len(points)} / {len(triangles)}")
    print(f"  boundary facets       : {len(facets)}  "
          f"(support/bottom={nb[1]} load/top={nb[3]} free={nb[4]})")
    print(f"  → {out_dir/'mesh.msh'}  (Gmsh 2.2; DOLFINx: gmshio.read_from_msh)")
    print(f"  → {out_dir/'mesh.npz'}  → {out_dir/'mesh.png'}  → {out_dir/'design_vs_mesh.png'}")
    return points, triangles, facets


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("design", nargs="?",
        default="outputs/augmented/mbb_beam_192x64/train_images/"
                "mma_seed1340_binary_gauss_s0.1_bin.png")
    p.add_argument("--out-dir", default="mesh_output")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--reconnect", default="none",
                   choices=["closing", "bridge", "none"])
    a = p.parse_args()
    convert(a.design, a.out_dir, mirror=not a.no_mirror, reconnect_mode=a.reconnect)
