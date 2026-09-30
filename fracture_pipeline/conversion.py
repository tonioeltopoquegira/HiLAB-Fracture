"""
Single source of truth for turning a design (a VAE latent, or a saved PNG/NPY
density grid) into the data structure the AT2 fracture solver
(`scripts/fracture_analysis.run_fracture`) and the DOLFINx mesher
(`fracture_solver_wrapper/grid_to_mesh.py`) actually consume.
"""

import itertools
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import label as nd_label, binary_dilation, binary_closing
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# latent -> raw decoder output 
def decode_latent(decoder, z):
    """latent (D,) -> (rgb HxWx3 in [0,1], gray HxW in [0,1]), raw decoder
    output with no rotation/resizing applied yet."""
    import torch
    zt = torch.from_numpy(np.asarray(z, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        recon = decoder.decode(zt)
    recon = recon[0].cpu().numpy()  # (3,H,W)
    rgb = np.clip(recon.transpose(1, 2, 0), 0, 1)
    gray = rgb.mean(axis=2)
    return rgb, gray


def rotate_and_resize(gray, nelx, nely):
    """decoder-space grayscale -> rot90 -> resized to the physics grid
    (nely, nelx), still continuous in [0,1]."""
    from PIL import Image
    rotated = np.rot90(gray, k=1)
    pil = Image.fromarray((np.clip(rotated, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((nelx, nely), resample=Image.BILINEAR)
    return np.asarray(pil).astype(np.float32) / 255.0


def binarize(image, threshold=0.5):
    """Pixel-intensity image in [0,1] (black = geometry) -> solid mask {0,1},
    1 = solid."""
    return (image < threshold).astype(np.float32)


def decode_to_mask(decoder, z, nelx, nely, threshold=0.5):
    """latent -> (rgb, gray, continuous, mask): every intermediate of
    decode_latent -> rotate_and_resize -> binarize, for plotting/saving."""
    rgb, gray = decode_latent(decoder, z)
    continuous = rotate_and_resize(gray, nelx, nely)
    return rgb, gray, continuous, binarize(continuous, threshold)


# loading an existing design file
def load_design_file(design_path):
    """Load a half- (or full-) beam design from PNG or NPY.

    Returns a solid mask {0,1} (1=solid) with row 0 = bottom (FEM
    convention: files are stored top-row-first, so we flip vertically).
    PNG: black pixels are geometry. NPY: a mask/density where 1 = solid.
    """
    design_path = str(design_path)
    is_png = design_path.lower().endswith((".png", ".jpg", ".jpeg"))
    if is_png:
        from PIL import Image
        img = np.array(Image.open(design_path).convert("L")).astype(float)
        rho = np.flipud(binarize(img / 255.0).astype(float))
    else:
        rho = np.load(design_path).astype(float)
        if rho.ndim == 3:
            rho = rho[0]
        rho = np.flipud((rho >= 0.5).astype(float))
    return rho


# half-beam -> full beam 
def mirror_half_beam(design):
    """Reflect a half-beam design (Ny, Nx_half) left-right into a full beam
    (Ny, 2*Nx_half); the centre column becomes the mirror/load axis."""
    return np.hstack([design[:, ::-1], design])


# connectivity repair 
def _bresenham(p0, p1):
    (y0, x0), (y1, x1) = p0, p1
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    y, x = y0, x0
    pts = []
    while True:
        pts.append((y, x))
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


def reconnect(solid, width=2):
    """Bridge disconnected solid pieces with a minimum-spanning-tree of thin
    connectors (Kruskal on nearest-pixel distances). Returns (new_solid,
    added) where `added` flags the inserted bridge pixels. Adds far less
    material than morphological closing (never fills concavities, only spans
    real gaps)."""
    Ny, Nx = solid.shape
    lab, n = nd_label(solid, structure=np.ones((3, 3), int))  # 8-connectivity
    if n <= 1:
        return solid.copy(), np.zeros_like(solid)
    coords = {c: np.argwhere(lab == c) for c in range(1, n + 1)}
    trees = {c: cKDTree(coords[c]) for c in range(1, n + 1)}
    edges = []
    for a, b in itertools.combinations(range(1, n + 1), 2):
        d, idx = trees[b].query(coords[a])
        k = int(np.argmin(d))
        edges.append((d[k], a, b, tuple(coords[a][k]), tuple(coords[b][idx[k]])))
    edges.sort(key=lambda e: e[0])
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    added = np.zeros_like(solid)
    for d, a, b, pa, pb in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            for (y, x) in _bresenham(pa, pb):
                if 0 <= y < Ny and 0 <= x < Nx:
                    added[y, x] = True
    if width > 1:
        added = binary_dilation(added, np.ones((width, width), bool)) & (~solid)
    return (solid | added), added


def reconnect_closing(solid, max_k=9):
    """Restore a load path by morphological closing with the SMALLEST square
    kernel that yields a single connected component. Closing re-thickens the
    members that augmentation thinned, so it gives a far stouter load path
    per added pixel than thin bridges. Returns (new_solid, added_pixels)."""
    for k in range(3, max_k + 1, 2):
        new = binary_closing(solid, np.ones((k, k), bool))
        _, n = nd_label(new, structure=np.ones((3, 3), int))
        if n == 1:
            return new, new & (~solid)
    new = binary_closing(solid, np.ones((max_k, max_k), bool))
    return new, new & (~solid)


def repair_connectivity(design, mode="closing", bridge_width=2):
    """design (any {0,1}-ish array) -> (repaired {0,1} float array, added mask).
    mode: 'closing' (default, stout) | 'bridge' (thin MST bridges) | 'none'."""
    solid = design >= 0.5
    if mode == "closing":
        new_solid, added = reconnect_closing(solid)
    elif mode == "bridge":
        new_solid, added = reconnect(solid, width=bridge_width)
    else:
        new_solid, added = solid.copy(), np.zeros_like(solid)
    return new_solid.astype(np.float32), added


# FEM connectivity + largest connected component 
def build_connectivity(Nx, Ny):
    """Q4 element -> global node ids (en) and dof ids (ed) for an Nx*Ny grid."""
    ne = Nx * Ny
    j_idx, i_idx = np.divmod(np.arange(ne), Nx)
    n0 = j_idx * (Nx + 1) + i_idx
    en = np.stack([n0, n0 + 1, n0 + (Nx + 2), n0 + (Nx + 1)], axis=1).astype(np.int64)
    ed = np.zeros((ne, 8), dtype=np.int64)
    for k in range(4):
        ed[:, 2 * k] = 2 * en[:, k]
        ed[:, 2 * k + 1] = 2 * en[:, k] + 1
    return en, ed


def largest_component(rho, en, n_nodes):
    """Return (active_node_ids, active_elem_ids) of the biggest solid blob,
    where `rho` is a (Ny,Nx) {0,1} array and `en` is build_connectivity's
    element->node table. This is what run_fracture actually meshes."""
    solid = rho.ravel() >= 0.5
    se = np.where(solid)[0]
    en_s = en[se]
    edges = np.vstack([en_s[:, [0, 1]], en_s[:, [1, 2]],
                        en_s[:, [2, 3]], en_s[:, [3, 0]]])
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(n_nodes, n_nodes))
    A = A + A.T
    _, lab = connected_components(A, directed=False)
    solid_nodes = np.unique(en_s.ravel())
    main = Counter(lab[solid_nodes]).most_common(1)[0][0]
    active_nodes = np.intersect1d(np.where(lab == main)[0], solid_nodes)
    active_elems = se[lab[en[se, 0]] == main]
    return active_nodes, active_elems


def largest_connected_component(design):
    """Convenience wrapper of `largest_component` that works directly on a
    (Ny,Nx) design array. Returns (kept, dropped) design arrays: `kept` is
    the single load-bearing blob the FEM solver will mesh, `dropped` is
    whatever solid material gets silently ignored (orphan islands) —
    inspect this to catch cases where a big chunk of the decoded design
    never participates in the physics."""
    Ny, Nx = design.shape
    en, _ = build_connectivity(Nx, Ny)
    n_nodes = (Nx + 1) * (Ny + 1)
    active_nodes, active_elems = largest_component(design, en, n_nodes)
    active_mask = np.zeros(Nx * Ny, dtype=bool)
    active_mask[active_elems] = True
    active_img = active_mask.reshape(Ny, Nx)
    solid = design >= 0.5
    kept = (solid & active_img).astype(np.float32)
    dropped = (solid & ~active_img).astype(np.float32)
    return kept, dropped


# display: geometry is always BLACK
SOLID_COLOR = (0.0, 0.0, 0.0)
VOID_COLOR = (1.0, 1.0, 1.0)
BRIDGE_COLOR = (0.0, 0.35, 1.0)  # material added by repair_connectivity
CRACK_CMAP = "autumn_r"          # crack field d: yellow (onset) -> red (fully broken)
DISPLAY_CMAP = "Greys"  


def to_display_rgb(design, extra_masks=None):
    """solid mask (Ny,Nx), 1=solid -> (Ny,Nx,3) float RGB in [0,1] with
    solid=black, void=white.

    `extra_masks`: optional list of (bool_mask, rgb_color) drawn on top, in
    order, for highlighting things like repaired/added pixels or dropped
    orphan islands (e.g. [(added, BRIDGE_COLOR)] for repaired bridges).
    Masks are drawn regardless of the base design's threshold, so pass a
    mask already intersected with `design >= 0.5` if it should only ever
    highlight solid pixels.
    """
    rgb = np.ones((*design.shape, 3))
    rgb[design >= 0.5] = SOLID_COLOR
    for mask, color in (extra_masks or []):
        rgb[mask] = color
    return rgb


def to_display_image(design, extra_masks=None):
    """Same as `to_display_rgb` but returns a uint8 PIL Image ready to save."""
    from PIL import Image
    rgb = to_display_rgb(design, extra_masks=extra_masks)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
