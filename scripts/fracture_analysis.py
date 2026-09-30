#!/usr/bin/env python3
"""
AT2 phase-field fracture analysis on a topology-optimized MBB design.

Robust pipeline (handles the fragmented binary augmented designs)
"""

import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse, time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fracture_pipeline.conversion import (
    build_connectivity, repair_connectivity, largest_component,
    load_design_file, mirror_half_beam, to_display_rgb, BRIDGE_COLOR, CRACK_CMAP,
)
from fracture_pipeline.plots import snapshot_strip

# ─── 2×2 Gauss quadrature for Q4 ──────────────────────────────────────────────
_G  = 1.0 / np.sqrt(3.0)
_GP = np.array([[-_G,-_G],[_G,-_G],[_G,_G],[-_G,_G]])


def _shape(xi, eta):
    N  = 0.25*np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])
    dN = 0.25*np.array([[-(1-eta),(1-eta),(1+eta),-(1+eta)],
                        [-(1-xi),-(1+xi),(1+xi),(1-xi)]])
    return N, dN


def base_matrices(hx, hy, nu):
    Ji = np.array([2./hx, 2./hy]); detJ = hx*hy/4.
    C0 = 1./(1-nu**2)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    K0=np.zeros((8,8)); M0=np.zeros((4,4)); Apf0=np.zeros((4,4)); f0=np.zeros(4)
    for xi,eta in _GP:
        N,dN_ref = _shape(xi,eta)
        dNx = dN_ref*Ji[:,None]
        B = np.zeros((3,8))
        B[0,0::2]=dNx[0]; B[1,1::2]=dNx[1]; B[2,0::2]=dNx[1]; B[2,1::2]=dNx[0]
        K0   += (B.T@C0@B)*detJ
        M0   += np.outer(N,N)*detJ
        Apf0 += (dNx.T@dNx)*detJ
        f0   += N*detJ
    return K0, M0, Apf0, f0


def run_fracture(
    design_path,
    E_solid=1.0, nu=0.3,
    Gc=5e-4, ell_factor=2.5,
    n_steps=100, plot_every=10,
    max_load=0.22,        # max prescribed downward displacement at load patch
    k_eps=1e-4,
    reconnect_mode="closing",  # 'closing' (stout, default) | 'bridge' | 'none'
    bridge_width=2,       # thin-bridge width when reconnect_mode='bridge'
    mirror=True,          # PNG half-beam → mirror to full beam
    center_frac=0.1,      # loaded top strip width, as a fraction of the full beam width (as in fracture_dolfinx.py)
    refine=1,             # split each pixel refine×refine after repair: h -> h/refine, eps unchanged
    out_dir="fracture_output",
    **_ignore,            # tolerate legacy kwargs (close_size, bc_mode, …)
):
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    for stale in out_dir.glob("fracture_step*.png"):   # avoid mixing old frames
        stale.unlink()

    rho = load_design_file(design_path)   # {0,1}, 1=solid, row0=bottom

    if mirror:
        rho = mirror_half_beam(rho)    # full beam; centre = load axis

    rho_orig = (rho >= 0.5)                      # true reflected geometry
    rho, bridge_b = repair_connectivity(rho, mode=reconnect_mode, bridge_width=bridge_width)

    if refine > 1:
        up = np.ones((refine, refine), np.uint8)
        rho_orig = np.kron(rho_orig.astype(np.uint8), up).astype(bool)
        bridge_b = np.kron(bridge_b.astype(np.uint8), up).astype(bool)
        rho = np.kron(rho, up)

    Ny, Nx = rho.shape
    hx = 1.0/(Nx//2 if mirror else Nx); hy = hx          # square elements
    ell = ell_factor*refine*hx; vol_e = hx*hy
    n_nodes = (Nx+1)*(Ny+1); n_dofs = 2*n_nodes; n_elems = Nx*Ny

    en, ed = build_connectivity(Nx, Ny)
    K0, M0, Apf0, f0 = base_matrices(hx, hy, nu)
    K0f = K0.flatten()

    active_nodes, active_elems = largest_component(rho, en, n_nodes)
    active_dofs = np.sort(np.concatenate([2*active_nodes, 2*active_nodes+1]))
    nj, ni = np.divmod(active_nodes, Nx+1)

    node_local = -np.ones(n_nodes, dtype=np.int64)
    node_local[active_nodes] = np.arange(len(active_nodes))
    en_a   = en[active_elems]                  # (nE,4) global node ids
    en_loc = node_local[en_a]                  # (nE,4) local node ids
    ed_a   = ed[active_elems]                  # (nE,8) global dof ids
    nE     = len(active_elems)

    # Boundary conditions
    jmin = nj.min()
    Lmask = (ni < Nx*0.30) & (nj <= jmin+2*refine)
    Rmask = (ni > Nx*0.70) & (nj <= jmin+2*refine)
    supL  = active_nodes[Lmask]               
    supR  = active_nodes[Rmask]              
    ctr   = Nx//2
    cmask = np.abs(ni-ctr) <= 0.5*center_frac*Nx
    jtop  = nj[cmask].max()
    load_nodes = active_nodes[cmask & (nj >= jtop-refine)]  
    load_dofs  = (2*load_nodes+1).astype(int)            # uy

    fixed = np.unique(np.concatenate([2*supL, 2*supL+1, 2*supR+1])).astype(int)
    bc_all = np.unique(np.concatenate([fixed, load_dofs])).astype(int)
    free   = np.setdiff1d(active_dofs, bc_all)
    load_in_bc = np.searchsorted(bc_all, load_dofs)

    n_bridge = int(bridge_b.sum()) // refine**2          # in design pixels
    print(f"Full beam {Nx}×{Ny} (refine={refine}, h/ell={hx/ell:.2f}) | "
          f"active {len(active_nodes)} nodes, {nE} elems | "
          f"bridges +{n_bridge}px (+{100*bridge_b.sum()/max(rho_orig.sum(),1):.1f}%) | "
          f"supports L={len(supL)} R={len(supR)} | load nodes={len(load_nodes)} "
          f"(j≈{jtop}) | ell={ell:.4f} Gc={Gc}")

    # ── Index helpers for vectorised assembly ─────────────────────────────────
    lp8 = np.array([(r,c) for r in range(8) for c in range(8)])
    ru, cu = ed_a[:,lp8[:,0]].ravel(), ed_a[:,lp8[:,1]].ravel()
    lp4 = np.array([(r,c) for r in range(4) for c in range(4)])
    rd, cd = en_loc[:,lp4[:,0]].ravel(), en_loc[:,lp4[:,1]].ravel()
    nA = len(active_nodes)
    M0f, Apf0f = M0.flatten(), Apf0.flatten()

    def assemble_K(g_e):
        vals = ((g_e*E_solid)[:,None]*K0f).ravel()
        return sp.csr_matrix((vals,(ru,cu)), shape=(n_dofs,n_dofs))

    def solve_u(g_e, delta):
        K = assemble_K(g_e)
        bvals = np.zeros(len(bc_all)); bvals[load_in_bc] = delta
        Kf  = K[free,:][:,free].tocsc()
        rhs = -K[free,:][:,bc_all].dot(bvals)
        u = np.zeros(n_dofs)
        u[bc_all] = bvals
        u[free]   = np.nan_to_num(spla.spsolve(Kf, rhs), nan=0., posinf=0., neginf=0.)
        F = float(K[load_dofs,:].dot(u).sum())      # total reaction at platen
        return u, F

    def update_H(u, H_old):
        ue  = u[ed_a]                       # (nE,8)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            psi = 0.5*E_solid*np.einsum("ij,ij->i", ue, ue@K0)/vol_e
        return np.maximum(H_old, np.nan_to_num(psi, nan=0., posinf=0.))

    def solve_d(H, d_prev):
        coef = (Gc/ell + 2*H)
        vals = (coef[:,None]*M0f + (Gc*ell)*Apf0f).ravel()
        Ad = sp.csr_matrix((vals,(rd,cd)), shape=(nA,nA)).tocsc()
        fd = np.zeros(nA); np.add.at(fd, en_loc, (2*H)[:,None]*f0)
        d_act = np.clip(spla.spsolve(Ad, fd), 0., 1.)
        return np.maximum(d_act, d_prev)

    # ── Plot helpers ──────────────────────────────────────────────────────────
    active_img = np.zeros((Ny, Nx), bool)
    active_img.ravel()[active_elems] = True         # analysed material
    ext = [0, Nx, 0, Ny]

    def material_rgb():
        """black = original solid, blue = inserted bridges, white = void."""
        orig = active_img & rho_orig
        brid = active_img & bridge_b
        return to_display_rgb(orig.astype(float), extra_masks=[(brid, BRIDGE_COLOR)])

    def bc_markers(ax):
        ax.plot(ni[Lmask], nj[Lmask], "g^", ms=4)
        ax.plot(ni[Rmask], nj[Rmask], "g^", ms=4)
        lj, li = np.divmod(load_nodes, Nx+1)
        ax.plot(li, lj, "rv", ms=3)            # loaded patch
        ax.annotate("", xy=(ctr, jtop-4*refine), xytext=(ctr, jtop+5*refine),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=2))

    mat_rgb = material_rgb()

    fig0, ax0 = plt.subplots(figsize=(11,4))
    ax0.imshow(mat_rgb, origin="lower", extent=ext, aspect="equal")
    ax0.set_title(f"Reflected design  (black=original solid, blue=+{n_bridge}px "
                  f"reconnection, white=void)  •  ▲ supports   ↓ load", fontsize=9)
    bc_markers(ax0); ax0.set_xlabel("x"); ax0.set_ylabel("y")
    plt.tight_layout(); fig0.savefig(out_dir/"00_design.png", dpi=130); plt.close(fig0)

    H_flat  = np.zeros(nE)
    d_active = np.zeros(nA)
    d_nodes  = np.zeros(n_nodes)
    disp_hist=[0.]; force_hist=[0.]; saved=[]

    def save_snap(step, t, delta, F, dmax):
        d_elem = np.zeros(n_elems)
        d_elem[active_elems] = np.mean(d_active[en_loc], axis=1)
        d_disp = d_elem.reshape(Ny, Nx)

        fig, ax = plt.subplots(figsize=(11,4))
        ax.imshow(mat_rgb, origin="lower", extent=ext, aspect="equal")
        dm = np.ma.masked_where(d_disp < 0.05, d_disp)
        im = ax.imshow(dm, cmap=CRACK_CMAP, origin="lower", vmin=0, vmax=1,
                       extent=ext, aspect="equal", alpha=0.9)
        plt.colorbar(im, ax=ax, fraction=0.025, label="crack d")
        bc_markers(ax)
        ax.set_title(f"step {step}/{n_steps}  t={t:.2f}  δ={delta:.4f}  "
                     f"F={F:.4f}  d_max={dmax:.3f}", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        fn = out_dir/f"fracture_step{step:04d}.png"
        fig.savefig(fn, dpi=130); plt.close(fig)
        saved.append(fn); print(f"    → {fn.name}")

    t0 = time.time()
    for step in range(1, n_steps+1):
        t = step/n_steps
        delta = -max_load * t                      # downward

        d_elem_mean = np.mean(d_nodes[en_a], axis=1)
        g_e = (1-d_elem_mean)**2 + k_eps
        u, F = solve_u(g_e, delta)
        H_flat   = update_H(u, H_flat)
        d_active = solve_d(H_flat, d_active)
        d_nodes[active_nodes] = d_active
        dmax = d_active.max()

        disp_hist.append(abs(delta)); force_hist.append(abs(F))
        print(f"  step {step:3d}/{n_steps}  δ={delta:+.4f}  F={F:+.4e}  "
              f"d_max={dmax:.4f}  ({time.time()-t0:.1f}s)", flush=True)

        broke = dmax > 0.98
        if step % plot_every == 0 or step == n_steps or broke:
            save_snap(step, t, delta, F, dmax)
        if broke:
            print("  ✓ fully fractured — stopping early."); break

    elapsed = time.time()-t0

    da = np.array(disp_hist); fa = np.array(force_hist)
    absorbed = float(np.trapz(fa, da))

    fig_fd, ax_fd = plt.subplots(figsize=(7,4))
    ax_fd.plot(da, fa, "k-o", ms=3, lw=1.5)
    ax_fd.fill_between(da, fa, alpha=0.15, color="steelblue")
    ax_fd.set_xlabel("|δ|  prescribed displacement")
    ax_fd.set_ylabel("|F|  reaction force")
    ax_fd.set_title(f"Force–displacement  •  absorbed energy W = {absorbed:.4e}")
    ax_fd.grid(alpha=0.3)
    plt.tight_layout(); fig_fd.savefig(out_dir/"force_displacement.png", dpi=130); plt.close(fig_fd)
    print(f"Absorbed energy = {absorbed:.4e}  |  time = {elapsed:.2f}s")

    snapshot_strip(out_dir, n_steps, f"absorbed energy W = {absorbed:.4e}",
                   out_dir/"overview.png", cols=min(len(saved), 5) or None)

    print(f"Results → {out_dir.resolve()}")
    return out_dir, absorbed, elapsed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("design", nargs="?",
        default="outputs/augmented/mbb_beam_192x64/train_images/"
                "mma_seed1340_binary_gauss_s0.1_bin.png")
    p.add_argument("--Gc",         type=float, default=5e-4)
    p.add_argument("--ell-factor", type=float, default=2.5)
    p.add_argument("--steps",      type=int,   default=100)
    p.add_argument("--plot-every", type=int,   default=10)
    p.add_argument("--max-load",     type=float, default=0.22)
    p.add_argument("--bridge-width", type=int,   default=2)
    p.add_argument("--nu",           type=float, default=0.3)
    p.add_argument("--no-mirror",    action="store_true")
    p.add_argument("--center-frac",  type=float, default=0.1)
    p.add_argument("--refine",       type=int,   default=1)
    p.add_argument("--out-dir",      default="fracture_output")
    a = p.parse_args()
    run_fracture(a.design, nu=a.nu, Gc=a.Gc, ell_factor=a.ell_factor,
                 n_steps=a.steps, plot_every=a.plot_every, max_load=a.max_load,
                 bridge_width=a.bridge_width, mirror=not a.no_mirror,
                 center_frac=a.center_frac, refine=a.refine, out_dir=a.out_dir)
