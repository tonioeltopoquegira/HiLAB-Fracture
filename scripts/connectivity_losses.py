"""Connectivity-based loss terms for structural designs.

This module provides two strategies to build a connectivity-dependent loss
on batches of images (e.g. VAE reconstructions):

1. A fully differentiable PyTorch surrogate based on iterative max-pooling
   reachability from the top boundary (or left/right, in principle).
2. A wrapped MatInverse Fourier PDE solver that computes an effective
   conductivity per design via JAX. This is *not* differentiable with
   respect to the input PyTorch tensors (gradients do not flow back), but
   is useful as a high-fidelity metric or constraint.

Both strategies expose a similar interface and accept batched images along
with tunable parameters for thresholds, smoothness, and solver resolution.

All tensors are assumed to be in [0, 1], with black (~0) representing solid
material and white (~1) representing void, consistent with the rest of the
codebase.
"""

from typing import Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_conductivity import evaluate_design_conductivity, _load_augmented_images


Reduction = Literal["none", "mean", "sum"]
Direction = Literal["vertical", "horizontal"]


def _to_grayscale_torch(images: torch.Tensor) -> torch.Tensor:
    """Convert input batch to grayscale in [0,1].

    Accepts tensors of shape:
      - (B, H, W)
      - (B, C, H, W) with C>=1
    Returns:
      - (B, 1, H, W) float32 in [0,1].
    """
    if images.ndim == 3:
        # (B, H, W) -> (B, 1, H, W)
        x = images.unsqueeze(1)
    elif images.ndim == 4:
        x = images
    else:
        raise ValueError(f"Expected images of shape (B,H,W) or (B,C,H,W), got {images.shape}")

    if x.size(1) == 1:
        gray = x
    else:
        gray = x.mean(dim=1, keepdim=True)

    return gray.clamp(0.0, 1.0).float()


def connectivity_loss_surrogate(
    images: torch.Tensor,
    solid_level: float = 0.5,
    softness: float = 40.0,
    target: float = 1.0,
    direction: Direction = "vertical",
    pool_kernel: int = 3,
    pool_iterations: int = 64,
    reduction: Reduction = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable connectivity loss via max-pool reachability.

    Args:
      images: Tensor of shape (B, H, W) or (B, C, H, W) with values in [0,1].
              Black (~0) is solid, white (~1) is void.
      solid_level: Intensity threshold where values below are considered solid.
      softness: Slope for the sigmoid that maps image intensities into a
                soft solid indicator in [0,1]. Larger -> sharper transition.
      target: Desired connectivity score (e.g. 1.0 means fully connected).
      direction: "vertical" for top-to-bottom connectivity, "horizontal" for
                 left-to-right (currently only vertical is implemented).
      pool_kernel: Kernel size for max-pooling when expanding the reachable set.
      pool_iterations: Number of propagation iterations. Should be >= max(H,W)
                       for strict convergence; smaller values approximate.
      reduction: One of {"none", "mean", "sum"} for the final loss.

    Returns:
      loss: Scalar loss (or per-sample loss if reduction="none").
      connectivity: Tensor of shape (B,) with connectivity scores in [0,1].

    Notes:
      - This function is fully differentiable with respect to `images`.
      - It operates on a soft solid indicator, so gradients highlight where
        adding or removing material changes connectivity.
    """
    gray = _to_grayscale_torch(images)  # (B,1,H,W)

    # Map intensities to soft solid indicator phi in [0,1]
    # Black (0) -> phi ~= 1, white (1) -> phi ~= 0.
    phi = torch.sigmoid(softness * (solid_level - gray))

    B, _, H, W = phi.shape

    if direction != "vertical":
        raise NotImplementedError("Only vertical (top-to-bottom) connectivity is implemented.")

    # Initialize reachability from the top boundary: only solid points on the
    # top row are initially reachable.
    reach = torch.zeros_like(phi)
    reach[:, :, 0:1, :] = phi[:, :, 0:1, :]

    # Iteratively expand the reachable set using max-pooling and masking by phi.
    pad = pool_kernel // 2
    for _ in range(pool_iterations):
        pooled = F.max_pool2d(reach, kernel_size=pool_kernel, stride=1, padding=pad)
        reach = torch.maximum(reach, pooled * phi)

    # Connectivity score: how much of the bottom boundary is reachable.
    bottom_reach = reach[:, :, -1, :]  # (B,1,W)
    # Require at least one connected path across width -> use max over width.
    connectivity = bottom_reach.view(B, -1).max(dim=-1).values  # (B,)

    # Define a hinge-like loss encouraging connectivity >= target.
    gap = F.relu(target - connectivity)  # (B,)

    if reduction == "none":
        loss = gap
    elif reduction == "mean":
        loss = gap.mean()
    elif reduction == "sum":
        loss = gap.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return loss, connectivity


def connectivity_loss_pde(
    images: torch.Tensor,
    solver_res: int = 64,
    kappa_solid: float = 1.0,
    kappa_void: float = 1e-3,
    binarize: bool = True,
    threshold: float = 0.5,
    target: Optional[float] = None,
    reduction: Reduction = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Connectivity loss based on MatInverse Fourier effective conductivity.

    Args:
      images: Tensor of shape (B, H, W) or (B, C, H, W) in [0,1].
      solver_res: Resolution of the homogenization cell passed to the PDE solver.
      kappa_solid: Conductivity of the solid phase.
      kappa_void: Conductivity of the void phase.
      binarize: Whether to threshold the solid fraction before solving.
      threshold: Threshold in [0,1] used when binarize=True.
      target: If provided, penalize designs whose effective conductivity is
              below this value with a hinge loss. If None, the loss is simply
              the negative mean effective conductivity (i.e. we maximize it).
      reduction: One of {"none", "mean", "sum"}.

    Returns:
      loss: Scalar loss (or per-sample loss if reduction="none"). This loss
            does *not* carry gradients back to `images`.
      kappa_eff: Tensor of shape (B,) with effective conductivities.

    Notes:
      - Computation is performed via JAX/MatInverse on detached NumPy arrays,
        so gradients will not flow to the PyTorch graph. Use this as a high-
        fidelity metric, constraint, or for analysis, not as a primary
        gradient source.
    """
    if images.ndim == 3:
        x = images.unsqueeze(1)
    elif images.ndim == 4:
        x = images
    else:
        raise ValueError(f"Expected images of shape (B,H,W) or (B,C,H,W), got {images.shape}")

    # Convert to NumPy on CPU for the JAX solver.
    x_np = x.detach().cpu().float().numpy()

    # Let evaluate_design_conductivity handle grayscale conversion and resizing.
    _, kappa_eff_np, _ = evaluate_design_conductivity(
        x_np,
        solver_res=solver_res,
        kappa_solid=kappa_solid,
        kappa_void=kappa_void,
        binarize=binarize,
        threshold=threshold,
    )

    # Back to a torch tensor (note: no grad).
    device = images.device
    kappa_eff = torch.from_numpy(kappa_eff_np.astype("float32")).to(device)

    if target is None:
        # Maximize effective conductivity: minimize negative mean.
        per_sample = -kappa_eff
    else:
        # Hinge-like penalty for designs with conductivity below target.
        per_sample = F.relu(target - kappa_eff)

    if reduction == "none":
        loss = per_sample
    elif reduction == "mean":
        loss = per_sample.mean()
    elif reduction == "sum":
        loss = per_sample.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return loss, kappa_eff


class _JAXPDEConnectivityFunction(torch.autograd.Function):
    """Torch autograd bridge around a JAX/MatInverse PDE connectivity loss.

    This expects images already resized to (solver_res, solver_res) and
    single-channel (B, 1, N, N) in [0,1]. Gradients are computed via
    JAX's autodiff and passed back to PyTorch.
    """

    @staticmethod
    def forward(
        ctx,
        images: torch.Tensor,
        solver_res_t: torch.Tensor,
        kappa_solid_t: torch.Tensor,
        kappa_void_t: torch.Tensor,
        binarize_t: torch.Tensor,
        threshold_t: torch.Tensor,
        target_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import numpy as np
        import jax
        import jax.numpy as jnp  # type: ignore
        from matinverse import Geometry2D, BoundaryConditions, Fourier  # type: ignore

        if images.dim() != 4 or images.size(1) != 1:
            raise ValueError(
                "JAX PDE connectivity expects images of shape (B,1,H,W); "
                f"got {tuple(images.shape)}"
            )

        solver_res = int(solver_res_t.item())
        kappa_solid = float(kappa_solid_t.item())
        kappa_void = float(kappa_void_t.item())
        binarize_flag = bool(int(binarize_t.item()))
        threshold = float(threshold_t.item())
        target_val = float(target_t.item())
        target: Optional[float] = None if target_val < 0.0 else target_val

        B, _, H, W = images.shape
        if H != solver_res or W != solver_res:
            raise ValueError(
                f"JAX PDE connectivity expects H=W=solver_res ({solver_res}), "
                f"got H={H}, W={W}"
            )

        # Move data to CPU / NumPy for JAX.
        gray_np = images.detach().cpu().float().numpy()[:, 0, :, :]  # (B, N, N)

        # Cache geometry and solver per resolution to avoid repeated setup.
        global _JAX_PDE_CACHE
        try:
            _ = _JAX_PDE_CACHE
        except NameError:  # pragma: no cover - first call
            _JAX_PDE_CACHE = {}

        if solver_res not in _JAX_PDE_CACHE:
            size = [1.0, 1.0]
            grid = [solver_res, solver_res]
            geo = Geometry2D(grid, size, periodic=[True, True])
            fourier = Fourier(geo)
            bcs = BoundaryConditions(geo)
            bcs.periodic("y", lambda batch, space, t: 1.0)
            bcs.periodic("x", lambda batch, space, t: 0.0)
            _JAX_PDE_CACHE[solver_res] = (geo, fourier, bcs)

        _, fourier, bcs = _JAX_PDE_CACHE[solver_res]

        kappa_bulk = jnp.eye(2, dtype=jnp.float64)

        def loss_and_aux(gray_in: jnp.ndarray):  # gray_in: (B, N, N)
            # Solid fraction: black (0) -> 1, white (1) -> 0.
            phi = 1.0 - gray_in
            if binarize_flag:
                phi = (phi >= threshold).astype(jnp.float64)
            cond = kappa_void + phi * (kappa_solid - kappa_void)  # (B, N, N)

            B_loc, N_loc, _ = cond.shape
            cond_flat = cond.reshape(B_loc, N_loc * N_loc)

            def kappa_map(batch, space, temp, t):
                k = cond_flat[batch, space]
                return kappa_bulk * k

            output = fourier(kappa_map, bcs, batch_size=B_loc)
            kappa_eff = jnp.asarray(output["kappa_effective"]).reshape(-1)  # (B,)

            if target is None:
                per_sample = -kappa_eff
            else:
                per_sample = jnp.maximum(0.0, target - kappa_eff)

            loss_val = per_sample.mean()
            return loss_val, kappa_eff

        # Value, gradient w.r.t. gray_np, and auxiliary kappa_eff.
        (loss_jax, kappa_eff_jax), grad_gray = jax.value_and_grad(
            loss_and_aux, has_aux=True
        )(jnp.asarray(gray_np, dtype=jnp.float64))

        # Convert outputs back to PyTorch tensors.
        device = images.device
        grad_images = torch.from_numpy(np.asarray(grad_gray, dtype=np.float32))
        grad_images = grad_images.unsqueeze(1).to(device)  # (B,1,N,N)

        kappa_eff = torch.from_numpy(
            np.asarray(kappa_eff_jax, dtype=np.float32)
        ).to(device)

        ctx.save_for_backward(grad_images)

        loss_tensor = torch.tensor(float(loss_jax), device=device, dtype=images.dtype)
        return loss_tensor, kappa_eff

    @staticmethod
    def backward(
        ctx,
        grad_loss: torch.Tensor,
        grad_kappa: torch.Tensor,  # noqa: ARG003 - unused
    ) -> Tuple[Optional[torch.Tensor], ...]:
        (grad_images,) = ctx.saved_tensors

        if grad_loss is None:
            grad_in = None
        else:
            grad_in = grad_loss.view(1, 1, 1, 1) * grad_images

        # No gradients for hyperparameter tensors.
        return (
            grad_in,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def connectivity_loss_pde_differentiable(
    images: torch.Tensor,
    solver_res: int = 64,
    kappa_solid: float = 1.0,
    kappa_void: float = 1e-3,
    binarize: bool = True,
    threshold: float = 0.5,
    target: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable PDE-based connectivity loss via JAX/MatInverse.

    This is similar in spirit to :func:`connectivity_loss_pde` but uses JAX's
    autodiff to propagate gradients back to `images`.

    Args:
      images: Tensor of shape (B, 1, N, N) in [0,1]. N must equal `solver_res`.
      solver_res: Resolution N of the PDE grid.
      kappa_solid, kappa_void, binarize, threshold, target: Same semantics as
          in :func:`connectivity_loss_pde`.

    Returns:
      loss: Scalar tensor with gradients w.r.t. `images`.
      kappa_eff: Tensor of shape (B,) with effective conductivities.
    """
    device = images.device

    solver_res_t = torch.tensor(float(solver_res), device=device)
    kappa_solid_t = torch.tensor(float(kappa_solid), device=device)
    kappa_void_t = torch.tensor(float(kappa_void), device=device)
    binarize_t = torch.tensor(1.0 if binarize else 0.0, device=device)
    threshold_t = torch.tensor(float(threshold), device=device)
    target_val = -1.0 if target is None else float(target)
    target_t = torch.tensor(target_val, device=device)

    return _JAXPDEConnectivityFunction.apply(
        images,
        solver_res_t,
        kappa_solid_t,
        kappa_void_t,
        binarize_t,
        threshold_t,
        target_t,
    )


class PDEConnectivityLoss(nn.Module):
    """`nn.Module` wrapper around the differentiable PDE connectivity loss.

    Example:

        loss_fn = PDEConnectivityLoss(solver_res=64, target=0.2)
        loss, kappa = loss_fn(images)
    """

    def __init__(
        self,
        solver_res: int = 64,
        kappa_solid: float = 1.0,
        kappa_void: float = 1e-3,
        binarize: bool = True,
        threshold: float = 0.5,
        target: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.solver_res = solver_res
        self.kappa_solid = kappa_solid
        self.kappa_void = kappa_void
        self.binarize = binarize
        self.threshold = threshold
        self.target = target

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return connectivity_loss_pde_differentiable(
            images,
            solver_res=self.solver_res,
            kappa_solid=self.kappa_solid,
            kappa_void=self.kappa_void,
            binarize=self.binarize,
            threshold=self.threshold,
            target=self.target,
        )


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    # Quick manual test on random designs.
    B, H, W = 4, 64, 64
    imgs = torch.rand(B, 1, H, W)

    print("Surrogate connectivity loss test:")
    loss_surr, conn = connectivity_loss_surrogate(imgs)
    print("  loss:", float(loss_surr), "connectivity:", conn.tolist())

    try:
        print("\nPDE-based connectivity loss test (small solver_res):")
        loss_pde, kappa = connectivity_loss_pde(imgs, solver_res=64)
        print("  loss:", float(loss_pde), "kappa_eff:", kappa.tolist())
    except Exception as e:
        print("  PDE solver test failed (likely missing MatInverse/JAX):", e)

    try:
        print("\nDifferentiable PDE-based connectivity loss test (binarize=False):")
        imgs_req = imgs.clone().requires_grad_(True)
        loss_pde_diff, kappa_diff = connectivity_loss_pde_differentiable(
            F.interpolate(imgs_req, size=(64, 64), mode="bilinear", align_corners=False),
            solver_res=64,
            binarize=False,
        )
        loss_pde_diff.backward()
        print(
            "  loss:", float(loss_pde_diff),
            "kappa_eff:", kappa_diff.tolist(),
            "|grad|:", float(imgs_req.grad.norm().item()),
        )
    except Exception as e:
        print("  Differentiable PDE solver test failed:", e)

    # ------------------------------------------------------------------
    # Dataset-based test on max design and its cut version
    # ------------------------------------------------------------------
    try:
        print("\nDataset-based test on max design and cut (grad fields):")
        DESIGN_DIRS = [
            "outputs/designs/mbb_beam_384x64_0.4-20260113-231654/images",
        ]
        paths, imgs_np = _load_augmented_images(DESIGN_DIRS, max_images=16)
        solver_res = 64

        # Use the non-differentiable evaluator to find the max-conductivity design
        _, kappa_all_np, _ = evaluate_design_conductivity(
            imgs_np,
            solver_res=solver_res,
            kappa_solid=1.0,
            kappa_void=1e-3,
            binarize=True,
            threshold=0.5,
        )
        idx_max = int(np.argmax(kappa_all_np))
        print("  Max design index:", idx_max, "path:", paths[idx_max])

        max_img = imgs_np[idx_max]  # (H, W) in [0,1]
        H_img, W_img = max_img.shape
        band_thickness = max(1, H_img // 16)
        band_start = H_img // 2 - band_thickness // 2
        band_end = band_start + band_thickness

        max_img_cut = max_img.copy()
        max_img_cut[band_start:band_end, :] = 1.0

        # Prepare tensors at solver resolution
        max_t = torch.from_numpy(max_img).view(1, 1, H_img, W_img).float()
        max_cut_t = torch.from_numpy(max_img_cut).view(1, 1, H_img, W_img).float()

        max_t = F.interpolate(max_t, size=(solver_res, solver_res), mode="bilinear", align_corners=False)
        max_cut_t = F.interpolate(max_cut_t, size=(solver_res, solver_res), mode="bilinear", align_corners=False)

        # Compare forward kappa between non-diff and diff versions
        loss_nd, kappa_nd = connectivity_loss_pde(
            max_t.detach(),
            solver_res=solver_res,
            kappa_solid=1.0,
            kappa_void=1e-3,
            binarize=False,
            target=None,
            reduction="mean",
        )
        loss_diff_max, kappa_diff_max = connectivity_loss_pde_differentiable(
            max_t.clone().requires_grad_(True),
            solver_res=solver_res,
            kappa_solid=1.0,
            kappa_void=1e-3,
            binarize=False,
            threshold=0.5,
            target=None,
        )
        print("  Max design kappa_eff non-diff:", kappa_nd.detach().cpu().numpy().tolist())
        print("  Max design kappa_eff diff:", kappa_diff_max.detach().cpu().numpy().tolist())

        # Temperature fields from the original evaluate_design_conductivity
        T_max_np, kappa_max_np, _ = evaluate_design_conductivity(
            max_img[None, ...],
            solver_res=solver_res,
            kappa_solid=1.0,
            kappa_void=1e-3,
            binarize=False,
            threshold=0.5,
        )
        T_cut_np, kappa_cut_np, _ = evaluate_design_conductivity(
            max_img_cut[None, ...],
            solver_res=solver_res,
            kappa_solid=1.0,
            kappa_void=1e-3,
            binarize=False,
            threshold=0.5,
        )

        # Gradient fields for max and cut designs
        import matplotlib.pyplot as plt  # local import

        # Compute gradients via differentiable PDE loss
        img_max_req = max_t.clone().requires_grad_(True)
        loss_max_d, kappa_max_d = connectivity_loss_pde_differentiable(
            img_max_req,
            solver_res=solver_res,
            kappa_solid=1.0,
            kappa_void=1e-3,
            binarize=False,
            threshold=0.5,
            target=None,
        )
        loss_max_d.backward()
        grad_max = img_max_req.grad.detach().cpu().numpy()[0, 0]

        img_cut_req = max_cut_t.clone().requires_grad_(True)
        loss_cut_d, kappa_cut_d = connectivity_loss_pde_differentiable(
            img_cut_req,
            solver_res=solver_res,
            kappa_solid=1.0,
            kappa_void=1e-3,
            binarize=False,
            threshold=0.5,
            target=None,
        )
        loss_cut_d.backward()
        grad_cut = img_cut_req.grad.detach().cpu().numpy()[0, 0]

        # Combined figure: geometries, temperatures, and gradients
        out_dir = "figures/connectivity_gradients"
        import os
        os.makedirs(out_dir, exist_ok=True)
        # Geometry images at solver resolution
        geom_max = max_t.detach().cpu().numpy()[0, 0]
        geom_cut = max_cut_t.detach().cpu().numpy()[0, 0]

        # Binary masks for outlining geometry boundaries
        geom_max_bin = (geom_max >= 0.5).astype(float)
        geom_cut_bin = (geom_cut >= 0.5).astype(float)

        fig, axes = plt.subplots(3, 2, figsize=(8, 9))

        # Row 0: geometries
        im_geom_max = axes[0, 0].imshow(geom_max, cmap="gray")
        axes[0, 0].contour(geom_max_bin, levels=[0.5], colors="k", linewidths=0.5)
        axes[0, 0].set_title("Geometry (max)")
        fig.colorbar(im_geom_max, ax=axes[0, 0])

        im_geom_cut = axes[0, 1].imshow(geom_cut, cmap="gray")
        axes[0, 1].contour(geom_cut_bin, levels=[0.5], colors="k", linewidths=0.5)
        axes[0, 1].set_title("Geometry (cut)")
        fig.colorbar(im_geom_cut, ax=axes[0, 1])

        # Row 1: temperatures with geometry outline
        im0 = axes[1, 0].imshow(T_max_np[0], cmap="viridis")
        axes[1, 0].contour(geom_max_bin, levels=[0.5], colors="k", linewidths=0.5)
        axes[1, 0].set_title(f"Max T, k_eff={float(kappa_max_np[0]):.4f}")
        fig.colorbar(im0, ax=axes[1, 0])

        im1 = axes[1, 1].imshow(T_cut_np[0], cmap="viridis")
        axes[1, 1].contour(geom_cut_bin, levels=[0.5], colors="k", linewidths=0.5)
        axes[1, 1].set_title(f"Cut T, k_eff={float(kappa_cut_np[0]):.4f}")
        fig.colorbar(im1, ax=axes[1, 1])

        # Row 2: gradients with geometry outline
        im2 = axes[2, 0].imshow(grad_max, cmap="bwr")
        axes[2, 0].contour(geom_max_bin, levels=[0.5], colors="k", linewidths=0.5)
        axes[2, 0].set_title("Grad (max)")
        fig.colorbar(im2, ax=axes[2, 0])

        im3 = axes[2, 1].imshow(grad_cut, cmap="bwr")
        axes[2, 1].contour(geom_cut_bin, levels=[0.5], colors="k", linewidths=0.5)
        axes[2, 1].set_title("Grad (cut)")
        fig.colorbar(im3, ax=axes[2, 1])

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "max_cut_geom_T_grad.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        print("  Saved combined geometry+T+grad plot to figures/connectivity_gradients/max_cut_geom_T_grad.png")

    except Exception as e:
        print("  Dataset-based gradient test failed:", e)

    # ------------------------------------------------------------------
    # Speed test: forward + backward for different batch sizes
    # ------------------------------------------------------------------
    try:
        import time

        print("\nSpeed test for differentiable PDE loss (forward+backward):")
        solver_res = 64
        for B_test in [8, 16]:
            x = torch.rand(B_test, 1, solver_res, solver_res, requires_grad=True)

            # Warm-up (to avoid including JAX compilation time)
            _ = connectivity_loss_pde_differentiable(
                x,
                solver_res=solver_res,
                kappa_solid=1.0,
                kappa_void=1e-3,
                binarize=False,
                threshold=0.5,
                target=None,
            )[0]

            x.grad = None
            t0 = time.time()
            loss_b, _ = connectivity_loss_pde_differentiable(
                x,
                solver_res=solver_res,
                kappa_solid=1.0,
                kappa_void=1e-3,
                binarize=False,
                threshold=0.5,
                target=None,
            )
            loss_b.backward()
            dt = time.time() - t0
            print(f"  B={B_test:2d}: {dt:.3f}s total, {dt / B_test:.4f}s per image")
    except Exception as e:
        print("  Speed test failed:", e)
