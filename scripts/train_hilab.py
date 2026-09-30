import os
import math
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
try:
    from transformers import ViTModel
    _USE_HF = True
except Exception:
    _USE_HF = False
    try:
        import timm
    except Exception:
        timm = None
from connectivity_losses import connectivity_loss_pde_differentiable


#  Utilities

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#  Model
class ViTVAE(nn.Module):
    """
    ViT encoder (CLS token -> latent) + lightweight ConvTranspose decoder to (3, 128, 256).
    """
    def __init__(self, latent_dim=16, vit_name='google/vit-base-patch16-224-in21k', size_decoder: str = "small"):
        super().__init__()
        self.resize_for_vit = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        if _USE_HF:
            self.encoder = ViTModel.from_pretrained(vit_name)
            self._enc_type = "hf"
            self.encoder_hidden_size = self.encoder.config.hidden_size
        else:
            if timm is None:
                raise ImportError("Neither 'transformers' nor 'timm' available. Install one to use ViT encoder.")
            # use timm's ViT model as a fallback
            self.encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
            self._enc_type = "timm"
            if hasattr(self.encoder, "head") and hasattr(self.encoder.head, "in_features"):
                hidden = self.encoder.head.in_features
            elif hasattr(self.encoder, "embed_dim"):
                hidden = getattr(self.encoder, "embed_dim")
            else:
                hidden = 768
            self.encoder_hidden_size = hidden

        self.fc_mu = nn.Linear(self.encoder_hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_hidden_size, latent_dim)

        # ======= CHANGED: one power-of-two smaller in channels =======
        # original: decoder_input = latent -> 512 * 4 * 8 (seed 512, 4x8)
        # new: seed channels 512 -> 256
        if size_decoder == "small":
            self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 8)   # seed (256, 4, 8)
            self.decoder = nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(256, 4, 8)),                 # (256, 4, 8)
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 16)
                nn.BatchNorm2d(128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (64, 16, 32)
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (32, 32, 64)
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # (16, 64, 128)
                nn.BatchNorm2d(16), nn.ReLU(True),
                nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),     # (3, 128, 256)
                nn.Sigmoid()
            )

        if size_decoder == "medium":
            self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 8)   # seed (512, 4, 8)
            self.decoder = nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(512, 4, 8)),                 # (512, 4, 8)
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 16)
                nn.BatchNorm2d(256), nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 32)
                nn.BatchNorm2d(128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (64, 32, 64)
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (32, 64, 128)
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # (3, 128, 256)
                nn.Sigmoid()
            )

        if size_decoder == "large":
            self.decoder_input = nn.Linear(latent_dim, 1024 * 4 * 8)   # seed (1024, 4, 8)

            self.decoder = nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(1024, 4, 8)),                # (1024, 4, 8)

                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), # (512, 8, 16)
                nn.BatchNorm2d(512), nn.ReLU(True),

                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (256, 16, 32)
                nn.BatchNorm2d(256), nn.ReLU(True),

                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 32, 64)
                nn.BatchNorm2d(128), nn.ReLU(True),

                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (64, 64, 128)
                nn.BatchNorm2d(64), nn.ReLU(True),

                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # (3, 128, 256)
                nn.Sigmoid()
            )

        if size_decoder == "xlarge":
            self.decoder_input = nn.Linear(latent_dim, 2048 * 4 * 8)  # seed (2048, 4, 8)

            self.decoder = nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(2048, 4, 8)),  # (2048, 4, 8)

                nn.ConvTranspose2d(2048, 1024, 4, 2, 1),  # (1024, 8, 16)
                nn.BatchNorm2d(1024), nn.ReLU(True),

                nn.ConvTranspose2d(1024, 512, 4, 2, 1),   # (512, 16, 32)
                nn.BatchNorm2d(512), nn.ReLU(True),

                nn.ConvTranspose2d(512, 256, 4, 2, 1),    # (256, 32, 64)
                nn.BatchNorm2d(256), nn.ReLU(True),

                nn.ConvTranspose2d(256, 128, 4, 2, 1),    # (128, 64, 128)
                nn.BatchNorm2d(128), nn.ReLU(True),

                nn.ConvTranspose2d(128, 3, 4, 2, 1),      # (3, 128, 256)
                nn.Sigmoid()
            )

        if size_decoder == "xxlarge":
            # 1.25x xlarge: extra capacity without exploding parameter count
            self.decoder_input = nn.Linear(latent_dim, 2560 * 4 * 8)  # seed (2560, 4, 8)

            self.decoder = nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(2560, 4, 8)),  # (2560, 4, 8)

                nn.ConvTranspose2d(2560, 1280, 4, 2, 1),  # (1280, 8, 16)
                nn.BatchNorm2d(1280), nn.ReLU(True),

                nn.ConvTranspose2d(1280, 640, 4, 2, 1),   # (640, 16, 32)
                nn.BatchNorm2d(640), nn.ReLU(True),

                nn.ConvTranspose2d(640, 320, 4, 2, 1),    # (320, 32, 64)
                nn.BatchNorm2d(320), nn.ReLU(True),

                nn.ConvTranspose2d(320, 160, 4, 2, 1),    # (160, 64, 128)
                nn.BatchNorm2d(160), nn.ReLU(True),

                nn.ConvTranspose2d(160, 3, 4, 2, 1),      # (3, 128, 256)
                nn.Sigmoid()
            )


    # ---- freezing helpers ----
    def freeze_all_vit(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_k_vit_blocks(self, k: int, also_unfreeze_ln_head: bool = True, also_unfreeze_embeddings: bool = False):
        # Freeze everything first
        self.freeze_all_vit()
        if self._enc_type == "hf":
            # Optionally unfreeze embeddings
            if also_unfreeze_embeddings and hasattr(self.encoder, "embeddings"):
                for p in self.encoder.embeddings.parameters():
                    p.requires_grad = True
            # Unfreeze last-k blocks
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                blocks = self.encoder.encoder.layer
                if k > 0:
                    for blk in blocks[-k:]:
                        for p in blk.parameters():
                            p.requires_grad = True
            # Optionally unfreeze final LN / pooler
            if also_unfreeze_ln_head:
                if hasattr(self.encoder, "layernorm"):
                    for p in self.encoder.layernorm.parameters():
                        p.requires_grad = True
                if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
                    for p in self.encoder.pooler.parameters():
                        p.requires_grad = True
        else:
            # timm variant
            if hasattr(self.encoder, "blocks") and k > 0:
                for blk in self.encoder.blocks[-k:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            if also_unfreeze_ln_head:
                if hasattr(self.encoder, "norm"):
                    for p in self.encoder.norm.parameters():
                        p.requires_grad = True

    # ---- VAE core ----
    def encode(self, x):
        # x: (B, C, H, W)
        # Pad to square (preserve aspect ratio) before resizing to ViT input size.
        import torch.nn.functional as F
        B, C, H, W = x.shape
        s = max(H, W)
        pad_h = s - H
        pad_w = s - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # F.pad expects (left, right, top, bottom)
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        x224 = self.resize_for_vit(x_padded)  # (B,3,224,224)

        if self._enc_type == "hf":
            enc_out = self.encoder(pixel_values=x224)
            cls = enc_out.last_hidden_state[:, 0, :]
        else:
            feats = self.encoder.forward_features(x224)
            if feats.dim() == 3:
                cls = feats[:, 0, :]
            else:
                cls = feats
        mu = self.fc_mu(cls)
        logvar = self.fc_logvar(cls)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)  # (B,3,128,256)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def load_vitvae(path, latent_dim=None):
    """Load a ViTVAE checkpoint ({'state', 'meta'} or a bare state dict) in
    eval mode. Architecture comes from meta; only shape-matching keys load."""
    ck = torch.load(path, map_location="cpu")
    meta = ck.get("meta", {})
    state = ck.get("state", ck)
    latent_dim = latent_dim or int(meta.get("latent_dim", 16))
    model = ViTVAE(latent_dim=latent_dim, size_decoder=meta.get("size_decoder", "xlarge"))
    model_state = model.state_dict()
    model_state.update({k: v for k, v in state.items()
                        if k in model_state and model_state[k].shape == v.shape})
    model.load_state_dict(model_state)
    return model.eval()


class VAELoss(nn.Module):
    def __init__(self, recon_type="mse", kl_weight=1e-3, recon_weight=1.0, binarization_weight=0.0):
        super().__init__()
        self.recon_type = recon_type
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.binarization_weight = binarization_weight

    def forward(self, recon, x, mu, logvar):
        if self.recon_type == "mse":
            recon_loss = F.mse_loss(recon, x, reduction="mean")
        elif self.recon_type == "l1":
            recon_loss = F.l1_loss(recon, x, reduction="mean")
        elif self.recon_type == "bce":
            recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
        else:
            raise ValueError("Unknown recon_type")

        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        bin_reg = torch.mean(recon * (1 - recon)) if self.binarization_weight > 0 else 0.0
        loss = self.recon_weight * recon_loss + self.kl_weight * kld + self.binarization_weight * bin_reg
        return loss, recon_loss.detach(), kld.detach()



def make_loaders_from_arrays_flexible(
    train_images,          # np.ndarray or torch.Tensor
    test_images,           # np.ndarray or torch.Tensor
    batch_size: int = 16,
    num_workers: int = 4,
    normalize_from_uint8: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Accepts:
      - NumPy or Torch arrays
      - Either NHWC (N, H, W, C) with C=3 and H=256, W=128
        or   NCHW (N, 3, 128, 256)
    Returns NCHW float tensors in [0,1].
    """

    def to_tensor_nchw(x):
        # Convert to torch tensor (no copy if already tensor)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Expected np.ndarray or torch.Tensor")

        if x.dtype == torch.uint8 and not normalize_from_uint8:
            raise ValueError("Input is uint8; pass normalize_from_uint8=True or normalize beforehand.")

        if normalize_from_uint8 and x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor/array, got shape {tuple(x.shape)}")

        if x.shape[1] == 3 and x.shape[2] == 128 and x.shape[3] == 256:
            # Already NCHW
            pass
        elif x.shape[-1] == 3 and x.shape[1] == 256 and x.shape[2] == 128:
            
            x = x.permute(0, 3, 2, 1)
        else:
            raise ValueError(
                f"Unexpected shape {tuple(x.shape)}. "
                "Supported: (N,3,128,256) or (N,256,128,3)."
            )

        
        if x.min() < -1e-6 or x.max() > 1.0 + 1e-6:
            raise ValueError("Values out of [0,1]. Normalize your inputs or set normalize_from_uint8=True.")

        return x.contiguous()

    train_t = to_tensor_nchw(train_images)
    test_t  = to_tensor_nchw(test_images)

    # dummy labels so loop can unpack (imgs, _)
    train_ds = TensorDataset(train_t, torch.zeros(len(train_t)))
    test_ds  = TensorDataset(test_t,  torch.zeros(len(test_t)))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader

def make_optimizer(model):
    vit_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder"):
            vit_params.append(p)
        else:
            head_params.append(p)

    return optim.AdamW(
        [
            {"params": vit_params, "lr": 7.5e-5},
            {"params": head_params, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )



def run_one_epoch(model, loader, optimizer, criterion, device, train=True, connectivity_cfg=None):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_mse_sum = 0.0
    total_pixels = 0
    total_kld = 0.0
    n_imgs = 0

    pbar = tqdm(loader, leave=False)

    for batch in pbar:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if train:
                # stochastic
                recon, mu, logvar = model(imgs)
                loss, recon_loss, kld = criterion(recon, imgs, mu, logvar)

                # Optional PDE-based connectivity regularizer
                conn_weight = 0.0
                if connectivity_cfg is not None:
                    conn_weight = float(connectivity_cfg.get("weight", 0.0))
                    solver_res = int(connectivity_cfg.get("solver_res", 32))
                else:
                    solver_res = 32

                if conn_weight > 0.0:
                    # Convert reconstruction to grayscale and resize to square for PDE loss
                    gray = recon.mean(dim=1, keepdim=True)  # (B,1,H,W)
                    gray_sq = F.interpolate(
                        gray,
                        size=(solver_res, solver_res),
                        mode="bilinear",
                        align_corners=False,
                    )

                    conn_loss, _ = connectivity_loss_pde_differentiable(
                        gray_sq,
                        solver_res=solver_res,
                        kappa_solid=1.0,
                        kappa_void=1e-3,
                        binarize=False,
                        threshold=0.5,
                        target=None,
                    )
                    loss = loss + conn_weight * conn_loss
            else:
                # deterministic validation
                mu, logvar = model.encode(imgs)
                z = mu
                recon = model.decode(z)

                recon_loss = F.mse_loss(recon, imgs, reduction="mean")
                kld = torch.zeros_like(recon_loss)
                loss = recon_loss  # NO KL in validation

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        bsz = imgs.size(0)
        total_loss += loss.item() * bsz
        total_kld += kld.item() * bsz
        total_mse_sum += F.mse_loss(recon, imgs, reduction="sum").item()
        total_pixels += imgs.numel()
        n_imgs += bsz

        pbar.set_description(
            f"{'Train' if train else 'Eval'} "
            f"MSE {recon_loss.item():.5f}"
        )

    avg_loss = total_loss / n_imgs
    avg_mse = total_mse_sum / total_pixels  # TRUE global MSE
    avg_psnr = psnr_from_mse(avg_mse)

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "kld": total_kld / n_imgs
    }


def save_grid(imgs, path, nrow=4, rotate90: bool = False):
    """imgs: Tensor (B,3,H,W) in [0,1]"""
    b, c, h, w = imgs.shape
    ncol = nrow
    nrow = int(math.ceil(b / ncol))
    # infer one image size after conversion to original layout to set figure aspect
    if h == 128 and w == 256:
        sample_h, sample_w = w, h  # original layout is (256,128) => (H_orig=W, W_orig=H)
    else:
        sample_h, sample_w = h, w
    # each tile: base size 2 inches height; adjust width to keep aspect ratio
    tile_h_in = 2.0
    tile_w_in = tile_h_in * (sample_w / sample_h)
    fig_w = ncol * tile_w_in
    fig_h = nrow * tile_h_in
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(1.0, fig_w), max(1.0, fig_h)))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i < b:
            
            if h == 128 and w == 256:
                img = imgs[i].permute(2, 1, 0).cpu().numpy()
            else:
                img = imgs[i].permute(1, 2, 0).cpu().numpy()
            
            if rotate90:
                import numpy as _np
                img = _np.rot90(img, k=1)
            axes[i].imshow(img, aspect='equal')
    # Save without cropping the tiles
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def sweep_thaw_depths_with_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    thaw_depths: List[int],
    epochs_per_setting: int = 5,
    latent_dim: int = 16,
    recon_type: str = "mse",
    kl_weight: float = 1e-3,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_grid_examples: bool = True,
    inspect_variance: bool = True,
    size_decoder: str = "small",
    init_model_path: str = None,
    use_pde_connectivity: bool = False,
    max_conn_weight: float = 1
):
    set_seed(123)

    results = []
    all_curves: Dict[int, List[float]] = {}

    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    for k in thaw_depths:
        print("\n" + "=" * 80)
        print(f"Setting: Unfreeze last {k} ViT blocks")
        print("=" * 80)

        model = ViTVAE(latent_dim=latent_dim, size_decoder=size_decoder)

        # Optionally load initial model weights from a provided checkpoint
        if init_model_path:
            try:
                print(f"Loading initial weights from '{init_model_path}'...")
                state = torch.load(init_model_path, map_location=device)

                # Normalize to a plain state_dict
                if isinstance(state, dict):
                    if "state_dict" in state and isinstance(state["state_dict"], dict):
                        state_dict = state["state_dict"]
                    elif "state" in state and isinstance(state["state"], dict):
                        state_dict = state["state"]
                    else:
                        state_dict = state
                else:
                    state_dict = state

                # First: try to load the full model (encoder + decoder) strictly.
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print("Successfully loaded full model (encoder + decoder) weights.")
                except Exception as e_full:
                    print("[WARN] Full model load failed, trying encoder-only vit weights.")
                    print("       Full load error:", e_full)

                    # Fallback: load only the encoder (ViT) weights if present.
                    # Expect keys with prefix 'encoder.' in the checkpoint.
                    encoder_prefix = "encoder."
                    encoder_state = {
                        k[len(encoder_prefix):]: v
                        for k, v in state_dict.items()
                        if isinstance(k, str) and k.startswith(encoder_prefix)
                    }

                    if not encoder_state:
                        print("[WARN] No 'encoder.' keys found in checkpoint; encoder-only load skipped.")
                    else:
                        try:
                            model.encoder.load_state_dict(encoder_state, strict=False)
                            print("Successfully loaded encoder ViT weights only.")
                        except Exception as e_enc:
                            print("[ERROR] Failed to load encoder-only ViT weights:", e_enc)

            except Exception as e:
                print(f"[ERROR] Failed to load initial model weights from '{init_model_path}': {e}")

        model.unfreeze_last_k_vit_blocks(k=k, also_unfreeze_ln_head=True, also_unfreeze_embeddings=False)
        model = model.to(device)

        print(f"Trainable parameters: {count_trainable_params(model):,}")

        criterion = VAELoss(recon_type=recon_type, kl_weight=kl_weight)
        optimizer = make_optimizer(model)

        # KL annealing schedule: epoch -> kl_weight
        kl_schedule = {
            1:  5e-2,
            2: 1e-1,
            3: 1e-1,
            4: 1.25e-1
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

        def get_kl_weight(epoch):
            return max(v for k, v in kl_schedule.items() if k <= epoch)

        # PDE connectivity schedule: epoch -> (weight, solver_res)
        # - weight is 0 until halfway through the epochs, then increases linearly
        #   up to max_weight.
        # - resolution starts at 32 and is bumped to 64 for the last 3 epochs.

        def get_connectivity_schedule(epoch: int, total_epochs: int):
            frac = epoch / float(total_epochs)
            '''if frac <= 0.5:
                w = 0.0
            else:'''
            w = max_conn_weight * (frac - 0.5) / 0.5

            if epoch >= total_epochs - 2:
                res = 64
            else:
                res = 32
            return w, res

        val_mse_curve = []
        best_val = float("inf")

        for epoch in range(1, epochs_per_setting + 1):

            # KL annealing
            current_kl = get_kl_weight(epoch)
            criterion.kl_weight = current_kl
            # PDE connectivity schedule for this epoch (training only)
            if use_pde_connectivity:
                #conn_weight, conn_res = get_connectivity_schedule(epoch, epochs_per_setting)
                conn_weight = max_conn_weight
                conn_res = 64
                connectivity_cfg = {"weight": conn_weight, "solver_res": conn_res}
                #connectivity_cfg = {"weight": max_conn_weight, "solver_res": 32}
            else:
                conn_weight, conn_res = 0.0, 64
                connectivity_cfg = None

            print(
                f"\n[Thaw={k}] Epoch {epoch}/{epochs_per_setting}  kl_weight={current_kl:.4f} "
                f"| conn_weight={conn_weight:.4f} res={conn_res}"
            )

            train_metrics = run_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                train=True,
                connectivity_cfg=connectivity_cfg,
            )
            val_metrics = run_one_epoch(
                model,
                val_loader,
                optimizer,
                criterion,
                device,
                train=False,
                connectivity_cfg=None,
            )

            print(f"Train: loss={train_metrics['loss']:.4f} | MSE={train_metrics['mse']:.6f} | PSNR={train_metrics['psnr']:.2f} dB")
            print(
                f"Val  : MSE={val_metrics['mse']:.6f} | "
                f"PSNR={val_metrics['psnr']:.2f} dB"
            )

            val_mse_curve.append(val_metrics["mse"])

            if save_grid_examples:
                model.eval()
                with torch.no_grad():
                    imgs, _ = next(iter(val_loader))
                    imgs = imgs[:8].to(device)
                    mu, _ = model.encode(imgs)
                    recon = model.decode(mu)

                    # Here add the reconstruction new
                    os.makedirs("examples", exist_ok=True)
                    os.makedirs(f"examples/thaw{k}_latent{latent_dim}_dec{size_decoder}_{ts}", exist_ok=True)
                    save_grid(imgs, f"examples/thaw{k}_latent{latent_dim}_dec{size_decoder}_{ts}/input_epoch{epoch}.png")
                    save_grid(recon, f"examples/thaw{k}_latent{latent_dim}_dec{size_decoder}_{ts}/recon_epoch{epoch}.png")
                    reconstruction_check(model, None, f"examples/thaw{k}_latent{latent_dim}_dec{size_decoder}_{ts}/epoch{epoch}_fullrecon.png") # CONFIG['val_dirs'][0]
                    
            if inspect_variance:
                with torch.no_grad():
                    imgs, _ = next(iter(val_loader))
                    imgs = imgs[:32].to(device)
                    mu, logvar = model.encode(imgs)
                    print(f"Latent mu mean {mu.mean().item():.4f}, std {mu.std().item():.4f}; logvar mean {logvar.mean().item():.4f}")


            # Save model parameters for this thaw setting
            os.makedirs("models", exist_ok=True)
            full_path = os.path.join("models", f"vitvae_thaw{k}_latent{latent_dim}_decodersize{size_decoder}_{ts}_{epoch}.pt")
            #if val_metrics["mse"] < best_val:
            #    print(f'Best validation performance so far for thaw={k}, saving model to', full_path)
            #    best_val = val_metrics["mse"]
            torch.save(model.state_dict(), full_path)

        all_curves[k] = val_mse_curve
        agg = sum(val_mse_curve[-2:]) / 2.0 if len(val_mse_curve) >= 2 else val_mse_curve[-1]
        results.append({"thaw_blocks": k, "val_mse": agg})

    
    xs = [r["thaw_blocks"] for r in results]
    ys = [r["val_mse"] for r in results]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]; ys = [ys[i] for i in order]

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of unfrozen ViT encoder blocks (last-k)")
    plt.ylabel("Validation reconstruction MSE (lower is better)")
    plt.title("Effect of partial ViT fine-tuning on reconstruction quality (your dataset)")
    plt.grid(True); plt.tight_layout()
    plt.savefig("figures/standard_curve_vit_thaw_vs_mse.png", dpi=180)
    print("\nSaved curve to figures/standard_curve_vit_thaw_vs_mse.png")

    
    plt.figure(figsize=(7,5))
    for k, curve in sorted(all_curves.items(), key=lambda kv: kv[0]):
        plt.plot(range(1, len(curve)+1), curve, marker=".", label=f"thaw={k}")
    plt.xlabel("Epoch"); plt.ylabel("Val MSE")
    plt.title("Per-setting validation curves (your dataset)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("figures/val_curves_by_thaw.png", dpi=180)
    print("Saved per-setting curves to figures/val_curves_by_thaw.png")

    print("\nSummary (smaller MSE is better):")
    for r in sorted(results, key=lambda d: d["thaw_blocks"]):
        print(f"  thaw={r['thaw_blocks']:>2d} -> Val MSE ~ {r['val_mse']:.6f}")

    return results, all_curves

from reconstruction_check import load_images_as_nhwc, stretch, save_side_by_side_2x4

def reconstruction_check(model, load_file, output_file):

    # -------------------------
    # Load images
    # -------------------------

    # load_file
    paths, orig_nhwc = load_images_as_nhwc('outputs/augmented/mbb_beam_192x64/test_images', resize=(128, 256))
    print("Found images:", len(paths), "example shape (H,W,C):", orig_nhwc.shape[1:])

    # Convert to NCHW using the SAME permutation 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    orig_nchw = torch.from_numpy(orig_nhwc).permute(0,3,2,1).contiguous().float().to(DEVICE)
    print("Converted to NCHW for model:", orig_nchw.shape, "  (N,C,H,W)")

    # -------------------------
    # Run model deterministic path: encode -> decode with z = mu (no sampling)
    # -------------------------
    recons = []
    mus = []
    logvars = []
    with torch.no_grad():
        for i in range(0, len(orig_nchw), 8):
            batch = orig_nchw[i:i+8]
           
            mu, logvar = model.encode(batch)
            z = mu
            recon = model.decode(z)   # (B,3,128,256)
            recons.append(recon.cpu())
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())

    recon_nchw = torch.cat(recons, dim=0)
    mu_all = torch.cat(mus, dim=0)
    logvar_all = torch.cat(logvars, dim=0)
    print("recon_nchw.shape:", recon_nchw.shape)
    print("mu mean/std:", float(mu_all.mean()), float(mu_all.std()))
    print("logvar mean:", float(logvar_all.mean()))

    # Convert recon back to ORIGINAL disk NHWC layout:
    recon_nhwc = recon_nchw.permute(0,3,2,1).numpy()

    assert orig_nhwc.shape == recon_nhwc.shape, f"shape mismatch orig={orig_nhwc.shape} recon={recon_nhwc.shape}"

    mse = float(np.mean((orig_nhwc - recon_nhwc) ** 2))
    psnr = psnr_from_mse(mse)
    print(f"Overall MSE: {mse:.6e}  PSNR: {psnr:.2f} dB")

    # Save side-by-side comparison
    # Stretch width to make square for visualization
    orig_sq  = stretch(orig_nhwc, target_height=128, target_width=256)
    recon_sq = stretch(recon_nhwc, target_height=128, target_width=256)

    save_side_by_side_2x4(orig_sq, recon_sq, output_file)
    save_side_by_side_2x4(orig_sq, recon_sq, output_file.replace(".png", "_gray.png"), binarize=False)

    print("Saved comparison to:", output_file)



if __name__ == "__main__":
    """
    Replace the two lines below with your actual arrays, or import them.
    Expected shapes:
      train_images: (N, 256, 128, 3)  | test_images: (M, 256, 128, 3)
      values in [0,1] or set normalize_from_uint8=True
    """

    """
    PAST TRAIN - TEST SETTINGS:

    # list of directories containing PNG/JPEG images for training
        'train_dirs':['outputs/augmented/mbb_beam_384x64_0.4-20260111-164330',
                    'outputs/augmented/mbb_beam_384x64_0.4-20260111-201703',
                    'outputs/augmented/mbb_beam_384x64_0.4-20260111-221249',
                    'outputs/augmented/mbb_beam_384x64_0.4-20260113-231654',

                      #'outputs/designs/mbb_beam_384x64_0.4-20260111-164330',
                      #'outputs/designs/mbb_beam_384x64_0.4-20260111-201703',
                      #'outputs/designs/mbb_beam_384x64_0.4-20260111-221249',
                      #'outputs/designs/mbb_beam_384x64_0.4-20260113-231654',
                      ],
        # list of directories for validation
        'val_dirs': ['outputs/augmented/mbb_beam_384x64_0.4-20260113-223144/images',
                     #'outputs/designs/mbb_beam_384x64_0.4-20260113-223144/images',
                     #'outputs/designs/mbb_beam_384x64_0.4-20260111-201703',
                     'outputs/augmented/mbb_beam_384x64_0.4-20260113-205758/images',
                     #'outputs/designs/mbb_beam_384x64_0.4-20260113-205758/images',
                    ],
    
    """
    
    CONFIG = {
        # list of directories containing PNG/JPEG images for training
        'train_dirs':['outputs/augmented/mbb_beam_192x64/train_images',
                      ],
        # list of directories for validation
        'val_dirs': ['outputs/augmented/mbb_beam_192x64/test_images', 
                    ],
        # desired in-memory image size: (width, height) for PIL resize
        'resize': (128, 256),
        # whether to treat input files as uint8 and normalize automatically
        'normalize_from_uint8': True,
        'batch_size': 16,
        'num_workers': 4,
        # Optional: path to a checkpoint with initial model weights
        # Example: 'models/vitvae_thaw1_latent16_decodersizemedium_20260111-120000.pt'
        #'init_model_path': 'models/vitvae_thaw4_latent8_decodersizelarge_20260327-132911_3.pt',
    }


    from glob import glob
    from PIL import Image


    def load_images_from_dirs(dirs, resize=(128, 256), max_images=None):
        """Load images from multiple directories, return NHWC uint8 float32 in [0,1].

        Args:
          dirs: list of directory paths (strings or Path) to search recursively.
          resize: (width, height) tuple for PIL.Image.resize.
        Returns:
          NumPy array shaped (N, H, W, C) with dtype float32 and values in [0,1].
        """
        paths = []
        for d in dirs:
            d = str(d)
            paths += glob(os.path.join(d, '**', '*.png'), recursive=True)
            paths += glob(os.path.join(d, '**', '*.jpg'), recursive=True)
            paths += glob(os.path.join(d, '**', '*.jpeg'), recursive=True)

        paths = sorted(set(paths))
        if max_images is not None:
            paths = paths[:max_images]

        imgs = []
        for p in paths:
            try:
                im = Image.open(p).convert('RGB')
                im = im.resize(resize, resample=Image.BILINEAR)
                arr = np.asarray(im).astype(np.float32) / 255.0
                # expected output NHWC: (H, W, C) -> (256,128,3)
                imgs.append(arr)
            except Exception as e:
                print(f"Skipping {p}: {e}")

        if not imgs:
            raise RuntimeError(f'No images found in {dirs}')
        return np.stack(imgs, axis=0)


    # Load training and validation images from configured folders
    train_images = load_images_from_dirs(CONFIG['train_dirs'], resize=CONFIG['resize'])
    # If val_dirs doesn't exist or is empty, reuse a small split from train
    try:
        val_images = load_images_from_dirs(CONFIG['val_dirs'], resize=CONFIG['resize'])
    except RuntimeError:
        # fallback: take last 5% of train as validation
        n = len(train_images)
        split = max(1, int(n * 0.05))
        val_images = train_images[-split:]
        train_images = train_images[:-split]

    print("Train images shape:", train_images.shape)
    print("Val images shape  :", val_images.shape)

    train_loader, val_loader = make_loaders_from_arrays_flexible(
        train_images, val_images,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        normalize_from_uint8=CONFIG['normalize_from_uint8'],
    )
    
    
    init_model_path = CONFIG.get('init_model_path')
    for thaw, decoder_size, latent in zip([8, 10], ["large", "large"], [10, 8]):

        results, curves = sweep_thaw_depths_with_loaders(
            train_loader, val_loader,
            thaw_depths=[thaw],
            epochs_per_setting=4,
            latent_dim=latent, # try 8 as well
            recon_type='bce', # try bce as well
            kl_weight=0.0,
            size_decoder=decoder_size,
            init_model_path='',
            use_pde_connectivity=False,
            max_conn_weight=100.0
            
        )


    '''for thaw, decoder_size, latent in zip([10, 12], ["large", "large"], [8, 8]):

        results, curves = sweep_thaw_depths_with_loaders(
            train_loader, val_loader,
            thaw_depths=[thaw],
            epochs_per_setting=3,
            latent_dim=latent, # try 8 as well
            recon_type='bce', # try bce as well
            kl_weight=0.0,
            size_decoder=decoder_size,
            init_model_path='models/vitvae_thaw7_latent8_decodersizelarge_20260330-234125_4.pt',
            use_pde_connectivity=False,
            max_conn_weight=100.0
            
        )'''