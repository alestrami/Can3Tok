import torch
import torch.nn.functional as F
import torch.nn as nn
from argparse import ArgumentParser, Namespace
import os
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
from PIL import Image
# from chamferdist import ChamferDistance
# from geomloss import SamplesLoss
from model.michelangelo import *
from model.michelangelo.models.tsal.tsal_base import ShapeAsLatentModule
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset import gs_dataset

import spconv.pytorch as spconv
import torch.utils.data as Data
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
import wandb

os.environ["WANDB_API_KEY"] = "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"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

# ===== HELPER: SAVE PLY =====
def save_gaussians_as_ply(gaussians_data, filepath):
    gaussians_data = np.asarray(gaussians_data)
    if gaussians_data.ndim == 1:
        gaussians_data = gaussians_data.reshape(-1, 14)
    N = gaussians_data.shape[0]
    vertex = np.zeros(N, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])
    vertex['x']      = gaussians_data[:, 0]
    vertex['y']      = gaussians_data[:, 1]
    vertex['z']      = gaussians_data[:, 2]
    vertex['f_dc_0'] = gaussians_data[:, 3]
    vertex['f_dc_1'] = gaussians_data[:, 4]
    vertex['f_dc_2'] = gaussians_data[:, 5]
    vertex['opacity']  = gaussians_data[:, 6]
    vertex['scale_0']  = gaussians_data[:, 7]
    vertex['scale_1']  = gaussians_data[:, 8]
    vertex['scale_2']  = gaussians_data[:, 9]
    vertex['rot_0']    = gaussians_data[:, 10]
    vertex['rot_1']    = gaussians_data[:, 11]
    vertex['rot_2']    = gaussians_data[:, 12]
    vertex['rot_3']    = gaussians_data[:, 13]
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(filepath)
    print(f"[INFO] Saved PLY to: {filepath}")


# ===== LOSS HELPER =====
# KL annealing: ramp weight from 0 → kl_weight_max over kl_anneal_epochs.
# This lets the reconstruction warm up before the KL regularisation kicks in,
# preventing posterior collapse caused by an under-weighted KL term.
kl_weight_max    = 1e-5   # target KL weight after annealing
kl_anneal_epochs = 2000   # number of epochs to ramp from 0 to kl_weight_max

def kl_weight(epoch):
    return kl_weight_max * min(1.0, epoch / kl_anneal_epochs)

def compute_loss(UV_gs_recover, UV_gs_batch, mu, log_var, epoch):
    KL_loss = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    batch_size = UV_gs_batch.shape[0]
    w_kl = kl_weight(epoch)

    if loss_usage == "L1":
        rec_loss = torch.norm(
            UV_gs_recover.reshape(batch_size, -1, 14) - UV_gs_batch[:, :, 4:], p=2
        ) / batch_size
        loss = rec_loss + w_kl * KL_loss

    elif loss_usage == "chamfer":
        rec_loss = (
            torch.mean(chamferDist(
                UV_gs_batch.reshape([batch_size, -1, 14])[:, :, :3],
                UV_gs_recover.reshape([batch_size, -1, 14])[:, :, :3]
            )) +
            0.01 * torch.mean(chamferDist(
                UV_gs_batch.reshape([batch_size, -1, 14])[:, :, 3:],
                UV_gs_recover.reshape([batch_size, -1, 14])[:, :, 3:]
            ))
        )
        loss = rec_loss + w_kl * KL_loss

    elif loss_usage == "sinkhorn":
        rec_loss = sinkhorn_eff(
            UV_gs_batch.contiguous().reshape([batch_size, -1, 14]),
            UV_gs_recover.contiguous().reshape([batch_size, -1, 14])
        ).mean()
        loss = rec_loss + w_kl * KL_loss

    else:
        raise ValueError(f"Unknown loss_usage: {loss_usage}")

    return loss, KL_loss, w_kl


# ===== DEVICE =====
# CUDA_VISIBLE_DEVICES is set at the top of the file (line ~25).
# cuda:0 is always the first visible GPU regardless of physical ID.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device} | GPUs visible: {torch.cuda.device_count()}")

# ===== REPRODUCIBILITY =====
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===== CONFIG =====
# Set MAX_SCENES = 1 to test the full pipeline on a single scene.
# Set MAX_SCENES = 1000 for full multi-scene training.
MAX_SCENES = 1000      

loss_usage   = "L1"     # "L1", "chamfer", "sinkhorn"
random_permute  = 0
random_rotation = 1
grad_clip    = 500.0     # clips only extreme outliers; typical grad_norm here is ~100-300

resol      = 200
data_path  = "/data04/alestrami/Points2/data/000-006"
save_path  = "./results_multiscene/"
os.makedirs(save_path, exist_ok=True)

# Set to a checkpoint path to resume, or None to start from scratch.
# For old-format checkpoints (plain state dict), also set resume_epoch_override
# to the epoch the checkpoint was saved at so the loop continues from there.
resume_checkpoint    = "./results_multiscene/model_epoch1000.pth"
resume_epoch_override = 1001   # only used for old-format checkpoints

num_epochs   = 4000 #2000 #200000
bch_size     = 1 if MAX_SCENES == 1 else 128   # scale up for full training
eval_every   = 1000    # run eval loop every N epochs
save_every   = 1000    # save periodic checkpoint every N epochs
eval_split   = 0.2     # fraction of scenes held out for eval (ignored when MAX_SCENES==1)

chamferDist  = ()      # placeholder — replace with ChamferDistance() if available
sinkhorn_eff = ()      # placeholder — replace with SamplesLoss(...) if available

# ===== WANDB =====
wandb.init(
    project="3DGSAE-training",
    name=f"can3tok-{MAX_SCENES}scenes-{loss_usage}",
    config={
        "loss_type":        loss_usage,
        "num_epochs":       num_epochs,
        "batch_size":       bch_size,
        "learning_rate":    1e-4,
        "resolution":       resol,
        "random_permute":   random_permute,
        "random_rotation":  random_rotation,
        "max_scenes":       MAX_SCENES,
        "eval_split":       eval_split if MAX_SCENES > 1 else "N/A (single scene)",
    },
    tags=["3DGSAE", "Can3Tok", "multi-scene"],
    mode="online",
)
print("[INFO] W&B initialized")

# ===== MODEL =====
print("[INFO] Loading Can3Tok model...")
config_path_perciever = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perciever = get_config_from_file(config_path_perciever)
if hasattr(model_config_perciever, "model"):
    model_config_perciever = model_config_perciever.model
perceiver_encoder_decoder = instantiate_from_config(model_config_perciever)

if torch.cuda.device_count() > 1:
    gs_autoencoder = nn.DataParallel(perceiver_encoder_decoder)
else:
    gs_autoencoder = perceiver_encoder_decoder
gs_autoencoder.to(device)
print(f"[INFO] Model loaded on {device}")

optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-4, betas=[0.9, 0.999])

start_epoch     = 0
best_eval_loss  = float("inf")
if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
    ckpt = torch.load(resume_checkpoint, map_location=device)
    # Support both new format {"epoch", "model", "optimizer"} and old format (plain state dict)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_state    = ckpt["model"]
        start_epoch    = ckpt["epoch"] + 1
        best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"[INFO] Resumed from {resume_checkpoint} — epoch {ckpt['epoch']}, best eval loss {best_eval_loss:.6f}")
        else:
            print(f"[INFO] Resumed model weights from {resume_checkpoint} — epoch {ckpt['epoch']} (no optimizer state)")
    else:
        # Old format: plain OrderedDict of model weights, no epoch/optimizer info
        model_state = ckpt
        # resume_epoch_override lets you tell the script which epoch this checkpoint is from
        start_epoch    = resume_epoch_override
        best_eval_loss = float("inf")
        print(f"[INFO] Resumed model weights from {resume_checkpoint} (old format) — starting at epoch {start_epoch}, optimizer reset")
    if isinstance(gs_autoencoder, nn.DataParallel):
        gs_autoencoder.module.load_state_dict(model_state)
    else:
        gs_autoencoder.load_state_dict(model_state)
else:
    print("[INFO] Starting from scratch")

# ===== DATASET & SPLIT =====
print(f"[INFO] Loading dataset from: {data_path} (max {MAX_SCENES} scenes)")
full_dataset = gs_dataset(
    root=data_path,
    resol=resol,
    random_permute=False,
    train=True,
    ply_subpath="ckpts/point_cloud_15000.ply",
)

scene_indices = list(range(min(MAX_SCENES, len(full_dataset))))
full_subset   = Data.Subset(full_dataset, scene_indices)

if MAX_SCENES == 1:
    # Single-scene pipeline test: train == eval, no split
    train_dataset = full_subset
    eval_dataset  = full_subset
    print("[INFO] Single-scene mode: train == eval (pipeline sanity check)")
else:
    n_total = len(full_subset)
    n_eval  = max(1, int(n_total * eval_split))
    n_train = n_total - n_eval
    train_dataset, eval_dataset = Data.random_split(
        full_subset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[INFO] Split: {n_train} train / {n_eval} eval scenes")

trainDataLoader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=bch_size,
    shuffle=True,
    num_workers=0,
)
evalDataLoader = Data.DataLoader(
    dataset=eval_dataset,
    batch_size=bch_size,
    shuffle=False,
    num_workers=0,
)
print(f"[INFO] Train batches/epoch: {len(trainDataLoader)} | Eval batches: {len(evalDataLoader)}")

# ===== VOXEL PE SETUP (must match gs_dataset.py: volume_dims=20, resolution=0.8) =====
volume_dims   = 20
resolution_v  = 16.0 / volume_dims
origin_offset = torch.tensor(
    np.array([(volume_dims - 1) / 2] * 3) * resolution_v, dtype=torch.float32
).to(device)

# ===== TRAINING LOOP =====
print("[INFO] Starting training...")

for epoch in tqdm(range(start_epoch, num_epochs)):

    gs_autoencoder.train()
    train_losses, train_kl_losses, train_grad_norms = [], [], []

    for i_batch, batch_data in enumerate(trainDataLoader):
        UV_gs_batch = batch_data[0].float().to(device)

        # Optional: random permutation of Gaussian order
        if random_permute == 1:
            UV_gs_batch = UV_gs_batch[:, torch.randperm(UV_gs_batch.size(1))]

        # Optional: random rotation of XYZ positions + re-compute voxel PE
        if epoch % 5 == 0 and epoch > 1 and random_rotation == 1:
            rand_rot_comp = special_ortho_group.rvs(3)
            rand_rot = torch.tensor(
                np.dot(rand_rot_comp, rand_rot_comp.T), dtype=torch.float32
            ).to(device)
            # Rotate XYZ positions (cols 4:7); match original col indexing
            UV_gs_batch[:, :, 4:7] = UV_gs_batch[:, :, 4:7] @ rand_rot
            # Re-compute voxel center PE after rotation
            for b in range(UV_gs_batch.shape[0]):
                shifted     = UV_gs_batch[b, :, 4:7] + origin_offset
                vox_idx     = torch.floor(shifted / resolution_v).clamp(0, volume_dims - 1)
                vox_centers = (vox_idx - (volume_dims - 1) / 2) * resolution_v
                UV_gs_batch[b, :, :3] = vox_centers

        optimizer.zero_grad()

        shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
        )

        loss, KL_loss, w_kl = compute_loss(UV_gs_recover, UV_gs_batch, mu, log_var, epoch)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(gs_autoencoder.parameters(), grad_clip)
        optimizer.step()

        train_losses.append(loss.item())
        train_kl_losses.append(KL_loss.item())
        train_grad_norms.append(grad_norm.item())

    mean_train_loss = np.mean(train_losses)
    mean_train_kl   = np.mean(train_kl_losses)
    mean_grad_norm  = np.mean(train_grad_norms)

    current_lr = optimizer.param_groups[0]["lr"]
    wandb.log({
        "epoch":           epoch,
        "loss/train":      mean_train_loss,
        "loss/train_kl":   mean_train_kl,
        "kl_weight":       kl_weight(epoch),
        "grad_norm":       mean_grad_norm,
        "learning_rate":   current_lr,
    })

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}] Train Loss: {mean_train_loss:.6f}  KL: {mean_train_kl:.6f}  kl_w: {kl_weight(epoch):.2e}  grad_norm: {mean_grad_norm:.4f}")
        gs_autoencoder.eval()
        with torch.no_grad():
            UV_s = next(iter(trainDataLoader))[0].float().to(device)
            _, _, _, _, UV_r = gs_autoencoder(UV_s, UV_s, UV_s, UV_s[:, :, :3])
            print(f"  [DIAG] recover  mean={UV_r.mean().item():.4f}  std={UV_r.std().item():.4f}  spatial_std={UV_r.std(dim=1).mean().item():.4f}")
            print(f"  [DIAG] input    mean={UV_s[:,:,4:].mean().item():.4f}  std={UV_s[:,:,4:].std().item():.4f}  spatial_std={UV_s[:,:,4:].std(dim=1).mean().item():.4f}")
        gs_autoencoder.train()

    a = 2
    # ===== EVAL LOOP =====
    # Also fires on the very first epoch of a resumed run so you get immediate feedback.
    if epoch % eval_every == 0 or epoch == start_epoch:
        print("Evaluating on validation set...")
        gs_autoencoder.eval()
        eval_losses, eval_kl_losses = [], []

        with torch.no_grad():
            for eval_batch in evalDataLoader:
                UV_gs_eval = eval_batch[0].float().to(device)
                _, mu_e, log_var_e, _, UV_gs_recover_e = gs_autoencoder(
                    UV_gs_eval, UV_gs_eval, UV_gs_eval, UV_gs_eval[:, :, :3]
                )
                e_loss, e_kl, _ = compute_loss(UV_gs_recover_e, UV_gs_eval, mu_e, log_var_e, epoch)
                eval_losses.append(e_loss.item())
                eval_kl_losses.append(e_kl.item())

        mean_eval_loss = np.mean(eval_losses)
        mean_eval_kl   = np.mean(eval_kl_losses)

        wandb.log({
            "epoch":         epoch,
            "loss/eval":     mean_eval_loss,
            "loss/eval_kl":  mean_eval_kl,
        })
        print(f"[Epoch {epoch}] Eval  Loss: {mean_eval_loss:.6f}  KL: {mean_eval_kl:.6f}")

        # Save 10 random samples (recovered + GT) from both splits
        ply_dir = os.path.join(save_path, "eval_samples", f"epoch{epoch:06d}")
        os.makedirs(ply_dir, exist_ok=True)
        n_vis = 10
        with torch.no_grad():
            for split_name, dataset in [("train", train_dataset), ("eval", eval_dataset)]:
                n_pick = min(n_vis, len(dataset))
                indices = np.random.choice(len(dataset), size=n_pick, replace=False).tolist()
                subset_loader = Data.DataLoader(
                    Data.Subset(dataset, indices), batch_size=n_pick, shuffle=False, num_workers=0
                )
                UV_vis = next(iter(subset_loader))[0].float().to(device)
                _, _, _, _, UV_vis_recover = gs_autoencoder(
                    UV_vis, UV_vis, UV_vis, UV_vis[:, :, :3]
                )
                for i in range(UV_vis.shape[0]):
                    prefix = os.path.join(ply_dir, f"{split_name}_{i:02d}")
                    save_gaussians_as_ply(UV_vis_recover[i].detach().cpu().numpy(), prefix + "_recovered.ply")
                    save_gaussians_as_ply(UV_vis[i, :, 4:].detach().cpu().numpy(), prefix + "_gt.ply")
        print(f"[Epoch {epoch}] Saved {n_vis} train + {n_vis} eval samples to {ply_dir}")

        # Save best checkpoint
        if mean_eval_loss < best_eval_loss:
            best_eval_loss = mean_eval_loss
            _state = (gs_autoencoder.module.state_dict()
                      if isinstance(gs_autoencoder, nn.DataParallel)
                      else gs_autoencoder.state_dict())
            torch.save({
                "epoch": epoch, "model": _state,
                "optimizer": optimizer.state_dict(),
                "best_eval_loss": best_eval_loss,
            }, os.path.join(save_path, "checkpoint_best.pth"))
            print(f"[Epoch {epoch}] Best eval loss {mean_eval_loss:.6f} — saved checkpoint_best.pth")

    # ===== PERIODIC CHECKPOINT =====
    if (epoch % save_every == 0 and epoch > 0) or epoch == start_epoch:
        _state = (gs_autoencoder.module.state_dict()
                  if isinstance(gs_autoencoder, nn.DataParallel)
                  else gs_autoencoder.state_dict())
        torch.save({
            "epoch": epoch, "model": _state,
            "optimizer": optimizer.state_dict(),
            "best_eval_loss": best_eval_loss,
        }, os.path.join(save_path, f"checkpoint_epoch{epoch}.pth"))

        # Save one train sample and one eval sample (recovered + GT) for visual inspection
        ply_dir = os.path.join(save_path, "eval_samples")
        os.makedirs(ply_dir, exist_ok=True)
        gs_autoencoder.eval()
        with torch.no_grad():
            train_sample = next(iter(trainDataLoader))[0].float().to(device)
            _, _, _, _, train_recover = gs_autoencoder(
                train_sample, train_sample, train_sample, train_sample[:, :, :3]
            )
            eval_sample = next(iter(evalDataLoader))[0].float().to(device)
            _, _, _, _, eval_recover = gs_autoencoder(
                eval_sample, eval_sample, eval_sample, eval_sample[:, :, :3]
            )
        prefix = os.path.join(ply_dir, f"epoch{epoch:06d}")
        save_gaussians_as_ply(train_recover[0].detach().cpu().numpy(), prefix + "_train_recovered.ply")
        save_gaussians_as_ply(train_sample[0, :, 4:].detach().cpu().numpy(), prefix + "_train_gt.ply")
        save_gaussians_as_ply(eval_recover[0].detach().cpu().numpy(),  prefix + "_eval_recovered.ply")
        save_gaussians_as_ply(eval_sample[0, :, 4:].detach().cpu().numpy(),  prefix + "_eval_gt.ply")
        gs_autoencoder.train()
        print(f"[Epoch {epoch}] Periodic checkpoint saved")

print("[INFO] Training complete!")

# ===== FINAL MODEL SAVE =====
state = (gs_autoencoder.module.state_dict()
         if isinstance(gs_autoencoder, nn.DataParallel)
         else gs_autoencoder.state_dict())
torch.save(state, os.path.join(save_path, "model_final.pth"))
print("[INFO] Final model saved.")


def run_final_eval(loader, split_name):
    """Run inference on every batch in loader, save latents + PLY per sample."""
    out_dir = os.path.join(save_path, f"final_{split_name}")
    os.makedirs(out_dir, exist_ok=True)

    all_losses, all_kl = [], []
    sample_idx = 0

    gs_autoencoder.eval()
    with torch.no_grad():
        for batch_data in loader:
            UV_gs = batch_data[0].float().to(device)
            shape_embed, mu, log_var, z, UV_recover = gs_autoencoder(
                UV_gs, UV_gs, UV_gs, UV_gs[:, :, :3]
            )
            loss, kl, _ = compute_loss(UV_recover, UV_gs, mu, log_var, num_epochs)
            all_losses.append(loss.item())
            all_kl.append(kl.item())

            # Save per-sample outputs
            for b in range(UV_gs.shape[0]):
                prefix = os.path.join(out_dir, f"sample_{sample_idx:04d}")
                #torch.save(mu[b],          prefix + "_mu.pt")
                #torch.save(log_var[b],     prefix + "_logvar.pt")
                #torch.save(z[b],           prefix + "_z.pt")
                #torch.save(shape_embed[b], prefix + "_shape_emb.pt")
                recovered_np = UV_recover[b].detach().cpu().numpy()
                save_gaussians_as_ply(recovered_np, prefix + "_recovered.ply")
                # Save GT: cols 4:18 are the 14 Gaussian parameters (skip voxel PE)
                gt_np = UV_gs[b, :, 4:].detach().cpu().numpy()
                save_gaussians_as_ply(gt_np, prefix + "_gt.ply")
                sample_idx += 1

    mean_loss = np.mean(all_losses)
    mean_kl   = np.mean(all_kl)
    print(f"[Final {split_name}] scenes={sample_idx}  loss={mean_loss:.6f}  KL={mean_kl:.6f}")
    wandb.log({f"final/{split_name}_loss": mean_loss, f"final/{split_name}_kl": mean_kl})
    return mean_loss


print("[INFO] Running final evaluation on train set...")
run_final_eval(trainDataLoader, "train")

print("[INFO] Running final evaluation on eval set...")
run_final_eval(evalDataLoader, "eval")

print(f"[INFO] All results saved under: {save_path}")
print("[INFO] Done!")
