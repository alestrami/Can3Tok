import torch
import torch.nn.functional as F
import torch.nn as nn
from argparse import ArgumentParser, Namespace
import os
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from random import randint
import random
#from chamferdist import ChamferDistance
#from geomloss import SamplesLoss
from model.michelangelo import *
from model.michelangelo.models.tsal.tsal_base import ShapeAsLatentModule
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset import gs_dataset

import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel
import torch.utils.data as Data
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt

import wandb


# ===== HELPER FUNCTION TO SAVE PLY =====
def save_gaussians_as_ply(gaussians_data, filepath):
    """
    Save Gaussian parameters as a PLY file.
    
    Args:
        gaussians_data: numpy array [N, 14] with columns:
                        [x, y, z, color_r, color_g, color_b, opacity, scale_x, scale_y, scale_z, rot_x, rot_y, rot_z, rot_w]
        filepath: path to save PLY file
    """
    # Ensure proper shape
    gaussians_data = np.asarray(gaussians_data)
    if gaussians_data.ndim == 1:
        gaussians_data = gaussians_data.reshape(-1, 14)
    
    N = gaussians_data.shape[0]
    
    # Create vertex array
    vertex = np.zeros(N, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                                ('opacity', 'f4'),
                                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')])
    
    vertex['x'] = gaussians_data[:, 0]
    vertex['y'] = gaussians_data[:, 1]
    vertex['z'] = gaussians_data[:, 2]
    vertex['f_dc_0'] = gaussians_data[:, 3]
    vertex['f_dc_1'] = gaussians_data[:, 4]
    vertex['f_dc_2'] = gaussians_data[:, 5]
    vertex['opacity'] = gaussians_data[:, 6]
    vertex['scale_0'] = gaussians_data[:, 7]
    vertex['scale_1'] = gaussians_data[:, 8]
    vertex['scale_2'] = gaussians_data[:, 9]
    vertex['rot_0'] = gaussians_data[:, 10]
    vertex['rot_1'] = gaussians_data[:, 11]
    vertex['rot_2'] = gaussians_data[:, 12]
    vertex['rot_3'] = gaussians_data[:, 13]
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filepath)
    print(f"[INFO] Saved PLY to: {filepath}")


# ===== DEVICE SETUP =====
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===== CONFIG =====
loss_usage = "L1"  # Options: "L1", "chamfer", "sinkhorn"
random_permute = 0
random_rotation = 1
random_shuffle = 1

resol = 200  # Resolution for VAE
num_epochs = 1000
bch_size = 1  # Batch size = 1 for single splat
k_rendering_loss = 1000
enable_rendering_loss = 0  # Disabled - no rendering loss

# Path to your single splat file
#ply_path = "/leonardo_work/IscrC_GEN-X3D/GS/exports/splat_3rdcar/splat.ply"
#ply_path = "/leonardo_work/IscrC_GEN-X3D/GS/exports/splatObj/splat.ply"
#ply_path = "/leonardo_work/IscrC_GEN-X3D/GS/exports/point_cloud_15000.ply"
ply_path = "/data04/alestrami/Points2/data/000-006/0ad3d5e1c18c4fa1ab096893b39f251e/ckpts/point_cloud_15000.ply"
# Save path
save_path = "./resultsObj/"
os.makedirs(save_path, exist_ok=True)
print(f"[INFO] Results will be saved to: {save_path}")

wandb.init(
    project="3DGSAE-training",
    name=f"can3tok-splat-{loss_usage}",
    config={
        "loss_type": loss_usage,
        "num_epochs": num_epochs,
        "batch_size": bch_size,
        "learning_rate": 1e-4,
        "resolution": resol,
        "random_permute": random_permute,
        "random_rotation": random_rotation,
        "random_shuffle": random_shuffle,
        "enable_rendering_loss": enable_rendering_loss,
        "device": str(device),
        "num_gpus": torch.cuda.device_count(),
    },
    tags=["3DGSAE", "Can3Tok", "single-splat"],
    mode="offline"
)
print("[INFO] W&B initialized")

# ===== LOSS FUNCTIONS =====
chamferDist = ()#ChamferDistance()
sinkhorn_eff = ()#SamplesLoss(loss="sinkhorn", p=2, blur=.05)

# ===== LOAD MODEL =====
print("[INFO] Loading Can3Tok model...")
config_path_perciever = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perciever = get_config_from_file(config_path_perciever)
if hasattr(model_config_perciever, "model"):
    model_config_perciever = model_config_perciever.model
perceiver_encoder_decoder = instantiate_from_config(model_config_perciever)

# Multi-GPU or single GPU
if torch.cuda.device_count() > 1:
    gs_autoencoder = nn.DataParallel(perceiver_encoder_decoder)
else:
    gs_autoencoder = perceiver_encoder_decoder
gs_autoencoder.to(device)
print(f"[INFO] Model loaded on {device}")

# ===== OPTIMIZER =====
optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-4, betas=[0.9, 0.999])
print("[INFO] Optimizer initialized")

# ===== LOAD DATASET (SINGLE PLY) =====
print(f"[INFO] Loading single splat from: {ply_path}")
gs_dataset_obj = gs_dataset(
    root="dummy",
    resol=resol,
    random_permute=False,  # No random permutation needed for single file
    train=True,
    single_ply_path=ply_path
)

trainDataLoader = Data.DataLoader(
    dataset=gs_dataset_obj,
    batch_size=bch_size,
    shuffle=False,  # No need to shuffle single file
    num_workers=0   # Avoid multiprocessing for single file
)

print(f"[INFO] DataLoader ready. Dataset length: {len(gs_dataset_obj)}")

# ===== TRAINING LOOP =====
print("[INFO] Starting training loop...")
print(f"[INFO] Loss type: {loss_usage}")
print(f"[INFO] Rendering loss enabled: {enable_rendering_loss}")
print(f"[INFO] Number of epochs: {num_epochs}")

for epoch in tqdm(range(num_epochs)):
    for i_batch, batch_data in enumerate(trainDataLoader):
        # Handle batch format: DataLoader returns (data, indices) as tuple
        if isinstance(batch_data, (list, tuple)):
            UV_gs_batch = batch_data[0]
            if isinstance(UV_gs_batch, np.ndarray):
                UV_gs_batch = torch.from_numpy(UV_gs_batch).float()
            else:
                UV_gs_batch = UV_gs_batch.float()
        else:
            UV_gs_batch = batch_data.float()
        
        UV_gs_batch = UV_gs_batch.to(device)
        
        # Shape: [batch_size, num_gaussians, num_features]
        # Expected: [1, 40000, 18]
        # Last 14 features are the Gaussian parameters: [xyz (3), color (3), opacity (1), scale (3), rot (4)]
        
        # Optional: Random permutation
        if epoch % 1 == 0 and random_permute == 1:
            UV_gs_batch = UV_gs_batch[:, torch.randperm(UV_gs_batch.size()[1])]
        
        # Optional: Random rotation
        if epoch % 5 == 0 and epoch > 1 and random_rotation == 1:
            rand_rot_comp = special_ortho_group.rvs(3)
            rand_rot = torch.tensor(np.dot(rand_rot_comp, rand_rot_comp.T), dtype=torch.float32).to(UV_gs_batch.device)
            #UV_gs_batch[:, :, 7:10] = UV_gs_batch[:, :, 7:10] @ rand_rot  # Rotate scale
            UV_gs_batch[:,:,4:7] = UV_gs_batch[:,:,4:7] @ rand_rot  # rotates XYZ positions ✓

        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
        )
        
        # KL divergence loss (VAE regularization)
        KL_loss = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # Reconstruction loss (parameter space only, no rendering)
        loss_render = 0.0  # Rendering loss disabled
        
        if loss_usage == "L1":
            # Compare reconstructed Gaussians to input Gaussians
            # Extract last 14 features (Gaussian parameters)
            rec_loss = torch.norm(
                UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14)
                - UV_gs_batch[:, :, 4:],  # Skip first 4 features (voxel centers + voxel id)
                p=2
            ) / UV_gs_batch.shape[0]
            loss = rec_loss + 1e-5 * KL_loss
        
        elif loss_usage == "chamfer":
            # Chamfer distance on both position and attributes
            batch_size = UV_gs_batch.shape[0]
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
            loss = rec_loss + 0.0001 * KL_loss
        
        elif loss_usage == "sinkhorn":
            # Sinkhorn distance
            sinkhorn_loss_ = sinkhorn_eff(
                UV_gs_batch.contiguous().reshape([bch_size, -1, 14]),
                UV_gs_recover.contiguous().reshape([bch_size, -1, 14])
            ).mean()
            loss = sinkhorn_loss_ + 0.001 * KL_loss
        
        else:
            raise ValueError(f"Unknown loss_usage: {loss_usage}")
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "loss/total": loss.item(),
            "loss/kl": KL_loss.item()
        })
        
        
        # Logging
        if epoch % 100 == 0:
            print(f"\n[Epoch {epoch}] Loss: {loss.item():.6f}, KL: {KL_loss.item():.6f}, Recon: {(loss - 1e-5 * KL_loss).item():.6f}")
        
        # Save checkpoints
        if epoch % 1000 == 0 and epoch > 0:
            print(f"[Epoch {epoch}] Saving checkpoint...")
            
            # Save model state
            if isinstance(gs_autoencoder, nn.DataParallel):
                torch.save(gs_autoencoder.module.state_dict(), os.path.join(save_path, f"model_{epoch}.pth"))
            else:
                torch.save(gs_autoencoder.state_dict(), os.path.join(save_path, f"model_{epoch}.pth"))

            # Print latent dimensions
            print(f"[Latent Shapes]")
            print(f"  z shape: {z.shape}")
            print(f"  mu shape: {mu.shape}")
            print(f"  log_var shape: {log_var.shape}")
            print(f"  shape_embed shape: {shape_embed.shape}")

            # Save latent codesß
            torch.save(mu, os.path.join(save_path, f"gs_mu_{epoch}.pt"))
            torch.save(log_var, os.path.join(save_path, f"gs_var_{epoch}.pt"))
            torch.save(z, os.path.join(save_path, f"gs_z_{epoch}.pt"))
            torch.save(shape_embed, os.path.join(save_path, f"gs_shape_emb_{epoch}.pt"))
            
            # Save reconstruction as numpy
            recovered_gs_np = UV_gs_recover[0].detach().cpu().numpy()  # [40000, 14]
            np.save(os.path.join(save_path, f"recovered_gaussians_{epoch}.npy"), recovered_gs_np)
            
            # Save reconstruction as PLY
            ply_path_save = os.path.join(save_path, f"recovered_gaussians_{epoch}.ply")
            save_gaussians_as_ply(recovered_gs_np, ply_path_save)
            
            print(f"[Epoch {epoch}] Checkpoint saved!")

print("[INFO] Training complete!")

# ===== FINAL EVALUATION & SAVE =====
print("[INFO] Saving final results...")

gs_autoencoder.to(device)
gs_autoencoder.eval()
with torch.no_grad():
    for batch_data in trainDataLoader:
        # Handle batch format - same logic as training
        if isinstance(batch_data, (list, tuple)):
            UV_gs_batch = batch_data[0]
            if isinstance(UV_gs_batch, np.ndarray):
                UV_gs_batch = torch.from_numpy(UV_gs_batch).float()
            else:
                UV_gs_batch = UV_gs_batch.float()
        else:
            # batch_data is a tensor
            if isinstance(batch_data, np.ndarray):
                UV_gs_batch = torch.from_numpy(batch_data).float()
            else:
                UV_gs_batch = batch_data.float()
        
              # Ensure correct shape [batch_size, num_gaussians, num_features]
        if UV_gs_batch.ndim == 2:
            # If shape is [num_gaussians, num_features], add batch dimension
            UV_gs_batch = UV_gs_batch.unsqueeze(0)
        
        print(f"[DEBUG] Eval batch shape: {UV_gs_batch.shape}")
        
        try:
            UV_gs_batch = UV_gs_batch.to(device)
            shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(
                UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
            )
        except Exception as e:
            print(f"[ERROR] Forward pass failed: {e}")
            print(f"[DEBUG] Input shape to model: {UV_gs_batch.shape}")
            raise
        
        # Save final latent codes
        torch.save(shape_embed, os.path.join(save_path, f"gs_shape_emb_final.pt"))
        torch.save(z, os.path.join(save_path, f"gs_z_final.pt"))
        #torch.save(mu, os.path.join(save_path, f"gs_mu_final.pt"))
        #torch.save(log_var, os.path.join(save_path, f"gs_var_final.pt"))
        #torch.save(z, os.path.join(save_path, f"gs_emb_final.pt"))
        
        # Save final reconstruction as numpy
        recovered_gs_np = UV_gs_recover[0].detach().cpu().numpy()  # [40000, 14]
        np.save(os.path.join(save_path, f"recovered_gaussians_final.npy"), recovered_gs_np)
        
        # Save final reconstruction as PLY
        ply_path_final = os.path.join(save_path, f"recovered_gaussians_final.ply")
        save_gaussians_as_ply(recovered_gs_np, ply_path_final)
        
        # Optional: Print reconstruction statistics
        input_gs = UV_gs_batch[0, :, 4:].detach().cpu().numpy()
        output_gs = UV_gs_recover[0].detach().cpu().numpy()
        
        #mae = np.mean(np.abs(input_gs - output_gs))
        #mse = np.mean((input_gs - output_gs) ** 2)
        
        print(f"[Final Stats]")
        #print(f"  Mean Absolute Error: {mae:.6f}")
        #print(f"  Mean Squared Error: {mse:.6f}")
        print(f"  Input shape: {input_gs.shape}")
        print(f"  Output shape: {output_gs.shape}")

print(f"[INFO] All results saved to: {save_path}")
print("[INFO] Done!")
