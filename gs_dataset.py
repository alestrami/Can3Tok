import os
import torch
from PIL import Image
from torch.utils import data
import numpy as np
from torch.utils.data import DataLoader
from voxelize import voxelize
from plyfile import PlyData, PlyElement
import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel


class gs_dataset(data.Dataset):
    def __init__(self, root, resol, random_permute=False, train=True, single_ply_path=None):
        """
        Args:
            root: Path to dataset folder (ignored if single_ply_path is provided)
            resol: Resolution for voxelization
            random_permute: Whether to randomly permute Gaussians
            train: Training mode flag
            single_ply_path: (Optional) Path to a single .ply file to load instead of dataset folder
        """
        self.data_path = root
        self.resol = resol
        self.random_permute = random_permute
        self.single_ply_path = single_ply_path
        
        # If single_ply_path is provided, use single-file mode
        if single_ply_path is not None:
            self.single_mode = True
            self.folder_path_each = [single_ply_path]  # Treat as single "scene"
            print(f"[gs_dataset] Single PLY mode: {single_ply_path}")
        else:
            self.single_mode = False
            self.folder_path_each = os.listdir(self.data_path)[:1000]
            print(f"[gs_dataset] Multi-scene mode: {len(self.folder_path_each)} scenes found")


    def __getitem__(self, index):
        if self.single_mode:
            # Single PLY file mode
            gs_params_path_each = self.single_ply_path
        else:
            # Multi-scene mode (original behavior)
            gs_params_path_each = self.data_path + self.folder_path_each[index] + f"/point_cloud/iteration_30000/gs_filtered.ply"
        
        # Load PLY file
        plydata = PlyData.read(gs_params_path_each)
        
        # Extract XYZ
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        # Extract color (DC component)
        color_rgb = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                              np.asarray(plydata.elements[0]["f_dc_1"]),
                              np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)
        
        # Extract opacity
        opacity = np.asarray(plydata.elements[0]["opacity"])
        
        # Extract scale
        scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                          np.asarray(plydata.elements[0]["scale_1"]),
                          np.asarray(plydata.elements[0]["scale_2"])),  axis=1)
        
        # Extract rotation
        rot = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                        np.asarray(plydata.elements[0]["rot_1"]),
                        np.asarray(plydata.elements[0]["rot_2"]),
                        np.asarray(plydata.elements[0]["rot_3"])),  axis=1)
        
        # ---- Positional Encoding based on voxel centers ----
        coord_min = np.min(xyz, 0)
        coord = xyz - coord_min

        out = voxelize(coord, 0.8, 'fnv')
        uniq_idx, count = out[0], out[1] 
        #uniq_idx, count = voxelize(coord, 0.8, 'fnv')

        num_gaussians = xyz.shape[0]

        # FIX: Handle voxelize output - ensure it's an array with same length as num_gaussians
        uniq_idx = np.asarray(uniq_idx, dtype=np.float32)

        # If scalar or wrong shape, create a dummy voxel ID array
        if uniq_idx.ndim == 0 or uniq_idx.size == 1:
            # voxelize returned scalar or single element - create sequential IDs instead
            uniq_idx = np.arange(num_gaussians, dtype=np.float32)
        elif len(uniq_idx) != num_gaussians:
            # Voxelize returned wrong number of IDs - use sequential instead
            print(f"[WARNING] voxelize returned {len(uniq_idx)} IDs for {num_gaussians} Gaussians. Using sequential IDs.")
            uniq_idx = np.arange(num_gaussians, dtype=np.float32)

        # Combine all Gaussian parameters
        gs_full_params = np.concatenate((xyz, color_rgb, opacity[:, None], scale, rot), axis=1)


        
        #uniq_idx, count = voxelize(coord, 0.8, 'fnv')  # voxel_size=0.8 -> resolution~20
        
        # Combine all Gaussian parameters
        #gs_full_params = np.concatenate((xyz, color_rgb, opacity[:, None], scale, rot), axis=1)
        
        # ---- Voxel center-based positional encoding ----
        volume_dims = 20  # 40
        resolution = 16.0 / volume_dims
        origin_offset = np.array([(volume_dims - 1) / 2, (volume_dims - 1) / 2, (volume_dims - 1) / 2]) * resolution
        shifted_points = xyz + origin_offset
        voxel_indices = np.floor(shifted_points / resolution).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, np.array(volume_dims) - 1)
        voxel_centers = (voxel_indices - (np.array(volume_dims) - 1) / 2) * resolution
        
        # Concatenate: [voxel_centers (3), voxel_id (1), xyz (3), color (3), opacity (1), scale (3), rot (4)] = 18 features
        gs_full_params = np.concatenate((voxel_centers, np.array(uniq_idx)[:, None], gs_full_params), axis=1)
        
        # ---- Padding to fixed size (40000 Gaussians) ----
        target_num_gs = 40000
        if gs_full_params.shape[0] != target_num_gs:
            # Create zero-padded array
            padded_params = np.zeros([target_num_gs, 18], dtype=np.float32)
            
            # Fill with actual data
            num_actual = gs_full_params.shape[0]
            padded_params[:num_actual, :] = gs_full_params
            
            # Optional: fill padding with last Gaussian (or zeros)
            if num_actual > 0:
                padded_params[num_actual:, :] = gs_full_params[-1, :]
            
            gs_full_params = padded_params
        
        return gs_full_params, index


    def __len__(self):
        return len(self.folder_path_each)


    @staticmethod
    def load_ply_data(ply_path):
        """
        Load 3DGS data from PLY file (generic loader)
        
        Args:
            ply_path: Path to .ply file
            
        Returns:
            parameters (torch.Tensor): [N, D] tensor of raw PLY data
            property_names (list): Names of properties in PLY
        """
        plydata = PlyData.read(ply_path)
        vertex_data = plydata['vertex']
        
        # Get property names
        property_names = [prop.name for prop in vertex_data.properties]
        print(f"[load_ply_data] Found {len(property_names)} properties: {property_names}")
        
        # Extract data
        structured_data = vertex_data.data
        parameters = np.array(structured_data.tolist(), dtype=np.float32)
        
        print(f"[load_ply_data] Loaded {len(parameters)} Gaussians with shape {parameters.shape}")
        
        return torch.tensor(parameters, dtype=torch.float32), property_names
