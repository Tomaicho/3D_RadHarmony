import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import nibabel as nib

# ============================================================================
# DATASET
# ============================================================================

class MultiSite_2_5D_Dataset(Dataset):
    """
    Loads 3D volumes into 2.5D slices (3 adjacent slices as channels)
    """
    def __init__(self, 
                 image_paths: Dict[int, List[str]], 
                 image_size=256):
        """
        Args:
            image_paths: Dictionary mapping site_idx -> list of image paths
            image_size: Size of 3D images (D, H, W)
        """
        self.image_paths = image_paths
        self.image_size = image_size
        
        # Create flat list of (site_idx, path) tuples
        self.samples = []
        for site_idx, paths in image_paths.items():
            for path in paths:
                # Load volume to get number of slices
                nii = nib.load(path)
                volume = nii.get_fdata()
                
                # Add each slice (except first and last)
                for slice_idx in range(1, volume.shape[2] - 1):
                    self.samples.append((site_idx, path, slice_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        site_idx, path, slice_idx = self.samples[idx]
        
        nii = nib.load(path)
        volume = nii.get_fdata()
        
        # Extract 2.5D slice (3 adjacent slices)
        slice_2_5d = np.stack([
            volume[:, :, slice_idx - 1],
            volume[:, :, slice_idx],
            volume[:, :, slice_idx + 1]
        ], axis=0)  # [3, H, W]
        
        # Normalization to [-1, 1]
        slice_2_5d = (slice_2_5d - slice_2_5d.min()) / (slice_2_5d.max() - slice_2_5d.min() + 1e-8)
        slice_2_5d = slice_2_5d * 2 - 1  # [-1, 1]
        
        # Resize to target size
        slice_tensor = torch.from_numpy(slice_2_5d).float()
        slice_tensor = F.interpolate(
            slice_tensor.unsqueeze(0), 
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return {
            'site': site_idx,
            'image': slice_tensor
        }