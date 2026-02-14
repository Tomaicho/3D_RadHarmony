"""
Inference module for MURD 2.5D model
Harmonizes full 3D volumes by processing them slice-by-slice
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# VOLUME-LEVEL INFERENCE
# ============================================================================

@torch.no_grad()
def harmonize_volume_2_5d(
    model,
    nifti_input_path: str,
    output_folder: str,
    source_site: int,
    target_site: int,
    device: str = "cuda",
    image_size: int = 256,
    axis: int = 2,
    consistent_style: bool = True,
    use_reference: bool = False,
    reference_path: Optional[str] = None,
    reference_slice_idx: Optional[int] = None,
    save_visualization: bool = True,
):
    """
    Harmonize a full 3D NIfTI volume using 2.5D MURD model.
    
    Args:
        model: Trained MURD2_5D model
        nifti_input_path: Path to input NIfTI file
        output_folder: Path to folder to save inference data
        source_site: Source site index (0, 1, etc.)
        target_site: Target site index
        device: 'cuda' or 'cpu'
        image_size: Size model was trained on (default 256)
        axis: Which axis to slice along (2 = axial for standard orientation)
        consistent_style: If True, uses same style for all slices
        use_reference: If True, extracts style from a reference image
        reference_path: Path to reference NIfTI (if use_reference=True)
        reference_slice_idx: Which slice to use from reference (middle if None)
        save_visualization: Save before/after comparison
    
    Returns:
        Harmonized volume as numpy array
    """
    
    print(f"\n{'='*60}")
    print(f"MURD 2.5D Volume Harmonization")
    print(f"{'='*60}")
    print(f"Input: {nifti_input_path}")
    print(f"Output folder: {output_folder}")
    print(f"Source site: {source_site} → Target site: {target_site}")
    print(f"Consistent style: {consistent_style}")
    print(f"Use reference: {use_reference}")
    
    model = model.to(device)
    model.eval()
    
    # Load input volume
    nii = nib.load(nifti_input_path)
    volume_orig = nii.get_fdata().astype(np.float32)
    
    print(f"Original volume shape: {volume_orig.shape}")
    
    # Move slicing axis to last dimension if needed
    if axis != 2:
        volume_orig = np.moveaxis(volume_orig, axis, 2)
    
    H, W, D = volume_orig.shape
    print(f"Processing shape (H, W, D): {H} × {W} × {D}")
    
    # --- Normalization ---
    volume = volume_orig.copy()
    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)
    volume = volume * 2 - 1  # Scale to [-1, 1]
    
    print(f"Normalized range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Pad along slice dimension for boundary slices
    volume_pad = np.pad(volume, ((0,0), (0,0), (1,1)), mode='edge')
    
    # Prepare output
    output_volume = np.zeros((image_size, image_size, D), dtype=np.float32)
    
    # --- Style extraction ---
    style = None
    if use_reference and reference_path is not None:
        print(f"\nExtracting style from reference: {reference_path}")
        style = extract_style_from_reference(
            model=model,
            reference_path=reference_path,
            reference_site=target_site,
            slice_idx=reference_slice_idx,
            image_size=image_size,
            device=device,
            axis=axis
        )
        print(f"Style extracted from reference (shape: {style.shape})")
    
    elif consistent_style:
        # Generate one random style for entire volume
        print(f"\nGenerating consistent random style for target site {target_site}")
        style = model.generate_style(batch_size=1, site_idx=target_site, device=device)
        print(f"Style generated (shape: {style.shape})")
    
    # --- Process each slice ---
    print(f"\nProcessing {D} slices...")
    for k in tqdm(range(D), desc="Harmonizing slices"):
        # Extract 2.5D stack (3 adjacent slices)
        slice_stack = np.stack([
            volume_pad[:, :, k],     # k-1 (due to padding)
            volume_pad[:, :, k+1],   # k
            volume_pad[:, :, k+2]    # k+1
        ], axis=0)  # [3, H, W]
        
        # Convert to tensor and add batch dimension
        x = torch.from_numpy(slice_stack).float().unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # Resize to training size if needed
        if (H, W) != (image_size, image_size):
            x = F.interpolate(
                x, 
                size=(image_size, image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Encode content
        content = model.encode_content(x)
        
        # Get or generate style
        if style is None:
            # Generate new style for each slice (diverse styles)
            current_style = model.generate_style(batch_size=1, site_idx=target_site, device=device)
        else:
            current_style = style
        
        # Decode harmonized image
        harmonized = model.decode(content, current_style)  # [1, 3, H, W]
        
        # Extract middle channel (corresponds to current slice k)
        harmonized_slice = harmonized[0, 1].cpu().numpy()  # [H, W]
        
        # Store in output volume
        output_volume[:, :, k] = harmonized_slice
    
    print("✓ All slices processed")
    
    # --- Post-processing ---
    # Output is in [-1, 1], map back to original range
    output_volume = (output_volume + 1) / 2  # [0, 1]
    output_volume = output_volume * (vmax - vmin) + vmin
    
    print(f"Output range: [{output_volume.min():.3f}, {output_volume.max():.3f}]")
    
    # Restore original axis order if needed
    if axis != 2:
        output_volume = np.moveaxis(output_volume, 2, axis)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # --- Save output ---
    output_nii = nib.Nifti1Image(output_volume, affine=nii.affine, header=nii.header)
    nifti_output_path = os.path.join(output_folder, os.path.basename(nifti_input_path))
    nib.save(output_nii, nifti_output_path)
    print(f"✓ Saved harmonized volume: {nifti_output_path}")
    
    # --- Visualization ---
    if save_visualization:
        vis_path = str(Path(nifti_output_path).with_suffix('')) + '_comparison.png'
        visualize_harmonization(
            original=volume_orig,
            harmonized=output_volume if axis == 2 else np.moveaxis(output_volume, axis, 2),
            save_path=vis_path,
            slice_idx=D // 2
        )
        print(f"✓ Saved visualization: {vis_path}")
    
    print(f"{'='*60}\n")
    
    return output_volume


# ============================================================================
# REFERENCE-GUIDED STYLE EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_style_from_reference(
    model,
    reference_path: str,
    reference_site: int,
    slice_idx: Optional[int] = None,
    image_size: int = 256,
    device: str = "cuda",
    axis: int = 2
) -> torch.Tensor:
    """
    Extract style vector from a reference volume.
    
    Args:
        model: MURD2_5D model
        reference_path: Path to reference NIfTI
        reference_site: Site index of reference
        slice_idx: Which slice to use (middle if None)
        image_size: Model input size
        device: 'cuda' or 'cpu'
        axis: Slicing axis
    
    Returns:
        Style tensor [1, style_dim]
    """
    model.eval()
    
    # Load reference
    ref_nii = nib.load(reference_path)
    ref_volume = ref_nii.get_fdata().astype(np.float32)
    
    if axis != 2:
        ref_volume = np.moveaxis(ref_volume, axis, 2)
    
    H, W, D = ref_volume.shape
    
    # Choose slice
    if slice_idx is None:
        slice_idx = D // 2  # Middle slice
    slice_idx = int(np.clip(slice_idx, 1, D - 2))
    
    # Normalize (same as training)
    vmin, vmax = ref_volume.min(), ref_volume.max()
    ref_volume = (ref_volume - vmin) / (vmax - vmin + 1e-8)
    ref_volume = ref_volume * 2 - 1
    
    # Extract 2.5D stack
    slice_stack = np.stack([
        ref_volume[:, :, slice_idx - 1],
        ref_volume[:, :, slice_idx],
        ref_volume[:, :, slice_idx + 1]
    ], axis=0)
    
    # Convert to tensor
    x = torch.from_numpy(slice_stack).float().unsqueeze(0).to(device)
    
    # Resize if needed
    if (H, W) != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode='bilinear', align_corners=False)
    
    # Extract style
    style = model.encode_style(x, reference_site)
    
    return style


# ============================================================================
# BATCH INFERENCE
# ============================================================================

def harmonize_batch(
    model,
    input_paths: list,
    output_dir: str,
    source_site: int,
    target_site: int,
    device: str = "cuda",
    **kwargs
):
    """
    Harmonize multiple volumes.
    
    Args:
        model: MURD2_5D model
        input_paths: List of input NIfTI paths
        output_dir: Directory to save outputs
        source_site: Source site index
        target_site: Target site index
        device: 'cuda' or 'cpu'
        **kwargs: Additional arguments for harmonize_volume_2_5d
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Batch Harmonization: {len(input_paths)} volumes")
    print(f"Source site: {source_site} → Target site: {target_site}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    for i, input_path in enumerate(input_paths, 1):
        print(f"\nProcessing {i}/{len(input_paths)}: {Path(input_path).name}")
        
        output_path = output_dir / f"{Path(input_path).stem}_harmonized.nii.gz"
        
        try:
            harmonize_volume_2_5d(
                model=model,
                nifti_input_path=input_path,
                nifti_output_path=str(output_path),
                source_site=source_site,
                target_site=target_site,
                device=device,
                **kwargs
            )
        except Exception as e:
            print(f"✗ Error processing {input_path}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Batch harmonization complete!")
    print(f"{'='*60}\n")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_harmonization(
    original: np.ndarray,
    harmonized: np.ndarray,
    save_path: str,
    slice_idx: Optional[int] = None,
    n_slices: int = 5
):
    """
    Create before/after comparison visualization.
    
    Args:
        original: Original volume [H, W, D]
        harmonized: Harmonized volume [H, W, D]
        save_path: Where to save figure
        slice_idx: Center slice (middle if None)
        n_slices: Number of slices to show
    """
    D = original.shape[2]
    
    if slice_idx is None:
        slice_idx = D // 2
    
    # Select evenly spaced slices around center
    slice_indices = np.linspace(
        max(0, slice_idx - n_slices//2),
        min(D-1, slice_idx + n_slices//2),
        n_slices,
        dtype=int
    )
    
    fig, axes = plt.subplots(2, n_slices, figsize=(3*n_slices, 6))
    
    for i, idx in enumerate(slice_indices):
        # Original
        axes[0, i].imshow(original[:, :, idx], cmap='gray')
        axes[0, i].set_title(f'Original\nSlice {idx}')
        axes[0, i].axis('off')
        
        # Harmonized
        axes[1, i].imshow(harmonized[:, :, idx], cmap='gray')
        axes[1, i].set_title(f'Harmonized\nSlice {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# HELPER: LOAD MODEL FROM CHECKPOINT
# ============================================================================

def load_model_for_inference(
    checkpoint_path: str,
    num_sites: int = 2,
    device: str = "cuda"
):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        num_sites: Number of sites in training
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded MURD2_5D model in eval mode
    """
    from model_2_5D import MURD2_5D
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model = MURD2_5D(
        in_channels=3,
        num_sites=num_sites,
        style_dim=64,
        latent_dim=16,
        base_channels=64
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Loaded model from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MURD 2.5D Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input NIfTI file')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--source-site', type=int, required=True, help='Source site index')
    parser.add_argument('--target-site', type=int, required=True, help='Target site index')
    parser.add_argument('--num-sites', type=int, default=2, help='Total number of sites')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--reference', type=str, default=None, help='Reference image for style')
    parser.add_argument('--consistent-style', action='store_true', help='Use consistent style')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_for_inference(
        checkpoint_path=args.checkpoint,
        num_sites=args.num_sites,
        device=args.device
    )
    
    # Harmonize with a reference image (style extracted from reference image)
    harmonize_volume_2_5d(
        model=model,
        nifti_input_path=args.input,
        output_folder=args.output,
        source_site=args.source_site,
        target_site=args.target_site,
        device=args.device,
        use_reference=(args.reference is not None),
        reference_path=args.reference,
        consistent_style=args.consistent_style,
        save_visualization=True
    )

    # # Harmonize without reference image (generation of random target site style)
    # harmonize_volume_2_5d(
    #     model=model,
    #     nifti_input_path=args.input,
    #     output_folder=args.output.replace('reference_image', 'no_reference_image'),
    #     source_site=args.source_site,
    #     target_site=args.target_site,
    #     device=args.device,
    #     use_reference=(args.reference is not None),
    #     reference_path=None,
    #     consistent_style=args.consistent_style,
    #     save_visualization=True
    # )