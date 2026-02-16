"""
Inference module for MURD 2.5D model
Harmonizes full 3D volumes by processing them slice-by-slice
"""

from utils import load_model_for_inference, harmonize_volume_2_5d


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MURD 2.5D Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input NIfTI file')
    parser.add_argument('--output-folder', type=str, required=True, help='Output folder')
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
        output_folder=args.output_folder,
        source_site=args.source_site,
        target_site=args.target_site,
        device=args.device,
        use_reference=(args.reference is not None),
        reference_path=args.reference,
        consistent_style=args.consistent_style,
        save_visualization=True
    )