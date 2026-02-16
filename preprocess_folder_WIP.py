import torch
import torchio as tio
import os
import nibabel as nib
import SimpleITK as sitk
import re
import subprocess

TARGET_SPACING = (0.8, 0.8, 0.8)  # Target voxel spacing
TARGET_SHAPE = (256, 256, 192)  # Target shape for the images



transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(target=TARGET_SPACING, image_interpolation="linear", label_interpolation="nearest"),
    tio.CropOrPad(target_shape=TARGET_SHAPE, mask_name="mask"),
    tio.ZNormalization(),
])

# Function to resample and save images
def resample_and_save_images(paths, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for path in paths:

        mask_path = path.replace('T2w.nii.gz', 'mask.nii.gz')
        
        # Load the image using TorchIO
        subject = tio.Subject(
            mri=tio.ScalarImage(path),
            mask=tio.LabelMap(mask_path) if os.path.exists(mask_path) else None
        )
        
        # Apply the transform
        preprocessed_subject = transform(subject)

        transformed_image = preprocessed_subject.mri
        
        # Save the transformed image
        output_path = os.path.join(output_folder, os.path.basename(path))
        transformed_image.save(output_path)
        print(f"Saved resampled image to {output_path}")


