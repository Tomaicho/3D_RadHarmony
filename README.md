# 3D_RadHarmony
Project for the harmonization of 3D radiology images (MRI, CT) across sites or sequences.

This work has been adapted from Liu, S. & Yap, P.T. (2024). "Learning multi-site harmonization of magnetic resonance images without traveling human phantoms." Communications Engineering.

Given a set of sites, 3D_RadHamony learns the universal structural content features of the images, and the site-specific style features. The content and style features are disentangled, in order to ensure no information loss and full site-transferability of the images.

Besides site-harmonization, 3D_RadHarmony can also be applied to cross-modality generation. We have implemented 3D_RadHarmony for the inter-generation of 3T and 7T brain MRI volumes.

## Requirements
- PyTorch > 2.2.1
Recommended: 
- CUDA > 12.1.1 (for performance reasons) - also possible to run in cpu, but slower.

## Preare your data
Create a folder for your data inside the data/ folder an place it there. Create a subfolder for each site. We recommend to subdivide the data into train and test/inference. The data structure should be something like:
```
data
├── site_i
│   ├── test
│   │   └── 10.nii.gz
│   └── train
│       ├── 01.nii.gz
│       ├── 02.nii.gz
│       ├── ...
└── site_j
    ├── test
    │   └── 10.nii.gz
    └── train
        ├── 01.nii.gz
        ├── 02.nii.gz
        ├── ...
```

3D_RadHarmony was optimized for volumes with 256x256xN dimensions. Make sure to prepare your data accordingly, or use the preprocess_folder.py script for preprocessing the selected folders at a time.

## Usage
**train.py** - train a new model. Define the data folders' paths. Checkpoints are saved to the checkpoints/ folder
**inference.py** - run inference on a file (or folder) using a selected checkpoint