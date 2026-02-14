# 2.5D_RadHarmony
Project for the harmonization of 3D radiology images (MRI, CT) across sites or sequences.

This work has been adapted from Liu, S. & Yap, P.T. (2024). Learning multi-site harmonization of magnetic resonance images without traveling human phantoms. Communications Engineering.

Given a set of sites, 2.5D_RadHamony learns the universal structural content features of the images, and the site-speific style features. The content and style features are disentangled, in order to ensure no information loss and full site-transferability of the images.

Besides site-harmonization. 2.5D_RadHarmony can also be applied to cross-modality generation. We have implemented 2.5D_RadHarmony for the inter-generation of 3T and 7T brain MRI volumes.
