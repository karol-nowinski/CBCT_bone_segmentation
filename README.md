# Semantic segmentation of bone structures in 3D computed tomography images

This repository contains the code that was used for the master thesis. The goal of the project is to develop a deep learning model capable of automatically identifying and segmenting anatomical regions such as the mandible, maxilla, and teeth in 3D CBCT data.

Semantic segmentation in the context of 3D imaging is a process of assigning a class label to each voxel, equivalent of pixel in 3D images, within a volumetric scan, such as a CBCT (Cone Beam Computed Tomography) volume. Unlike traditional 2D segmentation, which works on individual image slices, 3D semantic segmentation considers spatial context across all three dimensions, enabling more accurate delineation of anatomical structures. It is particuraly usefull in the field of medicne where accuracy and precisione of the localization is extreamly important.

Training a 3D model requires significantly more computational resources due to the high dimensionality of the input data, increased memory usage, and the complexity of 3D convolutional operations.

# Datasets

In the field of medical imaging, especially for 3D data, publicly available annotated datasets are limited due to privacy concerns, annotation costs, and the complexity of acquiring high-quality volumetric scans. This scarcity of data poses a significant challenge for training deep learning models, which typically require large amounts of labeled data to generalize well.

For the purposes of this project, the following CBCT datasets were used:

- ToothFairy2: A publicly available dataset consisting of 3D CBCT scans with pixel-level annotations for dental and jaw structures. It provides high-resolution volumes suitable for training and evaluating models on tasks such as mandible and teeth segmentation.
- China CBCT: A subset of the dataset used in the paper " fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images". Due to problem with labels only a 100 of CBCT scans could be used. It contains only teeth structures.

To increase the diversity of the training data and improve the generalization capability of the models, data augmentation techniques were applied during training. More about it is written in the master thesis.

# Implemented models

- 3D U-Net: A volumetric extension of the classic U-Net architecture, widely used in medical image segmentation tasks. This implementation is based on the paper “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation” by Çiçek et al., which adapts U-Net to work directly on 3D medical volumes using 3D convolutions.

- 3D U-Net++: A hybrid architecture combining elements of both 3D U-Net and U-Net++. Unlike the standard 3D U-Net, this model incorporates nested and dense skip pathways inspired by U-Net++, allowing for more efficient feature re-use and better gradient flow. In this project, the original U-Net++ was modified to operate entirely in 3D, making it suitable for volumetric CBCT segmentation.

A more detailed description of both models can be found in the master's thesis.

# Technology

The list below contains all the technology used in the project and requiered to run the models:

- Python 3.9+
- PyTorch
- Torchio
- SimpleITK
- MatplotLib
- Tensorflow board

# How to run

# Results
