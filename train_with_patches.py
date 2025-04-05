import torch
from torch.utils.data import DataLoader
import torchio as tio
from torchio import Subject, ScalarImage, LabelMap
from torchio.data import GridSampler
from Algorithms.Unet3D.unet3D import UNet3D  # lub UNetPP3D
from Unet3d_Trainer import UnetTrainer
import glob