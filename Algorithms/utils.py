import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np

# Pomocnicza kalsa reprezentująca pojedynczy blok  
class DoubleConv(nn.Module):
    """(Conv3D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, ker_size = 3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=ker_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=ker_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

def Get_biggest_target_shape(image_files):
    max_shape = np.array([0,0,0])

    for image in image_files:
        image  = sitk.ReadImage(image)
        image_size = sitk.GetArrayFromImage(image)
        max_shape = np.maximum(max_shape,image_size.shape)
    return max_shape

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, epsilon=1e-6):
        intersection = (input * target).sum(dim=(2, 3, 4))
        union = input.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = 2 * intersection / (union + epsilon)
        return 1 - dice.mean()
