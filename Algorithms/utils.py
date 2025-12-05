import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np

# Pomocnicza kalsa reprezentująca pojedynczy blok  
class DoubleConv3DBatch(nn.Module):
    """
    Blok wykonujący dwie następujące po sobie konwolucje. Preferowany w przypadku batch size > 1.
    """
    def __init__(self, in_channels, out_channels, ker_size = 3):
        super(DoubleConv3DBatch, self).__init__()
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



class DoubleConv3DInstance(nn.Module):
    """
    Blok wykonujący dwie następujące po sobie konwolucje. Preferowany w przypadku batch size = 1.
    """
    def __init__(self, in_channels, out_channels, ker_size = 3):
        super(DoubleConv3DInstance, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=ker_size, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=ker_size, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.block(x)