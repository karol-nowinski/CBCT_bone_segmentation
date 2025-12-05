import torch
import torch.nn as nn
import torch.nn.functional as F
import Algorithms.losses as crit
from ..utils import DoubleConv3DBatch, DoubleConv3DInstance
import datetime
import os


# Pomocnicza kalsa reprezentująca pojedynczy blok  
# class DoubleConv(nn.Module):
#     """(Conv3D -> BatchNorm -> ReLU) * 2"""
#     def __init__(self, in_channels, out_channels, ker_size = 3):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=ker_size, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=ker_size, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)


# wersja z głębokością równą 5
# class UNet3D(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, base_channels=32):
#         super(UNet3D, self).__init__()
        
#         self.deep_supervision = False

#         self.encoder1 = DoubleConv3DInstance(in_channels, base_channels)
#         self.encoder2 = DoubleConv3DInstance(base_channels, base_channels * 2)
#         self.encoder3 = DoubleConv3DInstance(base_channels * 2, base_channels * 4)
#         self.encoder4 = DoubleConv3DInstance(base_channels * 4, base_channels * 8)
        
#         self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
#         self.bottleneck = DoubleConv3DInstance(base_channels * 8, base_channels * 16)
        
#         self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
#         self.decoder4 = DoubleConv3DInstance(base_channels * 16, base_channels * 8)
#         self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
#         self.decoder3 = DoubleConv3DInstance(base_channels * 8, base_channels * 4)
#         self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
#         self.decoder2 = DoubleConv3DInstance(base_channels * 4, base_channels * 2)
#         self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
#         self.decoder1 = DoubleConv3DInstance(base_channels * 2, base_channels)
        
#         self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool(enc1))
#         enc3 = self.encoder3(self.pool(enc2))
#         enc4 = self.encoder4(self.pool(enc3))
        
#         bottleneck = self.bottleneck(self.pool(enc4))
        
#         dec4 = self.upconv4(bottleneck) # up sample
#         dec4 = torch.cat((enc4, dec4), dim=1) # skip conection
#         dec4 = self.decoder4(dec4)
        
#         dec3 = self.upconv3(dec4) # up sample
#         dec3 = torch.cat((enc3, dec3), dim=1) # skip conection
#         dec3 = self.decoder3(dec3)
        
#         dec2 = self.upconv2(dec3) # up sample
#         dec2 = torch.cat((enc2, dec2), dim=1) # skip conection
#         dec2 = self.decoder2(dec2)
        
#         dec1 = self.upconv1(dec2) # up sample
#         dec1 = torch.cat((enc1, dec1), dim=1) # skip conection
#         dec1 = self.decoder1(dec1)
        
#         return self.final_conv(dec1)

#     def GetName(self):
#         return "Unet3D"


# version with 4 depth
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(UNet3D, self).__init__()
        
        self.deep_supervision = False

        self.encoder1 = DoubleConv3DBatch(in_channels, base_channels)
        self.encoder2 = DoubleConv3DBatch(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv3DBatch(base_channels * 2, base_channels * 4)
        self.bottleneck = DoubleConv3DBatch(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3DBatch(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3DBatch(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3DBatch(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        
        bottleneck = self.bottleneck(self.pool(enc3))
               
        dec3 = self.upconv3(bottleneck) # up sample
        dec3 = torch.cat((enc3, dec3), dim=1) # skip conection
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3) # up sample
        dec2 = torch.cat((enc2, dec2), dim=1) # skip conection
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2) # up sample
        dec1 = torch.cat((enc1, dec1), dim=1) # skip conection
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

    def GetName(self):
        return "Unet3D"
