import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """
    A module to perform two consecutive 3D convolution operations.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    


class UNetPlusPlus3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(UNetPlusPlus3D, self).__init__()
        
        self.encoder1 = DoubleConv3D(in_channels, base_channels)
        self.encoder2 = DoubleConv3D(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.encoder4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)
        
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv3D(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

        # Nested skip connections
        self.nested_skip_1_1 = DoubleConv3D(base_channels, base_channels)
        self.nested_skip_2_1 = DoubleConv3D(base_channels * 2, base_channels)
        self.nested_skip_3_1 = DoubleConv3D(base_channels * 4, base_channels)
        self.nested_skip_4_1 = DoubleConv3D(base_channels * 8, base_channels)

        self.nested_skip_2_2 = DoubleConv3D(base_channels * 3, base_channels)
        self.nested_skip_3_2 = DoubleConv3D(base_channels * 5, base_channels)
        self.nested_skip_4_2 = DoubleConv3D(base_channels * 9, base_channels)

        self.nested_skip_3_3 = DoubleConv3D(base_channels * 6, base_channels)
        self.nested_skip_4_3 = DoubleConv3D(base_channels * 10, base_channels)

        self.nested_skip_4_4 = DoubleConv3D(base_channels * 11, base_channels)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder path with nested skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.decoder4(dec4)

        dec3_1 = self.upconv3(dec4)
        dec3_1 = torch.cat([enc3, dec3_1], dim=1)
        dec3_1 = self.decoder3(dec3_1)
        
        dec3_2 = torch.cat([dec3_1, dec4], dim=1)
        dec3_2 = self.nested_skip_2_2(dec3_2)

        dec2_1 = self.upconv2(dec3_1)
        dec2_1 = torch.cat([enc2, dec2_1], dim=1)
        dec2_1 = self.decoder2(dec2_1)
        
        dec2_2 = torch.cat([dec2_1, dec3_1], dim=1)
        dec2_2 = self.nested_skip_1_2(dec2_2)

        dec1_1 = self.upconv1(dec2_1)
        dec1_1 = torch.cat([enc1, dec1_1], dim=1)
        dec1_1 = self.decoder1(dec1_1)
        
        dec1_2 = torch.cat([dec1_1, dec2_1], dim=1)
        dec1_2 = self.nested_skip_1_1(dec1_2)

        return self.final_conv(dec1_1 + dec1_2)