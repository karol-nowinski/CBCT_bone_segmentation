import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """
    A module bloc to perform two consecutive 3D convolution operations.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.block(x)
    


class UNetPP3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32,deep_supervision = False):
        super(UNetPP3D, self).__init__()
        
        self.deep_supervision = deep_supervision

        # encoder layers
        self.enc0_0 = DoubleConv3D(in_channels, base_channels)
        self.enc1_0 = DoubleConv3D(base_channels, base_channels * 2)
        self.enc2_0 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.enc3_0 = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.enc4_0 = DoubleConv3D(base_channels * 8, base_channels * 16)
        
        # max pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        


        # Upsampling layers
        self.up4_0 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.up3_0 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up3_1 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up2_0 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up2_2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up1_0 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up1_1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up1_2 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up1_3 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)


        # decoder layers (containgn nested)
        self.dec3_1 = DoubleConv3D(base_channels * 16, base_channels * 8)
        self.dec2_1 = DoubleConv3D(base_channels * 8, base_channels * 4)
        self.dec2_2 = DoubleConv3D(base_channels * 12, base_channels * 4)
        self.dec1_1 = DoubleConv3D(base_channels * 4, base_channels * 2)
        self.dec1_2 = DoubleConv3D(base_channels * 6, base_channels * 2)
        self.dec1_3 = DoubleConv3D(base_channels * 8, base_channels * 2)
        self.dec1_4 = DoubleConv3D(base_channels * 10, base_channels * 2)
        self.dec0_1 = DoubleConv3D(base_channels * 2, base_channels)
        self.dec0_2 = DoubleConv3D(base_channels * 3, base_channels)
        self.dec0_3 = DoubleConv3D(base_channels * 4, base_channels)
        self.dec0_4 = DoubleConv3D(base_channels * 5, base_channels)


        # Final output
        if deep_supervision:
            self.final1 = nn.Conv3d(base_channels, out_channels, kernel_size=1)
            self.final2 = nn.Conv3d(base_channels, out_channels, kernel_size=1)
            self.final3 = nn.Conv3d(base_channels, out_channels, kernel_size=1)
            self.final4 = nn.Conv3d(base_channels,out_channels, kernel_size=1)
        else:
            self.final = nn.Conv3d(base_channels,out_channels,kernel_size=1)


    def forward(self,x):

        x0_0 = self.enc0_0(x)
        x1_0 = self.enc1_0(self.pool(x0_0))
        x2_0 = self.enc2_0(self.pool(x1_0))
        x3_0 = self.enc3_0(self.pool(x2_0))
        x4_0 = self.enc4_0(self.pool(x3_0))

        x3_1 = self.dec3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))

        x2_1 = self.dec2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x2_2 = self.dec2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))

        x1_1 = self.dec1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x1_2 = self.dec1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x1_3 = self.dec1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))

        x0_1 = self.dec0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x0_2 = self.dec0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))
        x0_3 = self.dec0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))
        x0_4 = self.dec0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))


        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final(x0_4)
        
    def GetName(self):
        return "UnetPP3D"
