import torch
import torch.nn as nn
import torch.nn.functional as F
import Algorithms.losses as crit
#from ..utils import DoubleConv
import datetime
import os


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


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(UNet3D, self).__init__()
        
        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.encoder4 = DoubleConv(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck) # up sample
        dec4 = torch.cat((enc4, dec4), dim=1) # skip conection
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4) # up sample
        dec3 = torch.cat((enc3, dec3), dim=1) # skip conection
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3) # up sample
        dec2 = torch.cat((enc2, dec2), dim=1) # skip conection
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2) # up sample
        dec1 = torch.cat((enc1, dec1), dim=1) # skip conection
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)




# class Unet3DTrainer:
#     def __init__(self,model,train_dataset,val_dataset,batch_size = 4, learning_rate = 1e-4, num_epochs = 50, device = 'cuda'):
#         self.device = device
#         self.model = model.to(device)
#         self.train_loader = train_dataset
#         self.val_loader = val_dataset
#         self.num_epochs = num_epochs
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.criterion = crit.DiceLoss()
#         self.best_loss = float('inf')
#         pass


#     def train_epoch(self):
#         self.model.train()
#         running_loss = 0.0

#         for image,label in self.train_loader:
#             images, labels = images.to(self.device), labels.to(self.device)
#             self.optimizer.zero_grad()
#             outputs = self.model(images)
#             loss = self.criterion(outputs, labels)
#             running_loss += loss.item()
#             loss.backward()
#             self.optimizer.step()
#         avg_loss = running_loss / len(self.train_loader)
#         return avg_loss
    
#     def validate(self):
#         self.model.eval() 
#         running_loss = 0.0
#         with torch.no_grad():  # Disable gradient computation during validation
#             for val_images, val_labels in self.val_loader:
#                 val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                
#                 # Forward pass
#                 val_outputs = self.model(val_images)
                
#                 # Calculate validation loss
#                 val_loss = self.criterion(val_outputs, val_labels)
#                 running_loss += val_loss.item()

#         avg_val_loss = running_loss / len(self.val_loader)
#         return avg_val_loss

#     def train(self):

#         model_dir = 'Models/Unet3D'
#         if not os.path.exists(model_dir):
#             os.makedirs(model_dir)  

#         for epoch in range(self.num_epochs):
#             print(f"Epoch {epoch + 1}/{self.num_epochs}:")

#             #training
#             train_loss= self.train_epoch()
#             print(f"Training loss: {train_loss}")

#             # validation
#             val_loss = self.validate()
#             print(f"Validation loss: {val_loss}")

#             if val_loss < self.best_loss:
#                 self.best_loss = val_loss
#                 print("Validation loss improved, saving model...")

#                 # Get current date and time to add to filename
#                 timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#                 model_filename = f"unet3d_model_{timestamp}.pth"
#                 torch.save(self.model.state_dict(), model_filename)
#                 print(f"Model saved as {model_filename}")



#     def load_model(self, model_path):
#         # Load the model weights
#         self.model.load_state_dict(torch.load(model_path))
#         print("Model loaded successfully from", model_path)


# Example usage
# if __name__ == "__main__":
#     print("Running test example Unet3D network")
#     model = UNet3D(in_channels=1, out_channels=1, base_channels=32)
#     x = torch.randn(1, 1, 64, 64, 64)  # Batch size = 1, Depth = 64, Height = 64, Width = 64
#     output = model(x)
#     print("Output shape:", output.shape)