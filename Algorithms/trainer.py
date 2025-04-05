
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ..utils import DoubleConv
import datetime
import os
import Algorithms.losses as crit

# Ze wzgledu na korzytsanie z overlaptilestrategy batch_size musi byc 1
class UnetTrainer:
    def __init__(self,model,train_dataset,val_dataset,classes_number,batch_size = 1, learning_rate = 1e-4, num_epochs = 50, device = 'cuda'):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = crit.MultiClassDiceLoss(classes_number)
        self.best_loss = float('inf')
        pass


    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for image,label in self.train_loader:
            images, labels = image.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        self.model.eval() 
        running_loss = 0.0
        with torch.no_grad():  # Disable gradient computation during validation
            for val_images, val_labels in self.val_loader:
                val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                
                # Forward pass
                val_outputs = self.model(val_images)
                
                # Calculate validation loss
                val_loss = self.criterion(val_outputs, val_labels)
                running_loss += val_loss.item()

        avg_val_loss = running_loss / len(self.val_loader)
        return avg_val_loss

    def train(self):

        model_dir = 'Models/Unet3D'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)  

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}:")

            #training
            train_loss= self.train_epoch()
            print(f"Training loss: {train_loss}")

            # validation
            val_loss = self.validate()
            print(f"Validation loss: {val_loss}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print("Validation loss improved, saving model...")

                # Get current date and time to add to filename
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                model_filename = f"unet3d_model_{timestamp}.pth"
                torch.save(self.model.state_dict(), model_filename)
                print(f"Model saved as {model_filename}")



    def load_model(self, model_path):
        # Load the model weights
        self.model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully from", model_path)

