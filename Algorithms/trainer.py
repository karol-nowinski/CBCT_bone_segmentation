
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ..utils import DoubleConv
import datetime
import os
import Algorithms.losses as crit
import torchio as tio
import time

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

        print("W train_epoch")
        for batch_idx, batch in enumerate(self.train_loader):

            print(f"\n📦 Batch {batch_idx + 1}/{len(self.train_loader)}")
            images = batch['image'][tio.DATA].to(self.device)
            labels = batch['mask'][tio.DATA].to(self.device)

            print(images.shape)
            print("Optimizer")
            #start = time.time()
            self.optimizer.zero_grad()
            #print(f"⏱️ zero_grad: {time.time() - start:.3f}s")

            #start = time.time()
            outputs = self.model(images)
            #print(f"⏱️ forward pass: {time.time() - start:.3f}s")

            #start = time.time()
            loss = self.criterion(outputs, labels)
            #print(f"⏱️ criterion compute: {time.time() - start:.3f}s")
            
            #start = time.time()
            torch.cuda.synchronize()
            running_loss += loss.item()
            #print(f"⏱️ loss item compute: {time.time() - start:.3f}s")

            #start = time.time()
            loss.backward()
            #print(f"⏱️ backward pass: {time.time() - start:.3f}s")

            #start = time.time()
            self.optimizer.step()
            #print(f"⏱️ optimizer step: {time.time() - start:.3f}s")
            # Zabezpieczenie: opcjonalnie czyszczenie cache
            torch.cuda.empty_cache()

            # (Opcjonalnie) monitorowanie GPU
            print(f"🧠 GPU użycie: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        self.model.eval() 
        running_loss = 0.0
        with torch.no_grad():  # Disable gradient computation during validation
            for batch_idx, batch in enumerate(self.val_loader):
                print(f"\n🧪 Walidacja — Batch {batch_idx + 1}/{len(self.val_loader)}")

                images = batch['image'][tio.DATA].to(self.device, non_blocking=True)
                labels = batch['mask'][tio.DATA].to(self.device, non_blocking=True)

                print(images.shape)

                # Forward pass
                val_outputs = self.model(images)
                
                # Calculate validation loss
                val_loss = self.criterion(val_outputs, labels)
                running_loss += val_loss.item()
                torch.cuda.empty_cache()
                print(f"🧠 GPU użycie: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"📉 Walidacja - Loss: {val_loss.item():.4f}")

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

