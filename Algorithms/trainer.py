
import torch
import datetime
import os
import Algorithms.losses as crit
import torchio as tio
import gc
import csv
import config

# Ze wzgledu na korzytsanie z overlaptilestrategy batch_size musi byc 1
class UnetTrainer:
    def __init__(self,model,train_dataset,val_dataset,classes_number,batch_size = 1, learning_rate = 1e-4, num_epochs = 50, device = 'cuda',model_path = None):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = crit.MultiClassDiceLoss(classes_number)
        self.best_loss = float('inf')
        self.start_epoch = 0

        if model_path is not None:
            self.load_checkpoint(model_path)
        


    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):

            print(f"\n📦 Batch {batch_idx + 1}/{len(self.train_loader)}")
            images = batch['image'][tio.DATA].to(self.device)
            labels = batch['mask'][tio.DATA].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            torch.cuda.synchronize()
            running_loss += loss.item()
            loss.backward()

            self.optimizer.step()
            # Zabezpieczenie: opcjonalnie czyszczenie cache
            del images, labels, outputs, loss
            torch.cuda.empty_cache()

            gc.collect()
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

                # Forward pass
                val_outputs = self.model(images)
                
                # Calculate validation loss
                val_loss = self.criterion(val_outputs, labels)
                running_loss += val_loss.item()

                del images, labels, val_outputs, val_loss
                torch.cuda.empty_cache()
                gc.collect()

        avg_val_loss = running_loss / len(self.val_loader)
        return avg_val_loss

    def train(self):


        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_folder = os.path.join(config.MODEL_PATH, f"experiment_{timestamp}")

        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)  

        log_file_path = os.path.join(experiment_folder, 'training_log.csv')

        with open(log_file_path,mode='w',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'training_loss', 'validation_loss'])
                
            for epoch in range(self.start_epoch,self.num_epochs):
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
                    model_filename = f"unet3d_model_{epoch+1}_{timestamp}.pth"
                    model_file_path = os.path.join(experiment_folder, model_filename)
                    torch.save(self.model.state_dict(), model_file_path)
                    print(f"Model saved as {model_filename}")

                writer.writerow([epoch+1,train_loss,val_loss])



    def save_checkpoint(self,epoch,folder_path,name):
        model_file_path = os.path.join(folder_path, name)
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint,model_file_path)
        print(f"Model saved as {name}")

    def load_checkpoint(self,path):
        checkpoint = torch.load(path,map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch',0)
        print(f"Model loaded successfully from {path}")


    def load_model(self, model_path):
        # Load the model weights
        self.model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully from", model_path)

