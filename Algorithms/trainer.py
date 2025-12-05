
import torch
import datetime
import os
import Algorithms.losses as crit
import torchio as tio
import gc
import csv
from tqdm import tqdm

# Ze wzgledu na korzytsanie z overlaptilestrategy batch_size musi byc 1
class UnetTrainer:
    def __init__(self,model,train_dataset,val_dataset,classes_number, model_save_dir, batch_size = 1, learning_rate = 1e-4, num_epochs = 50, device = 'cuda',model_path = None):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        weights = torch.tensor([0.1] + [1.0]*(classes_number-1), device='cuda')
        self.criterion = crit.CombinedLoss(classes_number,class_weights=weights) #crit.MultiClassDiceLoss(classes_number)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=10)
        self.start_epoch = 0
        self.batch_size = batch_size

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.experiment_folder = model_save_dir / f"experiment_{timestamp}"

        if model_path is not None:
            self.load_checkpoint(model_path)
            self.experiment_folder = model_path.parent

    def train_epoch(self):
        '''
        Metoda wykonująca pojedynczą epokę uczenia modelu.
        '''

        self.model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader,desc='Training - batch')):

            images = batch['image'][tio.DATA].to(self.device)
            labels = batch['mask'][tio.DATA].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            if(self.model.deep_supervision == True):
                loss = (
                    0.2 * self.criterion(outputs[0], labels) +
                    0.3 * self.criterion(outputs[1], labels) +
                    0.5 * self.criterion(outputs[2], labels)
                )
            else:
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
        '''
        Główna metoda wykonująca walidacje modelu.
        '''
        self.model.eval() 
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader,desc='Validation - batch')):

                images = batch['image'][tio.DATA].to(self.device, non_blocking=True)
                labels = batch['mask'][tio.DATA].to(self.device, non_blocking=True)

                val_outputs = self.model(images)
                
                if(self.model.deep_supervision == True):
                    val_loss = self.criterion(val_outputs[-1], labels)
                else:
                    val_loss = self.criterion(val_outputs, labels)
                running_loss += val_loss.item()

                del images, labels, val_outputs, val_loss
                torch.cuda.empty_cache()
                gc.collect()

        avg_val_loss = running_loss / len(self.val_loader)
        return avg_val_loss

    def train(self):
        '''
        Główna metoda uczenia zawierająca pełną pętle wraz z zapisywaniem checkpointów modelu.
        '''

        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)  

        # ścieżka do pliku zawierającego wszystkie wartości funkcji straty epok
        log_file_path = os.path.join(self.experiment_folder, 'training_log.csv') 

        for epoch in range(self.start_epoch,self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}:")

            #training
            train_loss= self.train_epoch()
            print(f"Training loss: {train_loss}")

            # validation
            val_loss = self.validate()
            print(f"Validation loss: {val_loss}")
            self.scheduler.step(val_loss)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            model_filename = f"{self.model.GetName()}_model_{epoch+1}_{timestamp}.pth"
            self.save_checkpoint(epoch+1,self.experiment_folder,model_filename,train_loss,val_loss)

            self.log_epoch_results(log_file_path,epoch+1,train_loss,val_loss)



    def save_checkpoint(self,epoch,folder_path,name,train_loss,val_loss):
        '''
        Metoda zapisująca model jako checkpoint.
        '''
        model_file_path = os.path.join(folder_path, name)
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint,model_file_path)
        print(f"Model zapisany jako: {name}")

    def load_checkpoint(self,path):
        '''
        Metoda wczytująca dany checkpoint.
        '''
        checkpoint = torch.load(path,map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch',0)
        self.start_epoch = self.start_epoch -1
        print(f"Epoka startowa: {self.start_epoch}")
        print(f"Model zaladowany z {path}")




    def log_epoch_results(self,file_path,epoch,train_loss,val_loss):
        '''
        Funkcja dopisująca na końcu pliku uzyskane wartości funkcji straty dla danej epoki.
        '''
        if not os.path.exists(file_path):
            with open(file_path,mode='w',newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch','training_loss','validation_loss'])
        
        with open(file_path,mode='a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch,train_loss,val_loss])

