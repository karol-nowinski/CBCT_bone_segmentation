import torch
import torch.nn as nn
import torchio as tio
import config
from torch.utils.data import DataLoader
from Algorithms.Unet3D.unet3D import UNet3D




def load_model(checkpoint_path,device):

    checkpoint = torch.load(checkpoint_path,map_location=device)

    # print(checkpoint)
    model = UNet3D(in_channels=1, out_channels=config.CLASS_NUMBER)

    state = checkpoint
    print(state)


    model.load_state_dict(state)
    print(f"Załaodwano state_dict do modelu")

    return model


def Load_subject():
    pass



if __name__ == "__main__":
    print("--- Uruchomienie skryptu inferencji ---")



    print("--- Ładowanie modelu ---")
    model_path = r"Models\Unet3D\experiment_2025-04-18_19-10-40\unet3d_model_27_2025-04-19_00-44-21.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(model_path,device=device)

    print("--- Tworzenie klasy inferencji ---")

