import torch
import torch.nn as nn
import torchio as tio
import config
from torch.utils.data import DataLoader
from Algorithms.Unet3D.unet3D import UNet3D
from Algorithms.UnetPP3D.unetPlusPlus3D import UNetPP3D
from Algorithms.inference import UnetInference
from pathlib import Path




def load_model(checkpoint_path,device):

    checkpoint = torch.load(checkpoint_path,map_location=device)
    model = UNet3D(in_channels=1, out_channels=config.CLASS_NUMBER)
    #model = UNetPP3D(in_channels=1,out_channels=config.CLASS_NUMBER,deep_supervision=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Załaodwano state_dict do modelu")

    return model


def prepare_paths(input_folder_path,output_folder_path,extension):

    input_dir = Path(input_folder_path)
    output_dir = Path(output_folder_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []

    for file_path in input_dir.glob('*'+extension):
        splited_name = file_path.name.split('.')

        output_name = splited_name[0] + "_prediction" + extension
        output_path = output_dir / output_name
        pairs.append((file_path,output_path))

    return pairs

    



if __name__ == "__main__":
    print("--- Uruchomienie skryptu inferencji ---")

    print("--- Ładowanie modelu ---")
    model_path = "Models\\Unet3D\\experiment_2025-04-30_07-35-51\\Unet3D_model_100_2025-05-01_10-15-46.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path,device=device)

    print("--- Przygotowanie danych testowych ---")
    output_dir = "Results\\CleanToothFairy_Unet3D96_mainclasses"
    pairs = prepare_paths(config.TEST_IMG_PATH,output_dir,config.FILE_FORMAT)

    print("--- Tworzenie klasy inferencji ---")
    infer = UnetInference(model,device,config.PATCH_SIZE,config.PATCH_OVERLAP,1)

    #print(pairs)
    for pair in pairs:
        print(pair)
        infer.predict_and_save(pair[0],pair[1],True)

    # output_path = "1001463689_20200506_result.nii.gz"
    # infer.predict_and_save(image_path,output_path,True)

