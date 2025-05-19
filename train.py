import torch
from torch.utils.data import DataLoader
import torchio as tio
import glob
from pathlib import Path
from Algorithms.Unet3D.unet3D import UNet3D
from Algorithms.UnetPP3D.unetPlusPlus3D import UNetPP3D
from Algorithms.trainer import UnetTrainer
import numpy as np
import config
import gc
import random


# === Tworzenie Subjectów ===
def create_subjects_by_exact_filename(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    # Mapa: nazwa pliku -> pełna ścieżka do maski
    mask_map = {mask_path.name: mask_path for mask_path in mask_dir.glob('*' + config.FILE_FORMAT)}

    subjects = []
    for image_path in image_dir.glob('*'+ config.FILE_FORMAT):
        image_filename = image_path.stem[:-5] + config.FILE_FORMAT
        if image_filename in mask_map:
            mask_path = mask_map[image_filename]
            subject = tio.Subject(
                image=tio.ScalarImage(str(image_path)),
                mask=tio.LabelMap(str(mask_path)),
            )
            subjects.append(subject)
        else:
            print(f"⚠️ Brak pasującej maski dla: {image_filename}")

    return subjects


def get_kfold_split(data,k,index):
    if not 0 <= index < k:
        raise ValueError("Nie można pobrac kfolda")
    
    size = len(data) // k   

    start = index * size + min(index,0)
    end = start + size

    val_data = data[start:end]
    train_data = data[:start] + data[end:]

    return train_data, val_data


def print_configuration():
    print("🔧 CONFIGURATION:")
    print(f" - PATCH_SIZE:     {config.PATCH_SIZE}")
    print(f" - BATCH_SIZE:     {config.BATCH_SIZE}")
    print(f" - EPOCH_COUNT:    {config.EPOCH_COUNT}")
    print(f" - CLASS_NUMBER:   {config.CLASS_NUMBER}")
    print(f" - LEARNING_RATE:  {config.LEARNING_RATE}")
    print(f" - RANDOM_STATE:   {config.RANDOM_STATE}")


if __name__ == "__main__":

    # pobranie 
    # image_paths = sorted(glob.glob("Data/ChinaCBCT/img/*.nii.gz")) 
    # mask_paths = sorted(glob.glob("Data/ChinaCBCT/label/*.nii.gz")) 
    print_configuration()

    #subject_list = create_subjects_by_exact_filename(config.IMG_PATH,config.LABEL_PATH)

    # podział danych na walidacje oraz treningowe
    train_subjects = create_subjects_by_exact_filename(config.TRAIN_IMG_PATH,config.TRAIN_LABEL_PATH)
    #val_subjects = create_subjects_by_exact_filename(config.VAL_IMG_PATH,config.VAL_LABEL_PATH)
    train_subjects = train_subjects[:150]
    
    random.seed(config.RANDOM_STATE)
    random.shuffle(train_subjects)

    # Podzielenie na k-fold
    train_subjects,val_subjects = get_kfold_split(train_subjects,config.K_FOLD,4)

    #train_subjects = train_subjects[:30]
    #val_subjects = val_subjects[:6]
    for subject in val_subjects:
        image_path = subject['image'].path
        mask_path = subject['mask'].path
        print((image_path, mask_path))



    paths = [str(s['image'].path) for s in train_subjects]
    #print(paths)


    # Transformacje wykonywane na danych treninigowych
    training_transform = tio.Compose([
        tio.RandomNoise(std=0.01),
        tio.RandomAffine(scales=0.1, degrees=10),
    ])

    #print(subject_list)
    print(f"Liczebność zbioru treningowego: {len(train_subjects)}")
    print(f"Liczebność zbioru treningowego: {len(val_subjects)}")

    background_weight = 0.2
    tooth_weight =  (1.0 - background_weight) / (config.CLASS_NUMBER-1)

    class_propabilities = {0: background_weight}
    for i in range(1, config.CLASS_NUMBER):
        class_propabilities[i] = tooth_weight

    print(class_propabilities)

    train_dataset = tio.SubjectsDataset(train_subjects, transform=training_transform)

    # Kolejka treningowa
    train_queue = tio.Queue(
        train_dataset,
        config.QUE_MAX_LENGTH_TRAIN,
        config.QUE_SAMPLES_PER_VOLUME_TRAIN,
        sampler=tio.data.LabelSampler(config.PATCH_SIZE,label_name='mask',label_probabilities=class_propabilities),
        num_workers=config.NUM_WORKERS_TRAIN,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    train_loader = DataLoader(train_queue,batch_size=config.BATCH_SIZE)

    print("--- Tworzenie modelu Unet3D ---")
    #model = UNet3D(in_channels=1, out_channels=config.CLASS_NUMBER)
    model = UNetPP3D(in_channels=1,out_channels=config.CLASS_NUMBER,deep_supervision=True)


    print("--- Tworzenie walidacyjnego datasetu---")
    val_dataset = tio.SubjectsDataset(val_subjects)
    val_queue = tio.Queue(
        val_dataset,
        config.QUE_MAX_LENGTH_VAL,
        config.QUE_SAMPLES_PER_VOLUME_VAL   ,
        sampler=tio.data.UniformSampler(config.PATCH_SIZE),
        num_workers=config.NUM_WORKERS_VAL,
        shuffle_subjects=True,
        shuffle_patches=True
    )
    val_loader = DataLoader(val_queue,batch_size=1,shuffle=False)


    print("--- Tworzenie trainera ---")

    print(torch.cuda.is_available())

    trainer = UnetTrainer(
        model = model,
        train_dataset=train_loader,
        val_dataset=val_loader,
        classes_number=config.CLASS_NUMBER,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_epochs=config.EPOCH_COUNT,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_path="Models\\Unet3D\\experiment_2025-05-18_20-36-43\\UnetPP3D_model_36_2025-05-19_12-23-33.pth"
    )

    print("\n--- Trenowanie---")
    trainer.train()
    # for batch in train_loader:
    #     print("Image shape:", batch['image'][tio.DATA].shape)
    #     print("Mask shape:", batch['mask'][tio.DATA].shape)
        
    pass