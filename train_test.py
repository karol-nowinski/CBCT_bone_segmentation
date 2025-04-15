import torch
from torch.utils.data import DataLoader
import torchio as tio
import glob
from pathlib import Path
from Algorithms.Unet3D.unet3D import UNet3D
from Algorithms.trainer import UnetTrainer
import numpy as np

# === Parametry ===
PATCH_SIZE = (128, 128, 128)
BATCH_SIZE = 1
histogram_landmarks_path = "landmarks.npy"

# === Tworzenie Subjectów ===
def create_subjects_by_exact_filename(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    # Mapa: nazwa pliku -> pełna ścieżka do maski
    mask_map = {mask_path.name: mask_path for mask_path in mask_dir.glob('*.nii.gz')}

    subjects = []
    for image_path in image_dir.glob('*.nii.gz'):
        image_filename = image_path.name
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




if __name__ == "__main__":

    # pobranie 
    # image_paths = sorted(glob.glob("Data/ChinaCBCT/img/*.nii.gz")) 
    # mask_paths = sorted(glob.glob("Data/ChinaCBCT/label/*.nii.gz")) 


    subject_list = create_subjects_by_exact_filename("Data/ChinaCBCT/img","Data/ChinaCBCT/label")

    # podział danych na walidacje oraz treningowe
    train_subjects = subject_list[:8]
    val_subjects = subject_list[8:10]


    paths = [str(s['image'].path) for s in train_subjects]
    print(paths)

    landmarks = tio.HistogramStandardization.train(
        [s['image'].path for s in train_subjects],
        output_path=histogram_landmarks_path
        )
    print('\nTrained landmarks:', landmarks)



    landmark_dict = {'image': landmarks}
    # Transformacje wykonywane na danych treninigowych
    training_transform = tio.Compose([
        tio.ToCanonical(),
        tio.HistogramStandardization(landmark_dict),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomNoise(std=0.01),
        tio.RandomAffine(scales=0.1, degrees=10),
    ])

    val_transform = tio.Compose(
        [
            tio.ToCanonical(),
            tio.HistogramStandardization(landmark_dict),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        ]
    )

    #print(subject_list)
    print(len(subject_list))
    print(f"Liczebność zbioru treningowego: {len(train_subjects)}")
    print(f"Liczebność zbioru treningowego: {len(val_subjects)}")

    train_dataset = tio.SubjectsDataset(train_subjects, transform=training_transform)

    # Kolejka treningowa
    train_queue = tio.Queue(
        train_dataset,
        64,
        8,
        sampler=tio.data.LabelSampler(PATCH_SIZE),
        num_workers=6,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    train_loader = DataLoader(train_queue,batch_size=BATCH_SIZE)

    print("--- Tworzenie modelu Unet3D ---")
    model = UNet3D(in_channels=1, out_channels=33)

    print("--- Tworzenie walidacyjnego datasetu---")
    val_dataset = tio.SubjectsDataset(val_subjects,transform=val_transform)
    val_queue = tio.Queue(
        val_dataset,
        24,
        8,
        sampler=tio.data.LabelSampler(PATCH_SIZE),
        num_workers=2,
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
        classes_number=33,
        batch_size=1,
        learning_rate=1e-4,
        num_epochs=30,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n--- Trenowanie---")
    trainer.train()
    # for batch in train_loader:
    #     print("Image shape:", batch['image'][tio.DATA].shape)
    #     print("Mask shape:", batch['mask'][tio.DATA].shape)
        
    pass