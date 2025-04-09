import torch
from torch.utils.data import DataLoader
import torchio as tio
import glob
from pathlib import Path


# === Parametry ===
PATCH_SIZE = (160, 160, 160)
BATCH_SIZE = 1


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

    # Transformacje wykonywane na danych treninigowych
    training_transform = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomNoise(std=0.01),
        tio.RandomAffine(scales=0.1, degrees=10),
    ])


    subject_list = create_subjects_by_exact_filename("Data/ChinaCBCT/img","Data/ChinaCBCT/label")

    # podział danych na walidacje oraz treningowe
    train_subjects = subject_list[:8]
    val_subjects = subject_list[8:10]

    #print(subject_list)
    print(len(subject_list))
    print(f"Liczebność zbioru treningowego: {len(train_subjects)}")
    print(f"Liczebność zbioru treningowego: {len(val_subjects)}")

    train_dataset = tio.SubjectsDataset(train_subjects, transform=training_transform)

    # Kolejka treningowa
    train_queue = tio.Queue(
        train_dataset,
        300,
        32,
        sampler=tio.data.LabelSampler(PATCH_SIZE),
        num_workers=2,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    train_loader = DataLoader(train_queue,batch_size=BATCH_SIZE)

    print("\n--- Treningowy batch ---")
    for batch in train_loader:
        print("Image shape:", batch['image'][tio.DATA].shape)
        print("Mask shape:", batch['mask'][tio.DATA].shape)
        



    pass