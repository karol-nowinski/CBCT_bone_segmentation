import torchio as tio
from pathlib import Path
import os
import torch
import numpy as np



# configuration
RAW_IMG_FOLDER_PATH = Path('Data\ChinaCBCTClean\img')
RAW_LABEL_FOLDER_PATH = Path('Data\ChinaCBCTClean\label')

OUT_IMG_FOLDER_PATH_TRAIN = Path('Data\ChinaCBCTClean\imgPrepared\\train')
OUT_IMG_FOLDER_PATH_VAL = Path('Data\ChinaCBCTClean\imgPrepared\\validation')
OUT_IMG_FOLDER_PATH_TEST = Path('Data\ChinaCBCTClean\imgPrepared\\test')
OUT_LABEL_FOLDER_PATH_TRAIN = Path('Data\ChinaCBCTClean\labelPrepared\\train')
OUT_LABEL_FOLDER_PATH_VAL = Path('Data\ChinaCBCTClean\labelPrepared\\validation')
OUT_LABEL_FOLDER_PATH_TEST = Path('Data\ChinaCBCTClean\labelPrepared\\test')
LANDMARKS_PATH = Path('Data\ChinaCBCTClean\hs_landmarks.npy')

FILE_FORMAT = ".nii.gz"


# subject creation
def create_subjects_by_exact_filename():

    # Mapa: nazwa pliku -> pełna ścieżka do maski
    mask_map = {mask_path.name: mask_path for mask_path in RAW_LABEL_FOLDER_PATH.glob('*' + FILE_FORMAT)}

    subjects = []
    for image_path in RAW_IMG_FOLDER_PATH.glob('*'+ FILE_FORMAT):
        image_filename = image_path.name
        if image_filename in mask_map:
            mask_path = mask_map[image_filename]
            subject = tio.Subject(
                image=tio.ScalarImage(str(image_path)),
                mask=tio.LabelMap(str(mask_path)),
            )
            subjects.append(subject)
        else:
            print(f"Missing mask file for: {image_filename}")

    return subjects


if __name__ == "__main__":

    # subject creation
    subjects = create_subjects_by_exact_filename()
    # for file in subjects:
    #     print(file['image'].path)


    training_subjects = subjects[:64]
    validation_subject = subjects[64:74]
    test_subjects = subjects[74:]

    print(f"Train count: {str(len(training_subjects))}")
    print(f"Validation count: {len(validation_subject)}")
    print(f"Test count: {len(test_subjects)}")

    # landmarks training or loading
    if not LANDMARKS_PATH.exists():
        print("Training landmarks")
        landmarks = tio.HistogramStandardization.train(
            [s['image'].path for s in training_subjects],
            output_path=LANDMARKS_PATH)
    else:
        print("Loading landmarks")
        landmarks = np.load(LANDMARKS_PATH)

    print(landmarks)
    landmarks_dict = {'image': landmarks}
    print("📦 Typ wczytanych danych:", type(landmarks_dict))
    train_transformations = tio.Compose([
            tio.ToCanonical(),
            tio.HistogramStandardization(landmarks_dict),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        ])

    # training subjects
    for el in training_subjects:

        sub = train_transformations(el)

        new_img_path = OUT_IMG_FOLDER_PATH_TRAIN / el['image'].path.name
        new_label_path = OUT_LABEL_FOLDER_PATH_TRAIN / el['mask'].path.name
        
        sub['image'].save(new_img_path)
        sub['mask'].save(new_label_path)

        print(f"File { el['image'].path.name} has been transformed")

    # validation subjects
    for el in validation_subject:
        sub = train_transformations(el)

        new_img_path = OUT_IMG_FOLDER_PATH_VAL / el['image'].path.name
        new_label_path = OUT_LABEL_FOLDER_PATH_VAL / el['mask'].path.name
        
        sub['image'].save(new_img_path)
        sub['mask'].save(new_label_path)

        print(f"File { el['image'].path.name} has been transformed")


    # test subjects
    for el in test_subjects:
        sub = train_transformations(el)

        new_img_path = OUT_IMG_FOLDER_PATH_TEST / el['image'].path.name
        new_label_path = OUT_LABEL_FOLDER_PATH_TEST / el['mask'].path.name
        
        sub['image'].save(new_img_path)
        sub['mask'].save(new_label_path)

        print(f"File { el['image'].path.name} has been transformed")

    pass
