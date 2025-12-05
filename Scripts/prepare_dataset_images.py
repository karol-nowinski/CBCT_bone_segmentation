import os
import numpy as np
import argparse
from pathlib import Path
from types import SimpleNamespace
import torchio as tio
import script_config

ARG_TO_CONFIG_KEYS = {
    "output_folder": "PREP_DATASET_IMAGE_OUTPUT_FOLDER",
    "input_folder": "PREP_DATASET_IMAGE_INPUT_FOLDER",
    "extension": "PREP_DATASET_IMAGE_EXTENSION",
    "histogram": "PREP_DATASET_IMAGE_HISTOGRAM_FILE"
}


def get_mask_filename(dataset_name : str, image_path : Path, extension : str):

    if dataset_name == "ToothFairy2":
        return image_path.stem[:-4] + extension
    
    return image_path.stem + extension


def get_image_subjects(folder_path : Path, extension : str):
    '''
    Metoda wyszukująca wszytskie pliki o danym rozszerzeniu w folderze.
    '''
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Podany folder nie istnieje: {folder_path}")
    
    paths = list(folder_path.glob(f"*{extension}"))

    subjects = []
    for img_path in paths:
        subject = tio.Subject(
            image=tio.ScalarImage(str(img_path))
        )
        subjects.append(subject)

    return subjects




# def create_subjects(dataset_name : str, images_path : Path, labels_path : Path, extension : str):
#     subjects = []

#     mask_map = {mask_path.name: mask_path for mask_path in labels_path.glob('*' + extension)}

#     for image_file in os.listdir(images_path):
#         if image_file.endswith(f"extension"):
#             input_path_image = os.path.join(images_path, image_file)
#             mask_name = get_mask_filename()
#             if mask_name in mask_map:
#                 mask_path = mask_map[mask_name]
#                 subject = tio.Subject(
#                     image=tio.ScalarImage(str(image_file)),
#                     mask=tio.LabelMap(str(mask_path)),
#                 )
#                 subjects.append(subject)
#             else:
#                 print(f"⚠️ Brak pasującej maski dla: {image_file}")
#     return subjects

def override_configuration(configuration : SimpleNamespace, args):
    '''
    Metoda nadpisująca konfiguracje na podstawie argumentów uruchomienia. W przypadku braku nadpisania
    pobierana jest domyślna wartość z pliku config.py
    '''

    args_dict = vars(args)

    for arg_name, arg_value in args_dict.items():
        if arg_value is None:
            continue  # argument nie podany -> nic nie nadpisujemy

        if arg_name in ARG_TO_CONFIG_KEYS:
            config_key = ARG_TO_CONFIG_KEYS[arg_name]
            setattr(configuration, config_key, arg_value)

    return configuration

def load_configuration():
    configuration = SimpleNamespace(
        PREP_DATASET_IMAGE_EXTENSION = script_config.PREP_DATASET_IMAGE_EXTENSION,
        PREP_DATASET_IMAGE_OUTPUT_FOLDER = script_config.PREP_DATASET_IMAGE_OUTPUT_FOLDER,
        PREP_DATASET_IMAGE_INPUT_FOLDER = script_config.PREP_DATASET_IMAGE_INPUT_FOLDER,
        PREP_DATASET_IMAGE_HISTOGRAM_FILE = script_config.PREP_DATASET_IMAGE_HISTOGRAM_FILE
    )
    return configuration

def print_configuration(configuration : SimpleNamespace):
    '''
    Metoda wypisująca wykorzystywaną konfiguracje
    '''
    print("------ Wczytana konfiguracja ------")
    for key, value in vars(configuration).items():
        print(f"{key}: {value}")
    print("-----------------------------------")


def parse_arguments():
    '''
    Metoda wczytująca parametry wejściowe
    '''
    parser = argparse.ArgumentParser(
        description="Skrypt przygotowujący dane do trenowania modeli 3D. "
        "Łączy obrazy z odpowiadającymi im maskami, obraz "
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        required=False,
        type=Path,
        help="Ścieżka do folderu wewnątrz którego zapisane przetworzone obrazy medyczne.",
    )

    parser.add_argument(
        "--input_images",
        "-i",
        required=False,
        type=Path,
        help="Ścieżka do folderu zawierającego obrazy medyczne.",
    )

    parser.add_argument(
        "--histogram",
        "-H",
        required=False,
        type=Path,
        help="Ścieżka do pliku .npy zawierającego wytrenowane parametry histogram standardization.",
    )

    parser.add_argument(
        "--extension",
        "-e",
        required=False,
        type=str,
        help="Rozszerzenie wyszukiwanych plików obrazów medycznych",
    )


    args = parser.parse_args()

    return args

if __name__ == "__main__":

    print("--- Uruchomienie skryptu do przygotowywanie wejściowych obrazów medycznych ---")

    args = parse_arguments()
    configuration = load_configuration()
    configuration = override_configuration(configuration, args)
    print_configuration(configuration)

    # 
    subjects = get_image_subjects(configuration.PREP_DATASET_IMAGE_INPUT_FOLDER, configuration.PREP_DATASET_IMAGE_EXTENSION)

    if not configuration.PREP_DATASET_IMAGE_HISTOGRAM_FILE.exists():
        print("Generowanie pliku .npy")
        landmarks = tio.HistogramStandardization.train(
            [s['image'].path for s in subjects],
            output_path=configuration.PREP_DATASET_IMAGE_HISTOGRAM_FILE)
    else:
        print(f"Wczytywanie pliku .npy {configuration.PREP_DATASET_IMAGE_HISTOGRAM_FILE}")
        landmarks = np.load(configuration.PREP_DATASET_IMAGE_HISTOGRAM_FILE)

    print(landmarks)
    landmarks_dict = {'image': landmarks}


    transformations = tio.Compose([
        tio.ToCanonical(),
        tio.HistogramStandardization(landmarks_dict),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    ])

    os.makedirs(configuration.PREP_DATASET_IMAGE_OUTPUT_FOLDER, exist_ok=True)

    for sb in subjects:

        s_transformed = transformations(sb)
        new_img_path = configuration.PREP_DATASET_IMAGE_OUTPUT_FOLDER / sb['image'].path.name

        s_transformed['image'].save(new_img_path)

        print(f"Plik { sb['image'].path.name} został przetworzony.")

    pass