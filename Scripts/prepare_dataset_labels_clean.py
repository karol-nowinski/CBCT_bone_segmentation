import argparse
from pathlib import Path
from types import SimpleNamespace
import script_config
import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk

ARG_TO_CONFIG_KEYS = {
    "dataset": "PREP_DATASET_LABELS_CLEAN_DATASET",
    "output_folder": "PREP_DATASET_LABELS_CLEAN_OUTPUT_FOLDER",
    "input_folder": "PREP_DATASET_LABELS_CLEAN_INPUT_FOLDER",
}

# Labele wykorzystywane w ToothFairy2
original_tooth_labels = (
        [8,9,10] +
        list(range(11, 19)) +  # Górne prawe
        list(range(21, 29)) +  # Górne lewe
        list(range(31, 39)) +  # Dolne lewe
        list(range(41, 49))    # Dolne prawe
    )

def prepare_chinacbct_clean(input_folder : Path, output_folder : Path):
    '''
    Metoda przygotowująca maski segmentacyjne dla ChinaCBCTClean.
    Ustawia wszystkie etykiety/zęby w jedną klasę. Przyjmuje że format danych jest NIFti
    '''

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(f".nii.gz"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            img = nib.load(input_path)
            data = img.get_fdata()
            binary_data = (data > 0).astype(np.uint8)
            binary_img = nib.Nifti1Image(binary_data, img.affine, img.header)
            nib.save(binary_img, output_path)

            print(f"Przetworzono: {filename}")


def prepare_toothfairy_clean(input_folder : Path, output_folder : Path):
    '''
    Metoda przygotowująca maski segmentacyjne dla ToothFairy2.
    Scala wszytskie zęby w pojedynczą klasę.
    '''
    
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(f".mha"):

            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = sitk.ReadImage(input_path)
            data = sitk.GetArrayFromImage(img)

            # Zgodnie z zbiorem danych, każda etykieta większa od 8 odpowiada zębom
            new_data = np.where(data >=8,8,data).astype(np.uint8)

            binary_img = sitk.GetImageFromArray(new_data)
            binary_img.CopyInformation(img)

            sitk.WriteImage(binary_img, output_path,useCompression=True)

            print(f"Przetworzono: {filename}")

    pass

def override_configuration(configuration : SimpleNamespace, args):
    '''
    Metoda nadpisująca konfiguracje na podstawie argumentów uruchomienia. W przypadku braku nadpisania
    pobierana jest domyślna wartość z pliku schema_config.py
    '''

    args_dict = vars(args)

    for arg_name, arg_value in args_dict.items():
        if arg_value is None:
            continue 

        if arg_name in ARG_TO_CONFIG_KEYS:
            config_key = ARG_TO_CONFIG_KEYS[arg_name]
            setattr(configuration, config_key, arg_value)

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
        description="Skrypt przygotowujący maski segmentacyjne wykorzystywane w eksperymentach. "
        "Modyfikuje wartości etykiet przypisane do danego wokselu (np. scalanie w jedną klasę wszytskich zębów)."
    )


    parser.add_argument(
        "--input_folder",
        "-i",
        required=False,
        type=Path,
        help="Ścieżka do folderu z plikami masek segmentacyjnych.",
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        required=False,
        type=Path,
        help="Ścieżka do folderu do któego zapisane zostaną zmodyfikowane maski segmentacyjne.",
    )

    parser.add_argument(
        "--extension",
        "-e",
        required=False,
        type=str,
        help="Rozszerzenie wyszukiwanych plików zawierających maski segmentacyjne.",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        required=False,
        type=str,
        choices=["ChinaCBCT", "ToothFairy2"],
        help="Nazwa wykorzystywanego zbioru danych (ChinaCBCT lub ToothFairy2)",
    )

    args = parser.parse_args()

    return args

def load_configuration():
    configuration = SimpleNamespace(
        PREP_DATASET_LABELS_CLEAN_EXTENSION = script_config.PREP_DATASET_LABELS_CLEAN_EXTENSION,
        PREP_DATASET_LABELS_CLEAN_OUTPUT_FOLDER = script_config.PREP_DATASET_LABELS_CLEAN_OUTPUT_FOLDER,
        PREP_DATASET_LABELS_CLEAN_INPUT_FOLDER = script_config.PREP_DATASET_LABELS_CLEAN_INPUT_FOLDER,
        PREP_DATASET_LABELS_CLEAN_DATASET = script_config.PREP_DATASET_LABELS_CLEAN_DATASET
    )
    return configuration

if __name__ == "__main__":

    print("--- Uruchomienie skryptu do przygotowywanie masek segmentacyjnych ---")

    args = parse_arguments()

    configuration = load_configuration()
    configuration = override_configuration(configuration, args)

    print_configuration(configuration)

    if configuration.PREP_DATASET_LABELS_CLEAN_DATASET == "ChinaCBCT":
        prepare_chinacbct_clean(configuration.PREP_DATASET_LABELS_CLEAN_INPUT_FOLDER, configuration.PREP_DATASET_LABELS_CLEAN_OUTPUT_FOLDER)
    else:
        prepare_toothfairy_clean(configuration.PREP_DATASET_LABELS_CLEAN_INPUT_FOLDER, configuration.PREP_DATASET_LABELS_CLEAN_OUTPUT_FOLDER)
    pass