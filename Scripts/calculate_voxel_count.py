import argparse
from pathlib import Path
from collections import defaultdict
import SimpleITK as sitk
import pandas as pd
import numpy as np
from types import SimpleNamespace
import script_config

ARG_TO_CONFIG_KEYS = {
    "extension": "CALC_VOXEL_COUNT_EXTENSION",
    "excel_name": "CALC_VOXEL_COUNT_EXCEL_NAME",
    "data_folder": "CALC_VOXEL_COUNT_DATA_FOLDER",
    "print_results": "CALC_VOXEL_COUNT_PRINT",
}


def compute_voxel_per_label(image: sitk.Image) -> dict:
    """
    Metoda zwracająca słownik z liczbą voxeli w danej masce przypisanych do
    danej klasy segmentacyjnej.

    """
    arr = sitk.GetArrayFromImage(image)
    labels, counts = np.unique(arr, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))

def prepare_path(folder_path : Path, file_format : str):
    '''
    Metoda wyszukujacą wszytskie pliki o określonym formacie z przekazanego folderu.
    '''
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Podany folder nie istnieje: {folder_path}")
    
    paths = list(folder_path.glob(f"*{file_format}"))
    return paths

def calculate_volume(voxel_count : int, spacing : tuple[float,float,float]):
    '''
    Metoda wyliczająca rzeczywista objetosc danej klasy segmentacyjnej wykorzystujacspacing obrazu.
    '''
    volume = voxel_count * spacing[0] * spacing[1] * spacing[2]
    return round(volume, 3)

def parse_arguments():
    '''
    Metoda wczytująca parametry wejściowe
    '''


    parser = argparse.ArgumentParser(
        description="Skrypt obliczający obnjętości etykiet na podstawie plików z maskami segmentacji. "
        "Wynik zapisywany jest jako plik Excel."
    )

    parser.add_argument(
        "--data_folder",
        "-d",
        required=False,
        type=Path,
        help="Ścieżka do folderu z plikami masek segmentacyjnych",
    )
    parser.add_argument(
        "--extension",
        "-e",
        required=False,
        help="Rozszerzenie plików, np. .mha, .nii.gz",
    )
    parser.add_argument(
        "--excel_name",
        "-o",
        required=False,
        help="Nazwa pliku wyjściowego Excel",
    )

    parser.add_argument(
        "--print_results",
        "-p",
        type=lambda x: str(x).lower() == "true",
        help="Jeśli podane, wyniki będą wypisywane w konsoli.",
    )

    args = parser.parse_args()

    return args

def print_configuration(configuration : SimpleNamespace):
    '''
    Metoda wypisująca wykorzystywaną konfiguracje
    '''
    print("------ Wczytana konfiguracja ------")
    for key, value in vars(configuration).items():
        print(f"{key}: {value}")
    print("-----------------------------------")


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
        CALC_VOXEL_COUNT_EXTENSION = script_config.CALC_VOXEL_COUNT_EXTENSION,
        CALC_VOXEL_COUNT_EXCEL_NAME = script_config.CALC_VOXEL_COUNT_EXCEL_NAME,
        CALC_VOXEL_COUNT_DATA_FOLDER = script_config.CALC_VOXEL_COUNT_DATA_FOLDER,
        CALC_VOXEL_COUNT_PRINT = script_config.CALC_VOXEL_COUNT_PRINT
    )
    return configuration

if __name__ == "__main__":

    print("--- Uruchomienie skryptu do liczenia średnich objetości wokseli ---")

    args = parse_arguments()

    configuration = load_configuration()
    configuration = override_configuration(configuration, args)

    print_configuration(configuration)

    files = prepare_path(configuration.CALC_VOXEL_COUNT_DATA_FOLDER, configuration.CALC_VOXEL_COUNT_EXTENSION)
    print(len(files))


    total_volume_per_label = defaultdict(float)
    records = []

    
    for p in files:
        image = sitk.ReadImage(str(p))
        spacing = image.GetSpacing()  # (sx, sy, sz)
        spacing = tuple(round(s, 5) for s in spacing)

        result = compute_voxel_per_label(image)

        if configuration.CALC_VOXEL_COUNT_PRINT:
            print(f"Plik: {p.name} -> {result}")

        

        for label, voxel_count in result.items():
            size = calculate_volume(voxel_count, spacing)

            if configuration.CALC_VOXEL_COUNT_PRINT:
                print(f"Objętość dla kla klasy {label} wynosi: {size} mm3")

            records.append({
                "Nazwa pliku": p.name,
                "Etykieta": label,
                "Liczba wokseli": voxel_count,
                "Objętość (mm3) ": size,
            })
            total_volume_per_label[label] += size


    # Zapis do excela
    labels = sorted(total_volume_per_label.keys())
    volumes = [total_volume_per_label[l] for l in labels]
    print(labels)


    df = pd.DataFrame(records)
    df.to_excel(configuration.CALC_VOXEL_COUNT_EXCEL_NAME, index=False)
