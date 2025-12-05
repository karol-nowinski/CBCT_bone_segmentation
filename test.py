from types import SimpleNamespace
from pathlib import Path
import argparse
import config
from Algorithms.inference import UnetInference
import torch
from Algorithms.Unet3D.unet3D import UNet3D
from Algorithms.UnetPP3D.unetPlusPlus3D import UNetPP3D


ARG_TO_CONFIG_KEYS = {
    "extension": "FILE_FORMAT",
    "output_folder": "OUTPUT_FOLDER",
    "input_data": "INPUT_PATH",
    "mode": "MODE",
    "model_path" : "MODEL_PATH"
}

def load_model(checkpoint_path : Path, device : str, model_type : str, configuration : SimpleNamespace):
    '''
    Metoda ładująca odpowiedni model
    '''
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path,map_location=device)
    if model_type == 'UnetPP3D':
        model = UNetPP3D(in_channels=1,out_channels=configuration.CLASS_NUMBER,deep_supervision=True)
    else:
        model = UNet3D(in_channels=1, out_channels=configuration.CLASS_NUMBER)
        pass
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_all_paths(folder : Path, extension : str):
    '''
    Metoda zwracająca wszytskie pliki z wskazanym rozszerzeniem wewnątrz folderu
    '''

    if not extension.startswith('.'):
        extension = '.' + extension

    files = []

    for image_path in folder.glob('*'+extension):
        files.append(image_path)
    
    return files

def prepare_prediction_paths(source_files, output_folder_path : Path, extension : str):
    '''
    Metoda przygotowująca ścieżki inferowanych obrazów
    '''
    output_folder_path.mkdir(parents=True, exist_ok=True)
    pairs = []

    for file in source_files:
        name = file.name

        for suf in reversed(file.suffixes):
            name = name.removesuffix(suf)

        inference_file_name = name + "_prediction" + extension
        output_path = output_folder_path / inference_file_name
        pairs.append((file, output_path))

    return pairs

def load_configuration():

    configuration = SimpleNamespace(
        BASE_DIR = config.BASE_DIR,
        INPUT_PATH = config.INF_INPUT,
        OUTPUT_FOLDER = config.OUTPUT_INF_FOLDER,
        FILE_FORMAT = config.FILE_FORMAT,
        PATCH_SIZE = config.PATCH_SIZE,
        PATCH_INF_OVERLAP = config.PATCH_INF_OVERLAP,
        MODEL_NAME = config.MODEL_NAME,
        MODEL_PATH = config.MODEL_PATH,
        MODEL_TYPE = config.MODEL_TYPE,
        MODE = config.INF_MODE,
        CLASS_NUMBER = config.CLASS_NUMBER
    )
    return configuration

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

def print_configuration(configuration : SimpleNamespace):
    '''
    Metoda wypisująca wykorzystywaną konfiguracje
    '''
    print("------ Wczytana konfiguracja ------")
    for key, value in vars(configuration).items():
        print(f"{key}: {value}")
    print("-----------------------------------")
    pass


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Skrypt służący do wykonania inferencji/ predykcji przekazanych obrazów medycznych. "
        " Generuje on maski segmenatycjne zapisywane we wskazanym folderze."
    )

    parser.add_argument(
        "--model_path",
        type=Path,
        help="Ścieżka do pliku zawierającego model"
    )

    parser.add_argument(
        "--input_data",
        type=Path,
        help="Ścieżka do pliku lub folderu z którego zczytywane będą obrazy medyczne."
    )

    parser.add_argument(
        "--extension",
        help="Rozszerzenie plików obrazów medycznych"
    )

    parser.add_argument(
        "--mode",
        choices=["file", "folder"],
        help="Tryb działania definiuje czy : file lub folder"
    )

    parser.add_argument(
        "--output_folder",
        type=Path,
        help="Scieżka do folderu do którego zapisane zostaną maski segmentacyjne uzyskane z inferencji"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # ------------------------------------
    # Wczytywanie argumentów uruchomienia
    # ------------------------------------

    args = parse_arguments()

    # ------------------------------------
    # Załadowanie konfiguracji
    # ------------------------------------
    configuration = load_configuration()
    configuration = override_configuration(configuration, args)

    print_configuration(configuration)
    
    print("--- Uruchomienie skryptu inferencji ---")

    files = []
    if configuration.MODE == "file":
        files.append(configuration.INPUT_PATH)
    else:
        files = get_all_paths(configuration.INPUT_PATH, configuration.FILE_FORMAT)

    pairs = prepare_prediction_paths(files, configuration.OUTPUT_FOLDER, configuration.FILE_FORMAT)

    print("--- Wczytanie modelu---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(configuration.MODEL_PATH, device, configuration.MODEL_TYPE, configuration)

    infer = UnetInference(model, device, configuration.PATCH_SIZE, configuration.PATCH_INF_OVERLAP)


    print("--- Inferencja---")
    for pair in pairs:
        infer.predict_and_save(pair[0],pair[1],True)


