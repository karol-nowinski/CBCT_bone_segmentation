import argparse
import numpy as np
import SimpleITK as sitk
import numpy as np 
from pathlib import Path
from types import SimpleNamespace
import script_config
import pandas as pd
from openpyxl import load_workbook

ARG_TO_CONFIG_KEYS = {
    "extension": "CALC_METRIC_EXTENSION",
    "prediction_folder": "CALC_METRIC_PREDICTION_MASK_FOLDER",
    "reference_folder": "CALC_METRIC_REFERENCE_MASK_FOLDER",
    "excel_name": "CALC_METRIC_EXCEL_NAME",
}

def compute_metrics_per_label(ground_truth, prediction, labels):
    '''
    Metoda wyliczająca podstawowe metryki 
    '''

    metrics = {}
    for label in labels:
        gt_label = sitk.GetArrayFromImage(ground_truth == label).astype(np.uint8) 
        pred_label = sitk.GetArrayFromImage(prediction == label).astype(np.uint8) 
        intersection = np.logical_and(gt_label, pred_label).sum()
        union = np.logical_or(gt_label, pred_label).sum() 
        gt_sum = gt_label.sum() 
        pred_sum = pred_label.sum() 
        total = gt_label.size 
        dice = 2 * intersection / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 1.0 
        iou = intersection / union if union > 0 else 1.0
        recall = intersection / gt_sum if gt_sum > 0 else 1.0
        accuracy = (gt_label == pred_label).sum() / total 
        metrics[label] = { 
        "Dice": dice,
        "IoU": iou,
        "Recall": recall,
        "Accuracy": accuracy
        } 
    return metrics


def get_ref_pred_pairs(referene_paths, prediction_folder_path : Path, extension : str):
    '''
    Metoda grupująca w pary ścieżki referencyjne z predykcyjnymi.
    '''
    pairs = []
    
    for ref_path in referene_paths:
        pred_path = get_prediction_path(ref_path, prediction_folder_path, extension)

        if not pred_path.exists():
            print(f" Nie znaleziono pliku predykcji: {pred_path}")
        else:
            pairs.append((ref_path, pred_path))

    return pairs

def get_reference_files(folder_path : Path, extension : str):
    '''
    Metoda znajdująca wszytskie pliki o danym rozszerzeniu w folderze
    '''
    if not extension.startswith('.'):
        extension = '.' + extension

    files = []
    for image_path in folder_path.glob('*'+extension):
        files.append(image_path)
    
    return files

def print_configuration(configuration : SimpleNamespace):
    '''
    Metoda wypisująca wykorzystywaną konfiguracje
    '''
    print("------ Wczytana konfiguracja ------")
    for key, value in vars(configuration).items():
        print(f"{key}: {value}")
    print("-----------------------------------")

def get_paths_from_folder(folder_path : Path, extension : str):
    '''
    Metoda pobierająca wszytskie ścieżki z foldera o danym rozszerzeniu
    '''
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Podany folder nie istnieje: {folder_path}")
    
    paths = list(folder_path.glob(f"*{extension}"))
    return paths


def get_prediction_path(reference_path : Path, prediction_folder_path : Path, extension : str):
    '''
    Metoda znajdująca ścieżkę maski predykcji dla danej maski referencyjnej.
    '''
    reference_file_name = reference_path.name
    base_name = reference_file_name[:-len(extension)]
    prediction_name = f"{base_name}_0000_prediction{extension}"
    return prediction_folder_path / prediction_name

def save_metrics_to_excel(results, excel_path):
    rows = []
    for file_info in results:
        gt_path = file_info["ground_truth_path"]
        pred_path = file_info["prediction_path"]
        metrics = file_info["metrics"]
        for label, values in metrics.items():
            row = { 
                "Ground Truth Path": gt_path,
                "Prediction Path": pred_path,
                "Label": label 
                }
            row.update(values)
            rows.append(row)
    df = pd.DataFrame(rows)
    try: 
        # Jeśli plik istnieje — dopisz do niego
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                writer.book = load_workbook(excel_path)
                writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
                start_row = writer.sheets['Arkusz1'].max_row
                df.to_excel(writer, index=False, header=False, startrow=start_row)
    except FileNotFoundError: 
        # Jeśli plik nie istnieje — zapisz jako nowy
        df.to_excel(excel_path, index=False)


def parse_arguments():
    '''
    Metoda wczytująca parametry wejściowe
    '''

    parser = argparse.ArgumentParser(
        description="Skrypt wyliczający podstawowe metryki segmentacji dla par masek predykcji oraz referencji. "
        "Wyliczane oraz zapisywane do excela są podstawowe metryki takie jak: Dice, IoU, Recall oraz Accuracy."
    )

    parser.add_argument(
        "--reference_folder",
        "-r",
        required=False,
        type=Path,
        help="Ścieżka do folderu z plikami referencyjnych masek segmentacyjnych.",
    )

    parser.add_argument(
        "--prediction_folder",
        "-p",
        required=False,
        type=Path,
        help="Ścieżka do folderu z plikami predykcji masek segmentacyjnych.",
    )


    parser.add_argument(
        "--extension",
        "-e",
        required=False,
        type=str,
        help="Rozszerzenie wyszukiwanych plików zawierających maski segmentacyjne lub referencyjne.",
    )

    parser.add_argument(
        "--excel_name",
        "-o",
        required=False,
        help="Nazwa pliku wyjściowego Excel",
    )


    args = parser.parse_args()
    return args

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

def load_configuration():
    configuration = SimpleNamespace(
        CALC_METRIC_EXTENSION = script_config.CALC_METRIC_EXTENSION,
        CALC_METRIC_PREDICTION_MASK_FOLDER = script_config.CALC_METRIC_PREDICTION_MASK_FOLDER,
        CALC_METRIC_REFERENCE_MASK_FOLDER = script_config.CALC_METRIC_REFERENCE_MASK_FOLDER,
        CALC_METRIC_EXCEL_NAME = script_config.CALC_METRIC_EXCEL_NAME,
        CALC_METRIC_CLASS_NUMBER = script_config.CALC_METRIC_CLASS_NUMBER
    )
    return configuration


if __name__ == "__main__":

    print(" Uruchomienie skryptu wyznaczającego metryki segmentacji")
    args = parse_arguments()

    configuration = load_configuration()
    configuration = override_configuration(configuration, args)
    print_configuration(configuration)

    reference_files = get_reference_files(configuration.CALC_METRIC_REFERENCE_MASK_FOLDER,configuration.CALC_METRIC_EXTENSION)
    print(f"Liczba znalezionych plików referencyjnych: {len(reference_files)}")

    pairs = get_ref_pred_pairs(reference_files,configuration.CALC_METRIC_PREDICTION_MASK_FOLDER, configuration.CALC_METRIC_EXTENSION)
    print(f"Liczba par referencja-predykcja: {len(pairs)}")

    results_per_file = []
    for ref,pred in pairs:
        print(f"Operowanie na pliku {ref}")

        ref_image = sitk.ReadImage(ref)
        pred_image = sitk.ReadImage(pred)

        metrics = compute_metrics_per_label(ref_image,pred_image,list(range(1,configuration.CALC_METRIC_CLASS_NUMBER)))
        print(metrics)
        results_per_file.append(
            { 
                "metrics": metrics,
                "ground_truth_path": ref,
                "prediction_path": pred
            }
        )



    save_metrics_to_excel(results_per_file,configuration.CALC_METRIC_EXCEL_NAME) 
    pass
