
import torch
import argparse
from pathlib import Path
import random
import config
from types import SimpleNamespace
import torchio as tio
from collections import defaultdict
import re
from torch.utils.data import DataLoader
from Algorithms.Unet3D.unet3D import UNet3D
from Algorithms.UnetPP3D.unetPlusPlus3D import UNetPP3D
from Algorithms.trainer import UnetTrainer


ARG_TO_CONFIG_KEYS = {
    "extension": "FILE_FORMAT",
    "checkpoint_folder": "OUTPUT_FOLDER",
    "input_data": "INPUT_PATH",
    "mode": "MODE",
}


def get_dict_key_name(dataset: str, name: Path):
    '''
    Metoda pobierajaca odpowiedni klucz do slownika
    '''

    if dataset == "ToothFairy2":
        st = name.stem
        return st[:-5] + name.suffix
    
    return name.name

def get_kfold_split(subjects, fold_number, index):
    '''
    Metoda dzieląca subjecty na zbiory treningowe oraz walidacyjne.
    Podział dokonywany jest zgodnie z K-Fold cross validation

    indeks - numer folda wykorzystywanego jako zbiór walidacyjny
    fold_number - liczba foldów
    subjects - lista subjectów
    '''

    size = len(subjects) // fold_number

    start = index * size + min(index,0)
    end = start + size

    val_data = subjects[start:end]
    train_data = subjects[:start] + subjects[end:]

    return train_data, val_data

def get_subjects_dictionary(subjects):
    '''
    Grupuje pliki w folderze na podstawie ID pacjenta zawartego w nazwie pliku.
    Zakłada format nazwy: ID_data.nii.gz, np. 1001382496_20180423.nii.gz 
    '''

    grouped_subjects = defaultdict(list)

    for subject in subjects:
        image_path = Path(subject['image'].path)
        match = re.match(r"(\d+)_", image_path.name)
        if match:
            patient_id = match.group(1)
            grouped_subjects[patient_id].append(subject)
        else:
            print(f" Nie udało się wyciągnąć ID pacjenta z: {image_path.name}")

    return dict(grouped_subjects)

def get_lopocv_split(subject_group_dict, val_patient_index):

    patient_ids = list(subject_group_dict.keys())
    val_group = patient_ids[val_patient_index]

    val_subjects = subject_group_dict[val_group]
    train_subjects = []

    for pid, subjects in subject_group_dict.items():
        if pid != val_group:
            train_subjects.extend(subjects)

    return train_subjects,val_subjects


def load_subjects_by_filename(image_folder_path : Path, label_folder_path : Path, configuration : SimpleNamespace):
    '''
    Metoda przygotowująca subjecty, łączy w pary obrazy z odpowiadającymi im maskami.
    '''

    mask_map = {mask_path.name: mask_path for mask_path in label_folder_path.glob('*' + configuration.FILE_FORMAT)}
    print(mask_map)
    subjects = []
    for image_path in image_folder_path.glob( "*" + configuration.FILE_FORMAT):
        image_filename = get_dict_key_name(config.DATASET_NAME,image_path)
        if image_filename in mask_map:
            mask_path = mask_map[image_filename]
            # print(f"{image_path} - {mask_path}")
            subject = tio.Subject(
                image=tio.ScalarImage(str(image_path)),
                mask=tio.LabelMap(str(mask_path)),
            )
            subjects.append(subject)
        else:
            print(f"Brak pasującej maski dla: {image_filename}")
    return subjects

def load_configuration():

    configuration = SimpleNamespace(
        #hiperparametry modelu
        RANDOM_STATE = config.RANDOM_STATE,
        LEARNING_RATE = config.LEARNING_RATE,
        EPOCH_COUNT = config.EPOCH_COUNT,
        BATCH_SIZE = config.BATCH_SIZE,
        PATCH_SIZE = config.PATCH_SIZE,
        CLASS_NUMBER = config.CLASS_NUMBER,
        MODEL_TYPE = config.MODEL_TYPE,

        # sciezki do danych/ modeli
        BASE_DIR = config.BASE_DIR,
        DATA_DIR_IMAGES = config.DATA_DIR_IMAGES,
        DATA_DIR_LABELS = config.DATA_DIR_LABELS,
        FILE_FORMAT = config.FILE_FORMAT,
        MODEL_SAVE_DIR = config.MODEL_SAVE_DIR,

        # kolejki
        NUM_WORKERS_TRAIN = config.NUM_WORKERS_TRAIN,
        QUE_MAX_LENGTH_TRAIN = config.QUE_MAX_LENGTH_TRAIN,
        QUE_SAMPLES_PER_VOLUME_TRAIN = config.QUE_SAMPLES_PER_VOLUME_TRAIN,
        NUM_WORKERS_VALIDATION = config.NUM_WORKERS_VALIDATION,
        QUE_MAX_LENGTH_VALIDATION = config.QUE_MAX_LENGTH_VALIDATION,
        QUE_SAMPLES_PER_VOLUME_VALIDATION = config.QUE_SAMPLES_PER_VOLUME_VALIDATION,

        # procedury
        PROCEDURE_MODE = config.PROCEDURE_MODE,
        K_FOLD = config.K_FOLD,
        VAL_KFOLD = config.VAL_KFOLD,
        VAL_LOPOCV = config.VAL_LOPOCV,

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
        "--procedure",
        choices=["normal", "kfold", "lopocv"],
        help="Rodzaj wykorzystywanej procedury : normal, kfold lub lopocv"
    )

    args = parser.parse_args()
    return args

def print_kfold_subject(train_subjects,val_subjects):
    print("--- Dane dotyczące procedury kfold ---")
    print("- Wypisanie subjectów treningowych -")
    print(f"Liczba subjectów: {len(train_subjects)}")
    for subject in train_subjects:
        image_path = Path(subject['image'].path).name
        print(f"{image_path}")

    print("- Wypisanie subjectów walidacyjnych -")
    print(f"Liczba subjectów: {len(val_subjects)}")
    for subject in val_subjects:
        image_path = Path(subject['image'].path).name
        print(f"{image_path}")




def print_lopocv_subjects(subject_dict):
    print("--- Dane dotyczące procedury lopo-cv ---")

    group_count = len(subject_dict)
    print(f"Łącznie występuje {group_count} grup.") 

    for patient_id, subject_list in subject_dict.items():
        print(f"\n  Pacjent ID: {patient_id} ({len(subject_list)} subjectów)")
        for subject in subject_list:
            image_path = Path(subject['image'].path).name
            print(f"  - {image_path}")


if __name__ == "__main__":

    # ------------------------------------
    # Załadowanie konfiguracji
    # ------------------------------------
    configuration = load_configuration()
    print_configuration(configuration)



    # ------------------------------------
    # Wczytywanie danych
    # ------------------------------------
    subjects = load_subjects_by_filename(configuration.DATA_DIR_IMAGES, configuration.DATA_DIR_LABELS, configuration)

    random.seed(configuration.RANDOM_STATE)
    random.shuffle(subjects)
    if len(subjects) > config.DATA_IMAGES_MAX_TRAIN_COUNT:
        subjects = subjects[:config.DATA_IMAGES_MAX_TRAIN_COUNT]
  
    if configuration.PROCEDURE_MODE == 'kfold':
        train_subjects, val_subjects = get_kfold_split(subjects,configuration.K_FOLD, configuration.VAL_KFOLD)
        print_kfold_subject(train_subjects,val_subjects)
        pass
    elif configuration.PROCEDURE_MODE == 'lopocv':
        subject_dict = get_subjects_dictionary(subjects)
        train_subjects, val_subjects = get_lopocv_split(subject_dict,configuration.VAL_LOPOCV)
        print_lopocv_subjects(subject_dict)
        
    else:
        print("Tryb normalny procedury")
        pass

    # ------------------------------------
    # Przygotowanie kolejek
    # ------------------------------------
    training_transform = tio.Compose([
        tio.RandomNoise(std=0.01),
        tio.RandomAffine(scales=0.1, degrees=10),
    ])

    # Przygotowanie prawdopodobieństwa wyboru klasy
    background_weight = 0.2
    other_weight =  (1.0 - background_weight) / (config.CLASS_NUMBER-1)
    class_propabilities = {0: background_weight}
    for i in range(1, config.CLASS_NUMBER):
        class_propabilities[i] = other_weight
    print(class_propabilities)

    train_dataset = tio.SubjectsDataset(train_subjects, transform=training_transform)
    train_queue = tio.Queue(
        train_dataset,
        configuration.QUE_MAX_LENGTH_TRAIN,
        configuration.QUE_SAMPLES_PER_VOLUME_TRAIN,
        sampler=tio.data.LabelSampler(configuration.PATCH_SIZE,label_name='mask',label_probabilities=class_propabilities),
        num_workers=configuration.NUM_WORKERS_TRAIN,
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    train_loader = DataLoader(train_queue,batch_size=config.BATCH_SIZE)


    val_dataset = tio.SubjectsDataset(val_subjects)
    val_queue = tio.Queue(
        val_dataset,
        configuration.QUE_MAX_LENGTH_VALIDATION,
        configuration.QUE_SAMPLES_PER_VOLUME_VALIDATION,
        sampler=tio.data.UniformSampler(configuration.PATCH_SIZE),
        num_workers=configuration.NUM_WORKERS_VALIDATION,
        shuffle_subjects=True,
        shuffle_patches=True
    )
    val_loader = DataLoader(val_queue,batch_size=1,shuffle=False)


    # ------------------------------------
    # Załadowanie modelu
    # ------------------------------------

    if configuration.MODEL_TYPE == 'UnetPP3D':
        model = UNetPP3D(in_channels=1,out_channels=configuration.CLASS_NUMBER,deep_supervision=True)
    else:
        model = UNet3D(in_channels=1, out_channels=configuration.CLASS_NUMBER)



    # ------------------------------------
    # Przygotowanie trainera
    # ------------------------------------

    trainer = UnetTrainer(
        model = model,
        model_save_dir=configuration.MODEL_SAVE_DIR,
        train_dataset=train_loader,
        val_dataset=val_loader,
        classes_number=configuration.CLASS_NUMBER,
        batch_size=configuration.BATCH_SIZE,
        learning_rate=configuration.LEARNING_RATE,
        num_epochs=configuration.EPOCH_COUNT,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        #model_path="Models\\UnetPP3D\\experiment_2025-06-15_22-51-43_k=3\\UnetPP3D_model_86_2025-06-16_18-30-26.pth"
    )
    print("\n--- Trenowanie---")
    trainer.train()