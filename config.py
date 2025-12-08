from pathlib import Path


# ------------------------------------
# Hiperparametry modelu treningowego
# ------------------------------------

PATCH_SIZE = (96,96,96)        # Rozmiar patcha wykorzystywanego do treningu
BATCH_SIZE = 1                 # Wielkośc batcha, w zaimplementowanej wersji ze względu na wielkośc dany obsługiwana jest wyłącznie wielkość 1
EPOCH_COUNT = 100              # Liczba epok treningowych
CLASS_NUMBER = 2               # Liczba klas (ChinaCBCT: 2, ToothFairy2: 9)
LEARNING_RATE = 1e-3           # Początkowy współczynnik uczenia
RANDOM_STATE = 42              # Ziarno generatora losowego

DATASET_NAME = "ChinaCBCT"    # Wykorzystywany zbiór danych [ToothFairy2, ChinaCBCT]
MODEL_TYPE = "UnetPP3D"
#MODEL_TYPE = "Unet3D"
# ------------------------------------
# Parametry kolejki i loaderów
# ------------------------------------

# Trening
NUM_WORKERS_TRAIN = 6
QUE_MAX_LENGTH_TRAIN = 128
QUE_SAMPLES_PER_VOLUME_TRAIN = 16

# Walidacja
NUM_WORKERS_VALIDATION = 2
QUE_MAX_LENGTH_VALIDATION  = 24
QUE_SAMPLES_PER_VOLUME_VALIDATION  = 8


# ------------------------------------
# Ścieżki do zbiorów danych, modeli oraz ich podstawowe informacje
# ------------------------------------



# Katalog główny projektu
BASE_DIR = Path(__file__).resolve().parent
DATA_IMAGES_MAX_TRAIN_COUNT = 150 # maksymalna liczebność zbioru treningowego

# Katalog danych
#DATA_DIR_IMAGES = BASE_DIR / "Data" / "ChinaCBCTClean" / "TEST_DATA_IMG" # Obrazy
#DATA_DIR_LABELS = BASE_DIR / "Data" / "ChinaCBCTClean" / "TEST_DATA_LABEL" # Labele

# DATA_DIR_LABELS = BASE_DIR / "Data" / "ChinaCBCTClean" / "labelPrepared" / "all"  # Obrazy
# DATA_DIR_IMAGES = BASE_DIR / "Data" / "ChinaCBCTClean" / "imgPrepared" / "all" # Labele

DATA_DIR_LABELS = BASE_DIR / "Data" / "CleanToothFairy2" / "labelsTeethAll" / "train"  # Obrazy
DATA_DIR_IMAGES = BASE_DIR / "Data" / "CleanToothFairy2" / "imagesTr" / "train" # Labele

# Format danych (rozszerzenie)
FILE_FORMAT = ".nii.gz"
# FILE_FORMAT = ".mha"


 # Plik z punktami charakterystycznymi histogramu (po normalizacji)
DATA_HISTOGRAM_FILE = BASE_DIR / "landmarks.npy"


MODEL_SAVE_DIR = BASE_DIR / "Models" / MODEL_TYPE






#  # Plik z punktami charakterystycznymi histogramu (po normalizacji)
# HISTOGRAM_LANDMARKS_FILE = "landmarks.npy"   

# Ścieżki do folderu zawierającego zbiór traningowy i walidacyjny (obrazy CBCT oraz maski referencyjne)
# ALL_IMG_PATH = "Data\\ChinaCBCTClean\\imgPrepared\\all"
# ALL_LABEL_PATH = "Data\\ChinaCBCTClean\\labelPrepared\\all"

# Format plików danych 
# FILE_FORMAT = ".nii.gz"


# Ścieżka do folderu gdzie zapisywane zostaną wytrenowane modele
#MODEL_PATH = "Models\\UnetPP3D"



# ------------------------------------
# Parametry procedury 
# ------------------------------------
#PROCEDURE_MODE = 'kfold'
PROCEDURE_MODE = 'kfold'       # rodzaje precedury: [normal, kfold, lopocv]
K_FOLD = 5                      # Liczba foldów w k-cross validation
VAL_KFOLD = 0                   # Numer folda wykorzystywanego jako zbiór walidacyjny
VAL_LOPOCV = 0

# ------------------------------------
# Inferencja
# ------------------------------------

# UnetPP3D
MODEL_NAME = "UnetPP3D_model_106_2025-07-21_23-42-27.pth"
MODEL_EXPERIMENT = "experiment_2025-07-20_22-35-14_k=36"
# Unet3D
# MODEL_NAME = "Unet3D_model_100_2025-06-14_22-31-38.pth"
# MODEL_EXPERIMENT = "experiment_2025-06-14_10-42-13_k=36_tooth"

INF_FILE_NAME = ""                                                              # Nazwa pliku zapisującego inferencje
PATCH_INF_OVERLAP = (48,48,48)                                                  # Overlap przy składaniu patchy podczas inferencji
OUTPUT_INF_FOLDER = BASE_DIR / "Results" / f"{MODEL_TYPE}_{DATASET_NAME}"
MODEL_PATH = BASE_DIR / "Models" / MODEL_TYPE / MODEL_EXPERIMENT / MODEL_NAME
INF_MODE = "folder"
INF_INPUT =  BASE_DIR / "Data" / "ChinaCBCTClean" / "imgPrepared" / "testall"
