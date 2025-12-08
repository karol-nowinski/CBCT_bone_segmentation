from pathlib import Path
# ------------------------------------
# Plik zawierający konfiguracje do skryptów
# ------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# calculate_voxel_count

CALC_VOXEL_COUNT_EXTENSION = ".nii.gz"
CALC_VOXEL_COUNT_EXCEL_NAME = "calculated_volume.xlsx"
CALC_VOXEL_COUNT_PRINT = False
CALC_VOXEL_COUNT_DATA_FOLDER = BASE_DIR / "Data"

# prepare_dataset_labels_clean

PREP_DATASET_LABELS_CLEAN_DATASET = "ChinaCBCT"
PREP_DATASET_LABELS_DATASET_NAME = "CleanChinaCBCT"
PREP_DATASET_LABELS_CLEAN_EXTENSION = ".nii.gz"
PREP_DATASET_LABELS_CLEAN_OUTPUT_FOLDER = BASE_DIR / "Data" / PREP_DATASET_LABELS_DATASET_NAME / "labels"
PREP_DATASET_LABELS_CLEAN_INPUT_FOLDER = BASE_DIR / "Data" / "ChinaCBCT" / "label"

# prepare_dataset_images

PREP_DATASET_IMAGE_HISTOGRAM_FILE = Path("hs_landmark.npy")
PREP_DATASET_IMAGE_EXTENSION = ".nii.gz"
PREP_DATASET_IMAGE_DATASET_NAME = "CleanChinaCBCT"
PREP_DATASET_IMAGE_OUTPUT_FOLDER = BASE_DIR / "Data" / PREP_DATASET_IMAGE_DATASET_NAME / "images"
PREP_DATASET_IMAGE_INPUT_FOLDER = BASE_DIR / "Data" / "ChinaCBCT" / "img"

# calculate_metric_group

CALC_METRIC_INPUT_FOLDER = BASE_DIR / ""
CALC_METRIC_EXCEL_NAME = "Metryki.xlsx"
CALC_METRIC_REFERENCE_MASK_FOLDER = BASE_DIR / ""
CALC_METRIC_PREDICTION_MASK_FOLDER = BASE_DIR / ""

