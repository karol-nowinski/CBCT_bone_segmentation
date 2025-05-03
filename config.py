
# hyperparameters
PATCH_SIZE = (96,96,96)
BATCH_SIZE = 1 # has to be 1 becouse of the huge memory usage
EPOCH_COUNT = 200
CLASS_NUMBER = 9#2 # 33 # 36
LEARNING_RATE = 1e-3
RANDOM_STATE = 42






# Transformations



# training
NUM_WORKERS_TRAIN = 6
QUE_MAX_LENGTH_TRAIN = 128
QUE_SAMPLES_PER_VOLUME_TRAIN = 16

# validation
NUM_WORKERS_VAL = 2
QUE_MAX_LENGTH_VAL = 24
QUE_SAMPLES_PER_VOLUME_VAL = 8

# paths
HISTOGRAM_LANDMARKS_FILE="landmarks.npy"
# IMG_PATH = "Data\ChinaCBCTClean\img"
# LABEL_PATH = "Data\ChinaCBCTClean\label"


# China CBCT
# TRAIN_IMG_PATH = "Data\ChinaCBCTClean\imgPrepared\\train"
# VAL_IMG_PATH = "Data\ChinaCBCTClean\imgPrepared\\validation"
# TEST_IMG_PATH = "Data\ChinaCBCTClean\imgPrepared\\test"
# TRAIN_LABEL_PATH = "Data\ChinaCBCTClean\labelPrepared\\train"
# VAL_LABEL_PATH = "Data\ChinaCBCTClean\labelPrepared\\validation"
# TEST_LABEL_PATH = "Data\ChinaCBCTClean\labelPrepared\\test"
# MODEL_PATH = "" # path to folder where models will be saved
# FILE_FORMAT = ".nii.gz"
# MODEL_PATH="Models\\Unet3D"

# CleanToothFairy2_teethAll
TRAIN_IMG_PATH = "Data\\CleanToothFairy2\\imagesTr\\train"
VAL_IMG_PATH = "Data\\CleanToothFairy2\\imagesTr\\validation"
TEST_IMG_PATH = "Data\\CleanToothFairy2\\imagesTr\\test"

# Dla scalonych zebów plus reszty
TRAIN_LABEL_PATH = "Data\\CleanToothFairy2\\labelsTeethAll\\train"
VAL_LABEL_PATH = "Data\\CleanToothFairy2\\labelsTeethAll\\validation"
TEST_LABEL_PATH = "Data\\CleanToothFairy2\\labelsTeethAll\\test"


# TRAIN_LABEL_PATH = "Data\\CleanToothFairy2\\labelsOnlyTeeth\\train"
# VAL_LABEL_PATH = "Data\\CleanToothFairy2\\labelsOnlyTeeth\\validation"
# TEST_LABEL_PATH = "Data\\CleanToothFairy2\\labelsOnlyTeeth\\test"

MODEL_PATH = "" # path to folder where models will be saved
FILE_FORMAT = ".mha"
MODEL_PATH="Models\\UnetPP3D"


# inferecne
INFERENCE_FILE = ""
PATCH_OVERLAP = (48,48,48)
