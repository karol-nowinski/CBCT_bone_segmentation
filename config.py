
# hyperparameters
PATCH_SIZE = (128,128,128)
BATCH_SIZE = 1 # has to be 1 becouse of the huge memory usage
EPOCH_COUNT = 30
CLASS_NUMBER = 2 # 33
LEARNING_RATE = 1e-4
RANDOM_STATE = 42






# Transformations



# training
NUM_WORKERS_TRAIN = 6
QUE_MAX_LENGTH_TRAIN = 64
QUE_SAMPLES_PER_VOLUME_TRAIN = 8

# validation
NUM_WORKERS_VAL = 2
QUE_MAX_LENGTH_VAL = 24
QUE_SAMPLES_PER_VOLUME_VAL = 8

# paths
HISTOGRAM_LANDMARKS_FILE="landmarks.npy"
# IMG_PATH = "Data\ChinaCBCTClean\img"
# LABEL_PATH = "Data\ChinaCBCTClean\label"
TRAIN_IMG_PATH = "Data\ChinaCBCTClean\imgPrepared\\train"
VAL_IMG_PATH = "Data\ChinaCBCTClean\imgPrepared\\validation"
TEST_IMG_PATH = "Data\ChinaCBCTClean\imgPrepared\\test"
TRAIN_LABEL_PATH = "Data\ChinaCBCTClean\labelPrepared\\train"
VAL_LABEL_PATH = "Data\ChinaCBCTClean\labelPrepared\\validation"
TEST_LABEL_PATH = "Data\ChinaCBCTClean\labelPrepared\\test"
MODEL_PATH = "" # path to folder where models will be saved
FILE_FORMAT = ".nii.gz"
MODEL_PATH="Models\\Unet3D"


# inferecne
INFERENCE_FILE = ""
PATCH_OVERLAP = (64,64,64)
