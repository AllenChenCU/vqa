# Paths
TASK = "MultipleChoice"
DATASET = "abstract_v002"
DATA_DIR = "toy"
#DATA_DIR = "data"
ANNOTATIONS_TRAIN_FILEPATH = f"{DATA_DIR}/Annotations/{DATASET}_train2015_annotations.json"
ANNOTATIONS_VAL_FILEPATH = f"{DATA_DIR}/Annotations/{DATASET}_val2015_annotations.json"
IMAGES_TRAIN_DIR = f"{DATA_DIR}/Images/{DATASET}/scene_img_{DATASET}_train2015/"
IMAGES_VAL_DIR = f"{DATA_DIR}/Images/{DATASET}/scene_img_{DATASET}_val2015/"
IMAGES_TEST_DIR = f"{DATA_DIR}/Images/{DATASET}/scene_img_{DATASET}_test2015/"
QUESTIONS_TRAIN_FILEPATH = f"{DATA_DIR}/Questions/Questions_Train_{DATASET}/{TASK}_{DATASET}_train2015_questions.json"
QUESTIONS_VAL_FILEPATH = f"{DATA_DIR}/Questions/Questions_Val_{DATASET}/{TASK}_{DATASET}_val2015_questions.json"
QUESTIONS_TEST_FILEPATH = f"{DATA_DIR}/Questions/Questions_Test_{DATASET}/{TASK}_{DATASET}_test2015_questions.json"
PREPROCESSED_TRAIN_FILEPATH = "./resnet_train_embeddings.h5"
PREPROCESSED_VAL_FILEPATH = "./resnet_val_embeddings.h5"
PREPROCESSED_TEST_FILEPATH = "./resnet_test_embeddings.h5"

# Preprocessing
PREPROCESS_BATCH_SIZE = 64
IMAGE_SIZE = 448 # scale shorter end of image to this size
OUTPUT_SIZE = IMAGE_SIZE // 32  # size of the feature maps after processing through a network
OUTPUT_FEATURES = 2048 # number of feature maps
CENTRAL_FRACTION = 0.875

# Training config
EPOCHS = 2
BATCH_SIZE = 128
INITIAL_LR = 1e-3
LR_HALFLIFE = 50000
DATA_WORKERS = 1
MAX_ANSWERS = 3000 # change this for multiple choice later
