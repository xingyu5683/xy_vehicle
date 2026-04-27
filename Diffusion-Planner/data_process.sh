###################################
# User Configuration Section
###################################
# NUPLAN_DATA_PATH="/mnt/data/dataset/nuplan-v1.1/splits/trainval" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_DATA_PATH="/mnt/datadownload"
NUPLAN_MAP_PATH="/mnt/data/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/mnt/data/test/test_data" # preprocess training data
###################################

python data_process_with_nextstate.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 16000