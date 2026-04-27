export CUDA_VISIBLE_DEVICES=0,1

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/opt/conda/envs/diffusion_planner/bin/python3.9" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="/mnt/data/output/preprocess_data" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/mnt/data/output/code_diffusion_planner/Diffusion-Planner/file_list.json"
###################################

$RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 2 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH