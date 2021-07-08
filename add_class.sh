# modify
NUM_CLASS = 'ls ./input_data/ |grep "^ｄ"|wc -l'
echo NUM_CLASS

DATASET_DIR=./input_data

python convert_customized_data.py \
    --dataset_name=customized \
    --dataset_dir="${DATASET_DIR}"

TRAIN_DIR=./train

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=nasnet_large

CHECKPOINT_PATH=./pre-trained/model.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=customized \
    --dataset_split_name=train \
    --model_name=nasnet_large \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=final_layer,aux_11 \
    --trainable_scopes=final_layer,aux_11
