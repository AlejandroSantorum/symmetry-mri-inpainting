#!/bin/bash
#
#SBATCH --output=/home/santorum/logs/output/%x_%j.log
#SBATCH --error=/home/santorum/logs/error/%x_%j.log
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

source /home/proyectos/ada2/santorum/venv_phd/bin/activate

MODEL_NAME="aot_gan_for_inpainting_1M"
DATA_TRAIN="IXI-skull-stripped-mni-AOT-GAN-brains-png"
MASK_TYPE_TRAIN="IXI-skull-stripped-mni-AOT-GAN-masks-png"

DATA_TEST="IXI-skull-stripped-mni-AOT-GAN-brains-png"  # or "openneuro-ds000228-adults-mni-AOT-GAN-brains-png" 
MASK_TYPE_TEST="IXI-skull-stripped-mni-AOT-GAN-masks-png"  # or "openneuro-ds000228-adults-mni-AOT-GAN-masks-png"

RESULTS_PATH_SUFFIX="_ixi_testset1"  # or "_ixi_testset2" or "_on228_testset1"
SPLIT_TYPE="test1"  # or "test2"
IMAGE_SIZE=224
SEED=10
CHECKPOINTS=(
    "G0160000"
    "G0320000"
    "G0480000"
    "G0640000"
    "G0800000"
    "G0960000"
    "G1000000"
)

for checkpoint in "${CHECKPOINTS[@]}"; do

    PYTHONPATH="/home/santorum/repos/symmetry-mri-inpainting/aot_gan_replication:${PYTHONPATH}" \
    python3 /home/santorum/repos/symmetry-mri-inpainting/aot_gan_replication/test_eval_excel.py \
        --pre_train "/scratch/santorum/checkpoints/${MODEL_NAME}/aotgan_${DATA_TRAIN}_${MASK_TYPE_TRAIN}${IMAGE_SIZE}/${checkpoint}.pt" \
        --dir_image "/scratch/santorum/data" \
        --data_train $DATA_TEST \
        --dir_mask "/scratch/santorum/data" \
        --mask_type $MASK_TYPE_TEST \
        --split_type $SPLIT_TYPE \
        --image_size $IMAGE_SIZE \
        --save_dir "/scratch/santorum/tmp/${MODEL_NAME}${RESULTS_PATH_SUFFIX}" \
        --seed $SEED
done
