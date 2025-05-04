#!/bin/bash
#
#SBATCH --output=/home/santorum/logs/output/%x_%j.log
#SBATCH --error=/home/santorum/logs/error/%x_%j.log
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

source /home/proyectos/ada2/santorum/venv_phd/bin/activate

MODEL_NAME="ddpm_v_sm"
INPUT_CHANNELS=3
TRAIN_SAMPLES=400
MODEL_FLAGS="--image_size 256 --num_in_channels ${INPUT_CHANNELS} --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
TRAIN_FLAGS="--save_interval 20000"
DATASET_SEED=42  # seed for the dataset split
TRAIN_SEED=42  # seed for training - other experiments used 10

PYTHONPATH="/home/santorum/repos/symmetry-mri-inpainting" \
python3 /home/santorum/repos/symmetry-mri-inpainting/symmetry_mri_inpainting/train.py \
    --data_dir /scratch/santorum/data/IXI-skull-stripped-mni-dm-inpainting-preprocessed-3d \
    --output_dir "/scratch/santorum/checkpoints/${MODEL_NAME}_ixi_seed${TRAIN_SEED}" \
    --input_img_types "voided,symm-mask,brain" \
    --output_img_types "voided,symm-mask,brain" \
    --reference_img_type "symm-mask" \
    --num_cutoff_samples $TRAIN_SAMPLES \
    --dataset_seed $DATASET_SEED \
    --training_seed $TRAIN_SEED \
    $MODEL_FLAGS $TRAIN_FLAGS
