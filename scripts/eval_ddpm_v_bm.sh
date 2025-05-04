#!/bin/bash
#
#SBATCH --output=/home/santorum/logs/output/%x_%j.log
#SBATCH --error=/home/santorum/logs/error/%x_%j.log
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

source /home/proyectos/ada2/santorum/venv_phd/bin/activate

# Model config
MODEL_NAME="ddpm_v_bm"
INPUT_CHANNELS=3
INPUT_IMG_TYPES="voided,mask,brain"
OUTPUT_IMG_TYPES="voided,mask,brain"
REF_IMG_TYPE="mask"

# Training config
NUM_CUTOFF_SAMPLES=30
NUM_MAX_SAMPLES=30
DATASET_SEED=42  # seed for the dataset split
TRAIN_SEED=42  # seed for training - other experiments used 10
EVAL_SEED=0  # seed for evaluation
CHECKPOINT_NAME="savedmodel080000"
CHECKPOINT_DIR="/scratch/santorum/checkpoints/${MODEL_NAME}_ixi_seed${TRAIN_SEED}"
EVAL_RESULTS_DIR="/scratch/santorum/inference/${MODEL_NAME}_ixi_seed${TRAIN_SEED}/${CHECKPOINT_NAME}"

# Run the evaluation script
PYTHONPATH="/home/santorum/repos/symmetry-mri-inpainting" \
python3 /home/santorum/repos/symmetry-mri-inpainting/symmetry_mri_inpainting/evaluate.py \
    --data_dir /scratch/santorum/data/IXI-skull-stripped-mni-dm-inpainting-preprocessed-3d \
    --model_pt_path "${CHECKPOINT_DIR}/${CHECKPOINT_NAME}.pt" \
    --png_output_dir "${EVAL_RESULTS_DIR}/validation_slices_png" \
    --npy_output_dir "${EVAL_RESULTS_DIR}/validation_slices_npy" \
    --xlsx_output_dir $EVAL_RESULTS_DIR \
    --input_img_types $INPUT_IMG_TYPES \
    --output_img_types $OUTPUT_IMG_TYPES \
    --reference_img_type $REF_IMG_TYPE \
    --num_cutoff_samples $NUM_CUTOFF_SAMPLES \
    --num_max_samples $NUM_MAX_SAMPLES \
    --model_image_size 256 \
    --actual_image_size 224 \
    --sample_batch_size 8 \
    --dataset_seed $DATASET_SEED \
    --evaluation_seed $EVAL_SEED
