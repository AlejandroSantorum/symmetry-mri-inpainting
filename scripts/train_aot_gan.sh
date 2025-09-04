#!/bin/bash
#
#SBATCH --output=/home/santorum/logs/output/%x_%j.log
#SBATCH --error=/home/santorum/logs/error/%x_%j.log
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

source /home/proyectos/ada2/santorum/venv_phd/bin/activate

# Model config
IMAGE_SIZE=224
TRAIN_ITERATIONS=1000000
TRAIN_SEED=10
CHECKPOINT_SUFFIX="_1M"  # ""

PYTHONPATH="/home/santorum/repos/symmetry-mri-inpainting/aot_gan_replication" \
python3 /home/santorum/repos/symmetry-mri-inpainting/aot_gan_replication/train.py \
    --dir_image "/scratch/santorum/data" \
    --data_train "IXI-skull-stripped-mni-AOT-GAN-brains-png" \
    --dir_mask "/scratch/santorum/data" \
    --mask_type "IXI-skull-stripped-mni-AOT-GAN-masks-png" \
    --image_size $IMAGE_SIZE \
    --iterations $TRAIN_ITERATIONS \
    --save_dir "/scratch/santorum/checkpoints/aot_gan_for_inpainting${CHECKPOINT_SUFFIX}" \
    --split_type "train" \
    --save_every 20000 \
    --print_every 10000 \
    --seed $TRAIN_SEED
