import argparse
import logging
import torch

from symmetry_mri_inpainting.utils.arguments import create_train_argparser
from symmetry_mri_inpainting.utils.reproducibility import set_seed


def train(
    rank: int,
    use_gpu: bool,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    """
    Train the model with the given arguments.
    """
    pass


def main(args: argparse.Namespace) -> None:
    """
    Main function for training the model.
    Determine the number of available GPUs and spawn the training process.
    """
    gpu_count = torch.cuda.device_count()
    logging.info(f"Number of CUDA GPU available devices: {gpu_count}")

    if gpu_count > 1:
        torch.multiprocessing.spawn(
            train,
            args=(True, gpu_count, args),
            nprocs=gpu_count,
            join=True,
        )
    else:
        train(rank=0, use_gpu=False, world_size=1, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a symmetry-aware denoising diffusion model on MRI images for inpainting."
    )
    
    parser.add_argument("--data_dir", default=None, type=str, help="The directory containing the training data.")
    parser.add_argument("--output_dir", default=None, type=str, help="The directory to save the trained model.")
    parser.add_argument("--input_img_types", default=None, type=str, help="The types of input images of the dataloader.")
    parser.add_argument("--output_img_types", default=None, type=str, help="The types of output images of the dataloader.")
    parser.add_argument("--reference_img_type", default=None, type=str, help="The type of reference image of the dataloader to determine a valid training sample.")
    parser.add_argument("--num_cutoff_samples", default=None, type=int, help="The number of cutoff samples to use for training. This is used to split the dataset into training and validation sets.")
    parser.add_argument("--num_max_samples", default=None, type=int, help="The maximum number of samples to use for training. This is used to limit the number of samples used for training.")
    parser.add_argument("--dataset_seed", default=None, type=int, help="The seed to use for the dataset sampler.")
    parser.add_argument("--training_seed", default=None, type=int, help="The seed to use for training the model.")
    
    parser = create_train_argparser(parser=parser)
    args = parser.parse_args()
    main(args=args)
