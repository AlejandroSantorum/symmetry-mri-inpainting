import argparse
import logging
import os

import torch
from torch.utils.data.distributed import DistributedSampler

from symmetry_mri_inpainting.dataloading.brain_dataset import BrainDataset
from symmetry_mri_inpainting.model.trainer import TrainLoop
from symmetry_mri_inpainting.utils import logger, dist_util
from symmetry_mri_inpainting.utils.arguments import get_create_train_argparser
from symmetry_mri_inpainting.utils.create import (
    create_gaussian_diffusion,
    create_named_schedule_sampler,
    create_unet_model,
)
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
    # configuring output for logging and for checkpoint storing
    if args.output_dir is not None:
        logger.configure(logger_dir=args.output_dir)
    else:
        logger.configure()

    dist_util.setup_dist(rank, world_size)

    if use_gpu:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cpu:{rank}")

    logger.log(f"Initializing model and diffusion using device {device} ...")

    diffusion = create_gaussian_diffusion(args=vars(args))
    model = create_unet_model(args=vars(args))
    model.to(device)

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=1000
    )

    if args.input_img_types is not None:
        input_img_types = args.input_img_types.split(",")
        logger.log("Using input image types: " + str(input_img_types))
    else:
        input_img_types = None

    if args.output_img_types is not None:
        output_img_types = args.output_img_types.split(",")
        logger.log("Using output image types: " + str(output_img_types))
    else:
        output_img_types = None

        logger.log(f"Creating brain dataset loading from '{args.data_dir}'")
    brain_dataset = BrainDataset(
        directory=args.data_dir,
        test_flag=False,  # training
        input_img_types=input_img_types,
        output_img_types=output_img_types,
        reference_img_type=args.reference_img_type,
        num_cutoff_samples=args.num_cutoff_samples,
        num_max_samples=args.num_max_samples,
        seed=args.dataset_seed,
    )

    # Create a distributed sampler
    sampler = DistributedSampler(brain_dataset, num_replicas=world_size, rank=rank)

    dataloader = torch.utils.data.DataLoader(
        brain_dataset, batch_size=args.batch_size, sampler=sampler
    )

    seed_info = ""
    if args.training_seed is not None:
        set_seed(int(args.training_seed))
        seed_info = f" with seed {args.training_seed} "

    logger.log(f"Initiating training {seed_info}...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,  # not used
        dataloader=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    logger.log("Training finished.")


def main(args: argparse.Namespace) -> None:
    """
    Main function for training the model.
    Determine the number of available GPUs and spawn the training process.
    """
    gpu_count = torch.cuda.device_count()
    logging.info(f"Number of CUDA GPU available devices: {gpu_count}")

    if gpu_count > 1:
        logging.info(f"IDs of CUDA available devices: {os.getenv('CUDA_VISIBLE_DEVICES')}")
        torch.multiprocessing.spawn(
            train,
            args=(True, gpu_count, args),
            nprocs=gpu_count,
            join=True,
        )
    else:
        logging.info("CPU training")
        train(rank=0, use_gpu=False, world_size=1, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a symmetry-aware denoising diffusion model on MRI images for inpainting."
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The directory containing the training data.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The directory to save the trained model.",
    )
    parser.add_argument(
        "--input_img_types",
        default=None,
        type=str,
        help="The types of input images of the dataloader.",
    )
    parser.add_argument(
        "--output_img_types",
        default=None,
        type=str,
        help="The types of output images of the dataloader.",
    )
    parser.add_argument(
        "--reference_img_type",
        default=None,
        type=str,
        help="The type of reference image of the dataloader to determine a valid training sample.",
    )
    parser.add_argument(
        "--num_cutoff_samples",
        default=None,
        type=int,
        help="The number of cutoff samples to use for training. This is used to split the dataset into training and validation sets.",
    )
    parser.add_argument(
        "--num_max_samples",
        default=None,
        type=int,
        help="The maximum number of samples to use for training. This is used to limit the number of samples used for training.",
    )
    parser.add_argument(
        "--dataset_seed",
        default=None,
        type=int,
        help="The seed to use for the dataset sampler.",
    )
    parser.add_argument(
        "--training_seed",
        default=None,
        type=int,
        help="The seed to use for training the model.",
    )

    parser = get_create_train_argparser(parser=parser)
    args = parser.parse_args()
    main(args=args)
