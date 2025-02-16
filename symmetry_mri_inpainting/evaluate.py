import argparse
import torch
import math
import os
import logging
from typing import List, Dict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from symmetry_mri_inpainting.utils import logger, dist_util
from symmetry_mri_inpainting.utils.arguments import get_create_eval_argparser
from symmetry_mri_inpainting.utils.create import (
    create_gaussian_diffusion,
    create_unet_model,
)
from symmetry_mri_inpainting.dataloading.brain_dataset import BrainDataset
from symmetry_mri_inpainting.utils.reproducibility import set_seed
from symmetry_mri_inpainting.metrics import mse_2d, snr_2d, psnr_2d, ssim_2d


def plot_save_slices(
    original_batch: torch.Tensor,
    inpainted_batch: torch.Tensor,
    actual_img_size: int,
    slice_indices: list,
    subject_name: str,
    output_dir: str,
    perf_metrics: List[Dict] = None,
):
    # Plot the original and inpainted slices
    for i in range(inpainted_batch.shape[0]):
        n_figs = original_batch.shape[1] + 2 # channels, inpainted, diff map
        width_ratios = ([1] * n_figs) + [0.05]
        fig, axs = plt.subplots(1, n_figs+1, figsize=(3.8*n_figs, 3.8), gridspec_kw={"width_ratios": width_ratios})
        for k in range(original_batch.shape[1]):
            _img_show = original_batch[i,k,...].view(actual_img_size, actual_img_size).numpy()
            axs[k].imshow(_img_show, cmap="gray")
            if k == original_batch.shape[1]-1:
                axs[k].set_title("Groundtruth")
            else:
                axs[k].set_title(f"Channel {k+1}")
        
        inpainted_img = inpainted_batch[i].view(actual_img_size, actual_img_size).numpy()
        axs[-3].imshow(inpainted_img, cmap="gray")
        axs[-3].set_title("Inpainted")

        groundtruth_img = original_batch[i,-1,...].view(actual_img_size, actual_img_size).numpy()
        diff_map = inpainted_img - groundtruth_img
        ax_cb = axs[-2].imshow(diff_map, norm=mpl.colors.CenteredNorm(), cmap="seismic")
        axs[-2].set_title("Residuals Map")
        fig.colorbar(ax_cb, ax=axs[-2], cax=axs[-1])

        if perf_metrics is not None:
            title_msg = " | ".join(f"{k} = {v:.5f}" for k, v in perf_metrics[i].items())
            plt.suptitle(title_msg)

        plt.savefig(os.path.join(output_dir, f"{subject_name}_slice_{slice_indices[i]}.png"))


def evaluate(rank: int, use_gpu: bool, world_size: int, args: argparse.Namespace) -> None:
    # Configure the logger
    if args.output_dir:
        logger.configure(dir=args.log_output_dir)
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
    with open(args.model_pt_path, "rb") as f:
        model_data = torch.load(f, map_location="cpu")
    model.load_state_dict(model_data)
    model.to(device)

    logger.log("Model loaded successfully.")

    logger.log(f"Creating brain dataset loading from '{args.data_dir}'")
    brain_dataset = BrainDataset(
        directory=args.data_dir,
        test_flag=True,  # testing
        input_img_types=args.input_img_types,
        output_img_types=args.output_img_types,
        reference_img_type=args.reference_img_type,
        num_cutoff_samples=args.num_cutoff_samples,
        num_max_samples=args.num_max_samples,
        seed=args.dataset_seed,
    )

    if len(brain_dataset) == 0:
        raise ValueError(f"No samples found in the dataset in {args.data_dir}")

    logger.log(f"Loaded {len(brain_dataset)} samples from brain dataset")

    if args.npy_output_dir:
        os.makedirs(os.path.join(args.npy_output_dir, "inpainted"), exist_ok=True)
        os.makedirs(os.path.join(args.npy_output_dir, "groundtruth"), exist_ok=True)
        os.makedirs(os.path.join(args.npy_output_dir, "ref_mask"), exist_ok=True)
    
    if args.png_output_dir:
        os.makedirs(os.path.join(args.png_output_dir, "inpainted"), exist_ok=True)
        os.makedirs(os.path.join(args.png_output_dir, "groundtruth"), exist_ok=True)
        os.makedirs(os.path.join(args.png_output_dir, "ref_mask"), exist_ok=True)
    
    if args.xlsx_output_dir:
        os.makedirs(args.xlsx_output_dir, exist_ok=True)
    
    logger.log(f"Starting evaluation with seed: {args.evaluation_seed}")
    set_seed(args.evaluation_seed)

    # Metadata lists
    subject_names, slice_indices = [], []
    # Lists to store the performance metrics
    mse_list, snr_list, psnr_list, ssim_list = [], [], [], []

    for i in range(len(brain_dataset)):
        batch_i, path_i, slicedict_i = brain_dataset[i]
        ref_mask_i = brain_dataset.get_reference_img(i)
        num_p_sample_loop_iters = math.ceil(len(slicedict_i) / args.sample_batch_size)

        for j in range(num_p_sample_loop_iters):
            # get the indices for the current batch
            start_idx = j * args.sample_batch_size
            end_idx = start_idx + args.sample_batch_size
            # get the slice indices to inpaint in the current batch
            slicedict_i_j = slicedict_i[start_idx:end_idx]
            # get the batch of images to inpaint based on the slice indices
            batch_i_j = batch_i[:,:,:,slicedict_i_j]
            ref_mask_i_j = ref_mask_i[:,:,slicedict_i_j]
            # permute the dimensions to match the model's input shape (batch size, channels, height, width)
            batch_i_j = torch.permute(batch_i_j, (3, 0, 1, 2))

            # perform inpainting on the current batch of images using the DDPM model
            inpainted_batch_i_j, x_noisy_batch_i_j, original_batch_i_j = diffusion.p_sample_loop_known(
                model=model,
                shape=(batch_i_j.shape[0], batch_i_j.shape[1], args.model_image_size, args.model_image_size),
                img=batch_i_j,
                clip_denoised=True,
                model_kwargs={},
                progress=True,
            )

            inpainted_batch_i_j = inpainted_batch_i_j.cpu()
            x_noisy_batch_i_j = x_noisy_batch_i_j.cpu()
            original_batch_i_j = original_batch_i_j.cpu()

            mse_list_batch, snr_list_batch, psnr_list_batch, ssim_list_batch = [], [], [], []
            # calculate the performance metrics for the inpainted images
            for k in range(inpainted_batch_i_j.shape[0]):
                inpainted_slice_k = inpainted_batch_i_j[k].view(args.actual_image_size, args.actual_image_size).numpy()
                groundtruth_slice_k = original_batch_i_j[k,-1,...].view(args.actual_image_size, args.actual_image_size).numpy()
                ref_mask_slice_k = ref_mask_i_j[:,:,k].view(args.actual_image_size, args.actual_image_size).numpy()

                if args.npy_output_dir:
                    subject_name = os.path.basename(path_i).replace(".nii.gz", "")
                    np.save(
                        file=os.path.join(args.npy_output_dir, "inpainted", f"{subject_name}_slice_{slicedict_i_j[k]}.npy"),
                        arr=inpainted_slice_k
                    )
                    np.save(
                        file=os.path.join(args.npy_output_dir, "groundtruth", f"{subject_name}_slice_{slicedict_i_j[k]}.npy"),
                        arr=groundtruth_slice_k
                    )
                    np.save(
                        file=os.path.join(args.npy_output_dir, "ref_mask", f"{subject_name}_slice_{slicedict_i_j[k]}.npy"),
                        arr=ref_mask_slice_k
                    )

                mse_k = mse_2d(test_img=inpainted_slice_k, ref_img=groundtruth_slice_k, mask=ref_mask_slice_k)
                snr_k = snr_2d(test_img=inpainted_slice_k, ref_img=groundtruth_slice_k, mask=ref_mask_slice_k)
                psnr_k = psnr_2d(test_img=inpainted_slice_k, ref_img=groundtruth_slice_k, mask=ref_mask_slice_k)
                ssim_k = ssim_2d(test_img=inpainted_slice_k, ref_img=groundtruth_slice_k, mask=ref_mask_slice_k)

                mse_list_batch.append(mse_k)
                snr_list_batch.append(snr_k)
                psnr_list_batch.append(psnr_k)
                ssim_list_batch.append(ssim_k)

                subject_names.append(os.path.basename(path_i).replace(".nii.gz", ""))
                slice_indices.append(slicedict_i_j[k])
            
            mse_list.extend(mse_list_batch)
            snr_list.extend(snr_list_batch)
            psnr_list.extend(psnr_list_batch)
            ssim_list.extend(ssim_list_batch)

            # Save the inpainted and original slices
            if args.png_output_dir:
                os.makedirs(args.png_output_dir, exist_ok=True)
                plot_save_slices(
                    original_batch=original_batch_i_j,
                    inpainted_batch=inpainted_batch_i_j,
                    actual_img_size=args.actual_image_size,
                    slice_indices=slicedict_i_j,
                    subject_name=os.path.basename(path_i).replace(".nii.gz", ""),
                    output_dir=args.png_output_dir,
                    perf_metrics=[
                        {
                            "MSE": mse_list_batch[k],
                            "SNR": snr_list_batch[k],
                            "PSNR": psnr_list_batch[k],
                            "SSIM": ssim_list_batch[k],
                        }
                        for k in range(len(slicedict_i_j))
                    ]
                )

    # Calculate the performance metrics
    mse_list = np.array(mse_list)
    snr_list = np.array(snr_list)
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)

    logger.info(f"Dropping {np.sum(np.isnan(mse_list))} NaN values from MSE array")
    pr_mse_list = mse_list[~np.isnan(mse_list)]
    logger.info(f"Dropping {np.sum(np.isnan(snr_list))} NaN values from SNR array")
    pr_snr_list = snr_list[~np.isnan(snr_list)]
    logger.info(f"Dropping {np.sum(np.isnan(psnr_list))} NaN values from PSNR array")
    pr_psnr_list = psnr_list[~np.isnan(psnr_list)]
    logger.info(f"Dropping {np.sum(np.isnan(ssim_list))} NaN values from SSIM array")
    pr_ssim_list = ssim_list[~np.isnan(ssim_list)]

    logger.info("====================================")
    logger.info("Performance Metrics:")
    logger.info(f"MSE: {np.mean(pr_mse_list)} ± {np.std(pr_mse_list)}")
    logger.info(f"SNR: {np.mean(pr_snr_list)} ± {np.std(pr_snr_list)}")
    logger.info(f"PSNR: {np.mean(pr_psnr_list)} ± {np.std(pr_psnr_list)}")
    logger.info(f"SSIM: {np.mean(pr_ssim_list)} ± {np.std(pr_ssim_list)}")
    logger.info("====================================")
    logger.info("Quantiles of Performance Metrics:")
    for quantile in [0.25, 0.5, 0.75]:
        logger.info(f"MSE {quantile}: {np.quantile(pr_mse_list, quantile)}")
        logger.info(f"SNR {quantile}: {np.quantile(pr_snr_list, quantile)}")
        logger.info(f"PSNR {quantile}: {np.quantile(pr_psnr_list, quantile)}")
        logger.info(f"SSIM {quantile}: {np.quantile(pr_ssim_list, quantile)}")
    logger.info("====================================")

    # Save the performance metrics to a Excel file
    performance_metrics = {
        "Subject Name": subject_names,
        "Slice Index": slice_indices,
        "MSE": mse_list,
        "SNR": snr_list,
        "PSNR": psnr_list,
        "SSIM": ssim_list,
    }
    performance_metrics_df = pd.DataFrame(performance_metrics)
    checkpoint_name = os.path.basename(args.model_pt_path).replace(".pt", "")
    performance_metrics_df.to_excel(
        os.path.join(
            os.path.dirname(args.model_pt_path),
            f"performance_metrics_{checkpoint_name}.xlsx"
        )
    )
    if args.xlsx_output_dir:
        os.makedirs(args.xlsx_output_dir, exist_ok=True)
        performance_metrics_df.to_excel(
            os.path.join(
                args.xlsx_output_dir,
                f"performance_metrics_{checkpoint_name}.xlsx"
            )
        )



def main(args: argparse.Namespace) -> None:
    """
    Main function for evaluating the model.
    Determine the number of available GPUs and spawn the training process.
    """
    gpu_count = torch.cuda.device_count()
    logging.info(f"Number of CUDA GPU available devices: {gpu_count}")

    if gpu_count > 1:
        logging.info(f"IDs of CUDA available devices: {os.getenv('CUDA_VISIBLE_DEVICES')}")
        torch.multiprocessing.spawn(
            evaluate,
            args=(True, gpu_count, args),
            nprocs=gpu_count,
            join=True,
        )
    else:
        logging.info("CPU training")
        evaluate(rank=0, use_gpu=False, world_size=1, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a symmetry-aware denoising diffusion model on MRI images for inpainting."
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The directory containing the training data.",
    )
    parser.add_argument(
        "--model_pt_path",
        default=None,
        type=str,
        help="The path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--model_image_size",
        default=None,
        type=int,
        help="The size of the model input and output images.",
    )
    parser.add_argument(
        "--actual_image_size",
        default=None,
        type=int,
        help="The actual size of the input images.",
    )
    parser.add_argument(
        "--sample_batch_size",
        default=4,
        type=int,
        help="The batch size to use for sampling.",
    )
    parser.add_argument(
        "--log_output_dir",
        default=None,
        type=str,
        help="The directory to save the evaluation logs.",
    )
    parser.add_argument(
        "--png_output_dir",
        default=None,
        type=str,
        help="The directory to save the evaluation PNG images.",
    )
    parser.add_argument(
        "--npy_output_dir",
        default=None,
        type=str,
        help="The directory to save the evaluation NPY images.",
    )
    parser.add_argument(
        "--xlsx_output_dir",
        default=None,
        type=str,
        help="The directory to save the evaluation XLSX file.",
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
        default="mask",
        type=str,
        help="The type of reference image of the dataloader to determine a valid training/validation sample.",
    )
    parser.add_argument(
        "--num_cutoff_samples",
        default=None,
        type=int,
        help="The number of cutoff samples to use for evaluation. This is used to split the dataset into training and validation sets.",
    )
    parser.add_argument(
        "--num_max_samples",
        default=None,
        type=int,
        help="The maximum number of samples to use for evaluation. This is used to limit the number of samples used for evaluation.",
    )
    parser.add_argument(
        "--dataset_seed",
        default=None,
        type=int,
        help="The seed to use for the dataset sampler.",
    )
    parser.add_argument(
        "--evaluation_seed",
        default=0,
        type=int,
        help="The seed to use for evaluating the model.",
    )

    parser = get_create_eval_argparser(parser=parser)
    args = parser.parse_args()
    main(args=args)
