import argparse
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert BraTSC 2020 dataset in the MNI space to be used by AOT-GAN"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        help="Path to BraTSC 2020 dataset in MNI space",
        default="/scratch/santorum/data/brats20-mni-dm-inpainting-preprocessed-3d",
    )
    parser.add_argument(
        "--dataset-brains-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output BraTSC 2020 dataset (brains) in MNI space after preprocessing",
        default="/scratch/santorum/data/brats20-mni-AOT-GAN-brains-png",
    )
    parser.add_argument(
        "--dataset-masks-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output BraTSC 2020 dataset (masks) in MNI space after preprocessing",
        default="/scratch/santorum/data/brats20-mni-AOT-GAN-masks-png",
    )

    args = parser.parse_args()

    subjects_to_process_fpath = os.path.join(args.dataset_input_path, "Training")
    for subject_name in os.listdir(subjects_to_process_fpath):
        brain_img_fpath = os.path.join(
            subjects_to_process_fpath,
            subject_name,
            f"{subject_name}-t1n.nii.gz",
        )
        mask_img_fpath = os.path.join(
            subjects_to_process_fpath,
            subject_name,
            f"{subject_name}-mask-unhealthy.nii.gz",
        )
        brain_img_3d = nib.load(brain_img_fpath).get_fdata(dtype=np.float32)
        mask_img_3d = nib.load(mask_img_fpath).get_fdata(dtype=np.float32)

        os.makedirs(
            os.path.join(args.dataset_brains_output_path, subject_name), exist_ok=True
        )
        os.makedirs(
            os.path.join(args.dataset_masks_output_path, subject_name), exist_ok=True
        )

        num_axial_slices = brain_img_3d.shape[2]
        for slice_idx in range(num_axial_slices):
            brain_slice = brain_img_3d[:, :, slice_idx]
            mask_slice = mask_img_3d[:, :, slice_idx]
            if np.sum(mask_slice) > 0.0:  # Skip empty mask slices
                brain_slice_filename = f"{subject_name}-slice{slice_idx}.png"
                brain_slice_filepath = os.path.join(
                    args.dataset_brains_output_path, subject_name, brain_slice_filename
                )
                mask_slice_filename = f"{subject_name}-slice{slice_idx}.png"
                mask_slice_filepath = os.path.join(
                    args.dataset_masks_output_path, subject_name, mask_slice_filename
                )
                plt.imsave(brain_slice_filepath, brain_slice, cmap="gray")
                plt.imsave(mask_slice_filepath, mask_slice, cmap="gray")
