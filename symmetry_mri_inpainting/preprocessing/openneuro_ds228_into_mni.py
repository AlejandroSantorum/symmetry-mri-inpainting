"""
Script to register OpenNeuro 228 dataset to MNI space.
"""
import argparse
import os
import time

import ants


def main(args: argparse.Namespace):
    """
    Register OpenNeuro 228 dataset to MNI space.
    """
    os.makedirs(args.dataset_output_path, exist_ok=True)
    os.makedirs(args.ants_tmp_path, exist_ok=True)

    mni_template_img = ants.image_read(args.mni_template_path, reorient="RAS")

    for subpixar_name in os.listdir(args.dataset_input_path):
        img_base_path = os.path.join(args.dataset_input_path, subpixar_name)

        # get OpenNeuro 228 brain image and mask
        normed_anat_path = os.path.join(
            img_base_path, f"{subpixar_name}_normed_anat.nii.gz"
        )
        brain_mask_path = os.path.join(
            img_base_path, f"{subpixar_name}_analysis_mask.nii.gz"
        )
        normed_anat_ants_img = ants.image_read(normed_anat_path, reorient=False)
        brain_mask_ants_img = ants.image_read(brain_mask_path, reorient=False)

        # apply brain mask
        brain_ants_img = normed_anat_ants_img * brain_mask_ants_img

        # registration to MNI space
        mni_transformation = ants.registration(
            fixed=mni_template_img,
            moving=brain_ants_img,
            type_of_transform="SyN",
            outprefix=f"{args.ants_tmp_path}/{int(time.time())}_",
            verbose=False,
        )
        registered_img = mni_transformation["warpedmovout"]

        # storing 3D MNI image as .nii.gz format
        registered_img.to_file(
            os.path.join(args.dataset_output_path, f"{subpixar_name}.nii.gz")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert OpenNeuro 228 dataset to MNI space"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        dest="dataset_input_path",
        help="Path to the input OpenNeuro 228 dataset",
        default="/home/proyectos/ada2/santorum/data/openneuro-ds000228/derivatives/preprocessed_data",
    )
    parser.add_argument(
        "--dataset-output-path",
        "-o",
        type=str,
        required=True,
        dest="dataset_output_path",
        help="Path to the output OpenNeuro 228 dataset in MNI space",
        default="/scratch/santorum/openneuro-ds000228-mni",
    )
    parser.add_argument(
        "--mni-template-path",
        "-m",
        type=str,
        required=True,
        dest="mni_template_path",
        help="Path to the MNI template",
        default="/home/santorum/phd/imgs/MNI152_T1_1mm_brain.nii.gz",
    )
    parser.add_argument(
        "--ants-tmp-path",
        "-t",
        type=str,
        required=True,
        dest="ants_tmp_path",
        help="Path to the ANTs temporary directory",
        default="/scratch/santorum/tmp/ants_mni_registration/openneuro_ds228",
    )

    args = parser.parse_args()
    main(args)
