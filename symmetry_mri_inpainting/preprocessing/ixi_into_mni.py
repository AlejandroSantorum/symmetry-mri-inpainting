"""
Script to register IXI dataset to MNI space.
"""
import argparse
import os
import time

import ants


def main(args: argparse.Namespace):
    """
    Register IXI dataset to MNI space.
    """
    os.makedirs(args.dataset_output_path, exist_ok=True)
    os.makedirs(args.ants_tmp_path, exist_ok=True)

    mni_template_img = ants.image_read(args.mni_template_path, reorient="RAS")

    for ixi_img_name in os.listdir(args.dataset_input_path):
        # IXI nifti file path
        nifti_file_path = os.path.join(args.dataset_input_path, ixi_img_name)
        # read nifti image
        ants_ixi_img = ants.image_read(nifti_file_path, reorient=False)

        # registration to MNI space
        mni_transformation = ants.registration(
            fixed=mni_template_img,
            moving=ants_ixi_img,
            type_of_transform="SyN",
            outprefix=f"{args.ants_tmp_path}/{int(time.time())}_",
            verbose=False,
        )
        registered_img = mni_transformation["warpedmovout"]

        # storing 3D MNI image as .nii.gz format
        registered_img.to_file(os.path.join(args.dataset_output_path, ixi_img_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert IXI dataset to MNI space"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        dest="dataset_input_path",
        help="Path to the input IXI dataset",
        default="/scratch/santorum/data/IXI-skull-stripped",
    )
    parser.add_argument(
        "--dataset-output-path",
        "-o",
        type=str,
        required=True,
        dest="dataset_output_path",
        help="Path to the output IXI dataset in MNI space",
        default="/scratch/santorum/data/IXI-skull-stripped-mni",
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
        default="/scratch/santorum/tmp/ants_mni_registration/ixi_dataset",
    )

    args = parser.parse_args()
    main(args)
