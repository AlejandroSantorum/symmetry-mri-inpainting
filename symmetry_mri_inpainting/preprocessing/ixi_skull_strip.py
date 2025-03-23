"""
Perform brain extraction to skull-strip IXI dataset.
"""
import argparse
import os

import ants
from antspynet.utilities import brain_extraction


def main(args: argparse.Namespace):
    """
    Perform brain extraction to skull-strip IXI dataset.
    """
    os.makedirs(args.dataset_output_path, exist_ok=True)
    os.makedirs(args.ants_tmp_path, exist_ok=True)

    for dir_name in os.listdir(args.dataset_input_path):
        if "DS_Store" in dir_name:
            continue

        # raw IXI nifti file path
        nifti_file_path = os.path.join(args.dataset_input_path, dir_name)
        # read nifti image
        ants_image = ants.image_read(nifti_file_path)

        # brain extraction
        prob_brain_mask_img = brain_extraction(
            ants_image,
            modality=args.image_modality,  # usually 't1'
            antsxnet_cache_directory=args.ants_tmp_path,
            verbose=False,
        )

        # get binary brain mask
        brain_mask_img = ants.get_mask(prob_brain_mask_img, low_thresh=0.5)
        # apply mask to original image
        skullstripped_brain_img = ants.mask_image(ants_image, brain_mask_img)
        # store masked skull-stripped image
        output_file_path = os.path.join(args.dataset_output_path, dir_name)
        ants.image_write(skullstripped_brain_img, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to skull strip IXI dataset")

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        dest="dataset_input_path",
        help="Path to the input IXI dataset",
        default="/home/proyectos/ada2/santorum/data/IXI-dataset/t1",
    )
    parser.add_argument(
        "--dataset-output-path",
        "-o",
        type=str,
        required=True,
        dest="dataset_output_path",
        help="Path to the output skull-stripped IXI dataset",
        default="/scratch/santorum/data/IXI-skull-stripped",
    )
    parser.add_argument(
        "--ants-tmp-path",
        "-t",
        type=str,
        required=True,
        dest="ants_tmp_path",
        help="Path to the ANTs temporary directory",
        default="/scratch/santorum/tmp/antsxnet_cache",
    )
    parser.add_argument(
        "--image-modality",
        "-m",
        type=str,
        required=False,
        dest="image_modality",
        help="Image modality to perform brain extraction",
        default="t1",
    )

    args = parser.parse_args()
    main(args)
