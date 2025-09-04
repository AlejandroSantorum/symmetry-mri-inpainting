import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_utils import slice_and_convert_png  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert OpenNeuro 228 dataset in the MNI space to be used by AOT-GAN"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        help="Path to OpenNeuro 228 dataset in MNI space",
        default="/scratch/santorum/data/openneuro-ds000228-mni-dm-inpainting-preprocessed-3d",
    )
    parser.add_argument(
        "--dataset-brains-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output OpenNeuro 228 dataset (brains) in MNI space after preprocessing",
        default="/scratch/santorum/data/openneuro-ds000228-mni-AOT-GAN-brains-png",
    )
    parser.add_argument(
        "--dataset-masks-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output OpenNeuro 228 dataset (masks) in MNI space after preprocessing",
        default="/scratch/santorum/data/openneuro-ds000228-mni-AOT-GAN-masks-png",
    )

    args = parser.parse_args()

    for subject_name in os.listdir(args.dataset_input_path):
        if subject_name.startswith("sub-pixar1") and os.path.isdir(
            os.path.join(args.dataset_input_path, subject_name)
        ):
            slice_and_convert_png(
                input_dirpath=args.dataset_input_path,
                subject_name=subject_name,
                brains_output_dirpath=args.dataset_brains_output_path,
                masks_output_dirpath=args.dataset_masks_output_path,
            )
