import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_utils import process_images  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert IXI dataset to MNI space"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        help="Path to IXI dataset in MNI space",
        default="/scratch/santorum/data/IXI-skull-stripped-mni",
    )
    parser.add_argument(
        "--dataset-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output IXI dataset in MNI space after preprocessing",
        default="/scratch/santorum/data/IXI-skull-stripped-mni-dm-inpainting-preprocessed-3d",
    )

    args = parser.parse_args()

    process_images(
        input_folder=args.dataset_input_path,
        output_folder=args.dataset_output_path,
    )
