import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_utils import process_images  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert BraTSC 2020 dataset to MNI space"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        help="Path to BraTSC 2020 dataset in MNI space",
        default="/scratch/santorum/data/bratsc2020-mni",
    )
    parser.add_argument(
        "--dataset-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output BraTSC 2020 dataset in MNI space after preprocessing",
        default="/scratch/santorum/data/brats2020-mni-dm-inpainting-preprocessed-3d",
    )

    args = parser.parse_args()

    process_images(
        input_folder=args.dataset_input_path,
        output_folder=args.dataset_output_path,
    )
