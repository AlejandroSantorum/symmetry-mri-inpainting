import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess_utils import process_images_for_ddpms  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert OpenNeuro 228 dataset in the MNI space to be used by the DDPMs"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        help="Path to OpenNeuro 228 dataset in MNI space",
        default="/scratch/santorum/data/openneuro-ds000228-mni",
    )
    parser.add_argument(
        "--dataset-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output OpenNeuro 228 dataset in MNI space after preprocessing",
        default="/scratch/santorum/data/openneuro-ds000228-dm-inpainting-preprocessed-3d",
    )

    args = parser.parse_args()

    process_images_for_ddpms(
        input_folder=args.dataset_input_path,
        output_folder=args.dataset_output_path,
    )
