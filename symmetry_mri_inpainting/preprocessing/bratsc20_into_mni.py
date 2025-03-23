"""
Script to register BraTSC 2020 dataset to MNI space.
"""
import argparse
import os
import time

import ants


def main(args: argparse.Namespace):
    """
    Register BraTSC 2020 dataset to MNI space.
    """
    os.makedirs(args.dataset_output_path, exist_ok=True)

    mni_template_img = ants.image_read(args.mni_template_path, reorient="RAS")

    for split_name in ["Training", "Validation"]:
        ants_tmp_split_path = os.path.join(args.ants_tmp_path, split_name)
        os.makedirs(ants_tmp_split_path, exist_ok=True)

        split_path = os.path.join(
            args.dataset_input_path, f"MICCAI_BraTS2020_{split_name}Data"
        )
        for img_name in os.listdir(split_path):
            if not img_name.startswith("BraTS"):
                continue

            img_base_path = os.path.join(split_path, img_name)

            # get BraTSC 2020 brain image and mask
            brain_img_path = os.path.join(img_base_path, f"{img_name}_t1.nii.gz")
            seg_img_path = os.path.join(img_base_path, f"{img_name}_seg.nii.gz")
            brain_ants_img = ants.image_read(brain_img_path, reorient=False)
            seg_ants_img = ants.image_read(seg_img_path, reorient=False)

            # register brain image to MNI space
            mni_registration = ants.registration(
                fixed=mni_template_img,
                moving=brain_ants_img,
                type_of_transform="SyN",
                outprefix=f"{ants_tmp_split_path}/{int(time.time())}_",
                verbose=False,
            )
            registered_brain_img = mni_registration["warpedmovout"]

            # store registered images
            output_img_base_path = os.path.join(
                args.dataset_output_path, split_name, img_name
            )
            os.makedirs(output_img_base_path, exist_ok=True)
            registered_brain_img.to_file(
                os.path.join(output_img_base_path, f"{img_name}-t1.nii.gz")
            )

            # segmentation masks are only available for the training split
            if split_name == "Training":
                # use the same transformation to register the segmentation image
                registered_seg_img = ants.apply_transforms(
                    moving=seg_ants_img,
                    fixed=mni_registration["warpedmovout"],
                    transformlist=mni_registration["fwdtransforms"],
                    interpolator="nearestNeighbor",
                    verbose=False,
                )
                registered_seg_img.to_file(
                    os.path.join(output_img_base_path, f"{img_name}-seg.nii.gz")
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert BraTSC 2020 dataset to MNI space"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        help="Path to BraTSC 2020 dataset",
        default="/home/proyectos/ada2/santorum/data/mbtsc2020_MICCAI_BraTS_2020/mbtsc2020_BraTS_dataset",
    )
    parser.add_argument(
        "--dataset-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output BraTSC 2020 dataset in MNI space",
        default="/scratch/santorum/data/bratsc2020-mni",
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
        default="/scratch/santorum/tmp/ants_mni_registration/bratsc2020",
    )

    args = parser.parse_args()
    main(args)
