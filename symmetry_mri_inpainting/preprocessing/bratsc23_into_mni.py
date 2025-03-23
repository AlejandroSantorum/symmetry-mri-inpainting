"""
Script to register BraTSC 2023 dataset to MNI space.
"""
import argparse
import os
import time

import ants


def main(args: argparse.Namespace):
    """
    Register BraTSC 2023 dataset to MNI space.
    """
    os.makedirs(args.dataset_output_path, exist_ok=True)

    mni_template_img = ants.image_read(args.mni_template_path, reorient="RAS")

    for split_name in ["Training", "Validation"]:
        ants_tmp_split_path = os.path.join(args.ants_tmp_path, split_name)
        os.makedirs(ants_tmp_split_path, exist_ok=True)

        split_path = os.path.join(
            args.dataset_input_path,
            f"ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-{split_name}",
        )
        for img_name in os.listdir(split_path):
            if not img_name.startswith("BraTS"):
                continue

            img_base_path = os.path.join(split_path, img_name)

            # get BraTSC 2023 brain images and masks
            t1n_img_path = os.path.join(img_base_path, f"{img_name}-t1n.nii.gz")
            t1n_voided_img_path = os.path.join(
                img_base_path, f"{img_name}-t1n-voided.nii.gz"
            )
            mask_img_path = os.path.join(img_base_path, f"{img_name}-mask.nii.gz")
            mask_unhealthy_img_path = os.path.join(
                img_base_path, f"{img_name}-mask-unhealthy.nii.gz"
            )
            mask_healthy_img_path = os.path.join(
                img_base_path, f"{img_name}-mask-healthy.nii.gz"
            )

            t1n_ants_img = ants.image_read(t1n_img_path, reorient=False)
            t1n_voided_ants_img = ants.image_read(t1n_voided_img_path, reorient=False)
            mask_ants_img = ants.image_read(mask_img_path, reorient=False)
            mask_unhealthy_ants_img = ants.image_read(
                mask_unhealthy_img_path, reorient=False
            )
            mask_healthy_ants_img = ants.image_read(
                mask_healthy_img_path, reorient=False
            )

            # register brain image to MNI space
            mni_registration = ants.registration(
                fixed=mni_template_img,
                moving=t1n_ants_img,
                type_of_transform="SyN",
                outprefix=f"{ants_tmp_split_path}/{int(time.time())}_",
                verbose=False,
            )
            registered_t1n_img = mni_registration["warpedmovout"]

            # use the same transformation to register the segmentation image
            registered_mask_img = ants.apply_transforms(
                moving=mask_ants_img,
                fixed=registered_t1n_img,
                transformlist=mni_registration["fwdtransforms"],
                interpolator="nearestNeighbor",
                verbose=False,
            )

            # store registered images
            output_img_base_path = os.path.join(
                args.dataset_output_path, split_name, img_name
            )
            os.makedirs(output_img_base_path, exist_ok=True)
            registered_t1n_img.to_file(
                os.path.join(output_img_base_path, f"{img_name}-t1n.nii.gz")
            )
            registered_mask_img.to_file(
                os.path.join(output_img_base_path, f"{img_name}-mask.nii.gz")
            )

            # some segmentation masks are only available for the training split
            if split_name == "Training":
                registered_t1n_voided_img = ants.apply_transforms(
                    moving=t1n_voided_ants_img,
                    fixed=registered_t1n_img,
                    transformlist=mni_registration["fwdtransforms"],
                    interpolator="nearestNeighbor",
                    verbose=False,
                )
                registered_mask_unhealthy_img = ants.apply_transforms(
                    moving=mask_unhealthy_ants_img,
                    fixed=registered_t1n_img,
                    transformlist=mni_registration["fwdtransforms"],
                    interpolator="nearestNeighbor",
                    verbose=False,
                )
                registered_mask_healthy_img = ants.apply_transforms(
                    moving=mask_healthy_ants_img,
                    fixed=registered_t1n_img,
                    transformlist=mni_registration["fwdtransforms"],
                    interpolator="nearestNeighbor",
                    verbose=False,
                )
                registered_t1n_voided_img.to_file(
                    os.path.join(output_img_base_path, f"{img_name}-t1n-voided.nii.gz")
                )
                registered_mask_unhealthy_img.to_file(
                    os.path.join(
                        output_img_base_path, f"{img_name}-mask-unhealthy.nii.gz"
                    )
                )
                registered_mask_healthy_img.to_file(
                    os.path.join(
                        output_img_base_path, f"{img_name}-mask-healthy.nii.gz"
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert BraTSC 2023 dataset to MNI space"
    )

    parser.add_argument(
        "--dataset-input-path",
        "-i",
        type=str,
        required=True,
        help="Path to BraTSC 2023 dataset",
        default="/home/proyectos/ada2/santorum/data/MICCAI_BraTS_2023_Local_Inpainting",  # pragma: allowlist secret
    )
    parser.add_argument(
        "--dataset-output-path",
        "-o",
        type=str,
        required=True,
        help="Path to the output BraTSC 2023 dataset in MNI space",
        default="/scratch/santorum/data/bratsc2023-mni",
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
        default="/scratch/santorum/tmp/ants_mni_registration/bratsc2023",
    )

    args = parser.parse_args()
    main(args)
