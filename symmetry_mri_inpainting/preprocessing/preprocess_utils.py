import argparse
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import rotate


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def crop_or_pad_image(image, target_shape):
    """
    Crop or pad the image to the target shape.
    """
    current_shape = image.shape
    cropped_padded_image = np.zeros(target_shape)

    # Calculate cropping/padding indices
    crop_start = [
        (current_dim - target_dim) // 2 if current_dim > target_dim else 0
        for current_dim, target_dim in zip(current_shape, target_shape)
    ]
    crop_end = [crop_start[i] + target_shape[i] for i in range(len(target_shape))]

    pad_start = [
        (target_dim - current_dim) // 2 if current_dim < target_dim else 0
        for current_dim, target_dim in zip(current_shape, target_shape)
    ]
    pad_end = [pad_start[i] + current_shape[i] for i in range(len(current_shape))]

    # Crop the image
    cropped_image = image[
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ]

    # Pad the image
    cropped_padded_image[
        pad_start[0] : pad_end[0], pad_start[1] : pad_end[1], pad_start[2] : pad_end[2]
    ] = cropped_image

    return cropped_padded_image


def symmetrize_image(image, mask, axis=2):
    symmetrized_slices = []
    for i in range(image.shape[axis]):
        image_slice = np.take(image, i, axis=axis)
        mask_slice = np.take(mask, i, axis=axis)

        image_center = int(
            np.nan_to_num(np.round(np.mean(np.nonzero(np.sum(image_slice, axis=0)))))
        )
        mask_center = int(
            np.nan_to_num(np.round(np.mean(np.nonzero(np.sum(mask_slice, axis=0)))))
        )

        _zeros = np.zeros((image_slice.shape[0], image_slice.shape[1] - image_center))
        _ones = np.ones((image_slice.shape[0], image_center))

        if not mask_center:
            # no healthy mask is present - randomly choose a side
            if np.random.rand() > 0.5:
                symmetry_mask = np.concatenate([_zeros, _ones], axis=1)
            else:
                symmetry_mask = np.concatenate([_ones, _zeros], axis=1)
        else:
            # healthy mask and image are non-null
            if mask_center < image_center:
                # the healthy mask is on the left side of the image
                symmetry_mask = np.concatenate([_zeros, _ones], axis=1)
            elif mask_center >= image_center:
                # the healthy mask is on the right side of the image
                # or the mask is null
                symmetry_mask = np.concatenate([_ones, _zeros], axis=1)

        symmetrized_image_slice = image_slice * symmetry_mask + image_slice[
            :, ::-1
        ] * np.logical_not(symmetry_mask)
        symmetrized_slices.append(symmetrized_image_slice)

    symmetrized_image = np.stack(symmetrized_slices, axis=axis)
    return symmetrized_image


def process_images_for_ddpms(input_folder, output_folder, seed=42):
    np.random.seed(seed)
    target_shape = (224, 224, 224)

    for subdir, _, files in os.walk(input_folder):
        t1n_files = [f for f in files if f.endswith("t1n.nii.gz")]  # t1 image

        for t1n_file in t1n_files:
            print(f"Processing {t1n_file}...")
            t1n_path = os.path.join(subdir, t1n_file)
            healthy_mask_path = os.path.join(
                subdir, t1n_file.replace("t1n", "mask-healthy")
            )  # healthy mask
            unhealthy_mask_path = os.path.join(
                subdir, t1n_file.replace("t1n", "mask-unhealthy")
            )  # unhealthy mask

            # Load images
            t1n_image = nib.load(t1n_path).get_fdata()
            healthy_mask_image = nib.load(healthy_mask_path).get_fdata()
            unhealthy_mask_image = nib.load(unhealthy_mask_path).get_fdata()

            # NEW: Rotate images in the axial plane
            t1n_image = rotate(t1n_image, axes=(1, 0), angle=90)
            healthy_mask_image = rotate(healthy_mask_image, axes=(1, 0), angle=90)
            unhealthy_mask_image = rotate(unhealthy_mask_image, axes=(1, 0), angle=90)

            # Clipping
            t1n_image = np.clip(
                t1n_image, np.quantile(t1n_image, 0.001), np.quantile(t1n_image, 0.999)
            )

            # Normalize images between 0 and 1
            t1n_image = normalize_image(t1n_image)
            healthy_mask_image = normalize_image(healthy_mask_image)
            unhealthy_mask_image = normalize_image(unhealthy_mask_image)

            # Crop or pad images to [224, 224, 224]
            t1n_image = crop_or_pad_image(t1n_image, target_shape)
            healthy_mask_image = crop_or_pad_image(healthy_mask_image, target_shape)
            unhealthy_mask_image = crop_or_pad_image(unhealthy_mask_image, target_shape)

            # Make sure masks are binary masks
            healthy_mask_image = np.round(healthy_mask_image)
            unhealthy_mask_image = np.round(unhealthy_mask_image)

            # Create corresponding subdir in output folder
            relative_subdir = os.path.relpath(subdir, input_folder)
            output_subdir = os.path.join(output_folder, relative_subdir)
            os.makedirs(output_subdir, exist_ok=True)

            # Save the processed t1n_image and healthy_image
            processed_t1n_image_path = os.path.join(output_subdir, t1n_file)
            processed_healthy_mask_image_path = os.path.join(
                output_subdir, t1n_file.replace("t1n", "mask")
            )
            processed_unhealthy_mask_image_path = os.path.join(
                output_subdir, t1n_file.replace("t1n", "mask-unhealthy")
            )
            nib.save(nib.Nifti1Image(t1n_image, np.eye(4)), processed_t1n_image_path)
            nib.save(
                nib.Nifti1Image(healthy_mask_image, np.eye(4)),
                processed_healthy_mask_image_path,
            )
            nib.save(
                nib.Nifti1Image(unhealthy_mask_image, np.eye(4)),
                processed_unhealthy_mask_image_path,
            )

            # Mask out values in t1n_image where healthy_image == 1 (to create "voided" image that needs inpainting)
            t1n_healthy_voided_image = t1n_image.copy()
            t1n_healthy_voided_image[healthy_mask_image == 1] = 0

            # Mask out values in t1n_image where unhealthy_image == 1 (to create "voided" image that needs inpainting)
            t1n_unhealthy_voided_image = t1n_image.copy()
            t1n_unhealthy_voided_image[unhealthy_mask_image == 1] = 0

            # Save the modified voided images
            healthy_voided_image_path = os.path.join(
                output_subdir, t1n_file.replace("t1n", "healthy-voided")
            )
            unhealthy_voided_image_path = os.path.join(
                output_subdir, t1n_file.replace("t1n", "unhealthy-voided")
            )
            nib.save(
                nib.Nifti1Image(t1n_healthy_voided_image, np.eye(4)),
                healthy_voided_image_path,
            )
            nib.save(
                nib.Nifti1Image(t1n_unhealthy_voided_image, np.eye(4)),
                unhealthy_voided_image_path,
            )

            # Symmetrize the images
            symmetrized_t1n_image = symmetrize_image(t1n_image, healthy_mask_image)
            symmetrized_healthy_mask_image = healthy_mask_image * t1n_image[:, ::-1]

            # Save the symmetrized images
            symmetrized_t1n_image_path = os.path.join(
                output_subdir, t1n_file.replace("t1n", "symm-t1n")
            )
            symmetrized_healthy_mask_image_path = os.path.join(
                output_subdir, t1n_file.replace("t1n", "symm-healthy-mask")
            )
            nib.save(
                nib.Nifti1Image(symmetrized_t1n_image, np.eye(4)),
                symmetrized_t1n_image_path,
            )
            nib.save(
                nib.Nifti1Image(symmetrized_healthy_mask_image, np.eye(4)),
                symmetrized_healthy_mask_image_path,
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--input_folder", type=str, required=True)
    argparser.add_argument("--output_folder", type=str, required=True)

    args = argparser.parse_args()

    process_images_for_ddpms(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
    )
