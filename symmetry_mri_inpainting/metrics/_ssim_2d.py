import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def ssim_2d(
    test_img: np.ndarray, ref_img: np.ndarray, mask: np.ndarray = None,
) -> float:
    """
    Compute the structural similarity index between two 2D images.

    Parameters
    ----------
    test_img: ndarray
        Image being compared.
    ref_img: ndarray
        Reference image.
    mask: ndarray, optional, default None
        Mask of the region to be compared. If None, the whole image is compared.

    Returns
    -------
    ssim: float
        The structural similarity index between the two 2D images.
    
    Reference
    ---------
    - "Structural Similarity Index Measure (SSIM)". PyTorch Lightning Metrics.
        https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html#torchmetrics.functional.image.structural_similarity_index_measure
    - BraTS 2023 Challenge. Generate Metrics for the BraTS 2023 Challenge. GitHub Repository.
        https://github.com/BraTS-inpainting/2023_challenge/commit/910ae267d54661412ed7ea4df609a27492511897#diff-710e308f4d9dd755c0ec0375b69b554caa89164b035ec7b15d079daac3a7b15c
    """
    return torch_ssim_2d(test_img=test_img, ref_img=ref_img, mask=mask)


def torch_ssim_2d(
    test_img: np.ndarray, ref_img: np.ndarray, mask: np.ndarray = None,
) -> float:
    """
    Compute the structural similarity index between two 2D images.

    Parameters
    ----------
    test_img: ndarray
        Image being compared.
    ref_img: ndarray
        Reference image.
    mask: ndarray, optional, default None
        Mask of the region to be compared. If None, the whole image is compared.

    Returns
    -------
    ssim: float
        The structural similarity index between the two 2D images.
    
    Reference
    ---------
    - "Structural Similarity Index Measure (SSIM)". PyTorch Lightning Metrics.
        https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html#torchmetrics.functional.image.structural_similarity_index_measure
    - BraTS 2023 Challenge. Generate Metrics for the BraTS 2023 Challenge. GitHub Repository.
        https://github.com/BraTS-inpainting/2023_challenge/commit/910ae267d54661412ed7ea4df609a27492511897#diff-710e308f4d9dd755c0ec0375b69b554caa89164b035ec7b15d079daac3a7b15c
    """
    # checking they are 2D images
    assert len(ref_img.shape) == 2
    # checking both images have the same size
    assert ref_img.shape == test_img.shape
    
    test_img_tensor = torch.from_numpy(test_img).unsqueeze(0).unsqueeze(0)
    ref_img_tensor = torch.from_numpy(ref_img).unsqueeze(0).unsqueeze(0)

    if mask is not None:
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        mask = (mask_tensor > 0.0)
        test_img_tensor = (test_img_tensor * mask).float()
        ref_img_tensor = (ref_img_tensor * mask).float()
    else:
        mask = torch.ones_like(ref_img_tensor, dtype=torch.bool)

    _, ssim_full_image = structural_similarity_index_measure(
        preds=ref_img_tensor, target=test_img_tensor, return_full_image=True,
    )
    try:

        _ssim = ssim_full_image[mask]
    except Exception as exc:
        print(f"WARNING: {exc}")
        if len(ssim_full_image.shape) == 0:
            _ssim = torch.zeros_like(mask) * ssim_full_image
    return _ssim.mean()
