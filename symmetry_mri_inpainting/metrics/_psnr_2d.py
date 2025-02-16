import numpy as np

from ._mse_2d import mse_2d


def psnr_2d(
    test_img: np.ndarray,
    ref_img: np.ndarray,
    mask: np.ndarray = None,
    max_val: float = None,
) -> float:
    """Calculate the Peak Signal to Noise Ratio between two 2D images.

    Parameters
    ----------
    test_img: ndarray
        Image being compared.
    ref_img: ndarray
        Reference image.
    max_val: float, optional
        Maximum value of the image data. If None, the maximum value of the
        image data type is used.

    Returns
    -------
    psnr: float
        The Peak Signal to Noise Ratio between the two images.

    Reference
    ---------
    - "Peak Signal-to-Noise Ratio (PSNR)". PyTorch Lightning Metrics.
        https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html#torchmetrics.functional.image.peak_signal_noise_ratio
    - BraTS 2023 Challenge. Generate Metrics for the BraTS 2023 Challenge. GitHub Repository.
        https://github.com/BraTS-inpainting/2023_challenge/commit/910ae267d54661412ed7ea4df609a27492511897#diff-710e308f4d9dd755c0ec0375b69b554caa89164b035ec7b15d079daac3a7b15c
    """
    # checking they are 2D images
    assert len(ref_img.shape) == 2
    # checking both images have the same size
    assert ref_img.shape == test_img.shape

    _mse = mse_2d(test_img=test_img, ref_img=ref_img, mask=mask)

    if mask is not None:
        mask = mask > 0.0
        ref_img = ref_img[mask]
        test_img = test_img[mask]

    if max_val is None:
        max_val = ref_img.max()

    return 10 * np.log10(max_val**2 / _mse)
