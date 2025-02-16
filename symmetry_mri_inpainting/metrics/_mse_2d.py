import numpy as np


def mse_2d(test_img: np.ndarray, ref_img: np.ndarray, mask: np.ndarray = None):
    """Calculate the Mean Squared Error between two 2D images.

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
    mse: float
        The Mean Squared Error between the two 2D images.

    Reference
    ---------
    - BraTS 2023 Challenge. Generate Metrics for the BraTS 2023 Challenge. GitHub Repository.
        https://github.com/BraTS-inpainting/2023_challenge/commit/910ae267d54661412ed7ea4df609a27492511897#diff-710e308f4d9dd755c0ec0375b69b554caa89164b035ec7b15d079daac3a7b15c
    """
    # checking they are 2D images
    assert len(ref_img.shape) == 2
    # checking both images have the same size
    assert ref_img.shape == test_img.shape

    if mask is not None:
        mask = mask > 0.0
        ref_img = ref_img[mask]
        test_img = test_img[mask]

    _mse = 1 / (np.prod(ref_img.shape)) * np.sum((ref_img - test_img) ** 2)
    return _mse
