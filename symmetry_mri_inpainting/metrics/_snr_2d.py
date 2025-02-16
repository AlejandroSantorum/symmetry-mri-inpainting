import numpy as np


def snr_2d(ref_img: np.ndarray, test_img: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Computes Signal-to-Noise ratio measured in dB between two 2D images.

    Input:
        - ref_img (np.ndarray): 2D numpy array.
        - test_img (np.ndarray): 2D numpy array.

    Returns:
        float: signal-to-noise ratio.

    Reference:
    - "SNR, PSNR, RMSE, MAE". Daniel Sage at the Biomedical Image Group, EPFL, Switzerland.
        http://bigwww.epfl.ch/sage/soft/snr/
    - D. Sage, M. Unser, "Teaching Image-Processing Programming in Java".
        IEEE Signal Processing Magazine, vol. 20, no. 6, pp. 43-52, November 2003.
        http://bigwww.epfl.ch/publications/sage0303.html
    - BraTS 2023 Challenge. Generate Metrics for the BraTS 2023 Challenge. GitHub Repository.
        https://github.com/BraTS-inpainting/2023_challenge/commit/910ae267d54661412ed7ea4df609a27492511897#diff-710e308f4d9dd755c0ec0375b69b554caa89164b035ec7b15d079daac3a7b15c
    """
    # checking they are 2D images
    assert len(ref_img.shape) == 2
    # checking both images have the same size
    assert ref_img.shape == test_img.shape

    if mask is not None:
        mask = (mask > 0.0)
        ref_img = ref_img[mask]
        test_img = test_img[mask]

    numerator = np.sum(ref_img**2)
    denominator = np.sum((ref_img - test_img) ** 2)
    return 10 * np.log10(numerator / denominator)
