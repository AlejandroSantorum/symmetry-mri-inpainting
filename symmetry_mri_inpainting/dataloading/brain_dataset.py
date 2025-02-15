import logging
import os
from typing import List

import nibabel as nib
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainDataset(torch.utils.data.Dataset):
    """
    Brain dataset for guided diffusion training and evaluation.

    Attributes:
        directory (str): Base path to the dataset directory.
        test_flag (bool, default=True): Flag to indicate if the dataset is for testing. Default is True.
        input_img_types (List[str], optional): List of input image types. Default is None.
        output_img_types (List[str], optional): List of output image types. Default is None.
        reference_img_type (str, default="mask"): Reference image type. Default is "mask".
        num_cutoff_samples (int, optional): Number of samples to split by. The first part is used
            if test_flag is False, the second part is used if test_flag is True. Default is None.
            If None, all samples are used.
        num_max_samples (int, optional): Maximum number of samples to use. Default is None.
            If None, all samples are used.
        seed (int, optional): Random seed for reproducibility. Default is None.
    """

    def __init__(
        self,
        directory: str,
        test_flag: bool = True,
        input_img_types: List[str] = None,
        output_img_types: List[str] = None,
        reference_img_type: str = "mask",
        num_cutoff_samples: int = None,
        num_max_samples: int = None,
        seed: int = None,
    ):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.test_flag = test_flag
        self.input_img_types = input_img_types
        self.output_img_types = output_img_types
        self.reference_img_type = reference_img_type
        self.num_cutoff_samples = num_cutoff_samples
        self.num_max_samples = num_max_samples
        self.seed = seed

        if seed is not None:
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))

        self.database = []
        self.mask_vis = []
        for root, dirs, _ in os.walk(self.directory):
            dirs_sorted = sorted(dirs)

            if seed is not None:
                np.random.shuffle(dirs_sorted)

            if num_cutoff_samples is not None:
                if test_flag:
                    logger.info(f"Reading the last {num_cutoff_samples} samples ...")
                    # using the last 'num_cutoff_samples' samples
                    dirs_sorted = dirs_sorted[-int(num_cutoff_samples) :]
                else:
                    logger.info(f"Reading the first {num_cutoff_samples} samples ...")
                    # using the first 'num_cutoff_samples' samples
                    dirs_sorted = dirs_sorted[: int(num_cutoff_samples)]

            if num_max_samples is not None:
                if num_max_samples > len(dirs_sorted):
                    raise ValueError(
                        "num_max_samples must be smaller than the number of available samples."
                    )

                logger.info(f"Considering only {num_max_samples} samples ...")
                dirs_sorted = dirs_sorted[: int(num_max_samples)]

            for dir_id in dirs_sorted:
                datapoint = {}
                for _, _, filenames in os.walk(os.path.join(root, str(dir_id))):
                    fi_sorted = sorted(filenames)
                    for f in fi_sorted:
                        seqtype = f[
                            f.find(dir_id) + len(dir_id) + 1 : f.rfind(".nii.gz")
                        ]
                        datapoint[seqtype] = os.path.join(root, dir_id, f)
                        # getting indices of slices with non-zero reference mask
                        if seqtype == reference_img_type:
                            slice_range = []
                            reference_mask_img = nib.load(
                                datapoint[reference_img_type]
                            ).get_fdata()
                            for i in range(reference_mask_img.shape[2]):
                                mask_slice = reference_mask_img[:, :, i]
                                if np.sum(mask_slice) != 0:
                                    slice_range.append(i)

                    if not set(input_img_types).issubset(set(datapoint.keys())):
                        raise AssertionError(
                            f"""
                            Datapoint is incomplete.\n
                            Datapoint image types are {datapoint.keys()}\n
                            Expected image types are {input_img_types}
                        """
                        )

                    self.database.append(datapoint)
                    self.mask_vis.append(slice_range)

            break

    def __getitem__(self, x):
        filedict = self.database[x]
        slicedict = self.mask_vis[x]

        output_single_img = []
        # reading all output images
        for seqtype in self.output_img_types:
            nib_img = nib.load(filedict[seqtype]).get_fdata(dtype=np.float32)
            tensor_img = torch.tensor(nib_img)
            output_single_img.append(tensor_img)

        output_single_img = torch.stack(output_single_img)
        last_channel_img_fpath = filedict[seqtype]

        if self.test_flag:
            return (output_single_img, last_channel_img_fpath, slicedict)

        # the target is the last image in the output_single_img
        target = output_single_img[-1, ...]
        target = target.unsqueeze(0)
        # the input is all other images
        output_single_img = output_single_img[:-1, ...]
        return (output_single_img, target, slicedict)

    def __len__(self):
        return len(self.database)

    def get_reference_img(self, x):
        filedict = self.database[x]
        nib_img = nib.load(filedict[self.reference_img_type]).get_fdata(
            dtype=np.float32
        )
        return torch.tensor(nib_img)
