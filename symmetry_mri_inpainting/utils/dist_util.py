"""
Helpers for distributed training.

Copied and adapted from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/dist_util.py
"""

import io
import os

import blobfile as bf
import torch as th
import torch.distributed as dist

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print("Unable to import mpi4py. MPI will not be available.")

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(rank, world_size):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices_str:
        port_suffix = visible_devices_str.split(",")[0]
    else:
        port_suffix = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"1234{port_suffix}"

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    except dist.DistNetworkError:
        os.environ["MASTER_PORT"] = "12349"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2**30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for param in params:
        # Ensure the parameter is not modified in-place
        p = param.data.clone()
        dist.broadcast(p, src=0)
        param.data.copy_(p)
