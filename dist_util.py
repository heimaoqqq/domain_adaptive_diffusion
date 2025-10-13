"""
Helpers for distributed training.
"""

import io
import os
import socket

# Try to import blobfile, but make it optional
try:
    import blobfile as bf
except ImportError:
    # Fallback to standard file operations
    class bf:
        @staticmethod
        def BlobFile(path, mode):
            return open(path, mode)
        
        @staticmethod
        def exists(path):
            return os.path.exists(path)

import torch as th
import torch.distributed as dist

# Try to import MPI, but make it optional
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI = None
    MPI_AVAILABLE = False
    # Create a dummy MPI object for single GPU training
    class DummyMPI:
        class COMM_WORLD:
            @staticmethod
            def Get_rank():
                return 0
            @staticmethod
            def Get_size():
                return 1
    MPI = DummyMPI()

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    
    # Single GPU mode when MPI is not available
    if not MPI_AVAILABLE:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        
        backend = "gloo" if not th.cuda.is_available() else "nccl"
        
        # For single GPU, we might not need to initialize process group
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            dist.init_process_group(backend=backend, init_method="env://")
        return
    
    # Original MPI-based setup
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    # Single GPU mode - just load directly
    if not MPI_AVAILABLE:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)
    
    # Original MPI-based loading
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
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
    # Skip sync in single GPU mode
    if not dist.is_initialized():
        return
    
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
