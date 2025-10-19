import torch
from typing import Tuple


def create_codebook(
    vector_shape: Tuple[int, ...],
    K: int = 256,
    seed: int = 1234,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a deterministic, normalized random codebook with K atoms.

    Args:
        vector_shape: The shape of a single latent vector (e.g., (C, H, W)).
        K: Number of codebook atoms.
        seed: RNG seed for determinism across compress/decompress.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        codebook: Tensor of shape [K, *vector_shape], L2-normalized per atom.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    flat_dim = 1
    for d in vector_shape:
        flat_dim *= int(d)

    # Sample standard normal and normalize each atom.
    cb = torch.randn((K, flat_dim), generator=gen, device=device, dtype=dtype)
    cb = torch.nn.functional.normalize(cb, p=2, dim=1, eps=1e-12)
    cb = cb.view((K, *vector_shape))
    return cb
