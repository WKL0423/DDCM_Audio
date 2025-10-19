"""
DDCM prototype modules for zero-training audio latent compression.

This package provides:
- Deterministic codebook generation (`codebook.py`)
- Greedy matching-pursuit style compressor (`runner.py`)
- High-level codec helpers (`audio_codec.py`)

Note: This is a first, training-free baseline inspired by DDCM. It compresses
the VAE latent directly via a fixed random codebook and reconstructs it without
using the diffusion UNet. Future iterations may add per-step diffusion-style
variance injection and model-guided decoding.
"""

from .codebook import create_codebook
from .runner import compress_latent, decompress_latent
from .audio_codec import (
    build_codebook,
    compress_audio_to_bitstream,
    decompress_audio_from_bitstream,
)

__all__ = [
    "create_codebook",
    "compress_latent",
    "decompress_latent",
    "build_codebook",
    "compress_audio_to_bitstream",
    "decompress_audio_from_bitstream",
]
