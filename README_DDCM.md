# DDCM Prototype (Zero-Training) for AudioLDM2 Latents

This prototype compresses audio by operating directly on the VAE latent using a fixed, deterministic random codebook and a greedy matching-pursuit selection. No retraining required.

- Code lives under `ddcm/`.
- CLIs: `step2_ddcm_compress_audio.py` and `step3_ddcm_decompress_audio.py`.
- Bitstream is saved as `<base>.json` (metadata) and `<base>.npz` (indices/coeffs/shape).

## Quickstart

1) Compress a WAV into a bitstream (default mode=coord yields better initial quality)

```
python step2_ddcm_compress_audio.py .\AudioLDM2_Music_output.wav .\evaluation_results\audioldm2_music_ddcm --T 16 --mode coord
```

2) Decompress to a WAV

```
python step3_ddcm_decompress_audio.py .\evaluation_results\audioldm2_music_ddcm --out-wav .\evaluation_results\audioldm2_music_ddcm_reconstructed.wav
```

Notes:
- The compressor internally loads `cvssp/audioldm2-music` (configurable via `--model`).
- Hugging Face cache is resolved via `HUGGINGFACE_HUB_CACHE` or `HF_HOME/hub`, falling back to `F:\Kasai_Lab\hf_cache\huggingface\hub`.
- Determinism: codebook is fixed by `K` and `seed`. Keep them identical for compression and decompression.
- Latent shape, T, K, seed, and model metadata are stored in the `.json` file.

## Parameters
- mode: coord | random
  - coord: 直接在 VAE latent 的坐标基上做稀疏编码（选取幅值最大的 T 个位置），质量稳健，推荐默认
  - random: 使用固定随机代码本做匹配追踪（需 K 与 seed），可能更稀疏但初版质量不稳定
- K: 代码本大小（仅在 mode=random 时使用）
- T: 选取的原子/坐标个数（越大越接近原始 latent）
- seed: 代码本随机种子（仅在 mode=random 时使用；解码需一致）

## Limitations and Next Steps
- This is a training-free baseline. It does not yet inject diffusion per-step variance or use the UNet to guide selection.
- Planned upgrades:
  - Per-step diffusion-style variance injection and residual-guided index selection (closer to DDCM).
  - Bitrate accounting and rate–distortion sweeps to choose (K, T) for a target bps.
  - Optional matching pursuit variants (orthogonalization, block-wise atoms, multi-head codebooks).

## Troubleshooting
- If the model attempts to download, ensure your HF cache env vars point to a local path with the required models.
- GPU is optional; CPU works but is slower. On GPU, the VAE runs in float16 automatically.