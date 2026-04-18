# DDCM Prototype (Zero-Training) for AudioLDM2 Latents

This prototype compresses audio by operating directly on the VAE latent using a fixed, deterministic random codebook and a greedy matching-pursuit selection. No retraining required.

- Code lives under `ddcm/`.
- CLIs: `step2_ddcm_compress_audio.py` and `step3_ddcm_decompress_audio.py`.
- Bitstream is saved as `<base>.json` (metadata) and `<base>.npz` (indices/coeffs/shape).

## Quickstart

1) 压缩 WAV 为比特流（仅支持 random 模式，已禁用 coord）

```
python step2_ddcm_compress_audio.py .\AudioLDM2_Music_output.wav .\evaluation_results\audioldm2_music_ddcm --T 16 --K 256 --seed 1234
```

2) Decompress to a WAV

```
python step3_ddcm_decompress_audio.py .\evaluation_results\audioldm2_music_ddcm --out-wav .\evaluation_results\audioldm2_music_ddcm_reconstructed.wav
```

Notes:
- The compressor internally loads `cvssp/audioldm2-music` (configurable via `--model`).
- Hugging Face cache is resolved via `HUGGINGFACE_HUB_CACHE` or `HF_HOME/hub`, falling back to `F:\\Kasai_Lab\\hf_cache\\huggingface\\hub`.
- Determinism: codebook is fixed by `K` and `seed`. Keep them identical for compression and decompression.
- Latent shape, T, K, seed, and model metadata are stored in the `.json` file.
- 实现细节：Mel 提取采用 torchaudio（不依赖 librosa）；coord 模式已移除。

## Parameters
- mode: random（唯一支持）
  - random: 使用固定随机码本做匹配追踪（需 K 与 seed）。
- K: 码本大小（random 模式必需）
- T: 选取的原子个数（越大越接近原始 latent）
- seed: 码本随机种子（解码需一致）

## Quality-First Evaluation Protocol (Frozen)

This repo now uses a single quality-first protocol for DDCM progress tracking. The protocol keeps every result directly comparable with the latent codec baseline.

- Primary objective: improve perceptual/objective quality first, then optimize rate.
- Baseline for comparison: latent codec baseline pack (fixed seed/config).
- Report both quality and size on every run (no quality-only claims).

### Required Metrics (report all)

- Waveform: `pearson_corr`, `snr_db`, `waveform_mse`
- Spectral: `mel_db_mae`, `stft_mag_mse`
- Route-level metrics (if available): `si_sdr_db`, `log_mel_mse`, `log_mel_corr`
- Size: `bitstream_total_bytes`, `ratio_vs_orig`
- Reproducibility: `model_id`, `T`, `K`, `seed`, `t_range`, and command line

### Mandatory Artifact Checklist per Run

Each run directory must include:

- `manifest.json` (full configuration + references)
- Reconstructed audio (`*_decomp.wav` or route-specific output wav)
- Objective metrics json
- Bitstream payload files
- AB package (`ab_session`) and filled score csv (human or clearly labeled proxy)

### Gate Rule

A run is considered "quality-promising" only when:

- flow-through is valid (`flow_corr >= 0.9999` and `flow_mse <= 1e-12`)
- and quality is non-regressive vs the fixed latent baseline on at least one key metric (`pearson_corr` or `mel_db_mae`) with no severe size explosion.

See `ddcm_experiments/EVAL_PROTOCOL_QUALITY_FIRST.md` for the operational checklist and reporting template.

## Limitations and Next Steps
- 当前为零训练基线：尚未在扩散每个 timestep 内进行离散噪声选择/注入，也未使用 UNet 进行前向一致性（方法三）引导。
- Planned upgrades:
  - 每步扩散的方差注入与前向一致性选择（DDCM 方法三）。
  - 码率统计与 (K, T) 的率失真扫描。
  - 匹配追踪变体（正交化、分块原子、多头码本等）。

## Troubleshooting
- If the model attempts to download, ensure your HF cache env vars point to a local path with the required models.
- GPU is optional; CPU works but is slower. On GPU, the VAE runs in float16 automatically.