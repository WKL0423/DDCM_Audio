from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .codebook import load_or_create_codebook
from .runner import compress_latent, decompress_latent


@dataclass
class CodecConfig:
    K: int = 256
    T: int = 16
    seed: int = 1234
    # 说明：严格禁用 coord 模式，默认改为 random（基于码本的匹配追踪）
    mode: str = "random"  # 可选："random"（需要码本）；不再使用 "coord"


def _resolve_cache_dir() -> str:
    import os

    env_hf_home = os.environ.get("HF_HOME")
    env_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_hub_cache:
        return env_hub_cache
    if env_hf_home:
        return str(Path(env_hf_home) / "hub")
    return r"F:\\Kasai_Lab\\hf_cache\\huggingface\\hub"


def build_codebook(latent_shape: torch.Size, cfg: CodecConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    构建/加载本地持久化码本：不存入比特流，仅在本地 codebooks/ 下缓存。
    """
    C, H, W = int(latent_shape[1]), int(latent_shape[2]), int(latent_shape[3])
    cb = load_or_create_codebook((C, H, W), K=cfg.K, seed=cfg.seed, device=device, dtype=dtype)
    return cb


def _load_audio_mel_and_latent(pipeline, wav_path: str, device: torch.device) -> Dict:
    """
    加载 wav 并提取与训练流程一致的对数 Mel 频谱（不依赖 librosa）。
    参考 STEP1：
    - 先做反射 padding，长度为 (n_fft - hop)//2
    - torch.stft(..., center=False, hann window, return_complex=True)
    - 取幅度谱 |STFT|
    - 乘以 Slaney 标定的 Mel 滤波器（torchaudio 实现），n_mels=64, fmax=8000
    - log(clamp(., 1e-5))
    - 按目标帧长 padding/裁剪
    输出形状 [1,1,T,64]
    """
    import torchaudio
    import torch.nn.functional as F

    sr_target = 16000
    duration = 10.24
    hop_length = 160
    n_fft = 1024
    win_length = 1024
    n_mels = 64
    f_min, f_max = 0.0, 8000.0

    # 读取并重采样到 16k，混合为单声道，裁剪到固定长度
    wav, sr = torchaudio.load(wav_path)
    if sr != sr_target:
        wav = torchaudio.transforms.Resample(sr, sr_target)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    max_len = int(duration * sr_target)
    if wav.shape[-1] > max_len:
        wav = wav[..., :max_len]
    wav_np = wav.squeeze(0).numpy()

    # 反射 padding（与训练一致）
    pad = int((n_fft - hop_length) // 2)
    wav_pad = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    # STFT 幅度谱（center=False）
    window = torch.hann_window(win_length, device=wav.device)
    stft = torch.stft(
        wav_pad,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )  # [1, freq, frames]
    mag = torch.abs(stft)

    # Mel 滤波（Slaney 标定）；将 [freq, frames] -> [n_mels, frames]
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sr_target,
        f_min=f_min,
        f_max=f_max,
        n_stft=n_fft // 2 + 1,
        norm="slaney",
        mel_scale="slaney",
    )
    mel_spec = mel_scale(mag)  # [1, n_mels, frames]
    mel_log = torch.log(torch.clamp(mel_spec, min=1e-5))

    # 按目标帧长 padding/裁剪
    target_len = int(duration * sr_target / hop_length)
    T_cur = mel_log.shape[-1]
    if T_cur < target_len:
        mel_log = F.pad(mel_log, (0, target_len - T_cur))
    elif T_cur > target_len:
        mel_log = mel_log[..., :target_len]

    # 调整为 [B,1,T,F]
    mel = mel_log.permute(0, 2, 1).unsqueeze(1).to(device)
    try:
        vae_dtype = next(pipeline.vae.parameters()).dtype
    except Exception:
        vae_dtype = torch.float16 if device.type == "cuda" else torch.float32
    mel = mel.to(dtype=vae_dtype)

    with torch.no_grad():
        enc = pipeline.vae.encode(mel)
        latent = (enc.latent_dist.mode() if hasattr(enc, "latent_dist") else enc.mode())
        latent = latent * pipeline.vae.config.scaling_factor
    return {"mel": mel, "latent": latent, "sr": sr_target, "wav_np": wav_np}


def _decode_latent_to_audio(pipeline, latent: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        lat_for_dec = latent / pipeline.vae.config.scaling_factor
        vae_dtype = next(pipeline.vae.parameters()).dtype
        lat_for_dec = lat_for_dec.to(vae_dtype)
        mel_rec = pipeline.vae.decode(lat_for_dec).sample
        wav = pipeline.mel_spectrogram_to_waveform(mel_rec).squeeze().cpu().numpy()
    return wav


def compress_audio_to_bitstream(
    wav_path: str,
    model_name: str = "cvssp/audioldm2-music",
    cfg: Optional[CodecConfig] = None,
) -> Dict:
    """
    Compress a waveform into a bitstream (JSON+NPZ dict) using fixed codebook and VAE latent.
    Returns a dict containing metadata and the stream, without writing to disk.
    """
    from New_pipeline_audioldm2 import AudioLDM2Pipeline

    cfg = cfg or CodecConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = AudioLDM2Pipeline.from_pretrained(model_name, torch_dtype=dtype, cache_dir=_resolve_cache_dir()).to(device)

    io = _load_audio_mel_and_latent(pipe, wav_path, device)
    latent = io["latent"]  # [1,C,H,W]

    # 仅允许 random 模式（基于固定码本）
    mode_lower = (cfg.mode or "random").lower()
    if mode_lower != "random":
        raise ValueError("当前实现禁止使用 coord 模式，请使用 --mode random")

    codebook = build_codebook(latent.shape, cfg, device, latent.dtype)
    stream = compress_latent(latent, codebook, T=cfg.T, mode="random")

    meta = {
        "model_name": model_name,
        "cfg": asdict(cfg),
        "latent_shape": list(latent.shape),
        "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
        "sr": int(io["sr"]),
    }
    return {"meta": meta, "stream": stream}


def decompress_audio_from_bitstream(bitstream: Dict, out_wav_path: Optional[str] = None) -> np.ndarray:
    """
    Reconstruct waveform from a bitstream produced by `compress_audio_to_bitstream`.
    If out_wav_path is provided, saves the waveform to disk.
    """
    from New_pipeline_audioldm2 import AudioLDM2Pipeline
    import soundfile as sf

    meta = bitstream["meta"]
    stream = bitstream["stream"]
    cfg = CodecConfig(**meta["cfg"])  # recreate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = AudioLDM2Pipeline.from_pretrained(meta["model_name"], torch_dtype=dtype, cache_dir=_resolve_cache_dir()).to(device)

    # rebuild codebook and latent
    C, H, W = stream["shape"]
    latent_shape = [1, C, H, W]
    # 仅允许 random 模式；为兼容旧数据，这里若遇到 coord 会给出清晰错误
    mode_lower = (cfg.mode or "random").lower()
    if mode_lower != "random":
        raise ValueError("比特流 meta 指示为 coord 模式，当前版本不再支持该模式")

    codebook = build_codebook(torch.Size(latent_shape), cfg, device, dtype)
    latent_rec = decompress_latent(stream, codebook, device=device, dtype=dtype)

    wav = _decode_latent_to_audio(pipe, latent_rec)

    if out_wav_path:
        sr = int(meta.get("sr", 16000))
        sf.write(str(out_wav_path), wav, sr)
    return wav


def save_bitstream(bitstream: Dict, out_path: str) -> None:
    out_path = str(out_path)
    base = Path(out_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    # Save meta JSON (include stream mode for convenience)
    meta = dict(bitstream["meta"])  # shallow copy
    meta["mode"] = bitstream["stream"].get("mode", meta.get("mode", "coord"))
    (base.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save stream arrays depending on mode
    mode = bitstream["stream"].get("mode", "coord")
    if mode == "coord":
        np.savez_compressed(
            base.with_suffix(".npz"),
            mode=np.array(["coord"]),
            positions=np.array(bitstream["stream"]["positions"], dtype=np.int64),
            values=np.array(bitstream["stream"]["values"], dtype=np.float32),
            shape=np.array(bitstream["stream"]["shape"], dtype=np.int32),
            T=np.array([bitstream["stream"]["T"]], dtype=np.int32),
        )
    else:  # random
        np.savez_compressed(
            base.with_suffix(".npz"),
            mode=np.array(["random"]),
            indices=np.array(bitstream["stream"]["indices"], dtype=np.int32),
            coeffs=np.array(bitstream["stream"]["coeffs"], dtype=np.float32),
            shape=np.array(bitstream["stream"]["shape"], dtype=np.int32),
            T=np.array([bitstream["stream"]["T"]], dtype=np.int32),
            K=np.array([bitstream["stream"].get("K", 0)], dtype=np.int32),
        )


def load_bitstream(base_path: str) -> Dict:
    base = Path(base_path)
    meta = json.loads((base.with_suffix(".json")).read_text(encoding="utf-8"))
    npz = np.load(base.with_suffix(".npz"), allow_pickle=True)
    mode = meta.get("mode")
    if mode is None:
        arr = npz.get("mode")
        if arr is None:
            mode = "coord"
        else:
            try:
                mode = arr[0].item() if hasattr(arr, "shape") else str(arr)
            except Exception:
                mode = str(arr)

    if mode == "coord":
        stream = {
            "mode": "coord",
            "positions": npz["positions"].tolist(),
            "values": npz["values"].tolist(),
            "shape": npz["shape"].tolist(),
            "T": int(npz["T"][0]),
        }
    else:
        stream = {
            "mode": "random",
            "indices": npz["indices"].tolist(),
            "coeffs": npz["coeffs"].tolist(),
            "shape": npz["shape"].tolist(),
            "T": int(npz["T"][0]),
            "K": int(npz.get("K", np.array([0]))[0]),
        }
    return {"meta": meta, "stream": stream}
