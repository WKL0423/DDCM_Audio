from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .codebook import create_codebook
from .runner import compress_latent, decompress_latent


@dataclass
class CodecConfig:
    K: int = 256
    T: int = 16
    seed: int = 1234
    mode: str = "coord"  # "coord" (coordinate basis) or "random" (requires codebook)


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
    # latent_shape is [B,C,H,W] from VAE encode; use [C,H,W]
    C, H, W = int(latent_shape[1]), int(latent_shape[2]), int(latent_shape[3])
    cb = create_codebook((C, H, W), K=cfg.K, seed=cfg.seed, device=device, dtype=dtype)
    return cb


def _load_audio_mel_and_latent(pipeline, wav_path: str, device: torch.device) -> Dict:
    import torchaudio

    # Settings aligned with step1_training_matched_vae_reconstruction_fixed
    sr_target = 16000
    duration = 10.24
    hop_length = 160

    wav, sr = torchaudio.load(wav_path)
    if sr != sr_target:
        wav = torchaudio.transforms.Resample(sr, sr_target)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    max_len = int(duration * sr_target)
    if wav.shape[-1] > max_len:
        wav = wav[..., :max_len]
    wav_np = wav.squeeze(0).numpy()

    # Reuse pipeline's VAE-friendly mel via its own methods if available.
    # Here we mimic by calling the VAE.encode directly on precomputed mel from step1 util if present.
    # For simplicity, we reuse the VAE path: compute mel via pipeline.vae expected input.
    # The pipeline expects log-mel [B,1,T,F]; we'll reconstruct it by using its own helper if exists.
    # Fallback: use a minimal mel extraction similar to step1.
    try:
        from step1_training_matched_vae_reconstruction_fixed import TrainingMatchedVAEReconstructor

        recon = TrainingMatchedVAEReconstructor()
        mel = recon.extract_mel_spectrogram(wav_np).to(device)
        if device.type == "cuda":
            mel = mel.half()
    except Exception:
        # Minimal mel using librosa
        import librosa
        mels = librosa.feature.melspectrogram(
            y=wav_np,
            sr=sr_target,
            n_fft=1024,
            hop_length=hop_length,
            win_length=1024,
            n_mels=64,
            fmin=0,
            fmax=8000,
            center=False,
            power=2.0,
        )
        mel = torch.log(torch.clamp(torch.from_numpy(mels).float(), min=1e-5)).t().unsqueeze(0).unsqueeze(0).to(device)
        if device.type == "cuda":
            mel = mel.half()

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

    codebook = None
    if (cfg.mode or "coord").lower() == "random":
        codebook = build_codebook(latent.shape, cfg, device, latent.dtype)
    stream = compress_latent(latent, codebook, T=cfg.T, mode=cfg.mode)

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
    codebook = None
    if (cfg.mode or "coord").lower() == "random":
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
