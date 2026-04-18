#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from ddcm.step_codec import StepCodecConfig, decode_step_bitstream, encode_step_bitstream


def save_step_bitstream(bitstream: dict, out_base: str):
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(bitstream["meta"])
    meta["mode"] = "ddcm_step"
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stream = bitstream["stream"]
    np_data = {
        "mode": np.array(["ddcm_step"], dtype=object),
        "k_indices": np.array(stream["k_indices"], dtype=object),
        "shape": np.array(stream["shape"], dtype=np.int32),
    }
    if stream.get("coeffs") is not None:
        np_data["coeffs"] = np.array(stream["coeffs"], dtype=object)
    if stream.get("signs") is not None:
        np_data["signs"] = np.array(stream["signs"], dtype=object)
    if stream.get("init_latent") is not None:
        np_data["init_latent"] = np.asarray(stream["init_latent"], dtype=np.float32)
    np.savez_compressed(base.with_suffix(".npz"), **np_data)


def load_audio(path: Path, target_sr: Optional[int]) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr and sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav.squeeze(0), sr


def stft_mag(wav: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=wav.device),
        return_complex=True,
        center=True,
    )
    return spec.abs()


def mel_db(wav: torch.Tensor, sr: int, n_fft: int, hop: int, n_mels: int) -> torch.Tensor:
    mel = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        n_mels=n_mels,
        f_min=0.0,
        f_max=sr / 2,
        power=2.0,
        center=True,
        norm=None,
        mel_scale="htk",
    )(wav)
    return T.AmplitudeToDB(stype="power", top_db=80.0)(mel)


def spectral_centroid(wav: torch.Tensor, sr: int, n_fft: int, hop: int) -> torch.Tensor:
    mag = stft_mag(wav, n_fft, hop)
    freqs = torch.linspace(0, sr / 2, mag.shape[0], device=wav.device)
    denom = mag.sum(dim=0).clamp_min(1e-9)
    return (mag.T * freqs).sum(dim=1) / denom


def compare_audio(ref_path: Path, test_path: Path, sr: int = 16000, max_secs: Optional[float] = 30.0) -> Dict:
    ref_wav, sr_ref = load_audio(ref_path, sr)
    test_wav, sr_test = load_audio(test_path, sr)
    if sr_ref != sr_test:
        raise ValueError("Sample rates mismatch after resampling")

    max_len = min(ref_wav.numel(), test_wav.numel())
    if max_secs is not None:
        max_len = min(max_len, int(max_secs * sr_ref))
    ref_wav = ref_wav[:max_len]
    test_wav = test_wav[:max_len]

    diff = ref_wav - test_wav
    mse = float(torch.mean(diff ** 2).item())
    mae = float(torch.mean(diff.abs()).item())
    ref_energy = float(torch.mean(ref_wav ** 2).item())
    snr = 10.0 * math.log10((ref_energy + 1e-12) / (mse + 1e-12))

    ref_center = ref_wav - ref_wav.mean()
    test_center = test_wav - test_wav.mean()
    corr = float((ref_center * test_center).sum().item() / (ref_center.norm() * test_center.norm() + 1e-12))

    n_fft, hop, n_mels = 1024, 256, 128
    ref_mag = stft_mag(ref_wav, n_fft, hop)
    test_mag = stft_mag(test_wav, n_fft, hop)
    min_frames = min(ref_mag.shape[1], test_mag.shape[1])
    stft_mse = float(torch.mean((ref_mag[:, :min_frames] - test_mag[:, :min_frames]) ** 2).item())

    ref_mel = mel_db(ref_wav, sr_ref, n_fft, hop, n_mels)
    test_mel = mel_db(test_wav, sr_ref, n_fft, hop, n_mels)
    min_mel_frames = min(ref_mel.shape[1], test_mel.shape[1])
    mel_mae_db = float(torch.mean((ref_mel[:, :min_mel_frames] - test_mel[:, :min_mel_frames]).abs()).item())

    ref_centroid = spectral_centroid(ref_wav, sr_ref, n_fft, hop)
    test_centroid = spectral_centroid(test_wav, sr_ref, n_fft, hop)
    min_c = min(ref_centroid.shape[0], test_centroid.shape[0])
    centroid_l1 = float(torch.mean((ref_centroid[:min_c] - test_centroid[:min_c]).abs()).item())

    return {
        "sr": sr_ref,
        "duration_sec": round(max_len / sr_ref, 3),
        "waveform_mse": mse,
        "waveform_mae": mae,
        "snr_db": snr,
        "pearson_corr": corr,
        "stft_mag_mse": stft_mse,
        "mel_db_mae": mel_mae_db,
        "spectral_centroid_l1": centroid_l1,
    }


def diagnostics_summary(meta: Dict) -> Dict:
    diag = meta.get("diagnostics") or {}
    rb = np.asarray(diag.get("residual_norm_before", []), dtype=np.float64)
    ra = np.asarray(diag.get("residual_norm_after", []), dtype=np.float64)
    inj = np.asarray(diag.get("injection_norm", []), dtype=np.float64)

    if rb.size == 0 or ra.size == 0:
        return {"has_diagnostics": False}

    improvement = (rb - ra) / np.maximum(rb, 1e-12)
    improved_steps = float(np.mean((ra < rb).astype(np.float64)))
    final_ratio = float(ra[-1] / max(rb[0], 1e-12))

    return {
        "has_diagnostics": True,
        "steps": int(min(rb.size, ra.size)),
        "mean_step_improvement_ratio": float(np.mean(improvement)),
        "median_step_improvement_ratio": float(np.median(improvement)),
        "improved_steps_ratio": improved_steps,
        "residual_final_over_initial": final_ratio,
        "injection_norm_mean": float(np.mean(inj)) if inj.size else None,
        "injection_norm_last": float(inj[-1]) if inj.size else None,
    }


def verdict(encode_metrics: Dict, diag_stats: Dict) -> Dict:
    cond_metrics = (
        encode_metrics["snr_db"] > 0.0
        and encode_metrics["pearson_corr"] > 0.15
        and encode_metrics["mel_db_mae"] < 8.0
    )

    cond_diag = False
    if diag_stats.get("has_diagnostics"):
        cond_diag = (
            diag_stats["improved_steps_ratio"] > 0.65
            and diag_stats["residual_final_over_initial"] < 0.75
            and diag_stats["mean_step_improvement_ratio"] > 0.01
        )

    success = bool(cond_metrics and cond_diag)
    return {
        "encoding_effective": success,
        "rules": {
            "metrics_pass": bool(cond_metrics),
            "diagnostics_pass": bool(cond_diag),
        },
    }


def parse_coeff_levels(s: Optional[str]):
    if not s:
        return None
    raw = [item.strip() for item in s.replace(";", ",").split(",") if item.strip()]
    return tuple(float(x) for x in raw) if raw else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=str)
    p.add_argument("out_base", type=str)
    p.add_argument("--model", type=str, default="cvssp/audioldm2-music")
    p.add_argument("--K-step", type=int, default=256)
    p.add_argument("--N-step", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--guidance-gain", type=float, default=4.0)
    p.add_argument("--atoms-per-step", type=int, default=1)
    p.add_argument("--coeff-levels", type=str, default=None)
    p.add_argument("--continuous-coeff", action="store_true")
    p.add_argument("--match-epsilon", action="store_true")
    p.add_argument("--run-decode", action="store_true")
    p.add_argument("--max-secs", type=float, default=30.0)
    args = p.parse_args()

    out_base = Path(args.out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    coeff_levels = parse_coeff_levels(args.coeff_levels)
    cfg_kwargs = dict(
        K_step=int(args.K_step),
        N_step=int(args.N_step),
        seed=int(args.seed),
        guidance_gain=float(args.guidance_gain),
        atoms_per_step=max(1, int(args.atoms_per_step)),
        record_diagnostics=True,
        continuous_coeff=bool(args.continuous_coeff),
        match_epsilon=bool(args.match_epsilon),
    )
    if coeff_levels is not None:
        cfg_kwargs["coeff_levels"] = coeff_levels
    cfg = StepCodecConfig(**cfg_kwargs)

    encode_recon_path = out_base.with_name(out_base.name + "_encode_recon.wav")
    bitstream = encode_step_bitstream(
        args.wav,
        model_name=args.model,
        cfg=cfg,
        save_recon_path=str(encode_recon_path),
    )
    save_step_bitstream(bitstream, str(out_base))

    decode_wav_path = None
    if args.run_decode:
        decode_wav_path = out_base.with_suffix(".wav")
        decode_step_bitstream(bitstream, out_wav_path=str(decode_wav_path))

    ref_path = Path(args.wav)
    encode_metrics = compare_audio(ref_path, encode_recon_path, sr=16000, max_secs=args.max_secs)
    decode_metrics = compare_audio(ref_path, decode_wav_path, sr=16000, max_secs=args.max_secs) if decode_wav_path else None

    meta = bitstream["meta"]
    diag_stats = diagnostics_summary(meta)
    judge = verdict(encode_metrics, diag_stats)

    report = {
        "input_wav": str(ref_path),
        "bitstream_base": str(out_base),
        "encode_recon_wav": str(encode_recon_path),
        "decode_wav": str(decode_wav_path) if decode_wav_path else None,
        "cfg": meta.get("cfg", {}),
        "encode_metrics": encode_metrics,
        "decode_metrics": decode_metrics,
        "diagnostics_summary": diag_stats,
        "verdict": judge,
    }

    report_path = out_base.with_name(out_base.name + "_validity_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved validity report to: {report_path}")


if __name__ == "__main__":
    main()
