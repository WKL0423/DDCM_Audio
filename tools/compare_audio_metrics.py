#!/usr/bin/env python3
"""
Compute objective comparison metrics between two audio files.

Metrics:
  - Length (seconds, aligned)
  - Waveform MSE / MAE
  - SNR (dB)
  - Pearson correlation (waveform)
  - STFT magnitude MSE (n_fft=1024, hop=256)
  - Mel Spectrogram dB MAE (n_mels=128, same STFT params)
  - Spectral centroid L1 difference (mean over frames)

Usage:
  python tools/compare_audio_metrics.py \
    --ref .\\AudioLDM2_Music_output.wav \
    --test .\\evaluation_results\\music_output_ddcm_step_1000x1000_gpu.wav \
    --sr 16000 --max-secs 30

Notes:
  - Resamples both to --sr if provided.
  - Crops both to min(len(ref), len(test), max-secs*sr) for fair comparison.
  - Prints a JSON-like block for easy copy.
"""
import argparse
from pathlib import Path
import json
import math
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional


def load_audio(path: Path, target_sr: Optional[int]):
    wav, sr = torchaudio.load(str(path))  # [C, N]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr and sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav.squeeze(0), sr  # [N], sr


def stft_mag(wav: torch.Tensor, n_fft: int, hop: int):
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=torch.hann_window(n_fft, device=wav.device), return_complex=True, center=True)
    mag = spec.abs()
    return mag  # [freq, frames]


def mel_db(wav: torch.Tensor, sr: int, n_fft: int, hop: int, n_mels: int):
    mel = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft, n_mels=n_mels, f_min=0.0, f_max=sr/2, power=2.0, center=True, norm=None, mel_scale="htk")(wav)
    mel_db = T.AmplitudeToDB(stype="power", top_db=80.0)(mel)
    return mel_db  # [n_mels, frames]


def spectral_centroid(wav: torch.Tensor, sr: int, n_fft: int, hop: int):
    # Use magnitude spectrum weights to compute centroid per frame
    mag = stft_mag(wav, n_fft, hop)
    freqs = torch.linspace(0, sr/2, mag.shape[0], device=wav.device)
    # Avoid division by zero
    denom = mag.sum(dim=0).clamp_min(1e-9)
    centroid = (mag.T * freqs).sum(dim=1) / denom  # [frames]
    return centroid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', type=str, required=True, help='Reference (original) audio file path')
    ap.add_argument('--test', type=str, required=True, help='Test (reconstructed) audio file path')
    ap.add_argument('--sr', type=int, default=16000, help='Target sample rate for comparison')
    ap.add_argument('--max-secs', type=float, default=None, help='Optional max seconds to crop')
    ap.add_argument('--n-fft', type=int, default=1024)
    ap.add_argument('--hop', type=int, default=256)
    ap.add_argument('--n-mels', type=int, default=128)
    args = ap.parse_args()

    ref_path = Path(args.ref)
    test_path = Path(args.test)
    if not ref_path.exists() or not test_path.exists():
        raise FileNotFoundError('Missing input file')

    ref_wav, sr_ref = load_audio(ref_path, args.sr)
    test_wav, sr_test = load_audio(test_path, args.sr)

    assert sr_ref == sr_test, 'Sample rates must match after resampling'
    sr = sr_ref

    max_len = min(ref_wav.numel(), test_wav.numel())
    if args.max_secs:
        max_allowed = int(args.max_secs * sr)
        max_len = min(max_len, max_allowed)
    ref_wav = ref_wav[:max_len]
    test_wav = test_wav[:max_len]

    # Waveform metrics
    diff = ref_wav - test_wav
    mse = float(torch.mean(diff ** 2).item())
    mae = float(torch.mean(diff.abs()).item())
    ref_energy = float(torch.mean(ref_wav ** 2).item())
    snr = 10.0 * math.log10((ref_energy + 1e-12) / (mse + 1e-12))
    # Pearson correlation
    ref_center = ref_wav - ref_wav.mean()
    test_center = test_wav - test_wav.mean()
    corr = float((ref_center * test_center).sum().item() / (ref_center.norm() * test_center.norm() + 1e-12))

    # STFT magnitude
    ref_mag = stft_mag(ref_wav, args.n_fft, args.hop)
    test_mag = stft_mag(test_wav, args.n_fft, args.hop)
    # Align frames (should match)
    min_frames = min(ref_mag.shape[1], test_mag.shape[1])
    ref_mag = ref_mag[:, :min_frames]
    test_mag = test_mag[:, :min_frames]
    stft_mse = float(torch.mean((ref_mag - test_mag) ** 2).item())

    # Mel dB
    ref_mel = mel_db(ref_wav, sr, args.n_fft, args.hop, args.n_mels)
    test_mel = mel_db(test_wav, sr, args.n_fft, args.hop, args.n_mels)
    min_mel_frames = min(ref_mel.shape[1], test_mel.shape[1])
    ref_mel = ref_mel[:, :min_mel_frames]
    test_mel = test_mel[:, :min_mel_frames]
    mel_mae_db = float(torch.mean((ref_mel - test_mel).abs()).item())

    # Spectral centroid
    ref_centroid = spectral_centroid(ref_wav, sr, args.n_fft, args.hop)
    test_centroid = spectral_centroid(test_wav, sr, args.n_fft, args.hop)
    min_c_frames = min(ref_centroid.shape[0], test_centroid.shape[0])
    ref_centroid = ref_centroid[:min_c_frames]
    test_centroid = test_centroid[:min_c_frames]
    centroid_l1 = float(torch.mean((ref_centroid - test_centroid).abs()).item())

    metrics = {
        'sr': sr,
        'duration_sec': round(max_len / sr, 3),
        'waveform_mse': mse,
        'waveform_mae': mae,
        'snr_db': snr,
        'pearson_corr': corr,
        'stft_mag_mse': stft_mse,
        'mel_db_mae': mel_mae_db,
        'spectral_centroid_l1': centroid_l1,
        'n_fft': args.n_fft,
        'hop': args.hop,
        'n_mels': args.n_mels,
    }

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
