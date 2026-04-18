#!/usr/bin/env python3
"""
Plot spectrogram/mel-spectrogram comparisons for multiple audio files.

Usage examples:
  python tools/plot_spectrogram_compare.py \
    --wav ".\\AudioLDM2_Music_output.wav:Original" \
    --wav ".\\evaluation_results\\music_output_ddcm_step_50_gpu.wav:DDCM 50-step" \
    --out .\\evaluation_results\\music_output_compare_50.png \
    --mel --sr 16000 --n-fft 1024 --hop 256 --top-db 80 --max-secs 10

Notes:
- Uses torchaudio only (no librosa). Works with mono mixdown if input is stereo.
- For fair comparison, you can optionally crop with --max-secs to focus on the first N seconds.
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt


def parse_wav_arg(arg: str) -> Tuple[Path, str]:
    # Format: path[:title]
    if ":" in arg:
        p, title = arg.split(":", 1)
    else:
        p, title = arg, Path(arg).stem
    return Path(p), title


def load_audio(path: Path, target_sr: int | None) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))  # [C, N]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr and sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav.squeeze(0), sr  # [N], sr


def compute_spec(
    wav: torch.Tensor,
    sr: int,
    mel: bool,
    n_fft: int,
    hop: int,
    win_length: int | None,
    n_mels: int,
    top_db: float,
) -> Tuple[torch.Tensor, List[float], List[float]]:
    win_length = win_length or n_fft
    if mel:
        spec = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sr / 2,
            power=2.0,
            center=True,
            norm=None,
            mel_scale="htk",
        )(wav)
        spec_db = T.AmplitudeToDB(stype="power", top_db=top_db)(spec)
        # y-axis is mel bins 0..n_mels-1; x-axis frames
        y_ticks = list(range(0, n_mels, max(1, n_mels // 10)))
        x_frames = spec_db.shape[-1]
        x_ticks = list(range(0, x_frames, max(1, x_frames // 10)))
        return spec_db, x_ticks, y_ticks
    else:
        spec = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            power=2.0,
            center=True,
        )(wav)
        spec_db = T.AmplitudeToDB(stype="power", top_db=top_db)(spec)
        # Frequency bins up to Nyquist
        n_freq = spec_db.shape[0]
        y_ticks = list(range(0, n_freq, max(1, n_freq // 10)))
        x_frames = spec_db.shape[-1]
        x_ticks = list(range(0, x_frames, max(1, x_frames // 10)))
        return spec_db, x_ticks, y_ticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", action="append", required=True, help="Path[:Title] for each audio; repeatable")
    ap.add_argument("--out", type=str, default=None, help="Output PNG path; default under evaluation_results/")
    ap.add_argument("--sr", type=int, default=None, help="Target sample rate (e.g., 16000); no resample if omitted")
    ap.add_argument("--mel", action="store_true", help="Use mel-spectrogram instead of linear spectrogram")
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win-length", type=int, default=None)
    ap.add_argument("--n-mels", type=int, default=128)
    ap.add_argument("--top-db", type=float, default=80.0)
    ap.add_argument("--max-secs", type=float, default=None, help="Crop to first N seconds for plotting")
    args = ap.parse_args()

    items = [parse_wav_arg(w) for w in args.wav]

    waves: list[torch.Tensor] = []
    srs: list[int] = []
    titles: list[str] = []
    min_len: int | None = None

    for p, title in items:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        wav, sr = load_audio(p, args.sr)
        if args.max_secs:
            max_len = int(args.max_secs * sr)
            wav = wav[:max_len]
        if min_len is None or wav.numel() < min_len:
            min_len = wav.numel()
        waves.append(wav)
        srs.append(sr)
        titles.append(title)

    # Align length for fair comparison (crop to min length)
    min_len = min_len or min(w.numel() for w in waves)
    waves = [w[:min_len] for w in waves]

    # Use the first SR for transforms (assumed same after optional resample)
    sr0 = srs[0]

    # Prepare plots
    n = len(waves)
    plt.figure(figsize=(12, 3.2 * n), dpi=120)

    for i, (wav, title) in enumerate(zip(waves, titles), start=1):
        spec_db, x_ticks, y_ticks = compute_spec(
            wav, sr0, args.mel, args.n_fft, args.hop, args.win_length, args.n_mels, args.top_db
        )
        ax = plt.subplot(n, 1, i)
        # Build time axis in seconds for x ticks
        n_frames = spec_db.shape[-1]
        duration = wav.numel() / sr0
        times = [int(round(x * duration / max(1, n_frames - 1))) for x in x_ticks]
        im = ax.imshow(
            spec_db.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel bin" if args.mel else "Freq bin")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(times)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="dB")

    plt.tight_layout()

    out = args.out
    if not out:
        names = "_vs_".join([Path(p).stem for p, _ in items])
        out_dir = Path("evaluation_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = str(out_dir / f"spec_compare_{'mel' if args.mel else 'linear'}_{names}.png")

    plt.savefig(out, bbox_inches="tight")
    print(f"Saved spectrogram comparison to: {out}")


if __name__ == "__main__":
    main()
