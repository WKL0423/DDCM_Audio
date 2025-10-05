import os
import json
import time
import itertools
import argparse
from typing import Dict, Any
import numpy as np
import librosa
import torch
import soundfile as sf
from pathlib import Path
from New_pipeline_audioldm2 import AudioLDM2Pipeline


def resolve_cache_dir():
    hf_home = os.environ.get("HF_HOME")
    hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        return hf_hub_cache
    if hf_home:
        return os.path.join(hf_home, "hub")
    return r"F:\\Kasai_Lab\\hf_cache\\huggingface\\hub"


def run_once(model, sampler, steps, gs, prompt, length, seed, cache_dir, device, out_dir):
    pipe = AudioLDM2Pipeline.from_pretrained(
        model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=cache_dir,
    ).to(device)

    if sampler != "default":
        try:
            pipe.set_sampler(sampler, use_karras_sigmas=True)
        except Exception as e:
            print(f"[benchmark] set_sampler failed for {sampler}: {e}")

    g = torch.Generator(device).manual_seed(seed)
    t0 = time.time()
    audio = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=gs,
        audio_length_in_s=length,
        generator=g,
        num_waveforms_per_prompt=1,
    ).audios[0]
    dt = time.time() - t0

    # Save wav
    model_tag = Path(str(model)).name.replace('/', '_') if isinstance(model, str) else "model"
    fname = f"{model_tag}__{sampler}_s{steps}_gs{gs}_seed{seed}.wav"
    out_path = out_dir / fname
    sf.write(str(out_path), audio, 16000)

    # Compute metrics
    metrics = compute_metrics(audio, sr=16000)

    return {
        "model": model,
        "sampler": sampler,
        "steps": steps,
        "guidance_scale": gs,
        "seed": seed,
        "length": length,
        "device": device,
        "runtime_sec": dt,
        "output_path": str(out_path),
        "metrics": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("AUDIO_LDM2_MODEL_DIR", "cvssp/audioldm2-music"))
    ap.add_argument("--samplers", nargs="*", default=["default", "dpmpp", "unipc"])
    ap.add_argument("--steps", nargs="*", type=int, default=[16, 32, 50, 100])
    ap.add_argument("--gs", type=float, default=3.5)
    ap.add_argument("--prompt", default="Techno music with a strong, upbeat tempo and high melodic riffs")
    ap.add_argument("--length", type=float, default=10.24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="evaluation_results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = resolve_cache_dir()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sampler, steps in itertools.product(args.samplers, args.steps):
        print(f"\n[benchmark] model={args.model} sampler={sampler} steps={steps} gs={args.gs} device={device}")
        metrics = run_once(
            model=args.model,
            sampler=sampler,
            steps=steps,
            gs=args.gs,
            prompt=args.prompt,
            length=args.length,
            seed=args.seed,
            cache_dir=cache_dir,
            device=device,
            out_dir=out_dir,
        )
        results.append(metrics)

    # Save json
    ts = int(time.time())
    json_path = out_dir / f"benchmark_sampler_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark to {json_path}")


def compute_metrics(audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """Compute objective metrics without reference.

    Includes:
      - amplitude stats: rms, peak, crest, clipping_ratio
      - zero-crossing rate (mean)
      - frequency stats via FFT: spectral centroid, rolloff@0.85, eff_bandwidth@0.95,
        energy ratios in low/mid/high bands (0-2k, 2-4k, 4-8k)
      - mel stats (64 bins, 0-8k): mean/std of log-mel, high-mel energy ratio, entropy
    """
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    eps = 1e-12
    # Amplitude stats
    rms = float(np.sqrt(np.mean(y ** 2) + eps))
    peak = float(np.max(np.abs(y) + eps))
    crest = float(peak / max(rms, eps))
    clipping_ratio = float(np.mean(np.abs(y) >= 0.999))

    # ZCR
    zc = librosa.feature.zero_crossing_rate(y=y, frame_length=1024, hop_length=256)
    zcr_mean = float(np.mean(zc))

    # FFT-based metrics
    n = len(y)
    Y = np.fft.rfft(y)
    mag = np.abs(Y)
    pow_spec = mag ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total_power = float(np.sum(pow_spec) + eps)

    centroid = float(np.sum(freqs * mag) / (np.sum(mag) + eps))
    cumsum = np.cumsum(pow_spec)
    roll_idx = int(np.searchsorted(cumsum, 0.85 * total_power))
    rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])
    bw_idx = int(np.searchsorted(cumsum, 0.95 * total_power))
    eff_bandwidth = float(freqs[min(bw_idx, len(freqs) - 1)])

    nyq = sr / 2.0
    low = pow_spec[freqs <= nyq * 0.25]
    mid = pow_spec[(freqs > nyq * 0.25) & (freqs <= nyq * 0.5)]
    high = pow_spec[freqs > nyq * 0.5]
    low_ratio = float(np.sum(low) / total_power)
    mid_ratio = float(np.sum(mid) / total_power)
    high_ratio = float(np.sum(high) / total_power)

    # Mel stats (training-aligned params)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        window="hann",
        center=False,
        n_mels=64,
        fmin=0,
        fmax=8000,
        power=2.0,
    )
    log_mel = np.log(np.maximum(mel, 1e-10))
    mel_mean = float(np.mean(log_mel))
    mel_std = float(np.std(log_mel))
    # High-mel energy ratio (top 1/3 bins)
    k = mel.shape[0] // 3
    mel_high_ratio = float(np.sum(mel[-k:, :]) / (np.sum(mel) + eps))
    # Spectral entropy on mel (frame-wise then average)
    mel_sum = np.sum(mel, axis=0, keepdims=True) + eps
    mel_p = mel / mel_sum
    mel_entropy = float(-np.mean(np.sum(mel_p * np.log(mel_p + eps), axis=0)))

    return {
        "amplitude": {
            "rms": rms,
            "peak": peak,
            "crest": crest,
            "clipping_ratio": clipping_ratio,
            "zcr_mean": zcr_mean,
        },
        "frequency": {
            "spectral_centroid_hz": centroid,
            "rolloff_0.85_hz": rolloff,
            "effective_bandwidth_0.95_hz": eff_bandwidth,
            "band_energy_ratio": {
                "low_0_2k": low_ratio,
                "mid_2k_4k": mid_ratio,
                "high_4k_8k": high_ratio,
            },
        },
        "mel": {
            "log_mel_mean": mel_mean,
            "log_mel_std": mel_std,
            "high_mel_energy_ratio": mel_high_ratio,
            "entropy": mel_entropy,
        },
    }


if __name__ == "__main__":
    main()
