#!/usr/bin/env python3
"""Generate spectrogram comparison without matplotlib (uses Pillow)."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from PIL import Image, ImageDraw, ImageFont


def load_mono(path: Path, target_sr: int, max_sec: float | None) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))  # [C,N]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
        sr = target_sr
    wav = wav.squeeze(0)
    if max_sec:
        max_len = int(max_sec * sr)
        wav = wav[:max_len]
    return wav, sr


def spectrogram_image(wav: torch.Tensor, sr: int, n_fft: int, hop: int, top_db: float, height: int) -> Image.Image:
    spec = T.Spectrogram(n_fft=n_fft, hop_length=hop, power=2.0, center=True)(wav)
    spec_db = T.AmplitudeToDB(stype="power", top_db=top_db)(spec)  # [F,T]
    arr = spec_db.numpy()
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.resize((img.width, height), resample=Image.BILINEAR)
    return img.convert("L")


def stack_with_label(img: Image.Image, label: str, width: int) -> Image.Image:
    font = ImageFont.load_default()
    text_h = font.getbbox(label)[3] - font.getbbox(label)[1] + 4
    canvas = Image.new("RGB", (width, img.height + text_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), label, fill=(255, 255, 255), font=font)
    canvas.paste(img.convert("RGB"), (0, text_h))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference wav path")
    ap.add_argument("--test", required=True, help="Test/Reconstructed wav path")
    ap.add_argument("--out", default="evaluation_results/spec_compare_pillow.png")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--top-db", type=float, default=80.0)
    ap.add_argument("--height", type=int, default=256)
    args = ap.parse_args()

    ref, sr = load_mono(Path(args.ref), args.sr, args.max_sec)
    test, sr2 = load_mono(Path(args.test), args.sr, args.max_sec)
    assert sr == sr2
    length = min(ref.numel(), test.numel())
    ref = ref[:length]
    test = test[:length]

    ref_img = spectrogram_image(ref, sr, args.n_fft, args.hop, args.top_db, args.height)
    test_img = spectrogram_image(test, sr, args.n_fft, args.hop, args.top_db, args.height)

    width = max(ref_img.width, test_img.width)
    ref_img = ref_img.resize((width, ref_img.height))
    test_img = test_img.resize((width, test_img.height))

    ref_block = stack_with_label(ref_img, "Original", width)
    test_block = stack_with_label(test_img, "VGGish Reconstructed", width)

    stacked = Image.new("RGB", (width, ref_block.height + test_block.height))
    stacked.paste(ref_block, (0, 0))
    stacked.paste(test_block, (0, ref_block.height))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stacked.save(out_path)
    print(f"Saved spectrogram to {out_path}")


if __name__ == "__main__":
    main()
