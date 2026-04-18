#!/usr/bin/env python3
"""Compute CLAP embedding similarity between two audio files.

Outputs cosine similarity (higher is better) and L2 distance (lower is better).
"""
import argparse
from pathlib import Path
import torch
import torchaudio
from transformers import ClapProcessor, ClapModel


def load_mono_audio(path: Path, target_sr: int):
    wav, sr = torchaudio.load(str(path))  # [C, N]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Compute CLAP similarities between reference and test audio")
    parser.add_argument("--ref", type=str, required=True, help="Reference audio path")
    parser.add_argument("--test", type=str, required=True, help="Test audio path")
    parser.add_argument("--model", type=str, default="laion/clap-htsat-unfused", help="CLAP model id")
    parser.add_argument("--max-secs", type=float, default=30.0, help="Max seconds to use from each clip")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    args = parser.parse_args()

    ref_path = Path(args.ref)
    test_path = Path(args.test)
    if not ref_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing input audio file")

    processor = ClapProcessor.from_pretrained(args.model)
    model = ClapModel.from_pretrained(args.model, use_safetensors=True).to(args.device)
    model.eval()

    target_sr = processor.feature_extractor.sampling_rate

    ref = load_mono_audio(ref_path, target_sr)
    test = load_mono_audio(test_path, target_sr)

    max_len = int(args.max_secs * target_sr)
    if ref.numel() > max_len:
        ref = ref[:max_len]
    if test.numel() > max_len:
        test = test[:max_len]

    ref_inputs = processor(audio=ref.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")
    test_inputs = processor(audio=test.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")

    with torch.no_grad():
        ref_emb = model.get_audio_features(**{k: v.to(args.device) for k, v in ref_inputs.items()})
        test_emb = model.get_audio_features(**{k: v.to(args.device) for k, v in test_inputs.items()})

    ref_emb = torch.nn.functional.normalize(ref_emb, dim=-1)
    test_emb = torch.nn.functional.normalize(test_emb, dim=-1)

    cosine = torch.sum(ref_emb * test_emb).item()
    l2 = torch.nn.functional.pairwise_distance(ref_emb, test_emb, p=2).item()

    print({
        "model": args.model,
        "sampling_rate": target_sr,
        "max_duration_sec": args.max_secs,
        "cosine_similarity": cosine,
        "l2_distance": l2,
    })


if __name__ == "__main__":
    main()
