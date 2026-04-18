#!/usr/bin/env python3
"""Compute VGGish embedding similarity between two audio files."""
import argparse
from pathlib import Path
import os
import torch
import torchaudio


def _prepare_vggish(device: torch.device):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torchvggish  # type: ignore
    from torchvggish import vggish_input  # type: ignore

    model = torchvggish.vggish().to(device)
    model.eval()
    return model, vggish_input.waveform_to_examples


def load_audio(path: Path, target_sr: int, max_secs: float) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.squeeze(0)
    if max_secs:
        max_len = int(max_secs * target_sr)
        if wav.numel() > max_len:
            wav = wav[:max_len]
    return wav


def embed_vggish(model, to_examples, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    audio = audio.to(torch.float32).cpu()
    with torch.no_grad():
        examples = to_examples(audio.numpy(), sample_rate)
        if isinstance(examples, torch.Tensor):
            examples_tensor = examples.to(next(model.parameters()).device, torch.float32)
        else:
            examples_tensor = torch.from_numpy(examples).to(next(model.parameters()).device, torch.float32)
        feats = model(examples_tensor)
    if feats.ndim == 1:
        feats = feats.unsqueeze(0)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    feats = torch.nn.functional.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
    return feats


def main():
    parser = argparse.ArgumentParser(description="Compute VGGish cosine similarity between two clips")
    parser.add_argument("--ref", required=True, type=str, help="Reference audio path")
    parser.add_argument("--test", required=True, type=str, help="Test audio path")
    parser.add_argument("--max-secs", type=float, default=30.0, help="Maximum audio duration in seconds")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ref_path = Path(args.ref)
    test_path = Path(args.test)
    if not ref_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing input audio file")

    target_sr = 16000
    ref_audio = load_audio(ref_path, target_sr, args.max_secs).clamp(-1.0, 1.0)
    test_audio = load_audio(test_path, target_sr, args.max_secs).clamp(-1.0, 1.0)

    device = torch.device(args.device)
    model, to_examples = _prepare_vggish(device)

    ref_embed = embed_vggish(model, to_examples, ref_audio, target_sr)
    test_embed = embed_vggish(model, to_examples, test_audio, target_sr)

    cosine = torch.sum(ref_embed * test_embed).item()
    l2 = torch.nn.functional.pairwise_distance(ref_embed, test_embed, p=2).item()

    print({
        "sampling_rate": target_sr,
        "max_duration_sec": args.max_secs,
        "cosine_similarity": cosine,
        "l2_distance": l2,
    })


if __name__ == "__main__":
    main()
