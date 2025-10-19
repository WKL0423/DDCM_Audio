#!/usr/bin/env python3
"""
Step 3: DDCM prototype decompression (zero-training)

Load a bitstream (json+npz) and reconstruct the waveform via the VAE.
"""

import argparse
from pathlib import Path
from ddcm.audio_codec import load_bitstream, decompress_audio_from_bitstream
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=str, help="bitstream base path (without extension)")
    p.add_argument("--out-wav", type=str, default=None, help="output wav path")

    args = p.parse_args()
    bitstream = load_bitstream(args.base)

    wav = decompress_audio_from_bitstream(bitstream, out_wav_path=args.out_wav)
    if args.out_wav is None:
        # Default name next to base: <base>_reconstructed.wav
        base_path = Path(args.base)
        out_path = str(base_path.parent / f"{base_path.name}_reconstructed.wav")
        sr = int(bitstream["meta"].get("sr", 16000))
        sf.write(out_path, wav, sr)
        print(f"Saved reconstructed wav to: {out_path}")
    else:
        print(f"Saved reconstructed wav to: {args.out_wav}")


if __name__ == "__main__":
    main()
