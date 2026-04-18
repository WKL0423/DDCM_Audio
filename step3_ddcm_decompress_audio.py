#!/usr/bin/env python3
"""
Step 3: DDCM 原型解码（零训练）

说明：严格仅支持 random（码本）模式。若比特流为 coord，将直接报错并退出。
从 json+npz 载入比特流，通过 VAE 重构音频。
"""

import argparse
from pathlib import Path
from ddcm.audio_codec import load_bitstream, decompress_audio_from_bitstream
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=str, help="比特流基路径（不含扩展名）")
    p.add_argument("--out-wav", type=str, default=None, help="输出 wav 路径（可选）")

    args = p.parse_args()
    bitstream = load_bitstream(args.base)

    # 模式校验：仅允许 random
    meta_mode = str(bitstream.get("meta", {}).get("mode", bitstream.get("stream", {}).get("mode", "random"))).lower()
    if meta_mode != "random":
        raise SystemExit("当前解码器仅支持 random 模式。请使用 --mode random 重新压缩生成比特流。")

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
