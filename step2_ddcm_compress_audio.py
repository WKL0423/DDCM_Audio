#!/usr/bin/env python3
"""
Step 2: DDCM prototype compression (zero-training)

Compress a waveform into a bitstream using a fixed random codebook over the VAE latent.
Outputs a pair of files: <base>.json (meta) and <base>.npz (stream arrays).
"""

import argparse
from pathlib import Path
from ddcm.audio_codec import CodecConfig, compress_audio_to_bitstream, save_bitstream


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=str, help="input wav path")
    p.add_argument("out", type=str, help="output bitstream base path (no extension)")
    p.add_argument("--model", type=str, default="cvssp/audioldm2-music")
    p.add_argument("--K", type=int, default=256, help="codebook size (random mode)")
    p.add_argument("--T", type=int, default=16, help="number of selected atoms")
    p.add_argument("--seed", type=int, default=1234, help="codebook seed (random mode)")
    # 严格禁用 coord，默认仅支持 random
    p.add_argument("--mode", type=str, default="random", choices=["random"], help="compression mode (only random)")

    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)#确定输出目录存在

    cfg = CodecConfig(K=args.K, T=args.T, seed=args.seed, mode=args.mode)#创建编解码配置对象（仅 random）
    bitstream = compress_audio_to_bitstream(args.wav, model_name=args.model, cfg=cfg)#压缩音频为比特流
    save_bitstream(bitstream, args.out)

    print(f"Saved bitstream to: {args.out}.json and {args.out}.npz")


if __name__ == "__main__":
    main()
