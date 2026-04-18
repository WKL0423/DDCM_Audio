#!/usr/bin/env python3
"""
Step 3b: DDCM per-step 解码（仅 random 码本，码本持久化本地，不进比特流）
读取 <base>.json/.npz 并按每步索引在扩散循环中注入噪声，输出重建 WAV。
"""
import argparse
from pathlib import Path
import json
import numpy as np

from ddcm.step_codec import decode_step_bitstream


def load_step_bitstream(base_path: str):
    base = Path(base_path)
    meta = json.loads((base.with_suffix('.json')).read_text(encoding='utf-8'))
    npz = np.load(base.with_suffix('.npz'), allow_pickle=True)
    mode = meta.get('mode') or (npz['mode'][0].item() if hasattr(npz['mode'], 'shape') else str(npz['mode']))
    if str(mode).lower() != 'ddcm_step':
        raise SystemExit('比特流模式不是 ddcm_step')
    # 命名与论文一致：k_i 序列。兼容旧字段名 'indices'。
    k_indices = None
    if 'k_indices' in npz.files:
        k_indices = npz['k_indices'].tolist()
    elif 'indices' in npz.files:
        k_indices = npz['indices'].tolist()
    else:
        raise SystemExit('比特流缺少 k_indices/indices 字段')

    stream = {
        'mode': 'ddcm_step',
        'k_indices': k_indices,
        'coeffs': npz['coeffs'].tolist() if 'coeffs' in npz.files else None,
        'signs': npz['signs'].tolist() if 'signs' in npz.files else None,
        'shape': npz['shape'].tolist(),
        'init_latent': npz['init_latent'] if 'init_latent' in npz.files else None,
    }
    return {'meta': meta, 'stream': stream}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base', type=str, help='比特流基路径（不含扩展名）')
    p.add_argument('--out-wav', type=str, default=None, help='输出 wav 路径（可选）')
    args = p.parse_args()

    bitstream = load_step_bitstream(args.base)
    out = decode_step_bitstream(bitstream, out_wav_path=args.out_wav)
    print(f'Saved reconstructed wav to: {out}')


if __name__ == '__main__':
    main()
