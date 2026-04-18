#!/usr/bin/env python3
"""
Step 2b: DDCM per-step 编码（方法三，贪心式(7) 按残差内积选 k_i），步数默认 1000。
使用本地固定码本（不入比特流），输出 JSON+NPZ（模式 ddcm_step）。

用法：
  python step2b_ddcm_step_encode.py <wav_path> <out_base> \
      --K-step 256 --N-step 1000 --seed 1234 --model cvssp/audioldm2-music

会生成：
  <out_base>.json / <out_base>.npz
"""
import argparse
import json
from pathlib import Path
import numpy as np

from ddcm.step_codec import StepCodecConfig, encode_step_bitstream

def save_step_bitstream(bitstream: dict, out_base: str):
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(bitstream["meta"])  # shallow copy
    meta["mode"] = "ddcm_step"
    (base.with_suffix('.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')

    stream = bitstream["stream"]
    np_data = {
        "mode": np.array(["ddcm_step"], dtype=object),
        "k_indices": np.array(stream["k_indices"], dtype=object),
        "shape": np.array(stream["shape"], dtype=np.int32),
    }
    if stream.get("coeffs") is not None:
        np_data["coeffs"] = np.array(stream["coeffs"], dtype=object)
    if stream.get("signs") is not None:
        np_data["signs"] = np.array(stream["signs"], dtype=object)
    if stream.get("init_latent") is not None:
        np_data["init_latent"] = np.asarray(stream["init_latent"], dtype=np.float32)
    np.savez_compressed(
        base.with_suffix('.npz'),
        **np_data,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('wav', type=str, help='输入 WAV 路径')
    p.add_argument('out_base', type=str, help='输出基路径（不含扩展名）')
    p.add_argument('--K-step', type=int, default=256, help='每步码本大小 K')
    p.add_argument('--N-step', type=int, default=1000, help='时间步数 T（默认 1000）')
    p.add_argument('--seed', type=int, default=1234, help='码本种子')
    p.add_argument('--model', type=str, default='cvssp/audioldm2-music', help='Hugging Face 模型名')
    p.add_argument('--atoms-per-step', type=int, default=1, help='Matching Pursuit 每步原子数 M（默认单原子）')
    p.add_argument('--coeff-levels', type=str, default=None, help='逗号分隔的系数候选列表，例如 "0.25,0.5,1.0"')
    p.add_argument('--guidance-gain', type=float, default=1.0, help='全局注入强度缩放（>1 提高噪声能量）')
    p.add_argument('--record-diagnostics', action='store_true', help='记录每步残差与系数诊断信息')
    p.add_argument('--continuous-coeff', action='store_true', help='使用连续系数（跳过量化，仅用于实验）')
    p.add_argument('--clap-weight', type=float, default=0.0, help='CLAP 语义相似度权重（>0 启用）')
    p.add_argument('--clap-topk', type=int, default=1, help='每次匹配评估的候选原子数量（用于 CLAP 复选）')
    p.add_argument('--clap-interval', type=int, default=0, help='每隔多少步启用一次 CLAP 复选（0 表示每步）')
    p.add_argument('--clap-model', type=str, default=None, help='可选：指定 CLAP 模型名（默认 laion/clap-htsat-unfused）')
    p.add_argument('--clap-max-secs', type=float, default=30.0, help='CLAP 评估时裁剪的最大秒数')
    p.add_argument('--vggish-weight', type=float, default=0.0, help='VGGish 语义相似度权重（>0 启用）')
    p.add_argument('--vggish-topk', type=int, default=1, help='VGGish 复选候选数量')
    p.add_argument('--vggish-interval', type=int, default=0, help='VGGish 评估间隔（0 表示每步）')
    p.add_argument('--vggish-max-secs', type=float, default=30.0, help='VGGish 评估裁剪秒数')
    p.add_argument('--match-epsilon', action='store_true', help='启用 ε/噪声域残差匹配（默认 x0 域）')
    p.add_argument('--save-recon-audio', type=str, default=None, help='可选：保存编码结束时的重建音频路径（默认 <out_base>_encode_recon.wav）')
    args = p.parse_args()

    coeff_levels = None
    if args.coeff_levels:
        raw = [item.strip() for item in args.coeff_levels.replace(';', ',').split(',') if item.strip()]
        if raw:
            coeff_levels = tuple(float(x) for x in raw)

    cfg_kwargs = dict(
        K_step=int(args.K_step),
        N_step=int(args.N_step),
        seed=int(args.seed),
        guidance_gain=float(args.guidance_gain),
        atoms_per_step=max(1, int(args.atoms_per_step)),
        record_diagnostics=bool(args.record_diagnostics),
        continuous_coeff=bool(args.continuous_coeff),
        clap_weight=float(args.clap_weight),
        clap_topk=max(1, int(args.clap_topk)),
        clap_interval=int(args.clap_interval),
        clap_model_name=args.clap_model,
        clap_max_duration=float(args.clap_max_secs),
        vggish_weight=float(args.vggish_weight),
        vggish_topk=max(1, int(args.vggish_topk)),
        vggish_interval=int(args.vggish_interval),
        vggish_max_duration=float(args.vggish_max_secs),
        match_epsilon=bool(args.match_epsilon),
    )
    if coeff_levels is not None:
        cfg_kwargs['coeff_levels'] = coeff_levels

    cfg = StepCodecConfig(**cfg_kwargs)

    recon_path = args.save_recon_audio
    if recon_path is None:
        recon_path = str(Path(args.out_base).with_name(Path(args.out_base).name + '_encode_recon.wav'))

    bitstream = encode_step_bitstream(args.wav, model_name=args.model, cfg=cfg, save_recon_path=recon_path)
    save_step_bitstream(bitstream, args.out_base)
    print(f"Saved ddcm_step bitstream to: {args.out_base}.json/.npz")


if __name__ == '__main__':
    main()
