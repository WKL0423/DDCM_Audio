#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import soundfile as sf
import torch

from ddcm.audio_models import load_audio_model
from ddcm.audio_runners import compress
from ddcm.util.audio_file import (
    get_args_from_filename,
    load_from_binary_bitwise,
    load_meta_json,
    save_as_binary_bitwise,
    save_meta_json,
)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gpu", type=int, default=0)
    common.add_argument("--cpu", action="store_true", help="Force CPU execution to avoid GPU OOM")
    common.add_argument("--float16", action="store_true")
    common.add_argument("--output_dir", type=str, required=True)

    cparser = argparse.ArgumentParser(add_help=False)
    cparser.add_argument("--input_path", type=str, required=True, help="Input wav path")
    cparser.add_argument("--model_id", type=str, default="cvssp/audioldm2-music")
    cparser.add_argument("-K", "--num_noises", dest="K", type=int, default=64)
    cparser.add_argument("--num-noises-late", type=int, default=0, help="Optional late-stage candidate count override")
    cparser.add_argument("--num-noises-switch-t", type=int, default=-1, help="Switch to num-noises-late when timestep <= this value")
    cparser.add_argument("-T", "--timesteps", dest="T", type=int, default=50)
    cparser.add_argument("--pursuit-noises", type=int, default=1, help="Noises per step in pursuit matching (>=1)")
    cparser.add_argument("--pursuit-coef-bits", type=int, default=3, help="Quantization bits for pursuit coefficient")
    cparser.add_argument("--no-pursuit-renorm", action="store_true", help="Disable std renormalization during pursuit mixing")
    cparser.add_argument("--score-mode", type=str, default="dot", choices=["dot", "cosine", "blend"], help="Candidate scoring objective")
    cparser.add_argument("--score-blend-lambda", type=float, default=0.5, help="Blend weight used when score-mode=blend")
    cparser.add_argument("--score-mode-late", type=str, default="", choices=["", "dot", "cosine", "blend"], help="Optional late-stage scoring objective")
    cparser.add_argument("--score-switch-t", type=int, default=-1, help="Switch to late score mode when timestep <= this value")
    cparser.add_argument("--eta", type=float, default=1.0, help="Base eta used in finish_step")
    cparser.add_argument("--eta-late", type=float, default=-1.0, help="Optional late-stage eta (>=0 to enable)")
    cparser.add_argument("--eta-switch-t", type=int, default=-1, help="Switch to eta-late when timestep <= this value")
    cparser.add_argument("--exact-rerank-topk", type=int, default=0, help="If >1, rerank top-k candidates by exact one-step residual")
    cparser.add_argument("--mel-proxy-topk", type=int, default=0, help="If >1, periodically rerank top-k by decoded mel MSE proxy")
    cparser.add_argument("--mel-proxy-interval", type=int, default=10, help="Apply mel proxy rerank every N optimized steps")
    cparser.add_argument("--audio-proxy-topk", type=int, default=0, help="If >1, periodically rerank top-k candidates by waveform MSE proxy")
    cparser.add_argument("--audio-proxy-interval", type=int, default=10, help="Apply audio proxy rerank every N optimized steps")
    cparser.add_argument("--t_range", nargs=2, type=int, default=[999, 0])

    subparsers.add_parser("compress", parents=[common, cparser])
    subparsers.add_parser("roundtrip", parents=[common, cparser])

    dparser = subparsers.add_parser("decompress", parents=[common])
    dparser.add_argument("--input_path", type=str, required=True, help="Input .bin path")

    args = parser.parse_args()

    if bool(args.cpu):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in {"compress", "roundtrip"}:
        model = load_audio_model(
            model_id=args.model_id,
            timesteps=int(args.T),
            device=device,
            float16=bool(args.float16),
            local_files_only=True,
        )

        rerank_suffix = f"_RK={int(args.exact_rerank_topk)}" if int(args.exact_rerank_topk) > 0 else ""
        mproxy_suffix = f"_MP={int(args.mel_proxy_topk)}x{int(args.mel_proxy_interval)}" if int(args.mel_proxy_topk) > 0 else ""
        aproxy_suffix = f"_AP={int(args.audio_proxy_topk)}x{int(args.audio_proxy_interval)}" if int(args.audio_proxy_topk) > 0 else ""
        score_suffix = f"_SM={str(args.score_mode).lower()}" if str(args.score_mode).lower() != "dot" else ""
        if str(args.score_mode).lower() == "blend":
            score_suffix += f"({float(args.score_blend_lambda):.3g})"
        late_suffix = ""
        if str(args.score_mode_late).lower() in {"dot", "cosine", "blend"} and int(args.score_switch_t) >= 0:
            late_suffix = f"_SL={str(args.score_mode_late).lower()}@{int(args.score_switch_t)}"
        eta_suffix = ""
        if float(args.eta) != 1.0 or float(args.eta_late) >= 0.0:
            eta_suffix = f"_ETA={float(args.eta):.3g}"
            if float(args.eta_late) >= 0.0 and int(args.eta_switch_t) >= 0:
                eta_suffix += f"-{float(args.eta_late):.3g}@{int(args.eta_switch_t)}"
        out_prefix = f"T={args.T}_in{args.t_range[0]}-{args.t_range[1]}_K={args.K}_P={args.pursuit_noises}_CB={args.pursuit_coef_bits}{rerank_suffix}{mproxy_suffix}{aproxy_suffix}{score_suffix}{late_suffix}{eta_suffix}_model={args.model_id.split('/')[-1]}_audio"
        if int(args.num_noises_late) > 0 and int(args.num_noises_switch_t) >= 0:
            out_prefix = out_prefix.replace("_model=", f"_KL={int(args.num_noises_late)}@{int(args.num_noises_switch_t)}_model=")
        out_dir = Path(args.output_dir) / out_prefix
        out_dir.mkdir(parents=True, exist_ok=True)

        comp = compress(
            model=model,
            wav_to_compress=args.input_path,
            num_noises=int(args.K),
            num_noises_late=int(args.num_noises_late),
            num_noises_switch_t=int(args.num_noises_switch_t),
            device=device,
            num_pursuit_noises=int(args.pursuit_noises),
            num_pursuit_coef_bits=int(args.pursuit_coef_bits),
            pursuit_renorm=not bool(args.no_pursuit_renorm),
            score_mode=str(args.score_mode).lower(),
            score_blend_lambda=float(args.score_blend_lambda),
            score_mode_late=str(args.score_mode_late).lower(),
            score_switch_t=int(args.score_switch_t),
            eta=float(args.eta),
            eta_late=float(args.eta_late),
            eta_switch_t=int(args.eta_switch_t),
            exact_rerank_topk=int(args.exact_rerank_topk),
            mel_proxy_topk=int(args.mel_proxy_topk),
            mel_proxy_interval=int(args.mel_proxy_interval),
            audio_proxy_topk=int(args.audio_proxy_topk),
            audio_proxy_interval=int(args.audio_proxy_interval),
            t_range=(int(args.t_range[0]), int(args.t_range[1])),
            decompress_indices=None,
        )

        stem = Path(args.input_path).stem
        comp_wav = out_dir / f"{stem}_comp.wav"
        bin_path = out_dir / f"{stem}_noise_indices.bin"
        meta_path = out_dir / f"{stem}_noise_indices.json"

        sf.write(str(comp_wav), comp["audio"], int(comp["sr"]))
        flat_indices = comp["stream"]["noise_indices"]
        if len(flat_indices) > 0 and isinstance(flat_indices[0], list):
            flat_indices = [int(step[0]) if len(step) > 0 else 0 for step in flat_indices]
        save_as_binary_bitwise(flat_indices, int(args.K), str(bin_path))
        save_meta_json(
            str(meta_path),
            {
                "mode": comp["stream"].get("mode", "ddcm_audio_v1"),
                "model_id": args.model_id,
                "timesteps": int(args.T),
                "num_noises": int(args.K),
                "num_noises_late": int(comp["stream"].get("num_noises_late", int(args.num_noises_late))),
                "num_noises_switch_t": int(comp["stream"].get("num_noises_switch_t", int(args.num_noises_switch_t))),
                "num_pursuit_noises": int(args.pursuit_noises),
                "num_pursuit_coef_bits": int(args.pursuit_coef_bits),
                "pursuit_renorm": bool(comp["stream"].get("pursuit_renorm", (not bool(args.no_pursuit_renorm)))),
                "score_mode": str(comp["stream"].get("score_mode", str(args.score_mode).lower())),
                "score_blend_lambda": float(comp["stream"].get("score_blend_lambda", float(args.score_blend_lambda))),
                "score_mode_late": str(comp["stream"].get("score_mode_late", str(args.score_mode_late).lower())),
                "score_switch_t": int(comp["stream"].get("score_switch_t", int(args.score_switch_t))),
                "eta": float(comp["stream"].get("eta", float(args.eta))),
                "eta_late": float(comp["stream"].get("eta_late", float(args.eta_late))),
                "eta_switch_t": int(comp["stream"].get("eta_switch_t", int(args.eta_switch_t))),
                "exact_rerank_topk": int(comp["stream"].get("exact_rerank_topk", int(args.exact_rerank_topk))),
                "mel_proxy_topk": int(comp["stream"].get("mel_proxy_topk", int(args.mel_proxy_topk))),
                "mel_proxy_interval": int(comp["stream"].get("mel_proxy_interval", int(args.mel_proxy_interval))),
                "audio_proxy_topk": int(comp["stream"].get("audio_proxy_topk", int(args.audio_proxy_topk))),
                "audio_proxy_interval": int(comp["stream"].get("audio_proxy_interval", int(args.audio_proxy_interval))),
                "t_range": [int(args.t_range[0]), int(args.t_range[1])],
                "shape": comp["stream"]["shape"],
                "optimized_steps": int(comp["stream"]["optimized_steps"]),
                "init_noise_seed": int(comp["stream"]["init_noise_seed"]),
                "step_noise_seed": int(comp["stream"]["step_noise_seed"]),
                "noise_indices": comp["stream"].get("noise_indices"),
                "coeff_indices": comp["stream"].get("coeff_indices"),
                "sr": int(comp["sr"]),
                "diagnostics": comp.get("diagnostics"),
            },
        )

        print(f"Saved compressed audio to: {comp_wav}")
        print(f"Saved bitstream to: {bin_path}")

        if args.mode == "compress":
            return

        # roundtrip decode
        if comp["stream"].get("mode") == "ddcm_audio_v2":
            indices = comp["stream"]["noise_indices"]
            coeff_indices = comp["stream"].get("coeff_indices")
        else:
            indices = load_from_binary_bitwise(str(bin_path), int(args.K), int(comp["stream"]["optimized_steps"]))
            coeff_indices = None
        renorm = bool(comp["stream"].get("pursuit_renorm", (not bool(args.no_pursuit_renorm))))
        dec = compress(
            model=model,
            wav_to_compress=None,
            num_noises=int(args.K),
            num_noises_late=int(comp["stream"].get("num_noises_late", int(args.num_noises_late))),
            num_noises_switch_t=int(comp["stream"].get("num_noises_switch_t", int(args.num_noises_switch_t))),
            device=device,
            num_pursuit_noises=int(args.pursuit_noises),
            num_pursuit_coef_bits=int(args.pursuit_coef_bits),
            pursuit_renorm=renorm,
            score_mode=str(comp["stream"].get("score_mode", str(args.score_mode).lower())),
            score_blend_lambda=float(comp["stream"].get("score_blend_lambda", float(args.score_blend_lambda))),
            score_mode_late=str(comp["stream"].get("score_mode_late", str(args.score_mode_late).lower())),
            score_switch_t=int(comp["stream"].get("score_switch_t", int(args.score_switch_t))),
            eta=float(comp["stream"].get("eta", float(args.eta))),
            eta_late=float(comp["stream"].get("eta_late", float(args.eta_late))),
            eta_switch_t=int(comp["stream"].get("eta_switch_t", int(args.eta_switch_t))),
            exact_rerank_topk=int(comp["stream"].get("exact_rerank_topk", int(args.exact_rerank_topk))),
            mel_proxy_topk=int(comp["stream"].get("mel_proxy_topk", int(args.mel_proxy_topk))),
            mel_proxy_interval=int(comp["stream"].get("mel_proxy_interval", int(args.mel_proxy_interval))),
            audio_proxy_topk=int(comp["stream"].get("audio_proxy_topk", int(args.audio_proxy_topk))),
            audio_proxy_interval=int(comp["stream"].get("audio_proxy_interval", int(args.audio_proxy_interval))),
            t_range=(int(args.t_range[0]), int(args.t_range[1])),
            decompress_indices={
                "noise_indices": indices,
                "coeff_indices": coeff_indices,
                "shape": comp["stream"]["shape"],
            },
            init_noise_seed=int(comp["stream"]["init_noise_seed"]),
            step_noise_seed=int(comp["stream"]["step_noise_seed"]),
        )
        dec_wav = out_dir / f"{stem}_decomp.wav"
        sf.write(str(dec_wav), dec["audio"], int(dec["sr"]))
        print(f"Saved roundtrip audio to: {dec_wav}")

    else:
        bin_path = Path(args.input_path)
        meta_path = bin_path.with_suffix(".json")
        meta = load_meta_json(str(meta_path))
        T, K, t_range, model_id = get_args_from_filename(str(bin_path))
        # prefer sidecar metadata
        model_id = meta.get("model_id", model_id)
        T = int(meta.get("timesteps", T))
        K = int(meta.get("num_noises", K))
        K_late = int(meta.get("num_noises_late", 0))
        K_switch_t = int(meta.get("num_noises_switch_t", -1))
        P = int(meta.get("num_pursuit_noises", 1))
        coef_bits = int(meta.get("num_pursuit_coef_bits", 3))
        renorm = bool(meta.get("pursuit_renorm", True))
        score_mode = str(meta.get("score_mode", "dot")).lower()
        score_blend_lambda = float(meta.get("score_blend_lambda", 0.5))
        score_mode_late = str(meta.get("score_mode_late", "")).lower()
        score_switch_t = int(meta.get("score_switch_t", -1))
        eta = float(meta.get("eta", 1.0))
        eta_late = float(meta.get("eta_late", -1.0))
        eta_switch_t = int(meta.get("eta_switch_t", -1))
        rerank_topk = int(meta.get("exact_rerank_topk", 0))
        mel_proxy_topk = int(meta.get("mel_proxy_topk", 0))
        mel_proxy_interval = int(meta.get("mel_proxy_interval", 10))
        audio_proxy_topk = int(meta.get("audio_proxy_topk", 0))
        audio_proxy_interval = int(meta.get("audio_proxy_interval", 10))
        t_range = tuple(meta.get("t_range", list(t_range)))

        model = load_audio_model(
            model_id=model_id,
            timesteps=T,
            device=device,
            float16=bool(args.float16),
            local_files_only=True,
        )

        if meta.get("noise_indices") is not None:
            indices = meta.get("noise_indices")
            coeff_indices = meta.get("coeff_indices")
        else:
            indices = load_from_binary_bitwise(str(bin_path), int(K), int(meta["optimized_steps"]))
            coeff_indices = None
        dec = compress(
            model=model,
            wav_to_compress=None,
            num_noises=int(K),
            num_noises_late=int(K_late),
            num_noises_switch_t=int(K_switch_t),
            device=device,
            num_pursuit_noises=int(P),
            num_pursuit_coef_bits=int(coef_bits),
            pursuit_renorm=renorm,
            score_mode=str(score_mode),
            score_blend_lambda=float(score_blend_lambda),
            score_mode_late=str(score_mode_late),
            score_switch_t=int(score_switch_t),
            eta=float(eta),
            eta_late=float(eta_late),
            eta_switch_t=int(eta_switch_t),
            exact_rerank_topk=int(rerank_topk),
            mel_proxy_topk=int(mel_proxy_topk),
            mel_proxy_interval=int(mel_proxy_interval),
            audio_proxy_topk=int(audio_proxy_topk),
            audio_proxy_interval=int(audio_proxy_interval),
            t_range=(int(t_range[0]), int(t_range[1])),
            decompress_indices={
                "noise_indices": indices,
                "coeff_indices": coeff_indices,
                "shape": meta["shape"],
            },
            init_noise_seed=int(meta.get("init_noise_seed", 100000)),
            step_noise_seed=int(meta.get("step_noise_seed", 0)),
        )

        out_wav = Path(args.output_dir) / f"{bin_path.stem}_decomp.wav"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), dec["audio"], int(dec["sr"]))
        print(f"Saved decompressed audio to: {out_wav}")


if __name__ == "__main__":
    main()
