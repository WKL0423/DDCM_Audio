from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddcm.audio_models import load_audio_model
from ddcm.audio_runners import compress


def max_abs_diff(a, b) -> float:
    if len(a) != len(b):
        return float("inf")
    if not a:
        return 0.0
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(arr_a - arr_b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_id", default="cvssp/audioldm2-music")
    ap.add_argument("-T", type=int, default=20)
    ap.add_argument("-K", type=int, default=32)
    ap.add_argument("--pursuit-noises", type=int, default=2)
    ap.add_argument("--pursuit-coef-bits", type=int, default=4)
    ap.add_argument("--t0", type=int, default=999)
    ap.add_argument("--t1", type=int, default=0)
    ap.add_argument("--float16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-pursuit-renorm", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    model = load_audio_model(
        model_id=args.model_id,
        timesteps=int(args.T),
        device=device,
        float16=bool(args.float16),
        local_files_only=True,
    )

    comp = compress(
        model=model,
        wav_to_compress=args.input,
        num_noises=int(args.K),
        device=device,
        num_pursuit_noises=int(args.pursuit_noises),
        num_pursuit_coef_bits=int(args.pursuit_coef_bits),
        pursuit_renorm=not bool(args.no_pursuit_renorm),
        t_range=(int(args.t0), int(args.t1)),
    )

    dec = compress(
        model=model,
        wav_to_compress=None,
        num_noises=int(args.K),
        device=device,
        num_pursuit_noises=int(args.pursuit_noises),
        num_pursuit_coef_bits=int(args.pursuit_coef_bits),
        pursuit_renorm=not bool(args.no_pursuit_renorm),
        t_range=(int(args.t0), int(args.t1)),
        decompress_indices={
            "noise_indices": comp["stream"]["noise_indices"],
            "coeff_indices": comp["stream"].get("coeff_indices"),
            "shape": comp["stream"]["shape"],
        },
        init_noise_seed=int(comp["stream"]["init_noise_seed"]),
        step_noise_seed=int(comp["stream"]["step_noise_seed"]),
    )

    comp_l2 = comp["stream"].get("selected_noise_l2", [])
    dec_l2 = dec["stream"].get("selected_noise_l2", [])
    comp_sum = comp["stream"].get("selected_noise_sum", [])
    dec_sum = dec["stream"].get("selected_noise_sum", [])

    report = {
        "config": {
            "T": int(args.T),
            "K": int(args.K),
            "pursuit_noises": int(args.pursuit_noises),
            "pursuit_coef_bits": int(args.pursuit_coef_bits),
            "t_range": [int(args.t0), int(args.t1)],
            "pursuit_renorm": not bool(args.no_pursuit_renorm),
            "device": str(device),
        },
        "consistency": {
            "step_count_encode": len(comp_l2),
            "step_count_decode": len(dec_l2),
            "selected_noise_l2_max_abs_diff": max_abs_diff(comp_l2, dec_l2),
            "selected_noise_sum_max_abs_diff": max_abs_diff(comp_sum, dec_sum),
            "audio_waveform_max_abs_diff": float(np.max(np.abs(comp["audio"][: min(len(comp["audio"]), len(dec["audio"]))] - dec["audio"][: min(len(comp["audio"]), len(dec["audio"]))]))),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out).replace('\\', '/'))
    print(json.dumps(report["consistency"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
