#!/usr/bin/env python3
"""
Plot DDCM per-step diagnostics saved in the encoder meta JSON.
Usage:
  python tools/plot_ddcm_diagnostics.py <meta.json> --out evaluation_results/diag.png
If --out is omitted, save alongside input with suffix _diag.png.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _get(seq: Dict, key: str) -> Sequence:
    val = seq.get(key)
    if val is None:
        raise KeyError(f"Missing diagnostics field: {key}")
    return val


def plot_diag(meta: Dict, out_path: Path):
    diag = meta.get("diagnostics")
    if not diag:
        raise ValueError("No diagnostics found in JSON")

    residual_before = np.asarray(_get(diag, "residual_norm_before"), dtype=float)
    residual_after = np.asarray(_get(diag, "residual_norm_after"), dtype=float)
    injection_norm = np.asarray(_get(diag, "injection_norm"), dtype=float)
    injection_scale = np.asarray(_get(diag, "injection_scale"), dtype=float)
    steps = np.arange(residual_before.shape[0])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"DDCM diagnostics | K={meta.get('cfg', {}).get('K_step')} N={meta.get('cfg', {}).get('N_step')} guidance_gain={meta.get('cfg', {}).get('guidance_gain', 1.0)}",
        fontsize=12,
    )

    axs[0, 0].plot(steps, residual_before, label="before")
    axs[0, 0].plot(steps, residual_after, label="after")
    axs[0, 0].set_title("Residual norm")
    axs[0, 0].set_xlabel("step")
    axs[0, 0].set_ylabel("L2 norm")
    axs[0, 0].legend()

    best_abs = np.asarray(diag.get("best_abs_score", []), dtype=float)
    if best_abs.size:
        axs[0, 1].plot(steps[: best_abs.shape[0]], best_abs, label="|score|")
        axs[0, 1].set_title("Best atom score")
        axs[0, 1].set_xlabel("step")
        axs[0, 1].set_ylabel("abs(score)")
    else:
        axs[0, 1].axis("off")

    axs[1, 0].plot(steps[: injection_norm.shape[0]], injection_norm, label="||z_t||")
    axs[1, 0].set_title("Injection norm")
    axs[1, 0].set_xlabel("step")
    axs[1, 0].set_ylabel("L2 norm")

    axs[1, 1].plot(steps[: injection_scale.shape[0]], injection_scale, label="scale")
    axs[1, 1].set_title("Injection scale")
    axs[1, 1].set_xlabel("step")
    axs[1, 1].set_ylabel("scale")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta", type=str, help="Path to ddcm_step meta JSON with diagnostics")
    parser.add_argument("--out", type=str, default=None, help="Optional output PNG path")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    out_path = Path(args.out) if args.out else meta_path.with_name(meta_path.stem + "_diag.png")
    out = plot_diag(meta, out_path)
    print(f"Saved diagnostics plot to: {out}")


if __name__ == "__main__":
    main()
