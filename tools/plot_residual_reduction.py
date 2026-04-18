from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    items = [
        ("T10K16P1", Path("runs/T=10_in999-0_K=16_model=audioldm2-music_audio/piano_noise_indices.json")),
        ("T10K16P2", Path("runs/T=10_in999-0_K=16_P=2_model=audioldm2-music_audio/piano_noise_indices.json")),
        ("T20K32P1", Path("runs/T=20_in999-0_K=32_P=1_model=audioldm2-music_audio/piano_noise_indices.json")),
        ("T20K32P2", Path("runs/T=20_in999-0_K=32_P=2_model=audioldm2-music_audio/piano_noise_indices.json")),
    ]

    series = []
    for name, path in items:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        diag = obj.get("diagnostics") or {}
        before = np.asarray(diag.get("residual_norm_before") or [], dtype=float)
        after = np.asarray(diag.get("residual_norm_after") or [], dtype=float)
        if before.size == 0 or after.size == 0:
            continue

        denom = before[0] if before[0] != 0 else 1.0
        before_norm = before / denom
        after_norm = after / denom
        delta = before_norm[: after_norm.shape[0]] - after_norm
        series.append((name, before_norm, after_norm, delta))

    if not series:
        raise RuntimeError("No diagnostics found to plot")

    out_path = Path("runs/baseline/residual_reduction_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for name, before_norm, after_norm, delta in series:
        x = np.arange(before_norm.shape[0])
        axs[0].plot(x, before_norm, linestyle="--", alpha=0.8, label=f"{name} before")
        axs[0].plot(x, after_norm, linestyle="-", alpha=0.95, label=f"{name} after")
        axs[1].plot(np.arange(delta.shape[0]), delta, label=name)

    axs[0].set_title("Residual norm by step (normalized)")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("residual / first_before")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=8, ncol=2)

    axs[1].set_title("Per-step improvement (before - after)")
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("delta (higher is better)")
    axs[1].axhline(0.0, color="black", linewidth=1)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(str(out_path).replace('\\', '/'))


if __name__ == "__main__":
    main()
