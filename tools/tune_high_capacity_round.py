from __future__ import annotations

import json
import subprocess
from pathlib import Path

PY = r"E:/ProgramData/Anaconda3/envs/audioldm_env/python.exe"


def run_cmd(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "") + "\n" + (proc.stdout or ""))
    return proc.stdout


def metric(ref: str, test: str) -> dict:
    out = run_cmd([PY, "tools/compare_audio_metrics.py", "--ref", ref, "--test", test, "--sr", "16000", "--max-secs", "30"])
    return json.loads(out)


def main() -> None:
    root = Path("runs/tune_high_capacity")
    root.mkdir(parents=True, exist_ok=True)

    P = 2
    CB = 4
    t_range = (999, 0)

    configs = [
        {"T": 20, "K": 32},
        {"T": 20, "K": 64},
        {"T": 50, "K": 32},
        {"T": 50, "K": 64},
    ]

    rows: list[dict] = []
    for cfg in configs:
        T = int(cfg["T"])
        K = int(cfg["K"])

        out_dir = Path("runs") / f"T={T}_in{t_range[0]}-{t_range[1]}_K={K}_P={P}_CB={CB}_model=audioldm2-music_audio"
        decomp = out_dir / "piano_decomp.wav"
        comp = out_dir / "piano_comp.wav"
        meta_path = out_dir / "piano_noise_indices.json"
        bin_path = out_dir / "piano_noise_indices.bin"

        if not decomp.exists():
            run_cmd(
                [
                    PY,
                    "-u",
                    "audio_compression.py",
                    "roundtrip",
                    "--output_dir",
                    "runs",
                    "--input_path",
                    "piano.wav",
                    "-T",
                    str(T),
                    "-K",
                    str(K),
                    "--pursuit-noises",
                    str(P),
                    "--pursuit-coef-bits",
                    str(CB),
                    "--t_range",
                    str(t_range[0]),
                    str(t_range[1]),
                    "--float16",
                ]
            )

        q = metric("piano.wav", str(decomp))
        flow = metric(str(comp), str(decomp))

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        diag = meta.get("diagnostics") or {}
        before = diag.get("residual_norm_before") or []
        after = diag.get("residual_norm_after") or []
        cumulative_improvement = float(sum((b - a) for b, a in zip(before, after))) if before and after else 0.0
        improved_steps = int(sum(1 for b, a in zip(before, after) if a <= b)) if before and after else 0
        total_steps = int(len(after))

        orig = Path("piano.wav").stat().st_size
        total_bytes = int(bin_path.stat().st_size + meta_path.stat().st_size)

        rows.append(
            {
                "T": T,
                "K": K,
                "P": P,
                "coef_bits": CB,
                "t_range": [t_range[0], t_range[1]],
                "corr": q["pearson_corr"],
                "snr_db": q["snr_db"],
                "mel_db_mae": q["mel_db_mae"],
                "stft_mag_mse": q["stft_mag_mse"],
                "flow_mse": flow["waveform_mse"],
                "flow_corr": flow["pearson_corr"],
                "cumulative_residual_improvement": cumulative_improvement,
                "improved_steps": improved_steps,
                "total_steps": total_steps,
                "bitstream_total_bytes": total_bytes,
                "ratio_vs_orig": (orig / total_bytes) if total_bytes > 0 else None,
                "out_dir": str(out_dir).replace("\\", "/"),
            }
        )

    # Pareto: maximize corr/snr/cum_improve; minimize mel_mae/bytes
    def dominates(a: dict, b: dict) -> bool:
        not_worse = (
            a["corr"] >= b["corr"]
            and a["snr_db"] >= b["snr_db"]
            and a["cumulative_residual_improvement"] >= b["cumulative_residual_improvement"]
            and a["mel_db_mae"] <= b["mel_db_mae"]
            and a["bitstream_total_bytes"] <= b["bitstream_total_bytes"]
        )
        strictly_better = (
            a["corr"] > b["corr"]
            or a["snr_db"] > b["snr_db"]
            or a["cumulative_residual_improvement"] > b["cumulative_residual_improvement"]
            or a["mel_db_mae"] < b["mel_db_mae"]
            or a["bitstream_total_bytes"] < b["bitstream_total_bytes"]
        )
        return bool(not_worse and strictly_better)

    pareto = []
    for i, r in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if dominates(other, r):
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    # Save outputs
    json_path = root / "high_capacity_results.json"
    pareto_path = root / "high_capacity_pareto.json"
    md_path = root / "high_capacity_results.md"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pareto_path.write_text(json.dumps(pareto, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| T | K | corr | snr_db | mel_db_mae | stft_mse | cum_improve | improved_steps | total_bytes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["corr"], x["snr_db"], -x["mel_db_mae"]), reverse=True):
        lines.append(
            f"| {r['T']} | {r['K']} | {r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | {r['cumulative_residual_improvement']:.3f} | {r['improved_steps']}/{r['total_steps']} | {r['bitstream_total_bytes']} |"
        )
    lines.append("")
    lines.append("## Pareto candidates")
    for r in pareto:
        lines.append(
            f"- T={r['T']}, K={r['K']}, corr={r['corr']:.4f}, snr={r['snr_db']:.4f}, mel={r['mel_db_mae']:.4f}, bytes={r['bitstream_total_bytes']}"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    best_quality = sorted(rows, key=lambda x: (x["corr"], x["snr_db"], -x["mel_db_mae"]), reverse=True)[0]
    best_direction = sorted(rows, key=lambda x: (x["cumulative_residual_improvement"], x["improved_steps"], x["corr"]), reverse=True)[0]

    print(
        json.dumps(
            {
                "best_quality": best_quality,
                "best_direction": best_direction,
                "pareto_count": len(pareto),
                "results_json": str(json_path).replace("\\", "/"),
                "pareto_json": str(pareto_path).replace("\\", "/"),
                "results_md": str(md_path).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
