from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ddcm_tool_env import PY


def run_cmd(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "") + "\n" + (proc.stdout or ""))
    return proc.stdout


def metric(ref: str, test: str) -> dict:
    out = run_cmd([PY, "tools/compare_audio_metrics.py", "--ref", ref, "--test", test, "--sr", "16000", "--max-secs", "30"])
    return json.loads(out)


def main() -> None:
    root = Path("runs/tune_t50_range")
    root.mkdir(parents=True, exist_ok=True)

    T, K, P, CB = 50, 32, 2, 4
    ranges = [(999, 0), (900, 100), (800, 100), (700, 100), (600, 100)]

    rows = []
    for t0, t1 in ranges:
        out_dir = Path("runs") / f"T={T}_in{t0}-{t1}_K={K}_P={P}_CB={CB}_model=audioldm2-music_audio"
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
                    str(t0),
                    str(t1),
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
                "t_range": [t0, t1],
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

    rows_sorted = sorted(rows, key=lambda x: (x["corr"], x["snr_db"], -x["mel_db_mae"]), reverse=True)
    best_quality = rows_sorted[0]
    best_direction = sorted(rows, key=lambda x: (x["cumulative_residual_improvement"], x["improved_steps"], x["corr"]), reverse=True)[0]

    json_path = root / "t50_range_results.json"
    md_path = root / "t50_range_results.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| t_range | corr | snr_db | mel_db_mae | stft_mse | cum_improve | improved_steps | total_bytes |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['t_range'][0]}-{r['t_range'][1]} | {r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | {r['cumulative_residual_improvement']:.3f} | {r['improved_steps']}/{r['total_steps']} | {r['bitstream_total_bytes']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "best_quality": best_quality,
                "best_direction": best_direction,
                "results_json": str(json_path).replace("\\", "/"),
                "results_md": str(md_path).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
