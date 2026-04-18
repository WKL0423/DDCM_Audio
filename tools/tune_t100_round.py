from __future__ import annotations

import json
import subprocess
from pathlib import Path

PY = r"E:/ProgramData/Anaconda3/envs/audioldm_env/python.exe"


def run_cmd(args: list[str]) -> str:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "") + "\n" + (p.stdout or ""))
    return p.stdout


def metric(ref: str, test: str) -> dict:
    out = run_cmd([PY, "tools/compare_audio_metrics.py", "--ref", ref, "--test", test, "--sr", "16000", "--max-secs", "30"])
    return json.loads(out)


def main() -> None:
    root = Path("runs/tune_t100")
    root.mkdir(parents=True, exist_ok=True)

    T = 100
    P = 2
    CB = 4
    t0, t1 = 999, 0

    rows = []
    for K in [32, 64]:
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

        orig = Path("piano.wav").stat().st_size
        total_bytes = int(bin_path.stat().st_size + meta_path.stat().st_size)

        rows.append(
            {
                "T": T,
                "K": K,
                "P": P,
                "coef_bits": CB,
                "t_range": [t0, t1],
                "corr": q["pearson_corr"],
                "snr_db": q["snr_db"],
                "mel_db_mae": q["mel_db_mae"],
                "stft_mag_mse": q["stft_mag_mse"],
                "flow_mse": flow["waveform_mse"],
                "flow_corr": flow["pearson_corr"],
                "cumulative_residual_improvement": cumulative_improvement,
                "improved_steps": improved_steps,
                "total_steps": len(after),
                "bitstream_total_bytes": total_bytes,
                "ratio_vs_orig": (orig / total_bytes) if total_bytes > 0 else None,
                "out_dir": str(out_dir).replace("\\", "/"),
            }
        )

    # compare with current best T50K32P2CB4 if exists
    baseline = None
    baseline_meta = Path("runs/T=50_in999-0_K=32_P=2_CB=4_model=audioldm2-music_audio/piano_noise_indices.json")
    baseline_decomp = Path("runs/T=50_in999-0_K=32_P=2_CB=4_model=audioldm2-music_audio/piano_decomp.wav")
    baseline_comp = Path("runs/T=50_in999-0_K=32_P=2_CB=4_model=audioldm2-music_audio/piano_comp.wav")
    baseline_bin = Path("runs/T=50_in999-0_K=32_P=2_CB=4_model=audioldm2-music_audio/piano_noise_indices.bin")

    if baseline_meta.exists() and baseline_decomp.exists() and baseline_comp.exists() and baseline_bin.exists():
        q = metric("piano.wav", str(baseline_decomp))
        f = metric(str(baseline_comp), str(baseline_decomp))
        meta = json.loads(baseline_meta.read_text(encoding="utf-8"))
        diag = meta.get("diagnostics") or {}
        before = diag.get("residual_norm_before") or []
        after = diag.get("residual_norm_after") or []
        cumulative_improvement = float(sum((b - a) for b, a in zip(before, after))) if before and after else 0.0
        baseline = {
            "T": 50,
            "K": 32,
            "corr": q["pearson_corr"],
            "snr_db": q["snr_db"],
            "mel_db_mae": q["mel_db_mae"],
            "stft_mag_mse": q["stft_mag_mse"],
            "flow_mse": f["waveform_mse"],
            "flow_corr": f["pearson_corr"],
            "cumulative_residual_improvement": cumulative_improvement,
            "bitstream_total_bytes": baseline_bin.stat().st_size + baseline_meta.stat().st_size,
        }

    json_path = root / "t100_results.json"
    md_path = root / "t100_results.md"
    json_path.write_text(json.dumps({"baseline": baseline, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| cfg | corr | snr_db | mel_db_mae | stft_mse | cum_improve | flow_mse | total_bytes |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if baseline is not None:
        lines.append(
            f"| T50K32(P2,CB4) | {baseline['corr']:.4f} | {baseline['snr_db']:.4f} | {baseline['mel_db_mae']:.4f} | {baseline['stft_mag_mse']:.4f} | {baseline['cumulative_residual_improvement']:.3f} | {baseline['flow_mse']:.2e} | {baseline['bitstream_total_bytes']} |"
        )
    for r in rows:
        lines.append(
            f"| T{r['T']}K{r['K']}(P2,CB4) | {r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | {r['cumulative_residual_improvement']:.3f} | {r['flow_mse']:.2e} | {r['bitstream_total_bytes']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    best = sorted(rows, key=lambda x: (x["corr"], x["snr_db"], -x["mel_db_mae"]), reverse=True)[0]
    print(json.dumps({"best_t100": best, "results_json": str(json_path).replace('\\', '/'), "results_md": str(md_path).replace('\\', '/')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
