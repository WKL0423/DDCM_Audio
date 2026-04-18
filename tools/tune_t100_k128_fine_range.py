from __future__ import annotations

import json
import subprocess
from pathlib import Path

PY = r"E:/ProgramData/Anaconda3/envs/audioldm_env/python.exe"


def run_cmd(args: list[str], capture: bool = False) -> str:
    env = None
    if not capture:
        env = dict(**__import__("os").environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.run(args, text=True, capture_output=capture, env=env)
    if proc.returncode != 0:
        if capture:
            raise RuntimeError((proc.stderr or "") + "\n" + (proc.stdout or ""))
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(args)}")
    return (proc.stdout or "") if capture else ""


def metric(ref: str, test: str) -> dict:
    out = run_cmd([PY, "tools/compare_audio_metrics.py", "--ref", ref, "--test", test, "--sr", "16000", "--max-secs", "30"], capture=True)
    return json.loads(out)


def main() -> None:
    root = Path("runs/tune_t100_k128_fine")
    root.mkdir(parents=True, exist_ok=True)

    T, K, P, CB = 100, 128, 2, 4
    ranges = [
        (850, 120),
        (880, 120),
        (900, 80),
        (900, 100),
        (920, 100),
        (940, 120),
    ]

    rows = []
    for t0, t1 in ranges:
        out_dir = Path("runs") / f"T={T}_in{t0}-{t1}_K={K}_P={P}_CB={CB}_model=audioldm2-music_audio"
        decomp = out_dir / "piano_decomp.wav"
        comp = out_dir / "piano_comp.wav"
        meta_path = out_dir / "piano_noise_indices.json"
        bin_path = out_dir / "piano_noise_indices.bin"

        if not decomp.exists():
            base_args = [
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
            try:
                run_cmd(base_args)
            except Exception:
                cpu_args = [
                    PY,
                    "-u",
                    "audio_compression.py",
                    "roundtrip",
                    "--cpu",
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
                ]
                run_cmd(cpu_args)

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

    best_quality = sorted(rows, key=lambda x: (x["corr"], x["snr_db"], -x["mel_db_mae"]), reverse=True)[0]

    baseline = {
        "cfg": "T100K128P2CB4_in900-100",
        "corr": 0.053009457886219025,
        "snr_db": -1.4707780359812443,
        "mel_db_mae": 5.7219014167785645,
    }

    improved = (
        (best_quality["corr"] > baseline["corr"])
        or (best_quality["snr_db"] > baseline["snr_db"])
        or (best_quality["mel_db_mae"] < baseline["mel_db_mae"])
    )

    json_path = root / "results.json"
    md_path = root / "results.md"

    payload = {
        "baseline": baseline,
        "best_quality": best_quality,
        "improved_vs_baseline": bool(improved),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| t_range | corr | snr_db | mel_db_mae | stft_mse | cum_improve | improved_steps | bytes |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["corr"], x["snr_db"], -x["mel_db_mae"]), reverse=True):
        lines.append(
            f"| {r['t_range'][0]}-{r['t_range'][1]} | {r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | {r['cumulative_residual_improvement']:.3f} | {r['improved_steps']}/{r['total_steps']} | {r['bitstream_total_bytes']} |"
        )
    lines.append("")
    lines.append(f"Improved vs baseline: {bool(improved)}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "improved_vs_baseline": bool(improved),
        "best_quality": best_quality,
        "results_json": str(json_path).replace('\\', '/'),
        "results_md": str(md_path).replace('\\', '/'),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
