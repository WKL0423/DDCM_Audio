from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from ddcm_tool_env import PY


def run_cmd(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "") + "\n" + (proc.stdout or ""))
    return proc.stdout


def metric(ref: str, test: str) -> dict[str, Any]:
    out = run_cmd([PY, "tools/compare_audio_metrics.py", "--ref", ref, "--test", test, "--sr", "16000", "--max-secs", "30"])
    return json.loads(out)


def quality_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    latent_cos = float(row.get("final_latent_cosine") or 0.0)
    return (
        float(row["corr"]),
        float(row["snr_db"]),
        -float(row["mel_db_mae"]),
        latent_cos,
    )


def run_one(
    *,
    input_wav: str,
    model_id: str,
    out_root: Path,
    T: int,
    K: int,
    P: int,
    CB: int,
    t_range: tuple[int, int],
    force: bool,
    phase: str,
) -> dict[str, Any]:
    out_dir = out_root / f"T={T}_in{t_range[0]}-{t_range[1]}_K={K}_P={P}_CB={CB}_model={model_id.split('/')[-1]}_audio"
    decomp = out_dir / "piano_decomp.wav"
    comp = out_dir / "piano_comp.wav"
    meta_path = out_dir / "piano_noise_indices.json"
    bin_path = out_dir / "piano_noise_indices.bin"

    if force or (not decomp.exists()):
        run_cmd(
            [
                PY,
                "-u",
                "audio_compression.py",
                "roundtrip",
                "--output_dir",
                str(out_root),
                "--input_path",
                input_wav,
                "--model_id",
                model_id,
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

    q = metric(input_wav, str(decomp))
    f = metric(str(comp), str(decomp))

    meta_obj = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    diag = meta_obj.get("diagnostics") or {}

    orig = Path(input_wav).stat().st_size
    b = bin_path.stat().st_size if bin_path.exists() else 0
    j = meta_path.stat().st_size if meta_path.exists() else 0
    total = int(b + j)

    return {
        "phase": phase,
        "T": int(T),
        "K": int(K),
        "P": int(P),
        "coef_bits": int(CB),
        "t_range": [int(t_range[0]), int(t_range[1])],
        "corr": float(q["pearson_corr"]),
        "snr_db": float(q["snr_db"]),
        "mel_db_mae": float(q["mel_db_mae"]),
        "stft_mag_mse": float(q["stft_mag_mse"]),
        "flow_mse": float(f["waveform_mse"]),
        "flow_corr": float(f["pearson_corr"]),
        "final_latent_mse": diag.get("final_latent_mse"),
        "final_latent_rel_l2": diag.get("final_latent_rel_l2"),
        "final_latent_cosine": diag.get("final_latent_cosine"),
        "bitstream_total_bytes": total,
        "ratio_vs_orig": (orig / total) if total > 0 else None,
        "out_dir": str(out_dir).replace("\\", "/"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="piano.wav")
    ap.add_argument("--model_id", type=str, default="cvssp/audioldm2-music")
    ap.add_argument("--output_root", type=str, default="runs/quality_push_large_codebook")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--phase3_timesteps", nargs="*", type=int, default=[200, 400], help="Extra T values to test on best K/t_range")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    P = 2
    CB = 4

    phase1 = [
        {"T": 100, "K": 128, "t_range": (900, 80)},
        {"T": 100, "K": 256, "t_range": (900, 80)},
        {"T": 100, "K": 512, "t_range": (900, 80)},
        {"T": 100, "K": 1000, "t_range": (900, 80)},
    ]

    rows: list[dict[str, Any]] = []
    for cfg in phase1:
        row = run_one(
            input_wav=args.input,
            model_id=args.model_id,
            out_root=out_root,
            T=int(cfg["T"]),
            K=int(cfg["K"]),
            P=P,
            CB=CB,
            t_range=cfg["t_range"],
            force=bool(args.force),
            phase="phase1_codebook",
        )
        rows.append(row)

    best_phase1 = sorted(rows, key=quality_sort_key, reverse=True)[0]
    best_k = int(best_phase1["K"])

    phase2_ranges = [(940, 120), (900, 80), (850, 120)]
    for tr in phase2_ranges:
        row = run_one(
            input_wav=args.input,
            model_id=args.model_id,
            out_root=out_root,
            T=100,
            K=best_k,
            P=P,
            CB=CB,
            t_range=tr,
            force=bool(args.force),
            phase="phase2_trange",
        )
        rows.append(row)

    best_after_phase2 = sorted(rows, key=quality_sort_key, reverse=True)[0]
    best_k2 = int(best_after_phase2["K"])
    best_t_range2 = (int(best_after_phase2["t_range"][0]), int(best_after_phase2["t_range"][1]))
    phase3_ts = [int(v) for v in args.phase3_timesteps if int(v) > 0]

    for t_val in phase3_ts:
        row = run_one(
            input_wav=args.input,
            model_id=args.model_id,
            out_root=out_root,
            T=int(t_val),
            K=best_k2,
            P=P,
            CB=CB,
            t_range=best_t_range2,
            force=bool(args.force),
            phase="phase3_timesteps",
        )
        rows.append(row)

    rows_sorted = sorted(rows, key=quality_sort_key, reverse=True)

    result_json = out_root / "quality_push_results.json"
    result_md = out_root / "quality_push_results.md"

    result_json.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| phase | T | K | t_range | corr | snr_db | mel_db_mae | stft_mse | latent_cos | latent_rel_l2 | flow_mse | bytes |",
        "|:--|--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows_sorted:
        latent_cos = r.get("final_latent_cosine")
        latent_rel = r.get("final_latent_rel_l2")
        lines.append(
            f"| {r['phase']} | {r['T']} | {r['K']} | {r['t_range'][0]}-{r['t_range'][1]} | "
            f"{r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | "
            f"{float(latent_cos) if latent_cos is not None else float('nan'):.4f} | "
            f"{float(latent_rel) if latent_rel is not None else float('nan'):.4f} | "
            f"{r['flow_mse']:.2e} | {r['bitstream_total_bytes']} |"
        )

    best = rows_sorted[0]
    lines.append("")
    lines.append("## best")
    lines.append(
        f"- phase={best['phase']}, T={best['T']}, K={best['K']}, t_range={best['t_range'][0]}-{best['t_range'][1]}, "
        f"corr={best['corr']:.4f}, snr={best['snr_db']:.4f}, mel={best['mel_db_mae']:.4f}, bytes={best['bitstream_total_bytes']}"
    )

    result_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "best": best,
                "best_phase1": best_phase1,
                "best_after_phase2": best_after_phase2,
                "result_json": str(result_json).replace("\\", "/"),
                "result_md": str(result_md).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
