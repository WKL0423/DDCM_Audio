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


def quality_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["corr"]),
        float(row["snr_db"]),
        -float(row["mel_db_mae"]),
        float(row.get("final_latent_cosine") or 0.0),
    )


def run_one(
    *,
    input_wav: str,
    model_id: str,
    output_root: Path,
    T: int,
    K: int,
    P: int,
    CB: int,
    t_range: tuple[int, int],
    rerank_topk: int,
    force: bool,
    phase: str,
) -> dict[str, Any]:
    rerank_suffix = f"_RK={rerank_topk}" if rerank_topk > 0 else ""
    out_dir = output_root / (
        f"T={T}_in{t_range[0]}-{t_range[1]}_K={K}_P={P}_CB={CB}{rerank_suffix}_model={model_id.split('/')[-1]}_audio"
    )

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
                str(output_root),
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
                "--exact-rerank-topk",
                str(rerank_topk),
                "--t_range",
                str(t_range[0]),
                str(t_range[1]),
                "--float16",
            ]
        )

    q = metric(input_wav, str(decomp))
    f = metric(str(comp), str(decomp))

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    diag = meta.get("diagnostics") or {}

    orig = Path(input_wav).stat().st_size
    total = int((bin_path.stat().st_size if bin_path.exists() else 0) + (meta_path.stat().st_size if meta_path.exists() else 0))

    return {
        "phase": phase,
        "T": int(T),
        "K": int(K),
        "P": int(P),
        "coef_bits": int(CB),
        "rerank_topk": int(rerank_topk),
        "t_range": [int(t_range[0]), int(t_range[1])],
        "corr": float(q["pearson_corr"]),
        "snr_db": float(q["snr_db"]),
        "mel_db_mae": float(q["mel_db_mae"]),
        "stft_mag_mse": float(q["stft_mag_mse"]),
        "flow_mse": float(f["waveform_mse"]),
        "flow_corr": float(f["pearson_corr"]),
        "final_latent_cosine": diag.get("final_latent_cosine"),
        "final_latent_rel_l2": diag.get("final_latent_rel_l2"),
        "bitstream_total_bytes": total,
        "ratio_vs_orig": (orig / total) if total > 0 else None,
        "out_dir": str(out_dir).replace("\\", "/"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="piano.wav")
    ap.add_argument("--model_id", type=str, default="cvssp/audioldm2-music")
    ap.add_argument("--output_root", type=str, default="runs/objective_alignment_push")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    P = 2
    CB = 4
    T = 100
    t_range = (900, 80)

    rows: list[dict[str, Any]] = []

    # Phase 1: objective-fix sweep by exact rerank
    rerank_candidates = [0, 4, 8, 16]
    for rk in rerank_candidates:
        rows.append(
            run_one(
                input_wav=args.input,
                model_id=args.model_id,
                output_root=output_root,
                T=T,
                K=128,
                P=P,
                CB=CB,
                t_range=t_range,
                rerank_topk=rk,
                force=bool(args.force),
                phase="phase1_rerank",
            )
        )

    best_phase1 = sorted([r for r in rows if r["phase"] == "phase1_rerank"], key=quality_key, reverse=True)[0]
    best_rk = int(best_phase1["rerank_topk"])

    # Phase 2: large codebook under best objective-fix
    for k_val in [128, 256, 512, 1000]:
        rows.append(
            run_one(
                input_wav=args.input,
                model_id=args.model_id,
                output_root=output_root,
                T=T,
                K=int(k_val),
                P=P,
                CB=CB,
                t_range=t_range,
                rerank_topk=best_rk,
                force=bool(args.force),
                phase="phase2_large_codebook",
            )
        )

    rows_sorted = sorted(rows, key=quality_key, reverse=True)
    best = rows_sorted[0]

    json_path = output_root / "objective_alignment_results.json"
    md_path = output_root / "objective_alignment_results.md"

    json_path.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| phase | T | K | rerank_topk | t_range | corr | snr_db | mel_db_mae | stft_mse | latent_cos | flow_mse | bytes |",
        "|:--|--:|--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows_sorted:
        lat_cos = r.get("final_latent_cosine")
        lines.append(
            f"| {r['phase']} | {r['T']} | {r['K']} | {r['rerank_topk']} | {r['t_range'][0]}-{r['t_range'][1]} | "
            f"{r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | "
            f"{float(lat_cos) if lat_cos is not None else float('nan'):.4f} | {r['flow_mse']:.2e} | {r['bitstream_total_bytes']} |"
        )

    lines.append("")
    lines.append("## best")
    lines.append(
        f"- phase={best['phase']}, T={best['T']}, K={best['K']}, rerank_topk={best['rerank_topk']}, "
        f"corr={best['corr']:.4f}, snr={best['snr_db']:.4f}, mel={best['mel_db_mae']:.4f}, bytes={best['bitstream_total_bytes']}"
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "best": best,
                "best_phase1": best_phase1,
                "result_json": str(json_path).replace("\\", "/"),
                "result_md": str(md_path).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
