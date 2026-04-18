from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

PY = r"E:/ProgramData/Anaconda3/envs/audioldm_env/python.exe"


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
    score_mode: str,
    rerank_topk: int,
    force: bool,
    phase: str,
) -> dict[str, Any]:
    rerank_suffix = f"_RK={rerank_topk}" if rerank_topk > 0 else ""
    score_suffix = f"_SM={score_mode}" if score_mode != "dot" else ""
    out_dir = output_root / (
        f"T={T}_in{t_range[0]}-{t_range[1]}_K={K}_P={P}_CB={CB}{rerank_suffix}{score_suffix}_model={model_id.split('/')[-1]}_audio"
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
                "--score-mode",
                score_mode,
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
        "score_mode": str(score_mode),
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
    ap.add_argument("--output_root", type=str, default="runs/score_mode_push")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    T = 100
    P = 2
    CB = 4
    t_range = (900, 80)
    rerank_topk = 0

    rows: list[dict[str, Any]] = []

    # phase1: objective mode sweep
    for score_mode in ["dot", "cosine"]:
        rows.append(
            run_one(
                input_wav=args.input,
                model_id=args.model_id,
                output_root=out_root,
                T=T,
                K=128,
                P=P,
                CB=CB,
                t_range=t_range,
                score_mode=score_mode,
                rerank_topk=rerank_topk,
                force=bool(args.force),
                phase="phase1_score_mode",
            )
        )

    best_phase1 = sorted([r for r in rows if r["phase"] == "phase1_score_mode"], key=quality_key, reverse=True)[0]
    best_mode = str(best_phase1["score_mode"])

    # phase2: large codebook with best score_mode
    for k_val in [128, 256, 512, 1000]:
        rows.append(
            run_one(
                input_wav=args.input,
                model_id=args.model_id,
                output_root=out_root,
                T=T,
                K=int(k_val),
                P=P,
                CB=CB,
                t_range=t_range,
                score_mode=best_mode,
                rerank_topk=rerank_topk,
                force=bool(args.force),
                phase="phase2_large_codebook",
            )
        )

    rows_sorted = sorted(rows, key=quality_key, reverse=True)
    best = rows_sorted[0]

    json_path = out_root / "score_mode_results.json"
    md_path = out_root / "score_mode_results.md"

    json_path.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| phase | T | K | score_mode | t_range | corr | snr_db | mel_db_mae | stft_mse | latent_cos | flow_mse | bytes |",
        "|:--|--:|--:|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows_sorted:
        lat_cos = r.get("final_latent_cosine")
        lines.append(
            f"| {r['phase']} | {r['T']} | {r['K']} | {r['score_mode']} | {r['t_range'][0]}-{r['t_range'][1]} | "
            f"{r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | "
            f"{float(lat_cos) if lat_cos is not None else float('nan'):.4f} | {r['flow_mse']:.2e} | {r['bitstream_total_bytes']} |"
        )

    lines.append("")
    lines.append("## best")
    lines.append(
        f"- phase={best['phase']}, T={best['T']}, K={best['K']}, score_mode={best['score_mode']}, "
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
