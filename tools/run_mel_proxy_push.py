from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

PY = r"E:/ProgramData/Anaconda3/envs/audioldm_env/python.exe"


def run_cmd(args: list[str]) -> str:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "") + "\n" + (p.stdout or ""))
    return p.stdout


def metric(ref: str, test: str) -> dict[str, Any]:
    out = run_cmd([PY, "tools/compare_audio_metrics.py", "--ref", ref, "--test", test, "--sr", "16000", "--max-secs", "30"])
    return json.loads(out)


def quality_key(r: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(r["corr"]),
        float(r["snr_db"]),
        -float(r["mel_db_mae"]),
        float(r.get("final_latent_cosine") or 0.0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="piano.wav")
    ap.add_argument("--model_id", type=str, default="cvssp/audioldm2-music")
    ap.add_argument("--output_root", type=str, default="runs/mel_proxy_push")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    T = 100
    K = 128
    P = 2
    CB = 4
    t_range = (900, 80)

    configs = [
        {"score_mode": "cosine", "mel_proxy_topk": 0, "mel_proxy_interval": 10},
        {"score_mode": "cosine", "mel_proxy_topk": 2, "mel_proxy_interval": 10},
        {"score_mode": "cosine", "mel_proxy_topk": 2, "mel_proxy_interval": 5},
        {"score_mode": "cosine", "mel_proxy_topk": 4, "mel_proxy_interval": 10},
    ]

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        score_mode = str(cfg["score_mode"])
        mpk = int(cfg["mel_proxy_topk"])
        mpi = int(cfg["mel_proxy_interval"])

        mp_suffix = f"_MP={mpk}x{mpi}" if mpk > 0 else ""
        sm_suffix = f"_SM={score_mode}" if score_mode != "dot" else ""
        out_dir = out_root / f"T={T}_in{t_range[0]}-{t_range[1]}_K={K}_P={P}_CB={CB}{mp_suffix}{sm_suffix}_model={args.model_id.split('/')[-1]}_audio"

        decomp = out_dir / "piano_decomp.wav"
        comp = out_dir / "piano_comp.wav"
        meta = out_dir / "piano_noise_indices.json"
        binf = out_dir / "piano_noise_indices.bin"

        if args.force or (not decomp.exists()):
            run_cmd([
                PY,
                "-u",
                "audio_compression.py",
                "roundtrip",
                "--output_dir",
                str(out_root),
                "--input_path",
                args.input,
                "--model_id",
                args.model_id,
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
                "0",
                "--mel-proxy-topk",
                str(mpk),
                "--mel-proxy-interval",
                str(mpi),
                "--audio-proxy-topk",
                "0",
                "--audio-proxy-interval",
                "10",
                "--t_range",
                str(t_range[0]),
                str(t_range[1]),
                "--float16",
            ])

        q = metric(args.input, str(decomp))
        f = metric(str(comp), str(decomp))

        mobj = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        diag = mobj.get("diagnostics") or {}

        total = int((binf.stat().st_size if binf.exists() else 0) + (meta.stat().st_size if meta.exists() else 0))

        rows.append(
            {
                "score_mode": score_mode,
                "mel_proxy_topk": mpk,
                "mel_proxy_interval": mpi,
                "corr": float(q["pearson_corr"]),
                "snr_db": float(q["snr_db"]),
                "mel_db_mae": float(q["mel_db_mae"]),
                "stft_mag_mse": float(q["stft_mag_mse"]),
                "flow_mse": float(f["waveform_mse"]),
                "final_latent_cosine": diag.get("final_latent_cosine"),
                "bitstream_total_bytes": total,
                "out_dir": str(out_dir).replace("\\", "/"),
            }
        )

    rows_sorted = sorted(rows, key=quality_key, reverse=True)
    best = rows_sorted[0]

    json_path = out_root / "mel_proxy_results.json"
    md_path = out_root / "mel_proxy_results.md"
    json_path.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| score_mode | mel_proxy_topk | mel_proxy_interval | corr | snr_db | mel_db_mae | stft_mse | latent_cos | flow_mse | bytes |",
        "|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['score_mode']} | {r['mel_proxy_topk']} | {r['mel_proxy_interval']} | {r['corr']:.4f} | {r['snr_db']:.4f} | "
            f"{r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | {float(r.get('final_latent_cosine') or float('nan')):.4f} | "
            f"{r['flow_mse']:.2e} | {r['bitstream_total_bytes']} |"
        )

    lines.append("")
    lines.append("## best")
    lines.append(
        f"- score_mode={best['score_mode']}, mel_proxy_topk={best['mel_proxy_topk']}, mel_proxy_interval={best['mel_proxy_interval']}, "
        f"corr={best['corr']:.4f}, snr={best['snr_db']:.4f}, mel={best['mel_db_mae']:.4f}, bytes={best['bitstream_total_bytes']}"
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "best": best,
                "result_json": str(json_path).replace("\\", "/"),
                "result_md": str(md_path).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
