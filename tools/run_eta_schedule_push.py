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
    ap.add_argument("--output_root", type=str, default="runs/eta_schedule_push")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    T = 100
    K = 128
    P = 2
    CB = 4
    t_range = (900, 80)

    # keep best objective schedule, only test eta architecture
    score_mode = "dot"
    score_mode_late = "cosine"
    score_switch_t = 300

    configs = [
        {"eta": 1.0, "eta_late": -1.0, "eta_switch_t": -1},
        {"eta": 0.8, "eta_late": -1.0, "eta_switch_t": -1},
        {"eta": 0.6, "eta_late": -1.0, "eta_switch_t": -1},
        {"eta": 1.0, "eta_late": 0.6, "eta_switch_t": 300},
        {"eta": 1.0, "eta_late": 0.4, "eta_switch_t": 300},
        {"eta": 0.8, "eta_late": 0.4, "eta_switch_t": 300},
        {"eta": 1.0, "eta_late": 0.8, "eta_switch_t": 120},
        {"eta": 1.0, "eta_late": 0.8, "eta_switch_t": 80},
        {"eta": 1.0, "eta_late": 0.8, "eta_switch_t": 40},
        {"eta": 1.0, "eta_late": 0.6, "eta_switch_t": 80},
        {"eta": 1.0, "eta_late": 0.9, "eta_switch_t": 80},
        {"eta": 1.0, "eta_late": 0.95, "eta_switch_t": 80},
        {"eta": 1.0, "eta_late": 0.92, "eta_switch_t": 80},
        {"eta": 1.0, "eta_late": 0.9, "eta_switch_t": 100},
        {"eta": 1.0, "eta_late": 0.9, "eta_switch_t": 60},
        {"eta": 1.0, "eta_late": 0.92, "eta_switch_t": 60},
    ]

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        eta = float(cfg["eta"])
        eta_late = float(cfg["eta_late"])
        eta_switch_t = int(cfg["eta_switch_t"])

        eta_suffix = ""
        if eta != 1.0 or eta_late >= 0.0:
            eta_suffix = f"_ETA={eta:.3g}"
            if eta_late >= 0.0 and eta_switch_t >= 0:
                eta_suffix += f"-{eta_late:.3g}@{eta_switch_t}"

        out_dir = out_root / (
            f"T={T}_in{t_range[0]}-{t_range[1]}_K={K}_P={P}_CB={CB}_SL=cosine@{score_switch_t}{eta_suffix}_model={args.model_id.split('/')[-1]}_audio"
        )

        decomp = out_dir / "piano_decomp.wav"
        comp = out_dir / "piano_comp.wav"
        meta = out_dir / "piano_noise_indices.json"
        binf = out_dir / "piano_noise_indices.bin"

        if args.force or (not decomp.exists()):
            run_cmd(
                [
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
                    "--score-mode-late",
                    score_mode_late,
                    "--score-switch-t",
                    str(score_switch_t),
                    "--eta",
                    str(eta),
                    "--eta-late",
                    str(eta_late),
                    "--eta-switch-t",
                    str(eta_switch_t),
                    "--exact-rerank-topk",
                    "0",
                    "--mel-proxy-topk",
                    "0",
                    "--audio-proxy-topk",
                    "0",
                    "--t_range",
                    str(t_range[0]),
                    str(t_range[1]),
                    "--float16",
                ]
            )

        q = metric(args.input, str(decomp))
        f = metric(str(comp), str(decomp))
        mobj = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        diag = mobj.get("diagnostics") or {}
        total = int((binf.stat().st_size if binf.exists() else 0) + (meta.stat().st_size if meta.exists() else 0))

        rows.append(
            {
                "eta": eta,
                "eta_late": eta_late,
                "eta_switch_t": eta_switch_t,
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

    json_path = out_root / "eta_schedule_results.json"
    md_path = out_root / "eta_schedule_results.md"
    json_path.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| eta | eta_late | eta_switch_t | corr | snr_db | mel_db_mae | stft_mse | latent_cos | flow_mse | bytes |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['eta']:.3g} | {r['eta_late']:.3g} | {r['eta_switch_t']} | {r['corr']:.4f} | {r['snr_db']:.4f} | "
            f"{r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | {float(r.get('final_latent_cosine') or float('nan')):.4f} | "
            f"{r['flow_mse']:.2e} | {r['bitstream_total_bytes']} |"
        )

    lines.append("")
    lines.append("## best")
    lines.append(
        f"- eta={best['eta']:.3g}, eta_late={best['eta_late']:.3g}, eta_switch_t={best['eta_switch_t']}, "
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
