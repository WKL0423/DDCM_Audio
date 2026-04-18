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
    score_switch_t: int,
    eta_late: float,
    eta_switch_t: int,
    force: bool,
    phase: str,
) -> dict[str, Any]:
    out_dir = out_root / (
        f"T={T}_in{t_range[0]}-{t_range[1]}_K={K}_P={P}_CB={CB}"
        f"_SL=cosine@{score_switch_t}_ETA=1-{eta_late:.3g}@{eta_switch_t}_model={model_id.split('/')[-1]}_audio"
    )

    decomp = out_dir / "piano_decomp.wav"
    comp = out_dir / "piano_comp.wav"
    meta = out_dir / "piano_noise_indices.json"
    binf = out_dir / "piano_noise_indices.bin"

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
                "--score-mode",
                "dot",
                "--score-mode-late",
                "cosine",
                "--score-switch-t",
                str(score_switch_t),
                "--score-blend-lambda",
                "0.5",
                "--eta",
                "1.0",
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

    q = metric(input_wav, str(decomp))
    f = metric(str(comp), str(decomp))
    mobj = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    diag = mobj.get("diagnostics") or {}
    total = int((binf.stat().st_size if binf.exists() else 0) + (meta.stat().st_size if meta.exists() else 0))

    return {
        "phase": phase,
        "T": T,
        "K": K,
        "P": P,
        "coef_bits": CB,
        "t_range": [t_range[0], t_range[1]],
        "score_switch_t": int(score_switch_t),
        "eta": 1.0,
        "eta_late": float(eta_late),
        "eta_switch_t": int(eta_switch_t),
        "corr": float(q["pearson_corr"]),
        "snr_db": float(q["snr_db"]),
        "mel_db_mae": float(q["mel_db_mae"]),
        "stft_mag_mse": float(q["stft_mag_mse"]),
        "flow_mse": float(f["waveform_mse"]),
        "final_latent_cosine": diag.get("final_latent_cosine"),
        "bitstream_total_bytes": total,
        "out_dir": str(out_dir).replace("\\", "/"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="piano.wav")
    ap.add_argument("--model_id", type=str, default="cvssp/audioldm2-music")
    ap.add_argument("--output_root", type=str, default="runs/joint_fine_push")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    T = 100
    K = 128
    P = 2
    CB = 4

    baseline_t_range = (900, 80)
    baseline_switch = 300
    baseline_eta_late = 0.9
    baseline_eta_switch_t = 80

    rows: list[dict[str, Any]] = []

    # phase1: score switch refinement
    for sw in [200, 250, 300, 350, 400]:
        rows.append(
            run_one(
                input_wav=args.input,
                model_id=args.model_id,
                out_root=out_root,
                T=T,
                K=K,
                P=P,
                CB=CB,
                t_range=baseline_t_range,
                score_switch_t=int(sw),
                eta_late=baseline_eta_late,
                eta_switch_t=baseline_eta_switch_t,
                force=bool(args.force),
                phase="phase1_switch",
            )
        )

    best_p1 = sorted([r for r in rows if r["phase"] == "phase1_switch"], key=quality_key, reverse=True)[0]

    # phase2: t_range refinement at best switch
    best_sw = int(best_p1["score_switch_t"])
    for tr in [(900, 80), (890, 80), (910, 90), (920, 100), (880, 80)]:
        rows.append(
            run_one(
                input_wav=args.input,
                model_id=args.model_id,
                out_root=out_root,
                T=T,
                K=K,
                P=P,
                CB=CB,
                t_range=tr,
                score_switch_t=best_sw,
                eta_late=baseline_eta_late,
                eta_switch_t=baseline_eta_switch_t,
                force=bool(args.force),
                phase="phase2_trange",
            )
        )

    best_p2 = sorted([r for r in rows if r["phase"] in {"phase1_switch", "phase2_trange"}], key=quality_key, reverse=True)[0]

    # phase3: eta tail refinement at best switch + t_range
    best_sw2 = int(best_p2["score_switch_t"])
    best_tr2 = (int(best_p2["t_range"][0]), int(best_p2["t_range"][1]))
    for el, esw in [(0.8, 80), (0.9, 80), (0.92, 80), (0.95, 80), (0.9, 60), (0.9, 100)]:
        rows.append(
            run_one(
                input_wav=args.input,
                model_id=args.model_id,
                out_root=out_root,
                T=T,
                K=K,
                P=P,
                CB=CB,
                t_range=best_tr2,
                score_switch_t=best_sw2,
                eta_late=float(el),
                eta_switch_t=int(esw),
                force=bool(args.force),
                phase="phase3_eta",
            )
        )

    rows_sorted = sorted(rows, key=quality_key, reverse=True)
    best = rows_sorted[0]

    json_path = out_root / "joint_fine_results.json"
    md_path = out_root / "joint_fine_results.md"
    json_path.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| phase | t_range | switch_t | eta_late@eta_switch | corr | snr_db | mel_db_mae | stft_mse | latent_cos | flow_mse | bytes |",
        "|:--|:--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['phase']} | {r['t_range'][0]}-{r['t_range'][1]} | {r['score_switch_t']} | {r['eta_late']:.3g}@{r['eta_switch_t']} | "
            f"{r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | "
            f"{float(r.get('final_latent_cosine') or float('nan')):.4f} | {r['flow_mse']:.2e} | {r['bitstream_total_bytes']} |"
        )

    lines.append("")
    lines.append("## best")
    lines.append(
        f"- t_range={best['t_range'][0]}-{best['t_range'][1]}, switch_t={best['score_switch_t']}, eta_late={best['eta_late']:.3g}@{best['eta_switch_t']}, "
        f"corr={best['corr']:.4f}, snr={best['snr_db']:.4f}, mel={best['mel_db_mae']:.4f}, bytes={best['bitstream_total_bytes']}"
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "best": best,
                "best_phase1": best_p1,
                "best_phase2": best_p2,
                "result_json": str(json_path).replace("\\", "/"),
                "result_md": str(md_path).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
