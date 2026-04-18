from __future__ import annotations

import argparse
import json
import math
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


def build_out_dir(output_root: Path, model_id: str, cfg: dict[str, Any]) -> Path:
    score_suffix = f"_SM={cfg['score_mode']}" if cfg["score_mode"] != "dot" else ""
    late_suffix = ""
    if cfg["score_mode_late"] and int(cfg["score_switch_t"]) >= 0:
        late_suffix = f"_SL={cfg['score_mode_late']}@{int(cfg['score_switch_t'])}"

    eta_suffix = ""
    if float(cfg["eta"]) != 1.0 or float(cfg["eta_late"]) >= 0.0:
        eta_suffix = f"_ETA={float(cfg['eta']):.3g}"
        if float(cfg["eta_late"]) >= 0.0 and int(cfg["eta_switch_t"]) >= 0:
            eta_suffix += f"-{float(cfg['eta_late']):.3g}@{int(cfg['eta_switch_t'])}"

    return output_root / (
        f"T={int(cfg['T'])}_in{int(cfg['t_range'][0])}-{int(cfg['t_range'][1])}"
        f"_K={int(cfg['K'])}_P={int(cfg['P'])}_CB={int(cfg['CB'])}"
        f"{score_suffix}{late_suffix}{eta_suffix}_model={model_id.split('/')[-1]}_audio"
    )


def ensure_roundtrip(output_root: Path, input_wav: str, model_id: str, cfg: dict[str, Any], force: bool) -> tuple[Path, Path, Path]:
    out_dir = build_out_dir(output_root, model_id, cfg)
    stem = Path(input_wav).stem
    decomp = out_dir / f"{stem}_decomp.wav"
    comp = out_dir / f"{stem}_comp.wav"
    meta = out_dir / f"{stem}_noise_indices.json"

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
                str(int(cfg["T"])),
                "-K",
                str(int(cfg["K"])),
                "--pursuit-noises",
                str(int(cfg["P"])),
                "--pursuit-coef-bits",
                str(int(cfg["CB"])),
                "--score-mode",
                str(cfg["score_mode"]),
                "--score-mode-late",
                str(cfg["score_mode_late"]),
                "--score-switch-t",
                str(int(cfg["score_switch_t"])),
                "--score-blend-lambda",
                str(float(cfg.get("score_blend_lambda", 0.5))),
                "--eta",
                str(float(cfg["eta"])),
                "--eta-late",
                str(float(cfg["eta_late"])),
                "--eta-switch-t",
                str(int(cfg["eta_switch_t"])),
                "--exact-rerank-topk",
                "0",
                "--mel-proxy-topk",
                "0",
                "--audio-proxy-topk",
                "0",
                "--t_range",
                str(int(cfg["t_range"][0])),
                str(int(cfg["t_range"][1])),
                "--float16",
            ]
        )

    return decomp, comp, meta


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    not_worse = (
        a["mean_corr"] >= b["mean_corr"]
        and a["mean_snr_db"] >= b["mean_snr_db"]
        and a["mean_mel_db_mae"] <= b["mean_mel_db_mae"]
        and a["mean_stft_mag_mse"] <= b["mean_stft_mag_mse"]
    )
    strictly_better = (
        a["mean_corr"] > b["mean_corr"]
        or a["mean_snr_db"] > b["mean_snr_db"]
        or a["mean_mel_db_mae"] < b["mean_mel_db_mae"]
        or a["mean_stft_mag_mse"] < b["mean_stft_mag_mse"]
    )
    return bool(not_worse and strictly_better)


def zscore(value: float, values: list[float]) -> float:
    mean = sum(values) / max(len(values), 1)
    var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
    std = math.sqrt(var)
    if std < 1e-12:
        return 0.0
    return (value - mean) / std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=str, default="runs/multiobjective_pareto_push")
    ap.add_argument("--model_id", type=str, default="cvssp/audioldm2-music")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = ["piano.wav", "techno.wav", "custom_techno.wav"]

    candidates: list[dict[str, Any]] = [
        {
            "name": "schedule_only_300",
            "T": 100,
            "K": 128,
            "P": 2,
            "CB": 4,
            "t_range": [900, 80],
            "score_mode": "dot",
            "score_mode_late": "cosine",
            "score_switch_t": 300,
            "score_blend_lambda": 0.5,
            "eta": 1.0,
            "eta_late": -1.0,
            "eta_switch_t": -1,
        },
        {
            "name": "eta_tail_200_0p9_80",
            "T": 100,
            "K": 128,
            "P": 2,
            "CB": 4,
            "t_range": [900, 80],
            "score_mode": "dot",
            "score_mode_late": "cosine",
            "score_switch_t": 200,
            "score_blend_lambda": 0.5,
            "eta": 1.0,
            "eta_late": 0.9,
            "eta_switch_t": 80,
        },
        {
            "name": "eta_tail_300_0p9_80",
            "T": 100,
            "K": 128,
            "P": 2,
            "CB": 4,
            "t_range": [900, 80],
            "score_mode": "dot",
            "score_mode_late": "cosine",
            "score_switch_t": 300,
            "score_blend_lambda": 0.5,
            "eta": 1.0,
            "eta_late": 0.9,
            "eta_switch_t": 80,
        },
        {
            "name": "eta_tail_200_0p92_80",
            "T": 100,
            "K": 128,
            "P": 2,
            "CB": 4,
            "t_range": [900, 80],
            "score_mode": "dot",
            "score_mode_late": "cosine",
            "score_switch_t": 200,
            "score_blend_lambda": 0.5,
            "eta": 1.0,
            "eta_late": 0.92,
            "eta_switch_t": 80,
        },
        {
            "name": "eta_tail_200_0p8_80",
            "T": 100,
            "K": 128,
            "P": 2,
            "CB": 4,
            "t_range": [900, 80],
            "score_mode": "dot",
            "score_mode_late": "cosine",
            "score_switch_t": 200,
            "score_blend_lambda": 0.5,
            "eta": 1.0,
            "eta_late": 0.8,
            "eta_switch_t": 80,
        },
        {
            "name": "schedule_only_200",
            "T": 100,
            "K": 128,
            "P": 2,
            "CB": 4,
            "t_range": [900, 80],
            "score_mode": "dot",
            "score_mode_late": "cosine",
            "score_switch_t": 200,
            "score_blend_lambda": 0.5,
            "eta": 1.0,
            "eta_late": -1.0,
            "eta_switch_t": -1,
        },
    ]

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for cfg in candidates:
        cfg_rows = []
        for inp in inputs:
            decomp, comp, meta = ensure_roundtrip(output_root, inp, args.model_id, cfg, bool(args.force))
            q = metric(inp, str(decomp))
            f = metric(str(comp), str(decomp))
            mobj = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
            diag = mobj.get("diagnostics") or {}
            row = {
                "config": cfg["name"],
                "input": inp,
                "corr": float(q["pearson_corr"]),
                "snr_db": float(q["snr_db"]),
                "mel_db_mae": float(q["mel_db_mae"]),
                "stft_mag_mse": float(q["stft_mag_mse"]),
                "flow_mse": float(f["waveform_mse"]),
                "final_latent_cosine": diag.get("final_latent_cosine"),
                "decomp": str(decomp).replace("\\", "/"),
            }
            rows.append(row)
            cfg_rows.append(row)

        n = max(len(cfg_rows), 1)
        summaries.append(
            {
                "config": cfg["name"],
                "mean_corr": sum(r["corr"] for r in cfg_rows) / n,
                "mean_snr_db": sum(r["snr_db"] for r in cfg_rows) / n,
                "mean_mel_db_mae": sum(r["mel_db_mae"] for r in cfg_rows) / n,
                "mean_stft_mag_mse": sum(r["stft_mag_mse"] for r in cfg_rows) / n,
                "worst_corr": min(r["corr"] for r in cfg_rows),
                "worst_snr_db": min(r["snr_db"] for r in cfg_rows),
            }
        )

    # Pareto front
    pareto = []
    for i, s in enumerate(summaries):
        dominated = False
        for j, t in enumerate(summaries):
            if i == j:
                continue
            if dominates(t, s):
                dominated = True
                break
        if not dominated:
            pareto.append(s)

    corr_vals = [s["mean_corr"] for s in summaries]
    snr_vals = [s["mean_snr_db"] for s in summaries]
    mel_vals = [s["mean_mel_db_mae"] for s in summaries]
    stft_vals = [s["mean_stft_mag_mse"] for s in summaries]

    for s in summaries:
        score = (
            zscore(s["mean_corr"], corr_vals)
            + zscore(s["mean_snr_db"], snr_vals)
            - zscore(s["mean_mel_db_mae"], mel_vals)
            - zscore(s["mean_stft_mag_mse"], stft_vals)
        )
        s["balanced_score"] = float(score)

    best_balanced = sorted(summaries, key=lambda x: x["balanced_score"], reverse=True)[0]

    out_json = output_root / "multiobjective_results.json"
    out_md = output_root / "multiobjective_results.md"

    out_json.write_text(
        json.dumps(
            {
                "candidates": candidates,
                "summaries": summaries,
                "pareto": pareto,
                "best_balanced": best_balanced,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "## Summary",
        "| config | mean_corr | mean_snr_db | mean_mel_db_mae | mean_stft_mse | worst_corr | worst_snr_db | balanced_score |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]

    for s in sorted(summaries, key=lambda x: x["balanced_score"], reverse=True):
        lines.append(
            f"| {s['config']} | {s['mean_corr']:.4f} | {s['mean_snr_db']:.4f} | {s['mean_mel_db_mae']:.4f} | {s['mean_stft_mag_mse']:.4f} | {s['worst_corr']:.4f} | {s['worst_snr_db']:.4f} | {s['balanced_score']:.4f} |"
        )

    lines += ["", "## Pareto"]
    for p in pareto:
        lines.append(
            f"- {p['config']}: corr={p['mean_corr']:.4f}, snr={p['mean_snr_db']:.4f}, mel={p['mean_mel_db_mae']:.4f}, stft={p['mean_stft_mag_mse']:.4f}"
        )

    lines += ["", "## Per-input", "| config | input | corr | snr_db | mel_db_mae | stft_mse | flow_mse |", "|:--|:--|--:|--:|--:|--:|--:|"]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['input']} | {r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | {r['flow_mse']:.2e} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "best_balanced": best_balanced,
                "pareto_count": len(pareto),
                "results_json": str(out_json).replace("\\", "/"),
                "results_md": str(out_md).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
