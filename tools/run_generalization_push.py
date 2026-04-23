from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from ddcm_tool_env import PY


def run_cmd(args: list[str]) -> str:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "") + "\n" + (p.stdout or ""))
    return p.stdout


def metric(ref: str, test: str) -> dict[str, Any]:
    out = run_cmd([PY, "tools/compare_audio_metrics.py", "--ref", ref, "--test", test, "--sr", "16000", "--max-secs", "30"])
    return json.loads(out)


def ensure_roundtrip(
    *,
    output_root: Path,
    input_wav: str,
    model_id: str,
    cfg: dict[str, Any],
    force: bool,
) -> tuple[Path, Path, Path]:
    stem = Path(input_wav).stem

    score_suffix = f"_SM={cfg['score_mode']}" if cfg["score_mode"] != "dot" else ""
    late_suffix = ""
    if cfg["score_mode_late"] and int(cfg["score_switch_t"]) >= 0:
        late_suffix = f"_SL={cfg['score_mode_late']}@{int(cfg['score_switch_t'])}"
    eta_suffix = ""
    if float(cfg["eta"]) != 1.0 or float(cfg["eta_late"]) >= 0.0:
        eta_suffix = f"_ETA={float(cfg['eta']):.3g}"
        if float(cfg["eta_late"]) >= 0.0 and int(cfg["eta_switch_t"]) >= 0:
            eta_suffix += f"-{float(cfg['eta_late']):.3g}@{int(cfg['eta_switch_t'])}"

    out_dir = output_root / (
        f"T={int(cfg['T'])}_in{int(cfg['t_range'][0])}-{int(cfg['t_range'][1])}"
        f"_K={int(cfg['K'])}_P={int(cfg['P'])}_CB={int(cfg['CB'])}"
        f"{score_suffix}{late_suffix}{eta_suffix}_model={model_id.split('/')[-1]}_audio"
    )

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=str, default="runs/generalization_push")
    ap.add_argument("--model_id", type=str, default="cvssp/audioldm2-music")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = ["piano.wav", "techno.wav", "custom_techno.wav"]

    cfg_old = {
        "name": "prev_best_schedule",
        "T": 100,
        "K": 128,
        "P": 2,
        "CB": 4,
        "t_range": [900, 80],
        "score_mode": "dot",
        "score_mode_late": "cosine",
        "score_switch_t": 300,
        "eta": 1.0,
        "eta_late": -1.0,
        "eta_switch_t": -1,
        "score_blend_lambda": 0.5,
    }

    cfg_new = {
        "name": "new_best_schedule_eta_tail",
        "T": 100,
        "K": 128,
        "P": 2,
        "CB": 4,
        "t_range": [900, 80],
        "score_mode": "dot",
        "score_mode_late": "cosine",
        "score_switch_t": 200,
        "eta": 1.0,
        "eta_late": 0.9,
        "eta_switch_t": 80,
        "score_blend_lambda": 0.5,
    }

    configs = [cfg_old, cfg_new]

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        for inp in inputs:
            decomp, comp, meta = ensure_roundtrip(
                output_root=output_root,
                input_wav=inp,
                model_id=args.model_id,
                cfg=cfg,
                force=bool(args.force),
            )

            q = metric(inp, str(decomp))
            f = metric(str(comp), str(decomp))
            mobj = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
            diag = mobj.get("diagnostics") or {}

            rows.append(
                {
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
            )

    # aggregate
    summary: dict[str, dict[str, float]] = {}
    for cfg in [cfg_old["name"], cfg_new["name"]]:
        sub = [r for r in rows if r["config"] == cfg]
        n = max(len(sub), 1)
        summary[cfg] = {
            "mean_corr": sum(r["corr"] for r in sub) / n,
            "mean_snr_db": sum(r["snr_db"] for r in sub) / n,
            "mean_mel_db_mae": sum(r["mel_db_mae"] for r in sub) / n,
            "mean_stft_mag_mse": sum(r["stft_mag_mse"] for r in sub) / n,
        }

    out_json = output_root / "generalization_results.json"
    out_md = output_root / "generalization_results.md"

    out_json.write_text(
        json.dumps(
            {
                "configs": [cfg_old, cfg_new],
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "## Summary (mean over inputs)",
        "| config | mean_corr | mean_snr_db | mean_mel_db_mae | mean_stft_mse |",
        "|:--|--:|--:|--:|--:|",
    ]
    for cfg_name, s in summary.items():
        lines.append(
            f"| {cfg_name} | {s['mean_corr']:.4f} | {s['mean_snr_db']:.4f} | {s['mean_mel_db_mae']:.4f} | {s['mean_stft_mag_mse']:.4f} |"
        )

    lines += [
        "",
        "## Per-input",
        "| config | input | corr | snr_db | mel_db_mae | stft_mse | flow_mse |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]

    for r in rows:
        lines.append(
            f"| {r['config']} | {r['input']} | {r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['stft_mag_mse']:.4f} | {r['flow_mse']:.2e} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": summary,
                "results_json": str(out_json).replace("\\", "/"),
                "results_md": str(out_md).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
