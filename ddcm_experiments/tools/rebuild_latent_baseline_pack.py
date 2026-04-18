#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "ddcm_experiments" / "baselines" / "latent_fixed_20260414"
    out_dir.mkdir(parents=True, exist_ok=True)

    step1_metrics = load_json(repo / "runs" / "baseline" / "piano_t10k16_metrics.json")
    step1_flow = load_json(repo / "runs" / "baseline" / "piano_t10k16_flow_metrics.json")
    step1_size = load_json(repo / "runs" / "baseline" / "piano_t10k16_size.json")
    step1_verdict = load_json(repo / "runs" / "baseline" / "step1_baseline_verdict.json")

    routeb_default_manifest = load_json(
        repo / "ddcm_experiments" / "routeB_runs" / "20260403_115112_p1_baseline_default" / "routeB_manifest.json"
    )
    routeb_default_quality = load_json(
        repo / "ddcm_experiments" / "routeB_runs" / "20260403_115112_p1_baseline_default" / "quality_metrics.json"
    )
    routeb_real_manifest = load_json(
        repo / "ddcm_experiments" / "routeB_runs" / "20260403_155644_p1_real_techno" / "routeB_manifest.json"
    )
    routeb_real_quality = load_json(
        repo / "ddcm_experiments" / "routeB_runs" / "20260403_155644_p1_real_techno" / "quality_metrics.json"
    )

    manifest = {
        "pack_name": "latent_fixed_20260414",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_role": "fixed latent baseline pack for quality-first tracking",
        "sources": {
            "step1": {
                "quality": "runs/baseline/piano_t10k16_metrics.json",
                "flow": "runs/baseline/piano_t10k16_flow_metrics.json",
                "size": "runs/baseline/piano_t10k16_size.json",
                "verdict": "runs/baseline/step1_baseline_verdict.json",
            },
            "routeB_phase1_default": {
                "manifest": "ddcm_experiments/routeB_runs/20260403_115112_p1_baseline_default/routeB_manifest.json",
                "quality": "ddcm_experiments/routeB_runs/20260403_115112_p1_baseline_default/quality_metrics.json",
            },
            "routeB_phase1_real_techno": {
                "manifest": "ddcm_experiments/routeB_runs/20260403_155644_p1_real_techno/routeB_manifest.json",
                "quality": "ddcm_experiments/routeB_runs/20260403_155644_p1_real_techno/quality_metrics.json",
            },
        },
        "records": [
            {
                "id": "step1_piano_t10_k16",
                "route": "audio_compression_roundtrip",
                "metrics": step1_metrics,
                "flow_metrics": step1_flow,
                "size": step1_size,
                "verdict": step1_verdict.get("verdict", {}),
            },
            {
                "id": "routeB_phase1_synthetic_techno_k512_t64",
                "route": "routeB_phase1_latent_codec_only",
                "codec": routeb_default_manifest.get("codec", {}),
                "size": routeb_default_manifest.get("bitstream_files", {}),
                "metrics": routeb_default_quality,
            },
            {
                "id": "routeB_phase1_real_techno_k512_t64",
                "route": "routeB_phase1_latent_codec_only",
                "codec": routeb_real_manifest.get("codec", {}),
                "size": routeb_real_manifest.get("bitstream_files", {}),
                "metrics": routeb_real_quality,
            },
        ],
    }

    summary_rows = [
        "record_id,route,pearson_corr,snr_db,mel_db_mae,si_sdr_db,log_mel_corr,bitstream_total_bytes,ratio_vs_orig",
        (
            f"step1_piano_t10_k16,audio_compression_roundtrip,"
            f"{step1_metrics.get('pearson_corr','')},{step1_metrics.get('snr_db','')},{step1_metrics.get('mel_db_mae','')},"
            f",,{step1_size.get('bitstream_total_bytes','')},{step1_size.get('ratio_vs_orig','')}"
        ),
        (
            f"routeB_phase1_synthetic_techno_k512_t64,routeB_phase1_latent_codec_only,"
            f",,,{routeb_default_quality.get('si_sdr_db','')},{routeb_default_quality.get('log_mel_corr','')},"
            f"{routeb_default_manifest.get('bitstream_files',{}).get('total_bytes','')},"
        ),
        (
            f"routeB_phase1_real_techno_k512_t64,routeB_phase1_latent_codec_only,"
            f",,,{routeb_real_quality.get('si_sdr_db','')},{routeb_real_quality.get('log_mel_corr','')},"
            f"{routeb_real_manifest.get('bitstream_files',{}).get('total_bytes','')},"
        ),
    ]

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    (out_dir / "metrics_summary.csv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "Latent fixed baseline pack",
                "- Use manifest.json for source-of-truth references.",
                "- Use metrics_summary.csv for side-by-side quick comparison.",
                "- This pack is generated from existing run artifacts (no re-run).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote baseline pack to: {out_dir}")


if __name__ == "__main__":
    main()

