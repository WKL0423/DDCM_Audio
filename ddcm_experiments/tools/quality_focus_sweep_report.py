#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "ddcm_experiments" / "analysis" / "quality_focus_20260414"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_json(repo / "runs" / "baseline" / "piano_t10k16_metrics.json")
    quality_push = load_json(repo / "runs" / "quality_push_large_codebook" / "quality_push_results.json")
    multiobjective = load_json(repo / "runs" / "multiobjective_pareto_push" / "multiobjective_results.json")
    generalization = load_json(repo / "runs" / "generalization_push" / "generalization_results.json")

    candidates = []
    for item in quality_push:
        candidates.append(
            {
                "source": "quality_push_large_codebook",
                "name": item.get("out_dir", "unknown"),
                "corr": float(item.get("corr", 0.0)),
                "snr_db": float(item.get("snr_db", 0.0)),
                "mel_db_mae": float(item.get("mel_db_mae", 0.0)),
                "bitstream_total_bytes": int(item.get("bitstream_total_bytes", 0)),
                "ratio_vs_orig": float(item.get("ratio_vs_orig", 0.0)),
            }
        )

    for item in multiobjective.get("summaries", []):
        candidates.append(
            {
                "source": "multiobjective_pareto_push",
                "name": item.get("config", "unknown"),
                "corr": float(item.get("mean_corr", 0.0)),
                "snr_db": float(item.get("mean_snr_db", 0.0)),
                "mel_db_mae": float(item.get("mean_mel_db_mae", 0.0)),
                "bitstream_total_bytes": -1,
                "ratio_vs_orig": -1.0,
            }
        )

    for cfg, item in generalization.get("summary", {}).items():
        candidates.append(
            {
                "source": "generalization_push",
                "name": cfg,
                "corr": float(item.get("mean_corr", 0.0)),
                "snr_db": float(item.get("mean_snr_db", 0.0)),
                "mel_db_mae": float(item.get("mean_mel_db_mae", 0.0)),
                "bitstream_total_bytes": -1,
                "ratio_vs_orig": -1.0,
            }
        )

    # Deduplicate repeated records (same source+name appears in multiple phases).
    unique = {}
    for c in candidates:
        unique[(c["source"], c["name"])] = c
    deduped_candidates = list(unique.values())

    # quality-first ranking: corr desc, mel mae asc, snr desc
    ranked = sorted(deduped_candidates, key=lambda x: (-x["corr"], x["mel_db_mae"], -x["snr_db"]))
    top5 = ranked[:5]

    baseline_corr = float(baseline.get("pearson_corr", 0.0))
    baseline_mel = float(baseline.get("mel_db_mae", 0.0))
    baseline_snr = float(baseline.get("snr_db", 0.0))

    for row in top5:
        row["delta_corr_vs_baseline"] = row["corr"] - baseline_corr
        row["delta_mel_db_mae_vs_baseline"] = row["mel_db_mae"] - baseline_mel
        row["delta_snr_db_vs_baseline"] = row["snr_db"] - baseline_snr

    best = top5[0] if top5 else None
    result = {
        "baseline": {
            "id": "step1_piano_t10_k16",
            "pearson_corr": baseline_corr,
            "mel_db_mae": baseline_mel,
            "snr_db": baseline_snr,
        },
        "ranking_rule": "corr desc, mel_db_mae asc, snr_db desc",
        "top_candidates": top5,
        "recommended_candidate": best,
    }

    csv_lines = [
        "rank,source,name,corr,snr_db,mel_db_mae,bitstream_total_bytes,ratio_vs_orig,delta_corr_vs_baseline,delta_mel_db_mae_vs_baseline,delta_snr_db_vs_baseline"
    ]
    for idx, row in enumerate(top5, start=1):
        csv_lines.append(
            ",".join(
                [
                    str(idx),
                    row["source"],
                    row["name"],
                    f"{row['corr']:.9f}",
                    f"{row['snr_db']:.9f}",
                    f"{row['mel_db_mae']:.9f}",
                    str(row["bitstream_total_bytes"]),
                    str(row["ratio_vs_orig"]),
                    f"{row['delta_corr_vs_baseline']:.9f}",
                    f"{row['delta_mel_db_mae_vs_baseline']:.9f}",
                    f"{row['delta_snr_db_vs_baseline']:.9f}",
                ]
            )
        )

    md_lines = [
        "# Quality-Focused Sweep Report",
        "",
        "## Baseline",
        "",
        f"- pearson_corr: {baseline_corr:.9f}",
        f"- mel_db_mae: {baseline_mel:.9f}",
        f"- snr_db: {baseline_snr:.9f}",
        "",
        "## Top Candidates",
        "",
        "| rank | source | name | corr | snr_db | mel_db_mae | delta_corr | delta_mel | delta_snr |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top5, start=1):
        md_lines.append(
            f"| {idx} | {row['source']} | {row['name']} | {row['corr']:.6f} | {row['snr_db']:.6f} | "
            f"{row['mel_db_mae']:.6f} | {row['delta_corr_vs_baseline']:.6f} | "
            f"{row['delta_mel_db_mae_vs_baseline']:.6f} | {row['delta_snr_db_vs_baseline']:.6f} |"
        )

    if best is not None:
        md_lines.extend(
            [
                "",
                "## Recommendation",
                "",
                f"- Recommended candidate: `{best['name']}` from `{best['source']}`.",
                "- Next run should keep this config as quality anchor and only tune one knob at a time.",
            ]
        )

    (out_dir / "quality_focus_sweep_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (out_dir / "quality_focus_sweep_report.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (out_dir / "quality_focus_sweep_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote sweep report to: {out_dir}")


if __name__ == "__main__":
    main()

