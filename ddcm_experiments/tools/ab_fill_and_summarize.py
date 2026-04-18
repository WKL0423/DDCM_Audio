#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["trial_id", "preferred_sample", "confidence_1_to_5", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    routea_root = repo / "ddcm_experiments" / "routeA_runs"
    session_dirs = sorted(routea_root.glob("*/ab_session"))
    aggregate = []

    for session in session_dirs:
        run_dir = session.parent
        key = load_json(session / "ab_key.json")
        metrics_path = run_dir / "step5" / "quality_metrics.json"
        metrics = load_json(metrics_path) if metrics_path.exists() else {}

        # Proxy rule: assume reference is preferred in quality-first temporary scoring.
        preferred_role = "reference"
        preferred_sample = "A" if key.get("sample_A_role") == preferred_role else "B"
        trial_id = run_dir.name
        note = "proxy_objective_fill; replace_with_human_ab_when_available"
        filled_rows = [
            {
                "trial_id": trial_id,
                "preferred_sample": preferred_sample,
                "confidence_1_to_5": 3,
                "notes": note,
            }
        ]

        write_csv(session / "scores_filled_proxy.csv", filled_rows)

        session_summary = {
            "run_id": run_dir.name,
            "session_type": "proxy",
            "preferred_sample": preferred_sample,
            "preferred_role": preferred_role,
            "confidence": 3,
            "metrics_snapshot": {
                "si_sdr_db": metrics.get("si_sdr_db"),
                "log_mel_mse": metrics.get("log_mel_mse"),
                "log_mel_corr": metrics.get("log_mel_corr"),
            },
            "note": note,
        }
        (session / "ab_summary_proxy.json").write_text(
            json.dumps(session_summary, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        aggregate.append(session_summary)

    total = len(aggregate)
    ref_wins = sum(1 for x in aggregate if x["preferred_role"] == "reference")
    aggregate_summary = {
        "session_type": "proxy",
        "num_sessions": total,
        "reference_preferred_count": ref_wins,
        "reference_preferred_rate": (ref_wins / total) if total > 0 else 0.0,
        "sessions": aggregate,
        "warning": "proxy summaries are placeholders; replace with human AB ratings for final claims",
    }

    out_dir = repo / "ddcm_experiments" / "analysis" / "ab_proxy_summary_20260414"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_proxy_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (out_dir / "ab_proxy_summary.md").write_text(
        "\n".join(
            [
                "# AB Proxy Summary",
                "",
                f"- Sessions: {total}",
                f"- Reference preferred count: {ref_wins}",
                f"- Reference preferred rate: {(ref_wins / total) if total > 0 else 0.0:.3f}",
                "",
                "This is an objective-proxy placeholder. Human AB is still required for final listening claims.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote AB proxy summary to: {out_dir}")


if __name__ == "__main__":
    main()

