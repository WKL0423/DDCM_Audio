import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_by(records, key):
    out = {}
    for r in records:
        out.setdefault(r[key], []).append(r)
    return out


def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_metric(records, metric_getter, title, ylabel, out_path: Path):
    # x: steps, series: sampler
    by_sampler = group_by(records, 'sampler')
    x_values = sorted({r['steps'] for r in records})

    plt.figure(figsize=(7, 4.5))
    for sampler, recs in by_sampler.items():
        recs = sorted(recs, key=lambda r: r['steps'])
        y = [metric_getter(r) for r in recs if r['steps'] in x_values]
        x = [r['steps'] for r in recs if r['steps'] in x_values]
        plt.plot(x, y, marker='o', label=sampler)

    plt.title(title)
    plt.xlabel('sampling steps')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark metrics from JSON results.')
    parser.add_argument('--json', type=str, required=True, help='Path to benchmark JSON file')
    parser.add_argument('--out', type=str, default='evaluation_results', help='Output directory for plots')
    args = parser.parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.out)
    ensure_out_dir(out_dir)

    records = load_json(json_path)

    # Runtime
    plot_metric(
        records,
        metric_getter=lambda r: r.get('runtime_sec', float('nan')),
        title='Runtime vs Steps',
        ylabel='seconds',
        out_path=out_dir / 'benchmark_runtime.png'
    )

    # Effective bandwidth
    plot_metric(
        records,
        metric_getter=lambda r: r['metrics']['frequency'].get('effective_bandwidth_0.95_hz', float('nan')),
        title='Effective Bandwidth (0.95) vs Steps',
        ylabel='Hz',
        out_path=out_dir / 'benchmark_effective_bandwidth.png'
    )

    # High mel energy ratio
    plot_metric(
        records,
        metric_getter=lambda r: r['metrics']['mel'].get('high_mel_energy_ratio', float('nan')),
        title='High Mel Energy Ratio vs Steps',
        ylabel='ratio',
        out_path=out_dir / 'benchmark_high_mel_ratio.png'
    )

    # Entropy
    plot_metric(
        records,
        metric_getter=lambda r: r['metrics']['mel'].get('entropy', float('nan')),
        title='Mel Entropy vs Steps',
        ylabel='entropy',
        out_path=out_dir / 'benchmark_entropy.png'
    )

    print(f'Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
