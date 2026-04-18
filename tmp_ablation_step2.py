import json
import subprocess
from pathlib import Path

py = r'E:/ProgramData/Anaconda3/envs/audioldm_env/python.exe'
root = Path('runs/ablation')
root.mkdir(parents=True, exist_ok=True)

configs = [
    {'T': 10, 'K': 16, 'P': 1, 'coef_bits': 3},
    {'T': 10, 'K': 16, 'P': 2, 'coef_bits': 3},
    {'T': 20, 'K': 32, 'P': 1, 'coef_bits': 3},
    {'T': 20, 'K': 32, 'P': 2, 'coef_bits': 3},
]

def run_cmd(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or '') + '\n' + (p.stdout or ''))
    return p.stdout

def metric(ref, test):
    out = run_cmd([py, 'tools/compare_audio_metrics.py', '--ref', ref, '--test', test, '--sr', '16000', '--max-secs', '30'])
    return json.loads(out)

rows = []
for cfg in configs:
    out_dir = Path('runs') / f"T={cfg['T']}_in999-0_K={cfg['K']}_P={cfg['P']}_model=audioldm2-music_audio"
    decomp = out_dir / 'piano_decomp.wav'
    comp = out_dir / 'piano_comp.wav'
    meta = out_dir / 'piano_noise_indices.json'
    binf = out_dir / 'piano_noise_indices.bin'

    if not decomp.exists():
        run_cmd([
            py, '-u', 'audio_compression.py', 'roundtrip',
            '--output_dir', 'runs',
            '--input_path', 'piano.wav',
            '-T', str(cfg['T']),
            '-K', str(cfg['K']),
            '--pursuit-noises', str(cfg['P']),
            '--pursuit-coef-bits', str(cfg['coef_bits']),
            '--t_range', '999', '0',
            '--float16',
        ])

    q = metric('piano.wav', str(decomp))
    f = metric(str(comp), str(decomp))

    orig = Path('piano.wav').stat().st_size
    b = binf.stat().st_size if binf.exists() else 0
    j = meta.stat().st_size if meta.exists() else 0
    total = b + j

    rows.append({
        'T': cfg['T'],
        'K': cfg['K'],
        'P': cfg['P'],
        'coef_bits': cfg['coef_bits'],
        'corr': q['pearson_corr'],
        'snr_db': q['snr_db'],
        'mel_db_mae': q['mel_db_mae'],
        'flow_mse': f['waveform_mse'],
        'flow_corr': f['pearson_corr'],
        'bitstream_total_bytes': total,
        'ratio_vs_orig': (orig / total) if total > 0 else None,
        'out_dir': str(out_dir).replace('\\\\','/'),
    })

(rows_path := root / 'ablation_results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

# markdown table
lines = []
lines.append('| T | K | P | corr | snr_db | mel_db_mae | flow_mse | total_bytes | ratio_vs_orig |')
lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    lines.append(f"| {r['T']} | {r['K']} | {r['P']} | {r['corr']:.4f} | {r['snr_db']:.4f} | {r['mel_db_mae']:.4f} | {r['flow_mse']:.2e} | {r['bitstream_total_bytes']} | {r['ratio_vs_orig']:.2f}x |")
(root / 'ablation_results.md').write_text('\n'.join(lines), encoding='utf-8')

best = sorted(rows, key=lambda x: (x['corr'], x['snr_db'], -x['mel_db_mae']), reverse=True)[0]
print(json.dumps({'best_by_quality': best, 'results_file': str(rows_path)}, ensure_ascii=False, indent=2))
