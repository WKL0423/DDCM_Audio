# DDCM One-Page Report (Quality-First, Latent Baseline)

## 1) Current Best Candidate

- Recommended config (from quality-focused sweep):
  - `runs/quality_push_large_codebook/T=100_in900-80_K=128_P=2_CB=4_model=audioldm2-music_audio`
- Key metrics:
  - `corr = 0.07101`
  - `mel_db_mae = 5.67902`
  - `snr_db = -1.38740`
  - `bitstream_total_bytes = 18905`
  - `ratio_vs_orig = 33.8565`

Source: `ddcm_experiments/analysis/quality_focus_20260414/quality_focus_sweep_report.json`

## 2) Gain vs Fixed Latent Baseline

Fixed baseline pack: `ddcm_experiments/baselines/latent_fixed_20260414/manifest.json`

Baseline anchor (`step1_piano_t10_k16`):
- `corr = 0.00779`
- `mel_db_mae = 10.56395`
- `snr_db = -1.23180`

Relative to baseline, current best candidate:
- `delta_corr = +0.06322` (improved)
- `delta_mel_db_mae = -4.88493` (improved)
- `delta_snr_db = -0.15560` (slightly worse)

Interpretation:
- Quality trend is improved on correlation and mel-domain reconstruction.
- SNR remains weak and is still below practical listening quality targets.

## 3) AB Status

- AB loop has been closed with proxy placeholders to keep artifacts complete:
  - `ddcm_experiments/routeA_runs/20260402_141100_techno/ab_session/scores_filled_proxy.csv`
  - `ddcm_experiments/routeA_runs/20260403_091911_ksweep_techno/ab_session/scores_filled_proxy.csv`
  - `ddcm_experiments/analysis/ab_proxy_summary_20260414/ab_proxy_summary.json`
- Current proxy summary:
  - sessions: `2`
  - reference preferred rate: `1.0`

Important:
- Proxy AB is not human listening evidence and must be replaced before final external claims.

## 4) Remaining Risks

- Reported best config has much larger payload than ultra-compressed baseline.
- Route B phase2 inversion is still not implemented, so full route-level quality ceiling is unknown.
- Existing AB data is proxy only (no human confidence-backed result yet).

## 5) Next Decision Gate

Promote to next phase only if a candidate simultaneously satisfies:

1. quality gain over fixed baseline on at least two key metrics;
2. no unacceptable size explosion for target deployment;
3. human AB preference confirms objective trend.

