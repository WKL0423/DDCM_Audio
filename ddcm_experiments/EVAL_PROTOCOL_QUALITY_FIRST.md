# DDCM Quality-First Evaluation Protocol

## Scope

This protocol standardizes how DDCM compression experiments are evaluated and compared against the latent codec baseline.

## 1) Fixed Objective and Baseline

- Objective priority: quality first, then rate-distortion tuning.
- Baseline for all comparisons: `runs/baseline/piano_t10k16_*` metrics bundle and canonical baseline pack generated under `ddcm_experiments/baselines/`.

## 2) Metrics and Direction

- Higher-is-better: `pearson_corr`, `snr_db`, `log_mel_corr`, `si_sdr_db`
- Lower-is-better: `mel_db_mae`, `stft_mag_mse`, `waveform_mse`, `log_mel_mse`
- Compression: `bitstream_total_bytes` and `ratio_vs_orig`

## 3) Minimum Run Artifacts

Every reported run must include:

1. `manifest.json`
2. Bitstream payload (`.bin/.json/.npz` as applicable)
3. Reconstructed audio waveform
4. Objective metrics json
5. AB folder with:
   - key mapping (`ab_key.json`)
   - score file (`scores_filled_*.csv`)
   - run-level AB summary (`ab_summary_*.json`)

## 4) Run Labeling Convention

- Include all effective knobs in run names: `T`, `K`, `P`, `CB`, `t_range`, scoring mode, eta schedule.
- Always store `model_id`, `seed`, and command line in the manifest.

## 5) Comparison Table Schema

Minimum columns:

- `candidate_name`
- `dataset_or_input`
- `pearson_corr`
- `snr_db`
- `mel_db_mae`
- `stft_mag_mse`
- `bitstream_total_bytes`
- `ratio_vs_orig`
- `delta_vs_baseline_*` columns for at least two key metrics

## 6) Promotion Gate (Quality-Promising)

Candidate qualifies for next stage only if:

- flow-through validity passes;
- and at least one quality metric improves versus baseline;
- and payload growth remains acceptable for the current experiment phase.

## 7) AB Reporting Rule

- If human scores are not yet available, results must be labeled `proxy` explicitly.
- Human and proxy AB results must never be mixed without tags.

