# Step1 Baseline Report (Flow-through Validation)

## Experiment
- Input: `piano.wav`
- Pipeline: `audio_compression.py roundtrip`
- Params: `T=10, K=16, t_range=999..0`
- Output dir: `runs/T=10_in999-0_K=16_model=audioldm2-music_audio/`

## Pass/Fail Criteria
- **Flow-through pass**
  - `comp vs decomp waveform_mse <= 1e-12`
  - `comp vs decomp pearson_corr >= 0.9999`
- **Quality pass**
  - `orig vs decomp pearson_corr >= 0.30`
  - `orig vs decomp snr_db >= 0.0`
  - `orig vs decomp mel_db_mae <= 4.0`

## Measured Results
- `orig vs decomp`
  - `pearson_corr = 0.00779`
  - `snr_db = -1.2318`
  - `mel_db_mae = 10.5640`
- `comp vs decomp`
  - `waveform_mse = 0.0`
  - `pearson_corr = 1.000002`
- Size
  - `orig_wav_bytes = 640058`
  - `bitstream_total_bytes = 1820`
  - `ratio_vs_orig = 351.68x`

## Verdict
- `flow_through_pass = true`
- `quality_pass = false`
- `overall_step1_pass = true` (flow-through objective met)
- `ready_for_step2_quality_optimization = true`

## Artifact Files
- `runs/baseline/piano_t10k16_metrics.json`
- `runs/baseline/piano_t10k16_flow_metrics.json`
- `runs/baseline/piano_t10k16_size.json`
- `runs/baseline/step1_baseline_verdict.json`
