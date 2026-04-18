# Quality-Focused Sweep Report

## Baseline

- pearson_corr: 0.007789812
- mel_db_mae: 10.563952446
- snr_db: -1.231804870

## Top Candidates

| rank | source | name | corr | snr_db | mel_db_mae | delta_corr | delta_mel | delta_snr |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | quality_push_large_codebook | runs/quality_push_large_codebook/T=100_in900-80_K=128_P=2_CB=4_model=audioldm2-music_audio | 0.071009 | -1.387404 | 5.679022 | 0.063219 | -4.884930 | -0.155599 |
| 2 | quality_push_large_codebook | runs/quality_push_large_codebook/T=100_in900-80_K=1000_P=2_CB=4_model=audioldm2-music_audio | 0.047076 | -1.637075 | 5.265000 | 0.039286 | -5.298952 | -0.405270 |
| 3 | multiobjective_pareto_push | eta_tail_200_0p9_80 | 0.025721 | -1.168553 | 6.648436 | 0.017932 | -3.915517 | 0.063252 |
| 4 | generalization_push | new_best_schedule_eta_tail | 0.025721 | -1.168553 | 6.648436 | 0.017932 | -3.915517 | 0.063252 |
| 5 | multiobjective_pareto_push | eta_tail_200_0p92_80 | 0.025499 | -1.168036 | 6.646799 | 0.017710 | -3.917154 | 0.063769 |

## Recommendation

- Recommended candidate: `runs/quality_push_large_codebook/T=100_in900-80_K=128_P=2_CB=4_model=audioldm2-music_audio` from `quality_push_large_codebook`.
- Next run should keep this config as quality anchor and only tune one knob at a time.
