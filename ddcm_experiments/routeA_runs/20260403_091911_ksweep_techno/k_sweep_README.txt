Route A — K sweep (noise-space rate vs distortion)
==================================================

What this measures
------------------
- One text-to-audio reference run records per-step DDIM variance_noise tensors.
- For each codebook size K, step6 finds the nearest Gaussian atom (same codebook_seed)
  and reports mean squared L2 error in noise space (mse_mean_l2sq) vs uniform index
  coding bitrate (bits_total_uniform, bits_per_sec).

This is NOT waveform RD: it is the DDCM-style tradeoff for discretizing the noise
trajectory. Listening quality is evaluated separately (step5 replay at a chosen K).

Recommended commands
--------------------
Full run (techno prompt, K list 64..2048, AB session):

  set DDCM_LOCAL_FILES_ONLY=1
  python routeA_k_sweep.py

Same but skip blind A/B (faster):

  python routeA_k_sweep.py --skip-ab

Reuse saved step_noises.npz from an old run (re-sweep only, no GPU step5):

  python step6_rd_sweep.py --load-noises path\to\step_noises.npz --Ks 64,128,256,512,1024,2048 --csv out.csv --audio-length 10.24 --codebook-seed 1234

Outputs per run directory (routeA_k_sweep / routeA_experiment)
--------------------------------------------------------------
  manifest.json          hyperparameters and K list
  step_noises.npz        recorded variance noises (for offline sweeps)
  step5/                 reference wav, matched wav at --K, indices, metrics
  step6_rd.csv           one row per K: MSE vs bits
  k_sweep_README.txt     copy of this file in each automated run
  ab_session/            optional blind A/B (sample_A/B.wav)

Future: no-prompt / unconditional runs
--------------------------------------
AudioLDM2 is text-conditioned; empty or minimal prompt experiments may behave
differently. When you add those, keep the same K list and seeds so curves compare.
