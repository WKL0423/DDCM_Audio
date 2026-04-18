import json, subprocess
from pathlib import Path
py = r'E:/ProgramData/Anaconda3/envs/audioldm_env/python.exe'
pairs = [
 ('T10K16P1','runs/T=10_in999-0_K=16_model=audioldm2-music_audio/piano_decomp.wav'),
 ('T10K16P2','runs/T=10_in999-0_K=16_P=2_model=audioldm2-music_audio/piano_decomp.wav'),
 ('T20K32P1','runs/T=20_in999-0_K=32_P=1_model=audioldm2-music_audio/piano_decomp.wav'),
 ('T20K32P2','runs/T=20_in999-0_K=32_P=2_model=audioldm2-music_audio/piano_decomp.wav'),
]
for name, wav in pairs:
    if not Path(wav).exists():
        print(f'MISSING|{name}|{wav}')
        continue
    p = subprocess.run([py,'tools/compare_audio_metrics.py','--ref','piano.wav','--test',wav,'--sr','16000','--max-secs','30'], capture_output=True, text=True)
    if p.returncode != 0:
        print(f'ERROR|{name}')
        continue
    m = json.loads(p.stdout)
    print(f"{name}|corr={m['pearson_corr']:.4f}|snr={m['snr_db']:.4f}|mel_mae={m['mel_db_mae']:.4f}|stft_mse={m['stft_mag_mse']:.4f}")
