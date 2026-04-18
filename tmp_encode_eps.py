import json
import numpy as np
import pathlib
import traceback
from ddcm.step_codec import StepCodecConfig, encode_step_bitstream

try:
    cfg = StepCodecConfig(
        K_step=1024,
        N_step=150,
        seed=1234,
        atoms_per_step=3,
        coeff_levels=(0.25, 0.5, 0.75, 1.0),
        vggish_weight=0.6,
        vggish_topk=8,
        vggish_interval=1,
        vggish_max_duration=20.0,
        match_epsilon=True,
    )
    bitstream = encode_step_bitstream('AudioLDM2_output.wav', 'cvssp/audioldm2-music', cfg)
    base = pathlib.Path('vggish_k1024_a3_150_eps_inline')
    json.dump(bitstream['meta'], open(base.with_suffix('.json'), 'w'), indent=2)
    stream = bitstream['stream']
    np_data = {
        'mode': np.array(['ddcm_step'], dtype=object),
        'k_indices': np.array(stream['k_indices'], dtype=object),
        'shape': np.array(stream['shape'], dtype=np.int32),
    }
    if stream.get('coeffs') is not None:
        np_data['coeffs'] = np.array(stream['coeffs'], dtype=object)
    if stream.get('signs') is not None:
        np_data['signs'] = np.array(stream['signs'], dtype=object)
    if stream.get('init_latent') is not None:
        np_data['init_latent'] = np.asarray(stream['init_latent'], dtype=np.float32)
    np.savez_compressed(base.with_suffix('.npz'), **np_data)
    print('done')
except Exception:
    traceback.print_exc()
