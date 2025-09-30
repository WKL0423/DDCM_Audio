from New_pipeline_audioldm2 import AudioLDM2Pipeline
import torch
import scipy

# Load the AudioLDM2 pipeline
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Generate audio using the pipeline
prompt = "Piano romantic melody."
audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0).audios[0]

# Save the generated audio to a WAV file
scipy.io.wavfile.write("piano.wav", rate=16000, data=audio)
