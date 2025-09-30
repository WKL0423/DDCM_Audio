"""
调试 AudioLDM2 的实际 latent 维度
"""

import torch
import torchaudio
from diffusers import AudioLDM2Pipeline
import warnings
warnings.filterwarnings("ignore")

def debug_audioldm2_latent_dimensions():
    """调试 AudioLDM2 的实际潜在空间维度"""
    
    print("🔍 调试 AudioLDM2 潜在空间维度")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载 pipeline
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    
    # 加载测试音频
    audio, sr = torchaudio.load("AudioLDM2_Music_output.wav")
    print(f"📊 原始音频: {audio.shape}, {sr}Hz")
    
    # 重采样到 48kHz
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr, 48000)
        audio = resampler(audio)
    
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    print(f"📊 处理后音频: {audio.shape}")
    
    # 编码为 latent
    with torch.no_grad():
        if audio.is_cuda:
            audio = audio.cpu()
        
        audio_numpy = audio.squeeze(0).numpy()
        
        # ClapFeatureExtractor
        inputs = pipeline.feature_extractor(
            audio_numpy,
            sampling_rate=48000,
            return_tensors="pt"
        )
        
        mel_features = inputs["input_features"].to(device)
        if mel_features.dim() == 3:
            mel_features = mel_features.unsqueeze(1)
        
        if device == "cuda":
            mel_features = mel_features.half()
        
        print(f"📊 Mel features: {mel_features.shape}")
        
        # VAE 编码
        latent_dist = pipeline.vae.encode(mel_features)
        latent = latent_dist.latent_dist.mode() if hasattr(latent_dist.latent_dist, 'mode') else latent_dist.latent_dist.sample()
        
        print(f"📊 Latent: {latent.shape}")
        print(f"📊 Latent 元素总数: {latent.numel()}")
        
        # 分析每个维度
        if latent.dim() == 4:
            batch, channels, height, width = latent.shape
            print(f"📊 Latent 维度分析:")
            print(f"   Batch: {batch}")
            print(f"   Channels: {channels}")
            print(f"   Height (时间): {height}")
            print(f"   Width (频率): {width}")
        
        # 测试不同音频长度
        test_lengths = [1, 5, 10]  # 秒
        print(f"\n🧪 测试不同音频长度的 latent 维度:")
        
        for length in test_lengths:
            test_samples = int(48000 * length)
            if test_samples <= audio.shape[-1]:
                test_audio = audio[..., :test_samples]
                test_audio_numpy = test_audio.squeeze(0).numpy()
                
                test_inputs = pipeline.feature_extractor(
                    test_audio_numpy,
                    sampling_rate=48000,
                    return_tensors="pt"
                )
                
                test_mel = test_inputs["input_features"].to(device)
                if test_mel.dim() == 3:
                    test_mel = test_mel.unsqueeze(1)
                
                if device == "cuda":
                    test_mel = test_mel.half()
                
                test_latent_dist = pipeline.vae.encode(test_mel)
                test_latent = test_latent_dist.latent_dist.mode() if hasattr(test_latent_dist.latent_dist, 'mode') else test_latent_dist.latent_dist.sample()
                
                print(f"   {length}s 音频 -> Latent: {test_latent.shape}")

if __name__ == "__main__":
    debug_audioldm2_latent_dimensions()
