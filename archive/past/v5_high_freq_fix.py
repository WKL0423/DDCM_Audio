#!/usr/bin/env python3
"""
AudioLDM2 高频修复专用脚本
V5版本：尝试高采样率输出和高频增强
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from diffusers import AudioLDM2Pipeline
from pathlib import Path
import time
import os

def save_audio_compatible(audio, path, sr=16000):
    """兼容的音频保存函数"""
    try:
        # 确保音频是numpy数组
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # 确保音频是一维的
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        # 归一化到[-1, 1]
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.max(np.abs(audio))
        
        # 保存为WAV文件
        sf.write(path, audio, sr)
        print(f"   💾 保存成功: {path}")
        return True
        
    except Exception as e:
        print(f"   ❌ 保存失败 {path}: {e}")
        return False

def high_freq_enhancement(audio, sr=16000, enhancement_factor=1.5):
    """高频增强处理"""
    try:
        # 计算频谱
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # 定义高频范围（4kHz以上）
        high_freq_mask = np.abs(freqs) > 4000
        
        # 对高频部分进行增强
        fft[high_freq_mask] *= enhancement_factor
        
        # 转换回时域
        enhanced_audio = np.real(np.fft.ifft(fft))
        
        return enhanced_audio
    except:
        return audio

def test_v5_high_sample_rate(audio_path, max_length=10.0):
    """
    V5版本：高采样率输出 + 高频增强
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🎯 V5: 高采样率输出 + 高频增强")
    print(f"📱 设备: {device}")
    
    # 加载AudioLDM2 pipeline
    print("📦 加载AudioLDM2模型...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",  # 使用音乐专用模型
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    print(f"✅ 模型加载完成")
    print(f"   VAE scaling_factor: {pipeline.vae.config.scaling_factor}")
    print(f"   Vocoder类型: {type(pipeline.vocoder).__name__}")
    
    # 获取feature extractor参数
    fe_sr = pipeline.feature_extractor.sampling_rate
    print(f"   FeatureExtractor采样率: {fe_sr} Hz")
    
    # 加载音频 - 使用高采样率
    print(f"📁 加载音频: {Path(audio_path).name}")
    
    # 使用48kHz保持更多高频信息
    audio_48k, _ = librosa.load(audio_path, sr=48000, duration=max_length)
    audio_fe_sr, _ = librosa.load(audio_path, sr=fe_sr, duration=max_length)
    
    print(f"   48kHz音频: {len(audio_48k)/48000:.2f}秒, 范围[{audio_48k.min():.3f}, {audio_48k.max():.3f}]")
    print(f"   {fe_sr}Hz音频: {len(audio_fe_sr)/fe_sr:.2f}秒, 范围[{audio_fe_sr.min():.3f}, {audio_fe_sr.max():.3f}]")
    
    # 分析原始音频的频谱特征
    print("🔍 分析原始音频频谱...")
    fft_orig = np.abs(np.fft.fft(audio_48k[:48000]))[:24000]  # 取1秒分析
    freqs = np.fft.fftfreq(48000, 1/48000)[:24000]
    
    # 计算不同频段的能量
    low_freq_mask = freqs < 4000
    mid_freq_mask = (freqs >= 4000) & (freqs < 8000)
    high_freq_mask = freqs >= 8000
    
    low_energy = np.sum(fft_orig[low_freq_mask])
    mid_energy = np.sum(fft_orig[mid_freq_mask])
    high_energy = np.sum(fft_orig[high_freq_mask])
    total_energy = low_energy + mid_energy + high_energy
    
    print(f"   低频能量 (0-4kHz): {low_energy/total_energy*100:.1f}%")
    print(f"   中频能量 (4-8kHz): {mid_energy/total_energy*100:.1f}%")
    print(f"   高频能量 (8-24kHz): {high_energy/total_energy*100:.1f}%")
    
    # V5特色：使用ClapFeatureExtractor处理
    print("🎵 V5: 使用ClapFeatureExtractor...")
    try:
        # 轻微归一化
        audio_input = audio_fe_sr.copy()
        peak_value = np.max(np.abs(audio_input))
        if peak_value > 0:
            audio_input = audio_input / peak_value * 0.98
        
        features = pipeline.feature_extractor(
            audio_input, 
            return_tensors="pt", 
            sampling_rate=fe_sr
        )
        
        mel_input = features.input_features.to(device)
        if device == "cuda":
            mel_input = mel_input.half()
        
        print(f"   ✅ ClapFeatureExtractor成功")
        print(f"   输入: {mel_input.shape}")
        print(f"   范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ ClapFeatureExtractor失败: {e}")
        return None
    
    # VAE编码解码
    print("🧠 V5: VAE编码解码...")
    with torch.no_grad():
        # 编码
        latent_dist = pipeline.vae.encode(mel_input)
        latent = latent_dist.latent_dist.mode()
        latent = latent * pipeline.vae.config.scaling_factor
        
        print(f"   编码latent: {latent.shape}")
        print(f"   Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
        
        # 解码
        latent_for_decode = latent / pipeline.vae.config.scaling_factor
        reconstructed_mel = pipeline.vae.decode(latent_for_decode).sample
        
        print(f"   解码输出: {reconstructed_mel.shape}")
        print(f"   解码范围: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
    
    # HiFiGAN vocoder处理
    print("🎤 V5: HiFiGAN vocoder...")
    try:
        waveform = pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
        reconstructed_audio = waveform.squeeze().cpu().detach().numpy()
        print(f"   ✅ Vocoder成功！输出: {len(reconstructed_audio)}样本")
        
        # 检查vocoder输出采样率
        vocoder_sr = len(reconstructed_audio) / max_length
        print(f"   Vocoder输出采样率: {vocoder_sr:.0f} Hz")
        
    except Exception as e:
        print(f"   ❌ Vocoder失败: {e}")
        return None
    
    # V5特色：高频增强后处理
    print("🔧 V5: 高频增强后处理...")
    
    # 上采样到48kHz（如果需要）
    if len(reconstructed_audio) < len(audio_48k):
        # 使用librosa上采样
        current_sr = len(reconstructed_audio) / max_length
        reconstructed_48k = librosa.resample(
            reconstructed_audio, 
            orig_sr=current_sr, 
            target_sr=48000
        )
        print(f"   上采样到48kHz: {len(reconstructed_48k)}样本")
    else:
        reconstructed_48k = reconstructed_audio[:len(audio_48k)]
    
    # 高频增强
    print("🎵 应用高频增强...")
    enhanced_audio = high_freq_enhancement(reconstructed_48k, sr=48000, enhancement_factor=2.0)
    
    # 分析增强后的频谱
    fft_enhanced = np.abs(np.fft.fft(enhanced_audio[:48000]))[:24000]
    
    enhanced_low_energy = np.sum(fft_enhanced[low_freq_mask])
    enhanced_mid_energy = np.sum(fft_enhanced[mid_freq_mask])
    enhanced_high_energy = np.sum(fft_enhanced[high_freq_mask])
    enhanced_total_energy = enhanced_low_energy + enhanced_mid_energy + enhanced_high_energy
    
    print(f"   增强后低频能量: {enhanced_low_energy/enhanced_total_energy*100:.1f}%")
    print(f"   增强后中频能量: {enhanced_mid_energy/enhanced_total_energy*100:.1f}%")
    print(f"   增强后高频能量: {enhanced_high_energy/enhanced_total_energy*100:.1f}%")
    
    # 音量匹配
    ref_rms = np.sqrt(np.mean(audio_48k ** 2))
    enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
    
    if enhanced_rms > 0:
        volume_ratio = ref_rms / enhanced_rms
        volume_ratio = np.clip(volume_ratio, 0.3, 3.0)
        enhanced_audio = enhanced_audio * volume_ratio
        print(f"   音量匹配: {enhanced_rms:.4f} -> {ref_rms:.4f} (比例: {volume_ratio:.2f})")
    
    # 保存结果
    output_dir = Path("vae_hifigan_ultimate_fix")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存48kHz高质量版本
    original_path = output_dir / f"{input_name}_original_48k_{timestamp}.wav"
    enhanced_path = output_dir / f"{input_name}_V5_Enhanced_48k_{timestamp}.wav"
    
    print("💾 保存结果...")
    save_audio_compatible(audio_48k, original_path, sr=48000)
    save_audio_compatible(enhanced_audio, enhanced_path, sr=48000)
    
    # 同时保存16kHz版本供对比
    audio_16k = librosa.resample(audio_48k, orig_sr=48000, target_sr=16000)
    enhanced_16k = librosa.resample(enhanced_audio, orig_sr=48000, target_sr=16000)
    
    original_16k_path = output_dir / f"{input_name}_original_16k_{timestamp}.wav"
    enhanced_16k_path = output_dir / f"{input_name}_V5_Enhanced_16k_{timestamp}.wav"
    
    save_audio_compatible(audio_16k, original_16k_path, sr=16000)
    save_audio_compatible(enhanced_16k, enhanced_16k_path, sr=16000)
    
    # 计算质量指标
    min_len = min(len(audio_48k), len(enhanced_audio))
    reference_audio = audio_48k[:min_len]
    enhanced_audio = enhanced_audio[:min_len]
    
    mse = np.mean((reference_audio - enhanced_audio) ** 2)
    snr = 10 * np.log10(np.mean(reference_audio ** 2) / (mse + 1e-10))
    correlation = np.corrcoef(reference_audio, enhanced_audio)[0, 1] if len(reference_audio) > 1 else 0
    
    # 高频保持率
    high_freq_preserve_rate = enhanced_high_energy / high_energy if high_energy > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"🎯 V5结果 (高采样率 + 高频增强)")
    print(f"{'='*60}")
    print(f"📁 原始音频 (48kHz): {original_path}")
    print(f"📁 增强音频 (48kHz): {enhanced_path}")
    print(f"📁 原始音频 (16kHz): {original_16k_path}")
    print(f"📁 增强音频 (16kHz): {enhanced_16k_path}")
    print(f"📊 SNR: {snr:.2f} dB")
    print(f"📊 MSE: {mse:.6f}")
    print(f"📊 相关系数: {correlation:.4f}")
    print(f"📊 高频保持率: {high_freq_preserve_rate:.3f}")
    print(f"\n🔬 V5特色:")
    print(f"   ✅ 48kHz高采样率输出")
    print(f"   ✅ 高频增强后处理")
    print(f"   ✅ 详细频谱分析")
    print(f"   ✅ 双采样率输出")
    
    if high_freq_preserve_rate > 0.5:
        print(f"🎉 V5高频保持良好！")
    elif high_freq_preserve_rate > 0.2:
        print(f"✅ V5高频有一定改善")
    else:
        print(f"⚠️ V5高频改善有限")
    
    print(f"\n💡 建议:")
    print(f"   1. 对比48kHz和16kHz版本的听感差异")
    print(f"   2. 使用频谱分析工具查看高频恢复效果")
    print(f"   3. 如果效果仍不理想，可能需要考虑其他方法")

def main():
    """主函数"""
    input_file = "AudioLDM2_Music_output.wav"
    
    print("🎵 V5: 高采样率输出 + 高频增强测试")
    print("=" * 50)
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件 {input_file} 不存在")
        return
    
    test_v5_high_sample_rate(input_file)

if __name__ == "__main__":
    main()
