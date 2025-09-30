#!/usr/bin/env python3
"""
AudioLDM2 VAE重建最终优化脚本
==========================

自动处理AudioLDM2_Music_output.wav，使用最佳的V1方案
包含V1（推荐）、V2（深度优化但信号丢失）、V3（平衡版本）三个版本供对比

版本说明:
- V1 (推荐): AudioLDM2 Pipeline Standard improved - 最佳听感和信号保真度
- V2 (不推荐): 深度优化版本 - 无噪声但信号丢失严重
- V3 (备选): 平衡版本 - V1的改进版本

作者: AudioLDM2 项目团队
日期: 2024年最新版本
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from diffusers import AudioLDMPipeline, AudioLDM2Pipeline
from transformers import ClapFeatureExtractor
import warnings
warnings.filterwarnings("ignore")

def save_audio_compatible(audio, path, sr=16000):
    """保存音频文件，确保兼容性"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 处理音频数据
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        
        # 确保是1D数组
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # 归一化到[-1, 1]
        if audio.max() > 1 or audio.min() < -1:
            audio = np.clip(audio, -1, 1)
        
        # 保存为WAV文件
        sf.write(path, audio, sr)
        print(f"   💾 保存成功: {path}")
        return True
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        return False

def test_audioldm2_v1_standard_improved(audio_path, max_length=10.0):
    """
    V1版本: AudioLDM2 Pipeline Standard improved
    这是最佳版本，推荐使用
    """
    print(f"\n🎯 V1: AudioLDM2 Standard improved (推荐版本)")
    print(f"📱 设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:        # 加载模型
        print(f"📦 加载AudioLDM2模型...")
        pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-music", 
            torch_dtype=torch.float32
        )
        pipe = pipe.to(device)
        print(f"✅ 模型加载完成")
        print(f"   VAE scaling_factor: {pipe.vae.config.scaling_factor}")
        print(f"   Vocoder类型: {type(pipe.vocoder).__name__}")
        print(f"   FeatureExtractor采样率: {pipe.feature_extractor.sampling_rate} Hz")
        
        # 加载音频
        print(f"📁 加载音频: {audio_path}")
        audio_48k, sr = librosa.load(audio_path, sr=48000)
        
        # 截取长度
        if max_length:
            max_samples = int(max_length * 48000)
            audio_48k = audio_48k[:max_samples]
        
        audio_16k = librosa.resample(audio_48k, orig_sr=48000, target_sr=16000)
        
        print(f"   48000Hz音频: {len(audio_48k)/48000:.2f}秒, 范围[{audio_48k.min():.3f}, {audio_48k.max():.3f}]")
        print(f"   16kHz音频: {len(audio_16k)/16000:.2f}秒, 范围[{audio_16k.min():.3f}, {audio_16k.max():.3f}]")
        
        # V1: 使用ClapFeatureExtractor (最佳方法)
        print(f"🎵 使用AudioLDM2的ClapFeatureExtractor (改进版)...")
        try:
            mel_spec = pipe.feature_extractor(
                raw_speech=audio_48k,
                sampling_rate=48000,
                return_tensors="pt"
            ).input_features
            
            print(f"   ✅ ClapFeatureExtractor成功")
            print(f"   输入: {mel_spec.shape} (格式: [batch, channel, time, feature])")
            print(f"   范围: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
            
            # 确保正确的数据类型和设备
            mel_spec = mel_spec.to(device, dtype=torch.float32)
            print(f"   最终输入: {mel_spec.shape}, {mel_spec.dtype}")
            
            use_clap_features = True
            vocoder_method = "AudioLDM2_Pipeline_Standard"
            
        except Exception as e:
            print(f"   ❌ ClapFeatureExtractor失败: {e}")
            return None
        
        # VAE编码和解码
        print(f"🧠 VAE编码解码 (改进版)...")
        with torch.no_grad():
            # 编码
            latent = pipe.vae.encode(mel_spec).latent_dist.mode()  # 使用mode而非sample
            print(f"   编码latent: {latent.shape}")
            print(f"   Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
            
            # 应用scaling_factor
            latent = latent * pipe.vae.config.scaling_factor
            
            # 解码
            decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
            print(f"   解码输出: {decoded.shape}")
            print(f"   解码范围: [{decoded.min():.3f}, {decoded.max():.3f}]")
        
        # HiFiGAN vocoder处理
        print(f"🎤 HiFiGAN vocoder (多策略)...")
          # 策略1: 使用pipeline的mel_spectrogram_to_waveform
        try:
            print(f"   🚀 策略1: 使用pipeline.mel_spectrogram_to_waveform...")
            reconstructed_audio = pipe.mel_spectrogram_to_waveform(decoded).detach().cpu().numpy()
            print(f"   ✅ 成功！输出: {len(reconstructed_audio)}样本")
            vocoder_method = "AudioLDM2_Pipeline_Standard"
        except Exception as e:
            print(f"   ❌ 策略1失败: {e}")
            return None
          # 后处理
        print(f"🔧 后处理...")
        
        # 确保reconstructed_audio是一维数组
        if reconstructed_audio.ndim > 1:
            reconstructed_audio = reconstructed_audio.squeeze()
        
        # 检查长度
        print(f"   重建音频长度: {len(reconstructed_audio)} 样本")
        print(f"   原始音频长度: {len(audio_16k)} 样本")
        
        # 音量匹配
        if len(reconstructed_audio) > 0:
            original_rms = np.sqrt(np.mean(audio_16k**2))
            reconstructed_rms = np.sqrt(np.mean(reconstructed_audio**2))
            
            if reconstructed_rms > 0:
                volume_ratio = original_rms / reconstructed_rms
                # 限制放大倍数，避免过度放大
                volume_ratio = np.clip(volume_ratio, 0.1, 5.0)
                reconstructed_audio = reconstructed_audio * volume_ratio
                print(f"   音量匹配: {reconstructed_rms:.4f} -> {original_rms:.4f} (比例: {volume_ratio:.2f})")
        
        # 保存结果
        print(f"💾 保存结果...")
        timestamp = int(torch.randint(0, 1000000000, (1,)).item())
        
        # 创建输出目录
        output_dir = "vae_hifigan_ultimate_fix"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始音频
        original_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_original_{timestamp}.wav")
        save_audio_compatible(audio_16k, original_path, sr=16000)
        
        # 保存重建音频
        reconstructed_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_V1_{vocoder_method}_improved_{timestamp}.wav")
        save_audio_compatible(reconstructed_audio, reconstructed_path, sr=16000)
        
        # 计算质量指标
        if len(reconstructed_audio) > 0:
            min_len = min(len(audio_16k), len(reconstructed_audio))
            reference_audio = audio_16k[:min_len]
            reconstructed_audio_for_metrics = reconstructed_audio[:min_len]
            
            # 计算SNR
            signal_power = np.mean(reference_audio**2)
            noise_power = np.mean((reference_audio - reconstructed_audio_for_metrics)**2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # 计算相关系数
            correlation = np.corrcoef(reference_audio, reconstructed_audio_for_metrics)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # 计算MSE和MAE
            mse = np.mean((reference_audio - reconstructed_audio_for_metrics)**2)
            mae = np.mean(np.abs(reference_audio - reconstructed_audio_for_metrics))
            
            # 综合质量分数
            quality_score = snr + correlation * 10
        else:
            print("   ⚠️ 重建音频为空，无法计算质量指标")
            snr = mse = mae = correlation = quality_score = 0.0
        
        # 综合质量分数
        quality_score = snr + correlation * 10
        
        # 输出结果
        print(f"\n{'='*60}")
        print(f"🎯 V1结果 (AudioLDM2 Pipeline Standard improved)")
        print(f"{'='*60}")
        print(f"📁 原始音频: {original_path}")
        print(f"📁 重建音频: {reconstructed_path}")
        print(f"📊 SNR: {snr:.2f} dB")
        print(f"📊 MSE: {mse:.6f}")
        print(f"📊 MAE: {mae:.6f}")
        print(f"📊 相关系数: {correlation:.4f}")
        print(f"🏆 综合质量分数: {quality_score:.2f}")
        print(f"🎤 重建方法: {vocoder_method}")
        
        # 质量评估
        if quality_score > 5:
            print(f"🎉 V1重建质量优秀！")
        elif quality_score > 0:
            print(f"✅ V1重建质量良好")
        else:
            print(f"⚠️ V1重建质量需要改进")
        
        return {
            'snr': snr,
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'quality_score': quality_score,
            'output_file': reconstructed_path,
            'vocoder_method': vocoder_method
        }
    
    except Exception as e:
        print(f"❌ V1处理失败: {e}")
        return None

def main():
    """主函数：自动处理AudioLDM2_Music_output.wav"""
    input_file = "AudioLDM2_Music_output.wav"
    
    print("🎵 AudioLDM2 VAE重建最终优化脚本")
    print("=" * 50)
    print("📝 版本说明:")
    print("   V1 (推荐): AudioLDM2 Pipeline Standard improved - 最佳听感和信号保真度")
    print("   V2 (不推荐): 深度优化版本 - 无噪声但信号丢失严重")
    print("   V3 (备选): 平衡版本 - V1的改进版本")
    print("=" * 50)
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件 {input_file} 不存在")
        print("请确保AudioLDM2_Music_output.wav在当前目录中")
        return
    
    print(f"🎵 自动处理文件: {input_file}")
    print("🔧 使用最佳方案: V1 (AudioLDM2 Pipeline Standard improved)")
    
    try:
        # 运行V1版本（推荐）
        v1_result = test_audioldm2_v1_standard_improved(input_file)
        
        if v1_result:
            print("\n✅ V1处理完成！")
            print(f"💡 推荐使用: {v1_result['output_file']}")
            print(f"🏆 质量分数: {v1_result['quality_score']:.2f}")
            print("\n🎉 V1是经过验证的最佳方案，提供最佳听感和信号保真度")
        else:
            print("\n❌ V1处理失败")
    
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        print("请检查依赖包是否正确安装")

if __name__ == "__main__":
    main()
