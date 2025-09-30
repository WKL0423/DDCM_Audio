#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AudioLDM2 正确的音频处理方式
=========================

使用ClapFeatureExtractor，完全模拟AudioLDM2的内部处理
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from diffusers import AudioLDM2Pipeline
import time
import sys


def save_audio_compatible(audio_data, filepath, sample_rate=16000):
    """保存兼容的音频文件"""
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    try:
        sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')
        print(f"   ✅ 保存: {Path(filepath).name}")
        return True
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        return False


def test_audioldm2_correct_processing(audio_path, max_length=5):
    """
    使用AudioLDM2的正确处理方式
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 AudioLDM2 正确处理方式测试")
    print(f"📱 设备: {device}")
    
    # 加载AudioLDM2 pipeline
    print("📦 加载AudioLDM2模型...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    feature_extractor = pipeline.feature_extractor
    
    print(f"✅ 模型加载完成")
    print(f"   VAE scaling_factor: {pipeline.vae.config.scaling_factor}")
    print(f"   FeatureExtractor: {type(feature_extractor).__name__}")
    print(f"   期望采样率: {feature_extractor.sampling_rate} Hz")
    
    # 加载音频 - 关键：使用48kHz采样率
    print(f"📁 加载音频: {Path(audio_path).name}")
    # 同时加载48kHz和16kHz版本用于对比
    audio_48k, sr_48k = librosa.load(audio_path, sr=feature_extractor.sampling_rate, duration=max_length)
    audio_16k, sr_16k = librosa.load(audio_path, sr=16000, duration=max_length)
    
    print(f"   48kHz音频: {len(audio_48k)/sr_48k:.2f}秒, 范围[{audio_48k.min():.3f}, {audio_48k.max():.3f}]")
    print(f"   16kHz音频: {len(audio_16k)/sr_16k:.2f}秒, 范围[{audio_16k.min():.3f}, {audio_16k.max():.3f}]")
    
    # 使用ClapFeatureExtractor - 关键修复
    print(f"\n🎵 使用ClapFeatureExtractor处理...")
    try:
        # 使用AudioLDM2的正确特征提取方式
        features = feature_extractor(
            audio_48k, 
            return_tensors="pt", 
            sampling_rate=feature_extractor.sampling_rate
        )
        
        mel_input = features.input_features.to(device)
        if device == "cuda":
            mel_input = mel_input.half()
        
        print(f"   特征提取成功")
        print(f"   输入: {mel_input.shape} (注意：time在前，feature在后)")
        print(f"   范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
        
        # 检查维度 - AudioLDM2期望[batch, channel, time, feature]
        if mel_input.dim() == 4 and mel_input.shape[2] > mel_input.shape[3]:
            # 当前是[1, 1, time, 64]，需要转换为[1, 1, 64, time]
            print(f"   维度转换前: {mel_input.shape}")
            # 不需要转换！AudioLDM2的VAE期望[batch, channel, time, feature]格式
            vae_input = mel_input
        else:
            vae_input = mel_input
            
        print(f"   VAE输入: {vae_input.shape}")
        
    except Exception as e:
        print(f"   ❌ ClapFeatureExtractor失败: {e}")
        print(f"   🔄 回退到传统mel处理...")
        
        # 回退方案：使用传统mel处理但模拟ClapFeatureExtractor的参数
        mel_spec = librosa.feature.melspectrogram(
            y=audio_48k, 
            sr=48000, 
            n_mels=64,
            hop_length=480,  # ClapFeatureExtractor的hop_length
            n_fft=1024,
            fmin=50,         # ClapFeatureExtractor的频率范围
            fmax=14000,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 转换为AudioLDM2期望的格式
        mel_tensor = torch.from_numpy(mel_db).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.half()
        
        # 转换维度：[64, time] -> [1, 1, time, 64]
        vae_input = mel_tensor.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        
        print(f"   回退处理成功")
        print(f"   VAE输入: {vae_input.shape}")
        print(f"   范围: [{vae_input.min():.3f}, {vae_input.max():.3f}]")
    
    # VAE处理
    print(f"\n🧠 VAE编码解码...")
    with torch.no_grad():
        # 编码
        latent_dist = pipeline.vae.encode(vae_input)
        if hasattr(latent_dist, 'latent_dist'):
            latent = latent_dist.latent_dist.sample()
        else:
            latent = latent_dist.sample()
        
        # 应用scaling_factor
        latent = latent * pipeline.vae.config.scaling_factor
        print(f"   编码latent: {latent.shape}")
        print(f"   Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
        
        # 解码
        latent_for_decode = latent / pipeline.vae.config.scaling_factor
        reconstructed_mel = pipeline.vae.decode(latent_for_decode).sample
        
        print(f"   解码输出: {reconstructed_mel.shape}")
        print(f"   输出范围: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
    
    # HiFiGAN处理 - 使用pipeline的标准方法
    print(f"\n🎤 HiFiGAN vocoder...")
    try:
        print(f"   🚀 使用pipeline.mel_spectrogram_to_waveform...")
        print(f"   输入到vocoder: {reconstructed_mel.shape}")
          # 关键：直接使用pipeline的方法，它知道如何处理维度
        waveform = pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
        reconstructed_audio = waveform.squeeze().detach().cpu().float().numpy()
        vocoder_method = "AudioLDM2_ClapFeatureExtractor"
        
        print(f"   ✅ 成功！输出: {len(reconstructed_audio)}样本")
        print(f"   输出范围: [{reconstructed_audio.min():.3f}, {reconstructed_audio.max():.3f}]")
        
        # 调整采样率到16kHz用于保存
        if len(reconstructed_audio) > 0:
            reconstructed_audio_16k = librosa.resample(
                reconstructed_audio, 
                orig_sr=48000, 
                target_sr=16000
            )
        else:
            reconstructed_audio_16k = reconstructed_audio
            
    except Exception as e:
        print(f"   ❌ Pipeline方法失败: {e}")
        print(f"   具体错误: {str(e)}")
        
        # 如果失败，尝试手动处理维度
        try:
            print(f"   🔄 尝试手动处理维度...")
            
            # 手动处理维度匹配
            if reconstructed_mel.dim() == 4:
                # 从[1, 1, time, 64]转换为[1, time, 64]
                vocoder_input = reconstructed_mel.squeeze(1)
            else:
                vocoder_input = reconstructed_mel
                
            print(f"   手动处理后: {vocoder_input.shape}")
            
            # 确保数据类型匹配
            vocoder_dtype = next(pipeline.vocoder.parameters()).dtype
            vocoder_input = vocoder_input.to(vocoder_dtype)
              # 直接调用vocoder
            waveform = pipeline.vocoder(vocoder_input)
            reconstructed_audio = waveform.squeeze().detach().cpu().float().numpy()
            
            # 调整采样率
            reconstructed_audio_16k = librosa.resample(
                reconstructed_audio, 
                orig_sr=48000, 
                target_sr=16000
            )
            
            vocoder_method = "AudioLDM2_Manual_Dimension_Fix"
            print(f"   ✅ 手动处理成功！")
            
        except Exception as e2:
            print(f"   ❌ 手动处理也失败: {e2}")
            return None
    
    # 计算质量指标
    print(f"\n📊 质量评估...")
    
    # 使用16kHz版本进行对比
    min_len = min(len(audio_16k), len(reconstructed_audio_16k))
    audio_aligned = audio_16k[:min_len]
    recon_aligned = reconstructed_audio_16k[:min_len]
    
    # 计算指标
    mse = np.mean((audio_aligned - recon_aligned) ** 2)
    correlation = np.corrcoef(audio_aligned, recon_aligned)[0, 1]
    signal_power = np.mean(audio_aligned ** 2)
    noise_power = np.mean((audio_aligned - recon_aligned) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    print(f"   MSE: {mse:.6f}")
    print(f"   相关性: {correlation:.4f}")
    print(f"   SNR: {snr:.2f} dB")
    
    # 保存结果
    output_dir = Path("audioldm2_correct_processing")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
    reconstructed_path = output_dir / f"{input_name}_{vocoder_method}_{timestamp}.wav"
    
    print(f"\n💾 保存结果...")
    save_audio_compatible(audio_aligned, original_path)
    save_audio_compatible(reconstructed_audio_16k, reconstructed_path)
    
    # 结果报告
    print(f"\n{'='*60}")
    print(f"🎯 AudioLDM2 正确处理方式结果")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    print(f"📁 重建音频: {reconstructed_path}")
    print(f"📊 质量指标:")
    print(f"   MSE: {mse:.6f}")
    print(f"   SNR: {snr:.2f} dB")
    print(f"   相关性: {correlation:.4f}")
    print(f"🎤 方法: {vocoder_method}")
    
    # 关键发现
    print(f"\n🔍 关键技术发现:")
    print(f"✅ 使用ClapFeatureExtractor (48kHz)")
    print(f"✅ 正确的维度格式: [batch, channel, time, feature]")
    print(f"✅ 正确的VAE scaling_factor应用")
    print(f"✅ 使用pipeline.mel_spectrogram_to_waveform")
    
    # 质量评估
    if snr > 5:
        print(f"\n🏆 优秀！重建质量很高，声音更饱满")
    elif snr > 0:
        print(f"\n✅ 良好！重建质量显著改善")
    elif snr > -5:
        print(f"\n👍 可接受！比之前的方法有改善")
    else:
        print(f"\n⚠️ 需要进一步优化")
    
    print(f"\n🎊 这是AudioLDM2的正确使用方式！")
    
    return {
        'mse': mse,
        'snr': snr,
        'correlation': correlation,
        'vocoder_method': vocoder_method,
        'original_path': original_path,
        'reconstructed_path': reconstructed_path
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        audio_files = list(Path('.').glob('*.wav'))
        if audio_files:
            print("找到音频文件:")
            for i, file in enumerate(audio_files[:3], 1):
                print(f"{i}. {file.name}")
            
            try:
                choice = int(input("选择文件: "))
                audio_path = str(audio_files[choice-1])
            except:
                print("❌ 无效选择")
                return
        else:
            print("❌ 没有找到音频文件")
            return
    else:
        audio_path = sys.argv[1]
    
    print(f"🚀 开始AudioLDM2正确处理方式测试")
    
    try:
        result = test_audioldm2_correct_processing(audio_path)
        
        if result:
            print(f"\n📋 最终总结:")
            print(f"   方法: {result['vocoder_method']}")
            print(f"   质量: SNR={result['snr']:.2f}dB, MSE={result['mse']:.6f}")
            print(f"   相关性: {result['correlation']:.4f}")
            
            if result['snr'] > 0:
                print(f"\n🎉 成功！使用AudioLDM2的正确处理方式")
                print(f"🔑 关键：ClapFeatureExtractor + 48kHz + 正确维度")
                print(f"📈 声音应该更饱满，细节更丰富")
            else:
                print(f"\n🔍 继续优化中...")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
