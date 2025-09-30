#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深入分析AudioLDM2的ClapFeatureExtractor
"""

from diffusers import AudioLDM2Pipeline
import torch
import numpy as np
import librosa
from pathlib import Path

def analyze_clap_feature_extractor(audio_path):
    """分析ClapFeatureExtractor的工作方式"""
    print("🔍 深入分析ClapFeatureExtractor...")
    
    # 加载pipeline
    pipe = AudioLDM2Pipeline.from_pretrained('cvssp/audioldm2', torch_dtype=torch.float32)
    fe = pipe.feature_extractor
    
    print(f"\n📊 ClapFeatureExtractor详细配置:")
    print(f"- 采样率: {fe.sampling_rate} Hz")
    print(f"- n_fft: {fe.n_fft}")
    print(f"- hop_length: {fe.hop_length}")
    print(f"- 频率范围: {fe.frequency_min}-{fe.frequency_max} Hz")
    print(f"- 特征维度: {fe.feature_size}")
    print(f"- 最大长度: {fe.max_length_s}秒")
    print(f"- chunk_length: {fe.chunk_length_s}秒")
    
    # 加载测试音频
    print(f"\n📁 加载测试音频: {Path(audio_path).name}")
    # 注意：ClapFeatureExtractor期望48kHz
    audio_48k, sr_48k = librosa.load(audio_path, sr=fe.sampling_rate, duration=3)
    audio_16k, sr_16k = librosa.load(audio_path, sr=16000, duration=3)
    
    print(f"- 48kHz音频: {len(audio_48k)/sr_48k:.2f}秒, {len(audio_48k)}样本")
    print(f"- 16kHz音频: {len(audio_16k)/sr_16k:.2f}秒, {len(audio_16k)}样本")
    
    # 测试feature extractor
    print(f"\n🧪 测试ClapFeatureExtractor:")
    try:
        # 使用feature extractor处理音频
        features = fe(audio_48k, return_tensors="pt", sampling_rate=fe.sampling_rate)
        print(f"- 输入音频: {len(audio_48k)}样本")
        print(f"- 输出特征: {features.input_features.shape}")
        print(f"- 特征范围: [{features.input_features.min():.3f}, {features.input_features.max():.3f}]")
        
        # 检查输出格式
        mel_from_fe = features.input_features
        print(f"- 特征维度: {mel_from_fe.shape}")
        
    except Exception as e:
        print(f"- 处理失败: {e}")
        mel_from_fe = None
    
    # 对比我们的mel处理
    print(f"\n📊 对比我们的mel处理:")
    
    # 我们的方法1: 16kHz
    mel_ours_16k = librosa.feature.melspectrogram(
        y=audio_16k, sr=16000, n_mels=64, hop_length=160, n_fft=1024
    )
    mel_ours_16k_db = librosa.power_to_db(mel_ours_16k, ref=np.max)
    print(f"- 我们的16kHz: {mel_ours_16k_db.shape}, 范围[{mel_ours_16k_db.min():.1f}, {mel_ours_16k_db.max():.1f}]dB")
    
    # 我们的方法2: 48kHz，但用ClapFeatureExtractor的参数
    mel_ours_48k = librosa.feature.melspectrogram(
        y=audio_48k, sr=48000, n_mels=64, hop_length=480, n_fft=1024,
        fmin=50, fmax=14000  # 使用ClapFeatureExtractor的频率范围
    )
    mel_ours_48k_db = librosa.power_to_db(mel_ours_48k, ref=np.max)
    print(f"- 我们的48kHz: {mel_ours_48k_db.shape}, 范围[{mel_ours_48k_db.min():.1f}, {mel_ours_48k_db.max():.1f}]dB")
    
    # 测试用不同的mel输入VAE
    print(f"\n🧠 测试不同mel输入的VAE效果:")
    
    methods = []
    
    # 方法1: 我们的16kHz方法
    if mel_ours_16k_db.shape[1] > 0:
        methods.append(("Our_16kHz", mel_ours_16k_db))
    
    # 方法2: 我们的48kHz方法
    if mel_ours_48k_db.shape[1] > 0:
        methods.append(("Our_48kHz", mel_ours_48k_db))
    
    # 方法3: ClapFeatureExtractor
    if mel_from_fe is not None:
        methods.append(("ClapFeatureExtractor", mel_from_fe.squeeze().numpy()))
    
    results = {}
    
    for method_name, mel_data in methods:
        try:
            print(f"\n--- 测试 {method_name} ---")
            
            # 转换为tensor
            mel_tensor = torch.from_numpy(mel_data).float()
            if mel_tensor.dim() == 2:
                mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
            else:
                mel_input = mel_tensor.unsqueeze(0)
            
            print(f"   输入: {mel_input.shape}")
            
            # VAE处理
            with torch.no_grad():
                # 编码
                latent_dist = pipe.vae.encode(mel_input)
                if hasattr(latent_dist, 'latent_dist'):
                    latent = latent_dist.latent_dist.sample()
                else:
                    latent = latent_dist.sample()
                
                latent = latent * pipe.vae.config.scaling_factor
                
                # 解码
                latent_for_decode = latent / pipe.vae.config.scaling_factor
                reconstructed_mel = pipe.vae.decode(latent_for_decode).sample
                
                print(f"   VAE输出: {reconstructed_mel.shape}")
                print(f"   输出范围: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
                
                # 尝试vocoder
                try:
                    waveform = pipe.mel_spectrogram_to_waveform(reconstructed_mel)
                    recon_audio = waveform.squeeze().cpu().numpy()
                    
                    # 计算SNR
                    if method_name == "Our_16kHz":
                        ref_audio = audio_16k
                    else:
                        ref_audio = audio_48k
                        
                    min_len = min(len(ref_audio), len(recon_audio))
                    ref_aligned = ref_audio[:min_len]
                    recon_aligned = recon_audio[:min_len]
                    
                    mse = np.mean((ref_aligned - recon_aligned) ** 2)
                    snr = 10 * np.log10(np.mean(ref_aligned ** 2) / (mse + 1e-10))
                    
                    print(f"   ✅ Vocoder成功: SNR={snr:.2f}dB")
                    
                    results[method_name] = {
                        'snr': snr,
                        'output_range': [recon_audio.min(), recon_audio.max()],
                        'success': True
                    }
                    
                except Exception as ve:
                    print(f"   ❌ Vocoder失败: {ve}")
                    results[method_name] = {'success': False, 'error': str(ve)}
                    
        except Exception as e:
            print(f"   ❌ 整体失败: {e}")
            results[method_name] = {'success': False, 'error': str(e)}
    
    # 总结结果
    print(f"\n📋 结果总结:")
    for method, result in results.items():
        if result['success']:
            print(f"✅ {method}: SNR={result['snr']:.2f}dB, 输出范围{result['output_range']}")
        else:
            print(f"❌ {method}: 失败 - {result['error']}")
    
    return results

def main():
    """主函数"""
    audio_files = list(Path('.').glob('*.wav'))
    if not audio_files:
        print("❌ 没有找到音频文件")
        return
    
    print("找到音频文件:")
    for i, file in enumerate(audio_files[:3], 1):
        print(f"{i}. {file.name}")
    
    try:
        choice = int(input("选择文件: "))
        audio_path = str(audio_files[choice-1])
        
        results = analyze_clap_feature_extractor(audio_path)
        
        print(f"\n🎯 关键发现:")
        print(f"- AudioLDM2使用ClapFeatureExtractor处理音频")
        print(f"- 期望48kHz采样率，不是16kHz")
        print(f"- 使用特定的hop_length=480和频率范围")
        
        # 找到最佳方法
        best_method = None
        best_snr = float('-inf')
        
        for method, result in results.items():
            if result['success'] and result['snr'] > best_snr:
                best_snr = result['snr']
                best_method = method
        
        if best_method:
            print(f"\n🏆 最佳方法: {best_method} (SNR={best_snr:.2f}dB)")
        else:
            print(f"\n⚠️ 没有找到完全成功的方法")
            
    except (ValueError, IndexError):
        print("❌ 无效选择")
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
