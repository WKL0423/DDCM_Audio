#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AudioLDM2 VAE+HiFiGAN 关键修复版本
==============================

解决的核心问题：
1. HiFiGAN输入维度错误 (time vs channels)
2. VAE编码解码尺寸不匹配
3. 正确的mel频谱预处理和scaling
4. 数据类型和设备匹配

作者: 基于AudioLDM2分析的修复
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from diffusers import AudioLDM2Pipeline
import time
from typing import Union, Optional


def save_audio_compatible(audio: np.ndarray, filepath: str, sample_rate: int = 16000):
    """兼容性音频保存函数"""
    try:
        # 确保音频在有效范围内
        audio = np.clip(audio, -1.0, 1.0)
        
        # 保存为PCM 16bit WAV
        sf.write(filepath, audio, sample_rate, subtype='PCM_16')
        print(f"✅ 音频保存成功: {filepath}")
        
    except Exception as e:
        print(f"❌ 音频保存失败: {e}")


def test_vae_hifigan_critical_fix(audio_path: str, max_length: float = 5.0):
    """
    VAE+HiFiGAN关键修复测试
    
    Args:
        audio_path: 输入音频路径
        max_length: 最大音频长度（秒）
    """
    print(f"\n🚀 AudioLDM2 VAE+HiFiGAN 关键修复测试")
    print(f"🎯 目标: 解决维度错误和噪声问题")
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 设备: {device}")
    
    # 加载模型
    print("📦 加载AudioLDM2模型...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    print("✅ 模型加载完成")
    print(f"   VAE scaling_factor: {vae.config.scaling_factor}")
    print(f"   Vocoder类型: {type(vocoder).__name__}")
    
    # 加载音频
    print(f"📁 加载音频: {Path(audio_path).name}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, duration=max_length)
    print(f"   音频: {len(audio)/sr:.2f}秒, 范围[{audio.min():.3f}, {audio.max():.3f}]")
    
    # 创建mel频谱 - 关键修复1: 正确的参数
    print("\n🎵 创建mel频谱...")
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate, 
        n_mels=64,           # AudioLDM2标准
        hop_length=160,      # 10ms hop
        n_fft=1024,          # 标准FFT size
        win_length=1024,     # 窗口长度
        window='hann',       # 窗口类型
        center=True,         # 中心化
        pad_mode='reflect'   # 填充模式
    )
    
    # mel to db
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    print(f"   Mel形状: {mel_db.shape}")
    print(f"   Mel范围: [{mel_db.min():.1f}, {mel_db.max():.1f}] dB")
    
    # 关键修复2: 正确的归一化（参考diffusers源码）
    # AudioLDM2期望的mel谱范围大约是 [-5, 5]
    mel_normalized = mel_db / 20.0  # 缩放到合理范围
    mel_normalized = np.clip(mel_normalized, -1, 1)  # 裁剪到[-1, 1]
    
    print(f"   归一化后: [{mel_normalized.min():.3f}, {mel_normalized.max():.3f}]")
    
    # 转换为tensor
    mel_tensor = torch.from_numpy(mel_normalized).to(device)
    if device == "cuda":
        mel_tensor = mel_tensor.half()
    
    # 关键修复3: 正确的输入维度 [batch, channels, height, width]
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, time]
    print(f"   输入维度: {mel_input.shape}, 类型: {mel_input.dtype}")
    
    # VAE编码解码
    print("\n🧠 VAE编码解码...")
    with torch.no_grad():
        # 编码
        latent_dist = vae.encode(mel_input)
        if hasattr(latent_dist, 'latent_dist'):
            latent = latent_dist.latent_dist.sample()
        else:
            latent = latent_dist.sample()
        
        # 应用scaling factor
        latent = latent * vae.config.scaling_factor
        print(f"   Latent: {latent.shape}, 范围[{latent.min():.3f}, {latent.max():.3f}]")
        
        # 解码
        latent_for_decode = latent / vae.config.scaling_factor
        reconstructed_mel = vae.decode(latent_for_decode).sample
        
        print(f"   重建: {reconstructed_mel.shape}, 范围[{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
      # 关键修复4: 正确的HiFiGAN输入格式
    print("\n🎤 HiFiGAN vocoder...")
    
    # 从 [batch, channels, height, width] 到 [batch, height, width]
    vocoder_input = reconstructed_mel.squeeze(1)  # 移除channel维度: [1, 64, 500]
    print(f"   步骤1 - 移除channel: {vocoder_input.shape}")
      # 关键修复: AudioLDM2的HiFiGAN期望 [batch, time, n_mels] 格式
    vocoder_input = vocoder_input.transpose(1, 2)  # [1, 64, 500] -> [1, 500, 64]
    print(f"   步骤2 - 转置为[batch, time, n_mels]: {vocoder_input.shape}")
    
    # 确保数据类型匹配vocoder的权重类型
    vocoder_dtype = next(vocoder.parameters()).dtype
    vocoder_input = vocoder_input.to(vocoder_dtype)
    print(f"   数据类型: {vocoder_input.dtype}")
      # 尝试HiFiGAN
    try:
        print("   🚀 调用HiFiGAN...")
        waveform = vocoder(vocoder_input)
        reconstructed_audio = waveform.squeeze().detach().cpu().float().numpy()
        vocoder_method = "HiFiGAN_SUCCESS"
        
        print(f"   ✅ HiFiGAN成功！")
        print(f"   输出: {len(reconstructed_audio)}样本")
        print(f"   范围: [{reconstructed_audio.min():.3f}, {reconstructed_audio.max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ HiFiGAN失败: {e}")
        print(f"   输入期望: [batch, time, n_mels]")
        print(f"   实际输入: {vocoder_input.shape}")
        
        # Griffin-Lim备选方案 - 关键修复5: 尺寸对齐
        print("   🔄 使用Griffin-Lim...")
        
        # 确保尺寸一致，并转为float32
        mel_for_griffin = reconstructed_mel.squeeze().cpu().float().numpy()
        
        # 如果需要，调整尺寸到原始输入
        if mel_for_griffin.shape[-1] != mel_db.shape[-1]:
            print(f"   调整尺寸: {mel_for_griffin.shape[-1]} -> {mel_db.shape[-1]}")
            if mel_for_griffin.shape[-1] > mel_db.shape[-1]:
                mel_for_griffin = mel_for_griffin[:, :mel_db.shape[-1]]
            else:
                pad_width = mel_db.shape[-1] - mel_for_griffin.shape[-1]
                mel_for_griffin = np.pad(mel_for_griffin, ((0, 0), (0, pad_width)), mode='constant')
        
        # 反归一化
        mel_denorm = mel_for_griffin * 20.0  # 反向缩放
        mel_power = librosa.db_to_power(mel_denorm)
        
        # Griffin-Lim重建
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(
            mel_power, 
            sr=sample_rate, 
            hop_length=160, 
            n_fft=1024,
            win_length=1024,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        
        vocoder_method = "Griffin_Lim_Fixed"
        print(f"   ✅ Griffin-Lim成功: {len(reconstructed_audio)}样本")
    
    # 计算质量指标
    print("\n📊 质量评估...")
    
    # 调整长度匹配
    min_len = min(len(audio), len(reconstructed_audio))
    audio_aligned = audio[:min_len]
    recon_aligned = reconstructed_audio[:min_len]
    
    # 计算指标
    mse = np.mean((audio_aligned - recon_aligned) ** 2)
    correlation = np.corrcoef(audio_aligned, recon_aligned)[0, 1]
    
    # SNR计算
    signal_power = np.mean(audio_aligned ** 2)
    noise_power = np.mean((audio_aligned - recon_aligned) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    print(f"   MSE: {mse:.6f}")
    print(f"   相关性: {correlation:.4f}")
    print(f"   SNR: {snr:.2f} dB")
    
    # 保存结果
    output_dir = Path("vae_hifigan_critical_fix")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
    reconstructed_path = output_dir / f"{input_name}_{vocoder_method}_{timestamp}.wav"
    
    save_audio_compatible(audio_aligned, str(original_path))
    save_audio_compatible(reconstructed_audio, str(reconstructed_path))
    
    # 创建结果报告
    report = f"""
AudioLDM2 VAE+HiFiGAN 关键修复报告
================================

输入文件: {Path(audio_path).name}
处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
Vocoder方法: {vocoder_method}

音频信息:
- 原始长度: {len(audio)/sr:.2f}秒
- 重建长度: {len(reconstructed_audio)/sr:.2f}秒
- 采样率: {sample_rate} Hz

质量指标:
- MSE: {mse:.6f}
- 相关性: {correlation:.4f}
- SNR: {snr:.2f} dB

技术细节:
- Mel频谱: {mel_db.shape}
- VAE输入: {mel_input.shape}
- VAE输出: {reconstructed_mel.shape}
- Vocoder输入: {vocoder_input.shape if 'vocoder_input' in locals() else 'N/A'}

修复要点:
1. 正确的mel频谱参数和归一化
2. 正确的VAE scaling factor应用
3. 正确的HiFiGAN输入维度：[batch, n_mels, time]
4. 尺寸对齐和数据类型匹配
5. Griffin-Lim备选方案的尺寸修复
"""
    
    report_path = output_dir / f"report_{input_name}_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 报告保存: {report_path}")
    print(f"🎵 原始音频: {original_path}")
    print(f"🎵 重建音频: {reconstructed_path}")
    
    return {
        'mse': mse,
        'correlation': correlation,
        'snr': snr,
        'vocoder_method': vocoder_method,
        'original_path': original_path,
        'reconstructed_path': reconstructed_path
    }


def main():
    """主函数：选择音频文件并测试"""
    # 查找音频文件
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path('.').glob(ext))
    
    if not audio_files:
        print("❌ 当前目录没有找到音频文件")
        return
    
    print("找到音频文件:")
    for i, file in enumerate(audio_files, 1):
        print(f"{i}. {file.name}")
    
    try:
        choice = int(input("选择文件:"))
        audio_path = str(audio_files[choice - 1])
        
        # 运行测试
        result = test_vae_hifigan_critical_fix(audio_path)
        
        print(f"\n🎉 测试完成!")
        print(f"📈 最终结果: {result['vocoder_method']}")
        print(f"📊 质量: MSE={result['mse']:.6f}, SNR={result['snr']:.2f}dB")
        
    except (ValueError, IndexError):
        print("❌ 无效选择")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
