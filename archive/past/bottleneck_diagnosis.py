"""
AudioLDM2 VAE重建质量瓶颈分析 - 修复版

针对当前"能听出联系但质量一般"的问题，深度分析瓶颈所在
"""

import torch
import librosa
import numpy as np
import os
import time
from pathlib import Path
import torchaudio
import torch.nn.functional as F

from diffusers import AudioLDM2Pipeline


def quick_bottleneck_analysis(audio_path):
    """
    快速瓶颈分析，找出质量不佳的根本原因
    """
    print(f"🔍 快速瓶颈分析: {audio_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float32).to(device)
    vae = pipeline.vae
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio[:5*16000]  # 5秒
    
    print(f"\\n📊 原始音频统计:")
    print(f"   RMS功率: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"   动态范围: {audio.max():.3f} 到 {audio.min():.3f}")
    print(f"   频谱带宽: 0-8000Hz")
    
    # 测试1: Mel变换信息损失
    print(f"\\n🔬 瓶颈1: Mel变换信息损失")
    
    # 原始STFT
    stft_orig = librosa.stft(audio, n_fft=1024, hop_length=160)
    mag_orig = np.abs(stft_orig)
    
    # Mel变换
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64, n_fft=1024, hop_length=160)
    
    # 从Mel重建线性频谱
    mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=64)
    mag_reconstructed = mel_basis.T @ mel_spec
    
    # 计算损失
    spectral_mse = np.mean((mag_orig - mag_reconstructed) ** 2)
    spectral_corr = np.corrcoef(mag_orig.flatten(), mag_reconstructed.flatten())[0,1]
    
    print(f"   频谱重建相关性: {spectral_corr:.4f}")
    print(f"   频谱信息保留: {spectral_corr*100:.1f}%")
    print(f"   ❌ Mel变换损失: {(1-spectral_corr)*100:.1f}%")
    
    # 测试2: VAE压缩损失
    print(f"\\n🔬 瓶颈2: VAE压缩损失")
    
    # 准备mel输入
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = 2.0 * (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) - 1.0
    mel_tensor = torch.from_numpy(mel_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        latent = vae.encode(mel_tensor).latent_dist.sample()
        reconstructed = vae.decode(latent).sample
    
    recon_mel = reconstructed.squeeze().cpu().numpy()
    
    # 形状对齐
    min_w = min(mel_norm.shape[1], recon_mel.shape[1])
    mel_aligned = mel_norm[:, :min_w]
    recon_aligned = recon_mel[:, :min_w]
    
    vae_corr = np.corrcoef(mel_aligned.flatten(), recon_aligned.flatten())[0,1]
    vae_mse = np.mean((mel_aligned - recon_aligned) ** 2)
    
    print(f"   VAE重建相关性: {vae_corr:.4f}")
    print(f"   VAE信息保留: {vae_corr*100:.1f}%")
    print(f"   ❌ VAE压缩损失: {(1-vae_corr)*100:.1f}%")
    print(f"   压缩比: {mel_tensor.numel() / latent.numel():.1f}:1")
    
    # 测试3: Griffin-Lim重建损失
    print(f"\\n🔬 瓶颈3: Griffin-Lim重建损失")
    
    # 反归一化并重建音频
    recon_denorm = (recon_aligned + 1) / 2 * (mel_db.max() - mel_db.min()) + mel_db.min()
    recon_power = librosa.db_to_power(recon_denorm)
    
    # Griffin-Lim重建
    audio_recon = librosa.feature.inverse.mel_to_audio(
        recon_power, sr=16000, hop_length=160, n_fft=1024, n_iter=32
    )
    
    # 长度对齐
    min_len = min(len(audio), len(audio_recon))
    audio_final = audio[:min_len]
    recon_final = audio_recon[:min_len]
    
    # 最终质量指标
    final_snr = 10 * np.log10(np.mean(audio_final**2) / (np.mean((audio_final - recon_final)**2) + 1e-10))
    final_corr = np.corrcoef(audio_final, recon_final)[0,1]
    
    print(f"   最终音频SNR: {final_snr:.2f} dB")
    print(f"   最终音频相关性: {final_corr:.4f}")
    print(f"   ❌ Griffin-Lim限制明显")
    
    # 整体分析
    total_retention = spectral_corr * vae_corr * abs(final_corr)
    print(f"\\n📉 累积信息损失分析:")
    print(f"   原始 → Mel: 保留 {spectral_corr*100:.1f}%")
    print(f"   Mel → VAE: 保留 {vae_corr*100:.1f}%")
    print(f"   VAE → 音频: 保留 {abs(final_corr)*100:.1f}%")
    print(f"   🔴 总体保留率: {total_retention*100:.1f}%")
    print(f"   🔴 总体损失: {(1-total_retention)*100:.1f}%")
    
    # 关键瓶颈识别
    print(f"\\n⚠️ 关键瓶颈排序:")
    losses = [
        ("Griffin-Lim相位重建", 1 - abs(final_corr)),
        ("VAE压缩损失", 1 - vae_corr),
        ("Mel变换损失", 1 - spectral_corr)
    ]
    losses.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, loss) in enumerate(losses, 1):
        print(f"   #{i} {name}: {loss*100:.1f}% 损失")
    
    return {
        'spectral_retention': spectral_corr,
        'vae_retention': vae_corr, 
        'final_snr': final_snr,
        'final_correlation': final_corr,
        'total_retention': total_retention,
        'primary_bottleneck': losses[0][0]
    }


def test_improvement_strategies(audio_path):
    """
    测试针对性改进策略
    """
    print(f"\\n🚀 测试改进策略")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float32).to(device)
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio[:5*16000]
    
    results = []
    output_dir = "bottleneck_improvement_test"
    os.makedirs(output_dir, exist_ok=True)
    input_name = Path(audio_path).stem
    
    # 策略1: 提高Mel分辨率
    print(f"\\n📈 策略1: 高分辨率Mel (128bins)")
    
    mel_hires = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_mels=128, n_fft=2048, hop_length=128
    )
    
    # 降采样到VAE可接受的64bins
    mel_hires_resized = F.interpolate(
        torch.from_numpy(mel_hires).unsqueeze(0).unsqueeze(0),
        size=(64, mel_hires.shape[1]),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()
    
    # 标准化处理
    mel_db = librosa.power_to_db(mel_hires_resized, ref=np.max)
    mel_norm = 2.0 * (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) - 1.0
    mel_tensor = torch.from_numpy(mel_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        latent = vae.encode(mel_tensor).latent_dist.sample()
        reconstructed = vae.decode(latent).sample
    
    recon_mel = reconstructed.squeeze().cpu().numpy()
    recon_denorm = (recon_mel + 1) / 2 * (mel_db.max() - mel_db.min()) + mel_db.min()
    recon_power = librosa.db_to_power(recon_denorm)
    
    # 高质量Griffin-Lim重建
    audio_strategy1 = librosa.feature.inverse.mel_to_audio(
        recon_power, sr=16000, hop_length=128, n_fft=2048, n_iter=64
    )
    
    # 长度对齐和评估
    min_len = min(len(audio), len(audio_strategy1))
    snr1 = 10 * np.log10(np.mean(audio[:min_len]**2) / (np.mean((audio[:min_len] - audio_strategy1[:min_len])**2) + 1e-10))
    corr1 = np.corrcoef(audio[:min_len], audio_strategy1[:min_len])[0,1]
    
    # 保存
    path1 = os.path.join(output_dir, f"{input_name}_hires_mel.wav")
    torchaudio.save(path1, torch.from_numpy(audio_strategy1[:min_len] / (np.max(np.abs(audio_strategy1[:min_len])) + 1e-8)).unsqueeze(0), 16000)
    
    results.append(("高分辨率Mel", snr1, corr1, path1))
    print(f"   ✅ SNR: {snr1:.2f}dB, 相关性: {corr1:.4f}")
    
    # 策略2: 跳过VAE直接重建（上限测试）
    print(f"\\n📈 策略2: 跳过VAE（理论上限）")
    
    mel_direct = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_mels=64, n_fft=1024, hop_length=160
    )
    
    audio_strategy2 = librosa.feature.inverse.mel_to_audio(
        mel_direct, sr=16000, hop_length=160, n_fft=1024, n_iter=64
    )
    
    min_len = min(len(audio), len(audio_strategy2))
    snr2 = 10 * np.log10(np.mean(audio[:min_len]**2) / (np.mean((audio[:min_len] - audio_strategy2[:min_len])**2) + 1e-10))
    corr2 = np.corrcoef(audio[:min_len], audio_strategy2[:min_len])[0,1]
    
    path2 = os.path.join(output_dir, f"{input_name}_direct_mel.wav")
    torchaudio.save(path2, torch.from_numpy(audio_strategy2[:min_len] / (np.max(np.abs(audio_strategy2[:min_len])) + 1e-8)).unsqueeze(0), 16000)
    
    results.append(("跳过VAE(上限)", snr2, corr2, path2))
    print(f"   ✅ SNR: {snr2:.2f}dB, 相关性: {corr2:.4f}")
    
    # 策略3: 修正Vocoder（如果可行）
    print(f"\\n📈 策略3: 修正Vocoder")
    
    try:
        # 使用之前成功的vocoder方法
        mel_for_vocoder = torch.from_numpy(recon_mel).unsqueeze(0).float().to(device)
        mel_transposed = mel_for_vocoder.transpose(-2, -1)
        
        with torch.no_grad():
            audio_vocoder = vocoder(mel_transposed).squeeze().cpu().numpy()
        
        min_len = min(len(audio), len(audio_vocoder))
        snr3 = 10 * np.log10(np.mean(audio[:min_len]**2) / (np.mean((audio[:min_len] - audio_vocoder[:min_len])**2) + 1e-10))
        corr3 = np.corrcoef(audio[:min_len], audio_vocoder[:min_len])[0,1]
        
        path3 = os.path.join(output_dir, f"{input_name}_fixed_vocoder.wav")
        torchaudio.save(path3, torch.from_numpy(audio_vocoder[:min_len] / (np.max(np.abs(audio_vocoder[:min_len])) + 1e-8)).unsqueeze(0), 16000)
        
        results.append(("修正Vocoder", snr3, corr3, path3))
        print(f"   ✅ SNR: {snr3:.2f}dB, 相关性: {corr3:.4f}")
        
    except Exception as e:
        print(f"   ❌ Vocoder失败: {e}")
    
    # 保存原始音频
    orig_path = os.path.join(output_dir, f"{input_name}_original.wav")
    torchaudio.save(orig_path, torch.from_numpy(audio / (np.max(np.abs(audio)) + 1e-8)).unsqueeze(0), 16000)
    
    # 结果对比
    print(f"\\n🏆 改进策略效果对比:")
    results.sort(key=lambda x: x[1], reverse=True)  # 按SNR排序
    
    for i, (method, snr, corr, path) in enumerate(results, 1):
        improvement = snr - (-0.01)  # 与基线对比
        print(f"   #{i} {method}: SNR={snr:.2f}dB (+{improvement:.2f}), 相关={corr:.4f}")
        print(f"       文件: {path}")
    
    if results:
        best_improvement = results[0][1] - (-0.01)
        print(f"\\n🚀 最佳改进: {best_improvement:+.2f} dB ({results[0][0]})")
        
        if best_improvement > 5:
            print(f"✅ 显著改善！建议采用该方法")
        elif best_improvement > 1:
            print(f"⚠️ 有改善但不显著，需进一步优化")
        else:
            print(f"❌ 改善有限，需要更根本的方法")
    
    return results


def main():
    """主函数"""
    import sys
    
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "AudioLDM2_Music_output.wav"
    
    print(f"🔍 AudioLDM2 VAE重建质量瓶颈诊断")
    print(f"=" * 60)
    print(f"🎯 目标: 找出'能听出联系但质量一般'的根本原因")
    
    # 步骤1: 快速瓶颈分析
    analysis = quick_bottleneck_analysis(audio_path)
    
    # 步骤2: 测试改进策略
    improvements = test_improvement_strategies(audio_path)
    
    # 步骤3: 总结和建议
    print(f"\\n💡 诊断总结:")
    print(f"   🔴 主要瓶颈: {analysis['primary_bottleneck']}")
    print(f"   📉 总体信息保留: {analysis['total_retention']*100:.1f}%")
    print(f"   📈 当前SNR: {analysis['final_snr']:.2f}dB")
    
    if improvements:
        best_snr = max(improvements, key=lambda x: x[1])[1]
        potential_gain = best_snr - analysis['final_snr']
        print(f"   🚀 改进潜力: +{potential_gain:.2f}dB")
    
    print(f"\\n🎯 下一步建议:")
    if analysis['total_retention'] < 0.3:
        print(f"   1. 信息损失过大，建议从架构层面改进")
        print(f"   2. 考虑使用更大的VAE或跳过VAE压缩")
        print(f"   3. 研究端到端的音频重建模型")
    else:
        print(f"   1. 优化mel参数配置")
        print(f"   2. 改进Griffin-Lim算法或使用神经vocoder")
        print(f"   3. 增加后处理步骤")
    
    print(f"\\n📁 测试结果保存在: bottleneck_improvement_test/")
    print(f"🎧 请播放对比音频以进行主观评估")


if __name__ == "__main__":
    main()
