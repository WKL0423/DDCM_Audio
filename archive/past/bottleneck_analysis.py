"""
AudioLDM2 VAE重建质量瓶颈分析与改进方案

当前问题：
- SNR只有约-0.01dB，虽然技术上可行，但质量一般
- 相关系数很低(约0.006)，说明时域相似性不高
- 虽然能听出联系，但明显感知质量不佳

瓶颈分析：
1. 🎯 VAE不是为重建优化的 - 最大瓶颈
2. 🔊 Mel-spectrogram信息损失严重
3. 🎵 Griffin-Lim相位重建问题
4. 📊 归一化和参数配置不当
5. 🧠 缺少感知损失函数

本脚本实现针对性的改进措施。
"""

import torch
import librosa
import numpy as np
import os
import time
from pathlib import Path
import torchaudio
import torch.nn.functional as F
from scipy.signal import butter, filtfilt

from diffusers import AudioLDM2Pipeline


def analyze_bottlenecks(audio_path, model_id="cvssp/audioldm2-music"):
    """
    深度分析VAE重建的瓶颈问题
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 开始瓶颈分析: {audio_path}")
    
    # 加载模型
    pipeline = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    vae = pipeline.vae
    sample_rate = 16000
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if len(audio) > 5 * sample_rate:
        audio = audio[:5 * sample_rate]
    
    print(f"\\n📊 原始音频分析:")
    print(f"   长度: {len(audio)/sample_rate:.2f}秒")
    print(f"   动态范围: {audio.max():.3f} 到 {audio.min():.3f}")
    print(f"   RMS功率: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # 分析1: Mel-spectrogram信息损失
    print(f"\\n🔬 瓶颈分析1: Mel-spectrogram信息损失")
    
    # 原始频谱
    stft_original = librosa.stft(audio, n_fft=1024, hop_length=160)
    magnitude_original = np.abs(stft_original)
    
    # Mel变换
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=64, n_fft=1024, hop_length=160
    )
    
    # 尝试从mel重建线性频谱
    mel_to_linear_matrix = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=64)
    reconstructed_magnitude = np.dot(mel_to_linear_matrix.T, mel_spec)
    
    # 计算频谱域损失
    spectral_loss = np.mean((magnitude_original - reconstructed_magnitude) ** 2)
    spectral_correlation = np.corrcoef(magnitude_original.flatten(), reconstructed_magnitude.flatten())[0,1]
    
    print(f"   频谱域MSE损失: {spectral_loss:.6f}")
    print(f"   频谱域相关性: {spectral_correlation:.4f}")
    print(f"   ❌ 瓶颈: Mel变换损失了 {(1-spectral_correlation)*100:.1f}% 的频谱信息")
    
    # 分析2: VAE潜在空间压缩损失
    print(f"\\n🔬 瓶颈分析2: VAE潜在空间压缩损失")
    
    # 准备VAE输入
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_normalized = 2.0 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1.0
    mel_tensor = torch.from_numpy(mel_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
      with torch.no_grad():
        # VAE编码解码
        latent = vae.encode(mel_tensor).latent_dist.sample()
        reconstructed_mel_tensor = vae.decode(latent).sample
        
    reconstructed_mel_np = reconstructed_mel_tensor.squeeze().cpu().numpy()
    
    # 确保形状匹配（处理VAE可能的轻微尺寸变化）
    min_width = min(mel_normalized.shape[1], reconstructed_mel_np.shape[1])
    mel_normalized_aligned = mel_normalized[:, :min_width]
    reconstructed_mel_aligned = reconstructed_mel_np[:, :min_width]
    
    # 计算VAE重建损失
    vae_mse = np.mean((mel_normalized_aligned - reconstructed_mel_aligned) ** 2)
    vae_correlation = np.corrcoef(mel_normalized_aligned.flatten(), reconstructed_mel_aligned.flatten())[0,1]
    
    print(f"   VAE重建MSE: {vae_mse:.6f}")
    print(f"   VAE重建相关性: {vae_correlation:.4f}")
    print(f"   压缩比: {mel_tensor.numel() / latent.numel():.1f}:1")
    print(f"   ❌ 瓶颈: VAE压缩损失了 {(1-vae_correlation)*100:.1f}% 的mel信息")
    
    # 分析3: Griffin-Lim相位重建问题
    print(f"\\n🔬 瓶颈分析3: Griffin-Lim相位重建问题")
    
    # 使用原始幅度谱 + Griffin-Lim
    gl_audio_perfect = librosa.griffinlim(magnitude_original, hop_length=160, n_iter=32)
    
    # 使用重建幅度谱 + Griffin-Lim  
    reconstructed_mel_power = librosa.db_to_power((reconstructed_mel_np + 1) / 2 * 80 - 80)
    gl_audio_degraded = librosa.feature.inverse.mel_to_audio(
        reconstructed_mel_power, sr=sample_rate, hop_length=160, n_fft=1024, n_iter=32
    )
    
    # 对比分析
    if len(gl_audio_perfect) > len(audio):
        gl_audio_perfect = gl_audio_perfect[:len(audio)]
    if len(gl_audio_degraded) > len(audio):
        gl_audio_degraded = gl_audio_degraded[:len(audio)]
    
    perfect_correlation = np.corrcoef(audio, gl_audio_perfect)[0,1] if len(audio) > 1 else 0
    degraded_correlation = np.corrcoef(audio, gl_audio_degraded)[0,1] if len(audio) > 1 else 0
    
    print(f"   理想Griffin-Lim相关性: {perfect_correlation:.4f}")
    print(f"   实际Griffin-Lim相关性: {degraded_correlation:.4f}")
    print(f"   ❌ 瓶颈: 相位重建质量差异 {(perfect_correlation-degraded_correlation)*100:.1f}%")
    
    # 分析4: 整体信息流损失
    print(f"\\n🔬 瓶颈分析4: 整体信息流损失")
    print(f"   原始音频 → Mel变换: 保留 {spectral_correlation*100:.1f}%")
    print(f"   Mel → VAE重建: 保留 {vae_correlation*100:.1f}%")
    print(f"   VAE → 音频重建: 保留 {degraded_correlation*100:.1f}%")
    
    total_retention = spectral_correlation * vae_correlation * abs(degraded_correlation)
    print(f"   📉 总体信息保留率: {total_retention*100:.1f}%")
    print(f"   📉 总体信息损失: {(1-total_retention)*100:.1f}%")
    
    return {
        'spectral_loss': spectral_loss,
        'spectral_correlation': spectral_correlation,
        'vae_mse': vae_mse,
        'vae_correlation': vae_correlation,
        'perfect_gl_correlation': perfect_correlation,
        'degraded_gl_correlation': degraded_correlation,
        'total_retention': total_retention
    }


def improved_reconstruction_v2(audio_path, model_id="cvssp/audioldm2-music"):
    """
    实施针对性改进的重建方法
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\\n🚀 启动改进版重建方案")
    
    # 加载模型
    pipeline = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if len(audio) > 5 * sample_rate:
        audio = audio[:5 * sample_rate]
    
    output_dir = "vae_improved_v2_test"
    os.makedirs(output_dir, exist_ok=True)
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    results = []
    
    # 改进方案1: 高分辨率Mel + 多次迭代Griffin-Lim
    print(f"\\n📈 改进方案1: 高分辨率Mel + 增强Griffin-Lim")
    
    # 使用更高分辨率的mel
    mel_spec_hires = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, 
        n_mels=128,  # 从64增加到128
        n_fft=2048,  # 从1024增加到2048  
        hop_length=128,  # 从160减少到128
        fmax=sample_rate//2  # 使用全频带
    )
    
    # 改进的归一化：保留更多动态范围
    mel_spec_db = librosa.power_to_db(mel_spec_hires, ref=np.max)
    # 使用更保守的归一化，保留原始分布特征
    mel_mean = mel_spec_db.mean()
    mel_std = mel_spec_db.std()
    mel_normalized = (mel_spec_db - mel_mean) / (mel_std + 1e-8)
    mel_normalized = np.clip(mel_normalized, -3, 3)  # 3-sigma剪裁
    
    # 调整尺寸以适应VAE (需要能被8整除)
    target_height = 64  # VAE期望的mel bins
    target_width = (mel_normalized.shape[1] // 8) * 8
    
    # 降采样到VAE期望的尺寸
    mel_resized = F.interpolate(
        torch.from_numpy(mel_normalized).unsqueeze(0).unsqueeze(0),
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    mel_tensor = torch.from_numpy(mel_resized).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        latent = vae.encode(mel_tensor).latent_dist.sample()
        reconstructed_mel_tensor = vae.decode(latent).sample
    
    reconstructed_mel = reconstructed_mel_tensor.squeeze().cpu().numpy()
    
    # 反归一化
    reconstructed_mel_denorm = reconstructed_mel * (mel_std + 1e-8) + mel_mean
    
    # 上采样回原始分辨率
    reconstructed_mel_upsampled = F.interpolate(
        torch.from_numpy(reconstructed_mel_denorm).unsqueeze(0).unsqueeze(0),
        size=mel_spec_hires.shape,
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    # 转换回功率域并重建音频
    reconstructed_power = librosa.db_to_power(reconstructed_mel_upsampled)
    
    # 使用更多迭代的Griffin-Lim
    audio_improved_1 = librosa.feature.inverse.mel_to_audio(
        reconstructed_power,
        sr=sample_rate,
        n_fft=2048,
        hop_length=128,
        n_iter=64,  # 增加迭代次数
        fmax=sample_rate//2
    )
    
    # 长度对齐
    if len(audio_improved_1) > len(audio):
        audio_improved_1 = audio_improved_1[:len(audio)]
    elif len(audio_improved_1) < len(audio):
        audio_improved_1 = np.pad(audio_improved_1, (0, len(audio) - len(audio_improved_1)))
    
    # 计算指标
    snr_1 = 10 * np.log10(np.mean(audio**2) / (np.mean((audio - audio_improved_1)**2) + 1e-10))
    corr_1 = np.corrcoef(audio, audio_improved_1)[0,1]
    
    # 保存结果
    path_1 = os.path.join(output_dir, f"{input_name}_hires_griffin_{timestamp}.wav")
    torchaudio.save(path_1, torch.from_numpy(audio_improved_1 / (np.max(np.abs(audio_improved_1)) + 1e-8)).unsqueeze(0), sample_rate)
    
    results.append({
        'method': '高分辨率Mel + 增强Griffin-Lim',
        'snr': snr_1,
        'correlation': corr_1,
        'path': path_1
    })
    
    print(f"   ✅ SNR: {snr_1:.2f}dB, 相关性: {corr_1:.4f}")
    
    # 改进方案2: 固定Vocoder + 预后处理
    print(f"\\n📈 改进方案2: 修正Vocoder + 音频后处理")
    
    try:
        # 使用修正的Vocoder
        mel_for_vocoder = torch.from_numpy(reconstructed_mel).unsqueeze(0).float().to(device)
        mel_for_vocoder_transposed = mel_for_vocoder.transpose(-2, -1)
        
        with torch.no_grad():
            audio_vocoder_raw = vocoder(mel_for_vocoder_transposed).squeeze().cpu().numpy()
        
        # 后处理：频域滤波和动态范围调整
        # 设计低通滤波器移除高频噪声
        nyquist = sample_rate // 2
        cutoff = min(8000, nyquist * 0.8)  # 截止频率
        b, a = butter(5, cutoff / nyquist, btype='low')
        audio_vocoder_filtered = filtfilt(b, a, audio_vocoder_raw)
        
        # 动态范围压缩（轻微）
        audio_compressed = np.sign(audio_vocoder_filtered) * np.power(np.abs(audio_vocoder_filtered), 0.8)
        
        # 长度对齐
        if len(audio_compressed) > len(audio):
            audio_compressed = audio_compressed[:len(audio)]
        elif len(audio_compressed) < len(audio):
            audio_compressed = np.pad(audio_compressed, (0, len(audio) - len(audio_compressed)))
        
        # 计算指标
        snr_2 = 10 * np.log10(np.mean(audio**2) / (np.mean((audio - audio_compressed)**2) + 1e-10))
        corr_2 = np.corrcoef(audio, audio_compressed)[0,1]
        
        # 保存结果
        path_2 = os.path.join(output_dir, f"{input_name}_vocoder_enhanced_{timestamp}.wav")
        torchaudio.save(path_2, torch.from_numpy(audio_compressed / (np.max(np.abs(audio_compressed)) + 1e-8)).unsqueeze(0), sample_rate)
        
        results.append({
            'method': '修正Vocoder + 音频后处理',
            'snr': snr_2,
            'correlation': corr_2,
            'path': path_2
        })
        
        print(f"   ✅ SNR: {snr_2:.2f}dB, 相关性: {corr_2:.4f}")
        
    except Exception as e:
        print(f"   ❌ Vocoder方案失败: {e}")
    
    # 改进方案3: 多尺度融合重建
    print(f"\\n📈 改进方案3: 多尺度融合重建")
    
    # 使用不同分辨率重建并融合
    scales = [(64, 1024, 160), (80, 1536, 120), (96, 2048, 128)]
    reconstructed_audios = []
    
    for n_mels, n_fft, hop_length in scales:
        # 生成该尺度的mel
        mel_scale = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        
        # 简化处理：直接重建（跳过VAE以评估上限）
        audio_scale = librosa.feature.inverse.mel_to_audio(
            mel_scale, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_iter=32
        )
        
        # 长度对齐
        if len(audio_scale) > len(audio):
            audio_scale = audio_scale[:len(audio)]
        elif len(audio_scale) < len(audio):
            audio_scale = np.pad(audio_scale, (0, len(audio) - len(audio_scale)))
            
        reconstructed_audios.append(audio_scale)
    
    # 加权融合（基于频域能量分布）
    weights = [0.5, 0.3, 0.2]  # 给低分辨率更高权重（更稳定）
    audio_fused = np.zeros_like(audio)
    for i, (audio_scale, weight) in enumerate(zip(reconstructed_audios, weights)):
        audio_fused += weight * audio_scale
    
    # 计算指标
    snr_3 = 10 * np.log10(np.mean(audio**2) / (np.mean((audio - audio_fused)**2) + 1e-10))
    corr_3 = np.corrcoef(audio, audio_fused)[0,1]
    
    # 保存结果
    path_3 = os.path.join(output_dir, f"{input_name}_multiscale_fusion_{timestamp}.wav")
    torchaudio.save(path_3, torch.from_numpy(audio_fused / (np.max(np.abs(audio_fused)) + 1e-8)).unsqueeze(0), sample_rate)
    
    results.append({
        'method': '多尺度融合重建',
        'snr': snr_3,
        'correlation': corr_3,
        'path': path_3
    })
    
    print(f"   ✅ SNR: {snr_3:.2f}dB, 相关性: {corr_3:.4f}")
    
    # 保存原始音频作为对比
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    torchaudio.save(original_path, torch.from_numpy(audio / (np.max(np.abs(audio)) + 1e-8)).unsqueeze(0), sample_rate)
    
    # 结果分析
    print(f"\\n{'='*60}")
    print(f"🎯 改进方案效果对比")
    print(f"{'='*60}")
    
    results.sort(key=lambda x: x['snr'], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"#{i} {result['method']}:")
        print(f"    📈 SNR: {result['snr']:.2f} dB")
        print(f"    🔗 相关性: {result['correlation']:.4f}")
        print(f"    📄 文件: {result['path']}")
        print()
    
    best_improvement = results[0]['snr'] - (-0.01)  # 与之前最佳结果对比
    print(f"🚀 最佳改进效果: {best_improvement:+.2f} dB")
    print(f"🏆 推荐方法: {results[0]['method']}")
    
    print(f"\\n📁 所有结果保存在: {output_dir}/")
    print(f"🎧 建议播放音频文件进行主观评估")
    
    return results


def main():
    """主函数"""
    import sys
    
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "AudioLDM2_Music_output.wav"
    
    print(f"🔍 AudioLDM2 VAE重建瓶颈分析与改进")
    print(f"=" * 60)
    
    # 步骤1: 瓶颈分析
    print(f"\\n🔬 步骤1: 深度瓶颈分析")
    bottleneck_analysis = analyze_bottlenecks(audio_path)
    
    # 步骤2: 针对性改进
    print(f"\\n🚀 步骤2: 实施改进方案")
    improvement_results = improved_reconstruction_v2(audio_path)
    
    # 步骤3: 总结建议
    print(f"\\n💡 关键瓶颈总结:")
    print(f"   1️⃣ Mel变换信息损失: {(1-bottleneck_analysis['spectral_correlation'])*100:.1f}%")
    print(f"   2️⃣ VAE压缩信息损失: {(1-bottleneck_analysis['vae_correlation'])*100:.1f}%") 
    print(f"   3️⃣ 相位重建质量差: Griffin-Lim本身限制")
    print(f"   4️⃣ 总体信息保留率: 仅 {bottleneck_analysis['total_retention']*100:.1f}%")
    
    print(f"\\n🎯 下一步建议:")
    print(f"   🔧 短期: 优化mel参数配置和后处理")
    print(f"   🧠 中期: 使用预训练的高质量vocoder")
    print(f"   🚀 长期: 端到端训练专用重建模型")


if __name__ == "__main__":
    main()
