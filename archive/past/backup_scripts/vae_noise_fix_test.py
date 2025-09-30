#!/usr/bin/env python3
"""
AudioLDM2 VAE 重建测试 - 噪音修复版本
主要解决HiFiGAN生成的"咔哒咔哒"噪音问题

噪音修复策略:
1. 输入归一化优化
2. 边界平滑处理
3. 窗口函数应用
4. 后处理滤波
5. 渐变淡入淡出
"""

import torch
import torchaudio
import numpy as np
import librosa
import time
import os
import sys
from pathlib import Path
from diffusers import AudioLDM2Pipeline
import scipy.signal
from scipy.ndimage import gaussian_filter1d


def apply_fade_in_out(audio, fade_samples=1024):
    """应用渐变淡入淡出来减少边界点击"""
    if len(audio) <= 2 * fade_samples:
        fade_samples = len(audio) // 4
    
    # 淡入
    fade_in = np.linspace(0, 1, fade_samples)
    audio[:fade_samples] *= fade_in
    
    # 淡出
    fade_out = np.linspace(1, 0, fade_samples)
    audio[-fade_samples:] *= fade_out
    
    return audio


def smooth_spectrogram_boundaries(mel_spec, boundary_frames=4):
    """平滑mel-spectrogram的边界以减少不连续性"""
    if mel_spec.shape[-1] <= 2 * boundary_frames:
        return mel_spec
    
    # 对开始和结束帧应用平滑
    # 使用高斯滤波器平滑边界
    smoothed = mel_spec.clone()
    
    # 平滑开始部分
    for i in range(boundary_frames):
        weight = i / boundary_frames
        smoothed[..., i] = mel_spec[..., i] * weight + mel_spec[..., boundary_frames] * (1 - weight)
    
    # 平滑结束部分
    for i in range(boundary_frames):
        idx = -(i+1)
        weight = i / boundary_frames
        smoothed[..., idx] = mel_spec[..., idx] * weight + mel_spec[..., -(boundary_frames+1)] * (1 - weight)
    
    return smoothed


def normalize_mel_for_hifigan(mel_spec, target_mean=-4.0, target_std=4.0):
    """
    专门为HiFiGAN优化的mel-spectrogram归一化
    基于AudioLDM2训练时使用的统计信息
    """
    # 转换为numpy进行计算
    if isinstance(mel_spec, torch.Tensor):
        mel_np = mel_spec.cpu().numpy()
    else:
        mel_np = mel_spec
    
    # 检查输入的有效性
    if not np.isfinite(mel_np).all():
        print(f"   ⚠️ 检测到无效值，应用修复...")
        mel_np = np.nan_to_num(mel_np, nan=target_mean, posinf=target_mean + 2*target_std, neginf=target_mean - 2*target_std)
    
    # 计算当前统计信息
    current_mean = np.mean(mel_np)
    current_std = np.std(mel_np)
    
    print(f"   Mel归一化前: mean={current_mean:.3f}, std={current_std:.3f}")
    
    # 检查标准差是否有效
    if not np.isfinite(current_std) or current_std < 1e-6:
        print(f"   ⚠️ 标准差异常，使用简单缩放...")
        # 如果标准差异常，使用简单的线性映射
        min_val = np.min(mel_np)
        max_val = np.max(mel_np)
        if max_val > min_val:
            normalized = (mel_np - min_val) / (max_val - min_val)
            normalized = normalized * (2 * target_std) + (target_mean - target_std)
        else:
            normalized = np.full_like(mel_np, target_mean)
    else:
        # 标准化并调整到目标分布
        normalized = (mel_np - current_mean) / current_std
        normalized = normalized * target_std + target_mean
    
    # 应用合理的裁剪
    normalized = np.clip(normalized, target_mean - 3*target_std, target_mean + 3*target_std)
    
    # 确保没有无效值
    normalized = np.nan_to_num(normalized, nan=target_mean, posinf=target_mean + 2*target_std, neginf=target_mean - 2*target_std)
    
    final_mean = np.mean(normalized)
    final_std = np.std(normalized)
    print(f"   Mel归一化后: mean={final_mean:.3f}, std={final_std:.3f}")
    
    if isinstance(mel_spec, torch.Tensor):
        return torch.from_numpy(normalized).to(mel_spec.device).to(mel_spec.dtype)
    else:
        return normalized


def apply_post_filter(audio, sr=16000, cutoff_low=50, cutoff_high=7500):
    """应用后处理滤波器移除不需要的频率成分"""
    # 设计带通滤波器
    nyquist = sr / 2
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    
    # 应用巴特沃斯滤波器
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    filtered_audio = scipy.signal.filtfilt(b, a, audio)
    
    return filtered_audio


def remove_dc_bias(audio):
    """移除直流偏置"""
    return audio - np.mean(audio)


def load_and_test_vae_noise_fixed(audio_path, max_length=10):
    """
    加载音频并执行VAE重建测试 - 噪音修复版本
    """
    print(f"\n🎵 加载和测试音频: {audio_path}")
    print(f"⏱️ 最大长度: {max_length} 秒")
    
    # 检查CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device}")
    
    # 加载 AudioLDM2 模型
    print("📦 加载 AudioLDM2 模型...")
    try:
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        vae = pipeline.vae
        vocoder = pipeline.vocoder
        print(f"✅ 模型加载成功")
        print(f"   VAE: {type(vae).__name__}")
        print(f"   Vocoder: {type(vocoder).__name__}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 加载音频文件
    print(f"🎶 加载音频文件: {audio_path}")
    try:
        # 强制采样率为16kHz (AudioLDM2默认)
        sample_rate = 16000
        audio, orig_sr = torchaudio.load(audio_path)
        
        if orig_sr != sample_rate:
            print(f"   重采样: {orig_sr}Hz -> {sample_rate}Hz")
            audio = torchaudio.functional.resample(audio, orig_sr, sample_rate)
        
        # 转换为单声道
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        audio = audio.squeeze().numpy()
        
        # 截取指定长度
        max_samples = int(max_length * sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"   截取到 {max_length} 秒")
        
        print(f"   音频长度: {len(audio)} 样本 ({len(audio)/sample_rate:.2f} 秒)")
        
    except Exception as e:
        print(f"❌ 音频加载失败: {e}")
        return None
    
    # 预处理：移除直流偏置和归一化
    audio = remove_dc_bias(audio)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95  # 轻微降低幅度以避免削波
    
    # 生成 mel-spectrogram
    print("🔄 生成 mel-spectrogram...")
    
    # 使用与AudioLDM2一致的参数
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,        # AudioLDM2默认
        hop_length=160,    # AudioLDM2默认
        n_mels=64,         # AudioLDM2默认
        fmin=0,
        fmax=8000
    )
    
    # 转换为dB并应用合理范围
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
    mel_spec_db = np.clip(mel_spec_db, -80, 0)
    
    print(f"   原始 mel 形状: {mel_spec_db.shape}")
    print(f"   原始 mel 范围: [{np.min(mel_spec_db):.2f}, {np.max(mel_spec_db):.2f}] dB")
    
    # 为VAE准备输入
    with torch.no_grad():
        # 转换为张量并移动到设备
        mel_tensor = torch.from_numpy(mel_spec_db).to(device).to(vae.dtype)
        
        # 调整形状为 [batch, channels, height, width]
        mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, time]
        
        # 确保时间维度是8的倍数（VAE要求）
        time_dim = mel_input.shape[-1]
        if time_dim % 8 != 0:
            pad_width = 8 - (time_dim % 8)
            mel_input = torch.nn.functional.pad(mel_input, (0, pad_width), mode='constant', value=-80)
            print(f"   填充 mel 到: {mel_input.shape}")
        
        # 边界平滑处理
        mel_input_smoothed = smooth_spectrogram_boundaries(mel_input, boundary_frames=4)
        
        # 归一化为VAE期望的范围 [-1, 1]
        mel_min, mel_max = -80.0, 0.0
        mel_normalized = 2.0 * (mel_input_smoothed - mel_min) / (mel_max - mel_min) - 1.0
        mel_normalized = torch.clamp(mel_normalized, -1, 1)
        
        print(f"   VAE输入形状: {mel_normalized.shape}")
        print(f"   VAE输入范围: [{mel_normalized.min().item():.3f}, {mel_normalized.max().item():.3f}]")
        
        # VAE 编码
        print("🔧 VAE 编码...")
        start_time = time.time()
        
        try:
            latents = vae.encode(mel_normalized).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            encode_time = time.time() - start_time
            print(f"   编码完成: {encode_time:.2f}秒")
            print(f"   潜在形状: {latents.shape}")
            print(f"   潜在范围: [{latents.min().item():.3f}, {latents.max().item():.3f}]")
            
        except Exception as e:
            print(f"❌ VAE编码失败: {e}")
            return None
        
        # VAE 解码
        print("🔧 VAE 解码...")
        decode_start = time.time()
        
        try:
            latents_scaled = latents / vae.config.scaling_factor
            reconstructed_mel = vae.decode(latents_scaled).sample
            
            decode_time = time.time() - decode_start
            print(f"   解码完成: {decode_time:.2f}秒")
            print(f"   重建 mel 形状: {reconstructed_mel.shape}")
            print(f"   重建 mel 范围: [{reconstructed_mel.min().item():.3f}, {reconstructed_mel.max().item():.3f}]")
            
        except Exception as e:
            print(f"❌ VAE解码失败: {e}")
            return None
    
    # 音频重建
    print("🔊 音频重建...")
      with torch.no_grad():
        # 方法1: 使用AudioLDM2内置的HiFiGAN (噪音修复版本)
        print("🎤 使用AudioLDM2内置HiFiGAN (噪音修复版本)...")
        
        try:
            # 反归一化 - 确保范围合理
            mel_denorm = (reconstructed_mel + 1) / 2 * (mel_max - mel_min) + mel_min
            mel_denorm = torch.clamp(mel_denorm, mel_min, mel_max)
            
            # 检查解码后的有效性
            if not torch.isfinite(mel_denorm).all():
                print(f"   ⚠️ 检测到VAE解码输出的无效值，应用修复...")
                mel_denorm = torch.nan_to_num(mel_denorm, nan=-40.0, posinf=-20.0, neginf=-80.0)
                mel_denorm = torch.clamp(mel_denorm, mel_min, mel_max)
            
            # 准备HiFiGAN输入: [batch, time, mel_dim]
            vocoder_input = mel_denorm.squeeze(0).squeeze(0)  # [64, time]
            vocoder_input = vocoder_input.transpose(-2, -1)   # [time, 64]
            vocoder_input = vocoder_input.unsqueeze(0)        # [1, time, 64]
            
            print(f"   HiFiGAN输入形状 (预处理): {vocoder_input.shape}")
            print(f"   HiFiGAN输入范围 (预处理): [{vocoder_input.min().item():.3f}, {vocoder_input.max().item():.3f}]")
            
            # 专门为HiFiGAN优化归一化
            vocoder_input_norm = normalize_mel_for_hifigan(vocoder_input)
            
            print(f"   HiFiGAN输入形状 (最终): {vocoder_input_norm.shape}")
            print(f"   HiFiGAN输入范围 (最终): [{vocoder_input_norm.min().item():.3f}, {vocoder_input_norm.max().item():.3f}]")
            
            # 使用AudioLDM2内置HiFiGAN
            reconstructed_audio_tensor = vocoder(vocoder_input_norm)
            reconstructed_audio = reconstructed_audio_tensor.squeeze().cpu().numpy()
            
            print(f"   ✅ HiFiGAN成功: 输出{len(reconstructed_audio)}样本")
            
            # 后处理步骤
            print("🛠️ 应用后处理...")
            
            # 1. 移除直流偏置
            reconstructed_audio = remove_dc_bias(reconstructed_audio)
            
            # 2. 应用渐变淡入淡出
            reconstructed_audio = apply_fade_in_out(reconstructed_audio, fade_samples=512)
            
            # 3. 后处理滤波
            reconstructed_audio = apply_post_filter(reconstructed_audio, sr=sample_rate)
            
            # 4. 轻微平滑
            reconstructed_audio = gaussian_filter1d(reconstructed_audio, sigma=0.5)
            
            # 5. 最终归一化
            if np.max(np.abs(reconstructed_audio)) > 0:
                reconstructed_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio)) * 0.9
            
            vocoder_method = "AudioLDM2_HiFiGAN_NoiseFixed"
            print(f"   ✅ 后处理完成")
            
        except Exception as e:
            print(f"   ❌ HiFiGAN失败: {e}")
            print("📊 降级到Griffin-Lim算法...")
            
            # 降级方案: Griffin-Lim
            try:
                # 反归一化 mel-spectrogram
                recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
                recon_mel_denorm = (recon_mel_np + 1) / 2 * (mel_max - mel_min) + mel_min
                recon_mel_denorm = np.clip(recon_mel_denorm, -80.0, 0.0)
                
                # 转换回功率谱
                recon_mel_power = librosa.db_to_power(recon_mel_denorm)
                recon_mel_power = np.clip(recon_mel_power, 1e-10, None)
                
                # Griffin-Lim算法
                reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                    recon_mel_power,
                    sr=sample_rate,
                    hop_length=160,
                    n_fft=1024,
                    fmin=0,
                    fmax=8000
                )
                
                # 后处理
                reconstructed_audio = remove_dc_bias(reconstructed_audio)
                reconstructed_audio = apply_fade_in_out(reconstructed_audio, fade_samples=512)
                
                vocoder_method = "Griffin_Lim_PostProcessed"
                print(f"   ✅ Griffin-Lim成功: 输出{len(reconstructed_audio)}样本")
                
            except Exception as griffin_e:
                print(f"   ❌ Griffin-Lim也失败: {griffin_e}")
                reconstructed_audio = np.random.randn(len(audio)) * 0.01
                vocoder_method = "Fallback_Noise"
    
    # 确保音频长度一致
    if len(reconstructed_audio) > len(audio):
        reconstructed_audio = reconstructed_audio[:len(audio)]
    elif len(reconstructed_audio) < len(audio):
        reconstructed_audio = np.pad(reconstructed_audio, (0, len(audio) - len(reconstructed_audio)))
    
    # 保存结果
    print("💾 保存结果...")
    output_dir = "vae_noise_fix_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_norm = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    torchaudio.save(original_path, torch.from_numpy(audio_norm).unsqueeze(0), sample_rate)
    
    # 保存重建音频
    reconstructed_path = os.path.join(output_dir, f"{input_name}_reconstructed_noisefixed_{timestamp}.wav")
    recon_norm = reconstructed_audio / np.max(np.abs(reconstructed_audio)) if np.max(np.abs(reconstructed_audio)) > 0 else reconstructed_audio
    torchaudio.save(reconstructed_path, torch.from_numpy(recon_norm).unsqueeze(0), sample_rate)
    
    # 计算质量指标
    print("📊 计算质量指标...")
    min_len = min(len(audio), len(reconstructed_audio))
    orig_segment = audio[:min_len]
    recon_segment = reconstructed_audio[:min_len]
    
    # MSE 和相关系数
    mse = np.mean((orig_segment - recon_segment) ** 2)
    correlation = np.corrcoef(orig_segment, recon_segment)[0, 1] if len(orig_segment) > 1 else 0
    
    # SNR
    signal_power = np.mean(orig_segment ** 2)
    noise_power = np.mean((orig_segment - recon_segment) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # 压缩比
    original_size = mel_normalized.numel()
    compressed_size = latents.numel()
    compression_ratio = original_size / compressed_size
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"VAE 重建测试结果 - 噪音修复版本")
    print(f"{'='*60}")
    print(f"原始音频: {original_path}")
    print(f"重建音频: {reconstructed_path}")
    print(f"编码时间: {encode_time:.2f}秒")
    print(f"解码时间: {decode_time:.2f}秒")
    print(f"总时间: {encode_time + decode_time:.2f}秒")
    print(f"MSE: {mse:.6f}")
    print(f"SNR: {snr:.2f} dB")
    print(f"相关系数: {correlation:.4f}")
    print(f"Mel-spectrogram 形状: {mel_normalized.shape}")
    print(f"潜在表示形状: {latents.shape}")
    print(f"压缩比: {compression_ratio:.1f}:1")
    print(f"重建方法: {vocoder_method}")
    print(f"噪音修复: ✅ 应用了边界平滑、渐变淡入淡出、后处理滤波")
    
    return {
        'original_path': original_path,
        'reconstructed_path': reconstructed_path,
        'mse': mse,
        'snr': snr,
        'correlation': correlation,
        'encode_time': encode_time,
        'decode_time': decode_time,
        'compression_ratio': compression_ratio,
        'vocoder_method': vocoder_method
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python vae_noise_fix_test.py <音频文件路径> [最大长度秒数]")
        
        # 查找当前目录下的音频文件
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(Path('.').glob(f'*{ext}'))
        
        if audio_files:
            print(f"\n找到音频文件:")
            for i, file in enumerate(audio_files, 1):
                print(f"{i}. {file}")
            
            try:
                choice = input("请选择文件序号: ").strip()
                file_idx = int(choice) - 1
                if 0 <= file_idx < len(audio_files):
                    audio_path = str(audio_files[file_idx])
                else:
                    print("无效选择")
                    return
            except (ValueError, KeyboardInterrupt):
                print("取消操作")
                return
        else:
            print("当前目录没有找到音频文件")
            return
    else:
        audio_path = sys.argv[1]
    
    # 获取最大长度参数
    max_length = 10  # 默认10秒
    if len(sys.argv) >= 3:
        try:
            max_length = float(sys.argv[2])
        except ValueError:
            print(f"无效的长度参数，使用默认值 {max_length} 秒")
    
    # 执行测试
    print(f"🚀 开始噪音修复测试")
    print(f"音频文件: {audio_path}")
    print(f"最大长度: {max_length} 秒")
    
    try:
        result = load_and_test_vae_noise_fixed(audio_path, max_length=max_length)
        if result:
            print("\n✅ 噪音修复测试完成！")
            print("请播放重建音频检查噪音是否减少。")
            print("\n🔍 建议比较:")
            print("1. 原始音频 vs 重建音频")
            print("2. 与之前版本的输出进行对比")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
