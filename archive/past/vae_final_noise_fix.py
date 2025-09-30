#!/usr/bin/env python3
"""
AudioLDM2 VAE 重建测试 - 最终噪音修复版本
专门解决HiFiGAN的"咔哒咔哒"噪音问题

主要修复:
1. 强制使用float32避免溢出
2. 简化但有效的归一化策略
3. 针对性的后处理降噪
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

# 尝试导入 soundfile 以获得更好的兼容性
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("⚠️ soundfile 不可用，将使用 torchaudio 保存（可能存在兼容性问题）")


def save_audio_compatible(audio_data, filepath, sample_rate=16000):
    """
    保存音频文件，优先使用 soundfile 以获得最大兼容性
    
    Args:
        audio_data: 音频数据 (numpy array)
        filepath: 输出文件路径
        sample_rate: 采样率
    """
    # 确保音频数据是正确的格式
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    
    # 确保是 1D 数组
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    
    # 清理数据
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 归一化到 [-1, 1] 范围
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    success = False
    
    if SOUNDFILE_AVAILABLE:
        try:
            # 使用 soundfile 保存为 PCM_16 格式（最高兼容性）
            sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')
            print(f"   ✅ 使用 soundfile (PCM_16) 保存: {filepath}")
            success = True
        except Exception as e:
            print(f"   ⚠️ soundfile 保存失败: {e}")
    
    if not success:
        try:
            # 回退到 torchaudio
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
            torchaudio.save(filepath, audio_tensor, sample_rate)
            print(f"   ✅ 使用 torchaudio 保存: {filepath}")
            success = True
        except Exception as e:
            print(f"   ❌ torchaudio 保存也失败: {e}")
    
    return success


def simple_safe_normalize(mel_tensor, target_range=(-10, 10)):
    """简单而安全的归一化策略"""
    if isinstance(mel_tensor, torch.Tensor):
        # 强制转换为float32
        mel_float32 = mel_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        mel_float32 = np.array(mel_tensor, dtype=np.float32)
    
    # 移除任何无效值
    mel_float32 = np.nan_to_num(mel_float32, nan=0.0, posinf=0.0, neginf=-50.0)
    
    # 简单的min-max归一化到目标范围
    current_min = np.min(mel_float32)
    current_max = np.max(mel_float32)
    
    print(f"   简单归一化: min={current_min:.3f}, max={current_max:.3f}")
    
    if current_max > current_min:
        # 归一化到[0, 1]然后映射到目标范围
        normalized = (mel_float32 - current_min) / (current_max - current_min)
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    else:
        # 如果所有值都相同，设置为目标范围的中点
        normalized = np.full_like(mel_float32, (target_range[0] + target_range[1]) / 2)
    
    final_min = np.min(normalized)
    final_max = np.max(normalized)
    print(f"   归一化结果: min={final_min:.3f}, max={final_max:.3f}")
    
    if isinstance(mel_tensor, torch.Tensor):
        return torch.from_numpy(normalized).to(mel_tensor.device).to(torch.float32)
    else:
        return normalized


def advanced_audio_denoising(audio, sr=16000):
    """高级音频降噪处理"""
    if len(audio) == 0:
        return audio
    
    # 1. 移除直流偏置
    audio = audio - np.mean(audio)
    
    # 2. 检测并移除异常值（可能的点击声）
    # 使用3-sigma规则检测异常值
    std_audio = np.std(audio)
    mean_audio = np.mean(audio)
    threshold = 3 * std_audio
    
    # 找到异常值并用中值替换
    outliers = np.abs(audio - mean_audio) > threshold
    if np.any(outliers):
        print(f"   检测到 {np.sum(outliers)} 个异常值点击，进行修复...")
        # 使用滑动窗口中值滤波器修复异常值
        from scipy.ndimage import median_filter
        audio_median = median_filter(audio, size=5)
        audio[outliers] = audio_median[outliers]
    
    # 3. 应用渐变边界
    fade_samples = min(1024, len(audio) // 8)
    if len(audio) > 2 * fade_samples:
        # 淡入
        fade_in = np.linspace(0, 1, fade_samples) ** 0.5  # 平方根淡入更平滑
        audio[:fade_samples] *= fade_in
        
        # 淡出
        fade_out = np.linspace(1, 0, fade_samples) ** 0.5
        audio[-fade_samples:] *= fade_out
    
    # 4. 轻微平滑（减少高频噪音）
    audio = gaussian_filter1d(audio, sigma=1.0)
    
    # 5. 高质量带通滤波
    try:
        nyquist = sr / 2
        # 更保守的滤波范围
        low_freq = 60   # 60Hz高通
        high_freq = 7000  # 7kHz低通
        
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # 使用椭圆滤波器（更陡峭的衰减）
        b, a = scipy.signal.ellip(5, 1, 60, [low, high], btype='band')
        audio = scipy.signal.filtfilt(b, a, audio)
        
    except Exception as e:
        print(f"   滤波器警告: {e}")
    
    # 6. 动态范围压缩（减少峰值产生的点击）
    threshold = 0.7
    ratio = 0.3
    audio = np.where(np.abs(audio) > threshold,
                     np.sign(audio) * (threshold + (np.abs(audio) - threshold) * ratio),
                     audio)
    
    return audio


def load_and_test_vae_final(audio_path, max_length=10):
    """最终的VAE测试版本，专注于噪音修复"""
    print(f"\n🎵 AudioLDM2 VAE最终噪音修复测试")
    print(f"音频文件: {audio_path}")
    print(f"最大长度: {max_length} 秒")
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # 强制使用float32
    print(f"🖥️ 使用设备: {device}, 数据类型: {dtype}")
    
    # 加载模型
    print("📦 加载 AudioLDM2 模型...")
    try:
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            torch_dtype=dtype
        ).to(device)
        
        vae = pipeline.vae
        vocoder = pipeline.vocoder
        print(f"✅ 模型加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 加载音频
    print(f"🎶 加载音频...")
    try:
        sample_rate = 16000
        audio, orig_sr = torchaudio.load(audio_path)
        
        if orig_sr != sample_rate:
            audio = torchaudio.functional.resample(audio, orig_sr, sample_rate)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        audio = audio.squeeze().numpy().astype(np.float32)
        
        # 截取
        max_samples = int(max_length * sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # 预处理
        audio = audio - np.mean(audio)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        print(f"   音频长度: {len(audio)} 样本")
        
    except Exception as e:
        print(f"❌ 音频加载失败: {e}")
        return None
    
    # 生成mel-spectrogram
    print("🔄 生成 mel-spectrogram...")
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=160,
        n_mels=64,
        fmin=0,
        fmax=8000
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
    mel_spec_db = np.clip(mel_spec_db, -80, 0).astype(np.float32)
    
    print(f"   Mel形状: {mel_spec_db.shape}")
    
    # VAE处理
    with torch.no_grad():
        # 准备输入
        mel_tensor = torch.from_numpy(mel_spec_db).to(device).to(dtype)
        mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
        
        # 填充
        time_dim = mel_input.shape[-1]
        if time_dim % 8 != 0:
            pad_width = 8 - (time_dim % 8)
            mel_input = torch.nn.functional.pad(mel_input, (0, pad_width), mode='constant', value=-80)
        
        # 归一化
        mel_min, mel_max = -80.0, 0.0
        mel_normalized = 2.0 * (mel_input - mel_min) / (mel_max - mel_min) - 1.0
        mel_normalized = torch.clamp(mel_normalized, -1, 1)
        
        # VAE编码
        print("🔧 VAE 编码...")
        start_time = time.time()
        latents = vae.encode(mel_normalized).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        encode_time = time.time() - start_time
        
        # VAE解码
        print("🔧 VAE 解码...")
        decode_start = time.time()
        latents_scaled = latents / vae.config.scaling_factor
        reconstructed_mel = vae.decode(latents_scaled).sample
        decode_time = time.time() - decode_start
        
        print(f"   编码: {encode_time:.2f}s, 解码: {decode_time:.2f}s")
    
    # 音频重建 - 重点修复
    print("🔊 音频重建 (噪音修复版本)...")
    
    try:
        with torch.no_grad():
            # 反归一化
            mel_denorm = (reconstructed_mel + 1) / 2 * (mel_max - mel_min) + mel_min
            mel_denorm = torch.clamp(mel_denorm, mel_min, mel_max)
            
            # 转换为float32确保兼容性
            mel_denorm = mel_denorm.to(torch.float32)
            
            # 检查有效性
            if not torch.isfinite(mel_denorm).all():
                print(f"   ⚠️ 修复无效值...")
                mel_denorm = torch.nan_to_num(mel_denorm, nan=-40.0, posinf=-20.0, neginf=-80.0)
            
            # 准备vocoder输入
            vocoder_input = mel_denorm.squeeze(0).squeeze(0)  # [64, time]
            vocoder_input = vocoder_input.transpose(-2, -1)   # [time, 64]
            vocoder_input = vocoder_input.unsqueeze(0)        # [1, time, 64]
            
            print(f"   Vocoder输入形状: {vocoder_input.shape}")
            print(f"   Vocoder输入范围: [{vocoder_input.min().item():.3f}, {vocoder_input.max().item():.3f}]")
            
            # 简单归一化
            vocoder_input_norm = simple_safe_normalize(vocoder_input, target_range=(-8, 2))
            
            # 确保vocoder也使用float32
            vocoder.to(dtype)
            
            # 使用vocoder
            reconstructed_audio_tensor = vocoder(vocoder_input_norm)
            reconstructed_audio = reconstructed_audio_tensor.squeeze().cpu().numpy().astype(np.float32)
            
            print(f"   ✅ HiFiGAN成功: {len(reconstructed_audio)}样本")
            
            # 高级降噪处理
            print("🛠️ 应用高级降噪处理...")
            reconstructed_audio = advanced_audio_denoising(reconstructed_audio, sr=sample_rate)
            
            vocoder_method = "AudioLDM2_HiFiGAN_FinalFix"
            
    except Exception as e:
        print(f"   ❌ HiFiGAN失败: {e}")
        print("   📊 使用Griffin-Lim...")
        
        # Griffin-Lim降级
        try:
            recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
            recon_mel_denorm = (recon_mel_np + 1) / 2 * (mel_max - mel_min) + mel_min
            recon_mel_denorm = np.clip(recon_mel_denorm, -80.0, 0.0)
            
            recon_mel_power = librosa.db_to_power(recon_mel_denorm)
            recon_mel_power = np.clip(recon_mel_power, 1e-10, None)
            
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                recon_mel_power,
                sr=sample_rate,
                hop_length=160,
                n_fft=1024,
                fmin=0,
                fmax=8000
            )
            
            reconstructed_audio = advanced_audio_denoising(reconstructed_audio, sr=sample_rate)
            vocoder_method = "Griffin_Lim_Advanced"
            
        except Exception as griffin_e:
            print(f"   ❌ Griffin-Lim失败: {griffin_e}")
            reconstructed_audio = np.random.randn(len(audio)) * 0.001
            vocoder_method = "Fallback"
    
    # 长度对齐
    if len(reconstructed_audio) > len(audio):
        reconstructed_audio = reconstructed_audio[:len(audio)]
    elif len(reconstructed_audio) < len(audio):
        reconstructed_audio = np.pad(reconstructed_audio, (0, len(audio) - len(reconstructed_audio)))
      # 保存结果（使用高兼容性保存方法）
    print("💾 保存结果...")
    output_dir = "vae_final_noise_fix"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    reconstructed_path = os.path.join(output_dir, f"{input_name}_final_noisefixed_{timestamp}.wav")
    
    # 归一化
    audio_norm = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    recon_norm = reconstructed_audio / np.max(np.abs(reconstructed_audio)) if np.max(np.abs(reconstructed_audio)) > 0 else reconstructed_audio
    
    # 使用兼容性保存函数
    save_audio_compatible(audio_norm, original_path, sample_rate)
    save_audio_compatible(recon_norm, reconstructed_path, sample_rate)
    
    # 计算指标
    min_len = min(len(audio), len(reconstructed_audio))
    orig_segment = audio[:min_len]
    recon_segment = reconstructed_audio[:min_len]
    
    mse = np.mean((orig_segment - recon_segment) ** 2)
    correlation = np.corrcoef(orig_segment, recon_segment)[0, 1] if len(orig_segment) > 1 else 0
    
    signal_power = np.mean(orig_segment ** 2)
    noise_power = np.mean((orig_segment - recon_segment) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    original_size = mel_normalized.numel()
    compressed_size = latents.numel()
    compression_ratio = original_size / compressed_size
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"VAE 重建测试 - 最终噪音修复版本")
    print(f"{'='*60}")
    print(f"原始音频: {original_path}")
    print(f"重建音频: {reconstructed_path}")
    print(f"编码时间: {encode_time:.2f}秒")
    print(f"解码时间: {decode_time:.2f}秒")
    print(f"总时间: {encode_time + decode_time:.2f}秒")
    print(f"MSE: {mse:.6f}")
    print(f"SNR: {snr:.2f} dB")
    print(f"相关系数: {correlation:.4f}")
    print(f"压缩比: {compression_ratio:.1f}:1")
    print(f"重建方法: {vocoder_method}")
    print(f"✅ 修复内容: float32兼容性、异常值检测、高级降噪、动态范围压缩")
    
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
        print("使用方法: python vae_final_noise_fix.py <音频文件路径> [最大长度秒数]")
        
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
    
    max_length = 10
    if len(sys.argv) >= 3:
        try:
            max_length = float(sys.argv[2])
        except ValueError:
            print(f"无效长度参数，使用默认值 {max_length} 秒")
    
    print(f"🚀 开始最终噪音修复测试")
    
    try:
        result = load_and_test_vae_final(audio_path, max_length=max_length)
        if result:
            print(f"\n✅ 最终测试完成！")
            print(f"请播放重建音频检查噪音修复效果。")
            print(f"如果仍有噪音，可能需要调整vocoder参数或使用更高级的后处理。")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
