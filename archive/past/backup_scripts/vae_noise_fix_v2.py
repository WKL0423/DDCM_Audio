#!/usr/bin/env python3
"""
AudioLDM2 VAE 重建测试 - 噪音修复版本 v2
主要解决HiFiGAN生成的"咔哒咔哒"噪音问题

重点修复:
1. VAE解码结果的数值稳定性
2. HiFiGAN输入归一化优化
3. 后处理降噪策略
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


def safe_normalize_mel(mel_tensor, target_mean=-5.0, target_std=5.0):
    """安全的mel-spectrogram归一化，避免数值问题"""
    if isinstance(mel_tensor, torch.Tensor):
        mel_np = mel_tensor.detach().cpu().numpy().astype(np.float64)  # 使用float64避免溢出
    else:
        mel_np = mel_tensor.astype(np.float64)
    
    # 检查并修复无效值
    if not np.isfinite(mel_np).all():
        print(f"   ⚠️ 检测到无效值，进行修复...")
        mel_np = np.nan_to_num(mel_np, nan=target_mean, posinf=0.0, neginf=-50.0)
    
    # 计算统计信息
    mean_val = np.mean(mel_np)
    std_val = np.std(mel_np)
    
    print(f"   归一化前: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # 安全的标准化
    if np.isfinite(std_val) and std_val > 1e-6:
        # 标准标准化
        normalized = (mel_np - mean_val) / std_val
        normalized = normalized * target_std + target_mean
    else:
        # 简单的线性映射
        min_val = np.min(mel_np)
        max_val = np.max(mel_np)
        if max_val > min_val:
            normalized = (mel_np - min_val) / (max_val - min_val)
            normalized = normalized * (2 * target_std) + (target_mean - target_std)
        else:
            normalized = np.full_like(mel_np, target_mean)
    
    # 裁剪到合理范围
    normalized = np.clip(normalized, target_mean - 3*target_std, target_mean + 3*target_std)
    
    # 最终检查
    normalized = np.nan_to_num(normalized, nan=target_mean, posinf=target_mean + target_std, neginf=target_mean - target_std)
    
    final_mean = np.mean(normalized)
    final_std = np.std(normalized)
    print(f"   归一化后: mean={final_mean:.3f}, std={final_std:.3f}")
    
    # 转换回适当的精度
    normalized = normalized.astype(np.float32)
    
    if isinstance(mel_tensor, torch.Tensor):
        return torch.from_numpy(normalized).to(mel_tensor.device).to(torch.float32)  # 强制使用float32
    else:
        return normalized


def apply_audio_postprocessing(audio, sr=16000, fade_samples=1024):
    """应用音频后处理以减少噪音"""
    if len(audio) == 0:
        return audio
    
    # 1. 移除直流偏置
    audio = audio - np.mean(audio)
    
    # 2. 渐变淡入淡出（减少点击）
    if len(audio) > 2 * fade_samples:
        # 淡入
        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in
        
        # 淡出
        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out
    
    # 3. 轻微平滑滤波（减少高频噪音）
    audio = gaussian_filter1d(audio, sigma=0.8)
    
    # 4. 带通滤波（移除极低和极高频率）
    try:
        nyquist = sr / 2
        low = 80 / nyquist    # 80Hz高通
        high = 7500 / nyquist # 7.5kHz低通
        b, a = scipy.signal.butter(3, [low, high], btype='band')
        audio = scipy.signal.filtfilt(b, a, audio)
    except:
        print(f"   ⚠️ 滤波器应用失败，跳过...")
    
    # 5. 轻微压缩（减少峰值）
    threshold = 0.8
    audio = np.where(np.abs(audio) > threshold, 
                     np.sign(audio) * (threshold + (np.abs(audio) - threshold) * 0.5),
                     audio)
    
    return audio


def load_and_test_vae_v2(audio_path, max_length=10):
    """VAE重建测试 - 稳定版本"""
    print(f"\n🎵 开始VAE重建测试 v2")
    print(f"音频文件: {audio_path}")
    print(f"最大长度: {max_length} 秒")
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device}")
    
    # 加载模型
    print("📦 加载 AudioLDM2 模型...")
    try:
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        vae = pipeline.vae
        vocoder = pipeline.vocoder
        print(f"✅ 模型加载成功: VAE={type(vae).__name__}, Vocoder={type(vocoder).__name__}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 加载音频
    print(f"🎶 加载音频文件...")
    try:
        sample_rate = 16000
        audio, orig_sr = torchaudio.load(audio_path)
        
        if orig_sr != sample_rate:
            audio = torchaudio.functional.resample(audio, orig_sr, sample_rate)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        audio = audio.squeeze().numpy()
        
        # 截取
        max_samples = int(max_length * sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # 预处理
        audio = audio - np.mean(audio)  # 移除直流偏置
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95  # 轻微降低幅度
        
        print(f"   音频长度: {len(audio)} 样本 ({len(audio)/sample_rate:.2f} 秒)")
        
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
    
    # 转换为dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
    mel_spec_db = np.clip(mel_spec_db, -80, 0)
    
    print(f"   Mel形状: {mel_spec_db.shape}")
    print(f"   Mel范围: [{np.min(mel_spec_db):.2f}, {np.max(mel_spec_db):.2f}] dB")
    
    # VAE处理
    with torch.no_grad():
        # 准备VAE输入
        mel_tensor = torch.from_numpy(mel_spec_db).to(device).to(vae.dtype)
        mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, time]
        
        # 填充到8的倍数
        time_dim = mel_input.shape[-1]
        if time_dim % 8 != 0:
            pad_width = 8 - (time_dim % 8)
            mel_input = torch.nn.functional.pad(mel_input, (0, pad_width), mode='constant', value=-80)
        
        # 归一化到[-1, 1]
        mel_min, mel_max = -80.0, 0.0
        mel_normalized = 2.0 * (mel_input - mel_min) / (mel_max - mel_min) - 1.0
        mel_normalized = torch.clamp(mel_normalized, -1, 1)
        
        print(f"   VAE输入形状: {mel_normalized.shape}")
        
        # VAE编码
        print("🔧 VAE 编码...")
        start_time = time.time()
        
        latents = vae.encode(mel_normalized).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        encode_time = time.time() - start_time
        print(f"   编码完成: {encode_time:.2f}秒, 潜在形状: {latents.shape}")
        
        # VAE解码
        print("🔧 VAE 解码...")
        decode_start = time.time()
        
        latents_scaled = latents / vae.config.scaling_factor
        reconstructed_mel = vae.decode(latents_scaled).sample
        
        decode_time = time.time() - decode_start
        print(f"   解码完成: {decode_time:.2f}秒, 重建mel形状: {reconstructed_mel.shape}")
        
        # 检查解码结果
        print(f"   重建mel范围: [{reconstructed_mel.min().item():.3f}, {reconstructed_mel.max().item():.3f}]")
    
    # 音频重建
    print("🔊 音频重建 (改进版本)...")
    
    try:
        with torch.no_grad():
            # 反归一化mel
            mel_denorm = (reconstructed_mel + 1) / 2 * (mel_max - mel_min) + mel_min
            mel_denorm = torch.clamp(mel_denorm, mel_min, mel_max)
            
            # 检查有效性
            if not torch.isfinite(mel_denorm).all():
                print(f"   ⚠️ 修复解码输出的无效值...")
                mel_denorm = torch.nan_to_num(mel_denorm, nan=-40.0, posinf=-20.0, neginf=-80.0)
            
            # 准备vocoder输入
            vocoder_input = mel_denorm.squeeze(0).squeeze(0)  # [64, time]
            vocoder_input = vocoder_input.transpose(-2, -1)   # [time, 64]
            vocoder_input = vocoder_input.unsqueeze(0)        # [1, time, 64]
            
            print(f"   Vocoder输入形状: {vocoder_input.shape}")
            print(f"   Vocoder输入范围: [{vocoder_input.min().item():.3f}, {vocoder_input.max().item():.3f}]")
            
            # 安全归一化
            vocoder_input_norm = safe_normalize_mel(vocoder_input, target_mean=-5.0, target_std=4.0)
            
            # 使用vocoder
            reconstructed_audio_tensor = vocoder(vocoder_input_norm)
            reconstructed_audio = reconstructed_audio_tensor.squeeze().cpu().numpy()
            
            print(f"   ✅ Vocoder成功: 输出{len(reconstructed_audio)}样本")
            
            # 应用后处理
            print("🛠️ 应用后处理降噪...")
            reconstructed_audio = apply_audio_postprocessing(reconstructed_audio, sr=sample_rate)
            
            vocoder_method = "AudioLDM2_HiFiGAN_v2_NoiseFixed"
            
    except Exception as e:
        print(f"   ❌ HiFiGAN失败: {e}")
        print("   📊 降级到Griffin-Lim...")
        
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
            
            reconstructed_audio = apply_audio_postprocessing(reconstructed_audio, sr=sample_rate)
            vocoder_method = "Griffin_Lim_PostProcessed"
            
        except Exception as griffin_e:
            print(f"   ❌ Griffin-Lim也失败: {griffin_e}")
            reconstructed_audio = np.random.randn(len(audio)) * 0.01
            vocoder_method = "Fallback"
    
    # 长度对齐
    if len(reconstructed_audio) > len(audio):
        reconstructed_audio = reconstructed_audio[:len(audio)]
    elif len(reconstructed_audio) < len(audio):
        reconstructed_audio = np.pad(reconstructed_audio, (0, len(audio) - len(reconstructed_audio)))
    
    # 保存结果
    print("💾 保存结果...")
    output_dir = "vae_noise_fix_v2_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存音频文件
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    reconstructed_path = os.path.join(output_dir, f"{input_name}_reconstructed_v2_{timestamp}.wav")
    
    # 归一化保存
    audio_norm = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    recon_norm = reconstructed_audio / np.max(np.abs(reconstructed_audio)) if np.max(np.abs(reconstructed_audio)) > 0 else reconstructed_audio
    
    torchaudio.save(original_path, torch.from_numpy(audio_norm).unsqueeze(0), sample_rate)
    torchaudio.save(reconstructed_path, torch.from_numpy(recon_norm).unsqueeze(0), sample_rate)
    
    # 计算指标
    print("📊 计算质量指标...")
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
    print(f"VAE 重建测试结果 - 噪音修复 v2")
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
    print(f"✅ 改进项目: 数值稳定性、后处理降噪、渐变处理")
    
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
        print("使用方法: python vae_noise_fix_v2.py <音频文件路径> [最大长度秒数]")
        
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
            print(f"无效的长度参数，使用默认值 {max_length} 秒")
    
    print(f"🚀 开始噪音修复测试 v2")
    
    try:
        result = load_and_test_vae_v2(audio_path, max_length=max_length)
        if result:
            print(f"\n✅ 测试完成！")
            print(f"请比较音频质量，检查噪音是否减少。")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
