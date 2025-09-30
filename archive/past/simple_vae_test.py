"""
AudioLDM2 VAE 快速重建测试脚本
简化版本，专注于核心 VAE 测试功能

使用方法:
python simple_vae_test.py [音频文件路径] [可选：音频长度秒数]

示例:
python simple_vae_test.py techno.wav 5
"""

import torch
import librosa
import numpy as np
import os
import sys
import time
from pathlib import Path
import torchaudio
import torch.nn.functional as F
import soundfile as sf

from diffusers import AudioLDM2Pipeline


def load_and_test_vae(audio_path, model_id="cvssp/audioldm2-music", max_length=10):
    """
    加载音频文件并测试 VAE 重建
    
    Args:
        audio_path: 音频文件路径
        model_id: AudioLDM2-Music 模型ID
        max_length: 最大音频长度（秒）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 检查输入文件
    if not os.path.exists(audio_path):
        print(f"错误: 找不到音频文件 {audio_path}")
        return
    
    print(f"正在加载 AudioLDM2 模型: {model_id}")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    print(f"正在加载音频: {audio_path}")
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # 限制音频长度
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"音频已裁剪到 {max_length} 秒")
    
    print(f"音频信息: 长度={len(audio)/sample_rate:.2f}秒, 样本数={len(audio)}")
    
    # 将音频转换为 mel-spectrogram
    print("转换音频为 mel-spectrogram...")
    
    # 使用 vocoder 将音频转换为 mel-spectrogram
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 使用 vocoder 将音频转换为 mel-spectrogram
        # 这里我们需要使用 vocoder 的逆过程来获得 mel-spectrogram
        # 但是 vocoder 通常只有 mel->audio 的过程，我们需要用不同的方法
        
        # 计算 mel-spectrogram（使用 librosa）
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=64,  # AudioLDM2 使用 64 mel bins
            hop_length=160,  # AudioLDM2 hop length
            n_fft=1024,
            fmin=0,
            fmax=8000
        )
          # 转换为对数尺度
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 归一化到 [-1, 1] 范围，使用更保守的方法
        mel_min = mel_spec_db.min()
        mel_max = mel_spec_db.max()
        mel_spec_normalized = 2 * (mel_spec_db - mel_min) / (mel_max - mel_min) - 1
        mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
        
        # 保存归一化参数以便后续反归一化
        norm_params = {'min_val': mel_min, 'max_val': mel_max}
          # 转换为张量并调整形状
        mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
        
        # 确保数据类型与模型一致
        if device == "cuda":
            mel_tensor = mel_tensor.to(torch.float16)
        else:
            mel_tensor = mel_tensor.to(torch.float32)
        
        # 调整为 VAE 期望的形状: (batch, channels, height, width)
        # mel_tensor 形状: (n_mels, time_frames) -> (1, 1, n_mels, time_frames)
        mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
        
        print(f"Mel-spectrogram 形状: {mel_input.shape}")
    
    # VAE 编码-解码测试
    print("开始 VAE 编码...")
    start_time = time.time()
    
    with torch.no_grad():
        # 编码
        try:
            latents = vae.encode(mel_input).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            encode_time = time.time() - start_time
            print(f"编码完成: {encode_time:.2f}秒, 潜在形状: {latents.shape}")
            
            # 解码
            print("开始 VAE 解码...")
            decode_start = time.time()
            
            latents = latents / vae.config.scaling_factor
            reconstructed_mel = vae.decode(latents).sample
            
            decode_time = time.time() - decode_start
            print(f"解码完成: {decode_time:.2f}秒, 重建 mel 形状: {reconstructed_mel.shape}")
            
        except Exception as e:
            print(f"VAE 编码/解码过程中发生错误: {str(e)}")
            print("尝试使用不同的输入格式...")
            
            # 尝试不同的输入格式
            # 可能需要调整 mel-spectrogram 的尺寸
            target_height = 64
            target_width = mel_input.shape[-1]
            
            # 确保尺寸是 VAE 要求的倍数
            pad_width = (8 - (target_width % 8)) % 8
            if pad_width > 0:
                mel_input = F.pad(mel_input, (0, pad_width))
                target_width = mel_input.shape[-1]
            
            print(f"调整后的 mel 输入形状: {mel_input.shape}")
            
            # 重试编码
            latents = vae.encode(mel_input).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            encode_time = time.time() - start_time
            print(f"编码完成: {encode_time:.2f}秒, 潜在形状: {latents.shape}")
            
            # 解码
            print("开始 VAE 解码...")
            decode_start = time.time()
            
            latents = latents / vae.config.scaling_factor
            reconstructed_mel = vae.decode(latents).sample
            
            decode_time = time.time() - decode_start
            print(f"解码完成: {decode_time:.2f}秒, 重建 mel 形状: {reconstructed_mel.shape}")    # 将重建的 mel-spectrogram 转换回音频
    print("将重建的 mel-spectrogram 转换回音频...")
    
    with torch.no_grad():
        # 方法1: 使用AudioLDM2内置的HiFiGAN (推荐)
        print("🎤 尝试使用AudioLDM2内置HiFiGAN...")
        try:
            # 准备AudioLDM2 HiFiGAN的输入格式: [batch, time, mel_dim]
            vocoder_input = reconstructed_mel.squeeze(0)  # [1, 64, time] -> [64, time]
            vocoder_input = vocoder_input.transpose(-2, -1)  # [64, time] -> [time, 64]
            vocoder_input = vocoder_input.unsqueeze(0)  # [time, 64] -> [1, time, 64]
            
            print(f"   HiFiGAN输入格式: {vocoder_input.shape}")
            
            # 使用AudioLDM2内置HiFiGAN
            reconstructed_audio_tensor = vocoder(vocoder_input)
            reconstructed_audio = reconstructed_audio_tensor.squeeze().cpu().numpy()
            
            print(f"   ✅ AudioLDM2 HiFiGAN成功: 输出{len(reconstructed_audio)}样本")
            vocoder_method = "AudioLDM2_HiFiGAN"
            
        except Exception as e:
            print(f"   ❌ AudioLDM2 HiFiGAN失败: {e}")
            print("📊 降级到Griffin-Lim算法...")
            
            # 方法2: Griffin-Lim算法 (降级方案)
            # 反归一化 mel-spectrogram
            recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
            
            # 使用保存的归一化参数进行反归一化
            recon_mel_denorm = (recon_mel_np + 1) / 2 * (norm_params['max_val'] - norm_params['min_val']) + norm_params['min_val']
            
            # 确保没有无效值
            recon_mel_denorm = np.nan_to_num(recon_mel_denorm, nan=-80.0, posinf=-20.0, neginf=-80.0)
            recon_mel_denorm = np.clip(recon_mel_denorm, -80.0, 0.0)
            
            # 转换回功率谱
            recon_mel_power = librosa.db_to_power(recon_mel_denorm)
            
            # 确保功率谱值合理
            recon_mel_power = np.clip(recon_mel_power, 1e-10, None)
            
            # 使用 Griffin-Lim 算法重建音频
            try:
                reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                    recon_mel_power,
                    sr=sample_rate,
                    hop_length=160,
                    n_fft=1024,
                    fmin=0,
                    fmax=8000
                )
                vocoder_method = "Griffin_Lim"
                print(f"   ✅ Griffin-Lim成功: 输出{len(reconstructed_audio)}样本")
            except Exception as e:
                print(f"   ❌ Griffin-Lim失败: {e}")
                print("   🔄 使用随机音频作为占位符...")
                
                # 如果都失败，使用占位符
                reconstructed_audio = np.random.randn(len(audio)) * 0.1
                vocoder_method = "Fallback_Noise"
        
        # 确保音频值合理
        if len(reconstructed_audio) > 0:
            reconstructed_audio = np.nan_to_num(reconstructed_audio, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            reconstructed_audio = np.zeros_like(audio)
    
    # 保存结果
    output_dir = "vae_quick_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频（处理后的版本）
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_save = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    save_audio_compatible(audio_save, original_path, sample_rate)
    
    # 保存重建音频
    reconstructed_path = os.path.join(output_dir, f"{input_name}_reconstructed_{timestamp}.wav")
    recon_audio_norm = reconstructed_audio / np.max(np.abs(reconstructed_audio)) if np.max(np.abs(reconstructed_audio)) > 0 else reconstructed_audio
    
    # 确保重建音频长度与原始音频一致
    if len(recon_audio_norm) > len(audio):
        recon_audio_norm = recon_audio_norm[:len(audio)]
    elif len(recon_audio_norm) < len(audio):
        recon_audio_norm = np.pad(recon_audio_norm, (0, len(audio) - len(recon_audio_norm)))
    
    save_audio_compatible(recon_audio_norm, reconstructed_path, sample_rate)
    
    # 计算简单指标
    min_len = min(len(audio), len(recon_audio_norm))
    orig_flat = audio[:min_len]
    recon_flat = recon_audio_norm[:min_len]
    
    # 计算 MSE 和相关系数
    mse = np.mean((orig_flat - recon_flat) ** 2)
    correlation = np.corrcoef(orig_flat, recon_flat)[0, 1] if len(orig_flat) > 1 else 0
    
    # 计算 SNR
    signal_power = np.mean(orig_flat ** 2)
    noise_power = np.mean((orig_flat - recon_flat) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
      # 计算压缩比
    original_size = mel_input.numel()
    compressed_size = latents.numel()
    compression_ratio = original_size / compressed_size
    
    print(f"\n{'='*50}")
    print(f"VAE 重建测试结果")
    print(f"{'='*50}")
    print(f"原始音频: {original_path}")
    print(f"重建音频: {reconstructed_path}")
    print(f"编码时间: {encode_time:.2f}秒")
    print(f"解码时间: {decode_time:.2f}秒")
    print(f"总时间: {encode_time + decode_time:.2f}秒")
    print(f"MSE: {mse:.6f}")
    print(f"SNR: {snr:.2f} dB")
    print(f"相关系数: {correlation:.4f}")
    print(f"Mel-spectrogram 形状: {mel_input.shape}")
    print(f"潜在表示形状: {latents.shape}")
    print(f"压缩比: {compression_ratio:.1f}:1")
    print(f"重建方法: {vocoder_method}")
    
    return {
        'original_path': original_path,
        'reconstructed_path': reconstructed_path,
        'mse': mse,
        'snr': snr,
        'correlation': correlation,
        'encode_time': encode_time,
        'decode_time': decode_time,
        'compression_ratio': compression_ratio
    }


def save_audio_compatible(audio_data, file_path, sample_rate=16000):
    """
    使用最佳兼容性保存音频文件
    确保生成的WAV文件可以被各种播放器正确播放
    """
    # 确保音频数据是numpy array
    if isinstance(audio_data, torch.Tensor):
        audio_np = audio_data.detach().cpu().numpy()
    else:
        audio_np = np.array(audio_data)
    
    # 确保是1D数组
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    
    # 移除任何无效值
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 确保在合理范围内 [-1, 1]
    max_val = np.max(np.abs(audio_np))
    if max_val > 1.0:
        audio_np = audio_np / max_val * 0.95
    elif max_val > 0 and max_val < 0.01:  # 太安静的话放大一点
        audio_np = audio_np / max_val * 0.5
    
    # 转换为float32确保精度
    audio_np = audio_np.astype(np.float32)
    
    print(f"   保存音频: {file_path}")
    print(f"   音频长度: {len(audio_np)} 样本 ({len(audio_np)/sample_rate:.2f} 秒)")
    print(f"   数值范围: [{np.min(audio_np):.6f}, {np.max(audio_np):.6f}]")
    
    try:
        # 方法1: 使用soundfile，PCM_16格式 (最通用)
        sf.write(file_path, audio_np, sample_rate, subtype='PCM_16')
        print(f"   ✅ soundfile保存成功 (PCM_16)")
        
        # 验证文件
        test_audio, test_sr = sf.read(file_path)
        print(f"   ✅ 验证成功: 长度={len(test_audio)}, 采样率={test_sr}")
        return True
        
    except Exception as e:
        print(f"   ❌ soundfile失败: {e}")
        
        # 方法2: 降级到torchaudio
        try:
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            torchaudio.save(file_path, audio_tensor, sample_rate)
            print(f"   ✅ torchaudio保存成功")
            return True
        except Exception as e2:
            print(f"   ❌ torchaudio也失败: {e2}")
            return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python simple_vae_test.py <音频文件路径> [最大长度秒数]")
        
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
    print(f"开始测试音频文件: {audio_path}")
    print(f"最大长度: {max_length} 秒")
    
    try:
        result = load_and_test_vae(audio_path, max_length=max_length)
        if result:
            print("\n测试完成！")
            print("可以播放原始音频和重建音频来比较质量。")
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
