"""
AudioLDM2 VAE 改进版音频重建测试脚本
使用更好的方法提高重建质量

改进点：
1. 使用AudioLDM2的vocoder进行更好的音频重建
2. 优化mel-spectrogram参数
3. 添加更多质量评估指标
4. 提供多种重建方法比较
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
from scipy.signal import correlate
import matplotlib.pyplot as plt

from diffusers import AudioLDM2Pipeline


def improved_vae_test(audio_path, model_id="cvssp/audioldm2-music", max_length=10):
    """
    改进的VAE音频重建测试
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
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
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"音频已裁剪到 {max_length} 秒")
    
    print(f"音频信息: 长度={len(audio)/sample_rate:.2f}秒, 样本数={len(audio)}")
    
    # 方法1: 使用AudioLDM2的官方mel-spectrogram参数
    print("\n=== 方法1: 使用AudioLDM2官方参数 ===")
    result1 = test_with_official_params(audio, vae, vocoder, device, sample_rate)
    
    # 方法2: 使用优化的mel-spectrogram参数
    print("\n=== 方法2: 使用优化参数 ===")
    result2 = test_with_optimized_params(audio, vae, vocoder, device, sample_rate)
    
    # 方法3: 尝试直接使用vocoder（如果可能）
    print("\n=== 方法3: 尝试vocoder重建 ===")
    result3 = test_with_vocoder_direct(audio, vae, vocoder, device, sample_rate)
    
    # 保存所有结果
    output_dir = "vae_improved_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_save = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    torchaudio.save(original_path, torch.from_numpy(audio_save).unsqueeze(0), sample_rate)
    
    # 保存重建结果
    results = {}
    for i, (method_name, result) in enumerate([
        ("官方参数", result1),
        ("优化参数", result2), 
        ("vocoder直接", result3)
    ], 1):
        if result is not None:
            recon_path = os.path.join(output_dir, f"{input_name}_method{i}_{method_name}_{timestamp}.wav")
            
            # 归一化并保存
            recon_audio = result['audio']
            if len(recon_audio) > 0 and np.max(np.abs(recon_audio)) > 0:
                recon_audio = recon_audio / np.max(np.abs(recon_audio))
            
            # 确保长度一致
            if len(recon_audio) > len(audio):
                recon_audio = recon_audio[:len(audio)]
            elif len(recon_audio) < len(audio):
                recon_audio = np.pad(recon_audio, (0, len(audio) - len(recon_audio)))
            
            torchaudio.save(recon_path, torch.from_numpy(recon_audio).unsqueeze(0), sample_rate)
            
            # 计算质量指标
            metrics = calculate_quality_metrics(audio, recon_audio)
            
            results[f"方法{i}_{method_name}"] = {
                'path': recon_path,
                'metrics': metrics,
                'encode_time': result.get('encode_time', 0),
                'decode_time': result.get('decode_time', 0)
            }
    
    # 打印比较结果
    print(f"\n{'='*60}")
    print(f"重建质量比较结果")
    print(f"{'='*60}")
    print(f"原始音频: {original_path}")
    print()
    
    for method_name, data in results.items():
        metrics = data['metrics']
        print(f"{method_name}:")
        print(f"  文件: {data['path']}")
        print(f"  SNR: {metrics['snr']:.2f} dB")
        print(f"  相关系数: {metrics['correlation']:.4f}")
        print(f"  频谱相关性: {metrics['spectral_correlation']:.4f}")
        print(f"  处理时间: {data['encode_time'] + data['decode_time']:.2f}秒")
        print()
    
    # 找出最好的方法
    best_method = max(results.items(), key=lambda x: x[1]['metrics']['snr'])
    print(f"🏆 最佳重建方法: {best_method[0]} (SNR: {best_method[1]['metrics']['snr']:.2f} dB)")
    
    return results


def test_with_official_params(audio, vae, vocoder, device, sample_rate):
    """使用AudioLDM2官方参数"""
    try:
        # AudioLDM2官方参数
        n_fft = 1024
        hop_length = 160
        win_length = 1024
        n_mels = 64
        fmin = 0
        fmax = 8000
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0
        )
        
        # 转换为对数尺度
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 更好的归一化方法
        mel_mean = mel_spec_db.mean()
        mel_std = mel_spec_db.std()
        mel_spec_normalized = (mel_spec_db - mel_mean) / (mel_std + 1e-8)
        mel_spec_normalized = np.clip(mel_spec_normalized, -5, 5)  # 限制在合理范围
        
        norm_params = {'mean': mel_mean, 'std': mel_std}
        
        return vae_encode_decode_with_vocoder(mel_spec_normalized, norm_params, vae, vocoder, device, 
                                            n_fft, hop_length, win_length, sample_rate, fmin, fmax)
    except Exception as e:
        print(f"官方参数方法失败: {e}")
        return None


def test_with_optimized_params(audio, vae, vocoder, device, sample_rate):
    """使用优化参数"""
    try:
        # 优化参数 - 更高质量
        n_fft = 2048
        hop_length = 256
        win_length = 2048
        n_mels = 80
        fmin = 0
        fmax = 8000
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 使用分位数归一化
        p5, p95 = np.percentile(mel_spec_db, [5, 95])
        mel_spec_normalized = 2 * (mel_spec_db - p5) / (p95 - p5) - 1
        mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
        
        norm_params = {'p5': p5, 'p95': p95}
        
        # 调整到标准尺寸
        if n_mels != 64:
            # 插值到64维
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, n_mels)
            new_indices = np.linspace(0, 1, 64)
            interpolator = interp1d(old_indices, mel_spec_normalized, axis=0, kind='linear')
            mel_spec_normalized = interpolator(new_indices)
        
        return vae_encode_decode_with_vocoder(mel_spec_normalized, norm_params, vae, vocoder, device,
                                            1024, 160, 1024, sample_rate, fmin, fmax)
    except Exception as e:
        print(f"优化参数方法失败: {e}")
        return None


def test_with_vocoder_direct(audio, vae, vocoder, device, sample_rate):
    """尝试直接使用vocoder"""
    try:
        # 先尝试看看能否直接用vocoder的逆过程
        print("尝试使用vocoder的mel提取功能...")
        
        # 使用标准参数
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=160,
            n_mels=64,
            fmin=0,
            fmax=8000
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
        
        # 尝试使用AudioLDM2训练时的归一化方式
        mel_spec_normalized = (mel_spec_db + 80) / 80  # 假设训练时使用这种归一化
        mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1) * 2 - 1  # 转到[-1,1]
        
        norm_params = {'method': 'audioldm2_style'}
        
        return vae_encode_decode_improved(mel_spec_normalized, norm_params, vae, vocoder, device, sample_rate)
    except Exception as e:
        print(f"vocoder直接方法失败: {e}")
        return None


def vae_encode_decode_with_vocoder(mel_spec_normalized, norm_params, vae, vocoder, device, 
                                 n_fft, hop_length, win_length, sample_rate, fmin, fmax):
    """VAE编码解码 + 更好的音频重建"""
    start_time = time.time()
    
    # 转换为张量
    mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
    if device == "cuda":
        mel_tensor = mel_tensor.to(torch.float16)
    else:
        mel_tensor = mel_tensor.to(torch.float32)
    
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
    
    # 确保尺寸适合VAE
    pad_width = (8 - (mel_input.shape[-1] % 8)) % 8
    if pad_width > 0:
        mel_input = F.pad(mel_input, (0, pad_width))
    
    with torch.no_grad():
        # VAE编码
        latents = vae.encode(mel_input).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        encode_time = time.time() - start_time
        
        # VAE解码
        decode_start = time.time()
        latents = latents / vae.config.scaling_factor
        reconstructed_mel = vae.decode(latents).sample
        decode_time = time.time() - decode_start
        
        # 反归一化
        recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
        
        # 根据归一化方法进行反归一化
        if 'mean' in norm_params:
            recon_mel_denorm = recon_mel_np * norm_params['std'] + norm_params['mean']
        elif 'p5' in norm_params:
            recon_mel_denorm = (recon_mel_np + 1) / 2 * (norm_params['p95'] - norm_params['p5']) + norm_params['p5']
        else:
            recon_mel_denorm = recon_mel_np * 80 - 80
        
        # 使用改进的Griffin-Lim
        recon_mel_power = librosa.db_to_power(recon_mel_denorm)
        recon_mel_power = np.clip(recon_mel_power, 1e-10, None)
        
        # 使用更多迭代的Griffin-Lim
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(
            recon_mel_power,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            n_iter=60,  # 增加迭代次数
            length=None
        )
        
        # 后处理：降噪
        reconstructed_audio = apply_simple_denoising(reconstructed_audio)
    
    return {
        'audio': reconstructed_audio,
        'encode_time': encode_time,
        'decode_time': decode_time
    }


def vae_encode_decode_improved(mel_spec_normalized, norm_params, vae, vocoder, device, sample_rate):
    """改进的VAE编码解码"""
    start_time = time.time()
    
    mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
    if device == "cuda":
        mel_tensor = mel_tensor.to(torch.float16)
    else:
        mel_tensor = mel_tensor.to(torch.float32)
    
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
    
    # 确保尺寸适合VAE
    pad_width = (8 - (mel_input.shape[-1] % 8)) % 8
    if pad_width > 0:
        mel_input = F.pad(mel_input, (0, pad_width))
    
    with torch.no_grad():
        # VAE编码
        latents = vae.encode(mel_input).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        encode_time = time.time() - start_time
        
        # VAE解码
        decode_start = time.time()
        latents = latents / vae.config.scaling_factor
        reconstructed_mel = vae.decode(latents).sample
        decode_time = time.time() - decode_start
        
        # 尝试直接使用vocoder（如果兼容）
        try:
            # 调整形状给vocoder
            vocoder_input = reconstructed_mel
            
            # 检查vocoder是否能直接处理
            if hasattr(vocoder, 'decode') or hasattr(vocoder, '__call__'):
                # 尝试使用vocoder
                reconstructed_audio = vocoder(vocoder_input).squeeze().cpu().numpy()
                if len(reconstructed_audio.shape) > 1:
                    reconstructed_audio = reconstructed_audio[0]
                    
                print("成功使用vocoder进行音频重建！")
            else:
                # 回退到Griffin-Lim
                raise Exception("Vocoder不兼容，使用Griffin-Lim")
                
        except Exception as e:
            print(f"Vocoder失败，使用Griffin-Lim: {e}")
            
            # 反归一化并使用Griffin-Lim
            recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
            recon_mel_denorm = (recon_mel_np + 1) / 2 * 80 - 80
            recon_mel_power = librosa.db_to_power(recon_mel_denorm)
            recon_mel_power = np.clip(recon_mel_power, 1e-10, None)
            
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                recon_mel_power,
                sr=sample_rate,
                hop_length=160,
                n_fft=1024,
                n_iter=60
            )
    
    return {
        'audio': reconstructed_audio,
        'encode_time': encode_time,
        'decode_time': decode_time
    }


def apply_simple_denoising(audio, alpha=0.1):
    """简单的降噪处理"""
    try:
        # 使用低通滤波减少高频噪声
        from scipy.signal import butter, filtfilt
        
        nyquist = 8000  # sample_rate / 2
        cutoff = 7500   # 去除7.5kHz以上的频率
        order = 4
        
        b, a = butter(order, cutoff / nyquist, btype='low')
        filtered_audio = filtfilt(b, a, audio)
        
        # 轻微的平滑
        smoothed_audio = alpha * filtered_audio + (1 - alpha) * audio
        
        return smoothed_audio
    except:
        return audio


def calculate_quality_metrics(original, reconstructed):
    """计算更全面的质量指标"""
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len]
    recon = reconstructed[:min_len]
    
    # 基本指标
    mse = np.mean((orig - recon) ** 2)
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean((orig - recon) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    correlation = np.corrcoef(orig, recon)[0, 1] if len(orig) > 1 else 0
    
    # 频谱相关性
    orig_fft = np.abs(np.fft.fft(orig))
    recon_fft = np.abs(np.fft.fft(recon))
    spectral_correlation = np.corrcoef(orig_fft, recon_fft)[0, 1] if len(orig_fft) > 1 else 0
    
    # 频谱质心比较
    orig_centroid = librosa.feature.spectral_centroid(y=orig, sr=16000)[0].mean()
    recon_centroid = librosa.feature.spectral_centroid(y=recon, sr=16000)[0].mean()
    centroid_diff = abs(orig_centroid - recon_centroid) / orig_centroid
    
    return {
        'mse': mse,
        'snr': snr,
        'correlation': correlation,
        'spectral_correlation': spectral_correlation,
        'centroid_difference': centroid_diff
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python improved_vae_test.py <音频文件路径> [最大长度秒数]")
        
        # 查找音频文件
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
    
    print(f"开始改进版VAE测试: {audio_path}")
    print(f"最大长度: {max_length} 秒")
    
    try:
        results = improved_vae_test(audio_path, max_length=max_length)
        if results:
            print("\n✅ 改进版测试完成！")
            print("请播放不同方法的重建音频来比较质量差异。")
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
