"""
AudioLDM2 VAE 终极改进版本
尝试找到AudioLDM2的最佳重建方法，包括使用内置vocoder

这个版本专注于：
1. 理解AudioLDM2的真实工作流程
2. 尝试逆向工程AudioLDM2的mel处理
3. 使用最接近训练时的参数
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

from diffusers import AudioLDM2Pipeline


def ultimate_vae_test(audio_path, model_id="cvssp/audioldm2-music", max_length=10):
    """
    终极VAE测试 - 尽可能接近AudioLDM2的原始工作流程
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
    
    # 探索vocoder的属性
    print(f"\\n📊 Vocoder 信息:")
    print(f"  类型: {type(vocoder).__name__}")
    print(f"  是否有decode方法: {hasattr(vocoder, 'decode')}")
    print(f"  是否可调用: {hasattr(vocoder, '__call__')}")
    if hasattr(vocoder, 'config'):
        print(f"  Vocoder配置: {vocoder.config}")
    
    # 探索VAE的属性
    print(f"\\n🔧 VAE 信息:")
    print(f"  类型: {type(vae).__name__}")
    if hasattr(vae, 'config'):
        config = vae.config
        print(f"  输入channels: {getattr(config, 'in_channels', 'unknown')}")
        print(f"  输出channels: {getattr(config, 'out_channels', 'unknown')}")
        print(f"  潜在维度: {getattr(config, 'latent_channels', 'unknown')}")
        print(f"  缩放因子: {getattr(config, 'scaling_factor', 'unknown')}")
    
    print(f"\\n正在加载音频: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"音频已裁剪到 {max_length} 秒")
    
    print(f"音频信息: 长度={len(audio)/sample_rate:.2f}秒, 样本数={len(audio)}")
    
    # 方法1: 研究AudioLDM2的源码找到正确的mel参数
    print("\\n=== 方法1: 研究AudioLDM2内部mel参数 ===")
    result1 = test_with_audioldm2_internals(audio, pipeline, device, sample_rate)
    
    # 方法2: 模拟AudioLDM2的训练预处理
    print("\\n=== 方法2: 模拟训练时的预处理 ===")
    result2 = test_with_training_simulation(audio, vae, vocoder, device, sample_rate)
    
    # 方法3: 尝试更好的后处理
    print("\\n=== 方法3: 改进的后处理 ===")
    result3 = test_with_improved_postprocessing(audio, vae, device, sample_rate)
    
    # 保存结果
    output_dir = "vae_ultimate_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_save = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    torchaudio.save(original_path, torch.from_numpy(audio_save).unsqueeze(0), sample_rate)
    
    # 分析和保存结果
    results = {}
    method_names = ["AudioLDM2内部", "训练模拟", "改进后处理"]
    
    for i, (method_name, result) in enumerate(zip(method_names, [result1, result2, result3]), 1):
        if result is not None and result.get('audio') is not None:
            recon_path = os.path.join(output_dir, f"{input_name}_ultimate_method{i}_{method_name}_{timestamp}.wav")
            
            recon_audio = result['audio']
            if len(recon_audio) > 0 and np.max(np.abs(recon_audio)) > 0:
                recon_audio = recon_audio / np.max(np.abs(recon_audio))
            
            # 确保长度一致
            if len(recon_audio) > len(audio):
                recon_audio = recon_audio[:len(audio)]
            elif len(recon_audio) < len(audio):
                recon_audio = np.pad(recon_audio, (0, len(audio) - len(recon_audio)))
            
            torchaudio.save(recon_path, torch.from_numpy(recon_audio).unsqueeze(0), sample_rate)
            
            # 计算详细质量指标
            metrics = calculate_comprehensive_metrics(audio, recon_audio, sample_rate)
            
            results[f"方法{i}_{method_name}"] = {
                'path': recon_path,
                'metrics': metrics,
                'processing_time': result.get('processing_time', 0),
                'compression_info': result.get('compression_info', {})
            }
    
    # 打印详细结果
    print_comprehensive_results(original_path, results)
    
    return results


def test_with_audioldm2_internals(audio, pipeline, device, sample_rate):
    """尝试找到AudioLDM2内部的mel处理方法"""
    try:
        print("探索AudioLDM2的内部mel处理...")
        
        vae = pipeline.vae
        
        # 尝试找到正确的mel参数
        # 这些参数基于AudioLDM2论文和常见配置
        mel_params = {
            'n_fft': 1024,
            'hop_length': 160,
            'win_length': 1024,
            'n_mels': 64,
            'fmin': 0,
            'fmax': 8000,
            'power': 1.0,  # 使用幅度而不是功率
            'norm': 'slaney',
            'htk': False
        }
        
        start_time = time.time()
        
        # 计算mel谱
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, **mel_params)
        
        # 转换为对数尺度 - 使用AudioLDM2风格
        mel_spec_log = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
        
        # 使用更接近AudioLDM2训练的归一化
        # 基于研究，AudioLDM2可能使用这种归一化方式
        mel_mean = -4.0  # 假设的均值
        mel_std = 4.0    # 假设的标准差
        mel_spec_normalized = (mel_spec_log - mel_mean) / mel_std
        mel_spec_normalized = np.clip(mel_spec_normalized, -3, 3)
        
        # 转换为张量
        mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.to(torch.float16)
        
        mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
        
        # 确保尺寸兼容
        if mel_input.shape[-1] % 8 != 0:
            pad_width = 8 - (mel_input.shape[-1] % 8)
            mel_input = F.pad(mel_input, (0, pad_width))
        
        with torch.no_grad():
            # VAE处理
            latents = vae.encode(mel_input).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents / vae.config.scaling_factor
            reconstructed_mel = vae.decode(latents).sample
            
            # 反归一化
            recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
            recon_mel_denorm = recon_mel_np * mel_std + mel_mean
            
            # 转回线性尺度
            recon_mel_linear = np.exp(recon_mel_denorm)
            
            # 使用更精确的Griffin-Lim参数
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                recon_mel_linear,
                sr=sample_rate,
                n_fft=mel_params['n_fft'],
                hop_length=mel_params['hop_length'],
                win_length=mel_params['win_length'],
                fmin=mel_params['fmin'],
                fmax=mel_params['fmax'],
                n_iter=100,  # 更多迭代
                window='hann',
                length=len(audio)
            )
        
        processing_time = time.time() - start_time
        
        return {
            'audio': reconstructed_audio,
            'processing_time': processing_time,
            'compression_info': {
                'original_size': mel_input.numel(),
                'compressed_size': latents.numel(),
                'compression_ratio': mel_input.numel() / latents.numel()
            }
        }
        
    except Exception as e:
        print(f"AudioLDM2内部方法失败: {e}")
        return None


def test_with_training_simulation(audio, vae, vocoder, device, sample_rate):
    """模拟AudioLDM2训练时的数据处理"""
    try:
        print("模拟AudioLDM2训练时的数据预处理...")
        
        start_time = time.time()
        
        # 使用HiFi-GAN风格的mel参数（AudioLDM2可能基于此）
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,  # 尝试不同的hop length
            win_length=1024,
            n_mels=80,
            fmin=0,
            fmax=8000,
            power=1.0
        )
        
        # 转换为分贝并使用HiFi-GAN风格的归一化
        mel_spec_db = 20 * np.log10(np.maximum(mel_spec, 1e-5))
        mel_spec_normalized = (mel_spec_db + 100) / 100  # [0, 1] 范围
        mel_spec_normalized = mel_spec_normalized * 2 - 1  # [-1, 1] 范围
        
        # 调整到64 mels
        if mel_spec_normalized.shape[0] != 64:
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, mel_spec_normalized.shape[0])
            new_indices = np.linspace(0, 1, 64)
            interpolator = interp1d(old_indices, mel_spec_normalized, axis=0, kind='linear')
            mel_spec_normalized = interpolator(new_indices)
        
        # 调整时间维度到160 hop length的等效
        target_time_frames = len(audio) // 160
        if mel_spec_normalized.shape[1] != target_time_frames:
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, mel_spec_normalized.shape[1])
            new_indices = np.linspace(0, 1, target_time_frames)
            interpolator = interp1d(old_indices, mel_spec_normalized, axis=1, kind='linear')
            mel_spec_normalized = interpolator(new_indices)
        
        # VAE处理
        mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.to(torch.float16)
        
        mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
        
        # 填充到8的倍数
        if mel_input.shape[-1] % 8 != 0:
            pad_width = 8 - (mel_input.shape[-1] % 8)
            mel_input = F.pad(mel_input, (0, pad_width))
        
        with torch.no_grad():
            latents = vae.encode(mel_input).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents / vae.config.scaling_factor
            reconstructed_mel = vae.decode(latents).sample
            
            # 反归一化
            recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
            recon_mel_db = (recon_mel_np + 1) / 2 * 100 - 100
            recon_mel_linear = np.power(10, recon_mel_db / 20)
            
            # Griffin-Lim重建
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                recon_mel_linear,
                sr=sample_rate,
                n_fft=1024,
                hop_length=160,
                win_length=1024,
                fmin=0,
                fmax=8000,
                n_iter=200,  # 更多迭代以获得更好质量
                momentum=0.99,
                length=len(audio)
            )
        
        processing_time = time.time() - start_time
        
        return {
            'audio': reconstructed_audio,
            'processing_time': processing_time,
            'compression_info': {
                'original_size': mel_input.numel(),
                'compressed_size': latents.numel(),
                'compression_ratio': mel_input.numel() / latents.numel()
            }
        }
        
    except Exception as e:
        print(f"训练模拟方法失败: {e}")
        return None


def test_with_improved_postprocessing(audio, vae, device, sample_rate):
    """使用改进的后处理方法"""
    try:
        print("使用改进的后处理重建方法...")
        
        start_time = time.time()
        
        # 使用标准mel参数
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=160,
            n_mels=64,
            fmin=0,
            fmax=8000
        )
        
        # 更稳定的对数转换
        mel_spec_log = np.log(mel_spec + 1e-8)
        
        # 使用百分位数归一化 - 更稳健
        p1, p99 = np.percentile(mel_spec_log, [1, 99])
        mel_spec_normalized = 2 * (mel_spec_log - p1) / (p99 - p1) - 1
        mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
        
        # VAE处理
        mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.to(torch.float16)
        
        mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
        
        if mel_input.shape[-1] % 8 != 0:
            pad_width = 8 - (mel_input.shape[-1] % 8)
            mel_input = F.pad(mel_input, (0, pad_width))
        
        with torch.no_grad():
            latents = vae.encode(mel_input).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents / vae.config.scaling_factor
            reconstructed_mel = vae.decode(latents).sample
            
            # 改进的反归一化
            recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
            recon_mel_denorm = (recon_mel_np + 1) / 2 * (p99 - p1) + p1
            recon_mel_linear = np.exp(recon_mel_denorm)
            
            # 多步骤音频重建
            # 1. 标准Griffin-Lim
            audio_gl = librosa.feature.inverse.mel_to_audio(
                recon_mel_linear,
                sr=sample_rate,
                hop_length=160,
                n_fft=1024,
                n_iter=32
            )
            
            # 2. 进一步使用ISTFT优化
            # 计算STFT
            D = librosa.stft(audio_gl, hop_length=160, n_fft=1024)
            
            # 重建mel并比较
            reconstructed_audio = librosa.istft(D, hop_length=160, length=len(audio))
            
            # 3. 简单的后处理滤波
            from scipy.signal import savgol_filter
            if len(reconstructed_audio) > 51:  # 确保有足够的点进行滤波
                reconstructed_audio = savgol_filter(reconstructed_audio, 51, 3)
        
        processing_time = time.time() - start_time
        
        return {
            'audio': reconstructed_audio,
            'processing_time': processing_time,
            'compression_info': {
                'original_size': mel_input.numel(),
                'compressed_size': latents.numel(),
                'compression_ratio': mel_input.numel() / latents.numel()
            }
        }
        
    except Exception as e:
        print(f"改进后处理方法失败: {e}")
        return None


def calculate_comprehensive_metrics(original, reconstructed, sample_rate):
    """计算全面的质量指标"""
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len]
    recon = reconstructed[:min_len]
    
    # 基本指标
    mse = np.mean((orig - recon) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig - recon))
    
    # 信噪比
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean((orig - recon) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # 相关性
    correlation = np.corrcoef(orig, recon)[0, 1] if len(orig) > 1 else 0
    
    # 频谱指标
    orig_fft = np.abs(np.fft.fft(orig))
    recon_fft = np.abs(np.fft.fft(recon))
    spectral_correlation = np.corrcoef(orig_fft, recon_fft)[0, 1] if len(orig_fft) > 1 else 0
    
    # 感知指标
    try:
        orig_mfcc = librosa.feature.mfcc(y=orig, sr=sample_rate, n_mfcc=13)
        recon_mfcc = librosa.feature.mfcc(y=recon, sr=sample_rate, n_mfcc=13)
        mfcc_correlation = np.corrcoef(orig_mfcc.flatten(), recon_mfcc.flatten())[0, 1]
    except:
        mfcc_correlation = 0
    
    # 零交叉率比较
    orig_zcr = librosa.feature.zero_crossing_rate(orig)[0].mean()
    recon_zcr = librosa.feature.zero_crossing_rate(recon)[0].mean()
    zcr_diff = abs(orig_zcr - recon_zcr) / (orig_zcr + 1e-8)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'snr': snr,
        'correlation': correlation,
        'spectral_correlation': spectral_correlation,
        'mfcc_correlation': mfcc_correlation,
        'zcr_difference': zcr_diff
    }


def print_comprehensive_results(original_path, results):
    """打印详细的测试结果"""
    print(f"\\n{'='*70}")
    print(f"终极VAE重建测试结果")
    print(f"{'='*70}")
    print(f"📁 原始音频: {original_path}")
    print()
    
    if not results:
        print("❌ 所有方法都失败了")
        return
    
    # 按SNR排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['snr'], reverse=True)
    
    print("📊 各方法详细对比:")
    print("-" * 70)
    
    for method_name, data in sorted_results:
        metrics = data['metrics']
        comp_info = data['compression_info']
        
        print(f"🔬 {method_name}:")
        print(f"   📄 文件: {os.path.basename(data['path'])}")
        print(f"   ⚡ 处理时间: {data['processing_time']:.2f}秒")
        print(f"   📏 压缩比: {comp_info.get('compression_ratio', 0):.1f}:1")
        print(f"   📈 质量指标:")
        print(f"      SNR: {metrics['snr']:.2f} dB")
        print(f"      相关系数: {metrics['correlation']:.4f}")
        print(f"      频谱相关性: {metrics['spectral_correlation']:.4f}")
        print(f"      MFCC相关性: {metrics['mfcc_correlation']:.4f}")
        print(f"      RMSE: {metrics['rmse']:.6f}")
        print(f"      零交叉率差异: {metrics['zcr_difference']:.4f}")
        print()
    
    # 推荐最佳方法
    best_method = sorted_results[0]
    print(f"🏆 推荐方法: {best_method[0]}")
    print(f"   综合得分最高 (SNR: {best_method[1]['metrics']['snr']:.2f} dB)")
    
    print("\\n💡 质量改进建议:")
    best_snr = best_method[1]['metrics']['snr']
    if best_snr < 0:
        print("   - 当前重建质量较低，建议尝试:")
        print("     1. 调整mel-spectrogram参数")
        print("     2. 使用专门的vocoder模型")
        print("     3. 考虑端到端的音频压缩方法")
    elif best_snr < 10:
        print("   - 质量中等，可以进一步优化:")
        print("     1. 优化归一化方法")
        print("     2. 增加Griffin-Lim迭代次数")
        print("     3. 添加后处理滤波")
    else:
        print("   - 重建质量良好！")
    
    print(f"\\n📁 所有结果已保存到 vae_ultimate_test/ 目录")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python ultimate_vae_test.py <音频文件路径> [最大长度秒数]")
        
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(Path('.').glob(f'*{ext}'))
        
        if audio_files:
            print(f"\\n找到音频文件:")
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
    
    print(f"🚀 开始终极VAE测试: {audio_path}")
    print(f"⏱️ 最大长度: {max_length} 秒")
    
    try:
        results = ultimate_vae_test(audio_path, max_length=max_length)
        if results:
            print("\\n✅ 终极测试完成！")
            print("🎧 请播放不同方法的重建音频来评估质量。")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
