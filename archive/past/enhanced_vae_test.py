"""
AudioLDM2 VAE 改进版本 - 使用内置Vocoder
实现最有前景的改进：使用AudioLDM2的SpeechT5HifiGan vocoder替代Griffin-Lim

主要改进：
1. 直接使用AudioLDM2内置vocoder进行mel到音频转换
2. 优化mel-spectrogram参数配置
3. 改进归一化策略
4. 添加多种重建方法对比
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


def enhanced_vae_test(audio_path, model_id="cvssp/audioldm2-music", max_length=10):
    """
    使用AudioLDM2内置vocoder的增强VAE测试
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 使用设备: {device}")
    
    if not os.path.exists(audio_path):
        print(f"❌ 错误: 找不到音频文件 {audio_path}")
        return
    
    print(f"🔄 正在加载 AudioLDM2 模型: {model_id}")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    # 打印vocoder信息
    print(f"🔧 Vocoder信息: {type(vocoder).__name__}")
    if hasattr(vocoder, 'config'):
        config = vocoder.config
        print(f"   - 输入维度: {config.model_in_dim}")
        print(f"   - 采样率: {config.sampling_rate}")
        print(f"   - 上采样率: {config.upsample_rates}")
    
    print(f"📁 正在加载音频: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"✂️ 音频已裁剪到 {max_length} 秒")
    
    print(f"📊 音频信息: 长度={len(audio)/sample_rate:.2f}秒, 样本数={len(audio)}")
    
    # 测试多种方法
    methods = [
        ("原始Griffin-Lim", test_original_griffinlim),
        ("AudioLDM2 Vocoder", test_audioldm2_vocoder), 
        ("优化参数+Vocoder", test_optimized_vocoder),
        ("最佳配置", test_best_config)
    ]
    
    results = {}
    output_dir = "vae_enhanced_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_save = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    torchaudio.save(original_path, torch.from_numpy(audio_save).unsqueeze(0), sample_rate)
    
    print(f"\\n🧪 开始测试 {len(methods)} 种重建方法...")
    
    for i, (method_name, method_func) in enumerate(methods, 1):
        print(f"\\n{'='*50}")
        print(f"🔬 方法 {i}: {method_name}")
        print(f"{'='*50}")
        
        try:
            result = method_func(audio, vae, vocoder, device, sample_rate)
            
            if result and result.get('audio') is not None:
                recon_path = os.path.join(output_dir, f"{input_name}_{method_name.replace(' ', '_')}_{timestamp}.wav")
                
                recon_audio = result['audio']
                
                # 安全的归一化
                if len(recon_audio) > 0:
                    max_val = np.max(np.abs(recon_audio))
                    if max_val > 0:
                        recon_audio = recon_audio / max_val
                
                # 确保长度一致
                target_length = len(audio)
                if len(recon_audio) > target_length:
                    recon_audio = recon_audio[:target_length]
                elif len(recon_audio) < target_length:
                    recon_audio = np.pad(recon_audio, (0, target_length - len(recon_audio)))
                
                # 保存音频
                torchaudio.save(recon_path, torch.from_numpy(recon_audio).unsqueeze(0), sample_rate)
                
                # 计算详细质量指标
                metrics = calculate_detailed_metrics(audio, recon_audio, sample_rate)
                
                results[method_name] = {
                    'path': recon_path,
                    'metrics': metrics,
                    'processing_time': result.get('processing_time', 0),
                    'compression_info': result.get('compression_info', {}),
                    'success': True
                }
                
                print(f"✅ {method_name} 成功!")
                print(f"   📈 SNR: {metrics['snr']:.2f} dB")
                print(f"   🔗 相关系数: {metrics['correlation']:.4f}")
                print(f"   ⏱️ 处理时间: {result.get('processing_time', 0):.2f}秒")
            else:
                print(f"❌ {method_name} 失败: 无音频输出")
                results[method_name] = {'success': False}
                
        except Exception as e:
            print(f"❌ {method_name} 出错: {str(e)}")
            results[method_name] = {'success': False, 'error': str(e)}
    
    # 打印最终对比结果
    print_enhanced_results(original_path, results)
    
    return results


def test_original_griffinlim(audio, vae, vocoder, device, sample_rate):
    """原始Griffin-Lim方法（基线）"""
    start_time = time.time()
    
    # 使用原始参数
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=1024, hop_length=160, n_mels=64,
        fmin=0, fmax=8000, power=2.0
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
    mel_spec_normalized = 2 * (mel_spec_db - mel_min) / (mel_max - mel_min) - 1
    mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
    
    # VAE处理
    recon_audio, comp_info = process_through_vae(
        mel_spec_normalized, vae, device, 
        norm_params={'min': mel_min, 'max': mel_max}
    )
    
    # Griffin-Lim重建
    recon_mel_denorm = (recon_audio + 1) / 2 * (mel_max - mel_min) + mel_min
    recon_mel_power = librosa.db_to_power(recon_mel_denorm)
    final_audio = librosa.feature.inverse.mel_to_audio(
        recon_mel_power, sr=sample_rate, hop_length=160, n_fft=1024, n_iter=32
    )
    
    processing_time = time.time() - start_time
    
    return {
        'audio': final_audio,
        'processing_time': processing_time,
        'compression_info': comp_info
    }


def test_audioldm2_vocoder(audio, vae, vocoder, device, sample_rate):
    """使用AudioLDM2内置vocoder"""
    start_time = time.time()
    
    # 标准mel参数
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=1024, hop_length=160, n_mels=64,
        fmin=0, fmax=8000, power=2.0
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
    mel_spec_normalized = 2 * (mel_spec_db - mel_min) / (mel_max - mel_min) - 1
    mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
    
    # VAE处理
    recon_mel, comp_info = process_through_vae(
        mel_spec_normalized, vae, device,
        norm_params={'min': mel_min, 'max': mel_max},
        return_mel=True
    )
    
    # 使用AudioLDM2 vocoder
    try:
        print("🎤 尝试使用AudioLDM2内置vocoder...")
        
        # 准备vocoder输入
        mel_tensor = torch.from_numpy(recon_mel).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.to(torch.float16)
        
        # 调整形状适配vocoder
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)  # 添加batch维度
        
        print(f"   Vocoder输入形状: {mel_tensor.shape}")
        
        with torch.no_grad():
            # 调用vocoder
            vocoder_output = vocoder(mel_tensor)
            
            if isinstance(vocoder_output, tuple):
                final_audio = vocoder_output[0]
            else:
                final_audio = vocoder_output
            
            # 转换为numpy
            if torch.is_tensor(final_audio):
                final_audio = final_audio.squeeze().cpu().numpy()
            
            print(f"✅ Vocoder重建成功! 输出形状: {final_audio.shape}")
        
    except Exception as e:
        print(f"❌ Vocoder失败: {e}")
        print("🔄 回退到Griffin-Lim...")
        
        # 回退到Griffin-Lim
        recon_mel_denorm = (recon_mel + 1) / 2 * (mel_max - mel_min) + mel_min
        recon_mel_power = librosa.db_to_power(recon_mel_denorm)
        final_audio = librosa.feature.inverse.mel_to_audio(
            recon_mel_power, sr=sample_rate, hop_length=160, n_fft=1024
        )
    
    processing_time = time.time() - start_time
    
    return {
        'audio': final_audio,
        'processing_time': processing_time,
        'compression_info': comp_info
    }


def test_optimized_vocoder(audio, vae, vocoder, device, sample_rate):
    """优化参数 + vocoder"""
    start_time = time.time()
    
    # 优化的mel参数
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, 
        n_fft=2048,      # 更高分辨率
        hop_length=256,  # 更高时间分辨率  
        win_length=2048,
        n_mels=80,       # 更多mel bins
        fmin=0, fmax=8000, power=1.0  # 使用幅度谱
    )
    
    # 改进的归一化
    mel_spec_log = np.log(mel_spec + 1e-8)
    p5, p95 = np.percentile(mel_spec_log, [5, 95])
    mel_spec_normalized = 2 * (mel_spec_log - p5) / (p95 - p5) - 1
    mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
    
    # 调整到64 mels (vocoder要求)
    if mel_spec_normalized.shape[0] != 64:
        from scipy.interpolate import interp1d
        old_indices = np.linspace(0, 1, mel_spec_normalized.shape[0])
        new_indices = np.linspace(0, 1, 64)
        interpolator = interp1d(old_indices, mel_spec_normalized, axis=0, kind='linear')
        mel_spec_normalized = interpolator(new_indices)
    
    # 调整时间维度到标准hop_length
    target_frames = len(audio) // 160
    if mel_spec_normalized.shape[1] != target_frames:
        from scipy.interpolate import interp1d
        old_indices = np.linspace(0, 1, mel_spec_normalized.shape[1])
        new_indices = np.linspace(0, 1, target_frames)
        interpolator = interp1d(old_indices, mel_spec_normalized, axis=1, kind='linear')
        mel_spec_normalized = interpolator(new_indices)
    
    # VAE处理
    recon_mel, comp_info = process_through_vae(
        mel_spec_normalized, vae, device,
        norm_params={'p5': p5, 'p95': p95, 'method': 'percentile'},
        return_mel=True
    )
    
    # 使用vocoder重建
    try:
        mel_tensor = torch.from_numpy(recon_mel).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.to(torch.float16)
        
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
        
        with torch.no_grad():
            vocoder_output = vocoder(mel_tensor)
            if isinstance(vocoder_output, tuple):
                final_audio = vocoder_output[0]
            else:
                final_audio = vocoder_output
            
            if torch.is_tensor(final_audio):
                final_audio = final_audio.squeeze().cpu().numpy()
        
        print("✅ 优化参数+Vocoder重建成功!")
        
    except Exception as e:
        print(f"❌ 优化vocoder失败: {e}, 使用Griffin-Lim")
        # 回退处理
        if 'method' in comp_info:
            recon_mel_denorm = (recon_mel + 1) / 2 * (p95 - p5) + p5
        else:
            recon_mel_denorm = recon_mel
        
        recon_mel_linear = np.exp(recon_mel_denorm)
        final_audio = librosa.feature.inverse.mel_to_audio(
            recon_mel_linear, sr=sample_rate, hop_length=160, n_fft=1024
        )
    
    processing_time = time.time() - start_time
    
    return {
        'audio': final_audio,
        'processing_time': processing_time,
        'compression_info': comp_info
    }


def test_best_config(audio, vae, vocoder, device, sample_rate):
    """最佳配置：结合所有优化"""
    start_time = time.time()
    
    # 最优mel参数配置
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate,
        n_fft=1024,      # 与vocoder匹配
        hop_length=160,  # 与vocoder匹配
        win_length=1024,
        n_mels=64,       # 与vocoder匹配
        fmin=0, fmax=8000,
        power=1.0,       # 幅度谱更好
        norm='slaney'    # 更好的mel滤波器
    )
    
    # 最稳健的归一化
    mel_spec_log = np.log(mel_spec + 1e-8)
    mel_mean = mel_spec_log.mean()
    mel_std = mel_spec_log.std()
    mel_spec_normalized = (mel_spec_log - mel_mean) / (mel_std + 1e-8)
    mel_spec_normalized = np.clip(mel_spec_normalized, -3, 3)  # 合理范围
    
    # VAE处理
    recon_mel, comp_info = process_through_vae(
        mel_spec_normalized, vae, device,
        norm_params={'mean': mel_mean, 'std': mel_std, 'method': 'zscore'},
        return_mel=True
    )
    
    # 使用vocoder + 后处理
    try:
        mel_tensor = torch.from_numpy(recon_mel).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.to(torch.float16)
        
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
        
        with torch.no_grad():
            vocoder_output = vocoder(mel_tensor)
            if isinstance(vocoder_output, tuple):
                final_audio = vocoder_output[0]
            else:
                final_audio = vocoder_output
            
            if torch.is_tensor(final_audio):
                final_audio = final_audio.squeeze().cpu().numpy()
        
        # 后处理：轻微平滑
        if len(final_audio) > 21:
            from scipy.signal import savgol_filter
            try:
                final_audio = savgol_filter(final_audio, 21, 3)
            except:
                pass
        
        print("✅ 最佳配置重建成功!")
        
    except Exception as e:
        print(f"❌ 最佳配置失败: {e}")
        # 高质量回退
        recon_mel_denorm = recon_mel * mel_std + mel_mean
        recon_mel_linear = np.exp(recon_mel_denorm)
        final_audio = librosa.feature.inverse.mel_to_audio(
            recon_mel_linear, sr=sample_rate, hop_length=160, n_fft=1024, n_iter=100
        )
    
    processing_time = time.time() - start_time
    
    return {
        'audio': final_audio,
        'processing_time': processing_time,
        'compression_info': comp_info
    }


def process_through_vae(mel_spec_normalized, vae, device, norm_params, return_mel=False):
    """通过VAE处理mel-spectrogram"""
    
    # 转换为张量
    mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
    if device == "cuda":
        mel_tensor = mel_tensor.to(torch.float16)
    
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, mels, time)
    
    # 填充到8的倍数
    if mel_input.shape[-1] % 8 != 0:
        pad_width = 8 - (mel_input.shape[-1] % 8)
        mel_input = F.pad(mel_input, (0, pad_width))
    
    with torch.no_grad():
        # VAE编码解码
        latents = vae.encode(mel_input).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        latents = latents / vae.config.scaling_factor
        reconstructed_mel = vae.decode(latents).sample
        
        # 转换回numpy
        recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
        
        if return_mel:
            return recon_mel_np, {
                'compression_ratio': mel_input.numel() / latents.numel(),
                'latent_shape': latents.shape
            }
        else:
            # 反归一化处理
            if norm_params.get('method') == 'percentile':
                p5, p95 = norm_params['p5'], norm_params['p95']
                recon_mel_denorm = (recon_mel_np + 1) / 2 * (p95 - p5) + p5
            elif norm_params.get('method') == 'zscore':
                mean, std = norm_params['mean'], norm_params['std']
                recon_mel_denorm = recon_mel_np * std + mean
            else:
                mel_min, mel_max = norm_params['min'], norm_params['max']
                recon_mel_denorm = (recon_mel_np + 1) / 2 * (mel_max - mel_min) + mel_min
            
            return recon_mel_denorm, {
                'compression_ratio': mel_input.numel() / latents.numel(),
                'latent_shape': latents.shape
            }


def calculate_detailed_metrics(original, reconstructed, sample_rate):
    """计算详细的质量指标"""
    min_len = min(len(original), len(reconstructed))
    if min_len == 0:
        return {'snr': -np.inf, 'correlation': 0}
    
    orig = original[:min_len]
    recon = reconstructed[:min_len]
    
    # 基础指标
    mse = np.mean((orig - recon) ** 2)
    signal_power = np.mean(orig ** 2)
    noise_power = mse
    
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    try:
        correlation = np.corrcoef(orig, recon)[0, 1]
        if np.isnan(correlation):
            correlation = 0
    except:
        correlation = 0
    
    # 频谱指标
    try:
        orig_fft = np.abs(np.fft.fft(orig))
        recon_fft = np.abs(np.fft.fft(recon))
        spectral_corr = np.corrcoef(orig_fft, recon_fft)[0, 1]
        if np.isnan(spectral_corr):
            spectral_corr = 0
    except:
        spectral_corr = 0
    
    # 感知指标
    try:
        orig_mfcc = librosa.feature.mfcc(y=orig, sr=sample_rate, n_mfcc=13)
        recon_mfcc = librosa.feature.mfcc(y=recon, sr=sample_rate, n_mfcc=13)
        mfcc_corr = np.corrcoef(orig_mfcc.flatten(), recon_mfcc.flatten())[0, 1]
        if np.isnan(mfcc_corr):
            mfcc_corr = 0
    except:
        mfcc_corr = 0
    
    return {
        'mse': float(mse),
        'snr': float(snr),
        'correlation': float(correlation),
        'spectral_correlation': float(spectral_corr),
        'mfcc_correlation': float(mfcc_corr)
    }


def print_enhanced_results(original_path, results):
    """打印增强测试结果"""
    print(f"\\n{'='*70}")
    print(f"🎯 AudioLDM2 VAE 增强重建测试结果")
    print(f"{'='*70}")
    print(f"📁 原始音频: {os.path.basename(original_path)}")
    print()
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("❌ 所有方法都失败了")
        return
    
    # 按SNR排序
    sorted_results = sorted(successful_results.items(), 
                           key=lambda x: x[1]['metrics']['snr'], reverse=True)
    
    print(f"📊 方法对比 (按SNR排序):")
    print("-" * 70)
    
    baseline_snr = None
    for i, (method_name, data) in enumerate(sorted_results, 1):
        metrics = data['metrics']
        
        if i == 1:
            baseline_snr = metrics['snr']
        
        improvement = metrics['snr'] - baseline_snr if baseline_snr is not None else 0
        
        print(f"🏆 #{i} {method_name}:")
        print(f"   📄 文件: {os.path.basename(data['path'])}")
        print(f"   📈 SNR: {metrics['snr']:.2f} dB ({improvement:+.2f})")
        print(f"   🔗 时域相关性: {metrics['correlation']:.4f}")
        print(f"   🎵 频谱相关性: {metrics['spectral_correlation']:.4f}")
        print(f"   🎤 MFCC相关性: {metrics['mfcc_correlation']:.4f}")
        print(f"   ⏱️ 处理时间: {data['processing_time']:.2f}秒")
        
        if 'compression_info' in data:
            comp_info = data['compression_info']
            print(f"   📦 压缩比: {comp_info.get('compression_ratio', 0):.1f}:1")
        print()
    
    # 总结改进效果
    if len(sorted_results) > 1:
        best_snr = sorted_results[0][1]['metrics']['snr']
        baseline_snr = sorted_results[-1][1]['metrics']['snr']
        improvement = best_snr - baseline_snr
        
        print(f"🚀 最佳改进效果:")
        print(f"   📈 SNR提升: {improvement:.2f} dB")
        print(f"   🏆 最佳方法: {sorted_results[0][0]}")
    
    print(f"\\n📁 所有结果保存在 vae_enhanced_test/ 目录")
    print("🎧 建议播放音频文件来主观评估改进效果")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python enhanced_vae_test.py <音频文件路径> [最大长度秒数]")
        
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
    
    max_length = 5  # 默认5秒用于快速测试
    if len(sys.argv) >= 3:
        try:
            max_length = float(sys.argv[2])
        except ValueError:
            print(f"无效的长度参数，使用默认值 {max_length} 秒")
    
    print(f"🚀 开始增强VAE测试: {audio_path}")
    print(f"⏱️ 最大长度: {max_length} 秒")
    
    try:
        results = enhanced_vae_test(audio_path, max_length=max_length)
        if results:
            print("\\n✅ 增强测试完成！")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
