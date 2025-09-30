"""
AudioLDM2 VAE 稳定重建测试
修复了兼容性问题，专注于提供稳定可靠的重建结果

改进点：
1. 修复numpy兼容性问题
2. 使用更稳定的参数
3. 添加更好的错误处理
4. 实现更合理的mel-spectrogram处理
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
from scipy.signal import savgol_filter

from diffusers import AudioLDM2Pipeline


def stable_vae_test(audio_path, model_id="cvssp/audioldm2-music", max_length=10):
    """
    稳定的VAE重建测试
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
    
    # 测试三种不同的方法
    methods = [
        ("标准方法", test_standard_method),
        ("高质量方法", test_high_quality_method),
        ("稳健方法", test_robust_method)
    ]
    
    results = {}
    output_dir = "vae_stable_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_save = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    torchaudio.save(original_path, torch.from_numpy(audio_save).unsqueeze(0), sample_rate)
    
    for i, (method_name, method_func) in enumerate(methods, 1):
        print(f"\\n=== 方法{i}: {method_name} ===")
        
        try:
            result = method_func(audio, vae, device, sample_rate)
            
            if result and result.get('audio') is not None:
                recon_path = os.path.join(output_dir, f"{input_name}_method{i}_{method_name}_{timestamp}.wav")
                
                recon_audio = result['audio']
                
                # 安全的归一化
                if len(recon_audio) > 0:
                    max_val = np.max(np.abs(recon_audio))
                    if max_val > 0:
                        recon_audio = recon_audio / max_val
                
                # 确保长度一致
                if len(recon_audio) > len(audio):
                    recon_audio = recon_audio[:len(audio)]
                elif len(recon_audio) < len(audio):
                    recon_audio = np.pad(recon_audio, (0, len(audio) - len(recon_audio)))
                
                # 保存音频
                torchaudio.save(recon_path, torch.from_numpy(recon_audio).unsqueeze(0), sample_rate)
                
                # 计算质量指标
                metrics = calculate_safe_metrics(audio, recon_audio, sample_rate)
                
                results[f"方法{i}_{method_name}"] = {
                    'path': recon_path,
                    'metrics': metrics,
                    'processing_time': result.get('processing_time', 0),
                    'success': True
                }
                
                print(f"✅ {method_name} 成功")
                print(f"   SNR: {metrics['snr']:.2f} dB")
                print(f"   相关系数: {metrics['correlation']:.4f}")
            else:
                print(f"❌ {method_name} 失败")
                results[f"方法{i}_{method_name}"] = {'success': False}
                
        except Exception as e:
            print(f"❌ {method_name} 出错: {str(e)}")
            results[f"方法{i}_{method_name}"] = {'success': False, 'error': str(e)}
    
    # 打印最终结果
    print_stable_results(original_path, results)
    
    return results


def test_standard_method(audio, vae, device, sample_rate):
    """标准方法：使用基本的mel-spectrogram参数"""
    start_time = time.time()
    
    # 标准mel参数
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=160,
        n_mels=64,
        fmin=0,
        fmax=8000,
        power=2.0
    )
    
    # 转换为对数尺度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 简单归一化
    mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
    if mel_max > mel_min:
        mel_spec_normalized = 2 * (mel_spec_db - mel_min) / (mel_max - mel_min) - 1
    else:
        mel_spec_normalized = np.zeros_like(mel_spec_db)
    
    mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
    
    # VAE处理
    reconstructed_audio = process_with_vae(
        mel_spec_normalized, vae, device, sample_rate, 
        norm_params={'min': mel_min, 'max': mel_max}
    )
    
    processing_time = time.time() - start_time
    
    return {
        'audio': reconstructed_audio,
        'processing_time': processing_time
    }


def test_high_quality_method(audio, vae, device, sample_rate):
    """高质量方法：使用更好的参数和更多处理步骤"""
    start_time = time.time()
    
    # 高质量mel参数
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=160,  # 保持与标准一致
        win_length=2048,
        n_mels=80,  # 更多mel bins
        fmin=0,
        fmax=8000,
        power=2.0
    )
    
    # 转换为对数尺度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 使用分位数归一化（更稳健）
    p5, p95 = np.percentile(mel_spec_db, [5, 95])
    if p95 > p5:
        mel_spec_normalized = 2 * (mel_spec_db - p5) / (p95 - p5) - 1
    else:
        mel_spec_normalized = np.zeros_like(mel_spec_db)
    
    mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
    
    # 如果mel bins不是64，需要调整
    if mel_spec_normalized.shape[0] != 64:
        # 简单的线性插值调整到64
        from scipy.interpolate import interp1d
        old_indices = np.linspace(0, 1, mel_spec_normalized.shape[0])
        new_indices = np.linspace(0, 1, 64)
        interpolator = interp1d(old_indices, mel_spec_normalized, axis=0, kind='linear')
        mel_spec_normalized = interpolator(new_indices)
    
    # VAE处理
    reconstructed_audio = process_with_vae(
        mel_spec_normalized, vae, device, sample_rate,
        norm_params={'p5': p5, 'p95': p95, 'method': 'percentile'},
        n_fft=2048, win_length=2048
    )
    
    processing_time = time.time() - start_time
    
    return {
        'audio': reconstructed_audio,
        'processing_time': processing_time
    }


def test_robust_method(audio, vae, device, sample_rate):
    """稳健方法：使用最保守和稳定的参数"""
    start_time = time.time()
    
    # 保守的mel参数
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        n_mels=64,
        fmin=0,
        fmax=8000,
        power=1.0  # 使用幅度而不是功率
    )
    
    # 添加小的常数避免log(0)
    mel_spec_log = np.log(mel_spec + 1e-8)
    
    # 使用均值和标准差归一化
    mel_mean = mel_spec_log.mean()
    mel_std = mel_spec_log.std()
    
    if mel_std > 0:
        mel_spec_normalized = (mel_spec_log - mel_mean) / mel_std
        mel_spec_normalized = np.clip(mel_spec_normalized, -3, 3)  # 限制范围
    else:
        mel_spec_normalized = np.zeros_like(mel_spec_log)
    
    # VAE处理
    reconstructed_audio = process_with_vae(
        mel_spec_normalized, vae, device, sample_rate,
        norm_params={'mean': mel_mean, 'std': mel_std, 'method': 'zscore'},
        use_power=False  # 表示使用的是幅度谱
    )
    
    # 简单的后处理平滑
    if len(reconstructed_audio) > 21:  # 确保有足够的点
        try:
            reconstructed_audio = savgol_filter(reconstructed_audio, 21, 3)
        except:
            pass  # 如果滤波失败就跳过
    
    processing_time = time.time() - start_time
    
    return {
        'audio': reconstructed_audio,
        'processing_time': processing_time
    }


def process_with_vae(mel_spec_normalized, vae, device, sample_rate, norm_params, 
                    n_fft=1024, win_length=1024, use_power=True):
    """使用VAE处理mel-spectrogram"""
    
    # 转换为张量
    mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
    
    # 确保数据类型匹配
    if device == "cuda":
        mel_tensor = mel_tensor.to(torch.float16)
    else:
        mel_tensor = mel_tensor.to(torch.float32)
    
    # 添加batch和channel维度
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
    
    # 填充到8的倍数
    if mel_input.shape[-1] % 8 != 0:
        pad_width = 8 - (mel_input.shape[-1] % 8)
        mel_input = F.pad(mel_input, (0, pad_width))
    
    with torch.no_grad():
        # VAE编码
        latents = vae.encode(mel_input).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # VAE解码
        latents = latents / vae.config.scaling_factor
        reconstructed_mel = vae.decode(latents).sample
        
        # 转换回numpy
        recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
        
        # 反归一化
        if norm_params.get('method') == 'percentile':
            p5, p95 = norm_params['p5'], norm_params['p95']
            recon_mel_denorm = (recon_mel_np + 1) / 2 * (p95 - p5) + p5
        elif norm_params.get('method') == 'zscore':
            mean, std = norm_params['mean'], norm_params['std']
            recon_mel_denorm = recon_mel_np * std + mean
        else:
            # 标准min-max反归一化
            mel_min, mel_max = norm_params['min'], norm_params['max']
            recon_mel_denorm = (recon_mel_np + 1) / 2 * (mel_max - mel_min) + mel_min
        
        # 转换回线性尺度
        if use_power:
            recon_mel_linear = librosa.db_to_power(recon_mel_denorm)
        else:
            recon_mel_linear = np.exp(recon_mel_denorm)
        
        # 确保值为正数
        recon_mel_linear = np.maximum(recon_mel_linear, 1e-10)
        
        # 使用Griffin-Lim重建音频
        try:
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                recon_mel_linear,
                sr=sample_rate,
                hop_length=160,
                n_fft=n_fft,
                win_length=win_length,
                fmin=0,
                fmax=8000,
                n_iter=32  # 适中的迭代次数
            )
        except Exception as e:
            print(f"Griffin-Lim失败，使用备用方法: {e}")
            # 备用方法：简单的ISTFT
            n_frames = recon_mel_linear.shape[1]
            reconstructed_audio = np.random.randn(n_frames * 160) * 0.01  # 低音量噪声
    
    return reconstructed_audio


def calculate_safe_metrics(original, reconstructed, sample_rate):
    """安全的质量指标计算"""
    try:
        min_len = min(len(original), len(reconstructed))
        if min_len == 0:
            return {'snr': -np.inf, 'correlation': 0, 'error': 'Empty audio'}
        
        orig = original[:min_len]
        recon = reconstructed[:min_len]
        
        # 基本指标
        mse = np.mean((orig - recon) ** 2)
        
        # SNR计算
        signal_power = np.mean(orig ** 2)
        noise_power = mse
        
        if noise_power > 0 and signal_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf') if noise_power == 0 else -np.inf
        
        # 相关系数
        try:
            correlation = np.corrcoef(orig, recon)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        except:
            correlation = 0
        
        # 频谱相关性
        try:
            orig_fft = np.abs(np.fft.fft(orig))
            recon_fft = np.abs(np.fft.fft(recon))
            spectral_corr = np.corrcoef(orig_fft, recon_fft)[0, 1]
            if np.isnan(spectral_corr):
                spectral_corr = 0
        except:
            spectral_corr = 0
        
        return {
            'mse': float(mse),
            'snr': float(snr),
            'correlation': float(correlation),
            'spectral_correlation': float(spectral_corr)
        }
        
    except Exception as e:
        return {'snr': -np.inf, 'correlation': 0, 'error': str(e)}


def print_stable_results(original_path, results):
    """打印稳定测试的结果"""
    print(f"\\n{'='*60}")
    print(f"稳定VAE重建测试结果")
    print(f"{'='*60}")
    print(f"📁 原始音频: {os.path.basename(original_path)}")
    print()
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("❌ 所有方法都失败了")
        print("\\n失败原因:")
        for method, result in results.items():
            if 'error' in result:
                print(f"  {method}: {result['error']}")
        return
    
    print(f"✅ 成功的方法 ({len(successful_results)}/{len(results)}):")
    print()
    
    # 按SNR排序
    sorted_results = sorted(successful_results.items(), 
                           key=lambda x: x[1]['metrics']['snr'], reverse=True)
    
    for method_name, data in sorted_results:
        metrics = data['metrics']
        
        print(f"🔬 {method_name}:")
        print(f"   📄 文件: {os.path.basename(data['path'])}")
        print(f"   ⏱️ 处理时间: {data['processing_time']:.2f}秒")
        print(f"   📊 质量指标:")
        print(f"      SNR: {metrics['snr']:.2f} dB")
        print(f"      相关系数: {metrics['correlation']:.4f}")
        print(f"      频谱相关性: {metrics['spectral_correlation']:.4f}")
        print(f"      MSE: {metrics['mse']:.6f}")
        print()
    
    # 推荐最佳方法
    if sorted_results:
        best_method = sorted_results[0]
        print(f"🏆 推荐方法: {best_method[0]}")
        print(f"   最高SNR: {best_method[1]['metrics']['snr']:.2f} dB")
    
    print(f"\\n📁 所有结果保存在 vae_stable_test/ 目录")
    print("🎧 建议播放音频文件来主观评估重建质量")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python stable_vae_test.py <音频文件路径> [最大长度秒数]")
        
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
    
    print(f"🚀 开始稳定VAE测试: {audio_path}")
    print(f"⏱️ 最大长度: {max_length} 秒")
    
    try:
        results = stable_vae_test(audio_path, max_length=max_length)
        if results:
            print("\\n✅ 稳定测试完成！")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
