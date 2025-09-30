"""
高质量Vocoder改进方案实施
专门解决Griffin-Lim的92.1%信息损失问题

主要改进：
1. 集成多种高质量vocoder
2. 优化mel-spectrogram参数配置
3. 添加后处理增强
4. 提供详细的性能对比
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


def load_high_quality_vocoders():
    """
    加载多种高质量vocoder进行对比
    """
    vocoders = {}
    
    # 1. AudioLDM2内置vocoder（已修正维度问题）
    try:
        pipeline = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float32)
        vocoders['audioldm2'] = pipeline.vocoder
        print("✅ AudioLDM2 SpeechT5HifiGan加载成功")
    except Exception as e:
        print(f"❌ AudioLDM2 vocoder加载失败: {e}")
    
    # 2. 尝试加载其他高质量vocoder
    try:
        from transformers import SpeechT5HifiGan
        hifigan = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        vocoders['hifigan'] = hifigan
        print("✅ Microsoft SpeechT5 HifiGan加载成功")
    except Exception as e:
        print(f"❌ SpeechT5 HifiGan加载失败: {e}")
    
    # 3. 可以添加更多vocoder...
    # 如WaveGlow, Parallel WaveGAN等
    
    return vocoders


def optimize_mel_parameters(audio, sr=16000):
    """
    针对高质量重建优化mel-spectrogram参数
    """
    # 多种mel配置用于测试
    configs = [
        # 标准配置
        {'n_mels': 64, 'n_fft': 1024, 'hop_length': 160, 'name': 'standard'},
        # 高分辨率配置
        {'n_mels': 80, 'n_fft': 2048, 'hop_length': 128, 'name': 'high_res'},
        # 平衡配置
        {'n_mels': 64, 'n_fft': 1536, 'hop_length': 144, 'name': 'balanced'},
    ]
    
    mel_variants = {}
    
    for config in configs:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=config['n_mels'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            window='hann',
            center=True,
            pad_mode='reflect',
            power=2.0,
            fmin=0,
            fmax=sr//2
        )
        
        mel_variants[config['name']] = {
            'spec': mel_spec,
            'config': config
        }
    
    return mel_variants


def advanced_post_processing(audio, sr=16000):
    """
    高级音频后处理
    """
    processed_variants = {}
    
    # 1. 基础后处理
    try:
        # 去噪滤波
        nyquist = sr // 2
        cutoff = min(7500, nyquist * 0.85)
        b, a = butter(4, cutoff / nyquist, btype='low')
        audio_filtered = filtfilt(b, a, audio)
        
        # 轻微动态范围压缩
        audio_compressed = np.sign(audio_filtered) * np.power(np.abs(audio_filtered), 0.85)
        
        processed_variants['basic'] = audio_compressed
        
    except Exception as e:
        print(f"基础后处理失败: {e}")
        processed_variants['basic'] = audio
    
    # 2. 频域增强
    try:
        # STFT域处理
        stft = librosa.stft(audio, n_fft=1024, hop_length=160)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 谱减法去噪
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        magnitude_cleaned = np.maximum(magnitude - 0.1 * noise_floor, 0.1 * magnitude)
        
        # 重建音频
        stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
        audio_enhanced = librosa.istft(stft_cleaned, hop_length=160)
        
        processed_variants['spectral'] = audio_enhanced
        
    except Exception as e:
        print(f"频域增强失败: {e}")
        processed_variants['spectral'] = audio
    
    # 3. 相位优化
    try:
        # 最小相位重建
        stft = librosa.stft(audio, n_fft=1024, hop_length=160)
        magnitude = np.abs(stft)
        
        # 使用最小相位
        log_magnitude = np.log(magnitude + 1e-8)
        cepstrum = np.fft.irfft(log_magnitude, axis=0)
        
        # 构造最小相位
        cepstrum_min = cepstrum.copy()
        cepstrum_min[1:cepstrum.shape[0]//2] *= 2
        cepstrum_min[cepstrum.shape[0]//2+1:] = 0
        
        magnitude_min_phase = np.exp(np.fft.rfft(cepstrum_min, axis=0))
        audio_min_phase = librosa.istft(magnitude_min_phase, hop_length=160)
        
        processed_variants['min_phase'] = audio_min_phase
        
    except Exception as e:
        print(f"相位优化失败: {e}")
        processed_variants['min_phase'] = audio
    
    return processed_variants


def comprehensive_vocoder_test(audio_path):
    """
    综合vocoder测试和对比
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 启动综合vocoder质量提升测试")
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio[:5*16000]  # 5秒
    
    print(f"📊 测试音频: {len(audio)/16000:.1f}秒")
    
    # 创建输出目录
    output_dir = "high_quality_vocoder_test"
    os.makedirs(output_dir, exist_ok=True)
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    orig_path = os.path.join(output_dir, f"{input_name}_original.wav")
    torchaudio.save(orig_path, torch.from_numpy(audio / (np.max(np.abs(audio)) + 1e-8)).unsqueeze(0), 16000)
    
    # 加载VAE
    pipeline = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float32).to(device)
    vae = pipeline.vae
    
    # 加载vocoders
    print(f"\\n🎤 加载高质量vocoders...")
    vocoders = load_high_quality_vocoders()
    
    # 优化mel参数
    print(f"\\n📊 优化mel-spectrogram参数...")
    mel_variants = optimize_mel_parameters(audio)
    
    results = []
    
    # 测试所有vocoder和mel配置组合
    for mel_name, mel_data in mel_variants.items():
        print(f"\\n🔬 测试mel配置: {mel_name}")
        
        mel_spec = mel_data['spec']
        config = mel_data['config']
        
        # 标准化mel-spectrogram
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = 2.0 * (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) - 1.0
        
        # 调整到VAE期望的64 mel bins
        if config['n_mels'] != 64:
            mel_resized = F.interpolate(
                torch.from_numpy(mel_norm).unsqueeze(0).unsqueeze(0),
                size=(64, mel_norm.shape[1]),
                mode='bilinear', align_corners=False
            ).squeeze().numpy()
        else:
            mel_resized = mel_norm
        
        # VAE编码/解码
        mel_tensor = torch.from_numpy(mel_resized).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            latent = vae.encode(mel_tensor).latent_dist.sample()
            reconstructed_mel_tensor = vae.decode(latent).sample
        
        reconstructed_mel = reconstructed_mel_tensor.squeeze().cpu().numpy()
        
        # 测试不同的vocoder
        for vocoder_name, vocoder in vocoders.items():
            print(f"   🎵 使用vocoder: {vocoder_name}")
            
            try:
                if vocoder_name == 'audioldm2':
                    # 使用修正的AudioLDM2 vocoder
                    mel_for_vocoder = torch.from_numpy(reconstructed_mel).unsqueeze(0).float().to(device)
                    mel_transposed = mel_for_vocoder.transpose(-2, -1)
                    
                    with torch.no_grad():
                        audio_vocoder = vocoder(mel_transposed).squeeze().cpu().numpy()
                
                elif vocoder_name == 'hifigan':
                    # 使用SpeechT5 HifiGan
                    # 需要调整输入格式
                    mel_for_hifigan = torch.from_numpy(reconstructed_mel).unsqueeze(0).float()
                    
                    with torch.no_grad():
                        audio_vocoder = vocoder(mel_for_hifigan).squeeze().cpu().numpy()
                
                else:
                    print(f"   ⚠️ 未实现的vocoder: {vocoder_name}")
                    continue
                
                # 长度对齐
                min_len = min(len(audio), len(audio_vocoder))
                audio_aligned = audio[:min_len]
                vocoder_aligned = audio_vocoder[:min_len]
                
                # 后处理测试
                post_processed = advanced_post_processing(vocoder_aligned)
                
                for post_name, post_audio in post_processed.items():
                    method_name = f"{mel_name}_{vocoder_name}_{post_name}"
                    
                    # 长度对齐
                    if len(post_audio) > len(audio_aligned):
                        post_audio = post_audio[:len(audio_aligned)]
                    elif len(post_audio) < len(audio_aligned):
                        post_audio = np.pad(post_audio, (0, len(audio_aligned) - len(post_audio)))
                    
                    # 计算指标
                    snr = 10 * np.log10(np.mean(audio_aligned**2) / (np.mean((audio_aligned - post_audio)**2) + 1e-10))
                    corr = np.corrcoef(audio_aligned, post_audio)[0,1] if len(audio_aligned) > 1 else 0
                    rmse = np.sqrt(np.mean((audio_aligned - post_audio)**2))
                    
                    # 保存音频
                    save_path = os.path.join(output_dir, f"{input_name}_{method_name}_{timestamp}.wav")
                    audio_norm = post_audio / (np.max(np.abs(post_audio)) + 1e-8)
                    torchaudio.save(save_path, torch.from_numpy(audio_norm).unsqueeze(0), 16000)
                    
                    results.append({
                        'method': method_name,
                        'mel_config': mel_name,
                        'vocoder': vocoder_name,
                        'post_process': post_name,
                        'snr': snr,
                        'correlation': corr,
                        'rmse': rmse,
                        'path': save_path
                    })
                    
                    print(f"       {post_name}: SNR={snr:.2f}dB, 相关={corr:.4f}")
                
            except Exception as e:
                print(f"   ❌ {vocoder_name} 失败: {e}")
                continue
    
    # 添加基线对比：直接Griffin-Lim
    print(f"\\n🎵 基线对比: 直接Griffin-Lim")
    for mel_name, mel_data in mel_variants.items():
        mel_spec = mel_data['spec']
        config = mel_data['config']
        
        try:
            # 直接Griffin-Lim重建
            audio_gl = librosa.feature.inverse.mel_to_audio(
                mel_spec,
                sr=16000,
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                n_iter=64
            )
            
            min_len = min(len(audio), len(audio_gl))
            snr_gl = 10 * np.log10(np.mean(audio[:min_len]**2) / (np.mean((audio[:min_len] - audio_gl[:min_len])**2) + 1e-10))
            corr_gl = np.corrcoef(audio[:min_len], audio_gl[:min_len])[0,1]
            
            gl_path = os.path.join(output_dir, f"{input_name}_baseline_{mel_name}_gl.wav")
            torchaudio.save(gl_path, torch.from_numpy(audio_gl[:min_len] / (np.max(np.abs(audio_gl[:min_len])) + 1e-8)).unsqueeze(0), 16000)
            
            results.append({
                'method': f"baseline_{mel_name}_griffin_lim",
                'mel_config': mel_name,
                'vocoder': 'griffin_lim',
                'post_process': 'none',
                'snr': snr_gl,
                'correlation': corr_gl,
                'rmse': np.sqrt(np.mean((audio[:min_len] - audio_gl[:min_len])**2)),
                'path': gl_path
            })
            
            print(f"   {mel_name} Griffin-Lim: SNR={snr_gl:.2f}dB, 相关={corr_gl:.4f}")
            
        except Exception as e:
            print(f"   ❌ {mel_name} Griffin-Lim失败: {e}")
    
    # 结果分析
    print(f"\\n{'='*70}")
    print(f"🎯 高质量Vocoder测试结果分析")
    print(f"{'='*70}")
    
    if results:
        # 按SNR排序
        results.sort(key=lambda x: x['snr'], reverse=True)
        
        print(f"\\n🏆 质量排名 (前10名):")
        for i, result in enumerate(results[:10], 1):
            improvement = result['snr'] - 0.02  # 与之前基线对比
            print(f"   #{i} {result['method'][:50]}...")
            print(f"       📈 SNR: {result['snr']:.2f}dB (+{improvement:.2f})")
            print(f"       🔗 相关性: {result['correlation']:.4f}")
            print(f"       📁 文件: {result['path']}")
            print()
        
        # 最佳结果分析
        best = results[0]
        best_improvement = best['snr'] - 0.02
        
        print(f"🚀 最佳改进效果:")
        print(f"   🏆 最优组合: {best['mel_config']} + {best['vocoder']} + {best['post_process']}")
        print(f"   📈 SNR提升: {best_improvement:+.2f} dB")
        print(f"   🔗 最高相关性: {best['correlation']:.4f}")
        
        if best_improvement > 5:
            print(f"   ✅ 显著改善！Griffin-Lim瓶颈基本解决")
        elif best_improvement > 2:
            print(f"   ⚠️ 有明显改善，但仍有优化空间")
        else:
            print(f"   ❌ 改善有限，需要探索其他方案")
        
        # 方法效果分析
        print(f"\\n📊 不同方法效果分析:")
        
        # 按vocoder分组
        vocoder_performance = {}
        for result in results:
            vocoder = result['vocoder']
            if vocoder not in vocoder_performance:
                vocoder_performance[vocoder] = []
            vocoder_performance[vocoder].append(result['snr'])
        
        for vocoder, snrs in vocoder_performance.items():
            avg_snr = np.mean(snrs)
            max_snr = np.max(snrs)
            print(f"   🎤 {vocoder}: 平均{avg_snr:.2f}dB, 最佳{max_snr:.2f}dB")
        
    else:
        print(f"\\n❌ 没有成功的测试结果")
    
    print(f"\\n📁 所有结果保存在: {output_dir}/")
    print(f"🎧 强烈建议播放最佳结果进行主观评估")
    print(f"\\n✅ 综合vocoder测试完成！")
    
    return results


def main():
    """主函数"""
    import sys
    
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "AudioLDM2_Music_output.wav"
    
    print(f"🎯 高质量Vocoder改进方案测试")
    print(f"=" * 60)
    print(f"目标: 解决Griffin-Lim的92.1%信息损失问题")
    print(f"方法: 集成高质量vocoder + 优化参数 + 后处理增强")
    
    results = comprehensive_vocoder_test(audio_path)
    
    if results and len(results) > 0:
        best_snr = max(results, key=lambda x: x['snr'])['snr']
        baseline_snr = 0.02  # 之前的基线
        total_improvement = best_snr - baseline_snr
        
        print(f"\\n🎉 测试完成总结:")
        print(f"   📈 最大SNR提升: {total_improvement:+.2f} dB")
        print(f"   🎯 Griffin-Lim瓶颈解决程度: {min(100, max(0, total_improvement/10*100)):.1f}%")
        
        if total_improvement > 8:
            print(f"   🎉 重大突破！质量显著提升")
        elif total_improvement > 3:
            print(f"   ✅ 明显改善，接近实用水平")
        else:
            print(f"   ⚠️ 部分改善，需要继续优化")


if __name__ == "__main__":
    main()
