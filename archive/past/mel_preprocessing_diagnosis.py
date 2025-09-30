#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AudioLDM2 VAE噪声问题深度诊断
==========================

测试不同的mel频谱预处理参数，找到最接近AudioLDM2训练时的设置
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from diffusers import AudioLDM2Pipeline
import time
import matplotlib.pyplot as plt


def save_audio_compatible(audio: np.ndarray, filepath: str, sample_rate: int = 16000):
    """兼容性音频保存"""
    try:
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(filepath, audio, sample_rate, subtype='PCM_16')
        print(f"✅ 保存: {Path(filepath).name}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def test_mel_preprocessing_variants(audio_path: str, max_length: float = 3.0):
    """
    测试不同的mel频谱预处理参数
    """
    print(f"\n🔬 AudioLDM2 Mel频谱预处理参数测试")
    print(f"🎯 目标: 找到最佳的mel频谱预处理参数")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 设备: {device}")
    
    # 加载模型
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sample_rate, duration=max_length)
    print(f"📁 音频: {len(audio)/sr:.2f}秒")
    
    # 测试不同的mel频谱参数组合
    test_configs = [
        {
            'name': 'Current',
            'n_mels': 64,
            'hop_length': 160,
            'n_fft': 1024,
            'win_length': 1024,
            'normalization': 'div_20',
            'db_ref': 'max'
        },
        {
            'name': 'Standard_16k',
            'n_mels': 64,
            'hop_length': 160,
            'n_fft': 1024,
            'win_length': 1024,
            'normalization': 'div_80_plus_1',
            'db_ref': 'max'
        },
        {
            'name': 'TacoCentric',
            'n_mels': 64,
            'hop_length': 160,
            'n_fft': 1024,
            'win_length': 1024,
            'normalization': 'minmax_sym',
            'db_ref': 'max'
        },
        {
            'name': 'AudioLDM_Style',
            'n_mels': 64,
            'hop_length': 160,
            'n_fft': 1024,
            'win_length': 1024,
            'normalization': 'minmax_0_1',
            'db_ref': 'max'
        },
        {
            'name': 'No_Normalization',
            'n_mels': 64,
            'hop_length': 160,
            'n_fft': 1024,
            'win_length': 1024,
            'normalization': 'none',
            'db_ref': 'max'
        }
    ]
    
    results = []
    output_dir = Path("mel_preprocessing_test")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🧪 开始测试 {len(test_configs)} 种配置...")
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n--- 测试 {i}/{len(test_configs)}: {config['name']} ---")
        
        try:
            # 创建mel频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=config['n_mels'],
                hop_length=config['hop_length'],
                n_fft=config['n_fft'],
                win_length=config['win_length'],
                window='hann',
                center=True,
                pad_mode='reflect'
            )
            
            # 转换为dB
            if config['db_ref'] == 'max':
                mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            else:
                mel_db = librosa.power_to_db(mel_spec)
            
            print(f"   原始mel范围: [{mel_db.min():.1f}, {mel_db.max():.1f}] dB")
            
            # 应用不同的归一化方法
            if config['normalization'] == 'div_20':
                mel_norm = mel_db / 20.0
                mel_norm = np.clip(mel_norm, -1, 1)
            elif config['normalization'] == 'div_80_plus_1':
                mel_norm = (mel_db + 80) / 80
                mel_norm = mel_norm * 2 - 1  # 转换到[-1, 1]
            elif config['normalization'] == 'minmax_sym':
                mel_min, mel_max = mel_db.min(), mel_db.max()
                mel_norm = 2 * (mel_db - mel_min) / (mel_max - mel_min) - 1
            elif config['normalization'] == 'minmax_0_1':
                mel_min, mel_max = mel_db.min(), mel_db.max()
                mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
            elif config['normalization'] == 'none':
                mel_norm = mel_db
            
            print(f"   归一化后: [{mel_norm.min():.3f}, {mel_norm.max():.3f}]")
            
            # 转换为tensor
            mel_tensor = torch.from_numpy(mel_norm).to(device)
            if device == "cuda":
                mel_tensor = mel_tensor.half()
            
            mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
            
            # VAE处理
            with torch.no_grad():
                # 编码
                latent_dist = vae.encode(mel_input)
                if hasattr(latent_dist, 'latent_dist'):
                    latent = latent_dist.latent_dist.sample()
                else:
                    latent = latent_dist.sample()
                
                latent = latent * vae.config.scaling_factor
                
                # 解码
                latent_for_decode = latent / vae.config.scaling_factor
                reconstructed_mel = vae.decode(latent_for_decode).sample
            
            # HiFiGAN处理
            vocoder_input = reconstructed_mel.squeeze(1).transpose(1, 2)
            vocoder_dtype = next(vocoder.parameters()).dtype
            vocoder_input = vocoder_input.to(vocoder_dtype)
            
            waveform = vocoder(vocoder_input)
            reconstructed_audio = waveform.squeeze().detach().cpu().float().numpy()
            
            # 计算质量指标
            min_len = min(len(audio), len(reconstructed_audio))
            audio_aligned = audio[:min_len]
            recon_aligned = reconstructed_audio[:min_len]
            
            mse = np.mean((audio_aligned - recon_aligned) ** 2)
            correlation = np.corrcoef(audio_aligned, recon_aligned)[0, 1]
            signal_power = np.mean(audio_aligned ** 2)
            noise_power = np.mean((audio_aligned - recon_aligned) ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            print(f"   结果: MSE={mse:.6f}, SNR={snr:.2f}dB, 相关性={correlation:.4f}")
            
            # 保存结果
            timestamp = int(time.time())
            output_path = output_dir / f"{config['name']}_snr{snr:.1f}_{timestamp}.wav"
            save_audio_compatible(reconstructed_audio, str(output_path))
            
            results.append({
                'config': config['name'],
                'mse': mse,
                'snr': snr,
                'correlation': correlation,
                'output_path': output_path
            })
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            results.append({
                'config': config['name'],
                'mse': float('inf'),
                'snr': float('-inf'),
                'correlation': 0,
                'output_path': None
            })
    
    # 分析结果
    print(f"\n📊 测试结果总结:")
    print(f"{'配置':<20} {'SNR(dB)':<10} {'MSE':<12} {'相关性':<10}")
    print("-" * 55)
    
    best_snr = float('-inf')
    best_config = None
    
    for result in results:
        snr_str = f"{result['snr']:.2f}" if result['snr'] != float('-inf') else "FAIL"
        mse_str = f"{result['mse']:.6f}" if result['mse'] != float('inf') else "FAIL"
        corr_str = f"{result['correlation']:.4f}" if result['correlation'] != 0 else "FAIL"
        
        print(f"{result['config']:<20} {snr_str:<10} {mse_str:<12} {corr_str:<10}")
        
        if result['snr'] > best_snr:
            best_snr = result['snr']
            best_config = result
    
    print(f"\n🏆 最佳配置:")
    print(f"   配置: {best_config['config']}")
    print(f"   SNR: {best_config['snr']:.2f} dB")
    print(f"   MSE: {best_config['mse']:.6f}")
    print(f"   相关性: {best_config['correlation']:.4f}")
    
    if best_config['output_path']:
        print(f"   输出: {best_config['output_path']}")
    
    # 创建详细报告
    report = f"""
AudioLDM2 Mel频谱预处理参数测试报告
================================

测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
输入音频: {Path(audio_path).name}
音频长度: {len(audio)/sr:.2f}秒

测试结果:
"""
    
    for result in results:
        report += f"""
{result['config']}:
  SNR: {result['snr']:.2f} dB
  MSE: {result['mse']:.6f}
  相关性: {result['correlation']:.4f}
  输出: {result['output_path']}
"""
    
    report += f"""
最佳配置: {best_config['config']}
最佳SNR: {best_config['snr']:.2f} dB

结论:
"""
    
    if best_snr > -5:
        report += "- 找到了相对较好的预处理参数\n"
    elif best_snr > -10:
        report += "- 预处理参数有所改善，但仍需优化\n"
    else:
        report += "- 预处理参数改善有限，可能需要更深层的修复\n"
    
    report += "- 建议进一步研究AudioLDM2的训练代码和论文\n"
    report += "- 考虑使用AudioLDM2的内置预处理方法\n"
    
    report_path = output_dir / f"mel_preprocessing_report_{int(time.time())}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 详细报告: {report_path}")
    
    return results, best_config


def main():
    """主函数"""
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path('.').glob(ext))
    
    if not audio_files:
        print("❌ 没有找到音频文件")
        return
    
    print("找到音频文件:")
    for i, file in enumerate(audio_files, 1):
        print(f"{i}. {file.name}")
    
    try:
        choice = int(input("选择文件:"))
        audio_path = str(audio_files[choice - 1])
        
        results, best_config = test_mel_preprocessing_variants(audio_path)
        
        print(f"\n🎉 测试完成！")
        print(f"📈 最佳配置: {best_config['config']}")
        print(f"📊 最佳SNR: {best_config['snr']:.2f} dB")
        
        if best_config['snr'] > -5:
            print(f"🏆 取得了显著改善！")
        elif best_config['snr'] > -10:
            print(f"✅ 有所改善，建议进一步优化")
        else:
            print(f"⚠️ 改善有限，需要更深层的修复")
            
    except (ValueError, IndexError):
        print("❌ 无效选择")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
