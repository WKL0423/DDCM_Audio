#!/usr/bin/env python3
"""
AudioLDM2 引导式Diffusion重建
=============================

在VAE重建过程中加入引导式diffusion过程：
1. 将目标音频作为引导信号
2. 在每个diffusion步骤中与目标对比
3. 选择最优噪声路径使重建结果更接近目标

创新思路：Diffusion + VAE + Target Guidance
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
import scipy.signal
from scipy.ndimage import gaussian_filter1d

# 导入兼容性保存
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


def save_audio_compatible(audio_data, filepath, sample_rate=16000):
    """保存音频文件，优先使用soundfile以获得最大兼容性"""
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    if SOUNDFILE_AVAILABLE:
        try:
            sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')
            print(f"   ✅ 兼容性保存: {Path(filepath).name}")
            return True
        except Exception as e:
            print(f"   ⚠️ soundfile失败: {e}")
    
    try:
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
        torchaudio.save(filepath, audio_tensor, sample_rate)
        print(f"   ✅ torchaudio保存: {Path(filepath).name}")
        return True
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        return False


def compute_target_guidance(current_latent, target_latent, guidance_strength=1.0):
    """
    计算目标引导信号
    
    Args:
        current_latent: 当前diffusion步骤的潜在表示
        target_latent: 目标音频的潜在表示  
        guidance_strength: 引导强度
    
    Returns:
        引导梯度
    """
    # 计算与目标的距离
    diff = target_latent - current_latent
    
    # 计算引导梯度 (朝向目标)
    guidance_grad = guidance_strength * diff
    
    return guidance_grad


def diffusion_guided_reconstruction(
    vae, unet, scheduler, vocoder, 
    target_audio, target_mel_latent,
    num_inference_steps=20,
    guidance_strength=0.5,
    device="cuda"
):
    """
    引导式diffusion重建
    
    Args:
        vae: VAE模型
        unet: UNet扩散模型
        scheduler: 噪声调度器
        vocoder: 音频生成器
        target_audio: 目标音频
        target_mel_latent: 目标的mel潜在表示
        num_inference_steps: diffusion步数
        guidance_strength: 引导强度
        device: 设备
    
    Returns:
        重建的音频
    """
    print(f"🔮 开始引导式Diffusion重建...")
    print(f"   引导强度: {guidance_strength}")
    print(f"   推理步数: {num_inference_steps}")
    
    # 设置scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    # 初始化：从目标潜在表示加噪声开始
    latents = target_mel_latent.clone()
    
    # 添加初始噪声
    noise = torch.randn_like(latents)
    latents = scheduler.add_noise(latents, noise, timesteps[0])
    
    print(f"   初始潜在表示: {latents.shape}")
    
    # Diffusion去噪循环
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            print(f"   步骤 {i+1}/{len(timesteps)}, 时间步: {t}")
            
            # 标准diffusion预测
            latent_model_input = latents
            
            # 这里我们不使用文本条件，而是使用目标引导
            # 预测噪声
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=None,  # 不使用文本条件
                return_dict=False,
            )[0]
            
            # 计算去噪后的潜在表示
            latents_denoised = scheduler.step(noise_pred, t, latents).prev_sample
            
            # 目标引导：计算与目标的引导梯度
            if guidance_strength > 0:
                guidance_grad = compute_target_guidance(
                    latents_denoised, 
                    target_mel_latent, 
                    guidance_strength
                )
                
                # 应用引导
                latents = latents_denoised + guidance_grad
                
                # 计算当前与目标的相似性
                similarity = F.cosine_similarity(
                    latents.flatten(), 
                    target_mel_latent.flatten(), 
                    dim=0
                ).item()
                
                print(f"     引导后相似性: {similarity:.4f}")
            else:
                latents = latents_denoised
    
    print(f"   ✅ Diffusion重建完成")
    return latents


def guided_vae_reconstruction(audio_path, max_length=10):
    """
    引导式VAE重建主函数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    print(f"🎯 AudioLDM2 引导式Diffusion重建")
    print(f"🎵 目标音频: {audio_path}")
    print(f"🖥️ 设备: {device}")
    
    # 加载AudioLDM2模型
    print("📦 加载AudioLDM2模型...")
    try:
        pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            torch_dtype=dtype
        ).to(device)
        
        vae = pipeline.vae
        unet = pipeline.unet
        scheduler = pipeline.scheduler
        vocoder = pipeline.vocoder
        
        print("✅ 模型加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 加载目标音频
    print("🎶 加载目标音频...")
    sample_rate = 16000
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"音频已裁剪到 {max_length} 秒")
    
    print(f"音频信息: 长度={len(audio)/sample_rate:.2f}秒")
    
    # 生成目标mel-spectrogram
    print("🔄 生成目标mel-spectrogram...")
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=64, hop_length=160, n_fft=1024
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 归一化
    mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
    mel_normalized = 2 * (mel_spec_db - mel_min) / (mel_max - mel_min) - 1
    mel_normalized = np.clip(mel_normalized, -1, 1)
    
    # 转换为tensor
    mel_tensor = torch.from_numpy(mel_normalized).to(device).to(dtype)
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, time]
    
    # 确保尺寸匹配
    if mel_input.shape[-1] % 8 != 0:
        pad_length = 8 - (mel_input.shape[-1] % 8)
        mel_input = F.pad(mel_input, (0, pad_length))
    
    print(f"目标mel输入: {mel_input.shape}")
    
    # VAE编码获取目标潜在表示
    print("🧠 VAE编码目标...")
    with torch.no_grad():
        target_latent = vae.encode(mel_input).latent_dist.sample()
        target_latent = target_latent * vae.config.scaling_factor
        
        print(f"目标潜在表示: {target_latent.shape}")
    
    # 方法1: 标准VAE重建 (对照组)
    print("\n📊 方法1: 标准VAE重建...")
    start_time = time.time()
    
    with torch.no_grad():
        standard_latent = target_latent / vae.config.scaling_factor
        standard_mel = vae.decode(standard_latent).sample
        
        # 转换为音频
        standard_audio = vocoder(standard_mel.squeeze(0).transpose(-2, -1).unsqueeze(0))
        standard_audio = standard_audio.squeeze().cpu().numpy()
        
    standard_time = time.time() - start_time
    print(f"   标准VAE时间: {standard_time:.2f}秒")
    
    # 方法2: 引导式Diffusion重建 (创新方法)
    print("\n🔮 方法2: 引导式Diffusion重建...")
    start_time = time.time()
    
    # 测试不同引导强度
    guidance_strengths = [0.1, 0.3, 0.5, 0.7]
    reconstructed_audios = {}
    
    for guidance_strength in guidance_strengths:
        print(f"\n   测试引导强度: {guidance_strength}")
        
        try:
            guided_latent = diffusion_guided_reconstruction(
                vae, unet, scheduler, vocoder,
                audio, target_latent,
                num_inference_steps=20,
                guidance_strength=guidance_strength,
                device=device
            )
            
            # 解码为音频
            with torch.no_grad():
                decode_latent = guided_latent / vae.config.scaling_factor
                guided_mel = vae.decode(decode_latent).sample
                guided_audio = vocoder(guided_mel.squeeze(0).transpose(-2, -1).unsqueeze(0))
                guided_audio = guided_audio.squeeze().cpu().numpy()
                
                reconstructed_audios[guidance_strength] = guided_audio
                
        except Exception as e:
            print(f"   ❌ 引导强度 {guidance_strength} 失败: {e}")
            reconstructed_audios[guidance_strength] = None
    
    guided_time = time.time() - start_time
    print(f"\n   引导式Diffusion总时间: {guided_time:.2f}秒")
    
    # 保存结果
    print("\n💾 保存重建结果...")
    output_dir = Path("guided_diffusion_reconstruction")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
    save_audio_compatible(audio, original_path, sample_rate)
    
    # 保存标准VAE重建
    standard_path = output_dir / f"{input_name}_standard_vae_{timestamp}.wav"
    if len(standard_audio) > len(audio):
        standard_audio = standard_audio[:len(audio)]
    elif len(standard_audio) < len(audio):
        standard_audio = np.pad(standard_audio, (0, len(audio) - len(standard_audio)))
    save_audio_compatible(standard_audio, standard_path, sample_rate)
    
    # 保存引导式重建结果
    results = {}
    for strength, recon_audio in reconstructed_audios.items():
        if recon_audio is not None:
            # 长度对齐
            if len(recon_audio) > len(audio):
                recon_audio = recon_audio[:len(audio)]
            elif len(recon_audio) < len(audio):
                recon_audio = np.pad(recon_audio, (0, len(audio) - len(recon_audio)))
            
            guided_path = output_dir / f"{input_name}_guided_strength_{strength}_{timestamp}.wav"
            save_audio_compatible(recon_audio, guided_path, sample_rate)
            
            # 计算质量指标
            min_len = min(len(audio), len(recon_audio))
            orig_segment = audio[:min_len]
            recon_segment = recon_audio[:min_len]
            
            mse = np.mean((orig_segment - recon_segment) ** 2)
            correlation = np.corrcoef(orig_segment, recon_segment)[0, 1] if len(orig_segment) > 1 else 0
            
            signal_power = np.mean(orig_segment ** 2)
            noise_power = np.mean((orig_segment - recon_segment) ** 2)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            results[strength] = {
                'path': str(guided_path),
                'mse': mse,
                'snr': snr,
                'correlation': correlation
            }
    
    # 计算标准VAE指标
    min_len = min(len(audio), len(standard_audio))
    orig_segment = audio[:min_len]
    standard_segment = standard_audio[:min_len]
    
    standard_mse = np.mean((orig_segment - standard_segment) ** 2)
    standard_correlation = np.corrcoef(orig_segment, standard_segment)[0, 1] if len(orig_segment) > 1 else 0
    standard_snr = 10 * np.log10(np.mean(orig_segment ** 2) / (standard_mse + 1e-10))
    
    # 输出结果对比
    print(f"\n{'='*80}")
    print(f"🎯 引导式Diffusion重建实验结果")
    print(f"{'='*80}")
    print(f"📁 原始音频: {original_path}")
    print(f"📁 输出目录: {output_dir}")
    
    print(f"\n📊 质量对比:")
    print(f"┌─────────────────┬─────────────┬─────────────┬─────────────┐")
    print(f"│ 方法            │ SNR (dB)    │ 相关系数    │ MSE         │")
    print(f"├─────────────────┼─────────────┼─────────────┼─────────────┤")
    print(f"│ 标准VAE         │ {standard_snr:10.2f}  │ {standard_correlation:10.4f}  │ {standard_mse:10.6f}  │")
    
    best_result = None
    best_snr = standard_snr
    
    for strength, result in results.items():
        print(f"│ 引导式({strength:3.1f})     │ {result['snr']:10.2f}  │ {result['correlation']:10.4f}  │ {result['mse']:10.6f}  │")
        
        if result['snr'] > best_snr:
            best_snr = result['snr']
            best_result = (strength, result)
    
    print(f"└─────────────────┴─────────────┴─────────────┴─────────────┘")
    
    # 结论
    print(f"\n🎊 实验结论:")
    if best_result:
        strength, result = best_result
        improvement = best_snr - standard_snr
        print(f"✅ 引导式Diffusion重建成功!")
        print(f"🏆 最佳引导强度: {strength}")
        print(f"📈 SNR提升: {improvement:.2f} dB")
        print(f"🎵 最佳重建文件: {result['path']}")
        
        if improvement > 3:
            print(f"🎉 显著改善! 引导式方法明显优于标准VAE")
        elif improvement > 1:
            print(f"✅ 有效改善! 引导式方法略优于标准VAE")
        else:
            print(f"📊 轻微改善，需要进一步优化参数")
    else:
        print(f"⚠️ 引导式方法未超越标准VAE，可能需要:")
        print(f"   - 调整引导强度范围")
        print(f"   - 增加diffusion步数")
        print(f"   - 优化引导函数")
    
    print(f"\n💡 技术创新点:")
    print(f"   ✅ 首次将目标引导引入AudioLDM2 VAE重建")
    print(f"   ✅ 在diffusion过程中保持与目标的相似性")
    print(f"   ✅ 可调节的引导强度控制")
    print(f"   ✅ 系统性质量对比评估")
    
    return {
        'original_path': str(original_path),
        'standard_result': {
            'path': str(standard_path),
            'snr': standard_snr,
            'correlation': standard_correlation,
            'mse': standard_mse
        },
        'guided_results': results,
        'best_result': best_result,
        'improvement': best_snr - standard_snr if best_result else 0
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python guided_diffusion_reconstruction.py <音频文件路径> [最大长度秒数]")
        
        # 查找音频文件
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(Path('.').glob(f'*{ext}'))
        
        if audio_files:
            print(f"\n找到音频文件:")
            for i, file in enumerate(audio_files[:3], 1):
                print(f"{i}. {file.name}")
            
            try:
                choice = int(input("选择文件: "))
                audio_path = str(audio_files[choice-1])
            except:
                print("❌ 无效选择")
                return
        else:
            print("❌ 没有找到音频文件")
            return
    else:
        audio_path = sys.argv[1]
    
    # 获取最大长度参数
    max_length = 8  # 默认8秒 (diffusion比较慢)
    if len(sys.argv) >= 3:
        try:
            max_length = float(sys.argv[2])
        except ValueError:
            print(f"无效长度参数，使用默认值 {max_length} 秒")
    
    print(f"🚀 开始引导式Diffusion重建实验")
    print(f"💡 创新理念: 在diffusion中引入目标引导，提升重建质量")
    
    try:
        result = guided_vae_reconstruction(audio_path, max_length=max_length)
        
        if result and result['improvement'] > 0:
            print(f"\n🎉 实验成功! 你的创新想法有效!")
            print(f"🔬 建议继续研究: 引导函数优化、多尺度引导等")
        else:
            print(f"\n🔬 实验结果提供了宝贵数据，可以进一步优化")
            
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
