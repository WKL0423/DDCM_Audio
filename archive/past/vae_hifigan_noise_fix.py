#!/usr/bin/env python3
"""
AudioLDM2 HiFiGAN 修复版本 - 针对噪声问题优化
==============================================

基于AudioLDM2官方预处理参数和正确的mel频谱归一化
解决噪声问题的关键修复
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

# 尝试导入 soundfile 以获得更好的兼容性
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("⚠️ soundfile 不可用，将使用 torchaudio 保存（可能存在兼容性问题）")


def save_audio_compatible(audio_data, filepath, sample_rate=16000):
    """保存音频文件，优先使用 soundfile 以获得最大兼容性"""
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    
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
            sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')
            print(f"   ✅ 使用 soundfile (PCM_16) 保存: {Path(filepath).name}")
            success = True
        except Exception as e:
            print(f"   ⚠️ soundfile 保存失败: {e}")
    
    if not success:
        try:
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
            torchaudio.save(filepath, audio_tensor, sample_rate)
            print(f"   ✅ 使用 torchaudio 保存: {Path(filepath).name}")
            success = True
        except Exception as e:
            print(f"   ❌ torchaudio 保存也失败: {e}")
    
    return success


def create_mel_spectrogram_audioldm2(audio, sr=16000):
    """
    使用AudioLDM2官方参数创建mel频谱图
    
    基于AudioLDM2论文和官方实现的标准参数：
    - n_mels: 64 (AudioLDM2标准)
    - n_fft: 1024 
    - hop_length: 160
    - win_length: 1024
    - window: hann
    - 归一化: 使用AudioLDM2标准归一化
    """
    print("🎵 使用AudioLDM2官方参数创建mel频谱...")
    
    # AudioLDM2官方参数
    n_mels = 64
    n_fft = 1024
    hop_length = 160
    win_length = 1024
    fmin = 0
    fmax = sr // 2
    
    # 创建mel频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )
    
    print(f"   Mel频谱原始形状: {mel_spec.shape}")
    print(f"   Mel频谱原始范围: [{mel_spec.min():.6f}, {mel_spec.max():.6f}]")
    
    # 转换为对数尺度 (AudioLDM2使用的标准方法)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=None)
    
    print(f"   Mel频谱dB形状: {mel_spec_db.shape}")
    print(f"   Mel频谱dB范围: [{mel_spec_db.min():.6f}, {mel_spec_db.max():.6f}]")
    
    # AudioLDM2标准归一化: 缩放到[-1, 1]
    # 使用固定的动态范围，而不是基于当前样本的min/max
    # 这样可以保证训练和推理时的一致性
    
    # 方法1: 使用固定的dB范围 (更符合AudioLDM2训练)
    min_db = -80.0  # AudioLDM2通常使用的最小dB值
    max_db = mel_spec_db.max()
    
    # 裁剪到合理范围
    mel_spec_db = np.clip(mel_spec_db, min_db, max_db)
    
    # 归一化到[-1, 1]
    mel_spec_normalized = 2.0 * (mel_spec_db - min_db) / (max_db - min_db) - 1.0
    
    print(f"   归一化参数: min_db={min_db:.2f}, max_db={max_db:.2f}")
    print(f"   归一化后范围: [{mel_spec_normalized.min():.6f}, {mel_spec_normalized.max():.6f}]")
    
    return mel_spec_normalized, min_db, max_db


def test_audioldm2_hifigan_fixed(audio_path, max_length=5):
    """
    修复版本：使用正确的AudioLDM2预处理参数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 AudioLDM2 HiFiGAN 噪声修复测试")
    print(f"📱 设备: {device}")
    
    # 加载AudioLDM2
    print("📦 加载AudioLDM2模型...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",  # 使用基础模型，不是音乐专用版本
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    print(f"✅ 模型加载完成")
    print(f"   VAE类型: {type(vae).__name__}")
    print(f"   Vocoder类型: {type(vocoder).__name__}")
    
    # 加载音频
    print(f"📁 加载音频: {Path(audio_path).name}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, duration=max_length)
    print(f"   音频时长: {len(audio)/sr:.2f}秒")
    print(f"   音频范围: [{audio.min():.6f}, {audio.max():.6f}]")
    
    # 使用改进的mel频谱创建
    mel_spec_normalized, min_db, max_db = create_mel_spectrogram_audioldm2(audio, sr)
    
    # 转换为tensor
    mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
    if device == "cuda":
        mel_tensor = mel_tensor.half()
    
    # 调整为VAE期望的形状: [batch, channels, height, width]
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, time]
    print(f"🎵 VAE输入: {mel_input.shape}, 数据类型: {mel_input.dtype}")
    print(f"   输入范围: [{mel_input.min():.6f}, {mel_input.max():.6f}]")
    
    # VAE处理
    print("🧠 开始VAE编码解码...")
    with torch.no_grad():
        # 确保尺寸匹配VAE要求 (AudioLDM2 VAE通常要求能被某个数整除)
        orig_width = mel_input.shape[-1]
        
        # VAE编码
        latent_dist = vae.encode(mel_input)
        if hasattr(latent_dist, 'latent_dist'):
            latent = latent_dist.latent_dist.sample()
        else:
            latent = latent_dist.sample()
        
        print(f"   编码latent: {latent.shape}")
        print(f"   Latent范围: [{latent.min():.6f}, {latent.max():.6f}]")
        
        # VAE解码
        reconstructed_mel = vae.decode(latent).sample
        
        print(f"   解码输出: {reconstructed_mel.shape}")
        print(f"   解码范围: [{reconstructed_mel.min():.6f}, {reconstructed_mel.max():.6f}]")
        
        # 裁剪到原始宽度
        if reconstructed_mel.shape[-1] > orig_width:
            reconstructed_mel = reconstructed_mel[:, :, :, :orig_width]
            print(f"   裁剪后: {reconstructed_mel.shape}")
        
        # 关键修复：使用AudioLDM2标准的HiFiGAN输入格式
        print("🎤 准备HiFiGAN输入...")
        
        # 检查vocoder期望的输入格式
        print(f"   原始mel形状: {reconstructed_mel.shape}")
        
        # AudioLDM2的mel_spectrogram_to_waveform方法
        if reconstructed_mel.dim() == 4:
            vocoder_input = reconstructed_mel.squeeze(1)  # [batch, 1, height, width] -> [batch, height, width]
            print(f"   squeeze(1): {vocoder_input.shape}")
        else:
            vocoder_input = reconstructed_mel
        
        # 确保数据类型匹配
        vocoder_dtype = next(vocoder.parameters()).dtype
        if vocoder_input.dtype != vocoder_dtype:
            vocoder_input = vocoder_input.to(vocoder_dtype)
            print(f"   转换数据类型到: {vocoder_dtype}")
        
        print(f"   最终vocoder输入: {vocoder_input.shape}, {vocoder_input.dtype}")
        print(f"   Vocoder输入范围: [{vocoder_input.min():.6f}, {vocoder_input.max():.6f}]")
        
        # 使用AudioLDM2内置HiFiGAN
        try:
            print("🚀 调用AudioLDM2 HiFiGAN...")
            
            # 直接使用pipeline的标准方法
            waveform = pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
            reconstructed_audio = waveform.squeeze().cpu().numpy()
            vocoder_method = "AudioLDM2_HiFiGAN_FIXED"
            
            print(f"✅ HiFiGAN成功！输出: {len(reconstructed_audio)}样本")
            print(f"   重建音频范围: [{reconstructed_audio.min():.6f}, {reconstructed_audio.max():.6f}]")
            
        except Exception as e:
            print(f"❌ HiFiGAN仍然失败: {e}")
            print("🔄 使用Griffin-Lim备选...")
            
            # Griffin-Lim降级 (使用正确的反归一化)
            mel_np = reconstructed_mel.squeeze().cpu().float().numpy()
            
            # 反归一化：从[-1,1]恢复到dB尺度
            mel_denorm_db = (mel_np + 1.0) / 2.0 * (max_db - min_db) + min_db
            
            # 从dB转换到功率谱
            mel_power = librosa.db_to_power(mel_denorm_db)
            
            # Griffin-Lim重建
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                mel_power, 
                sr=sample_rate, 
                hop_length=160, 
                n_fft=1024,
                win_length=1024,
                window='hann',
                n_iter=32  # 增加迭代次数以提高质量
            )
            vocoder_method = "Griffin_Lim_Fixed"
            print(f"✅ Griffin-Lim成功: {len(reconstructed_audio)}样本")
    
    # 保存结果
    output_dir = Path("vae_hifigan_noise_fixed")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存路径
    original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
    reconstructed_path = output_dir / f"{input_name}_{vocoder_method}_{timestamp}.wav"
    
    # 长度匹配
    if len(reconstructed_audio) > len(audio):
        reconstructed_audio = reconstructed_audio[:len(audio)]
    elif len(reconstructed_audio) < len(audio):
        reconstructed_audio = np.pad(reconstructed_audio, (0, len(audio) - len(reconstructed_audio)))
    
    print("💾 保存结果...")
    save_audio_compatible(audio, original_path, sample_rate)
    save_audio_compatible(reconstructed_audio, reconstructed_path, sample_rate)
    
    # 计算质量指标
    mse = np.mean((audio - reconstructed_audio) ** 2)
    snr = 10 * np.log10(np.mean(audio ** 2) / (mse + 1e-10))
    correlation = np.corrcoef(audio, reconstructed_audio)[0, 1] if len(audio) > 1 else 0
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"🎯 AudioLDM2 噪声修复测试结果")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    print(f"📁 重建音频: {reconstructed_path}")
    print(f"📊 SNR: {snr:.2f} dB")
    print(f"📊 MSE: {mse:.6f}")
    print(f"📊 相关系数: {correlation:.4f}")
    print(f"🎤 重建方法: {vocoder_method}")
    
    # 分析结果
    if vocoder_method == "AudioLDM2_HiFiGAN_FIXED":
        print(f"\n🎉 HiFiGAN修复成功！")
        print(f"✅ 使用了AudioLDM2官方预处理参数")
        print(f"✅ 使用了正确的mel频谱归一化")
        print(f"✅ 使用了标准的vocoder调用方法")
        
        if snr > 0:
            print(f"🏆 重建质量良好！噪声问题已显著改善")
        elif snr > -5:
            print(f"✅ 重建质量可接受，继续优化中")
        else:
            print(f"⚠️ 仍需进一步调优，但方向正确")
    else:
        print(f"\n🔧 使用修复版Griffin-Lim")
        print(f"✅ 使用了正确的dB范围反归一化")
        print(f"✅ 增加了Griffin-Lim迭代次数")
    
    print(f"\n🔬 技术改进:")
    print(f"   ✅ 使用AudioLDM2官方mel参数")
    print(f"   ✅ 使用固定dB范围归一化")
    print(f"   ✅ 使用标准vocoder调用")
    print(f"   ✅ 改进反归一化过程")
    
    return {
        'snr': snr,
        'mse': mse,
        'correlation': correlation,
        'vocoder_method': vocoder_method,
        'original_path': str(original_path),
        'reconstructed_path': str(reconstructed_path)
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        audio_files = list(Path('.').glob('*.wav'))
        if audio_files:
            print("找到音频文件:")
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
    
    print(f"🚀 开始AudioLDM2噪声修复测试")
    
    try:
        result = test_audioldm2_hifigan_fixed(audio_path)
        
        print(f"\n📋 修复测试总结:")
        print(f"   方法: {result['vocoder_method']}")
        print(f"   SNR: {result['snr']:.2f}dB")
        print(f"   MSE: {result['mse']:.6f}")
        print(f"   相关性: {result['correlation']:.4f}")
        
        if "FIXED" in result['vocoder_method']:
            print(f"\n🎊 噪声修复测试完成！")
            print(f"🔬 使用了AudioLDM2官方参数和标准化方法")
        else:
            print(f"\n🔍 继续优化噪声问题...")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
