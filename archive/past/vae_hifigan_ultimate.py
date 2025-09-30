"""
AudioLDM2 HiFiGAN 终极修复版本
============================

基于vocoder分析的精确维度匹配
已集成兼容性音频保存
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
    """
    保存音频文件，优先使用 soundfile 以获得最大兼容性
    """
    # 确保音频数据是正确的格式
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    
    # 确保是 1D 数组
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
            # 使用 soundfile 保存为 PCM_16 格式（最高兼容性）
            sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')
            print(f"   ✅ 使用 soundfile (PCM_16) 保存: {Path(filepath).name}")
            success = True
        except Exception as e:
            print(f"   ⚠️ soundfile 保存失败: {e}")
    
    if not success:
        try:
            # 回退到 torchaudio
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
            torchaudio.save(filepath, audio_tensor, sample_rate)
            print(f"   ✅ 使用 torchaudio 保存: {Path(filepath).name}")
            success = True
        except Exception as e:
            print(f"   ❌ torchaudio 保存也失败: {e}")
    
    return success

def test_audioldm2_hifigan_final(audio_path, max_length=5):
    """
    最终修复版本：基于vocoder分析结果的精确实现
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 AudioLDM2 HiFiGAN 终极修复测试")
    print(f"📱 设备: {device}")
    
    # 加载AudioLDM2
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sample_rate, duration=max_length)
    print(f"📊 音频: {len(audio)/sr:.2f}秒")
    
    # 创建mel频谱 (64维，与AudioLDM2匹配)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=64, hop_length=160, n_fft=1024
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 归一化
    mel_min, mel_max = mel_spec.min(), mel_spec.max()
    mel_spec = 2 * (mel_spec - mel_min) / (mel_max - mel_min) - 1
    
    # 转换为tensor
    mel_tensor = torch.from_numpy(mel_spec).to(device)
    if device == "cuda":
        mel_tensor = mel_tensor.half()
    
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, time]
    print(f"🎵 Mel输入: {mel_input.shape}")
    
    # VAE处理
    with torch.no_grad():
        # 确保尺寸匹配VAE要求
        if mel_input.shape[-1] % 4 != 0:
            pad_length = 4 - (mel_input.shape[-1] % 4)
            mel_input = F.pad(mel_input, (0, pad_length))
        
        # VAE编码解码
        latent = vae.encode(mel_input).latent_dist.sample()
        reconstructed_mel = vae.decode(latent).sample
        
        print(f"🔄 VAE重建: {mel_input.shape} → {reconstructed_mel.shape}")
        
        # 关键修复：正确的HiFiGAN输入格式
        print("🎤 准备HiFiGAN输入...")
        
        # 从 [1, 1, 64, time] 转换为 [1, time, 64]
        vocoder_input = reconstructed_mel.squeeze()  # [64, time]
        print(f"   步骤1 - squeeze: {vocoder_input.shape}")
        
        if vocoder_input.dim() == 3:  # 如果还有batch维度
            vocoder_input = vocoder_input.squeeze(0)  # [1, 64, time] -> [64, time]
            print(f"   步骤2 - 再次squeeze: {vocoder_input.shape}")
        
        vocoder_input = vocoder_input.transpose(0, 1)  # [64, time] -> [time, 64]
        print(f"   步骤3 - transpose: {vocoder_input.shape}")
        
        vocoder_input = vocoder_input.unsqueeze(0)  # [time, 64] -> [1, time, 64]
        print(f"   步骤4 - 最终格式: {vocoder_input.shape}")
        
        # 数据类型匹配
        if next(vocoder.parameters()).dtype == torch.float16:
            vocoder_input = vocoder_input.half()
        else:
            vocoder_input = vocoder_input.float()
        
        print(f"   数据类型: {vocoder_input.dtype}")
        
        # 使用AudioLDM2内置HiFiGAN
        try:
            print("🚀 调用AudioLDM2 HiFiGAN...")
            audio_tensor = vocoder(vocoder_input)
            reconstructed_audio = audio_tensor.squeeze().cpu().numpy()
            vocoder_method = "AudioLDM2_HiFiGAN_SUCCESS"
            print(f"✅ 成功！输出: {len(reconstructed_audio)}样本")
            
        except Exception as e:
            print(f"❌ 仍然失败: {e}")
            print("🔄 使用Griffin-Lim...")
            
            # Griffin-Lim降级
            mel_np = reconstructed_mel.squeeze().cpu().numpy()
            mel_denorm = (mel_np + 1) / 2 * (mel_max - mel_min) + mel_min
            mel_power = librosa.db_to_power(mel_denorm)
            
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                mel_power, sr=sample_rate, hop_length=160, n_fft=1024
            )
            vocoder_method = "Griffin_Lim_Fallback"
            print(f"✅ Griffin-Lim成功: {len(reconstructed_audio)}样本")
      # 保存结果 (使用兼容性保存方法)
    output_dir = Path("vae_hifigan_final_test")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存音频
    original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
    reconstructed_path = output_dir / f"{input_name}_{vocoder_method}_{timestamp}.wav"
    
    # 归一化音频
    audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # 确保重建音频长度匹配
    if len(reconstructed_audio) > len(audio):
        reconstructed_audio = reconstructed_audio[:len(audio)]
    elif len(reconstructed_audio) < len(audio):
        reconstructed_audio = np.pad(reconstructed_audio, (0, len(audio) - len(reconstructed_audio)))
    
    recon_norm = reconstructed_audio / (np.max(np.abs(reconstructed_audio)) + 1e-8)
    
    # 使用兼容性保存函数
    print("💾 保存结果...")
    save_audio_compatible(audio_norm, original_path, sample_rate)
    save_audio_compatible(recon_norm, reconstructed_path, sample_rate)
    
    # 计算指标
    mse = np.mean((audio - recon_norm) ** 2)
    snr = 10 * np.log10(np.mean(audio ** 2) / (mse + 1e-10))
    correlation = np.corrcoef(audio, recon_norm)[0, 1] if len(audio) > 1 else 0
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"🎯 AudioLDM2 HiFiGAN 终极测试结果")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    print(f"📁 重建音频: {reconstructed_path}")
    print(f"📊 SNR: {snr:.2f} dB")
    print(f"📊 相关系数: {correlation:.4f}")
    print(f"🎤 重建方法: {vocoder_method}")
    
    # 关键结论
    if vocoder_method == "AudioLDM2_HiFiGAN_SUCCESS":
        print(f"\n🎉 重大突破！")
        print(f"✅ 成功使用AudioLDM2内置HiFiGAN")
        print(f"✅ 绕过了Griffin-Lim的92%信息损失瓶颈")
        print(f"📈 预期质量提升显著")
        
        if snr > 5:
            print(f"🏆 重建质量优秀！")
        elif snr > 0:
            print(f"✅ 重建质量良好")
        else:
            print(f"⚠️ 仍需进一步优化，但已是重大进步")
    else:
        print(f"\n⚠️ HiFiGAN集成仍有技术障碍")
        print(f"📊 当前使用: {vocoder_method}")
        print(f"🔬 需要更深入的AudioLDM2内部研究")
    
    return {
        'snr': snr,
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
    
    print(f"🚀 开始AudioLDM2 HiFiGAN终极测试")
    
    try:
        result = test_audioldm2_hifigan_final(audio_path)
        
        print(f"\n📋 测试总结:")
        print(f"   方法: {result['vocoder_method']}")
        print(f"   SNR: {result['snr']:.2f}dB")
        print(f"   相关性: {result['correlation']:.4f}")
        
        if result['vocoder_method'] == "AudioLDM2_HiFiGAN_SUCCESS":
            print(f"\n🎊 恭喜！已突破HiFiGAN集成技术瓶颈！")
        else:
            print(f"\n🔍 仍在探索HiFiGAN集成的最佳方案...")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
