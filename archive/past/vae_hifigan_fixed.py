"""
AudioLDM2 VAE 修复版本 - 使用内置HiFiGAN
=============================================

正确使用AudioLDM2内置的HiFiGAN vocoder
解决维度不匹配问题
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

def load_and_test_vae_fixed(audio_path, model_id="cvssp/audioldm2-music", max_length=10):
    """
    修复版本：正确使用AudioLDM2内置HiFiGAN
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 使用设备: {device}")
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到音频文件 {audio_path}")
        return
    
    print(f"📦 加载 AudioLDM2 模型: {model_id}")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder  # 这是AudioLDM2内置的HiFiGAN
    sample_rate = 16000
    
    print(f"🎤 Vocoder类型: {type(vocoder)}")
    print(f"📊 加载音频: {audio_path}")
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"✂️ 音频裁剪到 {max_length} 秒")
    
    print(f"📈 音频信息: {len(audio)/sample_rate:.2f}秒, {len(audio)}样本")
    
    # 创建mel频谱图
    print("🎼 创建mel频谱图...")
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=64,  # AudioLDM2使用64维
        hop_length=160,
        n_fft=1024,
        fmin=0,
        fmax=8000
    )
    
    # 转换为对数尺度并归一化
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    mel_spec_normalized = 2 * (mel_spec_db - mel_min) / (mel_max - mel_min) - 1
    mel_spec_normalized = np.clip(mel_spec_normalized, -1, 1)
    
    # 保存归一化参数
    norm_params = {'min_val': mel_min, 'max_val': mel_max}
    
    # 转换为tensor
    mel_tensor = torch.from_numpy(mel_spec_normalized).to(device)
    if device == "cuda":
        mel_tensor = mel_tensor.to(torch.float16)
    else:
        mel_tensor = mel_tensor.to(torch.float32)
    
    # 调整为VAE格式: [batch, channels, height, width]
    mel_input = mel_tensor.unsqueeze(0).unsqueeze(0)
    print(f"🎵 Mel频谱形状: {mel_input.shape}")
    
    # VAE编码解码
    print("🔄 VAE编码解码...")
    start_time = time.time()
    
    with torch.no_grad():
        try:
            # 编码
            latents = vae.encode(mel_input).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            encode_time = time.time() - start_time
            print(f"✅ 编码完成: {encode_time:.3f}秒, 潜在形状: {latents.shape}")
            
            # 解码
            decode_start = time.time()
            latents = latents / vae.config.scaling_factor
            reconstructed_mel = vae.decode(latents).sample
            decode_time = time.time() - decode_start
            print(f"✅ 解码完成: {decode_time:.3f}秒, 重建形状: {reconstructed_mel.shape}")
            
        except Exception as e:
            print(f"❌ VAE失败: {e}")
            # 尝试调整尺寸
            pad_width = (8 - (mel_input.shape[-1] % 8)) % 8
            if pad_width > 0:
                mel_input = F.pad(mel_input, (0, pad_width))
            
            latents = vae.encode(mel_input).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            encode_time = time.time() - start_time
            
            decode_start = time.time()
            latents = latents / vae.config.scaling_factor
            reconstructed_mel = vae.decode(latents).sample
            decode_time = time.time() - decode_start
            print(f"✅ 重试成功: 编码{encode_time:.3f}s, 解码{decode_time:.3f}s")
    
    # 音频重建：优先使用AudioLDM2内置HiFiGAN
    print("🎤 音频重建...")
    vocoder_method = "Unknown"
    
    with torch.no_grad():
        try:
            # 方法1: AudioLDM2内置HiFiGAN
            print("   🎯 尝试AudioLDM2内置HiFiGAN...")
            
            # 准备输入: [batch, time, mel_dim]
            vocoder_input = reconstructed_mel.squeeze(0)  # [1, 64, time] -> [64, time]
            vocoder_input = vocoder_input.transpose(-2, -1)  # [64, time] -> [time, 64]
            vocoder_input = vocoder_input.unsqueeze(0)  # [time, 64] -> [1, time, 64]
            
            print(f"      HiFiGAN输入: {vocoder_input.shape}")
            
            # 确保数据类型匹配
            if hasattr(vocoder, 'dtype'):
                vocoder_input = vocoder_input.to(vocoder.dtype)
            elif next(vocoder.parameters()).dtype == torch.float16:
                vocoder_input = vocoder_input.half()
            
            audio_tensor = vocoder(vocoder_input)
            reconstructed_audio = audio_tensor.squeeze().cpu().numpy()
            vocoder_method = "AudioLDM2_HiFiGAN"
            print(f"   ✅ AudioLDM2 HiFiGAN成功: {len(reconstructed_audio)}样本")
            
        except Exception as e:
            print(f"   ❌ AudioLDM2 HiFiGAN失败: {e}")
            print("   🔄 降级到Griffin-Lim...")
            
            # 方法2: Griffin-Lim降级
            recon_mel_np = reconstructed_mel.squeeze().cpu().numpy().astype(np.float32)
            recon_mel_denorm = (recon_mel_np + 1) / 2 * (norm_params['max_val'] - norm_params['min_val']) + norm_params['min_val']
            recon_mel_denorm = np.clip(recon_mel_denorm, -80.0, 0.0)
            recon_mel_power = librosa.db_to_power(recon_mel_denorm)
            
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
                print(f"   ✅ Griffin-Lim成功: {len(reconstructed_audio)}样本")
            except Exception as e2:
                print(f"   ❌ Griffin-Lim也失败: {e2}")
                reconstructed_audio = np.random.randn(len(audio)) * 0.1
                vocoder_method = "Fallback_Noise"
        
        # 后处理
        if len(reconstructed_audio) > 0:
            reconstructed_audio = np.nan_to_num(reconstructed_audio, nan=0.0)
        else:
            reconstructed_audio = np.zeros_like(audio)
    
    # 保存结果
    output_dir = "vae_hifigan_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
    torchaudio.save(original_path, torch.from_numpy(audio_normalized).unsqueeze(0), sample_rate)
    
    # 保存重建音频
    reconstructed_path = os.path.join(output_dir, f"{input_name}_reconstructed_{vocoder_method}_{timestamp}.wav")
    recon_normalized = reconstructed_audio / (np.max(np.abs(reconstructed_audio)) + 1e-8)
    
    # 确保长度一致
    if len(recon_normalized) > len(audio):
        recon_normalized = recon_normalized[:len(audio)]
    elif len(recon_normalized) < len(audio):
        recon_normalized = np.pad(recon_normalized, (0, len(audio) - len(recon_normalized)))
    
    torchaudio.save(reconstructed_path, torch.from_numpy(recon_normalized).unsqueeze(0), sample_rate)
    
    # 计算指标
    min_len = min(len(audio), len(recon_normalized))
    orig = audio[:min_len]
    recon = recon_normalized[:min_len]
    
    mse = np.mean((orig - recon) ** 2)
    correlation = np.corrcoef(orig, recon)[0, 1] if len(orig) > 1 else 0
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean((orig - recon) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # 压缩比
    compression_ratio = mel_input.numel() / latents.numel()
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"🎯 AudioLDM2 VAE + HiFiGAN 测试结果")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    print(f"📁 重建音频: {reconstructed_path}")
    print(f"⏱️ 编码时间: {encode_time:.3f}秒")
    print(f"⏱️ 解码时间: {decode_time:.3f}秒")
    print(f"⏱️ 总时间: {encode_time + decode_time:.3f}秒")
    print(f"📊 MSE: {mse:.6f}")
    print(f"📊 SNR: {snr:.2f} dB")
    print(f"📊 相关系数: {correlation:.4f}")
    print(f"🎵 Mel频谱形状: {mel_input.shape}")
    print(f"🗜️ 潜在表示形状: {latents.shape}")
    print(f"📦 压缩比: {compression_ratio:.1f}:1")
    print(f"🎤 重建方法: {vocoder_method}")
    
    # 分析结果
    print(f"\n💡 结果分析:")
    if vocoder_method == "AudioLDM2_HiFiGAN":
        print("✅ 成功使用AudioLDM2内置HiFiGAN！")
        if snr > 0:
            print("🎉 重建质量良好")
        elif snr > -5:
            print("⚠️ 重建质量一般，但可识别")
        else:
            print("❌ 重建质量较差")
    else:
        print(f"⚠️ 使用了降级方法: {vocoder_method}")
    
    return {
        'original_path': original_path,
        'reconstructed_path': reconstructed_path,
        'mse': mse,
        'snr': snr,
        'correlation': correlation,
        'encode_time': encode_time,
        'decode_time': decode_time,
        'compression_ratio': compression_ratio,
        'vocoder_method': vocoder_method
    }

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python vae_hifigan_fixed.py <音频文件路径> [最大长度秒数]")
        
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
                    print("❌ 无效选择")
                    return
            except (ValueError, KeyboardInterrupt):
                print("❌ 取消操作")
                return
        else:
            print("❌ 当前目录没有找到音频文件")
            return
    else:
        audio_path = sys.argv[1]
    
    # 获取最大长度
    max_length = 5  # 默认5秒
    if len(sys.argv) >= 3:
        try:
            max_length = float(sys.argv[2])
        except ValueError:
            print(f"⚠️ 无效长度参数，使用默认值 {max_length} 秒")
    
    # 执行测试
    print(f"🚀 开始测试: {audio_path}")
    print(f"⏱️ 最大长度: {max_length} 秒")
    
    try:
        result = load_and_test_vae_fixed(audio_path, max_length=max_length)
        if result:
            print("\n✅ 测试完成！")
            print("🎧 请播放原始音频和重建音频来比较质量")
            if result['vocoder_method'] == "AudioLDM2_HiFiGAN":
                print("🎉 成功突破了Griffin-Lim瓶颈，使用了神经vocoder！")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
