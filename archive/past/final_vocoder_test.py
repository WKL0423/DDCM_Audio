"""
解决Vocoder维度问题的最终版本

根据错误分析：
- 期望输入: [1, 500, 64] (batch, time, channels)
- 实际输入: [1, 64, 500] (batch, channels, time)
- 解决方案: 转置最后两个维度

这个版本专门解决vocoder的维度匹配问题。
"""

import torch
import librosa
import numpy as np
import os
import time
from pathlib import Path
import torchaudio

from diffusers import AudioLDM2Pipeline


def calculate_metrics(original, reconstructed):
    """计算重建质量指标"""
    if len(original) != len(reconstructed):
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]
    
    # SNR计算
    noise = reconstructed - original
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # 相关系数
    correlation = np.corrcoef(original, reconstructed)[0, 1] if len(original) > 1 else 0
    
    return snr, correlation


def mel_to_audio_vocoder_corrected(mel_spec, vocoder, device):
    """
    修正维度的vocoder音频重建
    """
    print(f"🎤 使用修正维度的vocoder...")
    
    # 确保是numpy数组和float32
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.cpu().numpy()
    mel_spec = mel_spec.astype(np.float32)
    
    print(f"   原始mel形状: {mel_spec.shape}")
    
    # 转换为tensor
    mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float().to(device)
    print(f"   添加batch维度后: {mel_tensor.shape}")
    
    # 关键修正：转置最后两个维度
    # 从 [batch, channels, time] 转换为 [batch, time, channels]
    mel_tensor_transposed = mel_tensor.transpose(-2, -1)
    print(f"   转置后维度: {mel_tensor_transposed.shape}")
    
    try:
        with torch.no_grad():
            # 尝试vocoder
            audio_tensor = vocoder(mel_tensor_transposed)
            
            if isinstance(audio_tensor, tuple):
                audio_tensor = audio_tensor[0]
            
            audio = audio_tensor.squeeze().cpu().numpy()
            print(f"   ✅ Vocoder成功! 音频形状: {audio.shape}")
            return audio, "success"
            
    except Exception as e:
        print(f"   ❌ 转置后仍失败: {e}")
        
        # 尝试其他可能的维度组合
        try:
            print(f"   尝试其他维度安排...")
            # 尝试 [batch, time, channels] 但确保channels=64
            if mel_tensor.shape[-1] == 64:  # 如果最后一维是64
                mel_for_vocoder = mel_tensor.transpose(-2, -1)
            elif mel_tensor.shape[-2] == 64:  # 如果倒数第二维是64
                mel_for_vocoder = mel_tensor
            else:
                print(f"   找不到64通道维度")
                return None, f"dimension_error: {mel_tensor.shape}"
                
            print(f"   最终尝试维度: {mel_for_vocoder.shape}")
            
            with torch.no_grad():
                audio_tensor = vocoder(mel_for_vocoder)
                if isinstance(audio_tensor, tuple):
                    audio_tensor = audio_tensor[0]
                audio = audio_tensor.squeeze().cpu().numpy()
                print(f"   ✅ 备选方案成功! 音频形状: {audio.shape}")
                return audio, "alternative_success"
                
        except Exception as e2:
            print(f"   ❌ 所有vocoder尝试都失败: {e2}")
            return None, f"all_failed: {e} | {e2}"


def mel_to_audio_griffinlim_safe(mel_spec, sample_rate=16000):
    """
    安全的Griffin-Lim重建
    """
    try:
        # 确保是float32
        mel_spec = mel_spec.astype(np.float32)
        
        # 反归一化：从[-1,1] -> [min_db, 0]
        mel_spec_denorm = (mel_spec + 1.0) / 2.0
        mel_spec_denorm = mel_spec_denorm * 80.0 - 80.0
        
        # 转换到功率域
        mel_spec_power = librosa.db_to_power(mel_spec_denorm, ref=1.0)
        
        # Griffin-Lim重建
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec_power,
            sr=sample_rate,
            n_fft=1024,
            hop_length=160,
            window='hann',
            center=True,
            pad_mode='reflect',
            n_iter=32
        )
        
        return audio, "success"
        
    except Exception as e:
        return None, f"griffinlim_failed: {e}"


def final_vae_test(audio_path, model_id="cvssp/audioldm2-music", max_length=5):
    """
    最终的VAE测试，专注于解决vocoder维度问题
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 设备: {device}")
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到文件: {audio_path}")
        return
    
    print(f"🔄 加载模型: {model_id}")
    
    # 使用float32避免类型问题
    pipeline = AudioLDM2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    print(f"🔧 Vocoder信息:")
    print(f"   类型: {type(vocoder).__name__}")
    if hasattr(vocoder, 'config'):
        print(f"   输入维度: {vocoder.config.model_in_dim}")
        print(f"   采样率: {vocoder.config.sampling_rate}")
    
    # 加载音频
    print(f"📁 加载音频: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"✂️ 裁剪到 {max_length} 秒")
    
    print(f"📊 音频信息: {len(audio)/sample_rate:.2f}秒")
    
    # 创建输出目录
    output_dir = "vae_final_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_save = audio / (np.max(np.abs(audio)) + 1e-8)
    torchaudio.save(original_path, torch.from_numpy(audio_save).unsqueeze(0), sample_rate)
    print(f"💾 原始音频: {original_path}")
    
    print(f"\\n🔬 开始完整VAE流程测试...")
    
    # 1. 音频 -> Mel
    print(f"\\n1️⃣ 音频转Mel-spectrogram")
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=64,
        n_fft=1024,
        hop_length=160,
        power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = 2.0 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1.0
    mel_spec_norm = mel_spec_norm.astype(np.float32)
    
    print(f"   ✅ Mel形状: {mel_spec_norm.shape}")
    
    # 2. VAE编码/解码
    print(f"\\n2️⃣ VAE编码解码")
    mel_tensor = torch.from_numpy(mel_spec_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    print(f"   VAE输入: {mel_tensor.shape}")
    
    with torch.no_grad():
        latent = vae.encode(mel_tensor).latent_dist.sample()
        decoded = vae.decode(latent).sample
        
    decoded_mel = decoded.squeeze().cpu().float().numpy()
    print(f"   ✅ VAE输出: {decoded_mel.shape}")
    
    # 3. 多种重建方法测试
    print(f"\\n3️⃣ 音频重建对比测试")
    
    results = []
    
    # 方法A: 修正维度的Vocoder
    print(f"\\n🎤 方法A: 修正维度Vocoder")
    start_time = time.time()
    vocoder_audio, vocoder_status = mel_to_audio_vocoder_corrected(decoded_mel, vocoder, device)
    vocoder_time = time.time() - start_time
    
    if vocoder_audio is not None:
        snr_v, corr_v = calculate_metrics(audio, vocoder_audio)
        
        vocoder_path = os.path.join(output_dir, f"{input_name}_vocoder_corrected_{timestamp}.wav")
        audio_norm = vocoder_audio / (np.max(np.abs(vocoder_audio)) + 1e-8)
        torchaudio.save(vocoder_path, torch.from_numpy(audio_norm).unsqueeze(0), sample_rate)
        
        results.append({
            'method': 'Vocoder修正',
            'path': vocoder_path,
            'snr': snr_v,
            'correlation': corr_v,
            'time': vocoder_time,
            'status': vocoder_status
        })
        
        print(f"   ✅ 成功! SNR: {snr_v:.2f}dB, 相关: {corr_v:.4f}")
    else:
        print(f"   ❌ 失败: {vocoder_status}")
    
    # 方法B: Griffin-Lim
    print(f"\\n🎵 方法B: Griffin-Lim")
    start_time = time.time()
    gl_audio, gl_status = mel_to_audio_griffinlim_safe(decoded_mel, sample_rate)
    gl_time = time.time() - start_time
    
    if gl_audio is not None:
        snr_gl, corr_gl = calculate_metrics(audio, gl_audio)
        
        gl_path = os.path.join(output_dir, f"{input_name}_griffinlim_{timestamp}.wav")
        audio_norm = gl_audio / (np.max(np.abs(gl_audio)) + 1e-8)
        torchaudio.save(gl_path, torch.from_numpy(audio_norm).unsqueeze(0), sample_rate)
        
        results.append({
            'method': 'Griffin-Lim',
            'path': gl_path,
            'snr': snr_gl,
            'correlation': corr_gl,
            'time': gl_time,
            'status': gl_status
        })
        
        print(f"   ✅ 成功! SNR: {snr_gl:.2f}dB, 相关: {corr_gl:.4f}")
    else:
        print(f"   ❌ 失败: {gl_status}")
    
    # 总结结果
    print(f"\\n{'='*60}")
    print(f"🎯 最终VAE重建测试结果")
    print(f"{'='*60}")
    
    if results:
        # 按SNR排序
        results.sort(key=lambda x: x['snr'], reverse=True)
        
        print(f"\\n🏆 重建质量排名:")
        for i, result in enumerate(results, 1):
            print(f"   #{i} {result['method']}:")
            print(f"       📈 SNR: {result['snr']:.2f} dB")
            print(f"       🔗 相关系数: {result['correlation']:.4f}")
            print(f"       ⏱️ 处理时间: {result['time']:.2f}秒")
            print(f"       📄 文件: {result['path']}")
            print(f"       ✅ 状态: {result['status']}")
            print()
        
        best_result = results[0]
        print(f"🚀 最佳结果:")
        print(f"   🏆 最优方法: {best_result['method']}")
        print(f"   📈 最高SNR: {best_result['snr']:.2f} dB")
        print(f"   🔗 相关系数: {best_result['correlation']:.4f}")
        
        if len(results) > 1:
            improvement = best_result['snr'] - results[-1]['snr']
            print(f"   📊 方法间差异: {improvement:.2f} dB")
        
        # 检查vocoder是否成功
        vocoder_success = any(r['method'] == 'Vocoder修正' for r in results)
        if vocoder_success:
            print(f"\\n🎉 Vocoder维度问题已解决！")
        else:
            print(f"\\n⚠️ Vocoder仍有问题，建议使用Griffin-Lim")
            
    else:
        print(f"\\n❌ 所有重建方法都失败了")
    
    print(f"\\n📁 所有结果保存在: {output_dir}/")
    print(f"🎧 建议播放音频文件进行主观质量评估")
    print(f"\\n✅ 最终测试完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "AudioLDM2_Music_output.wav"
    
    print(f"🚀 开始最终VAE测试: {audio_path}")
    print(f"🎯 目标: 解决vocoder维度问题，实现高质量音频重建")
    final_vae_test(audio_path)
