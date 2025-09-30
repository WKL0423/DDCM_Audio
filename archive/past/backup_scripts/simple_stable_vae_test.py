"""
简化且稳定的VAE测试版本
专注于解决数据类型和维度问题，实现基本的VAE重建功能

主要改进：
1. 统一使用float32避免数据类型问题
2. 简化vocoder调用逻辑
3. 改进错误处理
4. 提供清晰的结果对比
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


def audio_to_mel_simple(audio, sample_rate=16000):
    """
    简化的音频转mel-spectrogram
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=64,
        n_fft=1024,
        hop_length=160,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0
    )
    
    # 转换到对数域并归一化
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = 2.0 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1.0
    
    return mel_spec_norm.astype(np.float32)


def mel_to_audio_simple(mel_spec, sample_rate=16000):
    """
    简化的mel到音频转换（Griffin-Lim）
    """
    # 确保是float32
    mel_spec = mel_spec.astype(np.float32)
    
    # 反归一化
    mel_spec_denorm = (mel_spec + 1.0) / 2.0
    mel_spec_denorm = mel_spec_denorm * 80.0 - 80.0
    
    # 转换回功率域
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
    
    return audio


def try_vocoder_reconstruction(mel_spec, vocoder, device):
    """
    尝试使用vocoder重建音频，包含多种方法
    """
    results = []
    
    # 确保输入是float32
    mel_spec = mel_spec.astype(np.float32)
    
    # 方法1: 直接使用mel_spec
    try:
        print(f"   尝试方法1: 直接vocoder...")
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float().to(device)
        
        # 检查vocoder期望的数据类型
        vocoder_dtype = next(vocoder.parameters()).dtype
        if vocoder_dtype == torch.float16:
            mel_tensor = mel_tensor.half()
        
        print(f"   输入形状: {mel_tensor.shape}, 类型: {mel_tensor.dtype}")
        
        with torch.no_grad():
            audio_tensor = vocoder(mel_tensor)
            if isinstance(audio_tensor, tuple):
                audio_tensor = audio_tensor[0]
            audio = audio_tensor.squeeze().cpu().float().numpy()
            
        results.append(("直接vocoder", audio, None))
        print(f"   ✅ 方法1成功！")
        
    except Exception as e:
        print(f"   ❌ 方法1失败: {e}")
    
    # 方法2: 调整mel_spec范围
    try:
        print(f"   尝试方法2: 调整范围...")
        # 将[-1,1]范围映射到[0,1]
        mel_adjusted = (mel_spec + 1.0) / 2.0
        mel_tensor = torch.from_numpy(mel_adjusted).unsqueeze(0).float().to(device)
        
        # 检查vocoder期望的数据类型
        vocoder_dtype = next(vocoder.parameters()).dtype
        if vocoder_dtype == torch.float16:
            mel_tensor = mel_tensor.half()
        
        with torch.no_grad():
            audio_tensor = vocoder(mel_tensor)
            if isinstance(audio_tensor, tuple):
                audio_tensor = audio_tensor[0]
            audio = audio_tensor.squeeze().cpu().float().numpy()
            
        results.append(("调整范围vocoder", audio, None))
        print(f"   ✅ 方法2成功！")
        
    except Exception as e:
        print(f"   ❌ 方法2失败: {e}")
    
    # 方法3: 使用Griffin-Lim作为备选
    try:
        print(f"   备选: Griffin-Lim...")
        audio = mel_to_audio_simple(mel_spec)
        results.append(("Griffin-Lim备选", audio, None))
        print(f"   ✅ Griffin-Lim成功！")
        
    except Exception as e:
        print(f"   ❌ Griffin-Lim失败: {e}")
    
    return results


def simple_vae_test(audio_path, model_id="cvssp/audioldm2-music", max_length=5):
    """
    简化的VAE重建测试
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 设备: {device}")
    
    if not os.path.exists(audio_path):
        print(f"❌ 找不到文件: {audio_path}")
        return
    
    print(f"🔄 加载模型: {model_id}")
    
    # 强制使用float32以避免数据类型问题
    pipeline = AudioLDM2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    ).to(device)
    
    vae = pipeline.vae
    vocoder = pipeline.vocoder
    sample_rate = 16000
    
    print(f"🔧 Vocoder: {type(vocoder).__name__}")
    
    # 加载音频
    print(f"📁 加载音频: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"✂️ 裁剪到 {max_length} 秒")
    
    print(f"📊 音频: {len(audio)/sample_rate:.2f}秒, {len(audio)}样本")
    
    # 创建输出目录
    output_dir = "vae_simple_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_save = audio / (np.max(np.abs(audio)) + 1e-8)
    torchaudio.save(original_path, torch.from_numpy(audio_save).unsqueeze(0), sample_rate)
    
    print(f"\\n🔬 开始VAE重建流程...")
    
    # 步骤1: 音频 -> Mel
    print(f"\\n1️⃣ 音频转Mel-spectrogram")
    start_time = time.time()
    mel_spec = audio_to_mel_simple(audio, sample_rate)
    print(f"   ✅ Mel形状: {mel_spec.shape} ({time.time()-start_time:.2f}秒)")
    
    # 步骤2: VAE编码/解码
    print(f"\\n2️⃣ VAE编码/解码")
    start_time = time.time()
    
    # 准备VAE输入（确保float32）
    mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).float().to(device)
    print(f"   VAE输入: {mel_tensor.shape}")
    
    with torch.no_grad():
        # 编码
        latent = vae.encode(mel_tensor).latent_dist.sample()
        print(f"   潜在空间: {latent.shape}")
        
        # 解码
        decoded = vae.decode(latent).sample
        print(f"   VAE输出: {decoded.shape}")
    
    # 转换为numpy（确保float32）
    decoded_mel = decoded.squeeze().cpu().float().numpy()
    print(f"   ✅ VAE完成 ({time.time()-start_time:.2f}秒)")
    
    # 步骤3: 多种重建方法
    print(f"\\n3️⃣ 音频重建测试")
    
    # 测试vocoder方法
    print(f"\\n🎤 测试Vocoder方法:")
    vocoder_results = try_vocoder_reconstruction(decoded_mel, vocoder, device)
    
    # 测试Griffin-Lim
    print(f"\\n🎵 测试Griffin-Lim:")
    try:
        gl_audio = mel_to_audio_simple(decoded_mel, sample_rate)
        gl_results = [("Griffin-Lim", gl_audio, None)]
        print(f"   ✅ Griffin-Lim成功")
    except Exception as e:
        print(f"   ❌ Griffin-Lim失败: {e}")
        gl_results = []
    
    # 合并所有结果
    all_results = vocoder_results + gl_results
    
    # 保存和评估结果
    print(f"\\n📊 结果评估:")
    final_results = []
    
    for i, (method_name, recon_audio, error) in enumerate(all_results):
        if recon_audio is None:
            continue
            
        try:
            # 计算指标
            snr, corr = calculate_metrics(audio, recon_audio)
            
            # 保存音频
            save_path = os.path.join(output_dir, f"{input_name}_{method_name.replace(' ', '_')}_{timestamp}.wav")
            audio_norm = recon_audio / (np.max(np.abs(recon_audio)) + 1e-8)
            torchaudio.save(save_path, torch.from_numpy(audio_norm).unsqueeze(0), sample_rate)
            
            final_results.append({
                'method': method_name,
                'path': save_path,
                'snr': snr,
                'correlation': corr
            })
            
            print(f"   ✅ {method_name}: SNR={snr:.2f}dB, 相关={corr:.4f}")
            
        except Exception as e:
            print(f"   ❌ {method_name}保存失败: {e}")
    
    # 打印最终总结
    print(f"\\n{'='*60}")
    print(f"🎯 VAE重建测试总结")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    
    if final_results:
        # 按SNR排序
        final_results.sort(key=lambda x: x['snr'], reverse=True)
        
        print(f"\\n🏆 结果排名 (按SNR):")
        for i, result in enumerate(final_results, 1):
            print(f"   #{i} {result['method']}: {result['snr']:.2f}dB (相关:{result['correlation']:.4f})")
            print(f"       文件: {result['path']}")
        
        best_snr = final_results[0]['snr']
        worst_snr = final_results[-1]['snr'] if len(final_results) > 1 else best_snr
        improvement = best_snr - worst_snr
        
        print(f"\\n🚀 性能分析:")
        print(f"   📈 最佳SNR: {best_snr:.2f}dB ({final_results[0]['method']})")
        print(f"   📉 最差SNR: {worst_snr:.2f}dB")
        print(f"   📊 方法差异: {improvement:.2f}dB")
    else:
        print(f"\\n❌ 没有成功的重建结果")
    
    print(f"\\n📁 结果保存在: {output_dir}/")
    print(f"🎧 请播放音频文件进行主观评估")
    print(f"\\n✅ 测试完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "AudioLDM2_Music_output.wav"
    
    print(f"🚀 开始简化VAE测试: {audio_path}")
    simple_vae_test(audio_path)
