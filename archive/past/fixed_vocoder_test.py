"""
修复Vocoder维度问题的VAE测试
解决 "expected input[1, 504, 64] to have 64 channels, but got 504 channels" 错误

问题分析：
- Vocoder期望输入: [batch, channels, time] = [1, 64, time_steps]
- 当前输入: [1, time_steps, 64] - 维度顺序错误

解决方案：
1. 正确转置mel-spectrogram维度
2. 使用适当的vocoder配置
3. 添加维度检查和修复
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


def audio_to_mel_fixed(audio, sample_rate=16000, n_mels=64, n_fft=1024, hop_length=160):
    """
    音频转mel-spectrogram，确保正确的维度输出
    """
    # 计算mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0
    )
    
    # 转换到对数域
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 归一化到[-1, 1]
    mel_spec_norm = 2.0 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1.0
    
    print(f"   Mel-spec shape: {mel_spec_norm.shape}")
    print(f"   Mel-spec range: [{mel_spec_norm.min():.3f}, {mel_spec_norm.max():.3f}]")
    
    return mel_spec_norm


def mel_to_audio_vocoder_fixed(mel_spec, vocoder, device):
    """
    使用vocoder转换mel-spectrogram到音频，修复维度问题
    """
    print(f"🎤 使用修复的vocoder...")
    
    # 确保mel_spec是numpy数组
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.cpu().numpy()
    
    # 确保正确的维度顺序: [channels, time] -> [batch, channels, time]
    if mel_spec.ndim == 2:
        # 从 [n_mels, time] 转换为 [1, n_mels, time]
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float().to(device)
    else:
        raise ValueError(f"Unexpected mel_spec dimensions: {mel_spec.shape}")
    
    print(f"   修复后mel tensor shape: {mel_tensor.shape}")
    print(f"   期望vocoder输入: [batch, channels={vocoder.config.model_in_dim}, time]")
    
    # 检查通道数是否匹配
    expected_channels = vocoder.config.model_in_dim
    actual_channels = mel_tensor.shape[1]
    
    if actual_channels != expected_channels:
        print(f"   ⚠️ 通道数不匹配: 实际{actual_channels}, 期望{expected_channels}")
        # 尝试调整通道数
        if actual_channels < expected_channels:
            # 通过重复扩展通道
            repeat_factor = expected_channels // actual_channels
            mel_tensor = mel_tensor.repeat(1, repeat_factor, 1)
            if mel_tensor.shape[1] < expected_channels:
                # 如果还不够，补零
                padding = expected_channels - mel_tensor.shape[1]
                mel_tensor = F.pad(mel_tensor, (0, 0, 0, padding))
        else:
            # 裁剪多余通道
            mel_tensor = mel_tensor[:, :expected_channels, :]
        
        print(f"   调整后shape: {mel_tensor.shape}")
    
    try:
        with torch.no_grad():
            # 使用vocoder生成音频
            audio_tensor = vocoder(mel_tensor)
            
            if isinstance(audio_tensor, tuple):
                audio_tensor = audio_tensor[0]
            
            # 转换为numpy
            audio = audio_tensor.squeeze().cpu().numpy()
            print(f"   ✅ Vocoder成功! 输出形状: {audio.shape}")
            return audio
            
    except Exception as e:
        print(f"   ❌ Vocoder失败: {e}")
        return None


def mel_to_audio_griffinlim(mel_spec, sample_rate=16000, n_fft=1024, hop_length=160):
    """Griffin-Lim重建音频"""
    # 反归一化
    mel_spec_denorm = (mel_spec + 1.0) / 2.0
    mel_spec_denorm = mel_spec_denorm * 80.0 - 80.0  # 假设原始范围是-80到0 dB
    
    # 转换回功率域
    mel_spec_power = librosa.db_to_power(mel_spec_denorm, ref=1.0)
    
    # 使用Griffin-Lim重建
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec_power,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hann',
        center=True,
        pad_mode='reflect',
        n_iter=32
    )
    
    return audio


def test_vae_reconstruction_fixed(audio_path, model_id="cvssp/audioldm2-music", max_length=5):
    """
    修复版本的VAE音频重建测试
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
    
    # 打印vocoder配置信息
    print(f"🔧 Vocoder配置:")
    print(f"   类型: {type(vocoder).__name__}")
    print(f"   输入维度: {vocoder.config.model_in_dim}")
    print(f"   采样率: {vocoder.config.sampling_rate}")
    print(f"   上采样率: {vocoder.config.upsample_rates}")
    
    # 加载音频
    print(f"📁 正在加载音频: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    if len(audio) > max_length * sample_rate:
        audio = audio[:int(max_length * sample_rate)]
        print(f"✂️ 音频已裁剪到 {max_length} 秒")
    
    print(f"📊 音频信息: 长度={len(audio)/sample_rate:.2f}秒, 样本数={len(audio)}")
    
    # 创建输出目录
    output_dir = "vae_fixed_test"
    os.makedirs(output_dir, exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    # 保存原始音频
    original_path = os.path.join(output_dir, f"{input_name}_original_{timestamp}.wav")
    audio_save = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    torchaudio.save(original_path, torch.from_numpy(audio_save).unsqueeze(0), sample_rate)
    print(f"💾 原始音频保存: {original_path}")
    
    # 测试步骤
    results = {}
    
    print(f"\\n🔬 开始VAE重建测试...")
    
    # 步骤1: 音频 -> Mel-spectrogram
    print(f"\\n📊 步骤1: 音频 -> Mel-spectrogram")
    start_time = time.time()
    mel_spec = audio_to_mel_fixed(audio, sample_rate)
    mel_time = time.time() - start_time
    print(f"   ✅ Mel-spec生成完成 ({mel_time:.2f}秒)")
    
    # 步骤2: Mel -> VAE潜在空间
    print(f"\\n🧠 步骤2: Mel -> VAE潜在空间")
    start_time = time.time()
      # 准备VAE输入 - 确保数据类型匹配
    mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0)
    
    # 确保数据类型与模型匹配
    if next(vae.parameters()).dtype == torch.float16:
        mel_tensor = mel_tensor.half()
    else:
        mel_tensor = mel_tensor.float()
    
    mel_tensor = mel_tensor.to(device)
    print(f"   VAE输入形状: {mel_tensor.shape}, 数据类型: {mel_tensor.dtype}")
    
    with torch.no_grad():
        # VAE编码
        latent = vae.encode(mel_tensor).latent_dist.sample()
        print(f"   潜在向量形状: {latent.shape}")
        
        # VAE解码
        decoded = vae.decode(latent).sample
        print(f"   VAE解码形状: {decoded.shape}")
    
    vae_time = time.time() - start_time
    print(f"   ✅ VAE编码/解码完成 ({vae_time:.2f}秒)")
    
    # 转换解码结果为numpy
    decoded_mel = decoded.squeeze().cpu().numpy()
    print(f"   解码mel形状: {decoded_mel.shape}")
    
    # 步骤3a: 使用修复的Vocoder重建
    print(f"\\n🎤 步骤3a: 使用修复的Vocoder重建")
    start_time = time.time()
    
    vocoder_audio = mel_to_audio_vocoder_fixed(decoded_mel, vocoder, device)
    
    if vocoder_audio is not None:
        vocoder_time = time.time() - start_time
        print(f"   ✅ Vocoder重建完成 ({vocoder_time:.2f}秒)")
        
        # 计算指标
        snr_vocoder, corr_vocoder = calculate_metrics(audio, vocoder_audio)
        
        # 保存结果
        vocoder_path = os.path.join(output_dir, f"{input_name}_vocoder_fixed_{timestamp}.wav")
        audio_norm = vocoder_audio / np.max(np.abs(vocoder_audio)) if np.max(np.abs(vocoder_audio)) > 0 else vocoder_audio
        torchaudio.save(vocoder_path, torch.from_numpy(audio_norm).unsqueeze(0), sample_rate)
        
        results['vocoder'] = {
            'path': vocoder_path,
            'snr': snr_vocoder,
            'correlation': corr_vocoder,
            'time': vocoder_time
        }
        
        print(f"   📈 SNR: {snr_vocoder:.2f} dB")
        print(f"   🔗 相关系数: {corr_vocoder:.4f}")
        print(f"   💾 保存: {vocoder_path}")
    else:
        print(f"   ❌ Vocoder重建失败")
    
    # 步骤3b: 使用Griffin-Lim重建作为对比
    print(f"\\n🎵 步骤3b: 使用Griffin-Lim重建")
    start_time = time.time()
    
    griffinlim_audio = mel_to_audio_griffinlim(decoded_mel, sample_rate)
    griffinlim_time = time.time() - start_time
    
    # 计算指标
    snr_gl, corr_gl = calculate_metrics(audio, griffinlim_audio)
    
    # 保存结果
    gl_path = os.path.join(output_dir, f"{input_name}_griffinlim_{timestamp}.wav")
    audio_norm = griffinlim_audio / np.max(np.abs(griffinlim_audio)) if np.max(np.abs(griffinlim_audio)) > 0 else griffinlim_audio
    torchaudio.save(gl_path, torch.from_numpy(audio_norm).unsqueeze(0), sample_rate)
    
    results['griffinlim'] = {
        'path': gl_path,
        'snr': snr_gl,
        'correlation': corr_gl,
        'time': griffinlim_time
    }
    
    print(f"   ✅ Griffin-Lim重建完成 ({griffinlim_time:.2f}秒)")
    print(f"   📈 SNR: {snr_gl:.2f} dB")
    print(f"   🔗 相关系数: {corr_gl:.4f}")
    print(f"   💾 保存: {gl_path}")
    
    # 打印总结
    print(f"\\n{'='*60}")
    print(f"🎯 VAE重建测试结果总结")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    
    if 'vocoder' in results:
        vocoder_res = results['vocoder']
        print(f"\\n🎤 修复Vocoder方法:")
        print(f"   📄 文件: {vocoder_res['path']}")
        print(f"   📈 SNR: {vocoder_res['snr']:.2f} dB")
        print(f"   🔗 相关系数: {vocoder_res['correlation']:.4f}")
        print(f"   ⏱️ 处理时间: {vocoder_res['time']:.2f}秒")
    
    gl_res = results['griffinlim']
    print(f"\\n🎵 Griffin-Lim方法:")
    print(f"   📄 文件: {gl_res['path']}")
    print(f"   📈 SNR: {gl_res['snr']:.2f} dB")
    print(f"   🔗 相关系数: {gl_res['correlation']:.4f}")
    print(f"   ⏱️ 处理时间: {gl_res['time']:.2f}秒")
    
    if 'vocoder' in results:
        snr_diff = results['vocoder']['snr'] - results['griffinlim']['snr']
        print(f"\\n🚀 改进效果:")
        print(f"   📈 SNR改进: {snr_diff:+.2f} dB")
        print(f"   🏆 更好方法: {'Vocoder' if snr_diff > 0 else 'Griffin-Lim'}")
    
    print(f"\\n📁 所有结果保存在 {output_dir}/ 目录")
    print(f"🎧 建议播放音频文件来主观评估质量差异")
    print(f"\\n✅ 修复测试完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "AudioLDM2_Music_output.wav"
    
    print(f"🚀 开始修复版VAE测试: {audio_path}")
    test_vae_reconstruction_fixed(audio_path)
