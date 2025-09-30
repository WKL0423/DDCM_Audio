#!/usr/bin/env python3
"""
AudioLDM2 VAE + HiFiGAN 终极修复版本
自动处理AudioLDM2_Music_output.wav文件
修复所有已知问题：VAE scaling、ClapFeatureExtractor、HiFiGAN输入格式
"""

import sys
import torch
import numpy as np
import librosa
from pathlib import Path
import time
import soundfile as sf
from diffusers import AudioLDM2Pipeline
import os

def save_audio_compatible(audio, path, sr=16000):
    """兼容的音频保存函数"""
    try:
        # 确保音频是numpy数组
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # 确保音频是一维的
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        # 归一化到[-1, 1]
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.max(np.abs(audio))
        
        # 保存为WAV文件
        sf.write(path, audio, sr)
        print(f"   💾 保存成功: {path}")
        return True
        
    except Exception as e:
        print(f"   ❌ 保存失败 {path}: {e}")
        return False

def test_audioldm2_ultimate_fix(audio_path, max_length=10.0):
    """
    AudioLDM2 VAE + HiFiGAN 终极修复测试
    自动处理AudioLDM2_Music_output.wav文件
    添加多种改进方法以提高重建质量
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🎯 AudioLDM2 最终修复测试 (改进版)")
    print(f"📱 设备: {device}")
    
    # 加载AudioLDM2 pipeline
    print("📦 加载AudioLDM2模型...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    print(f"✅ 模型加载完成")
    print(f"   VAE scaling_factor: {pipeline.vae.config.scaling_factor}")
    print(f"   Vocoder类型: {type(pipeline.vocoder).__name__}")
    
    # 获取feature extractor参数
    fe_sr = pipeline.feature_extractor.sampling_rate
    print(f"   FeatureExtractor采样率: {fe_sr} Hz")
    
    # 加载音频 - 使用多种采样率进行对比
    print(f"📁 加载音频: {Path(audio_path).name}")
    
    # 使用feature extractor的采样率
    audio_fe_sr, sr_fe = librosa.load(audio_path, sr=fe_sr, duration=max_length)
    audio_16k, sr_16k = librosa.load(audio_path, sr=16000, duration=max_length)
    
    print(f"   {fe_sr}Hz音频: {len(audio_fe_sr)/sr_fe:.2f}秒, 范围[{audio_fe_sr.min():.3f}, {audio_fe_sr.max():.3f}]")
    print(f"   16kHz音频: {len(audio_16k)/sr_16k:.2f}秒, 范围[{audio_16k.min():.3f}, {audio_16k.max():.3f}]")
    
    # 方法1: 使用AudioLDM2的ClapFeatureExtractor（改进版）
    print("\n🎵 方法1: 使用AudioLDM2的ClapFeatureExtractor (改进版)...")
    try:
        # 音频预处理 - 添加音量归一化
        audio_input = audio_fe_sr.copy()
        
        # 轻微的音量归一化，避免过度处理
        if np.max(np.abs(audio_input)) > 0:
            audio_input = audio_input / np.max(np.abs(audio_input)) * 0.95
        
        # 确保音频输入是正确的格式
        if len(audio_input.shape) > 1:
            audio_input = audio_input.squeeze()
        
        features = pipeline.feature_extractor(
            audio_input, 
            return_tensors="pt", 
            sampling_rate=fe_sr
        )
        
        mel_input = features.input_features.to(device)
        if device == "cuda":
            mel_input = mel_input.half()
        
        print(f"   ✅ ClapFeatureExtractor成功")
        print(f"   输入: {mel_input.shape} (格式: [batch, channel, time, feature])")
        print(f"   范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
        
        use_clap_features = True
        
    except Exception as e:
        print(f"   ❌ ClapFeatureExtractor失败: {e}")
        use_clap_features = False
    
    # 如果ClapFeatureExtractor失败，使用改进的传统方法
    if not use_clap_features:
        print("   🔄 使用改进的传统mel频谱处理...")
        
        # 使用更精确的mel-spectrogram参数
        mel_spec = librosa.feature.melspectrogram(
            y=audio_fe_sr, 
            sr=fe_sr, 
            n_mels=64,
            hop_length=int(fe_sr * 0.01),  # 10ms hop length
            n_fft=int(fe_sr * 0.025),      # 25ms window
            fmin=50,
            fmax=fe_sr // 2,  # Nyquist frequency
            window='hann',
            center=True,
            pad_mode='reflect',
            power=2.0  # 功率谱
        )
        
        # 改进的dB转换
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # 标准化到[-1, 1]范围
        mel_db = 2 * (mel_db + 80) / 80 - 1
        
        # 转换为AudioLDM2期望的格式
        mel_tensor = torch.from_numpy(mel_db).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.half()
        
        # 维度转换：[64, time] -> [1, 1, time, 64]
        mel_input = mel_tensor.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        
        print(f"   传统处理: {mel_input.shape}")
        print(f"   范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
    
    print(f"   最终输入: {mel_input.shape}, {mel_input.dtype}")
    
    # VAE处理 - 改进版本
    print("\n🧠 VAE编码解码 (改进版)...")
    with torch.no_grad():
        # 编码
        latent_dist = pipeline.vae.encode(mel_input)
        
        # 改进1: 使用更准确的latent采样
        if hasattr(latent_dist, 'latent_dist'):
            # 使用mode而不是sample可能更稳定
            latent = latent_dist.latent_dist.mode()
        else:
            latent = latent_dist.mode() if hasattr(latent_dist, 'mode') else latent_dist.sample()
        
        # 关键修复1: 编码后应用scaling_factor
        latent = latent * pipeline.vae.config.scaling_factor
        print(f"   编码latent: {latent.shape}")
        print(f"   Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
        
        # 改进2: 轻微的latent space regularization
        latent_std = torch.std(latent)
        if latent_std > 3.0:  # 如果latent过于分散
            latent = latent * (3.0 / latent_std)
            print(f"   Latent正则化: std {latent_std:.3f} -> 3.0")
        
        # 解码
        latent_for_decode = latent / pipeline.vae.config.scaling_factor
        reconstructed_mel = pipeline.vae.decode(latent_for_decode).sample
        
        print(f"   解码输出: {reconstructed_mel.shape}")
        print(f"   解码范围: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
    
    # HiFiGAN处理 - 多种策略
    print("\n🎤 HiFiGAN vocoder (多策略)...")
    
    # 策略1: 使用pipeline标准方法
    try:
        print("   🚀 策略1: 使用pipeline.mel_spectrogram_to_waveform...")
        waveform = pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
        reconstructed_audio = waveform.squeeze().cpu().detach().numpy()
        vocoder_method = "AudioLDM2_Pipeline_Standard"
        print(f"   ✅ 成功！输出: {len(reconstructed_audio)}样本")
        
    except Exception as e:
        print(f"   ❌ 策略1失败: {e}")
        
        # 策略2: 直接使用vocoder，但加入mel-spectrogram预处理
        try:
            print("   🔄 策略2: 预处理+直接vocoder调用...")
            
            # 预处理mel-spectrogram
            vocoder_input = reconstructed_mel.clone()
            
            # 维度调整
            if vocoder_input.dim() == 4:
                vocoder_input = vocoder_input.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            
            # 改进的mel-spectrogram预处理
            # 将范围调整到vocoder期望的范围
            vocoder_input = torch.clamp(vocoder_input, -10, 2)
            
            # 确保数据类型匹配
            vocoder_dtype = next(pipeline.vocoder.parameters()).dtype
            vocoder_input = vocoder_input.to(vocoder_dtype)
            
            print(f"   Vocoder输入: {vocoder_input.shape}, {vocoder_input.dtype}")
            print(f"   Vocoder输入范围: [{vocoder_input.min():.3f}, {vocoder_input.max():.3f}]")
            
            # 直接调用vocoder
            with torch.no_grad():
                waveform = pipeline.vocoder(vocoder_input)
                reconstructed_audio = waveform.squeeze().cpu().detach().numpy()
                
            vocoder_method = "AudioLDM2_Vocoder_Direct_Improved"
            print(f"   ✅ 策略2成功！输出: {len(reconstructed_audio)}样本")
            
        except Exception as e2:
            print(f"   ❌ 策略2失败: {e2}")
            
            # 策略3: 使用改进的Griffin-Lim
            print("   🔄 策略3: 使用改进的Griffin-Lim...")
            
            mel_np = reconstructed_mel.squeeze().cpu().detach().numpy()
            print(f"   Mel shape: {mel_np.shape}")
            
            # 维度调整
            if mel_np.ndim == 2 and mel_np.shape[1] == 64:
                mel_np = mel_np.T  # [time, 64] -> [64, time]
            
            # 改进的反归一化 - 使用更精确的映射
            if use_clap_features:
                # ClapFeatureExtractor的输出范围大约是[-100, 20]
                mel_db = (mel_np + 100) / 120 * 80 - 80
            else:
                # 传统方法的输出范围是[-1, 1]
                mel_db = (mel_np + 1) / 2 * 80 - 80
            
            mel_power = librosa.db_to_power(mel_db)
            
            # 使用更适合的参数
            reconstructed_audio = librosa.feature.inverse.mel_to_audio(
                mel_power,
                sr=16000,
                hop_length=160,
                n_fft=1024,
                win_length=1024,
                window='hann',
                n_iter=64,  # 更多迭代
                length=len(audio_16k)
            )
            vocoder_method = "Griffin_Lim_Improved"
            print(f"   ✅ 策略3成功: {len(reconstructed_audio)}样本")
    
    # 后处理改进
    print("\n🔧 后处理...")
    
    # 音量匹配
    reference_audio = audio_16k
    if len(reconstructed_audio) > len(reference_audio):
        reconstructed_audio = reconstructed_audio[:len(reference_audio)]
    elif len(reconstructed_audio) < len(reference_audio):
        reconstructed_audio = np.pad(reconstructed_audio, (0, len(reference_audio) - len(reconstructed_audio)))
    
    # 改进的音量匹配
    ref_rms = np.sqrt(np.mean(reference_audio ** 2))
    rec_rms = np.sqrt(np.mean(reconstructed_audio ** 2))
    
    if rec_rms > 0:
        volume_ratio = ref_rms / rec_rms
        # 限制音量调整范围，避免过度放大
        volume_ratio = np.clip(volume_ratio, 0.1, 10.0)
        reconstructed_audio = reconstructed_audio * volume_ratio
        print(f"   音量匹配: {rec_rms:.4f} -> {ref_rms:.4f} (比例: {volume_ratio:.2f})")
    
    # 保存结果
    output_dir = Path("vae_hifigan_ultimate_fix")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
    reconstructed_path = output_dir / f"{input_name}_{vocoder_method}_improved_{timestamp}.wav"
    
    print("\n💾 保存结果...")
    save_audio_compatible(reference_audio, original_path)
    save_audio_compatible(reconstructed_audio, reconstructed_path)
    
    # 计算质量指标
    mse = np.mean((reference_audio - reconstructed_audio) ** 2)
    snr = 10 * np.log10(np.mean(reference_audio ** 2) / (mse + 1e-10))
    correlation = np.corrcoef(reference_audio, reconstructed_audio)[0, 1] if len(reference_audio) > 1 else 0
    
    # 额外的质量指标
    mae = np.mean(np.abs(reference_audio - reconstructed_audio))
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"🎯 AudioLDM2 最终修复结果 (改进版)")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    print(f"📁 重建音频: {reconstructed_path}")
    print(f"📊 SNR: {snr:.2f} dB")
    print(f"📊 MSE: {mse:.6f}")
    print(f"📊 MAE: {mae:.6f}")
    print(f"📊 相关系数: {correlation:.4f}")
    print(f"🎤 重建方法: {vocoder_method}")
    
    # 诊断分析
    print(f"\n🔬 关键改进:")
    print(f"   ✅ VAE scaling_factor: {pipeline.vae.config.scaling_factor}")
    print(f"   ✅ 音量预处理和后处理")
    print(f"   ✅ 改进的latent采样 (mode vs sample)")
    print(f"   ✅ 精确的mel-spectrogram参数")
    print(f"   ✅ 多策略vocoder处理")
    print(f"   ✅ ClapFeatureExtractor使用: {'成功' if use_clap_features else '回退到传统方法'}")
    
    # 质量评估
    if "Standard" in vocoder_method or "Direct" in vocoder_method:
        print(f"\n🎉 HiFiGAN修复成功！")
        quality_score = snr + correlation * 10  # 综合质量分数
        if quality_score > 5:
            print(f"🏆 重建质量优秀！(综合分数: {quality_score:.2f}")
        else:
            print(f"🔧 重建质量良好，但仍有提升空间")
    else:
        print(f"⚠️ 使用了回退的Griffin-Lim vocoder")
        print(f"💡 建议: 检查HiFiGAN模型加载是否正常")


def test_audioldm2_v3_balanced(audio_path, max_length=10.0):
    """
    V3版本: 平衡优化版本
    基于V1但添加更精细的参数调整和处理策略
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🎯 V3: AudioLDM2 平衡优化版本")
    print(f"📱 设备: {device}")
    
    # 加载AudioLDM2 pipeline
    print("📦 加载AudioLDM2模型...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    print(f"✅ 模型加载完成")
    print(f"   VAE scaling_factor: {pipeline.vae.config.scaling_factor}")
    print(f"   Vocoder类型: {type(pipeline.vocoder).__name__}")
    
    # 获取feature extractor参数
    fe_sr = pipeline.feature_extractor.sampling_rate
    print(f"   FeatureExtractor采样率: {fe_sr} Hz")
    
    # 加载音频 - V3: 使用更精细的音频预处理
    print(f"📁 加载音频: {Path(audio_path).name}")
    
    # 使用feature extractor的采样率
    audio_fe_sr, sr_fe = librosa.load(audio_path, sr=fe_sr, duration=max_length)
    audio_16k, sr_16k = librosa.load(audio_path, sr=16000, duration=max_length)
    
    print(f"   {fe_sr}Hz音频: {len(audio_fe_sr)/sr_fe:.2f}秒, 范围[{audio_fe_sr.min():.3f}, {audio_fe_sr.max():.3f}]")
    print(f"   16kHz音频: {len(audio_16k)/sr_16k:.2f}秒, 范围[{audio_16k.min():.3f}, {audio_16k.max():.3f}]")
    
    # V3特色: 更精细的音频预处理
    print("🎵 V3: 精细化ClapFeatureExtractor处理...")
    try:
        # V3改进1: 动态音量调整
        audio_input = audio_fe_sr.copy()
        
        # 计算动态范围
        audio_rms = np.sqrt(np.mean(audio_input**2))
        audio_peak = np.max(np.abs(audio_input))
        
        # V3特色: 动态音量归一化策略
        if audio_peak > 0:
            # 如果音频很小，适度放大
            if audio_peak < 0.1:
                audio_input = audio_input * (0.3 / audio_peak)
            # 如果音频很大，适度缩小
            elif audio_peak > 0.95:
                audio_input = audio_input * (0.85 / audio_peak)
            # 中等音量，轻微调整
            else:
                audio_input = audio_input * (0.9 / audio_peak)
        
        print(f"   V3音频调整: {audio_peak:.3f} -> {np.max(np.abs(audio_input)):.3f}")
        
        # V3改进2: 添加轻微的平滑处理
        from scipy import signal
        # 使用很轻的低通滤波器，只去除极高频噪声
        sos = signal.butter(2, 0.95, 'low', output='sos')
        audio_input = signal.sosfilt(sos, audio_input)
        
        features = pipeline.feature_extractor(
            audio_input, 
            return_tensors="pt", 
            sampling_rate=fe_sr
        )
        
        mel_input = features.input_features.to(device)
        if device == "cuda":
            mel_input = mel_input.half()
        
        print(f"   ✅ V3 ClapFeatureExtractor成功")
        print(f"   输入: {mel_input.shape} (格式: [batch, channel, time, feature])")
        print(f"   范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
        
        use_clap_features = True
        
    except Exception as e:
        print(f"   ❌ V3 ClapFeatureExtractor失败: {e}")
        use_clap_features = False
    
    # 如果失败，使用传统方法
    if not use_clap_features:
        print("   🔄 V3: 使用传统方法...")
        # 使用与V1相同的传统方法
        mel_spec = librosa.feature.melspectrogram(
            y=audio_fe_sr, 
            sr=fe_sr, 
            n_mels=64,
            hop_length=int(fe_sr * 0.01),
            n_fft=int(fe_sr * 0.025),
            fmin=50,
            fmax=fe_sr // 2,
            window='hann',
            center=True,
            pad_mode='reflect',
            power=2.0
        )
        
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        mel_db = 2 * (mel_db + 80) / 80 - 1
        
        mel_tensor = torch.from_numpy(mel_db).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.half()
        
        mel_input = mel_tensor.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        print(f"   传统处理: {mel_input.shape}")
        print(f"   范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
    
    print(f"   最终输入: {mel_input.shape}, {mel_input.dtype}")
    
    # V3 VAE处理 - 更精细的参数控制
    print("\n🧠 V3: VAE编码解码 (精细优化)...")
    with torch.no_grad():
        # 编码
        latent_dist = pipeline.vae.encode(mel_input)
        
        # V3改进3: 混合采样策略
        if hasattr(latent_dist, 'latent_dist'):
            # 混合mode和sample，取加权平均
            latent_mode = latent_dist.latent_dist.mode()
            latent_sample = latent_dist.latent_dist.sample()
            # 70%mode + 30%sample，平衡确定性和随机性
            latent = 0.7 * latent_mode + 0.3 * latent_sample
        else:
            latent = latent_dist.mode() if hasattr(latent_dist, 'mode') else latent_dist.sample()
        
        # 应用scaling_factor
        latent = latent * pipeline.vae.config.scaling_factor
        print(f"   V3编码latent: {latent.shape}")
        print(f"   Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
        
        # V3改进4: 自适应latent调整
        latent_std = torch.std(latent)
        latent_mean = torch.mean(latent)
        
        # 如果latent分布异常，进行轻微调整
        if latent_std > 4.0:
            latent = latent * (3.5 / latent_std)
            print(f"   V3 Latent标准化: std {latent_std:.3f} -> 3.5")
        
        # 如果均值偏移过大，进行中心化
        if abs(latent_mean) > 2.0:
            latent = latent - latent_mean * 0.3
            print(f"   V3 Latent中心化: mean {latent_mean:.3f} -> {torch.mean(latent):.3f}")
        
        # 解码
        latent_for_decode = latent / pipeline.vae.config.scaling_factor
        reconstructed_mel = pipeline.vae.decode(latent_for_decode).sample
        
        print(f"   V3解码输出: {reconstructed_mel.shape}")
        print(f"   解码范围: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
    
    # V3 HiFiGAN处理 - 优化的vocoder策略
    print("\n🎤 V3: HiFiGAN vocoder (优化策略)...")
    
    # V3优先策略: 改进的pipeline方法
    try:
        print("   🚀 V3策略: 优化的pipeline.mel_spectrogram_to_waveform...")
        
        # V3改进5: mel-spectrogram后处理
        mel_processed = reconstructed_mel.clone()
        
        # 轻微的mel-spectrogram平滑
        if mel_processed.dim() == 4:
            # 对时间维度进行轻微平滑
            kernel = torch.ones(1, 1, 1, 3, device=mel_processed.device, dtype=mel_processed.dtype) / 3
            mel_processed = torch.nn.functional.conv2d(
                mel_processed, 
                kernel, 
                padding=(0, 1)
            )
        
        waveform = pipeline.mel_spectrogram_to_waveform(mel_processed)
        reconstructed_audio = waveform.squeeze().cpu().detach().numpy()
        vocoder_method = "V3_AudioLDM2_Pipeline_Balanced"
        print(f"   ✅ V3成功！输出: {len(reconstructed_audio)}样本")
        
    except Exception as e:
        print(f"   ❌ V3策略失败: {e}")
        
        # 回退到标准方法
        try:
            print("   🔄 V3回退: 标准pipeline方法...")
            waveform = pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
            reconstructed_audio = waveform.squeeze().cpu().detach().numpy()
            vocoder_method = "V3_AudioLDM2_Pipeline_Standard"
            print(f"   ✅ V3回退成功！输出: {len(reconstructed_audio)}样本")
            
        except Exception as e2:
            print(f"   ❌ V3回退失败: {e2}")
            return None
    
    # V3后处理 - 平衡的音质优化
    print("\n🔧 V3: 平衡后处理...")
    
    # 长度匹配
    reference_audio = audio_16k
    if len(reconstructed_audio) > len(reference_audio):
        reconstructed_audio = reconstructed_audio[:len(reference_audio)]
    elif len(reconstructed_audio) < len(reference_audio):
        reconstructed_audio = np.pad(reconstructed_audio, (0, len(reference_audio) - len(reconstructed_audio)))
    
    # V3改进6: 智能音量匹配
    ref_rms = np.sqrt(np.mean(reference_audio ** 2))
    rec_rms = np.sqrt(np.mean(reconstructed_audio ** 2))
    
    if rec_rms > 0:
        # V3特色: 更保守的音量匹配
        volume_ratio = ref_rms / rec_rms
        
        # 根据信噪比调整音量匹配强度
        initial_snr = 10 * np.log10(np.mean(reference_audio**2) / (np.mean((reference_audio - reconstructed_audio)**2) + 1e-10))
        
        if initial_snr > 0:  # 如果信噪比较好，轻微调整
            volume_ratio = 1.0 + (volume_ratio - 1.0) * 0.8
        else:  # 如果信噪比较差，更积极调整
            volume_ratio = 1.0 + (volume_ratio - 1.0) * 1.2
        
        # 限制音量调整范围
        volume_ratio = np.clip(volume_ratio, 0.2, 5.0)
        reconstructed_audio = reconstructed_audio * volume_ratio
        print(f"   V3音量匹配: {rec_rms:.4f} -> {ref_rms:.4f} (比例: {volume_ratio:.2f})")
    
    # V3改进7: 轻微的后处理滤波
    try:
        # 只对重建音频进行轻微的去噪
        from scipy import signal
        # 使用很轻的高通滤波器，去除低频噪声
        sos = signal.butter(1, 50/(16000/2), 'high', output='sos')
        reconstructed_audio = signal.sosfilt(sos, reconstructed_audio)
        print(f"   V3后处理滤波: 完成")
    except:
        print(f"   V3后处理滤波: 跳过")
    
    # 保存结果
    output_dir = Path("vae_hifigan_ultimate_fix")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
    reconstructed_path = output_dir / f"{input_name}_V3_{vocoder_method}_{timestamp}.wav"
    
    print("\n💾 V3保存结果...")
    save_audio_compatible(reference_audio, original_path)
    save_audio_compatible(reconstructed_audio, reconstructed_path)
    
    # 计算质量指标
    mse = np.mean((reference_audio - reconstructed_audio) ** 2)
    snr = 10 * np.log10(np.mean(reference_audio ** 2) / (mse + 1e-10))
    correlation = np.corrcoef(reference_audio, reconstructed_audio)[0, 1] if len(reference_audio) > 1 else 0
    mae = np.mean(np.abs(reference_audio - reconstructed_audio))
    
    # V3综合质量分数
    quality_score = snr + correlation * 8 + (1 / (mae + 0.01)) * 2
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"🎯 V3结果 (AudioLDM2 平衡优化版本)")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    print(f"📁 重建音频: {reconstructed_path}")
    print(f"📊 SNR: {snr:.2f} dB")
    print(f"📊 MSE: {mse:.6f}")
    print(f"📊 MAE: {mae:.6f}")
    print(f"📊 相关系数: {correlation:.4f}")
    print(f"🏆 V3综合质量分数: {quality_score:.2f}")
    print(f"🎤 重建方法: {vocoder_method}")
    
    # V3诊断分析
    print(f"\n🔬 V3特色改进:")
    print(f"   ✅ 动态音量预处理")
    print(f"   ✅ 混合采样策略 (70%mode + 30%sample)")
    print(f"   ✅ 自适应latent调整")
    print(f"   ✅ mel-spectrogram时间平滑")
    print(f"   ✅ 智能音量匹配")
    print(f"   ✅ 轻微后处理滤波")
    
    # 质量评估
    if quality_score > 8:
        print(f"🎉 V3重建质量优秀！")
    elif quality_score > 5:
        print(f"✅ V3重建质量良好")
    else:
        print(f"⚠️ V3重建质量需要改进")
    
    return {
        'snr': snr,
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'quality_score': quality_score,
        'output_file': str(reconstructed_path),
        'vocoder_method': vocoder_method
    }


def test_audioldm2_v4_highfreq_fix(audio_path, max_length=10.0):
    """
    V4版本: 高频信号修复版本
    专门解决mel频谱图高频信号丢失问题
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🎯 V4: AudioLDM2 高频信号修复版本")
    print(f"📱 设备: {device}")
    
    # 加载AudioLDM2 pipeline
    print("📦 加载AudioLDM2模型...")
    pipeline = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    print(f"✅ 模型加载完成")
    print(f"   VAE scaling_factor: {pipeline.vae.config.scaling_factor}")
    print(f"   Vocoder类型: {type(pipeline.vocoder).__name__}")
    
    # 获取feature extractor参数
    fe_sr = pipeline.feature_extractor.sampling_rate
    print(f"   FeatureExtractor采样率: {fe_sr} Hz")
    
    # V4改进1: 检查ClapFeatureExtractor的频率范围
    print(f"   ClapFeatureExtractor参数检查:")
    if hasattr(pipeline.feature_extractor, 'feature_extractor'):
        inner_extractor = pipeline.feature_extractor.feature_extractor
        print(f"   - fmin: {getattr(inner_extractor, 'fmin', 'N/A')}")
        print(f"   - fmax: {getattr(inner_extractor, 'fmax', 'N/A')}")
        print(f"   - n_mels: {getattr(inner_extractor, 'n_mels', 'N/A')}")
        print(f"   - hop_length: {getattr(inner_extractor, 'hop_length', 'N/A')}")
    
    # 加载音频
    print(f"📁 加载音频: {Path(audio_path).name}")
    
    # 使用feature extractor的采样率
    audio_fe_sr, sr_fe = librosa.load(audio_path, sr=fe_sr, duration=max_length)
    audio_16k, sr_16k = librosa.load(audio_path, sr=16000, duration=max_length)
    
    print(f"   {fe_sr}Hz音频: {len(audio_fe_sr)/sr_fe:.2f}秒, 范围[{audio_fe_sr.min():.3f}, {audio_fe_sr.max():.3f}]")
    print(f"   16kHz音频: {len(audio_16k)/sr_16k:.2f}秒, 范围[{audio_16k.min():.3f}, {audio_16k.max():.3f}]")
    
    # V4特色: 高频保护的特征提取
    print("🎵 V4: 高频保护的ClapFeatureExtractor处理...")
    
    # 先获取原始mel-spectrogram作为参考
    print("   🔍 分析原始音频的频谱特征...")
    original_mel = librosa.feature.melspectrogram(
        y=audio_fe_sr, 
        sr=fe_sr, 
        n_mels=64,
        hop_length=480,  # 使用AudioLDM2的hop_length
        n_fft=1024,
        fmin=0,  # 从0Hz开始
        fmax=fe_sr // 2,  # 到Nyquist频率
        window='hann',
        power=2.0
    )
    original_mel_db = librosa.power_to_db(original_mel, ref=np.max)
    
    print(f"   原始mel频谱: {original_mel.shape}")
    print(f"   原始mel范围: [{original_mel_db.min():.1f}, {original_mel_db.max():.1f}] dB")
    
    # 分析高频能量分布
    high_freq_bins = original_mel_db[48:, :]  # 高频部分（大约75%以上）
    mid_freq_bins = original_mel_db[16:48, :]  # 中频部分
    low_freq_bins = original_mel_db[:16, :]    # 低频部分
    
    print(f"   高频能量 (75-100%): 平均 {high_freq_bins.mean():.1f} dB, 最大 {high_freq_bins.max():.1f} dB")
    print(f"   中频能量 (25-75%): 平均 {mid_freq_bins.mean():.1f} dB, 最大 {mid_freq_bins.max():.1f} dB")
    print(f"   低频能量 (0-25%): 平均 {low_freq_bins.mean():.1f} dB, 最大 {low_freq_bins.max():.1f} dB")
    
    try:
        # V4改进2: 保护高频的音频预处理
        audio_input = audio_fe_sr.copy()
        
        # 温和的归一化，避免损失高频信息
        peak_value = np.max(np.abs(audio_input))
        if peak_value > 0:
            # 更保守的归一化，保持动态范围
            audio_input = audio_input / peak_value * 0.98
        
        features = pipeline.feature_extractor(
            audio_input, 
            return_tensors="pt", 
            sampling_rate=fe_sr
        )
        
        mel_input = features.input_features.to(device)
        if device == "cuda":
            mel_input = mel_input.half()
        
        print(f"   ✅ V4 ClapFeatureExtractor成功")
        print(f"   输入: {mel_input.shape} (格式: [batch, channel, time, feature])")
        print(f"   范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
        
        # V4改进3: 分析ClapFeatureExtractor的输出
        mel_clap = mel_input.squeeze().cpu().numpy()
        if mel_clap.ndim == 3:
            mel_clap = mel_clap[0]  # 取第一个通道
        
        print(f"   CLAP输出分析:")
        print(f"   - 形状: {mel_clap.shape}")
        print(f"   - 高频部分 (75-100%): 平均 {mel_clap[48:, :].mean():.1f}, 最大 {mel_clap[48:, :].max():.1f}")
        print(f"   - 中频部分 (25-75%): 平均 {mel_clap[16:48, :].mean():.1f}, 最大 {mel_clap[16:48, :].max():.1f}")
        print(f"   - 低频部分 (0-25%): 平均 {mel_clap[:16, :].mean():.1f}, 最大 {mel_clap[:16, :].max():.1f}")
        
        use_clap_features = True
        
    except Exception as e:
        print(f"   ❌ V4 ClapFeatureExtractor失败: {e}")
        use_clap_features = False
    
    # 如果失败，使用改进的传统方法
    if not use_clap_features:
        print("   🔄 V4: 使用高频保护的传统方法...")
        
        # V4改进4: 高频保护的mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_fe_sr, 
            sr=fe_sr, 
            n_mels=64,
            hop_length=480,  # 匹配AudioLDM2
            n_fft=1024,
            fmin=0,  # 从0Hz开始，不丢失低频
            fmax=fe_sr // 2,  # 到Nyquist频率，保持高频
            window='hann',
            center=True,
            pad_mode='reflect',
            power=2.0
        )
        
        # 更保守的dB转换，保持动态范围
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=120)  # 增加动态范围
        
        # 改进的归一化，保持高频信息
        mel_db_normalized = (mel_db + 120) / 120 * 2 - 1
        
        mel_tensor = torch.from_numpy(mel_db_normalized).to(device)
        if device == "cuda":
            mel_tensor = mel_tensor.half()
        
        mel_input = mel_tensor.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        
        print(f"   传统处理: {mel_input.shape}")
        print(f"   范围: [{mel_input.min():.3f}, {mel_input.max():.3f}]")
    
    print(f"   最终输入: {mel_input.shape}, {mel_input.dtype}")
    
    # V4 VAE处理 - 高频保护
    print("\n🧠 V4: VAE编码解码 (高频保护)...")
    with torch.no_grad():
        # 编码
        latent_dist = pipeline.vae.encode(mel_input)
        
        # V4改进5: 使用更稳定的采样策略
        if hasattr(latent_dist, 'latent_dist'):
            latent = latent_dist.latent_dist.mode()  # 使用确定性采样
        else:
            latent = latent_dist.mode() if hasattr(latent_dist, 'mode') else latent_dist.sample()
        
        # 应用scaling_factor
        latent = latent * pipeline.vae.config.scaling_factor
        print(f"   V4编码latent: {latent.shape}")
        print(f"   Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
        
        # V4改进6: 温和的latent调整，避免损失高频信息
        latent_std = torch.std(latent)
        if latent_std > 5.0:  # 更宽松的阈值
            latent = latent * (4.5 / latent_std)
            print(f"   V4 Latent轻微调整: std {latent_std:.3f} -> 4.5")
        
        # 解码
        latent_for_decode = latent / pipeline.vae.config.scaling_factor
        reconstructed_mel = pipeline.vae.decode(latent_for_decode).sample
        
        print(f"   V4解码输出: {reconstructed_mel.shape}")
        print(f"   解码范围: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
        
        # V4改进7: 分析重建mel的频谱特征
        recon_mel = reconstructed_mel.squeeze().cpu().numpy()
        if recon_mel.ndim == 3:
            recon_mel = recon_mel[0]  # 取第一个通道
        
        print(f"   重建mel分析:")
        print(f"   - 形状: {recon_mel.shape}")
        print(f"   - 高频部分 (75-100%): 平均 {recon_mel[48:, :].mean():.3f}, 最大 {recon_mel[48:, :].max():.3f}")
        print(f"   - 中频部分 (25-75%): 平均 {recon_mel[16:48, :].mean():.3f}, 最大 {recon_mel[16:48, :].max():.3f}")
        print(f"   - 低频部分 (0-25%): 平均 {recon_mel[:16, :].mean():.3f}, 最大 {recon_mel[:16, :].max():.3f}")
        
        # 检查高频信息丢失
        high_freq_loss = np.mean(recon_mel[48:, :] == recon_mel[48:, :].min())
        print(f"   - 高频信息丢失率: {high_freq_loss*100:.1f}%")
        
        if high_freq_loss > 0.5:
            print(f"   ⚠️ 检测到严重高频信息丢失！")
        elif high_freq_loss > 0.1:
            print(f"   ⚠️ 检测到轻微高频信息丢失")
        else:
            print(f"   ✅ 高频信息保持良好")
    
    # V4 HiFiGAN处理 - 高频优化
    print("\n🎤 V4: HiFiGAN vocoder (高频优化)...")
    
    # V4改进8: 高频增强的vocoder处理
    try:
        print("   🚀 V4策略: 高频增强的vocoder处理...")
        
        # 检查是否需要高频增强
        if high_freq_loss > 0.1:
            print("   🔧 应用高频增强...")
            
            # 方法1: 高频部分适度增强
            mel_enhanced = reconstructed_mel.clone()
            
            # 对高频部分进行轻微增强
            high_freq_mask = torch.zeros_like(mel_enhanced)
            high_freq_mask[:, :, :, 48:] = 1.0  # 高频部分
            
            # 计算高频增强因子
            high_freq_mean = torch.mean(mel_enhanced * high_freq_mask)
            mid_freq_mean = torch.mean(mel_enhanced * (1 - high_freq_mask))
            
            if high_freq_mean < mid_freq_mean - 2.0:  # 如果高频明显低于中频
                enhancement_factor = 1.0 + min(0.5, (mid_freq_mean - high_freq_mean) / 10.0)
                mel_enhanced = mel_enhanced * (1 + high_freq_mask * (enhancement_factor - 1))
                print(f"   高频增强因子: {enhancement_factor:.3f}")
            
            waveform = pipeline.mel_spectrogram_to_waveform(mel_enhanced)
        else:
            waveform = pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
        
        reconstructed_audio = waveform.squeeze().cpu().detach().numpy()
        vocoder_method = "V4_AudioLDM2_HighFreq_Enhanced"
        print(f"   ✅ V4成功！输出: {len(reconstructed_audio)}样本")
        
    except Exception as e:
        print(f"   ❌ V4策略失败: {e}")
        
        # 回退到标准方法
        try:
            print("   🔄 V4回退: 标准pipeline方法...")
            waveform = pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
            reconstructed_audio = waveform.squeeze().cpu().detach().numpy()
            vocoder_method = "V4_AudioLDM2_Pipeline_Standard"
            print(f"   ✅ V4回退成功！输出: {len(reconstructed_audio)}样本")
            
        except Exception as e2:
            print(f"   ❌ V4回退失败: {e2}")
            return None
    
    # V4后处理 - 高频保护
    print("\n🔧 V4: 高频保护后处理...")
    
    # 长度匹配
    reference_audio = audio_16k
    if len(reconstructed_audio) > len(reference_audio):
        reconstructed_audio = reconstructed_audio[:len(reference_audio)]
    elif len(reconstructed_audio) < len(reference_audio):
        reconstructed_audio = np.pad(reconstructed_audio, (0, len(reference_audio) - len(reconstructed_audio)))
    
    # V4改进9: 保护高频的音量匹配
    ref_rms = np.sqrt(np.mean(reference_audio ** 2))
    rec_rms = np.sqrt(np.mean(reconstructed_audio ** 2))
    
    if rec_rms > 0:
        volume_ratio = ref_rms / rec_rms
        # 限制音量调整范围，避免过度放大损失高频
        volume_ratio = np.clip(volume_ratio, 0.3, 3.0)
        reconstructed_audio = reconstructed_audio * volume_ratio
        print(f"   V4音量匹配: {rec_rms:.4f} -> {ref_rms:.4f} (比例: {volume_ratio:.2f})")
    
    # V4改进10: 高频分析
    print("   🔍 V4高频分析...")
    
    # 分析重建音频的频谱
    recon_spec = np.abs(np.fft.fft(reconstructed_audio[:8192]))[:4096]  # 取前8192样本做FFT
    ref_spec = np.abs(np.fft.fft(reference_audio[:8192]))[:4096]
    
    # 计算高频能量比
    high_freq_energy_ref = np.sum(ref_spec[2048:])  # 高频部分
    high_freq_energy_recon = np.sum(recon_spec[2048:])
    
    total_energy_ref = np.sum(ref_spec)
    total_energy_recon = np.sum(recon_spec)
    
    if total_energy_ref > 0 and total_energy_recon > 0:
        high_freq_ratio_ref = high_freq_energy_ref / total_energy_ref
        high_freq_ratio_recon = high_freq_energy_recon / total_energy_recon
        
        print(f"   原始音频高频能量比: {high_freq_ratio_ref:.3f}")
        print(f"   重建音频高频能量比: {high_freq_ratio_recon:.3f}")
        print(f"   高频保持率: {high_freq_ratio_recon/high_freq_ratio_ref*100:.1f}%")
    
    # 保存结果
    output_dir = Path("vae_hifigan_ultimate_fix")
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(audio_path).stem
    timestamp = int(time.time())
    
    original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
    reconstructed_path = output_dir / f"{input_name}_V4_{vocoder_method}_{timestamp}.wav"
    
    print("\n💾 V4保存结果...")
    save_audio_compatible(reference_audio, original_path)
    save_audio_compatible(reconstructed_audio, reconstructed_path)
    
    # 计算质量指标
    mse = np.mean((reference_audio - reconstructed_audio) ** 2)
    snr = 10 * np.log10(np.mean(reference_audio ** 2) / (mse + 1e-10))
    correlation = np.corrcoef(reference_audio, reconstructed_audio)[0, 1] if len(reference_audio) > 1 else 0
    mae = np.mean(np.abs(reference_audio - reconstructed_audio))
    
    # V4综合质量分数（考虑高频保持）
    high_freq_score = high_freq_ratio_recon / high_freq_ratio_ref if high_freq_ratio_ref > 0 else 0
    quality_score = snr + correlation * 8 + high_freq_score * 5
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"🎯 V4结果 (AudioLDM2 高频修复版本)")
    print(f"{'='*60}")
    print(f"📁 原始音频: {original_path}")
    print(f"📁 重建音频: {reconstructed_path}")
    print(f"📊 SNR: {snr:.2f} dB")
    print(f"📊 MSE: {mse:.6f}")
    print(f"📊 MAE: {mae:.6f}")
    print(f"📊 相关系数: {correlation:.4f}")
    print(f"📊 高频保持分数: {high_freq_score:.3f}")
    print(f"🏆 V4综合质量分数: {quality_score:.2f}")
    print(f"🎤 重建方法: {vocoder_method}")
    
    # V4诊断分析
    print(f"\n🔬 V4高频修复特色:")
    print(f"   ✅ 高频保护的音频预处理")
    print(f"   ✅ 频谱分析和监控")
    print(f"   ✅ 高频信息丢失检测")
    print(f"   ✅ 自适应高频增强")
    print(f"   ✅ 高频保护的音量匹配")
    print(f"   ✅ 详细的频谱分析")
    
    # 质量评估
    if high_freq_score > 0.8:
        print(f"🎉 V4高频保持优秀！")
    elif high_freq_score > 0.6:
        print(f"✅ V4高频保持良好")
    elif high_freq_score > 0.3:
        print(f"⚠️ V4高频有一定损失")
    else:
        print(f"❌ V4高频损失严重")
    
    return {
        'snr': snr,
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'quality_score': quality_score,
        'high_freq_score': high_freq_score,
        'output_file': str(reconstructed_path),
        'vocoder_method': vocoder_method
    }


def main():
    """主函数：自动处理AudioLDM2_Music_output.wav，并提供多个版本对比"""
    input_file = "AudioLDM2_Music_output.wav"
    
    print("🎵 AudioLDM2 VAE重建多版本对比")
    print("=" * 50)
    print("📝 版本说明:")
    print("   V1 (推荐): AudioLDM2 Pipeline Standard improved - 最佳听感和信号保真度")
    print("   V3 (新): 平衡优化版本 - 精细化参数调整和处理策略")
    print("   V4 (高频修复): 专门针对高频信号丢失问题")
    print("=" * 50)
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件 {input_file} 不存在")
        print("请确保AudioLDM2_Music_output.wav在当前目录中")
        return
    
    print(f"🎵 自动处理文件: {input_file}")
    
    # 运行V1版本（推荐）
    print("\n" + "="*60)
    print("🔧 运行V1版本 (推荐)")
    print("="*60)
    try:
        v1_result = test_audioldm2_ultimate_fix(input_file)
        if v1_result:
            print(f"✅ V1处理成功")
        else:
            print(f"❌ V1处理失败")
    except Exception as e:
        print(f"❌ V1处理失败: {e}")
        v1_result = None
    
    # 运行V3版本（新的平衡优化）
    print("\n" + "="*60)
    print("🔧 运行V3版本 (平衡优化)")
    print("="*60)
    try:
        v3_result = test_audioldm2_v3_balanced(input_file)
        if v3_result:
            print(f"✅ V3处理成功")
        else:
            print(f"❌ V3处理失败")
    except Exception as e:
        print(f"❌ V3处理失败: {e}")
        v3_result = None
    
    # 运行V4版本（高频修复）
    print("\n" + "="*60)
    print("🔧 运行V4版本 (高频修复)")
    print("="*60)
    try:
        v4_result = test_audioldm2_v4_highfreq_fix(input_file)
        if v4_result:
            print(f"✅ V4处理成功")
        else:
            print(f"❌ V4处理失败")
    except Exception as e:
        print(f"❌ V4处理失败: {e}")
        v4_result = None
    
    # 结果对比
    print("\n" + "="*60)
    print("� 版本对比结果")
    print("="*60)
    
    results = []
    if v1_result:
        results.append(("V1 (推荐)", v1_result))
    if v3_result:
        results.append(("V3 (平衡优化)", v3_result))
    if v4_result:
        results.append(("V4 (高频修复)", v4_result))
    
    if results:
        print(f"{'版本':<15} {'SNR(dB)':<10} {'相关性':<10} {'质量分数':<10} {'输出文件'}")
        print("-" * 80)
        for name, result in results:
            print(f"{name:<15} {result['snr']:<10.2f} {result['correlation']:<10.4f} {result['quality_score']:<10.2f} {Path(result['output_file']).name}")
        
        # 推荐最佳版本
        best_result = max(results, key=lambda x: x[1]['quality_score'])
        print(f"\n🏆 推荐使用: {best_result[0]}")
        print(f"   质量分数: {best_result[1]['quality_score']:.2f}")
        print(f"   输出文件: {best_result[1]['output_file']}")
        
        # 详细建议
        if best_result[0] == "V1 (推荐)":
            print(f"\n💡 V1继续保持最佳效果，建议使用V1结果")
        elif best_result[0] == "V3 (平衡优化)":
            print(f"\n🎉 V3平衡优化版本表现更好！")
            print(f"   V3的改进包括：动态音量调整、混合采样、智能后处理等")
        elif best_result[0] == "V4 (高频修复)":
            print(f"\n🎉 V4高频修复版本表现出色！")
            print(f"   V4专注于恢复高频细节，适合对高频要求较高的场景")
        
    else:
        print("❌ 所有版本都处理失败")
    
    print(f"\n✅ 处理完成！请检查输出文件并进行主观听感测试")
    print(f"💡 建议：客观指标仅供参考，最终效果请以主观听感为准")


if __name__ == "__main__":
    main()
