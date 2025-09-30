#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析AudioLDM2的内部mel频谱处理方式
"""

from diffusers import AudioLDM2Pipeline
import torch
import numpy as np
import librosa

def analyze_audioldm2_processing():
    """分析AudioLDM2的mel频谱处理"""
    print("🔍 分析AudioLDM2内部处理方式...")
    
    # 加载pipeline
    pipe = AudioLDM2Pipeline.from_pretrained('cvssp/audioldm2', torch_dtype=torch.float32)
    
    print("\n📦 Pipeline组件:")
    print(f"- Feature extractor: {hasattr(pipe, 'feature_extractor')}")
    print(f"- Mel spectrogram: {hasattr(pipe, 'mel_spectrogram')}")
    print(f"- Audio processor: {hasattr(pipe, 'audio_processor')}")
    
    print("\n🔧 Pipeline方法:")
    for attr in dir(pipe):
        if 'mel' in attr.lower() or 'audio' in attr.lower():
            print(f"- {attr}")
    
    print("\n⚙️ VAE配置:")
    print(f"- in_channels: {pipe.vae.config.in_channels}")
    print(f"- out_channels: {pipe.vae.config.out_channels}")
    print(f"- sample_size: {pipe.vae.config.sample_size}")
    print(f"- scaling_factor: {pipe.vae.config.scaling_factor}")
    
    print("\n🎤 Vocoder配置:")
    print(f"- model_in_dim: {pipe.vocoder.config.model_in_dim}")
    print(f"- sampling_rate: {pipe.vocoder.config.sampling_rate}")
    
    # 检查feature_extractor
    if hasattr(pipe, 'feature_extractor'):
        print("\n🎵 Feature Extractor:")
        fe = pipe.feature_extractor
        print(f"- Class: {type(fe).__name__}")
        print(f"- Config: {fe}")
    
    # 测试mel_spectrogram_to_waveform方法
    print("\n🧪 测试mel_spectrogram_to_waveform:")
    try:
        # 创建一个测试mel频谱
        test_mel = torch.randn(1, 1, 64, 100)
        print(f"- 输入: {test_mel.shape}")
        
        # 调用方法
        waveform = pipe.mel_spectrogram_to_waveform(test_mel)
        print(f"- 输出: {waveform.shape}")
        print(f"- 方法工作正常")
        
    except Exception as e:
        print(f"- 方法失败: {e}")


if __name__ == "__main__":
    analyze_audioldm2_processing()
