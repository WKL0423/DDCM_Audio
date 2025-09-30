"""
AudioLDM2 Vocoder 深度分析
=========================

分析AudioLDM2内置vocoder的结构和HiFiGAN集成问题
"""

import torch
from diffusers import AudioLDM2Pipeline
from transformers import SpeechT5HifiGan
import librosa
import numpy as np

def analyze_audioldm2_vocoder():
    """分析AudioLDM2内置vocoder"""
    print("🔍 AudioLDM2 Vocoder 深度分析")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载AudioLDM2
    print("📦 加载AudioLDM2...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # 分析vocoder
    vocoder = pipe.vocoder
    print(f"\n🎤 AudioLDM2内置Vocoder分析:")
    print(f"   类型: {type(vocoder)}")
    print(f"   模块: {vocoder.__module__}")
    
    # 检查配置
    if hasattr(vocoder, 'config'):
        config = vocoder.config
        print(f"   配置类型: {type(config)}")
        print(f"   配置内容: {config}")
    else:
        print("   ❌ 没有config属性")
    
    # 检查模型结构
    print(f"\n🔧 Vocoder模型结构:")
    for name, module in vocoder.named_children():
        print(f"   {name}: {type(module)}")
    
    # 检查参数
    total_params = sum(p.numel() for p in vocoder.parameters())
    print(f"   总参数量: {total_params:,}")
    
    # 检查输入输出期望
    print(f"\n📊 输入输出分析:")
    
    # 创建测试mel数据
    test_mel = torch.randn(1, 64, 100).to(device)  # [batch, mel_bins, time]
    if device == "cuda":
        test_mel = test_mel.half()
    
    try:
        print(f"   测试输入: {test_mel.shape} ({test_mel.dtype})")
        with torch.no_grad():
            output = vocoder(test_mel)
        print(f"   ✅ 输出成功: {output.shape} ({output.dtype})")
        print(f"   输入格式: [batch, mel_bins, time_frames]")
        
    except Exception as e:
        print(f"   ❌ 直接调用失败: {e}")
        
        # 尝试不同的输入格式
        formats_to_try = [
            ("transpose", test_mel.transpose(-2, -1)),  # [batch, time, mel_bins]
            ("squeeze", test_mel.squeeze(0)),  # [mel_bins, time]
            ("unsqueeze", test_mel.unsqueeze(1)),  # [batch, channels, mel_bins, time]
        ]
        
        for name, test_input in formats_to_try:
            try:
                print(f"   尝试格式 {name}: {test_input.shape}")
                with torch.no_grad():
                    output = vocoder(test_input)
                print(f"   ✅ {name}格式成功: {output.shape}")
                break
            except Exception as e:
                print(f"   ❌ {name}格式失败: {e}")
    
    return vocoder

def analyze_external_hifigan():
    """分析外部HiFiGAN"""
    print(f"\n🎤 外部HiFiGAN分析:")
    print("="*30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 加载外部HiFiGAN
        hifigan = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        
        print(f"   类型: {type(hifigan)}")
        print(f"   模块: {hifigan.__module__}")
        
        if hasattr(hifigan, 'config'):
            print(f"   配置: {hifigan.config}")
        
        # 测试输入
        test_mel = torch.randn(1, 80, 100).to(device)  # SpeechT5标准：80维mel
        
        try:
            print(f"   测试输入: {test_mel.shape}")
            with torch.no_grad():
                output = hifigan(test_mel)
            print(f"   ✅ 输出成功: {output.shape}")
            
        except Exception as e:
            print(f"   ❌ 调用失败: {e}")
            
    except Exception as e:
        print(f"❌ HiFiGAN加载失败: {e}")

def analyze_dimension_mismatch():
    """分析维度不匹配问题"""
    print(f"\n🔍 维度不匹配问题分析:")
    print("="*40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    hifigan = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    # 模拟VAE输出
    print("📊 模拟真实使用场景:")
    
    # 创建测试音频
    audio = np.random.randn(16000 * 3)  # 3秒音频
    
    # 创建mel频谱
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_mels=64, n_fft=1024, hop_length=256
    )
    mel_spec = librosa.power_to_db(mel_spec)
    mel_spec = 2.0 * (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) - 1.0
    
    mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(device)
    if device == "cuda":
        mel_tensor = mel_tensor.half()
    
    print(f"   原始mel: {mel_tensor.shape}")
    
    # VAE处理
    if mel_tensor.shape[-1] % 4 != 0:
        pad_length = 4 - (mel_tensor.shape[-1] % 4)
        mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_length))
    
    print(f"   填充后mel: {mel_tensor.shape}")
    
    with torch.no_grad():
        # VAE编码解码
        latent = pipe.vae.encode(mel_tensor).latent_dist.sample()
        reconstructed_mel = pipe.vae.decode(latent).sample
        
        print(f"   VAE输出: {reconstructed_mel.shape}")
        
        # 尝试AudioLDM2内置vocoder
        try:
            # 准备AudioLDM2 vocoder输入
            audioldm2_input = reconstructed_mel.squeeze(0).transpose(-2, -1).unsqueeze(0)
            print(f"   AudioLDM2输入: {audioldm2_input.shape}")
            audioldm2_output = pipe.vocoder(audioldm2_input)
            print(f"   ✅ AudioLDM2 vocoder成功: {audioldm2_output.shape}")
        except Exception as e:
            print(f"   ❌ AudioLDM2 vocoder失败: {e}")
        
        # 尝试外部HiFiGAN
        try:
            # 准备HiFiGAN输入 (需要80维)
            hifigan_input = reconstructed_mel.squeeze()
            if hifigan_input.dim() == 3:
                hifigan_input = hifigan_input.squeeze(0)
            
            print(f"   重建mel原始: {hifigan_input.shape}")
            
            # 转换到80维
            if hifigan_input.shape[0] != 80:
                hifigan_input = torch.nn.functional.interpolate(
                    hifigan_input.unsqueeze(0).unsqueeze(0),
                    size=(80, hifigan_input.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            
            hifigan_input = hifigan_input.unsqueeze(0)
            print(f"   HiFiGAN输入: {hifigan_input.shape}")
            
            hifigan_output = hifigan(hifigan_input)
            print(f"   ✅ HiFiGAN成功: {hifigan_output.shape}")
            
        except Exception as e:
            print(f"   ❌ HiFiGAN失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("🎯 AudioLDM2 + HiFiGAN 集成问题深度分析")
    print("="*60)
    
    # 分析AudioLDM2内置vocoder
    audioldm2_vocoder = analyze_audioldm2_vocoder()
    
    # 分析外部HiFiGAN
    analyze_external_hifigan()
    
    # 分析维度不匹配问题
    analyze_dimension_mismatch()
    
    print(f"\n💡 问题总结:")
    print("1. AudioLDM2内置的是SpeechT5HifiGan，但配置和接口可能不同")
    print("2. 外部HiFiGAN期望80维mel，而AudioLDM2 VAE输出可能是64维")
    print("3. 数据类型和维度转换需要精确匹配")
    print("4. AudioLDM2内置vocoder已经优化过，可能比外部HiFiGAN更适合")

if __name__ == "__main__":
    main()
