#!/usr/bin/env python3
"""
Windows音频兼容性修复工具
将音频文件转换为更兼容的格式，确保在Windows Media Player等播放器中正常播放
"""

import soundfile as sf
import numpy as np
import os
import sys
from pathlib import Path


def convert_to_windows_compatible(input_path, output_path=None):
    """
    将音频文件转换为Windows高度兼容的格式
    """
    print(f"🔄 转换音频文件: {input_path}")
    
    try:
        # 加载音频
        audio, sample_rate = sf.read(input_path)
        print(f"   原始: 长度={len(audio)}, 采样率={sample_rate}, 类型={audio.dtype}")
        
        # 确保是1D数组
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 转为单声道
        
        # 转换为int16格式（CD质量，兼容性最好）
        if sample_rate != 44100:
            # 重采样到44.1kHz（标准CD采样率）
            import librosa
            audio_resampled = librosa.resample(audio.astype(np.float32), 
                                             orig_sr=sample_rate, 
                                             target_sr=44100)
            sample_rate = 44100
            audio = audio_resampled
            print(f"   重采样到44.1kHz")
        
        # 确保在[-1, 1]范围内
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        elif max_val > 0 and max_val < 0.01:
            audio = audio / max_val * 0.5
        
        # 转换为int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 生成输出文件名
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_windows_compatible.wav"
        
        # 保存为16-bit WAV文件
        sf.write(str(output_path), audio_int16, sample_rate, subtype='PCM_16')
        
        print(f"   ✅ 转换成功: {output_path}")
        print(f"   输出: 长度={len(audio_int16)}, 采样率={sample_rate}, 类型=int16")
        
        # 验证文件
        verify_audio, verify_sr = sf.read(str(output_path))
        print(f"   ✅ 验证成功: 可以正常加载")
        
        return True, str(output_path)
        
    except Exception as e:
        print(f"   ❌ 转换失败: {e}")
        return False, None


def batch_convert_to_compatible():
    """批量转换所有VAE生成的音频文件"""
    print("🎯 批量转换为Windows兼容格式")
    
    # 查找所有输出目录
    output_dirs = [
        "vae_quick_test",
        "vae_final_noise_fix", 
        "vae_noise_fix_v2_test",
        "vae_improved_test",
        "vae_ultimate_test"
    ]
    
    converted_count = 0
    total_count = 0
    
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            continue
        
        print(f"\n📁 处理目录: {output_dir}")
        wav_files = list(Path(output_dir).glob("*.wav"))
        
        for wav_file in wav_files:
            # 跳过已经转换的文件
            if 'windows_compatible' in wav_file.name:
                continue
                
            total_count += 1
            print(f"\n--- 文件 {total_count}: {wav_file.name} ---")
            
            success, output_path = convert_to_windows_compatible(str(wav_file))
            if success:
                converted_count += 1
                
                # 建议用户测试播放
                print(f"   💡 请尝试播放: {output_path}")
    
    print(f"\n{'='*50}")
    print(f"批量转换完成")
    print(f"总文件数: {total_count}")
    print(f"转换成功: {converted_count}")
    print(f"{'='*50}")
    
    print(f"\n🎧 播放测试建议:")
    print(f"1. 用Windows Media Player打开 *_windows_compatible.wav 文件")
    print(f"2. 用VLC Player测试播放")
    print(f"3. 用任何音频编辑软件（如Audacity）打开")
    print(f"4. 双击文件应该能在默认播放器中打开")


def create_simple_test_tones():
    """创建简单的测试音调"""
    print("🎵 创建测试音调文件")
    
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 创建不同频率的纯音调
    frequencies = [220, 440, 880]  # A3, A4, A5
    names = ["low_tone", "mid_tone", "high_tone"]
    
    for freq, name in zip(frequencies, names):
        # 生成正弦波
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        # 添加淡入淡出
        fade_samples = int(0.1 * sample_rate)  # 0.1秒淡入淡出
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # 转换为int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 保存
        filename = f"test_{name}_{freq}hz.wav"
        sf.write(filename, audio_int16, sample_rate, subtype='PCM_16')
        
        print(f"   ✅ 创建: {filename} ({freq}Hz, {duration}s)")
    
    # 创建混合音调
    mixed_audio = np.zeros_like(t)
    for freq in frequencies:
        mixed_audio += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # 淡入淡出
    fade_samples = int(0.1 * sample_rate)
    mixed_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    mixed_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # 转换并保存
    mixed_int16 = (mixed_audio * 32767).astype(np.int16)
    sf.write("test_mixed_tones.wav", mixed_int16, sample_rate, subtype='PCM_16')
    
    print(f"   ✅ 创建: test_mixed_tones.wav (混合音调)")
    print(f"\n💡 如果这些测试文件能播放，说明系统音频功能正常")


def main():
    """主函数"""
    print("🎧 Windows音频兼容性修复工具")
    
    if len(sys.argv) < 2:
        print("\n选择操作:")
        print("1. 转换单个文件")
        print("2. 批量转换所有VAE生成的文件")
        print("3. 创建测试音调文件")
        
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == "1":
            # 查找音频文件
            audio_files = []
            for ext in ['.wav']:
                audio_files.extend(Path('.').glob(f'**/*{ext}'))
            
            if audio_files:
                print(f"\n找到音频文件:")
                for i, file in enumerate(audio_files[:20], 1):
                    print(f"{i}. {file}")
                
                try:
                    file_choice = input("请选择文件序号: ").strip()
                    file_idx = int(file_choice) - 1
                    if 0 <= file_idx < len(audio_files):
                        audio_path = str(audio_files[file_idx])
                        convert_to_windows_compatible(audio_path)
                    else:
                        print("无效选择")
                except (ValueError, KeyboardInterrupt):
                    print("取消操作")
            else:
                print("未找到音频文件")
        
        elif choice == "2":
            batch_convert_to_compatible()
        
        elif choice == "3":
            create_simple_test_tones()
        
        else:
            print("无效选择")
    
    elif sys.argv[1] == "--batch":
        batch_convert_to_compatible()
    elif sys.argv[1] == "--test-tones":
        create_simple_test_tones()
    else:
        # 转换指定文件
        input_file = sys.argv[1]
        convert_to_windows_compatible(input_file)


if __name__ == "__main__":
    main()
