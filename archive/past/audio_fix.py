#!/usr/bin/env python3
"""
音频文件诊断和修复工具
检查生成的WAV文件是否可以正常播放，并修复常见问题
"""

import torch
import torchaudio
import numpy as np
import os
import sys
from pathlib import Path
import librosa
import soundfile as sf


def diagnose_audio_file(file_path):
    """诊断音频文件的问题"""
    print(f"\n🔍 诊断音频文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    file_size = os.path.getsize(file_path)
    print(f"   文件大小: {file_size} 字节")
    
    if file_size == 0:
        print(f"❌ 文件为空")
        return None
    
    # 尝试用不同方法加载音频
    issues = []
    audio_data = None
    sample_rate = None
    
    # 方法1: torchaudio
    try:
        audio_torch, sr_torch = torchaudio.load(file_path)
        print(f"   ✅ torchaudio加载成功: 形状={audio_torch.shape}, 采样率={sr_torch}")
        audio_data = audio_torch.numpy()
        sample_rate = sr_torch
    except Exception as e:
        print(f"   ❌ torchaudio加载失败: {e}")
        issues.append("torchaudio_load_failed")
    
    # 方法2: librosa
    try:
        audio_librosa, sr_librosa = librosa.load(file_path, sr=None)
        print(f"   ✅ librosa加载成功: 长度={len(audio_librosa)}, 采样率={sr_librosa}")
        if audio_data is None:
            audio_data = audio_librosa
            sample_rate = sr_librosa
    except Exception as e:
        print(f"   ❌ librosa加载失败: {e}")
        issues.append("librosa_load_failed")
    
    # 方法3: soundfile
    try:
        audio_sf, sr_sf = sf.read(file_path)
        print(f"   ✅ soundfile加载成功: 形状={np.array(audio_sf).shape}, 采样率={sr_sf}")
        if audio_data is None:
            audio_data = audio_sf
            sample_rate = sr_sf
    except Exception as e:
        print(f"   ❌ soundfile加载失败: {e}")
        issues.append("soundfile_load_failed")
    
    if audio_data is None:
        print(f"❌ 所有加载方法都失败")
        return None
    
    # 分析音频数据
    audio_flat = audio_data.flatten() if audio_data.ndim > 1 else audio_data
    
    print(f"   音频长度: {len(audio_flat)} 样本 ({len(audio_flat)/sample_rate:.2f} 秒)")
    print(f"   数据类型: {audio_flat.dtype}")
    print(f"   数值范围: [{np.min(audio_flat):.6f}, {np.max(audio_flat):.6f}]")
    print(f"   RMS: {np.sqrt(np.mean(audio_flat**2)):.6f}")
    
    # 检查常见问题
    if np.all(audio_flat == 0):
        issues.append("all_zeros")
        print(f"   ⚠️ 音频全为零值")
    
    if not np.isfinite(audio_flat).all():
        issues.append("invalid_values")
        nan_count = np.isnan(audio_flat).sum()
        inf_count = np.isinf(audio_flat).sum()
        print(f"   ⚠️ 包含无效值: NaN={nan_count}, Inf={inf_count}")
    
    if np.max(np.abs(audio_flat)) > 1.0:
        issues.append("clipping")
        print(f"   ⚠️ 可能存在削波: 最大幅度={np.max(np.abs(audio_flat)):.6f}")
    
    if np.max(np.abs(audio_flat)) < 1e-6:
        issues.append("too_quiet")
        print(f"   ⚠️ 音频过于安静: 最大幅度={np.max(np.abs(audio_flat)):.6f}")
    
    if sample_rate not in [8000, 16000, 22050, 44100, 48000]:
        issues.append("unusual_sample_rate")
        print(f"   ⚠️ 非标准采样率: {sample_rate}")
    
    return {
        'file_path': file_path,
        'audio_data': audio_flat,
        'sample_rate': sample_rate,
        'issues': issues,
        'file_size': file_size
    }


def fix_audio_file(file_path, output_path=None):
    """修复音频文件"""
    print(f"\n🔧 修复音频文件: {file_path}")
    
    diagnosis = diagnose_audio_file(file_path)
    if diagnosis is None:
        print(f"❌ 无法诊断文件，修复失败")
        return False
    
    audio_data = diagnosis['audio_data']
    sample_rate = diagnosis['sample_rate']
    issues = diagnosis['issues']
    
    if not issues:
        print(f"✅ 文件没有检测到问题")
        return True
    
    print(f"   检测到问题: {issues}")
    
    # 修复音频数据
    fixed_audio = audio_data.copy()
    
    # 修复无效值
    if 'invalid_values' in issues:
        print(f"   🔧 修复无效值...")
        fixed_audio = np.nan_to_num(fixed_audio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 修复全零问题
    if 'all_zeros' in issues:
        print(f"   🔧 无法修复全零音频")
        return False
    
    # 修复音量问题
    if 'too_quiet' in issues:
        print(f"   🔧 增加音频音量...")
        max_val = np.max(np.abs(fixed_audio))
        if max_val > 0:
            fixed_audio = fixed_audio / max_val * 0.5
    
    # 修复削波问题
    if 'clipping' in issues:
        print(f"   🔧 修复削波...")
        fixed_audio = np.clip(fixed_audio, -1.0, 1.0)
    
    # 确保合理的采样率
    if 'unusual_sample_rate' in issues:
        print(f"   🔧 调整采样率到16000Hz...")
        sample_rate = 16000
    
    # 保存修复后的文件
    if output_path is None:
        base_path = Path(file_path)
        output_path = base_path.parent / f"{base_path.stem}_fixed{base_path.suffix}"
    
    try:
        # 使用多种方法保存以确保兼容性
        
        # 方法1: soundfile (推荐)
        sf.write(str(output_path), fixed_audio, sample_rate, subtype='PCM_16')
        print(f"   ✅ 使用soundfile保存成功: {output_path}")
        
        # 验证保存的文件
        test_audio, test_sr = sf.read(str(output_path))
        print(f"   ✅ 验证成功: 长度={len(test_audio)}, 采样率={test_sr}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ soundfile保存失败: {e}")
        
        # 方法2: librosa + soundfile
        try:
            sf.write(str(output_path), fixed_audio.astype(np.float32), sample_rate)
            print(f"   ✅ 使用float32保存成功: {output_path}")
            return True
        except Exception as e2:
            print(f"   ❌ float32保存也失败: {e2}")
            return False


def fix_all_generated_files():
    """修复所有生成的音频文件"""
    print("🎯 批量修复生成的音频文件")
    
    # 查找所有输出目录
    output_dirs = [
        "vae_quick_test",
        "vae_final_noise_fix", 
        "vae_noise_fix_v2_test",
        "vae_improved_test",
        "vae_ultimate_test"
    ]
    
    fixed_count = 0
    total_count = 0
    
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            continue
        
        print(f"\n📁 处理目录: {output_dir}")
        wav_files = list(Path(output_dir).glob("*.wav"))
        
        for wav_file in wav_files:
            total_count += 1
            print(f"\n--- 文件 {total_count}: {wav_file.name} ---")
            
            diagnosis = diagnose_audio_file(str(wav_file))
            if diagnosis and diagnosis['issues']:
                success = fix_audio_file(str(wav_file))
                if success:
                    fixed_count += 1
            elif diagnosis:
                print(f"   ✅ 文件正常，无需修复")
            else:
                print(f"   ❌ 文件严重损坏，无法修复")
    
    print(f"\n{'='*50}")
    print(f"批量修复完成")
    print(f"总文件数: {total_count}")
    print(f"修复文件数: {fixed_count}")
    print(f"{'='*50}")


def create_test_audio():
    """创建测试音频文件"""
    print("\n🎵 创建测试音频文件")
    
    # 生成简单的测试音频
    sample_rate = 16000
    duration = 3  # 3秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成混合频率的正弦波
    frequencies = [440, 880, 1320]  # A4, A5, E6
    audio = np.zeros_like(t)
    
    for i, freq in enumerate(frequencies):
        audio += 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t / (duration / (i + 1)))
    
    # 添加一些噪声
    audio += 0.05 * np.random.randn(len(t))
    
    # 确保在合理范围内
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # 保存测试音频
    test_path = "test_audio_reference.wav"
    sf.write(test_path, audio, sample_rate, subtype='PCM_16')
    
    print(f"   ✅ 创建测试音频: {test_path}")
    print(f"   长度: {len(audio)} 样本 ({duration} 秒)")
    print(f"   采样率: {sample_rate} Hz")
    print(f"   幅度范围: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
    
    # 验证可以加载
    verify_audio, verify_sr = sf.read(test_path)
    print(f"   ✅ 验证加载成功: 长度={len(verify_audio)}, 采样率={verify_sr}")
    
    return test_path


def main():
    """主函数"""
    print("🎧 音频文件诊断和修复工具")
    
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("1. 诊断单个文件: python audio_fix.py <音频文件路径>")
        print("2. 批量修复所有文件: python audio_fix.py --fix-all")
        print("3. 创建测试音频: python audio_fix.py --create-test")
        
        choice = input("\n请选择操作 (1/2/3): ").strip()
        
        if choice == "1":
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac']:
                audio_files.extend(Path('.').glob(f'**/*{ext}'))
            
            if audio_files:
                print(f"\n找到音频文件:")
                for i, file in enumerate(audio_files[:20], 1):  # 只显示前20个
                    print(f"{i}. {file}")
                
                try:
                    file_choice = input("请选择文件序号: ").strip()
                    file_idx = int(file_choice) - 1
                    if 0 <= file_idx < len(audio_files):
                        audio_path = str(audio_files[file_idx])
                        diagnose_audio_file(audio_path)
                        
                        fix_choice = input("\n是否修复此文件? (y/n): ").strip().lower()
                        if fix_choice == 'y':
                            fix_audio_file(audio_path)
                    else:
                        print("无效选择")
                except (ValueError, KeyboardInterrupt):
                    print("取消操作")
            else:
                print("未找到音频文件")
        
        elif choice == "2":
            fix_all_generated_files()
        
        elif choice == "3":
            create_test_audio()
        
        else:
            print("无效选择")
            
    elif sys.argv[1] == "--fix-all":
        fix_all_generated_files()
    elif sys.argv[1] == "--create-test":
        create_test_audio()
    else:
        audio_path = sys.argv[1]
        diagnose_audio_file(audio_path)
        
        fix_choice = input("\n是否修复此文件? (y/n): ").strip().lower()
        if fix_choice == 'y':
            fix_audio_file(audio_path)


if __name__ == "__main__":
    main()
