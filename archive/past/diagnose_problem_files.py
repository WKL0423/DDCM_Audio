#!/usr/bin/env python3
"""
诊断 vae_final_noise_fix 文件夹中的问题文件
"""

import os
import sys
import numpy as np
import soundfile as sf
import torchaudio
import librosa
from pathlib import Path

def diagnose_file(filepath):
    """诊断单个音频文件"""
    print(f"\n{'='*60}")
    print(f"诊断文件: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print("❌ 文件不存在")
        return False
    
    print(f"文件大小: {os.path.getsize(filepath)} 字节")
    
    # 1. 使用 soundfile 检查
    print("\n--- soundfile 检查 ---")
    try:
        info = sf.info(filepath)
        print(f"✅ 文件信息: {info}")
        
        audio, sr = sf.read(filepath)
        print(f"✅ 读取成功: shape={audio.shape}, sr={sr}, dtype={audio.dtype}")
        print(f"   值范围: min={audio.min():.6f}, max={audio.max():.6f}")
        print(f"   是否包含NaN: {np.isnan(audio).any()}")
        print(f"   是否包含inf: {np.isinf(audio).any()}")
        
        # 检查音频数据特征
        if len(audio) > 0:
            print(f"   平均值: {np.mean(audio):.6f}")
            print(f"   标准差: {np.std(audio):.6f}")
            print(f"   RMS: {np.sqrt(np.mean(audio**2)):.6f}")
            
            # 检查是否有异常值
            outliers = np.abs(audio) > 1.0
            if np.any(outliers):
                print(f"   ⚠️ 警告: {np.sum(outliers)} 个样本超出 [-1, 1] 范围")
            
            # 检查是否全为零
            if np.all(audio == 0):
                print(f"   ⚠️ 警告: 音频全为零")
            
            # 检查动态范围
            dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-8))
            print(f"   动态范围: {dynamic_range:.2f} dB")
        
    except Exception as e:
        print(f"❌ soundfile 失败: {e}")
    
    # 2. 使用 torchaudio 检查
    print("\n--- torchaudio 检查 ---")
    try:
        audio, sr = torchaudio.load(filepath)
        print(f"✅ 读取成功: shape={audio.shape}, sr={sr}, dtype={audio.dtype}")
        print(f"   值范围: min={audio.min():.6f}, max={audio.max():.6f}")
    except Exception as e:
        print(f"❌ torchaudio 失败: {e}")
    
    # 3. 使用 librosa 检查
    print("\n--- librosa 检查 ---")
    try:
        audio, sr = librosa.load(filepath, sr=None)
        print(f"✅ 读取成功: shape={audio.shape}, sr={sr}, dtype={audio.dtype}")
        print(f"   值范围: min={audio.min():.6f}, max={audio.max():.6f}")
    except Exception as e:
        print(f"❌ librosa 失败: {e}")
    
    # 4. 检查文件头部
    print("\n--- 文件头部检查 ---")
    try:
        with open(filepath, 'rb') as f:
            header = f.read(44)  # WAV 文件头通常是 44 字节
            print(f"文件头部 (前16字节): {header[:16].hex()}")
            
            # 检查是否是有效的 WAV 文件
            if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                print("✅ 有效的 WAV 文件头")
            else:
                print("❌ 无效的 WAV 文件头")
    except Exception as e:
        print(f"❌ 头部检查失败: {e}")
    
    return True

def fix_file_with_soundfile(filepath, output_dir="vae_final_noise_fix_repaired"):
    """使用 soundfile 重新保存文件"""
    try:
        # 读取原文件
        audio, sr = sf.read(filepath)
        
        # 清理音频数据
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 限制到 [-1, 1] 范围
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # 确保是单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        filename = Path(filepath).name
        output_path = os.path.join(output_dir, f"repaired_{filename}")
        
        # 使用 soundfile 保存为高兼容性格式
        sf.write(output_path, audio, sr, subtype='PCM_16')
        
        print(f"✅ 修复文件保存到: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return None

def main():
    """主函数"""
    # 检查 vae_final_noise_fix 文件夹中的所有文件
    problem_dir = "vae_final_noise_fix"
    
    if not os.path.exists(problem_dir):
        print(f"❌ 目录不存在: {problem_dir}")
        return
    
    wav_files = list(Path(problem_dir).glob("*.wav"))
    print(f"找到 {len(wav_files)} 个 WAV 文件")
    
    for wav_file in wav_files:
        diagnose_file(str(wav_file))
        
        # 尝试修复
        print(f"\n--- 尝试修复 {wav_file.name} ---")
        fixed_path = fix_file_with_soundfile(str(wav_file))
        if fixed_path:
            print(f"修复后文件: {fixed_path}")
            # 验证修复后的文件
            print("验证修复后的文件:")
            diagnose_file(fixed_path)

if __name__ == "__main__":
    main()
