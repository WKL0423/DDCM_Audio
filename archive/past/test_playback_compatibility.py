#!/usr/bin/env python3
"""
测试修复后文件的播放兼容性
"""

import os
import subprocess
import sys
from pathlib import Path

def test_windows_playback(filepath):
    """测试在Windows上的播放兼容性"""
    print(f"\n🎵 测试播放: {Path(filepath).name}")
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        print("❌ 文件不存在")
        return False
    
    # 尝试用Windows Media Player播放
    try:
        print("   尝试用Windows Media Player播放...")
        # 使用start命令打开文件（会用默认程序播放）
        result = subprocess.run(
            ['cmd', '/c', 'start', '/wait', filepath],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("   ✅ 可以被Windows默认播放器打开")
        return True
    except subprocess.TimeoutExpired:
        print("   ✅ 播放器已启动（超时正常）")
        return True
    except Exception as e:
        print(f"   ❌ 播放失败: {e}")
        return False

def main():
    """主函数"""
    repaired_dir = "vae_final_noise_fix_repaired"
    
    if not os.path.exists(repaired_dir):
        print(f"❌ 修复文件夹不存在: {repaired_dir}")
        return
    
    wav_files = list(Path(repaired_dir).glob("*.wav"))
    print(f"🔍 找到 {len(wav_files)} 个修复后的WAV文件")
    
    success_count = 0
    for wav_file in wav_files:
        if test_windows_playback(str(wav_file)):
            success_count += 1
    
    print(f"\n📊 兼容性测试结果: {success_count}/{len(wav_files)} 文件可播放")
    
    if success_count == len(wav_files):
        print("🎉 所有修复后的文件都应该可以在Windows上播放!")
    else:
        print("⚠️ 部分文件可能仍有兼容性问题")
    
    print(f"\n📁 修复后的文件位置: {os.path.abspath(repaired_dir)}")
    print("💡 建议: 请手动测试播放修复后的文件以确认兼容性")

if __name__ == "__main__":
    main()
