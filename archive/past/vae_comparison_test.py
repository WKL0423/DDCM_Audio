#!/usr/bin/env python3
"""
AudioLDM2 VAE噪音修复对比测试
同时运行原版和修复版，生成对比音频文件

对比项目:
1. 原版simple_vae_test.py的输出
2. 修复版vae_final_noise_fix.py的输出
3. 质量指标对比分析
"""

import subprocess
import os
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import torchaudio
import torch


def run_original_test(audio_path, max_length):
    """运行原版测试"""
    print("🔄 运行原版VAE测试...")
    try:
        result = subprocess.run([
            sys.executable, "simple_vae_test.py", audio_path, str(max_length)
        ], capture_output=True, text=True, cwd=".", timeout=300)
        
        if result.returncode == 0:
            print("✅ 原版测试成功")
            return True, result.stdout
        else:
            print(f"❌ 原版测试失败: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ 原版测试异常: {e}")
        return False, str(e)


def run_fixed_test(audio_path, max_length):
    """运行修复版测试"""
    print("🔄 运行修复版VAE测试...")
    try:
        result = subprocess.run([
            sys.executable, "vae_final_noise_fix.py", audio_path, str(max_length)
        ], capture_output=True, text=True, cwd=".", timeout=300)
        
        if result.returncode == 0:
            print("✅ 修复版测试成功")
            return True, result.stdout
        else:
            print(f"❌ 修复版测试失败: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ 修复版测试异常: {e}")
        return False, str(e)


def extract_metrics_from_output(output_text):
    """从输出文本中提取质量指标"""
    metrics = {}
    lines = output_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if 'MSE:' in line:
            try:
                metrics['mse'] = float(line.split('MSE:')[1].strip())
            except:
                pass
        elif 'SNR:' in line and 'dB' in line:
            try:
                metrics['snr'] = float(line.split('SNR:')[1].split('dB')[0].strip())
            except:
                pass
        elif '相关系数:' in line:
            try:
                metrics['correlation'] = float(line.split('相关系数:')[1].strip())
            except:
                pass
        elif '编码时间:' in line:
            try:
                metrics['encode_time'] = float(line.split('编码时间:')[1].split('秒')[0].strip())
            except:
                pass
        elif '解码时间:' in line:
            try:
                metrics['decode_time'] = float(line.split('解码时间:')[1].split('秒')[0].strip())
            except:
                pass
        elif '压缩比:' in line:
            try:
                metrics['compression_ratio'] = float(line.split('压缩比:')[1].split(':')[0].strip())
            except:
                pass
        elif '重建方法:' in line:
            try:
                metrics['method'] = line.split('重建方法:')[1].strip()
            except:
                pass
    
    return metrics


def find_latest_outputs():
    """查找最新的输出文件"""
    original_dir = "vae_quick_test"
    fixed_dir = "vae_final_noise_fix"
    
    original_files = {'original': None, 'reconstructed': None}
    fixed_files = {'original': None, 'reconstructed': None}
    
    # 查找原版输出
    if os.path.exists(original_dir):
        files = list(Path(original_dir).glob("*.wav"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for f in files:
            if 'original' in f.name and original_files['original'] is None:
                original_files['original'] = str(f)
            elif 'reconstructed' in f.name and original_files['reconstructed'] is None:
                original_files['reconstructed'] = str(f)
    
    # 查找修复版输出
    if os.path.exists(fixed_dir):
        files = list(Path(fixed_dir).glob("*.wav"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for f in files:
            if 'original' in f.name and fixed_files['original'] is None:
                fixed_files['original'] = str(f)
            elif 'final_noisefixed' in f.name and fixed_files['reconstructed'] is None:
                fixed_files['reconstructed'] = str(f)
    
    return original_files, fixed_files


def analyze_audio_quality(file1, file2, label1, label2):
    """分析两个音频文件的质量差异"""
    if not file1 or not file2 or not os.path.exists(file1) or not os.path.exists(file2):
        return None
    
    try:
        # 加载音频
        audio1, sr1 = torchaudio.load(file1)
        audio2, sr2 = torchaudio.load(file2)
        
        # 确保采样率一致
        if sr1 != sr2:
            audio2 = torchaudio.functional.resample(audio2, sr2, sr1)
            sr2 = sr1
        
        # 转换为numpy
        audio1 = audio1.squeeze().numpy()
        audio2 = audio2.squeeze().numpy()
        
        # 确保长度一致
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # 计算质量指标
        mse = np.mean((audio1 - audio2) ** 2)
        
        # RMS功率
        rms1 = np.sqrt(np.mean(audio1 ** 2))
        rms2 = np.sqrt(np.mean(audio2 ** 2))
        
        # 动态范围
        dynamic_range1 = np.max(np.abs(audio1)) - np.min(np.abs(audio1))
        dynamic_range2 = np.max(np.abs(audio2)) - np.min(np.abs(audio2))
        
        # 频谱中心
        freqs = np.fft.fftfreq(len(audio1), 1/sr1)
        fft1 = np.abs(np.fft.fft(audio1))
        fft2 = np.abs(np.fft.fft(audio2))
        
        centroid1 = np.sum(freqs[:len(freqs)//2] * fft1[:len(freqs)//2]) / np.sum(fft1[:len(freqs)//2])
        centroid2 = np.sum(freqs[:len(freqs)//2] * fft2[:len(freqs)//2]) / np.sum(fft2[:len(freqs)//2])
        
        return {
            'mse_diff': mse,
            'rms_ratio': rms2 / rms1 if rms1 > 0 else 0,
            'dynamic_range_ratio': dynamic_range2 / dynamic_range1 if dynamic_range1 > 0 else 0,
            'spectral_centroid_diff': abs(centroid2 - centroid1),
            f'{label1}_rms': rms1,
            f'{label2}_rms': rms2,
            f'{label1}_dynamic_range': dynamic_range1,
            f'{label2}_dynamic_range': dynamic_range2
        }
        
    except Exception as e:
        print(f"   ⚠️ 音频分析失败: {e}")
        return None


def generate_comparison_report(original_metrics, fixed_metrics, audio_analysis, original_files, fixed_files):
    """生成对比报告"""
    report = []
    report.append("=" * 80)
    report.append("AudioLDM2 VAE 噪音修复对比报告")
    report.append("=" * 80)
    report.append("")
    
    # 文件信息
    report.append("📁 文件信息:")
    report.append(f"   原版重建音频: {original_files.get('reconstructed', 'N/A')}")
    report.append(f"   修复版重建音频: {fixed_files.get('reconstructed', 'N/A')}")
    report.append("")
    
    # 质量指标对比
    report.append("📊 质量指标对比:")
    metrics_comparison = [
        ('MSE', 'mse', '越小越好'),
        ('SNR (dB)', 'snr', '越大越好'),
        ('相关系数', 'correlation', '越接近1越好'),
        ('编码时间 (秒)', 'encode_time', '越小越好'),
        ('解码时间 (秒)', 'decode_time', '越小越好'),
        ('压缩比', 'compression_ratio', '信息'),
        ('重建方法', 'method', '信息')
    ]
    
    for metric_name, metric_key, note in metrics_comparison:
        orig_val = original_metrics.get(metric_key, 'N/A')
        fixed_val = fixed_metrics.get(metric_key, 'N/A')
        
        report.append(f"   {metric_name:15} | 原版: {orig_val:>12} | 修复版: {fixed_val:>12} | {note}")
    
    report.append("")
    
    # 音频分析
    if audio_analysis:
        report.append("🎵 音频质量分析:")
        report.append(f"   RMS功率比值: {audio_analysis.get('rms_ratio', 'N/A'):.4f} (修复版/原版)")
        report.append(f"   动态范围比值: {audio_analysis.get('dynamic_range_ratio', 'N/A'):.4f} (修复版/原版)")
        report.append(f"   频谱中心差异: {audio_analysis.get('spectral_centroid_diff', 'N/A'):.2f} Hz")
        report.append("")
    
    # 改进总结
    report.append("✅ 修复版改进内容:")
    report.append("   1. 解决了数值溢出问题 (std=inf)")
    report.append("   2. 强制使用float32避免类型不兼容")
    report.append("   3. 异常值检测和修复 (点击噪音)")
    report.append("   4. 渐变淡入淡出处理")
    report.append("   5. 高级带通滤波")
    report.append("   6. 动态范围压缩")
    report.append("   7. 轻微平滑滤波")
    report.append("")
    
    # 建议
    report.append("💡 建议:")
    if fixed_metrics.get('snr', 0) > original_metrics.get('snr', 0):
        report.append("   ✅ 修复版SNR更高，噪音减少效果明显")
    else:
        report.append("   ⚠️ 修复版SNR未明显改善，可能需要进一步调优")
    
    if audio_analysis and audio_analysis.get('rms_ratio', 1) < 0.8:
        report.append("   ⚠️ 修复版音量显著降低，可能过度压缩")
    elif audio_analysis and audio_analysis.get('rms_ratio', 1) > 1.2:
        report.append("   ⚠️ 修复版音量显著增加，注意削波")
    else:
        report.append("   ✅ 修复版音量控制合理")
    
    report.append("")
    report.append("🎧 听觉测试建议:")
    report.append("   请用音频播放器比较两个重建文件，重点关注:")
    report.append("   - 点击/爆音噪音是否减少")
    report.append("   - 整体音质是否改善")
    report.append("   - 是否保持了原始音频的特征")
    
    return "\n".join(report)


def run_comparison_test(audio_path, max_length=10):
    """运行完整的对比测试"""
    print(f"\n🎯 开始AudioLDM2 VAE噪音修复对比测试")
    print(f"音频文件: {audio_path}")
    print(f"最大长度: {max_length} 秒")
    print("=" * 60)
    
    # 运行原版测试
    original_success, original_output = run_original_test(audio_path, max_length)
    time.sleep(2)  # 等待文件写入
    
    # 运行修复版测试
    fixed_success, fixed_output = run_fixed_test(audio_path, max_length)
    time.sleep(2)  # 等待文件写入
    
    if not original_success and not fixed_success:
        print("❌ 两个测试都失败了")
        return
    
    # 提取指标
    original_metrics = {}
    fixed_metrics = {}
    
    if original_success:
        original_metrics = extract_metrics_from_output(original_output)
    
    if fixed_success:
        fixed_metrics = extract_metrics_from_output(fixed_output)
    
    # 查找输出文件
    original_files, fixed_files = find_latest_outputs()
    
    # 音频质量分析
    audio_analysis = None
    if original_files['reconstructed'] and fixed_files['reconstructed']:
        print("🔍 分析音频质量差异...")
        audio_analysis = analyze_audio_quality(
            original_files['reconstructed'],
            fixed_files['reconstructed'],
            "original", "fixed"
        )
    
    # 生成报告
    report = generate_comparison_report(
        original_metrics, fixed_metrics, audio_analysis,
        original_files, fixed_files
    )
    
    # 保存报告
    report_path = f"vae_noise_fix_comparison_{int(time.time())}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{report}")
    print(f"\n📝 详细报告已保存至: {report_path}")
    
    return {
        'original_metrics': original_metrics,
        'fixed_metrics': fixed_metrics,
        'audio_analysis': audio_analysis,
        'report_path': report_path
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python vae_comparison_test.py <音频文件路径> [最大长度秒数]")
        
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(Path('.').glob(f'*{ext}'))
        
        if audio_files:
            print(f"\n找到音频文件:")
            for i, file in enumerate(audio_files, 1):
                print(f"{i}. {file}")
            
            try:
                choice = input("请选择文件序号: ").strip()
                file_idx = int(choice) - 1
                if 0 <= file_idx < len(audio_files):
                    audio_path = str(audio_files[file_idx])
                else:
                    print("无效选择")
                    return
            except (ValueError, KeyboardInterrupt):
                print("取消操作")
                return
        else:
            print("当前目录没有找到音频文件")
            return
    else:
        audio_path = sys.argv[1]
    
    max_length = 10
    if len(sys.argv) >= 3:
        try:
            max_length = float(sys.argv[2])
        except ValueError:
            print(f"无效长度参数，使用默认值 {max_length} 秒")
    
    try:
        result = run_comparison_test(audio_path, max_length)
        if result:
            print(f"\n✅ 对比测试完成！")
            print(f"请查看报告并播放音频文件进行主观评估。")
    except Exception as e:
        print(f"❌ 对比测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
