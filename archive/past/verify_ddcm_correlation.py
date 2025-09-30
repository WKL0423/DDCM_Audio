#!/usr/bin/env python3
"""
验证DDCM生成的音频与输入音频的相关性
通过频谱分析、波形相关性等指标验证
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
import torch

def analyze_audio_similarity(original_path, reconstructed_paths):
    """分析音频相似性"""
    
    print("🔍 分析音频相似性...")
    
    # 加载原始音频
    original, sr = librosa.load(original_path, sr=16000)
    
    results = {}
    
    for name, path in reconstructed_paths.items():
        print(f"\n📊 分析 {name}...")
        
        # 加载重建音频
        recon, _ = librosa.load(path, sr=16000)
        
        # 确保长度一致
        min_len = min(len(original), len(recon))
        orig = original[:min_len]
        rec = recon[:min_len]
        
        # 1. 波形相关性
        correlation = np.corrcoef(orig, rec)[0, 1]
        
        # 2. 频谱相关性
        orig_fft = np.abs(np.fft.fft(orig))
        rec_fft = np.abs(np.fft.fft(rec))
        spectral_correlation = np.corrcoef(orig_fft, rec_fft)[0, 1]
        
        # 3. Mel频谱相关性
        orig_mel = librosa.feature.melspectrogram(y=orig, sr=sr)
        rec_mel = librosa.feature.melspectrogram(y=rec, sr=sr)
        mel_correlation = np.corrcoef(orig_mel.flatten(), rec_mel.flatten())[0, 1]
        
        # 4. MFCC相关性
        orig_mfcc = librosa.feature.mfcc(y=orig, sr=sr, n_mfcc=13)
        rec_mfcc = librosa.feature.mfcc(y=rec, sr=sr, n_mfcc=13)
        mfcc_correlation = np.corrcoef(orig_mfcc.flatten(), rec_mfcc.flatten())[0, 1]
        
        # 5. 节奏相关性
        orig_tempo, orig_beats = librosa.beat.beat_track(y=orig, sr=sr)
        rec_tempo, rec_beats = librosa.beat.beat_track(y=rec, sr=sr)
        tempo_similarity = 1 - abs(orig_tempo - rec_tempo) / max(orig_tempo, rec_tempo)
        
        # 6. 频谱质心相关性
        orig_centroid = librosa.feature.spectral_centroid(y=orig, sr=sr)[0]
        rec_centroid = librosa.feature.spectral_centroid(y=rec, sr=sr)[0]
        min_len_centroid = min(len(orig_centroid), len(rec_centroid))
        centroid_correlation = np.corrcoef(
            orig_centroid[:min_len_centroid], 
            rec_centroid[:min_len_centroid]
        )[0, 1]
        
        # 7. 零交叉率相关性
        orig_zcr = librosa.feature.zero_crossing_rate(orig)[0]
        rec_zcr = librosa.feature.zero_crossing_rate(rec)[0]
        min_len_zcr = min(len(orig_zcr), len(rec_zcr))
        zcr_correlation = np.corrcoef(
            orig_zcr[:min_len_zcr], 
            rec_zcr[:min_len_zcr]
        )[0, 1]
        
        # 8. 综合相似性分数
        similarity_score = (
            correlation * 0.25 +
            spectral_correlation * 0.20 +
            mel_correlation * 0.20 +
            mfcc_correlation * 0.15 +
            tempo_similarity * 0.10 +
            centroid_correlation * 0.05 +
            zcr_correlation * 0.05
        )
        
        results[name] = {
            'wave_correlation': correlation,
            'spectral_correlation': spectral_correlation,
            'mel_correlation': mel_correlation,
            'mfcc_correlation': mfcc_correlation,
            'tempo_similarity': tempo_similarity,
            'centroid_correlation': centroid_correlation,
            'zcr_correlation': zcr_correlation,
            'similarity_score': similarity_score
        }
        
        print(f"   波形相关性: {correlation:.4f}")
        print(f"   频谱相关性: {spectral_correlation:.4f}")
        print(f"   Mel频谱相关性: {mel_correlation:.4f}")
        print(f"   MFCC相关性: {mfcc_correlation:.4f}")
        print(f"   节奏相似性: {tempo_similarity:.4f}")
        print(f"   频谱质心相关性: {centroid_correlation:.4f}")
        print(f"   零交叉率相关性: {zcr_correlation:.4f}")
        print(f"   🏆 综合相似性分数: {similarity_score:.4f}")
    
    return results

def main():
    """主函数"""
    print("🎯 DDCM音频相关性验证")
    print("=" * 50)
    
    # 文件路径
    original_path = "AudioLDM2_Music_output.wav"
    output_dir = Path("ddcm_input_based_output")
    
    if not Path(original_path).exists():
        print(f"❌ 原始文件不存在: {original_path}")
        return
    
    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        print("请先运行 audioldm2_ddcm_input_based_fixed.py")
        return
    
    # 找到最新的输出文件
    files = list(output_dir.glob("*.wav"))
    if not files:
        print(f"❌ 未找到输出文件")
        return
    
    # 按修改时间排序，取最新的
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    reconstructed_paths = {}
    
    # 找到三种重建方法的文件
    for file in files:
        if "Original_VAE" in file.name:
            reconstructed_paths["Original_VAE"] = str(file)
        elif "Quantized_VAE" in file.name:
            reconstructed_paths["Quantized_VAE"] = str(file)
        elif "DDCM_Diffusion" in file.name:
            reconstructed_paths["DDCM_Diffusion"] = str(file)
        
        # 只取最新的一组
        if len(reconstructed_paths) == 3:
            break
    
    if len(reconstructed_paths) < 3:
        print(f"❌ 未找到完整的三种重建文件")
        print(f"找到的文件: {list(reconstructed_paths.keys())}")
        return
    
    print(f"✅ 找到重建文件:")
    for name, path in reconstructed_paths.items():
        print(f"   {name}: {Path(path).name}")
    
    # 分析相似性
    results = analyze_audio_similarity(original_path, reconstructed_paths)
    
    # 总结结果
    print(f"\n{'='*70}")
    print(f"🎯 相关性验证总结")
    print(f"{'='*70}")
    
    print(f"{'方法':<20} {'波形':<8} {'频谱':<8} {'Mel':<8} {'MFCC':<8} {'综合分数':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<20} {result['wave_correlation']:<8.4f} "
              f"{result['spectral_correlation']:<8.4f} {result['mel_correlation']:<8.4f} "
              f"{result['mfcc_correlation']:<8.4f} {result['similarity_score']:<10.4f}")
    
    # 判断是否与输入相关
    print(f"\n🔍 相关性判断:")
    
    for name, result in results.items():
        score = result['similarity_score']
        if score > 0.8:
            print(f"   {name}: 🎉 高度相关 (分数: {score:.4f})")
        elif score > 0.5:
            print(f"   {name}: ✅ 中度相关 (分数: {score:.4f})")
        elif score > 0.2:
            print(f"   {name}: ⚠️ 低度相关 (分数: {score:.4f})")
        else:
            print(f"   {name}: ❌ 几乎无关 (分数: {score:.4f})")
    
    # 特别关注DDCM diffusion的结果
    if 'DDCM_Diffusion' in results:
        ddcm_score = results['DDCM_Diffusion']['similarity_score']
        print(f"\n🎯 DDCM关键发现:")
        if ddcm_score > 0.3:
            print(f"   ✅ DDCM diffusion生成的音频与输入音频确实相关！")
            print(f"   📊 相似性分数: {ddcm_score:.4f}")
            print(f"   💡 这证明了基于量化latent的diffusion保持了输入音频的特征")
        else:
            print(f"   ❌ DDCM diffusion与输入音频相关性较低")
            print(f"   📊 相似性分数: {ddcm_score:.4f}")
            print(f"   💡 可能需要优化码本大小或diffusion参数")
    
    # 量化效果分析
    if 'Original_VAE' in results and 'Quantized_VAE' in results:
        orig_score = results['Original_VAE']['similarity_score']
        quant_score = results['Quantized_VAE']['similarity_score']
        loss = orig_score - quant_score
        
        print(f"\n📚 码本量化分析:")
        print(f"   原始VAE重建相似性: {orig_score:.4f}")
        print(f"   量化VAE重建相似性: {quant_score:.4f}")
        print(f"   量化损失: {loss:.4f}")
        
        if loss < 0.1:
            print(f"   ✅ 量化损失很小，码本表示非常有效")
        elif loss < 0.3:
            print(f"   ⚠️ 量化有一定损失，但仍可接受")
        else:
            print(f"   ❌ 量化损失较大，建议增大码本大小")

if __name__ == "__main__":
    main()
