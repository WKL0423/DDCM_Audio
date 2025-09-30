#!/usr/bin/env python3
"""
Step 3: 简化的参数优化
修复了PyTorch兼容性问题的版本，专注于核心的参数测试
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from pathlib import Path
import time
from diffusers import AudioLDM2Pipeline
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

class SimpleParameterOptimizer:
    """
    简化的参数优化器
    避免复杂的PyTorch操作，专注于核心功能
    """
    
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        """初始化优化器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎯 初始化简化参数优化器")
        print(f"   📱 设备: {self.device}")
        
        # 加载AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        print(f"✅ 简化参数优化器初始化完成")
        
        # 创建输出目录
        self.output_dir = Path("simplified_parameter_optimization")
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
    
    def optimize_frequency_boost_parameters(self, audio_path: str) -> Dict:
        """
        优化频率增强参数
        测试不同的增强系数，找到最佳平衡点
        """
        print(f"\n🔬 开始频率增强参数优化: {Path(audio_path).name}")
        
        # 加载和预处理音频（只做一次）
        original_audio, processed_audio = self._load_and_preprocess_audio(audio_path)
        latent = self._encode_audio(processed_audio)
        vae_only_audio = self._decode_audio(latent.clone())
        
        # 定义测试的增强系数
        boost_factors = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]
        
        results = []
        timestamp = int(time.time())
        
        print(f"🧪 测试 {len(boost_factors)} 种增强系数...")
        
        for i, boost_factor in enumerate(boost_factors):
            print(f"\n   🔬 测试 {i+1}/{len(boost_factors)}: 增强系数 {boost_factor}x")
            
            # 应用增强
            enhanced_latent = self._apply_simple_frequency_boost(latent.clone(), boost_factor)
            enhanced_audio = self._decode_audio(enhanced_latent)
            
            # 分析质量
            quality_metrics = self._analyze_quality(original_audio, vae_only_audio, enhanced_audio)
            freq_metrics = self._analyze_frequency(original_audio, vae_only_audio, enhanced_audio)
            
            # 计算综合得分
            composite_score = self._calculate_composite_score(quality_metrics, freq_metrics)
            
            # 保存关键配置的音频
            audio_paths = None
            if boost_factor in [1.2, 1.5, 2.0] or composite_score > 6.0:  # 保存关键配置
                audio_paths = self._save_configuration_audio(
                    original_audio, vae_only_audio, enhanced_audio, 
                    timestamp, f"boost_{boost_factor}"
                )
            
            result = {
                "boost_factor": boost_factor,
                "quality_metrics": quality_metrics,
                "frequency_metrics": freq_metrics,
                "composite_score": composite_score,
                "audio_paths": audio_paths
            }
            
            results.append(result)
            
            print(f"      📊 综合得分: {composite_score:.2f}")
            print(f"      🎼 高频改进: {freq_metrics['improvements']['high_freq_improvement']*100:+.1f}%")
            print(f"      📈 SNR改进: {quality_metrics['improvements']['snr_improvement']:+.2f} dB")
        
        # 找到最佳配置
        best_result = max(results, key=lambda x: x['composite_score'])
        
        # 创建分析报告
        self._create_analysis_report(results, best_result, timestamp)
        
        # 创建可视化
        self._create_analysis_visualizations(results, timestamp)
        
        optimization_result = {
            "input_file": audio_path,
            "timestamp": timestamp,
            "all_results": results,
            "best_result": best_result,
            "total_configs_tested": len(boost_factors)
        }
        
        self._display_results(optimization_result)
        
        return optimization_result
    
    def _apply_simple_frequency_boost(self, latent: torch.Tensor, boost_factor: float) -> torch.Tensor:
        """
        应用简单的频率增强
        使用基础的Laplacian算子，避免复杂的stride操作
        """
        with torch.no_grad():
            # 简单的Laplacian核
            laplacian = torch.tensor([
                [[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]]
            ], dtype=latent.dtype, device=latent.device).unsqueeze(0).unsqueeze(0)
            
            enhanced_latent = latent.clone()
            
            # 对每个通道分别处理
            for c in range(latent.shape[1]):
                channel_latent = latent[:, c:c+1, :, :]  # [B, 1, H, W]
                
                # 应用高频检测（简化版，避免stride问题）
                # 手动实现padding
                padded_channel = F.pad(channel_latent, (1, 1, 1, 1), mode='reflect')
                
                # 手动卷积（避免stride参数问题）
                high_freq_response = F.conv2d(padded_channel, laplacian, padding=0)
                
                # 确保输出尺寸正确
                if high_freq_response.shape != channel_latent.shape:
                    high_freq_response = F.interpolate(
                        high_freq_response, 
                        size=channel_latent.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 应用增强
                enhanced_channel = channel_latent + high_freq_response * (boost_factor - 1.0)
                enhanced_latent[:, c:c+1, :, :] = enhanced_channel
            
            return enhanced_latent
    
    def _calculate_composite_score(self, quality_metrics: Dict, freq_metrics: Dict) -> float:
        """计算综合评分"""
        q_improvements = quality_metrics['improvements']
        f_improvements = freq_metrics['improvements']
        
        # 权重设计（高频恢复最重要）
        weights = {
            'snr': 0.2,
            'correlation': 0.3,
            'high_freq': 0.4,
            'overall_freq': 0.1
        }
        
        # 归一化得分
        snr_score = max(0, min(10, (q_improvements['snr_improvement'] + 5) * 2))
        corr_score = max(0, min(10, q_improvements['correlation_improvement'] * 100))
        high_freq_score = max(0, min(10, f_improvements['high_freq_improvement'] * 10))
        overall_freq_score = max(0, min(10, f_improvements['frequency_correlation_improvement'] * 20))
        
        composite_score = (
            weights['snr'] * snr_score +
            weights['correlation'] * corr_score +
            weights['high_freq'] * high_freq_score +
            weights['overall_freq'] * overall_freq_score
        )
        
        return composite_score
    
    def _load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """加载和预处理音频"""
        print(f"   📂 加载音频文件...")
        
        original_audio, sr = torchaudio.load(audio_path)
        
        if original_audio.shape[0] > 1:
            original_audio = original_audio.mean(dim=0, keepdim=True)
        
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            processed_audio = resampler(original_audio)
        else:
            processed_audio = original_audio.clone()
        
        max_length = 48000 * 10
        if processed_audio.shape[-1] > max_length:
            processed_audio = processed_audio[..., :max_length]
        
        original_audio_np = original_audio.squeeze().numpy()
        return original_audio_np, processed_audio
    
    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """VAE编码"""
        with torch.no_grad():
            audio_np = audio.squeeze().numpy()
            inputs = self.pipeline.feature_extractor(
                audio_np, sampling_rate=48000, return_tensors="pt"
            )
            mel_features = inputs["input_features"].to(self.device)
            
            if mel_features.dim() == 3:
                mel_features = mel_features.unsqueeze(1)
            
            if self.device == "cuda":
                mel_features = mel_features.half()
            
            latent_dist = self.pipeline.vae.encode(mel_features)
            latent = latent_dist.latent_dist.mode()
            latent = latent * self.pipeline.vae.config.scaling_factor
            
            return latent
    
    def _decode_audio(self, latent: torch.Tensor) -> np.ndarray:
        """VAE解码"""
        with torch.no_grad():
            latent_for_decode = latent / self.pipeline.vae.config.scaling_factor
            vae_dtype = next(self.pipeline.vae.parameters()).dtype
            latent_for_decode = latent_for_decode.to(vae_dtype)
            
            mel_spectrogram = self.pipeline.vae.decode(latent_for_decode).sample
            audio_tensor = self.pipeline.mel_spectrogram_to_waveform(mel_spectrogram)
            audio_np = audio_tensor.squeeze().cpu().numpy()
            
            return audio_np
    
    def _analyze_quality(self, original: np.ndarray, vae_only: np.ndarray, enhanced: np.ndarray) -> Dict:
        """质量分析"""
        original_16k = librosa.resample(original, orig_sr=48000, target_sr=16000)
        
        def calc_metrics(orig, recon):
            min_len = min(len(orig), len(recon))
            o, r = orig[:min_len], recon[:min_len]
            
            mse = np.mean((o - r) ** 2)
            snr = 10 * np.log10(np.mean(o ** 2) / (mse + 1e-10))
            correlation = np.corrcoef(o, r)[0, 1] if min_len > 1 else 0.0
            if np.isnan(correlation):
                correlation = 0.0
            
            return {"snr_db": snr, "correlation": correlation, "mse": mse}
        
        vae_metrics = calc_metrics(original_16k, vae_only)
        enhanced_metrics = calc_metrics(original_16k, enhanced)
        
        improvements = {
            "snr_improvement": enhanced_metrics["snr_db"] - vae_metrics["snr_db"],
            "correlation_improvement": enhanced_metrics["correlation"] - vae_metrics["correlation"],
            "mse_improvement": vae_metrics["mse"] - enhanced_metrics["mse"]
        }
        
        return {
            "vae_only": vae_metrics,
            "enhanced": enhanced_metrics,
            "improvements": improvements
        }
    
    def _analyze_frequency(self, original: np.ndarray, vae_only: np.ndarray, enhanced: np.ndarray) -> Dict:
        """频率分析"""
        original_16k = librosa.resample(original, orig_sr=48000, target_sr=16000)
        
        def analyze_bands(orig, recon):
            min_len = min(len(orig), len(recon))
            o, r = orig[:min_len], recon[:min_len]
            
            n_fft = 8192 if min_len >= 8192 else 2 ** int(np.log2(min_len))
            orig_fft = np.abs(np.fft.fft(o[:n_fft]))[:n_fft//2]
            recon_fft = np.abs(np.fft.fft(r[:n_fft]))[:n_fft//2]
            freqs = np.fft.fftfreq(n_fft, 1/16000)[:n_fft//2]
            
            low_mask = freqs < 500
            mid_mask = (freqs >= 500) & (freqs < 4000)
            high_mask = freqs >= 4000
            
            low_retention = np.sum(recon_fft[low_mask]) / (np.sum(orig_fft[low_mask]) + 1e-10)
            mid_retention = np.sum(recon_fft[mid_mask]) / (np.sum(orig_fft[mid_mask]) + 1e-10)
            high_retention = np.sum(recon_fft[high_mask]) / (np.sum(orig_fft[high_mask]) + 1e-10)
            
            freq_corr = np.corrcoef(orig_fft, recon_fft)[0, 1]
            if np.isnan(freq_corr):
                freq_corr = 0.0
            
            return {
                "low_freq_retention": low_retention,
                "mid_freq_retention": mid_retention,
                "high_freq_retention": high_retention,
                "frequency_correlation": freq_corr
            }
        
        vae_freq = analyze_bands(original_16k, vae_only)
        enhanced_freq = analyze_bands(original_16k, enhanced)
        
        improvements = {
            "low_freq_improvement": enhanced_freq["low_freq_retention"] - vae_freq["low_freq_retention"],
            "mid_freq_improvement": enhanced_freq["mid_freq_retention"] - vae_freq["mid_freq_retention"],
            "high_freq_improvement": enhanced_freq["high_freq_retention"] - vae_freq["high_freq_retention"],
            "frequency_correlation_improvement": enhanced_freq["frequency_correlation"] - vae_freq["frequency_correlation"]
        }
        
        return {
            "vae_only": vae_freq,
            "enhanced": enhanced_freq,
            "improvements": improvements
        }
    
    def _save_configuration_audio(self,
                                original: np.ndarray,
                                vae_only: np.ndarray,
                                enhanced: np.ndarray,
                                timestamp: int,
                                config_name: str) -> Dict[str, str]:
        """保存特定配置的音频"""
        
        paths = {}
        safe_name = config_name.replace(".", "_")
        
        original_16k = librosa.resample(original, orig_sr=48000, target_sr=16000)
        
        original_path = self.output_dir / f"original_{safe_name}_{timestamp}.wav"
        vae_path = self.output_dir / f"vae_only_{safe_name}_{timestamp}.wav"
        enhanced_path = self.output_dir / f"enhanced_{safe_name}_{timestamp}.wav"
        
        sf.write(str(original_path), original_16k, 16000)
        sf.write(str(vae_path), vae_only, 16000)
        sf.write(str(enhanced_path), enhanced, 16000)
        
        paths["original"] = str(original_path)
        paths["vae_only"] = str(vae_path)
        paths["enhanced"] = str(enhanced_path)
        
        return paths
    
    def _create_analysis_report(self, results: List[Dict], best_result: Dict, timestamp: int):
        """创建分析报告"""
        
        report_path = self.output_dir / f"boost_factor_analysis_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 频率增强系数优化报告\n\n")
            f.write(f"## 测试概况\n")
            f.write(f"- 测试配置数量: {len(results)}\n")
            f.write(f"- 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n\n")
            
            f.write("## 最佳配置\n")
            f.write(f"- **最佳增强系数**: {best_result['boost_factor']}x\n")
            f.write(f"- **综合得分**: {best_result['composite_score']:.2f}\n")
            f.write(f"- **SNR改进**: {best_result['quality_metrics']['improvements']['snr_improvement']:+.2f} dB\n")
            f.write(f"- **高频改进**: {best_result['frequency_metrics']['improvements']['high_freq_improvement']*100:+.1f}%\n\n")
            
            f.write("## 所有配置结果\n\n")
            f.write("| 增强系数 | SNR改进(dB) | 高频改进(%) | 相关性改进 | 综合得分 |\n")
            f.write("|----------|-------------|-------------|------------|----------|\n")
            
            # 按得分排序
            sorted_results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
            
            for result in sorted_results:
                boost = result['boost_factor']
                quality = result['quality_metrics']['improvements']
                freq = result['frequency_metrics']['improvements']
                score = result['composite_score']
                
                f.write(f"| {boost}x | {quality['snr_improvement']:+.2f} | "
                       f"{freq['high_freq_improvement']*100:+.1f} | "
                       f"{quality['correlation_improvement']:+.4f} | "
                       f"{score:.2f} |\n")
            
            f.write("\n## 关键发现\n\n")
            
            # 找到最佳范围
            high_scores = [r for r in results if r['composite_score'] > 6.0]
            if high_scores:
                boost_range = [r['boost_factor'] for r in high_scores]
                f.write(f"### 最佳增强系数范围\n")
                f.write(f"- 高分配置的增强系数范围: {min(boost_range):.1f}x - {max(boost_range):.1f}x\n")
                f.write(f"- 建议使用: {best_result['boost_factor']}x (最高得分)\n\n")
            
            # 趋势分析
            f.write("### 趋势分析\n")
            boost_factors = [r['boost_factor'] for r in results]
            scores = [r['composite_score'] for r in results]
            
            # 找到得分的峰值
            max_score_idx = scores.index(max(scores))
            optimal_boost = boost_factors[max_score_idx]
            
            f.write(f"- 最优增强系数: {optimal_boost}x\n")
            
            # 分析过度增强
            over_enhanced = [r for r in results if r['boost_factor'] > optimal_boost and r['composite_score'] < best_result['composite_score']]
            if over_enhanced:
                f.write(f"- 过度增强阈值: >{optimal_boost}x 后性能下降\n")
            
        print(f"   📄 分析报告已保存: {report_path}")
    
    def _create_analysis_visualizations(self, results: List[Dict], timestamp: int):
        """创建分析可视化"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('频率增强系数优化分析', fontsize=16, fontweight='bold')
            
            # 提取数据
            boost_factors = [r['boost_factor'] for r in results]
            composite_scores = [r['composite_score'] for r in results]
            snr_improvements = [r['quality_metrics']['improvements']['snr_improvement'] for r in results]
            high_freq_improvements = [r['frequency_metrics']['improvements']['high_freq_improvement'] * 100 for r in results]
            
            # 1. 综合得分 vs 增强系数
            axes[0, 0].plot(boost_factors, composite_scores, 'o-', linewidth=2, markersize=8)
            axes[0, 0].set_title('综合得分 vs 增强系数')
            axes[0, 0].set_xlabel('增强系数')
            axes[0, 0].set_ylabel('综合得分')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 标记最佳点
            best_idx = composite_scores.index(max(composite_scores))
            axes[0, 0].scatter([boost_factors[best_idx]], [composite_scores[best_idx]], 
                             color='red', s=100, zorder=5, label='最佳配置')
            axes[0, 0].legend()
            
            # 2. SNR改进 vs 增强系数
            axes[0, 1].plot(boost_factors, snr_improvements, 's-', linewidth=2, markersize=6, color='orange')
            axes[0, 1].set_title('SNR改进 vs 增强系数')
            axes[0, 1].set_xlabel('增强系数')
            axes[0, 1].set_ylabel('SNR改进 (dB)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # 3. 高频改进 vs 增强系数
            axes[1, 0].plot(boost_factors, high_freq_improvements, '^-', linewidth=2, markersize=6, color='green')
            axes[1, 0].set_title('高频改进 vs 增强系数')
            axes[1, 0].set_xlabel('增强系数')
            axes[1, 0].set_ylabel('高频改进 (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. SNR vs 高频改进散点图
            axes[1, 1].scatter(snr_improvements, high_freq_improvements, s=100, alpha=0.7)
            for i, boost in enumerate(boost_factors):
                axes[1, 1].annotate(f'{boost}x', (snr_improvements[i], high_freq_improvements[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 1].set_title('SNR改进 vs 高频改进')
            axes[1, 1].set_xlabel('SNR改进 (dB)')
            axes[1, 1].set_ylabel('高频改进 (%)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = self.output_dir / f"boost_factor_analysis_{timestamp}.png"
            plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 分析可视化已保存: {plot_path}")
            
        except Exception as e:
            print(f"   ⚠️ 可视化生成失败: {e}")
    
    def _display_results(self, result: Dict):
        """显示结果"""
        print(f"\n{'='*80}")
        print(f"🎯 频率增强系数优化完成！")
        print(f"{'='*80}")
        
        print(f"📁 输入文件: {result['input_file']}")
        print(f"🧪 测试配置数量: {result['total_configs_tested']}")
        
        best = result['best_result']
        best_quality = best['quality_metrics']['improvements']
        best_freq = best['frequency_metrics']['improvements']
        
        print(f"\n🏆 最佳配置:")
        print(f"   📈 最佳增强系数: {best['boost_factor']}x")
        print(f"   🏅 综合得分: {best['composite_score']:.2f}/10")
        
        print(f"\n📊 最佳配置性能:")
        print(f"   📈 SNR改进: {best_quality['snr_improvement']:+.2f} dB")
        print(f"   🔗 相关性改进: {best_quality['correlation_improvement']:+.4f}")
        print(f"   🎼 高频改进: {best_freq['high_freq_improvement']*100:+.1f}%")
        
        # Top 3配置
        sorted_results = sorted(result['all_results'], key=lambda x: x['composite_score'], reverse=True)
        print(f"\n🥇 Top 3 增强系数:")
        for i, r in enumerate(sorted_results[:3]):
            boost = r['boost_factor']
            score = r['composite_score']
            high_freq = r['frequency_metrics']['improvements']['high_freq_improvement'] * 100
            print(f"   {i+1}. {boost}x: 得分 {score:.2f}, 高频改进 {high_freq:+.1f}%")
        
        # 趋势分析
        print(f"\n📈 趋势分析:")
        
        high_performers = [r for r in result['all_results'] if r['composite_score'] > 6.0]
        if high_performers:
            boost_range = [r['boost_factor'] for r in high_performers]
            print(f"   ✅ 推荐增强系数范围: {min(boost_range):.1f}x - {max(boost_range):.1f}x")
        
        # 检查过度增强
        best_boost = best['boost_factor']
        over_enhanced = [r for r in result['all_results'] 
                        if r['boost_factor'] > best_boost and 
                        r['composite_score'] < best['composite_score'] * 0.9]
        
        if over_enhanced:
            min_over = min([r['boost_factor'] for r in over_enhanced])
            print(f"   ⚠️ 过度增强阈值: >{min_over:.1f}x 后效果下降")
        
        print(f"\n📁 输出文件:")
        print(f"   📄 分析报告: boost_factor_analysis_{result['timestamp']}.md")
        print(f"   📊 可视化图表: boost_factor_analysis_{result['timestamp']}.png")
        
        if best['audio_paths']:
            print(f"   🎵 最佳配置音频:")
            for name, path in best['audio_paths'].items():
                print(f"      {name}: {Path(path).name}")

def demo_simplified_optimization():
    """演示简化参数优化"""
    print("🎯 Step 3: 简化参数优化")
    print("=" * 50)
    
    # 检查输入文件
    input_file = "AudioLDM2_Music_output.wav"
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        print("请确保AudioLDM2_Music_output.wav在当前目录中")
        return
    
    # 初始化简化优化器
    optimizer = SimpleParameterOptimizer()
    
    # 执行频率增强参数优化
    result = optimizer.optimize_frequency_boost_parameters(input_file)
    
    print(f"\n✅ 简化参数优化完成！")
    print(f"📊 查看输出目录: simplified_parameter_optimization/")
    print(f"🎵 试听不同增强系数的音频效果")
    print(f"📄 查看详细的分析报告")
    
    print(f"\n💡 总结:")
    print(f"   🎯 我们系统性地测试了不同的频率增强系数")
    print(f"   📈 找到了质量和高频恢复的最佳平衡点")
    print(f"   🔧 现在可以使用最佳参数配置处理音频")

if __name__ == "__main__":
    demo_simplified_optimization()
