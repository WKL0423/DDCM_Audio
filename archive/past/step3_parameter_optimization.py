#!/usr/bin/env python3
"""
Step 3: 参数优化和精细调整
基于Step 2的成功结果，对最有效的频率增强方法进行参数优化
目标：找到质量和高频恢复的最佳平衡点，实现最优性能

优化策略：
1. 系统性地测试不同的增强系数
2. 优化高频检测核的设计
3. 添加自适应增强机制
4. 综合评估质量和感知效果
"""

import torch
import torch.nn as nn
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

class OptimizedLatentEnhancer:
    """
    优化的Latent增强器
    基于Step 2的发现，专注于频率增强方法的参数优化
    """
    
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        """初始化优化的Latent增强器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎯 初始化优化的Latent增强器")
        print(f"   📱 设备: {self.device}")
        print(f"   🤖 模型: {model_name}")
        
        # 加载AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        print(f"✅ 优化的Latent增强器初始化完成")
        
        # 创建输出目录
        self.output_dir = Path("optimized_latent_enhanced")
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
        
        # 预定义不同的高频检测核
        self.kernels = self._create_frequency_kernels()
    
    def _create_frequency_kernels(self) -> Dict[str, torch.Tensor]:
        """创建不同类型的高频检测核"""
        kernels = {}
        
        # 标准Laplacian核
        kernels["laplacian"] = torch.tensor([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ], dtype=torch.float32)
        
        # 强化的Laplacian核
        kernels["strong_laplacian"] = torch.tensor([
            [[-2, -2, -2],
             [-2, 16, -2],
             [-2, -2, -2]]
        ], dtype=torch.float32)
        
        # Sobel边缘检测核（水平）
        kernels["sobel_h"] = torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32)
        
        # Sobel边缘检测核（垂直）
        kernels["sobel_v"] = torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32)
        
        # 高通滤波器
        kernels["highpass"] = torch.tensor([
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]]
        ], dtype=torch.float32)
        
        # 锐化核
        kernels["sharpen"] = torch.tensor([
            [[ 0, -1,  0],
             [-1,  6, -1],
             [ 0, -1,  0]]
        ], dtype=torch.float32)
        
        return kernels
    
    def optimize_parameters(self, audio_path: str) -> Dict:
        """
        系统性地优化增强参数
        
        Args:
            audio_path: 输入音频文件路径
            
        Returns:
            优化结果和最佳参数
        """
        print(f"\n🔬 开始参数优化: {Path(audio_path).name}")
        
        # 加载和预处理音频（只做一次）
        original_audio, processed_audio = self._load_and_preprocess_audio(audio_path)
        latent = self._encode_audio(processed_audio)
        vae_only_audio = self._decode_audio(latent.clone())
        
        # 定义参数搜索空间
        test_configs = [
            # 基础测试：不同增强系数
            {"name": "轻微增强", "kernel": "laplacian", "boost": 1.2, "adaptive": False},
            {"name": "中等增强", "kernel": "laplacian", "boost": 1.5, "adaptive": False},
            {"name": "强度增强", "kernel": "laplacian", "boost": 2.0, "adaptive": False},
            {"name": "极强增强", "kernel": "laplacian", "boost": 3.0, "adaptive": False},
            
            # 不同核类型测试
            {"name": "强化Laplacian", "kernel": "strong_laplacian", "boost": 1.5, "adaptive": False},
            {"name": "Sobel水平", "kernel": "sobel_h", "boost": 1.5, "adaptive": False},
            {"name": "Sobel垂直", "kernel": "sobel_v", "boost": 1.5, "adaptive": False},
            {"name": "高通滤波", "kernel": "highpass", "boost": 1.5, "adaptive": False},
            {"name": "锐化核", "kernel": "sharpen", "boost": 1.5, "adaptive": False},
            
            # 自适应增强测试
            {"name": "自适应中等", "kernel": "laplacian", "boost": 1.5, "adaptive": True},
            {"name": "自适应强度", "kernel": "laplacian", "boost": 2.0, "adaptive": True},
        ]
        
        results = []
        timestamp = int(time.time())
        
        print(f"🧪 测试 {len(test_configs)} 种参数配置...")
        
        for i, config in enumerate(test_configs):
            print(f"\n   🔬 测试 {i+1}/{len(test_configs)}: {config['name']}")
            
            # 应用增强
            enhanced_latent = self._apply_optimized_enhancement(
                latent.clone(),
                kernel_name=config['kernel'],
                boost_factor=config['boost'],
                adaptive=config['adaptive']
            )
            
            # 解码
            enhanced_audio = self._decode_audio(enhanced_latent)
            
            # 分析质量
            quality_metrics = self._analyze_quality(original_audio, vae_only_audio, enhanced_audio)
            freq_metrics = self._analyze_frequency(original_audio, vae_only_audio, enhanced_audio)
            
            # 计算综合得分
            composite_score = self._calculate_composite_score(quality_metrics, freq_metrics)
            
            # 保存音频（仅保存最佳几个）
            audio_paths = None
            if i < 3 or composite_score > 7.0:  # 保存前3个和高分配置
                audio_paths = self._save_configuration_audio(
                    original_audio, vae_only_audio, enhanced_audio, 
                    timestamp, config['name']
                )
            
            result = {
                "config": config,
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
        
        # 创建优化报告
        self._create_optimization_report(results, best_result, timestamp)
        
        # 创建对比可视化
        self._create_optimization_visualizations(results, timestamp)
        
        optimization_result = {
            "input_file": audio_path,
            "timestamp": timestamp,
            "all_results": results,
            "best_result": best_result,
            "total_configs_tested": len(test_configs)
        }
        
        self._display_optimization_results(optimization_result)
        
        return optimization_result
    
    def _apply_optimized_enhancement(self,
                                   latent: torch.Tensor,
                                   kernel_name: str = "laplacian",
                                   boost_factor: float = 1.5,
                                   adaptive: bool = False) -> torch.Tensor:
        """应用优化的增强"""
        
        kernel = self.kernels[kernel_name].to(latent.device, latent.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        
        enhanced_latent = latent.clone()
        
        # 对每个通道分别处理
        for c in range(latent.shape[1]):
            channel_latent = latent[:, c:c+1, :, :]  # [B, 1, H, W]
              # 应用高频检测
            high_freq_response = F.conv2d(channel_latent, kernel, padding=1)
            
            # 自适应增强
            if adaptive:
                # 基于局部方差调整增强强度
                local_mean = F.avg_pool2d(channel_latent, kernel_size=3, stride=1, padding=1)
                local_var = F.avg_pool2d((channel_latent - local_mean)**2, kernel_size=3, stride=1, padding=1)
                adaptive_factor = torch.clamp(local_var * 10, 0.5, 2.0)  # 自适应系数
                high_freq_enhancement = high_freq_response * (boost_factor - 1.0) * adaptive_factor
            else:
                high_freq_enhancement = high_freq_response * (boost_factor - 1.0)
            
            # 应用增强
            enhanced_channel = channel_latent + high_freq_enhancement
            enhanced_latent[:, c:c+1, :, :] = enhanced_channel
        
        return enhanced_latent
    
    def _calculate_composite_score(self, quality_metrics: Dict, freq_metrics: Dict) -> float:
        """
        计算综合评分，平衡质量和频率改进
        """
        q_improvements = quality_metrics['improvements']
        f_improvements = freq_metrics['improvements']
        
        # 权重设计
        weights = {
            'snr': 0.2,           # SNR权重较低，因为可能因为恢复高频而下降
            'correlation': 0.3,    # 相关性重要
            'high_freq': 0.4,      # 高频恢复最重要
            'overall_freq': 0.1    # 整体频率相关性
        }
        
        # 计算各项得分（归一化到0-10）
        snr_score = max(0, min(10, (q_improvements['snr_improvement'] + 5) * 2))  # -5到+5dB映射到0-10
        corr_score = max(0, min(10, q_improvements['correlation_improvement'] * 100))  # 相关性改进
        high_freq_score = max(0, min(10, f_improvements['high_freq_improvement'] * 10))  # 高频改进
        overall_freq_score = max(0, min(10, f_improvements['frequency_correlation_improvement'] * 20))
        
        # 综合得分
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
        safe_name = config_name.replace(" ", "_").replace("/", "_")
        
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
    
    def _create_optimization_report(self, results: List[Dict], best_result: Dict, timestamp: int):
        """创建优化报告"""
        
        report_path = self.output_dir / f"optimization_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 参数优化报告\n\n")
            f.write(f"## 测试概况\n")
            f.write(f"- 测试配置数量: {len(results)}\n")
            f.write(f"- 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n\n")
            
            f.write("## 最佳配置\n")
            best_config = best_result['config']
            f.write(f"- **配置名称**: {best_config['name']}\n")
            f.write(f"- **检测核**: {best_config['kernel']}\n")
            f.write(f"- **增强系数**: {best_config['boost']}\n")
            f.write(f"- **自适应**: {best_config['adaptive']}\n")
            f.write(f"- **综合得分**: {best_result['composite_score']:.2f}\n\n")
            
            f.write("## 所有配置结果\n\n")
            f.write("| 配置名称 | 检测核 | 增强系数 | 自适应 | SNR改进(dB) | 高频改进(%) | 综合得分 |\n")
            f.write("|---------|--------|----------|--------|-------------|-------------|----------|\n")
            
            # 按得分排序
            sorted_results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
            
            for result in sorted_results:
                config = result['config']
                quality = result['quality_metrics']['improvements']
                freq = result['frequency_metrics']['improvements']
                score = result['composite_score']
                
                f.write(f"| {config['name']} | {config['kernel']} | {config['boost']} | "
                       f"{'是' if config['adaptive'] else '否'} | "
                       f"{quality['snr_improvement']:+.2f} | "
                       f"{freq['high_freq_improvement']*100:+.1f} | "
                       f"{score:.2f} |\n")
            
            f.write("\n## 关键发现\n\n")
            
            # 分析最佳核类型
            kernel_scores = {}
            for result in results:
                kernel = result['config']['kernel']
                if kernel not in kernel_scores:
                    kernel_scores[kernel] = []
                kernel_scores[kernel].append(result['composite_score'])
            
            f.write("### 检测核性能对比\n")
            for kernel, scores in kernel_scores.items():
                avg_score = np.mean(scores)
                f.write(f"- **{kernel}**: 平均得分 {avg_score:.2f}\n")
            
            # 分析增强系数效果
            f.write("\n### 增强系数效果\n")
            boost_analysis = {}
            for result in results:
                boost = result['config']['boost']
                if boost not in boost_analysis:
                    boost_analysis[boost] = []
                boost_analysis[boost].append(result['composite_score'])
            
            for boost, scores in sorted(boost_analysis.items()):
                avg_score = np.mean(scores)
                f.write(f"- **{boost}x**: 平均得分 {avg_score:.2f}\n")
        
        print(f"   📄 优化报告已保存: {report_path}")
    
    def _create_optimization_visualizations(self, results: List[Dict], timestamp: int):
        """创建优化可视化"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('参数优化结果可视化', fontsize=16, fontweight='bold')
            
            # 提取数据
            config_names = [r['config']['name'] for r in results]
            composite_scores = [r['composite_score'] for r in results]
            snr_improvements = [r['quality_metrics']['improvements']['snr_improvement'] for r in results]
            high_freq_improvements = [r['frequency_metrics']['improvements']['high_freq_improvement'] * 100 for r in results]
            
            # 1. 综合得分对比
            axes[0, 0].bar(range(len(config_names)), composite_scores, alpha=0.7)
            axes[0, 0].set_title('综合得分对比')
            axes[0, 0].set_xlabel('配置')
            axes[0, 0].set_ylabel('综合得分')
            axes[0, 0].set_xticks(range(len(config_names)))
            axes[0, 0].set_xticklabels(config_names, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. SNR改进 vs 高频改进散点图
            axes[0, 1].scatter(snr_improvements, high_freq_improvements, alpha=0.7, s=100)
            for i, name in enumerate(config_names):
                axes[0, 1].annotate(name, (snr_improvements[i], high_freq_improvements[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[0, 1].set_title('SNR改进 vs 高频改进')
            axes[0, 1].set_xlabel('SNR改进 (dB)')
            axes[0, 1].set_ylabel('高频改进 (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 核类型性能对比
            kernel_types = {}
            for result in results:
                kernel = result['config']['kernel']
                if kernel not in kernel_types:
                    kernel_types[kernel] = []
                kernel_types[kernel].append(result['composite_score'])
            
            kernel_names = list(kernel_types.keys())
            kernel_avg_scores = [np.mean(scores) for scores in kernel_types.values()]
            
            axes[1, 0].bar(kernel_names, kernel_avg_scores, alpha=0.7)
            axes[1, 0].set_title('不同检测核的平均性能')
            axes[1, 0].set_xlabel('检测核类型')
            axes[1, 0].set_ylabel('平均综合得分')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 增强系数效果
            boost_factors = {}
            for result in results:
                boost = result['config']['boost']
                if boost not in boost_factors:
                    boost_factors[boost] = []
                boost_factors[boost].append(result['composite_score'])
            
            boost_values = sorted(boost_factors.keys())
            boost_avg_scores = [np.mean(boost_factors[boost]) for boost in boost_values]
            
            axes[1, 1].plot(boost_values, boost_avg_scores, 'o-', linewidth=2, markersize=8)
            axes[1, 1].set_title('增强系数效果')
            axes[1, 1].set_xlabel('增强系数')
            axes[1, 1].set_ylabel('平均综合得分')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = self.output_dir / f"optimization_visualization_{timestamp}.png"
            plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 优化可视化已保存: {plot_path}")
            
        except Exception as e:
            print(f"   ⚠️ 可视化生成失败: {e}")
    
    def _display_optimization_results(self, result: Dict):
        """显示优化结果"""
        print(f"\n{'='*90}")
        print(f"🎯 参数优化完成！")
        print(f"{'='*90}")
        
        print(f"📁 输入文件: {result['input_file']}")
        print(f"🧪 测试配置数量: {result['total_configs_tested']}")
        
        best = result['best_result']
        best_config = best['config']
        best_quality = best['quality_metrics']['improvements']
        best_freq = best['frequency_metrics']['improvements']
        
        print(f"\n🏆 最佳配置:")
        print(f"   📛 名称: {best_config['name']}")
        print(f"   🔍 检测核: {best_config['kernel']}")
        print(f"   📈 增强系数: {best_config['boost']}x")
        print(f"   🧠 自适应: {'是' if best_config['adaptive'] else '否'}")
        print(f"   🏅 综合得分: {best['composite_score']:.2f}/10")
        
        print(f"\n📊 最佳配置性能:")
        print(f"   📈 SNR改进: {best_quality['snr_improvement']:+.2f} dB")
        print(f"   🔗 相关性改进: {best_quality['correlation_improvement']:+.4f}")
        print(f"   🎼 高频改进: {best_freq['high_freq_improvement']*100:+.1f}%")
        print(f"   🎵 整体频率相关性改进: {best_freq['frequency_correlation_improvement']:+.4f}")
        
        # Top 3配置
        sorted_results = sorted(result['all_results'], key=lambda x: x['composite_score'], reverse=True)
        print(f"\n🥇 Top 3 配置:")
        for i, r in enumerate(sorted_results[:3]):
            config = r['config']
            score = r['composite_score']
            high_freq = r['frequency_metrics']['improvements']['high_freq_improvement'] * 100
            print(f"   {i+1}. {config['name']}: 得分 {score:.2f}, 高频改进 {high_freq:+.1f}%")
        
        # 关键发现
        print(f"\n💡 关键发现:")
        
        # 最佳核类型
        kernel_scores = {}
        for r in result['all_results']:
            kernel = r['config']['kernel']
            if kernel not in kernel_scores:
                kernel_scores[kernel] = []
            kernel_scores[kernel].append(r['composite_score'])
        
        best_kernel = max(kernel_scores.keys(), key=lambda k: np.mean(kernel_scores[k]))
        print(f"   🔍 最佳检测核: {best_kernel} (平均得分 {np.mean(kernel_scores[best_kernel]):.2f})")
        
        # 最佳增强系数
        boost_scores = {}
        for r in result['all_results']:
            boost = r['config']['boost']
            if boost not in boost_scores:
                boost_scores[boost] = []
            boost_scores[boost].append(r['composite_score'])
        
        best_boost = max(boost_scores.keys(), key=lambda b: np.mean(boost_scores[b]))
        print(f"   📈 最佳增强系数: {best_boost}x (平均得分 {np.mean(boost_scores[best_boost]):.2f})")
        
        # 自适应效果
        adaptive_scores = [r['composite_score'] for r in result['all_results'] if r['config']['adaptive']]
        non_adaptive_scores = [r['composite_score'] for r in result['all_results'] if not r['config']['adaptive']]
        
        if adaptive_scores and non_adaptive_scores:
            adaptive_avg = np.mean(adaptive_scores)
            non_adaptive_avg = np.mean(non_adaptive_scores)
            if adaptive_avg > non_adaptive_avg:
                print(f"   🧠 自适应增强有效: 平均得分提升 {adaptive_avg - non_adaptive_avg:.2f}")
            else:
                print(f"   🔒 固定增强更稳定: 平均得分高出 {non_adaptive_avg - adaptive_avg:.2f}")
        
        print(f"\n📁 输出文件:")
        print(f"   📄 优化报告: optimization_report_{result['timestamp']}.md")
        print(f"   📊 可视化图表: optimization_visualization_{result['timestamp']}.png")
        
        if best['audio_paths']:
            print(f"   🎵 最佳配置音频:")
            for name, path in best['audio_paths'].items():
                print(f"      {name}: {Path(path).name}")

def demo_parameter_optimization():
    """演示参数优化"""
    print("🎯 Step 3: 参数优化和精细调整")
    print("=" * 70)
    
    # 检查输入文件
    input_file = "AudioLDM2_Music_output.wav"
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        print("请确保AudioLDM2_Music_output.wav在当前目录中")
        return
    
    # 初始化优化器
    optimizer = OptimizedLatentEnhancer()
    
    # 执行参数优化
    result = optimizer.optimize_parameters(input_file)
    
    print(f"\n✅ 参数优化完成！")
    print(f"📊 查看输出目录: optimized_latent_enhanced/")
    print(f"🎵 试听最佳配置的音频效果")
    print(f"📄 查看详细的优化报告")
    
    print(f"\n🚀 下一步建议:")
    print(f"   1. 使用最佳参数配置处理更多音频文件")
    print(f"   2. 考虑集成到生产环境中")
    print(f"   3. 探索更高级的增强技术")

if __name__ == "__main__":
    demo_parameter_optimization()
