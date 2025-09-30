#!/usr/bin/env python3
"""
Step 4: 批量验证和实际应用集成
基于Step 3的最佳参数（增强系数1.4x），在多个音频文件上验证性能
并提供可集成到生产环境的增强管道
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from pathlib import Path
import time
from diffusers import AudioLDM2Pipeline
from typing import Dict, Tuple, List, Optional
import warnings
import pandas as pd
from scipy import signal
warnings.filterwarnings("ignore")

class BatchValidationPipeline:
    """
    批量验证管道
    验证最佳增强参数在多个音频文件上的性能
    """
    
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        """初始化批量验证管道"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎯 初始化批量验证管道")
        print(f"   📱 设备: {self.device}")
          # 加载AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        # 统一使用float32避免类型不匹配问题
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        
        # 最佳参数（来自Step 3的优化结果）
        self.best_boost_factor = 1.4
        
        print(f"✅ 批量验证管道初始化完成")
        print(f"   🎯 最佳增强系数: {self.best_boost_factor}x")
        
        # 创建输出目录
        self.output_dir = Path("batch_validation_results")
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
    
    def enhanced_vae_reconstruction(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        使用最佳参数进行增强VAE重建
        """
        try:
            # 1. VAE编码
            latents = self.pipeline.vae.encode(audio_tensor).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor
            
            # 2. 应用最佳增强（基于Step 3的结果）
            enhanced_latents = self.apply_frequency_boost(latents, self.best_boost_factor)
            
            # 3. VAE解码
            enhanced_latents = enhanced_latents / self.pipeline.vae.config.scaling_factor
            reconstructed = self.pipeline.vae.decode(enhanced_latents).sample
            
            return reconstructed
            
        except Exception as e:
            print(f"❌ 增强重建过程出错: {e}")
            return None
    
    def apply_frequency_boost(self, latents: torch.Tensor, boost_factor: float) -> torch.Tensor:
        """
        应用频率增强（基于Step 3优化的方法）
        使用手动numpy实现避免PyTorch stride问题
        """
        try:
            # 转换为numpy进行处理
            latents_np = latents.detach().cpu().numpy()
            enhanced_np = np.zeros_like(latents_np)
            
            # 定义Laplacian增强核（手动实现）
            for b in range(latents_np.shape[0]):  # batch
                for c in range(latents_np.shape[1]):  # channel
                    channel_data = latents_np[b, c]
                    
                    # 计算Laplacian（边缘检测，增强高频细节）
                    laplacian = np.zeros_like(channel_data)
                    
                    # 手动计算2D Laplacian算子
                    for i in range(1, channel_data.shape[0] - 1):
                        for j in range(1, channel_data.shape[1] - 1):
                            laplacian[i, j] = (
                                4 * channel_data[i, j] - 
                                channel_data[i-1, j] - channel_data[i+1, j] - 
                                channel_data[i, j-1] - channel_data[i, j+1]
                            )
                    
                    # 应用增强
                    enhanced_np[b, c] = channel_data + boost_factor * 0.1 * laplacian
            
            # 转换回tensor
            enhanced_latents = torch.from_numpy(enhanced_np).to(latents.device, latents.dtype)
            return enhanced_latents
            
        except Exception as e:
            print(f"❌ 频率增强过程出错: {e}")
            return latents
    
    def calculate_metrics(self, original: np.ndarray, reconstructed: np.ndarray, 
                         sample_rate: int) -> Dict:
        """计算详细的音频质量指标"""
        try:
            # 基本指标
            mse = np.mean((original - reconstructed) ** 2)
            snr = 10 * np.log10(np.var(original) / (mse + 1e-10))
            correlation = np.corrcoef(original, reconstructed)[0, 1]
            
            # 频域分析
            freqs_orig, psd_orig = signal.welch(original, sample_rate, nperseg=1024)
            freqs_recon, psd_recon = signal.welch(reconstructed, sample_rate, nperseg=1024)
            
            # 频段保持率
            low_freq_mask = freqs_orig < 500
            mid_freq_mask = (freqs_orig >= 500) & (freqs_orig < 4000)
            high_freq_mask = freqs_orig >= 4000
            
            low_retention = np.sum(psd_recon[low_freq_mask]) / (np.sum(psd_orig[low_freq_mask]) + 1e-10)
            mid_retention = np.sum(psd_recon[mid_freq_mask]) / (np.sum(psd_orig[mid_freq_mask]) + 1e-10)
            high_retention = np.sum(psd_recon[high_freq_mask]) / (np.sum(psd_orig[high_freq_mask]) + 1e-10)
            
            return {
                'snr': snr,
                'correlation': correlation,
                'mse': mse,
                'low_freq_retention': low_retention,
                'mid_freq_retention': mid_retention,
                'high_freq_retention': high_retention
            }
            
        except Exception as e:
            print(f"❌ 指标计算出错: {e}")
            return {}
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """处理单个音频文件"""
        try:
            print(f"\n🎵 处理音频文件: {Path(audio_path).name}")
            
            # 加载音频
            audio, sr = torchaudio.load(audio_path)
            
            # 确保单声道
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
              # 重采样到48kHz (CLAP特征提取器要求)
            target_sr = 48000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                audio = resampler(audio)
                sr = target_sr
            
            # 裁剪到合适长度（10秒）
            max_length = 10 * sr
            if audio.shape[1] > max_length:
                audio = audio[:, :max_length]
              # 预处理（mel频谱）
            inputs = self.pipeline.feature_extractor(
                audio.squeeze().numpy(), 
                sampling_rate=sr, 
                return_tensors="pt"
            )
            audio_tensor = inputs.input_features.to(self.device).float()
            
            # 原始VAE重建
            start_time = time.time()
            with torch.no_grad():
                original_latents = self.pipeline.vae.encode(audio_tensor).latent_dist.sample()
                original_latents = original_latents * self.pipeline.vae.config.scaling_factor
                original_latents = original_latents / self.pipeline.vae.config.scaling_factor
                original_reconstructed = self.pipeline.vae.decode(original_latents).sample
            original_time = time.time() - start_time
            
            # 增强VAE重建
            start_time = time.time()
            with torch.no_grad():
                enhanced_reconstructed = self.enhanced_vae_reconstruction(audio_tensor)
            enhanced_time = time.time() - start_time
            
            if enhanced_reconstructed is None:
                return None
            
            # 转换为numpy进行分析
            original_np = audio.squeeze().numpy()
            original_recon_np = self.mel_to_audio(original_reconstructed)
            enhanced_recon_np = self.mel_to_audio(enhanced_reconstructed)
            
            # 计算指标
            original_metrics = self.calculate_metrics(original_np, original_recon_np, sr)
            enhanced_metrics = self.calculate_metrics(original_np, enhanced_recon_np, sr)
            
            # 计算改进
            improvements = {}
            for key in original_metrics:
                if key in ['snr', 'correlation']:
                    improvements[f"{key}_improvement"] = enhanced_metrics[key] - original_metrics[key]
                else:
                    improvements[f"{key}_improvement"] = (enhanced_metrics[key] / original_metrics[key] - 1) * 100
            
            return {
                'filename': Path(audio_path).name,
                'audio_length': audio.shape[1] / sr,
                'processing_time_original': original_time,
                'processing_time_enhanced': enhanced_time,
                'original_metrics': original_metrics,
                'enhanced_metrics': enhanced_metrics,
                'improvements': improvements,
                'audio_original': original_np,
                'audio_original_recon': original_recon_np,
                'audio_enhanced_recon': enhanced_recon_np,
                'sample_rate': sr
            }
              except Exception as e:
            print(f"❌ 处理文件 {audio_path} 时出错: {e}")
            return None
    
    def mel_to_audio(self, mel_tensor: torch.Tensor) -> np.ndarray:
        """将mel频谱转换回音频（使用AudioLDM2的vocoder）"""
        try:
            # 使用AudioLDM2的vocoder进行转换
            with torch.no_grad():
                # mel_tensor应该已经是正确格式的mel频谱
                audio_tensor = self.pipeline.vocoder(mel_tensor.unsqueeze(0))
                audio_np = audio_tensor.squeeze().detach().cpu().numpy()
                
            return audio_np
            
        except Exception as e:
            print(f"❌ mel转音频出错: {e}")
            # 如果vocoder失败，返回匹配长度的静音
            try:
                # 估算合理的音频长度
                estimated_length = 16384  # 默认长度
                return np.zeros(estimated_length)
            except:
                return np.zeros(16000)  # 回退到1秒静音
    
    def batch_validate(self, audio_files: List[str]) -> Dict:
        """批量验证多个音频文件"""
        print(f"\n🎯 开始批量验证")
        print(f"   📁 文件数量: {len(audio_files)}")
        print(f"   ⚡ 增强系数: {self.best_boost_factor}x")
        
        results = []
        successful_files = 0
        
        for i, audio_file in enumerate(audio_files):
            print(f"\n📊 进度: {i+1}/{len(audio_files)}")
            
            result = self.process_audio_file(audio_file)
            if result is not None:
                results.append(result)
                successful_files += 1
                
                # 保存单个文件的结果
                self.save_individual_result(result, i)
        
        if not results:
            print("❌ 没有成功处理的文件")
            return {}
        
        # 计算统计摘要
        summary = self.calculate_batch_summary(results)
        
        # 生成批量报告
        self.generate_batch_report(results, summary)
        
        print(f"\n✅ 批量验证完成")
        print(f"   ✅ 成功处理: {successful_files}/{len(audio_files)} 文件")
        print(f"   📊 平均SNR改进: {summary['avg_snr_improvement']:.2f} dB")
        print(f"   🎵 平均高频改进: {summary['avg_high_freq_improvement']:.1f}%")
        
        return summary
    
    def save_individual_result(self, result: Dict, index: int):
        """保存单个文件的结果"""
        try:
            # 保存音频文件
            filename_base = Path(result['filename']).stem
            
            # 原始重建
            sf.write(
                self.output_dir / f"{filename_base}_original_reconstruction.wav",
                result['audio_original_recon'],
                result['sample_rate']
            )
            
            # 增强重建
            sf.write(
                self.output_dir / f"{filename_base}_enhanced_reconstruction.wav",
                result['audio_enhanced_recon'],
                result['sample_rate']
            )
            
            # 生成对比图
            self.plot_comparison(result, index)
            
        except Exception as e:
            print(f"❌ 保存结果时出错: {e}")
    
    def plot_comparison(self, result: Dict, index: int):
        """生成对比图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 时域对比
            time_axis = np.linspace(0, len(result['audio_original']) / result['sample_rate'], 
                                  len(result['audio_original']))
            
            axes[0, 0].plot(time_axis, result['audio_original'], label='原始', alpha=0.7)
            axes[0, 0].plot(time_axis[:len(result['audio_original_recon'])], 
                           result['audio_original_recon'], label='VAE重建', alpha=0.7)
            axes[0, 0].plot(time_axis[:len(result['audio_enhanced_recon'])], 
                           result['audio_enhanced_recon'], label='增强重建', alpha=0.7)
            axes[0, 0].set_title('时域对比')
            axes[0, 0].set_xlabel('时间 (s)')
            axes[0, 0].set_ylabel('幅度')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 频域对比
            freqs, psd_orig = signal.welch(result['audio_original'], result['sample_rate'])
            _, psd_orig_recon = signal.welch(result['audio_original_recon'], result['sample_rate'])
            _, psd_enhanced = signal.welch(result['audio_enhanced_recon'], result['sample_rate'])
            
            axes[0, 1].semilogy(freqs, psd_orig, label='原始', alpha=0.7)
            axes[0, 1].semilogy(freqs, psd_orig_recon, label='VAE重建', alpha=0.7)
            axes[0, 1].semilogy(freqs, psd_enhanced, label='增强重建', alpha=0.7)
            axes[0, 1].set_title('功率谱密度对比')
            axes[0, 1].set_xlabel('频率 (Hz)')
            axes[0, 1].set_ylabel('功率谱密度')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 指标对比
            metrics = ['snr', 'correlation', 'high_freq_retention']
            original_values = [result['original_metrics'][m] for m in metrics]
            enhanced_values = [result['enhanced_metrics'][m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, original_values, width, label='原始VAE', alpha=0.7)
            axes[1, 0].bar(x + width/2, enhanced_values, width, label='增强VAE', alpha=0.7)
            axes[1, 0].set_title('关键指标对比')
            axes[1, 0].set_xlabel('指标')
            axes[1, 0].set_ylabel('数值')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(['SNR', '相关系数', '高频保持率'])
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 改进百分比
            improvements = [
                result['improvements']['snr_improvement'],
                result['improvements']['correlation_improvement'],
                result['improvements']['high_freq_retention_improvement']
            ]
            
            colors = ['green' if x > 0 else 'red' for x in improvements]
            axes[1, 1].bar(metrics, improvements, color=colors, alpha=0.7)
            axes[1, 1].set_title('改进百分比')
            axes[1, 1].set_xlabel('指标')
            axes[1, 1].set_ylabel('改进 (%/dB)')
            axes[1, 1].set_xticklabels(['SNR (dB)', '相关系数 (%)', '高频保持率 (%)'])
            axes[1, 1].grid(True)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            
            filename_base = Path(result['filename']).stem
            plt.savefig(self.output_dir / f"{filename_base}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ 生成对比图时出错: {e}")
    
    def calculate_batch_summary(self, results: List[Dict]) -> Dict:
        """计算批量处理的统计摘要"""
        try:
            # 收集所有指标
            snr_improvements = [r['improvements']['snr_improvement'] for r in results]
            correlation_improvements = [r['improvements']['correlation_improvement'] for r in results]
            high_freq_improvements = [r['improvements']['high_freq_retention_improvement'] for r in results]
            
            processing_times = [r['processing_time_enhanced'] for r in results]
            
            return {
                'total_files': len(results),
                'avg_snr_improvement': np.mean(snr_improvements),
                'std_snr_improvement': np.std(snr_improvements),
                'avg_correlation_improvement': np.mean(correlation_improvements),
                'std_correlation_improvement': np.std(correlation_improvements),
                'avg_high_freq_improvement': np.mean(high_freq_improvements),
                'std_high_freq_improvement': np.std(high_freq_improvements),
                'avg_processing_time': np.mean(processing_times),
                'std_processing_time': np.std(processing_times),
                'snr_improvements': snr_improvements,
                'correlation_improvements': correlation_improvements,
                'high_freq_improvements': high_freq_improvements,
                'boost_factor_used': self.best_boost_factor
            }
            
        except Exception as e:
            print(f"❌ 计算摘要时出错: {e}")
            return {}
    
    def generate_batch_report(self, results: List[Dict], summary: Dict):
        """生成批量验证报告"""
        try:
            timestamp = int(time.time())
            report_file = self.output_dir / f"batch_validation_report_{timestamp}.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"# AudioLDM2 VAE增强 - 批量验证报告\n\n")
                f.write(f"## 验证概况\n")
                f.write(f"- 验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- 处理文件数: {summary['total_files']}\n")
                f.write(f"- 使用增强系数: {summary['boost_factor_used']}x\n\n")
                
                f.write(f"## 总体性能\n")
                f.write(f"- **平均SNR改进**: {summary['avg_snr_improvement']:.2f} ± {summary['std_snr_improvement']:.2f} dB\n")
                f.write(f"- **平均相关系数改进**: {summary['avg_correlation_improvement']:.1f} ± {summary['std_correlation_improvement']:.1f}%\n")
                f.write(f"- **平均高频改进**: {summary['avg_high_freq_improvement']:.1f} ± {summary['std_high_freq_improvement']:.1f}%\n")
                f.write(f"- **平均处理时间**: {summary['avg_processing_time']:.2f} ± {summary['std_processing_time']:.2f} 秒\n\n")
                
                f.write(f"## 详细结果\n\n")
                f.write(f"| 文件名 | SNR改进(dB) | 相关性改进(%) | 高频改进(%) | 处理时间(s) |\n")
                f.write(f"|--------|-------------|---------------|-------------|-------------|\n")
                
                for result in results:
                    f.write(f"| {result['filename']} | "
                           f"{result['improvements']['snr_improvement']:.2f} | "
                           f"{result['improvements']['correlation_improvement']:.1f} | "
                           f"{result['improvements']['high_freq_retention_improvement']:.1f} | "
                           f"{result['processing_time_enhanced']:.2f} |\n")
                
                f.write(f"\n## 性能分析\n")
                
                # 性能评级
                avg_snr = summary['avg_snr_improvement']
                avg_high_freq = summary['avg_high_freq_improvement']
                
                if avg_high_freq > 50 and avg_snr > -5:
                    rating = "🏆 优秀"
                elif avg_high_freq > 30 and avg_snr > -7:
                    rating = "🥇 良好"
                elif avg_high_freq > 10:
                    rating = "🥉 可用"
                else:
                    rating = "❌ 需要改进"
                
                f.write(f"### 整体评级: {rating}\n\n")
                
                f.write(f"### 技术评估\n")
                f.write(f"- **高频恢复效果**: {'优秀' if avg_high_freq > 50 else '良好' if avg_high_freq > 20 else '一般'}\n")
                f.write(f"- **质量保持**: {'良好' if avg_snr > -5 else '可接受' if avg_snr > -8 else '需要改进'}\n")
                f.write(f"- **处理效率**: {'快速' if summary['avg_processing_time'] < 2 else '中等' if summary['avg_processing_time'] < 5 else '较慢'}\n\n")
                
                f.write(f"## 应用建议\n")
                f.write(f"基于批量验证结果，推荐在以下场景使用增强系数 {summary['boost_factor_used']}x：\n\n")
                
                if avg_high_freq > 50:
                    f.write(f"✅ **推荐用于生产环境**\n")
                    f.write(f"- 高频恢复效果显著\n")
                    f.write(f"- 质量权衡可接受\n")
                    f.write(f"- 适合音乐和语音增强应用\n\n")
                else:
                    f.write(f"⚠️ **建议进一步优化**\n")
                    f.write(f"- 可能需要调整增强参数\n")
                    f.write(f"- 考虑针对不同音频类型使用不同参数\n\n")
                
                f.write(f"## 文件输出\n")
                f.write(f"本次验证生成的文件：\n")
                f.write(f"- 重建音频: `*_original_reconstruction.wav`, `*_enhanced_reconstruction.wav`\n")
                f.write(f"- 对比图表: `*_comparison.png`\n")
                f.write(f"- 验证报告: `batch_validation_report_{timestamp}.md`\n")
            
            print(f"📝 批量验证报告已保存: {report_file}")
            
        except Exception as e:
            print(f"❌ 生成报告时出错: {e}")

def main():
    """主函数"""
    print("🎯 AudioLDM2 VAE增强 - Step 4: 批量验证")
    print("=" * 60)
    
    # 配置Python环境
    from pathlib import Path
    workspace_path = Path(__file__).parent
    
    # 初始化批量验证管道
    validator = BatchValidationPipeline()
    
    # 查找可用的音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    # 在当前目录查找音频文件
    for ext in audio_extensions:
        audio_files.extend(list(workspace_path.glob(f"*{ext}")))
    
    # 在输出目录中查找参考音频
    output_dirs = ['simple_vae_reconstruction', 'simple_latent_enhanced', 'final_parameter_optimization']
    for output_dir in output_dirs:
        output_path = workspace_path / output_dir
        if output_path.exists():
            for ext in audio_extensions:
                audio_files.extend(list(output_path.glob(f"*{ext}")))
    
    # 过滤掉重建文件，只保留原始文件
    original_files = []
    for f in audio_files:
        filename = f.name.lower()
        if not any(keyword in filename for keyword in ['reconstruction', 'enhanced', 'output', 'generated']):
            original_files.append(str(f))
    
    if not original_files:
        print("❌ 未找到音频文件进行验证")
        print("🔍 尝试创建一个测试音频文件...")
        
        # 创建一个简单的测试音频
        test_audio_path = workspace_path / "test_validation.wav"
        duration = 5  # 5秒
        sample_rate = 16000
        t = np.linspace(0, duration, duration * sample_rate)
        
        # 创建包含多个频率成分的测试信号
        signal_440 = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4音符
        signal_880 = 0.2 * np.sin(2 * np.pi * 880 * t)  # A5音符
        signal_1760 = 0.1 * np.sin(2 * np.pi * 1760 * t)  # A6音符（高频）
        
        test_signal = signal_440 + signal_880 + signal_1760
        
        # 添加一些噪声和包络
        envelope = np.exp(-t / 2)  # 衰减包络
        noise = 0.05 * np.random.randn(len(t))
        test_signal = (test_signal * envelope + noise) * 0.5
        
        sf.write(test_audio_path, test_signal, sample_rate)
        original_files = [str(test_audio_path)]
        print(f"✅ 创建测试音频: {test_audio_path}")
    
    print(f"\n📁 找到 {len(original_files)} 个音频文件用于验证:")
    for f in original_files[:5]:  # 显示前5个
        print(f"   📄 {Path(f).name}")
    if len(original_files) > 5:
        print(f"   ... 还有 {len(original_files) - 5} 个文件")
    
    # 执行批量验证
    try:
        summary = validator.batch_validate(original_files)
        
        if summary:
            print(f"\n🎉 批量验证成功完成！")
            print(f"📊 关键结果:")
            print(f"   📈 平均SNR改进: {summary['avg_snr_improvement']:.2f} dB")
            print(f"   🎵 平均高频改进: {summary['avg_high_freq_improvement']:.1f}%")
            print(f"   ⏱️ 平均处理时间: {summary['avg_processing_time']:.2f} 秒")
            
            # 评估整体性能
            if summary['avg_high_freq_improvement'] > 50:
                print(f"   🏆 评级: 优秀 - 推荐用于生产环境")
            elif summary['avg_high_freq_improvement'] > 20:
                print(f"   🥇 评级: 良好 - 适合大多数应用")
            else:
                print(f"   🥉 评级: 需要进一步优化")
        else:
            print("❌ 批量验证失败")
    
    except Exception as e:
        print(f"❌ 批量验证过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
