#!/usr/bin/env python3
"""
Step 1: 改进的VAE重建测试
基于AudioLDM训练代码的正确数据处理流程
专注于：
1. 正确的Mel频谱提取（符合训练时的格式）
2. 子频带处理（如果启用）
3. 准确的VAE编码解码
4. 详细的质量分析
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
# 使用自定义的AudioLDM2Pipeline
from New_pipeline_audioldm2 import AudioLDM2Pipeline
import torchaudio
from pathlib import Path
import time
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class ImprovedVAEReconstructor:
    """
    改进的VAE重建器
    基于AudioLDM训练代码的正确处理流程
    """
    
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        """初始化VAE重建器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎵 初始化改进VAE重建器")
        print(f"   📱 设备: {self.device}")
        
        # 加载AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # 使用float32避免类型问题
        ).to(self.device)
        
        print(f"✅ VAE重建器初始化完成")
        print(f"   📊 VAE scaling_factor: {self.pipeline.vae.config.scaling_factor}")
        print(f"   🎤 特征提取器采样率: {self.pipeline.feature_extractor.sampling_rate}Hz")
        
        # AudioLDM训练时的标准参数
        self.target_sr = 16000  # 训练时使用16kHz
        self.mel_bins = 64      # 训练时使用64个Mel bins
        self.mel_fmax = 8000    # 最大频率8kHz
        self.hop_length = 160   # 训练配置中的hop_length
        self.win_length = 1024  # 训练配置中的win_length
        self.n_fft = 1024       # 训练配置中的filter_length
        
        print(f"   🔧 使用训练时标准参数:")
        print(f"      采样率: {self.target_sr}Hz")
        print(f"      Mel频道: {self.mel_bins}")
        print(f"      最大频率: {self.mel_fmax}Hz")
    
    def reconstruct_audio(self, audio_path: str) -> Dict:
        """
        完整的VAE重建流程（改进版）
        """
        print(f"\n🎵 改进VAE重建处理: {Path(audio_path).name}")
        
        # 1. 加载和预处理输入音频（使用训练时的标准流程）
        original_audio, mel_spectrogram = self._load_and_extract_mel(audio_path)
        
        # 2. VAE编码（模拟训练时的处理）
        latent = self._encode_to_latent(mel_spectrogram)
        
        # 3. VAE解码重建
        reconstructed_mel = self._decode_latent_to_mel(latent)
        
        # 4. Mel转音频
        reconstructed_audio = self._mel_to_audio(reconstructed_mel)
        
        # 5. 保存结果和分析
        result = self._save_and_analyze(original_audio, reconstructed_audio, audio_path, 
                                      mel_spectrogram, reconstructed_mel)
        
        # 6. 显示结果
        self._display_results(result)
        
        return result
    
    def _load_and_extract_mel(self, audio_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """加载音频并提取Mel频谱（按训练时的标准流程）"""
        print(f"   📁 加载音频...")
        
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        
        # 重采样到训练时的采样率 (16kHz)
        if sr != self.target_sr:
            print(f"   🔄 重采样: {sr}Hz -> {self.target_sr}Hz")
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        # 转为单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 限制长度到10.24秒（训练时的标准长度）
        max_length = int(self.target_sr * 10.24)
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
            print(f"   ✂️ 截取到10.24秒")
        
        print(f"   ✅ 音频预处理完成: {audio.shape}, {audio.shape[-1]/self.target_sr:.2f}秒")
        
        # 提取Mel频谱（使用训练时的参数）
        print(f"   🔄 提取Mel频谱...")
        with torch.no_grad():
            audio_np = audio.squeeze(0).numpy()
            
            # 使用librosa提取Mel频谱（更接近训练时的处理）
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np,
                sr=self.target_sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.mel_bins,
                fmax=self.mel_fmax
            )
            
            # 转换为对数刻度
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 转换为tensor并调整维度
            log_mel_tensor = torch.from_numpy(log_mel).float()
            
            # 调整维度为 [batch, channels, time, frequency]
            mel_spectrogram = log_mel_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            print(f"   📊 Mel频谱: {mel_spectrogram.shape}")
            print(f"   📊 Mel范围: [{mel_spectrogram.min():.3f}, {mel_spectrogram.max():.3f}] dB")
            
        return audio_np, mel_spectrogram
    
    def _encode_to_latent(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """编码Mel频谱为latent（模拟训练时的处理）"""
        print(f"   🔄 VAE编码...")
        
        with torch.no_grad():
            # 检查VAE是否有子频带处理
            vae = self.pipeline.vae
            
            # 模拟训练时的encode过程
            # 1. 子频带分解（如果启用）
            if hasattr(vae, 'freq_split_subband'):
                print(f"   🔧 应用子频带分解...")
                mel_input = vae.freq_split_subband(mel_spectrogram)
            else:
                mel_input = mel_spectrogram
            
            print(f"   📊 输入到encoder: {mel_input.shape}")
            
            # 2. Encoder
            if hasattr(vae, 'encoder'):
                h = vae.encoder(mel_input)
            else:
                # 如果直接使用pipeline的VAE
                latent_dist = vae.encode(mel_input)
                latent = latent_dist.latent_dist.mode()
                latent = latent * vae.config.scaling_factor
                print(f"   ✅ VAE编码完成: {latent.shape}")
                print(f"   📊 Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
                print(f"   📊 Latent std: {latent.std():.3f}")
                return latent
            
            # 3. 量化卷积
            if hasattr(vae, 'quant_conv'):
                moments = vae.quant_conv(h)
                # 使用DiagonalGaussianDistribution
                from audioldm_train.modules.diffusionmodules.distributions import DiagonalGaussianDistribution
                posterior = DiagonalGaussianDistribution(moments)
                latent = posterior.mode()  # 使用mode()更稳定
            else:
                # 回退到标准处理
                latent_dist = vae.encode(mel_input)
                latent = latent_dist.latent_dist.mode()
            
            # 4. 应用scaling factor
            if hasattr(vae.config, 'scaling_factor'):
                latent = latent * vae.config.scaling_factor
            
            print(f"   ✅ VAE编码完成: {latent.shape}")
            print(f"   📊 Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
            print(f"   📊 Latent std: {latent.std():.3f}")
            
        return latent
    
    def _decode_latent_to_mel(self, latent: torch.Tensor) -> torch.Tensor:
        """解码latent为Mel频谱（模拟训练时的处理）"""
        print(f"   🔄 VAE解码...")
        
        with torch.no_grad():
            vae = self.pipeline.vae
            
            # 1. 反向scaling
            if hasattr(vae.config, 'scaling_factor'):
                latent_for_decode = latent / vae.config.scaling_factor
            else:
                latent_for_decode = latent
            
            print(f"   📊 解码latent: {latent_for_decode.shape}")
            print(f"   📊 解码latent范围: [{latent_for_decode.min():.3f}, {latent_for_decode.max():.3f}]")
            
            # 2. 解码过程
            if hasattr(vae, 'post_quant_conv') and hasattr(vae, 'decoder'):
                # 训练代码风格的解码
                z = vae.post_quant_conv(latent_for_decode)
                dec = vae.decoder(z)
                
                # 3. 子频带合并（如果启用）
                if hasattr(vae, 'freq_merge_subband'):
                    print(f"   🔧 应用子频带合并...")
                    reconstructed_mel = vae.freq_merge_subband(dec)
                else:
                    reconstructed_mel = dec
            else:
                # 标准pipeline解码
                reconstructed_mel = vae.decode(latent_for_decode).sample
            
            print(f"   ✅ VAE解码完成: {reconstructed_mel.shape}")
            print(f"   📊 重建Mel范围: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}] dB")
            
        return reconstructed_mel
    
    def _mel_to_audio(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """将Mel频谱转换为音频"""
        print(f"   🔄 Mel转音频...")
        
        with torch.no_grad():
            # 使用AudioLDM2的vocoder
            audio_tensor = self.pipeline.mel_spectrogram_to_waveform(mel_spectrogram)
            audio = audio_tensor.squeeze().detach().cpu().numpy()
            
            print(f"   ✅ Mel转音频完成: {len(audio)}样本 ({len(audio)/self.target_sr:.2f}秒)")
            
        return audio
    
    def _save_and_analyze(self, original_audio: np.ndarray, reconstructed_audio: np.ndarray, 
                         audio_path: str, original_mel: torch.Tensor, 
                         reconstructed_mel: torch.Tensor) -> Dict:
        """保存结果并进行质量分析"""
        print(f"   💾 保存结果和质量分析...")
        
        # 创建输出目录
        output_dir = Path("step_1_improved_vae_reconstruction")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        input_name = Path(audio_path).stem
        
        # 确保长度一致
        min_len = min(len(original_audio), len(reconstructed_audio))
        original_audio = original_audio[:min_len]
        reconstructed_audio = reconstructed_audio[:min_len]
        
        # 保存音频文件
        original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
        reconstructed_path = output_dir / f"{input_name}_improved_vae_reconstructed_{timestamp}.wav"
        
        sf.write(str(original_path), original_audio, self.target_sr)
        sf.write(str(reconstructed_path), reconstructed_audio, self.target_sr)
        
        # 音频质量分析
        audio_analysis = self._analyze_audio_quality(original_audio, reconstructed_audio)
        
        # 频谱分析
        spectral_analysis = self._analyze_frequency_content(original_audio, reconstructed_audio)
        
        # Mel频谱分析
        mel_analysis = self._analyze_mel_spectrogram(original_mel, reconstructed_mel)
        
        # 生成对比图
        self._plot_comprehensive_analysis(original_audio, reconstructed_audio, 
                                        original_mel, reconstructed_mel, 
                                        output_dir, timestamp)
        
        result = {
            "input_file": audio_path,
            "original_path": str(original_path),
            "reconstructed_path": str(reconstructed_path),
            "audio_length": min_len / self.target_sr,
            "audio_quality_metrics": audio_analysis,
            "frequency_analysis": spectral_analysis,
            "mel_analysis": mel_analysis,
            "timestamp": timestamp
        }
        
        return result
    
    def _analyze_audio_quality(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """分析音频质量指标"""
        # 基础质量指标
        mse = np.mean((original - reconstructed) ** 2)
        snr = 10 * np.log10(np.mean(original ** 2) / (mse + 1e-10))
        correlation = np.corrcoef(original, reconstructed)[0, 1] if len(original) > 1 else 0
        mae = np.mean(np.abs(original - reconstructed))
        
        # RMS比较
        rms_original = np.sqrt(np.mean(original ** 2))
        rms_reconstructed = np.sqrt(np.mean(reconstructed ** 2))
        rms_ratio = rms_reconstructed / (rms_original + 1e-10)
        
        return {
            "snr_db": snr,
            "correlation": correlation,
            "mse": mse,
            "mae": mae,
            "rms_original": rms_original,
            "rms_reconstructed": rms_reconstructed,
            "rms_ratio": rms_ratio
        }
    
    def _analyze_frequency_content(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """分析频谱内容"""
        if len(original) < 8192:
            return {"error": "音频太短，无法进行频谱分析"}
        
        # FFT分析
        fft_len = 8192
        orig_fft = np.abs(np.fft.fft(original[:fft_len]))[:fft_len//2]
        recon_fft = np.abs(np.fft.fft(reconstructed[:fft_len]))[:fft_len//2]
        
        # 频段分析
        freqs = np.fft.fftfreq(fft_len, 1/self.target_sr)[:fft_len//2]
        
        # 定义频段（更细致的划分）
        low_freq_mask = freqs < 500     # 极低频 0-500Hz
        mid_low_mask = (freqs >= 500) & (freqs < 1000)   # 低频 500Hz-1kHz
        mid_freq_mask = (freqs >= 1000) & (freqs < 2000)  # 中低频 1-2kHz
        mid_high_mask = (freqs >= 2000) & (freqs < 4000)  # 中高频 2-4kHz
        high_freq_mask = (freqs >= 4000) & (freqs < 6000) # 高频 4-6kHz
        ultra_high_mask = freqs >= 6000  # 超高频 6-8kHz
        
        # 计算各频段能量
        def calc_retention(orig_mask, recon_mask):
            orig_energy = np.sum(orig_fft[orig_mask])
            recon_energy = np.sum(recon_fft[recon_mask])
            return recon_energy / (orig_energy + 1e-10)
        
        return {
            "ultra_low_retention": calc_retention(low_freq_mask, low_freq_mask),
            "low_freq_retention": calc_retention(mid_low_mask, mid_low_mask),
            "mid_low_retention": calc_retention(mid_freq_mask, mid_freq_mask),
            "mid_high_retention": calc_retention(mid_high_mask, mid_high_mask),
            "high_freq_retention": calc_retention(high_freq_mask, high_freq_mask),
            "ultra_high_retention": calc_retention(ultra_high_mask, ultra_high_mask),
            "total_energy_ratio": np.sum(recon_fft) / (np.sum(orig_fft) + 1e-10)
        }
    
    def _analyze_mel_spectrogram(self, original_mel: torch.Tensor, 
                               reconstructed_mel: torch.Tensor) -> Dict:
        """分析Mel频谱的重建质量"""
        with torch.no_grad():
            orig_mel_np = original_mel.squeeze().detach().cpu().numpy()
            recon_mel_np = reconstructed_mel.squeeze().detach().cpu().numpy()
            
            # Mel频谱MSE
            mel_mse = np.mean((orig_mel_np - recon_mel_np) ** 2)
            
            # Mel频谱相关性
            flat_orig = orig_mel_np.flatten()
            flat_recon = recon_mel_np.flatten()
            mel_correlation = np.corrcoef(flat_orig, flat_recon)[0, 1]
            
            # 不同频率bin的保持情况
            freq_bin_retention = []
            for i in range(min(orig_mel_np.shape[-1], recon_mel_np.shape[-1])):
                orig_bin = orig_mel_np[..., i]
                recon_bin = recon_mel_np[..., i]
                bin_corr = np.corrcoef(orig_bin.flatten(), recon_bin.flatten())[0, 1]
                freq_bin_retention.append(bin_corr)
            
            return {
                "mel_mse": mel_mse,
                "mel_correlation": mel_correlation,
                "avg_freq_bin_retention": np.mean(freq_bin_retention),
                "low_freq_bins_retention": np.mean(freq_bin_retention[:16]),
                "mid_freq_bins_retention": np.mean(freq_bin_retention[16:48]),
                "high_freq_bins_retention": np.mean(freq_bin_retention[48:])
            }
    
    def _plot_comprehensive_analysis(self, original_audio: np.ndarray, reconstructed_audio: np.ndarray,
                                   original_mel: torch.Tensor, reconstructed_mel: torch.Tensor,
                                   output_dir: Path, timestamp: int):
        """生成综合分析图表"""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            
            # 原始和重建Mel频谱对比
            orig_mel_np = original_mel.squeeze().detach().cpu().numpy()
            recon_mel_np = reconstructed_mel.squeeze().detach().cpu().numpy()
            
            im1 = axes[0,0].imshow(orig_mel_np, aspect='auto', origin='lower', cmap='viridis')
            axes[0,0].set_title('原始Mel频谱')
            axes[0,0].set_ylabel('Mel频率bin')
            plt.colorbar(im1, ax=axes[0,0])
            
            im2 = axes[0,1].imshow(recon_mel_np, aspect='auto', origin='lower', cmap='viridis')
            axes[0,1].set_title('重建Mel频谱')
            axes[0,1].set_ylabel('Mel频率bin')
            plt.colorbar(im2, ax=axes[0,1])
            
            # Mel频谱差异
            mel_diff = orig_mel_np - recon_mel_np
            im3 = axes[1,0].imshow(mel_diff, aspect='auto', origin='lower', cmap='RdBu_r')
            axes[1,0].set_title('Mel频谱差异 (原始-重建)')
            axes[1,0].set_ylabel('Mel频率bin')
            plt.colorbar(im3, ax=axes[1,0])
            
            # 频率bin平均保持情况
            freq_bin_means_orig = np.mean(orig_mel_np, axis=0)
            freq_bin_means_recon = np.mean(recon_mel_np, axis=0)
            
            axes[1,1].plot(freq_bin_means_orig, label='原始', alpha=0.7)
            axes[1,1].plot(freq_bin_means_recon, label='重建', alpha=0.7)
            axes[1,1].set_title('各频率bin平均能量')
            axes[1,1].set_xlabel('Mel频率bin')
            axes[1,1].set_ylabel('平均能量 (dB)')
            axes[1,1].legend()
            axes[1,1].grid(True)
            
            # 时域波形对比
            time_axis = np.linspace(0, len(original_audio)/self.target_sr, len(original_audio))
            
            axes[2,0].plot(time_axis, original_audio, label='原始', alpha=0.7)
            min_len = min(len(original_audio), len(reconstructed_audio))
            axes[2,0].plot(time_axis[:min_len], reconstructed_audio[:min_len], label='重建', alpha=0.7)
            axes[2,0].set_title('时域波形对比')
            axes[2,0].set_xlabel('时间 (s)')
            axes[2,0].set_ylabel('幅度')
            axes[2,0].legend()
            axes[2,0].grid(True)
            
            # 频域对比
            fft_len = min(8192, len(original_audio), len(reconstructed_audio))
            orig_fft = np.abs(np.fft.fft(original_audio[:fft_len]))[:fft_len//2]
            recon_fft = np.abs(np.fft.fft(reconstructed_audio[:fft_len]))[:fft_len//2]
            freqs = np.fft.fftfreq(fft_len, 1/self.target_sr)[:fft_len//2]
            
            axes[2,1].semilogy(freqs, orig_fft, label='原始', alpha=0.7)
            axes[2,1].semilogy(freqs, recon_fft, label='重建', alpha=0.7)
            axes[2,1].set_xlabel('频率 (Hz)')
            axes[2,1].set_ylabel('幅度')
            axes[2,1].set_title('频域响应对比')
            axes[2,1].legend()
            axes[2,1].grid(True)
            axes[2,1].set_xlim([0, self.mel_fmax])
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = output_dir / f"comprehensive_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 综合分析图已保存: {plot_path}")
            
        except Exception as e:
            print(f"   ⚠️ 分析图生成失败: {e}")
    
    def _display_results(self, result: Dict):
        """显示详细结果"""
        print(f"\n{'='*80}")
        print(f"🎯 改进VAE重建结果")
        print(f"{'='*80}")
        print(f"📁 输入文件: {result['input_file']}")
        print(f"📁 原始音频: {result['original_path']}")
        print(f"📁 重建音频: {result['reconstructed_path']}")
        print(f"⏱️ 音频长度: {result['audio_length']:.2f}秒")
        
        # 音频质量指标
        audio_metrics = result['audio_quality_metrics']
        print(f"\n📊 音频质量指标:")
        print(f"   SNR: {audio_metrics['snr_db']:.2f} dB")
        print(f"   相关系数: {audio_metrics['correlation']:.4f}")
        print(f"   MSE: {audio_metrics['mse']:.6f}")
        print(f"   MAE: {audio_metrics['mae']:.6f}")
        print(f"   RMS比率: {audio_metrics['rms_ratio']:.4f}")
        
        # 频率分析
        freq_analysis = result['frequency_analysis']
        if 'error' not in freq_analysis:
            print(f"\n🎵 详细频率分析:")
            print(f"   极低频保持率 (0-500Hz): {freq_analysis['ultra_low_retention']:.3f}")
            print(f"   低频保持率 (500Hz-1kHz): {freq_analysis['low_freq_retention']:.3f}")
            print(f"   中低频保持率 (1-2kHz): {freq_analysis['mid_low_retention']:.3f}")
            print(f"   中高频保持率 (2-4kHz): {freq_analysis['mid_high_retention']:.3f}")
            print(f"   高频保持率 (4-6kHz): {freq_analysis['high_freq_retention']:.3f}")
            print(f"   超高频保持率 (6-8kHz): {freq_analysis['ultra_high_retention']:.3f}")
            print(f"   总能量比率: {freq_analysis['total_energy_ratio']:.3f}")
        
        # Mel频谱分析
        mel_analysis = result['mel_analysis']
        print(f"\n🔍 Mel频谱分析:")
        print(f"   Mel MSE: {mel_analysis['mel_mse']:.6f}")
        print(f"   Mel相关性: {mel_analysis['mel_correlation']:.4f}")
        print(f"   平均频率bin保持率: {mel_analysis['avg_freq_bin_retention']:.4f}")
        print(f"   低频bin保持率: {mel_analysis['low_freq_bins_retention']:.4f}")
        print(f"   中频bin保持率: {mel_analysis['mid_freq_bins_retention']:.4f}")
        print(f"   高频bin保持率: {mel_analysis['high_freq_bins_retention']:.4f}")
        
        # 诊断和建议
        self._provide_diagnosis_and_suggestions(result)
    
    def _provide_diagnosis_and_suggestions(self, result: Dict):
        """提供诊断和改进建议"""
        print(f"\n💡 诊断和建议:")
        
        audio_metrics = result['audio_quality_metrics']
        freq_analysis = result['frequency_analysis']
        mel_analysis = result['mel_analysis']
        
        # SNR诊断
        if audio_metrics['snr_db'] > 10:
            print(f"   ✅ SNR良好 ({audio_metrics['snr_db']:.1f}dB)，基础重建质量不错")
        elif audio_metrics['snr_db'] > 0:
            print(f"   ⚠️ SNR一般 ({audio_metrics['snr_db']:.1f}dB)，有改善空间")
        else:
            print(f"   ❌ SNR较低 ({audio_metrics['snr_db']:.1f}dB)，重建质量需要改进")
        
        # 高频诊断
        high_freq_retention = freq_analysis.get('high_freq_retention', 0)
        ultra_high_retention = freq_analysis.get('ultra_high_retention', 0)
        
        if high_freq_retention < 0.3:
            print(f"   ❌ 严重高频损失！4-6kHz保持率仅 {high_freq_retention*100:.1f}%")
        elif high_freq_retention < 0.6:
            print(f"   ⚠️ 明显高频损失，4-6kHz保持率 {high_freq_retention*100:.1f}%")
        else:
            print(f"   ✅ 高频保持较好，4-6kHz保持率 {high_freq_retention*100:.1f}%")
        
        if ultra_high_retention < 0.1:
            print(f"   ❌ 超高频几乎完全丢失！6-8kHz保持率仅 {ultra_high_retention*100:.1f}%")
        
        # Mel频谱诊断
        if mel_analysis['mel_correlation'] > 0.8:
            print(f"   ✅ Mel频谱重建良好 (相关性: {mel_analysis['mel_correlation']:.3f})")
        else:
            print(f"   ⚠️ Mel频谱重建有待改进 (相关性: {mel_analysis['mel_correlation']:.3f})")
        
        # 改进建议
        print(f"\n🔧 改进建议:")
        if high_freq_retention < 0.5:
            print(f"   🎯 需要在latent空间进行高频增强")
            print(f"   🎯 考虑频率感知的损失函数")
        
        if mel_analysis['high_freq_bins_retention'] < 0.7:
            print(f"   🎯 高频Mel bin重建不佳，需要针对性优化")
        
        print(f"   📊 详细分析图表已生成，可进一步分析")

def test_improved_vae():
    """测试改进的VAE重建"""
    print("🎵 改进VAE重建测试")
    print("=" * 60)
    
    # 检查输入文件
    input_file = "AudioLDM2_Music_output.wav"
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        print("请确保AudioLDM2_Music_output.wav在当前目录中")
        return
    
    # 创建改进的VAE重建器
    vae_reconstructor = ImprovedVAEReconstructor()
    
    # 执行重建
    result = vae_reconstructor.reconstruct_audio(input_file)
    
    print(f"\n✅ 改进VAE重建测试完成！")
    print(f"📁 查看输出目录: step_1_improved_vae_reconstruction/")
    print(f"🎵 对比原始、重建音频和详细分析图表")

if __name__ == "__main__":
    test_improved_vae()
