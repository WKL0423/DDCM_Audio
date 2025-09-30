#!/usr/bin/env python3
"""
Step 1: 与训练管道完全匹配的VAE重建测试（修正版）
基于训练代码的autoencoder.py和配置文件16k_64.yaml创建
确保：
1. 使用16kHz采样率（训练配置）
2. 使用与训练完全一致的mel频谱图提取方法
3. 不使用子频带处理（subband=1）
4. 正确的数据格式、padding和维度处理
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import torchaudio
from pathlib import Path
import time
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 尝试导入AudioLDM2Pipeline
try:
    from New_pipeline_audioldm2 import AudioLDM2Pipeline
except ImportError:
    print("Warning: 无法导入New_pipeline_audioldm2，尝试使用标准管道")
    from diffusers import AudioLDM2Pipeline

class TrainingMatchedVAEReconstructor:
    """
    与训练管道完全匹配的VAE重建器
    严格按照训练代码的数据处理流程
    """
    
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        """初始化VAE重建器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎵 初始化训练匹配VAE重建器")
        print(f"   📱 设备: {self.device}")
        
        # 训练配置参数（来自16k_64.yaml）
        self.sampling_rate = 16000  # 训练采样率
        self.mel_bins = 64          # mel频段数
        self.mel_fmin = 0           # mel最小频率
        self.mel_fmax = 8000        # mel最大频率
        self.n_fft = 1024           # FFT窗口大小
        self.hop_length = 160       # 跳跃长度
        self.win_length = 1024      # 窗口长度
        self.subband = 1            # 不使用子频带处理
        self.duration = 10.24       # 训练音频持续时间
        
        print(f"   📊 训练配置参数:")
        print(f"      采样率: {self.sampling_rate}Hz")
        print(f"      Mel bins: {self.mel_bins}")
        print(f"      Mel频率范围: {self.mel_fmin}-{self.mel_fmax}Hz")
        print(f"      子频带数: {self.subband}")
        print(f"      音频持续时间: {self.duration}s")
        
        # 加载AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        print(f"✅ VAE重建器初始化完成")
        print(f"   📊 VAE scaling_factor: {self.pipeline.vae.config.scaling_factor}")
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        """
        使用与训练完全一致的mel频谱图提取方法
        """
        # 确保音频是1D数组
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # 转为torch张量
        waveform = torch.FloatTensor(audio).unsqueeze(0)  # [1, samples]
        
        # 计算目标长度（与训练代码一致）
        target_length = int(self.duration * self.sampling_rate / self.hop_length)
        
        # 使用与训练代码一致的mel频谱图提取
        mel_spec, stft_spec = self._mel_spectrogram_train(waveform)
        
        # 转置并padding到目标长度（与训练代码一致）
        log_mel = mel_spec.transpose(-1, -2)  # [batch, time, freq]
        log_mel = self._pad_spec(log_mel, target_length)
        
        # 调整维度为[batch, channel, time, freq]
        log_mel_tensor = log_mel.unsqueeze(1)  # [1, 1, time, freq]
        
        return log_mel_tensor
    
    def _mel_spectrogram_train(self, y):
        """
        与训练代码完全一致的mel频谱图提取
        """
        # 创建mel基和hann窗口（如果不存在）
        if not hasattr(self, 'mel_basis'):
            mel = librosa.filters.mel(
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                n_mels=self.mel_bins,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )
            self.mel_basis = torch.from_numpy(mel).float()
            self.hann_window = torch.hann_window(self.win_length)
        
        # Padding（与训练代码一致）
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_length) / 2),
                int((self.n_fft - self.hop_length) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        
        # STFT（与训练代码一致）
        stft_spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        
        stft_spec = torch.abs(stft_spec)
        
        # 计算mel频谱图
        mel = torch.matmul(self.mel_basis, stft_spec)
        
        # Spectral normalization（与训练代码一致）
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return mel, stft_spec
    
    def _pad_spec(self, log_mel_spec, target_length):
        """
        与训练代码完全一致的padding方法
        """
        n_frames = log_mel_spec.shape[1]  # time维度
        p = target_length - n_frames
        
        # cut and pad
        if p > 0:
            # padding
            m = torch.nn.ZeroPad2d((0, 0, 0, p))  # (left, right, top, bottom)
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            # cut
            log_mel_spec = log_mel_spec[:, :target_length, :]
        
        # 确保频率维度是偶数（与训练代码一致）
        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]
        
        return log_mel_spec
    
    def reconstruct_audio(self, audio_path: str) -> Dict:
        """
        完整的VAE重建流程，严格匹配训练管道
        
        Args:
            audio_path: 输入音频路径
            
        Returns:
            result: 包含重建结果和分析的字典
        """
        print(f"\n🎵 训练匹配VAE重建: {Path(audio_path).name}")
        
        # 1. 加载和预处理输入音频
        original_audio, mel_features, input_latent = self._load_and_encode_audio(audio_path)
        
        # 2. VAE解码重建
        reconstructed_mel, reconstructed_audio = self._decode_latent_to_audio(input_latent)
        
        # 3. 保存结果和分析
        result = self._save_and_analyze(
            original_audio, reconstructed_audio, mel_features, reconstructed_mel, audio_path
        )
        
        # 4. 显示结果
        self._display_results(result)
        
        return result
    
    def _load_and_encode_audio(self, audio_path: str) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """加载音频并编码为latent，严格匹配训练流程"""
        print(f"   📁 加载音频...")
        
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        
        # 重采样到训练采样率 (16kHz)
        if sr != self.sampling_rate:
            print(f"   🔄 重采样: {sr}Hz -> {self.sampling_rate}Hz")
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)
        
        # 转为单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 限制长度到训练配置时长
        max_length = int(self.duration * self.sampling_rate)
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
            print(f"   ✂️ 截取到{self.duration}秒")
        
        # 转为numpy用于mel提取
        audio_np = audio.squeeze(0).numpy()
        
        print(f"   ✅ 音频预处理完成: {len(audio_np)}样本 ({len(audio_np)/self.sampling_rate:.2f}秒)")
        
        # 使用与训练一致的mel频谱图提取
        print(f"   🔄 提取mel频谱图...")
        mel_features = self.extract_mel_spectrogram(audio_np)
        
        print(f"   📊 Mel特征: {mel_features.shape}")
        print(f"   📊 Mel范围: [{mel_features.min():.3f}, {mel_features.max():.3f}]")
        
        # VAE编码
        print(f"   🔄 VAE编码...")
        with torch.no_grad():
            # 移到设备并转换数据类型
            mel_features = mel_features.to(self.device)
            if self.device == "cuda":
                mel_features = mel_features.half()
            
            # VAE编码（不需要子频带处理，因为subband=1）
            latent_dist = self.pipeline.vae.encode(mel_features)
            
            # 获取latent（使用mode而不是sample）
            if hasattr(latent_dist, 'latent_dist'):
                latent = latent_dist.latent_dist.mode()
            else:
                latent = latent_dist.mode()
            
            # 应用scaling factor
            latent = latent * self.pipeline.vae.config.scaling_factor
            
            print(f"   ✅ VAE编码完成: {latent.shape}")
            print(f"   📊 Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
            print(f"   📊 Latent std: {latent.std():.3f}")
        
        return audio_np, mel_features, latent
    
    def _decode_latent_to_audio(self, latent: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """将latent解码为音频，严格匹配训练流程"""
        print(f"   🔄 VAE解码...")
        
        with torch.no_grad():
            # 准备解码
            latent_for_decode = latent / self.pipeline.vae.config.scaling_factor
            
            # 确保数据类型匹配
            vae_dtype = next(self.pipeline.vae.parameters()).dtype
            latent_for_decode = latent_for_decode.to(vae_dtype)
            
            print(f"   📊 解码latent: {latent_for_decode.shape}, {latent_for_decode.dtype}")
            print(f"   📊 解码latent范围: [{latent_for_decode.min():.3f}, {latent_for_decode.max():.3f}]")
            
            # VAE解码
            reconstructed_mel = self.pipeline.vae.decode(latent_for_decode).sample
            
            print(f"   📊 重建mel: {reconstructed_mel.shape}")
            print(f"   📊 重建mel范围: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
            
            # 使用vocoder转换为音频
            audio_tensor = self.pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
            audio = audio_tensor.squeeze().cpu().numpy()
            
            print(f"   ✅ VAE解码完成: {len(audio)}样本 ({len(audio)/16000:.2f}秒)")
            
        return reconstructed_mel, audio
    
    def _save_and_analyze(self, original_audio: np.ndarray, reconstructed_audio: np.ndarray,
                         original_mel: torch.Tensor, reconstructed_mel: torch.Tensor,
                         audio_path: str) -> Dict:
        """保存结果并进行全面质量分析"""
        print(f"   💾 保存结果和质量分析...")
        
        # 创建输出目录
        output_dir = Path("step_1_training_matched_vae_reconstruction_fixed")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        input_name = Path(audio_path).stem
        
        # 确保长度一致
        min_len = min(len(original_audio), len(reconstructed_audio))
        original_audio_trimmed = original_audio[:min_len]
        reconstructed_audio_trimmed = reconstructed_audio[:min_len]
        
        # 保存音频文件
        original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
        reconstructed_path = output_dir / f"{input_name}_training_matched_reconstructed_{timestamp}.wav"
        
        sf.write(str(original_path), original_audio_trimmed, self.sampling_rate)
        sf.write(str(reconstructed_path), reconstructed_audio_trimmed, self.sampling_rate)
        
        # 计算音频质量指标
        snr = self._calculate_snr(original_audio_trimmed, reconstructed_audio_trimmed)
        mse = np.mean((original_audio_trimmed - reconstructed_audio_trimmed) ** 2)
          # 计算频域分析
        freq_analysis = self._analyze_frequency_content(original_audio_trimmed, reconstructed_audio_trimmed)
        
        # 计算mel频谱图差异（使用训练方式的padding）
        mel_analysis = self._analyze_mel_difference_training_way(original_mel, reconstructed_mel)
          # 计算详细的误差分析
        error_analysis = self._analyze_error_patterns(original_audio_trimmed, reconstructed_audio_trimmed)
        
        # 生成可视化图表
        plot_path = self._generate_analysis_plots(
            original_audio_trimmed, reconstructed_audio_trimmed,
            original_mel, reconstructed_mel,
            output_dir, input_name, timestamp
        )
          # 组织结果
        result = {
            "input_path": audio_path,
            "output_dir": str(output_dir),
            "original_path": str(original_path),
            "reconstructed_path": str(reconstructed_path),
            "plot_path": str(plot_path),
            "audio_metrics": {
                "snr_db": snr,
                "mse": mse,
                "original_length": len(original_audio_trimmed),
                "reconstructed_length": len(reconstructed_audio_trimmed),
                "sampling_rate": self.sampling_rate            },
            "frequency_analysis": freq_analysis,
            "mel_analysis": mel_analysis,
            "error_analysis": error_analysis
        }
        
        return result
    
    def _calculate_snr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """计算信噪比"""
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        if noise_power == 0:
            return float('inf')
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def _analyze_frequency_content(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """分析频域内容"""
        # 计算频谱
        orig_fft = np.abs(np.fft.rfft(original))
        recon_fft = np.abs(np.fft.rfft(reconstructed))
        
        # 频率轴
        freqs = np.fft.rfftfreq(len(original), 1/self.sampling_rate)
        
        # 计算不同频段的能量损失
        nyquist = self.sampling_rate // 2
        low_freq_mask = freqs <= nyquist * 0.25  # 0-2kHz
        mid_freq_mask = (freqs > nyquist * 0.25) & (freqs <= nyquist * 0.5)  # 2-4kHz
        high_freq_mask = freqs > nyquist * 0.5  # 4-8kHz
        
        def energy_ratio(mask):
            orig_energy = np.sum(orig_fft[mask] ** 2)
            recon_energy = np.sum(recon_fft[mask] ** 2)
            return recon_energy / orig_energy if orig_energy > 0 else 0
        
        return {
            "low_freq_ratio": energy_ratio(low_freq_mask),
            "mid_freq_ratio": energy_ratio(mid_freq_mask),
            "high_freq_ratio": energy_ratio(high_freq_mask),
            "spectral_correlation": np.corrcoef(orig_fft, recon_fft)[0, 1]
        }
    
    def _analyze_mel_difference_training_way(self, original_mel: torch.Tensor, reconstructed_mel: torch.Tensor) -> Dict:
        """
        使用训练代码的方式分析mel频谱图差异
        都应该已经padding到相同的target_length
        """
        # 转为numpy
        orig_mel_np = original_mel.squeeze().cpu().numpy()
        recon_mel_np = reconstructed_mel.squeeze().cpu().numpy()
        
        print(f"   📊 Mel分析维度: 原始{orig_mel_np.shape}, 重建{recon_mel_np.shape}")
        
        # 如果维度仍然不匹配，使用训练代码的padding方法
        if orig_mel_np.shape != recon_mel_np.shape:
            print(f"   ⚠️ Mel维度不匹配，使用训练方式padding...")
            target_length = int(self.duration * self.sampling_rate / self.hop_length)
            
            # 对原始mel应用相同的padding
            orig_mel_tensor = torch.from_numpy(orig_mel_np).unsqueeze(0)  # [1, time, freq]
            orig_mel_tensor = self._pad_spec(orig_mel_tensor, target_length)
            orig_mel_np = orig_mel_tensor.squeeze(0).numpy()
            
            # 对重建mel也应用padding（如果需要）
            recon_mel_tensor = torch.from_numpy(recon_mel_np).unsqueeze(0)  # [1, time, freq]  
            recon_mel_tensor = self._pad_spec(recon_mel_tensor, target_length)
            recon_mel_np = recon_mel_tensor.squeeze(0).numpy()
            
            print(f"   📊 Padding后维度: 原始{orig_mel_np.shape}, 重建{recon_mel_np.shape}")
        
        # 计算各种指标
        mse = np.mean((orig_mel_np - recon_mel_np) ** 2)
        mae = np.mean(np.abs(orig_mel_np - recon_mel_np))
        correlation = np.corrcoef(orig_mel_np.flatten(), recon_mel_np.flatten())[0, 1]
        
        # 计算不同频段的误差
        n_mels = orig_mel_np.shape[-1]
        low_mel_error = np.mean((orig_mel_np[..., :n_mels//3] - recon_mel_np[..., :n_mels//3]) ** 2)
        mid_mel_error = np.mean((orig_mel_np[..., n_mels//3:2*n_mels//3] - recon_mel_np[..., n_mels//3:2*n_mels//3]) ** 2)
        high_mel_error = np.mean((orig_mel_np[..., 2*n_mels//3:] - recon_mel_np[..., 2*n_mels//3:]) ** 2)
        
        return {
            "mel_mse": mse,
            "mel_mae": mae,
            "mel_correlation": correlation,
            "low_mel_error": low_mel_error,
            "mid_mel_error": mid_mel_error,
            "high_mel_error": high_mel_error,
            "time_frames_original": orig_mel_np.shape[0],
            "time_frames_reconstructed": recon_mel_np.shape[0],
            "padding_applied": orig_mel_np.shape != recon_mel_np.shape
        }
    
    def _generate_analysis_plots(self, original_audio: np.ndarray, reconstructed_audio: np.ndarray,
                               original_mel: torch.Tensor, reconstructed_mel: torch.Tensor,
                               output_dir: Path, input_name: str, timestamp: int) -> str:
        """生成综合分析图表"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'analyzing graph - {input_name}', fontsize=16)
        
        # 1. 波形对比
        time_axis = np.arange(len(original_audio)) / self.sampling_rate
        axes[0, 0].plot(time_axis, original_audio, label='original', alpha=0.7)
        axes[0, 0].plot(time_axis, reconstructed_audio, label='reconstructed', alpha=0.7)
        axes[0, 0].set_title('waveform comparison')
        axes[0, 0].set_xlabel('time (s)')
        axes[0, 0].set_ylabel('amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 误差波形
        error = original_audio - reconstructed_audio
        axes[0, 1].plot(time_axis, error)
        axes[0, 1].set_title('waveform error')
        axes[0, 1].set_xlabel('time (s)')
        axes[0, 1].set_ylabel('error amplitude')
        axes[0, 1].grid(True)
        
        # 3. 频谱对比
        orig_fft = np.abs(np.fft.rfft(original_audio))
        recon_fft = np.abs(np.fft.rfft(reconstructed_audio))
        freqs = np.fft.rfftfreq(len(original_audio), 1/self.sampling_rate)
        
        axes[1, 0].semilogy(freqs, orig_fft, label='original', alpha=0.7)
        axes[1, 0].semilogy(freqs, recon_fft, label='reconstructed', alpha=0.7)
        axes[1, 0].set_title('spectrum comparison')
        axes[1, 0].set_xlabel('frequency (Hz)')
        axes[1, 0].set_ylabel('magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 频谱误差
        freq_error = np.abs(orig_fft - recon_fft)
        axes[1, 1].semilogy(freqs, freq_error)
        axes[1, 1].set_title('spectrum error')
        axes[1, 1].set_xlabel('frequency (Hz)')
        axes[1, 1].set_ylabel('error magnitude')
        axes[1, 1].grid(True)
        
        # 5. & 6. Mel频谱图对比（使用训练方式处理维度）
        orig_mel_np = original_mel.squeeze().cpu().numpy()
        recon_mel_np = reconstructed_mel.squeeze().cpu().numpy()
        
        # 确保维度匹配（使用训练方式的padding）
        if orig_mel_np.shape != recon_mel_np.shape:
            target_length = int(self.duration * self.sampling_rate / self.hop_length)
            
            orig_mel_tensor = torch.from_numpy(orig_mel_np).unsqueeze(0)
            orig_mel_tensor = self._pad_spec(orig_mel_tensor, target_length)
            orig_mel_np = orig_mel_tensor.squeeze(0).numpy()
            
            recon_mel_tensor = torch.from_numpy(recon_mel_np).unsqueeze(0)
            recon_mel_tensor = self._pad_spec(recon_mel_tensor, target_length)
            recon_mel_np = recon_mel_tensor.squeeze(0).numpy()
        
        im1 = axes[2, 0].imshow(orig_mel_np.T, aspect='auto', origin='lower', cmap='viridis')
        axes[2, 0].set_title('original Mel spectrogram')
        axes[2, 0].set_xlabel('time frames')
        axes[2, 0].set_ylabel('Mel frequency bins')
        plt.colorbar(im1, ax=axes[2, 0])
        
        im2 = axes[2, 1].imshow(recon_mel_np.T, aspect='auto', origin='lower', cmap='viridis')
        axes[2, 1].set_title('reconstructed Mel spectrogram')
        axes[2, 1].set_xlabel('time frames')
        axes[2, 1].set_ylabel('Mel frequency bins')
        plt.colorbar(im2, ax=axes[2, 1])
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = output_dir / f"{input_name}_training_matched_analysis_fixed_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _display_results(self, result: Dict):
        """显示分析结果"""
        print(f"\n📊 训练匹配VAE重建分析结果（修正版）:")
        print(f"   📁 输出目录: {result['output_dir']}")
        
        metrics = result["audio_metrics"]
        print(f"   🎵 音频质量:")
        print(f"      SNR: {metrics['snr_db']:.2f} dB")
        print(f"      MSE: {metrics['mse']:.6f}")
        
        freq_analysis = result["frequency_analysis"]
        print(f"   📊 频域分析:")
        print(f"      低频保持: {freq_analysis['low_freq_ratio']:.3f}")
        print(f"      中频保持: {freq_analysis['mid_freq_ratio']:.3f}")
        print(f"      高频保持: {freq_analysis['high_freq_ratio']:.3f}")
        print(f"      频谱相关性: {freq_analysis['spectral_correlation']:.3f}")
        mel_analysis = result["mel_analysis"]
        print(f"   🎼 Mel频谱分析:")
        print(f"      Mel MSE: {mel_analysis['mel_mse']:.6f}")
        print(f"      Mel相关性: {mel_analysis['mel_correlation']:.3f}")
        print(f"      低频段误差: {mel_analysis['low_mel_error']:.6f}")
        print(f"      中频段误差: {mel_analysis['mid_mel_error']:.6f}")
        print(f"      高频段误差: {mel_analysis['high_mel_error']:.6f}")
        print(f"      时间帧数: 原始{mel_analysis['time_frames_original']}, 重建{mel_analysis['time_frames_reconstructed']}")
        
        # 显示详细的误差分析
        error_interpretation = self._interpret_error_analysis(result["error_analysis"])
        print(f"\n{error_interpretation}")
        
        # 误差分析结果
        error_analysis = result["error_analysis"]
        wf_error = error_analysis["waveform_error"]
        sp_error = error_analysis["spectrum_error"]
        pattern = wf_error["pattern"]
        
        print(f"   📉 误差分析:")
        print(f"      波形RMS误差: {wf_error['rms']:.4f}")
        print(f"      波形最大误差: {wf_error['max']:.4f}")
        print(f"      低频误差: {sp_error['low_freq_error']:.4f} (0-2kHz)")
        print(f"      中频误差: {sp_error['mid_freq_error']:.4f} (2-4kHz)")
        print(f"      高频误差: {sp_error['high_freq_error']:.4f} (4-8kHz)")
        print(f"      误差集中度: {sp_error['error_concentration']:.2f} (高频/低频)")
        
        if pattern["has_dc_offset"]:
            print(f"   ⚠️ 检测到DC偏移 - 重建信号存在直流分量偏差")
        
        if pattern["is_periodic"]:
            print(f"   🔄 检测到周期性误差 - 可能某些频率成分丢失")
        
        if pattern["has_spikes"]:
            print(f"   ⚡ 检测到突发误差({pattern['spike_ratio']:.1%}) - 瞬态信号处理不佳")
        
        if not any([pattern["has_dc_offset"], pattern["is_periodic"], pattern["has_spikes"]]):
            print(f"   ✅ 误差模式良好 - 主要为随机噪声")
        
        # 总体评价
        if wf_error['rms'] < 0.1 and sp_error['error_concentration'] < 3:
            print(f"   ✅ 重建质量良好")
        elif wf_error['rms'] < 0.2 and sp_error['error_concentration'] < 5:
            print(f"   ⚠️ 重建质量中等，有改善空间")
        else:
            print(f"   ❌ 重建质量较差，需要优化")
        
        print(f"📊 查看详细分析图表: {result['plot_path']}")

    def _analyze_error_patterns(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """分析误差模式，识别重建过程中的问题类型"""
        error = original - reconstructed
        
        # 波形误差分析
        waveform_error = {
            "rms": np.sqrt(np.mean(error ** 2)),
            "max": np.max(np.abs(error)),
            "mean": np.mean(error),
            "std": np.std(error)
        }
        
        # 误差模式识别
        pattern = {
            "has_dc_offset": abs(waveform_error["mean"]) > 0.01,  # DC偏移
            "is_periodic": self._detect_periodic_error(error),    # 周期性误差
            "has_spikes": self._detect_spike_error(error),        # 突发误差
            "spike_ratio": np.sum(np.abs(error) > 3 * waveform_error["std"]) / len(error)
        }
        
        waveform_error["pattern"] = pattern
        
        # 频谱误差分析
        orig_fft = np.abs(np.fft.rfft(original))
        recon_fft = np.abs(np.fft.rfft(reconstructed))
        freq_error = np.abs(orig_fft - recon_fft)
        
        freqs = np.fft.rfftfreq(len(original), 1/self.sampling_rate)
        nyquist = self.sampling_rate // 2
        
        # 不同频段的误差
        low_freq_mask = freqs <= nyquist * 0.25  # 0-2kHz
        mid_freq_mask = (freqs > nyquist * 0.25) & (freqs <= nyquist * 0.5)  # 2-4kHz  
        high_freq_mask = freqs > nyquist * 0.5  # 4-8kHz
        
        spectrum_error = {
            "low_freq_error": np.mean(freq_error[low_freq_mask]),
            "mid_freq_error": np.mean(freq_error[mid_freq_mask]),
            "high_freq_error": np.mean(freq_error[high_freq_mask]),
            "total_freq_error": np.mean(freq_error),
            "error_concentration": np.mean(freq_error[high_freq_mask]) / (np.mean(freq_error[low_freq_mask]) + 1e-8)
        }
        
        return {
            "waveform_error": waveform_error,
            "spectrum_error": spectrum_error
        }
    
    def _detect_periodic_error(self, error: np.ndarray) -> bool:
        """检测周期性误差"""
        # 计算自相关
        autocorr = np.correlate(error, error, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # 归一化
        
        # 寻找周期性峰值
        peaks = []
        for i in range(1, min(len(autocorr) // 4, 1000)):
            if autocorr[i] > 0.3 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(autocorr[i])
        
        return len(peaks) > 0 and max(peaks) > 0.5 if peaks else False
    
    def _detect_spike_error(self, error: np.ndarray) -> bool:
        """检测突发误差"""
        threshold = 3 * np.std(error)
        spike_count = np.sum(np.abs(error) > threshold)
        return spike_count > len(error) * 0.01  # 超过1%的样本是突发误差
    
    def _interpret_error_analysis(self, error_analysis: Dict) -> str:
        """解释误差分析结果，提供诊断信息"""
        wf_error = error_analysis["waveform_error"]
        sp_error = error_analysis["spectrum_error"]
        pattern = wf_error["pattern"]
        
        interpretation = []
        interpretation.append("🔍 误差分析解释:")
        
        # 波形误差解释
        if wf_error["rms"] < 0.05:
            interpretation.append("   ✅ 波形重建质量优秀 (RMS < 0.05)")
        elif wf_error["rms"] < 0.1:
            interpretation.append("   ✅ 波形重建质量良好 (RMS < 0.1)")
        elif wf_error["rms"] < 0.2:
            interpretation.append("   ⚠️ 波形重建质量中等 (RMS < 0.2)")
        else:
            interpretation.append("   ❌ 波形重建质量较差 (RMS ≥ 0.2)")
        
        # 频谱误差解释
        if sp_error["error_concentration"] > 5:
            interpretation.append("   ❌ 严重的高频损失 - VAE在高频重建上表现不佳")
        elif sp_error["error_concentration"] > 3:
            interpretation.append("   ⚠️ 中等程度的高频损失 - 可以通过后处理改善")
        elif sp_error["error_concentration"] > 1.5:
            interpretation.append("   ⚠️ 轻微的高频损失 - 这是VAE的典型特征")
        else:
            interpretation.append("   ✅ 频率响应均衡 - 各频段损失较为平均")
        
        # 模式分析解释
        if pattern["has_dc_offset"]:
            interpretation.append("   ⚠️ DC偏移: 重建信号有直流分量偏差，可能需要高通滤波")
            
        if pattern["is_periodic"]:
            interpretation.append("   🔄 周期性误差: 某些周期性成分丢失，可能是训练数据不足")
            
        if pattern["has_spikes"]:
            spike_ratio = pattern["spike_ratio"]
            if spike_ratio > 0.05:
                interpretation.append(f"   ⚡ 严重突发误差({spike_ratio:.1%}): 瞬态信号处理能力差")
            else:
                interpretation.append(f"   ⚡ 轻微突发误差({spike_ratio:.1%}): 少量瞬态失真")
        
        if not any([pattern["has_dc_offset"], pattern["is_periodic"], pattern["has_spikes"]]):
            interpretation.append("   ✅ 误差模式健康: 主要为白噪声，无明显系统性问题")
        
        # 总体建议
        interpretation.append("\n💡 改进建议:")
        if sp_error["error_concentration"] > 3:
            interpretation.append("   • 考虑使用高频增强后处理")
            interpretation.append("   • 可尝试diffusion模型进行latent增强")
        
        if wf_error["rms"] > 0.1:
            interpretation.append("   • 检查VAE训练质量和参数设置")
            interpretation.append("   • 考虑使用更先进的vocoder")
        
        if pattern["has_dc_offset"]:
            interpretation.append("   • 添加高通滤波去除DC偏移")
        
        return "\n".join(interpretation)


def main():
    """主函数：运行训练匹配的VAE重建测试"""
    
    # 初始化重建器
    reconstructor = TrainingMatchedVAEReconstructor()
    
    # 测试音频文件路径
    test_audio_path = "piano.wav"  # 使用现有的测试音频
    
    if not Path(test_audio_path).exists():
        print(f"❌ 测试音频文件不存在: {test_audio_path}")
        print("请将测试音频文件放在当前目录下，或修改test_audio_path变量")
        return
    
    try:
        # 运行重建测试
        result = reconstructor.reconstruct_audio(test_audio_path)
        
        print(f"\n✅ 训练匹配VAE重建测试完成（修正版）!")
        print(f"📁 查看结果: {result['output_dir']}")
        print(f"🎵 原始音频: {result['original_path']}")
        print(f"🎵 重建音频: {result['reconstructed_path']}")
        print(f"📊 分析图表: {result['plot_path']}")
        
        # 关键改进说明
        print(f"\n🔧 关键修正:")
        print(f"   ✅ 使用训练代码的mel频谱图提取方法")
        print(f"   ✅ 使用训练代码的padding策略")
        print(f"   ✅ 严格匹配训练配置参数")
        print(f"   ✅ 修复维度不匹配问题")
        
    except Exception as e:
        print(f"❌ 重建过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
