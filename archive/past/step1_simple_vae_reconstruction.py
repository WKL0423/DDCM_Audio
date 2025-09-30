#!/usr/bin/env python3
"""
Step 1: 单纯的VAE重建测试
重新开始构建DDCM管道，第一步先验证基础VAE重建效果
专注于：
1. 音频加载和预处理
2. VAE编码
3. VAE解码
4. 音频保存和质量分析
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

class SimpleVAEReconstructor:
    """
    简单的VAE重建器
    只做最基础的音频→VAE编码→VAE解码→音频的流程
    """
    
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        """初始化VAE重建器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎵 初始化简单VAE重建器")
        print(f"   📱 设备: {self.device}")
        
        # 加载AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        print(f"✅ VAE重建器初始化完成")
        print(f"   📊 VAE scaling_factor: {self.pipeline.vae.config.scaling_factor}")
        print(f"   🎤 特征提取器采样率: {self.pipeline.feature_extractor.sampling_rate}Hz")
    
    def reconstruct_audio(self, audio_path: str) -> Dict:
        """
        完整的VAE重建流程
        
        Args:
            audio_path: 输入音频路径
            
        Returns:
            result: 包含重建结果和分析的字典
        """
        print(f"\n🎵 VAE重建处理: {Path(audio_path).name}")
        
        # 1. 加载和预处理输入音频
        original_audio, input_latent = self._load_and_encode_audio(audio_path)
        
        # 2. VAE解码重建
        reconstructed_audio = self._decode_latent_to_audio(input_latent)
        
        # 3. 保存结果
        result = self._save_and_analyze(original_audio, reconstructed_audio, audio_path)
        
        # 4. 显示结果
        self._display_results(result)
        
        return result
    
    def _load_and_encode_audio(self, audio_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """加载音频并编码为latent"""
        print(f"   📁 加载音频...")
        
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        
        # 重采样到AudioLDM2的采样率 (48kHz)
        target_sr = self.pipeline.feature_extractor.sampling_rate
        if sr != target_sr:
            print(f"   🔄 重采样: {sr}Hz -> {target_sr}Hz")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
        
        # 转为单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 限制长度到10秒
        max_length = target_sr * 10
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
            print(f"   ✂️ 截取到10秒")
        
        print(f"   ✅ 音频预处理完成: {audio.shape}, {audio.shape[-1]/target_sr:.2f}秒")
        
        # 编码为latent
        print(f"   🔄 VAE编码...")
        with torch.no_grad():
            # 确保音频是2D张量 [channels, samples]
            audio_np = audio.squeeze(0).numpy()
            
            # 使用AudioLDM2的特征提取器
            inputs = self.pipeline.feature_extractor(
                audio_np,
                sampling_rate=target_sr,
                return_tensors="pt"
            )
            
            mel_features = inputs["input_features"].to(self.device)
            
            # 调整维度：确保是 [batch, channels, height, width]
            if mel_features.dim() == 3:
                mel_features = mel_features.unsqueeze(1)
            
            # 转换数据类型
            if self.device == "cuda":
                mel_features = mel_features.half()
            
            print(f"   📊 Mel特征: {mel_features.shape}, 范围[{mel_features.min():.3f}, {mel_features.max():.3f}]")
            
            # VAE编码
            latent_dist = self.pipeline.vae.encode(mel_features)
            latent = latent_dist.latent_dist.mode()  # 使用mode而不是sample，更稳定
            
            # 应用scaling factor
            latent = latent * self.pipeline.vae.config.scaling_factor
            
            print(f"   ✅ VAE编码完成: {latent.shape}")
            print(f"   📊 Latent范围: [{latent.min():.3f}, {latent.max():.3f}]")
            print(f"   📊 Latent std: {latent.std():.3f}")
        
        # 转换原始音频到16kHz用于分析
        audio_16k = torchaudio.functional.resample(audio, target_sr, 16000)
        original_audio = audio_16k.squeeze().numpy()
        
        return original_audio, latent
    
    def _decode_latent_to_audio(self, latent: torch.Tensor) -> np.ndarray:
        """将latent解码为音频"""
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
            mel_spectrogram = self.pipeline.vae.decode(latent_for_decode).sample
            
            print(f"   📊 解码mel: {mel_spectrogram.shape}")
            print(f"   📊 解码mel范围: [{mel_spectrogram.min():.3f}, {mel_spectrogram.max():.3f}]")
            
            # 使用HiFiGAN vocoder
            audio_tensor = self.pipeline.mel_spectrogram_to_waveform(mel_spectrogram)
            audio = audio_tensor.squeeze().cpu().numpy()
            
            print(f"   ✅ VAE解码完成: {len(audio)}样本 ({len(audio)/16000:.2f}秒)")
            
        return audio
    
    def _save_and_analyze(self, original_audio: np.ndarray, reconstructed_audio: np.ndarray, 
                         audio_path: str) -> Dict:
        """保存结果并进行质量分析"""
        print(f"   💾 保存结果和质量分析...")
        
        # 创建输出目录
        output_dir = Path("step_1_simple_vae_reconstruction")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        input_name = Path(audio_path).stem
        
        # 确保长度一致
        min_len = min(len(original_audio), len(reconstructed_audio))
        original_audio = original_audio[:min_len]
        reconstructed_audio = reconstructed_audio[:min_len]
        
        # 保存音频文件
        original_path = output_dir / f"{input_name}_original_{timestamp}.wav"
        reconstructed_path = output_dir / f"{input_name}_vae_reconstructed_{timestamp}.wav"
        
        sf.write(str(original_path), original_audio, 16000)
        sf.write(str(reconstructed_path), reconstructed_audio, 16000)
        
        # 质量分析
        analysis = self._analyze_audio_quality(original_audio, reconstructed_audio)
        
        # 频谱分析
        spectral_analysis = self._analyze_frequency_content(original_audio, reconstructed_audio)
        
        # 生成频谱图
        self._plot_spectrograms(original_audio, reconstructed_audio, output_dir, timestamp)
        
        result = {
            "input_file": audio_path,
            "original_path": str(original_path),
            "reconstructed_path": str(reconstructed_path),
            "audio_length": min_len / 16000,
            "quality_metrics": analysis,
            "frequency_analysis": spectral_analysis,
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
        freqs = np.fft.fftfreq(fft_len, 1/16000)[:fft_len//2]
        
        # 定义频段
        low_freq_mask = freqs < 1000    # 低频 0-1kHz
        mid_freq_mask = (freqs >= 1000) & (freqs < 4000)  # 中频 1-4kHz
        high_freq_mask = freqs >= 4000   # 高频 4-8kHz
        
        # 计算各频段能量
        low_orig = np.sum(orig_fft[low_freq_mask])
        mid_orig = np.sum(orig_fft[mid_freq_mask])
        high_orig = np.sum(orig_fft[high_freq_mask])
        
        low_recon = np.sum(recon_fft[low_freq_mask])
        mid_recon = np.sum(recon_fft[mid_freq_mask])
        high_recon = np.sum(recon_fft[high_freq_mask])
        
        # 计算保持率
        low_retention = low_recon / (low_orig + 1e-10)
        mid_retention = mid_recon / (mid_orig + 1e-10)
        high_retention = high_recon / (high_orig + 1e-10)
        
        return {
            "low_freq_retention": low_retention,
            "mid_freq_retention": mid_retention,
            "high_freq_retention": high_retention,
            "total_energy_ratio": np.sum(recon_fft) / (np.sum(orig_fft) + 1e-10)
        }
    
    def _plot_spectrograms(self, original: np.ndarray, reconstructed: np.ndarray, 
                          output_dir: Path, timestamp: int):
        """绘制频谱图对比"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 原始音频频谱图
            D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
            librosa.display.specshow(D_orig, sr=16000, x_axis='time', y_axis='hz', ax=axes[0,0])
            axes[0,0].set_title('原始音频频谱图')
            axes[0,0].set_ylim([0, 8000])
            
            # 重建音频频谱图
            D_recon = librosa.amplitude_to_db(np.abs(librosa.stft(reconstructed)), ref=np.max)
            librosa.display.specshow(D_recon, sr=16000, x_axis='time', y_axis='hz', ax=axes[0,1])
            axes[0,1].set_title('VAE重建音频频谱图')
            axes[0,1].set_ylim([0, 8000])
            
            # 频谱差异
            diff = D_orig - D_recon
            im = librosa.display.specshow(diff, sr=16000, x_axis='time', y_axis='hz', ax=axes[1,0], cmap='RdBu_r')
            axes[1,0].set_title('频谱差异 (原始 - 重建)')
            axes[1,0].set_ylim([0, 8000])
            plt.colorbar(im, ax=axes[1,0])
            
            # 频率能量对比
            fft_len = min(8192, len(original))
            orig_fft = np.abs(np.fft.fft(original[:fft_len]))[:fft_len//2]
            recon_fft = np.abs(np.fft.fft(reconstructed[:fft_len]))[:fft_len//2]
            freqs = np.fft.fftfreq(fft_len, 1/16000)[:fft_len//2]
            
            axes[1,1].semilogy(freqs, orig_fft, label='原始', alpha=0.7)
            axes[1,1].semilogy(freqs, recon_fft, label='重建', alpha=0.7)
            axes[1,1].set_xlabel('频率 (Hz)')
            axes[1,1].set_ylabel('幅度')
            axes[1,1].set_title('频率响应对比')
            axes[1,1].legend()
            axes[1,1].set_xlim([0, 8000])
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = output_dir / f"spectral_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 频谱图已保存: {plot_path}")
            
        except Exception as e:
            print(f"   ⚠️ 频谱图生成失败: {e}")
    
    def _display_results(self, result: Dict):
        """显示结果"""
        print(f"\n{'='*70}")
        print(f"🎯 简单VAE重建结果")
        print(f"{'='*70}")
        print(f"📁 输入文件: {result['input_file']}")
        print(f"📁 原始音频: {result['original_path']}")
        print(f"📁 重建音频: {result['reconstructed_path']}")
        print(f"⏱️ 音频长度: {result['audio_length']:.2f}秒")
        
        metrics = result['quality_metrics']
        print(f"\n📊 质量指标:")
        print(f"   SNR: {metrics['snr_db']:.2f} dB")
        print(f"   相关系数: {metrics['correlation']:.4f}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
        print(f"   RMS比率: {metrics['rms_ratio']:.4f}")
        
        freq_analysis = result['frequency_analysis']
        if 'error' not in freq_analysis:
            print(f"\n🎵 频率分析:")
            print(f"   低频保持率 (0-1kHz): {freq_analysis['low_freq_retention']:.3f}")
            print(f"   中频保持率 (1-4kHz): {freq_analysis['mid_freq_retention']:.3f}")
            print(f"   高频保持率 (4-8kHz): {freq_analysis['high_freq_retention']:.3f}")
            print(f"   总能量比率: {freq_analysis['total_energy_ratio']:.3f}")
            
            # 分析高频丢失情况
            high_freq_retention = freq_analysis['high_freq_retention']
            if high_freq_retention < 0.3:
                print(f"   ❌ 严重高频损失！保持率仅 {high_freq_retention*100:.1f}%")
            elif high_freq_retention < 0.6:
                print(f"   ⚠️ 明显高频损失，保持率 {high_freq_retention*100:.1f}%")
            elif high_freq_retention < 0.8:
                print(f"   ⚡ 轻微高频损失，保持率 {high_freq_retention*100:.1f}%")
            else:
                print(f"   ✅ 高频保持良好，保持率 {high_freq_retention*100:.1f}%")
        
        print(f"\n💡 下一步建议:")
        if metrics['snr_db'] > 10:
            print(f"   ✅ VAE重建质量良好，可以在此基础上添加diffusion过程")
        else:
            print(f"   ⚠️ VAE重建质量需要改进，建议先优化VAE参数")
            
        if freq_analysis.get('high_freq_retention', 0) < 0.5:
            print(f"   🎯 高频损失明显，diffusion过程应重点补充高频信息")
        
        print(f"   📊 可视化分析图已生成")

def test_simple_vae():
    """测试简单VAE重建"""
    print("🎵 简单VAE重建测试")
    print("=" * 50)
    
    # 检查输入文件
    input_file = "AudioLDM2_Music_output.wav"
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        print("请确保AudioLDM2_Music_output.wav在当前目录中")
        return
    
    # 创建VAE重建器
    vae_reconstructor = SimpleVAEReconstructor()
    
    # 执行重建
    result = vae_reconstructor.reconstruct_audio(input_file)
    
    print(f"\n✅ 简单VAE重建测试完成！")
    print(f"📁 查看输出目录: simple_vae_reconstruction/")
    print(f"🎵 对比原始和重建音频，为下一步添加diffusion做准备")

if __name__ == "__main__":
    test_simple_vae()
