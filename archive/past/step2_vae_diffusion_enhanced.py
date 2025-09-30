#!/usr/bin/env python3
"""
Step 2: VAE + Diffusion增强重建
在VAE编码器和解码器之间加入diffusion过程，专门用于增强高频内容
目标：解决VAE高频损失问题，提升音频重建质量

流程：
1. VAE编码音频到latent空间
2. 使用diffusion过程增强latent表示（特别是高频对应的部分）
3. VAE解码增强后的latent到音频
4. 对比分析VAE-only vs VAE+Diffusion的效果
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
from diffusers import AudioLDM2Pipeline, DDPMScheduler
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class VAEDiffusionEnhancer:
    """
    VAE + Diffusion增强器
    在VAE的latent空间中使用diffusion来增强特征，特别是高频信息
    """
    
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        """初始化VAE+Diffusion增强器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 初始化VAE+Diffusion增强器")
        print(f"   📱 设备: {self.device}")
        print(f"   🤖 模型: {model_name}")
        
        # 加载AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        # 设置diffusion调度器
        self.scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        
        print(f"✅ VAE+Diffusion增强器初始化完成")
        print(f"   📊 VAE scaling_factor: {self.pipeline.vae.config.scaling_factor}")
        print(f"   🔄 Diffusion调度器: {type(self.scheduler).__name__}")
        print(f"   📈 Diffusion步数: {self.scheduler.num_train_timesteps}")
        
        # 创建输出目录
        self.output_dir = Path("vae_diffusion_enhanced")
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
    
    def enhance_reconstruction(self, audio_path: str, 
                            diffusion_steps: int = 20,
                            guidance_scale: float = 3.0,
                            high_freq_boost: float = 1.5) -> Dict:
        """
        执行VAE+Diffusion增强重建
        
        Args:
            audio_path: 输入音频文件路径
            diffusion_steps: diffusion去噪步数
            guidance_scale: 引导尺度
            high_freq_boost: 高频增强系数
            
        Returns:
            包含重建结果和分析的字典
        """
        print(f"\n🔄 开始VAE+Diffusion增强重建: {Path(audio_path).name}")
        print(f"   🔢 Diffusion步数: {diffusion_steps}")
        print(f"   🎯 引导尺度: {guidance_scale}")
        print(f"   🎼 高频增强: {high_freq_boost}x")
        
        # 1. 加载和预处理音频
        original_audio, processed_audio = self._load_and_preprocess_audio(audio_path)
        
        # 2. VAE编码到latent空间
        latent = self._encode_audio(processed_audio)
        
        # 3. 创建对比：普通VAE重建
        vae_only_audio = self._decode_audio(latent.clone())
        
        # 4. Diffusion增强latent
        enhanced_latent = self._enhance_latent_with_diffusion(
            latent, 
            diffusion_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            high_freq_boost=high_freq_boost
        )
        
        # 5. VAE解码增强后的latent
        enhanced_audio = self._decode_audio(enhanced_latent)
        
        # 6. 保存结果
        timestamp = int(time.time())
        paths = self._save_audio_results(
            original_audio, vae_only_audio, enhanced_audio, timestamp
        )
        
        # 7. 质量对比分析
        quality_comparison = self._compare_quality(
            original_audio, vae_only_audio, enhanced_audio
        )
        
        # 8. 频率对比分析
        frequency_comparison = self._compare_frequency_content(
            original_audio, vae_only_audio, enhanced_audio
        )
        
        # 9. 可视化分析
        self._create_comparison_visualizations(
            original_audio, vae_only_audio, enhanced_audio, timestamp
        )
        
        # 10. 整合结果
        result = {
            "input_file": audio_path,
            "timestamp": timestamp,
            "parameters": {
                "diffusion_steps": diffusion_steps,
                "guidance_scale": guidance_scale,
                "high_freq_boost": high_freq_boost
            },
            "paths": paths,
            "quality_comparison": quality_comparison,
            "frequency_comparison": frequency_comparison,
            "latent_shapes": {
                "original": latent.shape,
                "enhanced": enhanced_latent.shape
            },
            "processing_device": str(self.device)
        }
        
        # 11. 显示结果
        self._display_comparison_results(result)
        
        return result
    
    def _load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """加载和预处理音频"""
        print(f"   📂 加载音频文件...")
        
        # 加载原始音频
        original_audio, sr = torchaudio.load(audio_path)
        
        # 转换为单声道
        if original_audio.shape[0] > 1:
            original_audio = original_audio.mean(dim=0, keepdim=True)
        
        # 重采样到48kHz
        if sr != 48000:
            print(f"   🔄 重采样: {sr}Hz -> 48000Hz")
            resampler = torchaudio.transforms.Resample(sr, 48000)
            processed_audio = resampler(original_audio)
        else:
            processed_audio = original_audio.clone()
        
        # 限制长度到10秒
        max_length = 48000 * 10
        if processed_audio.shape[-1] > max_length:
            print(f"   ✂️ 截取前10秒")
            processed_audio = processed_audio[..., :max_length]
        
        # 转换为numpy用于最终输出
        original_audio_np = original_audio.squeeze().numpy()
        
        print(f"   ✅ 音频预处理完成: {processed_audio.shape}")
        
        return original_audio_np, processed_audio
    
    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """使用VAE编码音频到latent空间"""
        print(f"   🔗 VAE编码...")
        
        with torch.no_grad():
            # 转换为numpy并使用feature extractor
            audio_np = audio.squeeze().numpy()
            
            # 使用ClapFeatureExtractor处理音频
            inputs = self.pipeline.feature_extractor(
                audio_np,
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            # 获取mel频谱特征
            mel_features = inputs["input_features"].to(self.device)
            
            # 确保是4D张量
            if mel_features.dim() == 3:
                mel_features = mel_features.unsqueeze(1)
            
            # 匹配VAE的数据类型
            if self.device == "cuda":
                mel_features = mel_features.half()
            
            # VAE编码
            latent_dist = self.pipeline.vae.encode(mel_features)
            latent = latent_dist.latent_dist.mode()
            
            # 应用scaling factor
            latent = latent * self.pipeline.vae.config.scaling_factor
            
            print(f"   ✅ 编码完成: {mel_features.shape} -> {latent.shape}")
            
            return latent
    
    def _enhance_latent_with_diffusion(self, 
                                     latent: torch.Tensor,
                                     diffusion_steps: int = 20,
                                     guidance_scale: float = 3.0,
                                     high_freq_boost: float = 1.5) -> torch.Tensor:
        """
        使用diffusion过程增强latent表示
        特别针对高频信息进行增强
        """
        print(f"   🌊 Diffusion增强latent...")
        
        with torch.no_grad():
            batch_size = latent.shape[0]
            
            # 设置调度器的时间步
            self.scheduler.set_timesteps(diffusion_steps)
            
            # 添加噪声到latent（模拟diffusion的forward过程）
            # 使用较少的噪声，因为我们不想完全破坏VAE学到的表示
            noise_level = 0.3  # 相对较小的噪声水平
            noise = torch.randn_like(latent) * noise_level
            
            # 选择一个中等的时间步（不是完全随机噪声）
            timestep_idx = len(self.scheduler.timesteps) // 3  # 使用1/3处的时间步
            timestep = self.scheduler.timesteps[timestep_idx:timestep_idx+1]
            
            # 添加噪声
            noisy_latent = self.scheduler.add_noise(latent, noise, timestep)
            
            print(f"   🔸 噪声水平: {noise_level}")
            print(f"   ⏰ 起始时间步: {timestep.item()}")
            
            # 准备输入到UNet
            latent_model_input = noisy_latent
            
            # 创建无条件输入（用于classifier-free guidance）
            uncond_embeddings = torch.zeros(
                (batch_size, 77, 1024),  # AudioLDM2的文本嵌入维度
                device=self.device,
                dtype=latent.dtype
            )
            
            # 创建条件输入（基于原始latent的特征）
            # 这里我们使用原始latent的统计信息作为"条件"
            cond_embeddings = self._create_latent_conditioning(latent)
            
            # Diffusion去噪过程
            enhanced_latent = latent_model_input.clone()
            
            for i, t in enumerate(self.scheduler.timesteps[timestep_idx:]):
                # 准备输入
                latent_input = enhanced_latent
                
                # 如果使用guidance
                if guidance_scale > 1.0:
                    # 条件和无条件预测
                    latent_input_expanded = torch.cat([latent_input] * 2)
                    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
                else:
                    latent_input_expanded = latent_input
                    text_embeddings = cond_embeddings
                
                # UNet预测噪声
                with torch.no_grad():
                    noise_pred = self.pipeline.unet(
                        latent_input_expanded,
                        t,
                        encoder_hidden_states=text_embeddings,
                        return_dict=False
                    )[0]
                
                # 应用classifier-free guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # 调度器去噪步骤
                enhanced_latent = self.scheduler.step(
                    noise_pred, t, enhanced_latent, return_dict=False
                )[0]
                
                # 高频增强（在latent空间中）
                enhanced_latent = self._apply_high_freq_boost(
                    enhanced_latent, latent, high_freq_boost
                )
            
            print(f"   ✅ Diffusion增强完成: {diffusion_steps}步")
            
            return enhanced_latent
    
    def _create_latent_conditioning(self, latent: torch.Tensor) -> torch.Tensor:
        """
        基于latent创建条件嵌入
        使用latent的统计特征作为条件信息
        """
        batch_size = latent.shape[0]
        
        # 计算latent的统计特征
        latent_mean = latent.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        latent_std = latent.std(dim=[2, 3], keepdim=True)    # [B, C, 1, 1]
        latent_max = latent.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]  # [B, C, 1, 1]
        
        # 组合统计特征
        stats_features = torch.cat([latent_mean, latent_std, latent_max], dim=1)  # [B, 3*C, 1, 1]
        
        # 展平并投影到文本嵌入维度
        stats_flat = stats_features.flatten(1)  # [B, 3*C]
        
        # 创建简单的线性投影到文本嵌入空间
        embed_dim = 1024
        seq_len = 77
        
        # 重复和投影到目标维度
        conditioning = stats_flat.unsqueeze(1).repeat(1, seq_len, 1)  # [B, 77, 3*C]
        
        # 如果维度不匹配，进行填充或截断
        if conditioning.shape[-1] < embed_dim:
            padding = torch.zeros(
                batch_size, seq_len, embed_dim - conditioning.shape[-1],
                device=conditioning.device, dtype=conditioning.dtype
            )
            conditioning = torch.cat([conditioning, padding], dim=-1)
        elif conditioning.shape[-1] > embed_dim:
            conditioning = conditioning[:, :, :embed_dim]
        
        return conditioning
    
    def _apply_high_freq_boost(self, 
                              enhanced_latent: torch.Tensor,
                              original_latent: torch.Tensor,
                              boost_factor: float) -> torch.Tensor:
        """
        在latent空间中应用高频增强
        通过分析latent的空间频率特征来增强高频部分
        """
        if boost_factor <= 1.0:
            return enhanced_latent
        
        # 在latent的空间维度上应用高通滤波来识别高频部分
        # 使用简单的边缘检测核来识别高频特征
        high_freq_kernel = torch.tensor([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ], dtype=enhanced_latent.dtype, device=enhanced_latent.device)
        
        high_freq_kernel = high_freq_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        
        # 对每个通道分别应用高频检测
        enhanced_hf = torch.zeros_like(enhanced_latent)
        
        for c in range(enhanced_latent.shape[1]):
            latent_channel = enhanced_latent[:, c:c+1, :, :]  # [B, 1, H, W]
            
            # 应用高频检测（边缘检测）
            hf_response = F.conv2d(
                latent_channel,
                high_freq_kernel,
                padding=1
            )
            
            enhanced_hf[:, c:c+1, :, :] = hf_response
        
        # 增强高频部分
        high_freq_enhancement = enhanced_hf * (boost_factor - 1.0)
        enhanced_result = enhanced_latent + high_freq_enhancement
        
        # 保持与原始latent的相似性（避免过度增强）
        blend_factor = 0.8  # 80%增强 + 20%原始
        final_result = blend_factor * enhanced_result + (1 - blend_factor) * original_latent
        
        return final_result
    
    def _decode_audio(self, latent: torch.Tensor) -> np.ndarray:
        """使用VAE解码latent到音频"""
        print(f"   🔄 VAE解码...")
        
        with torch.no_grad():
            # 反向scaling
            latent_for_decode = latent / self.pipeline.vae.config.scaling_factor
            
            # 确保数据类型匹配
            vae_dtype = next(self.pipeline.vae.parameters()).dtype
            latent_for_decode = latent_for_decode.to(vae_dtype)
            
            # VAE解码到mel频谱
            mel_spectrogram = self.pipeline.vae.decode(latent_for_decode).sample
            
            # 使用vocoder转换为音频波形
            audio_tensor = self.pipeline.mel_spectrogram_to_waveform(mel_spectrogram)
            audio_np = audio_tensor.squeeze().cpu().numpy()
            
            print(f"   ✅ 解码完成: {latent.shape} -> 音频长度 {len(audio_np)}")
            
            return audio_np
    
    def _save_audio_results(self, 
                          original: np.ndarray,
                          vae_only: np.ndarray,
                          enhanced: np.ndarray,
                          timestamp: int) -> Dict[str, str]:
        """保存音频结果"""
        print(f"   💾 保存音频文件...")
        
        paths = {}
        
        # 保存原始音频（重采样到16kHz）
        original_16k = librosa.resample(original, orig_sr=48000, target_sr=16000)
        original_path = self.output_dir / f"original_{timestamp}.wav"
        sf.write(str(original_path), original_16k, 16000)
        paths["original"] = str(original_path)
        
        # 保存VAE-only重建音频
        vae_only_path = self.output_dir / f"vae_only_{timestamp}.wav"
        sf.write(str(vae_only_path), vae_only, 16000)
        paths["vae_only"] = str(vae_only_path)
        
        # 保存VAE+Diffusion增强音频
        enhanced_path = self.output_dir / f"vae_diffusion_enhanced_{timestamp}.wav"
        sf.write(str(enhanced_path), enhanced, 16000)
        paths["enhanced"] = str(enhanced_path)
        
        print(f"   ✅ 音频文件已保存")
        
        return paths
    
    def _compare_quality(self, 
                        original: np.ndarray,
                        vae_only: np.ndarray,
                        enhanced: np.ndarray) -> Dict:
        """对比重建质量"""
        print(f"   📊 质量对比分析...")
        
        # 重采样原始音频到16kHz
        original_16k = librosa.resample(original, orig_sr=48000, target_sr=16000)
        
        def calculate_metrics(orig, recon):
            # 确保长度一致
            min_len = min(len(orig), len(recon))
            o = orig[:min_len]
            r = recon[:min_len]
            
            # 基本指标
            mse = np.mean((o - r) ** 2)
            mae = np.mean(np.abs(o - r))
            
            # SNR
            signal_power = np.mean(o ** 2)
            noise_power = mse
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # 相关性
            correlation = np.corrcoef(o, r)[0, 1] if min_len > 1 else 0.0
            if np.isnan(correlation):
                correlation = 0.0
            
            return {
                "snr_db": snr,
                "correlation": correlation,
                "mse": mse,
                "mae": mae,
                "signal_power": signal_power
            }
        
        # 计算VAE-only指标
        vae_metrics = calculate_metrics(original_16k, vae_only)
        
        # 计算增强版指标
        enhanced_metrics = calculate_metrics(original_16k, enhanced)
        
        # 计算改进量
        improvements = {
            "snr_improvement": enhanced_metrics["snr_db"] - vae_metrics["snr_db"],
            "correlation_improvement": enhanced_metrics["correlation"] - vae_metrics["correlation"],
            "mse_improvement": vae_metrics["mse"] - enhanced_metrics["mse"],  # 减少是改进
            "mae_improvement": vae_metrics["mae"] - enhanced_metrics["mae"]   # 减少是改进
        }
        
        comparison = {
            "vae_only": vae_metrics,
            "enhanced": enhanced_metrics,
            "improvements": improvements
        }
        
        print(f"   ✅ 质量对比分析完成")
        
        return comparison
    
    def _compare_frequency_content(self,
                                 original: np.ndarray,
                                 vae_only: np.ndarray,
                                 enhanced: np.ndarray) -> Dict:
        """对比频率内容"""
        print(f"   🎼 频率对比分析...")
        
        # 重采样原始音频到16kHz
        original_16k = librosa.resample(original, orig_sr=48000, target_sr=16000)
        
        def analyze_frequency_bands(orig, recon):
            # 确保长度一致
            min_len = min(len(orig), len(recon))
            o = orig[:min_len]
            r = recon[:min_len]
            
            # FFT分析
            if min_len >= 8192:
                n_fft = 8192
            else:
                n_fft = 2 ** int(np.log2(min_len))
            
            orig_fft = np.abs(np.fft.fft(o[:n_fft]))[:n_fft//2]
            recon_fft = np.abs(np.fft.fft(r[:n_fft]))[:n_fft//2]
            
            freqs = np.fft.fftfreq(n_fft, 1/16000)[:n_fft//2]
            
            # 频段定义
            low_mask = freqs < 500
            mid_mask = (freqs >= 500) & (freqs < 4000)
            high_mask = freqs >= 4000
            
            # 计算各频段能量保持率
            low_retention = np.sum(recon_fft[low_mask]) / (np.sum(orig_fft[low_mask]) + 1e-10)
            mid_retention = np.sum(recon_fft[mid_mask]) / (np.sum(orig_fft[mid_mask]) + 1e-10)
            high_retention = np.sum(recon_fft[high_mask]) / (np.sum(orig_fft[high_mask]) + 1e-10)
            
            # 总体频谱相关性
            freq_corr = np.corrcoef(orig_fft, recon_fft)[0, 1]
            if np.isnan(freq_corr):
                freq_corr = 0.0
            
            return {
                "low_freq_retention": low_retention,
                "mid_freq_retention": mid_retention,
                "high_freq_retention": high_retention,
                "frequency_correlation": freq_corr
            }
        
        # 分析VAE-only频率特性
        vae_freq = analyze_frequency_bands(original_16k, vae_only)
        
        # 分析增强版频率特性
        enhanced_freq = analyze_frequency_bands(original_16k, enhanced)
        
        # 计算频率改进
        freq_improvements = {
            "low_freq_improvement": enhanced_freq["low_freq_retention"] - vae_freq["low_freq_retention"],
            "mid_freq_improvement": enhanced_freq["mid_freq_retention"] - vae_freq["mid_freq_retention"],
            "high_freq_improvement": enhanced_freq["high_freq_retention"] - vae_freq["high_freq_retention"],
            "frequency_correlation_improvement": enhanced_freq["frequency_correlation"] - vae_freq["frequency_correlation"]
        }
        
        comparison = {
            "vae_only": vae_freq,
            "enhanced": enhanced_freq,
            "improvements": freq_improvements
        }
        
        print(f"   ✅ 频率对比分析完成")
        
        return comparison
    
    def _create_comparison_visualizations(self,
                                        original: np.ndarray,
                                        vae_only: np.ndarray,
                                        enhanced: np.ndarray,
                                        timestamp: int):
        """创建对比可视化"""
        print(f"   📈 生成对比可视化...")
        
        # 重采样原始音频到16kHz
        original_16k = librosa.resample(original, orig_sr=48000, target_sr=16000)
        
        # 确保长度一致
        min_len = min(len(original_16k), len(vae_only), len(enhanced))
        orig = original_16k[:min_len]
        vae = vae_only[:min_len]
        enh = enhanced[:min_len]
        
        # 创建对比图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('VAE vs VAE+Diffusion 音频重建对比', fontsize=16, fontweight='bold')
        
        # 1. 时域波形对比
        time_samples = min(16000, min_len)  # 显示前1秒
        axes[0, 0].plot(orig[:time_samples], label='原始音频', alpha=0.8)
        axes[0, 0].plot(vae[:time_samples], label='VAE-only', alpha=0.8)
        axes[0, 0].plot(enh[:time_samples], label='VAE+Diffusion', alpha=0.8)
        axes[0, 0].set_title('时域波形对比（前1秒）')
        axes[0, 0].set_xlabel('采样点')
        axes[0, 0].set_ylabel('振幅')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 频谱对比
        if min_len >= 8192:
            n_fft = 8192
            orig_fft = np.abs(np.fft.fft(orig[:n_fft]))[:n_fft//2]
            vae_fft = np.abs(np.fft.fft(vae[:n_fft]))[:n_fft//2]
            enh_fft = np.abs(np.fft.fft(enh[:n_fft]))[:n_fft//2]
            freqs = np.fft.fftfreq(n_fft, 1/16000)[:n_fft//2]
            
            axes[0, 1].loglog(freqs[1:], orig_fft[1:], label='原始音频', alpha=0.8)
            axes[0, 1].loglog(freqs[1:], vae_fft[1:], label='VAE-only', alpha=0.8)
            axes[0, 1].loglog(freqs[1:], enh_fft[1:], label='VAE+Diffusion', alpha=0.8)
            axes[0, 1].set_title('频谱对比')
            axes[0, 1].set_xlabel('频率 (Hz)')
            axes[0, 1].set_ylabel('幅度')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mel频谱图 - 原始
        orig_mel = librosa.feature.melspectrogram(y=orig, sr=16000, n_mels=128)
        orig_mel_db = librosa.power_to_db(orig_mel, ref=np.max)
        im1 = axes[1, 0].imshow(orig_mel_db, aspect='auto', origin='lower', cmap='viridis')
        axes[1, 0].set_title('原始音频Mel频谱')
        axes[1, 0].set_xlabel('时间帧')
        axes[1, 0].set_ylabel('Mel频率')
        plt.colorbar(im1, ax=axes[1, 0])
        
        # 4. Mel频谱图 - VAE only
        vae_mel = librosa.feature.melspectrogram(y=vae, sr=16000, n_mels=128)
        vae_mel_db = librosa.power_to_db(vae_mel, ref=np.max)
        im2 = axes[1, 1].imshow(vae_mel_db, aspect='auto', origin='lower', cmap='viridis')
        axes[1, 1].set_title('VAE-only重建Mel频谱')
        axes[1, 1].set_xlabel('时间帧')
        axes[1, 1].set_ylabel('Mel频率')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # 5. Mel频谱图 - VAE+Diffusion
        enh_mel = librosa.feature.melspectrogram(y=enh, sr=16000, n_mels=128)
        enh_mel_db = librosa.power_to_db(enh_mel, ref=np.max)
        im3 = axes[2, 0].imshow(enh_mel_db, aspect='auto', origin='lower', cmap='viridis')
        axes[2, 0].set_title('VAE+Diffusion增强Mel频谱')
        axes[2, 0].set_xlabel('时间帧')
        axes[2, 0].set_ylabel('Mel频率')
        plt.colorbar(im3, ax=axes[2, 0])
        
        # 6. 频段能量对比
        if min_len >= 8192:
            # 计算各频段能量
            low_mask = freqs < 500
            mid_mask = (freqs >= 500) & (freqs < 4000)
            high_mask = freqs >= 4000
            
            bands = ['Low\n(<500Hz)', 'Mid\n(500Hz-4kHz)', 'High\n(>4kHz)']
            orig_energies = [
                np.sum(orig_fft[low_mask]),
                np.sum(orig_fft[mid_mask]),
                np.sum(orig_fft[high_mask])
            ]
            vae_energies = [
                np.sum(vae_fft[low_mask]),
                np.sum(vae_fft[mid_mask]),
                np.sum(vae_fft[high_mask])
            ]
            enh_energies = [
                np.sum(enh_fft[low_mask]),
                np.sum(enh_fft[mid_mask]),
                np.sum(enh_fft[high_mask])
            ]
            
            x = np.arange(len(bands))
            width = 0.25
            
            axes[2, 1].bar(x - width, orig_energies, width, label='原始', alpha=0.8)
            axes[2, 1].bar(x, vae_energies, width, label='VAE-only', alpha=0.8)
            axes[2, 1].bar(x + width, enh_energies, width, label='VAE+Diffusion', alpha=0.8)
            
            axes[2, 1].set_title('频段能量对比')
            axes[2, 1].set_xlabel('频段')
            axes[2, 1].set_ylabel('能量')
            axes[2, 1].set_xticks(x)
            axes[2, 1].set_xticklabels(bands)
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        plot_path = self.output_dir / f"comparison_analysis_{timestamp}.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 对比可视化已保存: {plot_path}")
    
    def _display_comparison_results(self, result: Dict):
        """显示对比结果"""
        print(f"\n{'='*80}")
        print(f"🎯 VAE vs VAE+Diffusion 重建对比结果")
        print(f"{'='*80}")
        
        print(f"📁 输入文件: {result['input_file']}")
        print(f"📱 处理设备: {result['processing_device']}")
        print(f"📐 Latent形状: {result['latent_shapes']['original']}")
        
        params = result['parameters']
        print(f"\n⚙️ Diffusion参数:")
        print(f"   🔢 去噪步数: {params['diffusion_steps']}")
        print(f"   🎯 引导尺度: {params['guidance_scale']}")
        print(f"   🎼 高频增强: {params['high_freq_boost']}x")
        
        print(f"\n📊 质量对比:")
        quality = result['quality_comparison']
        
        vae_metrics = quality['vae_only']
        enh_metrics = quality['enhanced']
        improvements = quality['improvements']
        
        print(f"   🎵 信噪比 (SNR):")
        print(f"      VAE-only: {vae_metrics['snr_db']:.2f} dB")
        print(f"      VAE+Diffusion: {enh_metrics['snr_db']:.2f} dB")
        print(f"      改进: {improvements['snr_improvement']:+.2f} dB")
        
        print(f"   🔗 相关性:")
        print(f"      VAE-only: {vae_metrics['correlation']:.4f}")
        print(f"      VAE+Diffusion: {enh_metrics['correlation']:.4f}")
        print(f"      改进: {improvements['correlation_improvement']:+.4f}")
        
        print(f"\n🎼 频率对比:")
        freq = result['frequency_comparison']
        
        vae_freq = freq['vae_only']
        enh_freq = freq['enhanced']
        freq_improvements = freq['improvements']
        
        print(f"   🎶 低频保持率 (<500Hz):")
        print(f"      VAE-only: {vae_freq['low_freq_retention']:.3f}")
        print(f"      VAE+Diffusion: {enh_freq['low_freq_retention']:.3f}")
        print(f"      改进: {freq_improvements['low_freq_improvement']:+.3f}")
        
        print(f"   🎵 中频保持率 (500Hz-4kHz):")
        print(f"      VAE-only: {vae_freq['mid_freq_retention']:.3f}")
        print(f"      VAE+Diffusion: {enh_freq['mid_freq_retention']:.3f}")
        print(f"      改进: {freq_improvements['mid_freq_improvement']:+.3f}")
        
        print(f"   🎼 高频保持率 (>4kHz):")
        print(f"      VAE-only: {vae_freq['high_freq_retention']:.3f}")
        print(f"      VAE+Diffusion: {enh_freq['high_freq_retention']:.3f}")
        print(f"      改进: {freq_improvements['high_freq_improvement']:+.3f}")
        
        # 效果评估
        print(f"\n🎯 效果评估:")
        
        if improvements['snr_improvement'] > 1.0:
            print(f"   ✅ 显著的质量改进！SNR提升 {improvements['snr_improvement']:.1f}dB")
        elif improvements['snr_improvement'] > 0.5:
            print(f"   🟢 明显的质量改进，SNR提升 {improvements['snr_improvement']:.1f}dB")
        elif improvements['snr_improvement'] > 0:
            print(f"   🔶 轻微的质量改进，SNR提升 {improvements['snr_improvement']:.1f}dB")
        else:
            print(f"   ❌ 质量下降，SNR下降 {-improvements['snr_improvement']:.1f}dB")
        
        if freq_improvements['high_freq_improvement'] > 0.1:
            print(f"   🎼 高频显著增强！提升 {freq_improvements['high_freq_improvement']*100:.1f}%")
        elif freq_improvements['high_freq_improvement'] > 0.05:
            print(f"   🎵 高频明显增强，提升 {freq_improvements['high_freq_improvement']*100:.1f}%")
        elif freq_improvements['high_freq_improvement'] > 0:
            print(f"   🔸 高频轻微增强，提升 {freq_improvements['high_freq_improvement']*100:.1f}%")
        else:
            print(f"   ⚠️ 高频未改善或下降")
        
        print(f"\n📁 输出文件:")
        for name, path in result['paths'].items():
            print(f"   {name}: {path}")
        
        print(f"\n💡 总结和建议:")
        
        # 根据结果提供建议
        if improvements['snr_improvement'] > 0.5 and freq_improvements['high_freq_improvement'] > 0.05:
            print(f"   ✅ VAE+Diffusion增强效果良好！")
            print(f"   🎯 建议：可以尝试调整diffusion参数进一步优化")
        elif improvements['snr_improvement'] > 0:
            print(f"   🟢 增强有效果，但还有改进空间")
            print(f"   💡 建议：增加diffusion步数或调整高频增强系数")
        else:
            print(f"   ⚠️ 当前参数可能不够理想")
            print(f"   🔧 建议：减少噪声水平或调整引导尺度")

def demo_vae_diffusion_enhancement():
    """演示VAE+Diffusion增强"""
    print("🚀 Step 2: VAE + Diffusion增强重建")
    print("=" * 60)
    
    # 检查输入文件
    input_file = "AudioLDM2_Music_output.wav"
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        print("请确保AudioLDM2_Music_output.wav在当前目录中")
        return
    
    # 初始化VAE+Diffusion增强器
    enhancer = VAEDiffusionEnhancer()
    
    # 测试不同的参数组合
    test_configs = [
        {
            "name": "轻度增强",
            "diffusion_steps": 10,
            "guidance_scale": 2.0,
            "high_freq_boost": 1.2
        },
        {
            "name": "中度增强",
            "diffusion_steps": 20,
            "guidance_scale": 3.0,
            "high_freq_boost": 1.5
        },
        {
            "name": "强度增强",
            "diffusion_steps": 30,
            "guidance_scale": 4.0,
            "high_freq_boost": 2.0
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🔬 测试配置: {config['name']}")
        print(f"   参数: {config}")
        
        result = enhancer.enhance_reconstruction(
            input_file,
            diffusion_steps=config['diffusion_steps'],
            guidance_scale=config['guidance_scale'],
            high_freq_boost=config['high_freq_boost']
        )
        
        result['config_name'] = config['name']
        results.append(result)
    
    # 对比所有配置的结果
    print(f"\n{'='*80}")
    print(f"📊 所有配置对比总结")
    print(f"{'='*80}")
    
    for i, result in enumerate(results):
        config_name = result['config_name']
        improvements = result['quality_comparison']['improvements']
        freq_improvements = result['frequency_comparison']['improvements']
        
        print(f"\n{i+1}. {config_name}:")
        print(f"   📈 SNR改进: {improvements['snr_improvement']:+.2f} dB")
        print(f"   🎼 高频改进: {freq_improvements['high_freq_improvement']*100:+.1f}%")
        print(f"   🔗 相关性改进: {improvements['correlation_improvement']:+.4f}")
    
    print(f"\n✅ VAE+Diffusion增强测试完成！")
    print(f"📊 查看输出目录: vae_diffusion_enhanced/")
    print(f"🎵 对比不同配置的音频效果，选择最佳参数")

if __name__ == "__main__":
    demo_vae_diffusion_enhancement()
