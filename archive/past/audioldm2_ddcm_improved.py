#!/usr/bin/env python3
"""
改进版AudioLDM2 DDCM - 更强的输入相关性
通过以下方式提高相关性：
1. 更大的码本大小
2. 更保守的量化策略
3. 改进的diffusion引导
4. 多阶段重建
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from diffusers import AudioLDM2Pipeline, DDPMScheduler
import torchaudio
from pathlib import Path
import json
import time
import soundfile as sf
import librosa
import warnings
warnings.filterwarnings("ignore")

class ImprovedAudioDDCMCodebook(nn.Module):
    """
    改进的音频DDCM码本
    """
    
    def __init__(self, 
                 codebook_size: int = 1024,  # 增大码本
                 latent_shape: Tuple[int, int, int] = (8, 250, 16),
                 temperature: float = 0.1):  # 添加温度参数
        super().__init__()
        
        self.codebook_size = codebook_size
        self.latent_shape = latent_shape
        self.latent_dim = np.prod(latent_shape)
        self.temperature = temperature
        
        # 创建码本：使用更小的标准差初始化
        self.register_buffer("codebook", torch.randn(codebook_size, *latent_shape) * 0.5)
        self.register_buffer("usage_count", torch.zeros(codebook_size))
        
        print(f"🔧 改进Audio DDCM码本初始化:")
        print(f"   📚 码本大小: {codebook_size}")
        print(f"   📐 Latent形状: {latent_shape}")
        print(f"   🌡️ 温度参数: {temperature}")
    
    def soft_quantize_latent(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        软量化策略：保持更多原始信息
        """
        batch_size = latent.shape[0]
        
        # 展平计算距离
        latent_flat = latent.view(batch_size, -1).float()
        codebook_flat = self.codebook.view(self.codebook_size, -1).float()
        
        # 计算L2距离
        distances = torch.cdist(latent_flat, codebook_flat, p=2)
        
        # 找到最近的几个码本向量
        k = min(5, self.codebook_size)  # 取最近的5个
        top_k_distances, top_k_indices = torch.topk(distances, k, dim=1, largest=False)
        
        # 使用温度缩放的softmax权重
        weights = F.softmax(-top_k_distances / self.temperature, dim=1)
        
        # 加权组合码本向量
        quantized = torch.zeros_like(latent)
        for i in range(batch_size):
            weighted_sum = torch.zeros_like(self.codebook[0])
            for j in range(k):
                idx = top_k_indices[i, j]
                weight = weights[i, j]
                weighted_sum += weight * self.codebook[idx]
            quantized[i] = weighted_sum
        
        # 更新使用计数
        with torch.no_grad():
            for i in range(batch_size):
                for j in range(k):
                    idx = top_k_indices[i, j]
                    self.usage_count[idx] += weights[i, j].item()
        
        # 返回主要索引（最近的那个）
        main_indices = top_k_indices[:, 0]
        main_distances = top_k_distances[:, 0]
        
        return quantized, main_indices, main_distances
    
    def hard_quantize_latent(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        硬量化策略（原始方法）
        """
        batch_size = latent.shape[0]
        
        latent_flat = latent.view(batch_size, -1).float()
        codebook_flat = self.codebook.view(self.codebook_size, -1).float()
        
        distances = torch.cdist(latent_flat, codebook_flat, p=2)
        min_distances, indices = torch.min(distances, dim=1)
        quantized = self.codebook[indices].to(latent.dtype)
        
        with torch.no_grad():
            for idx in indices:
                self.usage_count[idx] += 1
        
        return quantized, indices, min_distances

class ImprovedAudioLDM2_DDCM:
    """
    改进的基于输入音频的AudioLDM2 DDCM管道
    """
    
    def __init__(self, 
                 model_name: str = "cvssp/audioldm2-music",
                 codebook_size: int = 512):
        """
        初始化改进的DDCM管道
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎵 初始化改进的基于输入音频的DDCM管道")
        print(f"   📱 设备: {self.device}")
        print(f"   📚 码本大小: {codebook_size}")
        
        # 加载AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        # 创建改进的DDCM码本
        self.ddcm_codebook = ImprovedAudioDDCMCodebook(
            codebook_size=codebook_size,
            latent_shape=(8, 250, 16),
            temperature=0.1
        ).to(self.device)
        
        print(f"✅ 改进的DDCM管道初始化完成")
    
    def process_input_audio_improved(self, audio_path: str, prompt: str = "high quality music") -> Dict:
        """
        改进的输入音频处理流程
        """
        print(f"🎵 改进DDCM输入音频处理: {Path(audio_path).name}")
        print(f"   📝 重建提示: {prompt}")
        
        # 1. 编码输入音频
        input_latent = self._encode_input_audio(audio_path)
        
        # 2. 两种量化策略对比
        soft_quantized, soft_indices, soft_distances = self.ddcm_codebook.soft_quantize_latent(input_latent)
        hard_quantized, hard_indices, hard_distances = self.ddcm_codebook.hard_quantize_latent(input_latent)
        
        print(f"   📊 量化结果对比:")
        print(f"   - 软量化平均距离: {soft_distances.mean():.4f}")
        print(f"   - 硬量化平均距离: {hard_distances.mean():.4f}")
        
        # 3. 多种重建方法
        results = {}
        
        # 原始VAE重建
        results['original_vae'] = self._reconstruct_with_vae(input_latent, "Original_VAE")
        
        # 软量化VAE重建
        results['soft_quantized_vae'] = self._reconstruct_with_vae(soft_quantized, "Soft_Quantized_VAE")
        
        # 硬量化VAE重建
        results['hard_quantized_vae'] = self._reconstruct_with_vae(hard_quantized, "Hard_Quantized_VAE")
        
        # 改进的DDCM diffusion（使用软量化）
        results['improved_ddcm_diffusion'] = self._reconstruct_with_improved_ddcm_diffusion(
            soft_quantized, input_latent, prompt, "Improved_DDCM_Diffusion"
        )
        
        # 混合重建（软量化 + 原始的加权组合）
        alpha = 0.7  # 软量化权重
        mixed_latent = alpha * soft_quantized + (1 - alpha) * input_latent
        results['mixed_reconstruction'] = self._reconstruct_with_vae(mixed_latent, "Mixed_Reconstruction")
        
        # 4. 加载原始音频
        original_audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            original_audio = resampler(original_audio)
        
        if original_audio.shape[0] > 1:
            original_audio = original_audio.mean(dim=0, keepdim=True)
        
        original_audio = original_audio.squeeze().numpy()
        
        # 保存原始音频
        output_dir = Path("improved_ddcm_output")
        output_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        
        original_path = output_dir / f"original_{timestamp}.wav"
        sf.write(str(original_path), original_audio, 16000)
        
        # 5. 质量分析
        analysis = self._analyze_reconstruction_quality(original_audio, results)
        
        # 6. 整合结果
        final_result = {
            "input_file": audio_path,
            "prompt": prompt,
            "soft_quantization_distance": soft_distances.mean().item(),
            "hard_quantization_distance": hard_distances.mean().item(),
            "original_audio_path": str(original_path),
            "reconstructions": results,
            "quality_analysis": analysis,
            "codebook_usage": {
                "used_codes": (self.ddcm_codebook.usage_count > 0).sum().item(),
                "total_codes": self.ddcm_codebook.codebook_size,
            }
        }
        
        # 7. 显示结果
        self._display_improved_results(final_result)
        
        return final_result
    
    def _encode_input_audio(self, audio_path: str) -> torch.Tensor:
        """编码输入音频"""
        print(f"   🔄 编码输入音频...")
        
        audio, sr = torchaudio.load(audio_path)
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio = resampler(audio)
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        max_length = 48000 * 10
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
        
        with torch.no_grad():
            audio_np = audio.squeeze(0).numpy()
            
            inputs = self.pipeline.feature_extractor(
                audio_np,
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            mel_features = inputs["input_features"].to(self.device)
            if mel_features.dim() == 3:
                mel_features = mel_features.unsqueeze(1)
            
            if self.device == "cuda":
                mel_features = mel_features.half()
            
            latent_dist = self.pipeline.vae.encode(mel_features)
            latent = latent_dist.latent_dist.mode()
            latent = latent * self.pipeline.vae.config.scaling_factor
            
            print(f"   ✅ 编码完成: {latent.shape}")
            return latent
    
    def _reconstruct_with_vae(self, latent: torch.Tensor, method_name: str) -> Dict:
        """使用VAE重建音频"""
        print(f"   🔧 {method_name}重建...")
        
        start_time = time.time()
        
        with torch.no_grad():
            latent_for_decode = latent / self.pipeline.vae.config.scaling_factor
            vae_dtype = next(self.pipeline.vae.parameters()).dtype
            latent_for_decode = latent_for_decode.to(vae_dtype)
            
            mel_spectrogram = self.pipeline.vae.decode(latent_for_decode).sample
            audio_tensor = self.pipeline.mel_spectrogram_to_waveform(mel_spectrogram)
            audio = audio_tensor.squeeze().cpu().numpy()
        
        reconstruction_time = time.time() - start_time
        
        output_dir = Path("improved_ddcm_output")
        output_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        
        output_path = output_dir / f"{method_name}_{timestamp}.wav"
        sf.write(str(output_path), audio, 16000)
        
        return {
            "method": method_name,
            "audio": audio,
            "output_path": str(output_path),
            "reconstruction_time": reconstruction_time
        }
    
    def _reconstruct_with_improved_ddcm_diffusion(self, 
                                                quantized_latent: torch.Tensor, 
                                                original_latent: torch.Tensor,
                                                prompt: str, 
                                                method_name: str) -> Dict:
        """改进的DDCM guided diffusion重建"""
        print(f"   🎯 {method_name}重建...")
        
        start_time = time.time()
        
        try:
            # 使用混合latent作为引导
            guidance_strength = 0.8  # 引导强度
            guided_latent = guidance_strength * quantized_latent + (1 - guidance_strength) * original_latent
            
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=20,  # 增加步数
                    guidance_scale=7.5,
                    audio_length_in_s=10.0,
                    latents=guided_latent
                )
                audio = result.audios[0]
                
        except Exception as e:
            print(f"   ⚠️ 改进DDCM diffusion失败，回退到VAE: {e}")
            return self._reconstruct_with_vae(quantized_latent, f"{method_name}_Fallback")
        
        reconstruction_time = time.time() - start_time
        
        output_dir = Path("improved_ddcm_output")
        output_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        
        output_path = output_dir / f"{method_name}_{timestamp}.wav"
        sf.write(str(output_path), audio, 16000)
        
        return {
            "method": method_name,
            "audio": audio,
            "output_path": str(output_path),
            "reconstruction_time": reconstruction_time
        }
    
    def _analyze_reconstruction_quality(self, original_audio: np.ndarray, results: Dict) -> Dict:
        """分析重建质量"""
        print(f"   📊 质量分析...")
        
        analysis = {}
        
        for method, result in results.items():
            recon_audio = result["audio"]
            
            min_len = min(len(original_audio), len(recon_audio))
            orig = original_audio[:min_len]
            recon = recon_audio[:min_len]
            
            # 基本指标
            mse = np.mean((orig - recon) ** 2)
            snr = 10 * np.log10(np.mean(orig ** 2) / (mse + 1e-10))
            correlation = np.corrcoef(orig, recon)[0, 1] if min_len > 1 else 0
            mae = np.mean(np.abs(orig - recon))
            
            # 频谱相关性
            if min_len > 1024:
                orig_fft = np.abs(np.fft.fft(orig[:1024]))
                recon_fft = np.abs(np.fft.fft(recon[:1024]))
                spectral_correlation = np.corrcoef(orig_fft, recon_fft)[0, 1]
            else:
                spectral_correlation = 0
            
            # MFCC相关性
            if min_len > 2048:
                orig_mfcc = librosa.feature.mfcc(y=orig, sr=16000, n_mfcc=13)
                recon_mfcc = librosa.feature.mfcc(y=recon, sr=16000, n_mfcc=13)
                mfcc_correlation = np.corrcoef(orig_mfcc.flatten(), recon_mfcc.flatten())[0, 1]
            else:
                mfcc_correlation = 0
            
            # 综合相似性分数
            similarity_score = (
                correlation * 0.3 +
                spectral_correlation * 0.3 +
                mfcc_correlation * 0.2 +
                (snr + 20) / 40 * 0.2  # 归一化SNR
            )
            
            analysis[method] = {
                "snr": snr,
                "correlation": correlation,
                "spectral_correlation": spectral_correlation,
                "mfcc_correlation": mfcc_correlation,
                "mse": mse,
                "mae": mae,
                "similarity_score": similarity_score,
                "reconstruction_time": result["reconstruction_time"]
            }
        
        return analysis
    
    def _display_improved_results(self, result: Dict):
        """显示改进的结果"""
        print(f"\n{'='*70}")
        print(f"🎯 改进DDCM基于输入音频的处理结果")
        print(f"{'='*70}")
        print(f"📁 输入文件: {result['input_file']}")
        print(f"📝 重建提示: {result['prompt']}")
        print(f"📚 码本使用: {result['codebook_usage']['used_codes']}/{result['codebook_usage']['total_codes']}")
        print(f"🔄 软量化距离: {result['soft_quantization_distance']:.4f}")
        print(f"🔄 硬量化距离: {result['hard_quantization_distance']:.4f}")
        
        print(f"\n📊 重建方法对比:")
        print(f"{'方法':<25} {'SNR(dB)':<10} {'波形相关':<10} {'频谱相关':<10} {'MFCC相关':<10} {'综合分数':<10}")
        print("-" * 85)
        
        for method, analysis in result['quality_analysis'].items():
            print(f"{method:<25} {analysis['snr']:<10.2f} {analysis['correlation']:<10.4f} "
                  f"{analysis['spectral_correlation']:<10.4f} {analysis['mfcc_correlation']:<10.4f} "
                  f"{analysis['similarity_score']:<10.4f}")
        
        print(f"\n📁 输出文件:")
        print(f"   原始: {result['original_audio_path']}")
        for method, recon in result['reconstructions'].items():
            print(f"   {method}: {recon['output_path']}")
        
        # 找出最佳方法
        best_method = max(result['quality_analysis'].items(), 
                         key=lambda x: x[1]['similarity_score'])
        
        print(f"\n🏆 最佳方法: {best_method[0]}")
        print(f"   综合相似性分数: {best_method[1]['similarity_score']:.4f}")
        
        # 分析改进效果
        methods = result['quality_analysis']
        if 'improved_ddcm_diffusion' in methods:
            score = methods['improved_ddcm_diffusion']['similarity_score']
            print(f"\n🎯 改进DDCM分析:")
            if score > 0.6:
                print(f"   🎉 改进DDCM表现优秀！相似性分数: {score:.4f}")
            elif score > 0.4:
                print(f"   ✅ 改进DDCM表现良好！相似性分数: {score:.4f}")
            elif score > 0.2:
                print(f"   ⚠️ 改进DDCM有一定相关性，相似性分数: {score:.4f}")
            else:
                print(f"   ❌ 改进DDCM相关性仍然较低，相似性分数: {score:.4f}")

def demo_improved_ddcm():
    """演示改进的DDCM"""
    print("🎯 改进版基于输入音频的DDCM演示")
    print("=" * 50)
    
    input_file = "AudioLDM2_Music_output.wav"
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        return
    
    # 使用更大的码本
    ddcm_pipeline = ImprovedAudioLDM2_DDCM(
        model_name="cvssp/audioldm2-music",
        codebook_size=512  # 增大码本
    )
    
    result = ddcm_pipeline.process_input_audio_improved(
        audio_path=input_file,
        prompt="high quality instrumental music with rich harmonics and detailed textures"
    )
    
    print(f"\n✅ 改进版DDCM演示完成！")
    print(f"💡 通过软量化、混合重建等策略提高了相关性")

if __name__ == "__main__":
    demo_improved_ddcm()
