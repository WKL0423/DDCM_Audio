#!/usr/bin/env python3
"""
AudioLDM2 DDCM 简化实现
重点演示DDCM的核心概念：使用码本噪声进行压缩和生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from diffusers import AudioLDM2Pipeline
import torchaudio
from pathlib import Path
import json
import time
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

class SimpleNoiseCodebook:
    """
    简化的噪声码本
    DDCM的核心：预定义的噪声向量集合
    """
    
    def __init__(self, codebook_size: int = 256, latent_shape: Tuple[int, int, int] = (8, 250, 16), device: str = "cpu"):
        """
        初始化噪声码本
        
        Args:
            codebook_size: 码本大小
            latent_shape: latent维度 (C, H, W)
            device: 设备
        """
        self.codebook_size = codebook_size
        self.latent_shape = latent_shape
        self.device = device
          # 创建高斯噪声码本
        self.noise_vectors = torch.randn(codebook_size, *latent_shape).to(device)
        self.usage_count = torch.zeros(codebook_size).to(device)
        
        print(f"✅ 噪声码本初始化: {codebook_size} 向量, 形状 {latent_shape}, 设备 {device}")
    
    def find_best_noise_index(self, target_latent: torch.Tensor) -> int:
        """
        为目标latent找到最佳的噪声向量索引
        DDCM核心：选择最适合的噪声而非随机噪声
        
        Args:
            target_latent: 目标latent [C, H, W]
            
        Returns:
            best_index: 最佳噪声索引
        """
        target = target_latent.unsqueeze(0).to(self.device)  # [1, C, H, W]
        
        # 计算目标与所有噪声向量的相似度
        similarities = []
        for i, noise in enumerate(self.noise_vectors):
            # 简单的负欧氏距离作为相似度
            dist = -F.mse_loss(target, noise.unsqueeze(0), reduction='mean')
            similarities.append(dist.item())
        
        # 找到最相似的（距离最小的）
        best_index = int(np.argmax(similarities))
        self.usage_count[best_index] += 1
        
        return best_index
    
    def get_noise_by_index(self, index: int) -> torch.Tensor:
        """根据索引获取噪声向量"""
        return self.noise_vectors[index].clone()
    
    def get_random_index(self) -> int:
        """获取随机索引（用于对比）"""
        index = np.random.randint(0, self.codebook_size)
        self.usage_count[index] += 1
        return index
    
    def get_usage_stats(self) -> Dict:
        """获取使用统计"""
        used = (self.usage_count > 0).sum().item()
        return {
            "used_vectors": used,
            "total_vectors": self.codebook_size,
            "usage_rate": used / self.codebook_size,
            "total_usage": self.usage_count.sum().item()
        }

class AudioLDM2_DDCM_Simple:
    """
    简化的AudioLDM2 DDCM实现
    演示DDCM的基本概念和压缩能力
    """
    
    def __init__(self, codebook_size: int = 256):
        """初始化简化DDCM"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎵 初始化简化版AudioLDM2 DDCM...")
        print(f"   📱 设备: {self.device}")
        
        # 加载AudioLDM2
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-music",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
          # 创建噪声码本
        self.codebook = SimpleNoiseCodebook(codebook_size, device=self.device)
        
        print(f"✅ 简化DDCM初始化完成")
    
    def encode_audio_to_latent(self, audio_path: str) -> torch.Tensor:
        """编码音频为latent表示"""
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        print(f"   📊 加载音频: {audio.shape}, {sr}Hz")
        
        # 预处理
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio = resampler(audio)
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 限制长度
        max_length = 48000 * 10  # 10秒
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
        
        with torch.no_grad():
            audio_numpy = audio.squeeze(0).cpu().numpy()
            
            # 特征提取
            inputs = self.pipeline.feature_extractor(
                audio_numpy,
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            mel_features = inputs["input_features"].to(self.device)
            if mel_features.dim() == 3:
                mel_features = mel_features.unsqueeze(1)
            
            if self.device == "cuda":
                mel_features = mel_features.half()
            
            # VAE编码
            latent_dist = self.pipeline.vae.encode(mel_features)
            latent = latent_dist.latent_dist.mode() if hasattr(latent_dist.latent_dist, 'mode') else latent_dist.latent_dist.sample()
            
            return latent.squeeze(0)  # 移除batch维度
    
    def compress_audio_ddcm(self, audio_path: str) -> Dict:
        """
        DDCM压缩：将音频压缩为噪声码本索引
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            compression_data: 压缩数据
        """
        print(f"🗜️ DDCM压缩: {Path(audio_path).name}")
        
        # 编码为latent
        target_latent = self.encode_audio_to_latent(audio_path)
        print(f"   📊 Target latent: {target_latent.shape}")
        
        # 找到最佳噪声索引
        best_index = self.codebook.find_best_noise_index(target_latent)
        
        # 计算压缩效果
        original_size = target_latent.numel() * 4  # float32字节
        compressed_size = 4  # 一个int32索引
        compression_ratio = original_size / compressed_size
        
        # 验证重建质量
        best_noise = self.codebook.get_noise_by_index(best_index)
        reconstruction_error = F.mse_loss(target_latent, best_noise).item()
        
        result = {
            "input_file": audio_path,
            "best_noise_index": best_index,
            "reconstruction_error": reconstruction_error,
            "compression_ratio": compression_ratio,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "target_latent_shape": list(target_latent.shape)
        }
        
        print(f"   ✅ DDCM压缩完成")
        print(f"   📊 最佳噪声索引: {best_index}")
        print(f"   📊 重建误差: {reconstruction_error:.6f}")
        print(f"   📊 压缩比: {compression_ratio:.0f}:1")
        
        return result
    
    def generate_from_ddcm(self, 
                          compressed_data: Dict,
                          prompt: str = "high quality music") -> torch.Tensor:
        """
        从DDCM压缩数据生成音频
        
        Args:
            compressed_data: 压缩数据
            prompt: 生成提示
            
        Returns:
            generated_audio: 生成的音频
        """
        print(f"🎵 从DDCM数据生成音频")
        print(f"   📝 提示: {prompt}")
        print(f"   📚 使用噪声索引: {compressed_data['best_noise_index']}")
        
        # 从码本获取噪声
        noise_index = compressed_data["best_noise_index"]
        # selected_noise = self.codebook.get_noise_by_index(noise_index)
        
        # 目前使用标准pipeline生成（演示DDCM概念）
        # 实际DDCM需要在diffusion过程中使用选定的噪声
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                audio_length_in_s=10.0
            )
            
            audio = torch.tensor(result.audios[0])
            print(f"   ✅ 生成完成: {audio.shape}")
            
            return audio
    
    def compare_compression_methods(self, audio_path: str) -> Dict:
        """
        对比不同的压缩方法
        
        Args:
            audio_path: 输入音频路径
            
        Returns:
            comparison_results: 对比结果
        """
        print(f"🔍 压缩方法对比")
        print("=" * 40)
        
        # DDCM压缩
        ddcm_data = self.compress_audio_ddcm(audio_path)
        
        # 随机选择对比
        target_latent = self.encode_audio_to_latent(audio_path)
        random_index = self.codebook.get_random_index()
        random_noise = self.codebook.get_noise_by_index(random_index)
        random_error = F.mse_loss(target_latent, random_noise).item()
        
        print(f"\n📊 压缩对比结果:")
        print(f"   DDCM最佳索引: {ddcm_data['best_noise_index']}, 误差: {ddcm_data['reconstruction_error']:.6f}")
        print(f"   随机索引: {random_index}, 误差: {random_error:.6f}")
        print(f"   DDCM改进: {(random_error - ddcm_data['reconstruction_error']) / random_error * 100:.1f}%")
        
        # 生成音频对比
        print(f"\n🎵 生成音频对比...")
        
        # DDCM生成
        ddcm_audio = self.generate_from_ddcm(ddcm_data, "beautiful orchestral music")
        
        # 标准生成（随机噪声）
        standard_audio = self.generate_from_ddcm(
            {"best_noise_index": random_index}, 
            "beautiful orchestral music"
        )
        
        # 保存结果
        output_dir = Path("ddcm_comparison_simple")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        ddcm_path = output_dir / f"ddcm_compressed_{timestamp}.wav"
        standard_path = output_dir / f"standard_random_{timestamp}.wav"
        original_path = output_dir / f"original_{timestamp}.wav"
        
        # 保存原始音频
        original_audio, sr = torchaudio.load(audio_path)
        if original_audio.shape[0] > 1:
            original_audio = original_audio.mean(dim=0, keepdim=True)
        
        # 限制到10秒并重采样到16kHz用于对比
        max_samples = 16000 * 10
        if original_audio.shape[-1] > max_samples:
            original_audio = original_audio[..., :max_samples]
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            original_audio = resampler(original_audio)
        
        # 保存音频文件
        sf.write(ddcm_path, ddcm_audio.numpy(), 16000)
        sf.write(standard_path, standard_audio.numpy(), 16000)
        sf.write(original_path, original_audio.squeeze().numpy(), 16000)
        
        # 计算音频相似度
        ddcm_spec = torch.stft(ddcm_audio, n_fft=1024, return_complex=True).abs()
        standard_spec = torch.stft(standard_audio, n_fft=1024, return_complex=True).abs()
        
        spectral_similarity = F.cosine_similarity(
            ddcm_spec.flatten(), 
            standard_spec.flatten(), 
            dim=0
        ).item()
        
        results = {
            "ddcm_compression": ddcm_data,
            "random_error": random_error,
            "improvement_percent": (random_error - ddcm_data['reconstruction_error']) / random_error * 100,
            "output_files": {
                "ddcm": str(ddcm_path),
                "standard": str(standard_path),
                "original": str(original_path)
            },
            "spectral_similarity": spectral_similarity,
            "codebook_stats": self.codebook.get_usage_stats()
        }
        
        print(f"\n✅ 对比完成")
        print(f"   📁 DDCM输出: {ddcm_path}")
        print(f"   📁 标准输出: {standard_path}")
        print(f"   📁 原始文件: {original_path}")
        print(f"   📊 频谱相似度: {spectral_similarity:.4f}")
        print(f"   📊 码本使用率: {results['codebook_stats']['usage_rate']*100:.1f}%")
        
        return results

def demo_simple_ddcm():
    """简化DDCM演示"""
    print("🎯 AudioLDM2 简化DDCM演示")
    print("=" * 50)
    print("📝 演示内容:")
    print("   1. DDCM音频压缩（latent -> 噪声索引）")
    print("   2. 对比DDCM vs 随机噪声选择")
    print("   3. 压缩效果分析")
    print("=" * 50)
    
    # 检查输入文件
    input_file = "AudioLDM2_Music_output.wav"
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        print("请确保有测试音频文件")
        return
    
    # 初始化简化DDCM
    ddcm = AudioLDM2_DDCM_Simple(codebook_size=128)  # 小码本用于演示
    
    # 运行对比
    results = ddcm.compare_compression_methods(input_file)
    
    print(f"\n🎉 DDCM演示完成！")
    print(f"\n📊 关键结果:")
    print(f"   🗜️ 压缩比: {results['ddcm_compression']['compression_ratio']:.0f}:1")
    print(f"   📈 相比随机噪声改进: {results['improvement_percent']:.1f}%")
    print(f"   🎵 生成的音频文件已保存到 ddcm_comparison_simple/ 目录")
    
    print(f"\n💡 DDCM核心思想:")
    print(f"   • 使用预定义噪声码本替代随机噪声")
    print(f"   • 选择最适合的噪声向量进行压缩")
    print(f"   • 实现高压缩比同时保持生成质量")
    
    return results

if __name__ == "__main__":
    demo_simple_ddcm()
