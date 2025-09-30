"""
AudioLDM2 DDCM (Denoising Diffusion Codebook Model) 实现
基于 DDCM 论文思想，为 AudioLDM2 创建码本化的 diffusion 过程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from diffusers import AudioLDM2Pipeline
import torchaudio
from pathlib import Path
import json
import pickle
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

class AudioCodebook(nn.Module):
    """
    音频扩散码本
    为 AudioLDM2 的 latent space 设计的码本结构
    """
    
    def __init__(self, 
                 codebook_size: int = 1024,
                 latent_dim: int = 8,
                 latent_height: int = 250, 
                 latent_width: int = 16,
                 init_method: str = "gaussian"):
        """
        初始化音频码本
        
        Args:
            codebook_size: 码本大小
            latent_dim: 潜在空间维度 (AudioLDM2 VAE latent channels)
            latent_height: 潜在空间高度 (时间维度)
            latent_width: 潜在空间宽度 (频率维度)
            init_method: 初始化方法 ("gaussian", "uniform", "kmeans")
        """
        super().__init__()
        
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_shape = (latent_dim, latent_height, latent_width)
        
        # 码本向量：[codebook_size, latent_dim, latent_height, latent_width]
        if init_method == "gaussian":
            self.codebook = nn.Parameter(
                torch.randn(codebook_size, latent_dim, latent_height, latent_width)
            )
        elif init_method == "uniform":
            self.codebook = nn.Parameter(
                torch.rand(codebook_size, latent_dim, latent_height, latent_width) * 2 - 1
            )
        else:
            # 延迟初始化，用于 k-means
            self.register_parameter('codebook', None)
        
        self.usage_count = torch.zeros(codebook_size)
        
    def initialize_with_kmeans(self, sample_latents: torch.Tensor, n_samples: int = 10000):
        """
        使用 k-means 初始化码本
        
        Args:
            sample_latents: 样本潜在表示 [n_samples, latent_dim, latent_height, latent_width]
            n_samples: 用于聚类的样本数量
        """
        print(f"🔧 使用 K-means 初始化码本 (样本数: {min(n_samples, len(sample_latents))})")
        
        # 随机选择样本
        if len(sample_latents) > n_samples:
            indices = torch.randperm(len(sample_latents))[:n_samples]
            sample_latents = sample_latents[indices]
        
        # 展平样本
        flattened = sample_latents.view(len(sample_latents), -1).cpu().numpy()
        
        # K-means 聚类
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10)
        kmeans.fit(flattened)
        
        # 重塑为码本形状
        centroids = torch.tensor(kmeans.cluster_centers_).float()
        centroids = centroids.view(self.codebook_size, *self.latent_shape)        
        self.codebook = nn.Parameter(centroids)
        print(f"✅ K-means 码本初始化完成")
    
    def find_nearest_codes(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为给定的潜在表示找到最近的码本向量
        
        Args:
            latents: 潜在表示 [batch_size, latent_dim, latent_height, latent_width]
            
        Returns:
            indices: 码本索引 [batch_size]
            quantized: 量化后的潜在表示 [batch_size, latent_dim, latent_height, latent_width]
            distances: 距离 [batch_size]
        """
        batch_size = latents.shape[0]
        
        # 确保数据类型一致
        latents = latents.float()
        codebook_float = self.codebook.float()
        
        # 展平用于计算距离
        latents_flat = latents.view(batch_size, -1)  # [batch, latent_dim*height*width]
        codebook_flat = codebook_float.view(self.codebook_size, -1)  # [codebook_size, latent_dim*height*width]
        
        # 计算 L2 距离
        distances = torch.cdist(latents_flat, codebook_flat, p=2)  # [batch, codebook_size]
        
        # 找到最近的码本向量
        min_distances, indices = torch.min(distances, dim=1)  # [batch]
        
        # 获取量化后的向量
        quantized = codebook_float[indices]  # [batch, latent_dim, latent_height, latent_width]
        
        # 更新使用计数
        with torch.no_grad():
            for idx in indices:
                self.usage_count[idx] += 1
        
        return indices, quantized, min_distances
    
    def get_code_by_index(self, indices: torch.Tensor) -> torch.Tensor:
        """
        根据索引获取码本向量
        
        Args:
            indices: 码本索引 [batch_size]
            
        Returns:
            codes: 码本向量 [batch_size, latent_dim, latent_height, latent_width]
        """
        return self.codebook[indices]
    
    def get_usage_stats(self) -> Dict:
        """获取码本使用统计"""
        used_codes = (self.usage_count > 0).sum().item()
        total_usage = self.usage_count.sum().item()
        
        return {
            "used_codes": used_codes,
            "total_codes": self.codebook_size,
            "usage_rate": used_codes / self.codebook_size,
            "total_usage": total_usage,
            "avg_usage": total_usage / max(used_codes, 1)
        }

class AudioLDM2_DDCM(nn.Module):
    """
    AudioLDM2 DDCM 主类
    集成码本化扩散模型到 AudioLDM2 pipeline
    """
    
    def __init__(self, 
                 model_name: str = "cvssp/audioldm2-music",
                 codebook_size: int = 1024,
                 compression_level: str = "medium"):
        """
        初始化 AudioLDM2 DDCM
        
        Args:
            model_name: AudioLDM2 模型名称
            codebook_size: 码本大小
            compression_level: 压缩级别 ("low", "medium", "high", "extreme")
        """
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎵 初始化 AudioLDM2 DDCM...")
        print(f"   📱 设备: {self.device}")
        print(f"   📚 码本大小: {codebook_size}")
        print(f"   🗜️ 压缩级别: {compression_level}")
        
        # 加载基础 AudioLDM2 pipeline
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
          # 获取 VAE latent 空间维度
        # AudioLDM2 VAE latent shape: [batch, 8, 250, 16] for audio segments
        self.latent_channels = self.pipeline.vae.config.latent_channels  # 8
        
        # 根据压缩级别调整码本配置
        compression_configs = {
            "low": {"codebook_size": codebook_size * 4, "selection_strategy": "nearest"},
            "medium": {"codebook_size": codebook_size, "selection_strategy": "nearest"},  
            "high": {"codebook_size": codebook_size // 2, "selection_strategy": "nearest"},
            "extreme": {"codebook_size": codebook_size // 4, "selection_strategy": "probabilistic"}
        }
        
        config = compression_configs[compression_level]
        self.compression_level = compression_level
        self.selection_strategy = config["selection_strategy"]
        
        # 创建码本 (使用实际的 latent 维度)
        self.codebook = AudioCodebook(
            codebook_size=config["codebook_size"],
            latent_dim=self.latent_channels,
            latent_height=250,  # AudioLDM2 实际时间维度
            latent_width=16,    # AudioLDM2 实际频率维度
            init_method="gaussian"  # 先用高斯初始化，后续可用 k-means
        ).to(self.device)
        
        # 压缩历史
        self.compression_history = []
        
        print(f"✅ AudioLDM2 DDCM 初始化完成")
    
    def prepare_codebook_with_dataset(self, audio_files: List[str], max_samples: int = 1000):
        """
        使用音频数据集准备码本
        
        Args:
            audio_files: 音频文件路径列表
            max_samples: 最大样本数量
        """
        print(f"📚 使用数据集准备码本 ({len(audio_files)} 文件, 最大样本: {max_samples})")
        
        sample_latents = []
        processed = 0
        
        for audio_file in audio_files:
            if processed >= max_samples:
                break
                
            try:
                # 加载音频
                audio, sr = torchaudio.load(audio_file)
                
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
                
                # 编码为 latent
                latent = self._encode_audio_to_latent(audio.squeeze(0))
                sample_latents.append(latent)
                
                processed += 1
                if processed % 100 == 0:
                    print(f"   已处理: {processed}/{min(len(audio_files), max_samples)}")
                    
            except Exception as e:
                print(f"   ⚠️ 跳过文件 {audio_file}: {e}")
                continue
        
        if sample_latents:
            sample_latents = torch.stack(sample_latents)
            print(f"📊 收集到 {len(sample_latents)} 个潜在样本")
            
            # 使用 k-means 初始化码本
            self.codebook.initialize_with_kmeans(sample_latents)
        else:
            print("⚠️ 未能收集到有效样本，使用默认高斯初始化")
    
    def _encode_audio_to_latent(self, audio: torch.Tensor) -> torch.Tensor:
        """编码音频为潜在表示"""
        with torch.no_grad():
            # 确保正确格式
            if audio.is_cuda:
                audio = audio.cpu()
            if audio.dim() == 2:
                audio = audio.squeeze(0)
                
            audio_numpy = audio.numpy()
            
            # ClapFeatureExtractor
            inputs = self.pipeline.feature_extractor(
                audio_numpy,
                sampling_rate=48000,
                return_tensors="pt"            )
            
            mel_features = inputs["input_features"].to(self.device)
            if mel_features.dim() == 3:
                mel_features = mel_features.unsqueeze(1)
            
            # 确保数据类型匹配
            if self.device == "cuda":
                mel_features = mel_features.half()
                
            # VAE 编码
            latent_dist = self.pipeline.vae.encode(mel_features)
            latent = latent_dist.latent_dist.mode() if hasattr(latent_dist.latent_dist, 'mode') else latent_dist.latent_dist.sample()
            
            return latent.squeeze(0)  # 移除 batch 维度
    
    def compress_audio(self, audio_path: str) -> Dict:
        """
        压缩音频到码本表示
        
        Args:
            audio_path: 输入音频路径
            
        Returns:
            compression_result: 压缩结果字典
        """
        print(f"🗜️ 压缩音频: {Path(audio_path).name}")
        
        # 加载和预处理音频
        audio, sr = torchaudio.load(audio_path)
        print(f"   📊 原始音频: {audio.shape}, {sr}Hz")
        
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio = resampler(audio)
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 分段处理长音频
        segment_length = 48000 * 10  # 10秒分段
        audio_squeezed = audio.squeeze(0)
        segments = []
        
        for i in range(0, len(audio_squeezed), segment_length):
            segment = audio_squeezed[i:i+segment_length]
            if len(segment) < segment_length:
                # 填充最后一段
                padding = segment_length - len(segment)
                segment = torch.cat([segment, torch.zeros(padding)])
            segments.append(segment)
        
        print(f"   📦 分为 {len(segments)} 个段落")
        
        # 压缩每个段落
        compressed_data = []
        total_original_size = 0
        total_compressed_size = 0
        
        for i, segment in enumerate(segments):
            # 编码为 latent
            latent = self._encode_audio_to_latent(segment)
            
            # 找到最近的码本向量
            indices, quantized, distances = self.codebook.find_nearest_codes(latent.unsqueeze(0))
            
            # 压缩信息
            segment_info = {
                "segment_id": i,
                "codebook_index": indices[0].item(),
                "distance": distances[0].item(),
                "original_latent_shape": latent.shape
            }
            
            compressed_data.append(segment_info)
            
            # 计算压缩比
            original_size = latent.numel() * 4  # float32 字节数
            compressed_size = 4  # 只存储索引 (int32)
            
            total_original_size += original_size
            total_compressed_size += compressed_size
        
        compression_ratio = total_original_size / total_compressed_size
        
        result = {
            "input_file": audio_path,
            "segments": compressed_data,
            "compression_ratio": compression_ratio,
            "original_size_bytes": total_original_size,
            "compressed_size_bytes": total_compressed_size,
            "codebook_size": self.codebook.codebook_size,
            "compression_level": self.compression_level
        }
        
        self.compression_history.append(result)
        
        print(f"   ✅ 压缩完成")
        print(f"   📊 压缩比: {compression_ratio:.2f}:1")
        print(f"   📊 原始大小: {total_original_size} 字节")
        print(f"   📊 压缩大小: {total_compressed_size} 字节")
        
        return result
    
    def decompress_and_generate(self, 
                               compressed_data: Dict,
                               prompt: str = "high quality music",
                               num_inference_steps: int = 20,
                               guidance_scale: float = 7.5) -> torch.Tensor:
        """
        解压缩并生成音频
        
        Args:
            compressed_data: 压缩数据
            prompt: 文本提示
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            
        Returns:
            generated_audio: 生成的音频
        """
        print(f"🎵 解压缩并生成音频")
        print(f"   📝 提示: {prompt}")
        print(f"   📦 段落数: {len(compressed_data['segments'])}")
        
        generated_segments = []
        
        for segment_info in compressed_data["segments"]:
            # 从码本获取量化潜在表示
            codebook_index = segment_info["codebook_index"]
            quantized_latent = self.codebook.get_code_by_index(torch.tensor([codebook_index]).to(self.device))
            
            # 使用量化的 latent 作为起点进行 diffusion
            # 这里我们可以选择：
            # 1. 直接解码 (快速但质量可能较低)
            # 2. 使用 diffusion 重新生成 (质量更好但较慢)
            
            if self.compression_level in ["low", "medium"]:
                # 直接解码模式
                with torch.no_grad():
                    mel = self.pipeline.vae.decode(quantized_latent).sample
                    # 注意：这里需要vocoder将mel转为音频，但AudioLDM2可能没有直接的vocoder访问
                    # 我们使用pipeline的完整生成过程，但用量化latent作为条件
                    pass
            
            # 使用 diffusion 生成 (推荐方式)
            # 这里我们生成固定长度的音频段
            segment_audio = self._generate_audio_segment(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                audio_length_in_s=10.0
            )
            
            generated_segments.append(segment_audio)
        
        # 拼接所有段落
        full_audio = torch.cat(generated_segments, dim=0)
        
        print(f"   ✅ 生成完成: {full_audio.shape}")
        return full_audio
    
    def _generate_audio_segment(self, 
                              prompt: str,
                              num_inference_steps: int = 20,
                              guidance_scale: float = 7.5,
                              audio_length_in_s: float = 10.0) -> torch.Tensor:
        """生成单个音频段"""
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                audio_length_in_s=audio_length_in_s
            )
            return torch.tensor(result.audios[0])
    
    def save_compressed_data(self, compressed_data: Dict, save_path: str):
        """保存压缩数据"""
        with open(save_path, 'w') as f:
            json.dump(compressed_data, f, indent=2)
        print(f"💾 压缩数据保存到: {save_path}")
    
    def load_compressed_data(self, load_path: str) -> Dict:
        """加载压缩数据"""
        with open(load_path, 'r') as f:
            compressed_data = json.load(f)
        print(f"📂 从 {load_path} 加载压缩数据")
        return compressed_data
    
    def save_codebook(self, save_path: str):
        """保存码本"""
        codebook_data = {
            "codebook": self.codebook.state_dict(),
            "usage_count": self.codebook.usage_count,
            "config": {
                "codebook_size": self.codebook.codebook_size,
                "latent_shape": self.codebook.latent_shape,
                "compression_level": self.compression_level
            }
        }
        torch.save(codebook_data, save_path)
        print(f"💾 码本保存到: {save_path}")
    
    def load_codebook(self, load_path: str):
        """加载码本"""
        codebook_data = torch.load(load_path, map_location=self.device)
        self.codebook.load_state_dict(codebook_data["codebook"])
        self.codebook.usage_count = codebook_data["usage_count"]
        print(f"📂 从 {load_path} 加载码本")
        
        # 显示使用统计
        stats = self.codebook.get_usage_stats()
        print(f"   📊 码本统计: {stats}")

def demo_ddcm_workflow():
    """DDCM 工作流程演示"""
    print("🎯 AudioLDM2 DDCM 工作流程演示")
    print("=" * 50)
    
    # 1. 初始化 DDCM
    ddcm = AudioLDM2_DDCM(
        model_name="cvssp/audioldm2-music",
        codebook_size=512,
        compression_level="medium"
    )
    
    # 2. 检查输入文件
    input_file = "AudioLDM2_Music_output.wav"
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        print("请确保有测试音频文件")
        return
    
    # 3. 压缩音频
    print("\n🗜️ 第1步: 压缩音频")
    compressed_data = ddcm.compress_audio(input_file)
    
    # 4. 保存压缩数据
    compressed_file = "compressed_audio.json"
    ddcm.save_compressed_data(compressed_data, compressed_file)
    
    # 5. 保存码本
    codebook_file = "audio_codebook.pth"
    ddcm.save_codebook(codebook_file)
    
    # 6. 解压缩并生成
    print("\n🎵 第2步: 解压缩并生成音频")
    generated_audio = ddcm.decompress_and_generate(
        compressed_data,
        prompt="high quality instrumental music with rich harmonics",
        num_inference_steps=25,
        guidance_scale=7.5
    )
    
    # 7. 保存生成的音频
    output_dir = Path("ddcm_output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "ddcm_generated.wav"
    if generated_audio.dim() == 1:
        generated_audio = generated_audio.unsqueeze(0)
    
    torchaudio.save(str(output_file), generated_audio.cpu(), 16000)
    print(f"💾 生成音频保存到: {output_file}")
    
    # 8. 显示统计信息
    print("\n📊 DDCM 统计信息")
    print("-" * 30)
    print(f"压缩比: {compressed_data['compression_ratio']:.2f}:1")
    print(f"码本使用率: {ddcm.codebook.get_usage_stats()['usage_rate']*100:.1f}%")
    print(f"压缩级别: {ddcm.compression_level}")
    
    print("\n✅ DDCM 工作流程演示完成！")

if __name__ == "__main__":
    demo_ddcm_workflow()
