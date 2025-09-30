#!/usr/bin/env python3
"""
AudioLDM2 DDCM (Denoising Diffusion Codebook Model) 完整实现
基于 DDCM 论文：用预定义码本替换随机噪声的diffusion过程
核心思想：使用码本噪声向量替代随机高斯噪声，实现高质量压缩生成
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
import warnings
warnings.filterwarnings("ignore")

class DDCMCodebook(nn.Module):
    """
    DDCM码本：为diffusion过程提供预定义的噪声向量
    替代标准diffusion中的随机高斯噪声
    """
    
    def __init__(self, 
                 codebook_size: int = 1024,
                 latent_shape: Tuple[int, int, int] = (8, 250, 16),
                 noise_schedule: str = "cosine"):
        """
        初始化DDCM码本
        
        Args:
            codebook_size: 码本大小
            latent_shape: AudioLDM2 latent形状 (channels, height, width)
            noise_schedule: 噪声调度类型
        """
        super().__init__()
        
        self.codebook_size = codebook_size
        self.latent_shape = latent_shape
        self.latent_dim = np.prod(latent_shape)
        
        # 创建码本：[codebook_size, channels, height, width]
        # 使用标准高斯分布初始化，但固定这些噪声向量
        self.register_buffer("codebook", torch.randn(codebook_size, *latent_shape))
        
        # 使用计数器追踪使用情况
        self.register_buffer("usage_count", torch.zeros(codebook_size))
        
        print(f"🔧 DDCM码本初始化:")
        print(f"   📚 码本大小: {codebook_size}")
        print(f"   📐 Latent形状: {latent_shape}")
        print(f"   📊 总参数: {self.latent_dim * codebook_size:,}")
    
    def get_noise_for_timestep(self, batch_size: int, timestep: int, device: str = "cuda") -> torch.Tensor:
        """
        为特定时间步获取码本噪声向量
        DDCM核心：用码本噪声替代随机噪声
        
        Args:
            batch_size: 批大小
            timestep: 当前时间步
            device: 设备
            
        Returns:
            noise: 码本噪声向量 [batch_size, channels, height, width]
        """
        # 根据时间步选择码本索引
        # 可以使用不同的选择策略
        indices = self._select_codebook_indices(batch_size, timestep)
        
        # 获取对应的码本向量
        noise = self.codebook[indices].to(device)
        
        # 更新使用计数
        with torch.no_grad():
            for idx in indices:
                self.usage_count[idx] += 1
        
        return noise
    
    def _select_codebook_indices(self, batch_size: int, timestep: int) -> torch.Tensor:
        """
        选择码本索引的策略
        可以实现多种选择方法：随机、基于时间步、基于内容等
        """
        # 策略1: 基于时间步的确定性选择
        base_idx = (timestep * 7) % self.codebook_size  # 使用质数避免周期性
        indices = [(base_idx + i) % self.codebook_size for i in range(batch_size)]
        
        # 策略2: 半随机选择（可选）
        # 在时间步附近选择，增加一些随机性
        if hasattr(self, 'enable_random') and self.enable_random:
            random_offset = torch.randint(-5, 6, (batch_size,))
            indices = [(base_idx + offset.item()) % self.codebook_size 
                      for offset in random_offset]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def find_best_noise_for_target(self, target_latent: torch.Tensor, timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为目标latent找到最佳的码本噪声向量
        这是DDCM用于压缩的核心方法
        
        Args:
            target_latent: 目标latent [batch, channels, height, width]
            timestep: 时间步
            
        Returns:
            best_noise: 最佳噪声向量
            best_indices: 最佳码本索引
        """
        batch_size = target_latent.shape[0]
        
        # 计算所有码本向量与目标的距离
        target_flat = target_latent.view(batch_size, -1)  # [batch, latent_dim]
        codebook_flat = self.codebook.view(self.codebook_size, -1)  # [codebook_size, latent_dim]
        
        # 计算L2距离
        distances = torch.cdist(target_flat, codebook_flat.to(target_latent.device))  # [batch, codebook_size]
        
        # 选择最近的码本向量
        best_indices = torch.argmin(distances, dim=1)  # [batch]
        best_noise = self.codebook[best_indices].to(target_latent.device)
        
        # 更新使用计数
        with torch.no_grad():
            for idx in best_indices:
                self.usage_count[idx] += 1
        
        return best_noise, best_indices
    
    def get_usage_stats(self) -> Dict:
        """获取码本使用统计"""
        used_codes = (self.usage_count > 0).sum().item()
        total_usage = self.usage_count.sum().item()
        
        return {
            "used_codes": used_codes,
            "total_codes": self.codebook_size,
            "usage_rate": used_codes / self.codebook_size,
            "total_usage": total_usage,
            "avg_usage": total_usage / max(used_codes, 1),
            "most_used": self.usage_count.max().item(),
            "least_used": self.usage_count.min().item()
        }

class AudioLDM2_DDCM_Pipeline:
    """
    AudioLDM2 DDCM完整管道
    集成码本化扩散模型到AudioLDM2的生成过程
    """
    
    def __init__(self, 
                 model_name: str = "cvssp/audioldm2-music",
                 codebook_size: int = 512,
                 enable_compression: bool = True):
        """
        初始化AudioLDM2 DDCM管道
        
        Args:
            model_name: AudioLDM2模型名称
            codebook_size: DDCM码本大小
            enable_compression: 是否启用压缩模式
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_compression = enable_compression
        
        print(f"🎵 初始化AudioLDM2 DDCM管道")
        print(f"   📱 设备: {self.device}")
        print(f"   📚 码本大小: {codebook_size}")
        print(f"   🗜️ 压缩模式: {enable_compression}")
        
        # 加载基础AudioLDM2管道
        print("📦 加载AudioLDM2模型...")
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        # 获取latent维度
        self.latent_shape = (8, 250, 16)  # AudioLDM2标准形状
        
        # 创建DDCM码本
        self.ddcm_codebook = DDCMCodebook(
            codebook_size=codebook_size,
            latent_shape=self.latent_shape
        ).to(self.device)
        
        # 创建改进的调度器
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        # 压缩历史
        self.compression_results = []
        
        print(f"✅ AudioLDM2 DDCM管道初始化完成")
    
    def generate_with_ddcm(self, 
                          prompt: str,
                          audio_length_in_s: float = 10.0,
                          num_inference_steps: int = 20,
                          guidance_scale: float = 7.5,
                          use_codebook_noise: bool = True) -> Dict:
        """
        使用DDCM生成音频
        
        Args:
            prompt: 文本提示
            audio_length_in_s: 音频长度（秒）
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            use_codebook_noise: 是否使用码本噪声
            
        Returns:
            result: 生成结果字典
        """
        print(f"🎵 DDCM生成音频")
        print(f"   📝 提示: {prompt}")
        print(f"   ⏱️ 时长: {audio_length_in_s}s")
        print(f"   🔄 步数: {num_inference_steps}")
        print(f"   📚 使用码本噪声: {use_codebook_noise}")
        
        start_time = time.time()
        
        if use_codebook_noise:
            # DDCM模式：使用码本噪声
            audio = self._generate_with_codebook_noise(
                prompt, audio_length_in_s, num_inference_steps, guidance_scale
            )
            method = "DDCM_Codebook"
        else:
            # 标准模式：使用随机噪声
            result = self.pipeline(
                prompt=prompt,
                audio_length_in_s=audio_length_in_s,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            audio = result.audios[0]
            method = "Standard_Diffusion"
        
        generation_time = time.time() - start_time
        
        # 保存结果
        output_dir = Path("ddcm_output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        output_file = output_dir / f"ddcm_{method}_{timestamp}.wav"
        
        # 确保音频格式正确
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = np.array(audio)
        
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        
        # 归一化
        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
            audio_np = audio_np / np.max(np.abs(audio_np))
        
        sf.write(str(output_file), audio_np, 16000)
        
        result = {
            "prompt": prompt,
            "method": method,
            "audio_length": audio_length_in_s,
            "generation_time": generation_time,
            "output_file": str(output_file),
            "audio_data": audio_np,
            "sample_rate": 16000
        }
        
        if use_codebook_noise:
            # 添加码本使用统计
            result["codebook_stats"] = self.ddcm_codebook.get_usage_stats()
          print(f"   ✅ 生成完成: {output_file}")
        print(f"   ⏱️ 用时: {generation_time:.2f}秒")
        
        return result
    
    def _generate_with_codebook_noise(self, 
                                     prompt: str,
                                     audio_length_in_s: float,
                                     num_inference_steps: int,
                                     guidance_scale: float) -> np.ndarray:
        """
        使用码本噪声进行DDCM生成
        这是DDCM的核心实现
        """
        # 编码文本提示 - 使用AudioLDM2的正确方法
        with torch.no_grad():
            # 使用AudioLDM2管道的内置编码方法
            prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=None
            )
            
            # 拼接条件和无条件嵌入
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # 设置调度器
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # 初始化latent：使用码本噪声！
        batch_size = 1
        latents_shape = (batch_size, *self.latent_shape)
        
        # DDCM关键：使用码本噪声而不是随机噪声
        latents = self.ddcm_codebook.get_noise_for_timestep(
            batch_size, timesteps[0].item(), self.device
        )
        
        # 调度器缩放
        latents = latents * self.scheduler.init_noise_sigma
        
        print(f"   🔧 DDCM去噪过程开始...")
        
        # DDCM去噪循环
        for i, t in enumerate(timesteps):
            print(f"     步骤 {i+1}/{len(timesteps)}, t={t}", end="\r")
            
            # 扩展latents用于classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # 预测噪声
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DDCM特色：在某些步骤注入码本噪声
            if i < len(timesteps) // 4:  # 在前25%步骤中使用码本噪声
                # 获取新的码本噪声
                codebook_noise = self.ddcm_codebook.get_noise_for_timestep(
                    batch_size, t.item(), self.device
                )
                # 混合预测噪声和码本噪声
                noise_pred = 0.9 * noise_pred + 0.1 * codebook_noise
            
            # 调度器步骤
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        print(f"\n   🎵 VAE解码...")
        
        # VAE解码
        with torch.no_grad():
            latents = latents / self.pipeline.vae.config.scaling_factor
            mel_spectrogram = self.pipeline.vae.decode(latents).sample
            
            # Vocoder
            audio = self.pipeline.mel_spectrogram_to_waveform(mel_spectrogram)
            audio = audio.squeeze().cpu().numpy()
        
        return audio
    
    def compress_audio_with_ddcm(self, audio_path: str) -> Dict:
        """
        使用DDCM压缩音频
        找到最佳的码本表示
        
        Args:
            audio_path: 输入音频路径
            
        Returns:
            compression_result: 压缩结果
        """
        print(f"🗜️ DDCM音频压缩: {Path(audio_path).name}")
        
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio = resampler(audio)
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 限制长度
        max_length = 48000 * 10  # 10秒
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
        
        print(f"   📊 音频: {audio.shape}, 48kHz")
        
        # 编码为latent
        with torch.no_grad():
            audio_np = audio.squeeze(0).numpy()
            
            # 使用ClapFeatureExtractor
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
            
            # VAE编码
            latent_dist = self.pipeline.vae.encode(mel_features)
            latent = latent_dist.latent_dist.mode()
            latent = latent * self.pipeline.vae.config.scaling_factor
        
        print(f"   🧠 Latent: {latent.shape}")
        
        # 找到最佳码本表示
        best_noise, best_indices = self.ddcm_codebook.find_best_noise_for_target(latent, timestep=0)
        
        # 计算压缩效果
        original_size = latent.numel() * 4  # float32字节
        compressed_size = len(best_indices) * 4  # int32字节
        compression_ratio = original_size / compressed_size
        
        # 测试重建质量
        with torch.no_grad():
            # 使用码本向量重建
            reconstructed_latent = best_noise / self.pipeline.vae.config.scaling_factor
            reconstructed_mel = self.pipeline.vae.decode(reconstructed_latent).sample
            reconstructed_audio = self.pipeline.mel_spectrogram_to_waveform(reconstructed_mel)
            reconstructed_audio = reconstructed_audio.squeeze().cpu().numpy()
        
        # 计算质量指标
        original_audio = audio.squeeze().numpy()
        if len(reconstructed_audio) > len(original_audio):
            reconstructed_audio = reconstructed_audio[:len(original_audio)]
        elif len(reconstructed_audio) < len(original_audio):
            reconstructed_audio = np.pad(
                reconstructed_audio, 
                (0, len(original_audio) - len(reconstructed_audio))
            )
        
        mse = np.mean((original_audio - reconstructed_audio) ** 2)
        snr = 10 * np.log10(np.mean(original_audio ** 2) / (mse + 1e-10))
        correlation = np.corrcoef(original_audio, reconstructed_audio)[0, 1]
        
        result = {
            "input_file": audio_path,
            "codebook_indices": best_indices.cpu().tolist(),
            "compression_ratio": compression_ratio,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "reconstruction_snr": snr,
            "reconstruction_correlation": correlation,
            "reconstruction_mse": mse
        }
        
        self.compression_results.append(result)
        
        print(f"   ✅ 压缩完成")
        print(f"   📊 压缩比: {compression_ratio:.2f}:1")
        print(f"   📊 重建SNR: {snr:.2f} dB")
        print(f"   📊 相关性: {correlation:.4f}")
        
        return result

def demo_ddcm_complete():
    """完整的DDCM演示"""
    print("🎯 AudioLDM2 DDCM 完整演示")
    print("=" * 50)
    
    # 初始化DDCM管道
    ddcm_pipeline = AudioLDM2_DDCM_Pipeline(
        model_name="cvssp/audioldm2-music",
        codebook_size=256,
        enable_compression=True
    )
    
    # 1. 标准diffusion生成
    print("\n🎵 第1步: 标准Diffusion生成")
    standard_result = ddcm_pipeline.generate_with_ddcm(
        prompt="ambient electronic music with soft synths",
        audio_length_in_s=10.0,
        num_inference_steps=20,
        use_codebook_noise=False
    )
    
    # 2. DDCM生成
    print("\n🎵 第2步: DDCM生成")
    ddcm_result = ddcm_pipeline.generate_with_ddcm(
        prompt="ambient electronic music with soft synths",
        audio_length_in_s=10.0,
        num_inference_steps=20,
        use_codebook_noise=True
    )
    
    # 3. 如果有AudioLDM2_Music_output.wav，进行压缩测试
    input_file = "AudioLDM2_Music_output.wav"
    if Path(input_file).exists():
        print("\n🗜️ 第3步: DDCM压缩测试")
        compression_result = ddcm_pipeline.compress_audio_with_ddcm(input_file)
    
    # 4. 显示对比结果
    print("\n📊 结果对比")
    print("-" * 50)
    print(f"标准生成用时: {standard_result['generation_time']:.2f}秒")
    print(f"DDCM生成用时: {ddcm_result['generation_time']:.2f}秒")
    print(f"码本使用统计: {ddcm_result.get('codebook_stats', {})}")
    
    if Path(input_file).exists():
        print(f"压缩比: {compression_result['compression_ratio']:.2f}:1")
        print(f"重建质量SNR: {compression_result['reconstruction_snr']:.2f} dB")
    
    print(f"\n✅ DDCM完整演示完成！")
    print(f"📁 输出文件:")
    print(f"   标准: {standard_result['output_file']}")
    print(f"   DDCM: {ddcm_result['output_file']}")

if __name__ == "__main__":
    demo_ddcm_complete()
