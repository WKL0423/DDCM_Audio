#!/usr/bin/env python3
"""
AudioLDM2 引导式Diffusion重建 - 简化稳定版本
===========================================

基于之前成功的VAE脚本，实现一个更稳定的guided diffusion reconstruction。
该版本专注于核心创新思想的实现，而不是复杂的音频处理细节。

核心创新：
1. 使用VAE encoder获取目标音频的latent representation
2. 从随机噪声开始，使用diffusion过程
3. 在每个diffusion步骤中，添加指向目标latent的引导力
4. 最终使用VAE decoder重建音频

Author: AI Assistant  
Date: 2025-01-27
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Tuple
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from New_pipeline_audioldm2 import AudioLDM2Pipeline
except ImportError:
    print("❌ 无法导入AudioLDM2Pipeline，请确保New_pipeline_audioldm2.py在当前目录")
    sys.exit(1)


class SimpleGuidedDiffusionReconstructor:
    """简化版引导式Diffusion重建器"""
    
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        print(f"🚀 初始化简化版引导式Diffusion重建器")
        print(f"   设备: {device}, 数据类型: {dtype}")
        
        # 加载模型
        print("📦 加载AudioLDM2模型...")
        repo_id = "cvssp/audioldm2"
        self.pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=dtype)
        self.pipe = self.pipe.to(device)
        print("✅ 模型加载成功")
        
        # 获取组件
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        
        # 设置为评估模式
        self.vae.eval()
        self.unet.eval()
    
    def save_audio_compatible(self, audio: np.ndarray, filepath: str, sr: int = 16000):
        """保存兼容的音频文件"""
        try:
            # 确保音频在[-1, 1]范围内
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            # 转换为16位PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            sf.write(filepath, audio_int16, sr, subtype='PCM_16')
            
            print(f"💾 保存音频: {Path(filepath).name}")
            
        except Exception as e:
            print(f"❌ 保存音频失败: {e}")
            raise
    
    def load_and_encode_audio(self, audio_path: str) -> torch.Tensor:
        """
        加载音频并编码到latent space
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            latent tensor
        """
        print(f"\n📁 加载音频: {Path(audio_path).name}")
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"   采样率: {sr}Hz, 时长: {len(audio)/sr:.2f}s")
        
        # 归一化
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # 确保音频长度符合模型要求
        target_length = 163840  # 16000 * 10.24s, AudioLDM2的标准长度
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]
        
        print(f"   处理后长度: {len(audio)/sr:.2f}s")
        
        # 生成mel频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=16000, 
            n_mels=64, 
            n_fft=1024, 
            hop_length=160
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 归一化mel频谱图到[-1, 1]
        mel_spec = (mel_spec + 80) / 80 * 2 - 1
        
        # 转换为tensor并调整形状
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).to(
            device=self.device, dtype=self.dtype
        )
        
        # 调整到标准尺寸 (1, 1, 64, 1024)
        if mel_tensor.shape[-1] != 1024:
            mel_tensor = torch.nn.functional.interpolate(
                mel_tensor, size=(64, 1024), mode='bilinear', align_corners=False
            )
        
        print(f"🎵 Mel频谱图形状: {mel_tensor.shape}")
        
        # VAE编码
        with torch.no_grad():
            latent_dist = self.vae.encode(mel_tensor)
            if hasattr(latent_dist, 'latent_dist'):
                latent = latent_dist.latent_dist.sample()
            else:
                latent = latent_dist.sample()
            
            latent = latent * self.vae.config.scaling_factor
        
        print(f"🔢 Latent表示形状: {latent.shape}")
        print(f"   范围: [{latent.min():.3f}, {latent.max():.3f}]")
        
        return latent
    
    def decode_latent_to_audio(self, latent: torch.Tensor) -> np.ndarray:
        """
        将latent解码为音频
        
        Args:
            latent: latent tensor
            
        Returns:
            音频数组
        """
        print(f"🔄 解码latent到音频...")
        
        with torch.no_grad():
            # VAE解码
            latent_scaled = latent / self.vae.config.scaling_factor
            mel_spec = self.vae.decode(latent_scaled).sample
            
            print(f"   解码的mel频谱图形状: {mel_spec.shape}")
              # 转换为numpy，确保使用float32
            mel_spec_np = mel_spec.squeeze().cpu().float().numpy()
            
            # 反归一化
            mel_spec_np = (mel_spec_np + 1) / 2 * 80 - 80
            
            # Griffin-Lim重建音频
            audio = librosa.feature.inverse.mel_to_audio(
                librosa.db_to_power(mel_spec_np),
                sr=16000,
                n_fft=1024,
                hop_length=160,
                n_iter=32
            )
            
            print(f"🔊 重建音频形状: {audio.shape}")
            
            return audio
    
    def compute_guidance_loss(self, current_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        """计算引导损失"""
        # 使用MSE损失
        loss = torch.nn.functional.mse_loss(current_latent, target_latent)
        return loss
    
    def guided_diffusion_reconstruction(self, 
                                      target_latent: torch.Tensor,
                                      num_steps: int = 50,
                                      guidance_scale: float = 0.1,
                                      guidance_decay: float = 0.95) -> torch.Tensor:
        """
        引导式diffusion重建
        
        Args:
            target_latent: 目标latent
            num_steps: diffusion步数
            guidance_scale: 引导强度
            guidance_decay: 引导强度衰减
            
        Returns:
            重建的latent
        """
        print(f"\n🌟 开始引导式Diffusion重建")
        print(f"   步数: {num_steps}")
        print(f"   引导强度: {guidance_scale}")
        print(f"   强度衰减: {guidance_decay}")
        
        # 初始化噪声
        latent_shape = target_latent.shape
        noise = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
        
        # 设置scheduler
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps
        
        # 从噪声开始
        latents = noise.clone()
        
        current_guidance = guidance_scale
        
        for i, t in enumerate(timesteps):
            if i % 10 == 0:
                print(f"   步骤 {i+1}/{num_steps}, 时间步: {t.item():.0f}, 引导强度: {current_guidance:.4f}")
            
            # 准备输入
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            # 准备条件嵌入（使用零嵌入进行无条件生成）
            batch_size = latent_model_input.shape[0]
            
            # 创建空的条件嵌入
            encoder_hidden_states = torch.zeros(
                (batch_size, 77, 768), 
                device=self.device, 
                dtype=self.dtype
            )
            encoder_hidden_states_1 = torch.zeros(
                (batch_size, 77, 1024), 
                device=self.device, 
                dtype=self.dtype
            )
            
            # 预测噪声
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_1=encoder_hidden_states_1,
                    return_dict=False,
                )[0]
            
            # 标准diffusion步骤
            step_output = self.scheduler.step(noise_pred, t, latents)
            latents = step_output.prev_sample
            
            # 应用引导（如果不是最后几步）
            if i < num_steps - 5 and current_guidance > 1e-4:
                # 计算到目标的梯度
                latents.requires_grad_(True)
                loss = self.compute_guidance_loss(latents, target_latent)
                grad = torch.autograd.grad(loss, latents)[0]
                
                # 应用引导
                with torch.no_grad():
                    latents = latents - current_guidance * grad
                    latents.requires_grad_(False)
                
                # 衰减引导强度
                current_guidance *= guidance_decay
        
        print(f"✅ 引导式重建完成")
        print(f"   最终latent范围: [{latents.min():.3f}, {latents.max():.3f}]")
        
        return latents
    
    def reconstruct_audio(self, 
                         input_path: str,
                         output_dir: str = "guided_diffusion_simple_output",
                         num_steps: int = 50,
                         guidance_scale: float = 0.1,
                         compare_vae: bool = True) -> Tuple[str, Optional[str]]:
        """
        完整的重建流程
        
        Args:
            input_path: 输入音频路径
            output_dir: 输出目录
            num_steps: diffusion步数
            guidance_scale: 引导强度
            compare_vae: 是否与纯VAE对比
            
        Returns:
            (guided_path, vae_path)
        """
        print(f"\n{'='*60}")
        print(f"🎯 简化版引导式Diffusion音频重建")
        print(f"{'='*60}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载并编码音频
        target_latent = self.load_and_encode_audio(input_path)
        
        # 执行引导式diffusion重建
        guided_latent = self.guided_diffusion_reconstruction(
            target_latent=target_latent,
            num_steps=num_steps,
            guidance_scale=guidance_scale
        )
        
        # 解码为音频
        guided_audio = self.decode_latent_to_audio(guided_latent)
        
        # 保存引导式重建结果
        input_name = Path(input_path).stem
        guided_path = os.path.join(
            output_dir, 
            f"{input_name}_guided_{num_steps}steps.wav"
        )
        self.save_audio_compatible(guided_audio, guided_path)
        
        vae_path = None
        
        # 纯VAE重建对比
        if compare_vae:
            print(f"\n🔄 执行纯VAE重建对比...")
            vae_audio = self.decode_latent_to_audio(target_latent)
            vae_path = os.path.join(
                output_dir, 
                f"{input_name}_vae_only.wav"
            )
            self.save_audio_compatible(vae_audio, vae_path)
        
        # 计算质量指标
        self.compute_quality_metrics(input_path, guided_audio, 
                                   self.decode_latent_to_audio(target_latent) if compare_vae else None)
        
        return guided_path, vae_path
    
    def compute_quality_metrics(self, original_path: str, guided_audio: np.ndarray, vae_audio: Optional[np.ndarray] = None):
        """计算重建质量指标"""
        print(f"\n📊 重建质量分析")
        print(f"─" * 40)
        
        # 加载原始音频
        original_audio, _ = librosa.load(original_path, sr=16000, mono=True)
        
        # 确保长度一致
        min_len = min(len(original_audio), len(guided_audio))
        original_audio = original_audio[:min_len]
        guided_audio = guided_audio[:min_len]
        
        # 归一化
        if np.max(np.abs(original_audio)) > 0:
            original_audio = original_audio / np.max(np.abs(original_audio))
        if np.max(np.abs(guided_audio)) > 0:
            guided_audio = guided_audio / np.max(np.abs(guided_audio))
        
        # 计算MSE和SNR
        mse_guided = np.mean((original_audio - guided_audio) ** 2)
        snr_guided = 10 * np.log10(np.var(original_audio) / mse_guided) if mse_guided > 0 else float('inf')
        
        # 计算相关系数
        correlation_guided = np.corrcoef(original_audio, guided_audio)[0, 1]
        
        print(f"引导式Diffusion重建:")
        print(f"  MSE: {mse_guided:.6f}")
        print(f"  SNR: {snr_guided:.2f} dB")
        print(f"  相关系数: {correlation_guided:.4f}")
        
        if vae_audio is not None:
            vae_audio = vae_audio[:min_len]
            if np.max(np.abs(vae_audio)) > 0:
                vae_audio = vae_audio / np.max(np.abs(vae_audio))
            
            mse_vae = np.mean((original_audio - vae_audio) ** 2)
            snr_vae = 10 * np.log10(np.var(original_audio) / mse_vae) if mse_vae > 0 else float('inf')
            correlation_vae = np.corrcoef(original_audio, vae_audio)[0, 1]
            
            print(f"\n纯VAE重建:")
            print(f"  MSE: {mse_vae:.6f}")
            print(f"  SNR: {snr_vae:.2f} dB")
            print(f"  相关系数: {correlation_vae:.4f}")
            
            print(f"\n改进程度:")
            print(f"  MSE改进: {((mse_vae - mse_guided) / mse_vae * 100):.1f}%")
            print(f"  SNR改进: {(snr_guided - snr_vae):.2f} dB")
            print(f"  相关系数改进: {(correlation_guided - correlation_vae):.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化版引导式Diffusion音频重建")
    parser.add_argument("input_path", type=str, help="输入音频文件路径")
    parser.add_argument("--output", "-o", type=str, default="guided_diffusion_simple_output", 
                       help="输出目录")
    parser.add_argument("--steps", "-s", type=int, default=50, 
                       help="Diffusion步数")
    parser.add_argument("--guidance", "-g", type=float, default=0.1, 
                       help="引导强度")
    parser.add_argument("--no-compare", action="store_true", 
                       help="不进行VAE对比")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="计算设备")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_path):
        print(f"❌ 输入文件不存在: {args.input_path}")
        return
    
    try:
        # 初始化重建器
        reconstructor = SimpleGuidedDiffusionReconstructor(device=args.device)
        
        # 执行重建
        guided_path, vae_path = reconstructor.reconstruct_audio(
            input_path=args.input_path,
            output_dir=args.output,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            compare_vae=not args.no_compare
        )
        
        print(f"\n✅ 重建完成!")
        print(f"   引导式重建: {guided_path}")
        if vae_path:
            print(f"   VAE重建: {vae_path}")
        
        print(f"\n💡 技术创新总结:")
        print(f"   - 结合了diffusion和VAE重建的优势")
        print(f"   - 在每个diffusion步骤中加入目标引导")
        print(f"   - 通过梯度下降优化重建质量")
        print(f"   - 可调节的引导强度和衰减策略")
        
    except Exception as e:
        print(f"❌ 重建失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
