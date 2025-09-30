"""
AudioLDM2 完整 Diffusion Pipeline 重建
包含真正的 diffusion 去噪过程，而不仅仅是 VAE encode/decode
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, List
from diffusers import AudioLDM2Pipeline, DDIMScheduler, PNDMScheduler
import warnings
warnings.filterwarnings("ignore")

class AudioLDM2FullDiffusion:
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        """
        初始化 AudioLDM2 完整 diffusion pipeline
        
        Args:
            model_name: 模型名称
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎵 初始化 AudioLDM2 完整 diffusion pipeline...")
        print(f"   📱 设备: {self.device}")
        
        # 加载完整的 AudioLDM2 pipeline
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        # 设置不同的调度器选项
        self.schedulers = {
            "ddim": DDIMScheduler.from_config(self.pipeline.scheduler.config),
            "pndm": PNDMScheduler.from_config(self.pipeline.scheduler.config),
            "default": self.pipeline.scheduler
        }
        
        print(f"   ✅ 模型加载完成: {model_name}")
        print(f"   📊 VAE channels: {self.pipeline.vae.config.latent_channels}")
        print(f"   🔄 调度器选项: {list(self.schedulers.keys())}")
          def encode_audio_to_latent(self, audio: torch.Tensor) -> torch.Tensor:
        """
        编码音频为潜在表示
        
        Args:
            audio: 音频张量 [batch_size, channels, samples]
            
        Returns:
            latent: 潜在表示 [batch_size, channels, height, width]
        """
        with torch.no_grad():
            # 确保音频格式正确
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # [1, samples]
            elif audio.dim() == 2 and audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)  # 转为单声道
            
            # 重采样到 48kHz（ClapFeatureExtractor 要求）
            if audio.shape[-1] != int(48000 * 10):  # 假设 10 秒音频
                target_length = min(48000 * 10, audio.shape[-1])
                audio = audio[..., :target_length]
            
            # 使用 feature_extractor 转换为 mel-spectrogram
            inputs = self.pipeline.feature_extractor(
                audio.squeeze(0).cpu().numpy(),
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            mel = inputs["input_features"].to(self.device)
            if mel.dim() == 3:
                mel = mel.unsqueeze(1)  # [batch, 1, freq, time]
            
            # 编码为潜在表示
            latent_dist = self.pipeline.vae.encode(mel)
            
            # 采样获取确定性潜在表示
            latent = latent_dist.latent_dist.mode() if hasattr(latent_dist.latent_dist, 'mode') else latent_dist.latent_dist.sample()
            
            return latent
    
    def add_noise_to_latent(self, latent: torch.Tensor, noise_level: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        向潜在表示添加噪声（模拟 diffusion 前向过程）
        
        Args:
            latent: 原始潜在表示
            noise_level: 噪声水平 (0-1)
            
        Returns:
            noisy_latent: 加噪后的潜在表示
            noise: 添加的噪声
        """
        # 生成高斯噪声
        noise = torch.randn_like(latent)
        
        # 使用调度器的前向过程添加噪声
        # 选择噪声步数
        timestep = int(noise_level * self.pipeline.scheduler.config.num_train_timesteps)
        timesteps = torch.tensor([timestep], device=latent.device)
        
        # 添加噪声
        noisy_latent = self.pipeline.scheduler.add_noise(latent, noise, timesteps)
        
        return noisy_latent, noise
    
    def denoise_latent(self, 
                      noisy_latent: torch.Tensor, 
                      prompt: str = "high quality music", 
                      num_inference_steps: int = 20,
                      guidance_scale: float = 7.5,
                      scheduler_name: str = "ddim") -> torch.Tensor:
        """
        使用 diffusion 去噪潜在表示
        
        Args:
            noisy_latent: 加噪的潜在表示
            prompt: 文本提示
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            scheduler_name: 调度器名称
            
        Returns:
            denoised_latent: 去噪后的潜在表示
        """
        # 设置调度器
        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.schedulers[scheduler_name]
        
        # 设置推理步数
        self.pipeline.scheduler.set_timesteps(num_inference_steps)
        
        # 编码文本提示
        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(
                self.pipeline.tokenizer(
                    prompt,
                    max_length=self.pipeline.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(self.device)
            )[0]
            
            # 无条件嵌入（用于classifier-free guidance）
            uncond_embeddings = self.pipeline.text_encoder(
                self.pipeline.tokenizer(
                    "",
                    max_length=self.pipeline.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(self.device)
            )[0]
            
            # 合并条件和无条件嵌入
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # 去噪循环
        latent = noisy_latent
        
        for i, t in enumerate(self.pipeline.scheduler.timesteps):
            # 扩展潜在表示用于 classifier-free guidance
            latent_model_input = torch.cat([latent] * 2)
            
            # 预测噪声
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # 执行 classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 去噪步骤
            latent = self.pipeline.scheduler.step(noise_pred, t, latent).prev_sample
        
        # 恢复原始调度器
        self.pipeline.scheduler = original_scheduler
        
        return latent
    
    def decode_latent_to_audio(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码潜在表示为音频
        
        Args:
            latent: 潜在表示
            
        Returns:
            audio: 音频张量
        """
        with torch.no_grad():
            # 解码为 mel-spectrogram
            mel = self.pipeline.vae.decode(latent).sample
            
            # 转换为音频
            audio = self.pipeline.mel_spectrogram_extractor.mel_spectrogram_to_waveform(mel)
            
            return audio
    
    def reconstruct_with_diffusion(self, 
                                  input_audio: torch.Tensor,
                                  prompt: str = "high quality music",
                                  noise_level: float = 0.8,
                                  num_inference_steps: int = 20,
                                  guidance_scale: float = 7.5,
                                  scheduler_name: str = "ddim") -> Tuple[torch.Tensor, dict]:
        """
        使用完整 diffusion 过程重建音频
        
        Args:
            input_audio: 输入音频
            prompt: 文本提示
            noise_level: 噪声水平
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            scheduler_name: 调度器名称
            
        Returns:
            reconstructed_audio: 重建的音频
            info: 处理信息
        """
        print(f"🎵 开始 AudioLDM2 完整 diffusion 重建...")
        print(f"   📝 提示: {prompt}")
        print(f"   🔊 噪声水平: {noise_level}")
        print(f"   🔄 推理步数: {num_inference_steps}")
        print(f"   🎚️ 引导强度: {guidance_scale}")
        print(f"   ⏰ 调度器: {scheduler_name}")
        
        # 1. 编码音频为潜在表示
        print("   1️⃣ 编码音频为潜在表示...")
        original_latent = self.encode_audio_to_latent(input_audio)
        print(f"      潜在表示形状: {original_latent.shape}")
        
        # 2. 添加噪声
        print("   2️⃣ 添加噪声...")
        noisy_latent, noise = self.add_noise_to_latent(original_latent, noise_level)
        print(f"      噪声强度: {noise.std().item():.4f}")
        
        # 3. Diffusion 去噪
        print("   3️⃣ Diffusion 去噪...")
        denoised_latent = self.denoise_latent(
            noisy_latent, prompt, num_inference_steps, guidance_scale, scheduler_name
        )
        print(f"      去噪完成")
        
        # 4. 解码为音频
        print("   4️⃣ 解码为音频...")
        reconstructed_audio = self.decode_latent_to_audio(denoised_latent)
        print(f"      重建音频形状: {reconstructed_audio.shape}")
        
        # 处理信息
        info = {
            "prompt": prompt,
            "noise_level": noise_level,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler_name,
            "original_latent_shape": original_latent.shape,
            "reconstructed_audio_shape": reconstructed_audio.shape,
            "noise_std": noise.std().item()
        }
        
        print("   ✅ 完整 diffusion 重建完成！")
        return reconstructed_audio, info
    
    def compare_methods(self, 
                       input_audio: torch.Tensor,
                       output_dir: str = "diffusion_comparison") -> dict:
        """
        对比不同重建方法
        
        Args:
            input_audio: 输入音频
            output_dir: 输出目录
            
        Returns:
            results: 对比结果
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        # 1. VAE-only 重建（无 diffusion）
        print("\n🔄 方法 1: VAE-only 重建（无 diffusion）")
        latent = self.encode_audio_to_latent(input_audio)
        vae_only_audio = self.decode_latent_to_audio(latent)
        results["vae_only"] = vae_only_audio
        
        # 2. 低噪声 diffusion 重建
        print("\n🔄 方法 2: 低噪声 diffusion 重建")
        low_noise_audio, _ = self.reconstruct_with_diffusion(
            input_audio, 
            prompt="high quality music",
            noise_level=0.3,
            num_inference_steps=10
        )
        results["low_noise_diffusion"] = low_noise_audio
        
        # 3. 中等噪声 diffusion 重建
        print("\n🔄 方法 3: 中等噪声 diffusion 重建")
        medium_noise_audio, _ = self.reconstruct_with_diffusion(
            input_audio,
            prompt="high quality music",
            noise_level=0.6,
            num_inference_steps=20
        )
        results["medium_noise_diffusion"] = medium_noise_audio
        
        # 4. 高噪声 diffusion 重建
        print("\n🔄 方法 4: 高噪声 diffusion 重建")
        high_noise_audio, _ = self.reconstruct_with_diffusion(
            input_audio,
            prompt="high quality music",
            noise_level=0.9,
            num_inference_steps=30
        )
        results["high_noise_diffusion"] = high_noise_audio
        
        # 保存所有结果
        sample_rate = 16000
        for method_name, audio in results.items():
            output_file = output_path / f"{method_name}_output.wav"
            if audio.dim() == 3:
                audio = audio.squeeze(0)
            torchaudio.save(str(output_file), audio.cpu(), sample_rate)
            print(f"   💾 保存: {output_file}")
        
        # 保存原始音频
        original_file = output_path / "original_input.wav"
        if input_audio.dim() == 1:
            input_audio = input_audio.unsqueeze(0)
        torchaudio.save(str(original_file), input_audio.cpu(), sample_rate)
        print(f"   💾 保存原始音频: {original_file}")
        
        return results

def save_audio_compatible(audio: torch.Tensor, 
                         filepath: str, 
                         sample_rate: int = 16000,
                         normalize: bool = True) -> None:
    """
    兼容性音频保存函数
    
    Args:
        audio: 音频张量
        filepath: 保存路径
        sample_rate: 采样率
        normalize: 是否归一化
    """
    # 确保音频在 CPU 上
    if audio.is_cuda:
        audio = audio.cpu()
    
    # 处理音频维度
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # 归一化
    if normalize:
        audio = audio / (audio.abs().max() + 1e-8)
        audio = audio * 0.8  # 防止削波
    
    # 保存
    torchaudio.save(filepath, audio, sample_rate)

def main():
    """主函数：处理 AudioLDM2_Music_output.wav"""
    
    # 输入文件
    input_file = "AudioLDM2_Music_output.wav"
    
    if not Path(input_file).exists():
        print(f"❌ 找不到输入文件: {input_file}")
        return
    
    print(f"🎵 处理文件: {input_file}")
    
    # 加载音频
    audio, sr = torchaudio.load(input_file)
    print(f"   📊 原始音频: {audio.shape}, 采样率: {sr}")
    
    # 重采样到 16kHz（AudioLDM2 标准）
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
        print(f"   🔄 重采样到 16kHz: {audio.shape}")
    
    # 转为单声道
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        print(f"   🔊 转为单声道: {audio.shape}")
    
    # 初始化 AudioLDM2 完整 diffusion pipeline
    try:
        reconstructor = AudioLDM2FullDiffusion("cvssp/audioldm2-music")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("   尝试使用基础模型...")
        try:
            reconstructor = AudioLDM2FullDiffusion("cvssp/audioldm2")
        except Exception as e2:
            print(f"❌ 基础模型也加载失败: {e2}")
            return
    
    # 执行对比测试
    print("\n" + "="*50)
    print("🎯 开始对比不同重建方法")
    print("="*50)
    
    results = reconstructor.compare_methods(audio.squeeze(0))
    
    # 计算基础指标
    def calculate_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
        """计算重建质量指标"""
        # 确保形状一致
        min_len = min(original.shape[-1], reconstructed.shape[-1])
        orig = original[..., :min_len]
        recon = reconstructed[..., :min_len]
        
        # 计算指标
        mse = torch.mean((orig - recon) ** 2).item()
        snr = 10 * torch.log10(torch.var(orig) / (mse + 1e-8)).item()
        correlation = torch.corrcoef(torch.stack([orig.flatten(), recon.flatten()]))[0, 1].item()
        
        return {
            "mse": mse,
            "snr": snr,
            "correlation": correlation
        }
    
    # 分析结果
    print("\n📊 重建质量分析")
    print("-" * 40)
    
    original_audio = audio.squeeze(0)
    for method_name, reconstructed_audio in results.items():
        if reconstructed_audio.dim() > 1:
            reconstructed_audio = reconstructed_audio.squeeze(0)
        
        metrics = calculate_metrics(original_audio, reconstructed_audio)
        
        print(f"\n🔍 {method_name.replace('_', ' ').title()}:")
        print(f"   SNR: {metrics['snr']:.2f} dB")
        print(f"   相关性: {metrics['correlation']:.4f}")
        print(f"   MSE: {metrics['mse']:.6f}")
    
    print("\n✅ 完整 diffusion 重建测试完成！")
    print("📁 查看 diffusion_comparison/ 目录获取结果文件")
    
    # 额外的单独测试
    print("\n🎯 额外测试：自定义 diffusion 重建")
    custom_audio, custom_info = reconstructor.reconstruct_with_diffusion(
        audio.squeeze(0),
        prompt="high quality classical music with rich harmonics",
        noise_level=0.7,
        num_inference_steps=25,
        guidance_scale=9.0,
        scheduler_name="ddim"
    )
    
    # 保存自定义结果
    save_audio_compatible(custom_audio, "custom_diffusion_output.wav")
    
    print(f"\n📋 自定义重建信息:")
    for key, value in custom_info.items():
        print(f"   {key}: {value}")
    
    print("\n🎉 AudioLDM2 完整 diffusion 重建测试完成！")

if __name__ == "__main__":
    main()
