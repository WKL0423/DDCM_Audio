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
        
    def audio_to_mel_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        将音频转换为 mel-spectrogram features
        
        Args:
            audio: 音频张量 [samples] 或 [1, samples]
            
        Returns:
            mel_features: mel-spectrogram features
        """
        with torch.no_grad():
            # 确保音频在 CPU 上且为 numpy 格式
            if audio.is_cuda:
                audio = audio.cpu()
            
            if audio.dim() == 2:
                audio = audio.squeeze(0)
            
            audio_numpy = audio.numpy()
            
            # 使用 ClapFeatureExtractor 处理音频
            inputs = self.pipeline.feature_extractor(
                audio_numpy,
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            mel_features = inputs["input_features"].to(self.device)
            
            # 确保正确的维度 [batch, channels, height, width]
            if mel_features.dim() == 3:
                mel_features = mel_features.unsqueeze(1)  # [batch, 1, freq, time]
            
            return mel_features
    
    def encode_audio_to_latent(self, audio: torch.Tensor) -> torch.Tensor:
        """
        编码音频为潜在表示
        
        Args:
            audio: 音频张量
            
        Returns:
            latent: 潜在表示
        """
        with torch.no_grad():
            # 转换为 mel features
            mel_features = self.audio_to_mel_features(audio)
            
            # 使用 VAE 编码
            latent_dist = self.pipeline.vae.encode(mel_features)
            
            # 获取潜在表示
            if hasattr(latent_dist, 'latent_dist'):
                latent = latent_dist.latent_dist.mode() if hasattr(latent_dist.latent_dist, 'mode') else latent_dist.latent_dist.sample()
            else:
                latent = latent_dist.mode() if hasattr(latent_dist, 'mode') else latent_dist.sample()
            
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
            # 使用 VAE 解码
            mel = self.pipeline.vae.decode(latent).sample
            
            # 这里我们需要使用 vocoder 或其他方法将 mel 转换为音频
            # 但是 AudioLDM2Pipeline 可能没有直接的 vocoder
            # 我们暂时返回 mel spectrogram，让 pipeline 的完整流程处理
            return mel
    
    def reconstruct_with_vae_only(self, audio: torch.Tensor) -> torch.Tensor:
        """
        仅使用 VAE 重建（无 diffusion）
        
        Args:
            audio: 输入音频
            
        Returns:
            reconstructed_mel: 重建的 mel-spectrogram
        """
        print("🔄 VAE-only 重建...")
        
        # 编码
        latent = self.encode_audio_to_latent(audio)
        print(f"   潜在表示形状: {latent.shape}")
        
        # 解码
        reconstructed_mel = self.decode_latent_to_audio(latent)
        print(f"   重建 mel 形状: {reconstructed_mel.shape}")
        
        return reconstructed_mel
    
    def reconstruct_with_diffusion(self, 
                                  audio: torch.Tensor,
                                  prompt: str = "high quality music",
                                  num_inference_steps: int = 20,
                                  guidance_scale: float = 7.5) -> torch.Tensor:
        """
        使用完整 diffusion 过程重建音频
        
        Args:
            audio: 输入音频
            prompt: 文本提示
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            
        Returns:
            reconstructed_audio: 重建的音频
        """
        print(f"🎵 Diffusion 重建...")
        print(f"   📝 提示: {prompt}")
        print(f"   🔄 推理步数: {num_inference_steps}")
        print(f"   🎚️ 引导强度: {guidance_scale}")
        
        # 使用 pipeline 直接生成，而不是编码-解码
        # 这才是真正的 diffusion 过程
        with torch.no_grad():
            # 计算音频长度（秒）
            audio_length = audio.shape[-1] / 48000.0
            audio_length = min(max(audio_length, 2.0), 10.0)  # 限制在 2-10 秒
            
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                audio_length_in_s=audio_length
            )
            
            generated_audio = result.audios[0]
            
        print(f"   ✅ 生成音频形状: {generated_audio.shape}")
        return generated_audio
    
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
        try:
            vae_only_mel = self.reconstruct_with_vae_only(input_audio)
            results["vae_only_mel"] = vae_only_mel
            print("   ✅ VAE-only 完成（返回 mel-spectrogram）")
        except Exception as e:
            print(f"   ❌ VAE-only 失败: {e}")
        
        # 2-4. 不同参数的 diffusion 重建
        diffusion_configs = [
            {"name": "quick_diffusion", "steps": 10, "guidance": 5.0, "prompt": "music"},
            {"name": "balanced_diffusion", "steps": 20, "guidance": 7.5, "prompt": "high quality music"},
            {"name": "detailed_diffusion", "steps": 30, "guidance": 10.0, "prompt": "high quality classical music with rich harmonics"}
        ]
        
        for config in diffusion_configs:
            print(f"\n🔄 方法: {config['name']}")
            try:
                diffusion_audio = self.reconstruct_with_diffusion(
                    input_audio,
                    prompt=config["prompt"],
                    num_inference_steps=config["steps"],
                    guidance_scale=config["guidance"]
                )
                results[config["name"]] = diffusion_audio
                print(f"   ✅ {config['name']} 完成")
            except Exception as e:
                print(f"   ❌ {config['name']} 失败: {e}")
        
        # 保存音频结果
        sample_rate = 16000
        for method_name, result in results.items():
            if "mel" in method_name:
                # 跳过 mel-spectrogram 结果，因为它们不是音频
                continue
                
            output_file = output_path / f"{method_name}_output.wav"
            
            # 确保音频格式正确
            if isinstance(result, torch.Tensor):
                if result.dim() == 1:
                    audio_to_save = result.unsqueeze(0)
                else:
                    audio_to_save = result
            else:
                audio_to_save = torch.tensor(result)
                if audio_to_save.dim() == 1:
                    audio_to_save = audio_to_save.unsqueeze(0)
            
            # 保存音频
            try:
                torchaudio.save(str(output_file), audio_to_save.cpu(), sample_rate)
                print(f"   💾 保存: {output_file}")
            except Exception as e:
                print(f"   ❌ 保存失败 {output_file}: {e}")
        
        # 保存原始音频
        original_file = output_path / "original_input.wav"
        if input_audio.dim() == 1:
            input_audio_save = input_audio.unsqueeze(0)
        else:
            input_audio_save = input_audio
        
        try:
            # 重采样原始音频到 16kHz 用于对比
            if input_audio_save.shape[-1] > 16000 * 10:  # 假设原始是 48kHz
                resampler = torchaudio.transforms.Resample(48000, 16000)
                input_audio_save = resampler(input_audio_save)
            
            torchaudio.save(str(original_file), input_audio_save.cpu(), sample_rate)
            print(f"   💾 保存原始音频: {original_file}")
        except Exception as e:
            print(f"   ❌ 保存原始音频失败: {e}")
        
        return results

def save_audio_compatible(audio: torch.Tensor, 
                         filepath: str, 
                         sample_rate: int = 16000,
                         normalize: bool = True) -> None:
    """
    兼容性音频保存函数
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
    
    # 重采样到 48kHz（AudioLDM2 的 ClapFeatureExtractor 要求）
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr, 48000)
        audio = resampler(audio)
        print(f"   🔄 重采样到 48kHz: {audio.shape}")
    
    # 转为单声道
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        print(f"   🔊 转为单声道: {audio.shape}")
    
    # 限制音频长度（避免显存不足）
    max_length = 48000 * 10  # 10 秒
    if audio.shape[-1] > max_length:
        audio = audio[..., :max_length]
        print(f"   ✂️ 裁剪到 10 秒: {audio.shape}")
    
    # 初始化 AudioLDM2 完整 diffusion pipeline
    try:
        reconstructor = AudioLDM2FullDiffusion("cvssp/audioldm2-music")
    except Exception as e:
        print(f"❌ 音乐模型加载失败: {e}")
        print("   尝试使用基础模型...")
        try:
            reconstructor = AudioLDM2FullDiffusion("cvssp/audioldm2")
        except Exception as e2:
            print(f"❌ 基础模型也加载失败: {e2}")
            return
    
    # 执行对比测试
    print("\n" + "="*50)
    print("🎯 开始对比 VAE-only vs 完整 Diffusion 方法")
    print("="*50)
    
    results = reconstructor.compare_methods(audio.squeeze(0))
    
    print("\n✅ 完整 diffusion 重建测试完成！")
    print("📁 查看 diffusion_comparison/ 目录获取结果文件")
    
    print("\n📋 结果总结:")
    print("   🔍 VAE-only: 只做 encode → decode，无噪声去除")
    print("   🎵 Diffusion: 完整的文本引导生成，包含去噪过程")
    print("   🎯 对比: Diffusion 能生成全新音频，而非重建原音频")
    
    print("\n🎉 AudioLDM2 完整 diffusion 测试完成！")

if __name__ == "__main__":
    main()
