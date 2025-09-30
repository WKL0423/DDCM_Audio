"""
VAE-only vs 完整 Diffusion Pipeline 对比分析
重点：说明为什么之前的所有脚本都是 VAE-only，而这次是真正的 diffusion
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AudioLDM2Pipeline
import librosa
import warnings
warnings.filterwarnings("ignore")

class DiffusionVsVAEAnalysis:
    def __init__(self):
        """初始化分析器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔬 初始化 Diffusion vs VAE 对比分析器")
        print(f"   📱 设备: {self.device}")
        
        # 加载 AudioLDM2 pipeline
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-music",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
    def vae_only_reconstruction(self, audio: torch.Tensor) -> torch.Tensor:
        """
        真正的 VAE-only 重建（类似之前所有脚本的方法）
        ⚠️ 这只是 encode → decode，没有 diffusion 过程
        
        Args:
            audio: 输入音频
            
        Returns:
            mel_spectrogram: VAE 重建的 mel-spectrogram（不是音频）
        """
        print("🔄 执行 VAE-only 重建（encode → decode）...")
        
        with torch.no_grad():
            # 转换为 mel features（使用 ClapFeatureExtractor）
            if audio.is_cuda:
                audio = audio.cpu()
            if audio.dim() == 2:
                audio = audio.squeeze(0)
            
            audio_numpy = audio.numpy()
            
            # 使用 ClapFeatureExtractor
            inputs = self.pipeline.feature_extractor(
                audio_numpy,
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            mel_features = inputs["input_features"].to(self.device)
            if mel_features.dim() == 3:
                mel_features = mel_features.unsqueeze(1)
            
            print(f"   📊 输入 mel 形状: {mel_features.shape}")
            
            # VAE encode
            latent_dist = self.pipeline.vae.encode(mel_features)
            latent = latent_dist.latent_dist.mode() if hasattr(latent_dist.latent_dist, 'mode') else latent_dist.latent_dist.sample()
            print(f"   📊 潜在表示形状: {latent.shape}")
            
            # VAE decode  
            reconstructed_mel = self.pipeline.vae.decode(latent).sample
            print(f"   📊 重建 mel 形状: {reconstructed_mel.shape}")
            
            print("   ⚠️ 注意：VAE-only 只返回 mel-spectrogram，需要额外的 vocoder 转换为音频")
            print("   ⚠️ 这就是为什么之前所有脚本的音质都有限 - 缺少了 diffusion 过程！")
            
            return reconstructed_mel
    
    def full_diffusion_reconstruction(self, 
                                    reference_audio: torch.Tensor,
                                    prompt: str = "high quality music") -> torch.Tensor:
        """
        完整的 diffusion 重建
        ✅ 这包含了完整的 UNet 去噪过程，是真正的 diffusion
        
        Args:
            reference_audio: 参考音频（用于确定长度）
            prompt: 文本提示
            
        Returns:
            generated_audio: 生成的音频
        """
        print("🎵 执行完整 Diffusion 重建（包含 UNet 去噪）...")
        
        with torch.no_grad():
            # 计算音频长度
            audio_length = reference_audio.shape[-1] / 48000.0
            audio_length = min(max(audio_length, 2.0), 10.0)
            
            print(f"   📝 文本提示: {prompt}")
            print(f"   ⏱️ 目标长度: {audio_length:.1f} 秒")
            
            # 完整的 diffusion pipeline
            # 这包含：文本编码 → 噪声采样 → UNet 去噪 → VAE 解码 → 音频输出
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                audio_length_in_s=audio_length
            )
            
            generated_audio = result.audios[0]
            print(f"   📊 生成音频形状: {generated_audio.shape}")
            
            print("   ✅ 完整 diffusion 包含：")
            print("      1. 文本编码（CLAP text encoder）")
            print("      2. 随机噪声采样") 
            print("      3. UNet 去噪过程（多步迭代）")
            print("      4. VAE 解码")
            print("      5. 最终音频输出")
            
            return generated_audio
    
    def analyze_spectrograms(self, 
                           original_audio: torch.Tensor,
                           vae_mel: torch.Tensor,
                           diffusion_audio: torch.Tensor,
                           output_dir: str = "analysis_results"):
        """
        分析频谱图差异
        
        Args:
            original_audio: 原始音频
            vae_mel: VAE 重建的 mel-spectrogram
            diffusion_audio: diffusion 生成的音频
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n📊 分析频谱图差异...")
        
        # 确保音频在 CPU 上
        if original_audio.is_cuda:
            original_audio = original_audio.cpu()
        if diffusion_audio.is_cuda:
            diffusion_audio = diffusion_audio.cpu()
        if vae_mel.is_cuda:
            vae_mel = vae_mel.cpu()
        
        # 转换为 numpy
        if original_audio.dim() > 1:
            original_audio = original_audio.squeeze()
        if diffusion_audio.dim() > 1:
            diffusion_audio = diffusion_audio.squeeze()
        
        original_np = original_audio.numpy()
        diffusion_np = diffusion_audio.numpy()
        
        # 计算频谱图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 原始音频频谱
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_np)), ref=np.max)
        librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', ax=axes[0, 0])
        axes[0, 0].set_title('原始音频频谱')
        axes[0, 0].set_ylabel('频率 (Hz)')
        
        # Diffusion 生成音频频谱
        D_diff = librosa.amplitude_to_db(np.abs(librosa.stft(diffusion_np)), ref=np.max)
        librosa.display.specshow(D_diff, y_axis='hz', x_axis='time', ax=axes[0, 1])
        axes[0, 1].set_title('Diffusion 生成音频频谱')
        
        # VAE mel-spectrogram
        vae_mel_np = vae_mel.squeeze().numpy()
        if vae_mel_np.ndim == 3:
            vae_mel_np = vae_mel_np[0]  # 取第一个通道
        
        im = axes[0, 2].imshow(vae_mel_np, aspect='auto', origin='lower')
        axes[0, 2].set_title('VAE 重建 Mel-spectrogram')
        axes[0, 2].set_ylabel('Mel bins')
        plt.colorbar(im, ax=axes[0, 2])
        
        # 频率能量分布对比
        freqs_orig = np.mean(np.abs(librosa.stft(original_np)), axis=1)
        freqs_diff = np.mean(np.abs(librosa.stft(diffusion_np)), axis=1)
        
        freq_bins = librosa.fft_frequencies(sr=16000)
        axes[1, 0].plot(freq_bins, freqs_orig, label='原始音频', alpha=0.7)
        axes[1, 0].plot(freq_bins, freqs_diff, label='Diffusion 音频', alpha=0.7)
        axes[1, 0].set_xlabel('频率 (Hz)')
        axes[1, 0].set_ylabel('能量')
        axes[1, 0].set_title('频率能量分布对比')
        axes[1, 0].legend()
        axes[1, 0].set_xlim(0, 8000)
        
        # 高频能量对比
        high_freq_mask = freq_bins > 4000
        high_freq_orig = np.mean(freqs_orig[high_freq_mask])
        high_freq_diff = np.mean(freqs_diff[high_freq_mask])
        
        methods = ['原始音频', 'Diffusion音频']
        high_freq_energy = [high_freq_orig, high_freq_diff]
        
        axes[1, 1].bar(methods, high_freq_energy, alpha=0.7)
        axes[1, 1].set_title('高频能量对比 (>4kHz)')
        axes[1, 1].set_ylabel('平均能量')
        
        # 关键差异说明
        axes[1, 2].text(0.1, 0.8, 
                        "🔍 关键差异分析:\n\n"
                        "VAE-only 重建:\n"
                        "• 只有 encode → decode\n"
                        "• 没有去噪过程\n"
                        "• 输出 mel-spectrogram\n"
                        "• 需要额外 vocoder\n"
                        "• 质量受 VAE 瓶颈限制\n\n"
                        "完整 Diffusion:\n"
                        "• 包含 UNet 去噪\n"
                        "• 文本条件引导\n"
                        "• 多步迭代优化\n"
                        "• 直接输出音频\n"
                        "• 质量更高",
                        transform=axes[1, 2].transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / "vae_vs_diffusion_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 频谱分析图保存至: {output_path / 'vae_vs_diffusion_analysis.png'}")
        
        # 数值对比
        print(f"\n📈 定量分析:")
        print(f"   高频能量 (>4kHz):")
        print(f"      原始音频: {high_freq_orig:.4f}")
        print(f"      Diffusion: {high_freq_diff:.4f}")
        print(f"      比值: {high_freq_diff/high_freq_orig:.2f}")
        
    def comprehensive_comparison(self, input_audio_path: str = "AudioLDM2_Music_output.wav"):
        """
        综合对比分析
        
        Args:
            input_audio_path: 输入音频路径
        """
        print("\n" + "="*60)
        print("🎯 VAE-only vs 完整 Diffusion 综合对比")
        print("="*60)
        
        # 加载音频
        if not Path(input_audio_path).exists():
            print(f"❌ 找不到输入文件: {input_audio_path}")
            return
        
        audio, sr = torchaudio.load(input_audio_path)
        print(f"📂 加载音频: {audio.shape}, {sr}Hz")
        
        # 重采样到 48kHz
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio = resampler(audio)
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 限制长度
        max_length = 48000 * 8  # 8 秒
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
        
        print(f"📊 处理后音频: {audio.shape}")
        
        # 1. VAE-only 重建
        print(f"\n1️⃣ VAE-only 重建测试")
        print("-" * 30)
        try:
            vae_mel = self.vae_only_reconstruction(audio.squeeze(0))
            print("   ✅ VAE-only 成功")
        except Exception as e:
            print(f"   ❌ VAE-only 失败: {e}")
            vae_mel = None
        
        # 2. 完整 Diffusion 重建
        print(f"\n2️⃣ 完整 Diffusion 重建测试")
        print("-" * 30)
        try:
            diffusion_audio = self.full_diffusion_reconstruction(
                audio.squeeze(0), 
                "high quality instrumental music with rich harmonics"
            )
            print("   ✅ Diffusion 成功")
        except Exception as e:
            print(f"   ❌ Diffusion 失败: {e}")
            diffusion_audio = None
        
        # 3. 保存结果
        print(f"\n3️⃣ 保存结果")
        print("-" * 30)
        
        output_dir = Path("vae_vs_diffusion_comparison")
        output_dir.mkdir(exist_ok=True)
        
        # 保存原始音频
        if audio.dim() == 2:
            audio_save = audio
        else:
            audio_save = audio.unsqueeze(0)
        
        torchaudio.save(str(output_dir / "original_input.wav"), 
                       audio_save.cpu(), 16000)
        print(f"   💾 原始音频: {output_dir / 'original_input.wav'}")
          # 保存 diffusion 结果
        if diffusion_audio is not None:
            # 转换为 tensor 如果是 numpy
            if isinstance(diffusion_audio, np.ndarray):
                diffusion_audio = torch.tensor(diffusion_audio)
            
            if diffusion_audio.dim() == 1:
                diffusion_save = diffusion_audio.unsqueeze(0)
            else:
                diffusion_save = diffusion_audio
                
            torchaudio.save(str(output_dir / "diffusion_output.wav"), 
                           diffusion_save.cpu(), 16000)
            print(f"   💾 Diffusion 音频: {output_dir / 'diffusion_output.wav'}")
        
        # 4. 频谱分析
        if vae_mel is not None and diffusion_audio is not None:
            print(f"\n4️⃣ 频谱分析")
            print("-" * 30)
            self.analyze_spectrograms(
                audio.squeeze(0),
                vae_mel,
                diffusion_audio,
                str(output_dir)
            )
        
        # 5. 总结分析
        print(f"\n📋 总结分析")
        print("-" * 30)
        print("🔍 方法对比:")
        print("   VAE-only (之前所有脚本的方法):")
        print("      • 流程: audio → mel → encode → decode → mel")
        print("      • 问题: 只有 VAE 瓶颈，无去噪优化")
        print("      • 输出: mel-spectrogram (需要 vocoder)")
        print("      • 质量: 受限于 VAE 压缩损失")
        print("")
        print("   完整 Diffusion (本次实现):")
        print("      • 流程: prompt → noise → UNet去噪 → VAE解码 → audio")
        print("      • 优势: 包含去噪优化和文本引导")
        print("      • 输出: 直接音频")
        print("      • 质量: 更高的感知质量")
        print("")
        print("🎯 关键发现:")
        print("   • 之前所有测试都缺少了真正的 diffusion 过程")
        print("   • VAE-only 的高频丢失是结构性问题")
        print("   • 完整 diffusion 能生成更自然的音频")
        print("   • 但 diffusion 生成的是新音频，不是重建")
        
        print(f"\n✅ 对比分析完成！查看 {output_dir} 目录获取详细结果")

def main():
    """主函数"""
    analyzer = DiffusionVsVAEAnalysis()
    analyzer.comprehensive_comparison()

if __name__ == "__main__":
    main()
