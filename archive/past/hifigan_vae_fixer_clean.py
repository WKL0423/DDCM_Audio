"""
AudioLDM2 VAE + HiFiGAN 专项修复
================================

针对HiFiGAN维度不匹配问题的专门解决方案
目标: 成功集成神经vocoder，突破Griffin-Lim的92%信息损失瓶颈
"""

import torch
import torchaudio
import librosa
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional
from diffusers import AudioLDM2Pipeline
from transformers import SpeechT5HifiGan
import scipy.io.wavfile
import warnings

warnings.filterwarnings("ignore")

class HiFiGANVAEFixer:
    """HiFiGAN + VAE 专项修复器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎯 HiFiGAN + VAE 专项修复器")
        print(f"📱 设备: {self.device}")
        
        # 加载AudioLDM2
        print("📦 加载AudioLDM2...")
        self.pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-music",
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16
        ).to(self.device)
        
        self.vae = self.pipe.vae
        
        # 加载HiFiGAN
        print("🎤 加载HiFiGAN...")
        self.hifigan = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self.device)
        
        print("✅ 模型加载完成")
    
    def load_audio(self, audio_path: str, duration: float = 5.0) -> tuple:
        """加载音频"""
        audio, sr = librosa.load(audio_path, sr=16000, duration=duration)
        print(f"📊 音频: {len(audio)/sr:.2f}秒")
        return audio, sr
    
    def create_hifigan_compatible_mel(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """创建HiFiGAN兼容的80维mel频谱"""
        # HiFiGAN专用参数 (基于SpeechT5的标准配置)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,  # HiFiGAN标准
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            fmin=0,
            fmax=8000
        )
        
        # 转换为对数尺度
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 归一化到[-1, 1] (AudioLDM2 VAE标准)
        mel_spec = 2.0 * (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) - 1.0
        
        # 转换为tensor
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        print(f"🎼 创建80维mel: {mel_tensor.shape}")
        
        return mel_tensor.to(self.device)
    
    def vae_process(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """VAE编码解码"""
        with torch.no_grad():
            # 数据类型匹配
            if next(self.vae.parameters()).dtype == torch.float16:
                mel_tensor = mel_tensor.half()
            
            # 确保尺寸是4的倍数 (VAE要求)
            if mel_tensor.shape[-1] % 4 != 0:
                pad_length = 4 - (mel_tensor.shape[-1] % 4)
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_length))
            
            # VAE编码解码
            latent = self.vae.encode(mel_tensor).latent_dist.sample()
            reconstructed = self.vae.decode(latent).sample
            
            print(f"🔄 VAE重建: {mel_tensor.shape} → {latent.shape} → {reconstructed.shape}")
            return reconstructed.float()
    
    def hifigan_vocoder_fixed(self, mel_tensor: torch.Tensor) -> np.ndarray:
        """修复的HiFiGAN vocoder调用"""
        with torch.no_grad():
            # 从VAE输出格式转换为HiFiGAN输入格式
            mel_data = mel_tensor.squeeze()  # [1, 1, 80, time] → [80, time]
            
            if mel_data.dim() == 3:
                mel_data = mel_data.squeeze(0)  # [1, 80, time] → [80, time]
            
            print(f"🔧 HiFiGAN输入准备: {mel_data.shape}")
            
            # 确保是80维
            if mel_data.shape[0] != 80:
                print(f"⚠️ 维度调整: {mel_data.shape[0]} → 80")
                mel_data = torch.nn.functional.interpolate(
                    mel_data.unsqueeze(0).unsqueeze(0),
                    size=(80, mel_data.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            
            # 反归一化 (从[-1,1]恢复到mel尺度)
            mel_data = (mel_data + 1.0) / 2.0  # [0, 1]
            mel_data = mel_data * 80 - 80  # [-80, 0] dB范围
            
            # HiFiGAN需要batch维度: [batch, mel_bins, time]
            mel_input = mel_data.unsqueeze(0)
            print(f"🎤 HiFiGAN输入: {mel_input.shape}")
            
            # 使用HiFiGAN生成音频
            audio_tensor = self.hifigan(mel_input)
            audio = audio_tensor.squeeze().cpu().numpy()
            
            print(f"🎵 HiFiGAN输出: {audio.shape}")
            return audio
    
    def griffin_lim_baseline(self, mel_tensor: torch.Tensor, sr: int) -> np.ndarray:
        """Griffin-Lim基线对比"""
        mel_np = mel_tensor.squeeze().cpu().numpy()
        
        # 反归一化
        mel_np = (mel_np + 1.0) / 2.0
        mel_np = mel_np * 80 - 80  # 恢复dB范围
        mel_linear = librosa.db_to_power(mel_np)
        
        # Griffin-Lim重建
        audio = librosa.feature.inverse.mel_to_audio(
            mel_linear,
            sr=sr,
            n_iter=64,
            hop_length=256,
            win_length=1024,
            n_fft=1024
        )
        
        return audio
    
    def calculate_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """计算音频质量指标"""
        min_len = min(len(original), len(reconstructed))
        orig = original[:min_len]
        recon = reconstructed[:min_len]
        
        mse = np.mean((orig - recon) ** 2)
        snr = 10 * np.log10(np.mean(orig ** 2) / (mse + 1e-8))
        correlation = np.corrcoef(orig, recon)[0, 1] if len(orig) > 1 else 0.0
        
        return {
            'mse': float(mse),
            'snr': float(snr),
            'correlation': float(correlation if not np.isnan(correlation) else 0)
        }
    
    def comprehensive_comparison(self, audio_path: str) -> Dict:
        """全面对比测试"""
        print("\n🎯 HiFiGAN vs Griffin-Lim 全面对比测试")
        print("="*60)
        
        # 创建输出目录
        output_dir = Path("hifigan_vae_fix_test")
        output_dir.mkdir(exist_ok=True)
        
        # 加载音频
        audio, sr = self.load_audio(audio_path)
        
        # 保存原始音频
        audio_name = Path(audio_path).stem
        timestamp = int(time.time())
        original_path = output_dir / f"{audio_name}_original_{timestamp}.wav"
        scipy.io.wavfile.write(original_path, sr, (audio * 32767).astype(np.int16))
        
        results = {}
        
        print("\n🧪 开始测试流程...")
        
        try:
            # 创建80维mel (HiFiGAN兼容)
            mel_tensor = self.create_hifigan_compatible_mel(audio, sr)
            
            # VAE重建
            print("\n🔄 执行VAE重建...")
            vae_reconstructed_mel = self.vae_process(mel_tensor)
            
            # 测试1: Griffin-Lim重建
            print("\n📊 测试1: Griffin-Lim重建")
            try:
                griffin_audio = self.griffin_lim_baseline(vae_reconstructed_mel, sr)
                griffin_audio = griffin_audio / (np.max(np.abs(griffin_audio)) + 1e-8)
                griffin_metrics = self.calculate_metrics(audio, griffin_audio)
                
                griffin_file = output_dir / f"{audio_name}_griffin_lim_{timestamp}.wav"
                scipy.io.wavfile.write(griffin_file, sr, (griffin_audio * 32767).astype(np.int16))
                
                results['griffin_lim'] = {
                    'metrics': griffin_metrics,
                    'file': str(griffin_file)
                }
                
                print(f"   ✅ Griffin-Lim: SNR={griffin_metrics['snr']:.2f}dB, 相关性={griffin_metrics['correlation']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Griffin-Lim失败: {e}")
            
            # 测试2: HiFiGAN重建
            print("\n🎤 测试2: HiFiGAN重建")
            try:
                hifigan_audio = self.hifigan_vocoder_fixed(vae_reconstructed_mel)
                hifigan_audio = hifigan_audio / (np.max(np.abs(hifigan_audio)) + 1e-8)
                hifigan_metrics = self.calculate_metrics(audio, hifigan_audio)
                
                hifigan_file = output_dir / f"{audio_name}_hifigan_{timestamp}.wav"
                scipy.io.wavfile.write(hifigan_file, sr, (hifigan_audio * 32767).astype(np.int16))
                
                results['hifigan'] = {
                    'metrics': hifigan_metrics,
                    'file': str(hifigan_file)
                }
                
                print(f"   ✅ HiFiGAN: SNR={hifigan_metrics['snr']:.2f}dB, 相关性={hifigan_metrics['correlation']:.4f}")
                
            except Exception as e:
                print(f"   ❌ HiFiGAN失败: {e}")
            
        except Exception as e:
            print(f"❌ 整体流程失败: {e}")
        
        # 生成对比报告
        self._generate_comparison_report(results, output_dir)
        
        return results
    
    def _generate_comparison_report(self, results: Dict, output_dir: Path):
        """生成对比报告"""
        print("\n" + "="*60)
        print("🎯 HiFiGAN vs Griffin-Lim 对比结果")
        print("="*60)
        
        if not results:
            print("❌ 没有成功的测试结果")
            return
        
        # 基线Griffin-Lim结果
        if 'griffin_lim' in results:
            gl_metrics = results['griffin_lim']['metrics']
            print(f"\n📊 Griffin-Lim (基线):")
            print(f"   📈 SNR: {gl_metrics['snr']:.2f}dB")
            print(f"   🔗 相关性: {gl_metrics['correlation']:.4f}")
            print(f"   📁 文件: {Path(results['griffin_lim']['file']).name}")
        
        # HiFiGAN结果
        if 'hifigan' in results:
            hg_metrics = results['hifigan']['metrics']
            print(f"\n🎤 HiFiGAN (神经vocoder):")
            print(f"   📈 SNR: {hg_metrics['snr']:.2f}dB")
            print(f"   🔗 相关性: {hg_metrics['correlation']:.4f}")
            print(f"   📁 文件: {Path(results['hifigan']['file']).name}")
            
            # 对比分析
            if 'griffin_lim' in results:
                snr_improvement = hg_metrics['snr'] - gl_metrics['snr']
                corr_improvement = hg_metrics['correlation'] - gl_metrics['correlation']
                
                print(f"\n🚀 HiFiGAN vs Griffin-Lim 改进:")
                print(f"   📈 SNR改进: {snr_improvement:+.2f}dB")
                print(f"   🔗 相关性改进: {corr_improvement:+.4f}")
                
                if snr_improvement > 5:
                    print("   ✅ 显著改善！神经vocoder效果明显")
                elif snr_improvement > 1:
                    print("   ✅ 有所改善，神经vocoder有效")
                else:
                    print("   ⚠️ 改善有限，需要进一步优化")
        
        print(f"\n📁 所有结果保存在: {output_dir}")
        print("🎧 建议播放对比音频进行主观评估")
        
        # 下一步建议
        print(f"\n💡 优化建议:")
        if 'hifigan' in results and results['hifigan']['metrics']['snr'] > -5:
            print("   ✅ HiFiGAN集成成功，可探索更多神经vocoder")
        else:
            print("   ⚠️ 尝试其他神经vocoder模型")
        print("   📈 考虑端到端重建模型")
        print("   🔧 优化VAE压缩参数")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python hifigan_vae_fixer.py <音频文件>")
        
        # 列出音频文件
        audio_files = list(Path('.').glob('*.wav'))
        if audio_files:
            print("找到音频文件:")
            for i, file in enumerate(audio_files[:3], 1):
                print(f"{i}. {file.name}")
            
            try:
                choice = int(input("选择文件: "))
                audio_path = str(audio_files[choice-1])
            except:
                print("❌ 无效选择")
                return
        else:
            print("❌ 没有找到音频文件")
            return
    else:
        audio_path = sys.argv[1]
    
    # 运行修复测试
    fixer = HiFiGANVAEFixer()
    results = fixer.comprehensive_comparison(audio_path)
    
    print("\n✅ HiFiGAN专项修复测试完成!")

if __name__ == "__main__":
    main()
