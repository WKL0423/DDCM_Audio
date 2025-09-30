"""
AudioLDM2 VAE高质量音频重建 - 修复版本
============================================

基于前期诊断结果的针对性解决方案:
- Griffin-Lim导致92%信息损失 → 使用神经vocoder
- VAE压缩15%损失 → 优化参数配置
- Mel变换14%损失 → 多分辨率对比测试

目标: 显著提升重建质量，解决"能听出联系但质量一般"问题
"""

import torch
import torchaudio
import librosa
import numpy as np
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from diffusers import AudioLDM2Pipeline
from transformers import SpeechT5HifiGan
import scipy.io.wavfile
from datetime import datetime

warnings.filterwarnings("ignore")

class VAEReconstructionFixer:
    """VAE重建质量修复器 - 专门解决质量瓶颈"""
    
    def __init__(self, model_name: str = "cvssp/audioldm2-music"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 VAE重建质量修复器启动")
        print(f"📱 设备: {self.device}")
        
        # 加载AudioLDM2
        print("📦 加载AudioLDM2模型...")
        self.pipe = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16
        ).to(self.device)
        
        self.vae = self.pipe.vae
        
        # 尝试加载高质量vocoder
        self.hifigan_vocoder = None
        try:
            print("🎤 加载HiFiGAN vocoder...")
            self.hifigan_vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan"
            ).to(self.device)
            print("✅ HiFiGAN加载成功")
        except Exception as e:
            print(f"❌ HiFiGAN加载失败: {e}")
        
        print("✅ 修复器初始化完成")
    
    def load_and_preprocess_audio(self, audio_path: str, max_duration: float = 5.0) -> Tuple[np.ndarray, int]:
        """加载和预处理音频"""
        audio, sr = librosa.load(audio_path, sr=16000, duration=max_duration)
        print(f"📊 音频: {len(audio)/sr:.2f}秒, {len(audio)}样本")
        return audio, sr
    
    def create_mel_spectrogram(self, audio: np.ndarray, sr: int, 
                              n_mels: int = 64, enhanced: bool = False) -> torch.Tensor:
        """创建mel频谱图"""
        if enhanced:
            # 高质量配置
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels, n_fft=2048, 
                hop_length=256, win_length=1024, fmin=0, fmax=8000
            )
        else:
            # 标准配置
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels, n_fft=1024,
                hop_length=256, win_length=1024, fmin=0, fmax=8000
            )
        
        # 转换为对数尺度并归一化
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = 2.0 * (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) - 1.0
        
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        print(f"🎼 Mel频谱: {mel_tensor.shape}")
        return mel_tensor.to(self.device)
    
    def vae_reconstruct(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """VAE重建过程"""
        with torch.no_grad():
            # 数据类型匹配
            if next(self.vae.parameters()).dtype == torch.float16:
                mel_tensor = mel_tensor.half()
            
            # 确保尺寸匹配VAE要求
            if mel_tensor.shape[-1] % 4 != 0:
                pad_length = 4 - (mel_tensor.shape[-1] % 4)
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_length))
            
            # VAE编码解码
            start_time = time.time()
            latent = self.vae.encode(mel_tensor).latent_dist.sample()
            reconstructed = self.vae.decode(latent).sample
            vae_time = time.time() - start_time
            
            print(f"🔄 VAE重建: {vae_time:.3f}s, 潜在空间: {latent.shape}")
            return reconstructed.float()
    
    def griffin_lim_improved(self, mel_tensor: torch.Tensor, sr: int = 16000) -> np.ndarray:
        """改进的Griffin-Lim算法"""
        mel_np = mel_tensor.squeeze().cpu().numpy()
        
        # 反归一化
        mel_np = (mel_np + 1.0) / 2.0
        mel_min, mel_max = -80, 0
        mel_np = mel_np * (mel_max - mel_min) + mel_min
        mel_linear = librosa.db_to_power(mel_np)
        
        # 改进的Griffin-Lim参数
        audio = librosa.feature.inverse.mel_to_audio(
            mel_linear, sr=sr, n_iter=64, hop_length=256,
            win_length=1024, n_fft=1024, fmin=0, fmax=8000
        )
        
        return audio
    
    def hifigan_reconstruct(self, mel_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """使用HiFiGAN重建音频"""
        if self.hifigan_vocoder is None:
            return None
        
        try:
            with torch.no_grad():
                mel_input = mel_tensor.squeeze()
                if mel_input.dim() == 3:
                    mel_input = mel_input.squeeze(0)
                
                # 转换到80维 (HiFiGAN标准)
                if mel_input.shape[0] != 80:
                    mel_input = torch.nn.functional.interpolate(
                        mel_input.unsqueeze(0).unsqueeze(0),
                        size=(80, mel_input.shape[1]),
                        mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                
                # HiFiGAN期望的输入格式: [batch, mel_bins, time]
                mel_input = mel_input.unsqueeze(0)
                
                audio_tensor = self.hifigan_vocoder(mel_input)
                audio = audio_tensor.squeeze().cpu().numpy()
                
                return audio
                
        except Exception as e:
            print(f"❌ HiFiGAN失败: {e}")
            return None
    
    def calculate_quality_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """计算质量指标"""
        min_len = min(len(original), len(reconstructed))
        orig = original[:min_len]
        recon = reconstructed[:min_len]
        
        mse = np.mean((orig - recon) ** 2)
        snr = 10 * np.log10(np.mean(orig ** 2) / (mse + 1e-8))
        correlation = np.corrcoef(orig, recon)[0, 1]
        
        return {
            'mse': float(mse),
            'snr': float(snr),
            'correlation': float(correlation if not np.isnan(correlation) else 0)
        }
    
    def comprehensive_test(self, audio_path: str) -> Dict:
        """全面测试不同重建策略"""
        print("🔧 开始VAE重建质量修复测试")
        print("="*50)
        
        # 创建输出目录
        output_dir = Path("vae_quality_fix_test")
        output_dir.mkdir(exist_ok=True)
        
        # 加载音频
        audio, sr = self.load_and_preprocess_audio(audio_path)
        
        # 保存原始音频
        audio_name = Path(audio_path).stem
        timestamp = int(time.time())
        original_path = output_dir / f"{audio_name}_original_{timestamp}.wav"
        scipy.io.wavfile.write(original_path, sr, (audio * 32767).astype(np.int16))
        
        results = {}
        
        # 测试策略配置
        test_configs = [
            {"name": "baseline_64mel_griffin", "n_mels": 64, "enhanced": False, "vocoder": "griffin_lim"},
            {"name": "enhanced_64mel_griffin", "n_mels": 64, "enhanced": True, "vocoder": "griffin_lim"},
            {"name": "baseline_80mel_griffin", "n_mels": 80, "enhanced": False, "vocoder": "griffin_lim"},
            {"name": "enhanced_80mel_griffin", "n_mels": 80, "enhanced": True, "vocoder": "griffin_lim"},
        ]
        
        # 如果HiFiGAN可用，添加神经vocoder测试
        if self.hifigan_vocoder:
            test_configs.extend([
                {"name": "baseline_80mel_hifigan", "n_mels": 80, "enhanced": False, "vocoder": "hifigan"},
                {"name": "enhanced_80mel_hifigan", "n_mels": 80, "enhanced": True, "vocoder": "hifigan"},
            ])
        
        print(f"🧪 测试 {len(test_configs)} 种配置...")
        
        for i, config in enumerate(test_configs):
            print(f"\n📊 配置 {i+1}/{len(test_configs)}: {config['name']}")
            
            try:
                # 创建mel频谱
                mel_tensor = self.create_mel_spectrogram(
                    audio, sr, config['n_mels'], config['enhanced']
                )
                
                # VAE重建
                reconstructed_mel = self.vae_reconstruct(mel_tensor)
                
                # 使用指定的vocoder
                if config['vocoder'] == 'griffin_lim':
                    final_audio = self.griffin_lim_improved(reconstructed_mel, sr)
                elif config['vocoder'] == 'hifigan':
                    final_audio = self.hifigan_reconstruct(reconstructed_mel)
                    if final_audio is None:
                        print("   ❌ HiFiGAN失败，跳过")
                        continue
                
                # 后处理: 归一化
                final_audio = final_audio / (np.max(np.abs(final_audio)) + 1e-8)
                
                # 计算指标
                metrics = self.calculate_quality_metrics(audio, final_audio)
                
                # 保存结果
                output_file = output_dir / f"{audio_name}_{config['name']}_{timestamp}.wav"
                scipy.io.wavfile.write(output_file, sr, (final_audio * 32767).astype(np.int16))
                
                results[config['name']] = {
                    'metrics': metrics,
                    'file': str(output_file),
                    'config': config
                }
                
                print(f"   ✅ SNR: {metrics['snr']:.2f}dB, 相关性: {metrics['correlation']:.4f}")
                
            except Exception as e:
                print(f"   ❌ 配置失败: {e}")
                continue
        
        # 生成分析报告
        self._generate_analysis_report(results, audio_path, output_dir)
        
        return results
    
    def _generate_analysis_report(self, results: Dict, audio_path: str, output_dir: Path):
        """生成详细分析报告"""
        print("\n" + "="*50)
        print("🔧 VAE重建质量修复测试结果")
        print("="*50)
        
        if not results:
            print("❌ 没有成功的测试结果")
            return
        
        # 按SNR排序
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['metrics']['snr'],
            reverse=True
        )
        
        print(f"\n🏆 质量排名:")
        baseline_snr = None
        for i, (config_name, data) in enumerate(sorted_results):
            metrics = data['metrics']
            config = data['config']
            
            if i == 0:
                baseline_snr = metrics['snr']
            
            improvement = metrics['snr'] - baseline_snr if baseline_snr else 0
            
            print(f"   #{i+1} {config_name}")
            print(f"       📈 SNR: {metrics['snr']:.2f}dB ({improvement:+.2f})")
            print(f"       🔗 相关性: {metrics['correlation']:.4f}")
            print(f"       🎤 Vocoder: {config['vocoder']}")
            print(f"       📁 文件: {Path(data['file']).name}")
        
        # 按vocoder类型分析
        vocoder_analysis = {}
        for config_name, data in results.items():
            vocoder = data['config']['vocoder']
            if vocoder not in vocoder_analysis:
                vocoder_analysis[vocoder] = []
            vocoder_analysis[vocoder].append(data['metrics']['snr'])
        
        print(f"\n📊 Vocoder效果对比:")
        for vocoder, snrs in vocoder_analysis.items():
            avg_snr = np.mean(snrs)
            max_snr = np.max(snrs)
            print(f"   🎤 {vocoder}: 平均{avg_snr:.2f}dB, 最佳{max_snr:.2f}dB")
        
        # 改进建议
        best_config = sorted_results[0][1]['config']
        print(f"\n💡 改进建议:")
        if best_config['vocoder'] == 'hifigan':
            print("   ✅ 神经vocoder显著改善质量")
        else:
            print("   ⚠️ 建议尝试更多神经vocoder选项")
        
        if best_config['enhanced']:
            print("   ✅ 高质量mel配置有效")
        else:
            print("   ⚠️ 可尝试更高分辨率mel配置")
        
        print(f"\n📁 所有结果保存在: {output_dir}")
        print("🎧 建议主观评估最佳结果")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python vae_quality_fixer.py <音频文件路径>")
        
        # 列出可用音频文件
        audio_files = list(Path('.').glob('*.wav')) + list(Path('.').glob('*.mp3'))
        
        if audio_files:
            print("找到音频文件:")
            for i, file in enumerate(audio_files[:5], 1):
                print(f"{i}. {file.name}")
            
            try:
                choice = int(input("选择文件序号: "))
                audio_path = str(audio_files[choice-1])
            except (ValueError, IndexError):
                print("❌ 无效选择")
                return
        else:
            print("❌ 没有找到音频文件")
            return
    else:
        audio_path = sys.argv[1]
    
    # 运行修复测试
    fixer = VAEReconstructionFixer()
    results = fixer.comprehensive_test(audio_path)
    
    print("\n✅ VAE重建质量修复测试完成!")

if __name__ == "__main__":
    main()
