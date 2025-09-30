"""
AudioLDM2 VAE 立即改进版本
=========================

基于瓶颈分析的快速优化方案
目标: 在现有架构下最大化重建质量
"""

import torch
import librosa
import numpy as np
import time
from pathlib import Path
from diffusers import AudioLDM2Pipeline
import scipy.io.wavfile
import scipy.signal
import warnings

warnings.filterwarnings("ignore")

class VAEQuickImprover:
    """VAE快速改进器 - 在现有限制下最大化质量"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚡ VAE快速改进器启动")
        print(f"📱 设备: {self.device}")
        
        # 加载AudioLDM2
        self.pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-music",
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16
        ).to(self.device)
        
        self.vae = self.pipe.vae
        print("✅ 模型加载完成")
    
    def load_audio(self, audio_path: str, duration: float = 5.0):
        """加载音频"""
        audio, sr = librosa.load(audio_path, sr=16000, duration=duration)
        print(f"📊 音频: {len(audio)/sr:.2f}秒")
        return audio, sr
    
    def create_optimal_mel(self, audio: np.ndarray, sr: int, enhanced: bool = True):
        """创建优化的mel频谱"""
        if enhanced:
            # 高质量配置
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, n_fft=2048,
                hop_length=256, win_length=1024, fmin=0, fmax=8000
            )
        else:
            # 标准配置
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, n_fft=1024,
                hop_length=256, win_length=1024, fmin=0, fmax=8000
            )
        
        # 动态范围优化
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 改进的归一化（保留更多动态信息）
        mel_spec = np.clip(mel_spec, -80, 0)  # 限制范围
        mel_spec = 2.0 * (mel_spec + 80) / 80 - 1.0  # 归一化到[-1,1]
        
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        return mel_tensor.to(self.device)
    
    def vae_reconstruct(self, mel_tensor):
        """VAE重建"""
        with torch.no_grad():
            if next(self.vae.parameters()).dtype == torch.float16:
                mel_tensor = mel_tensor.half()
            
            # 确保尺寸匹配
            if mel_tensor.shape[-1] % 4 != 0:
                pad_length = 4 - (mel_tensor.shape[-1] % 4)
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_length))
            
            # VAE编码解码
            latent = self.vae.encode(mel_tensor).latent_dist.sample()
            reconstructed = self.vae.decode(latent).sample
            
            return reconstructed.float()
    
    def enhanced_griffin_lim(self, mel_tensor, sr: int = 16000, method: str = "advanced"):
        """增强的Griffin-Lim算法"""
        mel_np = mel_tensor.squeeze().cpu().numpy()
        
        # 反归一化（改进版）
        mel_np = (mel_np + 1.0) / 2.0 * 80 - 80  # 恢复到[-80, 0]范围
        mel_linear = librosa.db_to_power(mel_np)
        
        if method == "basic":
            # 基础版本
            audio = librosa.feature.inverse.mel_to_audio(
                mel_linear, sr=sr, n_iter=64,
                hop_length=256, win_length=1024, n_fft=1024
            )
        elif method == "advanced":
            # 高级版本 - 更多迭代
            audio = librosa.feature.inverse.mel_to_audio(
                mel_linear, sr=sr, n_iter=128,
                hop_length=256, win_length=1024, n_fft=1024
            )
        elif method == "premium":
            # 顶级版本 - 最多迭代和更好参数
            audio = librosa.feature.inverse.mel_to_audio(
                mel_linear, sr=sr, n_iter=256,
                hop_length=256, win_length=1024, n_fft=2048
            )
        
        return audio
    
    def post_process_audio(self, audio: np.ndarray, method: str = "enhanced"):
        """音频后处理"""
        if method == "basic":
            # 基础后处理：归一化
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
        elif method == "enhanced":
            # 增强后处理
            # 1. 软限幅
            audio = np.tanh(audio * 1.2) * 0.9
            
            # 2. 轻微去噪（维纳滤波）
            try:
                audio = scipy.signal.wiener(audio, mysize=5)
            except:
                pass  # 如果滤波失败，跳过
            
            # 3. 归一化
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
        elif method == "premium":
            # 顶级后处理
            # 1. 动态范围压缩
            audio = np.sign(audio) * np.power(np.abs(audio), 0.8)
            
            # 2. 高通滤波（去除低频噪声）
            try:
                from scipy import signal
                b, a = signal.butter(3, 80, 'high', fs=16000)
                audio = signal.filtfilt(b, a, audio)
            except:
                pass
            
            # 3. 软限幅和归一化
            audio = np.tanh(audio * 1.5) * 0.85
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    
    def calculate_metrics(self, original, reconstructed):
        """计算质量指标"""
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
    
    def comprehensive_improvement_test(self, audio_path: str):
        """全面改进测试"""
        print("\n⚡ VAE快速改进测试")
        print("="*50)
        
        # 创建输出目录
        output_dir = Path("vae_quick_improvement_test")
        output_dir.mkdir(exist_ok=True)
        
        # 加载音频
        audio, sr = self.load_audio(audio_path)
        
        # 保存原始音频
        audio_name = Path(audio_path).stem
        timestamp = int(time.time())
        original_path = output_dir / f"{audio_name}_original_{timestamp}.wav"
        scipy.io.wavfile.write(original_path, sr, (audio * 32767).astype(np.int16))
        
        results = {}
        
        # 测试配置
        test_configs = [
            {"name": "baseline", "mel": False, "griffin": "basic", "post": "basic"},
            {"name": "enhanced_mel", "mel": True, "griffin": "basic", "post": "basic"},
            {"name": "enhanced_griffin", "mel": False, "griffin": "advanced", "post": "basic"},
            {"name": "enhanced_post", "mel": False, "griffin": "basic", "post": "enhanced"},
            {"name": "all_enhanced", "mel": True, "griffin": "advanced", "post": "enhanced"},
            {"name": "premium", "mel": True, "griffin": "premium", "post": "premium"},
        ]
        
        print(f"\n🧪 测试 {len(test_configs)} 种改进配置...")
        
        for i, config in enumerate(test_configs):
            print(f"\n📊 配置 {i+1}: {config['name']}")
            
            try:
                # 创建mel频谱
                mel_tensor = self.create_optimal_mel(audio, sr, config['mel'])
                
                # VAE重建
                reconstructed_mel = self.vae_reconstruct(mel_tensor)
                
                # Griffin-Lim重建
                audio_recon = self.enhanced_griffin_lim(
                    reconstructed_mel, sr, config['griffin']
                )
                
                # 后处理
                final_audio = self.post_process_audio(audio_recon, config['post'])
                
                # 计算指标
                metrics = self.calculate_metrics(audio, final_audio)
                
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
        
        # 生成改进报告
        self._generate_improvement_report(results, output_dir)
        
        return results
    
    def _generate_improvement_report(self, results, output_dir):
        """生成改进报告"""
        print("\n" + "="*50)
        print("⚡ VAE快速改进测试结果")
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
        
        baseline_snr = None
        if 'baseline' in results:
            baseline_snr = results['baseline']['metrics']['snr']
        
        print(f"\n🏆 改进效果排名:")
        for i, (config_name, data) in enumerate(sorted_results):
            metrics = data['metrics']
            config = data['config']
            
            improvement = metrics['snr'] - baseline_snr if baseline_snr else 0
            
            print(f"   #{i+1} {config_name}")
            print(f"       📈 SNR: {metrics['snr']:.2f}dB ({improvement:+.2f})")
            print(f"       🔗 相关性: {metrics['correlation']:.4f}")
            print(f"       📁 文件: {Path(data['file']).name}")
        
        # 最佳改进分析
        if len(sorted_results) > 0:
            best_config = sorted_results[0][1]['config']
            best_snr = sorted_results[0][1]['metrics']['snr']
            
            print(f"\n🚀 最佳改进组合: {sorted_results[0][0]}")
            if baseline_snr:
                total_improvement = best_snr - baseline_snr
                print(f"   📈 总体SNR提升: {total_improvement:+.2f}dB")
            
            # 分析有效的改进方法
            print(f"\n💡 有效改进方法:")
            if best_config['mel']:
                print("   ✅ 高质量mel配置有效")
            if best_config['griffin'] != 'basic':
                print("   ✅ 增强Griffin-Lim有效")
            if best_config['post'] != 'basic':
                print("   ✅ 音频后处理有效")
        
        print(f"\n📁 所有结果保存在: {output_dir}")
        print("🎧 建议播放最佳结果进行主观评估")
        
        # 下一步建议
        print(f"\n🎯 进一步优化建议:")
        best_snr = sorted_results[0][1]['metrics']['snr'] if sorted_results else -10
        if best_snr > 0:
            print("   ✅ 已达到可用水平，继续精细调优")
        elif best_snr > -3:
            print("   🔧 接近可用水平，尝试更激进的优化")
        else:
            print("   ⚠️ 仍需架构级改进 (神经vocoder等)")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python vae_quick_improver.py <音频文件>")
        
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
    
    # 运行改进测试
    improver = VAEQuickImprover()
    results = improver.comprehensive_improvement_test(audio_path)
    
    print("\n✅ VAE快速改进测试完成!")

if __name__ == "__main__":
    main()
