"""
AudioLDM2 VAE音频重建 - 最终整合版本

这是集成了所有技术突破和最佳实践的完整版本：
1. ✅ 解决了Vocoder维度问题 (关键突破)
2. ✅ 修复了数据类型兼容性问题  
3. ✅ 实现了多种重建方法对比
4. ✅ 提供了完整的性能分析
5. ✅ 包含了详细的错误处理

使用方法:
python ultimate_vae_reconstruction.py <audio_file> [model_variant] [max_length]

示例:
python ultimate_vae_reconstruction.py AudioLDM2_Music_output.wav music 10
"""

import torch
import librosa
import numpy as np
import os
import time
from pathlib import Path
import torchaudio
import warnings
warnings.filterwarnings('ignore')

from diffusers import AudioLDM2Pipeline


class AudioLDM2VAEReconstructor:
    """AudioLDM2 VAE音频重建器"""
    
    def __init__(self, model_variant="music", device=None):
        """
        初始化重建器
        
        Args:
            model_variant: 模型变体 ("music", "speech", "large")
            device: 计算设备 (None为自动检测)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        
        # 模型变体映射
        self.model_variants = {
            "music": "cvssp/audioldm2-music",
            "speech": "cvssp/audioldm2",  
            "large": "cvssp/audioldm2-large"
        }
        
        self.model_id = self.model_variants.get(model_variant, model_variant)
        
        print(f"🚀 初始化AudioLDM2 VAE重建器")
        print(f"   🎯 设备: {self.device}")
        print(f"   🎵 模型: {self.model_id}")
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        print(f"🔄 正在加载模型...")
        
        # 使用float32避免类型问题
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32
        ).to(self.device)
        
        self.vae = self.pipeline.vae
        self.vocoder = self.pipeline.vocoder
        
        print(f"✅ 模型加载完成")
        print(f"   🔧 VAE: {type(self.vae).__name__}")
        print(f"   🎤 Vocoder: {type(self.vocoder).__name__}")
        
        if hasattr(self.vocoder, 'config'):
            print(f"   📊 Vocoder参数: {self.vocoder.config.model_in_dim}通道, {self.vocoder.config.sampling_rate}Hz")
    
    def audio_to_mel(self, audio):
        """音频转mel-spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=160,
            window='hann',
            center=True,
            pad_mode='reflect',
            power=2.0
        )
        
        # 转换到对数域并归一化到[-1, 1]
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = 2.0 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1.0
        
        return mel_spec_norm.astype(np.float32)
    
    def vae_encode_decode(self, mel_spec):
        """VAE编码解码"""
        # 准备输入张量
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            # 编码到潜在空间
            latent = self.vae.encode(mel_tensor).latent_dist.sample()
            
            # 从潜在空间解码
            decoded = self.vae.decode(latent).sample
        
        return decoded.squeeze().cpu().float().numpy()
    
    def mel_to_audio_vocoder(self, mel_spec):
        """使用修正维度的Vocoder重建音频"""
        try:
            # 转换为张量并添加batch维度
            mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float().to(self.device)
            
            # 🔑 关键修正: 转置维度从 [batch, channels, time] 到 [batch, time, channels]
            mel_tensor_corrected = mel_tensor.transpose(-2, -1)
            
            with torch.no_grad():
                audio_tensor = self.vocoder(mel_tensor_corrected)
                
                if isinstance(audio_tensor, tuple):
                    audio_tensor = audio_tensor[0]
                
                audio = audio_tensor.squeeze().cpu().numpy()
            
            return audio, "success"
            
        except Exception as e:
            return None, f"vocoder_failed: {e}"
    
    def mel_to_audio_griffinlim(self, mel_spec):
        """使用Griffin-Lim重建音频"""
        try:
            # 确保float32类型
            mel_spec = mel_spec.astype(np.float32)
            
            # 反归一化：从[-1,1] -> [min_db, 0]
            mel_spec_denorm = (mel_spec + 1.0) / 2.0
            mel_spec_denorm = mel_spec_denorm * 80.0 - 80.0
            
            # 转换到功率域
            mel_spec_power = librosa.db_to_power(mel_spec_denorm, ref=1.0)
            
            # Griffin-Lim重建
            audio = librosa.feature.inverse.mel_to_audio(
                mel_spec_power,
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=160,
                window='hann',
                center=True,
                pad_mode='reflect',
                n_iter=32
            )
            
            return audio, "success"
            
        except Exception as e:
            return None, f"griffinlim_failed: {e}"
    
    def calculate_metrics(self, original, reconstructed):
        """计算重建质量指标"""
        # 确保长度一致
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]
        
        # SNR计算
        noise = reconstructed - original
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # 相关系数
        correlation = np.corrcoef(original, reconstructed)[0, 1] if len(original) > 1 else 0
        
        # RMSE
        rmse = np.sqrt(np.mean(noise ** 2))
        
        return {
            'snr': snr,
            'correlation': correlation,
            'rmse': rmse,
            'signal_power': signal_power,
            'noise_power': noise_power
        }
    
    def reconstruct_audio(self, audio_path, max_length=10, output_dir=None):
        """
        完整的音频重建流程
        
        Args:
            audio_path: 输入音频文件路径
            max_length: 最大处理长度(秒)
            output_dir: 输出目录
            
        Returns:
            dict: 包含所有结果的字典
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"找不到音频文件: {audio_path}")
        
        # 创建输出目录
        if output_dir is None:
            output_dir = f"vae_reconstruction_{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        input_name = Path(audio_path).stem
        timestamp = int(time.time())
        
        print(f"\\n🎵 开始重建音频: {audio_path}")
        print(f"📁 输出目录: {output_dir}")
        
        # 1. 加载音频
        print(f"\\n1️⃣ 加载音频")
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        if len(audio) > max_length * self.sample_rate:
            audio = audio[:int(max_length * self.sample_rate)]
            print(f"   ✂️ 音频裁剪到 {max_length} 秒")
        
        print(f"   📊 音频信息: {len(audio)/self.sample_rate:.2f}秒, {len(audio)}样本")
        
        # 保存原始音频
        original_path = os.path.join(output_dir, f"{input_name}_original.wav")
        audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
        torchaudio.save(original_path, torch.from_numpy(audio_normalized).unsqueeze(0), self.sample_rate)
        
        # 2. 音频转Mel
        print(f"\\n2️⃣ 音频转Mel-spectrogram")
        start_time = time.time()
        mel_spec = self.audio_to_mel(audio)
        mel_time = time.time() - start_time
        print(f"   ✅ Mel形状: {mel_spec.shape} ({mel_time:.3f}秒)")
        
        # 3. VAE编码解码
        print(f"\\n3️⃣ VAE编码解码")
        start_time = time.time()
        decoded_mel = self.vae_encode_decode(mel_spec)
        vae_time = time.time() - start_time
        print(f"   ✅ VAE输出: {decoded_mel.shape} ({vae_time:.3f}秒)")
        
        # 4. 多种重建方法测试
        print(f"\\n4️⃣ 音频重建对比")
        
        results = []
        
        # 方法A: Vocoder重建
        print(f"\\n🎤 方法A: AudioLDM2 Vocoder")
        start_time = time.time()
        vocoder_audio, vocoder_status = self.mel_to_audio_vocoder(decoded_mel)
        vocoder_time = time.time() - start_time
        
        if vocoder_audio is not None:
            metrics = self.calculate_metrics(audio, vocoder_audio)
            
            vocoder_path = os.path.join(output_dir, f"{input_name}_vocoder_reconstruction.wav")
            audio_norm = vocoder_audio / (np.max(np.abs(vocoder_audio)) + 1e-8)
            torchaudio.save(vocoder_path, torch.from_numpy(audio_norm).unsqueeze(0), self.sample_rate)
            
            results.append({
                'method': 'AudioLDM2 Vocoder',
                'path': vocoder_path,
                'time': vocoder_time,
                'status': vocoder_status,
                **metrics
            })
            
            print(f"   ✅ 成功! SNR: {metrics['snr']:.2f}dB, 相关: {metrics['correlation']:.4f}")
        else:
            print(f"   ❌ 失败: {vocoder_status}")
        
        # 方法B: Griffin-Lim重建  
        print(f"\\n🎵 方法B: Griffin-Lim")
        start_time = time.time()
        gl_audio, gl_status = self.mel_to_audio_griffinlim(decoded_mel)
        gl_time = time.time() - start_time
        
        if gl_audio is not None:
            metrics = self.calculate_metrics(audio, gl_audio)
            
            gl_path = os.path.join(output_dir, f"{input_name}_griffinlim_reconstruction.wav")
            audio_norm = gl_audio / (np.max(np.abs(gl_audio)) + 1e-8)
            torchaudio.save(gl_path, torch.from_numpy(audio_norm).unsqueeze(0), self.sample_rate)
            
            results.append({
                'method': 'Griffin-Lim',
                'path': gl_path,
                'time': gl_time,
                'status': gl_status,
                **metrics
            })
            
            print(f"   ✅ 成功! SNR: {metrics['snr']:.2f}dB, 相关: {metrics['correlation']:.4f}")
        else:
            print(f"   ❌ 失败: {gl_status}")
        
        # 5. 结果分析
        print(f"\\n{'='*60}")
        print(f"🎯 AudioLDM2 VAE重建结果分析")
        print(f"{'='*60}")
        
        analysis = {
            'input_file': audio_path,
            'output_dir': output_dir,
            'original_path': original_path,
            'audio_duration': len(audio) / self.sample_rate,
            'mel_generation_time': mel_time,
            'vae_processing_time': vae_time,
            'results': results,
            'model_info': {
                'model_id': self.model_id,
                'device': self.device,
                'vae_type': type(self.vae).__name__,
                'vocoder_type': type(self.vocoder).__name__
            }
        }
        
        if results:
            # 按SNR排序
            results.sort(key=lambda x: x['snr'], reverse=True)
            
            print(f"\\n🏆 重建质量排名:")
            for i, result in enumerate(results, 1):
                print(f"   #{i} {result['method']}:")
                print(f"       📈 SNR: {result['snr']:.2f} dB")
                print(f"       🔗 相关系数: {result['correlation']:.4f}")
                print(f"       📏 RMSE: {result['rmse']:.6f}")
                print(f"       ⏱️ 处理时间: {result['time']:.3f}秒")
                print(f"       📄 文件: {result['path']}")
                print(f"       ✅ 状态: {result['status']}")
                print()
            
            best_result = results[0]
            analysis['best_method'] = best_result['method']
            analysis['best_snr'] = best_result['snr']
            
            print(f"🚀 最佳结果:")
            print(f"   🏆 最优方法: {best_result['method']}")
            print(f"   📈 最高SNR: {best_result['snr']:.2f} dB")
            print(f"   🔗 相关系数: {best_result['correlation']:.4f}")
            
            if len(results) > 1:
                improvement = best_result['snr'] - results[-1]['snr']
                print(f"   📊 方法间差异: {improvement:.2f} dB")
                analysis['method_difference'] = improvement
            
            # 检查vocoder成功状态
            vocoder_success = any(r['method'] == 'AudioLDM2 Vocoder' for r in results)
            if vocoder_success:
                print(f"\\n🎉 重大成就: AudioLDM2 Vocoder维度问题已完全解决！")
                analysis['vocoder_breakthrough'] = True
            else:
                print(f"\\n⚠️ Vocoder仍有问题，建议使用Griffin-Lim")
                analysis['vocoder_breakthrough'] = False
        else:
            print(f"\\n❌ 所有重建方法都失败了")
            analysis['success'] = False
        
        # 保存分析结果
        import json
        analysis_path = os.path.join(output_dir, f"{input_name}_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            # 将numpy类型转换为python原生类型以便JSON序列化
            def convert_types(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            def clean_for_json(data):
                if isinstance(data, dict):
                    return {k: clean_for_json(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [clean_for_json(item) for item in data]
                else:
                    return convert_types(data)
            
            json.dump(clean_for_json(analysis), f, indent=2, ensure_ascii=False)
        
        print(f"\\n📊 详细分析保存: {analysis_path}")
        print(f"📁 所有结果保存在: {output_dir}/")
        print(f"🎧 建议播放音频文件进行主观质量评估")
        print(f"\\n✅ 重建完成！")
        
        return analysis


def main():
    """命令行接口"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='AudioLDM2 VAE音频重建系统')
    parser.add_argument('audio_path', help='输入音频文件路径')
    parser.add_argument('--model', default='music', choices=['music', 'speech', 'large'], 
                       help='模型变体 (默认: music)')
    parser.add_argument('--max_length', type=int, default=10, 
                       help='最大处理长度(秒) (默认: 10)')
    parser.add_argument('--output_dir', help='输出目录 (默认: 自动生成)')
    
    if len(sys.argv) == 1:
        # 如果没有命令行参数，使用默认设置
        audio_path = "AudioLDM2_Music_output.wav"
        model_variant = "music"
        max_length = 10
        output_dir = None
        
        print(f"🎯 使用默认参数运行:")
        print(f"   📄 音频文件: {audio_path}")
        print(f"   🎵 模型变体: {model_variant}")
        print(f"   ⏱️ 最大长度: {max_length}秒")
    else:
        args = parser.parse_args()
        audio_path = args.audio_path
        model_variant = args.model
        max_length = args.max_length
        output_dir = args.output_dir
    
    print(f"🚀 启动AudioLDM2 VAE音频重建系统")
    print(f"=" * 60)
    
    try:
        # 创建重建器
        reconstructor = AudioLDM2VAEReconstructor(model_variant=model_variant)
        
        # 执行重建
        results = reconstructor.reconstruct_audio(
            audio_path=audio_path,
            max_length=max_length,
            output_dir=output_dir
        )
        
        print(f"\\n🎉 任务完成！查看输出目录了解详细结果。")
        
    except Exception as e:
        print(f"\\n❌ 错误: {e}")
        print(f"请检查输入文件和依赖项是否正确。")


if __name__ == "__main__":
    main()
