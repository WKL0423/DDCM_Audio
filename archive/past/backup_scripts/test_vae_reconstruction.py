"""
AudioLDM2 VAE 音频重建测试脚本
测试 AudioLDM2 的 VAE 编码器-解码器功能，用于音频压缩研究

功能：
1. 加载音频文件
2. 使用 AudioLDM2 VAE 编码音频到潜在空间
3. 从潜在空间解码回音频
4. 保存重建的音频文件
5. 计算重建质量指标

作者：Wang
日期：2024
"""

import torch
import torchaudio
import numpy as np
import os
import time
from pathlib import Path
import librosa
from sklearn.metrics import mean_squared_error
from scipy.signal import correlate

from diffusers import AudioLDM2Pipeline


class AudioLDM2VAETester:
    def __init__(self, model_id="cvssp/audioldm2", device=None):
        """
        初始化 AudioLDM2 VAE 测试器
        
        Args:
            model_id: AudioLDM2 模型ID
            device: 计算设备，默认自动选择
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        print(f"正在加载 AudioLDM2 模型: {model_id}")
        start_time = time.time()
        
        # 加载完整的 AudioLDM2 pipeline
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # 获取 VAE 组件
        self.vae = self.pipeline.vae
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # VAE 参数
        self.sample_rate = 16000  # AudioLDM2 默认采样率
        self.hop_length = 160     # VAE hop length
        self.vae_scale_factor = 8  # VAE downsampling factor
        
    def load_audio(self, audio_path, target_length=None):
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            target_length: 目标长度（秒），None表示保持原长度
            
        Returns:
            torch.Tensor: 音频张量 (1, samples)
        """
        print(f"正在加载音频: {audio_path}")
        
        # 使用 librosa 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # 如果指定了目标长度，裁剪或填充音频
        if target_length is not None:
            target_samples = int(target_length * self.sample_rate)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            elif len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        
        # 转换为张量
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        
        print(f"音频加载完成: 采样率={sr}, 长度={len(audio)/sr:.2f}秒, 形状={audio_tensor.shape}")
        return audio_tensor
    
    def encode_audio(self, audio_tensor):
        """
        使用 VAE 编码器将音频编码到潜在空间
        
        Args:
            audio_tensor: 音频张量 (1, samples)
            
        Returns:
            torch.Tensor: 潜在表示
        """
        print("正在编码音频到潜在空间...")
        
        with torch.no_grad():
            # 确保音频长度是 VAE 处理长度的倍数
            audio_length = audio_tensor.shape[-1]
            pad_length = (self.vae_scale_factor - (audio_length % self.vae_scale_factor)) % self.vae_scale_factor
            if pad_length > 0:
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_length))
            
            # 添加批次维度如果需要
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # (1, 1, samples)
            
            # VAE 编码
            latents = self.vae.encode(audio_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
        print(f"编码完成，潜在表示形状: {latents.shape}")
        return latents, audio_length
    
    def decode_audio(self, latents, original_length):
        """
        使用 VAE 解码器从潜在空间解码回音频
        
        Args:
            latents: 潜在表示
            original_length: 原始音频长度
            
        Returns:
            torch.Tensor: 重建的音频张量
        """
        print("正在从潜在空间解码音频...")
        
        with torch.no_grad():
            # VAE 解码
            latents = latents / self.vae.config.scaling_factor
            audio_tensor = self.vae.decode(latents).sample
            
            # 裁剪到原始长度
            if audio_tensor.shape[-1] > original_length:
                audio_tensor = audio_tensor[..., :original_length]
        
        print(f"解码完成，音频形状: {audio_tensor.shape}")
        return audio_tensor
    
    def save_audio(self, audio_tensor, output_path):
        """
        保存音频文件
        
        Args:
            audio_tensor: 音频张量
            output_path: 输出路径
        """
        print(f"正在保存音频到: {output_path}")
        
        # 确保音频在 CPU 上且为正确形状
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()
        
        # 移除批次维度
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.squeeze(0)
        
        # 归一化音频
        audio_np = audio_tensor.numpy()
        audio_np = audio_np / np.max(np.abs(audio_np)) if np.max(np.abs(audio_np)) > 0 else audio_np
        
        # 使用 torchaudio 保存
        torchaudio.save(output_path, torch.from_numpy(audio_np).unsqueeze(0), self.sample_rate)
        print(f"音频保存完成")
    
    def calculate_metrics(self, original_audio, reconstructed_audio):
        """
        计算重建质量指标
        
        Args:
            original_audio: 原始音频张量
            reconstructed_audio: 重建音频张量
            
        Returns:
            dict: 指标字典
        """
        print("正在计算重建质量指标...")
        
        # 转换为 numpy 数组
        orig = original_audio.cpu().numpy().flatten()
        recon = reconstructed_audio.cpu().numpy().flatten()
        
        # 确保长度一致
        min_len = min(len(orig), len(recon))
        orig = orig[:min_len]
        recon = recon[:min_len]
        
        # 计算指标
        mse = mean_squared_error(orig, recon)
        rmse = np.sqrt(mse)
        
        # 计算信噪比 (SNR)
        signal_power = np.mean(orig ** 2)
        noise_power = np.mean((orig - recon) ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # 计算相关系数
        correlation = np.corrcoef(orig, recon)[0, 1] if len(orig) > 1 else 0
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'SNR_dB': snr,
            'Correlation': correlation
        }
        
        print("指标计算完成:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def test_reconstruction(self, input_audio_path, output_dir=".", audio_length=None):
        """
        执行完整的音频重建测试
        
        Args:
            input_audio_path: 输入音频文件路径
            output_dir: 输出目录
            audio_length: 音频长度限制（秒）
            
        Returns:
            dict: 测试结果和指标
        """
        print(f"\n{'='*50}")
        print(f"开始 AudioLDM2 VAE 重建测试")
        print(f"输入文件: {input_audio_path}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*50}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        input_name = Path(input_audio_path).stem
        timestamp = int(time.time())
        
        reconstructed_path = os.path.join(output_dir, f"{input_name}_vae_reconstructed_{timestamp}.wav")
        original_processed_path = os.path.join(output_dir, f"{input_name}_original_processed_{timestamp}.wav")
        
        try:
            # 1. 加载音频
            start_time = time.time()
            original_audio = self.load_audio(input_audio_path, target_length=audio_length)
            load_time = time.time() - start_time
            
            # 保存处理后的原始音频以便比较
            self.save_audio(original_audio, original_processed_path)
            
            # 2. 编码
            start_time = time.time()
            latents, original_length = self.encode_audio(original_audio)
            encode_time = time.time() - start_time
            
            # 3. 解码
            start_time = time.time()
            reconstructed_audio = self.decode_audio(latents, original_length)
            decode_time = time.time() - start_time
            
            # 4. 保存重建音频
            self.save_audio(reconstructed_audio, reconstructed_path)
            
            # 5. 计算指标
            metrics = self.calculate_metrics(original_audio, reconstructed_audio)
            
            # 6. 生成测试报告
            results = {
                'input_file': input_audio_path,
                'original_processed_file': original_processed_path,
                'reconstructed_file': reconstructed_path,
                'model_id': self.pipeline.config._name_or_path if hasattr(self.pipeline.config, '_name_or_path') else "unknown",
                'device': str(self.device),
                'audio_length_seconds': original_length / self.sample_rate,
                'latent_shape': latents.shape,
                'processing_times': {
                    'load_time': load_time,
                    'encode_time': encode_time,
                    'decode_time': decode_time,
                    'total_time': load_time + encode_time + decode_time
                },
                'quality_metrics': metrics
            }
            
            print(f"\n{'='*50}")
            print(f"VAE 重建测试完成!")
            print(f"{'='*50}")
            print(f"原始音频: {original_processed_path}")
            print(f"重建音频: {reconstructed_path}")
            print(f"音频长度: {results['audio_length_seconds']:.2f}秒")
            print(f"潜在表示形状: {results['latent_shape']}")
            print(f"总处理时间: {results['processing_times']['total_time']:.2f}秒")
            print(f"重建质量 SNR: {metrics['SNR_dB']:.2f} dB")
            print(f"相关系数: {metrics['Correlation']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"测试过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数：演示 VAE 重建测试"""
    print("AudioLDM2 VAE 音频重建测试")
    print("="*50)
    
    # 配置参数
    model_variants = {
        "1": ("cvssp/audioldm2", "标准版"),
        "2": ("cvssp/audioldm2-large", "大型版"),
        "3": ("cvssp/audioldm2-music", "音乐版"),
        "4": ("declare-lab/audioldm2-gigaspeech", "语音版")
    }
    
    # 让用户选择模型变体
    print("可用的 AudioLDM2 模型变体:")
    for key, (model_id, description) in model_variants.items():
        print(f"{key}. {description} ({model_id})")
    
    choice = input("请选择模型变体 (1-4, 默认1): ").strip() or "1"
    
    if choice not in model_variants:
        print("无效选择，使用默认标准版")
        choice = "1"
    
    model_id, model_name = model_variants[choice]
    print(f"选择的模型: {model_name} ({model_id})")
    
    # 初始化测试器
    try:
        tester = AudioLDM2VAETester(model_id=model_id)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
    # 查找可用的音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    current_dir = Path('.')
    
    for ext in audio_extensions:
        audio_files.extend(current_dir.glob(f'*{ext}'))
    
    if audio_files:
        print(f"\n找到 {len(audio_files)} 个音频文件:")
        for i, file in enumerate(audio_files, 1):
            print(f"{i}. {file.name}")
        
        # 让用户选择文件
        try:
            file_choice = input(f"请选择要测试的音频文件 (1-{len(audio_files)}, 默认1): ").strip() or "1"
            file_idx = int(file_choice) - 1
            
            if 0 <= file_idx < len(audio_files):
                input_audio = str(audio_files[file_idx])
            else:
                print("无效选择，使用第一个文件")
                input_audio = str(audio_files[0])
        except ValueError:
            print("无效输入，使用第一个文件")
            input_audio = str(audio_files[0])
    else:
        # 如果没有找到音频文件，提示用户
        input_audio = input("请输入音频文件路径: ").strip()
        if not input_audio or not os.path.exists(input_audio):
            print("未找到有效的音频文件，退出测试")
            return
    
    # 询问音频长度限制
    length_choice = input("是否限制音频长度？(输入秒数，或按回车跳过): ").strip()
    audio_length = None
    if length_choice:
        try:
            audio_length = float(length_choice)
            print(f"将音频限制为 {audio_length} 秒")
        except ValueError:
            print("无效输入，不限制音频长度")
    
    # 执行测试
    print(f"\n开始测试音频文件: {input_audio}")
    results = tester.test_reconstruction(
        input_audio_path=input_audio,
        output_dir="vae_test_results",
        audio_length=audio_length
    )
    
    if results:
        # 保存测试报告
        report_path = os.path.join("vae_test_results", f"test_report_{int(time.time())}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AudioLDM2 VAE 重建测试报告\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {model_name} ({model_id})\n")
            f.write(f"设备: {results['device']}\n")
            f.write(f"输入文件: {results['input_file']}\n")
            f.write(f"音频长度: {results['audio_length_seconds']:.2f}秒\n")
            f.write(f"潜在表示形状: {results['latent_shape']}\n\n")
            
            f.write("处理时间:\n")
            for key, value in results['processing_times'].items():
                f.write(f"  {key}: {value:.2f}秒\n")
            
            f.write("\n质量指标:\n")
            for key, value in results['quality_metrics'].items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write(f"\n输出文件:\n")
            f.write(f"  原始音频: {results['original_processed_file']}\n")
            f.write(f"  重建音频: {results['reconstructed_file']}\n")
        
        print(f"\n测试报告已保存到: {report_path}")
        print("\n测试完成！可以播放音频文件来比较原始音频和重建音频的质量。")
    else:
        print("测试失败")


if __name__ == "__main__":
    main()
