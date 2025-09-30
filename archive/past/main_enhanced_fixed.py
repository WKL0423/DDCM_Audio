from New_pipeline_audioldm2 import AudioLDM2Pipeline
import torch
import scipy
import time

def main():
    print("🎵 使用自定义 AudioLDM2 Pipeline 生成音频")
    print("=" * 50)
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载自定义 pipeline
    print("正在加载自定义 AudioLDM2 Pipeline...")
    repo_id = "cvssp/audioldm2"
    
    start_time = time.time()
    
    if device == "cuda":
        pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    else:
        pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
        pipe = pipe.to("cpu")
    
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 (耗时: {load_time:.2f}秒)")
    
    # 显示模型信息
    print(f"\n🔧 模型信息:")
    print(f"  - Checkpoint: {repo_id}")
    print(f"  - 模型类型: {type(pipe).__name__}")
    print(f"  - UNet 参数量: {sum(p.numel() for p in pipe.unet.parameters()) / 1e6:.1f}M")
    print(f"  - VAE 参数量: {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.1f}M")
    print(f"  - Text Encoder 类型: {type(pipe.text_encoder).__name__}")
    print(f"  - Text Encoder 2 类型: {type(pipe.text_encoder_2).__name__}")
    
    # 音频生成参数
    prompt = "Techno music with a strong, upbeat tempo and high melodic riffs."
    num_inference_steps = 200
    audio_length_in_s = 10.0
    
    print(f"\n📝 参数设置:")
    print(f"  - 提示词: {prompt}")
    print(f"  - 推理步数: {num_inference_steps}")
    print(f"  - 音频长度: {audio_length_in_s}秒")
    print(f"  - 数据类型: {pipe.unet.dtype}")
    
    # 生成音频
    print("\n正在生成音频...")
    gen_start_time = time.time()
    
    audio = pipe(
        prompt, 
        num_inference_steps=num_inference_steps,
        audio_length_in_s=audio_length_in_s
    ).audios[0]
    
    gen_time = time.time() - gen_start_time
    print(f"✓ 音频生成完成 (耗时: {gen_time:.2f}秒)")
    
    # 保存音频
    output_file = "custom_techno.wav"
    scipy.io.wavfile.write(output_file, rate=16000, data=audio)
    print(f"✓ 音频已保存到: {output_file}")
    
    # 性能统计
    total_time = load_time + gen_time
    print(f"\n📊 性能统计:")
    print(f"  - 模型加载时间: {load_time:.2f}秒")
    print(f"  - 音频生成时间: {gen_time:.2f}秒")
    print(f"  - 总耗时: {total_time:.2f}秒")
    print(f"  - 生成速度: {audio_length_in_s/gen_time:.2f}x 实时速度")

if __name__ == "__main__":
    main()
