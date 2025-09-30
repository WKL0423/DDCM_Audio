from New_pipeline_audioldm2 import AudioLDM2Pipeline
import torch
import scipy
import time

def main():
    print("🎵 使用自定义 AudioLDM2 Pipeline 生成音频")
    print("=" * 50)
    
    # AudioLDM2 变体选择
    checkpoints = {
        "1": {
            "name": "AudioLDM2 (标准版)",
            "repo_id": "cvssp/audioldm2",
            "task": "文本转音频",
            "unet_size": "350M",
            "total_size": "1.1B",
            "training_data": "1150k小时"
        },
        "2": {
            "name": "AudioLDM2-Large (大型版)",
            "repo_id": "cvssp/audioldm2-large",
            "task": "文本转音频",
            "unet_size": "750M", 
            "total_size": "1.5B",
            "training_data": "1150k小时"
        },
        "3": {
            "name": "AudioLDM2-Music (音乐专用)",
            "repo_id": "cvssp/audioldm2-music",
            "task": "文本到音乐",
            "unet_size": "350M",
            "total_size": "1.1B", 
            "training_data": "665k小时"
        },
        "4": {
            "name": "AudioLDM2-GigaSpeech (语音)",
            "repo_id": "anhnct/audioldm2_gigaspeech",
            "task": "文本转语音",
            "unet_size": "350M",
            "total_size": "1.1B",
            "training_data": "10k小时"
        }
    }
    
    print("📋 可用的 AudioLDM2 变体:")
    for key, info in checkpoints.items():
        print(f"  {key}. {info['name']}")
        print(f"     任务: {info['task']} | UNet: {info['unet_size']} | 总大小: {info['total_size']}")
        print(f"     训练数据: {info['training_data']}")
        print()
    
    # 选择模型 (默认使用标准版)
    choice = "3"  # 可以修改这里来选择不同的模型
    selected_model = checkpoints[choice]
    repo_id = selected_model["repo_id"]
    
    print(f"🎯 已选择: {selected_model['name']}")
    print(f"   Checkpoint: {repo_id}")
    print(f"   专用任务: {selected_model['task']}")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 加载自定义 pipeline
    print("正在加载自定义 AudioLDM2 Pipeline...")
    
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
    print(f"  - 专用任务: {selected_model['task']}")
    print(f"  - UNet 参数量: {sum(p.numel() for p in pipe.unet.parameters()) / 1e6:.1f}M")
    print(f"  - VAE 参数量: {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.1f}M")
    print(f"  - Text Encoder 类型: {type(pipe.text_encoder).__name__}")
    print(f"  - Text Encoder 2 类型: {type(pipe.text_encoder_2).__name__}")
    
    # 根据模型类型选择合适的提示词
    if "music" in repo_id.lower():
        prompt = "Upbeat electronic dance music with synthesizers and drum beats"
        transcription = None
    elif "gigaspeech" in repo_id.lower() or "speech" in selected_model['task'].lower():
        prompt = "A female speaker with clear pronunciation"
        transcription = "Hello, this is a test of text to speech generation."
    else:
        prompt = "Techno music with a strong, upbeat tempo and high melodic riffs."
        transcription = None
      # 音频生成参数
    num_inference_steps = 200
    audio_length_in_s = 10.0
    
    print(f"\n📝 参数设置:")
    print(f"  - 提示词: {prompt}")
    if transcription:
        print(f"  - 转录文本: {transcription}")
    print(f"  - 推理步数: {num_inference_steps}")
    print(f"  - 音频长度: {audio_length_in_s}秒")
    print(f"  - 数据类型: {pipe.unet.dtype}")
    
    # 生成音频
    print("\n正在生成音频...")
    gen_start_time = time.time()
    
    # 根据模型类型调用不同的参数
    if transcription:
        audio = pipe(
            prompt,
            transcription=transcription,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
            max_new_tokens=512  # TTS 模型需要这个参数
        ).audios[0]
    else:
        audio = pipe(
            prompt, 
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s
        ).audios[0]
      gen_time = time.time() - gen_start_time
    print(f"✓ 音频生成完成 (耗时: {gen_time:.2f}秒)")
    
    # 保存音频
    model_name = selected_model['name'].split('(')[0].strip().replace(' ', '_').replace('-', '_')
    output_file = f"{model_name}_output.wav"
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
