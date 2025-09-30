import os
import torch
import soundfile as sf

# 使用本地修改过的管道
from New_pipeline_audioldm2 import AudioLDM2Pipeline


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_or_path = os.environ.get("AUDIO_LDM2_MODEL_DIR", "cvssp/audioldm2")

    # 解析本地缓存目录（显式传给 from_pretrained）
    hf_home = os.environ.get("HF_HOME")
    hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        cache_dir = hf_hub_cache
    elif hf_home:
        cache_dir = os.path.join(hf_home, "hub")
    else:
        # 回退到 F 盘默认路径（与你当前机器一致），其他机器可通过环境变量覆盖
        cache_dir = r"F:\\Kasai_Lab\\hf_cache\\huggingface\\hub"

    # 优先离线：若本地已缓存则不访问网络
    try:
        pipe = AudioLDM2Pipeline.from_pretrained(
            repo_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception as e:
        print("[smoke_test_sampler] 离线加载失败，尝试联网加载。原因:", e)
        pipe = AudioLDM2Pipeline.from_pretrained(
            repo_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir=cache_dir,
        )
    pipe = pipe.to(device)

    # 使用我们新增的快捷采样器切换：DPMSolver++ + Karras sigmas
    try:
        pipe.set_sampler("dpmpp", use_karras_sigmas=True)
    except Exception as e:
        print("[smoke_test_sampler] set_sampler failed:", e)

    prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
    g = torch.Generator(device).manual_seed(1234)

    audio = pipe(
        prompt=prompt,
        num_inference_steps=16,
        guidance_scale=3.5,
        audio_length_in_s=10.24,
        generator=g,
        num_waveforms_per_prompt=1,
    ).audios[0]

    # 保存到根目录
    sf.write("smoke_test_sampler_output.wav", audio, 16000)
    print("Saved: smoke_test_sampler_output.wav")


if __name__ == "__main__":
    main()
