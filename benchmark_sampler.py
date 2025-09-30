import os
import json
import time
import itertools
import argparse
import torch
import soundfile as sf
from pathlib import Path
from New_pipeline_audioldm2 import AudioLDM2Pipeline


def resolve_cache_dir():
    hf_home = os.environ.get("HF_HOME")
    hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        return hf_hub_cache
    if hf_home:
        return os.path.join(hf_home, "hub")
    return r"F:\\Kasai_Lab\\hf_cache\\huggingface\\hub"


def run_once(model, sampler, steps, gs, prompt, length, seed, cache_dir, device, out_dir):
    pipe = AudioLDM2Pipeline.from_pretrained(
        model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=cache_dir,
    ).to(device)

    if sampler != "default":
        try:
            pipe.set_sampler(sampler, use_karras_sigmas=True)
        except Exception as e:
            print(f"[benchmark] set_sampler failed for {sampler}: {e}")

    g = torch.Generator(device).manual_seed(seed)
    t0 = time.time()
    audio = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=gs,
        audio_length_in_s=length,
        generator=g,
        num_waveforms_per_prompt=1,
    ).audios[0]
    dt = time.time() - t0

    # Save wav
    model_tag = Path(str(model)).name.replace('/', '_') if isinstance(model, str) else "model"
    fname = f"{model_tag}__{sampler}_s{steps}_gs{gs}_seed{seed}.wav"
    out_path = out_dir / fname
    sf.write(str(out_path), audio, 16000)

    return {
        "model": model,
        "sampler": sampler,
        "steps": steps,
        "guidance_scale": gs,
        "seed": seed,
        "length": length,
        "device": device,
        "runtime_sec": dt,
        "output_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("AUDIO_LDM2_MODEL_DIR", "cvssp/audioldm2-music"))
    ap.add_argument("--samplers", nargs="*", default=["default", "dpmpp", "unipc"])
    ap.add_argument("--steps", nargs="*", type=int, default=[16, 32, 50, 100])
    ap.add_argument("--gs", type=float, default=3.5)
    ap.add_argument("--prompt", default="Techno music with a strong, upbeat tempo and high melodic riffs")
    ap.add_argument("--length", type=float, default=10.24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="evaluation_results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = resolve_cache_dir()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sampler, steps in itertools.product(args.samplers, args.steps):
        print(f"\n[benchmark] model={args.model} sampler={sampler} steps={steps} gs={args.gs} device={device}")
        metrics = run_once(
            model=args.model,
            sampler=sampler,
            steps=steps,
            gs=args.gs,
            prompt=args.prompt,
            length=args.length,
            seed=args.seed,
            cache_dir=cache_dir,
            device=device,
            out_dir=out_dir,
        )
        results.append(metrics)

    # Save json
    ts = int(time.time())
    json_path = out_dir / f"benchmark_sampler_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark to {json_path}")


if __name__ == "__main__":
    main()
