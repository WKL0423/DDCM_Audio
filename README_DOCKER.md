# Dockerized Environment for AudioLDM2 Tooling

This repository provides Dockerfiles for both CPU-only and CUDA-enabled setups, so you can avoid Anaconda and run everything in reproducible containers.

## Images

- Dockerfile.cpu (Python 3.10 + PyTorch CPU)
- Dockerfile.cuda (PyTorch 2.2.1 + CUDA 11.8 runtime)

Both images configure a shared Hugging Face cache at `/workspace/hf_cache` for faster repeated runs.

## Build

- CPU image:
  ```powershell
  docker build -t audioldm2:cpu -f Dockerfile.cpu .
  ```
- CUDA image:
  ```powershell
  docker build -t audioldm2:cuda -f Dockerfile.cuda .
  ```

Alternatively, use docker-compose to build both:
```powershell
docker compose build
```

## Run

- CPU container (smoke test):
  ```powershell
  docker run --rm -it ^
    -v ${PWD}:/workspace ^
    -v audioldm2_hf_cache:/workspace/hf_cache ^
    --name audioldm2_cpu audioldm2:cpu ^
    python smoke_test_sampler.py --sampler dpmpp --steps 16 --prompt "a minimal techno beat" --output evaluation_results/docker_cpu_test.wav
  ```

- GPU container (requires NVIDIA Container Toolkit):
  ```powershell
  docker run --rm -it ^
    --gpus all ^
    -v ${PWD}:/workspace ^
    -v audioldm2_hf_cache:/workspace/hf_cache ^
    --name audioldm2_gpu audioldm2:cuda ^
    python smoke_test_sampler.py --sampler dpmpp --steps 16 --prompt "a minimal techno beat" --output evaluation_results/docker_gpu_test.wav
  ```

Or with docker-compose:
```powershell
docker compose up audioldm2-cpu
# or
# docker compose up audioldm2-gpu
```

## Notes

- CUDA base uses `pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime`. If your host drivers are for CUDA 12.x, consider updating the base to a 12.x tag.
- The project relies on Hugging Face models. The cache is mounted as a named volume so subsequent runs are faster and offline-friendly.
- If you need to run benchmarks:
  ```powershell
  docker run --rm -it -v ${PWD}:/workspace -v audioldm2_hf_cache:/workspace/hf_cache audioldm2:cuda ^
    python benchmark_sampler.py --model cvssp/audioldm2-music --samplers default dpmpp unipc --steps 16 32 --gs 3.5 --seed 1234 --out evaluation_results
  ```

## Troubleshooting

- If you see `librosa` warnings about `pkg_resources`, they are harmless. 
- For GPU: ensure NVIDIA drivers + Container Toolkit installed. Check with `docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi`.
- Windows PowerShell escaping uses `^` for line breaks. Replace with `\` on Linux/macOS.
