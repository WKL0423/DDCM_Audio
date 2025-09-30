# 新电脑快速使用指南（Windows）

本文档帮助你在一台新的 Windows 电脑上快速把本工程跑起来（含本地缓存设置）。

## 1. 准备 Conda 环境
- 安装 Anaconda 或 Miniconda
- 打开 PowerShell（建议管理员权限）

可选，一键脚本（推荐）：
- 路径：`scripts/setup_new_machine.ps1`
- 右键“使用 PowerShell 运行”，或在 PowerShell 中执行：
  - `powershell -ExecutionPolicy Bypass -File .\scripts\setup_new_machine.ps1`
- 参数（可选）：
  - `-EnvName audioldm_env` 指定环境名
  - `-PythonVersion 3.10` 指定 Python 版本
  - `-HfHome F:\Kasai_Lab\hf_cache\huggingface` 指定 HF 缓存
  - `-RepoRoot F:\Kasai_Lab\Code\AudioLDM_2_diffuser` 仓库目录

手动步骤（如不使用脚本）：
1) 创建环境并安装依赖
   - `conda create -n audioldm_env python=3.10 -y`
   - `conda activate audioldm_env`
   - 安装依赖（与项目当前版本匹配）：
     - `pip install torch==2.2.1+cpu torchvision==0.17.1+cpu torchaudio==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu`
     - `pip install diffusers==0.25.0 transformers==4.30.2 huggingface-hub==0.23.2 tokenizers==0.13.3 accelerate==0.23.0`
     - `pip install librosa==0.9.2 soundfile==0.13.1 matplotlib==3.8.1 pandas==2.1.3 einops tqdm`
     - 可选：`pip install pytorch-lightning==2.1.1 taming-transformers-rom1504 h5py==3.10.0 webdataset wandb kornia`

## 2. 配置 Hugging Face 本地缓存
- 推荐将缓存放到本地盘，避免 C 盘占用和重复下载。
- 设置用户级环境变量：
  - `HF_HOME=F:\Kasai_Lab\hf_cache\huggingface`
  - 可选：`HUGGINGFACE_HUB_CACHE=F:\Kasai_Lab\hf_cache\huggingface\hub`
- 将现有机器的 `F:\Kasai_Lab\hf_cache\huggingface\hub` 目录拷贝到新电脑同路径或你指定路径下（保留目录结构）。
- 重启 VS Code 或终端使其生效。

## 3. VS Code 设置
- Ctrl+Shift+P → `Python: Select Interpreter` → 选择 `audioldm_env`。
- 重新加载窗口（可选）以确保终端默认激活该环境。

## 4. 运行验证
- 在仓库根目录下运行：
  - `conda activate audioldm_env`
  - `python -X faulthandler -u .\smoke_test_sampler.py`
- 正常情况下会生成 `smoke_test_sampler_output.wav`。

## 5. 常见问题
- 首次运行仍尝试联网？
  - 本工程中的 `smoke_test_sampler.py` 和 `step1_training_matched_vae_reconstruction_fixed.py` 都会显式传入 `cache_dir`，并先尝试 `local_files_only=True`。若缓存不完整，会回退联网补齐；若你要完全离线，请确保缓存目录完整。
- 符号链接警告（Windows Developer Mode）
  - 未开启开发者模式时，HF 会提示降级缓存模式（不影响功能，只是更占空间）。
- 使用 GPU？
  - 若你的机器有 NVIDIA GPU，建议安装对应 CUDA 版本的 PyTorch：
    - 以 CUDA 11.8 为例：
      - `pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
    - 以 CUDA 12.1 为例：
      - `pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121`
  - 然后按原依赖安装 diffusers/transformers 等即可。
  - 代码中会自动选择 `device = 'cuda' if torch.cuda.is_available() else 'cpu'`，并在 GPU 时优先使用 float16 提升性能。

## 6. 自定义参数
- 模型目录：设置 `AUDIO_LDM2_MODEL_DIR` 指向本地权重目录（不设置则使用 Hugging Face 仓库名并走缓存）。
- 缓存位置：通过 `HF_HOME` 或 `HUGGINGFACE_HUB_CACHE` 调整，工程内脚本会自动解析并传给 `from_pretrained(..., cache_dir=...)`。

祝使用顺利！如需要，我可以添加一个一键健康检查脚本，自动校验 Python/依赖版本、HF 缓存可读性并跑一次 smoketest。