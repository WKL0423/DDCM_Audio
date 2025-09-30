# 变更与下载记录（Plan A）

日期: 2025-09-29
环境: audioldm_env (Conda, Python 3.10.18)

核心依赖版本
- torch 2.2.1+cpu
- diffusers 0.25.0
- transformers 4.30.2
- huggingface-hub 0.23.2
- tokenizers 0.13.3
- librosa 0.9.2, soundfile 0.13.1, accelerate 0.23.0

已下载的模型与缓存位置
- 模型源: cvssp/audioldm2（首次运行时联网下载）
- 缓存根目录 (Windows): C:\\Users\\TheOne\\.cache\\huggingface\\hub\\models--cvssp--audioldm2
  - 说明: 缓存路径由 huggingface_hub 管理，可通过环境变量 HF_HOME 修改。
- 下载日志中出现的文件（节选）：
  - 配置类: model_index.json, scheduler_config.json, config.json, preprocessor_config.json
  - tokenizer: merges.txt (~456KB), vocab.json (~798KB), tokenizer.json (~2.4MB), special_tokens_map.json, tokenizer_config.json, spiece.model (~792KB)
  - 核心权重（多组件）:
    - diffusion_pytorch_model.safetensors: ~4.74MB / ~222MB / ~1.39GB（不同子模块）
    - model.safetensors: ~498MB / ~221MB / ~776MB / ~1.36GB（不同子模块）
  - 注: 上述多份权重对应 UNet / VAE / 文本编码器 / 语言模型 / 声码器等不同组件。

代码修改摘要
1) New_pipeline_audioldm2.py
- 新增: set_sampler(sampler="dpmpp"|"unipc", use_karras_sigmas=True, timestep_spacing=None)
  - 支持在推理前一键切换 DPMSolver++ 或 UniPC 少步数采样器
  - 自动从现有 scheduler.config 迁移配置
- 兼容性:
  - GPT2 旧版 Transformers（无 _get_initial_cache_position / _update_model_kwargs_for_generation）提供无缓存回退生成路径
  - DPMSolver++ / UniPC 导入增加子模块回退，避免版本差异导致不可用
  - XLA 可用性检查调整为动态导入，避免编辑器静态报错

2) smoke_test_sampler.py
- 改为使用本地修改过的管道: from New_pipeline_audioldm2 import AudioLDM2Pipeline
- 优先离线加载: local_files_only=True（失败再联网）
- 新增采样器切换: pipe.set_sampler("dpmpp", use_karras_sigmas=True)
- 支持通过环境变量 AUDIO_LDM2_MODEL_DIR 指定本地权重目录

运行验证
- 在 audioldm_env 下以 16 步推理成功，输出: smoke_test_sampler_output.wav（16kHz）

备注
- 若需完全离线运行，请先确保上述缓存目录完整或将模型权重复制到本地目录，并设置 AUDIO_LDM2_MODEL_DIR 指向该目录。
