from __future__ import annotations

"""
DDCM per-step 编解码（方法三雏形，逐步注入 variance_noise）
- 解码：在扩散循环的每一步 i 注入离散噪声 C_i(k_i)（来自本地持久化码本的原子），复现编码端的选择。
- 编码：占位（TODO）——需基于 forward 一致性或 epsilon 引导逐步选择 k_i。

与论文对齐的命名：
- 时间步用 i 表示；每步选择的索引用 k_i 表示；对应的噪声向量为 C_i(k_i)。
- σ_i 的缩放、时间网格与更新公式均由调度器（DDPM）负责。

注意：
- 码本不写入比特流。使用 (C,H,W), K_step, seed 以及 dtype 重建或从本地缓存加载。
- 回调签名已在 New_pipeline_audioldm2.py 扩展，可获得 latents 与 noise_pred。
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .codebook import load_or_create_codebook


@dataclass
class StepCodecConfig:
    K_step: int = 256  # 每步码本大小（对应论文中的 K）
    N_step: int = 1000  # 总采样步数（对应论文中的 T；默认对齐论文使用 1000 步）
    seed: int = 1234   # 码本种子（决定 C_i 的固定实现）
    guidance_gain: float = 1.0  # 全局注入强度缩放（>1 提高噪声能量）
    match_epsilon: bool = False  # 默认使用 x0 域残差匹配；将 ε 域作为可选对比模式
    atoms_per_step: int = 1  # Matching Pursuit 的原子数量 M（默认单原子）
    coeff_levels: Tuple[float, ...] = (
        0.0,
        0.0625,
        0.125,
        0.25,
        0.375,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
    )  # 量化系数候选集合（含 0 表示可跳步）
    record_diagnostics: bool = False  # 是否记录每步残差/系数诊断
    continuous_coeff: bool = False  # 是否使用连续系数（跳过量化）
    clap_weight: float = 0.0  # CLAP 语义相似度权重（0 表示禁用）
    clap_topk: int = 1  # 每次匹配时使用 CLAP 评估的候选原子数量
    clap_interval: int = 0  # 每隔多少步触发 CLAP 评估（0 表示每步）
    clap_model_name: Optional[str] = None  # 可选：自定义 CLAP 模型名
    clap_max_duration: float = 30.0  # CLAP 评估时裁剪的最长音频长度（秒）
    vggish_weight: float = 0.0  # VGGish 语义权重（0 表示禁用）
    vggish_topk: int = 1  # VGGish 复选候选数量
    vggish_interval: int = 0  # VGGish 复选间隔（0 表示每步）
    vggish_max_duration: float = 30.0  # VGGish 评估裁剪秒数


def _resolve_cache_dir() -> str:
    import os
    env_hf_home = os.environ.get("HF_HOME")
    env_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_hub_cache:
        return env_hub_cache
    if env_hf_home:
        return str(Path(env_hf_home) / "hub")
    return r"F:\\Kasai_Lab\\hf_cache\\huggingface\\hub"


def _latent_shape_from_model(pipe) -> Tuple[int,int,int]:
    # 依据当前 UNet/latents 约定：in_channels = C
    C = int(pipe.unet.config.in_channels)
    # 高宽（时间帧与频带）在推理前未知，这里由 VAE/mel 配置与采样长度共同决定。
    # 我们在实际运行回调时以动态形状为准；此处占位由编码端记录。
    raise NotImplementedError


def build_step_codebook(shape: Tuple[int,int,int], cfg: StepCodecConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    C, H, W = shape
    return load_or_create_codebook((C, H, W), K=cfg.K_step, seed=cfg.seed, device=device, dtype=dtype)


def _sample_step_noise_candidates(
    step_index: int,
    k_step: int,
    shape: Tuple[int, int, int],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(step_index))
    return torch.randn((int(k_step), int(shape[0]), int(shape[1]), int(shape[2])), device=device, dtype=dtype, generator=generator)


def _make_seeded_injection_callback(
    k_indices: Sequence,
    seed: int,
    k_step: int,
    shape: Tuple[int, int, int],
    scale_inject: float = 1.0,
    per_step_scales: Optional[Sequence[float]] = None,
    use_raw_atoms: bool = False,
    guidance_gain: float = 1.0,
):
    k_nested = _as_step_list(k_indices) or []

    def fn(i: int, t: torch.Tensor, latents: torch.Tensor, noise_pred: torch.Tensor, latent_shape: Tuple[int,...], device, dtype, scheduler):
        if i >= len(k_nested):
            return None
        entries = k_nested[i]
        if not entries:
            return None
        idx = int(entries[0])
        if idx < 0 or idx >= int(k_step):
            return None
        candidates = _sample_step_noise_candidates(i, int(k_step), shape, int(seed), device=device, dtype=dtype)
        atom = candidates[idx:idx+1]
        sigma_t = float(per_step_scales[i]) if per_step_scales is not None and i < len(per_step_scales) else float(scale_inject)
        if use_raw_atoms:
            return atom * float(guidance_gain)
        return atom * sigma_t * float(guidance_gain)

    return fn


# ------------------------ 解码（per-step 注入） ------------------------

def _as_step_list(data: Optional[Sequence], default_value: float = 1.0) -> Optional[List[List[float]]]:
    if data is None:
        return None
    if len(data) == 0:
        return []
    first = data[0]
    if isinstance(first, (list, tuple)):
        return [list(map(float, item)) for item in data]
    return [[float(item)] for item in data]


def _make_injection_callback(
    k_indices: Sequence,
    coeffs: Optional[Sequence],
    signs: Optional[Sequence],
    codebook: torch.Tensor,
    scale_inject: float = 1.0,  # Fallback scalar
    per_step_scales: Optional[Sequence[float]] = None,
    use_raw_atoms: bool = False,
    guidance_gain: float = 1.0,
):
    """
    生成一个闭包回调：在第 i 步返回注入的噪声 Σ_j s_ij * c_ij * C(k_ij)。
    """
    K = int(codebook.shape[0])
    k_nested = _as_step_list(k_indices) or []
    coeff_nested = _as_step_list(coeffs) if coeffs is not None else None
    sign_nested = _as_step_list(signs) if signs is not None else None

    step_entries: List[List[Tuple[int, float, float]]] = []
    for step_idx, ks in enumerate(k_nested):
        cs = coeff_nested[step_idx] if coeff_nested is not None and step_idx < len(coeff_nested) else [1.0] * len(ks)
        ss = sign_nested[step_idx] if sign_nested is not None and step_idx < len(sign_nested) else [1.0] * len(ks)
        entries: List[Tuple[int, float, float]] = []
        for local_idx, idx in enumerate(ks):
            coeff = float(cs[local_idx]) if local_idx < len(cs) else 1.0
            sign = 1.0
            if local_idx < len(ss):
                sign = 1.0 if float(ss[local_idx]) >= 0 else -1.0
            entries.append((int(idx), coeff, sign))
        step_entries.append(entries)

    def fn(i: int, t: torch.Tensor, latents: torch.Tensor, noise_pred: torch.Tensor, latent_shape: Tuple[int,...], device, dtype, scheduler):
        if i >= len(step_entries):
            return None
        entries = step_entries[i]
        if not entries:
            return None
        chunks = []
        for idx, coeff, sign in entries:
            if not (0 <= idx < K):
                continue
            atom = codebook[idx:idx+1].to(device=device, dtype=dtype)
            chunks.append(sign * coeff * atom)
        if not chunks:
            return None
        
        # scale holds sigma_t (or fallback scale_inject)
        sigma_t = float(per_step_scales[i]) if per_step_scales is not None and i < len(per_step_scales) else float(scale_inject)
        
        # If using raw atoms (for variance_noise), we return "gain * atom".
        # Scheduler will multiply by sigma_t. Result: sigma_t * gain * atom.
        if use_raw_atoms:
            return torch.stack(chunks, dim=0).sum(dim=0) * float(guidance_gain)
            
        # If manually injecting, we need "sigma_t * gain * atom".
        return torch.stack(chunks, dim=0).sum(dim=0) * sigma_t * float(guidance_gain)

    return fn


def decode_step_bitstream(bitstream: Dict, out_wav_path: Optional[str] = None) -> str:
    """
    根据 per-step 比特流进行解码：在每步调度器调用时注入 z_t。
    返回输出 wav 路径。
    """
    from New_pipeline_audioldm2 import AudioLDM2Pipeline

    meta = bitstream["meta"]
    stream = bitstream["stream"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Force local load to avoid network flakiness during decode; assumes weights already cached.
    pipe = AudioLDM2Pipeline.from_pretrained(
        meta["model_name"],
        torch_dtype=dtype,
        cache_dir=_resolve_cache_dir(),
        local_files_only=True,
    ).to(device)

    # 调度器对齐论文：强制使用 DDPM 日程（VP 噪声日程），并在下方设置推理步数
    try:
        try:
            from diffusers import DDPMScheduler  # type: ignore
        except Exception:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # type: ignore
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)  # type: ignore
    except Exception:
        print("[ddcm.step_codec] Warning: DDPMScheduler not available, keep existing scheduler.")

    shape = tuple(stream["shape"])  # [C,H,W]
    cfg_dict = dict(meta.get("cfg", {}))
    default_cfg = StepCodecConfig()
    coeff_levels = cfg_dict.get("coeff_levels", default_cfg.coeff_levels)
    if isinstance(coeff_levels, list):
        coeff_levels = tuple(float(x) for x in coeff_levels)
    elif not isinstance(coeff_levels, tuple):
        coeff_levels = tuple(default_cfg.coeff_levels)
    guidance_gain_val = float(cfg_dict.get("guidance_gain", default_cfg.guidance_gain))
    match_epsilon_val = cfg_dict.get("match_epsilon", default_cfg.match_epsilon)
    if isinstance(match_epsilon_val, str):
        match_epsilon_val = match_epsilon_val.lower() in {"1", "true", "yes"}
    record_diag_val = cfg_dict.get("record_diagnostics", default_cfg.record_diagnostics)
    if isinstance(record_diag_val, str):
        record_diag_val = record_diag_val.lower() in {"1", "true", "yes"}
    continuous_coeff_val = cfg_dict.get("continuous_coeff", default_cfg.continuous_coeff)
    if isinstance(continuous_coeff_val, str):
        continuous_coeff_val = continuous_coeff_val.lower() in {"1", "true", "yes"}
    clap_weight_val = float(cfg_dict.get("clap_weight", default_cfg.clap_weight))
    clap_topk_val = int(cfg_dict.get("clap_topk", default_cfg.clap_topk))
    clap_interval_val = int(cfg_dict.get("clap_interval", default_cfg.clap_interval))
    clap_model_val = cfg_dict.get("clap_model_name", default_cfg.clap_model_name)
    if isinstance(clap_model_val, str) and not clap_model_val:
        clap_model_val = None
    clap_max_duration_val = float(cfg_dict.get("clap_max_duration", default_cfg.clap_max_duration))
    vggish_weight_val = float(cfg_dict.get("vggish_weight", default_cfg.vggish_weight))
    vggish_topk_val = int(cfg_dict.get("vggish_topk", default_cfg.vggish_topk))
    vggish_interval_val = int(cfg_dict.get("vggish_interval", default_cfg.vggish_interval))
    vggish_max_duration_val = float(cfg_dict.get("vggish_max_duration", default_cfg.vggish_max_duration))
    cfg = StepCodecConfig(
        K_step=int(cfg_dict.get("K_step", default_cfg.K_step)),
        N_step=int(cfg_dict.get("N_step", default_cfg.N_step)),
        seed=int(cfg_dict.get("seed", default_cfg.seed)),
        guidance_gain=float(guidance_gain_val),
        match_epsilon=bool(match_epsilon_val),
        atoms_per_step=int(cfg_dict.get("atoms_per_step", default_cfg.atoms_per_step)),
        coeff_levels=coeff_levels,
        record_diagnostics=bool(record_diag_val),
        continuous_coeff=bool(continuous_coeff_val),
        clap_weight=float(clap_weight_val),
        clap_topk=int(clap_topk_val),
        clap_interval=int(clap_interval_val),
        clap_model_name=clap_model_val,
        clap_max_duration=float(clap_max_duration_val),
        vggish_weight=float(vggish_weight_val),
        vggish_topk=int(vggish_topk_val),
        vggish_interval=int(vggish_interval_val),
        vggish_max_duration=float(vggish_max_duration_val),
    )

    codec_style = str(cfg_dict.get("codec_style", "codebook_v1"))
    codebook = None
    if codec_style != "latent_runners_audio_v1":
        codebook = build_step_codebook(shape, cfg, device, dtype)

    def _posterior_std_list(scheduler, timesteps_tensor: torch.Tensor) -> List[float]:
        # Compute posterior std (sqrt(beta_tilde_t)) per sampled timestep
        betas = scheduler.betas.to(torch.float64)
        alphas_cumprod = scheduler.alphas_cumprod.to(torch.float64)
        scales: List[float] = []
        ts = timesteps_tensor.tolist()
        for idx, t_val in enumerate(ts):
            t_int = int(t_val)
            prev_t = int(ts[idx + 1]) if idx + 1 < len(ts) else -1
            beta_t = betas[t_int]
            alpha_prod_t = alphas_cumprod[t_int]
            alpha_prod_prev = alphas_cumprod[prev_t] if prev_t >= 0 else alphas_cumprod[0]
            posterior_var = beta_t * (1.0 - alpha_prod_prev) / max(float(1.0 - alpha_prod_t), 1e-12)
            posterior_var = float(max(posterior_var, 1e-20))
            scales.append(float(posterior_var ** 0.5))
        return scales

    # Per-step injection scaling using posterior std
    scale_inject = float(cfg.guidance_gain)  # fallback scalar，允许整体放大

    # 设置采样步数，并裁剪潜在越界的 t（部分版本可能给出 1000 而表大小为 1000）
    if hasattr(pipe.scheduler, "set_timesteps"):
        pipe.scheduler.set_timesteps(cfg.N_step, device=device)
        try:
            max_t = int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000)) - 1
        except Exception:
            max_t = 999
        try:
            _ts = []
            for _t in pipe.scheduler.timesteps:
                ti = int(_t) if not isinstance(_t, int) else _t
                if ti > max_t:
                    ti = max_t
                if ti < 0:
                    ti = 0
                _ts.append(ti)
            import torch as _torch
            pipe.scheduler.timesteps = _torch.tensor(_ts, device=device, dtype=getattr(pipe.scheduler.timesteps, "dtype", _torch.long))
        except Exception:
            pass

    # 命名与论文一致：k_i 序列。兼容旧字段名 'indices'。
    k_indices = stream.get("k_indices") or stream.get("indices")
    coeffs = stream.get("coeffs")
    # Per-step injection scales derived from posterior variance (sqrt(beta_tilde_t))
    try:
        per_step_scales = _posterior_std_list(pipe.scheduler, pipe.scheduler.timesteps)
        # Note: Guidance gain handled in callback
    except Exception:
        per_step_scales = None
    if codec_style == "latent_runners_audio_v1":
        cb = _make_seeded_injection_callback(
            k_indices,
            seed=int(cfg.seed),
            k_step=int(cfg.K_step),
            shape=shape,
            scale_inject=scale_inject,
            per_step_scales=per_step_scales,
            use_raw_atoms=True,
            guidance_gain=float(cfg.guidance_gain),
        )
    else:
        cb = _make_injection_callback(
            k_indices,
            coeffs,
            stream.get("signs"),
            codebook,
            scale_inject=scale_inject,
            per_step_scales=per_step_scales,
            use_raw_atoms=True,  # variance_noise expects standard-normal-scaled noise
            guidance_gain=float(cfg.guidance_gain),
        )
    pipe.set_ddcm_step_callback(cb)

    # 固定初始噪声（论文在 i=T+1 处使用 K=1 的码本；这里取 codebook[0] 并按调度器初始化尺度缩放）
    if codebook is not None:
        init = codebook[0:1].to(device=device, dtype=dtype)  # [1,C,H,W]
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(cfg.seed) + 100000)
        init = torch.randn((1, int(shape[0]), int(shape[1]), int(shape[2])), device=device, dtype=dtype, generator=generator)
    try:
        sigma0 = pipe.scheduler.init_noise_sigma  # type: ignore
    except Exception:
        sigma0 = 1.0
    latents0 = init * float(sigma0)
    if stream.get("init_latent") is not None:
        latents0 = torch.from_numpy(np.asarray(stream["init_latent"], dtype=np.float32)).to(device=device, dtype=dtype)

    # 运行一次无条件采样（或给一个空 prompt）；我们只利用 per-step 注入来重建。
    # 注意：这一步生成的是 mel 再到 waveform 的路径。
    generator = None
    latent_seed = meta.get("generator_seed")
    if latent_seed is not None:
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(latent_seed))
        except Exception:
            generator = torch.Generator()
            generator.manual_seed(int(latent_seed))

    pipe.set_progress_bar_config(disable=False)
    audio = pipe(prompt="", num_inference_steps=cfg.N_step, latents=latents0, generator=generator).audios[0]

    # 保存输出
    out_path = out_wav_path or (Path("evaluation_results") / "ddcm_step_reconstructed.wav")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(out_path), audio, int(meta.get("sr", 16000)))
    return str(out_path)


# ------------------------ 编码（占位） ------------------------

def encode_step_bitstream_placeholder(*args, **kwargs):
    raise NotImplementedError("TODO: 方法三前向一致性编码尚未实现。")


# ------------------------ 编码（式(7)：贪心每步选择 k_i） ------------------------

def encode_step_bitstream(
    wav_path: str,
    model_name: str,
    cfg: Optional[StepCodecConfig] = None,
    save_recon_path: Optional[str] = None,
) -> Dict:
    """
    根据论文式(7)，在每个时间步 i 选择 k_i = argmax_k < C_i(k), x0 - x^0|i >，
    生成 ddcm_step 比特流（仅存 k_indices 与形状，码本固定本地，不入流）。
    返回 {meta, stream}（内存字典）。
    """
    from New_pipeline_audioldm2 import AudioLDM2Pipeline
    from .audio_codec import _load_audio_mel_and_latent, _decode_latent_to_audio  # 复用 STEP1 对齐的 mel/latent 提取

    cfg = cfg or StepCodecConfig()
    # 限制线程，避免 Windows/CPU 上的潜在并发崩溃（MKL/OMP 线程冲突）
    try:
        import os as _os
        if "OMP_NUM_THREADS" not in _os.environ:
            _os.environ["OMP_NUM_THREADS"] = "1"
        if "MKL_NUM_THREADS" not in _os.environ:
            _os.environ["MKL_NUM_THREADS"] = "1"
        if "TQDM_DISABLE" not in _os.environ:
            _os.environ["TQDM_DISABLE"] = "1"
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 1) 加载管线并对齐调度器为 DDPM
    try:
        pipe = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            cache_dir=_resolve_cache_dir(),
            local_files_only=True,
        ).to(device)
    except Exception:
        pipe = AudioLDM2Pipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            cache_dir=_resolve_cache_dir(),
        ).to(device)
    try:
        try:
            from diffusers import DDPMScheduler  # type: ignore
        except Exception:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # type: ignore
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)  # type: ignore
    except Exception:
        print("[ddcm.step_codec] Warning: DDPMScheduler not available, keep existing scheduler.")

    # 2) 目标 latent（x0*）
    # 禁用梯度，纯推理
    with torch.no_grad():
        io = _load_audio_mel_and_latent(pipe, wav_path, device)
    x0_target: torch.Tensor = io["latent"].to(device=device, dtype=dtype)  # [1,C,H,W]

    source_sr = int(io.get("sr", 16000))
    orig_audio_np = io.get("wav_np", np.zeros(1, dtype=np.float32))
    orig_audio_np = np.asarray(orig_audio_np, dtype=np.float32)
    if orig_audio_np.ndim == 0:
        orig_audio_np = orig_audio_np[None]
    if orig_audio_np.size == 0:
        orig_audio_np = np.zeros(1, dtype=np.float32)
    orig_audio_np = np.clip(orig_audio_np, -1.0, 1.0)

    clap_enabled = bool(cfg.clap_weight and cfg.clap_weight > 0.0)
    vggish_enabled = bool(cfg.vggish_weight and cfg.vggish_weight > 0.0)
    feature_states: Dict[str, Dict] = {}
    feature_configs: List[Dict[str, object]] = []

    def _vggish_waveform_to_examples_torch(waveform: torch.Tensor, sample_rate: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Torch-based replacement for torchvggish.vggish_input.waveform_to_examples to avoid Windows mel_features crash.
        Returns tensor [N, 1, 96, 64] on the given device (defaults to waveform.device).
        """
        import torchaudio

        target_sr = 16000
        dev = device or waveform.device
        if waveform.ndim == 0:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(torch.float32)
        if waveform.numel() == 0:
            waveform = torch.zeros(1, dtype=torch.float32)
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sample_rate, target_sr).squeeze(0)

        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=400,          # 25 ms window
            win_length=400,
            hop_length=160,      # 10 ms hop
            f_min=125,
            f_max=7500,
            n_mels=64,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
        ).to(dev)
        mel = mel_tf(waveform.to(dev))
        log_mel = torch.log(mel + 1e-6)
        log_mel = log_mel.transpose(0, 1)  # [frames, mels]

        frames_per_example = 96  # 0.96 s @10 ms hop
        if log_mel.shape[0] < frames_per_example:
            pad_frames = frames_per_example - log_mel.shape[0]
            pad = torch.zeros(pad_frames, log_mel.shape[1], dtype=log_mel.dtype, device=log_mel.device)
            log_mel = torch.cat([log_mel, pad], dim=0)

        chunks: List[torch.Tensor] = []
        start = 0
        while start < log_mel.shape[0]:
            end = start + frames_per_example
            chunk = log_mel[start:end]
            if chunk.shape[0] < frames_per_example:
                pad = torch.zeros(frames_per_example - chunk.shape[0], log_mel.shape[1], dtype=log_mel.dtype, device=log_mel.device)
                chunk = torch.cat([chunk, pad], dim=0)
            chunks.append(chunk)
            start += frames_per_example
        if not chunks:
            chunks.append(torch.zeros(frames_per_example, 64, dtype=log_mel.dtype, device=log_mel.device))

        stacked = torch.stack(chunks, dim=0)  # [N, 96, 64]
        return stacked.unsqueeze(1)  # [N, 1, 96, 64]

    if clap_enabled:
        from transformers import ClapModel, ClapProcessor
        import torchaudio

        clap_eval_interval = max(0, int(cfg.clap_interval))
        clap_topk = max(1, int(cfg.clap_topk))
        clap_model_name = cfg.clap_model_name or "laion/clap-htsat-unfused"
        clap_device = device if device.type == "cuda" else torch.device("cpu")
        clap_processor = ClapProcessor.from_pretrained(clap_model_name)
        clap_model = ClapModel.from_pretrained(clap_model_name, use_safetensors=True).to(clap_device)
        clap_model.eval()
        clap_sr = int(getattr(clap_processor.feature_extractor, "sampling_rate", 48000))
        clap_max_len = int(max(0.1, float(cfg.clap_max_duration)) * clap_sr)

        audio_for_clap = torch.from_numpy(orig_audio_np).to(torch.float32)
        if source_sr != clap_sr:
            audio_for_clap = torchaudio.functional.resample(audio_for_clap.unsqueeze(0), source_sr, clap_sr).squeeze(0)
        if audio_for_clap.numel() > clap_max_len:
            audio_for_clap = audio_for_clap[:clap_max_len]
        target_inputs = clap_processor(audio=[audio_for_clap.cpu().numpy()], sampling_rate=clap_sr, return_tensors="pt")
        target_inputs = {k: v.to(clap_device) for k, v in target_inputs.items()}
        with torch.no_grad():
            target_embed = clap_model.get_audio_features(**target_inputs)
        target_embed = torch.nn.functional.normalize(target_embed, dim=-1)

        feature_states["clap"] = {
            "processor": clap_processor,
            "model": clap_model,
            "device": clap_device,
            "target": target_embed,
            "sr": clap_sr,
            "max_len": clap_max_len,
        }
        feature_configs.append({
            "name": "clap",
            "weight": float(cfg.clap_weight),
            "interval": clap_eval_interval,
            "topk": clap_topk,
        })

    if vggish_enabled:
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torchaudio
        import torchvggish

        vggish_eval_interval = max(0, int(cfg.vggish_interval))
        vggish_topk = max(1, int(cfg.vggish_topk))
        vggish_sr = 16000
        vggish_max_len = int(max(0.1, float(cfg.vggish_max_duration)) * vggish_sr)
        vggish_device = device if device.type == "cuda" else torch.device("cpu")
        vggish_model = torchvggish.vggish().to(vggish_device)
        # torchvggish postprocess (PCA/quant) keeps params on CPU; disable to avoid device mismatch on CUDA
        vggish_model.postprocess = False
        vggish_model.eval()

        audio_for_vggish = torch.from_numpy(orig_audio_np).to(torch.float32)
        if source_sr != vggish_sr:
            audio_for_vggish = torchaudio.functional.resample(audio_for_vggish.unsqueeze(0), source_sr, vggish_sr).squeeze(0)
        if audio_for_vggish.numel() > vggish_max_len:
            audio_for_vggish = audio_for_vggish[:vggish_max_len]
        with torch.no_grad():
            examples_tensor = _vggish_waveform_to_examples_torch(audio_for_vggish.cpu(), vggish_sr, device=vggish_device).to(vggish_device, torch.float32)
            embed = vggish_model(examples_tensor)
            if embed.ndim == 1:
                embed = embed.unsqueeze(0)
            embed = torch.nn.functional.normalize(embed, dim=-1)
            target_vggish = torch.nn.functional.normalize(embed.mean(dim=0, keepdim=True), dim=-1)

        feature_states["vggish"] = {
            "model": vggish_model,
            "device": vggish_device,
            "target": target_vggish,
            "sr": vggish_sr,
            "max_len": vggish_max_len,
            "input_fn": _vggish_waveform_to_examples_torch,
        }
        feature_configs.append({
            "name": "vggish",
            "weight": float(cfg.vggish_weight),
            "interval": vggish_eval_interval,
            "topk": vggish_topk,
        })

    def _compute_feature_cosines(latent_tensor: torch.Tensor, need_clap: bool, need_vggish: bool) -> Dict[str, float]:
        results: Dict[str, float] = {}
        if not (need_clap or need_vggish):
            return results
        with torch.no_grad():
            audio_np = _decode_latent_to_audio(pipe, latent_tensor.to(device=device, dtype=dtype))
        audio = torch.from_numpy(audio_np).to(torch.float32)
        if audio.ndim == 0:
            audio = audio.unsqueeze(0)
        audio = audio.clamp(-1.0, 1.0)
        if need_clap and "clap" in feature_states:
            import torchaudio
            state = feature_states["clap"]
            clap_audio = audio
            if source_sr != state["sr"]:
                clap_audio = torchaudio.functional.resample(clap_audio.unsqueeze(0), source_sr, state["sr"]).squeeze(0)
            if clap_audio.numel() > state["max_len"]:
                clap_audio = clap_audio[:state["max_len"]]
            inputs = state["processor"](audio=[clap_audio.cpu().numpy()], sampling_rate=state["sr"], return_tensors="pt")
            inputs = {k: v.to(state["device"]) for k, v in inputs.items()}
            with torch.no_grad():
                embed = state["model"].get_audio_features(**inputs)
            embed = torch.nn.functional.normalize(embed, dim=-1)
            results["clap"] = float(torch.sum(embed * state["target"]).item())
        if need_vggish and "vggish" in feature_states:
            import torchaudio
            state = feature_states["vggish"]
            vgg_audio = audio
            if source_sr != state["sr"]:
                vgg_audio = torchaudio.functional.resample(vgg_audio.unsqueeze(0), source_sr, state["sr"]).squeeze(0)
            if vgg_audio.numel() > state["max_len"]:
                vgg_audio = vgg_audio[:state["max_len"]]
            with torch.no_grad():
                examples_tensor = state["input_fn"](vgg_audio.cpu(), state["sr"], device=state["device"]).to(state["device"], torch.float32)
                embed = state["model"](examples_tensor)
                if embed.ndim == 1:
                    embed = embed.unsqueeze(0)
                embed = torch.nn.functional.normalize(embed, dim=-1)
                embed = torch.nn.functional.normalize(embed.mean(dim=0, keepdim=True), dim=-1)
            results["vggish"] = float(torch.sum(embed * state["target"]).item())
        return results

    # 3) 形状（latent_runners 风格：每步按 seed 生成 K 个随机候选，不使用持久化码本）
    shape = (int(x0_target.shape[1]), int(x0_target.shape[2]), int(x0_target.shape[3]))

    # 4) 时间步 & 初始噪声（论文 i=T+1 处 K=1；取 codebook[0] 作为 x_T）
    if hasattr(pipe.scheduler, "set_timesteps"):
        pipe.scheduler.set_timesteps(int(cfg.N_step), device=device)
    timesteps = pipe.scheduler.timesteps

    def _posterior_std_list(scheduler, timesteps_tensor: torch.Tensor) -> List[float]:
        betas = scheduler.betas.to(torch.float64)
        alphas_cumprod = scheduler.alphas_cumprod.to(torch.float64)
        scales: List[float] = []
        ts = timesteps_tensor.tolist()
        for idx, t_val in enumerate(ts):
            t_int = int(t_val)
            prev_t = int(ts[idx + 1]) if idx + 1 < len(ts) else -1
            beta_t = betas[t_int]
            alpha_prod_t = alphas_cumprod[t_int]
            alpha_prod_prev = alphas_cumprod[prev_t] if prev_t >= 0 else alphas_cumprod[0]
            posterior_var = beta_t * (1.0 - alpha_prod_prev) / max(float(1.0 - alpha_prod_t), 1e-12)
            posterior_var = float(max(posterior_var, 1e-20))
            scales.append(float(posterior_var ** 0.5))
        return scales

    scale_inject = float((100.0 / float(cfg.N_step)) ** 0.5) if cfg.N_step > 0 else 1.0
    # Note: guidance_gain applied during loop logic
    # 一些 diffusers 版本可能在 full-steps 下给出越界的首元素（例如 1000），这里做安全裁剪
    try:
        max_t = int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000)) - 1
    except Exception:
        max_t = 999
    _ts = []
    for _t in timesteps:
        ti = int(_t) if not isinstance(_t, int) else _t
        if ti > max_t:
            ti = max_t
        if ti < 0:
            ti = 0
        _ts.append(ti)
    import torch as _torch
    timesteps = _torch.tensor(_ts, device=device, dtype=getattr(timesteps, "dtype", _torch.long))
    # 使用前向扩散公式对目标 x0 施加噪声，得到与调度器一致的初始 latent（替代固定 codebook[0]）。
    try:
        sigma0 = pipe.scheduler.init_noise_sigma  # type: ignore
    except Exception:
        sigma0 = 1.0

    # 取首个时间步的 α_bar 以对齐调度器时间网格
    t0 = int(timesteps[0]) if len(timesteps) > 0 else 0
    alpha_bar_t0 = pipe.scheduler.alphas_cumprod[t0] if hasattr(pipe.scheduler, "alphas_cumprod") else None
    gen = torch.Generator(device=device)
    try:
        gen.manual_seed(int(cfg.seed))
    except Exception:
        pass

    if alpha_bar_t0 is not None:
        sqrt_ab = torch.sqrt(alpha_bar_t0).to(device=device, dtype=dtype)
        sqrt_omb = torch.sqrt((1.0 - alpha_bar_t0).to(device=device, dtype=dtype) + torch.finfo(dtype).eps)
        noise = torch.randn(x0_target.shape, device=device, dtype=dtype, generator=gen)
        latents = sqrt_ab * x0_target + sqrt_omb * noise
    else:
        latents = torch.randn(x0_target.shape, device=device, dtype=dtype, generator=gen) * float(sigma0)

    init_latent_np = latents.detach().to(torch.float32).cpu().numpy()

    try:
        per_step_scales = _posterior_std_list(pipe.scheduler, timesteps)
    except Exception:
        per_step_scales = None

    # 5) 准备无条件文本嵌入（保持与解码端一致）
    prompt=""
    do_cfg = False
    with torch.inference_mode():
        prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            transcription=None,
            negative_prompt=None,
        )

    # 6) 主循环：每步估计 x^0|i，并按 latent_runners 逻辑选 k_i = argmax <noise_k, residual>
    cfg.atoms_per_step = max(1, int(cfg.atoms_per_step))

    step_indices: List[List[int]] = []
    step_coeffs: List[List[float]] = []
    step_signs: List[List[int]] = []

    diag_residual_before: List[float] = []
    diag_residual_after: List[float] = []
    diag_best_abs_score: List[float] = []
    diag_coeff_raw: List[List[float]] = []
    diag_coeff_quant: List[List[float]] = []
    diag_atoms_used: List[int] = []
    diag_injection_norm: List[float] = []
    diag_injection_scale: List[float] = []
    diag_clap_cos: List[Optional[float]] = []
    diag_vggish_cos: List[Optional[float]] = []
    record_diag = bool(cfg.record_diagnostics)

    import inspect as _inspect
    accepts_variance_noise = "variance_noise" in set(_inspect.signature(pipe.scheduler.step).parameters.keys())

    for i, t in enumerate(tqdm(timesteps, total=timesteps.shape[0] if hasattr(timesteps, 'shape') else len(timesteps), desc="ddcm encode")):
        # 每个扩散时间步：估计当前 x0|i，并在 step-specific 噪声候选中选择最佳 k_i
        with torch.inference_mode():
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=generated_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=attention_mask,
                return_dict=False,
            )[0]

            tmp = pipe.scheduler.step(noise_pred, t, latents)
            x0_hat = tmp.pred_original_sample

        use_x0 = True
        residual_target: torch.Tensor
        if cfg.match_epsilon:
            with torch.no_grad():
                alpha_bar_t = pipe.scheduler.alphas_cumprod[int(t)] if hasattr(pipe.scheduler, "alphas_cumprod") else None
                if alpha_bar_t is not None:
                    use_x0 = False
                    sqrt_ab = torch.sqrt(alpha_bar_t).to(device=device, dtype=dtype)
                    one_minus_ab = (1.0 - alpha_bar_t).to(device=device, dtype=dtype)
                    sqrt_omb = torch.sqrt(one_minus_ab + torch.finfo(dtype).eps)
                    eps_target = (latents - sqrt_ab * x0_target) / sqrt_omb
                    residual_target = (eps_target - noise_pred).to(device=device, dtype=dtype)
        if use_x0:
            residual_target = (x0_target - x0_hat).to(device=device, dtype=dtype)

        selected_indices: List[int] = []
        selected_coeffs: List[float] = []
        selected_signs: List[int] = []
        step_coeff_raw: List[float] = []
        step_coeff_quant: List[float] = []
        step_best_abs_score = 0.0
        step_clap_cos = None
        step_vggish_cos = None
        residual_norm_before = float(residual_target.detach().to(dtype=torch.float32).norm().item())

        sigma_t = float(per_step_scales[i]) if per_step_scales is not None and i < len(per_step_scales) else float(scale_inject)
        if accepts_variance_noise:
            step_scale = float(cfg.guidance_gain)
        else:
            step_scale = sigma_t * float(cfg.guidance_gain)

        if int(t) >= 1:
            candidates = _sample_step_noise_candidates(i, int(cfg.K_step), shape, int(cfg.seed), device=device, dtype=dtype)
            residual_flat32 = residual_target.reshape(-1).to(dtype=torch.float32)
            scores = torch.matmul(candidates.view(candidates.shape[0], -1).to(dtype=torch.float32), residual_flat32)
            best_idx = int(torch.argmax(scores).item())
            best_noise = candidates[best_idx:best_idx + 1]
            selected_indices.append(best_idx)
            step_best_abs_score = float(torch.abs(scores[best_idx]).item())
        else:
            # 与 latent_runners 一致：最后一步不优化，使用固定随机噪声
            candidates = _sample_step_noise_candidates(i, 1, shape, int(cfg.seed), device=device, dtype=dtype)
            best_noise = candidates[0:1]
            step_best_abs_score = 0.0

        z_t = best_noise * step_scale
        residual_work = residual_target - z_t

        step_indices.append(selected_indices)
        step_coeffs.append(selected_coeffs)
        step_signs.append(selected_signs)

        with torch.inference_mode():
            if accepts_variance_noise:
                out = pipe.scheduler.step(noise_pred, t, latents, variance_noise=z_t)
                latents = out.prev_sample
            else:
                out = pipe.scheduler.step(noise_pred, t, latents)
                latents = out.prev_sample
                try:
                    latents = latents + z_t
                except Exception:
                    pass

        if record_diag:
            residual_norm_after = float(residual_work.detach().to(dtype=torch.float32).norm().item())
            z_norm = float(z_t.detach().to(dtype=torch.float32).norm().item())
            diag_residual_before.append(residual_norm_before)
            diag_residual_after.append(residual_norm_after)
            diag_best_abs_score.append(step_best_abs_score)
            diag_coeff_raw.append(step_coeff_raw)
            diag_coeff_quant.append(step_coeff_quant)
            diag_atoms_used.append(len(selected_indices))
            diag_injection_norm.append(z_norm)
            diag_injection_scale.append(float(step_scale))
            if clap_enabled:
                diag_clap_cos.append(float(step_clap_cos) if step_clap_cos is not None else None)
            else:
                diag_clap_cos.append(None)
            if vggish_enabled:
                diag_vggish_cos.append(float(step_vggish_cos) if step_vggish_cos is not None else None)
            else:
                diag_vggish_cos.append(None)

    def _maybe_flatten(entries: List[List]) -> List:
        if all(len(item) == 1 for item in entries):
            return [item[0] for item in entries]
        return entries

    meta = {
        "mode": "ddcm_step",
        "model_name": model_name,
        "cfg": {
            "K_step": int(cfg.K_step),
            "N_step": int(cfg.N_step),
            "seed": int(cfg.seed),
            "codec_style": "latent_runners_audio_v1",
            "guidance_gain": float(cfg.guidance_gain),
            "match_epsilon": bool(cfg.match_epsilon),
            "atoms_per_step": int(cfg.atoms_per_step),
            "coeff_levels": list(cfg.coeff_levels),
            "continuous_coeff": bool(cfg.continuous_coeff),
            "clap_weight": float(cfg.clap_weight),
            "clap_topk": int(cfg.clap_topk),
            "clap_interval": int(cfg.clap_interval),
            "clap_model_name": cfg.clap_model_name or "",
            "clap_max_duration": float(cfg.clap_max_duration),
            "vggish_weight": float(cfg.vggish_weight),
            "vggish_topk": int(cfg.vggish_topk),
            "vggish_interval": int(cfg.vggish_interval),
            "vggish_max_duration": float(cfg.vggish_max_duration),
        },
        "sr": int(io.get("sr", 16000)),
        "generator_seed": int(cfg.seed),
        "scale_inject_mode": "posterior_std",
    }
    if record_diag:
        meta["diagnostics"] = {
            "residual_norm_before": diag_residual_before,
            "residual_norm_after": diag_residual_after,
            "best_abs_score": diag_best_abs_score,
            "coeff_raw": diag_coeff_raw,
            "coeff_quant": diag_coeff_quant,
            "atoms_used": diag_atoms_used,
            "injection_norm": diag_injection_norm,
            "injection_scale": diag_injection_scale,
            "clap_cos": diag_clap_cos,
            "vggish_cos": diag_vggish_cos,
        }

    flat_indices = _maybe_flatten(step_indices)
    coeff_entries = _maybe_flatten(step_coeffs) if any(len(item) > 0 for item in step_coeffs) else None
    sign_entries = _maybe_flatten(step_signs) if any(len(item) > 0 for item in step_signs) else None

    stream = {
        "mode": "ddcm_step",
        "k_indices": flat_indices,
        "coeffs": coeff_entries,
        "signs": sign_entries,
        "shape": [int(x0_target.shape[1]), int(x0_target.shape[2]), int(x0_target.shape[3])],
        "init_latent": init_latent_np,
        "init_from_noised_x0": True,
    }

    # 可选：将编码结束时的 latent 解码为波形，方便直接对比输入音频
    if save_recon_path:
        audio_np = _decode_latent_to_audio(pipe, latents.to(device=device, dtype=dtype))
        recon_path = Path(save_recon_path)
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        sf.write(str(recon_path), audio_np, int(io.get("sr", 16000)))
        meta["encode_recon_wav"] = str(recon_path)

    return {"meta": meta, "stream": stream}
