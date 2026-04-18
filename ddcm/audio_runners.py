from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .audio_models import AudioLDM2Wrapper


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimized_step_count(timesteps: torch.Tensor, t_range: Tuple[int, int]) -> int:
    count = 0
    for t in timesteps:
        ti = int(t)
        if ti >= 1 and (t_range[0] >= ti >= t_range[1]):
            count += 1
    return count


def _endpoint_latent_stats(x_final: torch.Tensor, x_target: torch.Tensor) -> Dict[str, float]:
    xf = x_final.detach().to(torch.float32)
    xt = x_target.detach().to(torch.float32)

    diff = xf - xt
    mse = float(torch.mean(diff * diff).item())
    l2 = float(torch.norm(diff).item())

    xf_flat = xf.reshape(-1)
    xt_flat = xt.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(xf_flat.unsqueeze(0), xt_flat.unsqueeze(0), dim=1)
    cosine = float(cos.item())

    target_norm = float(torch.norm(xt).item())
    final_norm = float(torch.norm(xf).item())
    rel_l2 = float(l2 / (target_norm + 1e-12))

    return {
        "final_latent_mse": mse,
        "final_latent_l2": l2,
        "final_latent_rel_l2": rel_l2,
        "final_latent_cosine": cosine,
        "final_latent_norm": final_norm,
        "target_latent_norm": target_norm,
    }


def compress(
    model: AudioLDM2Wrapper,
    wav_to_compress: Optional[str],
    num_noises: int,
    device: torch.device,
    num_noises_late: int = 0,
    num_noises_switch_t: int = -1,
    num_pursuit_noises: int = 1,
    num_pursuit_coef_bits: int = 3,
    pursuit_renorm: bool = True,
    score_mode: str = "dot",
    score_blend_lambda: float = 0.5,
    score_mode_late: str = "",
    score_switch_t: int = -1,
    eta: float = 1.0,
    eta_late: float = -1.0,
    eta_switch_t: int = -1,
    exact_rerank_topk: int = 0,
    mel_proxy_topk: int = 0,
    mel_proxy_interval: int = 10,
    audio_proxy_topk: int = 0,
    audio_proxy_interval: int = 10,
    t_range: Tuple[int, int] = (999, 0),
    decompress_indices: Optional[Dict[str, List[int]]] = None,
    init_noise_seed: int = 100000,
    step_noise_seed: int = 0,
) -> Dict:
    decompress = wav_to_compress is None
    if decompress and decompress_indices is None:
        raise ValueError("Either wav_to_compress or decompress_indices must be provided")

    model.set_timesteps(model.num_timesteps, device=device)
    prompt_state = model.encode_prompt("")

    sr = 16000
    if not decompress:
        io = model.encode_audio_to_latent(wav_to_compress)
        x0_target = io["latent"]
        mel_target = io.get("mel")
        wav_target = io.get("wav_np")
        latent_shape = tuple(int(x) for x in x0_target.shape[1:])
        sr = int(io.get("sr", 16000))
    else:
        shape = decompress_indices.get("shape")
        if shape is None:
            raise ValueError("decompress_indices missing shape")
        latent_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        x0_target = None
        mel_target = None
        wav_target = None

    set_seed(init_noise_seed)
    xt = torch.randn((1, *latent_shape), device=device, dtype=model.dtype)
    init_latent_np = xt.detach().to(torch.float32).cpu().numpy()

    noise_indices: List = []
    coeff_indices: List = []
    curr_idx = 0

    diag_before: List[float] = []
    diag_after: List[float] = []
    diag_best: List[float] = []
    diag_inj_norm: List[float] = []
    diag_inj_scale: List[float] = []
    diag_rerank_improve: List[float] = []
    diag_mel_proxy_mse: List[float] = []
    diag_audio_proxy_mse: List[float] = []
    selected_noise_l2: List[float] = []
    selected_noise_sum: List[float] = []

    optimized_steps = _optimized_step_count(model.timesteps, t_range)
    score_mode = str(score_mode).lower()
    if score_mode not in {"dot", "cosine", "blend"}:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    score_blend_lambda = float(score_blend_lambda)
    score_mode_late = str(score_mode_late).lower().strip()
    if score_mode_late and score_mode_late not in {"dot", "cosine", "blend"}:
        raise ValueError(f"Unsupported score_mode_late: {score_mode_late}")

    eta = float(eta)
    eta_late = float(eta_late)

    def _eta_for_t(timestep: int) -> float:
        cur = float(eta)
        if eta_late >= 0.0 and int(eta_switch_t) >= 0 and int(timestep) <= int(eta_switch_t):
            cur = float(eta_late)
        if cur < 0.0:
            cur = 0.0
        return cur

    def _num_noises_for_t(timestep: int) -> int:
        base = int(max(int(num_noises), 1))
        late = int(max(int(num_noises_late), 0))
        if late > 0 and int(num_noises_switch_t) >= 0 and int(timestep) <= int(num_noises_switch_t):
            return int(late)
        return int(base)

    pbar = tqdm(model.timesteps, desc="audio ddcm")
    for step_idx, t in enumerate(pbar):
        set_seed(step_noise_seed + step_idx)
        optimize_t = (t_range[0] >= int(t) >= t_range[1])
        step_num_noises = _num_noises_for_t(int(t))

        candidates = torch.randn(
            (step_num_noises if optimize_t else 1, *latent_shape),
            device=device,
            dtype=model.dtype,
        )

        with torch.inference_mode():
            epst = model.get_epst(xt, t, prompt_state)
            x0_hat = model.get_x0_hat(xt, epst, t)

        if x0_target is not None:
            residual = x0_target - x0_hat
            residual_norm_before = float(residual.detach().to(torch.float32).norm().item())
            diag_before.append(residual_norm_before)
        else:
            residual = None

        best_abs_score = 0.0
        if int(t) >= 1 and optimize_t:
            if not decompress:
                residual_flat = residual.view(-1).to(torch.float32)
                residual_norm = residual_flat.norm().clamp_min(1e-12)
                current_score_mode = score_mode
                if score_mode_late and int(score_switch_t) >= 0 and int(t) <= int(score_switch_t):
                    current_score_mode = score_mode_late

                def _score(noise_batch: torch.Tensor) -> torch.Tensor:
                    flat = noise_batch.view(noise_batch.shape[0], -1).to(torch.float32)
                    dots = torch.matmul(flat, residual_flat)
                    if current_score_mode == "cosine":
                        norms = flat.norm(dim=1).clamp_min(1e-12)
                        return dots / (norms * residual_norm)
                    if current_score_mode == "blend":
                        norms = flat.norm(dim=1).clamp_min(1e-12)
                        cos = dots / (norms * residual_norm)
                        dots_z = (dots - dots.mean()) / dots.std().clamp_min(1e-6)
                        cos_z = (cos - cos.mean()) / cos.std().clamp_min(1e-6)
                        return dots_z + float(score_blend_lambda) * cos_z
                    return dots

                cand_scores = _score(candidates)
                best_idx = int(torch.argmax(cand_scores).item())
                best_abs_score = float(torch.abs(cand_scores[best_idx]).item())

                rerank_gain = 0.0
                rerank_topk = int(min(max(int(exact_rerank_topk), 0), int(candidates.shape[0])))
                if rerank_topk > 1:
                    topk_idx = torch.topk(cand_scores, k=rerank_topk).indices.tolist()
                    base_score = float(cand_scores[best_idx].item())
                    best_rerank_idx = best_idx
                    best_rerank_res = float("inf")
                    with torch.inference_mode():
                        for cand_idx in topk_idx:
                            cand_noise = candidates[int(cand_idx) : int(cand_idx) + 1]
                            eta_cur = _eta_for_t(int(t))
                            xt_try = model.finish_step(xt, x0_hat, epst, t, cand_noise, eta=eta_cur)
                            epst_try = model.get_epst(xt_try, t, prompt_state)
                            x0_try = model.get_x0_hat(xt_try, epst_try, t)
                            res_try = x0_target - x0_try
                            res_norm_try = float(res_try.detach().to(torch.float32).norm().item())
                            if res_norm_try < best_rerank_res:
                                best_rerank_res = res_norm_try
                                best_rerank_idx = int(cand_idx)
                    best_idx = int(best_rerank_idx)
                    rerank_gain = float(cand_scores[best_idx].item() - base_score)

                mel_proxy_k = int(min(max(int(mel_proxy_topk), 0), int(candidates.shape[0])))
                use_mel_proxy = (
                    mel_proxy_k > 1
                    and mel_target is not None
                    and int(max(int(mel_proxy_interval), 1)) > 0
                    and (int(step_idx) % int(max(int(mel_proxy_interval), 1)) == 0)
                )
                mel_proxy_best_mse = float("nan")
                if use_mel_proxy:
                    topk_idx = torch.topk(cand_scores, k=mel_proxy_k).indices.tolist()
                    best_mel_idx = best_idx
                    best_mel_mse = float("inf")
                    with torch.inference_mode():
                        for cand_idx in topk_idx:
                            cand_noise = candidates[int(cand_idx) : int(cand_idx) + 1]
                            eta_cur = _eta_for_t(int(t))
                            xt_try = model.finish_step(xt, x0_hat, epst, t, cand_noise, eta=eta_cur)

                            lat_for_dec = xt_try / model.pipe.vae.config.scaling_factor
                            vae_dtype = next(model.pipe.vae.parameters()).dtype
                            mel_try = model.pipe.vae.decode(lat_for_dec.to(vae_dtype)).sample
                            mel_err = mel_try.to(torch.float32) - mel_target.to(torch.float32)
                            mel_mse = float(torch.mean(mel_err * mel_err).item())
                            if mel_mse < best_mel_mse:
                                best_mel_mse = mel_mse
                                best_mel_idx = int(cand_idx)

                    if best_mel_mse < float("inf"):
                        best_idx = int(best_mel_idx)
                        mel_proxy_best_mse = float(best_mel_mse)

                proxy_topk = int(min(max(int(audio_proxy_topk), 0), int(candidates.shape[0])))
                use_audio_proxy = (
                    proxy_topk > 1
                    and wav_target is not None
                    and int(max(int(audio_proxy_interval), 1)) > 0
                    and (int(step_idx) % int(max(int(audio_proxy_interval), 1)) == 0)
                )
                proxy_best_mse = float("nan")
                if use_audio_proxy:
                    topk_idx = torch.topk(cand_scores, k=proxy_topk).indices.tolist()
                    best_proxy_idx = best_idx
                    best_proxy_mse = float("inf")
                    for cand_idx in topk_idx:
                        with torch.inference_mode():
                            cand_noise = candidates[int(cand_idx) : int(cand_idx) + 1]
                            eta_cur = _eta_for_t(int(t))
                            xt_try = model.finish_step(xt, x0_hat, epst, t, cand_noise, eta=eta_cur)
                            wav_try = model.decode_latent_to_audio(xt_try)
                        n = min(len(wav_try), len(wav_target))
                        if n <= 0:
                            continue
                        mse = float(np.mean((wav_try[:n] - wav_target[:n]) ** 2))
                        if mse < best_proxy_mse:
                            best_proxy_mse = mse
                            best_proxy_idx = int(cand_idx)
                    if best_proxy_mse < float("inf"):
                        best_idx = int(best_proxy_idx)
                        proxy_best_mse = float(best_proxy_mse)

                best_noise = candidates[best_idx]
                t_noises: List[int] = [best_idx]
                t_coeffs: List[int] = []

                if int(num_pursuit_noises) > 1:
                    pursuit_levels = torch.linspace(
                        0.0,
                        1.0,
                        2 ** int(num_pursuit_coef_bits),
                        device=candidates.device,
                        dtype=torch.float32,
                    )[1:]
                    best_dot_val = cand_scores[best_idx].to(torch.float32)

                    for _ in range(int(num_pursuit_noises) - 1):
                        next_best_noise = best_noise
                        next_noise_idx = 0
                        next_coeff_idx = 0

                        for coeff_idx, pursuit_coef in enumerate(pursuit_levels, 1):
                            mixed = best_noise.unsqueeze(0) * torch.sqrt(pursuit_coef) + candidates * torch.sqrt(1.0 - pursuit_coef)
                            if bool(pursuit_renorm):
                                mixed_std = mixed.view(mixed.shape[0], -1).std(1).view(mixed.shape[0], 1, 1, 1).clamp_min(1e-6)
                                mixed = mixed / mixed_std
                            cur_dot = _score(mixed)
                            cur_best_idx = int(torch.argmax(cur_dot).item())
                            cur_best_val = cur_dot[cur_best_idx].to(torch.float32)
                            if cur_best_val > best_dot_val:
                                best_dot_val = cur_best_val
                                next_best_noise = mixed[cur_best_idx]
                                next_noise_idx = cur_best_idx
                                next_coeff_idx = coeff_idx

                        if next_coeff_idx == 0:
                            break
                        best_noise = next_best_noise
                        t_noises.append(int(next_noise_idx))
                        t_coeffs.append(int(next_coeff_idx))

                if int(num_pursuit_noises) > 1:
                    noise_indices.append(t_noises)
                    coeff_indices.append(t_coeffs)
                else:
                    noise_indices.append(best_idx)
                diag_rerank_improve.append(float(rerank_gain))
                diag_mel_proxy_mse.append(float(mel_proxy_best_mse))
                diag_audio_proxy_mse.append(float(proxy_best_mse))
                best_noise = best_noise.unsqueeze(0)
            else:
                step_noise_info = decompress_indices["noise_indices"][curr_idx]
                if isinstance(step_noise_info, list):
                    t_noises = [int(v) for v in step_noise_info]
                else:
                    t_noises = [int(step_noise_info)]
                step_coeff_info = []
                if decompress_indices.get("coeff_indices") is not None:
                    raw_coeff = decompress_indices["coeff_indices"][curr_idx]
                    if isinstance(raw_coeff, list):
                        step_coeff_info = [int(v) for v in raw_coeff]

                best_noise = candidates[t_noises[0]]
                if int(num_pursuit_noises) > 1 and len(t_noises) > 1 and len(step_coeff_info) > 0:
                    pursuit_levels = torch.linspace(
                        0.0,
                        1.0,
                        2 ** int(num_pursuit_coef_bits),
                        device=candidates.device,
                        dtype=torch.float32,
                    )
                    for pursuit_idx in range(min(len(step_coeff_info), len(t_noises) - 1)):
                        coeff_idx = int(step_coeff_info[pursuit_idx])
                        if coeff_idx <= 0:
                            break
                        coeff_val = pursuit_levels[coeff_idx]
                        pursuit_noise = candidates[t_noises[pursuit_idx + 1]]
                        best_noise = best_noise * torch.sqrt(coeff_val) + pursuit_noise * torch.sqrt(1.0 - coeff_val)
                        if bool(pursuit_renorm):
                            best_noise = best_noise / best_noise.std().clamp_min(1e-6)
                curr_idx += 1
                best_noise = best_noise.unsqueeze(0)
        else:
            best_noise = candidates[0:1]

        selected_noise_l2.append(float(torch.norm(best_noise.detach().to(torch.float32)).item()))
        selected_noise_sum.append(float(best_noise.detach().to(torch.float32).sum().item()))

        variance_t = model.get_variance(t)
        std_dev_t = float(1.0) * (variance_t ** 0.5)
        inj = std_dev_t * best_noise

        eta_cur = _eta_for_t(int(t))
        with torch.inference_mode():
            xt_next = model.finish_step(xt, x0_hat, epst, t, best_noise, eta=eta_cur)

        if x0_target is not None:
            with torch.inference_mode():
                epst_after = model.get_epst(xt_next, t, prompt_state)
                x0_hat_after = model.get_x0_hat(xt_next, epst_after, t)
                residual_after = x0_target - x0_hat_after
            diag_after.append(float(residual_after.detach().to(torch.float32).norm().item()))
            diag_best.append(best_abs_score)
            diag_inj_norm.append(float(inj.detach().to(torch.float32).norm().item()))
            diag_inj_scale.append(float(std_dev_t.detach().to(torch.float32).item()))

        xt = xt_next

    audio_np = model.decode_latent_to_audio(xt)

    stream = {
        "mode": "ddcm_audio_v2" if int(num_pursuit_noises) > 1 else "ddcm_audio_v1",
        "noise_indices": noise_indices,
        "coeff_indices": coeff_indices if int(num_pursuit_noises) > 1 else None,
        "num_pursuit_noises": int(num_pursuit_noises),
        "num_pursuit_coef_bits": int(num_pursuit_coef_bits),
        "pursuit_renorm": bool(pursuit_renorm),
        "score_mode": str(score_mode),
        "score_blend_lambda": float(score_blend_lambda),
        "score_mode_late": str(score_mode_late),
        "score_switch_t": int(score_switch_t),
        "eta": float(eta),
        "eta_late": float(eta_late),
        "eta_switch_t": int(eta_switch_t),
        "exact_rerank_topk": int(exact_rerank_topk),
        "mel_proxy_topk": int(mel_proxy_topk),
        "mel_proxy_interval": int(mel_proxy_interval),
        "audio_proxy_topk": int(audio_proxy_topk),
        "audio_proxy_interval": int(audio_proxy_interval),
        "selected_noise_l2": selected_noise_l2,
        "selected_noise_sum": selected_noise_sum,
        "shape": [int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2])],
        "num_noises": int(num_noises),
        "num_noises_late": int(num_noises_late),
        "num_noises_switch_t": int(num_noises_switch_t),
        "optimized_steps": int(optimized_steps),
        "init_noise_seed": int(init_noise_seed),
        "step_noise_seed": int(step_noise_seed),
        "t_range": [int(t_range[0]), int(t_range[1])],
    }

    diagnostics = None
    if x0_target is not None:
        endpoint = _endpoint_latent_stats(xt, x0_target)
        diagnostics = {
            "residual_norm_before": diag_before,
            "residual_norm_after": diag_after,
            "best_abs_score": diag_best,
            "rerank_dot_gain": diag_rerank_improve,
            "mel_proxy_mse": diag_mel_proxy_mse,
            "audio_proxy_mse": diag_audio_proxy_mse,
            "injection_norm": diag_inj_norm,
            "injection_scale": diag_inj_scale,
            **endpoint,
        }

    return {
        "audio": audio_np,
        "sr": int(sr),
        "stream": stream,
        "init_latent": init_latent_np,
        "diagnostics": diagnostics,
    }
