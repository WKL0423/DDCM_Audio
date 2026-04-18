from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _resolve_cache_dir() -> str:
    import os
    from pathlib import Path

    env_hf_home = os.environ.get("HF_HOME")
    env_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_hub_cache:
        return env_hub_cache
    if env_hf_home:
        return str(Path(env_hf_home) / "hub")
    return r"F:\\Kasai_Lab\\hf_cache\\huggingface\\hub"


@dataclass
class PromptState:
    prompt_embeds: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    generated_prompt_embeds: torch.Tensor


class AudioLDM2Wrapper(torch.nn.Module):
    def __init__(
        self,
        model_id: str,
        timesteps: int,
        device: torch.device,
        float16: bool = False,
        local_files_only: bool = True,
        allow_online_fallback: bool = False,
    ) -> None:
        super().__init__()
        from New_pipeline_audioldm2 import AudioLDM2Pipeline
        try:
            from diffusers import DDIMScheduler  # type: ignore
        except Exception:
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # type: ignore

        self.model_id = model_id
        self.num_timesteps = int(timesteps)
        self.device = device
        self.float16 = bool(float16)
        self.dtype = torch.float16 if self.float16 and self.device.type == "cuda" else torch.float32

        try:
            self.pipe = AudioLDM2Pipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                cache_dir=_resolve_cache_dir(),
                local_files_only=local_files_only,
            ).to(device)
        except Exception:
            if not allow_online_fallback:
                raise
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.pipe = AudioLDM2Pipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                cache_dir=_resolve_cache_dir(),
            ).to(device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)  # type: ignore
        self.set_timesteps(self.num_timesteps, device)

    @property
    def timesteps(self) -> torch.Tensor:
        return self.pipe.scheduler.timesteps

    def set_timesteps(self, timesteps: int, device: torch.device) -> None:
        self.pipe.scheduler.set_timesteps(int(timesteps), device=device)
        try:
            if int(self.pipe.scheduler.timesteps[0]) == int(getattr(self.pipe.scheduler.config, "num_train_timesteps", 1000)):
                self.pipe.scheduler.timesteps = self.pipe.scheduler.timesteps - 1
        except Exception:
            pass

    def encode_prompt(self, prompt: str = "") -> PromptState:
        with torch.inference_mode():
            prompt_embeds, attention_mask, generated_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                self.device,
                num_waveforms_per_prompt=1,
                do_classifier_free_guidance=False,
                transcription=None,
                negative_prompt=None,
            )
        return PromptState(
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            generated_prompt_embeds=generated_prompt_embeds,
        )

    def encode_audio_to_latent(self, wav_path: str) -> Dict:
        from .audio_codec import _load_audio_mel_and_latent

        with torch.inference_mode():
            io = _load_audio_mel_and_latent(self.pipe, wav_path, self.device)
        io["latent"] = io["latent"].to(device=self.device, dtype=self.dtype)
        return io

    def decode_latent_to_audio(self, latent: torch.Tensor):
        from .audio_codec import _decode_latent_to_audio

        return _decode_latent_to_audio(self.pipe, latent.to(device=self.device, dtype=self.dtype))

    def get_epst(self, latents: torch.Tensor, t: torch.Tensor, prompt_state: PromptState) -> torch.Tensor:
        latent_model_input = self.pipe.scheduler.scale_model_input(latents, t)
        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_state.generated_prompt_embeds,
            encoder_hidden_states_1=prompt_state.prompt_embeds,
            encoder_attention_mask_1=prompt_state.attention_mask,
            return_dict=False,
        )[0]
        return noise_pred

    def get_x0_hat(self, xt: torch.Tensor, epst: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (xt - beta_prod_t ** 0.5 * epst) / alpha_prod_t ** 0.5
        return pred_original_sample

    def get_variance(self, timestep: torch.Tensor) -> torch.Tensor:
        prev_timestep = timestep - self.pipe.scheduler.config.num_train_timesteps // self.pipe.scheduler.num_inference_steps
        return self.pipe.scheduler._get_variance(timestep, prev_timestep)

    def _get_alpha_prod_t_prev(self, prev_timestep: torch.Tensor) -> torch.Tensor:
        return (
            self.pipe.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.pipe.scheduler.final_alpha_cumprod
        )

    def finish_step(
        self,
        xt: torch.Tensor,
        pred_x0: torch.Tensor,
        epst: torch.Tensor,
        timestep: torch.Tensor,
        variance_noise: torch.Tensor,
        eta: Optional[float] = 1.0,
    ) -> torch.Tensor:
        prev_timestep = timestep - self.pipe.scheduler.config.num_train_timesteps // self.pipe.scheduler.num_inference_steps
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self._get_alpha_prod_t_prev(prev_timestep)

        variance = self.get_variance(timestep)
        std_dev_t = float(eta) * variance ** 0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * epst
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_sample_direction

        if float(eta) > 0:
            prev_sample = prev_sample + std_dev_t * variance_noise

        return prev_sample


def load_audio_model(
    model_id: str,
    timesteps: int,
    device: torch.device,
    float16: bool = False,
    local_files_only: bool = True,
    allow_online_fallback: bool = False,
) -> AudioLDM2Wrapper:
    return AudioLDM2Wrapper(
        model_id=model_id,
        timesteps=timesteps,
        device=device,
        float16=float16,
        local_files_only=local_files_only,
        allow_online_fallback=allow_online_fallback,
    )
