import torch
from typing import Tuple
from pathlib import Path


def create_codebook(
    vector_shape: Tuple[int, ...],
    K: int = 256,
    seed: int = 1234,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    normalize: bool = False,
) -> torch.Tensor:
    """Create a deterministic random codebook with K atoms.

    By default we keep the raw standard-normal amplitudes so that each atom
    statistically matches the diffusion noise scale (unit variance per
    dimension). Set ``normalize=True`` only if unit-norm vectors are desired.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    flat_dim = 1
    for d in vector_shape:
        flat_dim *= int(d)

    cb = torch.randn((K, flat_dim), generator=gen, device=device, dtype=dtype)
    if normalize:
        cb = torch.nn.functional.normalize(cb, p=2, dim=1, eps=1e-12)
    cb = cb.view((K, *vector_shape))
    return cb


def _codebook_dir() -> Path:
    # 将码本固定保存在仓库内的 codebooks/ 目录（相对 ddcm/ 上一级）
    root = Path(__file__).resolve().parent.parent
    d = root / "codebooks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def codebook_cache_path(vector_shape: Tuple[int, ...], K: int, seed: int) -> Path:
    C, H, W = vector_shape
    return _codebook_dir() / f"cb_C{C}_H{H}_W{W}_K{K}_seed{seed}.pt"


def load_or_create_codebook(
    vector_shape: Tuple[int, ...],
    K: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    本地持久化码本：若缓存文件存在则直接加载；否则创建后保存到本地。
    说明：按你的诉求，码本固定且不随比特流存储，可适当增大 K。
    """
    path = codebook_cache_path(vector_shape, K, seed)
    if path.exists():
        cb = torch.load(path, map_location=device)
        cb = cb.to(device=device, dtype=dtype)
        # 旧版本缓存使用单位范数，检测到后进行再生成以恢复噪声幅度。
        flat = cb.detach().to(dtype=torch.float32).view(cb.shape[0], -1)
        avg_norm = float(flat.norm(dim=1).mean().item()) if flat.numel() > 0 else 0.0
        expected = float(flat.shape[1] ** 0.5) if flat.shape[1] > 0 else 1.0
        if expected > 0 and avg_norm < expected * 0.2:
            cb = create_codebook(vector_shape, K=K, seed=seed, device=device, dtype=dtype, normalize=False)
            torch.save(cb.detach().to("cpu", dtype=torch.float32), path)
            cb = cb.to(device=device, dtype=dtype)
        return cb
    cb = create_codebook(vector_shape, K=K, seed=seed, device=device, dtype=dtype, normalize=False)
    # 保存为 CPU float32，兼顾可移植性
    torch.save(cb.detach().to("cpu", dtype=torch.float32), path)
    return cb
