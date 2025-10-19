from typing import Dict, List, Tuple, Optional
import torch


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


@torch.no_grad()
def compress_latent(
    latent: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    T: int = 16,
    mode: str = "coord",
) -> Dict:
    """
    Greedy matching-pursuit style compression of a single latent tensor using a fixed codebook.

    Contract:
    - Input latent shape: [B, C, H, W] (expect B==1 for now)
    - Codebook shape: [K, C, H, W]
    - Returns indices [T], coeffs [T], and meta for reconstruction.

    Error modes:
    - Shape mismatch -> ValueError
    - Empty/zero T -> returns no-op stream with zero indices/coeffs.
    """
    if latent.dim() != 4:
        raise ValueError(f"latent must be 4D [B,C,H,W], got {latent.shape}")
    if latent.size(0) != 1:
        raise ValueError("Only batch size 1 is supported in this prototype.")

    B, C, H, W = latent.shape
    D = C * H * W

    mode = (mode or "coord").lower()

    if mode == "coord":
        # Coordinate-basis sparse coding: keep top-|T| entries by magnitude
        vec = _flatten(latent).squeeze(0)  # [D]
        T_eff = max(0, min(int(T), D))
        if T_eff == 0:
            top_idx = torch.empty(0, dtype=torch.long, device=vec.device)
            top_val = torch.empty(0, dtype=vec.dtype, device=vec.device)
        else:
            top_val, top_idx = torch.topk(torch.abs(vec), k=T_eff, largest=True, sorted=False)
            # Recover signed values at those positions
            top_val = vec[top_idx]

        stream = {
            "mode": "coord",
            "positions": top_idx.detach().long().cpu().tolist(),
            "values": top_val.detach().cpu().tolist(),
            "T": int(T_eff),
            "shape": [int(C), int(H), int(W)],
        }
        return stream

    elif mode == "random":
        if codebook is None or codebook.dim() != 4:
            raise ValueError("random mode requires codebook of shape [K,C,H,W]")
        if (C, H, W) != (codebook.size(1), codebook.size(2), codebook.size(3)):
            raise ValueError("latent and codebook atoms must share (C,H,W)")

        K = codebook.size(0)
        # Flatten for fast correlation
        target = _flatten(latent)  # [1, D]
        atoms = _flatten(codebook)  # [K, D]

        # Normalize atoms (safety)
        atoms = torch.nn.functional.normalize(atoms, p=2, dim=1, eps=1e-12)
        residual = target.clone()

        indices: List[int] = []
        coeffs: List[float] = []

        for _ in range(int(T)):
            corr = torch.matmul(residual, atoms.t()).squeeze(0)  # [K]
            idx = int(torch.argmax(torch.abs(corr)).item())
            alpha = float(corr[idx].item())
            indices.append(idx)
            coeffs.append(alpha)
            residual = residual - alpha * atoms[idx : idx + 1]

        stream = {
            "mode": "random",
            "indices": indices,
            "coeffs": coeffs,
            "T": int(T),
            "K": int(K),
            "shape": [int(C), int(H), int(W)],
        }
        return stream

    else:
        raise ValueError(f"Unknown compression mode: {mode}")


@torch.no_grad()
def decompress_latent(stream: Dict, codebook: Optional[torch.Tensor] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Reconstruct latent from stream.
    - coord mode: fills top-T positions with stored values.
    - random mode: requires codebook and uses linear combination of atoms.
    Returns tensor of shape [1,C,H,W].
    """
    mode = stream.get("mode", "random")
    C, H, W = stream["shape"]

    if device is None:
        device = codebook.device if codebook is not None else torch.device("cpu")
    if dtype is None:
        dtype = (codebook.dtype if codebook is not None else torch.float32)

    if mode == "coord":
        positions = stream.get("positions", [])
        values = stream.get("values", [])
        if len(positions) != len(values):
            raise ValueError("positions and values length mismatch")
        D = C * H * W
        vec = torch.zeros((D,), device=device, dtype=dtype)
        if len(positions) > 0:
            idx = torch.tensor(positions, device=device, dtype=torch.long)
            val = torch.tensor(values, device=device, dtype=dtype)
            vec.index_copy_(0, idx, val)
        recon = vec.view(1, C, H, W)
        return recon

    elif mode == "random":
        if codebook is None:
            raise ValueError("random mode requires codebook for decompression")
        indices = stream.get("indices", [])
        coeffs = stream.get("coeffs", [])
        if len(indices) != len(coeffs):
            raise ValueError("indices and coeffs length mismatch")
        atoms = codebook.to(device=device, dtype=dtype)
        recon = torch.zeros((1, C, H, W), device=device, dtype=dtype)
        for idx, a in zip(indices, coeffs):
            recon = recon + float(a) * atoms[int(idx)][None, ...]
        return recon

    else:
        raise ValueError(f"Unknown decompression mode: {mode}")
