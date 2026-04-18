from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def get_args_from_filename(filename: str) -> Tuple[int, int, Tuple[int, int], str]:
    T = int(re.search(r"T=(\d+)", filename).group(1))
    K = int(re.search(r"K=(\d+)", filename).group(1))
    t_range_match = re.search(r"_in(\d+)-(\d+)", filename)
    t_range = (int(t_range_match.group(1)), int(t_range_match.group(2)))
    model_short = re.search(r"model=(.+?)_audio", filename).group(1)
    if model_short == "audioldm2-music":
        model_id = "cvssp/audioldm2-music"
    else:
        model_id = model_short
    return T, K, t_range, model_id


def save_as_binary_bitwise(noise_indices, K: int, filename: str):
    bits_per_index = int(np.ceil(np.log2(K)))
    bitstring = "".join(format(int(v), f"0{bits_per_index}b") for v in noise_indices)
    byte_array = int(bitstring or "0", 2).to_bytes((len(bitstring) + 7) // 8 or 1, byteorder="big")
    with open(filename, "wb") as f:
        f.write(byte_array)


def load_from_binary_bitwise(filename: str, K: int, optimized_steps: int):
    bits_per_index = int(np.ceil(np.log2(K)))
    with open(filename, "rb") as f:
        byte_data = f.read()
    bitstring = bin(int.from_bytes(byte_data, byteorder="big"))[2:]
    bits_amount = optimized_steps * bits_per_index
    bitstring = bitstring.zfill(bits_amount)
    indices = [int(bitstring[i : i + bits_per_index], 2) for i in range(0, bits_amount, bits_per_index)]
    return indices


def save_meta_json(path: str, payload: Dict):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_meta_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
