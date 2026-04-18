from __future__ import annotations

import argparse
from pathlib import Path
import sys

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddcm.audio_models import load_audio_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_id", default="cvssp/audioldm2-music")
    ap.add_argument("--float16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    model = load_audio_model(
        model_id=args.model_id,
        timesteps=10,
        device=device,
        float16=bool(args.float16),
        local_files_only=True,
    )

    io = model.encode_audio_to_latent(args.input)
    wav = model.decode_latent_to_audio(io["latent"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav, int(io.get("sr", 16000)))
    print(str(out).replace('\\', '/'))


if __name__ == "__main__":
    main()
