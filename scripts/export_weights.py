"""Strip a training checkpoint down to the EMA weights for release (bf16 .safetensors, config in metadata).

    uv run python scripts/export_weights.py out/s4_base/ckpt.pt out/minidance-base.safetensors
"""
import json
import sys

import torch
from safetensors.torch import save_file


def main(src, dst):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in ckpt["ema"].items()}
    meta = {"config": json.dumps(ckpt["dit_cfg"]), "step": str(ckpt["step"])}
    save_file(sd, dst, metadata=meta)
    n = sum(v.numel() for v in sd.values()) / 1e6
    print(f"{src} (step {ckpt['step']}) -> {dst}: {n:.1f}M params, bf16")


if __name__ == "__main__":
    main(*sys.argv[1:3])
