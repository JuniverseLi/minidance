"""Generate videos with a trained MiniDance model.

    uv run python scripts/sample.py --ckpt out/minidance-base.safetensors \
        --prompt "A waterfall cascading down mossy rocks." "海上日落，金色余晖映在水面。"

Each prompt -> one .mp4 (and .gif with --gif) in --out. Runs on an 8 GB GPU.
"""
import argparse
import os
import re
import sys

import imageio.v2 as imageio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dit import load_dit
from model.text_encoder import TextEncoder
from model.vae import WanVAE
from trainer.trainer_utils import generate, save_gif


def save_mp4(video, path, fps):
    """video: uint8 [T, H, W, 3] RGB -> H.264 mp4 (plays in browsers and on GitHub)."""
    imageio.mimwrite(path, video, fps=fps, codec="libx264", quality=9, macro_block_size=16)


def slug(text, n=40):
    s = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")
    return s[:n] or "empty"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="exported .safetensors or a training ckpt.pt")
    ap.add_argument("--prompt", nargs="+", required=True)
    ap.add_argument("--out", default="out/samples")
    ap.add_argument("--steps", type=int, default=30, help="sampling steps")
    ap.add_argument("--cfg", type=float, default=5.0, help="classifier-free guidance scale")
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gif", action="store_true", help="also save a .gif per prompt")
    args = ap.parse_args()

    dev = "cuda"
    model = load_dit(args.ckpt, dev)
    te = TextEncoder(device=dev)
    vae = WanVAE(dtype=torch.bfloat16)
    videos = generate(model, te, vae, args.prompt, steps=args.steps, cfg=args.cfg,
                      shift=args.shift, seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    for i, (p, v) in enumerate(zip(args.prompt, videos)):
        name = os.path.join(args.out, f"{i:02d}_{slug(p)}_s{args.seed}")
        save_mp4(v, name + ".mp4", args.fps)
        if args.gif:
            save_gif(v[None], name + ".gif", args.fps)
        print(f"{name}.mp4  <- {p}")


if __name__ == "__main__":
    main()
