"""Generate the clips shown in the README gallery and the demo video (listed in demo/clips.json).

    uv run python demo/make_clips.py --ckpt out/minidance-base-sft.safetensors

-> demo/public/clips/<name>.mp4 (for Remotion) and images/gallery/<name>.webp (for the README),
   plus demo/public/denoise/{xt,x0}_NN.jpg: the denoising steps of one clip, for the "magic" scene,
   and demo/public/logo_alpha.png: images/logo.png with its white background made transparent.
Each clip can also be reproduced on its own:  scripts/sample.py --prompt "<prompt>" --seed <seed>
"""
import argparse
import json
import os
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.dit import load_dit
from model.flow import sample
from model.text_encoder import TextEncoder
from model.vae import WanVAE
from scripts.sample import save_mp4
from trainer.trainer_utils import generate


def to_webp(mp4, webp):
    """Animated WebP: ~1/3 the size of a GIF at better quality, and GitHub renders it inline."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", mp4,
                    "-c:v", "libwebp", "-q:v", "80", "-loop", "0", webp], check=True)


@torch.no_grad()
def denoise_frames(model, te, vae, prompt, seed, steps, out_dir):
    """Save the first frame of the canvas x_t and of the model's guess x0 = x_t - t*v at every step."""
    os.makedirs(out_dir, exist_ok=True)
    y, mask = te([prompt])
    y0, m0 = te([""])
    shape = (1, model.cfg.in_channels, 9, 16, 16)
    noise = torch.randn(shape[1:], device=y.device, generator=torch.Generator(y.device).manual_seed(seed))[None]
    xt, x0 = [], []
    keep = lambda i, t, x, v: (xt.append(x[:, :, :1].clone()), x0.append((x - t * v)[:, :, :1].clone()))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = sample(model, shape, y, mask, y0, m0, steps=steps, cfg=5.0, shift=3.0, noise=noise, callback=keep)
    xt.append(z[:, :, :1])
    for name, lats in (("xt", xt), ("x0", x0)):
        for i, lat in enumerate(lats):
            f = vae.decode(vae.denormalize(lat))[0, :, 0]  # causal VAE: latent frame 0 alone -> video frame 0
            img = ((f.permute(1, 2, 0).float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
            Image.fromarray(img).save(os.path.join(out_dir, f"{name}_{i:02d}.jpg"), quality=92)


def logo_alpha(src, dst, max_hole=0.02):
    """Clear the white background and the white holes inside letters; keep big white shapes (the cloud)."""
    from scipy import ndimage
    rgb = np.asarray(Image.open(src).convert("RGB"))
    labels, n = ndimage.label(rgb.min(axis=2) > 236)
    sizes = ndimage.sum(np.ones(labels.shape), labels, range(1, n + 1))
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
    clear = [k for k in range(1, n + 1) if k in border or sizes[k - 1] < max_hole * labels.size]
    alpha = np.where(np.isin(labels, clear), 0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha])).save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--denoise", default="waterfall", help="clip whose denoising steps are saved")
    args = ap.parse_args()

    clips = json.load(open(os.path.join(ROOT, "demo", "clips.json"), encoding="utf-8"))
    mp4_dir, img_dir = os.path.join(ROOT, "demo", "public", "clips"), os.path.join(ROOT, "images", "gallery")
    os.makedirs(mp4_dir, exist_ok=True), os.makedirs(img_dir, exist_ok=True)

    logo_alpha(os.path.join(ROOT, "images", "logo.png"), os.path.join(ROOT, "demo", "public", "logo_alpha.png"))
    model, te, vae = load_dit(args.ckpt), TextEncoder(), WanVAE(dtype=torch.bfloat16)
    for seed in sorted({c["seed"] for c in clips}):
        group = [c for c in clips if c["seed"] == seed]
        videos = generate(model, te, vae, [c["prompt"] for c in group], steps=args.steps, seed=seed)
        for c, v in zip(group, videos):
            mp4 = os.path.join(mp4_dir, c["name"] + ".mp4")
            save_mp4(v, mp4, 12)
            to_webp(mp4, os.path.join(img_dir, c["name"] + ".webp"))
            print(f"{c['name']:24s} seed {seed}  {c['prompt']}")
    c = next(c for c in clips if c["name"] == args.denoise)
    denoise_frames(model, te, vae, c["prompt"], c["seed"], args.steps, os.path.join(ROOT, "demo", "public", "denoise"))
    print(f"denoising steps of {c['name']} -> demo/public/denoise/")


if __name__ == "__main__":
    main()
