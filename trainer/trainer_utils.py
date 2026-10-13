import copy
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from model.flow import sample


def get_lr(step, total, warmup, lr, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))  # ramp up early on
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)


def save_checkpoint(path, model, ema, opt, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                "step": step, "dit_cfg": vars(model.cfg), "args": vars(args)}, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def generate(model, text_encoder, vae, prompts, frames=9, steps=20, cfg=5.0, shift=3.0, seed=0):
    """prompts -> uint8 videos [N, T, H, W, 3]. frames = latent frames (9 -> 33 video frames)."""
    y, mask = text_encoder(prompts)
    y0, m0 = text_encoder([""])
    g = torch.Generator(device=y.device).manual_seed(seed)
    shape = (len(prompts), model.cfg.in_channels, frames, 16, 16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = sample(model, shape, y, mask, y0, m0, steps=steps, cfg=cfg, shift=shift, generator=g)
    videos = []
    for i in range(len(prompts)):  # decode one at a time to keep VRAM low
        x = vae.decode(vae.denormalize(z[i:i + 1]))
        videos.append(((x[0].permute(1, 2, 3, 0).float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy())
    return np.stack(videos)


def save_gif(videos, path, fps=12):
    """videos [N, T, H, W, 3] -> one GIF with the N clips side by side."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = np.concatenate(list(videos), axis=2)
    frames = [Image.fromarray(f) for f in grid]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f, self.t0 = open(path, "a", encoding="utf-8"), time.time()

    def __call__(self, msg):
        line = f"[{(time.time() - self.t0) / 60:6.1f} min] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()
