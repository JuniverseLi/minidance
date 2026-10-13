"""MiniDance pretraining: text-to-video flow matching on pre-encoded latents.

    uv run python trainer/train_pretrain.py --shards 0 1 --preset small --steps 3000      # S3 overfit test
"""
import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.latent_dataset import LatentDataset, truncate
from model.dit import build_dit
from model.flow import flow_loss
from model.text_encoder import TextEncoder
from model.vae import WanVAE, latent_stats
from trainer.trainer_utils import EMA, Logger, generate, get_lr, save_checkpoint, save_gif

PROMPTS = [
    "Ocean waves rolling onto a sandy beach at sunset, golden light on the water.",
    "White clouds drifting over green mountains under a blue sky.",
    "A waterfall cascading down mossy rocks into a clear pool.",
    "海上日落，小浪拍岸，金色余晖映在水面。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/latents")
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--short_captions", default="data/short_captions.jsonl")
    ap.add_argument("--max_clips", type=int, default=0, help="train on the first N clips only (overfit test)")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max_minutes", type=float, default=0, help="stop (sample + save) after this long")
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--p_drop", type=float, default=0.1, help="caption dropout for CFG")
    ap.add_argument("--p_image", type=float, default=0.2, help="prob. a batch is trained as images (frame 0)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--out", default="out/pretrain")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    log = Logger(os.path.join(args.out, "train.log"))

    ds = LatentDataset(args.data_dir, args.shards, args.short_captions)
    if args.max_clips:
        ds.index = ds.index[:args.max_clips]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=min(args.batch, len(ds)), shuffle=True, drop_last=True,
        collate_fn=lambda b: (torch.stack([z for z, _ in b]), [c for _, c in b]))
    mean, std = (s.view(1, -1, 1, 1, 1).to(dev) for s in latent_stats())

    te = TextEncoder(device=dev)
    model = build_dit(args.preset, text_dim=te.dim).to(dev)
    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step, ckpt = 0, os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=dev)
        model.load_state_dict(sd["model"]), ema.model.load_state_dict(sd["ema"]), opt.load_state_dict(sd["opt"])
        step = sd["step"]
        log(f"resumed from step {step}")
    train_model = torch.compile(model) if args.compile else model
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"DiT-{args.preset}: {n_params:.1f}M params | {len(ds):,d} clips | batch {args.batch} | args {vars(args)}")

    it, t_log, seen, t_start = iter(loader), time.time(), 0, time.time()
    while step < args.steps:
        try:
            z, caps = next(it)
        except StopIteration:
            it = iter(loader)
            z, caps = next(it)
        for g in opt.param_groups:
            g["lr"] = get_lr(step, args.steps, args.warmup, args.lr)
        x0 = (z.to(dev).float() - mean) / std
        if random.random() < args.p_image:
            x0 = x0[:, :, :1]  # causal VAE: latent frame 0 is the first frame alone = an image
        caps = ["" if random.random() < args.p_drop else c for c in caps]
        y, mask = te(caps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_loss(train_model, x0, y, mask, args.shift)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.update(model, step)
        step += 1
        seen += z.shape[0]
        timeout = args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes

        if step % args.log_every == 0:
            dt = time.time() - t_log
            log(f"step {step:6d} | loss {loss.item():.4f} | gnorm {gnorm.item():.2f} | "
                f"lr {opt.param_groups[0]['lr']:.2e} | {seen / dt:.1f} clips/s | "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            t_log, seen = time.time(), 0
        if step % args.sample_every == 0 or step == args.steps or timeout:
            # the VAE only lives on the GPU while sampling: an 8 GB card can't hold it during training
            vae = WanVAE(dtype=torch.bfloat16)
            prompts = PROMPTS + [truncate(ds.long[i]) for i in ds.index[:2]]  # last two: seen in training
            videos = generate(ema.model, te, vae, prompts, shift=args.shift)
            del vae
            torch.cuda.empty_cache()
            save_gif(videos, os.path.join(args.out, "samples", f"step_{step:06d}.gif"))
            log(f"samples -> {args.out}/samples/step_{step:06d}.gif | prompts: {prompts[-2:]}")
        if step % args.save_every == 0 or step == args.steps or timeout:
            save_checkpoint(ckpt, model, ema, opt, step, args)
        if timeout:
            log(f"reached --max_minutes {args.max_minutes} at step {step}")
            break


if __name__ == "__main__":
    main()
