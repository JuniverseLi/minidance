"""Stream filtered OpenVid clips -> Wan2.2 VAE latents, packed into resumable shards.

Raw videos are never kept: each batch is fetched from the HF-hosted Lance dataset,
decoded, encoded and dropped. Output (one pair per shard of --shard_size clips):

    <out>/shard_00000.safetensors   latents  bf16 [N, 48, 9, 16, 16]  (raw, un-normalized)
    <out>/shard_00000.parquet       rowid, caption, scene, ... (same order)

Clips are shuffled once (fixed seed) before sharding, so any prefix of shards is a
representative sample: S3 uses `--shards 0 1` (1024 clips) locally, Colab does the rest.

    python dataset/encode_latents.py --out data/latents --shards 0 1
    python dataset/encode_latents.py --out /content/drive/MyDrive/minidance/latents
"""
import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import lance
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vae import WanVAE

URI = "hf://datasets/lance-format/Openvid-1M/data/train.lance"
SIZE, FRAMES = 256, 33
META_COLS = ["_rowid", "caption", "scene", "source", "video_id", "camera_motion",
             "aesthetic_score", "motion_score"]


def decode_clip(data, tmp_dir):
    """mp4 bytes -> uint8 [FRAMES, SIZE, SIZE, 3] RGB, frames spread evenly over the clip."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=tmp_dir, delete=False) as f:
        f.write(data)
    try:
        cap = cv2.VideoCapture(f.name)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    finally:
        os.remove(f.name)
    if len(frames) < FRAMES // 2:
        return None
    idx = np.round(np.linspace(0, len(frames) - 1, FRAMES)).astype(int)
    return np.stack([cv2.resize(frames[i], (SIZE, SIZE), interpolation=cv2.INTER_AREA)[..., ::-1] for i in idx])


def fetch(ds, rowids, tmp_dir, retries=4):
    """Download + decode one batch; runs in a worker thread while the GPU encodes.

    Network errors (e.g. HF rate limits) are retried with backoff; a batch that keeps
    failing is skipped (all None) rather than stalling the whole run.
    """
    for attempt in range(retries):
        try:
            blobs = ds.take_blobs("video_blob", ids=rowids)
            clips = []
            for blob in blobs:
                with blob as f:
                    clips.append(decode_clip(f.read(), tmp_dir))
            return clips
        except Exception as e:
            print(f"  fetch failed ({attempt + 1}/{retries}): {str(e)[:200]}", flush=True)
            time.sleep(15 * (attempt + 1))
    return [None] * len(rowids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/scenery.parquet")
    ap.add_argument("--out", default="data/latents")
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--shards", type=int, nargs="*", help="only these shard ids (default: all)")
    ap.add_argument("--batch", type=int, default=8, help="clips per VAE forward")
    ap.add_argument("--workers", type=int, default=2, help="download threads = batches prefetched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.parquet).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_shards = (len(df) + args.shard_size - 1) // args.shard_size
    todo = [s for s in (args.shards if args.shards is not None else range(n_shards))
            if not os.path.exists(os.path.join(args.out, f"shard_{s:05d}.safetensors"))]
    print(f"{len(df):,d} clips -> {n_shards} shards of {args.shard_size}; {len(todo)} to do", flush=True)
    if not todo:
        return

    print("HF_TOKEN set" if os.environ.get("HF_TOKEN") else
          "HF_TOKEN not set: anonymous downloads may be rate-limited", flush=True)
    vae = WanVAE(dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    ds = lance.dataset(URI)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    tmp_dir = tempfile.mkdtemp(prefix="minidance_")
    t0, done, last_log = time.time(), 0, time.time()
    print("encoding... (progress every 30 s)", flush=True)

    for s in todo:
        part = df.iloc[s * args.shard_size:(s + 1) * args.shard_size]
        batches = [part._rowid.tolist()[i:i + args.batch] for i in range(0, len(part), args.batch)]
        futures = [pool.submit(fetch, ds, b, tmp_dir) for b in batches[:args.workers]]  # prefetch
        latents, keep = [], []
        for i in range(len(batches)):
            clips = futures[i].result()
            if i + args.workers < len(batches):
                futures.append(pool.submit(fetch, ds, batches[i + args.workers], tmp_dir))
            ok = [j for j, c in enumerate(clips) if c is not None]
            if ok:
                x = torch.from_numpy(np.stack([clips[j] for j in ok])).cuda()
                x = x.permute(0, 4, 1, 2, 3).float().div(127.5).sub(1)  # B 3 T H W in [-1, 1]
                latents.append(vae.encode(x).to(torch.bfloat16).cpu())
                keep += [i * args.batch + j for j in ok]
            done += len(clips)
            if time.time() - last_log > 30:
                last_log = time.time()
                print(f"  shard {s}: {(i + 1) * args.batch}/{len(part)} clips, "
                      f"{done / (time.time() - t0):.2f} clips/s", flush=True)
        z = torch.cat(latents)
        meta = part.iloc[keep][[c for c in META_COLS if c in part.columns]].reset_index(drop=True)

        base = os.path.join(args.out, f"shard_{s:05d}")
        meta.to_parquet(base + ".parquet")
        save_file({"latents": z.contiguous()}, base + ".tmp")
        os.replace(base + ".tmp", base + ".safetensors")  # written last: marks the shard complete
        rate = done / (time.time() - t0)
        left = sum(min(args.shard_size, len(df) - t * args.shard_size) for t in todo[todo.index(s) + 1:])
        print(f"shard {s}: {len(z)}/{len(part)} clips, latent {tuple(z.shape[1:])} | "
              f"{rate:.2f} clips/s, ETA {left / rate / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
