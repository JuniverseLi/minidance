"""Face SFT data, step 2: OpenVid face clips -> face-centred crops -> Wan2.2 latents + face masks.

Differences from encode_latents.py:
  - frame rate pinned to ~--fps (OpenVid's Lance blobs are all 64 frames at 24 fps, so this matches
    encode_latents; the check only matters for other sources with different lengths / fps)
  - one fixed square crop around the face (median box over the clip; a moving crop would add
    fake camera motion), sized so the detected face is ~--face_frac of the side
  - clips without exactly one clear face in >= 90% of the frames are dropped
  - each shard also stores face_mask uint8 [N, 9, 16, 16]: latent cells that contain the face,
    used to up-weight the face in the flow loss

    python dataset/encode_faces.py --parquet data/faces.parquet --out data/face_latents
"""
import argparse
import functools
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import lance
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.encode_latents import FRAMES, META_COLS, SIZE, URI, fetch
from model.vae import WanVAE

YUNET_REPO, YUNET_FILE = "opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx"
CELL = 16  # Wan2.2 VAE: 16x16 pixels per latent cell
_local = threading.local()


def detect(frame, yunet, max_side=640):
    """Face boxes (x, y, w, h) in frame pixels, largest first. One YuNet per thread (not thread-safe)."""
    if not hasattr(_local, "det"):
        _local.det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.7, 0.3, 50)
    h, w = frame.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
    _local.det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = _local.det.detect(small)
    if faces is None:
        return []
    return sorted((f[:4] / s for f in faces), key=lambda b: -b[2] * b[3])


def face_crop(boxes, H, W, face_frac):
    """Fixed square crop (x0, y0, side) around the median face, or None if the face leaves it."""
    arr = np.array([b for b in boxes if b is not None])
    size = np.median(np.maximum(arr[:, 2], arr[:, 3]))
    cx, cy = np.median(arr[:, 0] + arr[:, 2] / 2), np.median(arr[:, 1] + arr[:, 3] / 2)
    side = min(size / face_frac, H, W)
    x0 = float(np.clip(cx - side / 2, 0, W - side))
    y0 = float(np.clip(cy - side / 2, 0, H - side))
    centers = np.stack([arr[:, 0] + arr[:, 2] / 2 - x0, arr[:, 1] + arr[:, 3] / 2 - y0], 1) / side
    if centers.min() < 0.15 or centers.max() > 0.85:  # face drifts to the crop border
        return None
    return x0, y0, side, size


def latent_face_mask(boxes, x0, y0, side, grow=0.1):
    """Per-frame face boxes -> uint8 [latent frames, SIZE/CELL, SIZE/CELL] (causal VAE: 1 + 4k frames)."""
    n = SIZE // CELL
    fm = np.zeros((len(boxes), n, n), np.uint8)
    last = next(b for b in boxes if b is not None)
    for i, b in enumerate(boxes):
        last = b if b is not None else last  # a missed detection keeps the previous box
        x, y, w, h = last
        x, y, w, h = x - grow * w, y - grow * h, w * (1 + 2 * grow), h * (1 + 2 * grow)
        a = (np.array([x - x0, y - y0, x - x0 + w, y - y0 + h]) / side * n)
        c0, r0 = np.clip(np.floor(a[:2]).astype(int), 0, n)
        c1, r1 = np.clip(np.ceil(a[2:]).astype(int), 0, n)
        fm[i, r0:r1, c0:c1] = 1
    T = 1 + (len(boxes) - 1) // 4
    return np.stack([fm[0]] + [fm[4 * k - 3:4 * k + 1].max(0) for k in range(1, T)])


def pick_frames(n, src_fps, fps, tol=0.8):
    """FRAMES indices at ~fps from the middle of an n-frame clip, or None if it is too short.

    OpenVid's Lance blobs are 64 frames at 24 fps: 33 frames at 12 fps would need 65, so the
    step shrinks to fit (63/32 -> 12.2 fps); clips needing < tol of the target step are dropped.
    """
    step = min(src_fps / fps, (n - 1) / (FRAMES - 1))
    if n < FRAMES or step < tol * src_fps / fps:
        return None
    start = (n - 1 - (FRAMES - 1) * step) / 2
    return np.round(start + np.arange(FRAMES) * step).astype(int)


def decode_face_clip(data, tmp_dir, yunet, fps=12, face_frac=0.5, min_face=100):
    """mp4 bytes -> (uint8 [FRAMES, SIZE, SIZE, 3] RGB, uint8 face mask [9, 16, 16]) or None."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=tmp_dir, delete=False) as f:
        f.write(data)
    try:
        cap = cv2.VideoCapture(f.name)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    finally:
        os.remove(f.name)
    idx = pick_frames(len(frames), src_fps, fps)
    if idx is None:
        return None
    clip = [frames[i] for i in idx]

    boxes = []
    for frame in clip:
        b = detect(frame, yunet)
        if len(b) >= 2 and b[1][2] * b[1][3] > 0.2 * b[0][2] * b[0][3]:
            return None  # a second clear face
        boxes.append(b[0] if b else None)
    if sum(b is not None for b in boxes) < 0.9 * FRAMES:
        return None
    H, W = clip[0].shape[:2]
    crop = face_crop(boxes, H, W, face_frac)
    if crop is None or crop[3] < min_face:
        return None
    x0, y0, side, _ = crop
    xa, ya, s = int(round(x0)), int(round(y0)), int(round(side))
    video = np.stack([cv2.resize(fr[ya:ya + s, xa:xa + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)[..., ::-1]
                      for fr in clip])
    return video, latent_face_mask(boxes, x0, y0, side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/faces.parquet")
    ap.add_argument("--out", default="data/face_latents")
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--shards", type=int, nargs="*", help="only these shard ids (default: all)")
    ap.add_argument("--batch", type=int, default=8, help="clips per VAE forward")
    ap.add_argument("--workers", type=int, default=4, help="download + face detection threads")
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--face_frac", type=float, default=0.5, help="face box size / crop side")
    ap.add_argument("--min_face", type=float, default=100, help="min face box size in source pixels")
    ap.add_argument("--yunet", help="local face_detection_yunet_2023mar.onnx (default: download from HF)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.yunet:
        from huggingface_hub import hf_hub_download
        args.yunet = hf_hub_download(YUNET_REPO, YUNET_FILE)
    decode = functools.partial(decode_face_clip, yunet=args.yunet, fps=args.fps,
                               face_frac=args.face_frac, min_face=args.min_face)

    df = pd.read_parquet(args.parquet).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_shards = (len(df) + args.shard_size - 1) // args.shard_size
    todo = [s for s in (args.shards if args.shards is not None else range(n_shards))
            if not os.path.exists(os.path.join(args.out, f"shard_{s:05d}.safetensors"))]
    print(f"{len(df):,d} candidate clips -> {n_shards} shards; {len(todo)} to do", flush=True)
    if not todo:
        return

    vae = WanVAE(dtype=torch.bfloat16)
    ds = lance.dataset(URI)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    tmp_dir = tempfile.mkdtemp(prefix="minidance_face_")
    t0, done, kept, last_log = time.time(), 0, 0, time.time()
    print("encoding... (progress every 30 s; the first lines can take a few minutes)", flush=True)

    for s in todo:
        part = df.iloc[s * args.shard_size:(s + 1) * args.shard_size]
        batches = [part._rowid.tolist()[i:i + args.batch] for i in range(0, len(part), args.batch)]
        get = lambda b: pool.submit(fetch, ds, b, tmp_dir, decode=decode)
        futures = [get(b) for b in batches[:args.workers]]
        latents, masks, keep = [], [], []
        for i in range(len(batches)):
            clips = futures[i].result()
            if i + args.workers < len(batches):
                futures.append(get(batches[i + args.workers]))
            ok = [j for j, c in enumerate(clips) if c is not None]
            if ok:
                x = torch.from_numpy(np.stack([clips[j][0] for j in ok])).cuda()
                x = x.permute(0, 4, 1, 2, 3).float().div(127.5).sub(1)  # B 3 T H W in [-1, 1]
                latents.append(vae.encode(x).to(torch.bfloat16).cpu())
                masks += [torch.from_numpy(clips[j][1]) for j in ok]
                keep += [i * args.batch + j for j in ok]
            done += len(clips)
            if time.time() - last_log > 30:  # downloads are slow: show life inside a shard
                last_log = time.time()
                print(f"  shard {s}: {(i + 1) * args.batch}/{len(part)} candidates, {len(keep)} faces kept, "
                      f"{done / (time.time() - t0):.2f} clips/s", flush=True)
        kept += len(keep)
        base = os.path.join(args.out, f"shard_{s:05d}")
        if not keep:
            print(f"shard {s}: no face clips kept", flush=True)
            continue
        z, m = torch.cat(latents), torch.stack(masks)
        meta = part.iloc[keep][[c for c in META_COLS if c in part.columns]].reset_index(drop=True)
        meta.to_parquet(base + ".parquet")
        save_file({"latents": z.contiguous(), "face_mask": m.contiguous()}, base + ".tmp")
        os.replace(base + ".tmp", base + ".safetensors")  # written last: marks the shard complete
        rate = done / (time.time() - t0)
        left = sum(min(args.shard_size, len(df) - t * args.shard_size) for t in todo[todo.index(s) + 1:])
        print(f"shard {s}: kept {len(z)}/{len(part)} | total kept {kept:,d}/{done:,d} "
              f"({kept / done:.0%}) | {rate:.2f} clips/s, ETA {left / rate / 3600:.1f} h | "
              f"face cells {m.float().mean():.0%}", flush=True)


if __name__ == "__main__":
    main()
