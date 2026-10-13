"""Download selected OpenVid clips (by Lance row id) as mp4 files.

    python dataset/download_clips.py --parquet data/scenery.parquet --n 50 --out data/samples
"""
import argparse
import os

import lance
import pandas as pd

URI = "hf://datasets/lance-format/Openvid-1M/data/train.lance"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/scenery.parquet")
    ap.add_argument("--n", type=int, default=0, help="random sample size; 0 = all rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/clips")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    if args.n:
        df = df.sample(args.n, random_state=args.seed)
    df = df[[not os.path.exists(os.path.join(args.out, f"{r}.mp4")) for r in df._rowid]]

    ds = lance.dataset(URI)
    ids = df._rowid.tolist()
    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        for rid, blob in zip(chunk, ds.take_blobs("video_blob", ids=chunk)):
            path = os.path.join(args.out, f"{rid}.mp4")
            with blob as f, open(path + ".tmp", "wb") as out:
                out.write(f.read())
            os.replace(path + ".tmp", path)
        print(f"{min(i + args.batch, len(ids))}/{len(ids)}", flush=True)
    df.to_parquet(os.path.join(args.out, "index.parquet"))


if __name__ == "__main__":
    main()
