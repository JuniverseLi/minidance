"""MiniDance SFT: fine-tune the pretrained model on the best clips with short prompts.

Same training loop as pretraining, but
  - data:     only high-aesthetic clips that actually move (top ~30%), rare scenes sampled more often
  - captions: only our short English / Chinese prompts (what users actually type)
  - weights:  start from the pretrained EMA weights, lower learning rate, a few thousand steps

    uv run python trainer/train_sft.py --init out/s4_base/ckpt.pt --preset base --out out/sft
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.latent_dataset import LatentDataset
from trainer.train_pretrain import build_parser, train


def main():
    ap = build_parser()
    ap.add_argument("--min_aesthetic", type=float, default=6.0)
    ap.add_argument("--min_motion", type=float, default=0.5, help="drop near-static clips")
    ap.add_argument("--scene_balance", type=float, default=0.5, help="0 = natural scene mix, 1 = uniform")
    ap.set_defaults(lr=5e-5, warmup=100, steps=3000, p_image=0.1, batch=32, preset="base",
                    sample_every=500, save_every=500, out="out/sft")
    args = ap.parse_args()
    assert args.init or args.resume, "SFT starts from a pretrained model: pass --init <pretrain ckpt.pt>"

    ds = LatentDataset(args.data_dir, args.shards, args.short_captions, p_long=0.0)
    ds.select(args.min_aesthetic, args.min_motion, need_short=True)
    if args.max_clips:
        ds.index = ds.index[:args.max_clips]
    sampler = torch.utils.data.WeightedRandomSampler(ds.scene_weights(args.scene_balance), len(ds))
    train(args, ds, sampler)


if __name__ == "__main__":
    main()
