"""MiniDance Face: teach the released base model face close-up videos while keeping its scenery.

Same training loop as pretraining, but
  - data:    face-centred crops of single-person close-ups from dataset/encode_faces.py, mixed with
             the SFT-quality scenery clips (--scenery_ratio of every batch, short en / zh prompts) so
             the model keeps what it already knows
  - loss:    flow loss up-weighted on the face (1 + --face_weight on face latent cells)
  - weights: start from the pretrained EMA weights (minidance-base), no extra parameters

    python trainer/train_face.py --init out/minidance-base.safetensors --out out/face --compile

A second round can start from the first one's EMA weights (it already knows hair, skin, clothes):

    python trainer/train_face.py --init out/face/ckpt.pt --out out/face_tight --steps 10000
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.latent_dataset import LatentDataset
from trainer.train_pretrain import build_parser, train

# OpenVid caption style ("The video features ..."): the face clips are only captioned this way
PROMPTS = [
    "The video features a young woman with long dark hair, smiling at the camera.",
    "The video features a middle-aged man with glasses and short hair, talking calmly.",
    "The video shows an elderly woman with white hair, slowly turning her head.",
    "海上日落，小浪拍岸，金色余晖映在水面。",  # scenery must survive the face training
]


class FaceSceneryMix(torch.utils.data.Dataset):
    """Face items first, then scenery items with an all-zero face mask (= plain loss)."""

    def __init__(self, face, scenery):
        self.face, self.scenery = face, scenery
        self.zero_mask = torch.zeros_like(face.face_mask[0])
        self.index = [0] + ([len(face)] if scenery else [])  # one face + one scenery training prompt

    def __len__(self):
        return len(self.face) + (len(self.scenery) if self.scenery else 0)

    def __getitem__(self, k):
        if k < len(self.face):
            return self.face[k]
        return (*self.scenery[k - len(self.face)], self.zero_mask)

    def prompt(self, k):
        if k < len(self.face):
            return self.face.prompt(self.face.index[k])
        return self.scenery.prompt(self.scenery.index[k - len(self.face)])

    def weights(self, scenery_ratio):
        """Sampling weights so that ~scenery_ratio of the samples are scenery."""
        if not self.scenery:
            return torch.ones(len(self.face), dtype=torch.double)
        return torch.cat([torch.full((len(self.face),), (1 - scenery_ratio) / len(self.face), dtype=torch.double),
                          torch.full((len(self.scenery),), scenery_ratio / len(self.scenery), dtype=torch.double)])


def main():
    ap = build_parser()
    ap.add_argument("--scenery_dir", default="data/latents", help="released scenery latents ('' = faces only)")
    ap.add_argument("--scenery_captions", default="data/short_captions.jsonl")
    ap.add_argument("--scenery_ratio", type=float, default=0.25, help="share of scenery clips in each batch")
    ap.set_defaults(data_dir="data/face_latents_tight", short_captions=None, preset="base", batch=64,
                    lr=1e-4, warmup=500, steps=8000, p_image=0.1, face_weight=2.0,
                    sample_every=2000, save_every=1000, out="out/face")
    args = ap.parse_args()

    # face shards have no short captions: always the long OpenVid caption, nothing dropped; cut to a
    # random 1-3 sentences so one-line prompts work too (with 3 only, short prompts gave no face)
    face = LatentDataset(args.data_dir, args.shards, args.short_captions, p_long=1.0, drop_issues=set(),
                         min_sentences=1)
    assert face.face_mask is not None, f"{args.data_dir} has no face_mask: build it with dataset/encode_faces.py"
    if args.max_clips:
        face.index = face.index[:args.max_clips]
    scenery = None
    if args.scenery_dir and args.scenery_ratio > 0:
        # the same clips and captions as SFT: good-looking, moving scenery with short en / zh prompts
        scenery = LatentDataset(args.scenery_dir, None, args.scenery_captions, p_long=0.0)
        scenery.select(min_aesthetic=6.0, min_motion=0.5, need_short=True)
    ds = FaceSceneryMix(face, scenery)
    print(f"face {len(face):,d} clips + scenery {len(scenery) if scenery else 0:,d} clips, "
          f"scenery share {args.scenery_ratio if scenery else 0:.0%}")
    sampler = torch.utils.data.WeightedRandomSampler(ds.weights(args.scenery_ratio), len(ds))
    train(args, ds, sampler, prompts=PROMPTS)


if __name__ == "__main__":
    main()
