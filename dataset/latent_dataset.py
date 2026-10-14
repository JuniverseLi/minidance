"""Training data: pre-encoded latent shards + captions.

Each item is (raw latent [48, 9, 16, 16] bf16, caption str). The caption is drawn at random from
the long OpenVid caption (truncated) and, when available, our short English / Chinese prompts.
"""
import glob
import json
import os
import random
import re

import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

# clips whose main subject is one of these are dropped (labels come with the short captions)
BAD_ISSUES = {"animal", "person", "vehicle", "cg", "text"}


def truncate(caption, sentences=3):
    return " ".join(re.split(r"(?<=[.!?])\s+", caption.strip())[:sentences])


class LatentDataset(Dataset):
    def __init__(self, shard_dir, shards=None, short_captions=None, p_long=0.3, p_zh=0.3):
        """p_long: prob. of the (truncated) long caption; p_zh: share of Chinese among short ones."""
        files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.safetensors")))
        if shards is not None:
            files = [f for f in files if int(os.path.basename(f)[6:11]) in set(shards)]
        assert files, f"no shards in {shard_dir}"
        self.latents = torch.cat([load_file(f)["latents"] for f in files])
        meta = pd.concat([pd.read_parquet(f.replace(".safetensors", ".parquet")) for f in files])
        self.rowids, self.long = meta._rowid.tolist(), meta.caption.tolist()
        self.meta = meta.reset_index(drop=True)

        self.short = {}
        if short_captions and os.path.exists(short_captions):
            for line in open(short_captions, encoding="utf-8"):
                r = json.loads(line)
                self.short[r["rowid"]] = r
        keep = [i for i, rid in enumerate(self.rowids)
                if self.short.get(rid, {}).get("issue", "none") not in BAD_ISSUES]
        self.index = keep
        self.p_long, self.p_zh = p_long, p_zh
        print(f"LatentDataset: {len(files)} shards, {len(self.rowids):,d} clips, "
              f"{sum(r in self.short for r in self.rowids):,d} with short captions, "
              f"{len(self.rowids) - len(keep):,d} dropped by issue label -> {len(keep):,d}")

    def select(self, min_aesthetic=0.0, min_motion=0.0, need_short=False):
        """Keep only clips passing these thresholds (used for SFT)."""
        m = self.meta
        self.index = [i for i in self.index
                      if m.aesthetic_score[i] >= min_aesthetic and m.motion_score[i] >= min_motion
                      and (not need_short or self.rowids[i] in self.short)]
        print(f"select(aesthetic >= {min_aesthetic}, motion >= {min_motion}, need_short={need_short})"
              f" -> {len(self.index):,d} clips")

    def scene_weights(self, power=0.5, min_count=100):
        """Per-item sampling weights ~ count(scene)^-power: 0 = natural mix, 1 = every scene equally often.

        Counts are floored at min_count so a scene with a handful of clips isn't repeated endlessly.
        """
        scenes = self.meta.scene.iloc[self.index]
        counts = scenes.value_counts()
        print("scenes:", ", ".join(f"{k} {v}" for k, v in counts.items()))
        w = scenes.map(counts).clip(lower=min_count) ** -power
        return torch.tensor(w.to_numpy(), dtype=torch.double)

    def prompt(self, i):
        """A fixed caption for clip i, for the sample grid during training."""
        s = self.short.get(self.rowids[i])
        return s["en"] if s and self.p_long == 0 else truncate(self.long[i])

    def __len__(self):
        return len(self.index)

    def caption(self, i):
        s = self.short.get(self.rowids[i])
        if s is None or random.random() < self.p_long:
            return truncate(self.long[i])
        return s["zh"] if random.random() < self.p_zh else s["en"]

    def __getitem__(self, k):
        i = self.index[k]
        return self.latents[i], self.caption(i)
