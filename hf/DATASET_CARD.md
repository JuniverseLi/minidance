---
license: cc-by-4.0
language:
- en
- zh
task_categories:
- text-to-video
size_categories:
- 10K<n<100K
tags:
- video
- latents
- nature
- scenery
pretty_name: MiniDance Scenery Latents
---

# MiniDance Scenery Latents

**43,562 nature-scenery video clips**, pre-encoded with the [Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) and paired with English / Chinese short captions — ready to train a text-to-video model with no video decoding at all. Used to train [MiniDance](https://huggingface.co/JuniverseLi/MiniDance) ([GitHub](https://github.com/JuniverseLi/minidance)).

**43,562 条自然风光视频片段**，已用 Wan2.2 VAE 编码为 latent，并配有中英文短字幕，下载即可训练文生视频模型，无需处理任何视频。

## Contents

```text
latents/
├── shard_00000.safetensors   # {"latents": bf16 [512, 48, 9, 16, 16]}
├── shard_00000.parquet       # metadata, one row per clip, same order
├── ...                       # 86 shards, ~9 GB
short_captions.jsonl          # {"rowid", "en", "zh", "issue"}, one line per clip
```

* **Video → latent**: each 2.67 s clip → 33 evenly spaced frames (~12 fps), full frame resized to 256×256 (no crop) → Wan2.2 VAE (bf16, posterior mean) → `48 × 9 × 16 × 16`. The VAE is causal, so latent frame 0 encodes only video frame 0 (usable as an image). Latents are **raw** (not normalized); normalize with the per-channel `latents_mean` / `latents_std` from the VAE config.
* **Metadata** (`shard_*.parquet`): `_rowid` (row id in the OpenVid-1M Lance dataset), `caption` (original long caption), `scene`, `source`, `video_id`, `camera_motion`, `aesthetic_score`, `motion_score`.
* **Short captions** (`short_captions.jsonl`, joined on `rowid` = `_rowid`): one English and one Chinese sentence condensed from the long caption by an LLM, plus a subject label `issue` ∈ {`none`, `road`, `structure`, `animal`, `cg`, `person`, `vehicle`, `text`, …}. MiniDance drops `animal`, `person`, `vehicle`, `cg` and `text` (3,133 clips) at training time.

```json
{"rowid": 197568497560, "en": "A waterfall with brownish water cascades down a rocky cliff, surrounded by green trees and bushes.", "zh": "棕色水流瀑布从岩石悬崖倾泻而下，四周绿树灌木环绕。", "issue": "none"}
```

## Loading

```python
import json, pandas as pd
from safetensors.torch import load_file

z = load_file("latents/shard_00000.safetensors")["latents"]   # bf16 [512, 48, 9, 16, 16]
meta = pd.read_parquet("latents/shard_00000.parquet")
short = {r["rowid"]: r for r in map(json.loads, open("short_captions.jsonl", encoding="utf-8"))}
print(z.shape, meta.caption[0], short[meta._rowid[0]]["zh"])
```

To decode a latent back to video, use `model/vae.py` in the MiniDance repo (`WanVAE().decode(z)`).

## How it was built

Clips come from [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) (read from its [Lance version](https://huggingface.co/datasets/lance-format/Openvid-1M)) and were filtered using metadata only:

1. Sources: Pixabay, Pexels, Mixkit.
2. Scenes matched in the caption: forest, ocean, river/lake, clouds, mountain, sunset, snow, waterfall, night sky, fog, rain/storm, aurora, fire.
3. Excluded: people, animals, vehicles, buildings, text, CG / digital art.
4. Temporal consistency ≥ 0.993, motion score 0.3–30, static or slow camera.
5. At most 4 clips per source video.

| forest | ocean | river/lake | clouds | mountain | sunset | snow | waterfall | night sky | fog | rain/storm | aurora | fire |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,166 | 9,214 | 5,450 | 4,358 | 3,817 | 3,305 | 2,905 | 2,086 | 1,049 | 905 | 156 | 76 | 75 |

Code: [dataset/](https://github.com/JuniverseLi/minidance/tree/main/dataset) in the MiniDance repo.

## License

CC BY 4.0, following OpenVid-1M. The underlying videos come from Pixabay, Pexels and Mixkit and belong to their authors; this dataset contains only VAE latents, metadata and captions, not the original videos. Please cite OpenVid-1M if you use it:

```bibtex
@article{nan2024openvid,
  title={OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation},
  author={Nan, Kepan and Xie, Rui and Zhou, Penghao and Fan, Tiehan and Yang, Zhenheng and Chen, Zhijie and Li, Xiang and Yang, Jian and Tai, Ying},
  journal={arXiv preprint arXiv:2407.02371},
  year={2024}
}
```
