---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-to-video
library_name: pytorch
tags:
- text-to-video
- video-generation
- diffusion-transformer
- flow-matching
- from-scratch
datasets:
- JuniverseLi/minidance-scenery
---

<div align="center">

![logo](assets/logo.png)

[GitHub](https://github.com/JuniverseLi/minidance) | [Dataset](https://huggingface.co/datasets/JuniverseLi/minidance-scenery)

</div>

# MiniDance

A **119.5M-parameter text-to-video DiT trained from scratch** on nature scenery for about ¥12 (≈ $1.7) — roughly 11 hours on a single Colab L4. Give it an English or Chinese prompt and it generates a 33-frame, 256×256 clip (12 fps). Inference runs on an 8 GB GPU.

一个**从零训练的 1.2 亿参数文生视频模型**，专注自然风光，在一张 Colab L4 上训练约 11 小时、成本约 12 元。支持中英文提示词，生成 256×256、33 帧的短视频，8 GB 显卡即可推理。

| | |
|---|---|
| ![](assets/beach_sunset.webp) | ![](assets/waterfall.webp) |
| Ocean waves rolling onto a sandy beach at sunset, golden light on the water. | A waterfall cascading down mossy rocks into a clear pool. |
| ![](assets/aerial_river.webp) | ![](assets/sea_sunset_zh.webp) |
| Aerial view of a turquoise river winding through autumn forest. | 海上日落，金色余晖映在水面。 |

## Files

| File | Description |
|---|---|
| `minidance-base-sft.safetensors` | pretrain + SFT, **recommended** |
| `minidance-base.safetensors` | pretrain only (64k steps) |

Both are the EMA weights in bf16 (239 MB); the model config is stored in the safetensors metadata.

## Usage

```bash
git clone https://github.com/JuniverseLi/minidance && cd minidance
uv sync   # or: pip install -r requirements.txt (after installing CUDA PyTorch)
hf download JuniverseLi/MiniDance --local-dir ./out/MiniDance
uv run python scripts/sample.py --ckpt out/MiniDance/minidance-base-sft.safetensors \
    --prompt "A waterfall cascading down mossy rocks into a clear pool." "海上日落，金色余晖映在水面。" --seed 0
```

The first run also downloads the frozen [Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (~2.7 GB) and [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (~1.5 GB).

## Model

| Component | Model | Params | |
|---|---|---|---|
| Video VAE | Wan2.2 VAE (16× spatial, 4× temporal, 48 channels) | — | frozen |
| Text encoder | Qwen3-0.6B (last hidden state, 1024-d) | 0.6B | frozen |
| Denoiser | MiniDance DiT: dim 768, 12 blocks, 12 heads, 3D RoPE, QK-Norm, adaLN-single, text cross-attention | **119.5M** | **from scratch** |

Trained with rectified flow (logit-normal timesteps, shift 3, 10% caption dropout); sampled with 30 Euler steps and CFG 5.

## Training

| Stage | Data | Steps | LR | Time (L4) |
|---|---|---|---|---|
| Pretrain | 40,429 scenery clips, long + short captions | 64,000 × batch 32 | 2e-4 → 2e-5 | ≈ 10.5 h |
| SFT | 11,496 high-aesthetic clips with motion, short captions | 3,000 × batch 32 | 5e-5 → 5e-6 | ≈ 35 min |

Data: [JuniverseLi/minidance-scenery](https://huggingface.co/datasets/JuniverseLi/minidance-scenery), filtered from [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M).

## Limitations

A "live wallpaper" model: good at forests, waves, rivers and lakes, sunsets, aerial shots, clouds and mountains; weaker on rare scenes (glaciers, lightning, fire, aurora); cannot generate people, animals, vehicles, cities or text, which were excluded from the training data. Motion is mostly gentle (flowing water, drifting clouds, slow camera moves).

## License

Apache 2.0. The frozen Wan2.2 VAE and Qwen3-0.6B are Apache 2.0; the training data derives from OpenVid-1M (CC BY 4.0).
