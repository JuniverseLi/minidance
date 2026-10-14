<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/JuniverseLi/minidance?style=social)](https://github.com/JuniverseLi/minidance/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/JuniverseLi/minidance)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/JuniverseLi/minidance)](https://github.com/JuniverseLi/minidance/commits/main)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/JuniverseLi/minidance/pulls)
[![Model](https://img.shields.io/badge/🤗-MiniDance-blue)](https://huggingface.co/JuniverseLi/MiniDance)
[![Dataset](https://img.shields.io/badge/🤗-minidance--scenery-blue)](https://huggingface.co/datasets/JuniverseLi/minidance-scenery)

中文 | [English](./README_en.md)

</div>

* 此开源项目旨在完全从 0 开始，仅用**约 12 块钱**成本、一张 Colab L4 训练**约 11 小时**，训练出规模为 **120M** 的文生视频（Text-to-Video）模型 **MiniDance**。
* MiniDance 专注于**自然风光动态视频**：海浪、瀑布、森林、日落、云海……输入一句中文或英文提示词，生成一段 33 帧的短视频，8 GB 显存的显卡即可推理。
* 项目开源了视频 DiT 的极简结构与完整训练链路，覆盖数据筛选、VAE 选型、latent 预编码、短字幕构建、预训练（Pretrain）、监督微调（SFT）、推理与可视化的全过程代码。
* 视频 DiT、Rectified Flow 训练与采样等核心算法均使用 PyTorch 原生代码从 0 实现，不依赖 `diffusers` 等第三方库的高层封装（只借用冻结的 VAE 与文本编码器）。
* 希望此项目能成为视频生成方向的一个可复现、可理解、可扩展的起点，像 [MiniMind](https://github.com/jingyaogong/minimind) 之于语言模型那样。

> 注：本项目基于 Apache 2.0 协议开源，完全免费。"约 11 小时" 指预训练 + SFT 在单张 NVIDIA L4（Colab）上的实测训练时长，"约 12 块钱" 指对应时段的 Colab 计算单元费用（L4 每小时 1.54 个计算单元，100 单元 9.99 美元）。本项目与字节跳动 / Seedance 无关，名字仅为致敬。

---

https://github.com/user-attachments/assets/90097d0e-61e5-40c8-a22c-d9be2a5fdc7f

<div align="center">

<table>
  <tr>
    <td align="center"><img src="./images/gallery/beach_sunset.webp" width="190"/></td>
    <td align="center"><img src="./images/gallery/waterfall.webp" width="190"/></td>
    <td align="center"><img src="./images/gallery/aerial_river.webp" width="190"/></td>
    <td align="center"><img src="./images/gallery/wheat_field.webp" width="190"/></td>
  </tr>
  <tr>
    <td align="center"><sub>Ocean waves rolling onto a sandy beach at sunset, golden light on the water.</sub></td>
    <td align="center"><sub>A waterfall cascading down mossy rocks into a clear pool.</sub></td>
    <td align="center"><sub>Aerial view of a turquoise river winding through autumn forest.</sub></td>
    <td align="center"><sub>Golden wheat field swaying in the wind at sunset.</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="./images/gallery/misty_forest.webp" width="190"/></td>
    <td align="center"><img src="./images/gallery/waves_rocks.webp" width="190"/></td>
    <td align="center"><img src="./images/gallery/sea_sunset_zh.webp" width="190"/></td>
    <td align="center"><img src="./images/gallery/snow_mountain_lake_zh.webp" width="190"/></td>
  </tr>
  <tr>
    <td align="center"><sub>Sunlight streaming through tall pine trees in a misty forest.</sub></td>
    <td align="center"><sub>Ocean waves crashing against dark rocks.</sub></td>
    <td align="center"><sub>海上日落，金色余晖映在水面。</sub></td>
    <td align="center"><sub>湖面倒映着雪山，微风吹起涟漪。</sub></td>
  </tr>
</table>

<sub>以上均为 `minidance-base-sft` 的原始输出（256×256，33 帧，12 fps），未经挑图以外的任何后处理；提示词与随机种子见 [demo/clips.json](./demo/clips.json)。</sub>

</div>

---

# 📌 项目介绍

Sora、Wan、HunyuanVideo、Kling……视频生成模型的效果一次次让人惊叹，但它们动辄数十亿参数、上万张 GPU 的训练成本，让个人开发者只能停留在"调用 API"或"跑推理"的层面。打开视频生成的黑盒子，真正理解 latent 是怎么被一步步"去噪"成一段会动的画面，本应是一件很有意思的事。

MiniDance 想回答一个问题：**一个普通人，花一顿饭钱，能不能从零训练出一个能"动起来"的文生视频模型？**

答案是可以的——只要把问题收窄：

- **收窄题材**：只做自然风光（动态壁纸）。没有人物、动物和文字，模型只需学会水流、云动、光影这类"纹理型"运动。
- **收窄分辨率**：256×256、33 帧。借助 Wan2.2 VAE 的 16×16×4 压缩，一段视频只有 576 个 token。
- **站在巨人的肩膀上**：VAE（Wan2.2）和文本编码器（Qwen3-0.6B）直接使用冻结的预训练权重，从零训练的只有中间的视频 DiT。这和 [MiniMind-V](https://github.com/jingyaogong/minimind-v) 冻结 CLIP 视觉编码器是同一个思路——我们真正要学习和复现的，是"生成"这件事本身。

😊 一起感受让画面动起来的乐趣吧！

---

#### 🎉 本项目包含以下内容

- 完整的视频 DiT 结构代码（[model/dit.py](./model/dit.py)，约 200 行）：3D RoPE、QK-Norm、adaLN-single、文本 Cross-Attention、SwiGLU。
- Rectified Flow / Flow Matching 的训练与 Euler + CFG 采样（[model/flow.py](./model/flow.py)，约 50 行）。
- 数据管线：从 OpenVid-1M（约 94 万条）中按规则筛选出 43,562 条高质量自然风光片段，并预编码为 latent。
- 开源数据集：latent 分片 + 中英文短字幕，下载即可训练，无需再处理视频。
- 预训练、SFT 两阶段训练脚本，支持限时训练、断点续训、`torch.compile`、bf16、EMA。
- 可直接在 Colab（网页版或 VS Code 连接 Colab 运行时）上运行的 notebook。
- 推理脚本：8 GB 显卡即可生成 H.264 mp4 / GIF。
- 用 [Remotion](https://www.remotion.dev/) 制作的架构图与演示视频源码（[demo/](./demo)）。

#### 🎉 已发布模型列表

| 模型 | 参数量 | 说明 | Release |
|------|--------|------|---------|
| minidance-base-sft | 119.5M | 预训练 + SFT，**推荐使用** | 2026.10.14 |
| minidance-base | 119.5M | 仅预训练（64k 步） | 2026.10.14 |

---

#### 📝 更新日志

<details>
<summary> <b>🔥 2026-10-14</b> </summary>

- 项目首次开源：代码、权重、数据集
- 发布 `minidance-base` / `minidance-base-sft`（119.5M）
- 架构图与演示视频

</details>

---

# 📌 快速开始

<details>
<summary>本人的软硬件配置（供参考）</summary>

* 本地：NVIDIA GeForce RTX 4060 Laptop（8 GB），Windows 11，CUDA 12.8，Python 3.12（推理、小规模调试）
* 训练：Google Colab，NVIDIA L4（24 GB）
* 依赖见 [pyproject.toml](./pyproject.toml) / [requirements.txt](./requirements.txt)

</details>

## 第0步

```bash
# 克隆仓库
git clone --depth 1 https://github.com/JuniverseLi/minidance
cd minidance

# 方式1：uv（推荐，会自动安装 CUDA 版 PyTorch 与 Python 3.12）
uv sync
# 方式2：pip（先自行安装 CUDA 版 PyTorch）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> 下文命令以 `python xxx.py` 书写；使用 uv 时请写成 `uv run python xxx.py`。

## Ⅰ 🚀 模型推理

### 1' 下载模型

```bash
hf download JuniverseLi/MiniDance --local-dir ./out/MiniDance
```

> 首次运行还会自动下载冻结的 Wan2.2 VAE（约 2.7 GB）与 Qwen3-0.6B（约 1.5 GB）。国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com`。

### 2' 生成视频

```bash
python scripts/sample.py --ckpt out/MiniDance/minidance-base-sft.safetensors \
    --prompt "A waterfall cascading down mossy rocks into a clear pool." "海上日落，金色余晖映在水面。" \
    --seed 0 --gif
```

> 每条提示词输出一个 mp4（加 `--gif` 另存 GIF）到 `out/samples/`。常用参数：`--steps` 采样步数（默认 30）、`--cfg` 引导强度（默认 5）、`--seed` 随机种子。同一个 seed 下，每条提示词的结果与同批的其他提示词无关，便于复现。

### 3'（计划中）WebUI

基于 Gradio 的生成界面，以及类似 MiniMind 的训练过程可视化（loss、学习率、样例随步数变化），见 [TODO](#-todo)。

## Ⅱ 🛠️ 模型训练

<details>
<summary>注：提前确认 Torch 的可用后端</summary>

```python
import torch
print(torch.cuda.is_available())
```

训练需要 CUDA GPU。`base` 模型 batch 32 约占 14 GB 显存；8 GB 显卡可以用 `--batch 4` 或更小的 `--preset small` 跑通流程。

</details>

### 1' 下载数据

```bash
hf download JuniverseLi/minidance-scenery --repo-type dataset --local-dir ./data
```

> 下载后应有 `data/latents/shard_*.safetensors`（86 个分片，约 9 GB）与 `data/short_captions.jsonl`。
> 想从原始视频重新构建数据集，见下文 [数据介绍](#-数据介绍)。

### 2' 开始训练

<details>
<summary>💡 限时训练与断点续训</summary>

所有训练脚本都支持 `--max_minutes`：到时间后自动出样例、保存 checkpoint 再退出，适合 Colab 这类有时长限制的环境。加上 `--resume` 后会从 `<out>/ckpt.pt` 接着训练（模型、EMA、优化器与步数全部恢复）：

```bash
python trainer/train_pretrain.py ... --max_minutes 240 --resume
```

注意：学习率按 `--steps` 做余弦衰减。分多次跑完一个训练时，每次都要用**同一个** `--steps`。

</details>

#### 2.1 预训练（必须）

```bash
python trainer/train_pretrain.py --preset base --batch 32 --lr 2e-4 --warmup 1000 --steps 64000 \
    --sample_every 1000 --save_every 2000 --compile --resume --out out/pretrain
```

> 训练过程中每 `--sample_every` 步会在 `out/pretrain/samples/` 生成一张样例 GIF（4 条固定提示词 + 2 条训练集字幕）。

#### 2.2 监督微调（推荐）

```bash
python trainer/train_sft.py --init out/pretrain/ckpt.pt --compile --resume --out out/sft
```

> 默认只用美学分 ≥ 6.0、运动分 ≥ 0.5 的 11,496 条片段和短字幕，lr 5e-5，3000 步。

#### 2.3 导出与测试

```bash
python scripts/export_weights.py out/sft/ckpt.pt out/minidance-base-sft.safetensors   # 只保留 EMA 权重，bf16，239 MB
python scripts/sample.py --ckpt out/minidance-base-sft.safetensors --prompt "White clouds drifting over green mountains."
```

<details>
<summary>注：在 Colab 上训练</summary>

[notebooks/](./notebooks) 下提供了三个 notebook，数据放在 Google 网盘 `MyDrive/minidance/`：

- `train_colab.ipynb`：在 Colab 网页版里运行的预训练
- `train_vscode.ipynb`：在 VS Code 里连接 Colab 运行时的预训练
- `sft_vscode.ipynb`：SFT
- `encode_latents_colab.ipynb`：从原始视频编码 latent（只需运行一次）

</details>

---

# 📌 数据介绍

## Ⅰ 数据来源

原始数据来自 [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M)（使用其 [Lance 格式版本](https://huggingface.co/datasets/lance-format/Openvid-1M)，可以按行流式读取，无需下载全部视频）。每条片段带有一段较长的英文描述，以及美学分、运动分、时序一致性、镜头运动等标注。

## Ⅱ 筛选规则

[dataset/filter_openvid.py](./dataset/filter_openvid.py) 只用元数据做筛选，不需要下载视频：

1. **来源**：只保留 Pixabay / Pexels / Mixkit 的素材（画质高、无水印）。
2. **题材**：用正则在字幕中匹配森林、海洋、河湖、云、山、日落、雪、瀑布、星空、雾等 13 类场景。
3. **排除**：人物、动物、车辆、建筑、文字、CG / 数字艺术等（带否定语境的句子不误杀，例如 "no people"）。
4. **质量**：时序一致性 ≥ 0.993，运动分 0.3 ~ 30；只保留静止或缓慢运镜的片段。
5. **去重**：同一原视频最多取 4 条。

最终得到 **43,562** 条片段，场景分布如下：

| 森林 | 海洋 | 河湖 | 云 | 山 | 日落 | 雪 | 瀑布 | 星空 | 雾 | 雨 / 雷暴 | 极光 | 火 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,166 | 9,214 | 5,450 | 4,358 | 3,817 | 3,305 | 2,905 | 2,086 | 1,049 | 905 | 156 | 76 | 75 |

## Ⅲ 短字幕

OpenVid 的原始字幕很长（常有五六句），而用户实际输入的提示词通常只有一句话。我们用大语言模型把每条长字幕压缩成一句**英文**和一句**中文**短字幕，同时打上主体标签；带有人物、动物、车辆、CG、文字标签的 3,133 条在训练时被丢弃，剩下 **40,429** 条。

```json
{"rowid": 197568497560, "en": "A waterfall with brownish water cascades down a rocky cliff, surrounded by green trees and bushes.", "zh": "棕色水流瀑布从岩石悬崖倾泻而下，四周绿树灌木环绕。", "issue": "none"}
```

训练时每条样本的字幕随机取自：截断后的长字幕（30%）、英文短字幕、中文短字幕（短字幕中 30% 为中文）。SFT 阶段只用短字幕。

## Ⅳ latent 数据集

每段 2.67 秒的片段均匀抽取 33 帧（约 12 fps），整帧缩放到 256×256（不裁剪），用冻结的 Wan2.2 VAE 编码为 `48 × 9 × 16 × 16` 的 latent（bf16）。每 512 条存为一个分片：

```text
data/latents/
├── shard_00000.safetensors   # {"latents": bf16 [512, 48, 9, 16, 16]}
├── shard_00000.parquet       # _rowid, caption, scene, source, aesthetic_score, motion_score, ...
└── ...                       # 共 86 个分片，约 9 GB
```

训练时整个数据集直接载入内存，**训练过程完全不需要 VAE**（只在出样例时临时加载），所以 8 GB 显卡也能跑通训练。

<details>
<summary>从原始视频重新构建数据集</summary>

```bash
python dataset/fetch_openvid_meta.py --out data/openvid_meta                    # 导出元数据（约 94 万条，可断点续跑）
python dataset/filter_openvid.py --meta data/openvid_meta --out data/scenery.parquet
python dataset/encode_latents.py --parquet data/scenery.parquet --out data/latents --batch 16 --workers 8
```

编码是一次性的预处理，速度主要受视频下载速度限制，建议在 Colab 上运行 `notebooks/encode_latents_colab.ipynb`。短字幕的生成脚本未开源，可直接使用数据集中的 `short_captions.jsonl`。

</details>

---

# 📌 模型

## 结构

![pipeline](./images/arch_pipeline.png)

MiniDance 由三部分组成，其中只有中间的 DiT 是从零训练的：

| 组件 | 模型 | 参数量 | 状态 |
|---|---|---|---|
| 视频 VAE | [Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)（空间 16×、时间 4×、48 通道，因果结构） | — | 冻结 |
| 文本编码器 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)（最后一层 hidden，1024 维，中英文） | 0.6B | 冻结 |
| 去噪网络 | **MiniDance DiT** | **119.5M** | **从零训练** |

VAE 是在 LTX-Video 0.9.7、Wan2.2、Open-Sora DC-AE 三者中对比重建质量与压缩率后选定的（[scripts/s2_vae_bench.py](./scripts/s2_vae_bench.py)）：Wan2.2 VAE 在 256×256 上重建 PSNR 约 31.5 dB，bf16 编码与 fp32 几乎无差别、速度快 1.7 倍。

![dit](./images/arch_dit.png)

DiT 的主要设计：

* **时空联合注意力 + 3D RoPE**：576 个 token 之间做完整的 self-attention；head 维度按 1 : 1.5 : 1.5 分给时间、高、宽三个轴做旋转位置编码。
* **QK-Norm**：对 Q、K 做 RMSNorm，bf16 训练更稳定。
* **adaLN-single**（来自 PixArt-α）：时间步只经过一个共享 MLP 得到 6 组调制参数，每个 Block 只额外学习一张很小的偏置表，比原版 adaLN-Zero 省下约 1/4 的参数。
* **零初始化残差**：每个 Block 的输出投影和最后的输出层都初始化为 0，训练开始时整个网络是恒等映射。
* **图片 / 视频混合训练**：Wan2.2 VAE 是因果结构，latent 的第 0 帧只编码视频第 0 帧，等价于一张图片。训练时 20% 的批次只取第 0 帧当图片训练，帮助模型更快学会画面构图。

## 模型配置

| Preset | 参数量 | dim | depth | heads | 用途 |
|---|---|---|---|---|---|
| tiny | 7.1M | 256 | 6 | 4 | 调试 |
| small | 20.8M | 384 | 8 | 6 | 过拟合测试 |
| **base** | **119.5M** | **768** | **12** | **12** | **发布模型** |

## 训练目标

采用 Rectified Flow（Flow Matching）：

$$x_t = (1-t)\,x_0 + t\,\varepsilon,\qquad \mathcal{L} = \big\|\, v_\theta(x_t, t, y) - (\varepsilon - x_0) \,\big\|^2$$

* 时间步 $t$ 从 logit-normal 分布采样后再做 shift（shift = 3），把更多训练放在噪声较大、决定构图和运动的阶段。
* 10% 的字幕被置空，用于 classifier-free guidance；推理时用 30 步 Euler 采样，CFG = 5。

---

# 📌 实验

## Ⅰ 训练开销

- **硬件**：Google Colab，单张 NVIDIA L4（24 GB）
- **单价**：L4 每小时 1.54 个计算单元，100 单元 = 9.99 美元，约合 **1.1 元 / 小时**
- **速度**：base 模型，batch 32，`torch.compile` + bf16，约 57 clips/s（6,400 步 / 小时）

| 阶段 | 数据 | 步数 | 学习率 | 时长 | 成本 |
|---|---|---|---|---|---|
| 预训练 | 40,429 条，长 / 短字幕混合 | 64,000 | 2e-4 → 2e-5 | ≈ 10.5 h | ≈ 11.5 ￥ |
| SFT | 11,496 条，仅短字幕 | 3,000 | 5e-5 → 5e-6 | ≈ 0.6 h | ≈ 0.6 ￥ |
| **合计** | | | | **≈ 11 h** | **≈ 12 ￥** |

> latent 编码是一次性的预处理，主要耗时在下载视频上；直接下载我们开源的 latent 数据集可以跳过这一步。

## Ⅱ 训练过程

同一组提示词、同一个随机种子在不同训练步数下的生成结果（每段视频取中间帧）。从左到右依次是：海滩日落、云海青山、瀑布、中文提示词"海上日落"。

<div align="center">
<img src="./images/progress.jpg" width="640"/>
</div>

* **1k 步**：只有大致的色调。
* **5k 步**（约 1 小时）：题材全部正确，中文提示词也生效了，但画面像一块块色斑。
* **20k 步**：结构清晰起来，开始出现太阳倒影、瀑布分层这样的细节。
* **64k 步**：学习率衰减阶段继续提升清晰度，瀑布、草叶都出现了细节。
* **SFT 3k 步**：画面更干净、更贴合提示词（白云与山脊分离、日落落在海平线上），星空这类少见场景的提升最明显。

> flow matching 的 loss 主要由随机采到的噪声强度决定，所以训练后期 loss 一直在 0.2 ~ 0.3 之间波动、看起来"不怎么降"，这是正常的；画质要看样例。

## Ⅲ 能力边界

| 擅长 | 一般 | 较弱 / 不支持 |
|---|---|---|
| 森林、海浪、河湖、日落、航拍、花草、云、山 | 瀑布、雾、雪、秋叶、沙漠、星空 | 冰川、闪电、火、极光、彩虹、雨；人物、动物、车辆、城市、文字 |

MiniDance 是一个"动态壁纸"模型：运动以水流、云动、草木摇曳和缓慢运镜为主，不擅长剧烈的动作。数据里少见的题材（星空、极光等）效果较弱，训练数据中排除的题材（人、动物、车）基本无法生成。

---

# 📌 TODO

- [ ] 上传权重与数据集到 HuggingFace / ModelScope
- [ ] Gradio 生成 WebUI 与在线 Demo
- [ ] 训练可视化 WebUI：loss、学习率、样例随步数的变化（类似 MiniMind）
- [ ] 多卡训练（DDP）
- [ ] 更高分辨率、更长时长
- [ ] 图生视频（I2V）

---

# 📌 致谢

> [!NOTE]
> 如果 `MiniDance` 对您有所帮助，欢迎在 GitHub 上点亮一个 ⭐<br/>
> 欢迎通过 Issues 交流反馈，或提交 PR 一起改进项目！

感谢以下优秀的项目与论文：

- [MiniMind](https://github.com/jingyaogong/minimind)：本项目的灵感来源与文档结构参考
- [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M)：训练数据来源
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)：视频 VAE
- [Qwen3](https://github.com/QwenLM/Qwen3)：文本编码器
- [DiT](https://arxiv.org/abs/2212.09748)、[PixArt-α](https://arxiv.org/abs/2310.00426)（adaLN-single）、[Stable Diffusion 3](https://arxiv.org/abs/2403.03206)（Rectified Flow 与 logit-normal 采样）、[Flow Matching](https://arxiv.org/abs/2210.02747)
- [Remotion](https://www.remotion.dev/)：架构图与演示视频

# ⚖️ 开源协议

本项目代码与 MiniDance 权重采用 [Apache License 2.0](LICENSE) 开源协议。

使用时请同时遵守第三方组件的协议：Wan2.2 VAE 与 Qwen3-0.6B 为 Apache 2.0；训练数据来自 OpenVid-1M（CC BY 4.0），其视频素材来自 Pixabay、Pexels、Mixkit，版权归原作者所有。本项目只分发 VAE latent 与字幕，不分发原始视频。
