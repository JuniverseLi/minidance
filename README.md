# MiniDance

> 🚧 开发中，尚未发布模型权重。

从零训练一个小型**文生视频**模型，专注自然风光（云海、海浪、瀑布、日落……）。
生成模型（视频 DiT）完全用 PyTorch 手写、从随机初始化开始训练；视频 VAE（Wan2.2）和文本编码器使用冻结的开源权重。

**声明**：本项目与字节跳动 / Seedance 无关，名字仅为致敬。

## 进度

- [x] S1 数据筛选：从 OpenVid-1M 中选出 43,562 条图库来源（Pixabay / Pexels / Mixkit）的风景片段
- [x] S2 VAE 选型：对比 LTX-Video 0.9.7、Wan2.2、Open-Sora DC-AE，选定 Wan2.2 VAE
- [ ] S3 小规模训练：DiT + flow matching，过拟合测试
- [ ] S4 成本测算

## 数据处理

```bash
uv sync
uv run python dataset/fetch_openvid_meta.py      # 拉取 OpenVid-1M 元数据（约 10 分钟）
uv run python dataset/filter_openvid.py          # 筛选风景片段
uv run python dataset/encode_latents.py          # 流式下载 + Wan2.2 VAE 编码，按分片保存
```

全量编码建议在 Colab 上运行：[notebooks/encode_latents_colab.ipynb](notebooks/encode_latents_colab.ipynb)

## 数据来源

视频来自 [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M)（CC-BY-4.0），仅使用其中 Pixabay、Pexels、Mixkit 来源的片段。本项目不分发原始视频。
