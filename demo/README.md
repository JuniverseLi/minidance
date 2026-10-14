# MiniDance 演示视频与架构图

用 [Remotion](https://www.remotion.dev/)（React 写视频）制作，源码在 `src/`：

| Composition | 输出 | 说明 |
|---|---|---|
| `Demo` | 演示视频，1920×1080，30 fps，约 44 秒 | logo → 猜价格 → 变魔术（真实去噪过程）→ 提示词与生成结果 → 流程图动画 → 结尾 |
| `ArchPipeline` | `../images/arch_pipeline.png` | 训练与推理流程图 |
| `ArchDiT` | `../images/arch_dit.png` | DiT 结构图 |

```bash
# 1. 生成视频片段、去噪过程帧、透明背景 logo（demo/public/）和 README 用的 ../images/gallery/*.webp
cd ..
uv run python demo/make_clips.py --ckpt out/minidance-base-sft.safetensors

# 2. 安装依赖并渲染（需要 Node.js 18+）
cd demo
npm install
npm run stills                                   # 两张架构图
npx remotion render Demo ../out/minidance_demo.mp4
npm run studio                                   # 在浏览器里实时预览、调整
```

片段的提示词和随机种子在 `clips.json` 中，单个片段也可以用 `scripts/sample.py --prompt "..." --seed N` 复现。

注：Remotion 对个人和 3 人以下的团队免费，公司使用请参阅其[许可证](https://www.remotion.dev/license)。
