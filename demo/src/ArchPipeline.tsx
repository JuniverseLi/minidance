import React from "react";
import { AbsoluteFill } from "remotion";
import { Arrow, Box, Legend, Section } from "./diagram";
import { C, FONT } from "./theme";

const W = 262;

export const ArchPipeline: React.FC<{ bg?: string }> = ({ bg = C.bg }) => (
  <AbsoluteFill
    style={{
      background: bg,
      fontFamily: FONT,
      padding: "56px 70px",
      display: "flex",
      flexDirection: "column",
      gap: 46,
    }}
  >
    <div style={{ display: "flex", alignItems: "baseline", gap: 22 }}>
      <div style={{ fontSize: 44, fontWeight: 800, color: C.ink }}>MiniDance 训练与推理流程</div>
      <div style={{ fontSize: 24, color: C.muted }}>Training &amp; inference pipeline</div>
    </div>

    <Section i={0} label="数据准备">
      <Box i={0} w={W} title="OpenVid-1M" sub="约 100 万条视频片段" />
      <Arrow i={1} />
      <Box i={1} w={W} title="规则筛选" sub={<>自然风光 · 慢镜头 · 高质量<br />→ 43,562 条</>} />
      <Arrow i={2} />
      <Box i={2} w={W} title="短字幕" sub={<>LLM 把长字幕压缩成<br />中 / 英文短提示词</>} />
      <Arrow i={3} />
      <Box i={3} w={W} pal={C.frozen} tag="冻结" title="Wan2.2 VAE 编码器" sub={<>33×256×256 视频<br />→ 48×9×16×16 latent</>} />
      <Arrow i={4} />
      <Box i={4} w={W} title="latent 数据集" sub={<>86 个分片 · 约 9 GB<br />只需编码一次</>} />
    </Section>

    <div style={{ height: 96 }} />
    <Section i={5} label="训练">
      <Box i={5} w={W} title={<>latent x₀ + 噪声 ε</>} sub="视频 / 图片（第 0 帧）" />
      <Arrow i={6} />
      <Box i={6} w={W} title="加噪" sub={<>x<sub>t</sub> = (1−t)·x₀ + t·ε<br />t ~ logit-normal, shift 3</>} />
      <Arrow i={7} />
      <div style={{ position: "relative" }}>
        {/* text encoder sits above the DiT without pushing it off the row's center line */}
        <div style={{ position: "absolute", bottom: "100%", left: 0, display: "flex", flexDirection: "column", alignItems: "center", gap: 4, paddingBottom: 4 }}>
          <Box i={7} w={W} pal={C.frozen} tag="冻结" title="Qwen3-0.6B" sub="文本编码器 · 中英文" />
          <Arrow i={8} dir="down" len={30} />
        </div>
        <Box i={8} w={W} pal={C.ours} tag="从零训练" title="MiniDance DiT" sub="119.5M 参数" />
      </div>
      <Arrow i={9} />
      <Box i={9} w={W} pal={C.out} title={<>预测速度 v̂</>} sub="rectified flow" />
      <Arrow i={10} />
      <Box i={10} w={W} title="MSE 损失" sub={<>‖ v̂ − (ε − x₀) ‖²</>} />
    </Section>

    <Section i={11} label="训练阶段">
      <Box i={11} w={420} pal={C.ours} title="① 预训练 Pretrain" sub={<>40,429 条 · 长 / 短字幕混合<br />64k 步 · batch 32 · lr 2e-4 → 2e-5 · L4 约 10.5 小时</>} />
      <Arrow i={12} len={60} />
      <Box i={12} w={420} pal={C.ours} title="② 精调 SFT" sub={<>11,496 条高美学、有运动的片段 · 只用短字幕<br />少见场景多采样 · 3k 步 · lr 5e-5 · 约 35 分钟</>} />
      <Arrow i={13} len={60} />
      <Box i={13} w={W + 60} pal={C.out} title="EMA 权重" sub={<>bf16 safetensors · 239 MB</>} />
    </Section>

    <Section i={14} label="推理">
      <Box i={14} w={W} pal={C.cond} title="提示词" sub="中文 / English" />
      <Arrow i={15} />
      <Box i={15} w={W} pal={C.frozen} tag="冻结" title="Qwen3-0.6B" sub="文本特征 1024 维" />
      <Arrow i={16} />
      <Box i={16} w={W} pal={C.ours} title="DiT × 30 步" sub={<>从纯噪声出发<br />Euler 采样 + CFG 5</>} />
      <Arrow i={17} />
      <Box i={17} w={W} pal={C.frozen} tag="冻结" title="Wan2.2 VAE 解码器" sub="latent → 像素" />
      <Arrow i={18} />
      <Box i={18} w={W} pal={C.out} title="视频" sub={<>33 帧 · 256×256 · 12 fps<br />8 GB 显卡可运行</>} />
    </Section>

    <div style={{ marginTop: "auto", paddingLeft: 144 }}>
      <Legend />
    </div>
  </AbsoluteFill>
);
