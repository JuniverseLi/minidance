import React from "react";
import { AbsoluteFill } from "remotion";
import { Arrow, Box, Legend, useReveal } from "./diagram";
import { C, FONT, MONO } from "./theme";

const Op: React.FC<{ i: number; children: React.ReactNode; pal?: { fill: string; stroke: string }; note?: string }> = ({
  i,
  children,
  pal = C.ours,
  note,
}) => {
  const o = useReveal(i);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 14, opacity: o }}>
      <div
        style={{
          width: 340,
          padding: "10px 14px",
          boxSizing: "border-box",
          borderRadius: 10,
          background: pal.fill,
          border: `2px solid ${pal.stroke}`,
          fontSize: 19,
          fontWeight: 600,
          color: C.ink,
          textAlign: "center",
        }}
      >
        {children}
      </div>
      {note && <div style={{ fontFamily: MONO, fontSize: 15, color: C.muted, whiteSpace: "nowrap" }}>{note}</div>}
    </div>
  );
};

const Plus: React.FC<{ i: number }> = ({ i }) => {
  const o = useReveal(i);
  return (
    <div style={{ width: 340, display: "flex", justifyContent: "center", opacity: o }}>
      <div
        style={{
          width: 30,
          height: 30,
          borderRadius: 15,
          border: `2.5px solid ${C.line}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 22,
          fontWeight: 700,
          color: C.muted,
          lineHeight: 1,
        }}
      >
        +
      </div>
    </div>
  );
};

const Down: React.FC<{ i: number }> = ({ i }) => (
  <div style={{ width: 340, display: "flex", justifyContent: "center" }}>
    <Arrow i={i} dir="down" len={22} />
  </div>
);

export const ArchDiT: React.FC<{ bg?: string }> = ({ bg = C.bg }) => (
  <AbsoluteFill style={{ background: bg, fontFamily: FONT, padding: "50px 70px", display: "flex", flexDirection: "column" }}>
    <div style={{ display: "flex", alignItems: "baseline", gap: 22 }}>
      <div style={{ fontSize: 44, fontWeight: 800, color: C.ink }}>MiniDance DiT 结构</div>
      <div style={{ fontSize: 24, color: C.muted }}>dim 768 · 12 blocks · 12 heads · 119.5M params</div>
    </div>

    <div style={{ display: "flex", gap: 70, marginTop: 40, alignItems: "flex-start" }}>
      {/* main path */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, width: 330 }}>
        <Box i={0} w={330} title="带噪 latent x_t" sub="48 × 9 × 16 × 16" />
        <Arrow i={1} dir="down" len={30} />
        <Box i={1} w={330} pal={C.ours} title="Patchify + Linear" sub={<>patch (1,2,2) → 9×8×8 = 576 tokens<br />每个 token 768 维</>} />
        <Arrow i={2} dir="down" len={30} />
        <Box i={2} w={330} pal={C.ours} dashed title="DiT Block × 12" sub="详见右侧 →" h={150} />
        <Arrow i={9} dir="down" len={30} />
        <Box i={9} w={330} pal={C.ours} title="LayerNorm + adaLN" sub="shift / scale ← t_emb" />
        <Arrow i={10} dir="down" len={30} />
        <Box i={10} w={330} pal={C.ours} title="Linear + Unpatchify" sub="输出初始化为 0" />
        <Arrow i={11} dir="down" len={30} />
        <Box i={11} w={330} pal={C.out} title="预测速度 v̂" sub="48 × 9 × 16 × 16" />
      </div>

      {/* block detail */}
      <div
        style={{
          opacity: useReveal(2),
          border: `2.5px dashed ${C.ours.stroke}`,
          borderRadius: 18,
          padding: "26px 26px 22px",
          display: "flex",
          flexDirection: "column",
          gap: 4,
          position: "relative",
        }}
      >
        <div style={{ position: "absolute", top: -17, left: 22, background: bg, padding: "0 10px", fontSize: 22, fontWeight: 700, color: C.ours.stroke }}>
          DiT Block
        </div>
        <Op i={3} pal={C.op} note="shift₁, scale₁ ← t_mod">LayerNorm + 调制</Op>
        <Down i={3} />
        <Op i={3} note="3D RoPE (t / h / w), QK-Norm">时空 Self-Attention</Op>
        <Down i={3} />
        <Op i={4} pal={C.op} note="gate₁ ← t_mod，初始化为 0">× gate</Op>
        <Plus i={4} />
        <Down i={5} />
        <Op i={5} pal={C.op}>LayerNorm</Op>
        <Down i={5} />
        <Op i={5} pal={C.cond} note="K, V ← 文本 tokens（带 mask）">Cross-Attention</Op>
        <Plus i={6} />
        <Down i={6} />
        <Op i={6} pal={C.op} note="shift₂, scale₂ ← t_mod">LayerNorm + 调制</Op>
        <Down i={7} />
        <Op i={7} note="hidden = 2048">SwiGLU MLP</Op>
        <Down i={7} />
        <Op i={8} pal={C.op} note="gate₂ ← t_mod">× gate</Op>
        <Plus i={8} />
      </div>

      {/* conditioning */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "center", width: 330 }}>
        <Box i={12} w={330} pal={C.cond} title="时间步 t ∈ [0, 1]" />
        <Arrow i={12} dir="down" len={26} />
        <Box i={12} w={330} pal={C.ours} title="正弦编码 + MLP" sub="→ t_emb (768)" />
        <Arrow i={13} dir="down" len={26} />
        <Box i={13} w={330} pal={C.ours} title="adaLN-single" sub={<>一个共享 MLP → 6 × 768<br />每个 Block 再加自己的偏置表</>} />
        <div style={{ height: 34 }} />
        <Box i={14} w={330} pal={C.cond} title="提示词" sub="中文 / English" />
        <Arrow i={14} dir="down" len={26} />
        <Box i={14} w={330} pal={C.frozen} tag="冻结" title="Qwen3-0.6B" sub="最后一层 hidden，1024 维" />
        <Arrow i={15} dir="down" len={26} />
        <Box i={15} w={330} pal={C.ours} title="RMSNorm + MLP" sub="→ 文本 tokens (768)，送入每个 Cross-Attention" />
      </div>
    </div>

    <div style={{ marginTop: "auto" }}>
      <Legend />
    </div>
  </AbsoluteFill>
);
