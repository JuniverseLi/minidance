import React, { createContext, useContext } from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { C, FONT, MONO } from "./theme";

type Palette = { fill: string; stroke: string };

// When a diagram is shown inside the demo video its parts fade in one by one;
// rendered as a still (README image) everything is visible.
export const RevealContext = createContext<{ animate: boolean; start: number; gap: number }>({
  animate: false,
  start: 0,
  gap: 6,
});

export const useReveal = (i: number) => {
  const frame = useCurrentFrame();
  const { animate, start, gap } = useContext(RevealContext);
  if (!animate) return 1;
  const t0 = start + i * gap;
  return interpolate(frame, [t0, t0 + 10], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
};

export const Box: React.FC<{
  i?: number;
  title: React.ReactNode;
  sub?: React.ReactNode;
  pal?: Palette;
  w?: number;
  h?: number;
  dashed?: boolean;
  tag?: string;
  style?: React.CSSProperties;
}> = ({ i = 0, title, sub, pal = C.data, w = 220, h, dashed, tag, style }) => {
  const o = useReveal(i);
  return (
    <div
      style={{
        position: "relative",
        width: w,
        minHeight: h,
        padding: "14px 16px",
        boxSizing: "border-box",
        background: pal.fill,
        border: `2.5px ${dashed ? "dashed" : "solid"} ${pal.stroke}`,
        borderRadius: 14,
        fontFamily: FONT,
        color: C.ink,
        textAlign: "center",
        opacity: o,
        transform: `translateY(${(1 - o) * 12}px)`,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        gap: 6,
        ...style,
      }}
    >
      {tag && (
        <div
          style={{
            position: "absolute",
            top: -13,
            right: 12,
            fontSize: 15,
            fontWeight: 600,
            padding: "1px 8px",
            borderRadius: 8,
            background: pal.stroke,
            color: "#fff",
          }}
        >
          {tag}
        </div>
      )}
      <div style={{ fontSize: 22, fontWeight: 700, lineHeight: 1.25 }}>{title}</div>
      {sub && <div style={{ fontSize: 16, color: C.muted, lineHeight: 1.35 }}>{sub}</div>}
    </div>
  );
};

export const Arrow: React.FC<{ i?: number; len?: number; dir?: "right" | "down" | "left"; label?: string }> = ({
  i = 0,
  len = 46,
  dir = "right",
  label,
}) => {
  const o = useReveal(i);
  const vertical = dir === "down";
  const w = vertical ? 30 : len;
  const h = vertical ? len : 30;
  const path = vertical
    ? `M15 2 L15 ${len - 10}`
    : dir === "right"
      ? `M2 15 L${len - 10} 15`
      : `M${len - 2} 15 L10 15`;
  const head = vertical
    ? `M7 ${len - 14} L15 ${len - 2} L23 ${len - 14}`
    : dir === "right"
      ? `M${len - 14} 7 L${len - 2} 15 L${len - 14} 23`
      : `M14 7 L2 15 L14 23`;
  return (
    <div style={{ position: "relative", opacity: o, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <svg width={w} height={h} style={{ overflow: "visible" }}>
        <path d={path} stroke={C.line} strokeWidth={3} fill="none" />
        <path d={head} stroke={C.line} strokeWidth={3} fill="none" strokeLinejoin="round" />
      </svg>
      {label && (
        <div
          style={{
            position: "absolute",
            [vertical ? "left" : "top"]: vertical ? 34 : -22,
            fontFamily: MONO,
            fontSize: 14,
            color: C.muted,
            whiteSpace: "nowrap",
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
};

export const Section: React.FC<{ i?: number; label: string; children: React.ReactNode }> = ({
  i = 0,
  label,
  children,
}) => {
  const o = useReveal(i);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
      <div
        style={{
          width: 120,
          flexShrink: 0,
          fontFamily: FONT,
          fontSize: 22,
          fontWeight: 700,
          color: C.muted,
          opacity: o,
          textAlign: "right",
          lineHeight: 1.3,
        }}
      >
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>{children}</div>
    </div>
  );
};

export const Legend: React.FC = () => {
  const items: [string, Palette][] = [
    ["数据 / Data", C.data],
    ["冻结的预训练组件 / Frozen", C.frozen],
    ["从零训练 / Trained from scratch", C.ours],
    ["条件 / Conditioning", C.cond],
    ["输出 / Output", C.out],
  ];
  return (
    <div style={{ display: "flex", gap: 26, fontFamily: FONT, fontSize: 17, color: C.muted }}>
      {items.map(([name, p]) => (
        <div key={name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 20, height: 20, borderRadius: 5, background: p.fill, border: `2px solid ${p.stroke}` }} />
          {name}
        </div>
      ))}
    </div>
  );
};
