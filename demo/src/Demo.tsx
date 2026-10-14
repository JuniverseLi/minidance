import React from "react";
import { loadFont } from "@remotion/google-fonts/Fredoka";
import {
  AbsoluteFill,
  Img,
  Loop,
  OffthreadVideo,
  Series,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import clips from "../clips.json";
import { ArchPipeline } from "./ArchPipeline";
import { RevealContext } from "./diagram";

const logo = staticFile("logo_alpha.png"); // made by make_clips.py

const { fontFamily: FREDOKA } = loadFont("normal", { weights: ["500", "600", "700"], subsets: ["latin"] });
const FONT = `${FREDOKA}, 'Microsoft YaHei', 'PingFang SC', sans-serif`;

// colors picked from the logo
const L = {
  bg: "linear-gradient(160deg, #e8f5ff 0%, #fdf9f3 55%, #fff1e4 100%)",
  navy: "#1f3554",
  muted: "#6a7a93",
  sky: "#3fa9f0",
  peach: "#f58a4b",
  teal: "#2fb9be",
  sun: "#ffcf4a",
  card: "#ffffff",
};

const CLIP_FRAMES = 82; // 33 frames at 12 fps = 2.75 s
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

const usePop = (delay = 0, damping = 11) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: f - delay, fps, config: { damping, stiffness: 160 } });
};

const Clip: React.FC<{ name: string; size: number; radius?: number }> = ({ name, size, radius = 14 }) => (
  <div style={{ position: "relative", width: size, height: size, borderRadius: radius, overflow: "hidden", flexShrink: 0 }}>
    <Loop durationInFrames={CLIP_FRAMES}>
      <OffthreadVideo src={staticFile(`clips/${name}.mp4`)} muted style={{ width: size, height: size, objectFit: "cover" }} />
    </Loop>
  </div>
);

const Polaroid: React.FC<{ children: React.ReactNode; tilt: number; delay?: number; pad?: number }> = ({
  children,
  tilt,
  delay = 0,
  pad = 16,
}) => {
  const s = usePop(delay);
  return (
    <div
      style={{
        background: L.card,
        padding: pad,
        paddingBottom: pad * 1.6,
        borderRadius: 10,
        boxShadow: "0 24px 60px rgba(31,53,84,0.18)",
        transform: `rotate(${tilt * s}deg) scale(${0.6 + 0.4 * s})`,
        opacity: Math.min(1, s * 1.5),
      }}
    >
      {children}
    </div>
  );
};

const Bubble: React.FC<{ children: React.ReactNode; color?: string; delay?: number; style?: React.CSSProperties }> = ({
  children,
  color = L.card,
  delay = 0,
  style,
}) => {
  const s = usePop(delay, 13);
  return (
    <div
      style={{
        position: "relative",
        background: color,
        borderRadius: 26,
        padding: "18px 28px",
        fontSize: 30,
        lineHeight: 1.35,
        color: L.navy,
        boxShadow: "0 12px 30px rgba(31,53,84,0.12)",
        transform: `scale(${s})`,
        transformOrigin: "20% 100%",
        ...style,
      }}
    >
      {children}
      <div
        style={{
          position: "absolute",
          left: 46,
          bottom: -14,
          width: 28,
          height: 28,
          background: color,
          transform: "rotate(45deg)",
          borderRadius: 4,
        }}
      />
    </div>
  );
};

const Typed: React.FC<{ text: string; start: number; cps?: number }> = ({ text, start, cps = 1.4 }) => {
  const f = useCurrentFrame();
  const chars = Array.from(text);
  const n = Math.floor(Math.max(0, f - start) * cps);
  return (
    <span>
      {chars.slice(0, n).join("")}
      <span style={{ opacity: n < chars.length && Math.floor(f / 6) % 2 === 0 ? 1 : 0, color: L.peach }}>▍</span>
    </span>
  );
};

// soft cuts between scenes
const Scene: React.FC<{ dur: number; children: React.ReactNode; bg?: string }> = ({ dur, children, bg = L.bg }) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, 10, dur - 10, dur], [0, 1, 1, 0], clamp);
  return <AbsoluteFill style={{ background: bg, opacity: o, fontFamily: FONT, color: L.navy }}>{children}</AbsoluteFill>;
};

// ---------------------------------------------------------------- 1. logo
const Intro: React.FC = () => {
  const f = useCurrentFrame();
  const s = usePop(4, 9);
  const wobble = Math.sin(f / 9) * 1.5 * Math.min(1, f / 30);
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
      {clips.map((c, i) => {
        // little clips drifting upwards in the background
        const x = (i * 241) % 1700 + 40;
        const y = 1100 - ((f * (1.6 + (i % 3) * 0.5) + i * 160) % 1400);
        return (
          <div key={c.name} style={{ position: "absolute", left: x, top: y, opacity: 0.35, transform: `rotate(${(i % 2 ? 1 : -1) * 8}deg)` }}>
            <Clip name={c.name} size={150} radius={12} />
          </div>
        );
      })}
      <Img src={logo} style={{ width: 1300, transform: `scale(${s}) rotate(${wobble}deg)` }} />
      <div style={{ fontSize: 46, fontWeight: 600, marginTop: 30, opacity: interpolate(f, [30, 45], [0, 1], clamp) }}>
        <Typed text="一个从零训练的文生视频小模型" start={30} cps={0.9} />
      </div>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- 2. how much?
const Strike: React.FC<{ text: string; start: number }> = ({ text, start }) => {
  const f = useCurrentFrame();
  const s = usePop(start, 12);
  const line = interpolate(f, [start + 14, start + 24], [0, 100], clamp);
  return (
    <div style={{ position: "relative", fontSize: 72, fontWeight: 700, color: L.muted, transform: `scale(${s})` }}>
      {text}
      <div style={{ position: "absolute", left: -10, top: "52%", height: 9, width: `calc(${line}% + 20px)`, background: "#ef5a5a", borderRadius: 6, transform: "rotate(-4deg)" }} />
    </div>
  );
};

const Price: React.FC = () => {
  const f = useCurrentFrame();
  const big = usePop(70, 7);
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 46 }}>
      <div style={{ fontSize: 60, fontWeight: 700 }}>从零训练一个视频生成模型，要花多少钱？</div>
      <div style={{ display: "flex", gap: 80, alignItems: "center" }}>
        <Strike text="¥ 1,000,000" start={18} />
        <Strike text="¥ 10,000" start={42} />
        <div style={{ transform: `scale(${big}) rotate(${(1 - big) * -20}deg)`, textAlign: "center" }}>
          <div style={{ fontSize: 150, fontWeight: 700, color: L.peach, lineHeight: 1, textShadow: `6px 6px 0 ${L.sun}` }}>¥12</div>
        </div>
      </div>
      <div style={{ fontSize: 36, color: L.muted, opacity: interpolate(f, [90, 104], [0, 1], clamp) }}>
        一张 Colab L4 · 约 11 小时 · 差不多一顿饭钱 🍜
      </div>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- 3. magic: noise -> waterfall
const STEP0 = 40;   // first frame of the step counter
const PER_STEP = 5; // frames per sampling step
const MAGIC_END = STEP0 + 30 * PER_STEP;

const Magic: React.FC = () => {
  const f = useCurrentFrame();
  const c = clips.find((x) => x.name === "waterfall")!;
  const step = Math.min(30, Math.max(0, Math.floor((f - STEP0) / PER_STEP)));
  const reveal = usePop(MAGIC_END + 6, 10);
  const pad = (i: number) => String(i).padStart(2, "0");
  const flash = interpolate(f, [MAGIC_END, MAGIC_END + 4, MAGIC_END + 16], [0, 0.9, 0], clamp);
  const Card: React.FC<{ title: string; src: string; tilt: number; delay: number }> = ({ title, src, tilt, delay }) => (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 18 }}>
      <Polaroid tilt={tilt} delay={delay}>
        <Img src={staticFile(src)} style={{ width: 500, height: 500, display: "block", borderRadius: 6 }} />
      </Polaroid>
      <div style={{ fontSize: 30, fontWeight: 600 }}>{title}</div>
    </div>
  );
  return (
    <AbsoluteFill style={{ alignItems: "center", paddingTop: 50 }}>
      <div style={{ fontSize: 58, fontWeight: 700 }}>
        🪄 变个魔术：<span style={{ color: L.sky }}>噪声</span> → <span style={{ color: L.peach }}>瀑布</span>
      </div>
      <Bubble delay={8} style={{ marginTop: 26, fontSize: 28 }}>
        “{c.prompt}”
      </Bubble>
      {f < MAGIC_END + 6 ? (
        <div style={{ display: "flex", gap: 90, marginTop: 50, alignItems: "flex-start" }}>
          <Card title="画布上的样子 xₜ" src={`denoise/xt_${pad(step)}.jpg`} tilt={-3} delay={14} />
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14, marginTop: 170, width: 220 }}>
            <div style={{ fontSize: 26, color: L.muted }}>第</div>
            <div style={{ fontSize: 96, fontWeight: 700, color: L.peach, lineHeight: 1 }}>{step}</div>
            <div style={{ fontSize: 26, color: L.muted }}>/ 30 步</div>
            <div style={{ width: 200, height: 14, borderRadius: 7, background: "#dbe7f3", overflow: "hidden" }}>
              <div style={{ width: `${(step / 30) * 100}%`, height: "100%", background: L.teal }} />
            </div>
          </div>
          <Card title="模型脑补的成品 x̂₀" src={`denoise/x0_${pad(Math.min(step, 29))}.jpg`} tilt={3} delay={20} />
        </div>
      ) : (
        <div style={{ marginTop: 40, display: "flex", alignItems: "center", gap: 50 }}>
          <div style={{ transform: `scale(${reveal}) rotate(${(1 - reveal) * 10}deg)` }}>
            <Polaroid tilt={-2} pad={18}>
              <Clip name={c.name} size={560} radius={6} />
            </Polaroid>
          </div>
          <div style={{ fontSize: 120, fontWeight: 700, color: L.peach, transform: `scale(${reveal})` }}>变！✨</div>
        </div>
      )}
      <AbsoluteFill style={{ background: "#fff", opacity: flash, pointerEvents: "none" }} />
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- 4. prompt -> video
const PromptPair: React.FC<{ pair: typeof clips; start: number }> = ({ pair, start }) => (
  <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
    <div style={{ display: "flex", gap: 110 }}>
      {pair.map((c, i) => (
        <div key={c.name} style={{ width: 600, display: "flex", flexDirection: "column", gap: 34 }}>
          <Bubble delay={i * 6} color={i % 2 ? "#fff3e8" : "#e9f6ff"} style={{ minHeight: 96 }}>
            <Typed text={c.prompt} start={start + 4 + i * 6} cps={2.2} />
          </Bubble>
          <Polaroid tilt={i % 2 ? 2.5 : -2.5} delay={10 + i * 6}>
            <Clip name={c.name} size={568} radius={6} />
          </Polaroid>
        </div>
      ))}
    </div>
  </AbsoluteFill>
);

// ---------------------------------------------------------------- 5. how
const How: React.FC = () => (
  <AbsoluteFill style={{ background: "#ffffff" }}>
    <RevealContext.Provider value={{ animate: true, start: 6, gap: 6 }}>
      <AbsoluteFill style={{ transform: "scale(0.95)", transformOrigin: "50% 40%" }}>
        <ArchPipeline />
      </AbsoluteFill>
    </RevealContext.Provider>
  </AbsoluteFill>
);

// ---------------------------------------------------------------- 6. outro
const Chip: React.FC<{ text: string; color: string; i: number }> = ({ text, color, i }) => {
  const s = usePop(8 + i * 5, 10);
  return (
    <div style={{ fontSize: 32, fontWeight: 600, padding: "14px 30px", borderRadius: 40, background: color, color: "#fff", transform: `scale(${s}) rotate(${(i % 2 ? 1 : -1) * 2}deg)` }}>
      {text}
    </div>
  );
};

const Outro: React.FC = () => {
  const f = useCurrentFrame();
  const s = usePop(40, 9);
  const star = usePop(70, 6);
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 44 }}>
      <div style={{ display: "flex", gap: 22 }}>
        <Chip i={0} color={L.sky} text="119.5M 参数" />
        <Chip i={1} color={L.teal} text="预训练 64k 步 + SFT 3k 步" />
        <Chip i={2} color={L.peach} text="总成本 ≈ ¥12" />
        <Chip i={3} color="#8a7ff0" text="8 GB 显卡能推理" />
      </div>
      <Img src={logo} style={{ width: 1100, transform: `scale(${s}) rotate(${Math.sin(f / 10) * 1.2}deg)` }} />
      <div style={{ fontSize: 44, fontWeight: 600, opacity: s }}>
        github.com/<span style={{ color: L.sky }}>JuniverseLi</span>/<span style={{ color: L.peach }}>minidance</span>
      </div>
      <div style={{ fontSize: 36, color: L.muted, display: "flex", alignItems: "center", gap: 14, opacity: star }}>
        代码 · 权重 · 数据集全部开源，点个
        <span style={{ display: "inline-block", fontSize: 54, transform: `scale(${star}) rotate(${(1 - star) * 360}deg)` }}>⭐</span>
        吧！
      </div>
    </AbsoluteFill>
  );
};

export const SCENES = { intro: 110, price: 140, magic: 300, pair: 105, how: 170, outro: 190 };
const PAIRS = [0, 2, 4, 6].map((i) => clips.slice(i, i + 2));
export const DEMO_FRAMES =
  SCENES.intro + SCENES.price + SCENES.magic + PAIRS.length * SCENES.pair + SCENES.how + SCENES.outro;

export const Demo: React.FC = () => (
  <AbsoluteFill style={{ background: "#fdf9f3" }}>
    <Series>
      <Series.Sequence durationInFrames={SCENES.intro}>
        <Scene dur={SCENES.intro}><Intro /></Scene>
      </Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.price}>
        <Scene dur={SCENES.price}><Price /></Scene>
      </Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.magic}>
        <Scene dur={SCENES.magic}><Magic /></Scene>
      </Series.Sequence>
      {PAIRS.map((p) => (
        <Series.Sequence key={p[0].name} durationInFrames={SCENES.pair}>
          <Scene dur={SCENES.pair}><PromptPair pair={p} start={0} /></Scene>
        </Series.Sequence>
      ))}
      <Series.Sequence durationInFrames={SCENES.how}>
        <Scene dur={SCENES.how} bg="#ffffff"><How /></Scene>
      </Series.Sequence>
      <Series.Sequence durationInFrames={SCENES.outro}>
        <Scene dur={SCENES.outro}><Outro /></Scene>
      </Series.Sequence>
    </Series>
  </AbsoluteFill>
);
