export const FONT =
  "'Inter', 'Segoe UI', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans SC', sans-serif";
export const MONO = "'JetBrains Mono', 'Cascadia Code', Consolas, monospace";

// diagram palette (light background, used for README images)
export const C = {
  bg: "#ffffff",
  ink: "#1d2433",
  muted: "#5b6577",
  line: "#9aa4b5",
  data: { fill: "#eef6ff", stroke: "#4a8fe0" },     // data / preprocessing
  frozen: { fill: "#f2f2f4", stroke: "#8a8f99" },   // frozen pretrained parts
  ours: { fill: "#fff3e6", stroke: "#e8833a" },     // trained from scratch
  cond: { fill: "#eefaf1", stroke: "#3fa66a" },     // conditioning
  out: { fill: "#f6efff", stroke: "#8a5cd6" },      // outputs
  op: { fill: "#ffffff", stroke: "#b8c0cc" },       // parameter-free ops
};
