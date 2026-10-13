"""S2: compare frozen video VAEs on our scenery clips (reconstruction quality / tokens / speed).

    uv run python scripts/s2_vae_bench.py --vae ltx097 --clips data/samples data/samples2
    uv run python scripts/s2_vae_bench.py --compose        # side-by-side mp4s + html

Each run writes data/s2/<vae>/metrics.csv and keeps uint8 reconstructions of the
first --keep clips for the side-by-side preview.
"""
import argparse
import glob
import html
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SIZE, FRAMES = 256, 33
OUT = "data/s2"


# ---------------------------------------------------------------- VAEs
class LTX:
    """LTX-Video VAE: 32x32 spatial, 8x temporal, 128 channels."""
    patch = 1

    def __init__(self, repo, dtype=torch.bfloat16):
        from diffusers import AutoencoderKLLTXVideo
        self.vae = AutoencoderKLLTXVideo.from_pretrained(repo, subfolder="vae", torch_dtype=dtype).cuda().eval()
        self.dtype = dtype

    def encode(self, x):
        return self.vae.encode(x.to(self.dtype)).latent_dist.mode()

    def decode(self, z):
        if self.vae.config.timestep_conditioning:  # 0.9.5+ decoder is a 1-step denoiser
            t, s = 0.05, 0.025
            z = (1 - s) * z + s * torch.randn_like(z)
            return self.vae.decode(z, temb=torch.tensor([t], device=z.device, dtype=z.dtype)).sample
        return self.vae.decode(z).sample


class Wan:
    """Wan2.2 (TI2V-5B) VAE: 16x16 spatial, 4x temporal, 48 channels; DiT uses patch 2."""
    patch = 2

    def __init__(self, repo, dtype=torch.float32):
        from diffusers import AutoencoderKLWan
        self.vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=dtype).cuda().eval()
        self.dtype = dtype

    def encode(self, x):
        return self.vae.encode(x.to(self.dtype)).latent_dist.mode()

    def decode(self, z):
        return self.vae.decode(z).sample


class OSDCAE:
    """Open-Sora 2.0 Video DC-AE: 32x32 spatial, 4x temporal, 128 channels.

    Not in diffusers: needs Open-Sora's dc_ae package on sys.path (env OSDCAE_SRC).
    """
    patch = 1

    def __init__(self, dtype=torch.bfloat16):
        import sys
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sys.path.insert(0, os.environ["OSDCAE_SRC"])
        from opensora.models.dc_ae.models.dc_ae import DCAE, dc_ae_f32
        self.vae = DCAE(dc_ae_f32("dc-ae-f32t4c128", None))
        sd = load_file(hf_hub_download("hpcai-tech/Open-Sora-v2-Video-DC-AE", "F32T4C128_AE.safetensors"))
        missing, unexpected = self.vae.load_state_dict(sd, strict=False)
        assert not missing, f"missing keys: {missing[:5]}"
        print(f"osdcae: {len(unexpected)} unexpected keys ignored")
        # runtime attrs normally set by Open-Sora's DC_AE() factory
        self.vae.cfg.is_training = False
        self.vae.use_spatial_tiling = self.vae.use_temporal_tiling = False
        self.vae.decoder.disc_off_grad_ckpt = False
        self.vae = self.vae.to("cuda", dtype).eval()
        self.dtype = dtype

    def encode(self, x):  # non-causal: T must be a multiple of 4 -> use 32 of the 33 frames
        return self.vae.encode(x[:, :, :x.shape[2] // 4 * 4].to(self.dtype))

    def decode(self, z):
        return self.vae.decode(z)


VAES = {
    "ltx097": lambda: LTX("Lightricks/LTX-Video-0.9.7-dev"),
    "wan22": lambda: Wan("Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    "osdcae": lambda: OSDCAE(),
}


# ---------------------------------------------------------------- data / metrics
def load_clip(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    idx = np.round(np.linspace(0, len(frames) - 1, FRAMES)).astype(int)
    clip = [cv2.resize(frames[i], (SIZE, SIZE), interpolation=cv2.INTER_AREA)[..., ::-1] for i in idx]
    return np.stack(clip)  # T H W 3 uint8 RGB


def to_tensor(clip):  # -> 1 3 T H W in [-1, 1]
    return torch.from_numpy(clip.copy()).permute(3, 0, 1, 2)[None].float().div(127.5).sub(1)


def ssim(a, b):  # a, b: N 3 H W in [0, 1]
    k = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    k = (k[:, None] * k[None]) / k.sum() ** 2
    k = k.expand(3, 1, 11, 11)
    blur = lambda x: F.conv2d(x, k, groups=3)
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.mean().item()


def run(args):
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    vae = VAES[args.vae]()
    out_dir = os.path.join(OUT, args.vae)
    os.makedirs(out_dir, exist_ok=True)

    keep_ids = set(pd.read_parquet("data/scenery.parquet")._rowid)
    paths = sorted(p for d in args.clips for p in glob.glob(os.path.join(d, "*.mp4"))
                   if int(os.path.basename(p)[:-4]) in keep_ids)[:args.n]
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for i, p in enumerate(paths):
        rid = os.path.basename(p)[:-4]
        clip = load_clip(p)
        x = to_tensor(clip).cuda()
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.time()
            z = vae.encode(x)
            torch.cuda.synchronize(); t1 = time.time()
            y = vae.decode(z).float().clamp(-1, 1)
            torch.cuda.synchronize(); t2 = time.time()
            T = min(x.shape[2], y.shape[2])  # DC-AE reconstructs 32 frames
            x, y = x[:, :, :T], y[:, :, :T]
            a = x[0].permute(1, 0, 2, 3).add(1).div(2)  # T 3 H W in [0, 1]
            b = y[0].permute(1, 0, 2, 3).add(1).div(2)
            mse = F.mse_loss(a, b).item()
            row = dict(rowid=rid, psnr=10 * np.log10(1 / mse), ssim=ssim(a, b),
                       lpips=lp(a * 2 - 1, b * 2 - 1).mean().item(),
                       enc_s=t1 - t0, dec_s=t2 - t1, latent=str(tuple(z.shape[1:])),
                       tokens=z.shape[2] * (z.shape[3] // vae.patch) * (z.shape[4] // vae.patch))
        rows.append(row)
        if i < args.keep:
            rec = b.permute(0, 2, 3, 1).mul(255).round().byte().cpu().numpy()
            np.save(os.path.join(out_dir, f"{rid}.npy"), rec)
        print(f"[{i + 1}/{len(paths)}] {rid} psnr {row['psnr']:.2f} ssim {row['ssim']:.3f} "
              f"lpips {row['lpips']:.3f} latent {row['latent']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(f"\n{args.vae}: PSNR {df.psnr.mean():.2f}  SSIM {df.ssim.mean():.4f}  LPIPS {df.lpips.mean():.4f}  "
          f"tokens {df.tokens.iloc[0]}  latent {df.latent.iloc[0]}  "
          f"enc {df.enc_s[1:].mean():.2f}s dec {df.dec_s[1:].mean():.2f}s  "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


# ---------------------------------------------------------------- side-by-side preview
def compose(args):
    vaes = [v for v in VAES_ORDER if os.path.exists(os.path.join(OUT, v, "metrics.csv"))]
    summary = []
    for v in vaes:
        m = pd.read_csv(os.path.join(OUT, v, "metrics.csv"))
        summary.append(dict(vae=v, clips=len(m), psnr=m.psnr.mean(), ssim=m.ssim.mean(), lpips=m.lpips.mean(),
                            tokens=m.tokens.iloc[0], latent=m.latent.iloc[0],
                            enc_s=m.enc_s[1:].mean(), dec_s=m.dec_s[1:].mean()))
    summary = pd.DataFrame(summary)
    print(summary.round(4).to_string(index=False))

    ids = sorted(set.intersection(*[{os.path.basename(p)[:-4] for p in glob.glob(os.path.join(OUT, v, "*.npy"))}
                                    for v in vaes]))
    vid_dir = os.path.join(OUT, "compare")
    os.makedirs(vid_dir, exist_ok=True)
    clip_dirs = [d for d in glob.glob("data/samples*") if os.path.isdir(d)]
    for rid in ids:
        src = next(os.path.join(d, f"{rid}.mp4") for d in clip_dirs if os.path.exists(os.path.join(d, f"{rid}.mp4")))
        panels = [load_clip(src)] + [np.load(os.path.join(OUT, v, f"{rid}.npy")) for v in vaes]
        T = min(len(p) for p in panels)
        grid = np.concatenate([p[:T] for p in panels], axis=2)  # T H (W*n) 3
        tmp = os.path.join(vid_dir, f"{rid}.raw.mp4")
        w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 12, (grid.shape[2], grid.shape[1]))
        for f in grid:
            w.write(np.ascontiguousarray(f[..., ::-1]))
        w.release()
        os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -crf 16 -pix_fmt yuv420p "{tmp[:-8]}.mp4"')
        os.remove(tmp)

    labels = "".join(f"<span>{html.escape(n)}</span>" for n in ["original"] + vaes)
    table = summary.round(4).to_html(index=False)
    vids = "".join(f'<video src="compare/{r}.mp4" autoplay loop muted playsinline></video>' for r in ids)
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8>
<title>MiniDance S2 VAE</title><style>body{{font-family:sans-serif;background:#111;color:#ddd;margin:16px}}
table{{border-collapse:collapse;margin-bottom:16px}}td,th{{border:1px solid #444;padding:4px 10px}}
.l{{display:flex;max-width:{256 * (len(vaes) + 1)}px}}.l span{{flex:1;text-align:center;font-weight:bold}}
video{{display:block;width:100%;max-width:{256 * (len(vaes) + 1)}px;margin-bottom:10px}}</style>
<h2>S2: VAE reconstruction at {SIZE}x{SIZE}, {FRAMES} frames</h2>{table}<div class=l>{labels}</div>{vids}""")
    print(f"wrote {len(ids)} comparisons -> {OUT}/index.html")


VAES_ORDER = ["ltx097", "wan22", "osdcae"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", choices=list(VAES))
    ap.add_argument("--clips", nargs="+", default=["data/samples", "data/samples2"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--keep", type=int, default=12)
    ap.add_argument("--compose", action="store_true")
    args = ap.parse_args()
    compose(args) if args.compose else run(args)
