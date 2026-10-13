"""Frozen Wan2.2 video VAE (16x16 spatial, 4x temporal, 48 channels).

It is causal: latent frame 0 encodes only video frame 0, so a single image is
just a 1-frame video. 33 frames -> 9 latent frames.
"""
import torch

WAN_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class WanVAE:
    def __init__(self, repo=WAN_VAE_REPO, device="cuda", dtype=torch.float32):
        from diffusers import AutoencoderKLWan
        self.vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=dtype).to(device).eval()
        self.vae.requires_grad_(False)
        self.dtype = dtype
        c = self.vae.config
        self.mean = torch.tensor(c.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(c.latents_std, device=device).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def encode(self, x):
        """x: B 3 T H W in [-1, 1] -> raw latent B 48 T' H/16 W/16 (posterior mean)."""
        return self.vae.encode(x.to(self.dtype)).latent_dist.mode()

    @torch.no_grad()
    def decode(self, z):
        """raw latent -> B 3 T H W in [-1, 1]."""
        return self.vae.decode(z.to(self.dtype)).sample.clamp(-1, 1)

    # The DiT works on normalized latents (~zero mean, unit variance per channel).
    def normalize(self, z):
        return (z.float() - self.mean) / self.std

    def denormalize(self, z):
        return z.float() * self.std + self.mean
