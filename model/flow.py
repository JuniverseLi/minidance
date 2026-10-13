"""Rectified flow / flow matching.

x_t = (1 - t) * x_0 + t * noise,   t in [0, 1]  (t = 1 is pure noise)
The model predicts the velocity v = noise - x_0; sampling integrates dx/dt = v from t=1 to t=0.
"""
import torch
import torch.nn.functional as F


def shift_t(t, shift):
    """Push timesteps toward the noisy end. Larger shift = more effort on global layout / motion."""
    return shift * t / (1 + (shift - 1) * t)


def sample_t(batch, shift, device):
    """Training timesteps: logit-normal (dense around the middle), then shifted."""
    return shift_t(torch.sigmoid(torch.randn(batch, device=device)), shift)


def flow_loss(model, x0, y, mask, shift):
    t = sample_t(x0.shape[0], shift, x0.device)
    noise = torch.randn_like(x0)
    tt = t.view(-1, *([1] * (x0.dim() - 1)))
    xt = (1 - tt) * x0 + tt * noise
    v = model(xt, t, y, mask)
    return F.mse_loss(v.float(), (noise - x0).float())


@torch.no_grad()
def sample(model, shape, y, mask, y_null, mask_null, steps=20, cfg=5.0, shift=3.0, generator=None):
    """Euler sampler with classifier-free guidance. y_null/mask_null: encoding of the empty prompt."""
    B = shape[0]
    x = torch.randn(shape, device=y.device, generator=generator)
    ts = shift_t(torch.linspace(1, 0, steps + 1, device=y.device), shift)
    # pad cond / uncond text to the same length so they run as one batch
    L = max(y.shape[1], y_null.shape[1])
    pad = lambda a, m: (F.pad(a, (0, 0, 0, L - a.shape[1])), F.pad(m, (0, L - m.shape[1])))
    y, mask = pad(y, mask)
    y_null, mask_null = pad(y_null.expand(B, -1, -1), mask_null.expand(B, -1))
    yy, mm = torch.cat([y, y_null]), torch.cat([mask, mask_null])
    for i in range(steps):
        t = ts[i].expand(B)
        v_c, v_u = model(torch.cat([x, x]), torch.cat([t, t]), yy, mm).float().chunk(2)
        v = v_u + cfg * (v_c - v_u)
        x = x + (ts[i + 1] - ts[i]) * v
    return x
