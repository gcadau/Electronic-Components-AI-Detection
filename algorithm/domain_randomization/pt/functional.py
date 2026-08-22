import numpy as np
import torch
import torch.nn as nn
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg

_EPS = 1e-8



def _param(value, x):
    v = torch.as_tensor(value, dtype=x.dtype, device=x.device).reshape(-1)
    if x.ndim == 4:
        if v.numel() == 1:
            v = v.expand(x.shape[0])
        return v.reshape(-1, 1, 1, 1)
    return v[:1].reshape(1, 1, 1)


def rgb_to_hsv(x):
    r, g, b = x[..., 0, :, :], x[..., 1, :, :], x[..., 2, :, :]
    maxc = x.amax(dim=-3)
    minc = x.amin(dim=-3)
    delta = maxc - minc
    safe_delta = torch.where(delta > _EPS, delta, torch.ones_like(delta))

    v = maxc
    s = torch.where(maxc > _EPS, delta / torch.where(maxc > _EPS, maxc, torch.ones_like(maxc)),
                    torch.zeros_like(maxc))

    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta

    h = torch.where(maxc == r, bc - gc, torch.zeros_like(maxc))
    h = torch.where((maxc == g) & (maxc != r), 2.0 + rc - bc, h)
    h = torch.where((maxc == b) & (maxc != r) & (maxc != g), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = torch.where(delta > _EPS, h, torch.zeros_like(h))

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(x):
    h, s, v = x[..., 0, :, :], x[..., 1, :, :], x[..., 2, :, :]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i.to(torch.long) % 6).unsqueeze(-3)

    channels = torch.stack(
        (
            torch.stack((v, q, p, p, t, v), dim=-3),
            torch.stack((t, v, v, q, p, p), dim=-3),
            torch.stack((p, p, t, v, v, q), dim=-3),
        ),
        dim=-4,
    )  # (..., 3, 6, H, W)

    selector = torch.nn.functional.one_hot(i.squeeze(-3), num_classes=6)
    selector = selector.to(x.dtype).movedim(-1, -3).unsqueeze(-4)  # (..., 1, 6, H, W)
    return (channels * selector).sum(dim=-3)



def invert(x, maximum=255.0):
    return maximum - x


def adjust_brightness(x, delta):
    return x + _param(delta, x)


def adjust_contrast(x, contrast_factor):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return (x - mean) * _param(contrast_factor, x) + mean


def adjust_hue(x, delta):
    hsv = rgb_to_hsv(x)
    h = (hsv[..., 0:1, :, :] + _param(delta, x)) % 1.0
    return hsv_to_rgb(torch.cat((h, hsv[..., 1:, :, :]), dim=-3))


def adjust_saturation(x, saturation_factor):
    hsv = rgb_to_hsv(x)
    s = (hsv[..., 1:2, :, :] * _param(saturation_factor, x)).clamp(0.0, 1.0)
    return hsv_to_rgb(torch.cat((hsv[..., 0:1, :, :], s, hsv[..., 2:, :, :]), dim=-3))


def adjust_jpeg_quality(x, jpeg_quality):
    batched = x.ndim == 4
    imgs = x if batched else x.unsqueeze(0)
    mode = ImageReadMode.GRAY if imgs.shape[-3] == 1 else ImageReadMode.RGB

    q = torch.as_tensor(jpeg_quality, dtype=torch.float32).reshape(-1)
    if q.numel() == 1:
        q = q.expand(imgs.shape[0])

    out = []
    for i in range(imgs.shape[0]):
        quality = int(min(100, max(1, round(float(q[i].item())))))
        u8 = (imgs[i].detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
        rec = decode_jpeg(encode_jpeg(u8, quality=quality), mode=mode)
        out.append(rec.to(device=x.device, dtype=x.dtype) / 255.0)

    res = torch.stack(out)
    return res if batched else res[0]


def flip_left_right(x):
    return x.flip(-1)


def flip_up_down(x):
    return x.flip(-2)



class DRLayer(nn.Module):

    def __init__(self, seed=None, factor=0.9):
        super().__init__()
        self.seed = seed
        self.factor = factor
        self._generator = None
        self._rng = None

    @property
    def generator(self):
        if self._generator is None and self.seed is not None:
            self._generator = torch.Generator().manual_seed(int(self.seed))
        return self._generator

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng


    def _training(self, training):
        return self.training if training is None else bool(training)

    @staticmethod
    def _n(x):
        return x.shape[0] if x.ndim == 4 else 1


    def _coin(self, n, factor=None):
        p = self.factor if factor is None else factor
        return torch.rand(n, generator=self.generator) <= p

    def _uniform(self, n, low, high):
        return torch.rand(n, generator=self.generator) * (high - low) + low

    def _normal(self, n, mean, variance):
        return torch.randn(n, generator=self.generator) * variance + mean

    def _triangular(self, n, low, mode, high):
        return torch.as_tensor(self.rng.triangular(low, mode, high, size=n), dtype=torch.float32)

    def _multivariate_normal(self, n, mean, cov):
        return torch.as_tensor(self.rng.multivariate_normal(mean, cov, size=n), dtype=torch.float32)

    @staticmethod
    def _values(value, n):
        if value is None:
            raise ValueError("`value` must be provided when `rand=False`.")
        v = torch.as_tensor(np.asarray(value, dtype=np.float32)).reshape(-1)
        return v.expand(n) if v.numel() == 1 else v[:n]


    @staticmethod
    def _gate(values, mask, identity):
        return torch.where(mask, values, torch.full_like(values, float(identity)))

    @staticmethod
    def _where(mask, transformed, x):
        shape = (-1, 1, 1, 1) if x.ndim == 4 else (1, 1, 1)
        return torch.where(mask.reshape(shape).to(x.device), transformed, x)


    def get_config(self):
        return {"seed": self.seed, "factor": self.factor}

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.get_config().items())


class NoneTransformation(nn.Module):

    def forward(self, x, training=None):
        return x
